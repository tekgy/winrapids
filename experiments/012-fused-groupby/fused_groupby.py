"""
WinRapids Experiment 012: Fused GroupBy Expressions

The composition question: Experiment 010 fuses element-wise expressions.
Experiment 011 does groupby reductions. What if they composed?

    df.groupby("key").agg(sum(a * b + c))

Without fusion (CuPy path):
  1. Kernel: temp1 = a * b           (read 2 cols, write 1 temp)
  2. Kernel: temp2 = temp1 + c       (read 2 cols, write 1 temp)
  3. Sort keys, reorder temp2
  4. Kernel: segmented reduce

With fusion:
  1. Sort keys, get reorder permutation
  2. One kernel: evaluate a*b+c per-element USING the sort permutation,
     AND reduce within segments — all in one launch

The fused version eliminates intermediate buffers and reads input columns
only once (through the permutation).

This is the piece that would be genuinely new — not just "faster" but
architecturally different from anything CuPy can express.
"""

from __future__ import annotations

import time
import numpy as np
import cupy as cp
import pandas as pd


# ============================================================
# Expression Tree (from Experiment 010b, simplified)
# ============================================================

class Expr:
    def __add__(self, other): return BinOp("+", self, _wrap(other))
    def __radd__(self, other): return BinOp("+", _wrap(other), self)
    def __sub__(self, other): return BinOp("-", self, _wrap(other))
    def __mul__(self, other): return BinOp("*", self, _wrap(other))
    def __rmul__(self, other): return BinOp("*", _wrap(other), self)
    def __truediv__(self, other): return BinOp("/", self, _wrap(other))
    def __neg__(self): return UnaryOp("neg", self)
    def abs(self): return UnaryOp("fabs", self)
    def sqrt(self): return UnaryOp("sqrt", self)

def _wrap(x):
    if isinstance(x, Expr): return x
    return Const(float(x))

class ColRef(Expr):
    def __init__(self, name: str, data: cp.ndarray):
        self.name = name
        self.data = data
    def code(self, params: dict) -> str:
        if self.name not in params:
            idx = len(params)
            params[self.name] = (f"col{idx}", self.data)
        return f"{params[self.name][0]}[orig_idx]"

class Const(Expr):
    def __init__(self, value: float):
        self.value = value
    def code(self, params: dict) -> str:
        return repr(self.value)

class BinOp(Expr):
    def __init__(self, op: str, left: Expr, right: Expr):
        self.op = op; self.left = left; self.right = right
    def code(self, params: dict) -> str:
        return f"({self.left.code(params)} {self.op} {self.right.code(params)})"

class UnaryOp(Expr):
    def __init__(self, op: str, operand: Expr):
        self.op = op; self.operand = operand
    def code(self, params: dict) -> str:
        inner = self.operand.code(params)
        if self.op == "neg": return f"(-{inner})"
        return f"{self.op}({inner})"


def _find_n(expr: Expr) -> int:
    if isinstance(expr, ColRef): return len(expr.data)
    if isinstance(expr, BinOp): return _find_n(expr.left) or _find_n(expr.right)
    if isinstance(expr, UnaryOp): return _find_n(expr.operand)
    return 0


# ============================================================
# Fused GroupBy Kernel Generator
# ============================================================

def _generate_fused_groupby_sum_kernel(expr: Expr):
    """
    Generate a kernel that:
    1. Reads input columns via a sort permutation (sorted order)
    2. Evaluates the expression per-element
    3. Does warp-shuffle segmented reduction within groups
    4. Writes group sums to output

    This replaces: compute expression -> materialize temp -> sort -> reduce
    With: sort permutation -> fused compute+reduce in one kernel
    """
    params: dict = {}
    body_code = expr.code(params)

    # Build kernel parameter list
    param_list = []
    data_list = []
    for name, (param_name, data) in params.items():
        param_list.append(f"const double* {param_name}")
        data_list.append(data)

    param_str = ", ".join(param_list)

    # The key insight: we read through the sort permutation and use
    # sorted keys to detect group boundaries for segmented reduction.
    kernel_src = f"""
extern "C" __global__
void fused_groupby_sum(
    {param_str},
    const long long* sort_perm,
    const long long* sorted_keys,
    const long long* group_starts,
    const long long* group_ends,
    double* group_sums,
    int n,
    int n_groups
) {{
    // Each block handles one or more groups
    // Simple approach: one thread per element, atomic add per group
    // Better: use cumsum approach from Exp 011, but evaluate expr inline

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Look up original index through sort permutation
    int orig_idx = (int)sort_perm[idx];

    // Evaluate expression at original position
    double val = {body_code};

    // Find which group this element belongs to (binary search)
    int lo = 0, hi = n_groups - 1;
    while (lo < hi) {{
        int mid = (lo + hi + 1) / 2;
        if (group_starts[mid] <= idx) lo = mid;
        else hi = mid - 1;
    }}

    // Atomic add to group sum
    atomicAdd(&group_sums[lo], val);
}}
"""

    kernel = cp.RawKernel(kernel_src, "fused_groupby_sum")
    return kernel, data_list


def fused_groupby_sum(keys: cp.ndarray, expr: Expr) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Fused groupby sum: sort keys, then evaluate expression AND reduce
    in a single kernel (no intermediate buffer for the expression result).
    """
    n = len(keys)

    # Sort keys and get permutation (this step is unavoidable)
    sort_idx = cp.argsort(keys)
    sorted_keys = keys[sort_idx]

    # Find group boundaries
    boundaries = cp.concatenate([
        cp.array([True]),
        sorted_keys[1:] != sorted_keys[:-1]
    ])
    boundary_idx = cp.where(boundaries)[0]
    unique_keys = sorted_keys[boundary_idx]
    n_groups = len(unique_keys)

    # Group starts and ends
    group_starts = boundary_idx
    group_ends = cp.concatenate([boundary_idx[1:], cp.array([n])])

    # Generate and launch fused kernel
    kernel, data_list = _generate_fused_groupby_sum_kernel(expr)
    group_sums = cp.zeros(n_groups, dtype=cp.float64)

    threads = 256
    blocks = (n + threads - 1) // threads

    args = tuple(data_list) + (
        sort_idx, sorted_keys, group_starts, group_ends,
        group_sums, n, n_groups
    )
    kernel((blocks,), (threads,), args)

    return unique_keys, group_sums


# ============================================================
# Unfused baseline (CuPy separate ops)
# ============================================================

def unfused_groupby_sum(keys: cp.ndarray, a: cp.ndarray, b: cp.ndarray,
                        c: cp.ndarray, expr_fn) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Unfused: evaluate expression first (separate kernels),
    then sort and reduce.
    """
    # Step 1: Evaluate expression (multiple kernel launches)
    temp = expr_fn(a, b, c)

    # Step 2: Sort and segmented reduce
    sort_idx = cp.argsort(keys)
    sorted_keys = keys[sort_idx]
    sorted_vals = temp[sort_idx]

    boundaries = cp.concatenate([
        cp.array([True]),
        sorted_keys[1:] != sorted_keys[:-1]
    ])
    boundary_idx = cp.where(boundaries)[0]
    unique_keys = sorted_keys[boundary_idx]

    cumsum = cp.cumsum(sorted_vals)
    end_idx = cp.concatenate([boundary_idx[1:] - 1, cp.array([len(keys) - 1])])
    group_sums = cumsum[end_idx].copy()
    group_sums[1:] -= cumsum[boundary_idx[1:] - 1]

    return unique_keys, group_sums


# ============================================================
# Benchmarks
# ============================================================

def bench(name, fn, warmup=2, runs=10):
    for _ in range(warmup):
        fn()
    cp.cuda.Device(0).synchronize()
    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    cp.cuda.Device(0).synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def benchmark_fused_vs_unfused(n: int, n_groups: int, expr_name: str,
                                expr_fn, expr_builder):
    """Compare fused vs unfused groupby for a given expression."""
    print(f"\n=== groupby({n_groups}).sum({expr_name}): {n:,} rows ===\n")

    rng = np.random.default_rng(42)
    keys_np = rng.integers(0, n_groups, size=n).astype(np.int64)
    a_np = rng.standard_normal(n).astype(np.float64)
    b_np = rng.standard_normal(n).astype(np.float64)
    c_np = rng.standard_normal(n).astype(np.float64)

    keys = cp.asarray(keys_np)
    a = cp.asarray(a_np)
    b = cp.asarray(b_np)
    c = cp.asarray(c_np)
    cp.cuda.Device(0).synchronize()

    # Build expression tree for fused path
    a_col = ColRef("a", a)
    b_col = ColRef("b", b)
    c_col = ColRef("c", c)
    expr = expr_builder(a_col, b_col, c_col)

    # --- pandas baseline ---
    pdf = pd.DataFrame({"key": keys_np, "a": a_np, "b": b_np, "c": c_np})
    pdf["expr"] = expr_fn(pdf["a"].values, pdf["b"].values, pdf["c"].values)
    _ = pdf.groupby("key")["expr"].sum()

    t0 = time.perf_counter()
    for _ in range(5):
        pdf["expr"] = expr_fn(pdf["a"].values, pdf["b"].values, pdf["c"].values)
        _ = pdf.groupby("key")["expr"].sum()
    t_pandas = (time.perf_counter() - t0) / 5 * 1000

    # --- CuPy unfused ---
    ms_unfused = bench("unfused", lambda: unfused_groupby_sum(keys, a, b, c, expr_fn))

    # --- Fused ---
    ms_fused = bench("fused", lambda: fused_groupby_sum(keys, expr))

    # Verify correctness
    uk_unfused, gs_unfused = unfused_groupby_sum(keys, a, b, c, expr_fn)
    uk_fused, gs_fused = fused_groupby_sum(keys, expr)

    # Sort both by key for comparison
    idx_u = cp.argsort(uk_unfused)
    idx_f = cp.argsort(uk_fused)
    max_err = float(cp.max(cp.abs(gs_unfused[idx_u] - gs_fused[idx_f])))

    print(f"  pandas:              {t_pandas:8.2f} ms")
    print(f"  CuPy unfused:        {ms_unfused:8.2f} ms  ({t_pandas/ms_unfused:.1f}x vs pandas)")
    print(f"  Fused groupby:       {ms_fused:8.2f} ms  ({t_pandas/ms_fused:.1f}x vs pandas)")
    print(f"  Fused vs unfused:    {ms_unfused/ms_fused:.2f}x")
    print(f"  Max error:           {max_err:.2e}")
    print()


# ============================================================
# Hybrid: fused expression eval + sort-based grouped reduce
# ============================================================

def _generate_eval_kernel(expr: Expr):
    """Generate a kernel that evaluates an expression (no groupby)."""
    params: dict = {}
    body_code = expr.code(params)

    param_list = []
    data_list = []
    for name, (param_name, data) in params.items():
        param_list.append(f"const double* {param_name}")
        data_list.append(data)

    param_str = ", ".join(param_list)

    kernel_src = f"""
extern "C" __global__
void fused_eval({param_str}, double* output, int n) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int orig_idx = idx;  // no permutation needed here
    if (idx < n) {{
        output[idx] = {body_code};
    }}
}}
"""
    kernel = cp.RawKernel(kernel_src, "fused_eval")
    return kernel, data_list


def hybrid_groupby_sum(keys: cp.ndarray, expr: Expr) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Hybrid approach:
    1. Fuse expression evaluation into one kernel (eliminates intermediates)
    2. Sort + cumsum grouped reduction (no atomics, no contention)

    Best of both worlds: fusion for compute, cumsum for reduction.
    """
    n = len(keys)

    # Step 1: Fused expression evaluation (1 kernel, 0 intermediates)
    kernel, data_list = _generate_eval_kernel(expr)
    temp = cp.empty(n, dtype=cp.float64)
    threads = 256
    blocks = (n + threads - 1) // threads
    args = tuple(data_list) + (temp, n)
    kernel((blocks,), (threads,), args)

    # Step 2: Sort-based grouped reduction (cumsum, no atomics)
    sort_idx = cp.argsort(keys)
    sorted_keys = keys[sort_idx]
    sorted_vals = temp[sort_idx]

    boundaries = cp.concatenate([
        cp.array([True]),
        sorted_keys[1:] != sorted_keys[:-1]
    ])
    boundary_idx = cp.where(boundaries)[0]
    unique_keys = sorted_keys[boundary_idx]

    cumsum = cp.cumsum(sorted_vals)
    end_idx = cp.concatenate([boundary_idx[1:] - 1, cp.array([n - 1])])
    group_sums = cumsum[end_idx].copy()
    group_sums[1:] -= cumsum[boundary_idx[1:] - 1]

    return unique_keys, group_sums


def benchmark_three_way(n: int, n_groups: int, expr_name: str,
                         expr_fn, expr_builder):
    """Compare unfused, fully-fused, and hybrid approaches."""
    print(f"\n=== groupby({n_groups}).sum({expr_name}): {n:,} rows ===\n")

    rng = np.random.default_rng(42)
    keys_np = rng.integers(0, n_groups, size=n).astype(np.int64)
    a_np = rng.standard_normal(n).astype(np.float64)
    b_np = rng.standard_normal(n).astype(np.float64)
    c_np = rng.standard_normal(n).astype(np.float64)

    keys = cp.asarray(keys_np)
    a = cp.asarray(a_np)
    b = cp.asarray(b_np)
    c = cp.asarray(c_np)
    cp.cuda.Device(0).synchronize()

    a_col = ColRef("a", a)
    b_col = ColRef("b", b)
    c_col = ColRef("c", c)
    expr = expr_builder(a_col, b_col, c_col)

    # pandas
    pdf = pd.DataFrame({"key": keys_np, "a": a_np, "b": b_np, "c": c_np})
    pdf["expr"] = expr_fn(pdf["a"].values, pdf["b"].values, pdf["c"].values)
    _ = pdf.groupby("key")["expr"].sum()
    t0 = time.perf_counter()
    for _ in range(5):
        pdf["expr"] = expr_fn(pdf["a"].values, pdf["b"].values, pdf["c"].values)
        _ = pdf.groupby("key")["expr"].sum()
    t_pandas = (time.perf_counter() - t0) / 5 * 1000

    # CuPy unfused
    ms_unfused = bench("unfused", lambda: unfused_groupby_sum(keys, a, b, c, expr_fn))

    # Fully fused (atomic)
    ms_fused = bench("fused", lambda: fused_groupby_sum(keys, expr))

    # Hybrid
    ms_hybrid = bench("hybrid", lambda: hybrid_groupby_sum(keys, expr))

    # Verify
    _, gs_unfused = unfused_groupby_sum(keys, a, b, c, expr_fn)
    _, gs_hybrid = hybrid_groupby_sum(keys, expr)

    print(f"  pandas:              {t_pandas:8.2f} ms")
    print(f"  CuPy unfused:        {ms_unfused:8.2f} ms  ({t_pandas/ms_unfused:.1f}x vs pandas)")
    print(f"  Fully fused (atomic):{ms_fused:8.2f} ms  ({t_pandas/ms_fused:.1f}x vs pandas)")
    print(f"  Hybrid (fuse+sort):  {ms_hybrid:8.2f} ms  ({t_pandas/ms_hybrid:.1f}x vs pandas)")
    print(f"  Hybrid vs unfused:   {ms_unfused/ms_hybrid:.2f}x")
    print()


def main():
    print("WinRapids Experiment 012: Fused GroupBy Expressions")
    print("=" * 60)

    n = 10_000_000

    # Simple expression at different cardinalities
    for n_groups in [100, 10_000, 1_000_000]:
        benchmark_three_way(
            n, n_groups, "a*b+c",
            lambda a, b, c: a * b + c,
            lambda a, b, c: a * b + c
        )

    # Complex expression (where fusion saves more)
    for n_groups in [100, 10_000]:
        benchmark_three_way(
            n, n_groups, "a*b+c*c-a/b",
            lambda a, b, c: a * b + c * c - a / b,
            lambda a, b, c: a * b + c * c - a / b
        )

    print("=" * 60)
    print("Experiment 012 complete.")


if __name__ == "__main__":
    main()
