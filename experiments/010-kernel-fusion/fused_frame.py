"""
WinRapids Experiment 010b: Python-level Kernel Fusion via CuPy RawKernel codegen.

The C++ expression templates prove that fusion works and matches hand-written
kernels. But C++ templates require compile-time knowledge of the expression.
For a DataFrame, expressions are built at runtime in Python.

This module generates fused CUDA kernel source code from Python expression
trees, then compiles them via CuPy's RawKernel JIT. The generated kernels
are cached — pay JIT cost once, amortized over repeated evaluations.

Architecture:
  - FusedColumn wraps a CuPy array (like GpuColumn from Exp 004)
  - Arithmetic operators build an expression tree (lazy, no kernel launch)
  - .evaluate() walks the tree, generates CUDA source, compiles, launches
  - Kernel cache: same expression structure => reuse compiled kernel

Comparison targets:
  - CuPy eager: launches kernel per operation (Exp 010 baseline)
  - C++ fused:  0.19 ms for any expression (Exp 010 fused_ops.cu)
  - This:       should approach C++ performance after first JIT compile
"""

from __future__ import annotations

import time
from typing import Optional
import hashlib

import numpy as np
import cupy as cp


# ============================================================
# Expression Tree Nodes
# ============================================================

class Expr:
    """Base class for expression tree nodes."""

    def __add__(self, other):
        return BinOp("+", self, _wrap(other))

    def __radd__(self, other):
        return BinOp("+", _wrap(other), self)

    def __sub__(self, other):
        return BinOp("-", self, _wrap(other))

    def __rsub__(self, other):
        return BinOp("-", _wrap(other), self)

    def __mul__(self, other):
        return BinOp("*", self, _wrap(other))

    def __rmul__(self, other):
        return BinOp("*", _wrap(other), self)

    def __truediv__(self, other):
        return BinOp("/", self, _wrap(other))

    def __rtruediv__(self, other):
        return BinOp("/", _wrap(other), self)

    def __gt__(self, other):
        return Compare(">", self, _wrap(other))

    def __lt__(self, other):
        return Compare("<", self, _wrap(other))

    def __ge__(self, other):
        return Compare(">=", self, _wrap(other))

    def __le__(self, other):
        return Compare("<=", self, _wrap(other))

    def __neg__(self):
        return UnaryOp("neg", self)

    def __abs__(self):
        return UnaryOp("fabs", self)

    def sqrt(self):
        return UnaryOp("sqrt", self)

    def abs(self):
        return UnaryOp("fabs", self)


def _wrap(x):
    """Wrap a scalar into a Const node."""
    if isinstance(x, Expr):
        return x
    return Const(float(x))


class ColRef(Expr):
    """Reference to a GPU column (leaf node)."""

    def __init__(self, name: str, data: cp.ndarray):
        self.name = name
        self.data = data

    def code(self, params: dict) -> str:
        if self.name not in params:
            idx = len(params)
            params[self.name] = (f"col{idx}", self.data)
        param_name = params[self.name][0]
        return f"{param_name}[idx]"

    def signature_key(self, params: dict) -> str:
        if self.name not in params:
            idx = len(params)
            params[self.name] = f"col{idx}"
        return params[self.name]


class Const(Expr):
    """A scalar constant."""

    def __init__(self, value: float):
        self.value = value

    def code(self, params: dict) -> str:
        # Use repr for full precision
        return repr(self.value)

    def signature_key(self, params: dict) -> str:
        return f"const_{self.value}"


class BinOp(Expr):
    """Binary operation node."""

    def __init__(self, op: str, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right

    def code(self, params: dict) -> str:
        l = self.left.code(params)
        r = self.right.code(params)
        return f"({l} {self.op} {r})"

    def signature_key(self, params: dict) -> str:
        l = self.left.signature_key(params)
        r = self.right.signature_key(params)
        return f"({l}{self.op}{r})"


class Compare(Expr):
    """Comparison node (returns double 1.0/0.0 for masking)."""

    def __init__(self, op: str, left: Expr, right: Expr):
        self.op = op
        self.left = left
        self.right = right

    def code(self, params: dict) -> str:
        l = self.left.code(params)
        r = self.right.code(params)
        return f"(({l} {self.op} {r}) ? 1.0 : 0.0)"

    def signature_key(self, params: dict) -> str:
        l = self.left.signature_key(params)
        r = self.right.signature_key(params)
        return f"({l}{self.op}{r})"


class UnaryOp(Expr):
    """Unary operation node."""

    def __init__(self, op: str, operand: Expr):
        self.op = op
        self.operand = operand

    def code(self, params: dict) -> str:
        inner = self.operand.code(params)
        if self.op == "neg":
            return f"(-{inner})"
        return f"{self.op}({inner})"

    def signature_key(self, params: dict) -> str:
        inner = self.operand.signature_key(params)
        if self.op == "neg":
            return f"(-{inner})"
        return f"{self.op}({inner})"


class WhereOp(Expr):
    """Ternary where(cond, then, else)."""

    def __init__(self, cond: Expr, then_val: Expr, else_val: Expr):
        self.cond = cond
        self.then_val = then_val
        self.else_val = else_val

    def code(self, params: dict) -> str:
        c = self.cond.code(params)
        t = self.then_val.code(params)
        e = self.else_val.code(params)
        return f"({c} != 0.0 ? {t} : {e})"

    def signature_key(self, params: dict) -> str:
        c = self.cond.signature_key(params)
        t = self.then_val.signature_key(params)
        e = self.else_val.signature_key(params)
        return f"where({c},{t},{e})"


def where(cond: Expr, then_val, else_val) -> WhereOp:
    """Fused where expression."""
    return WhereOp(cond, _wrap(then_val), _wrap(else_val))


# ============================================================
# Kernel Code Generation and Caching
# ============================================================

_kernel_cache: dict[str, cp.RawKernel] = {}


def _generate_kernel(expr: Expr) -> tuple[cp.RawKernel, list[cp.ndarray]]:
    """Generate a fused CUDA kernel from an expression tree."""

    # Collect column references and generate code
    params: dict = {}  # name -> (param_name, data)
    body_code = expr.code(params)

    # Build kernel signature
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
    if (idx < n) {{
        output[idx] = {body_code};
    }}
}}
"""

    # Cache key based on expression structure
    sig_params: dict = {}
    cache_key = expr.signature_key(sig_params)
    key_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]

    if key_hash in _kernel_cache:
        return _kernel_cache[key_hash], data_list

    kernel = cp.RawKernel(kernel_src, "fused_eval")
    _kernel_cache[key_hash] = kernel
    return kernel, data_list


def _generate_reduce_kernel(expr: Expr) -> tuple[cp.RawKernel, list[cp.ndarray]]:
    """Generate a fused compute+reduce CUDA kernel."""

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
void fused_reduce({param_str}, double* partial, int n) {{
    extern __shared__ double sdata[];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    double val = 0.0;

    // Evaluate expression at two positions (grid-stride)
    if (i < n) {{
        int idx = i;
        val += {body_code};
    }}
    if (i + blockDim.x < n) {{
        int idx = i + blockDim.x;
        val += {body_code};
    }}

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);

    int lane = tid & 31;
    int warp_id = tid >> 5;
    if (lane == 0) sdata[warp_id] = val;
    __syncthreads();

    int num_warps = blockDim.x >> 5;
    if (warp_id == 0) {{
        val = (lane < num_warps) ? sdata[lane] : 0.0;
        for (int offset = 16; offset > 0; offset >>= 1)
            val += __shfl_down_sync(0xFFFFFFFF, val, offset);
        if (lane == 0) partial[blockIdx.x] = val;
    }}
}}
"""

    sig_params: dict = {}
    cache_key = "reduce_" + expr.signature_key(sig_params)
    key_hash = hashlib.md5(cache_key.encode()).hexdigest()[:12]

    if key_hash in _kernel_cache:
        return _kernel_cache[key_hash], data_list

    kernel = cp.RawKernel(kernel_src, "fused_reduce")
    _kernel_cache[key_hash] = kernel
    return kernel, data_list


# ============================================================
# FusedColumn — lazy evaluation with kernel fusion
# ============================================================

class FusedColumn(Expr):
    """
    A GPU column that builds expression trees lazily and fuses at evaluation.

    Usage:
        a = FusedColumn("a", cupy_array_a)
        b = FusedColumn("b", cupy_array_b)
        c = FusedColumn("c", cupy_array_c)

        # This builds an expression tree (no GPU work yet):
        result_expr = a * b + c

        # This generates a fused kernel, compiles, and launches:
        result = result_expr.evaluate()  # -> CuPy array

        # Fused reduction:
        total = result_expr.fused_sum()  # -> float
    """

    def __init__(self, name: str, data: cp.ndarray):
        self.name = name
        self.data = data
        self._n = len(data)

    def code(self, params: dict) -> str:
        if self.name not in params:
            idx = len(params)
            params[self.name] = (f"col{idx}", self.data)
        return f"{params[self.name][0]}[idx]"

    def signature_key(self, params: dict) -> str:
        if self.name not in params:
            idx = len(params)
            params[self.name] = f"col{idx}"
        return params[self.name]

    @staticmethod
    def _find_n(expr: Expr) -> int:
        """Walk tree to find array length from any ColRef/FusedColumn."""
        if isinstance(expr, (ColRef, FusedColumn)):
            return len(expr.data)
        if isinstance(expr, BinOp):
            return FusedColumn._find_n(expr.left) or FusedColumn._find_n(expr.right)
        if isinstance(expr, Compare):
            return FusedColumn._find_n(expr.left) or FusedColumn._find_n(expr.right)
        if isinstance(expr, UnaryOp):
            return FusedColumn._find_n(expr.operand)
        if isinstance(expr, WhereOp):
            return (FusedColumn._find_n(expr.cond)
                    or FusedColumn._find_n(expr.then_val)
                    or FusedColumn._find_n(expr.else_val))
        return 0


def evaluate(expr: Expr) -> cp.ndarray:
    """Evaluate an expression tree by generating and launching a fused kernel."""
    n = FusedColumn._find_n(expr)
    kernel, data_list = _generate_kernel(expr)

    output = cp.empty(n, dtype=cp.float64)
    threads = 256
    blocks = (n + threads - 1) // threads

    args = tuple(data_list) + (output, n)
    kernel((blocks,), (threads,), args)
    return output


def fused_sum(expr: Expr) -> float:
    """Evaluate expression AND reduce to scalar in a single fused kernel."""
    n = FusedColumn._find_n(expr)
    kernel, data_list = _generate_reduce_kernel(expr)

    threads = 256
    blocks = (n + threads * 2 - 1) // (threads * 2)
    smem = (threads // 32) * 8  # sizeof(double) per warp

    partial = cp.empty(blocks, dtype=cp.float64)
    args = tuple(data_list) + (partial, n)
    kernel((blocks,), (threads,), args, shared_mem=smem)
    return float(cp.sum(partial))


# ============================================================
# Benchmarks
# ============================================================

def bench(name, fn, warmup=2, runs=20):
    for _ in range(warmup):
        fn()
    cp.cuda.Device(0).synchronize()

    t0 = time.perf_counter()
    for _ in range(runs):
        fn()
    cp.cuda.Device(0).synchronize()
    return (time.perf_counter() - t0) / runs * 1000


def main():
    print("WinRapids Experiment 010b: Python Kernel Fusion (codegen)")
    print("=" * 60)

    n = 10_000_000
    print(f"\nData size: {n:,} elements ({n * 8 / 1e6:.1f} MB per column)\n")

    rng = np.random.default_rng(42)
    a_np = rng.standard_normal(n)
    b_np = rng.standard_normal(n)
    c_np = rng.standard_normal(n)

    a_cp = cp.asarray(a_np)
    b_cp = cp.asarray(b_np)
    c_cp = cp.asarray(c_np)
    cp.cuda.Device(0).synchronize()

    # Create FusedColumns
    a = FusedColumn("a", a_cp)
    b = FusedColumn("b", b_cp)
    c = FusedColumn("c", c_cp)

    # ---- Test 1: a * b + c ----
    print("=== Test 1: a * b + c ===")

    expr_fma = a * b + c

    # CuPy eager
    ms_cupy = bench("cupy", lambda: a_cp * b_cp + c_cp)
    print(f"  CuPy eager (2 kernels): {ms_cupy:.4f} ms")

    # Fused
    ms_fused = bench("fused", lambda: evaluate(expr_fma))
    print(f"  Fused (1 kernel):       {ms_fused:.4f} ms")
    print(f"  Speedup:                {ms_cupy/ms_fused:.2f}x")

    # Verify
    r_cupy = cp.asnumpy(a_cp * b_cp + c_cp)
    r_fused = cp.asnumpy(evaluate(expr_fma))
    print(f"  Max error:              {np.max(np.abs(r_cupy - r_fused)):.2e}")
    print()

    # ---- Test 2: a*b + c*c - a/b ----
    print("=== Test 2: a*b + c*c - a/b ===")

    expr_complex = a * b + c * c - a / b

    ms_cupy = bench("cupy", lambda: a_cp * b_cp + c_cp * c_cp - a_cp / b_cp)
    ms_fused = bench("fused", lambda: evaluate(expr_complex))
    print(f"  CuPy eager (5 kernels): {ms_cupy:.4f} ms")
    print(f"  Fused (1 kernel):       {ms_fused:.4f} ms")
    print(f"  Speedup:                {ms_cupy/ms_fused:.2f}x")
    print()

    # ---- Test 3: where(a > 0, b * c, -b * c) ----
    print("=== Test 3: where(a > 0, b * c, -b * c) ===")

    expr_where = where(a > 0, b * c, -b * c)

    ms_cupy = bench("cupy", lambda: cp.where(a_cp > 0, b_cp * c_cp, -b_cp * c_cp))
    ms_fused = bench("fused", lambda: evaluate(expr_where))
    print(f"  CuPy eager (5 kernels): {ms_cupy:.4f} ms")
    print(f"  Fused (1 kernel):       {ms_fused:.4f} ms")
    print(f"  Speedup:                {ms_cupy/ms_fused:.2f}x")
    print()

    # ---- Test 4: sum(a * b + c) — fused compute + reduce ----
    print("=== Test 4: sum(a * b + c) — fused compute + reduce ===")

    ms_cupy = bench("cupy", lambda: float(cp.sum(a_cp * b_cp + c_cp)))
    ms_fused = bench("fused", lambda: fused_sum(expr_fma))
    cupy_result = float(cp.sum(a_cp * b_cp + c_cp))
    fused_result = fused_sum(expr_fma)
    print(f"  CuPy eager (3 kernels): {ms_cupy:.4f} ms  result={cupy_result:.6f}")
    print(f"  Fused (1 kernel):       {ms_fused:.4f} ms  result={fused_result:.6f}")
    print(f"  Speedup:                {ms_cupy/ms_fused:.2f}x")
    print()

    # ---- Test 5: sqrt(abs(a*b + c*c - a)) ----
    print("=== Test 5: sqrt(abs(a*b + c*c - a)) ===")

    expr_deep = (a * b + c * c - a).abs().sqrt()

    ms_cupy = bench("cupy", lambda: cp.sqrt(cp.abs(a_cp * b_cp + c_cp * c_cp - a_cp)))
    ms_fused = bench("fused", lambda: evaluate(expr_deep))
    print(f"  CuPy eager (6 kernels): {ms_cupy:.4f} ms")
    print(f"  Fused (1 kernel):       {ms_fused:.4f} ms")
    print(f"  Speedup:                {ms_cupy/ms_fused:.2f}x")
    print()

    # ---- JIT overhead analysis ----
    print("=== JIT Compilation Overhead ===")
    _kernel_cache.clear()

    t0 = time.perf_counter()
    _ = evaluate(a * b + c)
    cp.cuda.Device(0).synchronize()
    t_first = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    _ = evaluate(a * b + c)
    cp.cuda.Device(0).synchronize()
    t_cached = (time.perf_counter() - t0) * 1000

    print(f"  First call (JIT compile): {t_first:.1f} ms")
    print(f"  Cached call:              {t_cached:.4f} ms")
    print(f"  JIT overhead:             {t_first - t_cached:.1f} ms (one-time cost)")
    print()

    print(f"Kernel cache size: {len(_kernel_cache)} compiled kernels")

    print("\n" + "=" * 60)
    print("Experiment 010b complete.")


if __name__ == "__main__":
    main()
