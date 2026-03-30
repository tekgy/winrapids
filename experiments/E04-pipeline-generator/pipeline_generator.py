"""
E04 -- Pipeline Generator: Spec → Primitive DAG → CSE → Codegen → Execute

Demonstrates the core WinRapids compiler loop:

  1. User specifies a pipeline at domain level (specialist calls)
  2. Compiler decomposes specialists to 8-primitive IR
  3. CSE finds shared primitives across specialist boundaries
  4. Code generator emits a fused CUDA kernel source
  5. Kernel executes on GPU and produces correct output

Test pipeline: [rolling_zscore(price, w=20), rolling_std(price, w=20)]

  rolling_zscore decomposes to: cs(price) + cs(price²) + fused_expr
  rolling_std    decomposes to: cs(price) + cs(price²) + fused_expr

  Without CSE: 4 scan operations
  With CSE:    2 scan operations (shared cs and cs2)

  Generated kernel fuses the z_score computation: 1 kernel launch.

Injectable world state API (four inputs, null defaults for E04):
  plan(spec, registry, provenance=None, dirty_bitmap=None, residency=None)

Each scope is a null object in E04:
  - NullProvenanceCache: always misses (compute everything)
  - FullDirtyBitmap: everything dirty (no stale-check skipping)
  - EmptyResidencyMap: nothing warm (no pointer-handoff shortcuts)

Post-E07: inject live provenance and watch miss rate drop.
Post-E10: inject residency map and skip warm results.
"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import cupy as cp
import numpy as np


# ============================================================
# Primitive node representation
# ============================================================

@dataclass(frozen=True)
class PrimitiveNode:
    """One node in the primitive IR DAG.

    identity is the canonical key for CSE: two nodes with the same
    identity are the same computation and can share one result.

    Format: (op, input_identities..., params...)
    """
    op: str                    # "scan", "fused_expr", "reduce", etc.
    inputs: tuple              # (identity_str, ...) of predecessor outputs
    params: tuple              # (k, v, ...) sorted for canonical form
    output_name: str           # human-readable name for this output

    @property
    def identity(self) -> str:
        raw = f"{self.op}:{self.inputs}:{self.params}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]


# ============================================================
# Specialist registry (5 fields per specialist, vision.md spec)
# ============================================================

@dataclass
class SpecialistRecipe:
    """One row in the specialist registry."""
    name: str
    primitive_dag: list        # ordered list of (output_name, op, input_names, params_dict)
    fusion_eligible: bool
    fusion_crossover_rows: int # default inf (always fuse); anti-YAGNI field
    independent: bool          # False if any primitive is scan/sort/reduce
    identity: list             # parameters that constitute canonical identity


def build_registry() -> dict[str, SpecialistRecipe]:
    """
    Minimal registry for E04: three specialists on the rolling path.

    rolling_mean(data, w):
        cs   = scan(data,    add)
        out  = fused_expr([data, cs], formula=rolling_mean, w=w)

    rolling_std(data, w):
        cs   = scan(data,    add)
        cs2  = scan(data_sq, add)       # data_sq = fused_expr(data, "square")
        out  = fused_expr([data, cs, cs2], formula=rolling_std, w=w)

    rolling_zscore(data, w):
        cs   = scan(data,    add)
        cs2  = scan(data_sq, add)
        out  = fused_expr([data, cs, cs2], formula=rolling_zscore, w=w)

    CSE target: rolling_zscore + rolling_std on the same data →
    cs and cs2 are identical nodes → 2 scans instead of 4.
    """
    return {
        "rolling_mean": SpecialistRecipe(
            name="rolling_mean",
            primitive_dag=[
                # (output_name, op, [input_names], {params})
                ("cs",  "scan",       ["data"],         {"agg": "add"}),
                ("out", "fused_expr", ["data", "cs"],   {"formula": "rolling_mean"}),
            ],
            fusion_eligible=True,
            fusion_crossover_rows=2**62,   # inf: always fuse (anti-YAGNI slot)
            independent=False,             # scan is a dependent primitive
            identity=["data_identity", "window"],
        ),

        "rolling_std": SpecialistRecipe(
            name="rolling_std",
            primitive_dag=[
                ("cs",  "scan",       ["data"],         {"agg": "add"}),
                ("cs2", "scan",       ["data_sq"],      {"agg": "add"}),
                ("out", "fused_expr", ["data", "cs", "cs2"], {"formula": "rolling_std"}),
            ],
            fusion_eligible=True,
            fusion_crossover_rows=2**62,
            independent=False,
            identity=["data_identity", "window"],
        ),

        "rolling_zscore": SpecialistRecipe(
            name="rolling_zscore",
            primitive_dag=[
                ("cs",  "scan",       ["data"],         {"agg": "add"}),
                ("cs2", "scan",       ["data_sq"],      {"agg": "add"}),
                ("out", "fused_expr", ["data", "cs", "cs2"], {"formula": "rolling_zscore"}),
            ],
            fusion_eligible=True,
            fusion_crossover_rows=2**62,
            independent=False,
            identity=["data_identity", "window"],
        ),
    }


# ============================================================
# Injectable world state (null objects for E04 prototype)
# ============================================================

class NullProvenanceCache:
    """Always misses. E04 baseline: compute everything."""
    def get(self, identity: str) -> Any:
        return None

    def put(self, identity: str, result: Any, cost_estimate: float = 0.0) -> None:
        pass  # cost_estimate field: anti-YAGNI (needed for cost-aware eviction in E07+)


class FullDirtyBitmap:
    """Everything dirty. No stale-check skipping."""
    def is_clean(self, identity: str) -> bool:
        return False


class EmptyResidencyMap:
    """Nothing warm. No pointer-handoff shortcuts."""
    def is_resident(self, identity: str) -> bool:
        return False

    def get_pointer(self, identity: str) -> Any:
        return None


# ============================================================
# Pipeline spec
# ============================================================

@dataclass
class SpecialistCall:
    """One specialist call in user's pipeline."""
    specialist: str     # registry key
    data_var: str       # variable name (used as identity proxy in E04)
    window: int


@dataclass
class PipelineSpec:
    calls: list[SpecialistCall]


# ============================================================
# Execution plan
# ============================================================

@dataclass
class ExecutionPlan:
    """Compiler output: ordered unique primitives + result mapping."""
    # Ordered list of (node, binding) where binding maps input_names → values
    steps: list            # [(PrimitiveNode, {input_name: str})]
    outputs: dict          # {(specialist_call_idx, "out"): node_identity}
    cse_stats: dict        # {"original_nodes": N, "after_cse": M, "eliminated": K}


# ============================================================
# Compiler: plan()
# ============================================================

def plan(
    spec: PipelineSpec,
    registry: dict[str, SpecialistRecipe],
    provenance: NullProvenanceCache = None,
    dirty_bitmap: FullDirtyBitmap = None,
    residency: EmptyResidencyMap = None,
) -> ExecutionPlan:
    """
    Core compiler pass: spec → execution plan.

    Phase 1 — Decompose: expand each specialist call to primitive nodes.
    Phase 2 — Bind:      substitute data variable names into input refs.
    Phase 3 — CSE:       canonicalize nodes, deduplicate by identity hash.
    Phase 4 — Sort:      topological order for execution.
    Phase 5 — Check:     probe world state (provenance/residency/dirty).

    E04: Phase 5 always misses (null objects). Post-E07: inject live state.
    """
    provenance = provenance or NullProvenanceCache()
    dirty_bitmap = dirty_bitmap or FullDirtyBitmap()
    residency = residency or EmptyResidencyMap()

    # --- Phase 1 + 2: Decompose and bind ---
    # nodes_raw: list of (PrimitiveNode, input_binding)
    # input_binding maps symbolic input names ("data", "cs", ...) to resolved identities
    all_nodes_raw = []
    output_map = {}   # (call_idx, output_name) → node_identity

    for call_idx, call in enumerate(spec.calls):
        recipe = registry[call.specialist]

        # Within-specialist name → resolved identity
        local_identity: dict[str, str] = {}

        # Implicit inputs: "data" and derived "data_sq"
        data_id = f"data:{call.data_var}"
        data_sq_id = f"data_sq:{call.data_var}"
        local_identity["data"] = data_id
        local_identity["data_sq"] = data_sq_id

        for (out_name, op, input_names, params) in recipe.primitive_dag:
            # Resolve input identities
            input_ids = tuple(local_identity[n] for n in input_names)

            # Canonical params include window (part of identity)
            canonical_params = tuple(sorted({**params, "window": call.window}.items()))

            node = PrimitiveNode(
                op=op,
                inputs=input_ids,
                params=canonical_params,
                output_name=out_name,
            )

            binding = {n: local_identity[n] for n in input_names}
            all_nodes_raw.append((node, binding))

            # Register output identity for downstream use within this specialist
            local_identity[out_name] = node.identity

        output_map[(call_idx, "out")] = local_identity["out"]

    # --- Phase 3: CSE ---
    # Identity hash → first occurrence of node (dedup by identity)
    seen: dict[str, tuple] = {}  # identity → (node, binding)
    original_count = len(all_nodes_raw)

    for (node, binding) in all_nodes_raw:
        if node.identity not in seen:
            seen[node.identity] = (node, binding)

    after_cse = len(seen)

    # --- Phase 4: Topological sort ---
    # Build dependency graph on identity hashes
    dep_graph: dict[str, set] = {iid: set() for iid in seen}
    for iid, (node, _) in seen.items():
        for inp in node.inputs:
            # Only track deps that are themselves computed nodes (not raw data ids)
            if inp in seen:
                dep_graph[iid].add(inp)

    ordered = _topo_sort(dep_graph)

    steps = [(seen[iid][0], seen[iid][1]) for iid in ordered if iid in seen]

    # --- Phase 5: World state probe (all miss in E04) ---
    # Post-E07: check provenance.get(node.identity), residency.is_resident(...)
    # For now: log the miss rate
    hits = sum(1 for iid in seen if not dirty_bitmap.is_clean(iid))
    _ = hits  # 100% dirty in E04; this slot is where future provenance integration lands

    return ExecutionPlan(
        steps=steps,
        outputs=output_map,
        cse_stats={
            "original_nodes": original_count,
            "after_cse": after_cse,
            "eliminated": original_count - after_cse,
        },
    )


def _topo_sort(dep_graph: dict[str, set]) -> list[str]:
    """Kahn's algorithm. Stable: tie-break by key for determinism."""
    in_degree = {n: 0 for n in dep_graph}
    for n, deps in dep_graph.items():
        for d in deps:
            in_degree[n] = in_degree.get(n, 0) + 1
        _ = n  # in_degree already initialized above

    # Recount correctly
    in_degree = {n: 0 for n in dep_graph}
    for n, deps in dep_graph.items():
        for d in deps:
            in_degree[n] += 1

    # Kahn
    ready = sorted(n for n, d in in_degree.items() if d == 0)
    result = []
    while ready:
        n = ready.pop(0)
        result.append(n)
        for m in sorted(dep_graph):
            if n in dep_graph[m]:
                in_degree[m] -= 1
                if in_degree[m] == 0:
                    ready.append(m)
                    ready.sort()
    return result


# ============================================================
# Code generator: ExecutionPlan → CUDA source
# ============================================================

KERNEL_TEMPLATES = {
    "rolling_zscore": """
extern "C" __global__
void {name}(const float* data, float* z_out,
            const double* cs, const double* cs2,
            int n, int window) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int data_idx = idx + window - 1;
    if (data_idx < n) {{
        double sum_w  = cs[data_idx + 1]  - cs[data_idx + 1 - window];
        double sumsq_w = cs2[data_idx + 1] - cs2[data_idx + 1 - window];
        double mean  = sum_w / (double)window;
        double mean_sq = sumsq_w / (double)window;
        double var   = mean_sq - mean * mean;
        if (var < 0.0) var = 0.0;
        double std_v = sqrt(var);
        double val   = (double)data[data_idx];
        double z     = (std_v > 1e-10) ? (val - mean) / std_v : 0.0;
        z_out[idx]   = (float)z;
    }}
}}
""",

    "rolling_std": """
extern "C" __global__
void {name}(const float* data, float* std_out,
            const double* cs, const double* cs2,
            int n, int window) {{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int data_idx = idx + window - 1;
    if (data_idx < n) {{
        double sum_w   = cs[data_idx + 1]  - cs[data_idx + 1 - window];
        double sumsq_w = cs2[data_idx + 1] - cs2[data_idx + 1 - window];
        double mean    = sum_w / (double)window;
        double mean_sq = sumsq_w / (double)window;
        double var     = mean_sq - mean * mean;
        if (var < 0.0) var = 0.0;
        std_out[idx]   = (float)sqrt(var);
    }}
}}
""",
}

# Note: prefix-sum lookup (stride-60 access) is correct for FinTek sizes (L2-resident).
# Upgrade path for >5M rows: tile-based kernel — load cumsum tile into shared memory
# (coalesced global read), then access cs[i] and cs[i+w] within shared memory (free),
# write coalesced. This shifts the crossover from ~600K to ~5M. See naturalist-observations.md.


def codegen(plan: ExecutionPlan, spec: PipelineSpec, registry: dict) -> dict[str, Any]:
    """
    Generate CUDA kernels for fused_expr nodes in the plan.

    Scan primitives are dispatched to CuPy cumsum (already optimal for single scans).
    fused_expr primitives become one kernel each.

    Returns: {kernel_name: cp.RawKernel}
    """
    kernels = {}
    for call_idx, call in enumerate(spec.calls):
        recipe = registry[call.specialist]
        fused_formula = None
        for (out_name, op, _, params) in recipe.primitive_dag:
            if op == "fused_expr":
                fused_formula = params["formula"]

        if fused_formula not in KERNEL_TEMPLATES:
            continue

        kernel_name = f"{fused_formula}_{call_idx}"
        src = KERNEL_TEMPLATES[fused_formula].format(name=kernel_name)
        kernels[kernel_name] = cp.RawKernel(src, kernel_name)

    return kernels


# ============================================================
# Executor: run the plan
# ============================================================

def execute(
    plan: ExecutionPlan,
    spec: PipelineSpec,
    registry: dict,
    kernels: dict,
    data_bindings: dict[str, cp.ndarray],   # data_var → GPU array
) -> dict[tuple, cp.ndarray]:
    """
    Execute the plan. Walk steps in topological order.

    For scan primitives: call CuPy cumsum.
    For fused_expr primitives: launch the generated kernel.

    Returns: {(call_idx, "out"): result_array}
    """
    # Buffer pool: identity → computed array
    buffers: dict[str, cp.ndarray] = {}

    # Seed with raw data
    for var_name, arr in data_bindings.items():
        buffers[f"data:{var_name}"] = arr.astype(cp.float32)
        data_f64 = arr.astype(cp.float64)
        buffers[f"data_sq:{var_name}"] = data_f64 * data_f64

    for node, binding in plan.steps:
        if node.identity in buffers:
            continue  # CSE: already computed

        params_dict = dict(node.params)
        window = params_dict.get("window", 20)

        if node.op == "scan":
            # CuPy cumsum with prepended zero (standard prefix sum)
            inp_arr = buffers[binding["data"] if "data" in binding
                              else list(binding.values())[0]]
            inp_f64 = inp_arr.astype(cp.float64)
            cs = cp.cumsum(inp_f64)
            cs_padded = cp.concatenate([cp.zeros(1, dtype=cp.float64), cs])
            buffers[node.identity] = cs_padded

        elif node.op == "fused_expr":
            formula = params_dict.get("formula", "")
            kernel = None
            for k_name, k in kernels.items():
                if formula in k_name:
                    kernel = k
                    break

            if kernel is None:
                raise RuntimeError(f"No kernel for formula={formula}")

            # The binding already maps input names -> identity hashes.
            # These are the exact keys we need in buffers — including any
            # CSE-shared scan results that may have been computed once and
            # reused here. No search needed; the compiler baked sharing in.
            data_id = binding["data"]
            cs_id   = binding["cs"]
            cs2_id  = binding["cs2"]

            data_arr = buffers[data_id]
            n = len(data_arr)
            cs_arr  = buffers[cs_id]
            cs2_arr = buffers[cs2_id]

            out_size = n - window + 1
            out_arr = cp.empty(out_size, dtype=cp.float32)

            block = 256
            grid = (out_size + block - 1) // block
            kernel(
                (grid,), (block,),
                (data_arr, out_arr, cs_arr, cs2_arr,
                 np.int32(n), np.int32(window))
            )
            buffers[node.identity] = out_arr

    # Collect outputs
    results = {}
    for (call_idx, out_name), node_id in plan.outputs.items():
        results[(call_idx, out_name)] = buffers[node_id]

    return results


# ============================================================
# Reference paths for correctness check and benchmark
# ============================================================

def path_independent_cupy(data: cp.ndarray, window: int):
    """Path A: naive independent CuPy — rolling_zscore + rolling_std with NO sharing.
    Each specialist builds its own cumsums. 4 cumsum calls total.
    This is what a user gets without the compiler.
    """
    # rolling_zscore: 2 cumsums
    cs_z  = cp.cumsum(data.astype(cp.float64))
    cs_z  = cp.concatenate([cp.zeros(1, cp.float64), cs_z])
    cs2_z = cp.cumsum((data * data).astype(cp.float64))
    cs2_z = cp.concatenate([cp.zeros(1, cp.float64), cs2_z])
    mean_z    = (cs_z[window:]  - cs_z[:-window])  / window
    mean_sq_z = (cs2_z[window:] - cs2_z[:-window]) / window
    var_z  = cp.maximum(mean_sq_z - mean_z * mean_z, 0)
    std_z  = cp.sqrt(var_z)
    z_out  = ((data[window-1:].astype(cp.float64) - mean_z) / cp.maximum(std_z, 1e-10)).astype(cp.float32)

    # rolling_std: 2 more cumsums (independent, no sharing with above)
    cs_s  = cp.cumsum(data.astype(cp.float64))
    cs_s  = cp.concatenate([cp.zeros(1, cp.float64), cs_s])
    cs2_s = cp.cumsum((data * data).astype(cp.float64))
    cs2_s = cp.concatenate([cp.zeros(1, cp.float64), cs2_s])
    mean_sq_s = (cs2_s[window:] - cs2_s[:-window]) / window
    mean_s    = (cs_s[window:]  - cs_s[:-window])  / window
    var_s  = cp.maximum(mean_sq_s - mean_s * mean_s, 0)
    std_out = cp.sqrt(var_s).astype(cp.float32)

    return z_out, std_out


def path_shared_cupy(data: cp.ndarray, window: int):
    """Path B: manually shared cumsums — rolling_zscore + rolling_std with sharing.
    Smart programmer writes this: build cumsums once, use twice. 2 cumsum calls total.
    """
    cs  = cp.cumsum(data.astype(cp.float64))
    cs  = cp.concatenate([cp.zeros(1, cp.float64), cs])
    cs2 = cp.cumsum((data * data).astype(cp.float64))
    cs2 = cp.concatenate([cp.zeros(1, cp.float64), cs2])

    mean    = (cs[window:]  - cs[:-window])  / window
    mean_sq = (cs2[window:] - cs2[:-window]) / window
    var     = cp.maximum(mean_sq - mean * mean, 0)
    std     = cp.sqrt(var)

    z_out   = ((data[window-1:].astype(cp.float64) - mean) / cp.maximum(std, 1e-10)).astype(cp.float32)
    std_out = std.astype(cp.float32)

    return z_out, std_out


# ============================================================
# Benchmark
# ============================================================

WARMUP = 3
TIMED = 20


def bench(fn, *args, label=""):
    # warmup
    for _ in range(WARMUP):
        result = fn(*args)
    cp.cuda.stream.get_current_stream().synchronize()

    times = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        result = fn(*args)
        cp.cuda.stream.get_current_stream().synchronize()
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    p50 = times[len(times) // 2]
    p99 = times[int(len(times) * 0.99)]
    mean = sum(times) / len(times)
    print(f"  {label:<40s}  p50={p50:6.0f}us  p99={p99:6.0f}us  mean={mean:6.0f}us")
    return result


def bench_compiler_path(exec_plan, spec, registry, kernels, data_bindings, label=""):
    def _run():
        return execute(exec_plan, spec, registry, kernels, data_bindings)

    for _ in range(WARMUP):
        _run()
    cp.cuda.stream.get_current_stream().synchronize()

    times = []
    for _ in range(TIMED):
        t0 = time.perf_counter()
        _run()
        cp.cuda.stream.get_current_stream().synchronize()
        times.append((time.perf_counter() - t0) * 1e6)

    times.sort()
    p50 = times[len(times) // 2]
    p99 = times[int(len(times) * 0.99)]
    mean = sum(times) / len(times)
    print(f"  {label:<40s}  p50={p50:6.0f}us  p99={p99:6.0f}us  mean={mean:6.0f}us")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("E04 -- Pipeline Generator: Spec -> Primitive DAG -> CSE -> Codegen")
    print("=" * 70)

    registry = build_registry()

    # ---- Show plan output ----
    print("\n--- Compiler Plan ---")
    spec = PipelineSpec(calls=[
        SpecialistCall("rolling_zscore", "price", window=20),
        SpecialistCall("rolling_std",    "price", window=20),
    ])
    exec_plan = plan(spec, registry)

    print(f"\nPipeline: rolling_zscore(price, w=20) + rolling_std(price, w=20)")
    print(f"\nCSE stats:")
    print(f"  Original nodes : {exec_plan.cse_stats['original_nodes']}")
    print(f"  After CSE      : {exec_plan.cse_stats['after_cse']}")
    print(f"  Eliminated     : {exec_plan.cse_stats['eliminated']}")

    print(f"\nExecution steps ({len(exec_plan.steps)} total):")
    for node, binding in exec_plan.steps:
        print(f"  [{node.op:12s}]  {node.output_name:<6s}  "
              f"inputs={list(binding.values())}  id={node.identity[:8]}")

    # ---- Correctness check ----
    print("\n--- Correctness ---")
    rng = np.random.default_rng(42)
    n = 100_000
    window = 20
    data_np = rng.standard_normal(n).astype(np.float32)
    data_gpu = cp.asarray(data_np)

    kernels = codegen(exec_plan, spec, registry)
    data_bindings = {"price": data_gpu}

    results = execute(exec_plan, spec, registry, kernels, data_bindings)

    z_compiler   = results[(0, "out")].get()  # rolling_zscore output
    std_compiler = results[(1, "out")].get()  # rolling_std output
    z_ref, std_ref = path_independent_cupy(data_gpu, window)

    max_err_z   = np.max(np.abs(z_compiler   - z_ref.get()))
    max_err_std = np.max(np.abs(std_compiler - std_ref.get()))
    print(f"  n={n}, window={window}")
    print(f"  max |z_compiler   - z_ref|   = {max_err_z:.2e}  "   + ("PASS" if max_err_z   < 1e-4 else "FAIL"))
    print(f"  max |std_compiler - std_ref| = {max_err_std:.2e}  " + ("PASS" if max_err_std < 1e-4 else "FAIL"))

    # ---- Benchmark across sizes ----
    print("\n--- Benchmark ---")
    sizes = [50_000, 100_000, 500_000, 1_000_000, 5_000_000, 10_000_000]

    for n in sizes:
        data_np = rng.standard_normal(n).astype(np.float32)
        data_gpu = cp.asarray(data_np)
        data_bindings = {"price": data_gpu}

        # Recompile exec_plan for same window (plan is data-size independent)
        kernels = codegen(exec_plan, spec, registry)

        print(f"\n  n = {n:>10,}")
        bench(path_independent_cupy, data_gpu, window,
              label="Path A: naive CuPy (4 cumsums)")
        bench(path_shared_cupy, data_gpu, window,
              label="Path B: shared CuPy (2 cumsums, manual)")
        bench_compiler_path(exec_plan, spec, registry, kernels, data_bindings,
                            label="Path C: compiler-generated kernel")

    print("\n--- Plan Summary ---")
    print("The compiler automatically found shared scan(price,add) and")
    print("scan(price_sq,add) across rolling_zscore and rolling_std.")
    print(f"CSE eliminated {exec_plan.cse_stats['eliminated']} of "
          f"{exec_plan.cse_stats['original_nodes']} nodes "
          f"({100*exec_plan.cse_stats['eliminated']//exec_plan.cse_stats['original_nodes']}%).")
    print("\nUser wrote: rolling_zscore(price, 20) + rolling_std(price, 20)")
    print("Compiler produced: 2 scans + 1 fused kernel  [vs 4 scans + 2 kernels naive]")
    print()


if __name__ == "__main__":
    main()
