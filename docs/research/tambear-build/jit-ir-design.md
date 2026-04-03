# .tbs JIT IR Design

**Date**: 2026-04-02  
**Status**: Architecture documentation — captures the design as implemented by pathmaker in session 2026-04-02  
**File**: `tambear/src/tbs_jit.rs`

---

## The Problem

A `.tbs` chain is a sequence of named operations:

```
describe().normalize().correlation()
```

The naive execution: one CPU call per step. Each step re-reads the full dataset from memory.

The JIT path: classify each step, dispatch GPU-eligible steps through `ComputeEngine` or `TiledEngine`, fall back to the CPU executor only for steps that don't map to a GPU primitive. The GPU steps touch memory once instead of once per step.

---

## The IR: `JitPass`

Every step in a `TbsChain` compiles to exactly one `JitPass`. The plan is a `Vec<JitPass>`.

```rust
enum JitPass {
    ColumnReduce {
        step_index: usize,
        phi_exprs: Vec<&'static str>,          // Pass 1 expressions (e.g. ["1.0", "v", "v * v"])
        needs_centered_pass: bool,              // Whether a Pass 2 with refs is needed
        centered_phi_exprs: Vec<&'static str>, // Pass 2 expressions (centered moments)
    },
    PipelineGpu {
        step_index: usize,   // Steps with built-in GPU (normalize, kmeans, knn, train.*)
    },
    TiledMatrix {
        step_index: usize,
        op: TiledMatrixOp,   // Correlation | Covariance → TiledEngine
    },
    CpuFallback {
        step_index: usize,   // Everything not yet GPU-mapped
    },
}
```

Compilation is pure classification — no optimization passes. Each step maps independently. There is no cross-step fusion yet (see "Future: window fusion" below).

---

## Mapping to the Accumulate Unification

From `accumulate-unification.md`: all computation is `accumulate(grouping, expr, op)`.

`ColumnReduce` is `accumulate` with **`ByKey(column_index)`** grouping:

| Accumulate parameter | ColumnReduce value |
|---|---|
| `grouping` | `ByKey(j)` where `j = i % d` (column index of element `i`) |
| `expr` | the phi expression string, JIT-compiled via NVRTC |
| `op` | `Add` (all scatter phi expressions are additive accumulators) |
| `refs` | per-group values passed as `r` in the phi expression (used for centered moments) |

The column-key construction: `col_keys[i * d + j] = j`. This generates group assignments `[0,1,...,d-1, 0,1,...,d-1, ...]` for `n` rows.

`TiledMatrix` is `accumulate` with **`Tiled(m, n)`** grouping — GEMM via `TiledEngine`.

`PipelineGpu` and `CpuFallback` bypass the IR's accumulate path entirely; they execute through the existing `TamPipeline` / executor infrastructure.

---

## The 5-Phi Practical Limit

`scatter_multi_phi_cuda` allocates one output buffer per phi expression: `n_phi` buffers in addition to keys/values/refs. CUDA has a limit of 32 kernel parameters (each `&Buffer` is one pointer). In practice, kernel register pressure and shared memory consumption become the real constraint:

- 3 phi expressions (`describe` pass 1: count, sum, sum_sq) → fine on any GPU
- 4–5 phi expressions → fine, registers available
- 6+ phi expressions → risk of register spilling on lower-end devices; untested

The practical design rule: **at most 5 phi expressions per scatter_multi_phi call**. The `describe` two-pass design was chosen with this limit in mind: pass 1 has 3 expressions (count, sum, sum_sq), pass 2 has 3 centered expressions (M2, M3, M4). Neither pass exceeds the limit.

---

## Step-to-Pass Classification Table

| `.tbs` step | `JitPass` variant | GPU primitive | Notes |
|---|---|---|---|
| `describe()` | `ColumnReduce` | `scatter_multi_phi` × 2 passes | Pass 1: {1, v, v²}; Pass 2 with refs: {Δ², Δ³, Δ⁴} |
| `mean()` | `ColumnReduce` | `scatter_multi_phi` × 1 pass | {1, v} |
| `variance()`, `std()` | `ColumnReduce` | `scatter_multi_phi` × 1 pass | {1, v, v²} |
| `normalize()` | `PipelineGpu` | `TamPipeline` (internal GPU) | Stats on GPU, elementwise transform also GPU-mapped |
| `correlation()` | `TiledMatrix` | `TiledEngine::CovarianceOp` | Centered dot products |
| `covariance()` | `TiledMatrix` | `TiledEngine::CovarianceOp` | Same op |
| `discover_clusters()` | `PipelineGpu` | `ClusteringEngine` | DBSCAN on GPU distance matrix |
| `kmeans()` | `PipelineGpu` | `KMeansEngine` | k-means++ init + GPU iterations |
| `knn()` | `PipelineGpu` | `TiledEngine::DistanceOp` | Pairwise distances via tiled GEMM |
| `train.linear` | `PipelineGpu` | `TiledEngine` | X'X, X'y via GEMM |
| `train.logistic` | `PipelineGpu` | `TiledEngine` | Gradient steps via GEMM |
| everything else | `CpuFallback` | CPU executor | Runs through `tbs_executor::execute` |

---

## Worked Example: `describe().normalize()`

```
chain = describe().normalize()
```

**Compilation → `JitPlan`:**
```
passes = [
    JitPass::ColumnReduce { step_index: 0, phi_exprs: ["1.0", "v", "v * v"],
                            needs_centered_pass: true,
                            centered_phi_exprs: ["(v-r)*(v-r)", "(v-r)^3", "(v-r)^4"] },
    JitPass::PipelineGpu  { step_index: 1 },
]
```

**Execution:**

1. `ColumnReduce` (step 0 = `describe`):
   - Build `col_keys`: `[0,1,...,d-1, 0,1,...,d-1, ...]` length `n*d`
   - **GPU kernel 1** — `scatter_multi_phi(["1.0", "v", "v*v"], col_keys, data, None, d)`:  
     → `pass1[0][j]` = count for column j  
     → `pass1[1][j]` = sum for column j  
     → `pass1[2][j]` = sum of squares for column j  
   - Compute `means[j] = pass1[1][j] / pass1[0][j]` (CPU, O(d), negligible)
   - **GPU kernel 2** — `scatter_multi_phi(["(v-r)²", "(v-r)³", "(v-r)⁴"], col_keys, data, means, d)`:  
     → `pass2[0][j]` = M2 (centered sum of squares)  
     → `pass2[1][j]` = M3  
     → `pass2[2][j]` = M4  
   - Assemble `DescriptiveResult` per column (CPU, O(d))
   - **Total data passes**: 2 (one per GPU kernel)

2. `PipelineGpu` (step 1 = `normalize`):
   - `pipeline.normalize()` → internally reuses `scatter_multi_phi` for mean/std per column, then elementwise subtract+divide
   - **Total data passes**: 2 (stats pass + transform pass, both GPU)

**Total GPU kernel launches**: 4 (two for describe, two for normalize).  
**Total data touches**: 4 passes over `n*d` elements — all on GPU, no CPU copies between steps.

Without JIT: 2 CPU function calls touching data 4+ times with Python/Rust overhead between each.

---

## Execution Path

```
TbsChain
    └── compile() → JitPlan { passes: Vec<JitPass> }
                        └── execute_plan(data, n, d, y) → TbsResult
                                ├── ColumnReduce  → ComputeEngine::scatter_multi_phi
                                ├── TiledMatrix   → TiledEngine::run
                                ├── PipelineGpu   → TamPipeline methods
                                └── CpuFallback   → tbs_executor::execute_step (CPU)
```

One `ComputeEngine` and one `TamPipeline` are shared across all passes in a plan — the kernel cache in `ComputeEngine` persists, so repeated `describe()` calls in different plans hit the cache on the second call.

---

## Future: Window-Based Fusion

The current design classifies steps independently. The natural next step is **consecutive fusion**: walk the pass list with a sliding window, merge adjacent `ColumnReduce` passes if their combined phi count ≤ 5.

```
describe().mean()  →  today: 3 kernel launches (2 for describe, 1 for mean)
                   →  fused: 2 kernel launches (describe pass 1 absorbs mean's {1, v},
                                                 describe pass 2 unchanged)
```

The fusion algorithm:
1. Walk passes left to right
2. If `passes[i]` is `ColumnReduce` with `n` phi exprs, check `passes[i+1]`
3. If `passes[i+1]` is also `ColumnReduce` with `m` phi exprs, and `n + m ≤ 5`, merge
4. Repeat until no merges possible in a window pass

The correct abstraction for this is `GroupingSpec` — a description of the key-generation strategy that makes two `ColumnReduce` passes mergeable iff they have the same `GroupingSpec`. Right now all `ColumnReduce` passes use `ByKey(column_index)` — so they're all mergeable. When row-level groupings are added (e.g., `group_by(key_column)`), the `GroupingSpec` determines whether fusion is valid.

Current priority: not needed yet. The existing per-step classification covers all shipped operations. File this when adding a third `ColumnReduce` operation type.

---

## Files

| File | Role |
|---|---|
| `tambear/src/tbs_jit.rs` | `JitPass`, `JitPlan`, `compile()`, `execute_plan()` |
| `tambear/src/tbs_parser.rs` | `TbsChain`, `TbsStep` — the input to the JIT |
| `tambear/src/tbs_executor.rs` | CPU executor — used by `CpuFallback` steps |
| `tambear/src/compute_engine.rs` | `ComputeEngine::scatter_multi_phi` — the scatter primitive |
| `winrapids-tiled/src/dispatch.rs` | `TiledEngine::run` — the tiled matrix primitive |
| `tambear/src/pipeline.rs` | `TamPipeline` — used by `PipelineGpu` steps |
