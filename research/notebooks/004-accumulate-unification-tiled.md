# Lab Notebook 004: Accumulate Unification — Tiled Grouping Proven

**Date**: 2026-03-31
**Author**: Pathmaker
**Branch**: main
**Status**: Active
**Hardware**: NVIDIA RTX PRO 6000 Blackwell, CUDA 13.1

---

## Context & Motivation

The accumulate unification spec claims ALL primitives collapse to `accumulate(data, grouping, expr, op)`. Three of the eight grouping patterns were already wired (All, ByKey, Masked). Tiled was marked "todo — use TiledEngine directly."

The question: can Tiled dispatch through the same `accumulate()` API? And can the logistic regression training loop (proven in notebook 002 via direct TiledEngine calls) be re-expressed as pure accumulate calls?

---

## Experiment 1: Tiled Grouping Dispatch

### Before

**Hypothesis**: `Grouping::Tiled` can be wired through the existing `accumulate()` function signature by carrying the second matrix (`b`) in the grouping enum, and adding `Op::DotProduct` / `Op::Distance` for kernel selection.

**Design challenge**: The `accumulate()` signature takes one `values` slice, but tiled GEMM needs two matrices. The solution: the first matrix comes from `values`, the second from `Grouping::Tiled { b }`. This is consistent with the spec's "gather(Tiled)" pattern — the grouping defines how input is staged.

### Results

**Implementation changes to `accumulate.rs`**:
- `Grouping::Tiled { b: &'a [f64], m, n, k }` — now carries the second matrix
- `Op::DotProduct`, `Op::Distance` — dispatch to `TiledEngine::run(DotProductOp/DistanceOp)`
- `AccResult::Matrix { data, m, n }` — tiled output type
- `AccumulateEngine` now owns a `TiledEngine` alongside ScatterJit and ReduceOp

**Test: tiled_dot_product_2x2** — A(2×3) × B(3×2) → C(2×2), exact values verified.
**Test: tiled_dot_product_vector** — A(3×2) × B(2×1) → C(3×1), matrix-vector product.
**Test: tiled_distance_self** — 3 points in 2D, L2Sq distance matrix verified.
**Test: tiled_invalid_op_errors** — `Tiled + Op::Add` returns error (must use DotProduct or Distance).

All pass.

### Surprise?

The API design was cleaner than expected. The second matrix naturally fits in the grouping — it defines the partitioning structure (tile dimensions). The `values` parameter is always "the data being accumulated."

---

## Experiment 2: Logistic Regression via Pure Accumulate

### Before

**Hypothesis**: The complete logistic regression training loop (forward, sigmoid, residual, backward, update) can be expressed using ONLY `accumulate()` calls for the matrix operations, with CPU element-wise ops for sigmoid/residual/update.

**Connection to notebook 002**: Notebook 002 proved the gradient duality using direct TiledEngine calls. This experiment re-expresses the same loop through the unified `accumulate()` API.

### Results

**The training loop**:
```
for each iteration:
    // FORWARD: z = X_aug * β
    z = accumulate(X_aug, Tiled{b: β, m: n, n: 1, k: p}, Value, DotProduct)

    // MAP: sigmoid + residual (CPU, fuses in lazy pipeline)
    residual[i] = sigmoid(z[i]) - y[i]

    // BACKWARD: ∇ = X_aug_T * residual
    grad = accumulate(X_aug_T, Tiled{b: residual, m: p, n: 1, k: n}, Value, DotProduct)

    // UPDATE: β -= lr * ∇ / n (CPU)
    β[j] -= lr * grad[j] / n
```

**Two `accumulate()` calls per iteration. Same op (DotProduct), different tiled grouping (different matrix dimensions). The gradient duality IS the transpose of the tiled grouping parameters.**

Forward: `Tiled { b: β, m: n, n: 1, k: p }` — n×p times p×1
Backward: `Tiled { b: residual, m: p, n: 1, k: n }` — p×n times n×1

The test: 20-point 1D classification, 100 iterations, converges to >85% accuracy.

### Discussion

**The accumulate unification holds at three levels:**

1. **Spec level** (unification doc): "GEMM = accumulate(Tiled, a*b, Add)"
2. **API level** (accumulate.rs): Same `accumulate()` function handles All, ByKey, Masked, and Tiled
3. **Algorithm level** (this test): Complete gradient descent training loop = two accumulate calls + CPU element-wise

**Architecture table now:**

| Grouping | Status |
|----------|--------|
| All + Add | ✓ (scatter_phi with keys=0) |
| All + ArgMin/ArgMax | ✓ (ReduceOp) |
| ByKey + Add | ✓ (ScatterJit) |
| Masked + ByKey + Add | ✓ (scatter_phi_masked) |
| Tiled + DotProduct | ✓ (TiledEngine) |
| Tiled + Distance | ✓ (TiledEngine) |
| Prefix | todo (winrapids-scan) |
| Segmented | todo (winrapids-scan) |
| Windowed | todo (prefix subtraction) |

5 of 9 grouping patterns wired. The remaining 3 (Prefix, Segmented, Windowed) all need winrapids-scan integration.

163/163 tests pass (full crate).
