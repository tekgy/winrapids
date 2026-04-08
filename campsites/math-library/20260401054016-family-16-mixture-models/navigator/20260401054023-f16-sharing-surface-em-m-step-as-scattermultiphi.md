# F16 Sharing Surface: EM M-Step as scatter_multi_phi

Created: 2026-04-01T05:40:23-05:00
By: navigator

Prerequisite: F06 complete (MomentStats, scatter_multi_phi working).

---

## The Core Insight

**The EM M-step IS scatter_multi_phi — generalized to fractional weights.**

This is not an approximation — it's an identity. Every GMM/EM M-step is a grouped variance computation with soft (fractional) group assignments. We already have `scatter_multi_phi`. Family 16 needs one small extension to that primitive.

---

## EM Algorithm Structure for GMM

### E-step (responsibility computation):

```
r_nk = π_k * N(x_n | μ_k, Σ_k) / Σ_j π_j * N(x_n | μ_j, Σ_j)
```

`r_nk` = soft assignment of point n to cluster k. Row sum over k = 1.0 per point.

**Tambear path**: evaluate Gaussian log-density per (point, cluster), apply log-sum-exp row normalize.

For spherical/diagonal Σ: `log N(x_n | μ_k, σ_k²) = -0.5 * Σ_d ((x_nd - μ_kd)² / σ_kd²) + const`. This IS a distance computation. For spherical Σ (isotropic): proportional to L2 distance squared.

**Connection to F01**: E-step Gaussian evaluation IS the RBF kernel. Once F01 provides `KernelMatrix(RBF)`, E-step becomes free for spherical Σ.

---

### M-step (parameter update):

```
N_k = Σ_n r_nk                           // effective count per cluster
μ_k = (Σ_n r_nk * x_n) / N_k            // weighted mean
σ²_k = (Σ_n r_nk * (x_n - μ_k)²) / N_k // weighted variance (diagonal case)
π_k = N_k / N                             // mixing weights
```

**This is exactly scatter_multi_phi with fractional weights:**

```rust
// Current scatter_multi_phi: keys[n]: u32 → integer group ID, weight = 1.0
// Extended: weights[n,k]: f64 → fractional contribution to each cluster

let results = jit.scatter_multi_phi_weighted(
    &["1.0", "v - r", "(v - r) * (v - r)"],  // count, centered_sum, centered_sq
    weights,      // shape (N, K) soft assignment matrix (r_nk)
    values_d,     // x[:,d] — one call per dimension
    ref_values,   // current μ_k per cluster (centering)
    n_clusters,
)?;
// results[0][k] = N_k  (effective count)
// results[1][k] = Σ_n r_nk * (x_nd - μ_kd)  (centered weighted sum)
// results[2][k] = Σ_n r_nk * (x_nd - μ_kd)² (weighted variance numerator)
```

The scatter becomes: `acc[k] += weights[n,k] * phi(v[n], ref[k])`. This is a 10-20 line change to ScatterJit — replace `keys[n]: u32 → index` with `weights[n,:]: f64 → contribution`.

**Do NOT take Path B (discretize to hard assignments)** — that's KMeans, not GMM. The soft weights are the point.

---

## MSR Types

### New type needed: `SoftAssignment`

```rust
pub struct SoftAssignment {
    pub n_points: usize,
    pub n_clusters: usize,
    /// r[n*n_clusters + k] = probability of point n belonging to cluster k.
    /// Row sums = 1.0 (after normalization).
    pub responsibilities: Arc<Vec<f64>>,
}
```

Add to `IntermediateTag`:
```rust
SoftAssignment {
    data_id: DataId,
    n_clusters: usize,
    iteration: u32,  // EM iteration — consumers can check freshness
},
```

### Reused types:
- `MomentStats` per cluster (same as F06 — μ_k and σ²_k are MomentStats extracted)
- `ClusterAssignment` (Level-2: hard assignments = argmax_k r_nk per point)

---

## New Primitive Needed: `LogSumExpOp`

Both E-step normalization and log-likelihood require log-sum-exp over k.

Add to `winrapids-tiled/src/ops.rs`:

```rust
/// Row-wise log-sum-exp: log(Σ_k exp(x_k)) per row. Used for E-step and log-likelihood.
pub struct LogSumExpOp;
```

- `cuda_identity()`: `-f64::INFINITY`
- Accumulate: numerically stable two-pass. Pass 1: find max per row. Pass 2: sum exp(x - max).
- Output: `max + log(sum)` per row.

LogSumExpOp is valuable across the library:
- F16: E-step normalization, log-likelihood
- F23: softmax (= exp(x - log_sum_exp(x)))
- F17: Baum-Welch forward-backward for HMMs
- F21: CRF training

Worth implementing as a first-class TiledOp now.

---

## Covariance Variants

**Diagonal Σ** (one σ²_kd per cluster per dimension): `scatter_multi_phi_weighted` per dimension independently. One pass per dimension. This is Phase 1.

**Full Σ** (d×d covariance matrix per cluster): weighted outer product — `Σ_n r_nk * (x_n - μ_k)(x_n - μ_k)ᵀ`. This is a weighted GramMatrix per cluster. O(N × K × d²). Phase 2.

**Tied Σ** (shared covariance): sum N_k * Σ_k over k, normalize. One extra reduction. Phase 2.

---

## Sharing Tree for F16

**What F16 produces** → who consumes:
| MSR | Consumer |
|-----|---------|
| `SoftAssignment` (r_nk) | F20 (soft silhouette), F34 (posterior) |
| `MomentStats` per cluster | F06 (reuses), F20 (inter-cluster distances) |
| π_k (mixing weights) | F21 (generative classifier) |
| Log-likelihood trajectory | Convergence monitoring, F34 Bayesian model comparison |

**What F16 consumes from TamSession**:
1. `MomentStats(order=2, ByKey)` — K-means init starting point (from F20 or F06)
2. `DistancePairs` — for initialization via K-means (F20)
3. `KernelMatrix(RBF)` — E-step Gaussian eval (once F01 provides)

---

## Build Order

1. **`scatter_multi_phi_weighted`** in `scatter_jit.rs` — extend existing primitive (~20 lines)
2. **`LogSumExpOp`** in `winrapids-tiled/src/ops.rs` (~30 lines)
3. **`SoftAssignment` MSR type** in `intermediates.rs` (~15 lines)
4. **GMM E-step**: evaluate log-density per (n,k), LogSumExp normalize → SoftAssignment (~80 lines)
5. **GMM M-step**: `scatter_multi_phi_weighted` → update μ_k, σ²_k, π_k (~50 lines)
6. **EM loop**: iterate E→M until Δlog(L) < 1e-6 (~40 lines)
7. **Initialization**: K-means warm start → hard-to-soft r_nk (~20 lines)
8. **Tests**: R `mclust` package as gold standard (~100 lines)

Total: ~360 lines for diagonal GMM. With the infrastructure, the math is straightforward.

---

## The Structural Insight for Lab Notebook

> F06 (grouped variance) and F16 (GMM M-step) are the SAME scatter_multi_phi, generalized from integer group IDs to fractional responsibility weights. Hard group membership is a special case of soft group membership where r_nk ∈ {0,1}. One JIT kernel extension handles both — and the connection reveals that "clustering" and "statistics per group" are not separate algorithms but different parameterizations of the same accumulate operation.

**Publishable framing**: "From hard grouping to soft: one accumulate primitive for descriptive statistics and EM."

---

## Implementation Estimate

With scatter_multi_phi_weighted built: F16 (diagonal GMM) = ~360 lines.
Compare to naive GMM with bespoke CUDA kernels: 600-900 lines. The sharing infrastructure wins.
