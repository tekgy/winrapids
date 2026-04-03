# Scale Ladder: Numerical Stability Analysis

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Reference document for benchmarks and Paper 6

---

## Overview

This document maps where numerical stability degrades across the scale ladder (n=10 to n=1T), where f32 breaks, where algorithm complexity crosses over, and where condition numbers blow up. These are the mathematical boundaries that the benchmark suite must probe and that Paper 6 must prove.

---

## 1. Catastrophic Cancellation Boundaries (The Destruction Gradient)

### The Core Problem

The naive variance formula `E[x²] - E[x]²` suffers from catastrophic cancellation when the mean is large relative to the standard deviation. Both terms are O(μ²), but their difference is O(σ²). When μ/σ is large, significant digits are lost.

### Measured Boundaries (from canary tests in gold_standard_parity.rs)

| Offset (μ) | Spread (σ) | μ/σ Ratio | f64 Naive Rel Error | f64 Centered Rel Error | f32 Status |
|-------------|------------|-----------|---------------------|------------------------|------------|
| 0 | 0.029 | 0 | 0 | 0 | OK |
| 1e4 | 0.029 | 3.5e5 | ~1e-6 | < 1e-10 | Marginal |
| 1e6 | 0.029 | 3.5e7 | ~1e-2 | < 1e-10 | **BROKEN** |
| 1e8 | 0.029 | 3.5e9 | **NEGATIVE** | < 1e-6 | **BROKEN** |
| 1e10 | 0.029 | 3.5e11 | **NEGATIVE** | ~1e-5 | N/A |
| 1e12 | 0.029 | 3.5e13 | **NEGATIVE** | ~1e-4 | N/A |
| 1e14 | 0.029 | 3.5e15 | **NEGATIVE** | ~1e-2 | N/A |

### Theoretical Error Analysis

**Naive formula condition number**: κ(naive) = μ²/σ² × ε_mach

For f64 (ε_mach ≈ 1.1e-16):
- Offset 1e8: κ ≈ (1e8)²/(0.029)² × 1.1e-16 ≈ 1.3e4 relative error digits lost → ~12 good digits remain
- Offset 1e12: κ ≈ (1e12)²/(0.029)² × 1.1e-16 ≈ 1.3e12 → ~4 good digits remain
- Offset 1e14: κ ≈ (1e14)²/(0.029)² × 1.1e-16 ≈ 1.3e16 → **0 good digits** (total loss)

For f32 (ε_mach ≈ 5.96e-8):
- Offset 1e4: κ ≈ (1e4)²/(0.029)² × 5.96e-8 ≈ 70 → **~5 good digits** (marginal for 7-digit f32)
- Offset 1e6: κ ≈ (1e6)²/(0.029)² × 5.96e-8 ≈ 7e5 → **TOTAL LOSS**

**Centered formula condition number**: κ(centered) = O(1) regardless of offset.
- Deviations from mean are O(σ), so the working values never exceed O(σ²)
- Centering reduces the condition number from O(μ²/σ²) to O(1)
- Degradation at extreme offsets (1e12+) comes from the mean computation itself losing precision

### f32 Boundary Summary

| Statistic | f32 Safe Through | f32 Breaks At | Mechanism |
|-----------|-----------------|---------------|-----------|
| Mean | 1e7 (centered sum) | 1e8+ (sum overflow for large n) | Sum exceeds 2²⁴ mantissa |
| Variance (naive) | **1e3** | 1e4 | μ²/σ² cancellation |
| Variance (centered) | 1e6 | 1e7 | Mean precision limits centering |
| Skewness | **1e2** | 1e3 | Higher moment amplifies cancellation |
| Kurtosis | **1e2** | 1e3 | Fourth power amplifies worse |
| Poincaré distance | 0.99 (curvature) | r → 1 | Conformal factor 4/(1-r²)² |
| Cosine similarity | Orthogonal vectors | Near-parallel vectors | 1 - dot ≈ 0 |
| SVD (one-sided Jacobi) | κ ≈ 1e3 | κ > 1e4 | 7 digits insufficient |

**Design rule**: f32 is ONLY safe for:
1. Element-wise operations (no accumulation)
2. Accumulations where μ/σ < 1e3
3. Storage format (compute in f64, store in f32)

---

## 2. Scale-Dependent Algorithm Complexity Crossovers

### The GPU Complexity Landscape

| Algorithm Class | Complexity | n at 1ms (GPU) | n at 1s (GPU) | Crossover |
|----------------|-----------|----------------|---------------|-----------|
| Element-wise (fused_expr) | O(n) | ~1e9 | ~1e12 | Never — always fastest |
| Reduce (All) | O(n) | ~1e9 | ~1e12 | Never |
| Scan (Prefix) | O(n) | ~5e8 | ~5e11 | Never |
| Hash scatter (ByKey) | O(n) | ~2e8 | ~2e11 | Beats sort-based at all n |
| Sort-based groupby | O(n log n) | ~1e7 | ~1e10 | 17x slower than scatter |
| Distance matrix | O(n²d) | ~3e4 (d=10) | ~1e6 (d=10) | Limits clustering to n < 1e6 |
| SVD (one-sided Jacobi) | O(mn²) | n < 1e3 | n < 1e4 | CPU-bound, small matrices |
| Cholesky | O(n³/3) | n < 500 | n < 5000 | Faster than SVD for SPD |
| Interpolation (Lagrange) | O(n²) | n < 1e4 | n < 1e5 | Newton preferred for n > 100 |
| Tridiagonal solve | O(n) | ~1e8 | ~1e11 | Always preferred for banded |

### Hash Scatter Variant Crossovers (from scatter_bench.rs)

Three GPU kernel variants with different contention characteristics:

| Variant | Best When | Mechanism |
|---------|-----------|-----------|
| Naive (3 global atomicAdd) | n_groups > 1024 | Low contention, no overhead |
| Shared-memory | 64 < n_groups < 1024 | Block-local privatization |
| Warp-aggregated | n_groups < 64 | `__match_any_sync` coalesces atomics |

The crossover depends on n_groups/n_warps ratio. At high contention (n/n_groups >> 32), warp aggregation wins because it reduces atomic traffic by ~32x.

### Distance Matrix as Scale Limiter

The O(n²d) distance matrix is the primary scale bottleneck for:
- DBSCAN clustering
- KNN graph construction
- Manifold distance computation

**At n=1e6, d=10**: n² = 1e12 distances × 10 FLOPs/distance = 1e13 FLOPs → ~10s on RTX 6000 Pro
**At n=1e7**: n² = 1e14 → ~1000s → impractical without approximation

**Architectural solution**: Content-addressed intermediate sharing. Compute distance matrix once, share across DBSCAN/KNN/manifold. This doesn't reduce O(n²d) but avoids multiplying it by algorithm count.

**Algorithmic solution for n > 1e6**: Approximate nearest neighbors (HNSW, vantage-point trees) — reduces O(n²d) to O(n·d·log n). Not yet implemented; belongs in future work.

---

## 3. Condition Number Scaling

### SVD Condition Number Sensitivity

The key insight from linear_algebra.rs: forming A^T A squares the condition number (κ → κ²).

| Matrix Type | Typical κ(A) | κ(A^T A) | f64 Digits Lost (AᵀA) | f64 Digits Lost (Jacobi) |
|-------------|-------------|----------|----------------------|------------------------|
| Well-conditioned | 1-10 | 1-100 | 0-2 | 0-1 |
| Moderate | 1e3 | 1e6 | 6 | 3 |
| Hilbert 4×4 | 1.55e4 | 2.4e8 | 8 | 4 |
| Hilbert 6×6 | >1e6 | >1e12 | 12 | 6 |
| Ill-conditioned | 1e8 | 1e16 | **TOTAL LOSS** | 8 |

**Design rule**: One-sided Jacobi SVD is mandatory for any matrix where κ might exceed 1e4. The A^T A approach is NEVER safe because you can't know κ in advance.

### Poincaré Conformal Factor

For points near the boundary of the Poincaré ball (||x|| → 1/√κ):

```
conformal_factor = 4 / (1 - κ·||x||²)²
```

| ||x||² (κ=1) | Conformal Factor | Rounding Amplification | f64 Safe? | f32 Safe? |
|--------------|-----------------|----------------------|-----------|-----------|
| 0.0 | 4 | 1x | Yes | Yes |
| 0.5 | 16 | 4x | Yes | Yes |
| 0.9 | 400 | 100x | Yes | Marginal |
| 0.99 | 40,000 | 10,000x | Yes | **No** |
| 0.999 | 4,000,000 | 1,000,000x | Marginal | **No** |
| 0.9999 | 4e8 | 1e8x | **Degraded** | **No** |

**Design rule**: Poincaré geometry in f32 is limited to ||x|| < 0.9. For embeddings near the boundary (which is where hyperbolic geometry is most useful), f64 is required.

### Cholesky Conditioning

For the normal equations X^T X in linear regression:

| Design Matrix Property | κ(X^T X) | Effect |
|-----------------------|----------|--------|
| Orthogonal columns | 1 | Perfect |
| Moderate correlation | 1e2-1e4 | Safe |
| High collinearity | 1e6+ | Cholesky may produce negative diagonal |
| Near-singular | 1e12+ | **Use SVD or QR instead** |

The adversarial tests (svd_adversarial.rs) verify that Hilbert matrices up to 6×6 produce accurate SVD. For larger Hilbert matrices, even f64 SVD degrades — Hilbert 12×12 has κ ≈ 1.6e16 ≈ 1/ε_mach.

---

## 4. Higher Moments: The Binomial Transform Instability

Central moments from raw power sums use the binomial transform:

```
μₖ = Σⱼ₌₀ᵏ C(k,j)·(-1)^(k-j)·m'ⱼ·μ^(k-j)
```

The alternating signs cause catastrophic cancellation that worsens with k:

| Moment Order k | # Terms | Max Binomial Coeff | Cancellation Severity | f64 Safe Through Offset | f32 Safe Through |
|---------------|---------|-------------------|---------------------|------------------------|-----------------|
| 2 (variance) | 2 | 1 | Moderate | 1e14 (centered) | 1e6 (centered) |
| 3 (skewness) | 3 | 3 | High | 1e10 | 1e3 |
| 4 (kurtosis) | 4 | 6 | Very high | 1e8 | **Never safe** |
| 5 | 5 | 10 | Extreme | 1e6 | **Never safe** |
| 6 | 6 | 20 | Extreme | 1e4 | **Never safe** |
| 7+ | 7+ | 35+ | **Unusable from raw sums** | — | — |

**Design rule**: For k ≥ 5, the binomial transform from raw power sums is essentially unusable. Use Pebay's parallel combining algorithm (which works with centered moments directly) or multi-pass centering.

**Implication for MSR**: The polynomial degree theorem (Paper 2, Section 4) has a hidden asterisk: the MSR {n, S₁, ..., Sₖ} is _algebraically_ sufficient but _numerically_ insufficient for k ≥ 5 in f32 and k ≥ 7 in f64. The _numerically sufficient_ MSR is {n, mean, M₂, M₃, M₄, ...} (centered moments via Pebay), not raw power sums.

---

## 5. Scale Ladder: n-Dependent Failure Modes

### Small n (n < 30)

- **Kurtosis G₂**: denominator (n-2)(n-3) → divide-by-zero at n ≤ 3, wild swings at n = 4-10
- **Skewness G₁**: denominator (n-2) → undefined at n = 2
- **L-moments**: PWM estimates noisy for n < 20
- **Permutation entropy**: needs n >> m! (m = embedding dimension). For m = 5: n >> 120
- **DFA**: needs at least 4 box sizes. For box range [4, n/4], need n ≥ 16
- **Correlation dimension**: needs n >> 2^d (embedding dimension). Typically n > 1000

### Medium n (1e3 < n < 1e6)

- **All moment-based stats**: numerically stable with centering
- **Distance matrices**: O(n²) feasible (1e6² = 1e12 entries manageable with streaming)
- **SVD**: feasible for matrices up to ~1000×1000
- **Clustering**: DBSCAN/KNN practical
- **Sort-based stats**: O(n log n) fast (< 1ms for n = 1e6)

### Large n (1e6 < n < 1e9)

- **Distance matrix**: IMPRACTICAL without approximation. n=1e7 → 1e14 entries
- **Sort**: GPU radix sort still feasible (1e9 keys in ~1s)
- **Accumulation**: O(n) operations remain < 1s
- **f32 sum overflow**: Sum of 1e9 values at ~100 each = ~1e11 > 2²⁴ (f32 integer precision) → **sum loses precision**. Use Kahan compensated summation or f64.

### Very large n (n > 1e9)

- **f64 sum precision**: Sum of 1e12 values accumulates O(√n) × ε_mach relative error with pairwise summation, O(n) × ε_mach with naive summation
  - At n = 1e12: naive sum has ~1e-4 relative error; pairwise has ~1e-10
  - Kahan compensated: ~ε_mach regardless of n
- **Integer overflow**: n > 2³¹ (2.1e9) overflows i32 indices. Use i64/u64.
- **Memory**: n=1e9 × 8 bytes = 8GB per column. Multiple columns → multi-GPU territory.
- **Two-pass algorithms**: Two passes over 1TB+ data may hit I/O bottleneck before compute bottleneck

---

## 6. Recommendations

### For Benchmark Suite

1. **Variance canary at every scale**: n ∈ {100, 1e4, 1e6, 1e8, 1e10, 1e12} with offset = n (financial prices at that scale)
2. **f32 vs f64 crossover**: Find exact n where f32 accumulation diverges from f64 for each statistic
3. **Distance matrix wall**: Measure actual throughput at n = {1e3, 1e4, 1e5, 1e6, 3e6} to find practical limit
4. **Hash scatter variant selection**: Automate kernel selection based on n_groups/n_warps ratio

### For Paper 6 (Numerical Stability by Construction)

1. **Prove**: κ(naive) = O(μ²/σ²), κ(centered) = O(1) — formal condition number analysis
2. **Prove**: Pebay merge preserves O(1) condition number across arbitrary partition merges
3. **Show**: The destruction gradient table as the paper's headline result
4. **Cite**: Higham (2002) "Accuracy and Stability of Numerical Algorithms" for framework; Pebay (2008) for parallel combining

### For Paper 2 (MSR)

1. **Distinguish**: algebraic MSR (raw power sums) from numerical MSR (centered moments)
2. **State**: f32 limits the practical MSR to degree ≤ 2 (variance); f64 to degree ≤ 4 (kurtosis)
3. **Note**: The MSR principle is exact in infinite precision; in finite precision, the "minimum" may need more fields for numerical stability (e.g., RefCentered carries 3 fields for what Add needs 2)

---

## References

- Higham, N.J. (2002). Accuracy and Stability of Numerical Algorithms. SIAM.
- Pebay, P. (2008). Formulas for robust, one-pass parallel computation of covariances and arbitrary-order statistical moments. Sandia Technical Report.
- Welford, B.P. (1962). Note on a method for calculating corrected sums of squares and products. Technometrics.
- Kahan, W. (1965). Pracniques: Further remarks on reducing truncation errors. CACM.
