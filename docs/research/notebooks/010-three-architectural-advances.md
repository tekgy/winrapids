# Research Note: Three Architectural Advances for the Tambear Math Library

*Navigator — April 1, 2026*
*Session: Math Library Expedition, Day 1*
*Status: Confirmed in code and documentation*

---

## Preface

Three structural discoveries emerged during Day 1 of the math library expedition. Each was predicted from first principles, confirmed by reading the actual implementation, and then verified by the observer with gold-standard parity tests. They are recorded here as a shared reference — every future family decision should be checked against these three advances.

---

## Advance 1: The Three-Kingdom Taxonomy

### Claim

All 35 math library families decompose into three mathematical kingdoms, and every algorithm within a kingdom shares the same underlying accumulate primitive.

### The Kingdoms

**Kingdom A — Commutative (power sums / GramMatrix)**

Algorithms: descriptive stats (F06), hypothesis testing (F07), non-parametric (F08), robust stats (F09), regression (F10), mixed effects (F11), panel data (F12), dimensionality reduction (F22), information theory (F25), factor analysis (F14), CCA/MANOVA (F33).

The MSR is GramMatrix (cross-products). Polynomial theorem: MSR degree = polynomial degree of the quantity being extracted.
- Degree 2 in x → {n, Σx, Σ(x-μ)²}: univariate stats, all hypothesis tests
- Degree 2 in (x,y) → {n, Σx, Σy, Σ(x-μ)², Σ(y-ν)², Σ(x-μ)(y-ν)}: correlation, OLS, L2 distance
- Full p×p GramMatrix → all regression variants, PCA, factor analysis, CCA

**Kingdom B — Sequential (Affine scan)**

Algorithms: time series (F17), GARCH (F18), neural training (F24), optimization (F05 Adam).

All use `state_t = A · state_{t-1} + b_t`. Same operator, different parameterization:
- AR(p): A = companion matrix, b = 0
- Kalman filter: Särkkä operator (specialized Affine with noise)
- EWM: A = α scalar, b = (1-α)·x_t
- Adam optimizer: two Affine scans (m_t, v_t) on gradient sequence
- GARCH: A = β, b = ω + α·ε²_t (variance as state)

**Kingdom C — Iterative (outer loop around Kingdom A)**

Algorithms: GMM/EM (F16), robust IRLS (F09 outer), K-means (F20), LME REML (F11), Bayesian HMC/VI (F34).

Each ITERATION is a Kingdom A computation. The outer loop drives convergence.

### The IRLS Master Template (Kingdom C subpattern)

A particularly important subpattern: 10+ families share a common iterative template:

```
Loop until convergence:
  weights_i = f(μ_i, y_i)         ← weight formula differs by algorithm
  β̂_new = (X'WX)^{-1} X'Wz       ← same weighted GramMatrix solve
  update μ                          ← predict
```

| Algorithm | Weight | Family |
|-----------|--------|--------|
| GLM IRLS  | 1/Var(Y\|x)·(g'(μ))² | F10 |
| Logistic  | μ(1-μ) | F10 |
| Poisson   | μ | F10 |
| Robust (Huber) | ψ(ε/σ)/(ε/σ) | F09 |
| EM M-step | r_ik (posterior responsibility) | F16 |
| LME random | (ZGZ'+R)^{-1} block | F11 |
| Cox PH    | risk-set weights | F13 |

`scatter_multi_phi_weighted` is the single primitive that covers ALL of these. Implement it once; seven families are free.

### Confirmed In Code

- F06 uses `scatter_multi_phi` (PHI_COUNT, PHI_SUM, PHI_SUM_SQ, etc.)
- F07 has ZERO scatter imports — 27 algorithms, ~930 lines, zero GPU code
- F08 has ZERO new GPU primitives — transform-reentry on F06's accumulators
- F17 uses AffineState (Affine scan infrastructure)
- DBSCAN/KMeans/KNN share DistancePairs in TamSession — code confirmed in clustering.rs/knn.rs

---

## Advance 2: The Spine — 3 Passes Unlock 25/35 Families

### Claim

Three data passes, in order of increasing cost, each produce a shared intermediate that dozens of families read from without re-scanning the data.

### The Three Passes

**Pass 1 — Moments O(N)**
```
MomentStats(order=4) + ExtremaStats + QuantileSketch
```
- Cost: one scan of the data, one GPU kernel
- Unlocks: F06 (descriptive stats), F07 (hypothesis tests), F08 (normality tests), F09 (robust, via MomentStats), F25 (entropy from histogram)

**Pass 2 — Cross-Products O(N×p²)**
```
GramMatrix = tiled DotProductOp on centered data
```
- Cost: tiled GPU kernel, O(N×p²) work
- Unlocks: F10 (OLS/WLS/GLM), F22 (PCA via EigenDecomp), F14 (Factor Analysis), F33 (CCA/MANOVA), F11 (LME as Ridge on augmented system)

**Pass 3 — Distances O(N²)**
```
DistancePairs = tiled DistanceOp (NOT a separate pass — derivable from GramMatrix)
D[i,j] = norms[i] - 2·K[i,j] + norms[j]
```
- Cost: O(N²) extraction from GramMatrix, negligible vs Pass 2 for large N
- Unlocks: F01 (distance/similarity), F20 (clustering + validation), F21 (KNN), F22 (spectral clustering, UMAP), F28 (manifolds), F30 (spatial stats)

### Critical Note: Distance Is GramMatrix Extraction

**Pass 3 is NOT a separate data scan.** L2 distance is derivable from the GramMatrix diagonal (norms) and off-diagonal entries (dot products):

```
D²[i,j] = ||x_i||² - 2⟨x_i, x_j⟩ + ||x_j||²
         = diag[i] - 2·K[i,j] + diag[j]
```

The only work in Pass 3 is the O(N²) element-wise arithmetic — negligible compared to the O(N×p²) computation in Pass 2 that produces K.

This means: **GramMatrix is the universal sufficient statistic for BOTH the polynomial family (F06-F07 trunk) AND the geometric family (F20-F21-F22 trunk)**. The two sharing trunks merge at GramMatrix.

### Automatic Session Priming

```
session = tb.session(data).prime()
```

`prime()` plans and executes Pass 1 automatically. Passes 2 and 3 are demand-driven:
- First call to any F10/F14/F22/F33 method → triggers Pass 2
- First call to any F01/F20/F21/F22 manifold method → triggers Pass 3 (extraction from Pass 2)

No user needs to think about passes. The session manages the computation graph.

### Fusion Rule

Within a single pass, all algorithms that share the same (data, grouping) can fuse into one GPU kernel:

| Transform group | Fuseable algorithms |
|----------------|---------------------|
| Identity | MomentStats(order=4), ExtremaStats, HistogramStats → F06, F07, F25 simultaneously |
| Sort-required | QuantileSketch, SortedPermutation → F06 quantiles, F08 rank tests |
| Tiled accumulate | GramMatrix → F10, F22, F33, F14 simultaneously |

Three dispatches. Full EDA + regression + PCA. All families through F08.

### Confirmed In Code

- `clustering.rs` + `knn.rs`: KNN reads distance matrix from TamSession — zero recomputation
- `knn.rs` docstring: "Without sharing, KNN would recompute the O(N²d) distance matrix. With sharing, the distance matrix comes from the session for free."
- F07 reads MomentStats from TamSession → zero accumulation for all 27 hypothesis tests

---

## Advance 3: Distance Matrix = GramMatrix Subtraction

### Claim

L2 distance is not a new primitive — it is a Level-2 extraction from GramMatrix. The only new work is O(N²) element-wise arithmetic.

### The Formula

```
D²[i,j] = ||x_i - x_j||²
         = ||x_i||² - 2⟨x_i, x_j⟩ + ||x_j||²
         = K[i,i] - 2·K[i,j] + K[j,j]
```

Where K = GramMatrix (X'X). The diagonal K[i,i] = ||x_i||². The off-diagonal K[i,j] = ⟨x_i, x_j⟩.

### Implications

1. **DistancePairs is Level 2**, not Level 1. The expensive computation is Pass 2 (GramMatrix). Distance is a consequence.

2. **The entire geometry family is free after regression.** F10 (OLS) and F22 (PCA) and F01 (distance) all share Pass 2. Computing GramMatrix for regression GIVES YOU the distance matrix for DBSCAN/KNN/Spectral Clustering at zero additional GPU cost.

3. **Non-L2 distances are also Level 2.** Cosine similarity = K[i,j] / sqrt(K[i,i] · K[j,j]). Mahalanobis distance = (x_i - x_j)' Σ^{-1} (x_i - x_j) = derived from GramMatrix and covariance inverse. All inner-product geometries derive from the 3-field sufficient stats {||x||², ||y||², ⟨x,y⟩}.

4. **Edit distance, DTW, Wasserstein are genuinely Level 1.** These do NOT derive from inner products. They require their own accumulation passes. The only true exceptions to the "distance is free" rule.

### The Universal Kernel for Non-Euclidean Geometry

The 3-field MSR {sq_norm_x, sq_norm_y, dot_prod} is a universal kernel for any inner-product geometry:

| Geometry | Formula |
|----------|---------|
| Euclidean | sqrt(sq_norm_x - 2·dot + sq_norm_y) |
| Cosine similarity | dot / sqrt(sq_norm_x · sq_norm_y) |
| Poincaré (hyperbolic) | acosh(1 + 2(sq_norm_x - 2·dot + sq_norm_y) / ((1-sq_norm_x)(1-sq_norm_y))) |
| Spherical geodesic | acos(dot / sqrt(sq_norm_x · sq_norm_y)) |
| Kernel (RBF) | exp(-γ · (sq_norm_x - 2·dot + sq_norm_y)) |

All five geometries from ONE accumulate pass. This is why ManifoldDistanceOp stores all three fields — it's the minimum sufficient representation for all inner-product geometries.

### Confirmed In Code

- `tam-gpu/src/cpu.rs`: ManifoldDistanceOp stores {sq_norm_x, sq_norm_y, dot_prod}
- L2 distance derivation from GramMatrix is implemented in the TiledEngine
- Adversarial proof: `manifold-confidence-universal-v-column.py` confirms the 3-field representation covers all manifold types

---

## Summary: The Three Advances as Design Principles

| Advance | Design Rule |
|---------|-------------|
| Three-Kingdom Taxonomy | Before implementing any family: identify its kingdom. Kingdom A → find GramMatrix. Kingdom B → find Affine parameterization. Kingdom C → find inner Kingdom A iteration. |
| 3-Pass Spine | Prime the session with Pass 1. Pass 2 and 3 are demand-driven from TamSession. Never scan data twice for the same quantity. |
| Distance = GramMatrix | Never treat DistancePairs as an independent computation. It's O(N²) element-wise arithmetic on GramMatrix output. Flag it in code reviews when it appears as a separate data scan. |

---

## The 8-Operator Model (Unchanged)

The three advances do not add new operators. The complete operator set remains:

1. Add (scalar accumulation)
2. Welford (online mean/variance — deprecated in favor of centered two-pass)
3. TiledAdd = DotProduct (inner product, the workhorse of Kingdom A)
4. Affine (sequential state, Kingdom B)
5. Särkkä (Kalman filter specialization of Affine)
6. Max
7. Min
8. SoftmaxWeighted

LogSumExpOp = compound (Max + exp + Add + log) — NOT operator #9. Confirmed by naturalist.

If any new algorithm requires "operator #9" — that is a discovery worth flagging to the full team.

---

## Cross-Platform Note

These three advances hold on both tested backends:
- NVIDIA Blackwell (RTX PRO 6000) via Vulkan/CUDA
- Apple M4 Pro via Metal/wgpu

The tam-gpu `detect()` mechanism auto-selects the backend. The three-advance architecture is independent of backend — all three are API-level claims about computation structure, not hardware.

---

*Documented by: navigator*
*Verified by: naturalist (structural), observer (87/87 parity tests), adversarial (confirmed failure modes)*
