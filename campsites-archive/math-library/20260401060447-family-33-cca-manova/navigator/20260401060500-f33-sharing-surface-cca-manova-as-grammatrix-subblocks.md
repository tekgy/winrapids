# F33 Sharing Surface: CCA/MANOVA as GramMatrix Subblocks

Created: 2026-04-01T06:05:00-05:00
By: navigator

Prerequisites: F10 complete (GramMatrix), F22 complete (EigenDecomposition).

---

## Core Insight: The Joint GramMatrix Contains Everything

CCA and MANOVA both need the cross-covariance structure between variable sets. This is a subblock
of the joint GramMatrix — not a new computation.

Stack X (N×p) and Y (N×q) horizontally into Z = [X | Y] (N×(p+q)).
Compute GramMatrix(Z): a (p+q)×(p+q) matrix.

**Read off the subblocks**:
```
GramMatrix(Z) = [ Σ_xx  Σ_xy ]
                [ Σ_yx  Σ_yy ]

Σ_xx = X'X   (p×p cross-products, same as GramMatrix(X))
Σ_yy = Y'Y   (q×q cross-products)
Σ_xy = X'Y   (p×q cross-products — the key new quantity)
```

One GramMatrix(Z) call gives all three subblocks for FREE.

---

## Canonical Correlation Analysis (CCA)

**What CCA computes**: linear projections u = Xa and v = Yb that maximize cor(u, v).

### Mathematical Reduction

```
Maximize: cor(Xa, Yb) = a'Σ_xy b / √(a'Σ_xx a · b'Σ_yy b)
```

This is a generalized eigenvalue problem. The canonical correlations ρ and canonical weights are:
```
Σ_xx^{-1} Σ_xy Σ_yy^{-1} Σ_yx a = ρ² a
```

Or equivalently (symmetric form after whitening):
```
M = Σ_xx^{-1/2} Σ_xy Σ_yy^{-1/2}
SVD(M) = U Σ V'
```

Canonical correlations = singular values of M.
Canonical weights: a_k = Σ_xx^{-1/2} u_k, b_k = Σ_yy^{-1/2} v_k.

**CCA = SVD of whitened cross-covariance matrix.**

### Tambear Decomposition

| Step | Primitive | Reuse |
|------|-----------|-------|
| Σ_xx, Σ_xy, Σ_yy | GramMatrix([X\|Y]) | F10 existing |
| Σ_xx^{-1/2} | Cholesky(Σ_xx), triangular solve | F10 existing |
| Σ_yy^{-1/2} | Cholesky(Σ_yy), triangular solve | F10 existing |
| M = Σ_xx^{-1/2} Σ_xy Σ_yy^{-1/2} | Matrix multiply | DotProductOp |
| SVD(M) | Thin SVD (p×q matrix) | F22 EigenDecomp infrastructure |
| Canonical scores | X · a_k, Y · b_k | gather/project |

Total new code: matrix multiply (trivially DotProductOp) + thin SVD (F22 infrastructure).
Most of it is wiring existing primitives, not new algorithms.

### CCA MSR Type

```rust
pub struct CcaModel {
    pub n_obs: usize,        // N
    pub n_x: usize,          // p
    pub n_y: usize,          // q
    pub n_components: usize, // k = min(p, q)

    /// Canonical correlations ρ₁ ≥ ρ₂ ≥ ... ≥ ρ_k
    pub canonical_correlations: Vec<f64>,   // shape (k,)

    /// X canonical weights A: columns are a_1, ..., a_k. Shape (p, k).
    pub x_weights: Vec<f64>,

    /// Y canonical weights B: columns are b_1, ..., b_k. Shape (q, k).
    pub y_weights: Vec<f64>,

    /// X canonical scores = X A. Shape (N, k).
    pub x_scores: Arc<Vec<f64>>,

    /// Y canonical scores = Y B. Shape (N, k).
    pub y_scores: Arc<Vec<f64>>,

    /// Wilks' lambda and p-values for each canonical variate.
    pub wilks_lambda: Vec<f64>,
    pub p_values: Vec<f64>,
}
```

---

## MANOVA: Multivariate ANOVA

**What MANOVA tests**: whether group mean VECTORS differ across multiple outcomes simultaneously.
Univariate ANOVA tests one outcome; MANOVA tests p outcomes jointly.

### Mathematical Reduction

MANOVA generalizes the F-test to matrices:
```
H = between-group scatter matrix (p×p)    // multivariate analog of SS_between
E = within-group scatter matrix (p×p)     // multivariate analog of SS_within
```

Four MANOVA test statistics (all functions of eigenvalues of HE^{-1}):
```
Wilks' Λ     = |E| / |H+E| = ∏ (1 - ρ²_k)       // most common
Pillai's V   = Σ ρ²_k / (1+ρ²_k)                  // most robust
Hotelling-Lawley T = Σ ρ²_k / (1-ρ²_k)            // most powerful under normality
Roy's max root = ρ²₁ / (1-ρ²₁)                    // most powerful vs specific alternatives
```

Where ρ_k are eigenvalues of (H+E)^{-1}H (= canonical correlations between Y and group indicators).

### Tambear Decomposition

MANOVA IS multivariate-output regression with a categorical predictor.

**H (between-group scatter)**:
```
H = Σ_k n_k (ȳ_k - ȳ)(ȳ_k - ȳ)'
  = scatter_multi_phi grouped-by-k, phi = "(v - r)^T(v - r)" (outer product)
```

This is RefCenteredStats extended to multivariate output — scatter of outer products.

**E (within-group scatter)**:
```
E = Σ_k Σ_{i in k} (y_i - ȳ_k)(y_i - ȳ_k)'
  = scatter_multi_phi grouped-by-k, phi = "(v - group_mean)^T(v - group_mean)"
```

Both H and E are symmetric p×p matrices built from grouped outer products.

**New primitive needed**: `scatter_outer_product(keys, values_matrix)` — accumulates outer products
into a p×p symmetric matrix per group. This is O(N·p²) — same complexity as GramMatrix.

Alternatively: note that H + E = total scatter = GramMatrix(Y_centered). And E = GramMatrix with
within-group centering. So:
```
E = GramMatrix(Y - group_means)        // F10 with group-centered Y
H = GramMatrix(Y) - E
```

Using GramMatrix infrastructure: subtract centered versions.

**Test statistics**:
```
eigenvalues of HE^{-1}     // solve generalized eigenvalue problem
→ F22 EigenDecomposition infrastructure
```

### MANOVA MSR Type

```rust
pub struct ManovaResult {
    pub n_groups: usize,
    pub n_variables: usize,   // p (number of outcome variables)
    pub n_obs: usize,

    pub h_matrix: Vec<f64>,   // p×p between-group scatter
    pub e_matrix: Vec<f64>,   // p×p within-group scatter

    /// Eigenvalues of HE^{-1}: the canonical discriminant correlations.
    pub eigenvalues: Vec<f64>,

    /// Test statistics:
    pub wilks_lambda: f64,
    pub pillai_trace: f64,
    pub hotelling_lawley: f64,
    pub roy_max_root: f64,

    /// Approximate F and p-values for each test statistic.
    pub wilks_f: f64,
    pub wilks_p: f64,
    pub pillai_f: f64,
    pub pillai_p: f64,
}
```

---

## SEM / CFA Stub (Phase 2)

Full Confirmatory Factor Analysis / Structural Equation Modeling is a larger undertaking:

```
Minimize: FML = log|Σ(θ)| + tr(S Σ^{-1}(θ)) - log|S| - p
where:
  S = GramMatrix / (N-1)     (observed covariance, from F10)
  Σ(θ) = Λ Φ Λ' + Θ         (model-implied covariance)
  θ = (Λ, Φ, Θ)             (loadings, factor correlations, uniquenesses)
```

SEM = optimize FML over θ using F05 (L-BFGS + CfaOracle).

**Phase 1 scope for F33**: CCA + MANOVA only. SEM deferred to Phase 2.
**What SEM needs from F33**: GramMatrix subblocks (Σ), Cholesky for |Σ(θ)| and tr(SΣ^{-1}(θ)).
**What SEM adds**: model-implied covariance parameterization, fit indices (RMSEA, CFI, TLI, SRMR).

---

## MSR Sharing Map

| Step | F10 | F22 | F33 | New? |
|------|-----|-----|-----|------|
| GramMatrix | ✓ | ✓ (uses F10) | ✓ (uses F10) | — |
| Cholesky | ✓ | — | ✓ (whitening) | — |
| EigenDecomposition | — | ✓ | ✓ (of HE^{-1}) | — |
| SVD of small matrix | — | ✓ | ✓ (CCA) | — |
| Group scatter | F06 (ByKey) | — | ✓ H and E matrices | extend to outer products |
| Canonical scores | — | ✓ (projections) | ✓ (CCA scores) | — |

**F33 adds zero new accumulate primitives for MANOVA** if we use the GramMatrix approach.
**CCA adds one**: thin SVD of a p×q matrix (p,q < 50 typically). This reuses F22's SVD infrastructure.

---

## Build Order

1. **GramMatrix([X|Y]) subblock extraction** — index arithmetic to extract Σ_xx, Σ_xy, Σ_yy
   from GramMatrix of joint matrix (~10 lines)
2. **CCA whitening** — Cholesky(Σ_xx) + triangular solve to get Σ_xx^{-1/2} (~use F10's Cholesky)
3. **Thin SVD of M** — reuse F22's EigenDecomp via SVD route for small matrix (~30 lines)
4. **CcaModel struct** + canonical scores computation (~40 lines)
5. **Wilks' Λ test** — Rao's approximation F-statistic (~20 lines)
6. **MANOVA H and E matrices** — grouped GramMatrix (within-group centered) (~40 lines)
7. **Eigenvalues of HE^{-1}** — solve generalized eigenvalue, reuse F22 (~20 lines)
8. **MANOVA test statistics + approximate F** — four statistics (~50 lines)
9. **ManovaResult struct** (~30 lines)
10. **Tests**: R's `cancor()` for CCA, `manova()` for MANOVA. Python: `sklearn.cross_decomposition.CCA`.

**Total new code: ~250 lines.** Nearly free after F10 + F22.

---

## Gold Standards

### CCA
```r
# R canonical (cleanest output):
cc_result <- cancor(X, Y)
cc_result$cor          # canonical correlations
cc_result$xcoef        # X canonical weights
cc_result$ycoef        # Y canonical weights

# CCA F-test (Wilks' lambda):
library(CCP)
p.asym(cc_result$cor, n=nrow(X), p=ncol(X), q=ncol(Y), tstat="Wilks")
```

```python
from sklearn.cross_decomposition import CCA
cca = CCA(n_components=k)
cca.fit(X, Y)
x_scores, y_scores = cca.transform(X, Y)
```

### MANOVA
```r
# R canonical (lavaan-like):
fit <- manova(cbind(y1, y2, y3) ~ group, data=df)
summary(fit, test="Wilks")   # Wilks' Λ
summary(fit, test="Pillai")  # Pillai's trace
summary(fit, test="Hotelling-Lawley")
summary(fit, test="Roy")     # Roy's max root
```

```python
from statsmodels.multivariate.manova import MANOVA
mv = MANOVA.from_formula('y1 + y2 + y3 ~ group', data=df)
mv.mv_test()   # all four test statistics
```

**Match target**: R's `manova()` Wilks' Λ to 6 decimal places; `cancor()` canonical correlations to 4 decimal places.

---

## Structural Rhyme

MANOVA : ANOVA :: CCA : Regression

- ANOVA: tests whether scalar group means differ (F-test on SS_between/SS_within)
- MANOVA: tests whether VECTOR group means differ (eigenvalues of H·E^{-1})
- Regression: finds linear combination of X predicting scalar Y (β = Σ_xx^{-1} Σ_xy)
- CCA: finds linear combination of X predicting each axis of Y (SVD of Σ_xx^{-1/2} Σ_xy Σ_yy^{-1/2})

All four are the same GramMatrix computation with different extraction.

---

## The Lab Notebook Claim

> CCA and MANOVA are GramMatrix subblock extraction followed by EigenDecomposition of a derived matrix. The joint GramMatrix of [X|Y] contains Σ_xx, Σ_yy, Σ_xy in its subblocks — one accumulate call, three matrices. CCA = SVD of the whitened cross-covariance (~30 new lines using F22's SVD). MANOVA = eigenvalues of the between/within scatter ratio (~50 new lines using F10's Cholesky and F22's EigenDecomp). F33 adds ~250 total lines after F10 and F22 — the architecture prediction was correct.
