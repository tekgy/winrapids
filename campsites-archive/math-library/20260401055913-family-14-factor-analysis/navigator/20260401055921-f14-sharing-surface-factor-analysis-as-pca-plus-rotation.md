# F14 Sharing Surface: Factor Analysis as PCA Plus Rotation

Created: 2026-04-01T05:59:21-05:00
By: navigator

Prerequisites: F10 complete (GramMatrix), F22 complete (EigenDecomposition).

---

## Core Insight

**Exploratory Factor Analysis (EFA) = PCA on the correlation matrix + rotation + communality estimation.**

Once F22 provides EigenDecomposition:
- The eigendecomp step is FREE (reuse F22's implementation on correlation matrix)
- The only new code: rotation algorithms (Varimax, Oblimin) + communality estimation

F14 adds ~250-400 lines on top of F22. Most of it is rotation mathematics — no new GPU needed.

---

## What Factor Analysis Computes

Given data X (N × p):
```
1. Correlation matrix: R = cor(X)  (from GramMatrix normalized)
2. Factor extraction: R = Λ Λᵀ + Ψ  where Λ = loadings (p × k), Ψ = uniquenesses (diagonal)
3. Rotation: Λ_rotated = Λ T  for orthogonal (T'T = I) or oblique (T general) rotation
4. Factor scores: F = X_std Λ (Λ'Λ)⁻¹  (regression-based scores)
```

**What's different from PCA**:
- PCA: maximize variance of projections (eigendecomp of covariance/correlation, keep k largest)
- EFA: model the correlation structure with communality adjustment (eigendecomp of R - Ψ)
- EFA iterates to estimate Ψ (iterative principal axis)
- EFA rotates loadings for interpretability

---

## MSR Sharing with F22

| Step | F22 (PCA) | F14 (FA) | Shared? |
|------|-----------|----------|:-------:|
| GramMatrix | X'X | X'X | ✓ (same) |
| Correlation matrix | optional | required | ✓ (same extraction) |
| EigenDecomposition | on covariance | on (R - Ψ) | ✓ same algorithm, different input |
| Projection | Y = X V[:,:k] | factor scores | ~ (similar) |
| Rotation | — | Varimax/Oblimin | New in F14 |
| Communality | — | h² = Σ λ²_ij | New in F14 |

---

## Extraction Methods

### 1. Principal Axis Factoring (PAF) — recommended for Phase 1

Starting values: Ψ⁰ = diag(R)⁻¹ (SMC = squared multiple correlations).

```
Iterate:
  R* = R - Ψ  (reduce correlation matrix by uniquenesses)
  Eigendecompose R*  (using F22's algorithm)
  Keep k largest eigenvalues/eigenvectors
  Update h²_i = Σ_{j=1}^k λ²_ij  (communality of variable i)
  Update Ψ_i = 1 - h²_i
  Repeat until ||Ψ_new - Ψ_old|| < tol
```

Each iteration reuses F22's EigenDecomposition on a slightly modified matrix. 3-5 iterations typically sufficient.

**Tambear path**: F22 EigenDecomp called once per iteration. Inner loop is CPU (p×p matrix operations). Kingdom C outer loop.

### 2. Maximum Likelihood Factor Analysis (MLFA)

Maximizes likelihood under normality assumption. More expensive — requires iterative optimization (F05 needed).

Phase 2 only.

### 3. Principal Components Extraction (PCE)

Just PCA. No iteration. Loadings = V[:, :k] * √eigenvalues[:k]. This is trivially F22 PCA.

Phase 1 can implement this as "FA-by-PCA" for immediate deployment.

---

## Rotation Algorithms

Rotation transforms loadings Λ → Λ T to maximize interpretability (simple structure).

### Varimax (orthogonal rotation — most common)

Maximizes sum of variances of squared loadings within factors:
```
Q = Σ_j [1/p Σ_i λ⁴_ij - (1/p Σ_i λ²_ij)²]  → maximize
```

Algorithm: iterative pairwise Jacobi rotations. Per iteration: optimize rotation of columns j and k (O(p²) iterations × O(p) per step = O(p³) total).

```rust
fn varimax(loadings: &mut Vec<f64>, p: usize, k: usize, max_iter: usize) {
    for _iter in 0..max_iter {
        for j in 0..k {
            for kk in (j+1)..k {
                // Compute optimal rotation angle for columns j, kk
                // Apply rotation
            }
        }
        if converged { break; }
    }
}
```

Gold standard: `psych::fa(R, nfactors=k, rotate="varimax")` in R, `sklearn.decomposition.FactorAnalysis` in Python.

### Oblimin (oblique rotation — allows correlated factors)

Similar structure, but T is not constrained to be orthogonal. Factors can correlate.

Phase 2.

---

## Confirmatory Factor Analysis (CFA) — SEM Integration

CFA specifies WHICH variables load on WHICH factors a priori. It's a constrained optimization:

```
Minimize: FML = log|Σ(θ)| + tr(S Σ⁻¹(θ)) - log|S| - p
where Σ(θ) = Λ Φ Λ' + Θ  (model-implied covariance)
      S = observed covariance = GramMatrix / (N-1)
      θ = (Λ, Φ, Θ) = factor loadings, factor correlations, uniquenesses
```

This is a continuous optimization problem (Kingdom C). Requires:
- F10 GramMatrix for S
- F05 optimization for MLE

Full CFA/SEM (Lavaan replacement) is a larger undertaking. F14 Phase 1 should be EFA only.

---

## MSR Types F14 Produces

```rust
pub struct FactorModel {
    pub n_variables: usize,    // p
    pub n_factors: usize,      // k
    pub loadings: Vec<f64>,    // Λ, shape (p, k)
    pub uniquenesses: Vec<f64>, // Ψ = 1 - communalities, shape (p,)
    pub communalities: Vec<f64>, // h² = 1 - Ψ, shape (p,)
    pub eigenvalues: Vec<f64>,   // k eigenvalues from extraction
    pub rotation: RotationType,
    pub factor_correlations: Option<Vec<f64>>, // φ, for oblique rotation
    pub fit_stats: FactorFitStats,  // RMSEA, CFI, TLI, χ², df, p
}
```

Add to `IntermediateTag`:
```rust
FactorModel {
    gram_id: DataId,
    n_factors: u32,
    method: ExtractionMethod,
    rotation: RotationType,
},
```

---

## What Consumes FactorModel

| Consumer | What it uses |
|---------|-------------|
| F15 (IRT/Psychometrics) | FA loadings as starting values for item parameters |
| F11 (Mixed Effects) | Factor scores as observed proxies for latent variables |
| F22 (PCA) | Same EigenDecomp — can compare PCA vs FA results |
| F33 (SEM/CFA) | FA loadings → CFA model specification |

---

## Build Order

1. **Correlation matrix from GramMatrix** — normalize GramMatrix diagonal to 1.0 (~5 lines)
2. **PAF iteration** — loop calling F22 EigenDecomp on (R - Ψ), update Ψ each pass (~50 lines)
3. **Varimax rotation** — pairwise Jacobi, criterion function, convergence check (~100 lines)
4. **FactorModel struct + IntermediateTag** in `intermediates.rs` (~30 lines)
5. **Factor scores** — regression-based: F = X_std Λ (Λ'Λ)⁻¹ (~20 lines)
6. **Fit statistics** — RMSEA, χ² test of model fit (~30 lines)
7. **Tests**: `psych::fa()` in R is the gold standard. Match loadings within 0.001 (after sign normalization).

---

## The Lab Notebook Claim

> Factor Analysis is PCA with iteration and rotation. Both use EigenDecomposition of a correlation matrix, derived from GramMatrix. F14 adds ~250 lines to F22's infrastructure — the rotation and communality estimation. The architecture prediction (F14 is nearly free after F22) was correct: the expensive computation is shared, and the differentiating math is CPU-side.
