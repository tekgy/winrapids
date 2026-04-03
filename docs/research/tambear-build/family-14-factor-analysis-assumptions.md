# Family 14: Factor Analysis & Structural Equation Modeling — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: C (iterative optimization) wrapping A (covariance computation + eigendecomposition)

---

## Core Insight: Factor Analysis = Structured Covariance Decomposition

All factor analysis methods answer: can the observed p×p covariance matrix Σ be explained by k << p latent factors?

```
Σ = Λ·Φ·Λ' + Ψ
```

where Λ is p×k (loadings), Φ is k×k (factor correlations, I if orthogonal), Ψ is p×p diagonal (uniquenesses).

The computation: (1) compute covariance matrix (Kingdom A, `accumulate(Tiled, x_i·x_j, Add)`), (2) extract factors (Kingdom A eigendecomposition or Kingdom C iterative optimization), (3) rotate for interpretability (Kingdom C optimization).

SEM extends this by adding a structural model: latent factors predict each other + observed outcomes.

**Structural rhyme**: FA loadings matrix Λ = PCA loadings V scaled by √eigenvalue. For ML factor analysis, the relationship is exact when Ψ → 0.

---

## 1. Exploratory Factor Analysis (EFA)

### The Factor Model
```
x = Λf + ε
```
where x is p×1 observed, f is k×1 latent factors (E[f]=0, Var[f]=Φ), ε is p×1 unique (E[ε]=0, Var[ε]=Ψ diagonal).

Implied covariance: Σ = ΛΦΛ' + Ψ.

### Extraction Methods

| Method | Objective | Properties |
|--------|----------|-----------|
| Principal Axis (PAF) | Iterate eigendecomp of reduced correlation (R-Ψ̂) | No distributional assumption. Most common. |
| ML (Maximum Likelihood) | Minimize log|Σ̂| + tr(SΣ̂⁻¹) - log|S| - p | Assumes multivariate normality. χ² test. |
| ULS (Unweighted LS) | Minimize ½·tr[(S-Σ̂)²] | No distributional assumption. |
| GLS (Generalized LS) | Minimize ½·tr[(I - Σ̂⁻¹S)²] | Asymptotically efficient if normal. |
| MINRES | Minimize Σ(s_ij - σ̂_ij)² for off-diagonal | Equivalent to ULS but clearer. |

### Principal Axis Factoring (PAF)
```
1. R = correlation matrix of X                    # F06 correlation
2. Initial communalities: h²_i = max(|r_ij|) or squared multiple correlation
3. Replace diagonal of R with h²: R_reduced = R - diag(1-h²)
4. Eigendecompose R_reduced → (λ, V)              # F02 eigen
5. Loadings: Λ = V[:,:k] · diag(√λ[:k])
6. New communalities: h²_i = Σ_j λ_ij²
7. If h² changed: go to 3                          # Kingdom C iteration
```

Convergence: typically 25-50 iterations. Monitor max|Δh²| < 0.001.

### ML Factor Analysis
Minimize the discrepancy function:
```
F_ML = log|ΛΦΛ' + Ψ| + tr(S·(ΛΦΛ' + Ψ)⁻¹) - log|S| - p
```

Optimization: EM algorithm or direct Newton-Raphson.

**EM for FA**:
- E-step: E[f|x] = Φ·Λ'·Σ⁻¹·x (posterior mean of factors)
- M-step: update Λ, Ψ from sufficient statistics

This IS the IRLS template — same weighted scatter as F10 (logistic), F11 (LME), F15 (IRT).

### Determining Number of Factors

| Method | Decision rule |
|--------|--------------|
| Kaiser criterion | Retain factors with eigenvalue > 1 |
| Scree plot | Visual "elbow" in eigenvalue plot |
| **Parallel analysis** | Compare eigenvalues to random data (BEST) |
| MAP (Velicer) | Minimize average partial correlation |
| BIC/AIC | Information criterion from ML fit |
| χ² test (ML only) | Test if k factors sufficient |

**CRITICAL: Kaiser overestimates. Parallel analysis is the gold standard.** (Horn 1965, endorsed by simulation studies)

### GPU decomposition
- Correlation matrix: `accumulate(Tiled{X', X}, corr_expr, Add)` — GramMatrix of standardized data
- Eigendecomposition: F02
- PAF iteration: parallel eigendecomposition per iteration
- ML optimization: F05

---

## 2. Factor Rotation

### Why Rotate?
Unrotated loadings are mathematically correct but uninterpretable. The factor model is rotationally indeterminate: if Λ is a solution, so is ΛT for any orthogonal T (or non-singular T for oblique).

### Orthogonal Rotations (factors uncorrelated, Φ = I)

| Rotation | Criterion | Interpretation |
|----------|----------|----------------|
| **Varimax** | Maximize Σ_j Var(Λ²_·j) | Simple structure (each variable loads on few factors) |
| Quartimax | Maximize Σ_i Var(Λ²_i·) | Each variable loads on one factor |
| Equamax | Weight between varimax and quartimax | Compromise |

### Oblique Rotations (factors correlated, Φ ≠ I)

| Rotation | Method | Output |
|----------|--------|--------|
| **Promax** | Raise varimax loadings to power (κ=4) → target, then oblique | Pattern + structure matrices |
| Oblimin | Minimize cross-products of loadings | Direct oblimin (δ=0 default) |
| Geomin | Geometric mean of squared loadings | Robust to complex structure |

### Rotation Mechanics
For orthogonal: find T (orthogonal) maximizing the criterion. Solved by iterative pairwise (Jacobi) rotations.

For oblique: find T (non-singular) minimizing the criterion. Pattern matrix = ΛT⁻¹', Structure matrix = ΛT'.

**Factor correlation matrix**: Φ = T'T (for orthogonal T, Φ = I).

### CRITICAL: Report Both Matrices for Oblique
- **Pattern matrix** (P = ΛT⁻¹'): regression weights (direct effects)
- **Structure matrix** (S = ΛT'): correlations between variables and factors
- P ≠ S when factors are correlated. Reporting only one is misleading.

### GPU decomposition
- Varimax: pairwise Jacobi rotations (parallel across rotation pairs)
- Promax: varimax → power transform → procrustes (F33) → oblique target

---

## 3. Polychoric and Polyserial Correlations

### Why Needed
Standard Pearson correlation on ordinal data (Likert scales) attenuates toward zero. Polychoric correlation estimates the correlation between the LATENT continuous variables underlying the ordinal responses.

### Polychoric Correlation (ordinal × ordinal)
1. Build contingency table (scatter_add)
2. Maximum likelihood: find ρ and thresholds τ such that bivariate normal probabilities match observed cell proportions
3. Optimization: usually 1D search over ρ (thresholds estimated independently from marginals)

### Polyserial Correlation (ordinal × continuous)
Similar: find ρ such that bivariate normal likelihood matches observed data.

### Tetrachoric Correlation (special case: 2×2)
```
cos(π·√(bc/(ad+bc))) where a,b,c,d are cell counts (Bonett & Price approximation)
```

### CRITICAL: Polychoric correlations can produce non-positive-definite correlation matrices.
Fix: nearest PD matrix via Higham (2002) alternating projection. F02 eigendecomposition → clamp negative eigenvalues to ε → reconstruct.

### GPU decomposition
- Contingency tables: scatter_add (parallel)
- Bivariate normal CDF: numerical integration or Drezner-Wesolowsky approximation (parallel per pair)
- Optimization per pair: independent → embarrassingly parallel

---

## 4. Confirmatory Factor Analysis (CFA)

### The CFA Model
Same as EFA but with CONSTRAINTS: specific loadings fixed to 0 (or other values). The researcher specifies which variables load on which factors.

```
x = Λ·f + ε    where Λ has known zero pattern
```

### Estimation
Minimize discrepancy between S (sample covariance) and Σ(θ) (model-implied covariance) where θ = free parameters in Λ, Φ, Ψ.

| Estimator | Discrepancy | Assumption |
|-----------|-------------|------------|
| ML | log|Σ| + tr(SΣ⁻¹) - log|S| - p | Multivariate normal |
| DWLS/WLSMV | Weighted LS on polychoric | Ordinal data |
| ULS | Unweighted LS | No distributional assumption |

### Fit Indices

| Index | Good fit | Formula |
|-------|----------|---------|
| χ² | p > 0.05 | n·F_ML |
| CFI | > 0.95 | 1 - max(χ²_model - df_model, 0) / max(χ²_null - df_null, 0) |
| TLI | > 0.95 | (χ²_null/df_null - χ²_model/df_model) / (χ²_null/df_null - 1) |
| RMSEA | < 0.06 | √(max(χ²_model - df_model, 0) / (df_model·(n-1))) |
| SRMR | < 0.08 | √(mean(standardized_residuals²)) |

### CRITICAL: χ² is sample-size dependent. For n > 500, almost always rejects. Use CFI/RMSEA instead.

---

## 5. Structural Equation Modeling (SEM)

### The Full SEM
Two sub-models:
1. **Measurement model** (CFA): x = Λ_x·ξ + δ, y = Λ_y·η + ε
2. **Structural model** (regression among latents): η = B·η + Γ·ξ + ζ

Implied covariance of observed variables:
```
Σ(θ) = [Λ_y(I-B)⁻¹(ΓΦΓ' + Ψ)(I-B)⁻¹'Λ_y' + Θ_ε,  Λ_y(I-B)⁻¹ΓΦΛ_x']
        [Λ_x Φ Γ'(I-B)⁻¹'Λ_y',                       Λ_x Φ Λ_x' + Θ_δ  ]
```

### Estimation
Same as CFA: minimize F(S, Σ(θ)). Same estimators (ML, DWLS, ULS).

### Path Analysis
SEM without latent variables. All variables observed. Just the structural part:
```
y = By + Γx + ζ
```
Direct, indirect, and total effects.

### Mediation/Moderation
- **Mediation**: X → M → Y. Indirect effect = a·b (product of paths).
- **Bootstrap CI for indirect effect** (Preacher & Hayes): resample, refit, extract a·b distribution.
- **Moderation**: X → Y depends on Z. Include interaction X·Z.

### Multi-Group SEM
Fit same model across groups (gender, age, etc.) with progressively constrained parameters:
1. Configural invariance: same structure, free parameters
2. Metric invariance: equal loadings
3. Scalar invariance: equal loadings + intercepts
4. Strict invariance: equal loadings + intercepts + residual variances

Chi-square difference test between nested models.

### GPU decomposition
- Σ(θ) computation: matrix algebra (F02)
- Gradient of F w.r.t. θ: chain rule through matrix operations
- Optimization: F05 (quasi-Newton, typically)
- Bootstrap: parallel (each resample independent)

---

## 6. Reliability

### Cronbach's Alpha
```
α = (k/(k-1)) · (1 - Σ σ²_i / σ²_total)
```

From F06 MomentStats: need item variances and total variance.

### McDonald's Omega (preferred)
```
ω = (Σ λᵢ)² / ((Σ λᵢ)² + Σ ψᵢ)
```

From factor loadings. Better than alpha when loadings differ (tau non-equivalence).

### CRITICAL: Cronbach's alpha assumes tau-equivalence (equal loadings). When loadings differ, alpha underestimates reliability. Use omega.

---

## 7. Numerical Stability

### Heywood Cases
Uniqueness ψᵢ < 0 (variance less than zero) or communality h²ᵢ > 1. Indicates:
- Too many factors
- Too few observations
- Multicollinearity

**Fix**: constrain ψᵢ ≥ ε (e.g., 0.005) during optimization.

### Improper Solutions
Model-implied Σ(θ) is not positive definite. Can happen with:
- Negative residual variances (Heywood)
- Correlations > 1 between latent variables
- (I-B) singular in structural model

**Fix**: barrier function in optimization to enforce PD constraint.

### Near-Singular Correlations
When variables are highly correlated (r > 0.95), the correlation matrix is ill-conditioned. Ridge FA: R_ridge = (1-λ)R + λI.

### GPU considerations
- Polychoric correlation matrix: O(p²) bivariate optimizations → GPU parallel across pairs
- Factor extraction: F02 eigendecomposition
- CFA/SEM optimization: F05 with analytic gradients

---

## 8. Edge Cases

| Algorithm | Edge Case | Expected |
|-----------|----------|----------|
| EFA | k > p (more factors than variables) | Error |
| EFA | n < p (more variables than observations) | Regularize or error |
| PAF | Communality > 1 during iteration | Clamp to 0.999 (Heywood) |
| ML FA | Non-convergence | Report last iterate + warning |
| Rotation | k = 1 | No rotation possible. Return unrotated. |
| Polychoric | Cell with 0 count | Add 0.5 continuity correction |
| Polychoric | All responses in one category | Cannot estimate. Return NA. |
| CFA | Model not identified (df < 0) | Error with diagnostic message |
| SEM | (I-B) singular | Feedback loop without solution. Error. |
| Reliability | k = 1 item | Alpha undefined. Error. |

---

## Sharing Surface

### Reuses from Other Families
- **F02 (Linear Algebra)**: eigendecomposition, matrix inverse, Cholesky, SVD
- **F05 (Optimization)**: ML estimation, CFA/SEM discrepancy minimization
- **F06 (Descriptive)**: correlation matrix, item variances, MomentStats
- **F07 (Hypothesis)**: χ² test for model fit, chi-square difference tests
- **F08 (Nonparametric)**: bootstrap for mediation CIs, parallel analysis
- **F33 (Multivariate)**: Procrustes rotation, canonical correlation

### Provides to Other Families
- **F15 (IRT)**: FA loadings initialize 2PL discrimination parameters
- **F22 (Dim Reduction)**: FA as alternative to PCA (accounts for unique variance)
- **F16 (Mixture)**: Factor mixture models (FA within latent classes)
- **F11 (Mixed Effects)**: Multilevel SEM

### Structural Rhymes
- **ML FA = EM iteration**: same IRLS template as F10/F11/F15
- **CFA discrepancy = regression residual**: same optimization structure
- **Varimax = simple structure pursuit**: same goal as sparse PCA (F22)
- **Path analysis = regression with direct/indirect effects**: F10 composition

---

## Implementation Priority

**Phase 1** — Core EFA (~120 lines):
1. Correlation matrix (Pearson, from F06)
2. Principal axis factoring (eigendecomposition iteration)
3. Varimax rotation (Jacobi pairwise)
4. Promax rotation (varimax → power → oblique target)
5. Parallel analysis for factor number

**Phase 2** — CFA/SEM (~200 lines):
6. CFA model specification (loading pattern matrix)
7. ML estimation of CFA
8. Fit indices (χ², CFI, TLI, RMSEA, SRMR)
9. SEM (measurement + structural model)
10. Path analysis (observed variable SEM)

**Phase 3** — Special correlations + reliability (~150 lines):
11. Polychoric correlation (bivariate normal MLE)
12. Tetrachoric correlation (approximation + exact)
13. DWLS/WLSMV estimation for ordinal data
14. Cronbach's alpha, McDonald's omega

**Phase 4** — Extensions (~100 lines):
15. Multi-group invariance testing
16. Mediation (bootstrap indirect effect)
17. Modification indices (Lagrange multiplier tests)
18. Bifactor models

---

## Composability Contract

```toml
[family_14]
name = "Factor Analysis & SEM"
kingdom = "C (iterative optimization) wrapping A (covariance + eigendecomposition)"

[family_14.shared_primitives]
efa = "Exploratory FA: extract + rotate loadings from correlation matrix"
cfa = "Confirmatory FA: constrained estimation with fit indices"
sem = "Full SEM: measurement + structural model"
rotation = "Varimax/promax/oblimin on loading matrix"
polychoric = "Polychoric/tetrachoric correlations for ordinal data"

[family_14.reuses]
f02_linear_algebra = "Eigendecomposition, matrix inverse, SVD"
f05_optimization = "ML estimation, discrepancy minimization"
f06_descriptive = "Correlation matrix, item variances"
f07_hypothesis = "Chi-square tests, model comparison"
f08_nonparametric = "Bootstrap for mediation, parallel analysis"
f33_multivariate = "Procrustes rotation"

[family_14.provides]
loadings = "Factor loading matrix (pattern + structure)"
factor_scores = "Estimated latent variable scores"
fit_indices = "CFI, TLI, RMSEA, SRMR, chi-square"
reliability = "Alpha, omega coefficients"
polychoric_matrix = "Polychoric correlation matrix"

[family_14.consumers]
f15_irt = "Loading initialization for 2PL discrimination"
f22_dim_reduction = "FA as alternative to PCA"
f16_mixture = "Factor mixture models"
f11_mixed_effects = "Multilevel SEM"
```
