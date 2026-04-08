# F10 Sharing Surface: GramMatrix as the Pass 2 Backbone

Created: 2026-04-01T05:47:15-05:00
By: navigator

Prerequisites: F06 complete (MomentStats). DotProductOp exists in winrapids-tiled (already proven on Blackwell+M4).

---

## Why F10 Is the Backbone

F10 (Regression) produces `GramMatrix` — the Level 1 MSR that unlocks 6 other families:

| Family | Unlocked by | What it uses |
|--------|------------|-------------|
| F14 (Factor Analysis) | GramMatrix → correlation → EigenDecomp | Σ_xx for loadings |
| F22 (PCA, t-SNE, UMAP) | GramMatrix → EigenDecomp | Σ for principal components |
| F33 (CCA, MANOVA) | GramMatrix subblocks Σ_xx, Σ_yy, Σ_xy | Generalized eigenvalue |
| F07 (ANOVA F-test) | FittedModel from GramMatrix | F-statistic = Regression F-test |
| F09 (Robust regression, IRLS) | FittedModel as warm start | β̂, residuals, σ² |
| F35 (Causal inference, IV, RD) | FittedModel (residualization) | Orthogonalized residuals |

GramMatrix is "Pass 2" in the spine. After Pass 1 (Moments) and Pass 2 (GramMatrix), 15+ families are either fully or partially unlocked.

---

## GramMatrix: The Tambear Decomposition

### What GramMatrix is

For a data matrix X (N × p):
```
GramMatrix = X'X = p×p matrix where GramMatrix[i,j] = Σ_n X[n,i] * X[n,j]
```

For regression, we also need X'y (p × 1) and y'y (scalar). So the full accumulate produces:
```
[X'X  X'y]   (p+1) × (p+1) augmented gram matrix
[y'X  y'y]
```

This is a **DotProductOp tiled accumulate** over the augmented data matrix [X | y].

### How it maps to tambear

```rust
// Augmented data: append y as the last column
let data_aug: &[f64] = ...;  // shape (N, p+1), row-major; last col = y

// GramMatrix via TiledEngine:
let gram = engine.accumulate_tiled(
    data_aug,
    n_rows,
    p + 1,  // n_cols (includes y)
    TiledGrouping::Full(p + 1),  // all column pairs
    DotProductOp,
)?;
// gram[i,j] = Σ_n data[n,i] * data[n,j]
// gram[p,p] = y'y
// gram[i,p] = x_i'y  (cross-terms with y)
// gram[i,j] (i,j < p) = X'X
```

**DotProductOp is already implemented** in `winrapids-tiled/src/ops.rs`. F10 needs the wire-up from `accumulate.rs`'s `Tiled` grouping to TiledEngine.

### Centering: the crucial step

For OLS, X typically includes an intercept column. But for computational stability, center the data first:

```
X̃[n,i] = X[n,i] - X̄[i]    (subtract column means)
ỹ[n] = y[n] - ȳ
```

Column means come from MomentStats(order=1, All). So:

**F10 depends on F06's MomentStats for centering before GramMatrix.**

After centering: `GramMatrix = X̃'X̃`, and the intercept is recovered as `β₀ = ȳ - β̃'X̄`.

---

## OLS: From GramMatrix to β̂

```
β̂ = (X'X)⁻¹ X'y
```

Step 1: Cholesky factorization of X'X (p × p symmetric positive definite matrix):
```
X'X = L L'  where L is lower triangular
```

Step 2: Solve L L' β̂ = X'y via forward/backward substitution (two triangular solves):
```
L z = X'y  →  z (forward solve)
L' β̂ = z   →  β̂ (backward solve)
```

Cost: O(p³/3) for Cholesky + O(p²) for solves. Negligible vs O(N×p²) for GramMatrix when N >> p.

**The Cholesky factorization is the ONLY non-accumulate step.** Everything else is scatter or element-wise.

---

## FittedModel: Level 2 Derived Type

After solving for β̂:

```rust
pub struct FittedModel {
    pub n_obs: usize,
    pub n_params: usize,         // p (not counting intercept separately)
    pub coefficients: Vec<f64>,  // β̂, shape (p,)
    pub intercept: f64,          // β₀
    pub fitted_values: Option<Arc<Vec<f64>>>,   // ŷ = Xβ̂, shape (N,) — lazy
    pub residuals: Option<Arc<Vec<f64>>>,       // e = y - ŷ, shape (N,) — lazy
    pub sigma_sq: f64,           // MSE = RSS / (N - p - 1)
    pub r_squared: f64,          // R² = 1 - RSS/TSS
    pub adj_r_squared: f64,      // Adjusted R²
    pub f_stat: f64,             // Overall F-statistic
    pub f_pvalue: f64,           // p-value for F-test
    pub se_coefficients: Vec<f64>,  // standard errors of β̂
    pub t_stats: Vec<f64>,          // t-statistics for each coefficient
    pub t_pvalues: Vec<f64>,        // p-values for each t-stat
}
```

Key insight: ŷ and residuals are lazy — they require an O(N×p) matrix-vector product to compute. For large N, don't compute them unless requested. All other fields derive from GramMatrix alone.

**From GramMatrix:**
- `RSS = y'y - β̂'(X'y)` — pure dot products from gram, no second data pass
- `TSS = Σ(y - ȳ)² = y'y - n·ȳ²` — from gram and mean
- `R² = 1 - RSS/TSS` — no data pass needed
- `F-stat = (R²/(p)) / ((1-R²)/(N-p-1))` — the ANOVA = Regression F-test rhyme

**ANOVA = Regression F-test**: the F-statistic in OLS regression IS the ANOVA F-statistic for the regression model. For F07's one-way ANOVA with k groups: code k-1 dummy variables, run OLS, extract F-stat. Same computation, same GramMatrix, same FittedModel.

---

## Regression Diagnostics

All extractable from FittedModel + GramMatrix:

| Diagnostic | Formula | Data pass needed? |
|-----------|---------|:----------------:|
| Variance Inflation Factor (VIF) | VIF_i = 1/(1-R²_i) where R²_i = regress x_i on other x_j | Yes — one regression per predictor |
| Leverage (hat matrix diagonal) | h_ii = x_i'(X'X)⁻¹x_i | No — from GramMatrix inverse |
| Cook's distance | D_i = e_i² · h_ii / (p · MSE · (1-h_ii)²) | Needs residuals (one data pass) |
| Durbin-Watson (serial correlation in residuals) | DW = Σ(e_t - e_{t-1})² / Σe_t² | Needs residuals |
| Breusch-Pagan (heteroskedasticity) | Regress e² on X, test R² | Needs residuals + second regression |
| RESET test (functional form) | Add ŷ², ŷ³ to model, test with F-test | Needs fitted values |

For Phase 1: implement β̂, standard errors, R², F-stat, t-stats. These all come from GramMatrix alone.
For Phase 2: diagnostics that need residuals.

---

## Regression Variants

### OLS (standard): done from GramMatrix as above.

### WLS (weighted least squares):
```
β̂_WLS = (X'WX)⁻¹ X'Wy
```
Same structure, but the gram is `WeightedGramMatrix` with element-wise weights. Extend DotProductOp to support weight vector: `acc[i,j] += w[n] * x[n,i] * x[n,j]`.

### Ridge regression (L2 regularization):
```
β̂_ridge = (X'X + λI)⁻¹ X'y
```
Same GramMatrix + add λ to diagonal before Cholesky. Zero extra work.

### LASSO (L1 regularization):
Requires iterative algorithm (coordinate descent or ADMM). Outside GramMatrix extraction. Kingdom C.

### GLM (logistic, Poisson, etc.):
Requires IRLS (iteratively reweighted least squares). Each IRLS step IS weighted OLS. Kingdom C — outer loop over Kingdom A steps.

### Quantile regression:
Not from GramMatrix. Requires LP or iterative algorithm. Outside polynomial MSR.

---

## What F10 Produces for TamSession

```rust
// Level 1 MSR:
IntermediateTag::GramMatrix { data_id, grouping_id }  → Arc<GramMatrix>

// Level 2 derived:
IntermediateTag::FittedModel { data_id, response_id, formula_hash }  → Arc<FittedModel>
IntermediateTag::EigenDecomposition { gram_id }  → Arc<EigenDecomposition>
// (EigenDecomp deferred to after Cholesky + eigenvalue algorithm — shared by PCA, FA)
```

**GramMatrix struct:**
```rust
pub struct GramMatrix {
    pub p: usize,        // number of columns (features)
    pub matrix: Arc<Vec<f64>>,  // p×p symmetric, row-major
    pub column_means: Arc<Vec<f64>>,  // for uncentering predictions
    pub column_stds: Arc<Vec<f64>>,   // for standardized coefficients
}

impl GramMatrix {
    /// Extract correlation matrix from covariance GramMatrix.
    pub fn to_correlation(&self) -> Vec<f64> { ... }

    /// Diagonal = squared column norms.
    pub fn diagonal(&self, i: usize) -> f64 { self.matrix[i * self.p + i] }

    /// For L2 distance: D[i,j] = gram[i,i] - 2*gram[i,j] + gram[j,j]
    pub fn to_l2_distance_matrix(&self) -> Vec<f64> { ... }
}
```

---

## Build Order

1. **Wire Tiled(p,p) grouping to TiledEngine** in `accumulate.rs` — DotProductOp already exists
2. **GramMatrix struct** in `intermediates.rs` — 30 lines
3. **GramMatrix accumulation** — call TiledEngine::run with DotProductOp + centering pass from F06
4. **Cholesky solve** — already in `cholesky.rs` (verify it handles p×p, not just 3×3)
5. **OLS extraction** — β̂, σ², R², F-stat, SE, t-stats from gram + solve
6. **FittedModel struct** in `intermediates.rs` — 20 lines
7. **IntermediateTag::GramMatrix + FittedModel** — extend enum
8. **Tests**: `lm()` in R is the gold standard. `scipy.stats.linregress()` for bivariate. sklearn for multivariate.
9. **Ridge regression** — add λ to diagonal, same code path (one extra line)
10. **WLS** — extend DotProductOp to weighted variant

---

## Verification Strategy

Gold standard: R's `lm()` function.

```r
x <- matrix(rnorm(100*3), ncol=3)
y <- rnorm(100)
model <- lm(y ~ x)
summary(model)
# Coefficients, SE, t-stats, p-values, R², F-stat
```

For tambear: same data → should match R's `lm()` to within 1e-10 (f64 precision).

---

## The Structural Insight for Lab Notebook

> GramMatrix is the heart of Pass 2 in the spine. It enables regression, PCA, factor analysis, CCA, and every linear method simultaneously. It takes one tiled accumulate over O(N×p²) products and computes the sufficient statistics for the entire Kingdom A linear algebra family. The Cholesky solve (O(p³)) is fast enough to forget: for typical p < 100, it takes microseconds while the accumulate takes milliseconds.

> The ANOVA = Regression F-test rhyme is the most pedagogically important structural insight in the library: the "different" statistical traditions of regression (continuous x) and ANOVA (categorical x) are identical computations. One-way ANOVA with k groups is OLS regression with k-1 dummy variables. The F-statistic formula is identical. The GramMatrix is block-diagonal in the ANOVA case. Both live in FittedModel.
