# Regression Algorithm Specifications
**Author**: math-researcher  
**Date**: 2026-04-06  
**Scope**: Complete mathematical reference for regression algorithms — OLS through ElasticNet.  
**Purpose**: Implementation blueprint for pathmaker. Every formula is paper-verifiable. Accumulate+gather decomposition given for each.

---

## Design Principle: All Regression = Augmented Gram Matrix

The sufficient statistics for every regression algorithm are:
```
G = X'X  (p×p)      ← accumulate(rows, x⊗x, sum)
c = X'y  (p×1)      ← accumulate(rows, x·y, sum)
S = y'y  (scalar)   ← accumulate(rows, y², sum)
n = count
```

Everything else (β estimates, residuals, standard errors, log-likelihoods) flows from these.

**Anti-YAGNI**: Always compute and cache (G, c, S, n). Streaming or re-accumulating is the failure mode.

---

## 1. Ordinary Least Squares (OLS)

### Source
Gauss-Markov theorem; Rao (1973) *Linear Statistical Inference*; Davidson & MacKinnon (2004) §1-3.

### Model
```
y = Xβ + ε,   ε ~ N(0, σ²I)
```
X is n×p (including intercept as first column if desired).

### Sufficient Statistics
```
G = X'X        ← accumulate(rows, x⊗x, sum)
c = X'y        ← accumulate(rows, x·y, sum)
S = y'y        ← accumulate(rows, y², sum)
n, p = dimensions
```

### Point Estimates
```
β̂ = G⁻¹c = (X'X)⁻¹X'y
ŷ = Xβ̂
e = y - ŷ

RSS = y'y - c'G⁻¹c = S - c'β̂          (no re-scan needed)
σ̂² = RSS / (n - p)
```

**Implementation**: Use Cholesky factorization of G (positive definite when X is full rank). Solve G·β̂ = c by forward/back substitution.

### Inference
```
Var(β̂) = σ̂² G⁻¹
SE(β̂_j) = σ̂ · sqrt(G⁻¹_{jj})
t_j = β̂_j / SE(β̂_j),   df = n-p

TSS = S - n·ȳ²     where ȳ = (sum of y) / n
ESS = TSS - RSS
R² = ESS / TSS = 1 - RSS/TSS
R²_adj = 1 - (RSS/(n-p)) / (TSS/(n-1))

F = (R²/（p-1)) / ((1-R²)/(n-p)),   df = (p-1, n-p)   [for intercept-included model]
```

### Log-likelihood
```
ℓ(β, σ²) = -n/2 · ln(2π) - n/2 · ln(σ²) - RSS/(2σ²)
```

### AIC/BIC
```
AIC = -2ℓ + 2k    where k = p + 1 (p coefficients + σ²)
BIC = -2ℓ + k·ln(n)
```

### Prediction
```
ŷ_new = x_new' β̂
SE_pred² = σ̂² (1 + x_new' G⁻¹ x_new)  [prediction interval — includes new obs noise]
SE_mean² = σ̂² (x_new' G⁻¹ x_new)       [confidence interval for E[y|x_new]]
```

### Accumulate+Gather Decomposition
```
1. accumulate(rows_i, (x_i ⊗ x_i, x_i·y_i, y_i²), sum) → (G, c, S)
2. Cholesky(G) → L (Triangle solve: no further accumulate needed)
3. gather(β̂, from Cholesky solve of G·β̂ = c)
4. gather(RSS, from S - c'β̂)
```
Pattern: OLS = **linear scan** (Kingdom A).

---

## 2. Weighted Least Squares (WLS)

### Source
Aitken (1935); identical to GLS when Ω = diag(w).

### Model
```
y = Xβ + ε,   ε ~ N(0, σ²W⁻¹)
```
where W = diag(w₁,...,wₙ) with given weights wᵢ > 0.

### Sufficient Statistics
```
G_w = X'WX     ← accumulate(rows, w_i · x_i⊗x_i, sum)
c_w = X'Wy     ← accumulate(rows, w_i · x_i·y_i, sum)
S_w = y'Wy     ← accumulate(rows, w_i · y_i², sum)
```

### Estimates (identical to OLS on weighted data)
```
β̂ = (X'WX)⁻¹X'Wy = G_w⁻¹ c_w
RSS_w = S_w - c_w' β̂
σ̂² = RSS_w / (n - p)
```

---

## 3. Logistic Regression

### Source
Cox (1958) *JRSS*; McCullagh & Nelder (1989) §4.4; Agresti (2013) §5.

### Model
```
P(y=1 | x) = π(x) = σ(x'β) = 1/(1 + exp(-x'β))
```

### Log-likelihood
```
ℓ(β) = Σᵢ [yᵢ·x'β - log(1 + exp(x'β))]
       = Σᵢ [yᵢ·ηᵢ - log(1 + exp(ηᵢ))]    where ηᵢ = x'β (linear predictor)
```

**Numerically stable** log(1+exp(η)) = log-sum-exp trick:
```
softplus(η) = log(1 + exp(η))
            = η + log(1 + exp(-η))   if η > 0
            = log(1 + exp(η))         if η ≤ 0
```

### IRLS (Newton-Raphson)
```
πᵢ = σ(x'β_old)
wᵢ = πᵢ(1 - πᵢ)                  [IRLS weight]
zᵢ = ηᵢ + (yᵢ - πᵢ)/wᵢ           [working response]
```

Each iteration solves WLS with weights wᵢ and response zᵢ:
```
gradient:  g = X'(y - π)           ← accumulate(rows, x_i·(y_i - π_i), sum)
Hessian:   H = -X'WX               ← accumulate(rows, -w_i · x_i⊗x_i, sum)
step:      β_new = β_old - H⁻¹g = β_old + (X'WX)⁻¹X'(y-π)
```

**Convergence**: |ℓ_new - ℓ_old| < tol (1e-8). Typically converges in 10-30 iterations.

### Standard Errors
```
Fisher information: I(β) = X'WX  (evaluated at β̂)
Var(β̂) = I(β̂)⁻¹
SE(β̂_j) = sqrt(I⁻¹_{jj})
z_j = β̂_j / SE(β̂_j)   [Wald test, ~ N(0,1)]
```

### Deviance
```
Deviance = -2ℓ(β̂)
Null deviance = -2ℓ(β̂₀)  [intercept only]
Deviance residuals: dᵢ = sign(yᵢ - πᵢ) · sqrt(-2·[yᵢ log(πᵢ) + (1-yᵢ)log(1-πᵢ)])
```

### Accumulate+Gather Decomposition
```
1. accumulate(rows_i, x_i·η_i terms, sum) → ℓ(β)   [gradient]
2. accumulate(rows_i, w_i·x_i⊗x_i, sum) → X'WX   [Hessian]
3. gather(β_new, from Newton step)
4. Repeat until convergence
```
Pattern: IRLS = **iterative linear scan** (Kingdom AC).

---

## 4. Poisson Regression

### Source
Nelder & Wedderburn (1972); McCullagh & Nelder (1989) §6.

### Model
```
P(y=k | x) = exp(-λ) λᵏ / k!,   λ = exp(x'β)   [log link, canonical]
```

### Log-likelihood
```
ℓ(β) = Σᵢ [yᵢ·x'β - exp(x'β) - log(yᵢ!)]
       = Σᵢ [yᵢ·ηᵢ - exp(ηᵢ)] + const
```

### IRLS
```
λᵢ = exp(x'β)                    [mean]
wᵢ = λᵢ                           [IRLS weight = variance for Poisson]
zᵢ = ηᵢ + (yᵢ - λᵢ)/λᵢ           [working response]

gradient:  g = X'(y - λ)         ← accumulate(rows, x_i·(y_i - λ_i), sum)
Hessian:   H = -X'ΛX             where Λ = diag(λᵢ)
step:      β_new = β + (X'ΛX)⁻¹X'(y-λ)
```

### Overdispersion Test (Poisson vs NB)
```
φ̂ = Σᵢ [(yᵢ - λᵢ)² - yᵢ] / (Σᵢ λᵢ²)   [Cameron & Trivedi 1990 score test]
```
If φ̂ >> 0: use Negative Binomial.

### Exposure Offset
With known exposure eᵢ:
```
log(λᵢ) = log(eᵢ) + x'β
```
Add `log(eᵢ)` as fixed offset to linear predictor; wᵢ unchanged.

---

## 5. Negative Binomial Regression (NB2)

### Source
Lawless (1987) *Technometrics*; Cameron & Trivedi (1986, 2013); Hilbe (2011).

### Model
```
P(y=k | μ, r) = Γ(k+r)/(Γ(r)k!) · (r/(r+μ))^r · (μ/(r+μ))^k
```
where r > 0 is the dispersion parameter (r → ∞ gives Poisson).

Log-PMF (numerically stable via log-gamma):
```
log P(y|μ,r) = lgamma(y+r) - lgamma(r) - lgamma(y+1)
             + r·log(r/(r+μ)) + y·log(μ/(r+μ))
```
where `lgamma` = log-Gamma function.

### NB2 parameterization
Variance: Var(y) = μ + μ²/r = μ(1 + μ/r).  
NB2 = quadratic mean-variance; NB1 = linear mean-variance (different parameterization).

### IRLS + Profile Likelihood for r
1. Fix r, run IRLS for β (like Poisson but with different weights):
   ```
   wᵢ = μᵢ / (1 + μᵢ/r)    [NB2 IRLS weight]
   zᵢ = ηᵢ + (yᵢ - μᵢ)/wᵢ
   ```
2. Update r by maximizing profile log-likelihood:
   ```
   ∂ℓ/∂r = Σᵢ [ψ(yᵢ+r) - ψ(r) + log(r/(r+μᵢ)) + (μᵢ-yᵢ)/(r+μᵢ)]  = 0
   ```
   where ψ = digamma function.  
   Solve with Newton-Raphson: 1D, safe to bound r ∈ (ε, 1e6).
3. Alternate β and r updates until both converge.

**Practical**: 5-10 outer iterations (r updates) × 5-15 inner IRLS steps.

### Log-likelihood for r Newton step
```
∂²ℓ/∂r² = Σᵢ [ψ₁(yᵢ+r) - ψ₁(r) - (yᵢ+r)/(r+μᵢ)² + (μᵢ-yᵢ)·r/(r+μᵢ)²]
```
where ψ₁ = trigamma function.

---

## 6. Ridge Regression (L2 Regularization)

### Source
Hoerl & Kennard (1970) *Technometrics*; Hastie, Tibshirani & Friedman (2009) §3.4.

### Model
```
β̂_ridge(λ) = argmin ||y - Xβ||² + λ||β||²
             = (X'X + λI)⁻¹ X'y
```
**Note**: regularization should NOT apply to intercept. Standard practice: center y and X, fit without intercept, then recover intercept from means.

### From Sufficient Statistics
```
β̂_ridge = (G + λI)⁻¹ c
```
where G = X'X, c = X'y. Requires only one additional scalar λ on diagonal.

**Accumulate**: same (G, c) as OLS — no re-scan needed for different λ values.

### Effective Degrees of Freedom
```
df(λ) = trace((X'X + λI)⁻¹ X'X) = Σⱼ dⱼ²/(dⱼ² + λ)
```
where dⱼ are singular values of X (eigenvalues of X'X).

```
RSS(λ) = ||y - Xβ̂_ridge||²
σ̂² = RSS / (n - df(λ))   [effective df correction]
```

### GCV for λ Selection
Generalized cross-validation (Golub, Heath & Wahba 1979):
```
GCV(λ) = RSS(λ) / n / (1 - df(λ)/n)²
```
Minimize over λ. This approximates leave-one-out CV without refitting.

Efficient computation: use SVD X = UDV'.
```
α = D'U'y    [rotation of y]
β̂_ridge = V · diag(dⱼ/(dⱼ²+λ)) · α
RSS(λ) = ||y - UDV'β̂_ridge||² = Σⱼ λ²αⱼ²/(dⱼ²+λ)² + ||y⊥||²
df(λ) = Σⱼ dⱼ²/(dⱼ²+λ)
```

### λ Search Strategy
Grid: 100 log-spaced values λ ∈ [λ_min, λ_max].
- λ_max ≈ ||X'y||∞ / n (at this λ, β̂ ≈ 0)
- λ_min = λ_max · 1e-4

Accumulate+Gather:
```
1. accumulate(rows, x_i⊗x_i, sum) → G; accumulate(rows, x_i·y_i, sum) → c
2. SVD of G (or X) — one-time cost
3. gather(β̂_ridge, from SVD rotated solve for each λ)  [O(p) per λ]
```

---

## 7. Lasso Regression (L1 Regularization)

### Source
Tibshirani (1996) *JRSS-B*; Efron et al. (2004) LARS; Friedman, Hastie & Tibshirani (2010) *JSS* (coordinate descent).

### Model
```
β̂_lasso(λ) = argmin (1/2n)||y - Xβ||² + λ||β||₁
```

### Soft-Thresholding Operator
The closed-form solution for a 1D lasso subproblem:
```
S(z, λ) = sign(z) · max(|z| - λ, 0)
```
This is the key primitive. Every coordinate descent step is just S applied to a partial residual.

### Coordinate Descent (Friedman et al. 2010)
For each j = 1,...,p (cycle until convergence):
```
rⱼ = y - Σ_{k≠j} xₖ βₖ    [partial residual]
z_j = xⱼ'rⱼ / n           [OLS update for coordinate j]
β_j ← S(z_j, λ) / (xⱼ'xⱼ/n)
```

**With precomputed gram matrix** (recommended for p << n):
```
Precompute: G = X'X/n, c = X'y/n
rⱼ correlations: c_j - Σ_{k≠j} G_{jk} β_k
β_j ← S(c_j - Σ_{k≠j} G_{jk} β_k, λ) / G_{jj}
```
Cost per pass: O(p²) with precomputed G (or O(np) if G not precomputed).

**Convergence criterion**: max_j |β_j_new - β_j_old| < tol (1e-6).

### Path Algorithm (warm starts)
For regularization path λ_max ≥ λ₁ ≥ ... ≥ λ_K:
- Start at β = 0 with λ = λ_max = ||X'y||∞ / n
- Decrease λ geometrically
- Warm-start from previous solution at each λ

This is O(K × iterations) but very fast because few coefficients change per λ step.

### LARS (Least Angle Regression)
Efron et al. (2004): gives exact lasso path without grid search.
```
Algorithm:
1. Start β = 0, residual r = y
2. Find j* = argmax_j |x_j'r|  (most correlated predictor)
3. Move β in direction of x_{j*} until correlation with some other j ties
4. Add new predictor to active set, continue
5. On "drop event" (a coefficient passes through 0): remove from active set
```
LARS path has at most min(n,p) steps, each O(p² + active_set³).

**For p >> n**: use coordinate descent. **For exact path**: use LARS.

### Accumulate+Gather
```
1. accumulate(rows, x_i⊗x_i, sum) → G; accumulate(rows, x_i·y_i, sum) → c
2. Coordinate descent: scatter(β, S(c_j - G_j'β + G_{jj}β_j, λ)/G_{jj}, j)
3. gather(β̂_lasso, from converged coordinate descent)
```
Pattern: **scatter+gather** per coordinate (Kingdom B with active set selection).

---

## 8. ElasticNet

### Source
Zou & Hastie (2005) *JRSS-B*.

### Model
```
β̂_en(λ, α) = argmin (1/2n)||y - Xβ||² + λ[α||β||₁ + (1-α)||β||²/2]
```
α=1: Lasso. α=0: Ridge.

### Coordinate Descent Update
```
β_j ← S(c_j - Σ_{k≠j} G_{jk} β_k, λα) / (G_{jj} + λ(1-α))
```
The soft-threshold threshold is `λα`; the denominator gains `λ(1-α)` from the L2 term.

### Naive ElasticNet
Zou & Hastie (2005) note that the "naive" ElasticNet double-shrinks. To correct:
```
β̂_en_corrected = (1 + λ(1-α)) · β̂_en
```
This is optional but gives better predictive performance.

### λ Path
Same warm-start strategy as Lasso. Two hyperparameters: (λ, α).
In practice: fix α ∈ {0.1, 0.5, 0.9, 1.0} and tune λ by CV for each α.

---

## 9. Regression Diagnostics (shared across all)

### Residuals
```
Raw: eᵢ = yᵢ - ŷᵢ
Standardized: eᵢ / (σ̂ · sqrt(1 - hᵢᵢ))    where hᵢᵢ = xᵢ'(X'X)⁻¹xᵢ  [leverage]
Studentized: eᵢ / (σ̂₍ᵢ₎ · sqrt(1 - hᵢᵢ))  where σ̂₍ᵢ₎ excludes obs i
```

### Leverage (hat matrix diagonal)
```
hᵢᵢ = xᵢ' (X'X)⁻¹ xᵢ
```
After Cholesky L = chol(X'X), solve L'vᵢ = xᵢ → hᵢᵢ = ||vᵢ||²
**No need to form full hat matrix.**

### Cook's Distance
```
Dᵢ = (eᵢ/(p·σ̂²)) · hᵢᵢ/(1-hᵢᵢ)²    (simplified form, obs i is influential if D > 1)
```

### VIF (Variance Inflation Factor)
For each predictor j:
```
VIF_j = 1 / (1 - R²_j)
```
where R²_j is R² from regressing xⱼ on all other predictors.  
VIF > 10 indicates severe multicollinearity.  
Computed via diagonal of (X'X)⁻¹ vs individual regressions.

---

## 10. Cross-Validation (all regression models)

### k-Fold CV
```
For fold = 1..k:
  Train on all-but-fold
  Predict on fold
  CV_error = mean(|yᵢ - ŷᵢ|²) over test obs

CV score = mean(CV_error over folds)
```
Standard: k=5 or k=10.

### Leave-One-Out (LOO) — OLS only
Using Sherman-Morrison-Woodbury (no refitting):
```
LOO error_i = eᵢ / (1 - hᵢᵢ)
LOO-CV = (1/n) Σᵢ [eᵢ/(1-hᵢᵢ)]²
```
This is equivalent to refitting n models but costs O(p²) after one Cholesky.

### AIC/BIC for Model Selection (prefer to CV when possible)
```
AIC = n·log(RSS/n) + 2k
BIC = n·log(RSS/n) + k·log(n)
AICc = AIC + 2k(k+1)/(n-k-1)    [small-sample corrected, use when n/k < 40]
```
Note: AICc should be preferred in virtually all practical cases.

---

## 11. Sufficient Statistics Master Table

| Algorithm | Sufficient Stats | Accumulate Expression |
|---|---|---|
| OLS | (X'X, X'y, y'y, n) | accumulate(rows, (x⊗x, xy, y², 1), sum) |
| WLS | (X'WX, X'Wy, y'Wy, n) | accumulate(rows, w·(x⊗x, xy, y², 1), sum) |
| Logistic | (X'WX, X'(y-π)) per iteration | accumulate(rows, (w_i·x⊗x, x·(y-π)), sum) |
| Poisson | (X'ΛX, X'(y-λ)) per iteration | accumulate(rows, (λ_i·x⊗x, x·(y-λ)), sum) |
| NB2 | Same as Poisson + r | Same + digamma sums for r update |
| Ridge | Same as OLS (G + λI) | Same G, add λ to diagonal |
| Lasso | (G, c) + soft-threshold | accumulate(rows, x⊗x, sum) once; scatter per coordinate |
| ElasticNet | Same as Lasso | Same, modified denominator |

---

## 12. Numerical Implementation Notes

### Centering and Scaling (CRITICAL for regularized regression)
Before Ridge/Lasso/ElasticNet:
```
x̃_j = (xⱼ - x̄ⱼ) / s_j    [standardize each predictor]
ỹ = y - ȳ                   [center response]
```
Fit on (x̃, ỹ) without intercept. Recover original β:
```
β_j_orig = β̃_j / s_j
β₀ = ȳ - Σⱼ β̃_j · x̄_j / s_j
```

**Why**: Without scaling, λ has different effective strength per predictor. The l1/l2 penalties treat all coefficients equally — they must be on the same scale.

### Cholesky vs Normal Equations
- When p < 1000 and X is full rank: Cholesky of G = X'X (O(p³) one-time)
- When p > n: use QR factorization of X instead (X = QR, β = R⁻¹Q'y)
- When p >> n or near-singular: use SVD (robust but expensive)
- Always: check condition number of G before solving

### IRLS Numerical Stability (GLMs)
- Clip πᵢ ∈ [ε, 1-ε] before computing weights wᵢ = πᵢ(1-πᵢ) to avoid division by zero
- Cap working response |zᵢ| at some bound (e.g., 10) to handle extreme cases
- Initialize with β = 0 and π = ȳ (sample mean) for logistic regression
- Log-sum-exp trick for log(1+exp(η)) stability

### Coordinate Descent Initialization
For cold start: β = 0 (automatically feasible for λ ≥ λ_max).
For warm start: previous λ solution (must be feasible, always is with gradient descent).

---

## 13. Accumulate+Gather Patterns Summary

| Algorithm | Kingdom | Pattern |
|---|---|---|
| OLS | A | linear scan → Cholesky solve |
| WLS | A | linear scan with weights |
| Logistic | AC | iterative linear scan (IRLS) |
| Poisson | AC | iterative linear scan (IRLS) |
| NB2 | AC | nested iterative (IRLS outer + r Newton) |
| Ridge | A | OLS gram + diagonal regularization |
| Lasso | BC | coordinate descent = scatter + gather, repeated |
| ElasticNet | BC | same as Lasso, modified soft-threshold |

**Kingdom A** = solve (linear system, one pass)  
**Kingdom C** = iterate (fixed-point convergence)  
**Kingdom B** = partition (active set management in coordinate descent)

---

*All formulas above are paper-verifiable against cited primary sources.*  
*This document is the implementation contract for pathmaker's Tier 1 regression work.*
