# F10 Regression — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 10 (Regression: all types + diagnostics).
This family is the GramMatrix spine — OLS, Ridge, diagnostics all derive from X'X.
Unblocks F09 (Robust IRLS starting values = OLS from GramMatrix).

---

## Critical Trap: sklearn vs R Coefficient Comparison

**sklearn and R produce identical OLS coefficients** when data is unscaled.
But there are several silent differences:

1. **Intercept handling**: R's `lm(y ~ x)` includes intercept by default. sklearn `LinearRegression` also includes intercept by default (`fit_intercept=True`). Both augment X with a column of ones — the difference is where the intercept lands.

2. **Normalization**: sklearn's `normalize` parameter was **removed in sklearn 1.2** (2022). Code using `normalize=True` will crash. Use `StandardScaler` pipeline instead. R's `lm()` never normalizes by default.

3. **Coefficient extraction order**: R returns `(Intercept), x1, x2, ...` sklearn returns `coef_` = (x1, x2, ...) and `intercept_` separately.

4. **Standard errors**: R `summary(fit)$coefficients[, "Std. Error"]` includes SE for intercept. sklearn does NOT compute SEs — use `statsmodels` for those.

---

## R: stats::lm — The OLS Gold Standard

```r
# Basic OLS:
fit <- lm(y ~ x1 + x2)          # with intercept
fit <- lm(y ~ x1 + x2 - 1)      # no intercept
fit <- lm(y ~ ., data=df)        # all columns as predictors

# Coefficient access:
coef(fit)                         # β̂ vector including intercept
fitted(fit)                        # ŷ = Xβ̂
residuals(fit)                     # ε = y - ŷ

# Full summary:
s <- summary(fit)
s$coefficients                    # β̂, SE(β̂), t-stats, p-values
s$r.squared                       # R²
s$adj.r.squared                   # Adjusted R²
s$sigma                           # residual standard error = sqrt(RSS/(n-p-1))
s$fstatistic                      # F-stat and its df1, df2
s$df                              # residual degrees of freedom

# Design matrix:
model.matrix(fit)                 # X matrix actually used (with intercept column)

# Residual analysis:
hatvalues(fit)                    # leverage h_ii = diag(H) = diag(X(X'X)^{-1}X')
cooks.distance(fit)               # Cook's D = influence measure
rstudent(fit)                     # externally studentized residuals
vif(fit)                          # variance inflation factors (from car package)
```

**R uses QR decomposition internally**, not normal equations. `qr(fit)` exposes the QR factor.
The normal equations path (X'X then solve) is LESS stable — Hilbert matrix X will expose this.

### R Diagnostic Functions

```r
# Tests for regression assumptions:
lmtest::dwtest(fit)               # Durbin-Watson (autocorrelation in residuals)
lmtest::bptest(fit)               # Breusch-Pagan (heteroscedasticity)
car::ncvTest(fit)                 # Non-constant variance test
shapiro.test(residuals(fit))      # Normality of residuals

# Confidence/prediction intervals:
predict(fit, newdata, interval="confidence")  # CI for E[y|X]
predict(fit, newdata, interval="prediction")  # PI for individual y|X
```

---

## Python: sklearn.linear_model

```python
from sklearn.linear_model import (
    LinearRegression,     # OLS — no regularization
    Ridge,                # L2 regularization
    Lasso,                # L1 regularization
    ElasticNet,           # L1+L2 (linear combination)
    RidgeCV,              # Ridge with cross-validated alpha
    LassoCV,              # Lasso with cross-validated alpha
    BayesianRidge,        # Bayesian regularization (marginal likelihood hyperparams)
    HuberRegressor,       # Robust M-estimator (F09 boundary)
    Lars,                 # Least Angle Regression
    OrthogonalMatchingPursuit,  # Sparse OMP
)

# OLS:
lr = LinearRegression(fit_intercept=True)
lr.fit(X, y)
lr.coef_           # shape (p,) — excludes intercept
lr.intercept_      # scalar
lr.score(X, y)     # R²

# NO standard errors in sklearn. Use statsmodels:
import statsmodels.api as sm
X_with_const = sm.add_constant(X)    # prepend intercept column
model = sm.OLS(y, X_with_const)
result = model.fit()
result.summary()                      # full table: β, SE, t, p, CI
result.params                         # β̂
result.bse                            # SE(β̂)
result.tvalues                        # t-statistics
result.pvalues                        # p-values
result.rsquared                       # R²
result.rsquared_adj                   # adjusted R²
result.fvalue                         # F-statistic
result.mse_resid                      # σ² = RSS/(n-p-1)
```

### Ridge Regression

```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1.0)    # α = regularization strength = λ in statistics
ridge.fit(X, y)
# Solves: β̂ = (X'X + αI)^{-1} X'y
# Same GramMatrix! Just add α to diagonal before Cholesky solve.
```

```r
# R ridge options:
library(MASS)
MASS::lm.ridge(y ~ ., data=df, lambda=1.0)  # lambda = α

library(glmnet)
cv_ridge <- cv.glmnet(X, y, alpha=0)         # alpha=0 = Ridge (glmnet convention)
ridge <- glmnet(X, y, alpha=0, lambda=0.1)
coef(ridge)                                   # β̂ (note: glmnet centers/scales internally)
```

**Glmnet convention**: `alpha=0` = Ridge, `alpha=1` = LASSO, `alpha=0.5` = ElasticNet.
**Sklearn convention**: `Ridge(alpha=λ)` uses α directly as the regularization weight.
These are the SAME algorithm, just different symbol choices.

### LASSO

```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=0.1)    # α = λ regularization strength
lasso.fit(X, y)
# Solves: β̂ = argmin ||y - Xβ||² + 2αn||β||₁
# Note: sklearn multiplies by 2n internally — watch for factor differences vs R
```

```r
cv_lasso <- cv.glmnet(X, y, alpha=1)
lasso <- glmnet(X, y, alpha=1, lambda=cv_lasso$lambda.min)
```

**Critical**: sklearn's LASSO objective has a factor of 2n that glmnet does NOT include.
`sklearn Lasso(alpha=0.1)` ≈ `glmnet(lambda=0.1/(2*nrow(X)))` for matching coefficients.
Always verify via `coef_` comparison before using either as oracle.

### ElasticNet

```python
from sklearn.linear_model import ElasticNet

en = ElasticNet(alpha=0.1, l1_ratio=0.5)
# Objective: (1/2n)||y-Xβ||² + α*l1_ratio*||β||₁ + α*(1-l1_ratio)/2*||β||²
```

```r
glmnet(X, y, alpha=0.5, lambda=...)  # alpha interpolates L1-L2
```

---

## GramMatrix Path: OLS from Normal Equations

This is the tambear-native path. Numerically inferior to QR but directly derived from
the tiled accumulate infrastructure.

```
X'X = accumulate(Tiled(p, p), DotProduct, Add)   // GramMatrix
X'y = accumulate(Tiled(p, 1), DotProduct, Add)   // cross-product vector
β̂   = (X'X)^{-1} X'y                             // Cholesky solve

Alternatively (augmented):
[X'X  X'y] = accumulate(Tiled(p+1, p+1), DotProduct, Add)  // one pass for both
```

**Stability caveat**: for ill-conditioned X (near-collinear features), X'X amplifies the
condition number — κ(X'X) = κ(X)². For well-conditioned X (typical financial features),
normal equations are fine. Add condition number check: warn if κ(X'X) > 1e12.

```python
# Check condition number:
import numpy as np
cond = np.linalg.cond(X.T @ X)
# κ > 1e12: consider QR or SVD for coefficient stability
```

---

## Diagnostic Quantities: All from GramMatrix + Residuals

Once β̂ is computed, all standard OLS diagnostics derive from the same GramMatrix.

```
ŷ = Xβ̂                          // fitted values
ε = y - ŷ                        // residuals
RSS = ε'ε = Σεᵢ²                  // residual sum of squares
SST = Σ(yᵢ - ȳ)²  = MomentStats(order=2, All).sum2  // total SS from MomentStats
TSS = SST                         // same thing, different names

R²  = 1 - RSS/SST
adj_R² = 1 - (RSS/(n-p-1)) / (SST/(n-1))
σ²  = RSS/(n-p-1)                 // residual variance
SE(β̂) = sqrt(σ² * diag((X'X)^{-1}))  // from (X'X)^{-1} already computed

F = (R²/p) / ((1-R²)/(n-p-1))   // F-test for overall regression significance
  = (MSR / MSE)
  where MSR = (SST - RSS)/p,  MSE = RSS/(n-p-1)

// This is the ANOVA F-test — same formula as F07's one-way ANOVA F
// For one predictor: F = t²  (same identity as F07 scout found)
```

**Hat matrix** (leverage):
```
H = X(X'X)^{-1}X'                // n×n hat matrix — expensive for large n
h_ii = diag(H)                    // leverage — only the diagonal needed
     = row-wise: h_ii = xᵢ'(X'X)^{-1}xᵢ  // one matvec per row — O(np²)
```

---

## Python: statsmodels — Full Diagnostic Output

statsmodels is the Python gold standard for OLS diagnostics (sklearn has no SEs):

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Formula API (R-like):
result = smf.ols('y ~ x1 + x2', data=df).fit()
result.summary()        # full R-style output table

# NumPy array API:
X = sm.add_constant(X_array)   # prepend 1s for intercept
result = sm.OLS(y, X).fit()

# Key diagnostic attributes:
result.params            # β̂
result.bse               # SE(β̂)
result.tvalues           # t-statistics
result.pvalues           # p-values for H₀: βᵢ = 0
result.conf_int()        # 95% confidence intervals for β
result.rsquared          # R²
result.rsquared_adj      # adjusted R²
result.fvalue            # F-statistic
result.f_pvalue          # F-test p-value
result.mse_resid         # σ² = RSS/(n-p-1)
result.resid             # ε = y - ŷ
result.fittedvalues      # ŷ
result.influence_table() # leverage, studentized residuals, Cook's D
```

---

## Validation Targets

### Simple OLS (verify GramMatrix path):

```python
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm

np.random.seed(42)
X = np.random.randn(100, 3)
true_beta = np.array([1.5, -2.0, 0.5])
y = X @ true_beta + 0.5 * np.random.randn(100)

# sklearn:
lr = LinearRegression(fit_intercept=False)  # no intercept for clean comparison
lr.fit(X, y)
print("sklearn β̂:", lr.coef_)

# statsmodels:
result = sm.OLS(y, X).fit()
print("statsmodels β̂:", result.params)
print("statsmodels SE:", result.bse)
print("statsmodels R²:", result.rsquared)

# Tambear oracle values: capture from numpy normal equations for debugging
beta_numpy = np.linalg.lstsq(X, y, rcond=None)[0]
print("numpy lstsq β̂:", beta_numpy)
```

**Expected**: all three should match to < 1e-10. Tambear GramMatrix path should match to < 1e-8.

### Ridge (verify regularization path):

```python
from sklearn.linear_model import Ridge
ridge = Ridge(alpha=1.0, fit_intercept=False)
ridge.fit(X, y)

# Manual verification:
beta_ridge_manual = np.linalg.solve(X.T @ X + 1.0 * np.eye(3), X.T @ y)
print("sklearn:", ridge.coef_)
print("manual:", beta_ridge_manual)
# Should match exactly (Ridge uses Cholesky internally, same formula)
```

---

## F-test Identity (confirm with code before hardcoding):

```python
# For one predictor: F = t² (from F07 scout)
X1 = X[:, :1]
result_1d = sm.OLS(y, sm.add_constant(X1)).fit()
print(f"t² = {result_1d.tvalues[1]**2:.6f}")
print(f"F  = {result_1d.fvalue:.6f}")
# These should match to numerical precision
```

---

## Cross-Validation Utilities

```python
from sklearn.model_selection import cross_val_score, KFold

cv = KFold(n_splits=5, shuffle=True, random_state=42)
r2_scores = cross_val_score(LinearRegression(), X, y, cv=cv, scoring='r2')
rmse_scores = cross_val_score(LinearRegression(), X, y, cv=cv,
                               scoring='neg_root_mean_squared_error')
```

---

## Key Sharing Opportunities (F09, F22)

| Consumer | What it needs from F10 | Notes |
|----------|----------------------|-------|
| F09 IRLS | OLS β̂ as starting values | Warm start for M-estimation |
| F09 IRLS | GramMatrix already computed | Weighted update per iteration |
| F14 Factor Analysis | GramMatrix = correlation matrix | Same X'X, normalized |
| F22 PCA | GramMatrix = covariance matrix | Center first, then X'X |
| F33 Multivariate | GramMatrix generalized | Multiple response variables |
| F07 ANOVA | F-statistic = regression F | Same extraction from GramMatrix |

Once GramMatrix is cached in TamSession, F09, F14, F22, F33, F07 all get it for free.
The GramMatrix IS the MSR for the linear algebra kingdom.
