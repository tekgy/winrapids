# F10 Regression — GLM, Logistic, Quantile, WLS/GLS, Diagnostics

Created: 2026-04-01
By: scout (supplement to gold-standards-f10-regression.md)
Session: Day 1 tambear-math expedition

---

## The Structural Insight: GLM IRLS = Robust IRLS = EM M-step

All three families (GLMs, robust regression F09, mixture models F16) are the SAME
weighted scatter accumulate with different weight functions:

```
Iteration:
  weights wᵢ = f(μᵢ, yᵢ)          // weight formula differs by algorithm
  β̂_new = (X'WX)^{-1} X'Wz        // weighted GramMatrix solve
  μ = g^{-1}(Xβ̂_new)              // predict (link function inversion)
```

Where:
- **GLM IRLS**: wᵢ = 1/Var(Y|xᵢ)·(g'(μᵢ))², z = adjusted dependent variable
- **Robust IRLS**: wᵢ = ψ(εᵢ/σ)/(εᵢ/σ), z = y (original response)
- **EM M-step**: wᵢₖ = rᵢₖ (posterior responsibility), z = x (original data)

`scatter_multi_phi_weighted` from the F16 navigator doc covers ALL THREE with different weight vectors.

---

## GLM Gold Standards: statsmodels is the Oracle

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# General GLM interface:
model = sm.GLM(y, X, family=sm.families.Binomial())
result = model.fit()
result.summary()
result.params          # β̂
result.bse             # SE(β̂)
result.tvalues         # z-statistics (not t — GLMs use z)
result.pvalues         # p-values
result.deviance        # -2*loglik (model deviance)
result.null_deviance   # -2*loglik (null model, intercept only)
result.df_resid        # residual degrees of freedom = n - p

# GLM families available:
sm.families.Binomial()          # logistic regression (binary outcome)
sm.families.Binomial(link=sm.families.links.Probit())  # probit regression
sm.families.Poisson()           # Poisson regression (count outcome)
sm.families.NegativeBinomial()  # negative binomial (overdispersed counts)
sm.families.Gamma()             # gamma regression (positive continuous)
sm.families.InverseGaussian()   # inverse Gaussian
sm.families.Gaussian()          # OLS (GLM with identity link = standard regression)
sm.families.Tweedie(link=..., var_power=1.5)  # Tweedie compound Poisson-Gamma
```

---

## Logistic Regression

### Binary Logistic (most common)

```python
# sklearn (fast, no SEs):
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1e9, solver='lbfgs', max_iter=1000)
# C=1e9 = very weak regularization ≈ unregularized OLS equivalent
# Default in sklearn: C=1.0 (L2 regularized!) — this is a TRAP
lr.fit(X, y)
lr.coef_          # β̂ (shape (1, p) for binary, not (p,))
lr.intercept_     # β₀
lr.predict_proba(X)  # sigmoid(Xβ): shape (n, 2)

# statsmodels (full inference, no regularization by default):
model = sm.Logit(y, sm.add_constant(X))
result = model.fit()
result.params     # β̂
result.bse        # SE (from Hessian, not OLS formula)
result.llf        # log-likelihood
np.exp(result.params)  # odds ratios (OR)
result.conf_int()  # CI for β; exp(result.conf_int()) = CI for OR
```

**Critical trap**: `sklearn.LogisticRegression` applies L2 regularization by default (C=1.0).
This is different from R's `glm(y ~ x, family=binomial)` which has NO regularization.
For matching R's output: use `C=1e8` in sklearn or use statsmodels.

### R: glm (canonical)

```r
# Binary logistic:
fit <- glm(y ~ x1 + x2, data=df, family=binomial(link="logit"))
fit <- glm(y ~ x1 + x2, data=df, family=binomial(link="probit"))

# Output:
coef(fit)                           # β̂
summary(fit)$coefficients           # β̂, SE, z, p
exp(coef(fit))                      # odds ratios
exp(confint(fit))                   # CI for OR

fitted(fit)                          # μ̂ = predicted probabilities
predict(fit, type="response")        # same as fitted()
predict(fit, type="link")            # linear predictor Xβ̂ (log-odds)

# Model comparison:
AIC(fit)                             # -2*loglik + 2*k
fit$null.deviance - fit$deviance     # deviance reduction (like R²)
fit$df.null - fit$df.residual        # df reduction
pchisq(fit$null.deviance - fit$deviance, df=fit$df.null - fit$df.residual, lower.tail=FALSE)
```

### Multinomial Logistic

```python
# sklearn:
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(multi_class='multinomial', solver='lbfgs', C=1e9)
lr.fit(X, y)            # y ∈ {0, 1, ..., K-1}
lr.coef_                # shape (K, p) — K coefficient vectors

# statsmodels:
from statsmodels.discrete.discrete_model import MNLogit
model = MNLogit(y, sm.add_constant(X))
result = model.fit()
result.params           # shape (p+1, K-1) — K-1 due to reference category
```

```r
library(nnet)
fit <- multinom(y ~ x1 + x2, data=df)  # multinomial logistic
```

### Ordinal Logistic (proportional odds model)

```python
# statsmodels:
from statsmodels.miscmodels.ordinal_model import OrderedModel
model = OrderedModel(y, X, distr='logit')  # or distr='probit'
result = model.fit()
result.params    # β̂ + threshold parameters
```

```r
library(MASS)
fit <- polr(y ~ x, data=df, method="logistic")  # proportional odds
```

### Conditional Logistic (matched case-control)

```python
from lifelines.statistics import conditional_logistic_regression
# OR:
from statsmodels.discrete.conditional_models import ConditionalLogit
model = ConditionalLogit(y, X, groups=strata_id)
```

```r
library(survival)
fit <- clogit(y ~ x + strata(case_id), data=df)
```

---

## Poisson and Count Models

### Poisson Regression

```python
model = sm.GLM(y, X, family=sm.families.Poisson())
result = model.fit()
# Link = log: E[Y|X] = exp(Xβ̂)
# Residual deviance: 2 * Σ [y_i * log(y_i/μ_i) - (y_i - μ_i)]
```

```r
fit <- glm(y ~ x, family=poisson(link="log"))
# Overdispersion check: deviance/df_resid > 2 → consider neg. binomial
```

**Overdispersion test**: `AER::dispersiontest(fit)` in R. If p < 0.05, use negative binomial.

### Negative Binomial

```python
from statsmodels.discrete.discrete_model import NegativeBinomial
model = NegativeBinomial(y, X)
result = model.fit()
result.params[-1]   # log(α) where α = overdispersion parameter
```

```r
library(MASS)
fit <- glm.nb(y ~ x)    # negative binomial, estimates θ (dispersion)
fit$theta              # overdispersion parameter θ (variance = μ + μ²/θ)
```

### Zero-Inflated Models

```python
from statsmodels.discrete.count_model import (
    ZeroInflatedPoisson,
    ZeroInflatedNegativeBinomialP,
    ZeroInflatedGeneralizedPoisson,
)
model = ZeroInflatedPoisson(y, X)
result = model.fit()
```

```r
library(pscl)
fit <- zeroinfl(y ~ x, data=df, dist="poisson")   # ZIP
fit <- zeroinfl(y ~ x, data=df, dist="negbin")    # ZINB
fit <- hurdle(y ~ x, data=df, dist="poisson")     # hurdle model
```

**ZIP vs Hurdle**:
- ZIP: mixture of Poisson and point mass at 0. Zeros can come from either component.
- Hurdle: separate models for zero/nonzero. Zero-truncated Poisson for counts > 0.

---

## Quantile Regression

Minimizes the **pinball (check) loss** instead of squared error:
```
L_τ(ε) = τ·ε·1(ε≥0) + (τ-1)·ε·1(ε<0)  = τ·max(ε,0) + (1-τ)·max(-ε,0)
```

For τ=0.5: median regression. For τ=0.9: 90th percentile regression.

```python
import statsmodels.formula.api as smf

# Linear quantile regression:
model = smf.quantreg('y ~ x1 + x2', data=df)
result = model.fit(q=0.5)   # median regression
result_90 = model.fit(q=0.9)  # 90th percentile
result.params    # β̂ at quantile q
result.summary()

# Fit multiple quantiles:
quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
results = [model.fit(q=q) for q in quantiles]
```

```r
library(quantreg)
fit <- rq(y ~ x, tau=0.5)     # median regression
fit <- rq(y ~ x, tau=0.9)     # 90th percentile
rq(y ~ x, tau=c(0.25, 0.5, 0.75))  # multiple quantiles at once
summary(fit, se="boot")       # bootstrap SEs
```

**Algorithm**: linear programming (simplex or interior point).
NOT a weighted accumulate — quantile regression is an LP problem.
For tambear: quantile regression may need a separate LP solver (not in accumulate family).
Phase 1: use a CPU-side LP library (GLPK, HiGHS). Phase 2: GPU-accelerated LP.

**Tambear decomposition**: quantile regression is Kingdom C (iterative) but the inner step
is an LP, not a weighted GramMatrix solve. Different algorithmic structure from GLM/IRLS.

---

## WLS and GLS

### Weighted Least Squares (WLS)

```python
# statsmodels:
model = sm.WLS(y, X, weights=w)     # w_i = inverse variance of i-th observation
result = model.fit()
```

```r
fit <- lm(y ~ x, weights=w)   # built into base lm()
```

WLS = OLS with W = diag(w) included. Same IRLS framework but weights are FIXED (not iterated).
`β̂_WLS = (X'WX)^{-1} X'Wy` — exactly the weighted GramMatrix solve from F09's IRLS.

**Tambear**: WLS = one-shot weighted GramMatrix + solve. Already in F09 infrastructure.

### Generalized Least Squares (GLS)

```python
model = sm.GLS(y, X, sigma=Sigma)   # Sigma = covariance matrix of errors
result = model.fit()
# GLS β̂ = (X'Σ^{-1}X)^{-1} X'Σ^{-1}y
```

GLS requires a known (or estimated) error covariance Σ. Special cases:
- WLS: Σ = diag(1/w) (known heteroscedasticity)
- Feasible GLS (FGLS): Σ is estimated, then GLS applied
- GLS with AR(1) errors: `statsmodels.GLSAR`

For tambear: GLS requires Σ^{-1}X — a matrix multiply. If Σ = diag(w), reduces to WLS.
Full non-diagonal Σ requires computing Cholesky(Σ), transforming X and y.

---

## Diagnostic Tests

### Variance Inflation Factor (VIF)

```python
from statsmodels.stats.outliers_influence import variance_inflation_factor
# VIF for each feature:
vif_data = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
# VIF > 10: severe multicollinearity. VIF > 5: moderate.
```

```r
library(car)
car::vif(lm_fit)
```

VIF = 1/(1-R²_j) where R²_j is R² from regressing feature j on all other features.
Requires running p separate regressions — GramMatrix can be reused for all of them.

### Durbin-Watson Test (autocorrelation in residuals)

```python
from statsmodels.stats.stattools import durbin_watson
dw = durbin_watson(result.resid)  # DW statistic: 0-4, 2 = no autocorrelation
# DW < 1.5 or DW > 2.5 → potential autocorrelation
```

```r
library(lmtest)
lmtest::dwtest(fit)  # exact DW test with p-value
```

DW = Σᵢ(εᵢ - εᵢ₋₁)² / Σεᵢ². Pure arithmetic on residuals. O(n). No new primitives.

### Breusch-Pagan Test (heteroscedasticity)

```python
from statsmodels.stats.diagnostic import het_breuschpagan
lm, lm_pvalue, fvalue, f_pvalue = het_breuschpagan(result.resid, X)
# H₀: homoscedasticity. Small p-value → heteroscedastic.
```

```r
library(lmtest)
lmtest::bptest(fit)
```

Algorithm: regress ε² on X, test if R² is significantly > 0 via F-test.
Requires: scatter of ε² = F06 accumulate on residuals, then F07 F-test.

### White Test (heteroscedasticity, robust)

```python
from statsmodels.stats.diagnostic import het_white
lm, lm_pvalue, fvalue, f_pvalue = het_white(result.resid, result.model.exog)
```

Extends Breusch-Pagan by including quadratic terms. Same accumulate structure.

### Cook's Distance (influence measure)

```python
from statsmodels.stats.outliers_influence import OLSInfluence
influence = OLSInfluence(result)
cooks_d = influence.cooks_distance[0]   # shape (n,)
# Cook's D > 4/n: potentially influential. > 1: very influential.
```

Cook's D = (β̂_{-i} - β̂)' X'X (β̂_{-i} - β̂) / (p·MSE) — leave-one-out coefficient change.
Efficient computation: `D_i = h_ii · r_i² / (p · (1-h_ii)²)` where r_i = studentized residual.
Requires leverage h_ii = x_i' (X'X)^{-1} x_i. O(np²) to compute all leverages.

---

## Parity Test Targets

```python
import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression

np.random.seed(42)
n = 200
X = np.random.randn(n, 2)
# Logistic:
p_true = 1 / (1 + np.exp(-(1.5 + 2.0*X[:,0] - 1.0*X[:,1])))
y_binary = (np.random.rand(n) < p_true).astype(float)

# Oracle:
logit = sm.Logit(y_binary, sm.add_constant(X)).fit(disp=0)
print("statsmodels:", logit.params)

# Compare sklearn (need high C to match unregularized):
lr = LogisticRegression(C=1e8, solver='lbfgs', max_iter=1000)
lr.fit(X, y_binary)
print("sklearn coef:", lr.coef_, lr.intercept_)
# These should be close but NOT identical (sklearn uses BFGS, statsmodels uses Newton-Raphson)
# Match to within 1e-4 (both should converge to same MLE)
```

---

## Tambear Decomposition Summary

| Algorithm | Primitive | New vs existing |
|-----------|----------|----------------|
| OLS | GramMatrix + Cholesky | F10 existing |
| WLS | Weighted GramMatrix | F09 extension |
| GLS | Transform + OLS | Cholesky pre-transform |
| Ridge | GramMatrix + λI + Cholesky | F10 existing |
| LASSO | Coordinate descent | NEW: LP/coord descent primitive |
| Logistic (IRLS) | Weighted GramMatrix (F09) | F09 extension, different w |
| Poisson (IRLS) | Weighted GramMatrix (F09) | F09 extension, different w |
| Negative binomial | IRLS + θ estimation | F09 extension |
| Quantile | Linear programming | NEW: LP primitive |
| VIF | p separate OLS (GramMatrix) | F10 reuse |
| Cook's D | Leverage from (X'X)^{-1} | F10 existing |
| Durbin-Watson | Arithmetic on residuals | trivial |
| Breusch-Pagan | Scatter ε², then F07 F-test | F06+F07 reuse |

**Phase 1 scope**: OLS + WLS + Logistic + Poisson (all IRLS, same primitive).
**Phase 2**: Negative binomial, zero-inflated, quantile, multinomial.
**Deferred**: GLS with non-diagonal Σ, conditional logistic.
