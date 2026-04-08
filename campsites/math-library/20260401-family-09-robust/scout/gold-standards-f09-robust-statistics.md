# F09 Robust Statistics — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 09 (Robust Statistics).
Prerequisite: F10 (Regression/GramMatrix) for IRLS starting values.
Key algorithms: trimmed mean, M-estimators, IRLS, robust location/scale.

---

## Why These Are Kingdom C (Iteration over Kingdom A)

Every robust estimator iterates over a weighted OLS step:
1. Compute weights wᵢ = ψ(εᵢ/σ) / (εᵢ/σ)   where ψ = influence function
2. Solve weighted OLS: β̂ₙₑₓₜ = (X'WX)^{-1} X'Wy   where W = diag(w)
3. Update residuals: εᵢ = yᵢ - xᵢ'β̂ₙₑₓₜ
4. Update scale: σ̂ = median(|εᵢ|) / 0.6745   (or similar robust scale)
5. Repeat until convergence

Kingdom A core: `X'WX = accumulate(Tiled, WeightedDotProduct, Add)` — same GramMatrix
kernel with per-row weights. Kingdom C wrapper: outer IRLS loop.

---

## Influence Functions / ψ Functions (the choices)

### Huber's ψ

```
ρ_Huber(e) = { e²/2                  if |e| ≤ k
              { k·|e| - k²/2          if |e| > k

ψ_Huber(e) = dρ/de = { e      if |e| ≤ k
                      { k·sign(e)  if |e| > k

w_Huber(e) = ψ(e)/e = { 1          if |e| ≤ k
                       { k/|e|      if |e| > k
```

Default k = 1.345 (95% efficiency under normality). This is `MASS::rlm(method="M")` default.

### Tukey Bisquare (biweight)

```
ρ_bisquare(e) = { k²/6 · [1 - (1 - (e/k)²)³]   if |e| ≤ k
                { k²/6                            if |e| > k

w_bisquare(e) = { (1 - (e/k)²)²   if |e| ≤ k
               { 0                 if |e| > k  ← hard rejection above cutoff
```

Default k = 4.685 (95% efficiency). Points with |e| > 4.685σ get ZERO weight — hard rejection.
This is asymptotically more efficient than Huber but can have multiple local optima.

### Andrews' Sine Wave

```
ψ_andrews(e) = { sin(πe/k)   if |e| ≤ k
               { 0             if |e| > k

w_andrews(e) = ψ(e)/e = { sin(πe/k)/(πe/k)   if |e| ≤ k
                         { 0                    if |e| > k
```

Default k = π (1.339). Hard rejection like bisquare.

---

## R: MASS::rlm — The Canonical M-Estimator

```r
library(MASS)

# Huber M-estimator:
fit <- rlm(y ~ x1 + x2, data=df)            # Huber, default k=1.345
fit <- rlm(y ~ x1 + x2, psi=psi.huber)      # explicit psi function
fit <- rlm(y ~ x1 + x2, psi=psi.bisquare)   # Tukey bisquare
fit <- rlm(y ~ x1 + x2, psi=psi.hampel)     # Hampel

# Output:
fit$coefficients     # β̂ (converged M-estimate)
fit$residuals        # εᵢ = yᵢ - ŷᵢ
fit$weights          # final wᵢ values
fit$s                # scale estimate σ̂
fit$wresid           # weighted residuals
summary(fit)         # coefficients with SE and t-values (robust SEs)

# Control convergence:
rlm(y ~ x, maxit=100, acc=1e-4)  # max iterations and convergence tolerance
```

**Key**: `rlm` uses IRLS internally — same algorithm as tambear's iterative accumulate loop.

```r
# Even more robust (S-estimator + MM-estimator):
library(robustbase)
lmrob(y ~ x1 + x2)
# Uses high-breakdown-point S-estimator for initial β̂, then MM-step for efficiency
```

---

## Python: statsmodels RLM — Most Complete M-Estimator Implementation

```python
import statsmodels.api as sm

X = sm.add_constant(X_array)
model = sm.RLM(y, X, M=sm.robust.norms.HuberT(t=1.345))
result = model.fit()

result.params          # β̂
result.resid           # residuals
result.weights         # final weights wᵢ
result.sresid          # standardized residuals εᵢ/σ̂
result.summary()       # full summary

# Norms available:
sm.robust.norms.HuberT(t=1.345)          # Huber
sm.robust.norms.TukeyBiweight(c=4.685)  # Tukey bisquare
sm.robust.norms.AndrewWave()             # Andrews sine
sm.robust.norms.RamsayE()                # Ramsay
sm.robust.norms.LeastSquares()           # OLS (special case, w=1)
sm.robust.norms.TrimmedMean(c=2.0)      # trimmed mean as M-estimator
```

### sklearn HuberRegressor (simpler but different objective)

```python
from sklearn.linear_model import HuberRegressor

# Note: sklearn's HuberRegressor uses a DIFFERENT objective than statsmodels RLM
# sklearn: min Σ huber_loss(y - Xβ) + α||β||²  (with optional L2 regularization)
# statsmodels RLM: IRLS with explicit ψ function, no regularization

hr = HuberRegressor(epsilon=1.35, max_iter=100)  # epsilon ≈ k in Huber ψ
hr.fit(X, y)
hr.coef_          # β̂
hr.intercept_
hr.outliers_      # bool array: True where weight=0
```

**Trap**: sklearn HuberRegressor and statsmodels RLM with Huber ψ give similar but NOT identical
coefficients. Use statsmodels RLM as the oracle for parity tests.

---

## Trimmed Mean and Winsorizing

### Trimmed Mean

Remove the lowest α and highest α fraction of data, compute mean of remainder.

```python
from scipy.stats import trim_mean
trim_mean(x, proportiontocut=0.1)  # trim 10% from each tail

# Manual for validation:
import numpy as np
x_sorted = np.sort(x)
n = len(x)
k = int(np.floor(n * 0.1))  # number to trim from each end
trimmed = x_sorted[k : n-k]
mean_trimmed = np.mean(trimmed)
```

```r
# R base:
mean(x, trim=0.1)   # trim=0.1 removes 10% from each tail
```

**Tambear path**: trimmed mean requires sorted order statistics → SortedPermutation MSR.
Alternatively: two-pass (ExtremaStats for quantile cutoffs, then filtered MomentStats).
Phase 1: use ExtremaStats min/max + alpha fraction computation (approximate).
Phase 2: exact trimmed mean via SortedPermutation.

### Winsorized Mean

Clamp extremes to cutoffs instead of removing them.

```python
from scipy.stats import mstats
mstats.winsorize(x, limits=[0.1, 0.1])  # clamp bottom/top 10%
np.mean(mstats.winsorize(x, limits=[0.1, 0.1]))  # winsorized mean
```

```r
library(DescTools)
DescTools::Winsorize(x, probs=c(0.05, 0.95))
mean(DescTools::Winsorize(x, probs=c(0.05, 0.95)))
```

**Tambear path**: same as trimmed mean — needs SortedPermutation for exact quantile cutoffs.

---

## Robust Scale Estimators

Used to estimate σ̂ in IRLS (denominator for standardized residuals).

### MAD (Median Absolute Deviation)

```
MAD = median(|xᵢ - median(x)|)
σ̂_MAD = MAD / 0.6745   (consistent estimator under normality)
```

```python
from scipy.stats import median_abs_deviation
sigma_hat = median_abs_deviation(x, scale='normal')  # divides by 0.6745
```

```r
mad(x)              # base R, scale=TRUE by default (multiplies by 1.4826 = 1/0.6745)
median(abs(x - median(x)))  # manual without scaling
```

**Tambear**: requires two passes — first compute median (SortedPermutation), then scatter |x - median| and compute median again.

### S_n Estimator (Rousseeuw & Croux 1993, breakdown 50%)

```
S_n = c · median_i { median_j |xᵢ - xⱼ| }
```

Requires O(n log n) time, breakdown point 50%. More efficient than MAD.
R: `robustbase::Sn(x)`.

### Q_n Estimator (Rousseeuw & Croux 1993)

```
Q_n = c · {|xᵢ - xⱼ| : i < j}_(kth smallest)   where k = C(n/2 + 1, 2)
```

O(n log n) time, breakdown 50%, 82% efficiency under normality.
R: `robustbase::Qn(x)`.

**Both S_n and Q_n require pairwise distance computation** — DistancePairs MSR (L1 norm, 1D).
For 1D data: DistancePairs is just the sorted absolute differences.

---

## IRLS Algorithm (Tambear Implementation Path)

```
Input: X (design matrix), y (response), psi function, max_iter, tol
Output: β̂ (converged M-estimate), σ̂, weights wᵢ

1. Initialize: β̂ = OLS solution (from F10 GramMatrix)
2. For iter in 1..max_iter:
   a. Residuals:  εᵢ = yᵢ - xᵢ'·β̂
   b. Scale:      σ̂ = mad(ε) / 0.6745    [or other robust scale]
   c. Weights:    wᵢ = psi(εᵢ/σ̂) / (εᵢ/σ̂)   [0 where denominator = 0]
   d. Weighted GramMatrix: X'WX = accumulate(Tiled, WeightedDotProduct(wᵢ), Add)
   e. Cross-product: X'Wy = accumulate(Tiled, WeightedCrossProduct(wᵢ), Add)
   f. β̂_new = (X'WX)^{-1} · X'Wy           [Cholesky solve]
   g. if ||β̂_new - β̂|| / (1 + ||β̂||) < tol: break
   h. β̂ = β̂_new
3. Return β̂, σ̂, wᵢ
```

**GPU kernel insight**: steps (d) and (e) are `accumulate(Tiled, ...)` with per-row weights.
This is the weighted version of GramMatrix — one kernel extension needed in TiledEngine.

**Step (b) scale estimation**: MAD requires sort → SortedPermutation. For Phase 1,
use `|ε|` sorted via CPU (residuals are small n = n_samples, not n_points).

---

## Validation Targets

```python
import numpy as np
import statsmodels.api as sm
from scipy.stats import norm

np.random.seed(42)
n = 100
X = np.column_stack([np.ones(n), np.random.randn(n, 2)])
true_beta = np.array([1.0, 2.0, -1.0])
# Add 10% contamination with large outliers:
y = X @ true_beta + norm.rvs(size=n)
outlier_mask = np.random.choice([False, True], n, p=[0.9, 0.1])
y[outlier_mask] += 10.0  # shift outliers by 10σ

# OLS (for comparison — will be pulled by outliers):
ols = sm.OLS(y, X).fit()
print("OLS β̂:", ols.params)

# Huber M-estimator:
rlm = sm.RLM(y, X, M=sm.robust.norms.HuberT(t=1.345)).fit()
print("RLM β̂:", rlm.params)
print("RLM weights (min):", rlm.weights.min())  # should be < 1 for outliers

# Expected: RLM should recover β̂ ≈ [1.0, 2.0, -1.0]; OLS will be biased
```

---

## Breakdown Point Summary

| Estimator | Breakdown | Efficiency | Notes |
|-----------|-----------|-----------|-------|
| OLS mean | 0% | 100% | One outlier → unbounded bias |
| Trimmed mean (10%) | 10% | ~99% | Simple, interpretable |
| MAD | 50% | 37% | Most robust location estimator |
| Winsorized mean | 10-50% | ~95% | More efficient than trimmed |
| Huber M-est | ~25% | 95% | Good balance for regression |
| Tukey bisquare | ~50% | 95% | Hard rejection, multiple optima |
| S-estimator | 50% | 29% | High breakdown for regression |
| MM-estimator | 50% | 95% | Gold standard: S-init + M-step |

**Tambear Phase 1 target**: Huber M-estimator (IRLS) with MAD scale.
Breakdown ~25%, 95% efficiency — best practical trade-off.
