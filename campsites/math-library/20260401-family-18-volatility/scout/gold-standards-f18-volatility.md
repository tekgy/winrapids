# F18 Volatility & Financial Time Series — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 18 (Volatility & Financial Time Series).
Primary domain: fintek signal farm. Every algorithm here produces leaf candidates
in Kingdom A (windowed accumulate), Kingdom B (Affine scan), or Kingdom C (iterative/MCMC).

The central thesis from the naturalist: volatility is a SIGNAL ABOUT THE SIGNAL.
Every volatility estimate is a second-order statistic computed over returns.
The tambear framing: every algorithm here is an `accumulate` of returns with a
grouping and operator that extracts variance structure.

---

## Tambear Kingdom Mapping (Overview)

| Algorithm | Grouping | Op | Kingdom | Notes |
|-----------|----------|----|---------|-------|
| GARCH(p,q) | Prefix(forward) | Affine | B | scalar scan, A=β |
| EGARCH | Prefix(forward) | Custom(nonlinear) | B* | log-variance, needs Custom op |
| GJR-GARCH | Prefix(forward) | Custom(threshold) | B* | adds I(εt<0) branch |
| FIGARCH | Prefix(forward) | Affine(fractional) | B | fractional differencing |
| HAR-RV | Windowed(1,5,22) | Add (→ regression) | A+B | RV computed first, then HAR-OLS |
| Realized Vol (RV) | Windowed(M) | Add (MomentStats, order=2) | A | sum of sq returns in window |
| BPV | Windowed(M) | Custom(adjacent product) | A | |π_{m-1}|·|π_m| pairs |
| TSRV | Windowed + Windowed | Add + sparse subsample | A | two-scale correction |
| RVAR (jump-robust) | Windowed(M) | Custom(threshold) | A | replace outliers with BPV |
| Stoch. Vol (SV) | Prefix(forward) | Affine + noise | B+C | MCMC estimation = Kingdom C |
| Black-Scholes IV | None (inversion) | Custom(root-find) | C | Newton-Raphson per option |
| VIX-style | Windowed(M) | Add | A | model-free, strip of options |

---

## Part 1: GARCH Family

### 1.1 The Affine Scan Structure of GARCH

GARCH(1,1) volatility recursion:
```
σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
```

In tambear's Affine scan language:
```
state_t = A · state_{t-1} + b_t
where:
  state_t = σ²_t          (variance IS the state — scalar)
  A       = β              (constant transition, scalar)
  b_t     = ω + α·ε²_{t-1}  (data-dependent input, computed from residuals)
```

This is exactly `accumulate(data, Prefix(forward), expr=ω + α·ε²_{t-1}, op=Affine(β))`.
Stationarity condition: β < 1 (same as |AR coefficient| < 1).

GARCH(p,q) generalization:
```
σ²_t = ω + Σ_{i=1}^{q} α_i·ε²_{t-i} + Σ_{j=1}^{p} β_j·σ²_{t-j}
```

State vector for GARCH(p,q): s_t = [σ²_t, σ²_{t-1}, ..., σ²_{t-p+1}, ε²_{t-1}, ..., ε²_{t-q+1}]
Companion matrix form — same pattern as ARIMA. Scalar GARCH(1,1) is simplest case.

### 1.2 Python: arch Package (Kevin Sheppard — gold standard)

```python
from arch import arch_model
import numpy as np

# ------ GARCH(1,1) on returns ------
returns = np.random.randn(1000) * 0.01  # daily returns, ~1% vol scale

model = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
result = model.fit(disp='off')  # suppress optimization output
result.summary()

# KEY OUTPUTS:
result.params          # dict-like: 'mu', 'omega', 'alpha[1]', 'beta[1]'
result.conditional_volatility   # σ_t (NOT σ²_t — see trap below)
result.resid           # ε_t = r_t - μ (mean-adjusted residuals)
result.std_resid       # standardized: z_t = ε_t / σ_t
result.loglikelihood   # scalar log-likelihood
result.aic             # Akaike information criterion
result.bic             # Bayesian information criterion
result.nobs            # number of observations used

# Variance (what tambear stores as state):
sigma2 = result.conditional_volatility ** 2   # σ²_t — THIS is the Affine state

# ------ GARCH(2,1) ------
model = arch_model(returns, vol='Garch', p=2, q=1)
result = model.fit(disp='off')
# params: 'omega', 'alpha[1]', 'alpha[2]', 'beta[1]'

# ------ GARCH(1,1) with Student-t errors ------
model = arch_model(returns, vol='Garch', p=1, q=1, dist='t')
result = model.fit(disp='off')
# params: adds 'nu' (degrees of freedom)

# ------ Forecasting ------
forecast = result.forecast(horizon=5, reindex=False)
forecast.variance      # shape (1, 5): σ²_{T+1}, ..., σ²_{T+5}
forecast.mean          # shape (1, 5): μ forecast
```

**TRAP 1 (critical)**: `result.conditional_volatility` returns **σ_t** (standard deviation),
NOT σ²_t (variance). The Affine scan state is σ²_t.
Always square it: `sigma2 = result.conditional_volatility ** 2`.

**TRAP 2 (rescaling)**: By default, arch rescales returns by 100 before fitting.
To disable: `arch_model(returns * 100, ...)` or pass pre-scaled data.
Or set `rescale=False`:
```python
model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)
```
If you see `omega ~ 1e-4` vs `omega ~ 1.0`, you hit the rescaling trap.
The rule: omega in unscaled fit is omega_scaled / 100².

**TRAP 3 (initial variance)**: arch initializes σ²_0 = variance of the full return series.
Tambear must match this initialization to get identical outputs.
Access it via `result.model._backcast` (internal method — check version compatibility).

### 1.3 R: rugarch Package (complementary gold standard)

```r
library(rugarch)

# ------ GARCH(1,1) ------
spec <- ugarchspec(
  variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(0, 0), include.mean = TRUE),
  distribution.model = "norm"
)
fit <- ugarchfit(spec = spec, data = returns)

# Key outputs:
coef(fit)              # named vector: mu, omega, alpha1, beta1
sigma(fit)             # σ_t (conditional volatility, NOT variance)
sigma(fit)^2           # σ²_t — the Affine state
residuals(fit)         # ε_t = r_t - μ
residuals(fit, standardize = TRUE)  # z_t = ε_t / σ_t
likelihood(fit)        # log-likelihood (scalar)
infocriteria(fit)      # AIC, BIC, SIC, HQIC

# Forecasting:
forc <- ugarchforecast(fit, n.ahead = 5)
sigma(forc)            # σ_{T+1}, ..., σ_{T+5}
fitted(forc)           # mean forecast

# Model diagnostics:
plot(fit, which = "all")  # standardized residual plots
```

### 1.4 EGARCH — Exponential GARCH (Nelson 1991)

EGARCH models log(σ²_t) instead of σ²_t:
```
log(σ²_t) = ω + α·(|z_{t-1}| - E[|z|]) + γ·z_{t-1} + β·log(σ²_{t-1})
```
where z_t = ε_t / σ_t (standardized residual).

**Why it matters**: γ captures the leverage effect — negative shocks increase volatility
more than positive shocks of equal magnitude (empirically true in equity markets).
γ < 0 = leverage effect present.

**Tambear implication**: EGARCH is NOT a clean Affine scan because the state is log(σ²_t),
and the input term involves |z_{t-1}| which requires σ²_{t-1} to compute. This creates
a non-linear dependency: b_t depends on state_t. Strictly speaking it is still sequential
(Kingdom B) but the custom operator needs access to the current state to compute z_{t-1}.

This is the key structural difference from GARCH(1,1):
- GARCH(1,1): b_t = ω + α·ε²_{t-1} — depends only on returns data, not on state
- EGARCH: b_t includes γ·z_{t-1} = γ·ε_{t-1}/σ_{t-1} — depends on the PREVIOUS state

EGARCH needs `Custom(state-dependent)` operator in tambear, not plain `Affine`.

```python
from arch import arch_model

# EGARCH(1,1):
model = arch_model(returns, vol='EGarch', p=1, q=1, mean='Constant', dist='Normal')
result = model.fit(disp='off')
result.params  # 'mu', 'omega', 'alpha[1]', 'gamma[1]', 'beta[1]'
# gamma[1] = leverage parameter (typically negative for equities)

# rugarch:
spec <- ugarchspec(
  variance.model = list(model = "eGARCH", garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(0, 0))
)
fit <- ugarchfit(spec, data = returns)
coef(fit)  # omega, alpha1, beta1, gamma1
```

### 1.5 GJR-GARCH — Threshold GARCH (Glosten-Jagannathan-Runkle 1993)

```
σ²_t = ω + (α + γ·I(ε_{t-1} < 0))·ε²_{t-1} + β·σ²_{t-1}
```
where I(·) = indicator function.

**Interpretation**: when ε_{t-1} < 0 (bad news), total ARCH coefficient is (α + γ), not just α.
If γ > 0: bad news amplifies variance more than good news = leverage effect.

**Tambear implication**: This IS still an Affine scan structure, but the effective A coefficient
changes depending on the sign of ε_{t-1}. The custom operator:
```
b_t = ω + (α + γ·[ε_{t-1} < 0])·ε²_{t-1}
A   = β (constant)
```
The indicator is applied to `b_t` (the input term), NOT the state. So this can be
implemented as Affine with a modified `b_t` computation — a fused_expr before the scan.

```python
from arch import arch_model

# GJR-GARCH(1,1,1): p=1 ARCH lags, o=1 asymmetric lags, q=1 GARCH lags
model = arch_model(returns, vol='Garch', p=1, o=1, q=1, mean='Constant', dist='Normal')
result = model.fit(disp='off')
result.params  # 'mu', 'omega', 'alpha[1]', 'gamma[1]', 'beta[1]'
# gamma[1] = threshold/leverage coefficient
```

```r
library(rugarch)
spec <- ugarchspec(
  variance.model = list(model = "gjrGARCH", garchOrder = c(1, 1)),
  mean.model     = list(armaOrder = c(0, 0))
)
fit <- ugarchfit(spec, data = returns)
coef(fit)  # mu, omega, alpha1, beta1, gamma1
```

### 1.6 FIGARCH — Fractionally Integrated GARCH (Baillie et al. 1996)

FIGARCH allows long memory in variance: shocks decay hyperbolically, not exponentially.
The variance equation involves fractional differencing of ε²_t:
```
σ²_t = ω·(1 - β(L))^{-1} + [1 - (1 - β(L))^{-1}·Φ(L)·(1 - L)^d] · ε²_t
```
where d ∈ (0, 1) is the fractional integration order, L is the lag operator.

**Tambear implication**: FIGARCH expands to an infinite-order ARCH representation.
In practice it's truncated. The truncated form IS an Affine scan but with many more
terms in the b_t vector (50-100 lags of ε²_t). Kingdom B, but large state vector.
d ≈ 0 → GARCH, d ≈ 1 → IGARCH (integrated, variance has unit root).

```r
# R: rugarch handles FIGARCH
library(rugarch)
spec <- ugarchspec(
  variance.model = list(model = "fiGARCH", garchOrder = c(1, d=0.5, 1)),
  mean.model     = list(armaOrder = c(0, 0))
)
fit <- ugarchfit(spec, data = returns)
coef(fit)  # omega, d, alpha1, beta1
```

```python
# Python: no clean FIGARCH in arch package as of 2026
# Alternative: statsmodels has ARCH models, but FIGARCH requires rugarch or specialized code
# For validation purposes, use rugarch as the oracle for FIGARCH
```

### 1.7 Validation Targets: GARCH(1,1)

Generate a controlled dataset with known parameters, fit, verify:

```python
import numpy as np
from arch import arch_model

# Simulate GARCH(1,1) with known parameters
np.random.seed(42)
n = 2000
omega, alpha, beta = 0.0001, 0.10, 0.85  # omega small for daily returns ~1%
mu = 0.0005  # tiny drift

sigma2 = np.zeros(n)
eps    = np.zeros(n)
r      = np.zeros(n)

sigma2[0] = omega / (1 - alpha - beta)  # unconditional variance
for t in range(1, n):
    sigma2[t] = omega + alpha * eps[t-1]**2 + beta * sigma2[t-1]
    eps[t]    = np.sqrt(sigma2[t]) * np.random.randn()
    r[t]      = mu + eps[t]

# Fit with arch:
model  = arch_model(r[1:], vol='Garch', p=1, q=1, mean='Constant', rescale=False)
result = model.fit(disp='off')

print("True    omega={:.6f}, alpha={:.4f}, beta={:.4f}".format(omega, alpha, beta))
print("Fitted  omega={:.6f}, alpha={:.4f}, beta={:.4f}".format(
    result.params['omega'],
    result.params['alpha[1]'],
    result.params['beta[1]']
))
# Expected: fitted values close to true (within ~0.01 for alpha/beta, within 1e-5 for omega)
# Persistence = alpha + beta ~ 0.95 (typical for financial data)
print("Persistence (should be ~0.95):", result.params['alpha[1]'] + result.params['beta[1]'])

# Tambear state comparison:
fitted_sigma2 = result.conditional_volatility ** 2
simulated_sigma2 = sigma2[1:]
corr = np.corrcoef(fitted_sigma2, simulated_sigma2)[0, 1]
print("Correlation of fitted vs true σ²:", corr)  # should be > 0.90
```

**Validation targets** (with seed=42, n=2000, omega=0.0001, alpha=0.10, beta=0.85):
- Fitted omega: approximately 0.000085 to 0.000120 (MLE variance in parameter)
- Fitted alpha: approximately 0.085 to 0.115
- Fitted beta: approximately 0.830 to 0.870
- Fitted persistence (alpha + beta): approximately 0.92 to 0.98
- Log-likelihood: approximately 5800 to 6200 (depends on realized draws)

---

## Part 2: Realized Volatility Estimators

### 2.1 The Core Idea: RV from Tick Data

Realized volatility uses high-frequency intraday data to estimate daily (or intraday) variance.
The classic estimator: sum of squared intraday returns over M sub-intervals of length Δ = 1/M:
```
RV_t = Σ_{j=1}^{M} r_{t,j}²   where r_{t,j} = log(p_{t,jΔ}/p_{t,(j-1)Δ})
```

As M → ∞ and Δ → 0, RV_t → ∫ σ²(s) ds (integrated variance over day t).
In practice: M = 78 (5-min bars in 6.5-hour US equity session), or M = 23 (30-min bars).

**Tambear framing**: RV = `accumulate(sq_returns_intraday, Windowed(M=78), identity, Add)`
= `MomentStats(order=2, Windowed(M))`. This is a windowed reduce. Kingdom A.

The daily RV time series itself is computed with grouping `ByKey(date)`.

### 2.2 R: highfrequency Package (gold standard for RV)

```r
library(highfrequency)
library(xts)

# ------ Basic Realized Variance (RV) ------
# Input: xts of log returns or prices, sub-daily frequency (e.g., 5-min)

# From prices (auto-computes returns):
rv <- rCov(rData = returns_5min)  # rCov = realized covariance (scalar = realized var)

# Or specify:
rv <- rCov(
  rData      = returns_5min,   # xts of log returns, one column per asset
  align.by   = "minutes",      # align to minute boundaries
  align.period = 5,            # 5-minute bars
  makeReturns = FALSE          # data is already returns
)
# rv is an xts with one value per day

# ------ Bipower Variation (BPV) — jump-robust ------
bpv <- rBPCov(rData = returns_5min)
# BPV_t = (π/2) · Σ_{j=2}^{M} |r_{t,j-1}| · |r_{t,j}|
# Jump component = RV - BPV (if positive)

# ------ Two-Scales Realized Volatility (TSRV) ------
tsrv <- rTSCov(
  rData      = returns_raw,    # FULL tick data, not pre-sampled
  K          = 5,              # subsample frequency (every K ticks)
  J          = 1               # averaging scale
)
# TSRV corrects for microstructure noise in ultra-high-frequency data

# ------ Jump detection ------
jump_test <- rAJump(rData = returns_5min)
jump_test$ztest   # Z-statistic for presence of jumps on each day
jump_test$jump    # jump component estimate

# ------ HAR model ------
har <- HARmodel(
  data   = rv_daily,    # xts of daily RV values
  periods = c(1, 5, 22),  # daily, weekly, monthly lags
  type   = "HAR",
  h      = 1             # 1-step forecast (h > 1 uses overlapping aggregation)
)
summary(har)
# Coefficients: beta_d, beta_w, beta_m (and intercept)
fitted(har)      # in-sample fit
residuals(har)   # HAR residuals
predict(har, newdata = tail(rv_daily, 22))  # forecast next day's RV
```

### 2.3 Python: realized volatility (manual, since highfrequency is R-only)

```python
import numpy as np
import pandas as pd

def realized_variance(returns_intraday: np.ndarray) -> float:
    """Standard RV: sum of squared 5-min returns over one day."""
    return float(np.sum(returns_intraday ** 2))

def bipower_variation(returns_intraday: np.ndarray) -> float:
    """BPV: jump-robust alternative to RV.
    BPV = (π/2) · Σ_{j=2}^{M} |r_{j-1}| · |r_j|
    Consistent for integrated variance even in presence of jumps.
    """
    mu1 = np.sqrt(2 / np.pi)  # E[|z|] for z ~ N(0,1)
    r = np.abs(returns_intraday)
    return float((1 / mu1**2) * np.sum(r[:-1] * r[1:]))

def tsrv(returns_tick: np.ndarray, K: int = 5) -> float:
    """Two-Scales Realized Variance (Zhang, Mykland, Ait-Sahalia 2005).
    Subsamples at frequency K, computes average RV across offsets,
    then subtracts a bias correction for microstructure noise.

    Returns: TSRV estimate of integrated variance.
    """
    n = len(returns_tick)

    # Slow scale: RV using every K-th return (K subsampled grids, averaged)
    rv_slow_grids = []
    for offset in range(K):
        r_sub = returns_tick[offset::K]
        rv_slow_grids.append(np.sum(r_sub ** 2))
    rv_slow = np.mean(rv_slow_grids)

    # Fast scale: RV using ALL tick-by-tick returns (the noisy estimator)
    rv_fast = np.sum(returns_tick ** 2)

    n_sub = n / K  # average number of returns per subsampled grid

    # Bias-corrected TSRV:
    tsrv_val = rv_slow - (n_sub / n) * rv_fast
    return float(tsrv_val)

def realized_variance_daily(prices_intraday: pd.Series, freq: str = '5min') -> pd.Series:
    """Compute daily RV from intraday prices."""
    log_ret = np.log(prices_intraday).diff().dropna()
    log_ret_resampled = log_ret.resample(freq).sum()  # returns at freq
    rv = (log_ret_resampled ** 2).resample('1D').sum()
    return rv
```

### 2.4 The Sampling Frequency Trap

**The microstructure noise problem**: at very high frequencies (tick-by-tick), bid-ask
bounce and price discretization create artificial variance (microstructure noise).
RV computed from 1-second returns is severely upward-biased vs true volatility.

**Resolution: the Aitt-Sahalia-Mykland "volatility signature plot"**:
```r
# R: plot RV vs sampling frequency to find the "sweet spot"
library(highfrequency)
vsp <- rRVar(rData = tick_data, freq.vec = c(1, 2, 5, 10, 15, 30))
plot(vsp)  # U-shape: too fast = noise bias, too slow = efficiency loss
```

**Rule of thumb**: 5-minute returns are widely used for US equities (eliminates most
microstructure noise while preserving most true variance signal).

**For tambear**: the sampling frequency is a parameter. The `RealizedVol` specialist should
expose `sampling_freq` as a configuration. The "right" value is data-dependent.

### 2.5 Realized Volatility Tambear Primitive Decomposition

```
# Daily RV from intraday returns:
rv_daily = accumulate(
    gather(intraday_returns, ByKey(date)),   # group returns by day
    grouping = ByKey(date),
    expr     = r²,                           # squared return
    op       = Add                           # sum within day
)

# More precisely — fused expr + reduce per day:
rv_daily = accumulate(intraday_returns, ByKey(date), expr=identity, op=MomentStats(order=2))

# BPV requires adjacent pairs — NOT a simple MomentStats:
bpv_daily = accumulate(
    gather(intraday_returns, Strided(offset=1, stride=1)),  # adjacent pairs
    grouping = ByKey(date),
    expr     = |r_{j-1}| * |r_j|,           # product of adjacent abs returns
    op       = Add
)
# Requires MultiOffset addressing to get adjacent pairs simultaneously
```

BPV needs `MultiOffset([-1, 0])` gathering — two adjacent values simultaneously.
This is one step beyond simple windowed accumulate.

---

## Part 3: HAR Model (Heterogeneous Autoregressive)

### 3.1 The HAR Model

Corsi (2009) HAR model for daily RV:
```
RV_{t+1} = β_0 + β_d · RV_t + β_w · RV^{(w)}_t + β_m · RV^{(m)}_t + ε_{t+1}
```
where:
- RV_t = daily realized variance (last day's value)
- RV^{(w)}_t = mean of past 5 daily RVs (proxy for weekly agents)
- RV^{(m)}_t = mean of past 22 daily RVs (proxy for monthly agents)

**Key**: RV^{(w)} and RV^{(m)} use OVERLAPPING windows (rolling means), not non-overlapping blocks.

**Tambear decomposition**:
```
# Step 1: compute rolling means of RV at different horizons
rv_weekly  = accumulate(rv_daily, Windowed(5),  identity, Add) / 5
rv_monthly = accumulate(rv_daily, Windowed(22), identity, Add) / 22

# Step 2: construct design matrix X = [1, rv_t, rv_weekly_t, rv_monthly_t]
# Step 3: OLS regression RV_{t+1} ~ X_t
# = a Windowed accumulate + affine regression = Kingdom A + B composition
```

Step 1 is Kingdom A (Windowed Add). Step 2-3 is linear regression — Kingdom A (normal equations).
The HAR model is purely Kingdom A — no sequential state needed once the rolling means are computed.

### 3.2 R: highfrequency::HARmodel

```r
library(highfrequency)

# rv_daily: xts or numeric vector of daily RV estimates
har <- HARmodel(
  data    = rv_daily,
  periods = c(1, 5, 22),   # daily, weekly, monthly
  type    = "HAR",          # 'HAR', 'HARRV', 'HARRVJ' (with jump component)
  h       = 1,              # 1-step ahead target
  transform = "none"        # 'sqrt' for log-HAR: fit on sqrt(RV) for normality
)

# Access coefficients:
coef(har)             # c(intercept, beta_d, beta_w, beta_m)
summary(har)$r.squared  # R² (HAR typically gets 0.40-0.65 on real data)
fitted(har)           # in-sample fitted values
residuals(har)        # residuals (should be heteroskedastic)

# HAR-J: HAR with jump component
# Jump = max(RV - BPV, 0) added as fourth regressor
harj <- HARmodel(data = rv_xts, periods = c(1, 5, 22), type = "HARRVJ")
coef(harj)  # adds 'beta_j' for jump component

# Log-HAR: fit on log(RV) for approximate normality
har_log <- HARmodel(data = log(rv_daily), periods = c(1, 5, 22), type = "HAR")
# In-sample R² is typically higher on log scale (more homoskedastic)
```

### 3.3 Python: HAR Model (manual OLS)

```python
import numpy as np
from sklearn.linear_model import LinearRegression

def har_model(rv_daily: np.ndarray, h: int = 1):
    """
    HAR-RV model (Corsi 2009).

    rv_daily: array of daily realized variance values
    h: forecast horizon (default 1 day)

    Returns: fitted coefficients (intercept, beta_d, beta_w, beta_m)
    """
    n = len(rv_daily)

    # Compute weekly (5-day) and monthly (22-day) rolling means
    # Using overlapping windows
    rv_weekly  = np.array([np.mean(rv_daily[max(0,i-5):i])  for i in range(1, n)])
    rv_monthly = np.array([np.mean(rv_daily[max(0,i-22):i]) for i in range(1, n)])
    rv_lagged  = rv_daily[:-1]  # RV_t as predictor of RV_{t+1}

    # Align: need at least 22 observations for full monthly window
    start = 22
    y  = rv_daily[start + h:]   # target: RV_{t+h}
    X_d = rv_lagged[start:-h] if h > 0 else rv_lagged[start:]
    X_w = rv_weekly[start:-h]  if h > 0 else rv_weekly[start:]
    X_m = rv_monthly[start:-h] if h > 0 else rv_monthly[start:]

    # OLS
    X = np.column_stack([X_d, X_w, X_m])
    reg = LinearRegression(fit_intercept=True).fit(X, y)

    return {
        'intercept': reg.intercept_,
        'beta_d':    reg.coef_[0],
        'beta_w':    reg.coef_[1],
        'beta_m':    reg.coef_[2],
        'r_squared': reg.score(X, y),
    }

# Typical coefficient signs (empirical):
# beta_d > 0 (positive persistence at daily scale)
# beta_w > 0 (positive persistence at weekly scale)
# beta_m > 0 (positive persistence at monthly scale)
# Typical R² on real equity data: 0.40 - 0.65
# Typical R² on log(RV): 0.50 - 0.75
```

**TRAP (overlapping windows)**: HAR uses OVERLAPPING windows.
`RV^{(w)}_t = (RV_t + RV_{t-1} + RV_{t-2} + RV_{t-3} + RV_{t-4}) / 5`.
NOT the non-overlapping weekly block. This is a rolling mean, not a block mean.
The tambear primitive is `Windowed(5)` with overlap — standard rolling window.

### 3.4 HAR Validation Targets

```python
import numpy as np
from sklearn.linear_model import LinearRegression

np.random.seed(123)
# Simulate simple RV process (GARCH-like dynamics in variance of variance)
n = 500
rv = np.zeros(n)
rv[0] = 0.0002
for t in range(1, n):
    # AR(1) in RV + noise (simple simulation)
    rv[t] = 0.0001 + 0.7 * rv[t-1] + 0.00005 * np.random.randn()
    rv[t] = max(rv[t], 1e-8)  # keep positive

coefs = har_model(rv)
print("HAR coefficients:", coefs)
# Expected: beta_d dominant (0.4-0.7), beta_w and beta_m positive but smaller
```

---

## Part 4: Stochastic Volatility Models

### 4.1 The SV Model Structure

The basic stochastic volatility model (Taylor 1986):
```
y_t = σ_t · ε_t           (observation equation)
log(σ²_t) = μ + φ·(log(σ²_{t-1}) - μ) + η_t   (state equation)
ε_t ~ N(0,1), η_t ~ N(0, σ²_η)   (independent)
```

Parameters: {μ, φ, σ²_η}
- μ = log of unconditional variance (level)
- φ = persistence (|φ| < 1 for stationarity)
- σ²_η = variance of the log-variance innovations

**Key difference from GARCH**: in GARCH, σ²_t is a DETERMINISTIC function of past data.
In SV, σ²_t has its OWN noise source (η_t). This makes σ²_t a LATENT state.

**Tambear implication**: The SV state equation IS an Affine scan:
```
log(σ²_t) = (1-φ)·μ + φ·log(σ²_{t-1}) + η_t
state_t = A·state_{t-1} + b + noise
A = φ, b = (1-φ)·μ, noise = η_t ~ N(0, σ²_η)
```
But: estimation requires integrating out the latent σ²_t — this is a MCMC or particle filter
problem. The STRUCTURE is Kingdom B; the ESTIMATION is Kingdom C (iterative/MCMC).

### 4.2 Kim-Shephard-Chib Parameterization

KSC (1998) is the standard Bayesian MCMC approach, used in R's `stochvol` package.
It transforms the observation equation via:
```
log(y²_t) = log(σ²_t) + log(ε²_t)
```
Since log(ε²_t) ≈ mixture of normals (KSC use a 10-component mixture approximation
to the log-χ²(1) distribution), the state space becomes Gaussian and can be handled
by the Kalman smoother.

This is a 2-step structure:
1. Sample log(σ²_t) states via Kalman smoother conditioned on mixture indicators
2. Sample mixture indicators conditioned on log(σ²_t)
3. Sample parameters {μ, φ, σ²_η} from conjugate posteriors

### 4.3 R: stochvol Package (gold standard for SV)

```r
library(stochvol)

# ------ Basic MCMC estimation ------
# y: numeric vector of log-returns (demeaned, centered at zero)
y <- returns - mean(returns)

sv_result <- svsample(
  y,
  draws     = 10000,    # MCMC draws to keep
  burnin    = 1000,     # burn-in iterations
  thin      = 1,        # thinning factor
  priormu   = c(-9, 1), # prior for mu: N(-9, 1) — log-variance scale
  priorphi  = c(5, 1.5),# prior for phi: truncated normal via Beta parameterization
  priorsigma = 0.1      # prior for sigma_eta: exponential
)

# Parameter posterior summaries:
summary(sv_result$para)
# mu:    log(unconditional variance level) — for daily log-returns, typically -9 to -7
# phi:   persistence — typically 0.95-0.99 for financial data (highly persistent)
# sigma: volatility of log-variance — typically 0.1-0.3

# Extract posterior means:
colMeans(sv_result$para)  # mu, phi, sigma

# Latent volatility states (posterior means):
# sv_result$latent: matrix (draws × T) of log(σ_t) draws
volatility_path <- exp(colMeans(sv_result$latent))  # posterior mean of σ_t (NOT σ²_t)

# Compare with GARCH: SV volatility path is smoother (latent smoothing via Kalman)

# ------ More expressive SV with heavy tails ------
sv_result_t <- svsample(y, draws = 10000, burnin = 1000, designmatrix = "ar0",
                         n_t = NULL)  # Student-t innovations
```

### 4.4 Python: stochvol (via pymc or manual)

```python
# Option 1: pymc (gold standard for MCMC in Python)
import pymc as pm
import numpy as np

with pm.Model() as sv_model:
    # Priors matching KSC parameterization
    mu    = pm.Normal('mu', mu=-9, sigma=1)           # log-variance level
    phi   = pm.Uniform('phi', lower=0, upper=1)       # persistence
    sigma = pm.HalfNormal('sigma', sigma=0.1)          # volatility of log-var

    # State space: AR(1) on log-variance
    log_h = pm.AR('log_h', rho=phi, noise=sigma, init_dist=pm.Normal.dist(mu=mu, sigma=sigma/np.sqrt(1-phi**2)), shape=len(y))

    # Observation model
    obs = pm.Normal('obs', mu=0, sigma=pm.math.exp(log_h / 2), observed=y)

    # MCMC
    trace = pm.sample(2000, tune=1000, target_accept=0.95, return_inferencedata=True)

# Extract volatility:
import arviz as az
log_h_samples = trace.posterior['log_h'].values  # shape (chains, draws, T)
sigma_path = np.exp(np.mean(log_h_samples, axis=(0,1)) / 2)  # posterior mean σ_t
```

### 4.5 SV vs GARCH: When Each is Better

| Property | GARCH | SV |
|----------|-------|-----|
| Estimation | MLE (fast) | MCMC (slow) |
| Vol path | deterministic given data | smoothed latent |
| Parameter uncertainty | profile CI | full posterior |
| Fit (returns) | good | typically better |
| Prediction | direct recursion | simulation required |
| Tambear kingdom | B (deterministic scan) | B+C (scan + sampling) |

For the fintek signal farm: GARCH estimates are FASTER and SUFFICIENT for most signals.
SV is better for option pricing and situations where parameter uncertainty matters.
Phase 1 recommendation: implement GARCH family. SV as Phase 2.

---

## Part 5: Implied Volatility

### 5.1 Black-Scholes IV Inversion

The Black-Scholes call price formula:
```
C(S, K, T, r, σ) = S·N(d₁) - K·e^{-rT}·N(d₂)
where:
  d₁ = [log(S/K) + (r + σ²/2)·T] / (σ·√T)
  d₂ = d₁ - σ·√T
```

Implied volatility σ_IV = the σ that equates the formula to the observed market price C_mkt.
This is a root-finding problem: `f(σ) = C(S,K,T,r,σ) - C_mkt = 0`.

**There is no closed form.** Must be solved numerically.
**Standard method**: Newton-Raphson using the Black-Scholes vega (∂C/∂σ).

**Tambear implication**: IV computation is per-option. For a strip of options:
`accumulate(options, All, root_find(BS_price - market_price), Custom(Newton))`.
But Newton-Raphson is iterative per element — Kingdom C.
For a single batch of options, can parallelize across options (independent root-finds).
This is embarrassingly parallel: one Newton iteration per GPU thread per option.

```python
from scipy.stats import norm
from scipy.optimize import brentq
import numpy as np

def black_scholes_call(S, K, T, r, sigma):
    """Black-Scholes call price."""
    if T <= 0 or sigma <= 0:
        return max(S - K * np.exp(-r * T), 0.0)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def black_scholes_vega(S, K, T, r, sigma):
    """Vega = ∂C/∂σ. Used in Newton-Raphson IV inversion."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)

def implied_vol_newton(S, K, T, r, C_mkt, tol=1e-8, max_iter=100):
    """Newton-Raphson IV inversion.

    S: spot price
    K: strike
    T: time to expiry (years)
    r: risk-free rate (continuous)
    C_mkt: observed call price

    Returns: implied volatility, or NaN if no solution.
    """
    sigma = 0.2  # initial guess (20% annualized)
    for _ in range(max_iter):
        C_est = black_scholes_call(S, K, T, r, sigma)
        vega  = black_scholes_vega(S, K, T, r, sigma)
        if vega < 1e-12:
            break
        diff  = C_est - C_mkt
        sigma = sigma - diff / vega
        if abs(diff) < tol:
            return sigma
    return float('nan')

def implied_vol_brent(S, K, T, r, C_mkt):
    """Brent method (safer, no derivative needed).
    Lower bound: min possible vol ~1e-6. Upper bound: 5.0 (500% annualized).
    """
    intrinsic = max(S - K * np.exp(-r * T), 0.0)
    if C_mkt <= intrinsic:
        return float('nan')
    try:
        return brentq(
            lambda sig: black_scholes_call(S, K, T, r, sig) - C_mkt,
            a=1e-6, b=5.0, xtol=1e-8, rtol=1e-10
        )
    except ValueError:
        return float('nan')

# Validation:
S, K, T, r, sigma_true = 100.0, 100.0, 0.25, 0.05, 0.20
C_true = black_scholes_call(S, K, T, r, sigma_true)
sigma_implied = implied_vol_newton(S, K, T, r, C_true)
print(f"True sigma: {sigma_true:.6f}, Recovered: {sigma_implied:.6f}")
# Should match to 8+ decimal places
```

```r
# R: RQuantLib package (QuantLib bindings) — most complete option pricing
library(RQuantLib)

# European call IV:
EuropeanOptionImpliedVolatility(
  type          = "call",
  value         = C_mkt,     # observed market price
  underlying    = S,          # spot price
  strike        = K,
  dividendYield = 0,
  riskFreeRate  = r,
  maturity      = T,          # years to expiry
  volatility    = 0.2         # initial guess
)
# Returns: implied volatility

# Or: fOptions package
library(fOptions)
GBSVolatility(
  price = C_mkt,
  TypeFlag = "c",   # "c" = call, "p" = put
  S = S, X = K, Time = T, r = r, b = r,  # b = cost of carry (= r for no dividends)
  tol = 1e-6
)
```

### 5.2 VIX-Style Variance Swap Replication (Model-Free IV)

The CBOE VIX methodology computes a model-free expectation of 30-day variance using a
strip of out-of-the-money options across all strikes. It does NOT assume Black-Scholes.

The core formula (simplified):
```
VIX² = (2/T) · Σ_K (ΔK/K²) · e^{rT} · Q(K) - (1/T) · (F/K₀ - 1)²
```
where:
- Q(K) = price of out-of-the-money option at strike K (call if K > F, put if K < F)
- F = forward price
- K₀ = first strike below forward price
- ΔK = spacing between strikes

**Tambear framing**: VIX is a weighted sum over the strike strip:
```
vix_variance = accumulate(options_strip, All, expr=(ΔK/K²)·Q(K), op=Add) · (2/T)
```
This is `accumulate(All, weighted_integral, Add)` — pure Kingdom A.

```python
import numpy as np

def vix_variance(strikes: np.ndarray, option_prices: np.ndarray,
                 forward: float, T: float, r: float) -> float:
    """
    CBOE VIX methodology (simplified).
    Computes model-free 30-day variance.

    strikes: sorted array of strikes
    option_prices: OTM option prices (put if K < F, call if K >= F)
    forward: forward price F = S * exp(r*T)
    T: time to expiry (years, should be ~30/365 for VIX)
    r: risk-free rate
    """
    # Trapezoid weights
    delta_K = np.gradient(strikes)

    # Contribution from each option
    contributions = (delta_K / strikes**2) * np.exp(r * T) * option_prices
    raw_sum = 2 / T * np.sum(contributions)

    # ATM correction
    K0_idx = np.searchsorted(strikes, forward) - 1
    K0 = strikes[K0_idx]
    atm_correction = (1 / T) * (forward / K0 - 1)**2

    return raw_sum - atm_correction

# VIX = 100 * sqrt(vix_variance)
# When VIX = 20, vix_variance ≈ (0.20)² = 0.04 (annualized)
```

---

## Part 6: Key Traps Reference

### Trap 1: arch `conditional_volatility` = σ_t NOT σ²_t

```python
result = arch_model(returns, vol='Garch', p=1, q=1).fit()
sigma_t    = result.conditional_volatility        # σ_t (std dev)
sigma2_t   = result.conditional_volatility ** 2   # σ²_t (variance) ← Affine state
```
Same trap in R's rugarch: `sigma(fit)` = σ_t. Square it for variance.

### Trap 2: arch Package Default Rescaling

```python
# arch scales returns by 100 by default:
# internal_returns = 100 * your_returns
# fitted omega is for scaled data: omega_true = omega_fitted / 10000

model = arch_model(returns, vol='Garch', p=1, q=1)  # rescales by 100
model = arch_model(returns, vol='Garch', p=1, q=1, rescale=False)  # no rescaling

# To diagnose: compare omega magnitudes
# omega ~ 1e-6 to 1e-4 for daily returns → correct unscaled
# omega ~ 1e-2 to 1.0 for daily returns → data was rescaled by 100
```

### Trap 3: HAR Uses Overlapping (Rolling) Windows

```python
# WRONG: non-overlapping weekly blocks
rv_weekly_wrong = rv_daily.reshape(-1, 5).mean(axis=1)  # block means

# CORRECT: rolling (overlapping) window
rv_weekly_correct = pd.Series(rv_daily).rolling(5).mean().values  # rolling mean
# highfrequency::HARmodel always uses overlapping — verify by inspecting source
```

### Trap 4: RV Sampling Frequency Choice

- At tick frequency: RV is severely upward-biased (microstructure noise)
- At 5-minute frequency: standard choice for US equities
- At 30-minute frequency: appropriate for markets with lower liquidity
- Too slow (daily): RV collapses to one squared return — useless estimator

```r
# Use volatility signature plot to diagnose:
library(highfrequency)
vsp <- rRVar(rData = tick_data, freq.vec = c(1, 2, 5, 10, 15, 30))
plot(vsp)
# Sweet spot = frequency where RV curve flattens out before rising from noise
```

### Trap 5: GARCH Stationarity — alpha + beta < 1

If `alpha + beta >= 1`, the GARCH process is nonstationary (IGARCH: alpha + beta = 1).
- arch package DOES NOT enforce stationarity by default; it allows IGARCH
- rugarch allows specifying stationarity constraints
- Tambear should warn if fitted persistence >= 0.999 (near-integrated behavior)

```python
result = arch_model(returns, vol='Garch', p=1, q=1, rescale=False).fit()
alpha = result.params['alpha[1]']
beta  = result.params['beta[1]']
persistence = alpha + beta
if persistence >= 1.0:
    print("WARNING: IGARCH — non-stationary, infinite unconditional variance")
elif persistence > 0.999:
    print("WARNING: Near-integrated GARCH, very long memory")
```

### Trap 6: EGARCH State Dependency

EGARCH requires σ²_{t-1} to compute z_{t-1} = ε_{t-1}/σ_{t-1}.
This means the input b_t is NOT purely data-driven — it depends on the previous state.
EGARCH is NOT a clean Affine scan; it needs a `Custom(state-dependent)` operator
where b_t = f(state_{t-1}, data_{t-1}).

```
# GARCH(1,1): A = β (constant), b_t = ω + α·ε²_{t-1} (data only)  ← clean Affine scan
# EGARCH:     A = β (constant), b_t = ω + α·|z_{t-1}| + γ·z_{t-1}  ← state-dependent
#             where z_{t-1} = ε_{t-1} / σ_{t-1} = ε_{t-1} / sqrt(state_{t-1})
```

### Trap 7: SV Latent State — Not Observable

In SV models, σ²_t is LATENT. You cannot extract σ²_t from the data without MCMC.
The `svsample()` function returns a matrix of MCMC DRAWS from p(log σ²_t | data).
The "volatility path" is the posterior mean — NOT a deterministic reconstruction.

```r
# Wrong: treating latent path as point estimate
vol_path <- exp(sv_result$latent[1,] / 2)  # just ONE MCMC draw — wrong

# Correct: posterior mean
vol_path <- apply(sv_result$latent, 2, function(x) exp(mean(x) / 2))
# Or equivalently:
vol_path <- exp(colMeans(sv_result$latent) / 2)
```

---

## Part 7: Kingdom Classification Summary

### Kingdom A (Windowed / ByKey Accumulate)

| Algorithm | Grouping | Expr | Op | Notes |
|-----------|----------|------|----|-------|
| Realized Variance (RV) | ByKey(date) or Windowed(M) | r² | Add | Core: MomentStats(order=2) |
| Bipower Variation (BPV) | ByKey(date) or Windowed(M) | \|r_{j-1}\|·\|r_j\| | Add | Needs MultiOffset addressing |
| TSRV | Windowed + subsample | r² (subsampled) | Add | Two-scale correction |
| HAR Step 1: Rolling RV | Windowed(5), Windowed(22) | identity | Add / count | Rolling means of RV |
| HAR Step 2: Regression | All (normal equations) | XᵀX, Xᵀy | Add | OLS via accumulate |
| VIX replication | All (strike strip) | (ΔK/K²)·Q(K) | Add | Model-free variance |

### Kingdom B (Affine Scan / Sequential)

| Algorithm | A (transition) | b_t (input) | Notes |
|-----------|---------------|-------------|-------|
| GARCH(1,1) | β (constant, scalar) | ω + α·ε²_{t-1} | Cleanest Affine scan |
| GARCH(p,q) | companion matrix | offset + α terms | Vector Affine scan |
| GJR-GARCH | β (constant) | ω + (α + γ·I(ε<0))·ε² | Modified b_t — still clean Affine |
| FIGARCH | φ(L)(1-L)^d matrix (truncated) | ω | Large state vector |
| EGARCH | β (constant) | ω + f(state_{t-1}, ε_{t-1}) | State-dependent — Custom op |
| SV (state eq.) | φ (scalar) | (1-φ)·μ + η_t | Latent; estimation is Kingdom C |

### Kingdom C (Iterative / MCMC / Root-Finding)

| Algorithm | Method | Iteration | Notes |
|-----------|--------|-----------|-------|
| GARCH estimation | L-BFGS-B, MLE | outer optimization | Inner evaluation IS Kingdom B |
| SV estimation | MCMC (KSC) | Gibbs sampling | Kalman smoother in each step = Kingdom B |
| Implied Vol | Newton-Raphson / Brent | per-option root-find | Embarrassingly parallel |
| FIGARCH estimation | MLE | outer optimization | Inner evaluation IS Kingdom B |

**Note on Kingdom C**: GARCH's MLE outer optimization (finding ω, α, β) is iterative,
but once parameters are fixed, computing σ²_t is a pure Kingdom B scan.
For the signal FARM (fixed parameters, compute vol path daily), it's Kingdom B only.
Kingdom C appears only during FIT — which happens once, not per-day.

---

## Part 8: Primitive Gaps for F18

### Gap 1: Adjacent-Pair Accumulate (for BPV)

BPV requires multiplying adjacent absolute returns: `|r_{j-1}| · |r_j|`.
This requires `MultiOffset([-1, 0])` gather — reading two adjacent positions simultaneously.
Current tambear has `Strided` but not `MultiOffset` for the adjacent-pairs pattern.

**Proposed**: `gather(returns, MultiOffset(offsets=[-1, 0]))` producing pairs, then
`accumulate(pairs, ByKey(date), |a|·|b|, Add)`.

### Gap 2: Subsample Accumulate (for TSRV)

TSRV requires computing RV on K different subsampled grids (every K-th return).
This is `K` parallel windowed accumulates with different strides.
Can be expressed as `gather(returns, Strided(offset=k, stride=K))` for k = 0..K-1,
then `K` parallel accumulates.

No new primitive needed — this is composition of existing Strided gather + Windowed accumulate.

### Gap 3: Per-Element Root-Finding (for Implied Vol)

IV computation is Newton-Raphson per option. This is a per-element iterative computation —
cannot be expressed as a single accumulate.
Proposed: `CustomIterative` kernel template — parallel Newton steps across options.
Each option converges independently. Convergence is broadcast-AND across threads.

---

## Part 9: Specific Validation Reference Data

### GARCH(1,1) Reference Parameters

Fit on the canonical SPY daily returns from 2010-01-01 to 2020-12-31 (11 years, ~2770 obs):

| Parameter | Typical Value (SPY daily log-returns) | Notes |
|-----------|--------------------------------------|-------|
| omega     | 1e-6 to 5e-6 (unscaled) | Small — daily variance baseline |
| alpha     | 0.08 to 0.15 | ARCH coefficient |
| beta      | 0.82 to 0.92 | GARCH coefficient |
| alpha+beta | 0.92 to 0.98 | High persistence typical for equities |
| mu (mean) | 3e-4 to 6e-4 | Tiny daily drift |
| Unconditional vol | 0.012 to 0.015 | ~1.2-1.5% daily = ~19-24% annual |

```python
# Download and fit (requires yfinance):
import yfinance as yf
import numpy as np
from arch import arch_model

spy = yf.download("SPY", start="2010-01-01", end="2020-12-31", auto_adjust=True)
returns = 100 * np.log(spy["Close"] / spy["Close"].shift(1)).dropna()  # percentage returns

model  = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant', dist='Normal')
result = model.fit(disp='off')
print(result.summary())

# Validation: result.params should be in ranges above (for percentage returns: scale omega by 10000)
```

### HAR Reference Coefficients

Typical HAR-RV coefficients on S&P 500 daily realized variance (5-min sampling):

| Coefficient | Typical Range | Interpretation |
|-------------|--------------|----------------|
| Intercept   | 1e-5 to 1e-4 | Unconditional RV baseline |
| beta_d (daily) | 0.35 to 0.55 | Day-to-day persistence |
| beta_w (weekly) | 0.25 to 0.45 | Week-to-week persistence |
| beta_m (monthly) | 0.10 to 0.25 | Month-to-month persistence |
| R² | 0.45 to 0.65 | HAR fits equity RV well |

Sum (beta_d + beta_w + beta_m) ≈ 0.90 to 0.95 for typical equity data.

---

## Summary: Gold Standard Package Reference

| Algorithm Family | Python Gold Standard | R Gold Standard |
|-----------------|---------------------|-----------------|
| GARCH(p,q), GJR, EGARCH | `arch` (Kevin Sheppard, pypi: arch) | `rugarch` |
| FIGARCH | — (no clean Python impl) | `rugarch` |
| Realized Variance (RV) | manual (highfrequency is R-only) | `highfrequency` |
| BPV, TSRV | manual | `highfrequency` |
| HAR-RV | manual OLS | `highfrequency::HARmodel` |
| Stochastic Volatility | `pymc` (flexible MCMC) | `stochvol` (KSC, efficient) |
| Implied Vol | `scipy.optimize.brentq` | `RQuantLib`, `fOptions` |
| VIX-style | manual (CBOE formula) | manual |

**Primary oracles**:
- Python `arch` for GARCH family (version: arch >= 5.0.0 for `rescale` parameter)
- R `highfrequency` for all realized measures (version: >= 0.6.0)
- R `stochvol` for stochastic volatility (version: >= 3.0.0)
- `scipy.optimize.brentq` for IV (part of scipy, always available)
