# F17 Time Series Models — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 17 (Time Series: AR, ARIMA, GARCH, Kalman).
All members of Kingdom B — Affine scan with varying A, b matrices.
The naturalist's claim: ALL of these are the same Affine scan, differently parameterized.
This document verifies that claim against the gold standards.

---

## The Kingdom B Claim: All Time Series = Affine Scan

```
State update: sₜ = Aₜ · sₜ₋₁ + bₜ + noise
```

Where Aₜ, bₜ may be:
- Fixed matrix, zero input: AR(p) state space form
- State-dependent matrix: GARCH (variance as state, nonlinear A)
- Data-dependent matrix: Kalman filter (A = system dynamics matrix F)
- Learned matrix: LSTM/RNN (A = learned recurrent weights)

**Verification from source**: this section confirms each algorithm fits the Affine scan structure.

---

## ARIMA

### The State Space Representation

ARIMA(p,d,q) can be written in state space form with:
```
State sₜ = [xₜ, xₜ₋₁, ..., xₜ₋ₚ₊₁, εₜ, εₜ₋₁, ..., εₜ₋q₊₁]'  (p+q dimensional)

Aₜ = A (constant companion matrix):
A = [[φ₁, φ₂, ..., φₚ, θ₁, θ₂, ..., θq],
     [1,   0, ...,  0,  0, ...,       0],
     ...] (companion matrix form)

bₜ = [εₜ, 0, ..., 0]'  (noise enters first component only)
```

**Confirms Affine scan**: A is constant, b is the noise vector. This is the simplest
case of the Affine scan — same A matrix at every step.

### R: stats::arima

```r
# Fit ARIMA:
fit <- arima(x, order=c(p, d, q))
fit <- arima(x, order=c(2, 1, 1))  # ARIMA(2,1,1)

# From arima object:
fit$coef            # AR and MA coefficients (φ₁...φₚ, θ₁...θq)
fit$sigma2          # estimated noise variance
fit$loglik          # log-likelihood
fit$aic             # AIC

# forecast package (auto.arima):
library(forecast)
fit <- auto.arima(x)            # automatic model selection
fitted(fit)                      # in-sample predictions
residuals(fit)                   # innovations εₜ
forecast(fit, h=12)              # h-step ahead forecasts
```

### Python: statsmodels ARIMA

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

# Fit:
model = ARIMA(endog=x, order=(2, 1, 1))
result = model.fit()

# Access:
result.params              # AR/MA coefficients
result.resid               # innovations
result.fittedvalues        # in-sample fitted values
result.predict(start, end) # predictions
result.forecast(steps=12)  # out-of-sample forecasts
result.summary()           # full summary table

# Seasonal ARIMA:
model = SARIMAX(x, order=(p,d,q), seasonal_order=(P,D,Q,s))
```

**Important**: statsmodels ARIMA uses Kalman filter internally for state space estimation.
The `result.filter_results` exposes the Kalman filter quantities directly.

---

## Exponential Weighted Mean (EWM)

The simplest Affine scan. `sₜ = α·xₜ + (1-α)·sₜ₋₁`.

```
A = (1-α), b = α·xₜ  (scalar case)
```

```python
import pandas as pd
s = pd.Series(x)
s.ewm(alpha=0.1).mean()      # EWM mean, A = 1-α
s.ewm(alpha=0.1).std()       # EWM std (more complex)
s.ewm(span=10).mean()        # span=10 ↔ α = 2/(10+1) ≈ 0.182
s.ewm(halflife=5).mean()     # halflife: α = 1 - exp(-log(2)/halflife)
```

```r
# R: filter() or TTR package:
filter(x, filter=(1-alpha), method="recursive", init=x[1])
TTR::EMA(x, n=10)   # exponential moving average
```

**Verification**: this is literally A = (1-α), b = α·xₜ. The most degenerate case of Affine scan.

---

## Kalman Filter

### State Space Form (general linear dynamical system)

```
State equation:   sₜ = F · sₜ₋₁ + G · uₜ + Q^{1/2} · wₜ    (wₜ ~ N(0,I))
Observation:      yₜ = H · sₜ + R^{1/2} · vₜ                (vₜ ~ N(0,I))

where:
  F = state transition matrix (p×p)
  G = input matrix (p×m)
  H = observation matrix (d×p)
  Q = process noise covariance (p×p)
  R = observation noise covariance (d×d)
```

**This IS an Affine scan** with Aₜ = F (constant for linear time-invariant systems),
bₜ = G·uₜ + Kalman gain update. The predict-update cycle is a structured Affine step.

### Kalman Filter Equations

```
PREDICT:
  s_{t|t-1} = F · s_{t-1|t-1}                          (state prediction)
  P_{t|t-1} = F · P_{t-1|t-1} · F' + Q                 (covariance prediction)

UPDATE:
  K_t = P_{t|t-1} · H' · (H · P_{t|t-1} · H' + R)^{-1}  (Kalman gain)
  s_{t|t} = s_{t|t-1} + K_t · (y_t - H · s_{t|t-1})       (state update)
  P_{t|t} = (I - K_t · H) · P_{t|t-1}                      (covariance update)
```

The state (sₜ, Pₜ) = AffineState in tambear.
The Kalman gain Kₜ = P_{t|t-1}H'(...)^{-1} requires a matrix inversion per step.
This is a more complex Affine update than AR — it involves a matrix inversion.

**Tambear's Särkka operator** (mentioned in naturalist's document) handles this.
Särkka (2013) "Bayesian Filtering and Smoothing" — the standard reference.

### R: dlm Package (Kalman)

```r
library(dlm)

# Define model (local level model, random walk + noise):
model <- dlmModPoly(order=1, dV=0.5, dW=0.1)  # obs noise + process noise

# Or manually:
model <- dlm(
  FF = matrix(1),         # H observation matrix
  GG = matrix(1),         # F state transition
  V  = matrix(0.5),       # R observation noise variance
  W  = matrix(0.1),       # Q process noise variance
  m0 = 0,                 # initial state mean
  C0 = matrix(1000)       # initial state covariance
)

# Kalman filter:
filtered <- dlmFilter(y, model)
filtered$m    # filtered state estimates s_{t|t}
filtered$C    # filtered state covariances P_{t|t} (as list)

# Kalman smoother (backward pass = reverse Affine scan):
smoothed <- dlmSmooth(filtered)
smoothed$s    # smoothed state estimates
```

### Python: filterpy

```python
from filterpy.kalman import KalmanFilter

kf = KalmanFilter(dim_x=2, dim_z=1)  # 2D state, 1D observation
kf.F = np.array([[1, 1], [0, 1]])     # state transition (constant velocity)
kf.H = np.array([[1, 0]])             # observation matrix
kf.R = np.array([[5.]])               # observation noise
kf.Q = np.array([[0.1, 0.], [0., 0.1]])  # process noise
kf.x = np.array([[0.], [0.]])         # initial state
kf.P = np.eye(2) * 1000.              # initial covariance

for z in measurements:
    kf.predict()
    kf.update(np.array([[z]]))
    print(kf.x, kf.P)

# Or batch processing:
from filterpy.kalman import predict, update
```

### Python: statsmodels State Space (most complete)

```python
import statsmodels.api as sm

# Custom linear state space model:
class LocalLevelModel(sm.tsa.statespace.MLEModel):
    def __init__(self, endog):
        super().__init__(endog, k_states=1, k_posdef=1)
        self['design'] = 1.0
        self['transition'] = 1.0
        self['selection'] = 1.0

# Pre-built models:
result = sm.tsa.UnobservedComponents(y, level='local level').fit()
result.filtered_state      # s_{t|t} — shape (p, T)
result.filtered_state_cov  # P_{t|t} — shape (p, p, T)
result.smoothed_state      # backward-smoothed state
result.smoothed_state_cov  # backward-smoothed covariance
result.loglikelihood_burn  # log-likelihood
```

**Key**: statsmodels uses the Durbin-Koopman (2001) disturbance smoother internally —
more efficient than the classic Kalman smoother. For tambear validation, the filtered states
should match exactly; smoothed states differ only by backward-pass precision.

---

## GARCH

### Why GARCH is Kingdom B (with a nonlinearity)

GARCH(1,1) for variance:
```
σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁
```

This is a scalar Affine recurrence: `sₜ = β·sₜ₋₁ + (ω + α·ε²ₜ₋₁)`.
State = σ²ₜ (variance), A = β (constant), bₜ = ω + α·ε²ₜ₋₁ (data-dependent input).

**It IS an Affine scan**: constant A = β, time-varying b = ω + α·ε²ₜ₋₁.
The e²ₜ₋₁ term is computed from the previous residual (not a nonlinearity of the state).

The constraint β < 1 ensures stationarity (just like |AR coefficient| < 1 for AR stability).

### Python: arch Package (gold standard for GARCH)

```python
from arch import arch_model

# GARCH(1,1):
model = arch_model(returns, vol='Garch', p=1, q=1)
result = model.fit()
result.summary()
result.conditional_volatility    # σₜ estimates
result.resid                     # standardized residuals
result.params                    # ω, α, β
result.loglikelihood             # log-likelihood

# GJR-GARCH (asymmetric, leverage effect):
model = arch_model(returns, vol='Garch', p=1, o=1, q=1)  # o=asymmetric term

# EGARCH:
model = arch_model(returns, vol='EGarch', p=1, q=1)
```

```r
library(rugarch)
spec <- ugarchspec(
  variance.model=list(model="sGARCH", garchOrder=c(1,1)),
  mean.model=list(armaOrder=c(0,0), include.mean=TRUE)
)
fit <- ugarchfit(spec=spec, data=returns)
fit@fit$coef                    # ω, α, β
sigma(fit)                      # σₜ
residuals(fit, standardize=TRUE) # standardized residuals
likelihood(fit)                  # log-likelihood
```

**Trap**: arch package `conditional_volatility` returns σₜ (not σ²ₜ). R's rugarch `sigma(fit)` also returns σₜ. Tambear should store σ²ₜ (the state) and derive σₜ = sqrt(σ²ₜ) on output.

---

## The Kalman = Recursive Least Squares Identity (naturalist's rhyme)

For Kalman with:
- F = I (identity state transition, random walk prior)
- H = xₜ' (current feature vector as observation matrix)
- Q = 0 (no process noise — static coefficient assumption)
- R = σ²ε (observation noise)

The Kalman filter reduces to Recursive Least Squares (RLS):
```
βₜ = βₜ₋₁ + K_t · (yₜ - xₜ'·βₜ₋₁)
K_t = Pₜ₋₁·xₜ / (xₜ'·Pₜ₋₁·xₜ + σ²ε)
Pₜ = (I - K_t·xₜ')·Pₜ₋₁
```

This IS recursive OLS — updating the regression coefficient as new data arrives.
The state is the coefficient vector β and its covariance matrix P.
The Affine update: Aₜ = I - K_t·xₜ' (depends on data), bₜ = K_t·yₜ.

**Verification**: R's `dlm` with F=I, H=xₜ' reproduces recursive least squares.

```r
# Verify Kalman = RLS identity:
library(dlm)
# Dynamic regression model where state = regression coefficients:
model <- dlmModReg(X=design_matrix, dV=sigma2, dW=0)  # process noise = 0 → static β
filtered <- dlmFilter(y, model)
# filtered$m[T,] should match lm(y ~ X)$coef (final batch OLS)
```

---

## State MSR for Kingdom B

For each family member, the AffineState contains different quantities:

| Algorithm | State sₜ | Covariance Pₜ | A (transition) |
|-----------|----------|--------------|----------------|
| AR(p) | [xₜ, ..., xₜ₋ₚ₊₁] | scalar σ² | companion matrix |
| EWM | μₜ (scalar) | none | (1-α) |
| Kalman | system state (latent) | full p×p | F (system dynamics) |
| RLS | β vector | p×p inv-info | I - K·x' (data-dependent) |
| GARCH | σ²ₜ (scalar) | none | β (constant) |
| RNN/LSTM | hidden state h | none | learned W |

All produce time-indexed state sequences. The AffineState MSR = {sₜ, Pₜ, t ∈ 1..T}.

---

## Validation: AR(1) as Simplest Case

```python
from statsmodels.tsa.ar_model import AutoReg
import numpy as np

np.random.seed(42)
n = 200
phi = 0.8
e = np.random.randn(n)
x = np.zeros(n)
for t in range(1, n):
    x[t] = phi * x[t-1] + e[t]

# Fit AR(1):
result = AutoReg(x, lags=1).fit()
print(f"Fitted φ₁: {result.params[1]:.4f}")  # should be ≈ 0.8
print(f"True φ₁:   {phi}")

# Verify against Kalman (state space representation):
from statsmodels.tsa.statespace.sarimax import SARIMAX
kf_result = SARIMAX(x, order=(1,0,0)).fit(disp=False)
print(f"Kalman φ₁: {kf_result.params[0]:.4f}")
# Both should give ≈ 0.8 — confirms AR = special case of Kalman
```

---

## Parallel Prefix Scan (GPU Implementation Note)

The Affine scan `sₜ = Aₜ · sₜ₋₁ + bₜ` is parallelizable via parallel prefix scan
using the associative operator `(A₂, b₂) ∘ (A₁, b₁) = (A₂·A₁, A₂·b₁ + b₂)`.

This IS associative — the classic result from Blelloch (1990).
O(log N) depth on GPU, O(N) work.

For time-varying (A, b): each element has its own (Aₜ, bₜ) pair.
Parallel prefix over these pairs gives all states simultaneously.

**R reference**: no direct parallel scan in base R. Used sequentially.
**Python reference**: JAX's `jax.lax.associative_scan` implements this exactly.

```python
import jax.numpy as jnp
import jax

def affine_combine(carry, x):
    A_prev, b_prev = carry
    A_curr, b_curr = x
    return (A_curr @ A_prev, A_curr @ b_prev + b_curr)

# Parallel prefix over sequence of (A, b) pairs:
jax.lax.associative_scan(affine_combine, (A_seq, b_seq))
```

This is the JAX gold standard for parallel Affine scan.
Tambear's Affine operator should produce identical results.
