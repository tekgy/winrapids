# Family 17: Time Series Models — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: B (Affine scan) — the entire family is parameterized instances of `accumulate(Prefix(forward), input, Affine(A,b))`

---

## Core Insight: Every Time Series Model Is an Affine Scan

The structural rhyme that unifies this family: every classical time series model is a linear recurrence of the form:

```
state_{t+1} = A · state_t + b_t
```

This is `accumulate(Prefix(forward), b_t, Affine(A, b))`. The Affine operator IS the time series family.

| Model | State | A | b_t |
|-------|-------|---|-----|
| EWM (mean) | mean_t | α | (1-α)·x_t |
| SES | level_t | α | (1-α)·x_t |
| Holt (trend) | [level, trend] | [[α, α], [0, 1-β]] | [(1-α)·x_t, β·...] |
| Holt-Winters | [level, trend, seasonal[s]] | 3×3 block | ... |
| AR(p) | [x_t, ..., x_{t-p+1}] | companion matrix | [ε_t, 0, ..., 0] |
| ARMA(p,q) | augmented state | augmented companion | ... |
| Kalman filter | [state, covariance] | state transition | [innovation, ...] |
| Adam optimizer | [m, v] | [β₁, β₂] | [(1-β₁)·g, (1-β₂)·g²] |

**Structural rhyme**: Adam optimizer (F05) = 4 independent EWM channels. Same Affine scan, different parameters.

---

## 1. ARIMA(p,d,q)

### Model
```
φ(B) · (1-B)^d · x_t = θ(B) · ε_t
```
where B is the backshift operator, φ(B) = 1 - φ₁B - ... - φ_pB^p (AR), θ(B) = 1 + θ₁B + ... + θ_qB^q (MA).

### Decomposition
1. **Differencing** (d times): `x'_t = x_t - x_{t-1}`. This is `accumulate(Prefix(forward), x_t, Diff)` or gather with offset -1 then subtract.
2. **AR part**: linear combination of past values. State = [x'_t, ..., x'_{t-p+1}]. Update = companion matrix multiply. **Affine scan.**
3. **MA part**: linear combination of past errors. Requires error estimation (iterative). **Kingdom C wrapper** for estimation, Kingdom B for filtering.

### State Space Form (canonical)
```
state_t = [x_t, x_{t-1}, ..., x_{t-p+1}, ε_t, ε_{t-1}, ..., ε_{t-q+1}]

Transition: F = [[φ₁, φ₂, ..., φ_p, θ₁, θ₂, ..., θ_q],
                  [1,   0,  ..., 0,   0,   0,  ..., 0  ],
                  ...
                  [0,   0,  ..., 0,   1,   0,  ..., 0  ]]
```

### Parameter Estimation

| Method | Complexity | Use when |
|--------|-----------|----------|
| Conditional least squares | O(n·p) per iteration | Quick and dirty |
| Exact MLE (Kalman filter) | O(n·p²) per iteration | Gold standard |
| CSS-MLE (conditional start) | O(n·p) per iteration | Large n |
| Burg (AR only) | O(n·p) one pass | AR models, fast |
| Yule-Walker (AR only) | O(n + p²) | AR models, closed-form |
| Hannan-Rissanen (ARMA) | O(n·(p+q)) 3-stage | ARMA, fast initial |

### CRITICAL: Stationarity and Invertibility
- **AR stationarity**: all roots of φ(z) = 0 must lie OUTSIDE the unit circle
- **MA invertibility**: all roots of θ(z) = 0 must lie OUTSIDE the unit circle
- Check after estimation. If violated: reflect roots inside unit circle to outside (statsmodels approach).
- **Differencing** (d): makes non-stationary series stationary. Unit root tests (ADF, KPSS, PP) determine d.

### SARIMA(p,d,q)(P,D,Q)[s]
Seasonal ARIMA: additional seasonal AR/MA terms at lag s.
```
Φ(B^s) · φ(B) · (1-B^s)^D · (1-B)^d · x_t = Θ(B^s) · θ(B) · ε_t
```

Same state space form, larger state vector. Same Affine scan.

### GPU decomposition
- Differencing: gather(offset=-1) + subtract. Parallel.
- Filtering (given parameters): Affine scan. Kingdom B.
- Parameter estimation: optimization (F05) or Kalman recursion. Kingdom C.
- Forecasting: extend the Affine scan forward.

---

## 2. Exponential Smoothing (ETS)

### SES (Simple Exponential Smoothing)
```
l_t = α·x_t + (1-α)·l_{t-1}
```
This IS `accumulate(Prefix(forward), α·x_t, Affine(1-α, 0))`.

Forecast: ŷ_{t+h} = l_t (flat).

### Holt (Linear Trend)
```
l_t = α·x_t + (1-α)·(l_{t-1} + b_{t-1})
b_t = β·(l_t - l_{t-1}) + (1-β)·b_{t-1}
```
State = [l_t, b_t]. 2×2 Affine scan.

Forecast: ŷ_{t+h} = l_t + h·b_t.

### Holt-Winters (Seasonal)
**Additive seasonal:**
```
l_t = α·(x_t - s_{t-m}) + (1-α)·(l_{t-1} + b_{t-1})
b_t = β·(l_t - l_{t-1}) + (1-β)·b_{t-1}
s_t = γ·(x_t - l_t) + (1-γ)·s_{t-m}
```

State = [l_t, b_t, s_t, s_{t-1}, ..., s_{t-m+1}]. (2+m)-dimensional Affine scan.

**Multiplicative seasonal:** replace addition/subtraction with multiplication/division for seasonal component. NOT a linear recurrence — requires log transform to make Affine.

### Damped Trend
```
b_t = β·(l_t - l_{t-1}) + (1-β)·φ·b_{t-1}
```
Forecast: ŷ_{t+h} = l_t + (φ + φ² + ... + φ^h)·b_t. Prevents explosive forecasts.

### ETS Taxonomy (Hyndman et al.)
30 models from 3 choices: Error (Additive/Multiplicative) × Trend (None/Additive/Damped) × Seasonal (None/Additive/Multiplicative).

Model selection: minimize AIC/AICc/BIC.

### GPU decomposition
- All ETS models: Affine scan with appropriate state dimension
- Parameter optimization: F05 (minimize forecast error)

---

## 3. State Space Models (SSM)

### General Linear Gaussian SSM
```
State equation:     α_{t+1} = T·α_t + R·η_t     η_t ~ N(0, Q)
Observation:        y_t = Z·α_t + ε_t             ε_t ~ N(0, H)
```

### Kalman Filter (already documented as Särkkä operator)
```
Prediction:  a_{t|t-1} = T·a_{t-1|t-1}
             P_{t|t-1} = T·P_{t-1|t-1}·T' + R·Q·R'

Innovation:  v_t = y_t - Z·a_{t|t-1}
             F_t = Z·P_{t|t-1}·Z' + H

Update:      K_t = P_{t|t-1}·Z'·F_t⁻¹
             a_{t|t} = a_{t|t-1} + K_t·v_t
             P_{t|t} = (I - K_t·Z)·P_{t|t-1}
```

### Relationship to Other Models
- **SES** = SSM with Z=1, T=1, R=1 (local level model)
- **Holt** = SSM with Z=[1,0], T=[[1,1],[0,1]] (local linear trend)
- **ARIMA(p,d,q)** = SSM in companion form
- **All ETS** = SSM with specific T, Z, R matrices

**Everything is a Kalman filter.** The different models are just different parameterizations of T, Z, R, Q, H.

### Kalman Smoother
Run filter forward, then backward pass:
```
a_{t|T} = a_{t|t} + L_t·(a_{t+1|T} - a_{t+1|t})
P_{t|T} = P_{t|t} + L_t·(P_{t+1|T} - P_{t+1|t})·L_t'
```
where L_t = P_{t|t}·T'·P_{t+1|t}⁻¹.

**This is `accumulate(Prefix(reverse), smoother_state, Affine)`** — a REVERSE Affine scan. Same operator, reverse direction.

### GPU decomposition
- Forward filter: `accumulate(Prefix(forward), innovation, Särkkä)` — sequential per time series
- Backward smoother: `accumulate(Prefix(reverse), smoother_state, Affine)` — reverse scan
- **Batch across time series**: many independent time series in parallel. This is where GPU wins — filter 10,000 tickers simultaneously.

---

## 4. Unit Root Tests

### ADF (Augmented Dickey-Fuller)
Test H₀: φ = 1 (unit root) in:
```
Δx_t = (φ-1)·x_{t-1} + Σ_{j=1}^p γ_j·Δx_{t-j} + deterministic + ε_t
```

Implementation: F10 OLS regression → t-statistic on (φ-1) coefficient → compare to DF critical values (NOT Student-t).

### KPSS (Kwiatkowski-Phillips-Schmidt-Shin)
Test H₀: series IS stationary. Complementary to ADF.

```
η = (1/T²) · Σ_{t=1}^T S²_t / σ̂²_∞
```
where S_t = partial sum of residuals, σ̂²_∞ = Newey-West long-run variance estimator.

### PP (Phillips-Perron)
Like ADF but uses Newey-West correction instead of augmented lags. Non-parametric.

### GPU decomposition
All unit root tests: F10 OLS regression (parallel across multiple time series).

---

## 5. Structural Break Detection

### CUSUM
```
CUSUM_t = Σ_{k=1}^t (x_k - x̄) / σ̂
```
Prefix sum of centered observations. Flag when |CUSUM_t| > boundary.

**GPU**: `accumulate(Prefix(forward), (x_t - mean) / std, Add)` — prefix scan.

### Bai-Perron (multiple structural breaks)
Dynamic programming to find m breakpoints minimizing total SSR:
```
min_{b₁,...,bₘ} Σ_{j=0}^m SSR(b_j+1, ..., b_{j+1})
```

O(n²·m) with dynamic programming. Each SSR = F06 moment accumulation on segment.

### PELT (Pruned Exact Linear Time)
Changepoint detection with pruning:
```
F(t) = min_{s<t} [F(s) + C(y_{s+1:t}) + β]
```
where C is a cost function (e.g., negative log-likelihood) and β is a penalty (BIC, MBIC, Hannan-Quinn).

### GPU decomposition
- CUSUM: prefix scan (parallel)
- PELT cost: segment-wise accumulate (parallelizable across segments)
- DP: sequential over breakpoint count (Kingdom C), but cost evaluation is parallel

---

## 6. Forecasting

### Point Forecasts
Given fitted model, extend the Affine scan forward with no new observations:
```
ŷ_{t+h} = Z · T^h · a_{t|t}     (SSM form)
```

For ARIMA: companion matrix powers.
For ETS: recursive application of smoothing equations with x_t replaced by forecast.

### Prediction Intervals
```
Var(e_{t+h}) = σ² · Σ_{j=0}^{h-1} ||ψ_j||²
```
where ψ_j are MA(∞) coefficients (derived from AR/MA parameters).

PI = ŷ_{t+h} ± z_{α/2} · √Var(e_{t+h})

### CRITICAL: Prediction intervals widen with horizon. Report them alongside point forecasts (V columns).

---

## 7. Model Selection

### Information Criteria
```
AIC = -2·logL + 2·k
AICc = AIC + 2k(k+1)/(n-k-1)     (corrected for small samples)
BIC = -2·logL + k·log(n)
```
where k = number of parameters, n = sample size.

### Auto-ARIMA (Hyndman-Khandakar algorithm)
1. Determine d via unit root tests (KPSS)
2. Determine D via seasonal unit root test (OCSB)
3. Try initial models based on ACF/PACF
4. Stepwise search: vary p,q by ±1, keep best AICc
5. Also try p,q ∈ {0,1,2} × {0,1,2} for seasonal P,Q

### GPU: evaluate many candidate models in parallel (each model = independent Kalman filter).

---

## 8. Numerical Stability

### Kalman Filter
- **Covariance matrix must remain positive definite.** Joseph form update prevents numerical drift:
  ```
  P_{t|t} = (I-KZ)P_{t|t-1}(I-KZ)' + K·H·K'
  ```
  More stable than P = P - KZP.
- **Square-root filter**: propagate Cholesky factor of P instead of P. Halves the condition number.
- **Innovation form**: monitor v_t/√F_t. Should be approximately N(0,1). Large values indicate model misspecification.

### ARIMA
- **Near-unit-root AR**: φ ≈ 1 makes companion matrix ill-conditioned. Use exact MLE (Kalman), not CSS.
- **MA invertibility boundary**: θ ≈ -1 causes flat likelihood. Constrain optimization.
- **Long seasonal periods** (s > 24): state vector becomes large. Use concentrated likelihood or Fourier approximation.

### General
- **Log-likelihood in log-space**: sum log-likelihoods, don't multiply likelihoods (underflow).
- **Scaling**: standardize data before fitting (prevents coefficient magnitude issues).

---

## 9. Edge Cases

| Algorithm | Edge Case | Expected |
|-----------|----------|----------|
| ARIMA | n < 2(p+q)+d | Too few observations. Error. |
| ARIMA | All values identical | Variance = 0, model undefined. Return constant. |
| ARIMA | Missing values | Kalman filter handles naturally (skip update step) |
| ETS | α = 0 or α = 1 | Boundary: no smoothing or no memory. Valid but degenerate. |
| Kalman | P → 0 (filter lock) | Steady-state Kalman. No further learning. |
| Kalman | Divergent P | Model misspecification. Flag via V column. |
| ADF | Trend in data | Include deterministic trend in regression. |
| CUSUM | No break | CUSUM stays within boundaries. Return null. |
| Forecast | h > n | Very wide PI. Warn user. |

---

## Sharing Surface

### Reuses from Other Families
- **F02 (Linear Algebra)**: matrix operations for state space, companion matrix
- **F05 (Optimization)**: MLE parameter estimation (L-BFGS, Newton)
- **F06 (Descriptive)**: mean, variance for centering, residual diagnostics
- **F07 (Hypothesis)**: t-statistics for parameter significance, ADF critical values
- **F10 (Regression)**: OLS for ADF test, trend estimation

### Provides to Other Families
- **F18 (Volatility)**: ARMA mean equation (GARCH models = F17 mean + F18 variance)
- **F19 (Spectral TS)**: AR spectral density estimation (Burg PSD = F17 AR fit → F19 spectral)
- **F12 (Panel)**: time series components for panel data models
- **F05 (Optimization)**: Adam = 4 EWM channels (structural rhyme — same Affine scan)

### Structural Rhymes
- **EWM = SES** = simplest Affine scan
- **Adam = 4 EWM channels**: m = EWM of gradient, v = EWM of squared gradient
- **Kalman forward + backward = bidirectional scan**: `accumulate(Prefix(bidirectional), ...)`
- **CUSUM = prefix sum of residuals**: same as F06 cumulative statistics
- **GARCH = EWM with intercept**: Affine(β₁, ω + α₁·ε²_t) — same operator, different b

---

## Implementation Priority

**Phase 1** — Core Affine scan models (~150 lines):
1. SES, Holt, Holt-Winters (all ETS via Affine scan)
2. AR(p) via companion matrix (Yule-Walker + Burg estimation)
3. Kalman filter (Särkkä operator, forward pass)
4. Unit root tests: ADF, KPSS (via F10 OLS)

**Phase 2** — ARIMA + state space (~200 lines):
5. ARIMA(p,d,q) estimation (exact MLE via Kalman)
6. SARIMA extension
7. Kalman smoother (reverse Affine scan)
8. Auto-ARIMA (Hyndman-Khandakar stepwise)

**Phase 3** — Structural breaks + forecasting (~150 lines):
9. CUSUM (prefix scan)
10. PELT changepoint detection
11. Bai-Perron multiple breaks
12. Forecast with prediction intervals

**Phase 4** — Extensions (~100 lines):
13. VAR (Vector AR) — multivariate extension
14. VECM (Vector Error Correction)
15. Granger causality (F-test on restricted vs unrestricted VAR)
16. Cointegration (Johansen trace/max eigenvalue tests)

---

## Composability Contract

```toml
[family_17]
name = "Time Series Models"
kingdom = "B (Affine scan — all models are parameterized linear recurrences)"

[family_17.shared_primitives]
affine_scan = "accumulate(Prefix(forward), input, Affine(A, b)) — the universal time series primitive"
kalman_filter = "Särkkä operator for exact transient Kalman filtering"
companion_matrix = "AR(p) state transition matrix"
differencing = "gather(offset=-1) + subtract for integration order d"

[family_17.reuses]
f02_linear_algebra = "Matrix operations for state space, companion matrix"
f05_optimization = "MLE for parameter estimation"
f06_descriptive = "Centering, residual statistics"
f07_hypothesis = "ADF critical values, parameter significance"
f10_regression = "OLS for ADF test, trend estimation"

[family_17.provides]
filtered_state = "Kalman filter state estimates"
forecasts = "Point forecasts with prediction intervals"
residuals = "Model residuals for diagnostic checking"
smoothed_state = "Kalman smoother full-sample estimates"
ar_coefficients = "AR/ARMA parameters for spectral estimation (F19)"

[family_17.consumers]
f18_volatility = "Mean equation for GARCH models"
f19_spectral = "AR coefficients for Burg PSD"
f12_panel = "Time series dynamics in panel models"
f05_optimization = "Adam = same Affine scan (structural rhyme)"
```
