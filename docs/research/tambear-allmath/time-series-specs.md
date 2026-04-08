# Time Series Algorithm Specifications
**Author**: math-researcher  
**Date**: 2026-04-06  
**Scope**: ARIMA, Kalman filter, GARCH diagnostics — implementation blueprint for pathmaker.  
**Prerequisites**: `time_series.rs` already has AR(p), differencing, ACF/PACF, ADF, exponential smoothing.

---

## Existing (verified correct):
- `ar_fit`: Yule-Walker + Levinson-Durbin ✅
- `difference`: d-order differencing ✅
- `adf_test`: Augmented Dickey-Fuller ✅
- `acf`, `pacf`: sample ACF/PACF via Levinson-Durbin ✅
- `garch11_fit`, `garch11_forecast` in `volatility.rs` ✅

## Gaps to Implement:
1. MA(q) fitting (needed for ARMA)
2. ARMA(p,q) via conditional log-likelihood
3. ARIMA(p,d,q) pipeline
4. Information criteria for order selection
5. Ljung-Box portmanteau test
6. Linear Kalman filter + smoother
7. GARCH diagnostic tests (Ljung-Box on squared residuals)

---

## 1. MA(q) Model

### Source
Box & Jenkins (1976) *Time Series Analysis*; Brockwell & Davis (1991) §3.

### Model
```
y_t = μ + ε_t + θ₁ε_{t-1} + θ₂ε_{t-2} + ... + θ_q ε_{t-q}
ε_t ~ i.i.d. N(0, σ²)
```

### Log-likelihood (conditional on ε_{1-q} = ... = ε_0 = 0)
```
ε_t = y_t - μ - Σⱼ θⱼ ε_{t-j}     [innovation recursion]

ℓ(θ, σ², μ) = -n/2 · ln(2π) - n/2 · ln(σ²) - (1/2σ²) Σ ε²_t
```

Maximize over μ first: μ̂ = ȳ (mean of y). Then maximize over θ, σ².

### Optimization
Because the innovation recursion is sequential, MA parameters require numerical MLE (no closed form).

**Algorithm**:
1. Initialize θ = 0, σ² = var(y)
2. For each candidate θ, evaluate ℓ(θ) by running innovation recursion
3. Minimize -ℓ(θ) using L-BFGS or Nelder-Mead
4. σ̂² = (1/n) Σ ε̂²_t

**Gradient** (for L-BFGS):
```
∂ℓ/∂θⱼ = (1/σ²) Σ_t ε_t · (∂ε_t/∂θⱼ)
```
where the partial derivatives satisfy the recursion:
```
∂ε_t/∂θⱼ = -ε_{t-j} - Σ_{k=1}^{q} θₖ · (∂ε_{t-k}/∂θⱼ)
```
Initial: ∂ε_{t}/∂θⱼ = 0 for t ≤ 0.

This recursive gradient is O(nq) per evaluation.

### Stationarity/Invertibility
MA(q) is always stationary. Invertibility requires roots of θ(z) = 1 + θ₁z + ... + θ_q z^q outside the unit circle. Enforce during optimization by transforming to unconstrained parameterization if needed.

### Sufficient Statistics
None in closed form — MA requires the full innovation sequence. The MSR is the recursive filter state {ε_{t-q}, ..., ε_{t-1}}.

---

## 2. ARMA(p,q) Model

### Source
Box & Jenkins (1976); Brockwell & Davis §3.3; Hamilton (1994) §4.

### Model
```
y_t = c + φ₁y_{t-1} + ... + φ_p y_{t-p} + ε_t + θ₁ε_{t-1} + ... + θ_q ε_{t-q}
```

### Conditional Log-likelihood
Given initial values y₁₋ₚ,...,y₀ (use actuals or backcast to mean):
```
ε_t = y_t - c - Σⱼ φⱼ y_{t-j} - Σⱼ θⱼ ε_{t-j}

ℓ(φ,θ,σ²) = -n/2 · ln(2π) - n/2 · ln(σ²) - (1/2σ²) Σ_{t=1}^{n} ε²_t
```

σ̂²(φ,θ) = (1/n) Σ ε²_t, so concentrate out:
```
ℓ_concentrated(φ,θ) = -n/2 · (1 + ln(2π) + ln(σ̂²))
```

Minimize -ℓ_concentrated over (φ,θ) ∈ ℝ^{p+q}.

### Initialization
- AR parameters: Yule-Walker solution (from `ar_fit`)
- MA parameters: 0
- Initial innovations: 0
- Optional: backcast using the unconditional mean

### Order of Optimization
1. Fix p and q (user-specified or from information criteria)
2. Run L-BFGS on (φ₁,...,φₚ,θ₁,...,θ_q) with concentrated ℓ
3. Compute σ̂² from final residuals
4. Compute standard errors from numerical Hessian (or finite-difference approximation)

### Accumulate+Gather Decomposition
```
1. AR recursion: accumulate(lags 1..p, phi_j * y_{t-j}, sum) → AR prediction
2. MA recursion: accumulate(innovations 1..q, theta_j * eps_{t-j}, sum) → MA correction
3. Innovation: eps_t = y_t - AR_pred - MA_corr   [gather operation]
4. accumulate(t=1..n, eps_t², sum) → SSE  [for log-likelihood]
```
Pattern: sequential scan (Kingdom C, iteration over t cannot be parallelized).

---

## 3. ARIMA(p,d,q)

### Source
Box & Jenkins (1976) §6.

### Pipeline
```
1. Difference y d times: Δᵈy_t = difference(y, d)    [from existing difference()]
2. Fit ARMA(p,q) to Δᵈy                               [from new arma_fit()]
3. Forecast Δᵈy at horizon h
4. Undo differencing: integrate back d times
```

**Integration (undoing differences)**:
```
Given d=1 and last d observations of original series y[n-1]:
y_hat[t] = y[n-1] + Σ_{s=1}^{t} Δy_hat[s]
```
For d=2, apply twice.

```
// For d=1:
fn integrate(diff_forecast: &[f64], last_obs: f64) -> Vec<f64> {
    let mut result = vec![0.0; diff_forecast.len()];
    let mut prev = last_obs;
    for (i, &d) in diff_forecast.iter().enumerate() {
        prev += d;
        result[i] = prev;
    }
    result
}
```

### Forecast Confidence Intervals
For ARIMA forecast, the h-step-ahead variance grows as:
```
Var(y_{n+h} - ŷ_{n+h}) = σ² Σ_{j=0}^{h-1} ψ²_j
```
where ψⱼ are the MA(∞) coefficients of the ARMA process.

The ψ-weights can be computed recursively from (φ, θ):
```
ψ_0 = 1
ψ_j = θⱼ + Σₖ φₖ ψ_{j-k}   for j ≥ 1
     (with θⱼ = 0 for j > q, φₖ = 0 for k > p)
```

**95% confidence interval**: ŷ_{n+h} ± 1.96 · σ̂ · √(Σ_{j=0}^{h-1} ψ²_j)

---

## 4. Information Criteria for ARIMA Order Selection

### Source
Akaike (1974); Schwarz (1978); Hurvich & Tsai (1989) for AICc.

### Formulas
For ARIMA(p,d,q) fitted on differenced series of length n' = n-d:
```
k = p + q + 1     [parameters: p AR, q MA, σ²]

AIC  = -2ℓ + 2k
BIC  = -2ℓ + k·ln(n')
AICc = AIC + 2k(k+1)/(n'-k-1)    [small-sample corrected — use always]
```

### Auto-ARIMA Algorithm
Grid search over (p, d, q):
```
For d in 0..2:  [check ADF test]
  Δᵈy = difference(y, d)
  For p in 0..P_max (e.g., P_max=5):
    For q in 0..Q_max (e.g., Q_max=5):
      Fit ARMA(p,q) to Δᵈy
      Compute AICc
Best model = argmin AICc
```

**Practical constraint**: p + q ≤ min(5, n/4) to avoid overfit.

**Heuristic for starting d**: 
- d=0 if ADF rejects unit root at α=0.05
- d=1 if ADF fails on y but succeeds on Δy
- d=2 otherwise (rare for financial data)

---

## 5. Ljung-Box Portmanteau Test

### Source
Ljung & Box (1978) *Biometrika*; Box & Pierce (1970).

### Purpose
Tests whether the residual ACF is white noise. For ARIMA residuals.

### Statistic
```
Q(m) = n(n+2) Σ_{k=1}^{m} r²_k / (n-k)
```
where rₖ = sample autocorrelation of residuals at lag k, n = number of residuals, m = number of lags tested.

Under H₀ (residuals are white noise): Q ~ χ²(m - p - q).

**Degrees of freedom**: m - p - q (subtract fitted model parameters).

### Accumulate+Gather
```
r_k = accumulate(t=k+1..n, (ε_t - ε̄)(ε_{t-k} - ε̄), sum) / accumulate(t=1..n, (ε_t - ε̄)², sum)
Q = n(n+2) accumulate(k=1..m, r²_k/(n-k), sum)
```

### Recommended Test Lags
- Minimum: m = max(p+q+4, 10)
- Typical: m = 20 for quarterly/monthly data
- Large n: m = ⌊log(n)⌋ ... use multiple m values

---

## 6. Linear Kalman Filter

### Source
Kalman (1960) *J. Basic Eng.*; Hamilton (1994) §13; Shumway & Stoffer (2000).

### State-Space Model
```
State:       x_t = F x_{t-1} + B u_t + w_t,     w_t ~ N(0, Q)
Observation: y_t = H x_t + v_t,                  v_t ~ N(0, R)
```
where:
- x_t ∈ ℝ^m — latent state (m-dimensional)
- y_t ∈ ℝ^p — observation (p-dimensional)
- F ∈ ℝ^{m×m} — state transition matrix
- H ∈ ℝ^{p×m} — observation matrix
- Q ∈ ℝ^{m×m} — process noise covariance (SPD)
- R ∈ ℝ^{p×p} — observation noise covariance (SPD)
- B u_t — optional control input (often 0)

### Kalman Filter (Forward Pass)

**Initialization**: x̂₀|₀ = μ₀, P₀|₀ = Σ₀

**Predict step** (time update):
```
x̂_{t|t-1} = F x̂_{t-1|t-1} + B u_t
P_{t|t-1} = F P_{t-1|t-1} F' + Q
```

**Update step** (measurement update):
```
innovation:     v_t = y_t - H x̂_{t|t-1}
innovation cov: S_t = H P_{t|t-1} H' + R
Kalman gain:    K_t = P_{t|t-1} H' S_t⁻¹
state update:   x̂_{t|t} = x̂_{t|t-1} + K_t v_t
cov update:     P_{t|t} = (I - K_t H) P_{t|t-1}
```

**Numerically stable covariance update** (Joseph form, avoids asymmetry due to roundoff):
```
P_{t|t} = (I - K_t H) P_{t|t-1} (I - K_t H)' + K_t R K_t'
```

### Log-likelihood (for parameter estimation)
```
ℓ = Σ_t [-½ p·ln(2π) - ½ ln|S_t| - ½ v_t' S_t⁻¹ v_t]
```
This is the **prediction error decomposition** of the likelihood.

### Accumulate+Gather Decomposition
```
For t=1..n (sequential, cannot parallelize):
  1. accumulate: predict x̂, P forward one step
  2. gather: K_t, v_t from S_t⁻¹ (solve linear system for gain)
  3. scatter: update x̂_{t|t}, P_{t|t}
  4. accumulate: log-likelihood sum
```
Pattern: **sequential scan with symmetric PSD matrix arithmetic** (Kingdom C, linear in n).

### Sufficient Statistics
The filter state (x̂_{t|t}, P_{t|t}) is the MSR of all data y₁,...,y_t. Once you have this, you have everything needed to:
- Continue filtering forward
- Start the smoother backward
- Evaluate the likelihood

### Implementation Notes
- Store the full sequence {x̂_{t|t-1}, P_{t|t-1}, v_t, S_t, K_t, x̂_{t|t}, P_{t|t}} for the smoother
- Use Cholesky of S_t to solve for K_t: `S_t · K_t' = (H P_{t|t-1})'` → back-solve
- Joseph form is recommended; the simple form accumulates floating-point errors over many steps

---

## 7. Kalman Smoother (RTS Smoother)

### Source
Rauch, Tung & Striebel (1965) *AIAA J.*; Shumway & Stoffer (2000) §6.3.

### Purpose
Computes the smoothed state estimates x̂_{t|n} = E[x_t | y₁,...,y_n] and their covariances P_{t|n}. The filter only uses data up to t; the smoother uses ALL data.

### Backward Pass (run after filter completes)

**Initialize**: x̂_{n|n} = x̂_{n|n} (from filter), P_{n|n} = P_{n|n} (from filter)

**For t = n-1, n-2, ..., 1**:
```
Smoother gain:       G_t = P_{t|t} F' [P_{t+1|t}]⁻¹
Smoothed state:      x̂_{t|n} = x̂_{t|t} + G_t (x̂_{t+1|n} - x̂_{t+1|t})
Smoothed covariance: P_{t|n} = P_{t|t} + G_t (P_{t+1|n} - P_{t+1|t}) G_t'
```

where P_{t+1|t} = F P_{t|t} F' + Q (stored from the filter's predict step).

### Numerical Note
Solving P_{t+1|t} · G_t' = F P_{t|t}' via Cholesky avoids explicit matrix inversion.
```
P_{t+1|t} · X = F P_{t|t}     [solve for X = G_t']
G_t = X'
```

### Accumulate+Gather Pattern
```
Backward scan (sequential, cannot parallelize):
  1. gather: G_t from Cholesky solve
  2. scatter: x̂_{t|n}, P_{t|n} backward
```

---

## 8. ARIMA as State-Space + Kalman Filter

Every ARIMA(p,d,q) has an exact state-space representation (the *innovation state-space form*). This connection is important for:
- Exact likelihood (not conditional LL) via the Kalman filter's prediction error decomposition
- Missing data handling (Kalman handles missing y_t by skipping the update step)
- Seasonal ARIMA (SARIMA) — same machinery with seasonal lags

The ARIMA(p,d,q) state-space form (Harvey 1989, §3.3):
```
State vector: x_t = [y_t, y_{t+1|t}, ..., y_{t+m-1|t}]   (m = max(p+d, q+1))
F: companion matrix built from φ and θ coefficients
H: [1, 0, 0, ..., 0]
Q: σ² κκ' where κ = [1, θ₁, ..., θ_{q}] padded to length m
R = 0 (no observation noise — all randomness through state)
```

**For tambear**: the conditional LL approach (Section 2 above) is simpler to implement first. The Kalman/state-space approach is for missing data and exact likelihood.

---

## 9. Implementation Priority

| Function | Priority | Depends on |
|---|---|---|
| `ma_fit` (MA(q) via L-BFGS) | HIGH | `optimization::lbfgs`, `acf` |
| `arma_fit` (ARMA via concentrated LL) | HIGH | `ar_fit`, `ma_fit` (for init) |
| `arima_fit` (ARIMA pipeline) | HIGH | `difference`, `arma_fit` |
| `arima_forecast` (with CI) | HIGH | `arima_fit` |
| `arima_aic_bic` | HIGH | `arima_fit` |
| `ljung_box` | MEDIUM | `acf` |
| `kalman_filter` | HIGH | `cholesky`, `cholesky_solve` |
| `kalman_smoother` (RTS) | MEDIUM | `kalman_filter` |
| `auto_arima` | MEDIUM | `arima_fit`, `adf_test` |

---

## 10. GARCH Diagnostic Tests (for `volatility.rs`)

The GARCH fit should be validated with:

### Ljung-Box on Squared Standardized Residuals
```
z_t = ε_t / σ_t      [standardized residuals]
Test H₀: z²_t are white noise (ARCH effects removed)
Q(m) on z²_t with df = m (no ARMA parameters in GARCH residuals)
```

### Sign Bias Test (Engle & Ng 1993)
Tests whether positive/negative shocks have differential impact on volatility:
```
z²_t = β₀ + β₁ S⁻_{t-1} + β₂ S⁻_{t-1} ε_{t-1} + β₃ S⁺_{t-1} ε_{t-1} + u_t
```
where S⁻_{t-1} = 1 if ε_{t-1} < 0 (indicator for negative shock).
F-test on β₁=β₂=β₃=0 → if rejected, consider EGARCH or GJR-GARCH.

### ARCH-LM Test (Engle 1982)
```
1. Fit OLS: ε²_t = c + α₁ε²_{t-1} + ... + α_p ε²_{t-p} + u_t
2. LM statistic = n · R² ~ χ²(p)
3. Reject H₀ (no ARCH) if LM > χ²(p, α)
```

These tests live in the diagnostics section of `volatility.rs`.

---

*All formulas paper-verifiable against Box & Jenkins (1976), Hamilton (1994), Kalman (1960), Rauch et al. (1965), Ljung & Box (1978).*
