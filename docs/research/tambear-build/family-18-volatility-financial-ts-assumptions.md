# Family 18: Volatility & Financial Time Series — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: Mixed — A (GARCH filter = affine prefix scan with constant linear coefficient), C (GARCH MLE via F05 optimizer), A (realized measures = accumulates)
**Note (2026-04-10)**: GARCH was previously labeled Kingdom B. Corrected — see affine scan analysis below.

---

## Core Insight: Two Worlds of Volatility

1. **Model-based** (GARCH family): parametric models estimated via MLE → Kingdom B (state scan) + C (optimization)
2. **Realized measures** (from high-frequency data): non-parametric, computed directly from returns/prices → Kingdom A (accumulates)

The fintek pipeline uses BOTH: realized measures as ground truth, GARCH as forecast.

---

## 1. GARCH(p,q) — Generalized Autoregressive Conditional Heteroskedasticity

### Model
Returns: r_t = μ + ε_t, where ε_t = σ_t · z_t, z_t ~ N(0,1) (or Student-t, or skewed-t)

Variance equation:
```
σ²_t = ω + Σ_{i=1}^{q} α_i · ε²_{t-i} + Σ_{j=1}^{p} β_j · σ²_{t-j}
```

GARCH(1,1) (the workhorse, covers >90% of use cases):
```
σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}
```

### As Affine Scan (Kingdom A — not B)
State = σ²_t. Input = ε²_t = r²_{t-1} (KNOWN DATA, not state). Update:
```
σ²_t = ω + α · ε²_{t-1} + β · σ²_{t-1}
     = (ω + α · input_t) + β · state_{t-1}
```
This is f_t(x) = β·x + b_t where b_t = ω + α·ε²_{t-1}.

KEY: β is CONSTANT across all steps. Only the offset b_t varies, and it depends on DATA
(the return at t-1), not the current state σ²_{t-1}. The linear coefficient β is fixed.

Affine maps with constant linear coefficient compose as a semigroup:
  (f_s ∘ f_t)(x) = β²·x + (β·b_t + b_s)
The composition parameters depend only on the data sequence b_0, b_1, ..., not on the state.
This makes GARCH a prefix product of data-determined maps — **Kingdom A**.

Compare to true Kingdom B: an affine scan where a_t (the linear coefficient) itself
varies per step as a function of the state. That would break the semigroup composition.

**Corrected**: GARCH filter = Kingdom A. Only the MLE outer optimization loop = Kingdom C.

### Constraints
- ω > 0, α ≥ 0, β ≥ 0
- α + β < 1 (stationarity — ensures unconditional variance σ² = ω/(1-α-β) is finite)
- IGARCH: α + β = 1 (integrated GARCH — infinite unconditional variance, unit root in variance)

### Log-Likelihood
```
log L = -½ Σ_t [log(2π) + log(σ²_t) + ε²_t/σ²_t]
```

**MLE via F05 optimizer** (L-BFGS or Adam). The GradientOracle wraps this likelihood.

### Starting Values
- ω = Var(ε)·(1-α₀-β₀), α₀ = 0.05, β₀ = 0.90 (typical starting point)
- σ²₁ = sample variance of first 50 returns (or unconditional variance)
- **CRITICAL**: σ²_t must be initialized. Common choice: backcast using exponential smoothing.

### Edge Cases
- ε²_t very large (outlier): σ²_{t+1} spikes → can cause numerical overflow. Clamp σ²_t to [ε_floor, σ²_max].
- α + β → 1: near-IGARCH, very slow mean reversion. Forecast converges slowly to unconditional variance.
- All returns zero: σ²_t = ω for all t (constant variance — degenerate)

---

## 2. EGARCH (Exponential GARCH — Nelson 1991)

### Model
```
log(σ²_t) = ω + Σ α_i · [|z_{t-i}| - E|z_{t-i}|] + Σ γ_i · z_{t-i} + Σ β_j · log(σ²_{t-j})
```
where z_t = ε_t/σ_t (standardized residuals).

### Key Properties
- **No positivity constraints needed** — log(σ²) can be any real number
- **Asymmetric**: γ captures leverage effect (bad news → more volatility)
- E|z_t| = √(2/π) for z ~ N(0,1)

### As Affine Scan
State = log(σ²_t). Input = function of z_t. Still Affine(1,1) for EGARCH(1,1).

---

## 3. GJR-GARCH (Glosten-Jagannathan-Runkle 1993)

### Model
```
σ²_t = ω + (α + γ · I_{ε_{t-1}<0}) · ε²_{t-1} + β · σ²_{t-1}
```
where I_{ε<0} = 1 if ε < 0 (indicator of negative shock).

### Key Properties
- **Simplest asymmetric GARCH**: one extra parameter γ
- γ > 0 means negative returns increase volatility more (leverage effect)
- News Impact Curve: parabolic but steeper for negative ε

### Constraints
- ω > 0, α ≥ 0, β ≥ 0, α + γ ≥ 0
- α + β + γ/2 < 1 (stationarity)

---

## 4. TGARCH (Threshold GARCH — Zakoian 1994)

### Model (on σ_t, not σ²_t)
```
σ_t = ω + α⁺ · max(ε_{t-1}, 0) + α⁻ · max(-ε_{t-1}, 0) + β · σ_{t-1}
```

### Difference from GJR
TGARCH models conditional standard deviation (not variance). The threshold is on ε (not ε²).

---

## 5. FIGARCH (Fractionally Integrated GARCH — Baillie, Bollerslev, Mikkelsen 1996)

### Model
```
σ²_t = ω + [1 - β(L) - φ(L)(1-L)^d] · ε²_t + β(L) · σ²_t
```
where d ∈ (0, 1) is the fractional differencing parameter.

### Key Property
Long memory in volatility: autocorrelation of ε² decays hyperbolically (not exponentially as in GARCH). Empirically observed in financial data.

### CRITICAL: Requires truncated binomial series expansion of (1-L)^d:
```
(1-L)^d = Σ_{k=0}^{∞} C(d,k)(-L)^k    where C(d,k) = Γ(k-d)/(Γ(-d)Γ(k+1))
```
Truncate at ~1000 lags.

### Kingdom: B (scan) but with long filter (1000+ taps). Computationally heavier than GARCH.

---

## 6. Realized Volatility (RV)

### Definition
```
RV_t = Σ_{i=1}^{M} r²_{t,i}
```
where r_{t,i} are M intraday returns on day t.

### This is `accumulate(ByKey(day), r², Add)` — Kingdom A.

### Sampling Frequency Trade-off
- High frequency (1s): more data but microstructure noise dominates
- Low frequency (5min): less noise but fewer observations
- **Optimal**: 5-minute returns are standard; 1-minute with noise correction is better

### Noise Correction: Two-Scale RV (Zhang, Mykland, Aït-Sahalia 2005)
```
TSRV = RV^{slow} - (n_bar/n_fast) · RV^{fast}
```
Subsamples at two frequencies and debias.

### Kernel Realized Volatility (Barndorff-Nielsen et al. 2008)
```
KRV = Σ_{h=-H}^{H} k(h/H) · γ̂_h
```
where γ̂_h = Σ r_{t,i} · r_{t,i-h} is the autocovariance at lag h, k is a kernel (Parzen, Bartlett).

---

## 7. Bipower Variation (BPV)

### Definition
```
BPV_t = (π/2) · Σ_{i=2}^{M} |r_{t,i}| · |r_{t,i-1}|
```

### Key Property
BPV converges to integrated variance even in the presence of jumps. RV converges to integrated variance + jump variation. Therefore:

### Jump Detection (Barndorff-Nielsen & Shephard)
```
J_t = RV_t - BPV_t
```
Under H₀ (no jumps): J_t / √(Var) → N(0,1).

Relative jump statistic:
```
z_J = (RV - BPV) / √((π²/4 + π - 5) · TPQ / M)
```
where TPQ = (M/(M-2)) · Σ |r_i|^{4/3} · |r_{i-1}|^{4/3} · |r_{i-2}|^{4/3} (tri-power quarticity).

### Kingdom: A — all are accumulates over intraday returns.

---

## 8. Stochastic Volatility (SV)

### Model
```
r_t = exp(h_t/2) · ε_t,    ε_t ~ N(0,1)
h_t = μ + φ(h_{t-1} - μ) + σ_η · η_t,    η_t ~ N(0,1)
```
where h_t = log-volatility follows an AR(1) process.

### Estimation
- **Kalman filter** (after linearization): log(r²_t) = h_t + log(ε²_t). But log(ε²_t) ~ log-χ²(1), not Gaussian.
- **Particle filter**: Sequential Monte Carlo. Kingdom C (iterative).
- **MCMC** (standard): Sample h_{1:T} via block Gibbs or Hamiltonian MC. Kingdom C.

### Connection to GARCH
SV and GARCH are not nested. SV has two independent shocks (ε, η); GARCH has one (ε drives both return and volatility). SV is more flexible but harder to estimate.

---

## 9. VPIN (Volume-Synchronized Probability of Informed Trading)

### Definition
Partition trading volume into V-bars (equal volume buckets). For each bucket:
```
V_S = volume classified as sell
V_B = volume classified as buy
OI = |V_S - V_B| / V    (order imbalance)
```

```
VPIN = (1/n) Σ_{i=1}^{n} OI_i    (moving average of order imbalance over n buckets)
```

### Trade Classification
**Bulk Volume Classification** (BVC):
```
V_B = V · Φ(Z),    V_S = V · (1 - Φ(Z))
```
where Z = ΔP / σ_ΔP (normalized price change within the bar).

### Kingdom: A (accumulate over volume buckets)

---

## 10. Roll Spread (Roll 1984)

### Estimator
```
Spread = 2√(-Cov(Δp_t, Δp_{t-1}))
```
If Cov > 0 (violates model assumption): set spread = 0.

### From Sufficient Stats
Needs: lag-1 autocovariance of price changes. This is `accumulate(All, ΔpΔp_lag1, Add)` — one pass.

### CRITICAL: The Roll model assumes: (1) efficient market, (2) constant spread, (3) no inventory effects. Violations are common. Use as rough estimate only.

---

## 11. Kyle's Lambda (Kyle 1985)

### Estimator
Regress price change on signed order flow:
```
Δp_t = λ · OFI_t + ε_t
```
where OFI = order flow imbalance (buy volume - sell volume).

λ measures price impact: higher λ = less liquid market.

### Implementation: F10 regression (OLS). One coefficient.

---

## 12. DFA / MF-DFA / Hurst R/S

### Already covered in F26 (Complexity & Chaos) assumption document.

These are shared with F26 — the implementations live in F26, F18 consumes them for financial time series specifically.

### Key for finance:
- H > 0.5: trending (momentum)
- H < 0.5: mean-reverting
- H = 0.5: random walk (efficient market)

---

## Error Distributions for GARCH

### Student-t Innovations
```
z_t ~ t(ν),    f(z) = Γ((ν+1)/2) / (√(πν)Γ(ν/2)) · (1+z²/ν)^{-(ν+1)/2}
```
Log-likelihood adds: log Γ((ν+1)/2) - log Γ(ν/2) - ½log(π(ν-2)) - (ν+1)/2 · log(1 + z²/(ν-2))

### Skewed-t (Hansen 1994)
Adds asymmetry parameter λ ∈ (-1, 1):
```
f(z|ν,λ) = {bc(1 + 1/(ν-2)·((bz+a)/(1-λ))²)^{-(ν+1)/2}  if z < -a/b
            {bc(1 + 1/(ν-2)·((bz+a)/(1+λ))²)^{-(ν+1)/2}  if z ≥ -a/b
```
where a, b, c are normalizing constants depending on ν, λ.

### GED (Generalized Error Distribution)
```
f(z|ν) = ν · exp(-½|z/λ|^ν) / (λ · 2^{1+1/ν} · Γ(1/ν))
```
ν = 2: Gaussian. ν < 2: heavier tails. ν > 2: lighter tails.

---

## Sharing Surface

### Reuse from Other Families
- **F05 (Optimization)**: GradientOracle for GARCH MLE (L-BFGS primary, Adam for difficult likelihoods)
- **F06 (Descriptive)**: Sample variance for RV, means for VPIN buckets
- **F07 (Hypothesis Testing)**: Jump detection z-tests, Ljung-Box on standardized residuals
- **F10 (Regression)**: Kyle's lambda (OLS), DFA detrending
- **F17 (Time Series)**: ARMA for mean equation, AR(1) in SV model
- **F26 (Complexity)**: DFA, MF-DFA, Hurst — shared implementations

### Consumers of F18
- **Fintek pipeline**: ALL microstructure leaves (RV, BPV, jumps, VPIN, Roll, Kyle)
- **F05 optimization**: GARCH likelihood as GradientOracle consumer
- **Risk management**: VaR, CVaR from GARCH forecasts

### Structural Rhymes
- **GARCH(1,1) variance = EMA of squared returns** (with intercept): structural rhyme with F17 exponential smoothing
- **GARCH Affine scan = Adam Affine scan** (F05): same Kingdom B machinery
- **RV = sample variance of intraday returns**: same as F06 but at intraday resolution
- **BPV = product accumulate**: structural rhyme with Kendall's tau (F08, product of adjacent ranks)
- **VPIN = windowed mean of order imbalance**: same as F06 windowed statistics

---

## Implementation Priority

**Phase 1** — Core GARCH + realized measures (~200 lines):
1. GARCH(1,1) variance recursion (Affine scan)
2. GARCH log-likelihood (Gaussian + Student-t innovations)
3. GARCH MLE via F05 GradientOracle
4. Realized Volatility (accumulate over intraday returns)
5. Bipower Variation + jump detection (BNS test)

**Phase 2** — GARCH extensions (~150 lines):
6. EGARCH (log-variance, no positivity constraints)
7. GJR-GARCH (asymmetric / leverage)
8. GARCH-t, GARCH-skewed-t (error distributions)
9. GARCH forecasting (multi-step ahead)

**Phase 3** — Microstructure leaves (~100 lines):
10. VPIN (volume buckets + BVC classification)
11. Roll spread estimator
12. Kyle's lambda (F10 OLS wrapper)
13. Two-Scale RV / Kernel RV (noise correction)

**Phase 4** — Advanced (~150 lines):
14. FIGARCH (long memory, truncated binomial series)
15. Stochastic Volatility (Kalman filter approach)
16. MF-DFA (via F26)
17. GARCH-MIDAS (mixed frequency)

---

## Composability Contract

```toml
[family_18]
name = "Volatility & Financial Time Series"
kingdom = "A (realized measures) + B (GARCH scan) + C (MLE optimization)"

[family_18.shared_primitives]
garch_scan = "Affine(1,1) scan for σ²_t recursion"
garch_likelihood = "GradientOracle wrapping GARCH log-likelihood"
realized_vol = "accumulate(ByKey(day), r², Add)"
bipower_var = "accumulate(ByKey(day), |r_i|·|r_{i-1}|, Add)"

[family_18.reuses]
f05_optimizer = "L-BFGS/Adam for GARCH MLE"
f06_descriptive = "Sample variance, windowed means"
f07_hypothesis = "Jump detection tests, Ljung-Box"
f10_regression = "Kyle's lambda (OLS), DFA detrending"
f17_time_series = "ARMA mean equation, AR(1) in SV"
f26_complexity = "DFA, MF-DFA, Hurst exponent"

[family_18.provides]
garch_forecast = "Conditional variance forecasts (1-step and multi-step)"
realized_measures = "RV, BPV, TSRV, KRV"
jump_detection = "BNS test for intraday jumps"
microstructure = "VPIN, Roll spread, Kyle's lambda"

[family_18.consumers]
fintek = "ALL microstructure and volatility leaves"
risk = "VaR, CVaR from GARCH forecasts"

[family_18.session_intermediates]
garch_state = "GarchState(model_id) — current σ²_t, parameters"
realized_vol = "RV(data_id, day) — daily realized volatility"
jump_flags = "JumpFlags(data_id, day) — boolean jump indicators"
```
