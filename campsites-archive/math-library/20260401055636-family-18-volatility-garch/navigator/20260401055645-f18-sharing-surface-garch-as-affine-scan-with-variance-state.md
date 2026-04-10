# F18 Sharing Surface: GARCH as Affine Scan with Variance State

Created: 2026-04-01T05:56:45-05:00
By: navigator

Prerequisite: F17 complete (Affine scan infrastructure).

---

## Core Insight: GARCH Is F17 With a Different State

F17 implemented the Affine scan for AR, ARIMA, EWM, and Kalman. F18 (volatility models) uses the EXACT SAME Affine scan infrastructure. The state is now σ²_t (conditional variance) rather than the signal value.

For all GARCH-family models: `σ²_t = A_t · σ²_{t-1} + b_t`

The only difference from F17: `b_t` depends on past residuals ε²_t — it's data-driven rather than a constant input vector.

---

## Model Family and Affine Parameterization

### EWM Variance (simplest)

```
σ²_t = λ · σ²_{t-1} + (1-λ) · ε²_t
A_t = λ  (constant decay factor, λ = 1 - 2/(span+1))
b_t = (1-λ) · ε²_t  (data-dependent)
```

State: scalar σ²_t. This is the simplest case — scalar Affine with data-dependent b.

**Tambear path**: same as EWM in F17, with the phi expression applied to ε²_t.

### GARCH(1,1)

```
σ²_t = ω + α · ε²_t + β · σ²_{t-1}
A_t = β  (constant)
b_t = ω + α · ε²_t  (data-dependent, varies per step)
```

State: scalar σ²_t. A is constant, b varies with data.

**Tambear path**: AffineState with scalar A = β, b_t = ω + α·ε²_t computed per step.

### GARCH(p,q)

```
σ²_t = ω + Σ_{i=1}^q α_i · ε²_{t-i} + Σ_{j=1}^p β_j · σ²_{t-j}
```

State: vector [σ²_t, σ²_{t-1}, ..., σ²_{t-p+1}, ε²_t, ..., ε²_{t-q+1}]' — companion form.
A is the (p+q) × (p+q) companion matrix (constant).
b_t = [ω + α_1 · ε²_t, 0, ..., 0]'

**Tambear path**: same vector Affine scan as ARIMA, companion matrix form.

### EGARCH(1,1) (exponential GARCH — handles leverage effect)

```
log(σ²_t) = ω + α · (|z_t| - E[|z_t|]) + γ · z_t + β · log(σ²_{t-1})
where z_t = ε_t / σ_t
```

State: log(σ²_t) — still a scalar.
A_t = β (constant), b_t = ω + α·(|z_t| - √(2/π)) + γ·z_t

**Tambear path**: Affine scan with log(σ²) as state. The nonlinearity is only in b_t computation. Map from linear state to variance: σ²_t = exp(state_t).

### GJR-GARCH (asymmetric, for leverage/down-market)

```
σ²_t = ω + (α + γ · 1{ε_{t-1}<0}) · ε²_{t-1} + β · σ²_{t-1}
```

The indicator function 1{ε_{t-1}<0} makes b_t piecewise. Still Affine scan — just with a conditional in b_t computation. `ScatterJit` can express this as a phi with conditional: `"(v < 0) ? alpha_plus_gamma * v * v : alpha * v * v"`.

---

## AffineState MSR Type for Volatility

```rust
/// Affine scan result for volatility models.
/// Stores the conditional variance sequence and fitted parameters.
pub struct VolatilityState {
    pub model: VolatilityModel,  // which model was fit
    pub params: Vec<f64>,        // fitted ω, α, β, ... via MLE
    /// Conditional variances σ²_t for all t. Shape: (n_obs,).
    pub conditional_variance: Arc<Vec<f64>>,
    /// Standardized residuals z_t = ε_t / σ_t. Shape: (n_obs,).
    pub std_residuals: Arc<Vec<f64>>,
    /// Log-likelihood at fitted parameters.
    pub log_likelihood: f64,
    /// AIC, BIC for model comparison.
    pub aic: f64,
    pub bic: f64,
}

pub enum VolatilityModel {
    EwmVariance { lambda: f64 },
    Garch { p: u32, q: u32 },
    Egarch { p: u32, q: u32 },
    GjrGarch { p: u32, q: u32 },
}
```

---

## Parameter Estimation: MLE

**GARCH estimation requires MLE** (not OLS — the likelihood is non-Gaussian because variance changes over time).

The GARCH(1,1) log-likelihood:
```
L(ω, α, β | ε) = -0.5 · Σ_t [log(2π) + log(σ²_t) + ε²_t / σ²_t]
```

Where σ²_t depends on parameters via the Affine scan.

**This is Kingdom C** (iterative over gradient): each gradient evaluation is an Affine scan pass + O(n) likelihood sum. The outer optimizer is gradient descent or L-BFGS.

**Tambear decomposition**:
1. Forward scan: compute σ²_t sequence from parameters (Affine scan from F17)
2. Likelihood: `-0.5 · scatter_phi("log(v) + x_sq / v")` where v = σ²_t, x_sq = ε²_t
3. Gradient: derivative of likelihood w.r.t. (ω, α, β) via chain rule through scan
4. Optimizer: L-BFGS or Adam (F05 optimization)

**For Phase 1**: use EWM variance (no MLE needed — λ is fixed or estimated from exponential smoothing). This gives a working implementation with zero optimization infrastructure.

**For Phase 2**: GARCH(1,1) with MLE — requires F05 (optimization) or a specialized BFGS implementation.

---

## What F18 Produces for TamSession

```rust
IntermediateTag::VolatilityState {
    data_id: DataId,      // the return series
    model: VolatilityModel,
    params_hash: DataId,  // hash of fitted parameters (for cache invalidation)
}
```

**What F18 provides to downstream families**:
- Conditional variance series σ²_t → F25 (conditional entropy, information)
- Standardized residuals z_t → F06/F07 (normality testing of standardized residuals)
- Log-likelihood → F34 (Bayesian model comparison)
- AIC/BIC → Model selection

---

## Sharing with F17

F17 (AR, ARIMA, EWM, Kalman) already implements the core Affine scan. F18 shares:
- The AffineState data structure
- The forward scan computation
- The companion matrix parameterization for GARCH(p,q)

**F18 adds**:
- ε²_t computation (conditional on F10/F09 for residuals from fitted mean model)
- MLE objective (log-likelihood as scatter_phi sum)
- Gradient of log-likelihood through scan (reverse Affine scan)

The reverse Affine scan (gradient through scan) is the same gradient duality principle:
`∂L/∂params` through `σ²_t = A·σ²_{t-1} + b_t` is the reversed scan with transposed A.
This was proved in the March 31 session and is documented in the gradient duality docs.

---

## Build Order

**Phase 1 (EWM variance, no MLE)**:
1. `fn ewm_variance(returns: &[f64], lambda: f64) -> Vec<f64>` — scalar Affine scan (~20 lines)
2. `VolatilityState` struct in `intermediates.rs`
3. Tests: match `pd.DataFrame.ewm(span=...).var()` in pandas

**Phase 2 (GARCH(1,1) with MLE)**:
1. Forward scan: compute σ²_t given (ω, α, β) — uses F17's AffineState infrastructure
2. Log-likelihood: `scatter_phi("-0.5 * (log(v) + x_sq / v)", all)` over t
3. Gradient: reverse-mode differentiation through the scan (documented in gradient duality)
4. Optimizer: L-BFGS or coordinate-wise Brent (simple for 3 parameters)
5. Tests: match `arch.arch_model(returns, vol='Garch', p=1, q=1).fit()` in Python arch package

**Phase 3 (EGARCH, GJR-GARCH, GARCH(p,q))**:
- Same MLE structure, different state dynamics
- These are incremental once Phase 2 is working

---

## Gold Standard Libraries

- Python: `arch` library — `arch.arch_model()` for all GARCH variants
- R: `rugarch` package — `ugarchspec()` + `ugarchfit()`
- R: `fGarch` package — simpler, GARCH(1,1)
- Python: `statsmodels.tsa.statespace.SARIMAX` for ARMA-GARCH (combined mean + variance models)

**Match target**: arch Python library for GARCH parameters, AIC/BIC, and conditional variance sequence. Tolerance: parameters within 1e-4 (MLE estimates vary slightly by optimization path).

---

## The Lab Notebook Claim

> GARCH volatility modeling is F17 (Time Series, Kingdom B) with the state variable changed from signal level to conditional variance. The same Affine scan operator, the same companion matrix parameterization, the same infrastructure. F18 adds only the MLE objective and its gradient — which is itself an Affine scan in reverse (gradient duality principle). One forward scan + one reverse scan = full GARCH estimation.
