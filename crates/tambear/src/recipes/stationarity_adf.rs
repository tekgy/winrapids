//! Augmented Dickey–Fuller test for unit-root / stationarity.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over a
//! regression-design gather + Kulisch-backed OLS via Cholesky + a
//! MacKinnon critical-value lookup. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! The augmented Dickey–Fuller (Said & Dickey 1984) test fits the
//! regression with intercept:
//!
//! ```text
//! Δy[t]  =  α  +  γ · y[t−1]  +  Σ_{j=1..p} δ_j · Δy[t−j]  +  ε[t]
//! ```
//!
//! and reports the `t`-statistic on `γ`:
//!
//! ```text
//! ADF_t  =  γ̂  /  SE(γ̂)
//! ```
//!
//! Under the null of a unit root (`γ = 0`, series non-stationary), the
//! statistic does **not** follow a standard `t` distribution — instead
//! it follows the Dickey–Fuller distribution. We compare `ADF_t`
//! against MacKinnon (2010) response-surface critical values at 1%,
//! 5%, and 10%; reject the unit-root null (conclude stationarity) when
//! `ADF_t < critical_value_α`.
//!
//! More-negative statistics are stronger evidence of stationarity.
//!
//! # SIP context
//!
//! Per `R:\ternyx-sip\docs\signal-compute-spec-for-tambear.md` the per-
//! hour `stationarity_pvalue` header field uses the ADF statistic on
//! the bucket-level price (or return) series. The closest-available
//! approximation to a p-value is a linear interpolation between the
//! 10%/5%/1% critical values; this recipe returns the statistic plus
//! all three critical values so the caller can make that interpolation
//! or report the raw statistic directly.
//!
//! # Reference
//!
//! Dickey, D. A., & Fuller, W. A. (1979). Distribution of the
//! estimators for autoregressive time series with a unit root.
//! *Journal of the American Statistical Association* 74(366): 427–431.
//!
//! Said, S. E., & Dickey, D. A. (1984). Testing for unit roots in
//! autoregressive-moving average models of unknown order. *Biometrika*
//! 71(3): 599–607.
//!
//! MacKinnon, J. G. (2010). Critical values for cointegration tests.
//! *Queen's University Economics Department Working Paper* 1227.
//!
//! # Composition
//!
//! - **Difference gather** — `Δy[t] = y[t] − y[t−1]`. Lowers to
//!   `gather(y, Addressing::Offset{-1})` + pointwise subtraction.
//! - **Regression-design assembly** — stacks the `1`, `y[t−1]`,
//!   `Δy[t−1..t−p]` columns into an `(nobs × p)` matrix. Lowers to a
//!   set of gather operations indexed by the lag schedule.
//! - **Normal equations** — `X'X = accumulate(x·xᵀ, All, Op::Add)` and
//!   `X'y = accumulate(x·y, All, Op::Add)`. Both solved via Cholesky
//!   (positive-definite by construction).
//! - **t-statistic** — residual SS → MSE; diagonal of `(X'X)⁻¹` via a
//!   second Cholesky solve → SE(γ̂); `γ̂ / SE(γ̂)`.
//! - **MacKinnon critical-value lookup** — closed-form response surface
//!   `cv(n) = β_∞ + β₁/n + β₂/n²` per MacKinnon (2010) Table 1 case
//!   "c" (constant, no trend, 1 regressor).
//!
//! # NaN/Inf policy
//!
//! Follows the standard accumulate-layer contract: `!is_finite(x) →
//! skip` per element. The recipe pre-filters the input to its finite
//! subset before running the regression. Because ADF depends on
//! time ordering, the filter is a copy — the regression runs on the
//! compact finite-only sequence and the caller's original indexing is
//! lost inside the recipe.
//!
//! - Insufficient finite data (`n_finite ≤ n_lags + 2`) → statistic is
//!   NaN; critical values are still returned (they depend only on the
//!   sample size).
//! - Singular normal equations (e.g., perfectly collinear lagged
//!   differences) → statistic is NaN.
//! - All-NaN / all-Inf input → same as empty; statistic is NaN.
//!
//! # Default parameters
//!
//! - `n_lags` — caller-supplied. A common default is
//!   `floor(12·(n/100)^0.25)` (Schwert 1989); SIP uses `n_lags = 0`
//!   for the simple DF test within a bucket and small positive lags
//!   for the hour-level test. This recipe takes `n_lags` explicitly to
//!   avoid a hidden heuristic.

/// ADF test result: statistic + MacKinnon critical values + p-value.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StationarityAdfResult {
    /// ADF `t`-statistic on the `y[t−1]` coefficient.
    pub statistic: f64,
    /// Piecewise-linear-interpolated p-value on the Dickey–Fuller
    /// distribution. See `adf_p_value_from_statistic` for the
    /// construction; this is an approximation, not a true DF p-value.
    /// Small p means reject the unit-root null (conclude stationary).
    pub p_value: f64,
    /// Number of augmenting lags used.
    pub n_lags: usize,
    /// Effective sample size after filtering non-finite inputs and
    /// accounting for lag loss. 0 when the statistic cannot be
    /// computed.
    pub n_observations: u64,
    /// MacKinnon critical value at 1% significance.
    pub critical_1pct: f64,
    /// MacKinnon critical value at 5% significance.
    pub critical_5pct: f64,
    /// MacKinnon critical value at 10% significance.
    pub critical_10pct: f64,
}

/// Augmented Dickey–Fuller test on a price (or level) series.
///
/// Fits `Δy[t] = α + γ·y[t−1] + Σ δ_j·Δy[t−j] + ε[t]` and returns the
/// `t`-statistic on `γ` plus MacKinnon critical values at 1/5/10%.
///
/// Interpret: reject the unit-root null (conclude stationary) when
/// `statistic < critical_value_α`. More-negative is stronger evidence
/// of stationarity.
pub fn stationarity_adf(data: &[f64], n_lags: usize) -> StationarityAdfResult {
    // Pre-filter to the finite subset per the accumulate NaN contract.
    // ADF depends on time ordering, so we compact the finite values
    // into a contiguous sequence and run the regression on that.
    let filtered: Vec<f64> = data.iter().copied().filter(|v| v.is_finite()).collect();
    let n = filtered.len();
    if n <= n_lags + 2 {
        let (c1, c5, c10) = mackinnon_adf_critical_values(n.max(1));
        return StationarityAdfResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            n_lags,
            n_observations: 0,
            critical_1pct: c1,
            critical_5pct: c5,
            critical_10pct: c10,
        };
    }
    let data: &[f64] = &filtered;

    // First-differences: Δy[t] = y[t] − y[t−1].
    let dy: Vec<f64> = data.windows(2).map(|w| w[1] - w[0]).collect();
    let m = dy.len();

    // Regression design: Δy[t] = α + γ·y[t−1] + Σ δ_j·Δy[t−j] + ε[t]
    //   columns:  0 = intercept, 1 = y[t−1], 2..2+p = Δy[t−1..t−p]
    // The row index `t` runs n_lags..m so that all Δy[t−j] are in range.
    let p = 2 + n_lags; // # of regressors
    let nobs = m - n_lags;
    let mut x = vec![0.0; nobs * p];
    let mut y_reg = vec![0.0; nobs];

    for t in n_lags..m {
        let row = t - n_lags;
        y_reg[row] = dy[t];
        x[row * p] = 1.0; // intercept
        x[row * p + 1] = data[t]; // y[t−1]  (data[t] is y at the start of Δy[t])
        for j in 0..n_lags {
            x[row * p + 2 + j] = dy[t - 1 - j];
        }
    }

    if nobs <= p {
        let (c1, c5, c10) = mackinnon_adf_critical_values(nobs.max(1));
        return StationarityAdfResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            n_lags,
            n_observations: nobs as u64,
            critical_1pct: c1,
            critical_5pct: c5,
            critical_10pct: c10,
        };
    }

    // Normal equations X'X β = X'y, via explicit accumulation.
    let mut xtx = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for i in 0..nobs {
        for j in 0..p {
            xty[j] += x[i * p + j] * y_reg[i];
            for k in 0..p {
                xtx[j * p + k] += x[i * p + j] * x[i * p + k];
            }
        }
    }

    let (c1, c5, c10) = mackinnon_adf_critical_values(nobs);
    let a = crate::linear_algebra::Mat::from_vec(p, p, xtx);
    let l = match crate::linear_algebra::cholesky(&a) {
        Some(l) => l,
        None => {
            return StationarityAdfResult {
                statistic: f64::NAN,
                p_value: f64::NAN,
                n_lags,
                n_observations: nobs as u64,
                critical_1pct: c1,
                critical_5pct: c5,
                critical_10pct: c10,
            };
        }
    };
    let beta = crate::linear_algebra::cholesky_solve(&l, &xty);

    // Residual sum of squares → MSE.
    let mut ss_resid = 0.0;
    for i in 0..nobs {
        let fitted: f64 = (0..p).map(|j| x[i * p + j] * beta[j]).sum();
        let e = y_reg[i] - fitted;
        ss_resid += e * e;
    }
    let df_resid = (nobs - p) as f64;
    if df_resid <= 0.0 {
        return StationarityAdfResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            n_lags,
            n_observations: nobs as u64,
            critical_1pct: c1,
            critical_5pct: c5,
            critical_10pct: c10,
        };
    }
    let mse = ss_resid / df_resid;

    // Diagonal element of (X'X)⁻¹ at the γ column (index 1) via a
    // Cholesky solve against the unit vector e_1.
    let mut ej = vec![0.0; p];
    ej[1] = 1.0;
    let inv_col = crate::linear_algebra::cholesky_solve(&l, &ej);
    let var_gamma = mse * inv_col[1];
    if !var_gamma.is_finite() || var_gamma <= 0.0 {
        return StationarityAdfResult {
            statistic: f64::NAN,
            p_value: f64::NAN,
            n_lags,
            n_observations: nobs as u64,
            critical_1pct: c1,
            critical_5pct: c5,
            critical_10pct: c10,
        };
    }
    let se_gamma = var_gamma.sqrt();
    let statistic = beta[1] / se_gamma;
    let p_value = adf_p_value_from_statistic(statistic, c1, c5, c10);

    StationarityAdfResult {
        statistic,
        p_value,
        n_lags,
        n_observations: nobs as u64,
        critical_1pct: c1,
        critical_5pct: c5,
        critical_10pct: c10,
    }
}

/// Piecewise-linear approximation of the Dickey–Fuller p-value from
/// the ADF statistic and the three MacKinnon critical values.
///
/// Anchor points:
///
/// ```text
///   statistic ≤ critical_1pct    →  p ≤ 0.01   (extrapolated linearly toward 0)
///   critical_1pct < s ≤ critical_5pct   →  linear between 0.01 and 0.05
///   critical_5pct < s ≤ critical_10pct  →  linear between 0.05 and 0.10
///   statistic > critical_10pct   →  linear between 0.10 and 1.0 as s → 0
///                                   (clamped at 1.0 for s ≥ 0)
/// ```
///
/// This is the conventional "which critical value did we cross"
/// reporting dressed up as a continuous p, accurate enough for the
/// reject/fail-to-reject decision and monotone in the statistic.
/// Tambear does not ship a true DF quantile function (that requires
/// tabulated Monte Carlo or polynomial response surfaces); callers
/// needing higher-precision p-values should consult the critical
/// values directly.
#[inline]
pub fn adf_p_value_from_statistic(
    statistic: f64,
    critical_1pct: f64,
    critical_5pct: f64,
    critical_10pct: f64,
) -> f64 {
    if !statistic.is_finite() {
        return f64::NAN;
    }
    // The ADF statistic is typically negative; critical_1pct is the
    // most-negative (strictest) cutoff and critical_10pct the least.
    if statistic <= critical_1pct {
        // Extrapolate the 0.01 → 0 slope beyond the 1% cutoff.
        // slope_1_to_5 = (0.05 - 0.01) / (critical_5pct - critical_1pct)
        let denom = critical_5pct - critical_1pct;
        if denom.abs() < 1e-15 {
            return 0.0;
        }
        let slope = 0.04 / denom;
        let p = 0.01 + slope * (statistic - critical_1pct);
        p.clamp(0.0, 0.01)
    } else if statistic <= critical_5pct {
        let denom = critical_5pct - critical_1pct;
        if denom.abs() < 1e-15 {
            return 0.05;
        }
        let frac = (statistic - critical_1pct) / denom;
        (0.01 + frac * (0.05 - 0.01)).clamp(0.01, 0.05)
    } else if statistic <= critical_10pct {
        let denom = critical_10pct - critical_5pct;
        if denom.abs() < 1e-15 {
            return 0.10;
        }
        let frac = (statistic - critical_5pct) / denom;
        (0.05 + frac * (0.10 - 0.05)).clamp(0.05, 0.10)
    } else {
        // statistic > critical_10pct — cannot reject at any standard
        // level. Linearly ramp p from 0.10 at critical_10pct to 1.0
        // as statistic → 0 (and clamp at 1.0 for positive statistics).
        let denom = -critical_10pct; // critical_10pct is negative
        if denom.abs() < 1e-15 {
            return 1.0;
        }
        let frac = (statistic - critical_10pct) / denom;
        (0.10 + frac * (1.0 - 0.10)).clamp(0.10, 1.0)
    }
}

/// MacKinnon (2010) finite-sample critical values for ADF, model "c"
/// (constant, no trend, 1 unit-root regressor).
///
/// Response surface: `cv(n) = β_∞ + β₁/n + β₂/n²` with coefficients
/// from MacKinnon (2010) Table 1.
///
/// Exposed for downstream tests that need the same asymptotic-
/// correction curve (Phillips–Perron, DF-GLS, KPSS comparison).
#[inline]
pub fn mackinnon_adf_critical_values(n: usize) -> (f64, f64, f64) {
    let nf = n as f64;
    let inv_n = 1.0 / nf;
    let inv_n2 = inv_n * inv_n;
    let c1 = -3.4336 + (-5.999) * inv_n + (-29.25) * inv_n2;
    let c5 = -2.8621 + (-2.738) * inv_n + (-8.36) * inv_n2;
    let c10 = -2.5671 + (-1.438) * inv_n + (-4.48) * inv_n2;
    (c1, c5, c10)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_series_gives_nan_statistic() {
        let data = vec![1.0, 2.0, 3.0];
        let r = stationarity_adf(&data, 2);
        assert!(r.statistic.is_nan());
        assert!(r.p_value.is_nan());
        assert_eq!(r.n_observations, 0);
        // Critical values are still emitted.
        assert!(r.critical_1pct < r.critical_5pct);
        assert!(r.critical_5pct < r.critical_10pct);
    }

    #[test]
    fn p_value_interpolation_hits_anchor_points() {
        // At each critical value, the p-value should equal the nominal
        // significance level.
        let (c1, c5, c10) = mackinnon_adf_critical_values(1000);
        assert!((adf_p_value_from_statistic(c1, c1, c5, c10) - 0.01).abs() < 1e-12);
        assert!((adf_p_value_from_statistic(c5, c1, c5, c10) - 0.05).abs() < 1e-12);
        assert!((adf_p_value_from_statistic(c10, c1, c5, c10) - 0.10).abs() < 1e-12);
    }

    #[test]
    fn p_value_monotone_in_statistic() {
        // More-negative statistic → smaller p.
        let (c1, c5, c10) = mackinnon_adf_critical_values(1000);
        let p_strong = adf_p_value_from_statistic(-10.0, c1, c5, c10);
        let p_mid = adf_p_value_from_statistic(c5, c1, c5, c10);
        let p_weak = adf_p_value_from_statistic(-1.0, c1, c5, c10);
        let p_pos = adf_p_value_from_statistic(1.0, c1, c5, c10);
        assert!(p_strong < p_mid);
        assert!(p_mid < p_weak);
        assert!(p_weak <= p_pos);
        assert!((0.0..=1.0).contains(&p_strong));
        assert!((0.0..=1.0).contains(&p_pos));
    }

    #[test]
    fn p_value_nan_for_nan_statistic() {
        let (c1, c5, c10) = mackinnon_adf_critical_values(1000);
        assert!(adf_p_value_from_statistic(f64::NAN, c1, c5, c10).is_nan());
    }

    #[test]
    fn critical_values_ordered_and_sensible() {
        let (c1, c5, c10) = mackinnon_adf_critical_values(1000);
        assert!(c1 < c5 && c5 < c10, "{c1} {c5} {c10}");
        // Asymptotic values around -3.43 / -2.86 / -2.57.
        assert!((c1 - (-3.4336)).abs() < 0.1);
        assert!((c5 - (-2.8621)).abs() < 0.1);
        assert!((c10 - (-2.5671)).abs() < 0.1);
    }

    #[test]
    fn pure_unit_root_statistic_not_very_negative() {
        // Cumulative sum of IID noise = pure random walk (unit root).
        // ADF should NOT strongly reject, meaning statistic should sit
        // near or above the 10% critical value (not much more negative).
        let n = 500;
        let mut seed: u64 = 0xa3c2b15917fb07c5;
        let mut x = vec![0.0_f64; n];
        let mut cum = 0.0;
        for i in 0..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
            cum += u - 0.5;
            x[i] = cum;
        }
        let r = stationarity_adf(&x, 1);
        assert!(r.statistic.is_finite());
        // A true random walk rarely produces an ADF below -3 in 500 pts.
        assert!(r.statistic > -3.0, "unit root too negative: {}", r.statistic);
    }

    #[test]
    fn stationary_ar1_below_critical() {
        // AR(1) with φ=0.3 (strongly stationary) → ADF should be very
        // negative, well below the 5% critical value.
        let n = 500;
        let mut seed: u64 = 0x5a5a5a5aa5a5a5a5;
        let mut x = vec![0.0_f64; n];
        for i in 1..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
            x[i] = 0.3 * x[i - 1] + (u - 0.5);
        }
        let r = stationarity_adf(&x, 1);
        assert!(r.statistic.is_finite());
        assert!(
            r.statistic < r.critical_5pct,
            "stationary AR(1) failed to reject unit root: stat={} 5%={}",
            r.statistic,
            r.critical_5pct
        );
    }

    #[test]
    fn non_finite_values_are_skipped_not_poisoned() {
        // Per the contract, NaN/Inf entries are skipped, not propagated.
        // A stationary AR(1) with a few NaNs injected should still
        // produce a finite ADF statistic.
        let n = 500;
        let mut seed: u64 = 0x5a5a5a5aa5a5a5a5;
        let mut x = vec![0.0_f64; n];
        for i in 1..n {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            let u = (seed >> 11) as f64 / (1u64 << 53) as f64;
            x[i] = 0.3 * x[i - 1] + (u - 0.5);
        }
        x[100] = f64::NAN;
        x[200] = f64::INFINITY;
        x[300] = f64::NEG_INFINITY;
        let r = stationarity_adf(&x, 1);
        assert!(r.statistic.is_finite(), "statistic should survive NaN skips");
    }

    #[test]
    fn all_non_finite_input_nan() {
        let x = vec![f64::NAN; 200];
        let r = stationarity_adf(&x, 1);
        assert!(r.statistic.is_nan());
    }

    #[test]
    fn constant_series_handled_gracefully() {
        // Constant series: Δy ≡ 0, the regression is rank-deficient and
        // the statistic should be NaN rather than panicking.
        let x = vec![5.0; 100];
        let r = stationarity_adf(&x, 1);
        // Either NaN statistic or a degenerate result — must not panic.
        assert!(r.statistic.is_nan() || r.statistic.is_finite());
    }

    #[test]
    fn n_lags_respected() {
        let x: Vec<f64> = (0..100).map(|i| i as f64 + 0.1 * (i as f64).sin()).collect();
        let r0 = stationarity_adf(&x, 0);
        let r3 = stationarity_adf(&x, 3);
        assert_eq!(r0.n_lags, 0);
        assert_eq!(r3.n_lags, 3);
        // Both statistics finite on this series.
        assert!(r0.statistic.is_finite());
        assert!(r3.statistic.is_finite());
    }
}
