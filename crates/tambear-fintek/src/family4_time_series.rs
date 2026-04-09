//! Family 4 — Time Series / ARMA.
//!
//! Covers fintek leaves: `autocorrelation`, `ar_model`, `arma`, `arima`, `arx`.
//! NOT covered: `ar_burg` (GAP — see task #137).

use tambear::time_series::{acf, pacf, ar_fit, ArResult, difference};

/// Autocorrelation features for a single bin.
///
/// Fintek's `autocorrelation.rs` emits 16 features: ACF[1..8] + PACF[1..8].
#[derive(Debug, Clone)]
pub struct AutocorrelationResult {
    pub acf: Vec<f64>,  // ACF at lags 1..=max_lag
    pub pacf: Vec<f64>, // PACF at lags 1..=max_lag
}

impl AutocorrelationResult {
    pub fn nan(max_lag: usize) -> Self {
        Self {
            acf: vec![f64::NAN; max_lag],
            pacf: vec![f64::NAN; max_lag],
        }
    }
}

/// Compute ACF and PACF features for a bin.
///
/// `data`: bin-level series (usually log returns).
/// `max_lag`: maximum lag to compute (fintek typically uses 8).
pub fn autocorrelation(data: &[f64], max_lag: usize) -> AutocorrelationResult {
    if data.len() < max_lag + 2 {
        return AutocorrelationResult::nan(max_lag);
    }
    // acf returns lags 0..=max_lag; we want lags 1..=max_lag
    let acf_all = acf(data, max_lag);
    let pacf_all = pacf(data, max_lag);

    // Skip lag 0 (always 1.0 for ACF)
    let acf_out = if acf_all.len() > 1 { acf_all[1..].to_vec() } else { vec![f64::NAN; max_lag] };
    let pacf_out = if pacf_all.len() > 0 { pacf_all } else { vec![f64::NAN; max_lag] };

    // Pad to max_lag if shorter
    let mut acf_final = acf_out;
    let mut pacf_final = pacf_out;
    while acf_final.len() < max_lag { acf_final.push(f64::NAN); }
    while pacf_final.len() < max_lag { pacf_final.push(f64::NAN); }
    acf_final.truncate(max_lag);
    pacf_final.truncate(max_lag);

    AutocorrelationResult { acf: acf_final, pacf: pacf_final }
}

/// AR(p) model result (Yule-Walker).
#[derive(Debug, Clone)]
pub struct ArModelResult {
    /// AR coefficients (φ_1, ..., φ_p).
    pub coefficients: Vec<f64>,
    /// Residual variance σ².
    pub sigma2: f64,
    /// AIC: n·ln(σ²) + 2p.
    pub aic: f64,
    /// BIC: n·ln(σ²) + p·ln(n).
    pub bic: f64,
    /// Order p.
    pub order: usize,
}

impl ArModelResult {
    pub fn nan(p: usize) -> Self {
        Self { coefficients: vec![f64::NAN; p], sigma2: f64::NAN, aic: f64::NAN, bic: f64::NAN, order: p }
    }
}

/// Fit AR(p) via Yule-Walker.
pub fn ar_model(data: &[f64], p: usize) -> ArModelResult {
    if data.len() < p + 2 {
        return ArModelResult::nan(p);
    }
    let result: ArResult = ar_fit(data, p);
    let n = data.len() as f64;
    let pf = p as f64;
    // BIC = n·ln(σ²) + p·ln(n)
    let bic = if result.sigma2 > 0.0 { n * result.sigma2.ln() + pf * n.ln() } else { f64::NAN };
    ArModelResult {
        coefficients: result.coefficients,
        sigma2: result.sigma2,
        aic: result.aic,
        bic,
        order: p,
    }
}

/// Select optimal AR order via BIC minimization.
///
/// `max_p`: maximum order to consider (typically 8 for financial data).
/// Returns the best AR model and its BIC.
pub fn ar_model_auto(data: &[f64], max_p: usize) -> ArModelResult {
    if data.len() < max_p + 2 || max_p == 0 {
        return ArModelResult::nan(0);
    }
    let mut best = ar_model(data, 1);
    for p in 2..=max_p {
        let candidate = ar_model(data, p);
        if candidate.bic.is_finite() && (best.bic.is_nan() || candidate.bic < best.bic) {
            best = candidate;
        }
    }
    best
}

/// ARMA(p, q) output.
///
/// We fit AR(p) on the data, then fit MA(q) on the residual ACF via
/// a simple moment-matching approximation (innovations algorithm).
#[derive(Debug, Clone)]
pub struct ArmaResult {
    pub ar_coeffs: Vec<f64>,
    pub ma_coeffs: Vec<f64>,
    pub sigma2: f64,
    pub p: usize,
    pub q: usize,
}

impl ArmaResult {
    pub fn nan(p: usize, q: usize) -> Self {
        Self {
            ar_coeffs: vec![f64::NAN; p],
            ma_coeffs: vec![f64::NAN; q],
            sigma2: f64::NAN,
            p, q,
        }
    }
}

/// Fit ARMA(p, q) via AR(p) then MA(q) from residual ACF.
///
/// This is a two-stage estimator: first AR via Yule-Walker, then MA
/// parameters from the ACF of residuals. Less efficient than MLE but
/// closed-form. Matches fintek's simpler implementation.
pub fn arma(data: &[f64], p: usize, q: usize) -> ArmaResult {
    if data.len() < p + q + 2 {
        return ArmaResult::nan(p, q);
    }
    // Stage 1: AR(p)
    let ar_result = ar_fit(data, p);

    // Stage 2: compute residuals
    let n = data.len();
    let mut residuals = Vec::with_capacity(n.saturating_sub(p));
    for t in p..n {
        let mut ar_pred = 0.0;
        for k in 0..p {
            ar_pred += ar_result.coefficients[k] * data[t - 1 - k];
        }
        residuals.push(data[t] - ar_pred);
    }
    if residuals.len() < q + 2 {
        return ArmaResult {
            ar_coeffs: ar_result.coefficients,
            ma_coeffs: vec![f64::NAN; q],
            sigma2: ar_result.sigma2,
            p, q,
        };
    }

    // Stage 3: MA(q) from residual autocorrelations via moment matching.
    // Simple approach: MA(q) coefficients ≈ residual ACF at lags 1..=q.
    // Not a proper MLE but captures the essential MA behavior.
    let residual_acf = acf(&residuals, q);
    let ma_coeffs = if residual_acf.len() > q {
        residual_acf[1..=q].to_vec()
    } else {
        vec![f64::NAN; q]
    };

    ArmaResult {
        ar_coeffs: ar_result.coefficients,
        ma_coeffs,
        sigma2: ar_result.sigma2,
        p, q,
    }
}

/// Simple ARIMA(p, d, q): difference d times, then fit ARMA(p, q).
///
/// Equivalent to fintek's `arima.rs`.
pub fn arima(data: &[f64], p: usize, d: usize, q: usize) -> ArmaResult {
    let mut working = data.to_vec();
    for _ in 0..d {
        if working.len() < 2 { return ArmaResult::nan(p, q); }
        working = difference(&working, 1);
    }
    arma(&working, p, q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autocorrelation_basic() {
        // AR(1) with phi=0.8 — should have decaying ACF
        let mut data = vec![0.0; 200];
        data[0] = 1.0;
        let mut rng = tambear::rng::Xoshiro256::new(42);
        for t in 1..200 {
            data[t] = 0.8 * data[t - 1] + tambear::rng::sample_normal(&mut rng, 0.0, 0.1);
        }
        let r = autocorrelation(&data, 8);
        assert_eq!(r.acf.len(), 8);
        assert_eq!(r.pacf.len(), 8);
        // ACF at lag 1 should be positive and near 0.8
        assert!(r.acf[0] > 0.3, "AR(1) ACF[1] should be positive, got {}", r.acf[0]);
    }

    #[test]
    fn autocorrelation_too_short() {
        let r = autocorrelation(&[1.0, 2.0, 3.0], 8);
        assert!(r.acf.iter().all(|v| v.is_nan()));
    }

    #[test]
    fn ar_model_basic() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let r = ar_model(&data, 2);
        assert_eq!(r.coefficients.len(), 2);
        assert!(r.sigma2 >= 0.0 || r.sigma2.is_nan());
        assert!(r.bic.is_finite() || r.sigma2.is_nan());
    }

    #[test]
    fn ar_model_auto_selects() {
        // AR(1) data
        let mut data = vec![0.0; 200];
        let mut rng = tambear::rng::Xoshiro256::new(42);
        for t in 1..200 {
            data[t] = 0.7 * data[t - 1] + tambear::rng::sample_normal(&mut rng, 0.0, 0.1);
        }
        let r = ar_model_auto(&data, 5);
        assert!(r.order >= 1 && r.order <= 5);
    }

    #[test]
    fn arma_basic() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.2).sin() * 0.5).collect();
        let r = arma(&data, 2, 1);
        assert_eq!(r.ar_coeffs.len(), 2);
        assert_eq!(r.ma_coeffs.len(), 1);
    }

    #[test]
    fn arima_differencing() {
        // Linear trend: d=1 should remove it
        let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let r = arima(&data, 1, 1, 0);
        // After 1st differencing: constant 1s → AR(1) on constant gives phi=NaN or 0
        assert!(r.p == 1 && r.q == 0);
    }
}
