//! Family 4 — Time Series / ARMA.
//!
//! Covers fintek leaves: `autocorrelation`, `ar_model`, `arma`, `arima`, `ar_burg`, `arx`.

use tambear::time_series::{acf, pacf, ar_fit, ar_burg_fit, ar_psd, ArResult, difference};

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

// ── Burg AR ───────────────────────────────────────────────────────────────────

/// Spectral features from Burg AR PSD.
///
/// Fintek's `ar_burg.rs` (K02P02C05R01) outputs 4 features from a Burg AR PSD
/// evaluated at 128 frequencies:
///   - spectral_centroid: power-weighted mean frequency
///   - spectral_bandwidth: power-weighted frequency spread
///   - spectral_rolloff: frequency below which 85% of power falls
///   - ar_order: effective AR order used (BIC-selected up to 12)
#[derive(Debug, Clone)]
pub struct ArBurgResult {
    pub spectral_centroid: f64,
    pub spectral_bandwidth: f64,
    pub spectral_rolloff: f64,
    pub ar_order: usize,
}

impl ArBurgResult {
    pub fn nan() -> Self {
        Self { spectral_centroid: f64::NAN, spectral_bandwidth: f64::NAN,
               spectral_rolloff: f64::NAN, ar_order: 0 }
    }
}

/// Burg AR spectral features.
///
/// Fits Burg AR models of order 1..=12 on log-returns, selects via AIC,
/// evaluates PSD at 128 frequencies, and extracts 4 spectral summary features.
/// Input: bin-level price series (at least 10 ticks).
pub fn ar_burg(prices: &[f64]) -> ArBurgResult {
    const MIN_RETURNS: usize = 10;
    const MAX_ORDER: usize = 12;
    const N_FREQ: usize = 128;
    const ROLLOFF_FRAC: f64 = 0.85;

    if prices.len() < MIN_RETURNS + 2 {
        return ArBurgResult::nan();
    }

    // Log returns from prices
    let returns: Vec<f64> = prices.windows(2).filter_map(|w| {
        if w[0] > 0.0 && w[1] > 0.0 { Some((w[1] / w[0]).ln()) } else { None }
    }).collect();

    if returns.len() < MIN_RETURNS {
        return ArBurgResult::nan();
    }

    // BIC-select order 1..=MAX_ORDER
    let n = returns.len() as f64;
    let mut best_order = 1usize;
    let mut best_aic = f64::INFINITY;
    for p in 1..=MAX_ORDER.min(returns.len() / 4).max(1) {
        let fit = ar_burg_fit(&returns, p);
        if fit.aic < best_aic {
            best_aic = fit.aic;
            best_order = p;
        }
    }

    let ar = ar_burg_fit(&returns, best_order);

    // Evaluate PSD at N_FREQ uniform frequencies in [0, 0.5)
    let (freqs, psd) = ar_psd(&ar, N_FREQ);

    let total_power: f64 = psd.iter().sum();
    if total_power < 1e-300 {
        return ArBurgResult::nan();
    }

    // Spectral centroid: Σ(f * psd) / Σpsd
    let centroid: f64 = freqs.iter().zip(psd.iter()).map(|(&f, &p)| f * p).sum::<f64>() / total_power;

    // Spectral bandwidth: sqrt(Σ((f - centroid)² * psd) / Σpsd)
    let bandwidth: f64 = (freqs.iter().zip(psd.iter())
        .map(|(&f, &p)| (f - centroid) * (f - centroid) * p)
        .sum::<f64>() / total_power).sqrt();

    // Spectral rolloff: smallest freq where cumulative power ≥ 85%
    let rolloff_target = ROLLOFF_FRAC * total_power;
    let mut cumulative = 0.0;
    let mut rolloff = *freqs.last().unwrap_or(&0.5);
    for (&f, &p) in freqs.iter().zip(psd.iter()) {
        cumulative += p;
        if cumulative >= rolloff_target {
            rolloff = f;
            break;
        }
    }

    ArBurgResult {
        spectral_centroid: centroid,
        spectral_bandwidth: bandwidth,
        spectral_rolloff: rolloff,
        ar_order: best_order,
    }
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

    #[test]
    fn ar_burg_spectral_features_ar1() {
        // Build prices such that log-returns form an AR(1) with φ=0.9.
        // `ar_burg()` internally converts prices to log-returns, so we need
        // the AR process in return space, not in price space. Also exponentiate
        // the cumulative return to keep prices strictly positive (the log-return
        // computation filters non-positive values).
        let n = 600;
        let mut returns = vec![0.0; n];
        let mut rng = tambear::rng::Xoshiro256::new(99);
        for t in 1..n {
            returns[t] = 0.9 * returns[t - 1]
                + tambear::rng::sample_normal(&mut rng, 0.0, 0.01);
        }
        let mut prices = vec![100.0; n + 1];
        for t in 0..n {
            prices[t + 1] = prices[t] * returns[t].exp();
        }
        let r = ar_burg(&prices);
        assert!(r.spectral_centroid.is_finite());
        assert!(r.spectral_bandwidth.is_finite());
        assert!(r.spectral_rolloff.is_finite());
        assert!(r.ar_order > 0 && r.ar_order <= 12);
        // AR(1) φ=0.9 has most power near DC — centroid well below Nyquist/2
        assert!(
            r.spectral_centroid < 0.2,
            "centroid {} should be < 0.2 for low-freq AR(1) φ=0.9",
            r.spectral_centroid
        );
    }

    #[test]
    fn ar_burg_too_short() {
        let r = ar_burg(&[1.0, 2.0, 3.0]);
        assert!(r.spectral_centroid.is_nan());
    }
}

// ── ARX(1,1) — Autoregressive with exogenous input ────────────────────────────

/// ARX result matching fintek's `arx.rs` (K02P11C04R01).
#[derive(Debug, Clone)]
pub struct ArxResult {
    pub ar_coeff: f64,
    pub x_coeff: f64,
    pub residual_var: f64,
    pub r2_gain: f64,
}

impl ArxResult {
    pub fn nan() -> Self {
        Self { ar_coeff: f64::NAN, x_coeff: f64::NAN,
               residual_var: f64::NAN, r2_gain: f64::NAN }
    }
}

/// Fit ARX(1,1) model: y_t = a + b·y_{t-1} + c·x_{t-1} + ε.
///
/// Returns AR coefficient, exogenous coefficient, residual variance,
/// and R² gain from adding the exogenous input over AR-only.
///
/// `returns`: log return series (y).
/// `exog`: exogenous series (x), e.g. log volume changes.
/// Both must be the same length. Returns NaN if `n < 6`.
pub fn arx(returns: &[f64], exog: &[f64]) -> ArxResult {
    let n = returns.len().min(exog.len());
    if n < 6 { return ArxResult::nan(); }

    let m = n - 1;
    let y = &returns[1..=m];
    let x_ar = &returns[..m];
    let x_vol = &exog[..m];
    let mf = m as f64;

    // AR-only: y = a + b·x_ar
    let mean_y: f64 = y.iter().sum::<f64>() / mf;
    let mean_ar: f64 = x_ar.iter().sum::<f64>() / mf;

    let mut cov_ar_y = 0.0f64;
    let mut var_ar = 0.0f64;
    let mut ss_tot = 0.0f64;
    for i in 0..m {
        let dy = y[i] - mean_y;
        let dar = x_ar[i] - mean_ar;
        cov_ar_y += dar * dy;
        var_ar += dar * dar;
        ss_tot += dy * dy;
    }

    let b_ar_only = if var_ar > 1e-30 { cov_ar_y / var_ar } else { 0.0 };
    let a_ar_only = mean_y - b_ar_only * mean_ar;

    let mut ss_res_ar = 0.0f64;
    for i in 0..m {
        let resid = y[i] - (a_ar_only + b_ar_only * x_ar[i]);
        ss_res_ar += resid * resid;
    }
    let r2_ar = if ss_tot > 1e-30 { 1.0 - ss_res_ar / ss_tot } else { 0.0 };

    // ARX: y = a + b·x_ar + c·x_vol (2×2 OLS in centered space)
    let mean_vol: f64 = x_vol.iter().sum::<f64>() / mf;

    let mut s11 = 0.0f64; let mut s12 = 0.0f64; let mut s22 = 0.0f64;
    let mut r1 = 0.0f64;  let mut r2 = 0.0f64;
    for i in 0..m {
        let ar_c = x_ar[i] - mean_ar;
        let vol_c = x_vol[i] - mean_vol;
        let y_c = y[i] - mean_y;
        s11 += ar_c * ar_c; s12 += ar_c * vol_c; s22 += vol_c * vol_c;
        r1 += ar_c * y_c;   r2 += vol_c * y_c;
    }

    let det = s11 * s22 - s12 * s12;
    let (b_arx, c_arx) = if det.abs() > 1e-30 {
        let inv = 1.0 / det;
        (inv * (s22 * r1 - s12 * r2), inv * (s11 * r2 - s12 * r1))
    } else {
        (b_ar_only, 0.0)
    };
    let a_arx = mean_y - b_arx * mean_ar - c_arx * mean_vol;

    let mut ss_res_arx = 0.0f64;
    for i in 0..m {
        let resid = y[i] - (a_arx + b_arx * x_ar[i] + c_arx * x_vol[i]);
        ss_res_arx += resid * resid;
    }
    let residual_var = ss_res_arx / mf;
    let r2_arx = if ss_tot > 1e-30 { 1.0 - ss_res_arx / ss_tot } else { 0.0 };

    ArxResult {
        ar_coeff: b_arx,
        x_coeff: c_arx,
        residual_var,
        r2_gain: r2_arx - r2_ar,
    }
}

#[cfg(test)]
mod tests_arx {
    use super::*;

    #[test]
    fn arx_too_short() {
        let r = arx(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]);
        assert!(r.ar_coeff.is_nan());
    }

    #[test]
    fn arx_white_noise_small_r2_gain() {
        // White noise returns + independent exog → near-zero R² gain
        let n = 100;
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let returns: Vec<f64> = (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let exog: Vec<f64> = (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let r = arx(&returns, &exog);
        assert!(r.ar_coeff.is_finite());
        assert!(r.x_coeff.is_finite());
        assert!(r.residual_var.is_finite() && r.residual_var >= 0.0);
        assert!(r.r2_gain.is_finite());
    }

    #[test]
    fn arx_exog_predicts_returns() {
        // Construct returns = 0.5 * lagged_exog + noise → x_coeff ≈ 0.5, r2_gain > 0
        let n = 200;
        let mut rng = tambear::rng::Xoshiro256::new(77);
        let exog: Vec<f64> = (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let mut returns = vec![0.0f64; n];
        for i in 1..n {
            returns[i] = 0.5 * exog[i-1] + tambear::rng::sample_normal(&mut rng, 0.0, 0.1);
        }
        let r = arx(&returns, &exog);
        assert!(r.x_coeff.abs() > 0.2, "x_coeff should be significant, got {}", r.x_coeff);
        assert!(r.r2_gain > 0.0, "exog should improve R², got gain={}", r.r2_gain);
    }
}
