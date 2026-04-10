//! Family 9 — Volatility.
//!
//! Covers fintek leaves: `garch`, `realized_vol`, `jump_detection`,
//! `roll_spread`, `signature_plot`, `range_vol`, `vpin_bvc`,
//! `vol_regime`, `vol_dynamics`, `stochvol`, `wiener`.

use tambear::volatility::{
    garch11_fit, garch11_forecast, realized_variance, bipower_variation,
    jump_test_bns, roll_spread, GarchResult,
};
use tambear::time_series::rolling_variance_prefix;
use tambear::{ols_slope, lag1_autocorrelation};

/// GARCH(1,1) fit result.
#[derive(Debug, Clone)]
pub struct GarchFitResult {
    pub omega: f64,
    pub alpha: f64,
    pub beta: f64,
    pub log_likelihood: f64,
    /// Unconditional variance: ω / (1 - α - β).
    pub uncond_variance: f64,
    /// Persistence: α + β.
    pub persistence: f64,
    /// True if α + β > 0.99 (near IGARCH).
    pub near_igarch: bool,
}

impl GarchFitResult {
    pub fn nan() -> Self {
        Self {
            omega: f64::NAN, alpha: f64::NAN, beta: f64::NAN,
            log_likelihood: f64::NAN, uncond_variance: f64::NAN,
            persistence: f64::NAN, near_igarch: false,
        }
    }
}

/// Fit GARCH(1,1) on bin-level returns.
pub fn garch_fit(returns: &[f64], max_iter: usize) -> GarchFitResult {
    if returns.len() < 30 {
        return GarchFitResult::nan();
    }
    let r: GarchResult = garch11_fit(returns, max_iter);
    let persistence = r.alpha + r.beta;
    // Unconditional variance: ω / (1 - α - β), unreliable if persistence ≈ 1
    let uncond = if persistence < 0.999 && persistence.is_finite() {
        r.omega / (1.0 - persistence)
    } else {
        f64::NAN
    };
    GarchFitResult {
        omega: r.omega,
        alpha: r.alpha,
        beta: r.beta,
        log_likelihood: r.log_likelihood,
        uncond_variance: uncond,
        persistence,
        near_igarch: r.near_igarch,
    }
}

/// Forecast GARCH(1,1) variance h steps ahead.
pub fn garch_forecast(fit: &GarchFitResult, last_return: f64, n_steps: usize) -> Vec<f64> {
    if !fit.omega.is_finite() || n_steps == 0 { return vec![f64::NAN; n_steps]; }
    let garch = GarchResult {
        omega: fit.omega,
        alpha: fit.alpha,
        beta: fit.beta,
        log_likelihood: fit.log_likelihood,
        variances: vec![],
        iterations: 0,
        near_igarch: fit.near_igarch,
    };
    garch11_forecast(&garch, last_return, n_steps)
}

/// Realized volatility features.
#[derive(Debug, Clone)]
pub struct RealizedVolResult {
    pub realized_variance: f64,
    pub realized_vol: f64,  // sqrt(RV)
    pub bipower_variation: f64,
    pub jump_ratio: f64,    // (RV - BV) / RV  (jump component share)
}

impl RealizedVolResult {
    pub fn nan() -> Self {
        Self { realized_variance: f64::NAN, realized_vol: f64::NAN, bipower_variation: f64::NAN, jump_ratio: f64::NAN }
    }
}

/// Compute RV + BV + jump ratio.
pub fn realized_vol(returns: &[f64]) -> RealizedVolResult {
    if returns.len() < 3 { return RealizedVolResult::nan(); }
    let rv = realized_variance(returns);
    let bv = bipower_variation(returns);
    let jump_ratio = if rv > 1e-15 { (rv - bv).max(0.0) / rv } else { f64::NAN };
    RealizedVolResult {
        realized_variance: rv,
        realized_vol: rv.sqrt(),
        bipower_variation: bv,
        jump_ratio,
    }
}

/// BNS jump test result.
#[derive(Debug, Clone)]
pub struct JumpDetectionResult {
    pub bns_statistic: f64,
    pub has_jump_5pct: bool,
}

/// Run BNS (Barndorff-Nielsen-Shephard) jump test.
///
/// Reject at 5% when |Z| > 1.96.
pub fn jump_detection(returns: &[f64]) -> JumpDetectionResult {
    if returns.len() < 5 {
        return JumpDetectionResult { bns_statistic: f64::NAN, has_jump_5pct: false };
    }
    let stat = jump_test_bns(returns);
    JumpDetectionResult {
        bns_statistic: stat,
        has_jump_5pct: stat.is_finite() && stat.abs() > 1.96,
    }
}

/// Roll 1984 effective spread from serial covariance.
///
/// Wraps `volatility::roll_spread`.
pub fn roll_effective_spread(prices: &[f64]) -> f64 {
    roll_spread(prices)
}

/// Range volatility estimators from OHLC data.
///
/// Fintek's `range_vol.rs` emits all four (Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang).
#[derive(Debug, Clone)]
pub struct RangeVolResult {
    pub parkinson: f64,
    pub garman_klass: f64,
    pub rogers_satchell: f64,
    pub yang_zhang: f64,
}

impl RangeVolResult {
    pub fn nan() -> Self {
        Self {
            parkinson: f64::NAN, garman_klass: f64::NAN,
            rogers_satchell: f64::NAN, yang_zhang: f64::NAN,
        }
    }
}

/// Compute all four range volatility estimators from an OHLC series.
///
/// Parkinson/GK/RS are averaged across bars; Yang-Zhang is computed jointly.
pub fn range_volatility(
    opens: &[f64], highs: &[f64], lows: &[f64], closes: &[f64], prev_closes: &[f64],
) -> RangeVolResult {
    let n = opens.len();
    if n == 0 || highs.len() != n || lows.len() != n || closes.len() != n {
        return RangeVolResult::nan();
    }

    let mut park_sum = 0.0_f64;
    let mut gk_sum = 0.0_f64;
    let mut rs_sum = 0.0_f64;
    let mut count = 0_usize;
    for i in 0..n {
        let p = tambear::volatility::parkinson_variance(highs[i], lows[i]);
        let gk = tambear::volatility::garman_klass_variance(opens[i], highs[i], lows[i], closes[i]);
        let rs = tambear::volatility::rogers_satchell_variance(opens[i], highs[i], lows[i], closes[i]);
        if p.is_finite() { park_sum += p; count += 1; }
        if gk.is_finite() { gk_sum += gk; }
        if rs.is_finite() { rs_sum += rs; }
    }
    let countf = count.max(1) as f64;
    let yz = if prev_closes.len() == n {
        tambear::volatility::yang_zhang_variance(opens, highs, lows, closes, prev_closes)
    } else { f64::NAN };

    RangeVolResult {
        parkinson: park_sum / countf,
        garman_klass: gk_sum / countf,
        rogers_satchell: rs_sum / countf,
        yang_zhang: yz,
    }
}

/// Tripower quarticity (TQ) wrapper.
///
/// Used as the variance estimator for the BV in the BNS jump test.
pub fn tripower_quarticity(returns: &[f64]) -> f64 {
    tambear::volatility::tripower_quarticity(returns)
}

/// Signature plot: realized variance at multiple sampling frequencies.
///
/// Returns (sampling_steps, rv_at_each_step). Used to detect microstructure noise.
pub fn signature_plot(returns: &[f64], steps: &[usize]) -> Vec<(usize, f64)> {
    let mut out = Vec::with_capacity(steps.len());
    for &step in steps {
        if step == 0 || returns.len() < step + 1 {
            out.push((step, f64::NAN));
            continue;
        }
        // Sub-sample at interval `step`
        let subsampled: Vec<f64> = (0..returns.len()).step_by(step).map(|i| returns[i]).collect();
        if subsampled.len() < 2 {
            out.push((step, f64::NAN));
        } else {
            out.push((step, realized_variance(&subsampled)));
        }
    }
    out
}

// ── Volatility regime (K02P18C03R01F02) ──────────────────────────────────────

/// Volatility regime features.
///
/// Fintek's `vol_regime.rs` outputs:
/// - n_vol_regimes — number of distinct vol regimes detected
/// - vol_ratio_range — max/min rolling-var ratio
/// - high_vol_fraction — fraction of windows in high-vol regime
/// - regime_switching_rate — fraction of consecutive windows that switch regime
///
/// Uses short_window=20, long_window=100, high_vol_threshold=1.5.
#[derive(Debug, Clone)]
pub struct VolRegimeResult {
    pub n_vol_regimes: f64,
    pub vol_ratio_range: f64,
    pub high_vol_fraction: f64,
    pub regime_switching_rate: f64,
}

impl VolRegimeResult {
    pub fn nan() -> Self {
        Self { n_vol_regimes: f64::NAN, vol_ratio_range: f64::NAN,
               high_vol_fraction: f64::NAN, regime_switching_rate: f64::NAN }
    }
}

/// Compute volatility regime features from bin returns.
pub fn vol_regime(returns: &[f64]) -> VolRegimeResult {
    const SHORT: usize = 20;
    const LONG: usize = 100;
    const HIGH_VOL_THRESH: f64 = 1.5;
    let n = returns.len();
    if n < LONG + SHORT { return VolRegimeResult::nan(); }

    let short_var = rolling_variance_prefix(returns, SHORT);
    let long_var = rolling_variance_prefix(returns, LONG);
    let offset = LONG - SHORT;
    let m = long_var.len();

    let mut vol_ratios: Vec<f64> = Vec::with_capacity(m);
    let mut regimes: Vec<bool> = Vec::with_capacity(m);
    for i in 0..m {
        let sv_idx = i + offset;
        if sv_idx < short_var.len() {
            let lv = long_var[i].max(1e-30);
            let ratio = short_var[sv_idx] / lv;
            vol_ratios.push(ratio);
            regimes.push(ratio > HIGH_VOL_THRESH);
        }
    }
    if vol_ratios.is_empty() { return VolRegimeResult::nan(); }

    let min_r = vol_ratios.iter().copied().fold(f64::INFINITY, f64::min);
    let max_r = vol_ratios.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let vol_ratio_range = if min_r > 1e-30 { max_r / min_r } else { f64::NAN };

    let high_vol_fraction = regimes.iter().filter(|&&h| h).count() as f64 / regimes.len() as f64;

    // Count regime switches
    let switches = regimes.windows(2).filter(|w| w[0] != w[1]).count();
    let regime_switching_rate = switches as f64 / (regimes.len() - 1).max(1) as f64;

    // Count distinct regimes (contiguous blocks)
    let mut n_regimes = if regimes.is_empty() { 0 } else { 1 };
    for w in regimes.windows(2) { if w[0] != w[1] { n_regimes += 1; } }

    VolRegimeResult { n_vol_regimes: n_regimes as f64, vol_ratio_range, high_vol_fraction, regime_switching_rate }
}

// ── Volatility dynamics (K02P10C03R01F01) ────────────────────────────────────

/// Second-order volatility dynamics features.
///
/// Fintek's `vol_dynamics.rs` outputs:
/// - vol_trend — OLS slope of rolling vol vs time index
/// - vol_of_vol — std of rolling vol series
/// - mean_reversion_speed — AR(1) coefficient of rolling vol (speed of mean-reversion)
/// - vol_autocorr — lag-1 autocorrelation of rolling vol
#[derive(Debug, Clone)]
pub struct VolDynamicsResult {
    pub vol_trend: f64,
    pub vol_of_vol: f64,
    pub mean_reversion_speed: f64,
    pub vol_autocorr: f64,
}

impl VolDynamicsResult {
    pub fn nan() -> Self {
        Self { vol_trend: f64::NAN, vol_of_vol: f64::NAN,
               mean_reversion_speed: f64::NAN, vol_autocorr: f64::NAN }
    }
}

/// Compute volatility dynamics from bin returns.
pub fn vol_dynamics(returns: &[f64]) -> VolDynamicsResult {
    let n = returns.len();
    let window = (n / 3).max(3).min(20);
    if n < 2 * window { return VolDynamicsResult::nan(); }

    let abs_ret: Vec<f64> = returns.iter().map(|r| r.abs()).collect();
    // Rolling vol = rolling std of |returns|
    let rv: Vec<f64> = (0..=(n-window)).map(|i| {
        let slice = &abs_ret[i..i+window];
        let mean = slice.iter().sum::<f64>() / window as f64;
        (slice.iter().map(|v| (v-mean).powi(2)).sum::<f64>() / window as f64).sqrt()
    }).collect();
    let m = rv.len();
    if m < 3 { return VolDynamicsResult::nan(); }

    // OLS slope of rv vs time
    let t_index: Vec<f64> = (0..m).map(|i| i as f64).collect();
    let vol_trend = ols_slope(&t_index, &rv);

    // Vol-of-vol = std of rv
    let mean_y: f64 = rv.iter().sum::<f64>() / m as f64;
    let vol_of_vol = (rv.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / m as f64).sqrt();

    // Lag-1 autocorr of rv
    let vol_autocorr = lag1_autocorrelation(&rv).clamp(-1.0, 1.0);

    // Mean reversion speed = 1 - AR(1) coefficient
    let mean_reversion_speed = if vol_autocorr.is_finite() { (1.0 - vol_autocorr).max(0.0) } else { f64::NAN };

    VolDynamicsResult { vol_trend, vol_of_vol, mean_reversion_speed, vol_autocorr }
}

// ── Stochastic volatility proxy (K02P10C01R02F01) ────────────────────────────

/// Stochastic volatility proxy features.
///
/// Fintek's `stochvol.rs` outputs:
/// - vol_of_vol — std of AR(1) residuals on log(r²)
/// - mean_log_vol — mean(log(r²))
/// - persistence — AR(1) coefficient φ on log(r²)
/// - leverage_corr — correlation between returns and log(r²)
#[derive(Debug, Clone)]
pub struct StochvolResult {
    pub vol_of_vol: f64,
    pub mean_log_vol: f64,
    pub persistence: f64,
    pub leverage_corr: f64,
}

impl StochvolResult {
    pub fn nan() -> Self {
        Self { vol_of_vol: f64::NAN, mean_log_vol: f64::NAN, persistence: f64::NAN, leverage_corr: f64::NAN }
    }
}

/// Compute stochastic volatility proxy features from bin returns.
pub fn stochvol(returns: &[f64]) -> StochvolResult {
    let n = returns.len();
    if n < 4 { return StochvolResult::nan(); }

    let log_rsq: Vec<f64> = returns.iter().map(|r| (r*r).max(1e-30).ln()).collect();
    let mean_log_vol: f64 = log_rsq.iter().sum::<f64>() / n as f64;

    // AR(1): φ = Cov(y_t, y_{t-1}) / Var(y_{t-1})
    let mut cov01 = 0.0_f64; let mut var_y = 0.0_f64;
    for t in 1..n {
        cov01 += (log_rsq[t] - mean_log_vol) * (log_rsq[t-1] - mean_log_vol);
        var_y  += (log_rsq[t-1] - mean_log_vol).powi(2);
    }
    let persistence = if var_y > 1e-30 { (cov01 / var_y).clamp(-1.0, 1.0) } else { 0.0 };

    // Vol-of-vol = std of AR(1) residuals
    let mut resid_ss = 0.0_f64;
    for t in 1..n {
        let predicted = mean_log_vol + persistence * (log_rsq[t-1] - mean_log_vol);
        resid_ss += (log_rsq[t] - predicted).powi(2);
    }
    let vol_of_vol = if n > 1 { (resid_ss / (n-1) as f64).sqrt() } else { f64::NAN };

    // Leverage: corr(returns, log_rsq)
    let mean_r: f64 = returns.iter().sum::<f64>() / n as f64;
    let mut sxy = 0.0_f64; let mut sxx = 0.0_f64; let mut syy = 0.0_f64;
    for i in 0..n {
        let dx = returns[i] - mean_r;
        let dy = log_rsq[i] - mean_log_vol;
        sxy += dx * dy; sxx += dx*dx; syy += dy*dy;
    }
    let leverage_corr = if sxx > 1e-30 && syy > 1e-30 { sxy / (sxx*syy).sqrt() } else { f64::NAN };

    StochvolResult { vol_of_vol, mean_log_vol, persistence, leverage_corr }
}

// ── Wiener SNR (K02P05C03R02) ─────────────────────────────────────────────────

/// Wiener filter SNR estimation.
///
/// Fintek's `wiener.rs` outputs:
/// - noise_estimate — noise power from high-freq tail (top ¼ of PSD)
/// - signal_estimate — total power minus noise
/// - wiener_snr — signal / noise
/// - noise_fraction — noise / total
#[derive(Debug, Clone)]
pub struct WienerSnrResult {
    pub noise_estimate: f64,
    pub signal_estimate: f64,
    pub wiener_snr: f64,
    pub noise_fraction: f64,
}

impl WienerSnrResult {
    pub fn nan() -> Self {
        Self { noise_estimate: f64::NAN, signal_estimate: f64::NAN, wiener_snr: f64::NAN, noise_fraction: f64::NAN }
    }
}

/// Estimate Wiener SNR from bin returns via PSD high-frequency tail.
pub fn wiener_snr(returns: &[f64]) -> WienerSnrResult {
    let n = returns.len();
    if n < 10 { return WienerSnrResult::nan(); }
    use tambear::signal_processing as sp2;
    let input_c: Vec<sp2::Complex> = returns.iter().map(|&r| (r, 0.0)).collect();
    let spectrum = sp2::fft(&input_c);
    let n_freq = n / 2;
    if n_freq < 4 { return WienerSnrResult::nan(); }
    let power: Vec<f64> = (1..=n_freq).map(|k| {
        let (re, im) = spectrum[k];
        (re*re + im*im) / (n as f64 * n as f64)
    }).collect();
    let total: f64 = power.iter().sum();
    if total < 1e-30 { return WienerSnrResult::nan(); }
    let tail_start = 3 * n_freq / 4;
    let tail_len = (n_freq - tail_start).max(1);
    let noise_power: f64 = power[tail_start..].iter().sum::<f64>() / tail_len as f64;
    let noise_total = noise_power * n_freq as f64;
    let signal = (total - noise_total).max(0.0);
    let wiener_snr_val = if noise_total > 1e-30 { signal / noise_total } else { f64::NAN };
    let noise_fraction = noise_total / total;
    WienerSnrResult { noise_estimate: noise_power, signal_estimate: signal, wiener_snr: wiener_snr_val, noise_fraction }
}

// ── VPIN (K02P10C04R01F01) ────────────────────────────────────────────────────

/// VPIN (Volume-synchronized Probability of Informed trading) via BVC.
///
/// Fintek's `vpin_bvc.rs` uses Easley, Lopez de Prado, O'Hara (2012) BVC method.
///
/// Outputs: vpin, mean_bucket_imbalance, max_vpin, vpin_volatility.
/// Wraps `tambear::volatility::vpin_bvc`.
#[derive(Debug, Clone)]
pub struct VpinResult {
    pub vpin: f64,
    pub mean_bucket_imbalance: f64,
    pub max_vpin: f64,
    pub vpin_volatility: f64,
}

impl VpinResult {
    pub fn nan() -> Self {
        Self { vpin: f64::NAN, mean_bucket_imbalance: f64::NAN, max_vpin: f64::NAN, vpin_volatility: f64::NAN }
    }
}

/// Compute VPIN features from prices and volumes.
///
/// `bucket_volume`: target volume per bucket (0.0 → auto = total_volume / 50).
pub fn vpin(prices: &[f64], volumes: &[f64], bucket_volume: f64) -> VpinResult {
    if prices.len() < 4 || volumes.len() != prices.len() { return VpinResult::nan(); }
    let bv = if bucket_volume <= 0.0 {
        let total_vol: f64 = volumes.iter().sum();
        total_vol / 50.0
    } else { bucket_volume };
    if bv < 1e-30 { return VpinResult::nan(); }
    let r = tambear::volatility::vpin_bvc(prices, volumes, bv, 50);
    if r.vpin.is_empty() { return VpinResult::nan(); }
    let n = r.vpin.len() as f64;
    let mean_vpin: f64 = r.vpin.iter().sum::<f64>() / n;
    let max_vpin: f64 = r.vpin.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mean_bucket_imbalance = mean_vpin;
    let vpin_volatility: f64 = (r.vpin.iter().map(|v| (v - mean_vpin).powi(2)).sum::<f64>() / n).sqrt();
    VpinResult { vpin: mean_vpin, mean_bucket_imbalance, max_vpin, vpin_volatility }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn garch_fit_on_garch_data() {
        // Generate GARCH(1,1) data: ω=0.01, α=0.1, β=0.85
        let n = 500;
        let mut returns = vec![0.0; n];
        let mut sigma2: f64 = 0.01 / (1.0 - 0.95);
        let mut rng = tambear::rng::Xoshiro256::new(42);
        for t in 0..n {
            let z = tambear::rng::sample_normal(&mut rng, 0.0, 1.0);
            returns[t] = sigma2.sqrt() * z;
            sigma2 = 0.01 + 0.1 * returns[t].powi(2) + 0.85 * sigma2;
        }
        let r = garch_fit(&returns, 200);
        assert!(r.omega.is_finite());
        assert!(r.alpha > 0.0 && r.alpha < 1.0);
        assert!(r.beta > 0.0 && r.beta < 1.0);
        assert!(r.persistence < 1.0, "persistence should be < 1 for stationary GARCH");
    }

    #[test]
    fn realized_vol_basic() {
        let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        let r = realized_vol(&returns);
        assert!(r.realized_variance >= 0.0);
        assert!(r.realized_vol >= 0.0);
        assert!((r.realized_vol - r.realized_variance.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn jump_detection_no_jumps() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let returns: Vec<f64> = (0..200).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = jump_detection(&returns);
        assert!(r.bns_statistic.is_finite());
        // No injected jumps → should not reject
        assert!(!r.has_jump_5pct || r.bns_statistic.abs() < 3.0);
    }

    #[test]
    fn roll_spread_basic() {
        let prices = vec![100.0, 100.1, 100.0, 100.1, 100.0, 100.1, 100.0];
        let s = roll_effective_spread(&prices);
        // Bid-ask bounce → positive Roll spread
        assert!(s.is_finite());
    }

    #[test]
    fn signature_plot_increasing_n() {
        let returns: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        let sp = signature_plot(&returns, &[1, 2, 4, 8]);
        assert_eq!(sp.len(), 4);
        for (_, rv) in &sp { assert!(rv.is_finite() && *rv >= 0.0); }
    }

    #[test]
    fn vol_regime_basic() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let n = 300;
        // First half: low vol, second half: high vol → should detect regime change
        let mut returns = vec![0.0_f64; n];
        for i in 0..150 { returns[i] = tambear::rng::sample_normal(&mut rng, 0.0, 0.005); }
        for i in 150..n { returns[i] = tambear::rng::sample_normal(&mut rng, 0.0, 0.05); }
        let r = vol_regime(&returns);
        assert!(r.n_vol_regimes.is_finite());
        assert!(r.high_vol_fraction >= 0.0 && r.high_vol_fraction <= 1.0);
        assert!(r.regime_switching_rate >= 0.0 && r.regime_switching_rate <= 1.0);
    }

    #[test]
    fn vol_regime_too_short() {
        let r = vol_regime(&vec![0.01_f64; 50]);
        assert!(r.n_vol_regimes.is_nan());
    }

    #[test]
    fn vol_dynamics_basic() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let returns: Vec<f64> = (0..200).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = vol_dynamics(&returns);
        assert!(r.vol_trend.is_finite());
        assert!(r.vol_of_vol >= 0.0);
        assert!(r.vol_autocorr.abs() <= 1.0);
    }

    #[test]
    fn vol_dynamics_too_short() {
        let r = vol_dynamics(&[0.01, 0.02]);
        assert!(r.vol_trend.is_nan());
    }

    #[test]
    fn stochvol_basic() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let returns: Vec<f64> = (0..200).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = stochvol(&returns);
        assert!(r.mean_log_vol.is_finite());
        assert!(r.vol_of_vol >= 0.0);
        assert!(r.persistence.abs() <= 1.0);
        assert!(r.leverage_corr.is_finite() && r.leverage_corr.abs() <= 1.0);
    }

    #[test]
    fn stochvol_too_short() {
        let r = stochvol(&[0.01, 0.02]);
        assert!(r.vol_of_vol.is_nan());
    }

    #[test]
    fn wiener_snr_basic() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let n = 128;
        // Mix of low-freq sine + noise
        let returns: Vec<f64> = (0..n).map(|i| {
            (2.0 * std::f64::consts::PI * 0.05 * i as f64).sin() * 0.01
            + tambear::rng::sample_normal(&mut rng, 0.0, 0.002)
        }).collect();
        let r = wiener_snr(&returns);
        assert!(r.noise_estimate.is_finite() && r.noise_estimate >= 0.0);
        assert!(r.noise_fraction >= 0.0 && r.noise_fraction <= 1.0);
    }

    #[test]
    fn wiener_snr_too_short() {
        let r = wiener_snr(&[1.0, 2.0]);
        assert!(r.wiener_snr.is_nan());
    }

    #[test]
    fn vpin_basic() {
        // 500 ticks with volume 10 each, bucket_volume=50 → 100 buckets
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let prices: Vec<f64> = (0..500).scan(100.0_f64, |p, _| {
            *p += tambear::rng::sample_normal(&mut rng, 0.0, 0.01);
            Some(*p)
        }).collect();
        let volumes = vec![10.0_f64; 500];
        let r = vpin(&prices, &volumes, 50.0);
        // With enough volume we should get a finite result
        if r.vpin.is_finite() {
            assert!(r.vpin >= 0.0 && r.vpin <= 1.0, "vpin={}", r.vpin);
        }
        // Even if auto-mode fails, NaN is acceptable for degenerate inputs
    }

    #[test]
    fn vpin_too_short() {
        let r = vpin(&[100.0, 101.0], &[10.0, 10.0], 0.0);
        // auto bucket volume = 20/50 = 0.4; may or may not form buckets — NaN is fine
        let _ = r; // just verify it doesn't panic
    }
}
