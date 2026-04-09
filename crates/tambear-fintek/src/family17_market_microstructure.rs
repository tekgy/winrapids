//! Family 17 — Market microstructure and regime leaves.
//!
//! Covers fintek leaves: `ou_process`, `kalman`, `variability`, `vol_regime`,
//! `vpin_bvc`, `returns` (multi-feature result).
//!
//! All functions are pure: `&[f64]` slices in, named result structs out.
//! No dependency on fintek's Leaf trait or ExecutionContext.

// ── OU process ───────────────────────────────────────────────────────────────

/// Ornstein-Uhlenbeck process parameter estimates for a single bin.
///
/// Fintek's `ou_process.rs` (K02P11C01R01) outputs:
///   DO01: theta — mean reversion speed
///   DO02: mu — long-run equilibrium
///   DO03: sigma — diffusion volatility
///   DO04: half_life — ln(2) / theta (in ticks)
///   DO05: r_squared — OLS fit quality
#[derive(Debug, Clone)]
pub struct OuProcessResult {
    pub theta: f64,
    pub mu: f64,
    pub sigma: f64,
    pub half_life: f64,
    pub r_squared: f64,
}

impl OuProcessResult {
    pub fn nan() -> Self {
        Self { theta: f64::NAN, mu: f64::NAN, sigma: f64::NAN,
               half_life: f64::NAN, r_squared: f64::NAN }
    }
}

/// Fit Ornstein-Uhlenbeck parameters to a price series via OLS regression
/// of Δx_t on x_t: Δx = a + b·x + ε → θ = -b, μ = -a/b.
///
/// `prices`: raw bin-level price ticks (at least 5).
pub fn ou_process(prices: &[f64]) -> OuProcessResult {
    let n = prices.len();
    if n < 5 { return OuProcessResult::nan(); }

    let m = n - 1; // pairs (x_t, Δx_t)
    let mut sum_x = 0.0f64;
    let mut sum_dx = 0.0f64;
    let mut sum_x2 = 0.0f64;
    let mut sum_xdx = 0.0f64;

    for i in 0..m {
        let x = prices[i];
        let dx = prices[i + 1] - prices[i];
        sum_x += x;
        sum_dx += dx;
        sum_x2 += x * x;
        sum_xdx += x * dx;
    }

    let mf = m as f64;
    let mean_x = sum_x / mf;
    let mean_dx = sum_dx / mf;

    let cov_xdx = sum_xdx / mf - mean_x * mean_dx;
    let var_x = sum_x2 / mf - mean_x * mean_x;

    if var_x < 1e-30 { return OuProcessResult::nan(); }

    let b = cov_xdx / var_x;
    let a = mean_dx - b * mean_x;

    let theta = -b;
    let mu = if b.abs() < 1e-15 {
        prices.iter().sum::<f64>() / n as f64
    } else {
        -a / b
    };

    // Residuals for sigma and R²
    let mut ss_res = 0.0f64;
    let mut ss_tot = 0.0f64;
    for i in 0..m {
        let predicted = a + b * prices[i];
        let dx = prices[i + 1] - prices[i];
        ss_res += (dx - predicted) * (dx - predicted);
        ss_tot += (dx - mean_dx) * (dx - mean_dx);
    }

    let sigma = if m > 1 { (ss_res / (m - 1) as f64).sqrt() } else { f64::NAN };
    let r_squared = if ss_tot > 1e-300 { 1.0 - ss_res / ss_tot } else { f64::NAN };
    let half_life = if theta > 1e-300 { std::f64::consts::LN_2 / theta } else { f64::INFINITY };

    OuProcessResult { theta, mu, sigma, half_life, r_squared }
}

// ── Kalman filter ─────────────────────────────────────────────────────────────

/// Kalman filter diagnostics for a single bin.
///
/// Fintek's `kalman.rs` (K02P05C04R01) outputs:
///   DO01: innovation_var — variance of Kalman innovations
///   DO02: gain_mean — average Kalman gain
///   DO03: smoothing_improvement — smoother MSE / filter MSE
///   DO04: prediction_error — one-step prediction error variance (filter MSE)
#[derive(Debug, Clone)]
pub struct KalmanResult {
    pub innovation_var: f64,
    pub gain_mean: f64,
    pub smoothing_improvement: f64,
    pub prediction_error: f64,
}

impl KalmanResult {
    pub fn nan() -> Self {
        Self { innovation_var: f64::NAN, gain_mean: f64::NAN,
               smoothing_improvement: f64::NAN, prediction_error: f64::NAN }
    }
}

/// Scalar Kalman filter + RTS smoother diagnostics on log-returns.
///
/// Noise parameters estimated from data:
///   R (observation noise) = var(returns) / 2
///   Q (state noise) = max(var(Δreturns) / 2 - R, 1e-30)
///
/// Returns innovation variance, mean gain, smoother/filter MSE ratio, filter MSE.
/// Input: bin-level price series (at least 20 ticks for meaningful estimates).
pub fn kalman(prices: &[f64]) -> KalmanResult {
    const MIN_RETURNS: usize = 20;

    if prices.len() < MIN_RETURNS + 2 { return KalmanResult::nan(); }

    // Log returns
    let returns: Vec<f64> = prices.windows(2).map(|w| {
        if w[0] > 0.0 && w[1] > 0.0 { (w[1] / w[0]).ln() } else { 0.0 }
    }).collect();

    let n = returns.len();
    if n < MIN_RETURNS { return KalmanResult::nan(); }

    // Noise parameter estimation
    let m_ret: f64 = returns.iter().sum::<f64>() / n as f64;
    let var_ret: f64 = returns.iter().map(|r| (r - m_ret) * (r - m_ret)).sum::<f64>() / n as f64;
    if var_ret < 1e-30 { return KalmanResult::nan(); }

    let diff_ret: Vec<f64> = returns.windows(2).map(|w| w[1] - w[0]).collect();
    let m_diff: f64 = diff_ret.iter().sum::<f64>() / diff_ret.len() as f64;
    let var_diff: f64 = diff_ret.iter().map(|d| (d - m_diff) * (d - m_diff)).sum::<f64>() / diff_ret.len() as f64;

    let r_noise = var_ret / 2.0;
    let q_state = (var_diff / 2.0 - r_noise).max(1e-30);

    // Forward Kalman filter
    let mut x_filt = vec![0.0f64; n];
    let mut p_filt = vec![0.0f64; n];
    let mut gains = vec![0.0f64; n];
    let mut innovations = vec![0.0f64; n];

    x_filt[0] = returns[0];
    p_filt[0] = r_noise;

    for t in 1..n {
        let p_pred = p_filt[t - 1] + q_state;
        let innov = returns[t] - x_filt[t - 1];
        let s = p_pred + r_noise;
        let k = if s > 1e-30 { p_pred / s } else { 0.0 };
        x_filt[t] = x_filt[t - 1] + k * innov;
        p_filt[t] = (1.0 - k) * p_pred;
        gains[t] = k;
        innovations[t] = innov;
    }

    let innov_mean: f64 = innovations[1..].iter().sum::<f64>() / (n - 1) as f64;
    let innov_var: f64 = innovations[1..].iter()
        .map(|i| (i - innov_mean) * (i - innov_mean)).sum::<f64>() / (n - 1) as f64;
    let gain_mean: f64 = gains[1..].iter().sum::<f64>() / (n - 1) as f64;

    let filter_mse: f64 = returns.iter().zip(x_filt.iter())
        .map(|(&r, &f)| (r - f) * (r - f)).sum::<f64>() / n as f64;

    // RTS backward smoother
    let mut x_smooth = x_filt.clone();
    for t in (0..n - 1).rev() {
        let p_pred = p_filt[t] + q_state;
        let l = if p_pred > 1e-30 { p_filt[t] / p_pred } else { 0.0 };
        x_smooth[t] = x_filt[t] + l * (x_smooth[t + 1] - x_filt[t]);
    }

    let smoother_mse: f64 = returns.iter().zip(x_smooth.iter())
        .map(|(&r, &s)| (r - s) * (r - s)).sum::<f64>() / n as f64;

    let smoothing_improvement = if filter_mse > 1e-30 { smoother_mse / filter_mse } else { f64::NAN };

    KalmanResult { innovation_var: innov_var, gain_mean, smoothing_improvement, prediction_error: filter_mse }
}

// ── Variability index ─────────────────────────────────────────────────────────

/// Rolling feature variability diagnostics.
///
/// Fintek's `variability.rs` (K02P19C5R1) outputs:
///   DO01: rolling_var_cv — CV of rolling variance
///   DO02: rolling_mean_cv — CV of rolling mean
///   DO03: range_variation — CV of rolling range
///   DO04: stability_index — 1 - max(var_cv, mean_cv)
#[derive(Debug, Clone)]
pub struct VariabilityResult {
    pub rolling_var_cv: f64,
    pub rolling_mean_cv: f64,
    pub range_variation: f64,
    pub stability_index: f64,
}

impl VariabilityResult {
    pub fn nan() -> Self {
        Self { rolling_var_cv: f64::NAN, rolling_mean_cv: f64::NAN,
               range_variation: f64::NAN, stability_index: f64::NAN }
    }
}

/// Coefficient of variation of a slice. Uses population std / |mean|.
/// Falls back to std / RMS when mean is near zero.
fn cv(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 { return f64::NAN; }
    let mean: f64 = values.iter().sum::<f64>() / n as f64;
    let std: f64 = (values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / n as f64).sqrt();
    if mean.abs() > 1e-30 {
        std / mean.abs()
    } else {
        let rms: f64 = (values.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
        if rms > 1e-30 { std / rms } else { 0.0 }
    }
}

/// Rolling feature variability from log-returns.
///
/// `returns`: log-returns (at least 20 required).
/// Window = max(5, n/10). Returns NaN if series is too short.
pub fn variability(returns: &[f64]) -> VariabilityResult {
    const MIN_RETURNS: usize = 20;
    let n = returns.len();
    if n < MIN_RETURNS { return VariabilityResult::nan(); }

    let w = 5usize.max(n / 10);
    if w >= n { return VariabilityResult::nan(); }
    let n_windows = n - w + 1;

    let mut rolling_means = Vec::with_capacity(n_windows);
    let mut rolling_vars = Vec::with_capacity(n_windows);
    let mut rolling_ranges = Vec::with_capacity(n_windows);

    for j in 0..n_windows {
        let win = &returns[j..j + w];
        let m: f64 = win.iter().sum::<f64>() / w as f64;
        let v: f64 = win.iter().map(|r| (r - m) * (r - m)).sum::<f64>() / w as f64;
        let min = win.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = win.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        rolling_means.push(m);
        rolling_vars.push(v);
        rolling_ranges.push(max - min);
    }

    let var_cv = cv(&rolling_vars);
    let mean_cv = cv(&rolling_means);
    let range_cv = cv(&rolling_ranges);
    let stability = if var_cv.is_finite() && mean_cv.is_finite() {
        (1.0 - var_cv.max(mean_cv)).max(0.0).min(1.0)
    } else { f64::NAN };

    VariabilityResult {
        rolling_var_cv: var_cv,
        rolling_mean_cv: mean_cv,
        range_variation: range_cv,
        stability_index: stability,
    }
}

// ── Volatility regime ─────────────────────────────────────────────────────────

/// Volatility regime classification result.
///
/// Fintek's `vol_regime.rs` (K02P18C03R01F02) outputs:
///   DO01: n_vol_regimes — number of distinct volatility regimes detected
///   DO02: vol_ratio_range — range of short/long variance ratio
///   DO03: high_vol_fraction — fraction of time in high-vol regime
///   DO04: regime_switching_rate — fraction of consecutive-window transitions
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

/// Compute rolling variance using prefix sums. Returns values starting at index window-1.
fn rolling_variance_prefix(returns: &[f64], window: usize) -> Vec<f64> {
    let n = returns.len();
    if n < window { return Vec::new(); }
    let mut cum = vec![0.0f64; n + 1];
    let mut cum2 = vec![0.0f64; n + 1];
    for i in 0..n {
        cum[i + 1] = cum[i] + returns[i];
        cum2[i + 1] = cum2[i] + returns[i] * returns[i];
    }
    let wf = window as f64;
    (0..=(n - window)).map(|i| {
        let s = cum[i + window] - cum[i];
        let s2 = cum2[i + window] - cum2[i];
        let mean = s / wf;
        (s2 / wf - mean * mean).max(0.0)
    }).collect()
}

/// Volatility regime features from log-returns.
///
/// `returns`: log-returns (at least 120 required for reliable short+long windows).
/// Short window = 20, long window = 100.
pub fn vol_regime(returns: &[f64]) -> VolRegimeResult {
    const SHORT: usize = 20;
    const LONG: usize = 100;
    const MIN_RETURNS: usize = 120;
    const HIGH_VOL_THRESHOLD: f64 = 1.5;

    if returns.len() < MIN_RETURNS { return VolRegimeResult::nan(); }

    let short_var = rolling_variance_prefix(returns, SHORT);
    let long_var  = rolling_variance_prefix(returns, LONG);

    // Align: long starts at index LONG-1, short at SHORT-1.
    // Offset so we can index them together for the same point in returns.
    let offset = LONG - SHORT;
    let n_aligned = long_var.len().min(short_var.len().saturating_sub(offset));
    if n_aligned == 0 { return VolRegimeResult::nan(); }

    // Variance ratio (short / long). Classify into regimes.
    let mut ratios = Vec::with_capacity(n_aligned);
    for i in 0..n_aligned {
        let lv = long_var[i];
        let sv = short_var[i + offset];
        let ratio = if lv > 1e-30 { sv / lv } else { 1.0 };
        ratios.push(ratio);
    }

    // High-vol = ratio > HIGH_VOL_THRESHOLD
    let high_vol_count = ratios.iter().filter(|&&r| r > HIGH_VOL_THRESHOLD).count();
    let high_vol_frac = high_vol_count as f64 / n_aligned as f64;

    // Count regime transitions (consecutive windows with different classification)
    let mut transitions = 0usize;
    let mut prev_high = ratios[0] > HIGH_VOL_THRESHOLD;
    for &r in &ratios[1..] {
        let is_high = r > HIGH_VOL_THRESHOLD;
        if is_high != prev_high { transitions += 1; }
        prev_high = is_high;
    }
    let switching_rate = transitions as f64 / (n_aligned - 1) as f64;

    // Count distinct regimes by quantizing ratio into terciles
    let mut sorted_ratios = ratios.clone();
    sorted_ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q33 = sorted_ratios[n_aligned / 3];
    let q67 = sorted_ratios[2 * n_aligned / 3];

    let mut regime_labels: Vec<u8> = ratios.iter().map(|&r| {
        if r < q33 { 0 } else if r < q67 { 1 } else { 2 }
    }).collect();
    let n_regimes = {
        regime_labels.sort();
        regime_labels.dedup();
        regime_labels.len()
    };

    let ratio_range = sorted_ratios.last().unwrap_or(&0.0) - sorted_ratios.first().unwrap_or(&0.0);

    VolRegimeResult {
        n_vol_regimes: n_regimes as f64,
        vol_ratio_range: ratio_range,
        high_vol_fraction: high_vol_frac,
        regime_switching_rate: switching_rate,
    }
}

// ── VPIN via Bulk Volume Classification ──────────────────────────────────────

/// VPIN (Volume-synchronized Probability of Informed Trading) features.
///
/// Fintek's `vpin_bvc.rs` (K02P10C04R01F01) outputs:
///   DO01: vpin — mean order imbalance across volume buckets
///   DO02: mean_bucket_imbalance — mean |buy_vol - sell_vol| / bucket_vol
///   DO03: max_vpin — maximum rolling VPIN
///   DO04: vpin_volatility — std dev of bucket imbalances
#[derive(Debug, Clone)]
pub struct VpinResult {
    pub vpin: f64,
    pub mean_bucket_imbalance: f64,
    pub max_vpin: f64,
    pub vpin_volatility: f64,
}

impl VpinResult {
    pub fn nan() -> Self {
        Self { vpin: f64::NAN, mean_bucket_imbalance: f64::NAN,
               max_vpin: f64::NAN, vpin_volatility: f64::NAN }
    }
}

fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + libm_erf(x / std::f64::consts::SQRT_2))
}

fn libm_erf(x: f64) -> f64 {
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let ax = x.abs();
    const P: f64 = 0.3275911;
    const A1: f64 = 0.254829592;
    const A2: f64 = -0.284496736;
    const A3: f64 = 1.421413741;
    const A4: f64 = -1.453152027;
    const A5: f64 = 1.061405429;
    let t = 1.0 / (1.0 + P * ax);
    let t2 = t * t; let t3 = t2 * t; let t4 = t3 * t; let t5 = t4 * t;
    let y = 1.0 - (A1*t + A2*t2 + A3*t3 + A4*t4 + A5*t5) * (-ax * ax).exp();
    sign * y
}

/// VPIN via Bulk Volume Classification.
///
/// `prices`: tick-level prices.
/// `sizes`: tick-level sizes (volumes).
/// Uses 20 equal-volume buckets.
pub fn vpin_bvc(prices: &[f64], sizes: &[f64]) -> VpinResult {
    assert_eq!(prices.len(), sizes.len(), "prices and sizes must match");
    const N_BUCKETS: usize = 20;

    let n = prices.len();
    if n < 4 { return VpinResult::nan(); }

    // Log price changes
    let dp: Vec<f64> = (1..n).map(|i| {
        if prices[i] > 0.0 && prices[i-1] > 0.0 { (prices[i] / prices[i-1]).ln() }
        else { 0.0 }
    }).collect();

    let m_dp: f64 = dp.iter().sum::<f64>() / dp.len() as f64;
    let var_dp: f64 = dp.iter().map(|&d| (d - m_dp) * (d - m_dp)).sum::<f64>()
        / (dp.len().saturating_sub(1)).max(1) as f64;
    let std_dp = var_dp.sqrt();
    if std_dp < 1e-15 { return VpinResult::nan(); }

    // BVC: classify each trade's volume as buy/sell
    let trade_vol: Vec<f64> = sizes[1..].iter().map(|&s| s.max(1e-10)).collect();
    let total_vol: f64 = trade_vol.iter().sum();
    if total_vol < 1e-10 { return VpinResult::nan(); }

    let bucket_vol = total_vol / N_BUCKETS as f64;

    let buy_frac: Vec<f64> = dp.iter().map(|&d| standard_normal_cdf(d / std_dp)).collect();

    // Fill equal-volume buckets
    let mut bucket_imbalances = Vec::new();
    let mut cum_vol = 0.0f64;
    let mut bucket_buy = 0.0f64;
    let mut bucket_sell = 0.0f64;
    let mut bucket_total = 0.0f64;

    for i in 0..dp.len() {
        let v = trade_vol[i];
        let bfrac = buy_frac[i];
        let buy_v = v * bfrac;
        let sell_v = v * (1.0 - bfrac);

        bucket_buy += buy_v;
        bucket_sell += sell_v;
        bucket_total += v;
        cum_vol += v;

        if cum_vol >= bucket_vol || i == dp.len() - 1 {
            if bucket_total > 1e-10 {
                bucket_imbalances.push((bucket_buy - bucket_sell).abs() / bucket_total);
            }
            cum_vol = 0.0;
            bucket_buy = 0.0;
            bucket_sell = 0.0;
            bucket_total = 0.0;
        }
    }

    if bucket_imbalances.is_empty() { return VpinResult::nan(); }

    let nb = bucket_imbalances.len() as f64;
    let vpin_mean = bucket_imbalances.iter().sum::<f64>() / nb;
    let max_vpin = bucket_imbalances.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let var_imb = bucket_imbalances.iter()
        .map(|&x| (x - vpin_mean) * (x - vpin_mean)).sum::<f64>() / nb;
    let vpin_vol = var_imb.sqrt();

    VpinResult {
        vpin: vpin_mean,
        mean_bucket_imbalance: vpin_mean,
        max_vpin,
        vpin_volatility: vpin_vol,
    }
}

// ── Returns multi-feature ─────────────────────────────────────────────────────

/// Multi-feature return summary for a single bin.
///
/// Fintek's `returns.rs` (K02P01C05) emits 6 features using cross-bin
/// context (prev_close). This pure function handles the within-bin case
/// where prev_close is provided as a parameter.
///
///   open_return: first price / prev_close - 1
///   close_return: last price / first price - 1
///   high_low_range: (high - low) / first_price
///   log_return: ln(last / first)
///   abs_return: |last - first| / first
///   signed_volume: uptick vol - downtick vol (requires sizes)
#[derive(Debug, Clone)]
pub struct ReturnsResult {
    pub open_return: f64,
    pub close_return: f64,
    pub high_low_range: f64,
    pub log_return: f64,
    pub abs_return: f64,
    pub signed_volume: f64,
}

impl ReturnsResult {
    pub fn nan() -> Self {
        Self { open_return: f64::NAN, close_return: f64::NAN, high_low_range: f64::NAN,
               log_return: f64::NAN, abs_return: f64::NAN, signed_volume: f64::NAN }
    }
}

/// Compute 6 return features for a single bin.
///
/// `prices`: tick-level prices in the bin (at least 2).
/// `sizes`: tick-level sizes (for signed_volume). Pass empty slice to get NaN for that field.
/// `prev_close`: previous bin's closing price (for open_return). Pass NaN if unavailable.
pub fn returns_features(prices: &[f64], sizes: &[f64], prev_close: f64) -> ReturnsResult {
    if prices.len() < 2 { return ReturnsResult::nan(); }

    let open = prices[0];
    let close = *prices.last().unwrap();

    if open <= 0.0 || close <= 0.0 { return ReturnsResult::nan(); }

    let high = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let low = prices.iter().cloned().fold(f64::INFINITY, f64::min);

    let open_return = if prev_close > 0.0 { open / prev_close - 1.0 } else { f64::NAN };
    let close_return = close / open - 1.0;
    let high_low_range = (high - low) / open;
    let log_return = (close / open).ln();
    let abs_return = (close - open).abs() / open;

    let signed_volume = if sizes.len() >= prices.len() && prices.len() >= 2 {
        let mut sv = 0.0f64;
        for i in 1..prices.len() {
            let dir = if prices[i] > prices[i - 1] { 1.0 }
                      else if prices[i] < prices[i - 1] { -1.0 }
                      else { 0.0 };
            sv += dir * sizes[i];
        }
        sv
    } else { f64::NAN };

    ReturnsResult { open_return, close_return, high_low_range, log_return, abs_return, signed_volume }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── OU process tests ────────────────────────────────────────────────────

    #[test]
    fn ou_process_mean_reverting() {
        // Simulate OU: dx = θ(μ - x)dt + σdW, θ=0.5, μ=100, σ=0.5
        let mut prices = vec![100.0f64; 200];
        let mut rng = tambear::rng::Xoshiro256::new(42);
        for t in 1..200 {
            let x = prices[t - 1];
            prices[t] = x + 0.5 * (100.0 - x) + tambear::rng::sample_normal(&mut rng, 0.0, 0.5);
        }
        let r = ou_process(&prices);
        assert!(r.theta.is_finite() && r.theta > 0.0, "theta={} should be > 0 for mean-reverting", r.theta);
        assert!((r.mu - 100.0).abs() < 10.0, "mu={} should be near 100", r.mu);
        assert!(r.half_life.is_finite() && r.half_life > 0.0);
    }

    #[test]
    fn ou_process_random_walk_near_zero_theta() {
        // Random walk: dx = 0 (θ ≈ 0)
        let mut prices = vec![100.0f64; 200];
        let mut rng = tambear::rng::Xoshiro256::new(7);
        for t in 1..200 {
            prices[t] = prices[t - 1] + tambear::rng::sample_normal(&mut rng, 0.0, 1.0);
        }
        let r = ou_process(&prices);
        // Random walk: theta can be negative or near zero
        assert!(r.theta.is_finite());
    }

    #[test]
    fn ou_process_too_short() {
        let r = ou_process(&[100.0, 101.0, 102.0]);
        assert!(r.theta.is_nan());
    }

    // ── Kalman filter tests ─────────────────────────────────────────────────

    #[test]
    fn kalman_basic_outputs() {
        let mut prices = vec![100.0f64; 100];
        let mut rng = tambear::rng::Xoshiro256::new(1);
        for t in 1..100 {
            prices[t] = prices[t-1] * (1.0 + tambear::rng::sample_normal(&mut rng, 0.0, 0.01));
        }
        let r = kalman(&prices);
        assert!(r.innovation_var >= 0.0, "innovation_var={}", r.innovation_var);
        assert!(r.gain_mean > 0.0 && r.gain_mean < 1.0, "gain_mean={}", r.gain_mean);
        assert!(r.smoothing_improvement.is_finite());
        assert!(r.prediction_error >= 0.0);
    }

    #[test]
    fn kalman_too_short() {
        let r = kalman(&vec![100.0; 10]);
        assert!(r.innovation_var.is_nan());
    }

    #[test]
    fn kalman_smoothing_improvement_finite() {
        // smoothing_improvement = smoother_mse / filter_mse.
        // With estimated noise params (not oracle), this ratio can exceed 1
        // in finite samples — the RTS guarantee only holds under the true model.
        // We just assert the output is a finite positive scalar.
        let mut prices = vec![100.0f64; 200];
        let mut rng = tambear::rng::Xoshiro256::new(42);
        for t in 1..200 {
            prices[t] = prices[t-1] * (1.0 + tambear::rng::sample_normal(&mut rng, 0.0, 0.01));
        }
        let r = kalman(&prices);
        assert!(r.smoothing_improvement.is_finite() && r.smoothing_improvement > 0.0,
            "smoothing_improvement={} should be finite and positive", r.smoothing_improvement);
    }

    // ── Variability tests ───────────────────────────────────────────────────

    #[test]
    fn variability_constant_returns_low_cv() {
        // Constant returns → rolling variance = 0 → CV is degenerate (0 or NaN)
        let returns = vec![0.01f64; 100];
        let r = variability(&returns);
        // CV of constants = 0
        assert!(r.rolling_mean_cv.is_finite() && r.rolling_mean_cv < 1e-10,
            "rolling_mean_cv={} should be ≈ 0 for constant returns", r.rolling_mean_cv);
        assert!(r.stability_index.is_finite());
    }

    #[test]
    fn variability_noisy_returns() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let returns: Vec<f64> = (0..100).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = variability(&returns);
        assert!(r.rolling_var_cv.is_finite());
        assert!(r.rolling_mean_cv.is_finite());
        assert!(r.stability_index >= 0.0 && r.stability_index <= 1.0,
            "stability_index={} out of [0,1]", r.stability_index);
    }

    #[test]
    fn variability_too_short() {
        let r = variability(&[0.01; 5]);
        assert!(r.rolling_var_cv.is_nan());
    }

    // ── Volatility regime tests ─────────────────────────────────────────────

    #[test]
    fn vol_regime_basic() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let returns: Vec<f64> = (0..200).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = vol_regime(&returns);
        assert!(r.n_vol_regimes.is_finite() && r.n_vol_regimes >= 1.0);
        assert!(r.high_vol_fraction >= 0.0 && r.high_vol_fraction <= 1.0);
        assert!(r.regime_switching_rate >= 0.0 && r.regime_switching_rate <= 1.0);
    }

    #[test]
    fn vol_regime_too_short() {
        let r = vol_regime(&vec![0.01; 50]);
        assert!(r.n_vol_regimes.is_nan());
    }

    // ── VPIN tests ──────────────────────────────────────────────────────────

    #[test]
    fn vpin_basic() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let n = 200;
        let mut prices = vec![100.0f64; n];
        let mut sizes = vec![1.0f64; n];
        for t in 1..n {
            prices[t] = prices[t-1] + tambear::rng::sample_normal(&mut rng, 0.0, 0.1);
            sizes[t] = 100.0 + tambear::rng::sample_normal(&mut rng, 0.0, 10.0).abs();
        }
        let r = vpin_bvc(&prices, &sizes);
        assert!(r.vpin >= 0.0 && r.vpin <= 1.0, "vpin={} out of [0,1]", r.vpin);
        assert!(r.max_vpin >= r.vpin - 1e-10);
        assert!(r.vpin_volatility >= 0.0);
    }

    #[test]
    fn vpin_too_short() {
        let r = vpin_bvc(&[100.0, 101.0], &[1.0, 1.0]);
        assert!(r.vpin.is_nan());
    }

    // ── Returns features tests ──────────────────────────────────────────────

    #[test]
    fn returns_features_basic() {
        let prices = vec![100.0, 101.0, 102.0, 101.5, 103.0];
        let sizes = vec![0.0, 10.0, 20.0, 5.0, 15.0];
        let r = returns_features(&prices, &sizes, 99.0);
        assert!((r.open_return - (100.0/99.0 - 1.0)).abs() < 1e-10);
        assert!((r.close_return - (103.0/100.0 - 1.0)).abs() < 1e-10);
        assert!((r.log_return - (103.0_f64/100.0).ln()).abs() < 1e-10);
        assert!(r.signed_volume.is_finite());
    }

    #[test]
    fn returns_features_no_prev_close() {
        let prices = vec![100.0, 101.0, 102.0];
        let r = returns_features(&prices, &[], f64::NAN);
        assert!(r.open_return.is_nan());
        assert!(r.close_return.is_finite());
        assert!(r.signed_volume.is_nan()); // no sizes
    }

    #[test]
    fn returns_features_too_short() {
        let r = returns_features(&[100.0], &[], f64::NAN);
        assert!(r.log_return.is_nan());
    }
}
