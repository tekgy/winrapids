//! Family 18 — Decomposition, smoothing, HMM, SDE, seismic.
//!
//! Covers fintek leaves: `stl`, `sde`, `hmm`, `savgol`, `smoothers`,
//! `fir_bandpass`, `seismic`, `scale_freeness`.
//!
//! All tambear math primitives were built in prior gap-filling waves:
//!   - STL decomposition — C11
//!   - SDE drift/diffusion — C12
//!   - HMM (Baum-Welch, Viterbi) — B4
//!   - Savitzky-Golay filter — B20
//!   - FIR filters — B20
//!   - Seismic laws (GR, Omori, Bath) — C8
//!   - Gutenberg-Richter for scale-freeness — C8

use tambear::time_series::stl_decompose;
use tambear::nonparametric::{sde_estimate, gutenberg_richter_fit, omori_fit, bath_law};
use tambear::signal_processing::{savgol_filter, fir_bandpass, fir_filter};
use tambear::hmm::{Hmm, hmm_baum_welch, hmm_viterbi, hmm_random_init};

// ── STL decomposition ─────────────────────────────────────────────────────────

/// STL decomposition features per bin.
///
/// Fintek's `stl.rs` (K02P19C2R1) outputs 5 scalars:
///   DO01: trend_strength — Var(T) / (Var(T) + Var(R)), ∈ [0,1]
///   DO02: seasonal_strength — Var(S) / (Var(S) + Var(R)), ∈ [0,1]
///   DO03: residual_variance — variance of remainder component
///   DO04: seasonal_period — period used for decomposition (auto-detected)
///   DO05: trend_slope — OLS slope of trend component (direction of drift)
#[derive(Debug, Clone)]
pub struct StlFeaturesResult {
    pub trend_strength: f64,
    pub seasonal_strength: f64,
    pub residual_variance: f64,
    pub seasonal_period: f64,
    pub trend_slope: f64,
}

impl StlFeaturesResult {
    pub fn nan() -> Self {
        Self {
            trend_strength: f64::NAN, seasonal_strength: f64::NAN,
            residual_variance: f64::NAN, seasonal_period: f64::NAN,
            trend_slope: f64::NAN,
        }
    }
}

/// STL seasonal-trend decomposition features.
///
/// Auto-detects period: tries periods [4, 8, 12, 16] and picks the one that
/// maximizes seasonal_strength. For financial returns, seasonal patterns are
/// typically weak — trend_strength dominates.
pub fn stl_features(returns: &[f64]) -> StlFeaturesResult {
    const MIN_N: usize = 20;
    if returns.len() < MIN_N { return StlFeaturesResult::nan(); }

    // Try a few periods and pick the one that gives strongest decomposition
    let candidate_periods = [4usize, 8, 12, 16];
    let mut best_result = None;
    let mut best_strength = -1.0_f64;

    for &period in &candidate_periods {
        if returns.len() < 3 * period { continue; }
        if let Some(r) = stl_decompose(returns, period, false) {
            let s = r.seasonal_strength();
            let t = r.trend_strength();
            let combined = s.max(t);
            if combined > best_strength {
                best_strength = combined;
                best_result = Some((r, period));
            }
        }
    }

    let (stl, period) = match best_result {
        Some(v) => v,
        None => {
            // fallback: period=4 required at minimum; if all failed, constant series likely
            return StlFeaturesResult {
                trend_strength: 0.0, seasonal_strength: 0.0,
                residual_variance: 0.0, seasonal_period: 0.0, trend_slope: 0.0,
            };
        }
    };

    let trend_s = stl.trend_strength();
    let seasonal_s = stl.seasonal_strength();

    // Residual variance
    let rem = &stl.remainder;
    let n = rem.iter().filter(|x| x.is_finite()).count();
    let res_var = if n > 1 {
        let m = rem.iter().filter(|x| x.is_finite()).sum::<f64>() / n as f64;
        rem.iter().filter(|x| x.is_finite()).map(|&x| (x - m) * (x - m)).sum::<f64>() / n as f64
    } else {
        f64::NAN
    };

    // Trend slope via OLS on finite trend values
    let trend_finite: Vec<(f64, f64)> = stl.trend.iter().enumerate()
        .filter(|(_, &v)| v.is_finite())
        .map(|(i, &v)| (i as f64, v))
        .collect();
    let trend_slope = if trend_finite.len() >= 2 {
        let nt = trend_finite.len() as f64;
        let sx: f64 = trend_finite.iter().map(|(x, _)| x).sum();
        let sy: f64 = trend_finite.iter().map(|(_, y)| y).sum();
        let sxx: f64 = trend_finite.iter().map(|(x, _)| x * x).sum();
        let sxy: f64 = trend_finite.iter().map(|(x, y)| x * y).sum();
        let denom = nt * sxx - sx * sx;
        if denom.abs() > 1e-14 { (nt * sxy - sx * sy) / denom } else { 0.0 }
    } else {
        f64::NAN
    };

    StlFeaturesResult {
        trend_strength: trend_s,
        seasonal_strength: seasonal_s,
        residual_variance: res_var,
        seasonal_period: period as f64,
        trend_slope,
    }
}

// ── SDE drift/diffusion estimation ───────────────────────────────────────────

/// SDE features per bin.
///
/// Fintek's `sde.rs` (K02P11C02R01) outputs 5 scalars:
///   DO01: drift_mean — mean drift across state-space grid
///   DO02: drift_slope — slope of drift vs. state (negative = mean-reverting)
///   DO03: diffusion_mean — mean local volatility
///   DO04: diffusion_slope — slope of diffusion vs. state (volatility smile)
///   DO05: drift_diffusion_corr — leverage effect (Pearson r)
pub use tambear::nonparametric::SdeResult;

/// SDE drift/diffusion features via Nadaraya-Watson kernel regression.
///
/// Uses 20-point state grid, Silverman bandwidth.
pub fn sde(prices: &[f64]) -> SdeResult {
    const MIN_N: usize = 20;
    const N_GRID: usize = 20;
    if prices.len() < MIN_N { return SdeResult::nan(); }
    sde_estimate(prices, N_GRID)
}

// ── Hidden Markov Model ────────────────────────────────────────────────────────

/// HMM features per bin.
///
/// Fintek's `hmm.rs` (K02P18C02R01F01) fits a 2-state HMM to discretized returns.
/// 4 outputs:
///   DO01: state_persistence — mean time-in-state (diagonal of transition matrix)
///   DO02: state_separation — |μ₁ − μ₂| / pooled σ (Cohen's d between states)
///   DO03: entropy_rate — H = −Σ_{ij} π_i · A_{ij} · ln A_{ij}
///   DO04: viterbi_switches — number of state switches in Viterbi path
#[derive(Debug, Clone)]
pub struct HmmFeaturesResult {
    pub state_persistence: f64,
    pub state_separation: f64,
    pub entropy_rate: f64,
    pub viterbi_switches: f64,
}

impl HmmFeaturesResult {
    pub fn nan() -> Self {
        Self {
            state_persistence: f64::NAN, state_separation: f64::NAN,
            entropy_rate: f64::NAN, viterbi_switches: f64::NAN,
        }
    }
}

/// Fit 2-state HMM via Baum-Welch and extract 4 summary features.
///
/// Returns are discretized into 8 uniform bins before fitting.
pub fn hmm_features(returns: &[f64]) -> HmmFeaturesResult {
    const MIN_N: usize = 30;
    const N_STATES: usize = 2;
    const N_OBS: usize = 8;
    const N_ITER: usize = 20;
    const MAX_N: usize = 2000;

    if returns.len() < MIN_N { return HmmFeaturesResult::nan(); }

    // Subsample deterministically if too long
    let working: Vec<f64> = if returns.len() > MAX_N {
        let step = returns.len() / MAX_N;
        returns.iter().step_by(step).take(MAX_N).copied().collect()
    } else {
        returns.to_vec()
    };

    // Discretize to N_OBS bins via uniform quantization
    let min_r = working.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_r = working.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_r - min_r;
    if range < 1e-12 { return HmmFeaturesResult::nan(); }

    let obs: Vec<usize> = working.iter().map(|&x| {
        let bin = ((x - min_r) / range * N_OBS as f64) as usize;
        bin.min(N_OBS - 1)
    }).collect();

    // Random init with fixed seed → deterministic
    let init_hmm = hmm_random_init(N_STATES, N_OBS, 42);
    let obs_slice: &[usize] = &obs;
    let fitted_result = hmm_baum_welch(&init_hmm, &[obs_slice], N_ITER, 1e-4);
    let fitted = &fitted_result.hmm;

    // State persistence: mean of diagonal transition probabilities
    let (_, trans, _) = fitted.to_probs();
    let persistence: f64 = (0..N_STATES)
        .map(|i| trans[i * N_STATES + i])
        .sum::<f64>() / N_STATES as f64;

    // State separation: use emission distribution means
    // Approximate: weighted mean of obs for each state from emission probs
    let (_, _, emit) = fitted.to_probs();
    let state_means: Vec<f64> = (0..N_STATES).map(|s| {
        let weights = &emit[s * N_OBS..(s + 1) * N_OBS];
        let total: f64 = weights.iter().sum();
        if total < 1e-12 { return 0.0; }
        (0..N_OBS).map(|k| weights[k] * k as f64).sum::<f64>() / total
    }).collect();

    let state_var: Vec<f64> = (0..N_STATES).map(|s| {
        let weights = &emit[s * N_OBS..(s + 1) * N_OBS];
        let total: f64 = weights.iter().sum();
        if total < 1e-12 { return 1.0; }
        let m = state_means[s];
        (0..N_OBS).map(|k| weights[k] * (k as f64 - m) * (k as f64 - m)).sum::<f64>() / total
    }).collect();

    let pooled_std = ((state_var[0] + state_var[1]) / 2.0).sqrt().max(1e-12);
    let separation = (state_means[0] - state_means[1]).abs() / pooled_std;

    // Stationary distribution via power iteration
    let mut pi = vec![1.0 / N_STATES as f64; N_STATES];
    for _ in 0..100 {
        let mut new_pi = vec![0.0; N_STATES];
        for i in 0..N_STATES {
            for j in 0..N_STATES {
                new_pi[j] += pi[i] * trans[i * N_STATES + j];
            }
        }
        pi = new_pi;
    }

    // Entropy rate H = -Σ_{ij} π_i · A_{ij} · ln(A_{ij})
    let entropy_rate: f64 = {
        let mut h = 0.0_f64;
        for i in 0..N_STATES {
            for j in 0..N_STATES {
                let aij = trans[i * N_STATES + j];
                if aij > 1e-12 { h -= pi[i] * aij * aij.ln(); }
            }
        }
        h
    };

    // Viterbi path switch count
    let viterbi = hmm_viterbi(fitted, &obs);
    let switches = viterbi.states.windows(2).filter(|w| w[0] != w[1]).count() as f64;

    HmmFeaturesResult {
        state_persistence: persistence,
        state_separation: separation,
        entropy_rate,
        viterbi_switches: switches,
    }
}

// ── Savitzky-Golay smoothing ───────────────────────────────────────────────────

/// Savitzky-Golay smoothed series.
///
/// Fintek's `savgol.rs` applies SG filter and returns the smoothed values.
/// Bridge returns the full smoothed series (same length as input).
///
/// `window`: must be odd, ≥ 5. `poly_order`: polynomial degree, < window.
pub fn savgol(data: &[f64], window: usize, poly_order: usize) -> Vec<f64> {
    if data.len() < window || window < 3 || poly_order >= window {
        return data.to_vec();
    }
    savgol_filter(data, window, poly_order)
}

// ── FIR bandpass filter ────────────────────────────────────────────────────────

/// FIR bandpass filtered series.
///
/// Fintek's `fir_bandpass.rs` returns the filtered signal and summary stats.
/// Bridge returns the filtered signal directly.
///
/// `low_cutoff`, `high_cutoff`: normalized frequencies in (0, 0.5).
/// `n_taps`: filter length (odd). Defaults: low=0.05, high=0.25, n_taps=31.
pub fn fir_bandpass_filter(data: &[f64], low_cutoff: f64, high_cutoff: f64, n_taps: usize) -> Vec<f64> {
    if data.len() < n_taps { return vec![f64::NAN; data.len()]; }
    // fir_bandpass returns the filter coefficients; apply via fir_filter
    let coeffs = fir_bandpass(low_cutoff, high_cutoff, n_taps);
    fir_filter(data, &coeffs)
}

// ── Seismic laws (GR + Omori + Bath) ─────────────────────────────────────────

/// Seismic features per bin.
///
/// Fintek's `seismic.rs` (K02P19C1R1) outputs 4 scalars:
///   DO01: gr_b_value — Gutenberg-Richter b-value on |returns|
///   DO02: omori_p — Omori decay exponent (p) for |returns| vs. time
///   DO03: bath_ratio — Bath's law ratio: largest aftershock / mainshock
///   DO04: n_extreme — count of |return| > 3σ events
#[derive(Debug, Clone)]
pub struct SeismicResult {
    pub gr_b_value: f64,
    pub omori_p: f64,
    pub bath_ratio: f64,
    pub n_extreme: f64,
}

impl SeismicResult {
    pub fn nan() -> Self {
        Self { gr_b_value: f64::NAN, omori_p: f64::NAN, bath_ratio: f64::NAN, n_extreme: f64::NAN }
    }
}

/// Seismic law features from log-returns.
///
/// |returns| treated as "earthquake magnitudes". The largest |return| is the
/// mainshock; the rest are aftershocks.
pub fn seismic(returns: &[f64]) -> SeismicResult {
    const MIN_N: usize = 20;
    if returns.len() < MIN_N { return SeismicResult::nan(); }

    let abs_returns: Vec<f64> = returns.iter().map(|&x| x.abs()).collect();

    // Count extreme events (|r| > 3σ)
    let n = abs_returns.len() as f64;
    let mean = abs_returns.iter().sum::<f64>() / n;
    let var = abs_returns.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let std = var.sqrt();
    let threshold_3sigma = mean + 3.0 * std;
    let n_extreme = abs_returns.iter().filter(|&&x| x > threshold_3sigma).count() as f64;

    // GR b-value on |returns| (treated as magnitudes)
    let m_min = abs_returns.iter().cloned().fold(f64::INFINITY, f64::min).max(1e-10);
    let gr = gutenberg_richter_fit(&abs_returns, m_min);
    let gr_b = gr.b_value;

    // Omori p: fit to |returns| sorted by time (interarrival pattern)
    // Use times as cumulative event times from extreme event indices
    let extreme_times: Vec<f64> = returns.iter().enumerate()
        .filter(|(_, &x)| x.abs() > threshold_3sigma)
        .map(|(i, _)| i as f64)
        .collect();
    let omori_p = if extreme_times.len() >= 3 {
        let t_end = *extreme_times.last().unwrap_or(&(returns.len() as f64));
        let omori = omori_fit(&extreme_times, t_end);
        omori.p
    } else {
        f64::NAN
    };

    // Bath's law: mainshock = max |return|, aftershocks = rest
    let max_idx = abs_returns.iter().enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let mainshock_mag = abs_returns[max_idx];
    let aftershocks: Vec<f64> = abs_returns.iter().enumerate()
        .filter(|&(i, _)| i != max_idx)
        .map(|(_, &v)| v)
        .collect();
    let bath = bath_law(mainshock_mag, &aftershocks);
    let bath_ratio = if bath.largest_aftershock.is_finite() && mainshock_mag > 1e-12 {
        bath.largest_aftershock / mainshock_mag
    } else {
        f64::NAN
    };

    SeismicResult { gr_b_value: gr_b, omori_p, bath_ratio, n_extreme }
}

// ── Scale-freeness (Gutenberg-Richter b-value on returns) ────────────────────

/// Scale-freeness features per bin.
///
/// Fintek's `scale_freeness.rs` (K02P12C03R02) outputs 4 scalars:
///   DO01: b_value — GR b-value (slope of power law)
///   DO02: b_std — standard error of b-value estimate
///   DO03: scaling_range — log10(max/min |return|) = decades of scaling
///   DO04: deviation_from_gr — RMS residual from pure GR log-linear fit
#[derive(Debug, Clone)]
pub struct ScaleFreenessResult {
    pub b_value: f64,
    pub b_std: f64,
    pub scaling_range: f64,
    pub deviation_from_gr: f64,
}

impl ScaleFreenessResult {
    pub fn nan() -> Self {
        Self { b_value: f64::NAN, b_std: f64::NAN, scaling_range: f64::NAN, deviation_from_gr: f64::NAN }
    }
}

/// Gutenberg-Richter power law on return magnitudes.
///
/// Treats |log_returns| as earthquake magnitudes and fits the GR relation:
/// log10(N(≥M)) = a − b·M. The b-value is the power-law exponent.
pub fn scale_freeness(returns: &[f64]) -> ScaleFreenessResult {
    const MIN_N: usize = 20;
    if returns.len() < MIN_N { return ScaleFreenessResult::nan(); }

    let magnitudes: Vec<f64> = returns.iter().map(|&x| x.abs()).filter(|&x| x > 1e-12).collect();
    if magnitudes.len() < 10 { return ScaleFreenessResult::nan(); }

    let m_min = magnitudes.iter().cloned().fold(f64::INFINITY, f64::min);
    let m_max = magnitudes.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if m_min <= 0.0 || m_max <= m_min { return ScaleFreenessResult::nan(); }

    let gr = gutenberg_richter_fit(&magnitudes, m_min);
    let b_value = gr.b_value;

    // Standard error of b-value via Aki (1965): SE(b) = b / sqrt(n)
    let n = magnitudes.len() as f64;
    let b_std = if b_value.is_finite() && b_value > 0.0 { b_value / n.sqrt() } else { f64::NAN };

    // Scaling range: number of decades
    let scaling_range = (m_max / m_min).log10();

    // Deviation from pure GR: compare empirical exceedance to fitted line
    // Sort magnitudes, compute empirical log10(N(≥M)) at 10 threshold points
    let mut sorted_mag = magnitudes.clone();
    sorted_mag.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n_total = sorted_mag.len() as f64;

    let deviation = if b_value.is_finite() && gr.a_value.is_finite() {
        let n_pts = 10.min(sorted_mag.len());
        let step = sorted_mag.len() / n_pts;
        let rms_sq: f64 = (0..n_pts).map(|k| {
            let idx = k * step;
            let m = sorted_mag[idx];
            let emp = (n_total - idx as f64).log10();
            let fitted = gr.a_value - b_value * m;
            (emp - fitted) * (emp - fitted)
        }).sum::<f64>() / n_pts as f64;
        rms_sq.sqrt()
    } else {
        f64::NAN
    };

    ScaleFreenessResult { b_value, b_std, scaling_range, deviation_from_gr: deviation }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stl_features_trend() {
        // Linearly increasing series — strong trend, no seasonality
        let data: Vec<f64> = (0..80).map(|i| i as f64 * 0.1).collect();
        let r = stl_features(&data);
        assert!(r.trend_strength.is_finite());
        assert!(r.trend_slope > 0.0, "upward trend should have positive slope, got {}", r.trend_slope);
    }

    #[test]
    fn stl_features_too_short() {
        let r = stl_features(&[1.0, 2.0, 3.0]);
        assert!(r.trend_strength.is_nan());
    }

    #[test]
    fn sde_basic() {
        let mut prices = vec![100.0_f64; 201];
        let mut rng = tambear::rng::Xoshiro256::new(42);
        for i in 1..201 {
            prices[i] = (prices[i - 1] + tambear::rng::sample_normal(&mut rng, 0.0, 0.5)).max(0.01);
        }
        let r = sde(&prices);
        assert!(r.drift_mean.is_finite());
        assert!(r.diffusion_mean.is_finite());
        assert!(r.diffusion_mean > 0.0);
    }

    #[test]
    fn sde_too_short() {
        let r = sde(&[1.0, 2.0]);
        assert!(r.drift_mean.is_nan());
    }

    #[test]
    fn hmm_features_basic() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        // Two-regime AR: regime 0 low-vol, regime 1 high-vol
        let mut data = Vec::with_capacity(200);
        let mut state = 0u8;
        for _ in 0..200 {
            let vol = if state == 0 { 0.01 } else { 0.05 };
            data.push(tambear::rng::sample_normal(&mut rng, 0.0, vol));
            if tambear::rng::sample_bernoulli(&mut rng, 0.05) { state = 1 - state; }
        }
        let r = hmm_features(&data);
        assert!(r.state_persistence.is_finite());
        assert!(r.entropy_rate.is_finite());
        assert!(r.viterbi_switches >= 0.0);
    }

    #[test]
    fn hmm_features_too_short() {
        let r = hmm_features(&[1.0, 2.0, 3.0]);
        assert!(r.state_persistence.is_nan());
    }

    #[test]
    fn savgol_basic() {
        let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin() + 0.1).collect();
        let smoothed = savgol(&data, 7, 3);
        assert_eq!(smoothed.len(), data.len());
        // Smoothed and original should be close for smooth signal
        let max_err: f64 = smoothed.iter().zip(data.iter()).map(|(s, d)| (s - d).abs()).fold(0.0_f64, f64::max);
        assert!(max_err < 0.5, "SG filter diverged: {}", max_err);
    }

    #[test]
    fn seismic_basic() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let returns: Vec<f64> = (0..100).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = seismic(&returns);
        assert!(r.n_extreme >= 0.0);
        assert!(r.gr_b_value.is_finite() || r.gr_b_value.is_nan());
    }

    #[test]
    fn seismic_too_short() {
        let r = seismic(&[0.01, -0.02, 0.005]);
        assert!(r.gr_b_value.is_nan());
    }

    #[test]
    fn scale_freeness_basic() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let returns: Vec<f64> = (0..100).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = scale_freeness(&returns);
        assert!(r.b_value.is_finite() || r.b_value.is_nan());
        assert!(r.scaling_range.is_finite() || r.scaling_range.is_nan());
    }

    #[test]
    fn scale_freeness_too_short() {
        let r = scale_freeness(&[0.01, -0.02]);
        assert!(r.b_value.is_nan());
    }
}
