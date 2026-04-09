//! Family 8 — Correlation / Dependence distance measures.
//!
//! Covers fintek leaves: `dtw`, `edit_distance`, `dist_distance`,
//! `msplit_temporal_coherence`.

use tambear::nonparametric::{dtw, dtw_banded, levenshtein, quantile_symbolize, edit_distance_on_series, pearson_r};

/// DTW distance between two bin-level series.
///
/// Equivalent to fintek's `dtw.rs`. Returns NaN for empty inputs.
pub fn dtw_distance(x: &[f64], y: &[f64]) -> f64 {
    dtw(x, y)
}

/// DTW with Sakoe-Chiba band constraint.
///
/// Faster for long series when the warping is near-diagonal.
pub fn dtw_banded_distance(x: &[f64], y: &[f64], window: usize) -> f64 {
    dtw_banded(x, y, window)
}

/// Edit distance on symbolized series (quantile-based alphabet).
///
/// Equivalent to fintek's `edit_distance.rs`. Symbolizes both series
/// into `n_symbols` quantile bins, then computes Levenshtein distance.
pub fn edit_distance(x: &[f64], y: &[f64], n_symbols: usize) -> usize {
    edit_distance_on_series(x, y, n_symbols)
}

/// Wasserstein-1 distance between two 1D distributions.
///
/// For 1D data, W1 = ∫ |F⁻¹_X(t) - F⁻¹_Y(t)| dt = mean absolute
/// difference of sorted samples (for equal-length samples).
///
/// For unequal lengths, we interpolate the ECDF.
pub fn wasserstein_1d(x: &[f64], y: &[f64]) -> f64 {
    let nx = x.len();
    let ny = y.len();
    if nx == 0 || ny == 0 { return f64::NAN; }

    let mut sx: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let mut sy: Vec<f64> = y.iter().copied().filter(|v| v.is_finite()).collect();
    sx.sort_by(|a, b| a.total_cmp(b));
    sy.sort_by(|a, b| a.total_cmp(b));

    if sx.is_empty() || sy.is_empty() { return f64::NAN; }

    // Resample both to common grid (use the larger of the two lengths)
    let n = sx.len().max(sy.len());
    let mut sum = 0.0_f64;
    for i in 0..n {
        // Interpolated quantile at rank (i+0.5)/n
        let t = (i as f64 + 0.5) / n as f64;
        let qx = sample_quantile(&sx, t);
        let qy = sample_quantile(&sy, t);
        sum += (qx - qy).abs();
    }
    sum / n as f64
}

/// Linear-interpolated quantile from a sorted slice.
fn sample_quantile(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 { return f64::NAN; }
    if n == 1 { return sorted[0]; }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(n - 1);
    let frac = pos - lo as f64;
    sorted[lo] * (1.0 - frac) + sorted[hi] * frac
}

/// Pearson cross-correlation at a single lag.
///
/// For `lag > 0`: Pearson(x[lag..], y[..n-lag]).
/// For `lag < 0`: Pearson(x[..n-|lag|], y[|lag|..]).
pub fn cross_correlation_at_lag(x: &[f64], y: &[f64], lag: i64) -> f64 {
    let nx = x.len();
    let ny = y.len();
    if nx == 0 || ny == 0 { return f64::NAN; }
    let n = nx.min(ny);
    if (lag.unsigned_abs() as usize) >= n { return f64::NAN; }

    let (xs, ys): (&[f64], &[f64]) = if lag >= 0 {
        let l = lag as usize;
        (&x[l..n], &y[..n - l])
    } else {
        let l = (-lag) as usize;
        (&x[..n - l], &y[l..n])
    };
    if xs.len() != ys.len() || xs.len() < 2 { return f64::NAN; }
    pearson_r(xs, ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dtw_distance_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert!((dtw_distance(&x, &x) - 0.0).abs() < 1e-15);
    }

    #[test]
    fn dtw_banded_constraint() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 4.0, 3.0, 2.0, 1.0];
        let d_full = dtw_distance(&x, &y);
        let d_band = dtw_banded_distance(&x, &y, 2);
        assert!(d_full.is_finite() && d_band.is_finite());
    }

    #[test]
    fn edit_distance_identical() {
        let x: Vec<f64> = (0..30).map(|i| (i as f64 * 0.1).sin()).collect();
        assert_eq!(edit_distance(&x, &x, 5), 0);
    }

    #[test]
    fn wasserstein_identical() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let w = wasserstein_1d(&x, &x);
        assert!((w - 0.0).abs() < 1e-12);
    }

    #[test]
    fn wasserstein_shifted() {
        // Distributions shifted by 5 → W1 = 5
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![6.0, 7.0, 8.0, 9.0, 10.0];
        let w = wasserstein_1d(&x, &y);
        assert!((w - 5.0).abs() < 0.5, "W1 of shift-5 should be ~5, got {}", w);
    }

    #[test]
    fn wasserstein_empty() {
        assert!(wasserstein_1d(&[], &[1.0, 2.0]).is_nan());
    }

    #[test]
    fn cross_correlation_zero_lag_equals_pearson() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // y = 2x
        let cc = cross_correlation_at_lag(&x, &y, 0);
        assert!((cc - 1.0).abs() < 1e-10, "CC at lag 0 for linear should be 1, got {}", cc);
    }

    #[test]
    fn cross_correlation_positive_lag() {
        // y is x shifted forward by 1
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        // At lag=1: correlate x[1..] with y[..n-1]
        // x[1..] = [2,3,4,5,6], y[..5] = [0,1,2,3,4] — still perfect linear
        let cc = cross_correlation_at_lag(&x, &y, 1);
        assert!((cc - 1.0).abs() < 1e-10, "CC at lag 1 should find the shift, got {}", cc);
    }

    #[test]
    fn cross_correlation_bad_lag() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![4.0, 5.0, 6.0];
        assert!(cross_correlation_at_lag(&x, &y, 10).is_nan());
    }
}

// ── M-split temporal coherence (K02P07C03) ────────────────────────────────────

/// M-split temporal coherence features.
///
/// Regularizes a variable-length return series to M equispaced points, then
/// computes temporal correlation features.
///
/// Fintek's `msplit_temporal_coherence.rs` outputs:
/// - mean_autocorr — lag-1 autocorrelation of the regularized series
/// - split_half_corr — Pearson r between first and second halves
/// - temporal_coherence — R² of linear fit to the regularized series
/// - decorrelation_scale — first lag where |ACF| drops below 1/e
///
/// `m`: regularization target length. For fintek leaf variants: 4, 8, 16, 32.
/// `data`: raw variable-length return series.
#[derive(Debug, Clone)]
pub struct MsplitCoherenceResult {
    pub mean_autocorr: f64,
    pub split_half_corr: f64,
    pub temporal_coherence: f64,
    pub decorrelation_scale: usize,
}

impl MsplitCoherenceResult {
    pub fn nan() -> Self {
        Self { mean_autocorr: f64::NAN, split_half_corr: f64::NAN,
               temporal_coherence: f64::NAN, decorrelation_scale: 0 }
    }
}

/// Regularize data to m equispaced points using linear interpolation.
fn regularize_interp_m(data: &[f64], m: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 { return vec![f64::NAN; m]; }
    if n == 1 { return vec![data[0]; m]; }
    (0..m).map(|i| {
        let t = (i as f64 / m as f64) * (n - 1) as f64;
        let lo = t as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = t - lo as f64;
        data[lo] * (1.0 - frac) + data[hi] * frac
    }).collect()
}

/// Compute M-split temporal coherence features.
pub fn msplit_temporal_coherence(data: &[f64], m: usize) -> MsplitCoherenceResult {
    if data.len() < 4 || m < 4 { return MsplitCoherenceResult::nan(); }
    let reg = regularize_interp_m(data, m);
    if reg.iter().any(|v| v.is_nan()) { return MsplitCoherenceResult::nan(); }

    let mean = reg.iter().sum::<f64>() / m as f64;
    let centered: Vec<f64> = reg.iter().map(|v| v - mean).collect();
    let var: f64 = centered.iter().map(|c| c * c).sum::<f64>() / m as f64;
    if var < 1e-30 { return MsplitCoherenceResult::nan(); }

    // Lag-1 autocorrelation
    let s01: f64 = (0..m-1).map(|t| centered[t] * centered[t+1]).sum();
    let mean_autocorr = s01 / (m as f64 * var);

    // Split-half correlation
    let half = m / 2;
    let h1 = &reg[..half];
    let h2 = &reg[half..half*2];
    let split_half_corr = pearson_r(h1, h2);

    // Temporal coherence: R² of OLS linear fit to reg vs index
    let mean_x = (m - 1) as f64 / 2.0;
    let mean_y = mean;
    let mut sxy = 0.0_f64; let mut sxx = 0.0_f64; let mut syy = 0.0_f64;
    for i in 0..m {
        let dx = i as f64 - mean_x; let dy = reg[i] - mean_y;
        sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
    }
    let temporal_coherence = if sxx > 1e-30 && syy > 1e-30 {
        (sxy * sxy / (sxx * syy)).clamp(0.0, 1.0)
    } else { f64::NAN };

    // Decorrelation scale: first lag where |ACF| < 1/e
    let inv_e = 1.0 / std::f64::consts::E;
    let mut decorrelation_scale = m;
    for lag in 1..m {
        let s: f64 = (0..m-lag).map(|t| centered[t] * centered[t+lag]).sum();
        let acf = s / (m as f64 * var);
        if acf.abs() < inv_e { decorrelation_scale = lag; break; }
    }

    MsplitCoherenceResult { mean_autocorr, split_half_corr, temporal_coherence, decorrelation_scale }
}

// ── Struct break (K02P18C03R02F01) ────────────────────────────────────────────

/// Structural break features (Chow-like scan).
///
/// Fintek's `struct_break.rs` outputs:
/// - max_f_stat — maximum F-statistic across all candidate breakpoints
/// - break_location — fractional position of the best break [0,1]
/// - pre_post_var_ratio — variance ratio post/pre break
/// - pre_post_mean_shift — mean shift at the break point
#[derive(Debug, Clone)]
pub struct StructBreakResult {
    pub max_f_stat: f64,
    pub break_location: f64,
    pub pre_post_var_ratio: f64,
    pub pre_post_mean_shift: f64,
}

impl StructBreakResult {
    pub fn nan() -> Self {
        Self { max_f_stat: f64::NAN, break_location: f64::NAN,
               pre_post_var_ratio: f64::NAN, pre_post_mean_shift: f64::NAN }
    }
}

/// Compute structural break features via Chow-like scan.
pub fn struct_break(returns: &[f64]) -> StructBreakResult {
    const MIN_SEG: usize = 20;
    let n = returns.len();
    if n < 2 * MIN_SEG { return StructBreakResult::nan(); }

    // Prefix sums for O(1) segment stats
    let mut cumsum = vec![0.0_f64; n + 1];
    let mut cumsum2 = vec![0.0_f64; n + 1];
    for i in 0..n { cumsum[i+1] = cumsum[i] + returns[i]; cumsum2[i+1] = cumsum2[i] + returns[i]*returns[i]; }

    let nf = n as f64;
    let total_mean = cumsum[n] / nf;
    let total_var = (cumsum2[n] / nf - total_mean * total_mean).max(1e-30);
    let ss_total = total_var * nf;

    let mut max_f = 0.0_f64;
    let mut best_k = n / 2;
    for k in MIN_SEG..(n - MIN_SEG + 1) {
        let kf = k as f64; let nkf = (n - k) as f64;
        let pre_mean = cumsum[k] / kf;
        let post_mean = (cumsum[n] - cumsum[k]) / nkf;
        // Between-group SS: n*var between = kf*(pre_mean - total_mean)² + nkf*(post_mean - total_mean)²
        let ss_between = kf * (pre_mean - total_mean).powi(2) + nkf * (post_mean - total_mean).powi(2);
        let ss_within = ss_total - ss_between;
        let f_stat = if ss_within > 1e-30 { (ss_between / 1.0) / (ss_within / (nf - 2.0)) } else { 0.0 };
        if f_stat > max_f { max_f = f_stat; best_k = k; }
    }

    let kf = best_k as f64; let nkf = (n - best_k) as f64;
    let pre_mean = cumsum[best_k] / kf;
    let post_mean = (cumsum[n] - cumsum[best_k]) / nkf;
    let pre_var = (cumsum2[best_k] / kf - pre_mean * pre_mean).max(1e-30);
    let post_var = ((cumsum2[n] - cumsum2[best_k]) / nkf - post_mean * post_mean).max(1e-30);

    StructBreakResult {
        max_f_stat: max_f,
        break_location: best_k as f64 / n as f64,
        pre_post_var_ratio: post_var / pre_var,
        pre_post_mean_shift: post_mean - pre_mean,
    }
}

#[cfg(test)]
mod tests_extra {
    use super::*;

    #[test]
    fn msplit_coherence_sine() {
        let n = 100;
        let data: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 10.0).sin()).collect();
        let r = msplit_temporal_coherence(&data, 16);
        assert!(r.mean_autocorr.is_finite());
        assert!(r.split_half_corr.abs() <= 1.0);
        assert!(r.temporal_coherence >= 0.0 && r.temporal_coherence <= 1.0);
    }

    #[test]
    fn msplit_coherence_too_short() {
        let r = msplit_temporal_coherence(&[1.0, 2.0], 16);
        assert!(r.mean_autocorr.is_nan());
    }

    #[test]
    fn struct_break_with_clear_break() {
        let mut data = vec![0.0_f64; 100];
        let mut rng = tambear::rng::Xoshiro256::new(42);
        for i in 0..50 { data[i] = tambear::rng::sample_normal(&mut rng, 0.0, 0.01); }
        for i in 50..100 { data[i] = tambear::rng::sample_normal(&mut rng, 0.5, 0.01); }
        let r = struct_break(&data);
        assert!(r.max_f_stat > 0.0);
        assert!(r.break_location > 0.0 && r.break_location < 1.0);
        assert!(r.pre_post_mean_shift.abs() > 0.1, "should detect the mean shift");
    }

    #[test]
    fn struct_break_too_short() {
        let r = struct_break(&[0.0; 10]);
        assert!(r.max_f_stat.is_nan());
    }
}
