//! Family 8 — Correlation / Dependence distance measures.
//!
//! Covers fintek leaves: `dtw`, `edit_distance`, `dist_distance`,
//! `msplit_temporal_coherence`.

use tambear::nonparametric::{dtw, dtw_banded, levenshtein, quantile_symbolize, edit_distance_on_series, pearson_r};
use tambear::{quantile, QuantileMethod, lag1_autocorrelation};
use tambear::signal_processing::regularize_interp;

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
        let qx = quantile(&sx, t, QuantileMethod::Linear);
        let qy = quantile(&sy, t, QuantileMethod::Linear);
        sum += (qx - qy).abs();
    }
    sum / n as f64
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

// ── Distributional distances (K02P16C02) ─────────────────────────────────────

/// DistDistance result matching fintek's `dist_distance.rs` (K02P16C02R01).
///
/// Compares the first-half vs. second-half distribution of a return series.
#[derive(Debug, Clone)]
pub struct DistDistanceResult {
    pub wasserstein: f64,
    pub energy_distance: f64,
    pub ks_stat: f64,
    pub ks_p: f64,
}

impl DistDistanceResult {
    pub fn nan() -> Self {
        Self { wasserstein: f64::NAN, energy_distance: f64::NAN,
               ks_stat: f64::NAN, ks_p: f64::NAN }
    }
}

/// Distributional distance between first and second half of a return series.
///
/// Outputs:
/// - Wasserstein-1 distance (sorted quantile interpolation)
/// - Szekely-Rizzo energy distance (subsample ≤ 200 each half for O(n²))
/// - KS statistic
/// - KS p-value (Kolmogorov approximation)
///
/// Returns NaN if total returns < 20 or either half < 5.
pub fn dist_distance(returns: &[f64]) -> DistDistanceResult {
    let n = returns.len();
    if n < 20 { return DistDistanceResult::nan(); }

    let mid = n / 2;
    let mut first: Vec<f64> = returns[..mid].iter().copied().filter(|v| v.is_finite()).collect();
    let mut second: Vec<f64> = returns[mid..].iter().copied().filter(|v| v.is_finite()).collect();
    first.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    second.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n1 = first.len();
    let n2 = second.len();
    if n1 < 5 || n2 < 5 { return DistDistanceResult::nan(); }

    // Wasserstein-1: quantile-grid interpolation
    let grid = n1.max(n2);
    let mut w_sum = 0.0f64;
    for g in 0..grid {
        let q = (g as f64 + 0.5) / grid as f64;
        let idx1 = ((q * n1 as f64) as usize).min(n1 - 1);
        let idx2 = ((q * n2 as f64) as usize).min(n2 - 1);
        w_sum += (first[idx1] - second[idx2]).abs();
    }
    let wasserstein = w_sum / grid as f64;

    // Energy distance (Szekely-Rizzo): subsample both halves to ≤ 200
    const MAX_SUB: usize = 200;
    let f_sub: Vec<f64> = if n1 > MAX_SUB {
        let step = n1 / MAX_SUB;
        first.iter().step_by(step).copied().collect()
    } else { first.clone() };
    let s_sub: Vec<f64> = if n2 > MAX_SUB {
        let step = n2 / MAX_SUB;
        second.iter().step_by(step).copied().collect()
    } else { second.clone() };

    let ns1 = f_sub.len(); let ns2 = s_sub.len();
    let mut e_xy = 0.0f64;
    for a in &f_sub { for b in &s_sub { e_xy += (a - b).abs(); } }
    e_xy /= (ns1 * ns2) as f64;

    let mut e_xx = 0.0f64;
    for j in 0..ns1 { for k in (j+1)..ns1 { e_xx += (f_sub[j] - f_sub[k]).abs(); } }
    if ns1 > 1 { e_xx *= 2.0 / (ns1 * (ns1 - 1)) as f64; }

    let mut e_yy = 0.0f64;
    for j in 0..ns2 { for k in (j+1)..ns2 { e_yy += (s_sub[j] - s_sub[k]).abs(); } }
    if ns2 > 1 { e_yy *= 2.0 / (ns2 * (ns2 - 1)) as f64; }

    let energy_distance = (2.0 * e_xy - e_xx - e_yy).max(0.0);

    // KS two-sample statistic
    let mut ks_max = 0.0f64;
    let mut i1 = 0usize;
    let mut i2 = 0usize;
    let mut all: Vec<f64> = first.iter().chain(second.iter()).copied().collect();
    all.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    all.dedup();
    for &v in &all {
        while i1 < n1 && first[i1] <= v { i1 += 1; }
        while i2 < n2 && second[i2] <= v { i2 += 1; }
        let diff = (i1 as f64 / n1 as f64 - i2 as f64 / n2 as f64).abs();
        if diff > ks_max { ks_max = diff; }
    }

    // Kolmogorov approximation for KS p-value
    let en = ((n1 * n2) as f64 / (n1 + n2) as f64).sqrt();
    let lambda = (en + 0.12 + 0.11 / en) * ks_max;
    let ks_p = (2.0 * (-2.0 * lambda * lambda).exp()).clamp(0.0, 1.0);

    DistDistanceResult { wasserstein, energy_distance, ks_stat: ks_max, ks_p }
}

#[cfg(test)]
mod tests_dist {
    use super::*;

    #[test]
    fn dist_distance_too_short() {
        let r = dist_distance(&[0.1, 0.2, 0.3]);
        assert!(r.wasserstein.is_nan());
    }

    #[test]
    fn dist_distance_identical_distribution() {
        // Same distribution both halves → small distances, high KS p-value
        let n = 100;
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let returns: Vec<f64> = (0..n).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = dist_distance(&returns);
        assert!(r.wasserstein.is_finite() && r.wasserstein >= 0.0);
        assert!(r.energy_distance.is_finite() && r.energy_distance >= 0.0);
        assert!(r.ks_stat.is_finite() && r.ks_stat >= 0.0 && r.ks_stat <= 1.0);
        assert!(r.ks_p.is_finite() && r.ks_p >= 0.0 && r.ks_p <= 1.0);
    }

    #[test]
    fn dist_distance_shifted_halves() {
        // Clear mean shift between halves → large Wasserstein, small KS p
        let mut returns = vec![0.0f64; 50];
        for i in 0..25 { returns[i] = -0.05; }
        for i in 25..50 { returns[i] = 0.05; }
        let r = dist_distance(&returns);
        assert!(r.wasserstein > 0.0, "shifted halves should have W > 0, got {}", r.wasserstein);
        assert!(r.ks_stat > 0.5, "KS stat should be high for shifted halves, got {}", r.ks_stat);
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

/// Compute M-split temporal coherence features.
pub fn msplit_temporal_coherence(data: &[f64], m: usize) -> MsplitCoherenceResult {
    if data.len() < 4 || m < 4 { return MsplitCoherenceResult::nan(); }
    let reg = regularize_interp(data, m);
    if reg.iter().any(|v| v.is_nan()) { return MsplitCoherenceResult::nan(); }

    let mean = reg.iter().sum::<f64>() / m as f64;
    let centered: Vec<f64> = reg.iter().map(|v| v - mean).collect();
    let var: f64 = centered.iter().map(|c| c * c).sum::<f64>() / m as f64;
    if var < 1e-30 { return MsplitCoherenceResult::nan(); }

    // Lag-1 autocorrelation
    let mean_autocorr = lag1_autocorrelation(&reg);

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
