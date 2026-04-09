//! Family 8 — Correlation / Dependence distance measures.
//!
//! Partial coverage. Adds DTW and edit distance for fintek's `dtw.rs`,
//! `edit_distance.rs`, and `dist_distance.rs`.

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
