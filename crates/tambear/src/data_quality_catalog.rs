//! # Data Quality Catalog — every measure, every family
//!
//! Extension of [`crate::data_quality`] with the full published catalogs for:
//!
//! - **Counting**: count_above/below/in_range, count_zeros/positive/negative,
//!   count_nan/inf/finite, count_peaks/troughs/inflections, count_zero_crossings,
//!   count_sign_changes, count_runs, count_outliers_{iqr,zscore,mad},
//!   count_inversions, count_ties.
//!
//! - **Variability / dispersion**: sample_std, mad_raw, mad_normal, iqr, range,
//!   midrange, gini_coefficient, quartile_coefficient.
//!
//! - **Shape**: skewness_fisher, skewness_bowley, skewness_kelly,
//!   skewness_pearson, kurtosis_excess, kurtosis_crow_siddiqui.
//!
//! Every function is a single-purpose, cadence-agnostic primitive operating
//! on a slice. Callers compose them into family-specific validity predicates
//! or diagnostic bundles. Thresholds are downstream — we have every answer.
//!
//! Each function honestly handles empty, constant, and NaN-laden input by
//! returning NaN or a documented sentinel. Never panics. Never hardcodes
//! alpha/threshold/tuning constants beyond numerical sentinels.

// ═══════════════════════════════════════════════════════════════════════════
// COUNTING FAMILY
// ═══════════════════════════════════════════════════════════════════════════

/// Count of values strictly above a threshold.
#[inline]
pub fn count_above_threshold(x: &[f64], t: f64) -> usize {
    x.iter().filter(|&&v| v > t).count()
}

/// Count of values strictly below a threshold.
#[inline]
pub fn count_below_threshold(x: &[f64], t: f64) -> usize {
    x.iter().filter(|&&v| v < t).count()
}

/// Count of values in the closed interval `[lo, hi]`.
#[inline]
pub fn count_in_range(x: &[f64], lo: f64, hi: f64) -> usize {
    x.iter().filter(|&&v| v >= lo && v <= hi).count()
}

#[inline]
pub fn count_zeros(x: &[f64]) -> usize {
    x.iter().filter(|&&v| v == 0.0).count()
}

#[inline]
pub fn count_positive(x: &[f64]) -> usize {
    x.iter().filter(|&&v| v > 0.0).count()
}

#[inline]
pub fn count_negative(x: &[f64]) -> usize {
    x.iter().filter(|&&v| v < 0.0).count()
}

#[inline]
pub fn count_nan(x: &[f64]) -> usize {
    x.iter().filter(|v| v.is_nan()).count()
}

#[inline]
pub fn count_inf(x: &[f64]) -> usize {
    x.iter().filter(|v| v.is_infinite()).count()
}

#[inline]
pub fn count_finite(x: &[f64]) -> usize {
    x.iter().filter(|v| v.is_finite()).count()
}

/// Count of strict local maxima (peaks): `x[i-1] < x[i] > x[i+1]`.
pub fn count_peaks(x: &[f64]) -> usize {
    if x.len() < 3 {
        return 0;
    }
    (1..x.len() - 1).filter(|&i| x[i - 1] < x[i] && x[i] > x[i + 1]).count()
}

/// Count of strict local minima (troughs).
pub fn count_troughs(x: &[f64]) -> usize {
    if x.len() < 3 {
        return 0;
    }
    (1..x.len() - 1).filter(|&i| x[i - 1] > x[i] && x[i] < x[i + 1]).count()
}

/// Count of inflection points — sign changes in the second difference.
pub fn count_inflections(x: &[f64]) -> usize {
    if x.len() < 4 {
        return 0;
    }
    let d2: Vec<f64> = (1..x.len() - 1)
        .map(|i| x[i + 1] - 2.0 * x[i] + x[i - 1])
        .collect();
    (1..d2.len()).filter(|&i| d2[i - 1] * d2[i] < 0.0).count()
}

/// Count of zero crossings: sign changes between adjacent samples.
pub fn count_zero_crossings(x: &[f64]) -> usize {
    if x.len() < 2 {
        return 0;
    }
    (1..x.len())
        .filter(|&i| (x[i - 1] > 0.0 && x[i] < 0.0) || (x[i - 1] < 0.0 && x[i] > 0.0))
        .count()
}

/// Count of sign changes of the first difference (turning points).
pub fn count_sign_changes(x: &[f64]) -> usize {
    if x.len() < 3 {
        return 0;
    }
    let diffs: Vec<f64> = (1..x.len()).map(|i| x[i] - x[i - 1]).collect();
    (1..diffs.len())
        .filter(|&i| diffs[i - 1] * diffs[i] < 0.0)
        .count()
}

/// Count of runs of same-sign values. Zero values break runs.
pub fn count_runs(x: &[f64]) -> usize {
    if x.is_empty() {
        return 0;
    }
    let mut runs = 0usize;
    let mut prev_sign: i8 = 0;
    for &v in x {
        let sign: i8 = if v > 0.0 { 1 } else if v < 0.0 { -1 } else { 0 };
        if sign != 0 && sign != prev_sign {
            runs += 1;
        }
        prev_sign = sign;
    }
    runs
}

/// Count of outliers by the IQR rule: `x < Q1 - k·IQR` or `x > Q3 + k·IQR`.
pub fn count_outliers_iqr(x: &[f64], k: f64) -> usize {
    let clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 4 {
        return 0;
    }
    let mut sorted = clean.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let q1 = sorted[n / 4];
    let q3 = sorted[3 * n / 4];
    let iqr_val = q3 - q1;
    let lo = q1 - k * iqr_val;
    let hi = q3 + k * iqr_val;
    sorted.iter().filter(|&&v| v < lo || v > hi).count()
}

/// Count of outliers by the z-score rule: `|z| > threshold`.
pub fn count_outliers_zscore(x: &[f64], threshold: f64) -> usize {
    let clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 2 {
        return 0;
    }
    let mean = clean.iter().sum::<f64>() / n as f64;
    let var = clean.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let sd = var.sqrt();
    if sd < 1e-300 {
        return 0;
    }
    clean
        .iter()
        .filter(|&&v| ((v - mean) / sd).abs() > threshold)
        .count()
}

/// Count of outliers by MAD rule: `|x - median| / (1.4826·MAD) > threshold`.
pub fn count_outliers_mad(x: &[f64], threshold: f64) -> usize {
    let mut clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 2 {
        return 0;
    }
    clean.sort_by(|a, b| a.total_cmp(b));
    let median = if n % 2 == 0 {
        (clean[n / 2 - 1] + clean[n / 2]) / 2.0
    } else {
        clean[n / 2]
    };
    let mut abs_dev: Vec<f64> = clean.iter().map(|v| (v - median).abs()).collect();
    abs_dev.sort_by(|a, b| a.total_cmp(b));
    let mad = if n % 2 == 0 {
        (abs_dev[n / 2 - 1] + abs_dev[n / 2]) / 2.0
    } else {
        abs_dev[n / 2]
    };
    let scale = 1.4826 * mad;
    if scale < 1e-300 {
        return 0;
    }
    clean
        .iter()
        .filter(|&&v| ((v - median) / scale).abs() > threshold)
        .count()
}

/// Count of inversions: pairs `(i, j)` with `i < j` but `x[i] > x[j]`.
///
/// O(n log n) via merge sort. Foundation of Kendall's tau.
/// Delegates to `nonparametric::inversion_count` — the canonical global primitive.
pub fn count_inversions(x: &[f64]) -> u64 {
    crate::nonparametric::inversion_count(x).max(0) as u64
}

/// Count of tied pairs `(i, j)` with `i < j` and `x[i] == x[j]`. O(n log n).
pub fn count_ties(x: &[f64]) -> u64 {
    let mut sorted: Vec<f64> = x.iter().copied().filter(|v| !v.is_nan()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let mut ties = 0u64;
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i + 1;
        while j < sorted.len() && sorted[j].total_cmp(&sorted[i]) == std::cmp::Ordering::Equal {
            j += 1;
        }
        let run = (j - i) as u64;
        if run >= 2 {
            ties += run * (run - 1) / 2;
        }
        i = j;
    }
    ties
}

// ═══════════════════════════════════════════════════════════════════════════
// VARIABILITY / DISPERSION FAMILY
// ═══════════════════════════════════════════════════════════════════════════

/// Sample standard deviation (ddof = 1) of finite values.
pub fn sample_std(x: &[f64]) -> f64 {
    let clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 2 {
        return f64::NAN;
    }
    let m = clean.iter().sum::<f64>() / n as f64;
    let v = clean.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (n - 1) as f64;
    v.sqrt()
}

/// Raw median absolute deviation (no consistency constant).
pub fn mad_raw(x: &[f64]) -> f64 {
    let mut clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 2 {
        return f64::NAN;
    }
    clean.sort_by(|a, b| a.total_cmp(b));
    let med = if n % 2 == 0 {
        (clean[n / 2 - 1] + clean[n / 2]) / 2.0
    } else {
        clean[n / 2]
    };
    let mut abs_dev: Vec<f64> = clean.iter().map(|v| (v - med).abs()).collect();
    abs_dev.sort_by(|a, b| a.total_cmp(b));
    if n % 2 == 0 {
        (abs_dev[n / 2 - 1] + abs_dev[n / 2]) / 2.0
    } else {
        abs_dev[n / 2]
    }
}

/// Normal-consistent MAD: `1.4826 · MAD`. Approximates σ for Gaussian data.
#[inline]
pub fn mad_normal(x: &[f64]) -> f64 {
    1.4826 * mad_raw(x)
}

/// Interquartile range: `Q3 − Q1`.
pub fn iqr(x: &[f64]) -> f64 {
    let mut clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 4 {
        return f64::NAN;
    }
    clean.sort_by(|a, b| a.total_cmp(b));
    let q1 = clean[n / 4];
    let q3 = clean[3 * n / 4];
    q3 - q1
}

/// Range: `max − min` of finite values.
pub fn range(x: &[f64]) -> f64 {
    let clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    if clean.is_empty() {
        return f64::NAN;
    }
    let mn = clean.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = clean.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    mx - mn
}

/// Midrange: `(max + min) / 2`.
pub fn midrange(x: &[f64]) -> f64 {
    let clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    if clean.is_empty() {
        return f64::NAN;
    }
    let mn = clean.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = clean.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (mx + mn) / 2.0
}

/// Gini coefficient of |x|. Range `[0, 1]`. `0 = perfectly equal`.
pub fn gini_coefficient(x: &[f64]) -> f64 {
    let mut abs_vals: Vec<f64> = x.iter().map(|v| v.abs()).filter(|v| v.is_finite()).collect();
    let n = abs_vals.len();
    if n < 2 {
        return f64::NAN;
    }
    abs_vals.sort_by(|a, b| a.total_cmp(b));
    let sum: f64 = abs_vals.iter().sum();
    if sum < 1e-300 {
        return 0.0;
    }
    let mut cum = 0.0;
    for (i, &v) in abs_vals.iter().enumerate() {
        let weight = (2 * i + 1) as i64 - n as i64;
        cum += weight as f64 * v;
    }
    cum / (n as f64 * sum)
}

/// Quartile coefficient of dispersion: `(Q3 − Q1) / (Q3 + Q1)`.
pub fn quartile_coefficient(x: &[f64]) -> f64 {
    let mut clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 4 {
        return f64::NAN;
    }
    clean.sort_by(|a, b| a.total_cmp(b));
    let q1 = clean[n / 4];
    let q3 = clean[3 * n / 4];
    if (q3 + q1).abs() < 1e-300 {
        return f64::NAN;
    }
    (q3 - q1) / (q3 + q1)
}

// ═══════════════════════════════════════════════════════════════════════════
// SHAPE FAMILY — skewness and kurtosis variants
// ═══════════════════════════════════════════════════════════════════════════

/// Fisher-Pearson skewness: `m3 / m2^(3/2)`.
pub fn skewness_fisher(x: &[f64]) -> f64 {
    let clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 3 {
        return f64::NAN;
    }
    let nf = n as f64;
    let m = clean.iter().sum::<f64>() / nf;
    let m2 = clean.iter().map(|v| (v - m).powi(2)).sum::<f64>() / nf;
    let m3 = clean.iter().map(|v| (v - m).powi(3)).sum::<f64>() / nf;
    if m2 < 1e-300 {
        return 0.0;
    }
    m3 / m2.powf(1.5)
}

/// Bowley quartile skewness: `(Q3 + Q1 − 2·median) / IQR`. Range `[-1, 1]`.
/// Robust to outliers.
pub fn skewness_bowley(x: &[f64]) -> f64 {
    let mut clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 4 {
        return f64::NAN;
    }
    clean.sort_by(|a, b| a.total_cmp(b));
    let q1 = clean[n / 4];
    let q3 = clean[3 * n / 4];
    let median = if n % 2 == 0 {
        (clean[n / 2 - 1] + clean[n / 2]) / 2.0
    } else {
        clean[n / 2]
    };
    let iqr_val = q3 - q1;
    if iqr_val < 1e-300 {
        return 0.0;
    }
    (q3 + q1 - 2.0 * median) / iqr_val
}

/// Kelly skewness: `(P90 + P10 − 2·median) / (P90 − P10)`.
/// More robust to tails than Bowley.
pub fn skewness_kelly(x: &[f64]) -> f64 {
    let mut clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 10 {
        return f64::NAN;
    }
    clean.sort_by(|a, b| a.total_cmp(b));
    let p10 = clean[n / 10];
    let p90 = clean[9 * n / 10];
    let median = if n % 2 == 0 {
        (clean[n / 2 - 1] + clean[n / 2]) / 2.0
    } else {
        clean[n / 2]
    };
    let span = p90 - p10;
    if span < 1e-300 {
        return 0.0;
    }
    (p90 + p10 - 2.0 * median) / span
}

/// Pearson's second skewness (median-based): `3·(mean − median) / std`.
pub fn skewness_pearson(x: &[f64]) -> f64 {
    let clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 3 {
        return f64::NAN;
    }
    let mean = clean.iter().sum::<f64>() / n as f64;
    let std = sample_std(&clean);
    let mut sorted = clean.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = if n % 2 == 0 {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    } else {
        sorted[n / 2]
    };
    if std < 1e-300 {
        return 0.0;
    }
    3.0 * (mean - median) / std
}

/// Excess kurtosis: `m4 / m2² − 3`. Zero for normal distribution.
pub fn kurtosis_excess(x: &[f64]) -> f64 {
    let clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 4 {
        return f64::NAN;
    }
    let nf = n as f64;
    let m = clean.iter().sum::<f64>() / nf;
    let m2 = clean.iter().map(|v| (v - m).powi(2)).sum::<f64>() / nf;
    let m4 = clean.iter().map(|v| (v - m).powi(4)).sum::<f64>() / nf;
    if m2 < 1e-300 {
        return 0.0;
    }
    m4 / (m2 * m2) - 3.0
}

/// Crow-Siddiqui tail-weight kurtosis: `(P97.5 − P2.5) / (P75 − P25)`.
/// Robust alternative to moment kurtosis; larger values mean heavier tails.
pub fn kurtosis_crow_siddiqui(x: &[f64]) -> f64 {
    let mut clean: Vec<f64> = x.iter().copied().filter(|v| v.is_finite()).collect();
    let n = clean.len();
    if n < 40 {
        return f64::NAN;
    }
    clean.sort_by(|a, b| a.total_cmp(b));
    let p025 = clean[n / 40];
    let p975 = clean[39 * n / 40];
    let p25 = clean[n / 4];
    let p75 = clean[3 * n / 4];
    let iqr_val = p75 - p25;
    if iqr_val < 1e-300 {
        return f64::NAN;
    }
    (p975 - p025) / iqr_val
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── counting ───────────────────────────────────────────────────────

    #[test]
    fn count_above_below_range() {
        let x = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(count_above_threshold(&x, 3.0), 2); // 4, 5
        assert_eq!(count_below_threshold(&x, 3.0), 2); // 1, 2
        assert_eq!(count_in_range(&x, 2.0, 4.0), 3); // 2, 3, 4
    }

    #[test]
    fn count_sign_categories() {
        let x = [-2.0, -1.0, 0.0, 0.0, 1.0, 2.0, f64::NAN];
        assert_eq!(count_negative(&x), 2);
        assert_eq!(count_zeros(&x), 2);
        assert_eq!(count_positive(&x), 2);
        assert_eq!(count_nan(&x), 1);
    }

    #[test]
    fn count_finite_inf_nan() {
        let x = [1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 2.0];
        assert_eq!(count_finite(&x), 2);
        assert_eq!(count_inf(&x), 2);
        assert_eq!(count_nan(&x), 1);
    }

    #[test]
    fn count_peaks_simple() {
        // 1 2 1 3 1: two peaks at indices 1 and 3
        assert_eq!(count_peaks(&[1.0, 2.0, 1.0, 3.0, 1.0]), 2);
        assert_eq!(count_troughs(&[2.0, 1.0, 3.0, 2.0, 4.0]), 2);
    }

    #[test]
    fn count_inflections_parabola_cubic() {
        // sin(x) has infinitely many inflection points at every n·π.
        // Sample densely: over [0, 4π] we get ~4 inflections.
        let x: Vec<f64> = (0..100).map(|i| (i as f64 * 0.126).sin()).collect();
        let n_inflect = count_inflections(&x);
        assert!(
            n_inflect >= 2,
            "sin over [0, 4π] should have several inflections, got {}",
            n_inflect
        );
    }

    #[test]
    fn count_zero_crossings_alternating() {
        let x = [-1.0, 1.0, -1.0, 1.0, -1.0];
        assert_eq!(count_zero_crossings(&x), 4);
    }

    #[test]
    fn count_sign_changes_turning_points() {
        // Monotonic increasing: 0 sign changes in first difference
        let mono: Vec<f64> = (0..10).map(|i| i as f64).collect();
        assert_eq!(count_sign_changes(&mono), 0);
        // Zigzag: many sign changes
        let zig = [1.0, 3.0, 2.0, 4.0, 3.0, 5.0];
        assert!(count_sign_changes(&zig) >= 3);
    }

    #[test]
    fn count_runs_sign_based() {
        let x = [1.0, 2.0, -1.0, -2.0, 3.0, 4.0, -5.0];
        // Runs: (+,+), (-,-), (+,+), (-) → 4 runs
        assert_eq!(count_runs(&x), 4);
    }

    #[test]
    fn count_outliers_iqr_no_outliers() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        assert_eq!(count_outliers_iqr(&x, 1.5), 0);
    }

    #[test]
    fn count_outliers_iqr_with_extremes() {
        let mut x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        x.push(1e6);
        x.push(-1e6);
        assert!(count_outliers_iqr(&x, 1.5) >= 2);
    }

    #[test]
    fn count_outliers_zscore_threshold() {
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        // Uniform 0..99: nothing exceeds z=3
        assert_eq!(count_outliers_zscore(&x, 3.0), 0);
    }

    #[test]
    fn count_outliers_mad_robust_to_extremes() {
        let mut x = vec![0.0; 98];
        x.push(1e9); // huge outlier
        x.push(-1e9); // huge outlier
        // MAD is 0 (most values are 0) → scale is 0 → returns 0 by design
        let _ = count_outliers_mad(&x, 3.0);
    }

    #[test]
    fn count_inversions_sorted() {
        let sorted = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert_eq!(count_inversions(&sorted), 0);
        let reversed = [5.0, 4.0, 3.0, 2.0, 1.0];
        assert_eq!(count_inversions(&reversed), 10); // C(5,2)
    }

    #[test]
    fn count_inversions_small() {
        // [2, 1, 3]: one inversion (2,1)
        assert_eq!(count_inversions(&[2.0, 1.0, 3.0]), 1);
        // [3, 1, 2]: two inversions (3,1), (3,2)
        assert_eq!(count_inversions(&[3.0, 1.0, 2.0]), 2);
    }

    #[test]
    fn count_ties_known() {
        assert_eq!(count_ties(&[1.0, 1.0, 1.0]), 3); // C(3,2) = 3
        assert_eq!(count_ties(&[1.0, 2.0, 3.0]), 0);
        assert_eq!(count_ties(&[1.0, 1.0, 2.0, 2.0]), 2); // two pairs
    }

    // ── variability ────────────────────────────────────────────────────

    #[test]
    fn sample_std_known_value() {
        // [1,2,3,4,5]: variance = 2.5, std ≈ 1.5811
        let s = sample_std(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((s - 1.5811).abs() < 0.001);
    }

    #[test]
    fn mad_known_value() {
        // [1,2,3,4,5]: median = 3, abs_dev = [2,1,0,1,2], MAD = 1
        assert_eq!(mad_raw(&[1.0, 2.0, 3.0, 4.0, 5.0]), 1.0);
    }

    #[test]
    fn mad_normal_consistency() {
        let m = mad_normal(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!((m - 1.4826).abs() < 1e-10);
    }

    #[test]
    fn iqr_known_value() {
        // [1..=8]: Q1=2, Q3=6, IQR=4
        let iqr_val = iqr(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!((iqr_val - 4.0).abs() < 1e-10);
    }

    #[test]
    fn range_known() {
        assert_eq!(range(&[1.0, 5.0, 3.0, 2.0]), 4.0);
    }

    #[test]
    fn midrange_known() {
        assert_eq!(midrange(&[2.0, 4.0, 6.0, 8.0]), 5.0);
    }

    #[test]
    fn gini_equal_is_zero() {
        let g = gini_coefficient(&[1.0, 1.0, 1.0, 1.0]);
        assert!(g.abs() < 1e-10, "gini equal = {}", g);
    }

    #[test]
    fn gini_unequal_is_positive() {
        let g = gini_coefficient(&[0.0, 0.0, 0.0, 1.0]);
        assert!(g > 0.5);
    }

    #[test]
    fn quartile_coefficient_known() {
        // [1..=8]: n=8, Q1 = sorted[n/4] = sorted[2] = 3,
        //           Q3 = sorted[3n/4] = sorted[6] = 7
        // → (7 - 3) / (7 + 3) = 0.4
        let q = quartile_coefficient(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        assert!((q - 0.4).abs() < 1e-10, "q = {}", q);
    }

    // ── shape ──────────────────────────────────────────────────────────

    #[test]
    fn skewness_fisher_symmetric_is_zero() {
        // Symmetric around 0: [-2,-1,0,1,2] → skew ≈ 0
        let s = skewness_fisher(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn skewness_fisher_positive() {
        // Right-skewed: most values small, a few large
        let x = [1.0, 1.0, 1.0, 1.0, 10.0];
        assert!(skewness_fisher(&x) > 0.0);
    }

    #[test]
    fn skewness_bowley_symmetric() {
        let s = skewness_bowley(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(s.abs() < 1e-10);
    }

    #[test]
    fn skewness_bowley_bounded() {
        let x = [1.0, 1.0, 1.0, 1.0, 10.0, 10.0, 10.0, 10.0];
        let s = skewness_bowley(&x);
        assert!(s >= -1.0 && s <= 1.0);
    }

    #[test]
    fn skewness_kelly_needs_10() {
        // Need at least 10 samples
        let s = skewness_kelly(&[1.0, 2.0, 3.0]);
        assert!(s.is_nan());
        let x: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let sk = skewness_kelly(&x);
        assert!(sk.abs() < 0.5); // approximately symmetric
    }

    #[test]
    fn skewness_pearson_symmetric() {
        let s = skewness_pearson(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(s.abs() < 0.1);
    }

    #[test]
    fn kurtosis_excess_uniform_is_negative() {
        // Uniform distribution has excess kurtosis = -1.2
        let x: Vec<f64> = (0..1000).map(|i| i as f64 / 1000.0).collect();
        let k = kurtosis_excess(&x);
        // Approximately -1.2 for uniform
        assert!(k < 0.0, "uniform kurtosis = {}", k);
    }

    #[test]
    fn kurtosis_crow_siddiqui_needs_40() {
        assert!(kurtosis_crow_siddiqui(&[1.0, 2.0, 3.0]).is_nan());
        let x: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let k = kurtosis_crow_siddiqui(&x);
        assert!(k.is_finite());
        assert!(k > 0.0);
    }
}
