//! Adversarial tests for new prerequisite methods:
//! Levene's test, KPSS, Ljung-Box, Durbin-Watson, Welch's ANOVA
//!
//! Each test asserts what the math SHOULD produce. Tests that FAIL
//! document real bugs — the pathmaker's work queue.

use tambear::hypothesis::*;
use tambear::time_series::*;

// ═══════════════════════════════════════════════════════════════════════════
// LEVENE'S TEST
// ═══════════════════════════════════════════════════════════════════════════

/// Levene with single group: k=1 → should return NaN (undefined).
#[test]
fn levene_single_group() {
    let g1 = vec![1.0, 2.0, 3.0];
    let result = levene_test(&[&g1], LeveneCenter::Median);
    assert!(result.f_statistic.is_nan(),
        "Levene with k=1 should return NaN F, got {}", result.f_statistic);
}

/// Levene with two identical groups: F=0 (no difference in variances).
#[test]
fn levene_identical_groups() {
    let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let g2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = levene_test(&[&g1, &g2], LeveneCenter::Median);
    assert!(result.f_statistic.is_finite(),
        "Levene with identical groups should be finite, got {}", result.f_statistic);
    assert!((result.f_statistic - 0.0).abs() < 1e-10,
        "Levene with identical groups should have F=0, got {}", result.f_statistic);
}

/// Levene with constant groups: all z_ij = 0 → F undefined (0/0).
#[test]
fn levene_constant_groups() {
    let g1 = vec![5.0; 10];
    let g2 = vec![5.0; 10];
    let result = levene_test(&[&g1, &g2], LeveneCenter::Median);
    // All deviations from median = 0 → ANOVA on zeros → F = 0/0
    assert!(result.f_statistic.is_nan() || result.f_statistic == 0.0,
        "Levene with constant groups should be NaN or 0, got {}", result.f_statistic);
}

/// Levene with empty group: should not panic.
#[test]
fn levene_empty_group() {
    let g1 = vec![1.0, 2.0, 3.0];
    let g2: Vec<f64> = vec![];
    let result = std::panic::catch_unwind(|| {
        levene_test(&[&g1, &g2[..]], LeveneCenter::Median)
    });
    // Empty group: median of empty slice panics or produces garbage.
    // Should handle gracefully.
    assert!(result.is_ok(),
        "Levene should not panic with an empty group");
}

/// Levene with known unequal variances: should detect heteroscedasticity.
#[test]
fn levene_known_unequal_variances() {
    // Group 1: low variance (std ≈ 0.82)
    let g1 = vec![4.0, 5.0, 6.0, 4.5, 5.5];
    // Group 2: high variance (std ≈ 4.47)
    let g2 = vec![0.0, 10.0, 5.0, 1.0, 9.0];
    let result = levene_test(&[&g1, &g2], LeveneCenter::Median);
    assert!(result.f_statistic > 1.0,
        "Levene should detect unequal variances: F={}", result.f_statistic);
    assert!(result.p_value < 0.2,
        "Levene p-value should be small for unequal variances, got {}", result.p_value);
}

/// Levene: Brown-Forsythe (median) vs original (mean) should give different F.
#[test]
fn levene_mean_vs_median() {
    let g1 = vec![1.0, 2.0, 3.0, 100.0]; // outlier in g1
    let g2 = vec![1.0, 2.0, 3.0, 4.0];
    let r_mean = levene_test(&[&g1, &g2], LeveneCenter::Mean);
    let r_median = levene_test(&[&g1, &g2], LeveneCenter::Median);
    assert!(r_mean.f_statistic.is_finite() && r_median.f_statistic.is_finite());
    // With an outlier, mean and median centers should give different F values
    assert!((r_mean.f_statistic - r_median.f_statistic).abs() > 0.01,
        "Mean and median Levene should differ with outlier: F_mean={}, F_median={}",
        r_mean.f_statistic, r_median.f_statistic);
}

/// Levene with single-element group: variance undefined.
#[test]
fn levene_singleton_group() {
    let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let g2 = vec![10.0]; // single element
    let result = std::panic::catch_unwind(|| {
        levene_test(&[&g1, &g2[..]], LeveneCenter::Median)
    });
    assert!(result.is_ok(),
        "Levene should not panic with singleton group");
}

// ═══════════════════════════════════════════════════════════════════════════
// LJUNG-BOX TEST
// ═══════════════════════════════════════════════════════════════════════════

/// Ljung-Box on white noise: should not reject (high p-value).
#[test]
fn ljung_box_white_noise() {
    // Deterministic "white noise" via LCG
    let mut rng = 12345u64;
    let data: Vec<f64> = (0..200).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5
    }).collect();
    let result = ljung_box(&data, 10, 0);
    assert!(result.statistic.is_finite(),
        "Ljung-Box Q should be finite, got {}", result.statistic);
    // White noise → autocorrelations near 0 → Q small → large p-value
    assert!(result.p_value > 0.01,
        "Ljung-Box on white noise should have p > 0.01, got {}", result.p_value);
}

/// Ljung-Box on AR(1) data: should detect autocorrelation.
#[test]
fn ljung_box_ar1() {
    let n = 200;
    let mut data = vec![0.0; n];
    let mut rng = 54321u64;
    for t in 1..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.1;
        data[t] = 0.9 * data[t - 1] + noise; // strong AR(1)
    }
    let result = ljung_box(&data, 10, 0);
    assert!(result.statistic > 10.0,
        "Ljung-Box on AR(1) should have large Q, got {}", result.statistic);
    assert!(result.p_value < 0.05,
        "Ljung-Box on AR(1) should reject (p < 0.05), got {}", result.p_value);
}

/// Ljung-Box with constant data: ACF at all lags = 1 → should detect.
#[test]
fn ljung_box_constant_data() {
    let data = vec![5.0; 100];
    let result = ljung_box(&data, 5, 0);
    // Constant data has zero variance → ACF is undefined (returns 1.0 by convention)
    // Q should be very large or NaN
    assert!(result.statistic.is_finite() || result.statistic.is_nan(),
        "Ljung-Box on constant data should not be Inf, got {}", result.statistic);
}

/// Ljung-Box with n_lags=0: degenerate → should return Q=0.
#[test]
fn ljung_box_zero_lags() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = ljung_box(&data, 0, 0);
    assert!((result.statistic - 0.0).abs() < 1e-10,
        "Ljung-Box with 0 lags should have Q=0, got {}", result.statistic);
}

/// Ljung-Box with n_lags > n: should clamp or handle gracefully.
#[test]
fn ljung_box_lags_exceed_n() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = std::panic::catch_unwind(|| {
        ljung_box(&data, 100, 0)
    });
    assert!(result.is_ok(),
        "Ljung-Box should not panic when n_lags > n");
}

/// Ljung-Box with fitted_params > n_lags: df = 0 or negative.
#[test]
fn ljung_box_fitted_exceeds_lags() {
    let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
    let result = ljung_box(&data, 5, 10); // fitted_params=10 > n_lags=5
    // df = max(5 - 10, 1) = 1 (clamped)
    assert_eq!(result.df, 1,
        "Ljung-Box with fitted_params > n_lags should clamp df to 1, got {}", result.df);
}

/// Ljung-Box with very short data (n=3).
#[test]
fn ljung_box_short_data() {
    let data = vec![1.0, 2.0, 3.0];
    let result = std::panic::catch_unwind(|| {
        ljung_box(&data, 2, 0)
    });
    assert!(result.is_ok(),
        "Ljung-Box should not panic on very short data");
}

// ═══════════════════════════════════════════════════════════════════════════
// KPSS TEST
// ═══════════════════════════════════════════════════════════════════════════

/// KPSS on stationary data: should not reject (small statistic).
#[test]
fn kpss_stationary() {
    let mut rng = 12345u64;
    let data: Vec<f64> = (0..200).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5
    }).collect();
    let result = kpss_test(&data, false, None);
    assert!(result.statistic.is_finite(),
        "KPSS statistic should be finite, got {}", result.statistic);
    // Stationary data → statistic < critical value at 5%
    assert!(result.statistic < result.critical_5pct,
        "KPSS on stationary data should not reject at 5%: stat={} < cv={}",
        result.statistic, result.critical_5pct);
}

/// KPSS on random walk (unit root): should reject.
#[test]
fn kpss_random_walk() {
    let n = 300;
    let mut data = vec![0.0; n];
    let mut rng = 54321u64;
    for t in 1..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        data[t] = data[t - 1] + ((rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.1;
    }
    let result = kpss_test(&data, false, None);
    assert!(result.statistic.is_finite(),
        "KPSS statistic should be finite, got {}", result.statistic);
    // Random walk → statistic >> critical value
    assert!(result.statistic > result.critical_10pct,
        "KPSS on random walk should reject at 10%: stat={} > cv={}",
        result.statistic, result.critical_10pct);
}

/// KPSS with constant data: residuals = 0 → statistic = 0.
#[test]
fn kpss_constant_data() {
    let data = vec![5.0; 50];
    let result = kpss_test(&data, false, None);
    // Constant data: all residuals = 0, partial sums = 0 → stat = 0
    assert!(result.statistic >= 0.0 || result.statistic.is_nan(),
        "KPSS on constant data should be >= 0, got {}", result.statistic);
}

/// KPSS with too-short data (n < 4): should return NaN gracefully.
#[test]
fn kpss_short_data() {
    let data = vec![1.0, 2.0, 3.0];
    let result = kpss_test(&data, false, None);
    assert!(result.statistic.is_nan(),
        "KPSS with n < 4 should return NaN, got {}", result.statistic);
}

/// KPSS trend vs level: trend test on trended data should not reject.
#[test]
fn kpss_trend_on_linear_trend() {
    let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.5 + 10.0).collect();
    let result_trend = kpss_test(&data, true, None);
    // Linear trend is trend-stationary → should not reject
    assert!(result_trend.statistic < result_trend.critical_1pct,
        "KPSS(trend) on linear data should not reject: stat={} < cv={}",
        result_trend.statistic, result_trend.critical_1pct);
}

/// KPSS with explicit lag=0: no Newey-West correction.
#[test]
fn kpss_lag_zero() {
    let mut rng = 99999u64;
    let data: Vec<f64> = (0..100).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as f64 / (1u64 << 31) as f64
    }).collect();
    let result = kpss_test(&data, false, Some(0));
    assert!(result.statistic.is_finite(),
        "KPSS with lag=0 should be finite, got {}", result.statistic);
    assert_eq!(result.n_lags, 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// DURBIN-WATSON TEST
// ═══════════════════════════════════════════════════════════════════════════

/// DW with all-zero residuals: 0/0 → should return d=2 (no autocorrelation).
#[test]
fn durbin_watson_zero_residuals() {
    let residuals = vec![0.0; 20];
    let result = durbin_watson(&residuals);
    assert!((result.statistic - 2.0).abs() < 1e-10,
        "DW of zero residuals should be 2.0 (guarded), got {}", result.statistic);
}

/// DW of constant non-zero residuals: d = 0 (perfect positive autocorrelation).
#[test]
fn durbin_watson_constant_residuals() {
    let residuals = vec![5.0; 20];
    let result = durbin_watson(&residuals);
    // e_t - e_{t-1} = 0 for all t → num = 0, den = 20*25 = 500 → d = 0
    assert!((result.statistic - 0.0).abs() < 1e-10,
        "DW of constant residuals should be 0.0, got {}", result.statistic);
    assert!((result.rho_hat - 1.0).abs() < 1e-10,
        "DW rho_hat for constant should be 1.0, got {}", result.rho_hat);
}

/// DW of alternating residuals: d ≈ 4 (perfect negative autocorrelation).
#[test]
fn durbin_watson_alternating() {
    let residuals: Vec<f64> = (0..20).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
    let result = durbin_watson(&residuals);
    // Alternating: e_t - e_{t-1} = ±2, sum of squares of differences = 19*4 = 76
    // sum of squares of residuals = 20 * 1 = 20, d = 76/20 = 3.8
    assert!(result.statistic > 3.5,
        "DW of alternating should be near 4, got {}", result.statistic);
    assert!(result.rho_hat < -0.5,
        "DW rho_hat for alternating should be negative, got {}", result.rho_hat);
}

/// DW with n=1: too short, should return d=2 (default).
#[test]
fn durbin_watson_single_residual() {
    let result = durbin_watson(&[5.0]);
    assert!((result.statistic - 2.0).abs() < 1e-10,
        "DW with n=1 should return 2.0 (default), got {}", result.statistic);
}

/// DW with empty residuals.
#[test]
fn durbin_watson_empty() {
    let result = durbin_watson(&[]);
    assert!((result.statistic - 2.0).abs() < 1e-10,
        "DW with empty should return 2.0 (default), got {}", result.statistic);
}

/// DW with NaN residuals.
#[test]
fn durbin_watson_nan() {
    let residuals = vec![1.0, f64::NAN, 3.0];
    let result = durbin_watson(&residuals);
    // NaN propagates through arithmetic
    assert!(result.statistic.is_nan(),
        "DW with NaN residuals should produce NaN, got {}", result.statistic);
}

// ═══════════════════════════════════════════════════════════════════════════
// WELCH'S ANOVA
// ═══════════════════════════════════════════════════════════════════════════

/// Welch ANOVA with identical groups: F=0.
#[test]
fn welch_anova_identical_groups() {
    let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let g2 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = welch_anova(&[&g1, &g2]);
    assert!(result.f_statistic.is_finite(),
        "Welch ANOVA identical groups F should be finite, got {}", result.f_statistic);
    assert!(result.f_statistic < 0.01,
        "Welch ANOVA identical groups F should be ~0, got {}", result.f_statistic);
}

/// Welch ANOVA with well-separated groups: should reject.
#[test]
fn welch_anova_separated_groups() {
    let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let g2 = vec![100.0, 101.0, 102.0, 103.0, 104.0];
    let result = welch_anova(&[&g1, &g2]);
    assert!(result.f_statistic > 10.0,
        "Welch ANOVA separated groups should have large F, got {}", result.f_statistic);
    assert!(result.p_value < 0.01,
        "Welch ANOVA separated groups should reject, got p={}", result.p_value);
}

/// Welch ANOVA with single group: k=1 → NaN.
#[test]
fn welch_anova_single_group() {
    let g1 = vec![1.0, 2.0, 3.0];
    let result = welch_anova(&[&g1]);
    assert!(result.f_statistic.is_nan(),
        "Welch ANOVA with k=1 should return NaN, got {}", result.f_statistic);
}

/// Welch ANOVA with constant group (zero variance): should return NaN.
/// Weight would explode (w_j = n/0 = Inf), making F statistic meaningless.
/// The correct behavior is to return NaN rather than a misleading finite value.
#[test]
fn welch_anova_constant_group() {
    let g1 = vec![5.0; 10]; // zero variance
    let g2 = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let result = std::panic::catch_unwind(|| {
        welch_anova(&[&g1, &g2])
    });
    assert!(result.is_ok(),
        "Welch ANOVA should not panic with constant group");
    let r = result.unwrap();
    // Zero-variance group causes weight explosion; NaN is the documented correct result
    assert!(r.f_statistic.is_nan(),
        "Welch ANOVA with constant group should return NaN (weight explosion guarded), got {}", r.f_statistic);
}

/// Welch ANOVA with all constant groups: 0/0.
#[test]
fn welch_anova_all_constant() {
    let g1 = vec![5.0; 10];
    let g2 = vec![5.0; 10];
    let result = welch_anova(&[&g1, &g2]);
    // Both groups have zero variance and same mean → F should be 0 or NaN
    assert!(result.f_statistic.is_nan() || result.f_statistic == 0.0 || result.f_statistic.is_finite(),
        "Welch ANOVA all-constant should not crash, got {}", result.f_statistic);
}

/// Welch ANOVA with many groups (k=5): verify degrees of freedom.
#[test]
fn welch_anova_five_groups() {
    let groups: Vec<Vec<f64>> = (0..5).map(|k| {
        (0..10).map(|i| (k * 10 + i) as f64).collect()
    }).collect();
    let refs: Vec<&[f64]> = groups.iter().map(|g| g.as_slice()).collect();
    let result = welch_anova(&refs);
    assert!((result.df_between - 4.0).abs() < 1e-10,
        "df_between should be k-1=4, got {}", result.df_between);
    assert!(result.df_within > 0.0,
        "df_within should be positive, got {}", result.df_within);
    assert!(result.f_statistic > 5.0,
        "F should be large for well-separated groups, got {}", result.f_statistic);
}

/// Welch ANOVA with singleton group: n_j - 1 = 0 → variance undefined.
#[test]
fn welch_anova_singleton() {
    let g1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let g2 = vec![100.0]; // singleton
    let result = std::panic::catch_unwind(|| {
        welch_anova(&[&g1, &g2[..]])
    });
    assert!(result.is_ok(),
        "Welch ANOVA should not panic with singleton group");
    let r = result.unwrap();
    assert!(r.f_statistic.is_finite() || r.f_statistic.is_nan(),
        "F should be finite or NaN, got {}", r.f_statistic);
}

/// Welch ANOVA with empty group: should not panic.
#[test]
fn welch_anova_empty_group() {
    let g1 = vec![1.0, 2.0, 3.0];
    let g2: Vec<f64> = vec![];
    let result = std::panic::catch_unwind(|| {
        welch_anova(&[&g1, &g2[..]])
    });
    // Empty group has n=0 → mean = 0/0 = NaN, var = NaN
    // Should handle gracefully (skip empty groups or return NaN)
    assert!(result.is_ok(),
        "Welch ANOVA should not panic with empty group");
}
