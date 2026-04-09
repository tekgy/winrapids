//! Adversarial tests — Wave 3: WLS, Anderson-Darling, Fisher's exact,
//! Friedman test, plus gold-standard verifications.

use tambear::linear_algebra::Mat;

// ═══════════════════════════════════════════════════════════════════════════
// WLS (Weighted Least Squares)
// ═══════════════════════════════════════════════════════════════════════════

/// WLS with uniform weights: should equal OLS.
#[test]
fn wls_uniform_weights_equals_ols() {
    let x = Mat::from_vec(4, 2, vec![
        1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0,
    ]);
    let y = vec![3.0, 5.0, 7.0, 9.0]; // y = 2x + 1
    let w = vec![1.0; 4];
    let result = tambear::multivariate::wls(&x, &y, &w);
    assert!((result.beta[0] - 1.0).abs() < 0.01,
        "WLS uniform intercept should be ~1, got {}", result.beta[0]);
    assert!((result.beta[1] - 2.0).abs() < 0.01,
        "WLS uniform slope should be ~2, got {}", result.beta[1]);
}

/// WLS with zero weight: excludes that observation.
#[test]
fn wls_zero_weight_excludes() {
    let x = Mat::from_vec(5, 2, vec![
        1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 100.0,
    ]);
    let y = vec![3.0, 5.0, 7.0, 9.0, 999.0]; // last is outlier
    let w = vec![1.0, 1.0, 1.0, 1.0, 0.0]; // zero-weight the outlier
    let result = tambear::multivariate::wls(&x, &y, &w);
    // Should recover y = 2x + 1 ignoring the outlier
    assert!((result.beta[1] - 2.0).abs() < 0.1,
        "WLS with zero-weighted outlier: slope should be ~2, got {}", result.beta[1]);
}

/// WLS with all-zero weights: degenerate.
#[test]
fn wls_all_zero_weights() {
    let x = Mat::from_vec(3, 1, vec![1.0, 2.0, 3.0]);
    let y = vec![1.0, 2.0, 3.0];
    let w = vec![0.0; 3];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tambear::multivariate::wls(&x, &y, &w)
    }));
    assert!(result.is_ok(), "WLS should not panic with all-zero weights");
}

/// WLS R² in [0, 1].
#[test]
fn wls_r2_range() {
    let x = Mat::from_vec(5, 2, vec![
        1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0,
    ]);
    let y = vec![2.0, 4.1, 5.9, 8.0, 10.1];
    let w = vec![1.0, 2.0, 1.0, 2.0, 1.0];
    let result = tambear::multivariate::wls(&x, &y, &w);
    assert!(result.r2 >= 0.0 && result.r2 <= 1.0,
        "WLS R² should be in [0,1], got {}", result.r2);
}

// ═══════════════════════════════════════════════════════════════════════════
// ANDERSON-DARLING
// ═══════════════════════════════════════════════════════════════════════════

/// AD on normal data: should not reject (large p).
#[test]
fn anderson_darling_normal() {
    let mut rng = 12345u64;
    let data: Vec<f64> = (0..100).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u1 = (rng >> 11) as f64 / (1u64 << 53) as f64;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u2 = (rng >> 11) as f64 / (1u64 << 53) as f64;
        (-2.0 * u1.max(1e-300).ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }).collect();
    let result = tambear::nonparametric::anderson_darling(&data);
    assert!(result.statistic.is_finite(), "AD statistic should be finite");
    assert!(result.p_value > 0.05,
        "AD on normal data should not reject: p={}", result.p_value);
}

/// AD on uniform data: should reject (non-normal).
#[test]
fn anderson_darling_uniform() {
    let data: Vec<f64> = (0..100).map(|i| i as f64 / 100.0).collect();
    let result = tambear::nonparametric::anderson_darling(&data);
    assert!(result.statistic > 0.5,
        "AD on uniform should have large A²*, got {}", result.statistic);
    assert!(result.p_value < 0.05,
        "AD on uniform should reject normality: p={}", result.p_value);
}

/// AD on constant data: W=1, degenerate normal.
#[test]
fn anderson_darling_constant() {
    let data = vec![5.0; 50];
    let result = tambear::nonparametric::anderson_darling(&data);
    assert!((result.statistic - 0.0).abs() < 1e-10,
        "AD on constant should be 0, got {}", result.statistic);
    assert!((result.p_value - 1.0).abs() < 1e-10,
        "AD on constant should have p=1, got {}", result.p_value);
}

/// AD with n < 3: NaN.
#[test]
fn anderson_darling_too_few() {
    let result = tambear::nonparametric::anderson_darling(&[1.0, 2.0]);
    assert!(result.statistic.is_nan(), "AD with n=2 should be NaN");
}

/// AD with all NaN: NaN.
#[test]
fn anderson_darling_all_nan() {
    let data = vec![f64::NAN; 10];
    let result = tambear::nonparametric::anderson_darling(&data);
    assert!(result.statistic.is_nan(), "AD on all-NaN should be NaN");
}

/// AD statistic should be non-negative.
#[test]
fn anderson_darling_non_negative() {
    let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
    let result = tambear::nonparametric::anderson_darling(&data);
    assert!(result.statistic >= 0.0,
        "A²* should be non-negative, got {}", result.statistic);
}

// ═══════════════════════════════════════════════════════════════════════════
// FISHER'S EXACT TEST
// ═══════════════════════════════════════════════════════════════════════════

/// Fisher's exact: classic tea-tasting example.
/// Lady correctly identified all 4 cups of each type.
/// Table: [[4,0],[0,4]]. One-sided p = 1/70, two-sided p = 2/70 ≈ 0.0286.
/// (Both extreme tables [4,0,0,4] and [0,4,4,0] have the same probability.)
#[test]
fn fisher_exact_tea_tasting() {
    let result = tambear::hypothesis::fisher_exact(&[4, 0, 0, 4]);
    assert!((result.p_value - 2.0 / 70.0).abs() < 0.01,
        "Tea tasting two-sided p should be 2/70 ≈ 0.0286, got {}", result.p_value);
    assert!(result.odds_ratio.is_infinite(),
        "Perfect association → OR = Inf, got {}", result.odds_ratio);
}

/// Fisher's exact: no association → p ≈ 1.
#[test]
fn fisher_exact_no_association() {
    // Balanced table: [[5,5],[5,5]]
    let result = tambear::hypothesis::fisher_exact(&[5, 5, 5, 5]);
    assert!(result.p_value > 0.9,
        "Balanced table should have p near 1, got {}", result.p_value);
    assert!((result.odds_ratio - 1.0).abs() < 0.01,
        "Balanced table OR should be 1, got {}", result.odds_ratio);
}

/// Fisher's exact: strong association.
#[test]
fn fisher_exact_strong_association() {
    let result = tambear::hypothesis::fisher_exact(&[10, 1, 1, 10]);
    assert!(result.p_value < 0.01,
        "Strong association should have small p, got {}", result.p_value);
    assert!(result.odds_ratio > 10.0,
        "Strong association OR should be large, got {}", result.odds_ratio);
}

/// Fisher's exact with zero cell.
#[test]
fn fisher_exact_zero_cell() {
    let result = tambear::hypothesis::fisher_exact(&[5, 0, 3, 7]);
    assert!(result.p_value.is_finite(),
        "Fisher with zero cell should be finite, got {}", result.p_value);
    assert!(result.odds_ratio.is_infinite(),
        "Zero cell → OR = Inf, got {}", result.odds_ratio);
}

/// Fisher's exact with all zeros: degenerate.
#[test]
fn fisher_exact_all_zero() {
    let result = std::panic::catch_unwind(|| {
        tambear::hypothesis::fisher_exact(&[0, 0, 0, 0])
    });
    assert!(result.is_ok(), "Fisher should not panic on all-zero table");
}

/// Fisher's exact: p in [0, 1].
#[test]
fn fisher_exact_p_range() {
    for table in &[[3, 1, 1, 3], [10, 2, 3, 15], [1, 1, 1, 1], [0, 5, 5, 0]] {
        let result = tambear::hypothesis::fisher_exact(table);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0 + 1e-10,
            "Fisher p should be in [0,1] for {:?}, got {}", table, result.p_value);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// FRIEDMAN TEST
// ═══════════════════════════════════════════════════════════════════════════

/// Friedman with identical treatments: Q = 0, p ≈ 1.
#[test]
fn friedman_identical_treatments() {
    // 5 subjects, 3 treatments, all same values
    let data = vec![
        5.0, 5.0, 5.0,
        3.0, 3.0, 3.0,
        7.0, 7.0, 7.0,
        1.0, 1.0, 1.0,
        9.0, 9.0, 9.0,
    ];
    let result = tambear::nonparametric::friedman_test(&data, 5, 3);
    assert!((result.statistic - 0.0).abs() < 1e-10,
        "Friedman identical treatments Q should be 0, got {}", result.statistic);
    assert!(result.p_value > 0.9,
        "Friedman identical p should be ~1, got {}", result.p_value);
}

/// Friedman with clear treatment effect: should reject.
#[test]
fn friedman_clear_effect() {
    // Treatment 3 always ranks highest
    let data = vec![
        1.0, 2.0, 10.0,
        1.0, 3.0, 11.0,
        2.0, 3.0, 12.0,
        1.0, 2.0, 9.0,
        2.0, 3.0, 13.0,
        1.0, 2.0, 8.0,
    ];
    let result = tambear::nonparametric::friedman_test(&data, 6, 3);
    assert!(result.statistic > 5.0,
        "Friedman should detect treatment effect: Q={}", result.statistic);
    assert!(result.p_value < 0.05,
        "Friedman should reject H0: p={}", result.p_value);
}

/// Friedman with k=1: degenerate → NaN.
#[test]
fn friedman_single_treatment() {
    let data = vec![1.0, 2.0, 3.0]; // 3 subjects, 1 treatment
    let result = tambear::nonparametric::friedman_test(&data, 3, 1);
    assert!(result.statistic.is_nan(),
        "Friedman with k=1 should be NaN, got {}", result.statistic);
}

/// Friedman with n=1: too few subjects → NaN.
#[test]
fn friedman_single_subject() {
    let data = vec![1.0, 2.0, 3.0]; // 1 subject, 3 treatments
    let result = tambear::nonparametric::friedman_test(&data, 1, 3);
    assert!(result.statistic.is_nan(),
        "Friedman with n=1 should be NaN, got {}", result.statistic);
}

/// Friedman: Q should be non-negative.
#[test]
fn friedman_non_negative() {
    let data: Vec<f64> = (0..15).map(|i| (i as f64 * 0.7).sin()).collect();
    let result = tambear::nonparametric::friedman_test(&data, 5, 3);
    assert!(result.statistic >= 0.0,
        "Friedman Q should be non-negative, got {}", result.statistic);
}

/// Friedman with 2 treatments: should be equivalent to sign test in spirit.
#[test]
fn friedman_two_treatments() {
    let data = vec![
        1.0, 5.0,
        2.0, 6.0,
        3.0, 4.0,
        1.0, 7.0,
    ];
    let result = tambear::nonparametric::friedman_test(&data, 4, 2);
    assert!(result.statistic.is_finite(),
        "Friedman with k=2 should be finite, got {}", result.statistic);
    // With treatment 2 consistently higher, should detect
    assert!(result.p_value < 0.2,
        "Friedman k=2 with effect: p={}", result.p_value);
}
