//! Adversarial tests for second wave of prerequisite methods:
//! Dunn's test, Tukey HSD, KMO+Bartlett, Hopkins, Cook's distance
//!
//! Also includes parameterization audit tests that verify user-tunable parameters
//! are not hardcoded.

use tambear::linear_algebra::Mat;
use tambear::descriptive::MomentStats;
use tambear::hypothesis::*;
use tambear::nonparametric::*;

// ═══════════════════════════════════════════════════════════════════════════
// DUNN'S TEST
// ═══════════════════════════════════════════════════════════════════════════

/// Dunn's with two groups: should produce exactly 1 comparison.
#[test]
fn dunn_two_groups() {
    let data = vec![1.0, 2.0, 3.0, 10.0, 11.0, 12.0];
    let group_sizes = vec![3, 3];
    let result = dunn_test(&data, &group_sizes);
    assert_eq!(result.len(), 1, "Two groups → 1 comparison, got {}", result.len());
    assert!(result[0].z_statistic.is_finite(),
        "Dunn z should be finite, got {}", result[0].z_statistic);
    assert!(result[0].p_value >= 0.0 && result[0].p_value <= 1.0,
        "Dunn p should be in [0,1], got {}", result[0].p_value);
}

/// Dunn's with identical groups: z = 0, p ≈ 1.
#[test]
fn dunn_identical_groups() {
    let data = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    let group_sizes = vec![3, 3];
    let result = dunn_test(&data, &group_sizes);
    assert_eq!(result.len(), 1);
    // Same data → same mean ranks → z ≈ 0
    assert!(result[0].z_statistic.abs() < 1.0,
        "Dunn identical groups z should be near 0, got {}", result[0].z_statistic);
}

/// Dunn's with single group: no pairwise comparisons.
#[test]
fn dunn_single_group() {
    let data = vec![1.0, 2.0, 3.0];
    let group_sizes = vec![3];
    let result = dunn_test(&data, &group_sizes);
    assert!(result.is_empty(), "Single group → 0 comparisons, got {}", result.len());
}

/// Dunn's with empty group: should not panic.
#[test]
fn dunn_empty_group() {
    let data = vec![1.0, 2.0, 3.0];
    let group_sizes = vec![3, 0, 0];
    let result = std::panic::catch_unwind(|| {
        dunn_test(&data, &group_sizes)
    });
    assert!(result.is_ok(), "Dunn should not panic with empty groups");
}

/// Dunn's with all tied data: ranks are all tied, SE should handle gracefully.
#[test]
fn dunn_all_tied() {
    let data = vec![5.0; 9];
    let group_sizes = vec![3, 3, 3];
    let result = std::panic::catch_unwind(|| {
        dunn_test(&data, &group_sizes)
    });
    assert!(result.is_ok(), "Dunn should not panic with all-tied data");
    let comparisons = result.unwrap();
    // All tied → tie correction = N*(N²-1)/12 → SE could be 0 → z = NaN
    for c in &comparisons {
        assert!(!c.z_statistic.is_infinite(),
            "Dunn all-tied z should not be Inf, got {}", c.z_statistic);
    }
}

/// Dunn's with many groups: k=5 → C(5,2) = 10 comparisons.
#[test]
fn dunn_five_groups() {
    let data: Vec<f64> = (0..50).map(|i| i as f64).collect();
    let group_sizes = vec![10, 10, 10, 10, 10];
    let result = dunn_test(&data, &group_sizes);
    assert_eq!(result.len(), 10, "5 groups → 10 comparisons, got {}", result.len());
}

// ═══════════════════════════════════════════════════════════════════════════
// TUKEY HSD
// ═══════════════════════════════════════════════════════════════════════════

/// Tukey with two groups: should match t-test in spirit (single comparison).
#[test]
fn tukey_two_groups() {
    let stats: Vec<MomentStats> = vec![
        tambear::descriptive::moments_ungrouped(&[1.0, 2.0, 3.0, 4.0, 5.0]),
        tambear::descriptive::moments_ungrouped(&[6.0, 7.0, 8.0, 9.0, 10.0]),
    ];
    let ms_error = 2.5; // typical within-group MS
    let df_error = 8.0;
    let result = tukey_hsd(&stats, ms_error, df_error);
    assert_eq!(result.len(), 1, "Two groups → 1 comparison");
    assert!(result[0].q_statistic > 0.0, "q should be positive");
    assert!(result[0].p_value.is_finite(), "p should be finite");
}

/// Tukey with all identical groups: q = 0, p ≈ 1.
#[test]
fn tukey_identical_groups() {
    let stats: Vec<MomentStats> = vec![
        tambear::descriptive::moments_ungrouped(&[5.0, 5.0, 5.0, 5.0, 5.0]),
        tambear::descriptive::moments_ungrouped(&[5.0, 5.0, 5.0, 5.0, 5.0]),
        tambear::descriptive::moments_ungrouped(&[5.0, 5.0, 5.0, 5.0, 5.0]),
    ];
    let result = tukey_hsd(&stats, 0.001, 12.0);
    for c in &result {
        assert!((c.mean_diff).abs() < 1e-10,
            "Tukey identical groups: mean_diff should be 0, got {}", c.mean_diff);
    }
}

/// Tukey: significant_at should be tunable (parameterization test).
#[test]
fn tukey_significant_at_custom_alpha() {
    let stats: Vec<MomentStats> = vec![
        tambear::descriptive::moments_ungrouped(&[1.0, 2.0, 3.0, 4.0, 5.0]),
        tambear::descriptive::moments_ungrouped(&[3.0, 4.0, 5.0, 6.0, 7.0]),
    ];
    let result = tukey_hsd(&stats, 2.5, 8.0);
    let c = &result[0];
    // The `significant` field uses hardcoded 0.05.
    // The `significant_at` method should allow custom alpha.
    if c.p_value > 0.01 && c.p_value < 0.10 {
        // This is a good range to test: significant at 0.10 but not at 0.01
        assert!(c.significant_at(0.10), "Should be significant at alpha=0.10");
        assert!(!c.significant_at(0.01), "Should not be significant at alpha=0.01");
    }
}

/// Tukey with zero ms_error: q = Inf.
#[test]
fn tukey_zero_mse() {
    let stats: Vec<MomentStats> = vec![
        tambear::descriptive::moments_ungrouped(&[1.0, 2.0, 3.0]),
        tambear::descriptive::moments_ungrouped(&[4.0, 5.0, 6.0]),
    ];
    let result = tukey_hsd(&stats, 0.0, 4.0);
    assert_eq!(result.len(), 1);
    // ms_error=0 → se=0 → q=Inf
    assert!(result[0].q_statistic.is_infinite(),
        "Tukey with ms_error=0 should have q=Inf, got {}", result[0].q_statistic);
}

// ═══════════════════════════════════════════════════════════════════════════
// KMO + BARTLETT
// ═══════════════════════════════════════════════════════════════════════════

/// KMO on identity correlation matrix: degenerate case (0/0).
/// Identity means no correlations AND no partial correlations → KMO = 0/0.
/// The implementation guards this as 1.0 (or 0.0); either is acceptable for degenerate input.
#[test]
fn kmo_identity_matrix() {
    let corr = Mat::eye(4);
    let result = tambear::factor_analysis::kmo_bartlett(&corr, 100);
    // Identity: r_ij = 0 and a_ij = 0 for all i≠j → KMO = 0/(0+0) = degenerate
    // Implementation returns 1.0 (denominator guard). This is the 0/0 case.
    assert!(result.kmo_overall.is_finite(),
        "KMO of identity should be finite (guarded 0/0), got {}", result.kmo_overall);
}

/// KMO on 2x2 matrix: minimal case.
#[test]
fn kmo_2x2() {
    let corr = Mat::from_vec(2, 2, vec![1.0, 0.8, 0.8, 1.0]);
    let result = tambear::factor_analysis::kmo_bartlett(&corr, 50);
    assert!(result.kmo_overall.is_finite(),
        "KMO 2x2 should be finite, got {}", result.kmo_overall);
    assert!(result.kmo_overall > 0.0,
        "KMO 2x2 with r=0.8 should be positive, got {}", result.kmo_overall);
    assert_eq!(result.kmo_per_variable.len(), 2);
}

/// Bartlett on identity: should NOT reject (no correlations to explain).
#[test]
fn bartlett_identity_not_reject() {
    let corr = Mat::eye(3);
    let result = tambear::factor_analysis::kmo_bartlett(&corr, 100);
    // Identity → ln|R| = 0 → Bartlett stat = 0 → p ≈ 1
    assert!((result.bartlett_statistic - 0.0).abs() < 1e-6,
        "Bartlett on identity should have stat=0, got {}", result.bartlett_statistic);
}

/// KMO on singular matrix: should handle gracefully (return 0 or NaN).
#[test]
fn kmo_singular_matrix() {
    // All rows identical → singular
    let corr = Mat::from_vec(3, 3, vec![
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
        1.0, 1.0, 1.0,
    ]);
    let result = tambear::factor_analysis::kmo_bartlett(&corr, 50);
    // Singular → Cholesky fails → kmo=0, bartlett=NaN
    assert!(result.kmo_overall == 0.0 || result.kmo_overall.is_nan(),
        "KMO singular should be 0 or NaN, got {}", result.kmo_overall);
}

// ═══════════════════════════════════════════════════════════════════════════
// HOPKINS STATISTIC
// ═══════════════════════════════════════════════════════════════════════════

/// Hopkins on clustered data: should be > 0.5 (clustering tendency).
#[test]
fn hopkins_clustered() {
    // Two tight clusters far apart
    let mut data = Vec::new();
    for _ in 0..20 { data.push(0.0); data.push(0.0); }
    for _ in 0..20 { data.push(100.0); data.push(100.0); }
    let h = tambear::clustering::hopkins_statistic(&data, 40, 2, 10, 42);
    assert!(h > 0.5,
        "Hopkins on clustered data should be > 0.5, got {}", h);
}

/// Hopkins on single point: n < 2 → returns 0.5.
#[test]
fn hopkins_single_point() {
    let data = vec![1.0, 2.0]; // 1 point in 2D
    let h = tambear::clustering::hopkins_statistic(&data, 1, 2, 1, 42);
    assert!((h - 0.5).abs() < 1e-10,
        "Hopkins with n=1 should return 0.5 (degenerate), got {}", h);
}

/// Hopkins with all identical points: u_sum = 0 → H = w / (w + 0) = 1.
#[test]
fn hopkins_identical_points() {
    let data = vec![5.0; 20]; // 10 identical points in 2D
    let h = tambear::clustering::hopkins_statistic(&data, 10, 2, 5, 42);
    // All data points at same location → u_sum = 0
    // w_sum from uniform random points to nearest data → some positive value
    // H = w_sum / (w_sum + 0) = 1.0 (if w_sum > 0) or 0.5 (if both 0)
    assert!(h >= 0.5 && h <= 1.0,
        "Hopkins identical should be in [0.5, 1.0], got {}", h);
}

/// Hopkins with m=0: returns 0.5.
#[test]
fn hopkins_m_zero() {
    let data = vec![1.0, 2.0, 3.0, 4.0]; // 2 points in 2D
    let h = tambear::clustering::hopkins_statistic(&data, 2, 2, 0, 42);
    assert!((h - 0.5).abs() < 1e-10,
        "Hopkins with m=0 should return 0.5, got {}", h);
}

// ═══════════════════════════════════════════════════════════════════════════
// COOK'S DISTANCE
// ═══════════════════════════════════════════════════════════════════════════

/// Cook's distance: no outliers → all distances small.
#[test]
fn cooks_no_outliers() {
    let n = 20;
    let mut x_data = vec![0.0; n * 2];
    let mut residuals = Vec::new();
    let mut rng = 12345u64;
    for i in 0..n {
        x_data[i * 2] = 1.0;
        x_data[i * 2 + 1] = i as f64;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        residuals.push((rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5);
    }
    let x = Mat::from_vec(n, 2, x_data);
    let result = cooks_distance(&x, &residuals);
    assert_eq!(result.cooks_distance.len(), n);
    assert_eq!(result.leverage.len(), n);
    // No outliers → Cook's distances should all be small
    let max_cook = result.cooks_distance.iter().cloned().fold(0.0_f64, f64::max);
    assert!(max_cook < 2.0,
        "No outliers: max Cook's D should be small, got {}", max_cook);
}

/// Cook's distance with high-leverage outlier: should detect it.
#[test]
fn cooks_outlier_detected() {
    let n = 20;
    let mut x_data = vec![0.0; n * 2];
    let mut residuals = vec![0.1; n]; // small residuals
    for i in 0..n {
        x_data[i * 2] = 1.0;
        x_data[i * 2 + 1] = i as f64;
    }
    // Make last point an extreme outlier
    residuals[n - 1] = 100.0; // huge residual
    let x = Mat::from_vec(n, 2, x_data);
    let result = cooks_distance(&x, &residuals);
    // Outlier should have much larger Cook's D
    let max_idx = result.cooks_distance.iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
    assert_eq!(max_idx, n - 1,
        "Outlier at index {} should have max Cook's D, but index {} does", n - 1, max_idx);
    assert!(result.n_influential >= 1,
        "Should detect at least 1 influential point, got {}", result.n_influential);
}

/// Cook's with zero residuals: all distances = 0.
#[test]
fn cooks_zero_residuals() {
    let n = 10;
    let mut x_data = vec![0.0; n * 2];
    for i in 0..n { x_data[i * 2] = 1.0; x_data[i * 2 + 1] = i as f64; }
    let x = Mat::from_vec(n, 2, x_data);
    let residuals = vec![0.0; n];
    let result = cooks_distance(&x, &residuals);
    for (i, &d) in result.cooks_distance.iter().enumerate() {
        assert!(!d.is_nan(),
            "Cook's D[{}] should not be NaN with zero residuals (0/0 guarded)", i);
    }
}

/// Cook's leverage should be in [0, 1] for well-conditioned X.
#[test]
fn cooks_leverage_range() {
    let n = 20;
    let mut x_data = vec![0.0; n * 2];
    for i in 0..n { x_data[i * 2] = 1.0; x_data[i * 2 + 1] = i as f64; }
    let x = Mat::from_vec(n, 2, x_data);
    let residuals: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
    let result = cooks_distance(&x, &residuals);
    for (i, &h) in result.leverage.iter().enumerate() {
        assert!(h >= 0.0 && h <= 1.0 + 1e-10,
            "Leverage[{}]={} should be in [0, 1]", i, h);
    }
    // Sum of leverage = p (trace of hat matrix)
    let lev_sum: f64 = result.leverage.iter().sum();
    assert!((lev_sum - 2.0).abs() < 0.1,
        "Sum of leverage should be p=2, got {}", lev_sum);
}

/// Cook's with singular X'X: fallback to equal leverage.
#[test]
fn cooks_singular_xtx() {
    let n = 5;
    // All rows identical → X'X is rank 1
    let x = Mat::from_vec(n, 2, vec![[1.0, 1.0]; n].into_iter().flatten().collect());
    let residuals = vec![1.0, -1.0, 0.5, -0.5, 0.0];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        cooks_distance(&x, &residuals)
    }));
    assert!(result.is_ok(), "Cook's should not panic with singular X'X");
}

// ═══════════════════════════════════════════════════════════════════════════
// PARAMETERIZATION AUDIT
// ═══════════════════════════════════════════════════════════════════════════

/// Cook's distance threshold is hardcoded at 4/n. Test that we get consistent
/// results and document the gap.
#[test]
fn param_audit_cooks_threshold() {
    let n = 20;
    let mut x_data = vec![0.0; n * 2];
    let mut residuals = vec![0.0; n];
    for i in 0..n {
        x_data[i * 2] = 1.0;
        x_data[i * 2 + 1] = i as f64;
        residuals[i] = if i == n - 1 { 10.0 } else { 0.1 };
    }
    let x = Mat::from_vec(n, 2, x_data);
    let result = cooks_distance(&x, &residuals);
    // The threshold is 4/n = 0.2. Verify the count uses this.
    let threshold = 4.0 / n as f64;
    let manual_count = result.cooks_distance.iter().filter(|&&d| d > threshold).count();
    assert_eq!(result.n_influential, manual_count,
        "n_influential should match 4/n threshold count: {} vs {}", result.n_influential, manual_count);
    // PARAMETERIZATION GAP: threshold should be a parameter, not hardcoded 4/n.
    // Some texts use 1.0 as threshold, or 4/(n-k-1), or percentile-based.
}

/// Tukey significant_at method exists and works.
#[test]
fn param_audit_tukey_significant_at() {
    let stats: Vec<MomentStats> = vec![
        tambear::descriptive::moments_ungrouped(&[1.0, 2.0, 3.0]),
        tambear::descriptive::moments_ungrouped(&[10.0, 11.0, 12.0]),
    ];
    let result = tukey_hsd(&stats, 1.0, 4.0);
    let c = &result[0];
    // Verify significant_at is available and works independently of hardcoded field
    let sig_005 = c.significant_at(0.05);
    let sig_001 = c.significant_at(0.01);
    assert_eq!(sig_005, c.p_value < 0.05, "significant_at(0.05) should match p < 0.05");
    assert_eq!(sig_001, c.p_value < 0.01, "significant_at(0.01) should match p < 0.01");
}
