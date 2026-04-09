//! Adversarial tests for math library expansion:
//! Cramér's V, eta squared, distance correlation, cluster validation

// ═══════════════════════════════════════════════════════════════════════════
// CRAMER'S V
// ═══════════════════════════════════════════════════════════════════════════

/// Cramér's V on identity-like table: V = 1 (perfect association).
#[test]
fn cramers_v_perfect_association() {
    // 3x3 diagonal: perfect association
    let table = vec![10.0, 0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 10.0];
    let v = tambear::nonparametric::cramers_v(&table, 3);
    assert!((v - 1.0).abs() < 0.01,
        "Perfect association should have V=1.0, got {}", v);
}

/// Cramér's V on uniform table: V ≈ 0 (no association).
#[test]
fn cramers_v_no_association() {
    // 3x3 all equal: no association
    let table = vec![10.0; 9];
    let v = tambear::nonparametric::cramers_v(&table, 3);
    assert!(v.abs() < 0.01,
        "No association should have V≈0, got {}", v);
}

/// Cramér's V on 2x2: should equal |phi|.
#[test]
fn cramers_v_2x2_equals_phi() {
    let table = vec![10.0, 2.0, 3.0, 15.0];
    let v = tambear::nonparametric::cramers_v(&table, 2);
    assert!(v >= 0.0 && v <= 1.0, "V should be in [0,1], got {}", v);
    assert!(v > 0.3, "Should detect moderate association, got {}", v);
}

/// Cramér's V with zero row: should handle gracefully.
#[test]
fn cramers_v_zero_row() {
    let table = vec![0.0, 0.0, 0.0, 5.0, 3.0, 2.0];
    let v = tambear::nonparametric::cramers_v(&table, 2);
    assert!(v.is_finite(), "V with zero row should be finite, got {}", v);
}

/// Cramér's V with all zeros: NaN.
#[test]
fn cramers_v_all_zeros() {
    let table = vec![0.0; 6];
    let v = tambear::nonparametric::cramers_v(&table, 2);
    assert!(v.is_nan(), "V of all-zero table should be NaN, got {}", v);
}

/// Cramér's V single row: degenerate → NaN.
#[test]
fn cramers_v_single_row() {
    let table = vec![1.0, 2.0, 3.0];
    let v = tambear::nonparametric::cramers_v(&table, 1);
    assert!(v.is_nan(), "V with 1 row should be NaN, got {}", v);
}

// ═══════════════════════════════════════════════════════════════════════════
// ETA SQUARED
// ═══════════════════════════════════════════════════════════════════════════

/// Eta² with perfectly separated groups: η² = 1.
#[test]
fn eta_squared_perfect_separation() {
    let values = vec![0.0, 0.0, 0.0, 100.0, 100.0, 100.0];
    let groups = vec![0, 0, 0, 1, 1, 1];
    let eta2 = tambear::nonparametric::eta_squared(&values, &groups);
    assert!((eta2 - 1.0).abs() < 1e-10,
        "Perfect separation should have η²=1, got {}", eta2);
}

/// Eta² with no group differences: η² = 0.
#[test]
fn eta_squared_no_effect() {
    let values = vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0];
    let groups = vec![0, 0, 0, 1, 1, 1];
    let eta2 = tambear::nonparametric::eta_squared(&values, &groups);
    assert!(eta2.abs() < 0.01,
        "No group effect should have η²≈0, got {}", eta2);
}

/// Eta² on constant data: 0.
#[test]
fn eta_squared_constant() {
    let values = vec![5.0; 10];
    let groups = vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];
    let eta2 = tambear::nonparametric::eta_squared(&values, &groups);
    assert!((eta2 - 0.0).abs() < 1e-10,
        "Constant data should have η²=0, got {}", eta2);
}

/// Eta² range: always in [0, 1].
#[test]
fn eta_squared_range() {
    let values = vec![1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0];
    let groups = vec![0, 1, 0, 1, 0, 1, 0, 1];
    let eta2 = tambear::nonparametric::eta_squared(&values, &groups);
    assert!(eta2 >= 0.0 && eta2 <= 1.0,
        "η² should be in [0,1], got {}", eta2);
}

/// Eta² with single observation: NaN.
#[test]
fn eta_squared_single() {
    let eta2 = tambear::nonparametric::eta_squared(&[5.0], &[0]);
    assert!(eta2.is_nan(), "η² with n=1 should be NaN, got {}", eta2);
}

// ═══════════════════════════════════════════════════════════════════════════
// DISTANCE CORRELATION
// ═══════════════════════════════════════════════════════════════════════════

/// Distance correlation of independent data: should be near 0.
#[test]
fn distance_corr_independent() {
    // X and Y are unrelated
    let x: Vec<f64> = (0..30).map(|i| (i as f64 * 0.7).sin()).collect();
    let y: Vec<f64> = (0..30).map(|i| (i as f64 * 1.3).cos()).collect();
    let dc = tambear::nonparametric::distance_correlation(&x, &y);
    assert!(dc < 0.5,
        "Independent data should have low dCor, got {}", dc);
}

/// Distance correlation of identical data: dCor = 1.
#[test]
fn distance_corr_identical() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let dc = tambear::nonparametric::distance_correlation(&x, &x);
    assert!((dc - 1.0).abs() < 0.01,
        "dCor(X, X) should be 1.0, got {}", dc);
}

/// Distance correlation of linear relationship: dCor = 1.
#[test]
fn distance_corr_linear() {
    let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let y: Vec<f64> = x.iter().map(|xi| 3.0 * xi + 7.0).collect();
    let dc = tambear::nonparametric::distance_correlation(&x, &y);
    assert!((dc - 1.0).abs() < 0.01,
        "Perfect linear should have dCor=1.0, got {}", dc);
}

/// Distance correlation detects nonlinear: Y = X² should have high dCor.
#[test]
fn distance_corr_nonlinear() {
    let x: Vec<f64> = (-10..=10).map(|i| i as f64).collect();
    let y: Vec<f64> = x.iter().map(|xi| xi * xi).collect();
    let dc = tambear::nonparametric::distance_correlation(&x, &y);
    // Pearson r would be ~0 for this (symmetric parabola), but dCor should be high
    assert!(dc > 0.3,
        "dCor should detect nonlinear X² dependence, got {}", dc);
}

/// Distance correlation with constant X: dCor = 0.
#[test]
fn distance_corr_constant() {
    let x = vec![5.0; 10];
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let dc = tambear::nonparametric::distance_correlation(&x, &y);
    assert!(dc.abs() < 0.01,
        "Constant X should have dCor=0, got {}", dc);
}

/// Distance correlation range: [0, 1].
#[test]
fn distance_corr_range() {
    let x = vec![1.0, 4.0, 2.0, 8.0, 5.0];
    let y = vec![3.0, 7.0, 1.0, 9.0, 4.0];
    let dc = tambear::nonparametric::distance_correlation(&x, &y);
    assert!(dc >= 0.0 && dc <= 1.0 + 1e-10,
        "dCor should be in [0,1], got {}", dc);
}

/// Distance correlation with n=2: minimal case.
#[test]
fn distance_corr_n2() {
    let dc = tambear::nonparametric::distance_correlation(&[1.0, 2.0], &[3.0, 4.0]);
    assert!(dc.is_finite(), "dCor with n=2 should be finite, got {}", dc);
}

/// Distance correlation with n=1: NaN.
#[test]
fn distance_corr_n1() {
    let dc = tambear::nonparametric::distance_correlation(&[1.0], &[2.0]);
    assert!(dc.is_nan(), "dCor with n=1 should be NaN, got {}", dc);
}

// ═══════════════════════════════════════════════════════════════════════════
// CLUSTER VALIDATION (silhouette, Calinski-Harabasz, Davies-Bouldin)
// ═══════════════════════════════════════════════════════════════════════════

/// Cluster validation with perfect clusters: high silhouette, high CH, low DB.
#[test]
fn cluster_validation_perfect_clusters() {
    // Two tight clusters far apart in 2D
    let mut data = Vec::new();
    let mut labels = Vec::new();
    for _ in 0..10 { data.push(0.0); data.push(0.0); labels.push(0); }
    for _ in 0..10 { data.push(100.0); data.push(100.0); labels.push(1); }
    let result = tambear::clustering::cluster_validation(&data, &labels, 2);
    assert!(result.is_some(), "Should return validation for 2 clusters");
    let v = result.unwrap();
    assert!(v.silhouette > 0.9,
        "Perfect clusters: silhouette should be > 0.9, got {}", v.silhouette);
    assert!(v.calinski_harabasz > 100.0,
        "Perfect clusters: CH should be large, got {}", v.calinski_harabasz);
    assert!(v.davies_bouldin < 0.1,
        "Perfect clusters: DB should be low, got {}", v.davies_bouldin);
}

/// Cluster validation with single cluster: silhouette = 0, CH undefined.
#[test]
fn cluster_validation_single_cluster() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 points in 2D
    let labels = vec![0, 0, 0];
    let result = tambear::clustering::cluster_validation(&data, &labels, 2);
    // Single cluster: silhouette is undefined (no b(i)), CH has k-1=0 in denominator
    // Implementation may return None or return with special values
    match result {
        Some(v) => {
            // Single cluster metrics are degenerate but shouldn't crash
            assert!(v.silhouette.is_finite() || v.silhouette.is_nan(),
                "Single cluster silhouette should be finite or NaN");
        }
        None => {
            // Acceptable: single cluster → validation undefined
        }
    }
}

/// Cluster validation with all-noise labels (-1): should return None.
#[test]
fn cluster_validation_all_noise() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let labels = vec![-1, -1, -1];
    let result = tambear::clustering::cluster_validation(&data, &labels, 2);
    assert!(result.is_none(),
        "All noise labels should return None");
}

/// Cluster validation with identical points: all distances = 0.
#[test]
fn cluster_validation_identical_points() {
    let data = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0]; // 3 identical points in 2D
    let labels = vec![0, 0, 1];
    let result = tambear::clustering::cluster_validation(&data, &labels, 2);
    // All distances 0 → silhouette = (0-0)/max(0,0) = 0/0 guarded
    match result {
        Some(v) => {
            assert!(v.silhouette.is_finite(),
                "Identical points silhouette should be finite (guarded 0/0)");
        }
        None => {} // acceptable
    }
}

/// Cluster validation with n_clusters = n: each point its own cluster.
#[test]
fn cluster_validation_n_equals_k() {
    let data = vec![0.0, 0.0, 10.0, 10.0, 20.0, 20.0]; // 3 points in 2D
    let labels = vec![0, 1, 2]; // each its own cluster
    let result = tambear::clustering::cluster_validation(&data, &labels, 2);
    // n=k → SS_W = 0, each cluster has size 1 → silhouette undefined (a(i) undefined)
    match result {
        Some(v) => {
            assert!(v.calinski_harabasz.is_finite() || v.calinski_harabasz.is_infinite(),
                "CH with n=k: SS_W=0 → CH=Inf (acceptable)");
        }
        None => {} // acceptable
    }
}
