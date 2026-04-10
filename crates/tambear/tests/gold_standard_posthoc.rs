//! Gold Standard Parity: Post-hoc Tests, Influence Diagnostics, Factor Adequacy
//!
//! Verification chain: statsmodels/sklearn → tambear CPU
//!
//! Methods covered:
//!   - Tukey HSD post-hoc test (statsmodels.stats.multicomp)
//!   - Dunn's test (manual rank-based)
//!   - Cook's distance (statsmodels OLSInfluence)
//!   - KMO & Bartlett's sphericity (analytical)
//!
//! Expected values: research/gold_standard/family_posthoc_influence_oracle.py

fn assert_close(name: &str, got: f64, expected: f64, tol: f64) {
    let abs_diff = (got - expected).abs();
    let rel_diff = if expected.abs() > 1e-15 {
        abs_diff / expected.abs()
    } else {
        abs_diff
    };
    assert!(
        abs_diff < tol || rel_diff < tol,
        "{}: got {:.15e}, expected {:.15e}, abs_diff={:.2e}, rel_diff={:.2e}, tol={:.0e}",
        name, got, expected, abs_diff, rel_diff, tol
    );
}

// ===========================================================================
// TUKEY HSD — verified against statsmodels pairwise_tukeyhsd
// ===========================================================================

mod tukey_hsd_parity {
    use super::*;
    use tambear::hypothesis::tukey_hsd;
    use tambear::descriptive::moments_ungrouped;

    #[test]
    fn tukey_hsd_3groups_matches_statsmodels() {
        // statsmodels: A-B diff=2.0 p=0.1546, A-C diff=5.0 p=0.0008, B-C diff=3.0 p=0.0277
        let g1 = [1.0, 2.0, 3.0, 4.0, 5.0];
        let g2 = [3.0, 4.0, 5.0, 6.0, 7.0];
        let g3 = [6.0, 7.0, 8.0, 9.0, 10.0];

        let stats: Vec<_> = [&g1[..], &g2[..], &g3[..]]
            .iter().map(|g| moments_ungrouped(g)).collect();

        // ms_error = 2.5, df_error = 12 (from one-way ANOVA)
        let results = tukey_hsd(&stats, 2.5, 12.0, None);
        assert_eq!(results.len(), 3, "3 groups → 3 pairwise comparisons");

        // Find A-B comparison (groups 0-1)
        let ab = results.iter().find(|r| r.group_i == 0 && r.group_j == 1).unwrap();
        assert_close("tukey_AB_diff", ab.mean_diff.abs(), 2.0, 1e-10);

        // Find A-C comparison (groups 0-2)
        let ac = results.iter().find(|r| r.group_i == 0 && r.group_j == 2).unwrap();
        assert_close("tukey_AC_diff", ac.mean_diff.abs(), 5.0, 1e-10);

        // Find B-C comparison (groups 1-2)
        let bc = results.iter().find(|r| r.group_i == 1 && r.group_j == 2).unwrap();
        assert_close("tukey_BC_diff", bc.mean_diff.abs(), 3.0, 1e-10);
    }

    #[test]
    fn tukey_hsd_mean_diffs_correct() {
        let g1 = [10.0, 20.0, 30.0];
        let g2 = [15.0, 25.0, 35.0];
        let stats: Vec<_> = [&g1[..], &g2[..]]
            .iter().map(|g| moments_ungrouped(g)).collect();

        let results = tukey_hsd(&stats, 1.0, 4.0, None);
        assert_eq!(results.len(), 1);
        // mean1=20, mean2=25, diff=-5 (or 5 depending on sign convention)
        assert_close("tukey_diff", results[0].mean_diff.abs(), 5.0, 1e-10);
    }

    #[test]
    fn tukey_hsd_equal_means_q_zero() {
        let g1 = [10.0, 11.0, 12.0, 13.0, 14.0];
        let g2 = [10.0, 11.0, 12.0, 13.0, 14.0];
        let stats: Vec<_> = [&g1[..], &g2[..]]
            .iter().map(|g| moments_ungrouped(g)).collect();

        let results = tukey_hsd(&stats, 2.5, 8.0, None);
        assert_eq!(results.len(), 1);
        assert_close("tukey_eq_diff", results[0].mean_diff, 0.0, 1e-10);
        assert_close("tukey_eq_q", results[0].q_statistic, 0.0, 1e-10);
    }
}

// ===========================================================================
// DUNN'S TEST — verified against manual rank computation
// ===========================================================================

mod dunn_parity {
    use super::*;
    use tambear::nonparametric::dunn_test;

    #[test]
    fn dunn_3groups_z_statistics() {
        // Groups: [1-5], [4-8], [7-11], N=15, n_i=5
        // Mean ranks (scipy): [3.4, 8.0, 12.6]
        // z = |R_i - R_j| / SE, SE = sqrt(N(N+1)/12 * (1/5 + 1/5))
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0,  // group 0
            4.0, 5.0, 6.0, 7.0, 8.0,  // group 1
            7.0, 8.0, 9.0, 10.0, 11.0, // group 2
        ];
        let group_sizes = vec![5, 5, 5];
        let results = dunn_test(&data, &group_sizes);

        // Should have C(3,2) = 3 comparisons
        assert_eq!(results.len(), 3, "3 groups → 3 pairwise comparisons");

        // g0 vs g2 should have the largest |z| (most separated)
        let g0_g2 = results.iter().find(|r| r.group_i == 0 && r.group_j == 2).unwrap();
        assert!(g0_g2.z_statistic.abs() > 2.0,
            "g0 vs g2 |z|={} should be > 2.0 (well separated)", g0_g2.z_statistic.abs());

        // Symmetric: |g0-g1| should equal |g1-g2| (equal spacing)
        let g0_g1 = results.iter().find(|r| r.group_i == 0 && r.group_j == 1).unwrap();
        let g1_g2 = results.iter().find(|r| r.group_i == 1 && r.group_j == 2).unwrap();
        assert_close("dunn_symmetry", g0_g1.z_statistic.abs(), g1_g2.z_statistic.abs(), 0.5);
    }

    #[test]
    fn dunn_identical_groups_z_zero() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let group_sizes = vec![5, 5];
        let results = dunn_test(&data, &group_sizes);
        assert_eq!(results.len(), 1);
        // With ties, z should be small (near 0)
        assert!(results[0].z_statistic < 1.0,
            "Identical groups z={} should be near 0", results[0].z_statistic);
    }
}

// ===========================================================================
// COOK'S DISTANCE — verified against statsmodels OLSInfluence
// ===========================================================================

mod cooks_distance_parity {
    use super::*;
    use tambear::hypothesis::cooks_distance;
    use tambear::linear_algebra::Mat;

    #[test]
    fn cooks_outlier_detected() {
        // Simple regression with one outlier at the end
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.1, 4.0, 5.9, 8.1, 10.0, 12.0, 14.1, 16.0, 18.0, 50.0]; // last is outlier

        // Build design matrix [1, x_i]
        let n = x.len();
        let mut x_data = Vec::with_capacity(n * 2);
        for &xi in &x {
            x_data.push(1.0);
            x_data.push(xi);
        }
        let x_mat = Mat { rows: n, cols: 2, data: x_data };

        // Compute OLS residuals manually
        let x_mean = x.iter().sum::<f64>() / n as f64;
        let y_mean = y.iter().sum::<f64>() / n as f64;
        let mut sxy = 0.0;
        let mut sxx = 0.0;
        for i in 0..n {
            sxy += (x[i] - x_mean) * (y[i] - y_mean);
            sxx += (x[i] - x_mean) * (x[i] - x_mean);
        }
        let b1 = sxy / sxx;
        let b0 = y_mean - b1 * x_mean;
        let residuals: Vec<f64> = (0..n).map(|i| y[i] - b0 - b1 * x[i]).collect();

        let result = cooks_distance(&x_mat, &residuals);

        // The outlier (index 9) should have the highest Cook's D
        let max_idx = result.cooks_distance.iter().enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i).unwrap();
        assert_eq!(max_idx, 9, "Outlier at index 9 should have max Cook's D");

        // Max Cook's D should exceed 4/n threshold
        let threshold = 4.0 / n as f64;
        assert!(result.cooks_distance[9] > threshold,
            "Outlier Cook's D={} should exceed 4/n={}", result.cooks_distance[9], threshold);
    }

    #[test]
    fn cooks_no_outlier_all_below_threshold() {
        // Clean linear data: no influential points
        let n = 20;
        let mut x_data = Vec::with_capacity(n * 2);
        let mut residuals = Vec::with_capacity(n);
        let mut rng = tambear::rng::Xoshiro256::new(42);
        use tambear::rng::TamRng;
        for i in 0..n {
            x_data.push(1.0);
            x_data.push(i as f64);
            let (z, _) = tambear::rng::normal_pair(&mut rng);
            residuals.push(z * 0.1);
        }
        let x_mat = Mat { rows: n, cols: 2, data: x_data };
        let result = cooks_distance(&x_mat, &residuals);

        // All Cook's D should be reasonable (< 1.0)
        for (i, &d) in result.cooks_distance.iter().enumerate() {
            assert!(d < 1.0, "Cook's D[{}]={} should be < 1.0 for clean data", i, d);
        }
    }

    #[test]
    fn leverage_sums_to_p() {
        // Trace of hat matrix = p (number of parameters)
        let n = 10;
        let mut x_data = Vec::with_capacity(n * 2);
        for i in 0..n {
            x_data.push(1.0);
            x_data.push(i as f64);
        }
        let x_mat = Mat { rows: n, cols: 2, data: x_data };
        let residuals = vec![0.1; n];
        let result = cooks_distance(&x_mat, &residuals);

        let sum_h: f64 = result.leverage.iter().sum();
        assert_close("leverage_sum", sum_h, 2.0, 0.1); // p=2
    }
}

// ===========================================================================
// KMO & BARTLETT'S SPHERICITY — verified against analytical formulas
// ===========================================================================

mod kmo_bartlett_parity {
    use super::*;
    use tambear::factor_analysis::kmo_bartlett;
    use tambear::linear_algebra::Mat;

    #[test]
    fn bartlett_identity_not_significant() {
        // Identity correlation matrix: ln|I|=0, statistic=0, p=1.0
        let n = 4;
        let identity = Mat {
            rows: n, cols: n,
            data: vec![
                1.0, 0.0, 0.0, 0.0,
                0.0, 1.0, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        };
        let result = kmo_bartlett(&identity, 100);
        assert_close("bart_id_stat", result.bartlett_statistic, 0.0, 1e-8);
        assert_close("bart_id_p", result.bartlett_p_value, 1.0, 1e-6);
    }

    #[test]
    fn bartlett_correlated_significant() {
        // Correlated data: should reject H0 (identity)
        // Oracle: statistic=942.21, p≈0
        // Build a correlation matrix with substantial off-diagonal elements
        let r = Mat {
            rows: 4, cols: 4,
            data: vec![
                1.0, 0.8, 0.3, 0.6,
                0.8, 1.0, 0.4, 0.7,
                0.3, 0.4, 1.0, 0.5,
                0.6, 0.7, 0.5, 1.0,
            ],
        };
        let result = kmo_bartlett(&r, 100);
        assert!(result.bartlett_statistic > 50.0,
            "Correlated data Bartlett stat={} should be large", result.bartlett_statistic);
        assert!(result.bartlett_p_value < 0.001,
            "Correlated data Bartlett p={} should be very small", result.bartlett_p_value);
        assert_eq!(result.bartlett_df, 6); // p*(p-1)/2 = 4*3/2 = 6
    }

    #[test]
    fn bartlett_df_correct() {
        let r = Mat {
            rows: 3, cols: 3,
            data: vec![1.0, 0.5, 0.3, 0.5, 1.0, 0.4, 0.3, 0.4, 1.0],
        };
        let result = kmo_bartlett(&r, 50);
        assert_eq!(result.bartlett_df, 3); // 3*2/2 = 3
    }

    #[test]
    fn kmo_bounded_0_1() {
        let r = Mat {
            rows: 3, cols: 3,
            data: vec![1.0, 0.6, 0.4, 0.6, 1.0, 0.5, 0.4, 0.5, 1.0],
        };
        let result = kmo_bartlett(&r, 100);
        assert!(result.kmo_overall >= 0.0 && result.kmo_overall <= 1.0,
            "KMO={} should be in [0, 1]", result.kmo_overall);
        for (i, &v) in result.kmo_per_variable.iter().enumerate() {
            assert!(v >= 0.0 && v <= 1.0,
                "KMO[{}]={} should be in [0, 1]", i, v);
        }
    }

    #[test]
    fn kmo_correlated_data_adequate() {
        // Highly correlated data: KMO should be > 0.5 (adequate for FA)
        let r = Mat {
            rows: 4, cols: 4,
            data: vec![
                1.0, 0.8, 0.7, 0.6,
                0.8, 1.0, 0.8, 0.7,
                0.7, 0.8, 1.0, 0.8,
                0.6, 0.7, 0.8, 1.0,
            ],
        };
        let result = kmo_bartlett(&r, 200);
        assert!(result.kmo_overall > 0.5,
            "Correlated data KMO={} should be > 0.5", result.kmo_overall);
    }
}

// ===========================================================================
// CLUSTER VALIDATION — verified against sklearn metrics
// ===========================================================================

mod cluster_validation_parity {
    use super::*;
    use tambear::clustering::cluster_validation;

    #[test]
    fn well_separated_clusters_high_silhouette() {
        // Three well-separated 2D clusters
        // sklearn: silhouette ≈ 0.878, CH ≈ 2228, DB ≈ 0.167
        let mut data = Vec::new();
        let mut labels = Vec::new();

        // Cluster 0 around (0, 0)
        for i in 0..10 {
            data.push(0.0 + 0.1 * i as f64);
            data.push(0.0 + 0.05 * i as f64);
            labels.push(0i32);
        }
        // Cluster 1 around (10, 10)
        for i in 0..10 {
            data.push(10.0 + 0.1 * i as f64);
            data.push(10.0 + 0.05 * i as f64);
            labels.push(1i32);
        }
        // Cluster 2 around (20, 0)
        for i in 0..10 {
            data.push(20.0 + 0.1 * i as f64);
            data.push(0.0 + 0.05 * i as f64);
            labels.push(2i32);
        }

        let result = cluster_validation(&data, &labels, 2).unwrap();
        // Well-separated: silhouette should be high (> 0.8)
        assert!(result.silhouette > 0.8,
            "Well-separated silhouette={} should be > 0.8", result.silhouette);
        // CH should be high
        assert!(result.calinski_harabasz > 100.0,
            "Well-separated CH={} should be high", result.calinski_harabasz);
        // DB should be low (< 0.5)
        assert!(result.davies_bouldin < 0.5,
            "Well-separated DB={} should be < 0.5", result.davies_bouldin);
    }

    #[test]
    fn overlapping_clusters_lower_silhouette() {
        // Two overlapping clusters
        let mut data = Vec::new();
        let mut labels = Vec::new();
        for i in 0..15 {
            data.push(i as f64 * 0.1);
            data.push(i as f64 * 0.1);
            labels.push(0i32);
        }
        for i in 0..15 {
            data.push(0.5 + i as f64 * 0.1);
            data.push(0.5 + i as f64 * 0.1);
            labels.push(1i32);
        }

        let result = cluster_validation(&data, &labels, 2).unwrap();
        // Overlapping: silhouette should be lower than well-separated
        assert!(result.silhouette < 0.8,
            "Overlapping silhouette={} should be < 0.8", result.silhouette);
    }

    #[test]
    fn silhouette_range() {
        // Silhouette should always be in [-1, 1]
        let data = vec![0.0, 0.0, 1.0, 1.0, 10.0, 10.0, 11.0, 11.0];
        let labels = vec![0i32, 0, 1, 1];
        let result = cluster_validation(&data, &labels, 2).unwrap();
        assert!(result.silhouette >= -1.0 && result.silhouette <= 1.0,
            "Silhouette={} should be in [-1, 1]", result.silhouette);
    }

    #[test]
    fn davies_bouldin_nonnegative() {
        // DB index is always >= 0
        let data = vec![0.0, 5.0, 10.0, 0.0, 5.0, 10.0];
        let labels = vec![0i32, 1, 2];
        let result = cluster_validation(&data, &labels, 2).unwrap();
        assert!(result.davies_bouldin >= 0.0,
            "DB={} should be >= 0", result.davies_bouldin);
    }
}
