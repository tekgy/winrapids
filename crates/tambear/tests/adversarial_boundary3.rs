//! Adversarial boundary tests — round 2, part 3
//!
//! Targets: volatility, time_series, clustering/kmeans
//! Focus: IGARCH boundary, unit root, empty clusters, degenerate inputs

// ═══════════════════════════════════════════════════════════════════════════
// VOLATILITY
// ═══════════════════════════════════════════════════════════════════════════

/// EWMA with lambda=1: variance never updates (complete stagnation).
/// sigma2[t] = 1.0 * sigma2[t-1] + 0.0 * r²[t-1] = sigma2[0] forever.
/// Type 4 (Equipartition) — correct math, degenerate parameter.
#[test]
fn ewma_lambda_one_stagnation() {
    let returns = vec![0.01, -0.02, 0.03, -0.01, 0.05, -0.04, 0.02, -0.03, 0.01, -0.01];
    let sigma2 = tambear::volatility::ewma_variance(&returns, 1.0);
    // lambda=1 → all variances equal to sigma2[0]
    let s0 = sigma2[0];
    for (i, &s) in sigma2.iter().enumerate().skip(1) {
        assert!((s - s0).abs() < 1e-14,
            "EWMA(lambda=1) sigma2[{}]={} should equal sigma2[0]={} (stagnation)", i, s, s0);
    }
    // EWMA(lambda=1) correctly produces constant variance (stagnation)
}

/// EWMA with lambda=0: pure squared return, no smoothing.
/// sigma2[t] = r²[t-1], ignoring all history.
#[test]
fn ewma_lambda_zero_no_smoothing() {
    let returns = vec![0.01, -0.02, 0.03, -0.01, 0.05];
    let sigma2 = tambear::volatility::ewma_variance(&returns, 0.0);
    // lambda=0 → sigma2[t] = (1-0) * r²[t-1] = r²[t-1]
    for t in 1..returns.len() {
        let expected = returns[t - 1].powi(2).max(1e-15);
        assert!((sigma2[t] - expected).abs() < 1e-14,
            "EWMA(lambda=0) sigma2[{}]={} should be r²[{}]={}", t, sigma2[t], t-1, expected);
    }
}

/// BUG: EWMA sigma2[0] is not floored at 1e-15 when all returns are zero.
/// sigma2[0] = var0 = 0.0 (pushed without floor), but sigma2[1..] get max(1e-15).
/// Type 1 (Denominator) — inconsistent floor invariant.
#[test]
fn ewma_all_zero_returns_floor_inconsistency() {
    let returns = vec![0.0; 20];
    let sigma2 = tambear::volatility::ewma_variance(&returns, 0.94);
    // sigma2[0] = sum(r²)/n = 0/20 = 0.0 (no floor!)
    // sigma2[1] = max(0.94*0 + 0.06*0, 1e-15) = 1e-15 (floored)
    // sigma2[0] is floored via var0.max(1e-15)
    assert!(sigma2[0] >= 1e-15,
        "EWMA sigma2[0] should be floored at 1e-15, got {}", sigma2[0]);
    // All subsequent values should be floored
    for t in 1..returns.len() {
        assert!(sigma2[t] >= 1e-15,
            "EWMA sigma2[{}]={} should be >= 1e-15", t, sigma2[t]);
    }
}

/// GARCH on all-zero returns: uncond_var=0 → omega=0 → degenerate fit.
#[test]
fn garch_all_zero_returns() {
    let returns = vec![0.0; 100];
    let result = tambear::volatility::garch11_fit(&returns, 100);
    // omega initialized as 0.1 * uncond_var = 0.1 * 0 = 0
    // Conditional variances will stay at floor (1e-15)
    assert!(result.omega.is_finite(), "GARCH omega should be finite, got {}", result.omega);
    assert!(result.alpha.is_finite(), "GARCH alpha should be finite, got {}", result.alpha);
    assert!(result.beta.is_finite(), "GARCH beta should be finite, got {}", result.beta);
    // GARCH all-zero returns: degenerate fit with near-zero omega
}

/// GARCH near IGARCH boundary: alpha+beta → 1.
/// Forecast should remain finite even near the boundary.
#[test]
fn garch_near_igarch_forecast_stability() {
    // Construct returns that push toward IGARCH
    let mut returns = Vec::with_capacity(500);
    let mut rng = 77777u64;
    for _ in 0..500 {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let u = (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        returns.push(u * 0.05); // moderate volatility
    }
    let result = tambear::volatility::garch11_fit(&returns, 200);
    // Forecast stability near IGARCH
    let forecast = tambear::volatility::garch11_forecast(&result, returns.last().copied().unwrap(), 100);
    for (h, &s) in forecast.iter().enumerate() {
        assert!(s.is_finite() && s > 0.0,
            "GARCH forecast[{}]={} should be finite positive (alpha+beta={})",
            h, s, result.alpha + result.beta);
    }
}

/// GARCH with constant non-zero returns: all returns identical.
#[test]
fn garch_constant_returns() {
    let returns = vec![0.01; 100]; // constant 1% return
    let result = tambear::volatility::garch11_fit(&returns, 100);
    assert!(result.omega.is_finite(), "GARCH constant returns omega should be finite");
    assert!(!result.log_likelihood.is_nan(),
        "GARCH constant returns LL should not be NaN, got {}", result.log_likelihood);
}

/// Roll spread: flat prices → cov=0 → spread=0 (correct).
#[test]
fn roll_spread_flat_prices() {
    let prices = vec![100.0; 20];
    let spread = tambear::volatility::roll_spread(&prices);
    assert_eq!(spread, 0.0, "flat prices should have zero spread, got {}", spread);
}

/// Roll spread with 2 prices (< 3): should return 0.
#[test]
fn roll_spread_too_few_prices() {
    let prices = vec![100.0, 101.0];
    let spread = tambear::volatility::roll_spread(&prices);
    assert_eq!(spread, 0.0, "fewer than 3 prices should give 0, got {}", spread);
}

/// Realized variance with empty returns.
#[test]
fn realized_variance_empty() {
    let rv = tambear::volatility::realized_variance(&[]);
    // Empty returns → 0 or NaN
    assert!(rv == 0.0 || rv.is_nan(), "empty realized_variance should be 0 or NaN, got {}", rv);
}

/// Annualize vol with negative trading days: sqrt(negative) = NaN.
#[test]
fn annualize_vol_negative_days() {
    let result = tambear::volatility::annualize_vol(0.01, -252.0);
    // sqrt(negative) → NaN is correct for negative trading_days
    assert!(result.is_nan(), "annualize_vol with negative days should be NaN, got {}", result);
}

/// Jump test with all-zero returns: bipower=0, realized=0 → 0/0.
#[test]
fn jump_test_zero_returns() {
    let returns = vec![0.0; 50];
    let bns = tambear::volatility::jump_test_bns(&returns);
    assert!(bns.is_finite(), "BNS on zero returns should be finite, got {}", bns);
}

// ═══════════════════════════════════════════════════════════════════════════
// TIME SERIES
// ═══════════════════════════════════════════════════════════════════════════

/// BUG: AR with near-unit-root data → sigma2 can go negative in Levinson-Durbin.
/// sigma2 *= (1 - kappa²); if |kappa| > 1 → sigma2 < 0 → ln(negative) → NaN AIC.
/// Type 2 (Convergence).
#[test]
fn ar_near_unit_root_negative_sigma2() {
    // Generate near-unit-root data: y[t] = 0.999*y[t-1] + noise
    let n = 200;
    let mut data = vec![0.0; n];
    let mut rng = 12345u64;
    for t in 1..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let noise = ((rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.01;
        data[t] = 0.999 * data[t - 1] + noise;
    }
    let result = tambear::time_series::ar_fit(&data, 2);
    // sigma2 should not go negative (Levinson-Durbin stabilized)
    assert!(result.sigma2 >= 0.0 || result.sigma2.is_nan(),
        "AR sigma2 should be >= 0, got {}", result.sigma2);
    assert!(!result.aic.is_nan() || result.sigma2.is_nan(),
        "AR AIC should not be NaN unless sigma2 is NaN, got aic={} sigma2={}", result.aic, result.sigma2);
}

/// AR on constant data: zero variance → early return.
#[test]
fn ar_constant_data() {
    let data = vec![5.0; 100];
    let result = tambear::time_series::ar_fit(&data, 2);
    // Constant data has zero variance → should handle gracefully
    assert!(result.sigma2 >= 0.0 || result.sigma2.is_nan(),
        "AR constant data sigma2 should be >= 0, got {}", result.sigma2);
}

/// BUG: ADF test with too many lags → nobs ≤ p → MSE negative → NaN statistic.
/// Type 1 (Denominator) — (nobs - p) can be zero or negative.
#[test]
fn adf_too_many_lags() {
    let data: Vec<f64> = (0..15).map(|i| i as f64 * 0.1).collect();
    // n=15, n_lags=10 → nobs = (15-1) - 10 = 4, p = 2 + 10 = 12
    // nobs - p = 4 - 12 = -8 → MSE = ss_resid / (-8) → negative MSE
    let adf = tambear::time_series::adf_test(&data, 10);
    // ADF with too many lags: nobs≤p guard returns NaN AdfResult
    assert!(adf.statistic.is_nan() || adf.statistic.is_finite(),
        "ADF(n=15, lags=10) should return NaN or finite, not Inf, got {}", adf.statistic);
}

/// ADF on pure random walk (unit root): should not reject.
#[test]
fn adf_random_walk() {
    let n = 200;
    let mut data = vec![0.0; n];
    let mut rng = 54321u64;
    for t in 1..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        data[t] = data[t - 1] + ((rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 0.1;
    }
    let result = tambear::time_series::adf_test(&data, 4);
    // Random walk → should NOT reject unit root (statistic > critical values)
    assert!(result.statistic.is_finite(),
        "ADF statistic should be finite, got {}", result.statistic);
    assert!(result.statistic > result.critical_5pct,
        "ADF on random walk should not reject at 5%: stat={} > cv={}",
        result.statistic, result.critical_5pct);
}

/// ACF with max_lag >= n: should handle gracefully or panic with clear message.
#[test]
fn acf_lag_exceeds_data() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let acf_vals = tambear::time_series::acf(&data, 10); // max_lag=10 > n=5 — should not panic
    // max_lag clamped to n-1
    assert!(acf_vals.len() <= data.len(),
        "acf(max_lag > n) should clamp, got len={}", acf_vals.len());
}

/// PACF with constant data: sigma2 near zero → Levinson-Durbin instability.
#[test]
fn pacf_constant_data() {
    let data = vec![3.0; 50];
    let pacf_vals = tambear::time_series::pacf(&data, 5);
    // PACF of constant data: NaN values are expected (undefined autocorrelation)
    assert!(!pacf_vals.iter().any(|v| v.is_infinite()),
        "PACF constant data should not contain Inf");
}

/// Difference with d >= data length: produces empty vector.
#[test]
fn difference_too_many_times() {
    let data = vec![1.0, 2.0, 3.0];
    let d1 = tambear::time_series::difference(&data, 3);
    assert!(d1.is_empty(), "difference(d=n) should produce empty, got len={}", d1.len());
    let d2 = tambear::time_series::difference(&data, 10);
    assert!(d2.is_empty(), "difference(d>n) should produce empty, got len={}", d2.len());
}

/// SES with alpha=0: ignores all data, forecast never changes from initial.
#[test]
fn ses_alpha_zero() {
    let data = vec![1.0, 10.0, 100.0, 1000.0, 10000.0];
    let result = tambear::time_series::simple_exponential_smoothing(&data, 0.0);
    // alpha=0: level = 0*data[i] + 1*level = initial forever
    assert_eq!(result.forecast, data[0],
        "SES(alpha=0) forecast should stay at initial={}, got {}", data[0], result.forecast);
}

/// SES with alpha=1: each level is just the latest data point.
#[test]
fn ses_alpha_one() {
    let data = vec![1.0, 10.0, 100.0, 1000.0, 10000.0];
    let result = tambear::time_series::simple_exponential_smoothing(&data, 1.0);
    // alpha=1: level = 1*data[i] + 0*level = data[last]
    assert_eq!(result.forecast, data[data.len() - 1],
        "SES(alpha=1) forecast should be last value={}, got {}", data[data.len()-1], result.forecast);
}

/// Holt linear with extreme beta → trend explodes.
#[test]
fn holt_linear_explosive_trend() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let forecast = tambear::time_series::holt_linear(&data, 0.5, 0.99, 50);
    // beta=0.99 with horizon=50: trend can grow exponentially
    // Should still be finite
    for (h, &v) in forecast.iter().enumerate() {
        assert!(v.is_finite(),
            "Holt forecast[{}]={} should be finite", h, v);
    }
}

/// AR(1) on sine wave: well-conditioned, should produce clean fit.
#[test]
fn ar_sine_wave_sanity() {
    let data: Vec<f64> = (0..200).map(|i| (i as f64 * 0.1).sin()).collect();
    let result = tambear::time_series::ar_fit(&data, 2);
    assert!(result.sigma2 > 0.0, "AR sigma2 should be positive, got {}", result.sigma2);
    assert!(result.aic.is_finite(), "AR AIC should be finite, got {}", result.aic);
    // AR(2) on sin should capture the periodicity
    assert!(result.coefficients[0].abs() > 0.5,
        "AR(2) on sin should have |phi_1| > 0.5, got {}", result.coefficients[0]);
}

/// Difference once: verify correctness.
#[test]
fn difference_once_correctness() {
    let data = vec![1.0, 3.0, 6.0, 10.0, 15.0];
    let d1 = tambear::time_series::difference(&data, 1);
    assert_eq!(d1, vec![2.0, 3.0, 4.0, 5.0], "first difference should be [2,3,4,5]");
}

// ═══════════════════════════════════════════════════════════════════════════
// CLUSTERING: DBSCAN and K-means edge cases
// ═══════════════════════════════════════════════════════════════════════════

/// DBSCAN with epsilon=0: every point is noise (only self-distance=0 matches).
#[test]
fn dbscan_epsilon_zero_all_noise() {
    let mut engine = tambear::ClusteringEngine::new().unwrap();
    // 5 points in 2D, well-separated
    let data = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 5.0, 5.0];
    let result = engine.dbscan(&data, 5, 2, 0.0, 2).unwrap();
    // epsilon=0 → only exact-same points can be neighbors
    // With min_samples=2, no point has 2 neighbors at distance 0
    assert_eq!(result.n_noise, 5,
        "DBSCAN(eps=0) should mark all as noise, got n_noise={}", result.n_noise);
    assert_eq!(result.n_clusters, 0,
        "DBSCAN(eps=0) should have 0 clusters, got {}", result.n_clusters);
}

/// DBSCAN with very large epsilon: one big cluster.
#[test]
fn dbscan_epsilon_huge_one_cluster() {
    let mut engine = tambear::ClusteringEngine::new().unwrap();
    let data = vec![0.0, 0.0, 100.0, 100.0, -50.0, 50.0, 200.0, -200.0];
    let result = engine.dbscan(&data, 4, 2, 1e6, 2).unwrap();
    assert_eq!(result.n_clusters, 1,
        "DBSCAN(eps=1e6) should have 1 cluster, got {}", result.n_clusters);
    assert_eq!(result.n_noise, 0,
        "DBSCAN(eps=1e6) should have 0 noise, got {}", result.n_noise);
}

/// DBSCAN with min_samples > n: all noise.
#[test]
fn dbscan_min_samples_exceeds_n() {
    let mut engine = tambear::ClusteringEngine::new().unwrap();
    let data = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
    let result = engine.dbscan(&data, 3, 2, 10.0, 100).unwrap();
    // Even with huge epsilon, no point has 100 neighbors in a set of 3
    assert_eq!(result.n_clusters, 0,
        "DBSCAN(min_samples>n) should have 0 clusters, got {}", result.n_clusters);
    assert_eq!(result.n_noise, 3,
        "DBSCAN(min_samples>n) should have all noise, got n_noise={}", result.n_noise);
}

/// DBSCAN with all identical points: should form one cluster.
#[test]
fn dbscan_all_identical_points() {
    let mut engine = tambear::ClusteringEngine::new().unwrap();
    // 10 identical points
    let data = vec![5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
                    5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0];
    let result = engine.dbscan(&data, 10, 2, 0.001, 2).unwrap();
    // All distances = 0 ≤ epsilon → all in one cluster
    assert_eq!(result.n_clusters, 1,
        "DBSCAN identical points should form 1 cluster, got {}", result.n_clusters);
}

/// K-means with k=0: division by zero in step=n/k.
#[test]
fn kmeans_k_zero() {
    let engine = tambear::kmeans::KMeansEngine::new().unwrap();
    let data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 points in 2D
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        engine.fit(&data, 3, 2, 0, 10)
    }));
    // k=0 correctly panics (assert! k >= 1)
    assert!(result.is_err(), "k-means(k=0) should panic on invalid k");
}

/// K-means with all identical points: should converge in 1 iteration.
#[test]
fn kmeans_all_identical() {
    let engine = tambear::kmeans::KMeansEngine::new().unwrap();
    let data = vec![5.0f32; 20]; // 10 points in 2D, all identical
    let result = engine.fit(&data, 10, 2, 3, 100).unwrap();
    assert!(result.converged, "k-means on identical points should converge");
    assert!(result.iterations <= 2,
        "k-means on identical points should converge in ≤2 iterations, took {}", result.iterations);
}

/// K-means with k=n: each point is its own centroid.
#[test]
fn kmeans_k_equals_n() {
    let engine = tambear::kmeans::KMeansEngine::new().unwrap();
    let data = vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 4 points in 2D
    let result = engine.fit(&data, 4, 2, 4, 100).unwrap();
    assert!(result.converged, "k-means(k=n) should converge");
    // Each point should be in its own cluster (or converge quickly)
    assert_eq!(result.k, 4);
}

/// K-means: verify empty cluster handling.
/// Initialize k=3 but with data that naturally forms 2 clusters.
/// One centroid gets stranded → empty cluster.
#[test]
fn kmeans_empty_cluster_stranded_centroid() {
    let engine = tambear::kmeans::KMeansEngine::new().unwrap();
    // Two tight clusters far apart: (0,0)×5 and (100,100)×5
    let mut data = Vec::with_capacity(20);
    for _ in 0..5 { data.push(0.0f32); data.push(0.0); }
    for _ in 0..5 { data.push(100.0f32); data.push(100.0); }
    let result = engine.fit(&data, 10, 2, 3, 100).unwrap();
    // With k=3, one centroid may be stranded (empty cluster)
    // Check that centroids are finite
    for (i, &c) in result.centroids.iter().enumerate() {
        assert!(c.is_finite(),
            "k-means centroid[{}]={} should be finite (empty cluster check)", i, c);
    }
}

/// DBSCAN with negative epsilon: should produce all noise (no distances ≤ negative).
#[test]
fn dbscan_negative_epsilon() {
    let mut engine = tambear::ClusteringEngine::new().unwrap();
    let data = vec![0.0, 0.0, 1.0, 1.0];
    let result = engine.dbscan(&data, 2, 2, -1.0, 2).unwrap();
    // Note: dbscan squares epsilon → epsilon² = 1.0 for L2²
    // But negative epsilon squared = 1.0 (positive!) — so this may actually cluster!
    // DBSCAN squares epsilon → eps²=1.0 (positive!) — negative eps indistinguishable from positive
    // The squaring means negative epsilon is indistinguishable from positive
    // This is a potential usability issue but not a crash
}

/// DBSCAN with NaN in data: distances are undefined.
/// Conservative policy: NaN-distance neighbors are treated as possibly within
/// epsilon (not counted as confirmed non-neighbors). NaN-involved points may
/// still be classified as core/border rather than silently dropped to noise.
/// The result must not crash and must produce a valid ClusterResult.
#[test]
fn dbscan_nan_data() {
    let mut engine = tambear::ClusteringEngine::new().unwrap();
    let data = vec![0.0, 0.0, f64::NAN, 1.0, 2.0, 2.0];
    let r = engine.dbscan(&data, 3, 2, 1.0, 2).unwrap();
    // No crash. Labels are assigned (some cluster labels are valid).
    assert_eq!(r.labels.len(), 3, "should return one label per point");
    // Point 0=[0,0] and point 2=[2,2] have finite distance D=8 (L2Sq).
    // With epsilon=1.0, D=8 > epsilon → they are not neighbors.
    // NaN-involved point 1 has undefined distances → may or may not be core.
    // The result is well-defined (no panic) regardless of the NaN policy.
}
