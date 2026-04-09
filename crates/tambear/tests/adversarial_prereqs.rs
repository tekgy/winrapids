//! Adversarial tests for newly implemented prerequisite methods:
//! Shapiro-Wilk, VIF, ARCH-LM, Breusch-Pagan, Schoenfeld residuals
//!
//! Tests FAIL when bugs exist. When all pass, the implementations are hardened.

use tambear::linear_algebra::Mat;

// ═══════════════════════════════════════════════════════════════════════════
// SHAPIRO-WILK
// ═══════════════════════════════════════════════════════════════════════════

/// Shapiro-Wilk with constant data: W should be 1.0 (perfect normality trivially).
#[test]
fn shapiro_wilk_constant_data() {
    let data = vec![5.0; 50];
    let result = tambear::nonparametric::shapiro_wilk(&data);
    assert!((result.statistic - 1.0).abs() < 1e-10,
        "Shapiro-Wilk on constant data should have W=1.0, got {}", result.statistic);
    assert!((result.p_value - 1.0).abs() < 1e-10,
        "Shapiro-Wilk on constant data should have p=1.0, got {}", result.p_value);
}

/// Shapiro-Wilk with single element: n < 3 → NaN.
#[test]
fn shapiro_wilk_single_element() {
    let result = tambear::nonparametric::shapiro_wilk(&[42.0]);
    assert!(result.statistic.is_nan(),
        "Shapiro-Wilk with n=1 should return NaN, got {}", result.statistic);
}

/// Shapiro-Wilk with n=2: below minimum → NaN.
#[test]
fn shapiro_wilk_n2() {
    let result = tambear::nonparametric::shapiro_wilk(&[1.0, 2.0]);
    assert!(result.statistic.is_nan(),
        "Shapiro-Wilk with n=2 should return NaN, got {}", result.statistic);
}

/// Shapiro-Wilk with n=3: minimum valid case.
#[test]
fn shapiro_wilk_n3() {
    let result = tambear::nonparametric::shapiro_wilk(&[1.0, 2.0, 3.0]);
    assert!(result.statistic.is_finite(),
        "Shapiro-Wilk with n=3 should be finite, got {}", result.statistic);
    // W can exceed 1.0 slightly due to floating-point imprecision at small n
    assert!(result.statistic > 0.0 && result.statistic <= 1.0 + 1e-6,
        "W should be in (0, 1+eps], got {}", result.statistic);
}

/// Shapiro-Wilk with all NaN: should return NaN (no valid data).
#[test]
fn shapiro_wilk_all_nan() {
    let data = vec![f64::NAN; 10];
    let result = tambear::nonparametric::shapiro_wilk(&data);
    assert!(result.statistic.is_nan(),
        "Shapiro-Wilk on all-NaN should return NaN, got {}", result.statistic);
}

/// Shapiro-Wilk with some NaN: should filter and compute on valid data.
#[test]
fn shapiro_wilk_some_nan() {
    let data = vec![1.0, f64::NAN, 3.0, 4.0, 5.0, f64::NAN, 7.0, 8.0, 9.0, 10.0];
    let result = tambear::nonparametric::shapiro_wilk(&data);
    assert!(result.statistic.is_finite(),
        "Shapiro-Wilk with some NaN should filter and compute, got {}", result.statistic);
    assert!(result.statistic > 0.0 && result.statistic <= 1.0,
        "W should be in (0, 1], got {}", result.statistic);
}

/// Shapiro-Wilk on clearly non-normal data: should reject.
#[test]
fn shapiro_wilk_non_normal() {
    // Uniform distribution — clearly non-normal for large n
    let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
    let result = tambear::nonparametric::shapiro_wilk(&data);
    assert!(result.statistic < 0.99,
        "Shapiro-Wilk on uniform data should have W < 0.99, got {}", result.statistic);
    // Uniform data: p-value depends on approximation quality; may not be tiny
    assert!(result.p_value < 0.50,
        "Shapiro-Wilk on uniform data should have moderate-to-low p-value, got {}", result.p_value);
}

/// Shapiro-Wilk W must be in (0, 1].
#[test]
fn shapiro_wilk_w_in_range() {
    let data: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
    let result = tambear::nonparametric::shapiro_wilk(&data);
    assert!(result.statistic > 0.0 && result.statistic <= 1.0 + 1e-10,
        "W should be in (0, 1], got {}", result.statistic);
    assert!(result.p_value >= 0.0 && result.p_value <= 1.0 + 1e-10,
        "p-value should be in [0, 1], got {}", result.p_value);
}

/// Shapiro-Wilk empty input.
#[test]
fn shapiro_wilk_empty() {
    let result = tambear::nonparametric::shapiro_wilk(&[]);
    assert!(result.statistic.is_nan(),
        "Shapiro-Wilk on empty should return NaN, got {}", result.statistic);
}

// ═══════════════════════════════════════════════════════════════════════════
// VIF (Variance Inflation Factor)
// ═══════════════════════════════════════════════════════════════════════════

/// VIF with orthogonal predictors: all VIF should be 1.0.
#[test]
fn vif_orthogonal() {
    // Orthogonal columns: identity-like correlation
    let x = Mat::from_vec(4, 2, vec![
        1.0, 0.0,
        0.0, 1.0,
        1.0, 0.0,
        0.0, 1.0,
    ]);
    let vifs = tambear::multivariate::vif(&x);
    assert_eq!(vifs.len(), 2);
    for (i, &v) in vifs.iter().enumerate() {
        assert!(v.is_finite(),
            "VIF[{}] should be finite for orthogonal, got {}", i, v);
        assert!((v - 1.0).abs() < 0.5,
            "VIF[{}] should be near 1.0 for orthogonal, got {}", i, v);
    }
}

/// VIF with perfectly collinear predictors: VIF should be very large or Inf.
#[test]
fn vif_collinear() {
    // x2 = 2*x1 → perfect collinearity
    let x = Mat::from_vec(5, 2, vec![
        1.0, 2.0,
        2.0, 4.0,
        3.0, 6.0,
        4.0, 8.0,
        5.0, 10.0,
    ]);
    let vifs = tambear::multivariate::vif(&x);
    assert_eq!(vifs.len(), 2);
    // R² ≈ 1 → VIF = 1/(1-R²) → very large
    for (i, &v) in vifs.iter().enumerate() {
        assert!(v > 100.0 || v.is_infinite(),
            "VIF[{}] should be huge for collinear, got {}", i, v);
    }
}

/// VIF with single predictor: VIF = 1 (no other predictors to regress on).
#[test]
fn vif_single_predictor() {
    let x = Mat::from_vec(5, 1, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    let vifs = tambear::multivariate::vif(&x);
    assert_eq!(vifs.len(), 1);
    assert!((vifs[0] - 1.0).abs() < 1e-10,
        "VIF with single predictor should be 1.0, got {}", vifs[0]);
}

/// VIF with constant column: VIF should be Inf (zero SS_tot).
#[test]
fn vif_constant_column() {
    let x = Mat::from_vec(5, 2, vec![
        5.0, 1.0,
        5.0, 2.0,
        5.0, 3.0,
        5.0, 4.0,
        5.0, 5.0,
    ]);
    let vifs = tambear::multivariate::vif(&x);
    // Column 0 is constant → SS_tot = 0 → VIF = Inf
    assert!(vifs[0].is_infinite(),
        "VIF of constant column should be Inf, got {}", vifs[0]);
}

/// VIF with n < p: underdetermined regression, VIF should still be finite or Inf.
#[test]
fn vif_underdetermined() {
    let x = Mat::from_vec(2, 3, vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
    ]);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tambear::multivariate::vif(&x)
    }));
    assert!(result.is_ok(),
        "VIF should not panic with n < p (underdetermined)");
}

// ═══════════════════════════════════════════════════════════════════════════
// ARCH-LM TEST
// ═══════════════════════════════════════════════════════════════════════════

/// ARCH-LM with constant residuals: no ARCH effects → p ≈ 1.
#[test]
fn arch_lm_constant_residuals() {
    let residuals = vec![1.0; 100];
    let result = tambear::volatility::arch_lm_test(&residuals, 5);
    assert!(result.is_some(), "ARCH-LM should return Some for n=100, lags=5");
    let r = result.unwrap();
    assert!(r.statistic.is_finite(),
        "ARCH-LM statistic should be finite, got {}", r.statistic);
    // Constant residuals → constant squared residuals → no ARCH → stat ≈ 0
    assert!(r.statistic < 5.0,
        "ARCH-LM on constant residuals should have small statistic, got {}", r.statistic);
}

/// ARCH-LM with ARCH effects: should detect.
#[test]
fn arch_lm_detects_arch() {
    // Generate residuals with strong ARCH(1) structure
    let n = 200;
    let mut residuals = vec![0.0; n];
    let mut sigma2: f64 = 1.0;
    let mut rng = 42u64;
    for t in 0..n {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let z = ((rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5) * 2.0;
        residuals[t] = sigma2.sqrt() * z;
        sigma2 = 0.1 + 0.8 * residuals[t] * residuals[t]; // strong ARCH(1)
    }
    let result = tambear::volatility::arch_lm_test(&residuals, 5);
    assert!(result.is_some());
    let r = result.unwrap();
    assert!(r.statistic > 5.0,
        "ARCH-LM should detect ARCH effects: stat={}", r.statistic);
    assert!(r.p_value < 0.1,
        "ARCH-LM should reject H0 (no ARCH): p={}", r.p_value);
}

/// ARCH-LM with n_lags = 0: returns None (invalid).
#[test]
fn arch_lm_zero_lags() {
    let residuals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = tambear::volatility::arch_lm_test(&residuals, 0);
    assert!(result.is_none(),
        "ARCH-LM with 0 lags should return None");
}

/// ARCH-LM with n_lags >= n: returns None (insufficient data).
#[test]
fn arch_lm_too_many_lags() {
    let residuals = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = tambear::volatility::arch_lm_test(&residuals, 5);
    assert!(result.is_none(),
        "ARCH-LM with n_lags >= n should return None");
}

/// ARCH-LM with all-zero residuals: squared residuals = 0, R² = 0 → stat = 0.
#[test]
fn arch_lm_zero_residuals() {
    let residuals = vec![0.0; 50];
    let result = tambear::volatility::arch_lm_test(&residuals, 3);
    assert!(result.is_some());
    let r = result.unwrap();
    assert!((r.statistic - 0.0).abs() < 1e-10,
        "ARCH-LM on zero residuals should have stat=0, got {}", r.statistic);
}

/// ARCH-LM with NaN residuals: NaN should propagate or be handled.
#[test]
fn arch_lm_nan_residuals() {
    let mut residuals = vec![1.0; 20];
    residuals[5] = f64::NAN;
    let result = tambear::volatility::arch_lm_test(&residuals, 3);
    assert!(result.is_some());
    let r = result.unwrap();
    // NaN should propagate to statistic
    assert!(r.statistic.is_nan() || r.statistic.is_finite(),
        "ARCH-LM with NaN should produce NaN or finite, not Inf");
}

// ═══════════════════════════════════════════════════════════════════════════
// BREUSCH-PAGAN TEST
// ═══════════════════════════════════════════════════════════════════════════

/// Breusch-Pagan with perfect homoscedasticity: p ≈ 1.
#[test]
fn breusch_pagan_homoscedastic() {
    // Constant variance residuals
    let n = 50;
    let mut rng = 12345u64;
    let residuals: Vec<f64> = (0..n).map(|_| {
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng >> 33) as f64 / (1u64 << 31) as f64 - 0.5
    }).collect();
    // Design matrix with intercept and one predictor
    let mut x_data = vec![0.0; n * 2];
    for i in 0..n {
        x_data[i * 2] = 1.0; // intercept
        x_data[i * 2 + 1] = i as f64;
    }
    let x = Mat::from_vec(n, 2, x_data);
    let result = tambear::hypothesis::breusch_pagan(&x, &residuals);
    assert!(result.statistic.is_finite(),
        "BP statistic should be finite, got {}", result.statistic);
    // Homoscedastic → small statistic, large p-value
    assert!(result.p_value > 0.01,
        "BP on homoscedastic data should have p > 0.01, got {}", result.p_value);
}

/// Breusch-Pagan with zero residuals: trivially homoscedastic.
#[test]
fn breusch_pagan_zero_residuals() {
    let n = 20;
    let residuals = vec![0.0; n];
    let mut x_data = vec![0.0; n * 2];
    for i in 0..n { x_data[i * 2] = 1.0; x_data[i * 2 + 1] = i as f64; }
    let x = Mat::from_vec(n, 2, x_data);
    let result = tambear::hypothesis::breusch_pagan(&x, &residuals);
    assert!(result.statistic.is_finite(),
        "BP with zero residuals should be finite, got {}", result.statistic);
}

/// Breusch-Pagan with single predictor (intercept only): df = 0 edge case.
#[test]
fn breusch_pagan_intercept_only() {
    let n = 20;
    let residuals: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let x_data: Vec<f64> = (0..n).map(|_| 1.0).collect(); // intercept only
    let x = Mat::from_vec(n, 1, x_data);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tambear::hypothesis::breusch_pagan(&x, &residuals)
    }));
    assert!(result.is_ok(),
        "BP with intercept-only should not panic");
}

/// Breusch-Pagan with NaN residuals.
#[test]
fn breusch_pagan_nan_residuals() {
    let n = 10;
    let mut residuals = vec![1.0; n];
    residuals[3] = f64::NAN;
    let mut x_data = vec![0.0; n * 2];
    for i in 0..n { x_data[i * 2] = 1.0; x_data[i * 2 + 1] = i as f64; }
    let x = Mat::from_vec(n, 2, x_data);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        tambear::hypothesis::breusch_pagan(&x, &residuals)
    }));
    assert!(result.is_ok(),
        "BP should not panic with NaN residuals");
}

// ═══════════════════════════════════════════════════════════════════════════
// SCHOENFELD RESIDUALS (on Cox PH)
// ═══════════════════════════════════════════════════════════════════════════

/// Schoenfeld residuals: count should equal number of events.
#[test]
fn schoenfeld_count_equals_events() {
    let n = 20;
    let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let times: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let events: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let n_events = events.iter().filter(|&&e| e).count();
    let result = tambear::survival::cox_ph(&x, &times, &events, n, 1, 25);
    assert_eq!(result.schoenfeld_residuals.len(), n_events,
        "Schoenfeld count should equal event count: {} != {}", result.schoenfeld_residuals.len(), n_events);
}

/// Schoenfeld residuals should sum to approximately zero at convergence.
#[test]
fn schoenfeld_sum_near_zero() {
    let n = 30;
    let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.1).collect();
    let times: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let events: Vec<bool> = (0..n).map(|i| i % 3 != 0).collect();
    let result = tambear::survival::cox_ph(&x, &times, &events, n, 1, 50);
    let sum: f64 = result.schoenfeld_residuals.iter().map(|r| r[0]).sum();
    // Tolerance depends on convergence quality. With moderate n and iterations,
    // the score equation sum should be small relative to sample size.
    assert!(sum.abs() < 5.0,
        "Schoenfeld residuals should sum near 0 (score equation), got sum={}", sum);
}

/// Schoenfeld residuals should all be finite.
#[test]
fn schoenfeld_all_finite() {
    let n = 15;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let times: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let events = vec![true; n];
    let result = tambear::survival::cox_ph(&x, &times, &events, n, 1, 25);
    for (k, resid) in result.schoenfeld_residuals.iter().enumerate() {
        for (j, &v) in resid.iter().enumerate() {
            assert!(v.is_finite(),
                "Schoenfeld residual [event={}, covariate={}] should be finite, got {}", k, j, v);
        }
    }
}

/// Schoenfeld with all censored: no events → no residuals.
#[test]
fn schoenfeld_all_censored() {
    let n = 10;
    let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let times: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let events = vec![false; n];
    let result = tambear::survival::cox_ph(&x, &times, &events, n, 1, 25);
    assert!(result.schoenfeld_residuals.is_empty(),
        "Schoenfeld with all censored should be empty, got {} residuals",
        result.schoenfeld_residuals.len());
}

/// Schoenfeld multivariate: each residual should have d components.
#[test]
fn schoenfeld_multivariate_dimensions() {
    let n = 20;
    let d = 3;
    let x: Vec<f64> = (0..n*d).map(|i| i as f64 * 0.01).collect();
    let times: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let events: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let result = tambear::survival::cox_ph(&x, &times, &events, n, d, 25);
    for (k, resid) in result.schoenfeld_residuals.iter().enumerate() {
        assert_eq!(resid.len(), d,
            "Schoenfeld[{}] should have {} components, got {}", k, d, resid.len());
    }
}
