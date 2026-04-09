//! Adversarial tests for regularized regression: Ridge, Lasso, Elastic Net
//!
//! Tests assert mathematical properties and boundary behavior.

use tambear::linear_algebra::Mat;
use tambear::multivariate::*;

// ═══════════════════════════════════════════════════════════════════════════
// RIDGE REGRESSION
// ═══════════════════════════════════════════════════════════════════════════

/// Ridge with lambda=0: should match OLS exactly.
#[test]
fn ridge_lambda_zero_equals_ols() {
    // y = 2*x1 + 3*x2 + 1 (exact linear, no noise)
    let x = Mat::from_vec(5, 2, vec![
        1.0, 0.0,
        0.0, 1.0,
        1.0, 1.0,
        2.0, 1.0,
        1.0, 2.0,
    ]);
    let y = vec![3.0, 4.0, 6.0, 8.0, 9.0]; // 2*x1 + 3*x2 + 1
    let result = ridge(&x, &y, 0.0);
    assert!((result.beta[0] - 2.0).abs() < 0.1,
        "Ridge(λ=0) beta[0] should be ~2.0 (OLS), got {}", result.beta[0]);
    assert!((result.beta[1] - 3.0).abs() < 0.1,
        "Ridge(λ=0) beta[1] should be ~3.0 (OLS), got {}", result.beta[1]);
    assert!((result.intercept - 1.0).abs() < 0.1,
        "Ridge(λ=0) intercept should be ~1.0, got {}", result.intercept);
    assert!(result.r2 > 0.99,
        "Ridge(λ=0) R² should be ~1.0, got {}", result.r2);
}

/// Ridge with large lambda: coefficients should shrink toward 0.
#[test]
fn ridge_large_lambda_shrinks() {
    let x = Mat::from_vec(5, 2, vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0,
    ]);
    let y = vec![3.0, 4.0, 6.0, 8.0, 9.0];
    let r_small = ridge(&x, &y, 0.01);
    let r_large = ridge(&x, &y, 1000.0);
    // Larger lambda → smaller coefficients
    let norm_small: f64 = r_small.beta.iter().map(|b| b * b).sum::<f64>().sqrt();
    let norm_large: f64 = r_large.beta.iter().map(|b| b * b).sum::<f64>().sqrt();
    assert!(norm_large < norm_small,
        "Larger λ should shrink: ‖β(λ=1000)‖={} < ‖β(λ=0.01)‖={}", norm_large, norm_small);
}

/// Ridge with collinear predictors: should not panic (λ regularizes).
#[test]
fn ridge_collinear_predictors() {
    // x2 = 2*x1 → X'X singular, but X'X + λI is not
    let x = Mat::from_vec(5, 2, vec![
        1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0, 5.0, 10.0,
    ]);
    let y = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = ridge(&x, &y, 1.0);
    assert!(result.beta[0].is_finite() && result.beta[1].is_finite(),
        "Ridge with collinear should produce finite betas: {:?}", result.beta);
}

/// Ridge with constant y: all betas should be ~0.
#[test]
fn ridge_constant_y() {
    let x = Mat::from_vec(5, 2, vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    ]);
    let y = vec![5.0; 5];
    let result = ridge(&x, &y, 1.0);
    for (j, &b) in result.beta.iter().enumerate() {
        assert!(b.abs() < 0.1,
            "Ridge constant y: beta[{}] should be ~0, got {}", j, b);
    }
    assert!((result.intercept - 5.0).abs() < 0.1,
        "Ridge constant y: intercept should be ~5, got {}", result.intercept);
}

/// Ridge with single predictor: closed-form verification.
#[test]
fn ridge_single_predictor() {
    // y = 3*x + 2, lambda=1
    let x = Mat::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let y = vec![5.0, 8.0, 11.0, 14.0]; // y = 3x + 2
    let result = ridge(&x, &y, 0.0);
    assert!((result.beta[0] - 3.0).abs() < 0.01,
        "Ridge(λ=0) single predictor should recover slope=3, got {}", result.beta[0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// LASSO REGRESSION
// ═══════════════════════════════════════════════════════════════════════════

/// Lasso with lambda=0: should match OLS (no sparsity).
#[test]
fn lasso_lambda_zero_equals_ols() {
    let x = Mat::from_vec(5, 2, vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0,
    ]);
    let y = vec![3.0, 4.0, 6.0, 8.0, 9.0];
    let result = lasso(&x, &y, 0.0, 1000, 1e-8);
    assert!((result.beta[0] - 2.0).abs() < 0.2,
        "Lasso(λ=0) beta[0] should be ~2.0, got {}", result.beta[0]);
    assert!((result.beta[1] - 3.0).abs() < 0.2,
        "Lasso(λ=0) beta[1] should be ~3.0, got {}", result.beta[1]);
}

/// Lasso with large lambda: all coefficients should be exactly 0 (sparsity).
#[test]
fn lasso_large_lambda_sparsity() {
    let x = Mat::from_vec(5, 2, vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0,
    ]);
    let y = vec![3.0, 4.0, 6.0, 8.0, 9.0];
    let result = lasso(&x, &y, 100.0, 1000, 1e-8);
    // Large λ → all betas exactly 0
    assert_eq!(result.n_nonzero, 0,
        "Lasso with huge λ should zero all coefficients, got {} nonzero", result.n_nonzero);
    for (j, &b) in result.beta.iter().enumerate() {
        assert!(b.abs() < 1e-10,
            "Lasso(λ=100) beta[{}] should be 0, got {}", j, b);
    }
}

/// Lasso with moderate lambda: some coefficients zero (feature selection).
#[test]
fn lasso_feature_selection() {
    // y depends on x1 only, x2 and x3 are noise
    let n = 50;
    let mut x_data = Vec::with_capacity(n * 3);
    let mut y = Vec::with_capacity(n);
    let mut rng = 12345u64;
    for i in 0..n {
        let x1 = i as f64;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x2 = (rng >> 33) as f64 / (1u64 << 31) as f64;
        rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
        let x3 = (rng >> 33) as f64 / (1u64 << 31) as f64;
        x_data.push(x1);
        x_data.push(x2);
        x_data.push(x3);
        y.push(2.0 * x1 + 1.0);
    }
    let x = Mat::from_vec(n, 3, x_data);
    let result = lasso(&x, &y, 0.5, 1000, 1e-8);
    // x1 should have large coefficient, x2 and x3 should be near 0
    assert!(result.beta[0].abs() > 1.0,
        "Lasso should keep x1: beta[0]={}", result.beta[0]);
    // x2, x3 might be zeroed or very small
    assert!(result.beta[1].abs() < 0.5 && result.beta[2].abs() < 0.5,
        "Lasso should shrink noise: beta[1]={}, beta[2]={}", result.beta[1], result.beta[2]);
}

/// Lasso convergence: should terminate before max_iter for well-conditioned problems.
#[test]
fn lasso_converges() {
    let x = Mat::from_vec(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
    let y = vec![2.0, 4.0, 6.0, 8.0]; // y = 2x
    let result = lasso(&x, &y, 0.01, 1000, 1e-10);
    assert!(result.iterations < 100,
        "Lasso should converge quickly, took {} iterations", result.iterations);
}

/// Lasso with constant y: all betas 0.
#[test]
fn lasso_constant_y() {
    let x = Mat::from_vec(5, 2, vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
    ]);
    let y = vec![5.0; 5];
    let result = lasso(&x, &y, 0.1, 1000, 1e-8);
    assert_eq!(result.n_nonzero, 0,
        "Lasso constant y should zero all, got {} nonzero", result.n_nonzero);
}

// ═══════════════════════════════════════════════════════════════════════════
// ELASTIC NET
// ═══════════════════════════════════════════════════════════════════════════

/// Elastic Net alpha=1: should equal Lasso.
#[test]
fn elastic_net_alpha1_equals_lasso() {
    let x = Mat::from_vec(5, 2, vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0,
    ]);
    let y = vec![3.0, 4.0, 6.0, 8.0, 9.0];
    let en = elastic_net(&x, &y, 0.1, 1.0, 1000, 1e-10);
    let la = lasso(&x, &y, 0.1, 1000, 1e-10);
    for j in 0..2 {
        assert!((en.beta[j] - la.beta[j]).abs() < 1e-6,
            "EN(α=1) should equal Lasso: beta[{}] {} vs {}", j, en.beta[j], la.beta[j]);
    }
}

/// Elastic Net alpha=0: should be close to Ridge (scaled differently).
#[test]
fn elastic_net_alpha0_ridge_like() {
    let x = Mat::from_vec(5, 2, vec![
        1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0,
    ]);
    let y = vec![3.0, 4.0, 6.0, 8.0, 9.0];
    let result = elastic_net(&x, &y, 1.0, 0.0, 1000, 1e-10);
    // Alpha=0 → pure L2 penalty → no sparsity, all coefficients nonzero
    assert!(result.n_nonzero == 2,
        "EN(α=0) should keep all coefficients, got {} nonzero", result.n_nonzero);
}

/// Elastic Net with NaN in data: should propagate or handle.
#[test]
fn elastic_net_nan_data() {
    let x = Mat::from_vec(3, 1, vec![1.0, f64::NAN, 3.0]);
    let y = vec![1.0, 2.0, 3.0];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        elastic_net(&x, &y, 0.1, 0.5, 100, 1e-8)
    }));
    assert!(result.is_ok(), "Elastic Net should not panic on NaN data");
}

/// Elastic Net with empty data: should not panic.
#[test]
fn elastic_net_empty() {
    let x = Mat::from_vec(0, 2, vec![]);
    let y: Vec<f64> = vec![];
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        elastic_net(&x, &y, 0.1, 0.5, 100, 1e-8)
    }));
    // May panic on assertion (n=0 is degenerate) — acceptable
    // Just shouldn't produce garbage silently
    if let Ok(r) = result {
        assert!(r.beta.is_empty() || r.beta.iter().all(|b| b.is_finite()),
            "Empty EN should produce empty or finite betas");
    }
}

/// Ridge R² should be in [0, 1].
#[test]
fn ridge_r2_range() {
    let x = Mat::from_vec(10, 2, (0..20).map(|i| i as f64 * 0.1).collect());
    let y: Vec<f64> = (0..10).map(|i| (i as f64 * 0.3).sin()).collect();
    for &lambda in &[0.0, 0.1, 1.0, 10.0, 100.0] {
        let r = ridge(&x, &y, lambda);
        assert!(r.r2 >= 0.0 && r.r2 <= 1.0,
            "Ridge(λ={}) R² should be in [0,1], got {}", lambda, r.r2);
    }
}

/// Gold standard: Ridge with known analytical solution.
/// For y = Xβ* where X is orthogonal and standardized:
/// β_ridge = β_ols / (1 + λ/n)
#[test]
fn ridge_gold_standard_orthogonal() {
    // Orthogonal X (after centering): use [1,-1,1,-1] and [1,1,-1,-1]
    let x = Mat::from_vec(4, 2, vec![
        1.0,  1.0,
       -1.0,  1.0,
        1.0, -1.0,
       -1.0, -1.0,
    ]);
    // y = 3*x1 + 5*x2 (no intercept needed, y mean = 0)
    let y = vec![8.0, 2.0, -2.0, -8.0]; // 3*1+5*1, 3*(-1)+5*1, 3*1+5*(-1), 3*(-1)+5*(-1)
    let n = 4.0;
    let lambda = 2.0;
    let result = ridge(&x, &y, lambda);
    // For orthogonal centered X: X'X = n*I, so (X'X + λI)^{-1} = I/(n+λ)
    // β_ridge = X'y / (n + λ)
    // X'y = [3*4, 5*4] = [12, 20] (since X is orthogonal with X'X = 4I)
    let expected_0 = 12.0 / (n + lambda); // 12/6 = 2.0
    let expected_1 = 20.0 / (n + lambda); // 20/6 = 3.333
    assert!((result.beta[0] - expected_0).abs() < 0.01,
        "Ridge orthogonal: beta[0] should be {}, got {}", expected_0, result.beta[0]);
    assert!((result.beta[1] - expected_1).abs() < 0.01,
        "Ridge orthogonal: beta[1] should be {}, got {}", expected_1, result.beta[1]);
}
