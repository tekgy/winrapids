//! Adversarial tests for confidence intervals on hypothesis tests.
//!
//! Key property: the 95% CI contains the null value iff the test does NOT reject at α=0.05.
//! This is the duality between CIs and hypothesis tests.

use tambear::descriptive::MomentStats;
use tambear::hypothesis::*;

// ═══════════════════════════════════════════════════════════════════════════
// ONE-SAMPLE T-TEST CI
// ═══════════════════════════════════════════════════════════════════════════

/// One-sample CI on mean: x̄ ± t_{0.975, n-1} · SE.
#[test]
fn one_sample_t_ci_contains_mean() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let stats = tambear::descriptive::moments_ungrouped(&data);
    // Test against the sample mean itself; CI should be centered on it
    let r = one_sample_t(&stats, stats.mean());
    // The CI is on the mean; it should contain the sample mean (exactly)
    assert!(r.ci_lower < stats.mean() && r.ci_upper > stats.mean(),
        "CI [{}, {}] should contain sample mean {}", r.ci_lower, r.ci_upper, stats.mean());
    assert!((r.ci_level - 0.95).abs() < 1e-10, "CI level should be 0.95");
}

/// Duality: CI contains null → test does not reject at α = 0.05.
#[test]
fn one_sample_t_ci_duality() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]; // mean=5.5
    let stats = tambear::descriptive::moments_ungrouped(&data);

    // Null=5.5 is the sample mean → CI contains it, p ≈ 1
    let r1 = one_sample_t(&stats, 5.5);
    assert_eq!(r1.ci_contains(5.5), Some(true));
    assert!(!r1.significant_at(0.05));

    // Null=100 is way outside → CI does not contain, p < 0.05
    let r2 = one_sample_t(&stats, 100.0);
    assert_eq!(r2.ci_contains(100.0), Some(false));
    assert!(r2.significant_at(0.05));
}

/// CI width shrinks with increasing n (SE decreases).
#[test]
fn one_sample_t_ci_width_shrinks_with_n() {
    let data_small: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let data_large: Vec<f64> = (1..=100).map(|i| (i as f64 / 10.0)).collect();
    let r_small = one_sample_t(&tambear::descriptive::moments_ungrouped(&data_small), 5.5);
    let r_large = one_sample_t(&tambear::descriptive::moments_ungrouped(&data_large), 5.05);
    let w_small = r_small.ci_upper - r_small.ci_lower;
    let w_large = r_large.ci_upper - r_large.ci_lower;
    assert!(w_large < w_small,
        "CI width should shrink with n: {} (n=10) vs {} (n=100)", w_small, w_large);
}

/// One-sample t with n < 2: CI should be NaN.
#[test]
fn one_sample_t_ci_too_few() {
    let stats = MomentStats {
        count: 1.0, sum: 5.0, min: 5.0, max: 5.0,
        m2: 0.0, m3: 0.0, m4: 0.0,
    };
    let r = one_sample_t(&stats, 0.0);
    assert!(r.ci_lower.is_nan() && r.ci_upper.is_nan(),
        "CI should be NaN with n<2, got [{}, {}]", r.ci_lower, r.ci_upper);
}

// ═══════════════════════════════════════════════════════════════════════════
// TWO-SAMPLE T-TEST CI
// ═══════════════════════════════════════════════════════════════════════════

/// Two-sample CI on mean difference (μ₁ - μ₂).
#[test]
fn two_sample_t_ci_contains_diff() {
    let d1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let d2 = vec![3.0, 4.0, 5.0, 6.0, 7.0]; // mean diff = -2
    let s1 = tambear::descriptive::moments_ungrouped(&d1);
    let s2 = tambear::descriptive::moments_ungrouped(&d2);
    let r = two_sample_t(&s1, &s2);
    let diff = s1.mean() - s2.mean();
    assert!(r.ci_lower < diff && r.ci_upper > diff,
        "CI [{}, {}] should contain mean diff {}", r.ci_lower, r.ci_upper, diff);
    // Null=0 should be contained (small sample, modest separation)
    let contains_zero = r.ci_contains(0.0).unwrap_or(false);
    let rejects = r.significant_at(0.05);
    // Duality: contains(0) ↔ NOT reject
    assert_eq!(contains_zero, !rejects,
        "Duality violated: contains(0)={}, rejects={}", contains_zero, rejects);
}

/// Two-sample CI: strong separation → CI excludes 0.
#[test]
fn two_sample_t_ci_strong_separation() {
    let d1 = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let d2 = vec![100.0, 101.0, 102.0, 103.0, 104.0];
    let s1 = tambear::descriptive::moments_ungrouped(&d1);
    let s2 = tambear::descriptive::moments_ungrouped(&d2);
    let r = two_sample_t(&s1, &s2);
    // CI should be far from zero
    assert!(r.ci_upper < 0.0,
        "Strong negative separation: CI upper should be < 0, got {}", r.ci_upper);
    assert_eq!(r.ci_contains(0.0), Some(false));
    assert!(r.significant_at(0.01));
}

// ═══════════════════════════════════════════════════════════════════════════
// WELCH T-TEST CI
// ═══════════════════════════════════════════════════════════════════════════

/// Welch's t-test CI duality holds.
#[test]
fn welch_t_ci_duality() {
    let d1: Vec<f64> = (1..=20).map(|i| i as f64).collect();
    let d2: Vec<f64> = (1..=20).map(|i| i as f64 + 10.0).collect();
    let s1 = tambear::descriptive::moments_ungrouped(&d1);
    let s2 = tambear::descriptive::moments_ungrouped(&d2);
    let r = welch_t(&s1, &s2);
    let rejects_at_05 = r.significant_at(0.05);
    let contains_zero = r.ci_contains(0.0).unwrap_or(false);
    assert_eq!(rejects_at_05, !contains_zero,
        "Welch duality: rejects={}, contains(0)={}", rejects_at_05, contains_zero);
}

// ═══════════════════════════════════════════════════════════════════════════
// PROPORTION CIs
// ═══════════════════════════════════════════════════════════════════════════

/// Wilson score CI for one proportion: should contain p̂.
#[test]
fn one_proportion_ci_contains_phat() {
    let r = one_proportion_z(60.0, 100.0, 0.5);
    let p_hat = 0.6;
    assert!(r.ci_lower < p_hat && r.ci_upper > p_hat,
        "Wilson CI [{}, {}] should contain p̂={}", r.ci_lower, r.ci_upper, p_hat);
    // CI should be within [0, 1]
    assert!(r.ci_lower >= 0.0 && r.ci_upper <= 1.0,
        "Wilson CI should be in [0,1], got [{}, {}]", r.ci_lower, r.ci_upper);
}

/// Wilson CI near 0: should not go below 0 (unlike normal approximation).
#[test]
fn one_proportion_wilson_boundary() {
    let r = one_proportion_z(2.0, 100.0, 0.5);
    assert!(r.ci_lower > 0.0,
        "Wilson CI lower should be > 0 (unlike Wald), got {}", r.ci_lower);
}

/// Two-proportion CI: well-separated proportions.
#[test]
fn two_proportion_ci_separation() {
    let r = two_proportion_z(80.0, 100.0, 20.0, 100.0);
    // p1 - p2 = 0.6, should be well-separated from 0
    assert!(r.ci_lower > 0.3,
        "CI lower should be far from 0, got {}", r.ci_lower);
    assert_eq!(r.ci_contains(0.0), Some(false));
    assert!(r.significant_at(0.001));
}

// ═══════════════════════════════════════════════════════════════════════════
// GOLD STANDARD VERIFICATION
// ═══════════════════════════════════════════════════════════════════════════

/// Gold standard: n=10 sample with mean=5, SD=2, test at μ=5.
/// 95% CI = 5 ± t_{0.975, 9} · (2/√10) = 5 ± 2.262 · 0.632 = 5 ± 1.430
#[test]
fn one_sample_t_ci_gold_standard() {
    // Construct MomentStats directly: n=10, mean=5, variance=4
    let stats = MomentStats {
        count: 10.0, sum: 50.0, min: 1.0, max: 9.0,
        m2: 36.0, // (n-1) * var = 9 * 4 = 36
        m3: 0.0, m4: 0.0,
    };
    // variance(1) = m2/(n-1) = 36/9 = 4.0 ✓
    let r = one_sample_t(&stats, 5.0);
    // Expected CI: 5 ± 2.262 · 0.6325 ≈ 5 ± 1.4305
    let expected_half = 2.262 * (2.0 / (10.0_f64).sqrt());
    let half = (r.ci_upper - r.ci_lower) / 2.0;
    assert!((half - expected_half).abs() < 0.05,
        "Gold: half-width should be ~{}, got {}", expected_half, half);
    assert!((r.ci_lower - (5.0 - expected_half)).abs() < 0.05);
    assert!((r.ci_upper - (5.0 + expected_half)).abs() < 0.05);
}

/// CI level should always be 0.95 when computed.
#[test]
fn ci_level_is_0_95() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let stats = tambear::descriptive::moments_ungrouped(&data);
    let r = one_sample_t(&stats, 0.0);
    assert!((r.ci_level - 0.95).abs() < 1e-10,
        "CI level should be 0.95, got {}", r.ci_level);
}

/// ci_contains returns None when CI not computed.
#[test]
fn ci_contains_none_when_nan() {
    // Cochran's Q doesn't compute a CI on the estimand
    let data = vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0]; // 3 subjects × 2 treatments
    let r = cochran_q(&data, 3, 2);
    assert_eq!(r.ci_contains(0.0), None,
        "ci_contains should be None when ci_lower/upper are NaN");
}
