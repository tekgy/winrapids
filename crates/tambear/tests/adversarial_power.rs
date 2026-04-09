//! Adversarial tests for power analysis.
//!
//! Gold standards from Cohen (1988) and G*Power:
//! - t-test, d=0.5 (medium), α=0.05 two-sided, power=0.80 → n_per_group ≈ 64
//! - t-test, d=0.8 (large), α=0.05 two-sided, power=0.80 → n_per_group ≈ 26
//! - ANOVA, f=0.25 (medium), k=3, α=0.05, power=0.80 → n_per_group ≈ 52
//! - correlation, r=0.3, α=0.05 two-sided, power=0.80 → n ≈ 85

use tambear::hypothesis::*;

// ═══════════════════════════════════════════════════════════════════════════
// POWER ↔ SAMPLE SIZE DUALITY
// ═══════════════════════════════════════════════════════════════════════════

/// Core duality: if sample_size returns n*, then power(n*) ≥ target.
#[test]
fn power_sample_size_duality_two_sample_t() {
    let d = 0.5;
    let target_power = 0.80;
    let n = sample_size_two_sample_t(d, target_power, 0.05, true);
    let achieved = power_two_sample_t(d, n, 0.05, true);
    assert!(achieved >= target_power - 0.01,
        "Computed n={} should achieve power ≥ {}, got {}", n, target_power, achieved);
    // One fewer should give less than target
    let achieved_minus = power_two_sample_t(d, n - 1.0, 0.05, true);
    assert!(achieved_minus < achieved,
        "Power should be monotone in n: {} vs {}", achieved_minus, achieved);
}

// ═══════════════════════════════════════════════════════════════════════════
// TWO-SAMPLE T-TEST POWER
// ═══════════════════════════════════════════════════════════════════════════

/// Gold standard: two-sample t, d=0.5, α=0.05 two-sided, power=0.80 → n≈64/group.
/// G*Power value: 64 per group.
#[test]
fn power_two_sample_t_gold_medium() {
    let n = sample_size_two_sample_t(0.5, 0.80, 0.05, true);
    assert!((n - 64.0).abs() < 5.0,
        "G*Power says n≈64 for d=0.5, got {}", n);
}

/// Gold standard: two-sample t, d=0.8, α=0.05 two-sided, power=0.80 → n≈26/group.
#[test]
fn power_two_sample_t_gold_large() {
    let n = sample_size_two_sample_t(0.8, 0.80, 0.05, true);
    assert!((n - 26.0).abs() < 3.0,
        "G*Power says n≈26 for d=0.8, got {}", n);
}

/// Gold standard: two-sample t, d=0.2, α=0.05 two-sided, power=0.80 → n≈394/group.
#[test]
fn power_two_sample_t_gold_small() {
    let n = sample_size_two_sample_t(0.2, 0.80, 0.05, true);
    assert!((n - 394.0).abs() < 10.0,
        "G*Power says n≈394 for d=0.2, got {}", n);
}

/// Zero effect size: can never reject reliably → NaN sample size.
#[test]
fn power_sample_size_zero_effect() {
    let n = sample_size_two_sample_t(0.0, 0.80, 0.05, true);
    assert!(n.is_nan(), "Zero effect size should give NaN n, got {}", n);
}

/// Power at n=2 per group: very low.
#[test]
fn power_two_sample_t_tiny_n() {
    let p = power_two_sample_t(0.5, 2.0, 0.05, true);
    assert!(p < 0.15,
        "Power with n=2 per group should be very low, got {}", p);
}

/// Power monotone in effect size.
#[test]
fn power_monotone_in_effect_size() {
    let p_small = power_two_sample_t(0.2, 50.0, 0.05, true);
    let p_med = power_two_sample_t(0.5, 50.0, 0.05, true);
    let p_large = power_two_sample_t(0.8, 50.0, 0.05, true);
    assert!(p_small < p_med && p_med < p_large,
        "Power should increase with effect size: {} < {} < {}", p_small, p_med, p_large);
}

/// Power monotone in n.
#[test]
fn power_monotone_in_n() {
    let p_10 = power_two_sample_t(0.5, 10.0, 0.05, true);
    let p_50 = power_two_sample_t(0.5, 50.0, 0.05, true);
    let p_200 = power_two_sample_t(0.5, 200.0, 0.05, true);
    assert!(p_10 < p_50 && p_50 < p_200,
        "Power should increase with n: {} < {} < {}", p_10, p_50, p_200);
}

/// Power approaches alpha as effect size → 0 (type-I error rate).
#[test]
fn power_approaches_alpha() {
    let p = power_two_sample_t(1e-10, 100.0, 0.05, true);
    assert!((p - 0.05).abs() < 0.01,
        "Power at d=0 should approach alpha=0.05, got {}", p);
}

/// Power approaches 1 for huge effect size.
#[test]
fn power_approaches_one() {
    let p = power_two_sample_t(5.0, 100.0, 0.05, true);
    assert!(p > 0.99,
        "Power for d=5 should be ~1, got {}", p);
}

// ═══════════════════════════════════════════════════════════════════════════
// ONE-SAMPLE T-TEST POWER
// ═══════════════════════════════════════════════════════════════════════════

/// One-sample needs roughly half the n of two-sample for the same power (same d).
#[test]
fn power_one_sample_vs_two_sample() {
    let n_one = sample_size_one_sample_t(0.5, 0.80, 0.05, true);
    let n_two = sample_size_two_sample_t(0.5, 0.80, 0.05, true);
    // Two-sample needs 2x (plus a bit)
    assert!(n_two > n_one,
        "Two-sample should need more n than one-sample: {} vs {}", n_two, n_one);
    let ratio = n_two / n_one;
    assert!((ratio - 2.0).abs() < 0.2,
        "Ratio should be ~2, got {}", ratio);
}

/// Gold standard: one-sample t, d=0.5, α=0.05 two-sided, power=0.80 → n ≈ 34.
#[test]
fn power_one_sample_t_gold() {
    let n = sample_size_one_sample_t(0.5, 0.80, 0.05, true);
    assert!((n - 34.0).abs() < 4.0,
        "G*Power says n≈34 for one-sample d=0.5, got {}", n);
}

// ═══════════════════════════════════════════════════════════════════════════
// ANOVA POWER
// ═══════════════════════════════════════════════════════════════════════════

/// Gold standard: ANOVA, f=0.25 (medium), k=3, α=0.05, power=0.80 → n≈52/group.
/// G*Power value: 52 per group (total N=156).
#[test]
fn power_anova_gold_medium() {
    let n = sample_size_anova(0.25, 3.0, 0.80, 0.05);
    assert!((n - 52.0).abs() < 10.0,
        "G*Power says n≈52 for ANOVA f=0.25, k=3, got {}", n);
}

/// Gold standard: ANOVA, f=0.40 (large), k=4, α=0.05, power=0.80 → n≈19/group.
#[test]
fn power_anova_gold_large() {
    let n = sample_size_anova(0.40, 4.0, 0.80, 0.05);
    assert!((n - 19.0).abs() < 5.0,
        "G*Power says n≈19 for ANOVA f=0.40, k=4, got {}", n);
}

/// ANOVA power monotone in n.
#[test]
fn power_anova_monotone() {
    let p_small = power_anova(0.25, 3.0, 20.0, 0.05);
    let p_large = power_anova(0.25, 3.0, 100.0, 0.05);
    assert!(p_small < p_large,
        "ANOVA power should increase with n: {} vs {}", p_small, p_large);
}

/// ANOVA with k=1: degenerate → NaN.
#[test]
fn power_anova_single_group() {
    let p = power_anova(0.25, 1.0, 20.0, 0.05);
    assert!(p.is_nan(), "ANOVA k=1 should be NaN, got {}", p);
}

// ═══════════════════════════════════════════════════════════════════════════
// CORRELATION POWER
// ═══════════════════════════════════════════════════════════════════════════

/// Gold standard: correlation, r=0.3, α=0.05 two-sided, power=0.80 → n ≈ 85.
#[test]
fn power_correlation_gold() {
    let n = sample_size_correlation(0.3, 0.80, 0.05, true);
    assert!((n - 85.0).abs() < 5.0,
        "G*Power says n≈85 for r=0.3, got {}", n);
}

/// Correlation power monotone in r.
#[test]
fn power_correlation_monotone() {
    let p_01 = power_correlation(0.1, 50.0, 0.05, true);
    let p_03 = power_correlation(0.3, 50.0, 0.05, true);
    let p_05 = power_correlation(0.5, 50.0, 0.05, true);
    assert!(p_01 < p_03 && p_03 < p_05,
        "Correlation power should increase with r: {} < {} < {}", p_01, p_03, p_05);
}

/// Correlation with r ≥ 1: invalid → NaN.
#[test]
fn power_correlation_r_one() {
    let p = power_correlation(1.0, 50.0, 0.05, true);
    assert!(p.is_nan(), "r=1 should be NaN, got {}", p);
}

/// Correlation with n ≤ 3: Fisher z undefined → NaN.
#[test]
fn power_correlation_too_few() {
    let p = power_correlation(0.3, 3.0, 0.05, true);
    assert!(p.is_nan(), "n=3 should be NaN, got {}", p);
}

/// Correlation duality: sample_size(r) gives n* with power ≥ target.
#[test]
fn power_correlation_duality() {
    let r = 0.3;
    let target = 0.80;
    let n = sample_size_correlation(r, target, 0.05, true);
    let achieved = power_correlation(r, n, 0.05, true);
    assert!(achieved >= target - 0.02,
        "Correlation sample_size/power duality: n={}, achieved={}", n, achieved);
}

// ═══════════════════════════════════════════════════════════════════════════
// CROSS-TEST SANITY
// ═══════════════════════════════════════════════════════════════════════════

/// Power in [0, 1] for all valid inputs.
#[test]
fn power_in_unit_interval() {
    for &d in &[0.1, 0.3, 0.5, 0.8, 1.2] {
        for &n in &[10.0, 30.0, 100.0, 500.0] {
            for &alpha in &[0.01, 0.05, 0.10] {
                let p = power_two_sample_t(d, n, alpha, true);
                assert!(p >= 0.0 && p <= 1.0,
                    "Power out of [0,1] for d={}, n={}, α={}: got {}", d, n, alpha, p);
            }
        }
    }
}

/// Larger alpha → more power (easier to reject).
#[test]
fn power_monotone_in_alpha() {
    let p_01 = power_two_sample_t(0.5, 50.0, 0.01, true);
    let p_05 = power_two_sample_t(0.5, 50.0, 0.05, true);
    let p_10 = power_two_sample_t(0.5, 50.0, 0.10, true);
    assert!(p_01 < p_05 && p_05 < p_10,
        "Power should increase with alpha: {} < {} < {}", p_01, p_05, p_10);
}

/// One-sided has more power than two-sided at same effect.
#[test]
fn power_one_sided_greater() {
    let p_two = power_two_sample_t(0.5, 50.0, 0.05, true);
    let p_one = power_two_sample_t(0.5, 50.0, 0.05, false);
    assert!(p_one > p_two,
        "One-sided power should be > two-sided: {} vs {}", p_one, p_two);
}
