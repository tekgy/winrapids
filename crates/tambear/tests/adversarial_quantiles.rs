//! Adversarial tests for quantile functions: t, chi2, F.
//!
//! These unlock confidence intervals for every test. Each is verified by
//! round-trip (CDF(quantile(p)) = p) plus known-value checks.

use tambear::special_functions::*;

// ═══════════════════════════════════════════════════════════════════════════
// T QUANTILE
// ═══════════════════════════════════════════════════════════════════════════

/// t_quantile round-trip: CDF(quantile(p, df)) should equal p.
#[test]
fn t_quantile_roundtrip() {
    for &df in &[1.0, 2.0, 5.0, 10.0, 30.0, 100.0] {
        for &p in &[0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975] {
            let q = t_quantile(p, df);
            let p_back = t_cdf(q, df);
            assert!((p - p_back).abs() < 1e-6,
                "t_quantile roundtrip failed: p={}, df={}, q={}, cdf(q)={}",
                p, df, q, p_back);
        }
    }
}

/// t_quantile(0.975, Inf) should equal normal quantile ≈ 1.96.
#[test]
fn t_quantile_large_df_approaches_normal() {
    let t_crit = t_quantile(0.975, 10000.0);
    let z_crit = normal_quantile(0.975);
    assert!((t_crit - z_crit).abs() < 0.001,
        "t(df=10000) should converge to normal: {} vs {}", t_crit, z_crit);
    assert!((t_crit - 1.96).abs() < 0.01,
        "z_{{0.975}} should be 1.96, got {}", t_crit);
}

/// t_quantile gold standard: t_{0.025, df=10} = 2.228 (from tables).
#[test]
fn t_quantile_gold_standard() {
    let t = t_quantile(0.975, 10.0);
    assert!((t - 2.228).abs() < 0.01,
        "t_{{0.975, df=10}} should be 2.228, got {}", t);
}

/// t_quantile(p=0.5) = 0 for all df (symmetry).
#[test]
fn t_quantile_median_is_zero() {
    for &df in &[1.0, 5.0, 50.0] {
        let q = t_quantile(0.5, df);
        assert!(q.abs() < 1e-10,
            "t_quantile(0.5, df={}) should be 0, got {}", df, q);
    }
}

/// t_quantile symmetry: q(p, df) = -q(1-p, df).
#[test]
fn t_quantile_symmetry() {
    for &p in &[0.1, 0.25, 0.4] {
        let q_lo = t_quantile(p, 10.0);
        let q_hi = t_quantile(1.0 - p, 10.0);
        assert!((q_lo + q_hi).abs() < 1e-6,
            "t symmetry: q({})=-q({}), got {} vs {}", p, 1.0 - p, q_lo, q_hi);
    }
}

/// t_quantile edge cases.
#[test]
fn t_quantile_edge_cases() {
    assert!(t_quantile(0.0, 5.0).is_infinite() && t_quantile(0.0, 5.0) < 0.0);
    assert!(t_quantile(1.0, 5.0).is_infinite() && t_quantile(1.0, 5.0) > 0.0);
    assert!(t_quantile(0.5, 0.0).is_nan(), "df=0 should be NaN");
    assert!(t_quantile(0.5, -1.0).is_nan(), "df<0 should be NaN");
}

// ═══════════════════════════════════════════════════════════════════════════
// CHI2 QUANTILE
// ═══════════════════════════════════════════════════════════════════════════

/// chi2_quantile round-trip.
#[test]
fn chi2_quantile_roundtrip() {
    for &k in &[1.0, 2.0, 5.0, 10.0, 30.0] {
        for &p in &[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95] {
            let q = chi2_quantile(p, k);
            let p_back = chi2_cdf(q, k);
            assert!((p - p_back).abs() < 1e-5,
                "chi2 roundtrip: p={}, k={}, q={}, cdf(q)={}", p, k, q, p_back);
        }
    }
}

/// chi2 gold standard: chi²_{0.95, df=1} = 3.841.
#[test]
fn chi2_quantile_gold_standard_df1() {
    let q = chi2_quantile(0.95, 1.0);
    assert!((q - 3.841).abs() < 0.01,
        "chi2_{{0.95, df=1}} should be 3.841, got {}", q);
}

/// chi2 gold standard: chi²_{0.95, df=10} = 18.307.
#[test]
fn chi2_quantile_gold_standard_df10() {
    let q = chi2_quantile(0.95, 10.0);
    assert!((q - 18.307).abs() < 0.05,
        "chi2_{{0.95, df=10}} should be 18.307, got {}", q);
}

/// chi2 quantile is non-negative.
#[test]
fn chi2_quantile_non_negative() {
    for &p in &[0.001, 0.01, 0.1, 0.5, 0.9, 0.999] {
        for &k in &[1.0, 5.0, 20.0] {
            let q = chi2_quantile(p, k);
            assert!(q >= 0.0,
                "chi2 quantile should be >= 0: q({}, {}) = {}", p, k, q);
        }
    }
}

/// chi2 quantile edge cases.
#[test]
fn chi2_quantile_edge_cases() {
    assert_eq!(chi2_quantile(0.0, 5.0), 0.0);
    assert!(chi2_quantile(1.0, 5.0).is_infinite());
    assert!(chi2_quantile(0.5, 0.0).is_nan());
    assert!(chi2_quantile(0.5, -1.0).is_nan());
}

// ═══════════════════════════════════════════════════════════════════════════
// F QUANTILE
// ═══════════════════════════════════════════════════════════════════════════

/// f_quantile round-trip.
#[test]
fn f_quantile_roundtrip() {
    for &(d1, d2) in &[(1.0, 5.0), (2.0, 10.0), (5.0, 20.0), (10.0, 30.0)] {
        for &p in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.95] {
            let q = f_quantile(p, d1, d2);
            let p_back = f_cdf(q, d1, d2);
            assert!((p - p_back).abs() < 1e-4,
                "F roundtrip: p={}, d1={}, d2={}, q={}, cdf(q)={}",
                p, d1, d2, q, p_back);
        }
    }
}

/// F gold standard: F_{0.95, 5, 10} = 3.326.
#[test]
fn f_quantile_gold_standard() {
    let q = f_quantile(0.95, 5.0, 10.0);
    assert!((q - 3.326).abs() < 0.05,
        "F_{{0.95, 5, 10}} should be 3.326, got {}", q);
}

/// F gold standard: F_{0.95, 1, ∞} = 3.841 (same as chi2_{0.95, 1}).
#[test]
fn f_quantile_limit_chi2() {
    let q = f_quantile(0.95, 1.0, 10000.0);
    assert!((q - 3.841).abs() < 0.05,
        "F(1, inf) should match chi2(1)/1 = 3.841, got {}", q);
}

/// F quantile is non-negative.
#[test]
fn f_quantile_non_negative() {
    for &p in &[0.01, 0.1, 0.5, 0.9, 0.99] {
        let q = f_quantile(p, 5.0, 10.0);
        assert!(q >= 0.0,
            "F quantile should be >= 0: q({}) = {}", p, q);
    }
}

/// F quantile edge cases.
#[test]
fn f_quantile_edge_cases() {
    assert_eq!(f_quantile(0.0, 5.0, 10.0), 0.0);
    assert!(f_quantile(1.0, 5.0, 10.0).is_infinite());
    assert!(f_quantile(0.5, 0.0, 10.0).is_nan());
    assert!(f_quantile(0.5, 5.0, 0.0).is_nan());
}

// ═══════════════════════════════════════════════════════════════════════════
// CROSS-VERIFICATION (connecting distributions)
// ═══════════════════════════════════════════════════════════════════════════

/// t²_{df} = F_{1, df}: the square of a t variable with df degrees of freedom
/// is an F with (1, df) degrees of freedom.
#[test]
fn t_squared_equals_f() {
    let t_95 = t_quantile(0.975, 10.0); // two-sided 5%
    let f_95 = f_quantile(0.95, 1.0, 10.0);
    assert!((t_95 * t_95 - f_95).abs() < 0.05,
        "t²_{{0.975,10}} should equal F_{{0.95,1,10}}: {} vs {}", t_95 * t_95, f_95);
}
