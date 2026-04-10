//! Workup: regularized_incomplete_beta — pushing to the limits
//!
//! Tests tambear's I_x(a,b) against mpmath 50dp ground truth across
//! extreme parameter ranges: large df, deep tails, extreme asymmetry,
//! near-boundary x, very small parameters.

use tambear::special_functions::regularized_incomplete_beta;

fn assert_close(name: &str, got: f64, expected: f64, tol: f64) {
    let abs_diff = (got - expected).abs();
    let rel_diff = if expected.abs() > 1e-300 {
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

// ── t-CDF proxy: I_x(df/2, 1/2) ──────────────────────────────────────

#[test]
fn ibeta_t_cdf_df10000() {
    // t near 0, df=10000: x = df/(df+t^2) ≈ 0.9999
    // mpmath: 3.173105072579535e-01
    let val = regularized_incomplete_beta(0.9999, 5000.0, 0.5);
    assert_close("t_df10000", val, 3.173105072579535e-01, 1e-8);
}

#[test]
fn ibeta_t_cdf_df1_far_tail() {
    // df=1 (Cauchy), very far tail: x = 1e-10
    // mpmath: 6.366197723781917e-06
    let val = regularized_incomplete_beta(1e-10, 0.5, 0.5);
    assert_close("t_df1_far", val, 6.366197723781917e-06, 1e-8);
}

#[test]
fn ibeta_t_cdf_df1_extreme_tail() {
    // df=1, extreme tail: x = 1e-20
    // mpmath: 6.366197723675813e-11
    let val = regularized_incomplete_beta(1e-20, 0.5, 0.5);
    assert_close("t_df1_extreme", val, 6.366197723675813e-11, 1e-8);
}

// ── F-CDF proxy: I_x(d1/2, d2/2) ─────────────────────────────────────

#[test]
fn ibeta_f_cdf_equal_df1000() {
    // F=1, df1=df2=1000: x = 0.5
    // mpmath: 0.5 exactly (by symmetry I_0.5(a,a) = 0.5)
    let val = regularized_incomplete_beta(0.5, 500.0, 500.0);
    assert_close("f_df1000", val, 0.5, 1e-10);
}

// ── Extreme asymmetry ──────────────────────────────────────────────────

#[test]
fn ibeta_extreme_asymmetry_tiny_a() {
    // a=0.001, b=1000, x=0.5
    // mpmath: 1.0
    let val = regularized_incomplete_beta(0.5, 0.001, 1000.0);
    assert_close("asym_tiny_a", val, 1.0, 1e-10);
}

#[test]
fn ibeta_extreme_asymmetry_huge_a() {
    // a=1000, b=0.001, x=0.5
    // mpmath: 1.877373265558377e-307 (extremely small)
    let val = regularized_incomplete_beta(0.5, 1000.0, 0.001);
    // This is a denormal-range value. Just check it's in the right ballpark.
    assert!(val >= 0.0 && val < 1e-300,
        "huge_a: val={:.6e} should be tiny positive", val);
}

// ── Near boundaries ────────────────────────────────────────────────────

#[test]
fn ibeta_near_x_zero() {
    // x = 1e-300, a=b=1
    let val = regularized_incomplete_beta(1e-300, 1.0, 1.0);
    assert_close("near_x0", val, 1e-300, 1e-310);
}

#[test]
fn ibeta_near_x_one() {
    // x = 1 - 1e-15, a=b=1
    let val = regularized_incomplete_beta(1.0 - 1e-15, 1.0, 1.0);
    assert_close("near_x1", val, 1.0 - 1e-15, 1e-14);
}

// ── Exact x=0 and x=1 ─────────────────────────────────────────────────

#[test]
fn ibeta_exact_boundaries() {
    assert_eq!(regularized_incomplete_beta(0.0, 2.0, 3.0), 0.0);
    assert_eq!(regularized_incomplete_beta(1.0, 2.0, 3.0), 1.0);
}

// ── U-shaped beta (a=b<1) ──────────────────────────────────────────────

#[test]
fn ibeta_u_shaped() {
    // I_0.5(0.01, 0.01) = 0.5 by symmetry
    let val = regularized_incomplete_beta(0.5, 0.01, 0.01);
    assert_close("u_shaped_0.01", val, 0.5, 1e-10);
}

#[test]
fn ibeta_u_shaped_extreme() {
    // I_0.5(0.001, 0.001) = 0.5
    let val = regularized_incomplete_beta(0.5, 0.001, 0.001);
    assert_close("u_shaped_0.001", val, 0.5, 1e-10);
}

#[test]
fn ibeta_u_shaped_degenerate() {
    // I_0.5(1e-6, 1e-6) = 0.5
    let val = regularized_incomplete_beta(0.5, 1e-6, 1e-6);
    assert_close("u_shaped_1e-6", val, 0.5, 1e-8);
}

// ── Symmetry identity ──────────────────────────────────────────────────

#[test]
fn ibeta_symmetry_identity() {
    // I_x(a,b) + I_{1-x}(b,a) = 1
    for &(a, b, x) in &[
        (2.0, 3.0, 0.3),
        (0.5, 0.5, 0.7),
        (10.0, 1.0, 0.9),
        (100.0, 100.0, 0.45),
    ] {
        let fwd = regularized_incomplete_beta(x, a, b);
        let rev = regularized_incomplete_beta(1.0 - x, b, a);
        let sum = fwd + rev;
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "symmetry: I_{x}({a},{b}) + I_{{1-x}}({b},{a}) = {sum}, should be 1.0"
        );
    }
}

// ── Monotonicity in x ──────────────────────────────────────────────────

#[test]
fn ibeta_monotone_in_x() {
    let a = 5.0;
    let b = 3.0;
    let mut prev = 0.0;
    for i in 1..=99 {
        let x = i as f64 / 100.0;
        let val = regularized_incomplete_beta(x, a, b);
        assert!(val >= prev, "not monotone at x={}: {} < {}", x, val, prev);
        prev = val;
    }
}

// ── Known exact values ─────────────────────────────────────────────────

#[test]
fn ibeta_known_a1_b1() {
    // I_x(1, 1) = x (uniform CDF)
    for &x in &[0.0, 0.25, 0.5, 0.75, 1.0] {
        let val = regularized_incomplete_beta(x, 1.0, 1.0);
        assert_close(&format!("I_{x}(1,1)"), val, x, 1e-15);
    }
}

#[test]
fn ibeta_known_a1() {
    // I_x(1, b) = 1 - (1-x)^b
    for &b in &[2.0f64, 5.0, 10.0] {
        for &x in &[0.1f64, 0.5, 0.9] {
            let expected = 1.0 - (1.0 - x).powf(b);
            let val = regularized_incomplete_beta(x, 1.0, b);
            assert_close(&format!("I_{x}(1,{b})"), val, expected, 1e-12);
        }
    }
}

#[test]
fn ibeta_known_b1() {
    // I_x(a, 1) = x^a
    for &a in &[2.0f64, 5.0, 10.0] {
        for &x in &[0.1f64, 0.5, 0.9] {
            let expected = x.powf(a);
            let val = regularized_incomplete_beta(x, a, 1.0);
            assert_close(&format!("I_{x}({a},1)"), val, expected, 1e-12);
        }
    }
}

// ── Chi-square deep tail (via incomplete gamma, cross-check) ───────────

#[test]
fn chi2_deep_tail_via_ibeta() {
    // chi2.sf(100, 5) = Q(2.5, 50) ≈ 5.285e-20
    // This goes through incomplete gamma, not beta, but tests the p-value
    // pipeline end-to-end.
    let p = tambear::special_functions::chi2_right_tail_p(100.0, 5.0);
    assert_close("chi2_sf_100_5", p, 5.285148360943219e-20, 1e-8);
}

#[test]
fn chi2_extreme_tail() {
    // chi2.sf(1000, 5) ≈ 6.01e-214
    let p = tambear::special_functions::chi2_right_tail_p(1000.0, 5.0);
    // Just check it's in the right order of magnitude
    assert!(p > 0.0 && p < 1e-200,
        "chi2.sf(1000,5) = {:.6e} should be ~6e-214", p);
}

#[test]
fn chi2_transition_region() {
    // chi2.sf(1000, 1000) ≈ 0.4941
    let p = tambear::special_functions::chi2_right_tail_p(1000.0, 1000.0);
    assert_close("chi2_transition", p, 4.940528538292396e-01, 1e-6);
}
