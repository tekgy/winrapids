//! Adversarial tests for atan / atan2 / acot / asec / acsc (TRIG-17).
//! Compiles when pathmaker ships the atan and inv_recip modules.
//!
//! Key adversarial patterns:
//! - atan2: 20+ IEEE 754-2019 §9.2.1 exact cases for every ±0/±∞ combination.
//! - atan near algorithm branch at |x|=1: implementation discontinuity risk.
//! - acot, asec, acsc domain edges.

use tambear::recipes::libm::atan::{atan_strict, atan2_strict};
use tambear::recipes::libm::inv_recip::{acot_strict, asec_strict, acsc_strict};
use tambear::primitives::oracle::ulps_between;

macro_rules! assert_ulps {
    ($actual:expr, $expected:expr, $max:expr, $ctx:literal) => {{
        let a = $actual as f64;
        let e = $expected as f64;
        let d = ulps_between(a, e);
        assert!(d <= $max, "{}: got {:e}, expected {:e}, {} ulps (max {})", $ctx, a, e, d, $max);
    }};
}

macro_rules! assert_bits {
    ($actual:expr, $expected:expr, $ctx:literal) => {{
        let a = ($actual) as f64;
        let e = ($expected) as f64;
        assert_eq!(a.to_bits(), e.to_bits(), "{}: got {:e}, expected {:e}", $ctx, a, e);
    }};
}

// ── atan special cases ────────────────────────────────────────────────────────

#[test]
fn atan_zero_is_zero() {
    assert_bits!(atan_strict(0.0), 0.0_f64, "atan(0) must be +0");
}

#[test]
fn atan_neg_zero_is_neg_zero() {
    let v = atan_strict(-0.0_f64);
    assert!(v.is_sign_negative() && v == 0.0, "atan(-0) must be -0, got {v}");
}

#[test]
fn atan_pos_inf_is_pi_over_2() {
    // atan(+∞) = π/2 exactly.
    assert_ulps!(atan_strict(f64::INFINITY), std::f64::consts::FRAC_PI_2, 0, "atan(+∞)");
}

#[test]
fn atan_neg_inf_is_neg_pi_over_2() {
    assert_ulps!(atan_strict(f64::NEG_INFINITY), -std::f64::consts::FRAC_PI_2, 0, "atan(-∞)");
}

#[test]
fn atan_nan_is_nan() {
    assert!(atan_strict(f64::NAN).is_nan(), "atan(NaN) must be NaN");
}

#[test]
fn atan_one_is_pi_over_4() {
    assert_ulps!(atan_strict(1.0), std::f64::consts::FRAC_PI_4, 1, "atan(1)");
}

#[test]
fn atan_neg_one_is_neg_pi_over_4() {
    assert_ulps!(atan_strict(-1.0), -std::f64::consts::FRAC_PI_4, 1, "atan(-1)");
}

#[test]
fn atan_is_odd() {
    for x in [0.5_f64, 1.0, 10.0, 1e6] {
        let a = atan_strict(-x);
        let b = -atan_strict(x);
        assert_eq!(a.to_bits(), b.to_bits(), "atan(-{x}) != -atan({x})");
    }
}

#[test]
fn atan_large_x_approaches_pi_over_2() {
    let pio2 = std::f64::consts::FRAC_PI_2;
    for x in [1e15_f64, 1e100, 1e300, f64::MAX] {
        let got = atan_strict(x);
        let d = ulps_between(got, pio2);
        assert!(d <= 4, "atan({x:e}) should be within 4 ulps of π/2, {d} ulps");
    }
}

#[test]
fn atan_algorithm_branch_at_one() {
    // Many atan implementations change algorithm at |x|=1. Test the neighborhood.
    for x in [0.9_f64, 0.99, 0.999, 1.001, 1.01, 1.1] {
        let got = atan_strict(x);
        let expected = x.atan();
        let d = ulps_between(got, expected);
        assert!(d <= 2, "atan({x}) at branch: {d} ulps, {got:e} vs {expected:e}");
    }
}

// ── atan2 — IEEE 754-2019 §9.2.1 exact cases ─────────────────────────────────
//
// Every combination of {+0, -0, finite+, finite-, +∞, -∞, NaN} × 2
// has a specified result. These tests are non-negotiable.

#[test]
fn atan2_nan_propagation() {
    assert!(atan2_strict(f64::NAN, 1.0).is_nan(), "atan2(NaN, 1)");
    assert!(atan2_strict(1.0, f64::NAN).is_nan(), "atan2(1, NaN)");
    assert!(atan2_strict(f64::NAN, f64::NAN).is_nan(), "atan2(NaN, NaN)");
}

#[test]
fn atan2_pos_zero_pos_x() {
    // atan2(+0, x) for x > 0 = +0
    let v = atan2_strict(0.0, 1.0);
    assert!(!v.is_sign_negative() && v == 0.0, "atan2(+0, 1) must be +0, got {v}");
    let v = atan2_strict(0.0, f64::INFINITY);
    assert!(!v.is_sign_negative() && v == 0.0, "atan2(+0, +∞) must be +0, got {v}");
}

#[test]
fn atan2_neg_zero_pos_x() {
    // atan2(-0, x) for x > 0 = -0
    let v = atan2_strict(-0.0, 1.0);
    assert!(v.is_sign_negative() && v == 0.0, "atan2(-0, 1) must be -0, got {v}");
    let v = atan2_strict(-0.0, f64::INFINITY);
    assert!(v.is_sign_negative() && v == 0.0, "atan2(-0, +∞) must be -0, got {v}");
}

#[test]
fn atan2_pos_zero_neg_x() {
    // atan2(+0, x) for x < 0 = +π
    assert_ulps!(atan2_strict(0.0, -1.0), std::f64::consts::PI, 0, "atan2(+0, -1)");
    assert_ulps!(atan2_strict(0.0, f64::NEG_INFINITY), std::f64::consts::PI, 0, "atan2(+0, -∞)");
}

#[test]
fn atan2_neg_zero_neg_x() {
    // atan2(-0, x) for x < 0 = -π
    assert_ulps!(atan2_strict(-0.0, -1.0), -std::f64::consts::PI, 0, "atan2(-0, -1)");
    assert_ulps!(atan2_strict(-0.0, f64::NEG_INFINITY), -std::f64::consts::PI, 0, "atan2(-0, -∞)");
}

#[test]
fn atan2_pos_y_zero_x() {
    // atan2(y, +0) for y > 0 = +π/2
    assert_ulps!(atan2_strict(1.0, 0.0), std::f64::consts::FRAC_PI_2, 0, "atan2(1, +0)");
    assert_ulps!(atan2_strict(f64::INFINITY, 0.0), std::f64::consts::FRAC_PI_2, 0, "atan2(+∞, +0)");
}

#[test]
fn atan2_neg_y_zero_x() {
    // atan2(y, +0) for y < 0 = -π/2
    assert_ulps!(atan2_strict(-1.0, 0.0), -std::f64::consts::FRAC_PI_2, 0, "atan2(-1, +0)");
    assert_ulps!(atan2_strict(f64::NEG_INFINITY, 0.0), -std::f64::consts::FRAC_PI_2, 0, "atan2(-∞, +0)");
}

#[test]
fn atan2_pos_y_neg_zero_x() {
    assert_ulps!(atan2_strict(1.0, -0.0), std::f64::consts::FRAC_PI_2, 0, "atan2(1, -0)");
}

#[test]
fn atan2_neg_y_neg_zero_x() {
    assert_ulps!(atan2_strict(-1.0, -0.0), -std::f64::consts::FRAC_PI_2, 0, "atan2(-1, -0)");
}

#[test]
fn atan2_zero_zero_four_cases() {
    let pi = std::f64::consts::PI;
    // atan2(+0, +0) = +0
    let v = atan2_strict(0.0, 0.0);
    assert!(!v.is_sign_negative() && v == 0.0, "atan2(+0,+0) must be +0, got {v}");
    // atan2(-0, +0) = -0
    let v = atan2_strict(-0.0, 0.0);
    assert!(v.is_sign_negative() && v == 0.0, "atan2(-0,+0) must be -0, got {v}");
    // atan2(+0, -0) = +π
    assert_ulps!(atan2_strict(0.0, -0.0), pi, 0, "atan2(+0,-0)");
    // atan2(-0, -0) = -π
    assert_ulps!(atan2_strict(-0.0, -0.0), -pi, 0, "atan2(-0,-0)");
}

#[test]
fn atan2_inf_inf_four_corners() {
    let pi = std::f64::consts::PI;
    let pio4 = std::f64::consts::FRAC_PI_4;
    // atan2(+∞, +∞) = +π/4
    assert_ulps!(atan2_strict(f64::INFINITY, f64::INFINITY), pio4, 0, "atan2(+∞,+∞)");
    // atan2(+∞, -∞) = +3π/4
    assert_ulps!(atan2_strict(f64::INFINITY, f64::NEG_INFINITY), 3.0 * pi / 4.0, 0, "atan2(+∞,-∞)");
    // atan2(-∞, +∞) = -π/4
    assert_ulps!(atan2_strict(f64::NEG_INFINITY, f64::INFINITY), -pio4, 0, "atan2(-∞,+∞)");
    // atan2(-∞, -∞) = -3π/4
    assert_ulps!(atan2_strict(f64::NEG_INFINITY, f64::NEG_INFINITY), -3.0 * pi / 4.0, 0, "atan2(-∞,-∞)");
}

#[test]
fn atan2_finite_inf_x() {
    // atan2(y, +∞) = +0 for finite y (regardless of sign of y, but with sign matching y).
    // Actually: atan2(y, +∞) = +0 for y > 0, -0 for y < 0, per IEEE 754.
    let v_pos = atan2_strict(1.0, f64::INFINITY);
    let v_neg = atan2_strict(-1.0, f64::INFINITY);
    assert!(!v_pos.is_sign_negative() && v_pos == 0.0, "atan2(1,+∞) must be +0");
    assert!(v_neg.is_sign_negative() && v_neg == 0.0, "atan2(-1,+∞) must be -0");
    // atan2(y, -∞): ±π
    assert_ulps!(atan2_strict(1.0, f64::NEG_INFINITY), std::f64::consts::PI, 0, "atan2(1,-∞)");
    assert_ulps!(atan2_strict(-1.0, f64::NEG_INFINITY), -std::f64::consts::PI, 0, "atan2(-1,-∞)");
}

#[test]
fn atan2_accuracy_normal_inputs() {
    let ys = [-1e10_f64, -1.0, -0.001, 0.001, 1.0, 1e10];
    let xs = [-1e10_f64, -1.0, -0.001, 0.001, 1.0, 1e10];
    for y in ys {
        for x in xs {
            let got = atan2_strict(y, x);
            let expected = y.atan2(x);
            let d = ulps_between(got, expected);
            assert!(d <= 2, "atan2({y}, {x}): {d} ulps, {got:e} vs {expected:e}");
        }
    }
}

// ── acot / asec / acsc ────────────────────────────────────────────────────────

#[test]
fn acot_one_is_pi_over_4() {
    assert_ulps!(acot_strict(1.0), std::f64::consts::FRAC_PI_4, 1, "acot(1)");
}

#[test]
fn acot_zero_is_pi_over_2() {
    assert_ulps!(acot_strict(0.0), std::f64::consts::FRAC_PI_2, 1, "acot(+0)");
}

#[test]
fn acot_nan_is_nan() {
    assert!(acot_strict(f64::NAN).is_nan(), "acot(NaN) must be NaN");
}

#[test]
fn acot_pos_inf_is_zero() {
    // acot(+∞) = 0.
    assert_ulps!(acot_strict(f64::INFINITY), 0.0, 0, "acot(+∞)");
}

#[test]
fn asec_one_is_zero() {
    assert_bits!(asec_strict(1.0), 0.0_f64, "asec(1) must be 0");
}

#[test]
fn asec_neg_one_is_pi() {
    assert_ulps!(asec_strict(-1.0), std::f64::consts::PI, 1, "asec(-1)");
}

#[test]
fn asec_inside_domain_is_nan() {
    assert!(asec_strict(0.0).is_nan(), "asec(0) must be NaN");
    assert!(asec_strict(0.5).is_nan(), "asec(0.5) must be NaN");
    assert!(asec_strict(-0.5).is_nan(), "asec(-0.5) must be NaN");
    assert!(asec_strict(f64::NAN).is_nan(), "asec(NaN) must be NaN");
}

#[test]
fn acsc_one_is_pi_over_2() {
    assert_ulps!(acsc_strict(1.0), std::f64::consts::FRAC_PI_2, 1, "acsc(1)");
}

#[test]
fn acsc_neg_one_is_neg_pi_over_2() {
    assert_ulps!(acsc_strict(-1.0), -std::f64::consts::FRAC_PI_2, 1, "acsc(-1)");
}

#[test]
fn acsc_inside_domain_is_nan() {
    assert!(acsc_strict(0.0).is_nan(), "acsc(0) must be NaN");
    assert!(acsc_strict(0.5).is_nan(), "acsc(0.5) must be NaN");
    assert!(acsc_strict(f64::NAN).is_nan(), "acsc(NaN) must be NaN");
}

// ── roundtrip ─────────────────────────────────────────────────────────────────

#[test]
fn atan_tan_roundtrip_in_range() {
    use tambear::recipes::libm::tan::tan_strict;
    for x in [-1.4_f64, -1.0, -0.5, 0.0, 0.5, 1.0, 1.4] {
        let roundtrip = atan_strict(tan_strict(x));
        let d = ulps_between(roundtrip, x);
        assert!(d <= 8, "atan(tan({x})): {d} ulps, got {roundtrip:e}");
    }
}
