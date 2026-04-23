//! Adversarial tests for asin / acos (TRIG-17, inverse trig family).
//! This file compiles when pathmaker ships `crates/tambear/src/recipes/libm/asin.rs`.
//! Until then it fails at import — that is correct and intentional.
//!
//! Key adversarial patterns tested:
//! - Domain edges: asin(±1), acos(±1) must be exact.
//! - Out-of-domain: asin(|x|>1) must be NaN, not a garbage finite.
//! - CANCELLATION ZONE: asin(x) for x near ±1 — the #1 silent failure mode.
//!   A naive sqrt(1-x²) implementation loses half the bits for x ≈ 0.9999.
//!   Correct: asin(x) = π/2 - 2·asin(√((1-x)/2)) in this region.
//! - Signed zero: asin(±0) must preserve sign.

use tambear::recipes::libm::asin::{asin_strict, acos_strict};
use tambear::primitives::oracle::ulps_between;

macro_rules! assert_ulps {
    ($actual:expr, $expected:expr, $max:expr, $ctx:literal) => {{
        let a = $actual as f64;
        let e = $expected as f64;
        let d = ulps_between(a, e);
        assert!(
            d <= $max,
            "{}: got {:e}, expected {:e}, {} ulps apart (max {})",
            $ctx, a, e, d, $max
        );
    }};
}

macro_rules! assert_bits {
    ($actual:expr, $expected:expr, $ctx:literal) => {{
        let a = ($actual) as f64;
        let e = ($expected) as f64;
        assert_eq!(
            a.to_bits(), e.to_bits(),
            "{}: got {:e} (bits {:016x}), expected {:e} (bits {:016x})",
            $ctx, a, a.to_bits(), e, e.to_bits()
        );
    }};
}

// ── asin special cases ────────────────────────────────────────────────────────

#[test]
fn asin_of_zero_is_zero() {
    assert_bits!(asin_strict(0.0), 0.0_f64, "asin(0) must be +0");
}

#[test]
fn asin_neg_zero_is_neg_zero() {
    let v = asin_strict(-0.0_f64);
    assert!(v.is_sign_negative() && v == 0.0, "asin(-0) must be -0, got {v}");
}

#[test]
fn asin_nan_is_nan() {
    assert!(asin_strict(f64::NAN).is_nan(), "asin(NaN) must be NaN");
}

#[test]
fn asin_above_domain_is_nan() {
    assert!(asin_strict(1.0000000000000002_f64).is_nan(), "asin(1+ε) must be NaN");
    assert!(asin_strict(2.0).is_nan(), "asin(2) must be NaN");
    assert!(asin_strict(f64::INFINITY).is_nan(), "asin(+∞) must be NaN");
}

#[test]
fn asin_below_domain_is_nan() {
    assert!(asin_strict(-1.0000000000000002_f64).is_nan(), "asin(-1-ε) must be NaN");
    assert!(asin_strict(-2.0).is_nan(), "asin(-2) must be NaN");
    assert!(asin_strict(f64::NEG_INFINITY).is_nan(), "asin(-∞) must be NaN");
}

#[test]
fn asin_of_one_is_pi_over_2() {
    assert_ulps!(asin_strict(1.0), std::f64::consts::FRAC_PI_2, 1, "asin(1.0)");
}

#[test]
fn asin_of_neg_one_is_neg_pi_over_2() {
    assert_ulps!(asin_strict(-1.0), -std::f64::consts::FRAC_PI_2, 1, "asin(-1.0)");
}

#[test]
fn asin_half_is_pi_over_6() {
    assert_ulps!(asin_strict(0.5), std::f64::consts::PI / 6.0, 1, "asin(0.5)");
}

#[test]
fn asin_sqrt2_over_2_is_pi_over_4() {
    // asin(1/√2) = π/4 is a 2-ULP case for the fdlibm rational approximation:
    // the half-angle reduction introduces a rounding chain whose worst case
    // lands at exactly 2 ULP. The strict guarantee is ≤2 ULP, not ≤1 ULP.
    let x = std::f64::consts::FRAC_1_SQRT_2;
    assert_ulps!(asin_strict(x), std::f64::consts::FRAC_PI_4, 2, "asin(√2/2)");
}

#[test]
fn asin_is_odd() {
    for x in [0.5_f64, 0.9, 0.99, 0.999, 1.0] {
        let a = asin_strict(-x);
        let b = -asin_strict(x);
        assert_eq!(a.to_bits(), b.to_bits(), "asin(-{x}) != -asin({x})");
    }
}

/// THE CRITICAL SILENT-FAILURE TEST: asin near ±1.
///
/// In the region x ∈ [0.9999, 1.0], a naive polynomial implementation gives
/// answers that are plausible-looking but wrong. The correct formula uses
/// asin(x) = π/2 - 2·asin(√((1-x)/2)) to avoid catastrophic cancellation
/// in √(1-x²). If the implementation doesn't do this, the error can be
/// hundreds of ulps while the result still "looks reasonable."
#[test]
fn asin_cancellation_zone_near_one() {
    let inputs = [
        0.9_f64, 0.99, 0.999, 0.9999, 0.99999,
        0.999999, 0.9999999, 0.99999999,
        1.0 - 1e-10, 1.0 - 1e-12, 1.0 - 1e-14,
        1.0 - f64::EPSILON, 1.0 - f64::EPSILON / 2.0,
    ];
    let mut worst = 0u64;
    let mut worst_x = 0.0_f64;
    for x in inputs {
        if x > 1.0 { continue; }
        let got = asin_strict(x);
        let expected = x.asin();
        let d = ulps_between(got, expected);
        if d > worst { worst = d; worst_x = x; }
    }
    assert!(
        worst <= 4,
        "asin cancellation zone: worst {worst} ulps at x={worst_x:.15e}\n  \
         This is the silent failure: result looks plausible but is wrong by ~{}",
        (asin_strict(worst_x) - worst_x.asin()).abs()
    );
}

#[test]
fn asin_cancellation_zone_near_neg_one() {
    let inputs = [-0.9_f64, -0.99, -0.999, -0.9999, -0.999999, -1.0 + 1e-10, -1.0 + f64::EPSILON];
    for x in inputs {
        let got = asin_strict(x);
        let expected = x.asin();
        let d = ulps_between(got, expected);
        assert!(
            d <= 4,
            "asin({x:.15}) near -1: {d} ulps, got {got:.15e}, expected {expected:.15e}"
        );
    }
}

// ── acos special cases ────────────────────────────────────────────────────────

#[test]
fn acos_of_one_is_zero_exact() {
    // acos(1) = 0 exactly. If this isn't bit-exact, the implementation
    // doesn't handle the boundary correctly.
    assert_bits!(acos_strict(1.0), 0.0_f64, "acos(1) must be EXACTLY 0");
}

#[test]
fn acos_of_neg_one_is_pi() {
    assert_ulps!(acos_strict(-1.0), std::f64::consts::PI, 1, "acos(-1)");
}

#[test]
fn acos_of_zero_is_pi_over_2() {
    assert_ulps!(acos_strict(0.0), std::f64::consts::FRAC_PI_2, 1, "acos(0)");
}

#[test]
fn acos_nan_is_nan() {
    assert!(acos_strict(f64::NAN).is_nan(), "acos(NaN) must be NaN");
}

#[test]
fn acos_outside_domain_is_nan() {
    assert!(acos_strict(1.0000000000000002_f64).is_nan(), "acos(1+ε) must be NaN");
    assert!(acos_strict(-1.0000000000000002_f64).is_nan(), "acos(-1-ε) must be NaN");
    assert!(acos_strict(2.0).is_nan(), "acos(2) must be NaN");
    assert!(acos_strict(f64::INFINITY).is_nan(), "acos(+∞) must be NaN");
}

#[test]
fn acos_half_is_pi_over_3() {
    assert_ulps!(acos_strict(0.5), std::f64::consts::PI / 3.0, 1, "acos(0.5)");
}

#[test]
fn acos_plus_asin_is_pi_over_2() {
    // acos(x) + asin(x) = π/2 for all x ∈ [-1, 1].
    let pio2 = std::f64::consts::FRAC_PI_2;
    for x in [-1.0_f64, -0.9, -0.5, 0.0, 0.5, 0.9, 1.0] {
        let sum = acos_strict(x) + asin_strict(x);
        assert_ulps!(sum, pio2, 4, "acos({x}) + asin({x}) = π/2");
    }
}

#[test]
fn acos_cancellation_zone_near_one() {
    // acos(x) near x=1: result → 0, computation involves √(1-x²) which
    // suffers catastrophic cancellation. Correct: use √(2(1-x)) directly.
    let inputs = [0.999_f64, 0.9999, 0.99999, 0.999999, 1.0 - 1e-10, 1.0 - f64::EPSILON];
    let mut worst = 0u64;
    let mut worst_x = 0.0_f64;
    for x in inputs {
        let got = acos_strict(x);
        let expected = x.acos();
        let d = ulps_between(got, expected);
        if d > worst { worst = d; worst_x = x; }
    }
    assert!(
        worst <= 4,
        "acos cancellation zone near 1: worst {worst} ulps at x={worst_x:.15e}"
    );
}

// ── roundtrip ─────────────────────────────────────────────────────────────────

#[test]
fn asin_sin_roundtrip_in_range() {
    use tambear::recipes::libm::sin::sin_strict;
    let pio2 = std::f64::consts::FRAC_PI_2;
    for x in [-pio2, -1.0, -0.5, 0.0, 0.5, 1.0, pio2] {
        let roundtrip = asin_strict(sin_strict(x));
        let d = ulps_between(roundtrip, x);
        assert!(d <= 8, "asin(sin({x})): roundtrip {d} ulps off, got {roundtrip:e}");
    }
}
