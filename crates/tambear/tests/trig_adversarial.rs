//! Adversarial test battery for the full trig family — TRIG-17.
//!
//! This file covers the IMPLEMENTED functions: sin, cos, tan, cot, sec, csc, sincos.
//! Each subsequent family gets its own file that compiles when pathmaker ships it:
//!   trig_adversarial_asin.rs    — asin/acos
//!   trig_adversarial_atan.rs    — atan/atan2/acot/asec/acsc
//!   trig_adversarial_hyp.rs     — sinh/cosh/tanh + asinh/acosh/atanh
//!   trig_adversarial_pi.rs      — sinpi/cospi/tanpi + sincospi
//!   trig_adversarial_rare.rs    — versin/haversin/gudermannian
//!
//! # Test philosophy
//!
//! Every test MUST FAIL when the bug exists. No tolerance-fishing to make
//! broken code look green. Each test asserts what the math REQUIRES.
//!
//! Silent failures are the worst failure mode: asin(0.9999) returning 1.56
//! when it should return 1.5693... is wrong by <1% and looks plausible.
//! The adversarial inputs here are specifically chosen to reveal that mode.

use tambear::recipes::libm::sin::{sin_strict, cos_strict};
use tambear::recipes::libm::tan::{
    tan_strict, cot_strict, sec_strict, csc_strict, sincos_strict,
};
use tambear::primitives::oracle::ulps_between;

// ── Helpers ──────────────────────────────────────────────────────────────────

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
            a.to_bits(),
            e.to_bits(),
            "{}: got {:e} (bits {:016x}), expected {:e} (bits {:016x})",
            $ctx, a, a.to_bits(), e, e.to_bits()
        );
    }};
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── SIN — IEEE 754 special cases ─────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn sin_neg_zero_preserves_sign() {
    let v = sin_strict(-0.0_f64);
    assert!(v.is_sign_negative(), "sin(-0) must be -0, got {v}");
    assert_eq!(v, 0.0, "sin(-0) must have magnitude 0");
}

#[test]
fn sin_pos_zero_is_pos_zero() {
    assert_bits!(sin_strict(0.0), 0.0_f64, "sin(+0) must be +0");
}

#[test]
fn sin_nan_is_nan() {
    assert!(sin_strict(f64::NAN).is_nan(), "sin(NaN) must be NaN");
}

#[test]
fn sin_pos_inf_is_nan() {
    assert!(sin_strict(f64::INFINITY).is_nan(), "sin(+∞) must be NaN");
}

#[test]
fn sin_neg_inf_is_nan() {
    assert!(sin_strict(f64::NEG_INFINITY).is_nan(), "sin(-∞) must be NaN");
}

#[test]
fn sin_half_pi_is_one() {
    assert_ulps!(sin_strict(std::f64::consts::FRAC_PI_2), 1.0, 1, "sin(π/2)");
}

#[test]
fn sin_pi_matches_platform() {
    let got = sin_strict(std::f64::consts::PI);
    assert_ulps!(got, std::f64::consts::PI.sin(), 2, "sin(π)");
}

#[test]
fn sin_is_odd_identity() {
    for x in [0.5_f64, 1.0, 2.0, std::f64::consts::PI / 3.0, 7.3, 100.0] {
        let a = sin_strict(-x);
        let b = -sin_strict(x);
        assert_eq!(a.to_bits(), b.to_bits(), "sin(-{x}) != -sin({x})");
    }
}

#[test]
fn sin_pythagorean_identity_near_zero() {
    for x in [1e-300_f64, 1e-100, 1e-15, 1e-8, 1e-4] {
        let s = sin_strict(x);
        let c = cos_strict(x);
        let sum = s * s + c * c;
        assert_ulps!(sum, 1.0, 4, "sin²+cos²=1 at x={x:e}");
    }
}

#[test]
fn sin_pythagorean_identity_large_x() {
    for x in [1e4_f64, 1e6, 1e8, 1e10, 1e15] {
        let s = sin_strict(x);
        let c = cos_strict(x);
        let sum = s * s + c * c;
        assert_ulps!(sum, 1.0, 8, "sin²+cos²=1 at x={x:e}");
    }
}

#[test]
fn sin_subnormal_equals_input() {
    // For |x| << 1, sin(x) = x exactly (cubic term underflows).
    let tiny = 5e-324_f64;
    assert_eq!(sin_strict(tiny).to_bits(), tiny.to_bits(), "sin(2^-1074) must equal 2^-1074");
}

#[test]
fn sin_min_positive_normal_equals_input() {
    let x = f64::MIN_POSITIVE;
    assert_eq!(sin_strict(x).to_bits(), x.to_bits(), "sin(2^-1022) must equal 2^-1022");
}

#[test]
fn sin_payne_hanek_regime() {
    let samples = [1.65e6_f64, 1e7, 1e10, 1e15, 1.234_567_891_234_567e17];
    for x in samples {
        let got = sin_strict(x);
        let expected = x.sin();
        let d = ulps_between(got, expected);
        assert!(
            d <= 64,
            "sin({x:e}) Payne-Hanek: {d} ulps from platform, got {got:e}, expected {expected:e}"
        );
    }
}

/// Hard range-reduction case: 355/113 ≈ π to 6 decimal places.
/// sin(355) has a tiny reduced argument. A Cody-Waite implementation with
/// insufficient π/2 precision returns garbage here while still producing
/// a number in [-1, 1] that "looks fine."
#[test]
fn sin_355_hard_case() {
    let got = sin_strict(355.0_f64);
    let expected = 355.0_f64.sin();
    let d = ulps_between(got, expected);
    assert!(
        d <= 4,
        "sin(355) hard range-reduction: {d} ulps, got {got:e}, expected {expected:e}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── COS — IEEE 754 special cases ─────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn cos_zero_is_one_exact() {
    assert_bits!(cos_strict(0.0), 1.0_f64, "cos(0) must be exactly 1");
}

#[test]
fn cos_neg_zero_is_one_exact() {
    assert_bits!(cos_strict(-0.0), 1.0_f64, "cos(-0) must be exactly 1");
}

#[test]
fn cos_nan_is_nan() {
    assert!(cos_strict(f64::NAN).is_nan(), "cos(NaN) must be NaN");
}

#[test]
fn cos_pos_inf_is_nan() {
    assert!(cos_strict(f64::INFINITY).is_nan(), "cos(+∞) must be NaN");
}

#[test]
fn cos_neg_inf_is_nan() {
    assert!(cos_strict(f64::NEG_INFINITY).is_nan(), "cos(-∞) must be NaN");
}

#[test]
fn cos_is_even_identity() {
    for x in [0.5_f64, 1.0, 2.0, std::f64::consts::PI / 3.0, 7.3, 100.0] {
        let a = cos_strict(-x);
        let b = cos_strict(x);
        assert_eq!(a.to_bits(), b.to_bits(), "cos(-{x}) != cos({x})");
    }
}

#[test]
fn cos_pi_is_neg_one() {
    assert_ulps!(cos_strict(std::f64::consts::PI), -1.0, 1, "cos(π)");
}

#[test]
fn cos_half_pi_matches_platform() {
    let got = cos_strict(std::f64::consts::FRAC_PI_2);
    assert_ulps!(got, std::f64::consts::FRAC_PI_2.cos(), 2, "cos(π/2)");
}

#[test]
fn cos_subnormal_is_one() {
    assert_bits!(cos_strict(5e-324_f64), 1.0_f64, "cos(2^-1074) must be 1.0");
}

#[test]
fn cos_355_hard_case() {
    let got = cos_strict(355.0_f64);
    let expected = 355.0_f64.cos();
    let d = ulps_between(got, expected);
    assert!(d <= 4, "cos(355) hard range-reduction: {d} ulps");
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── TAN — poles and near-pole silent failures ─────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn tan_zero_is_zero() {
    assert_bits!(tan_strict(0.0), 0.0_f64, "tan(0) must be +0");
}

#[test]
fn tan_neg_zero_is_neg_zero() {
    let v = tan_strict(-0.0_f64);
    assert!(v.is_sign_negative() && v == 0.0, "tan(-0) must be -0, got {v}");
}

#[test]
fn tan_nan_is_nan() {
    assert!(tan_strict(f64::NAN).is_nan(), "tan(NaN) must be NaN");
}

#[test]
fn tan_inf_is_nan() {
    assert!(tan_strict(f64::INFINITY).is_nan(), "tan(+∞) must be NaN");
    assert!(tan_strict(f64::NEG_INFINITY).is_nan(), "tan(-∞) must be NaN");
}

#[test]
fn tan_pi_over_4_is_one() {
    assert_ulps!(tan_strict(std::f64::consts::FRAC_PI_4), 1.0, 1, "tan(π/4)");
}

#[test]
fn tan_neg_pi_over_4_is_neg_one() {
    assert_ulps!(tan_strict(-std::f64::consts::FRAC_PI_4), -1.0, 1, "tan(-π/4)");
}

#[test]
fn tan_pi_matches_platform() {
    let got = tan_strict(std::f64::consts::PI);
    assert_ulps!(got, std::f64::consts::PI.tan(), 2, "tan(π)");
}

#[test]
fn tan_is_odd() {
    for x in [0.5_f64, 1.0, 1.2, 2.9, 10.0] {
        let a = tan_strict(-x);
        let b = -tan_strict(x);
        assert_eq!(a.to_bits(), b.to_bits(), "tan(-{x}) != -tan({x})");
    }
}

/// CRITICAL: tan near π/2 must be very large, not ~1.0.
/// A silent failure: wrong range reduction returns a value near 1 for input
/// near π/2 because it reduces x to a number near π/4 instead of near 0.
#[test]
fn tan_at_f64_pi_over_2_is_very_large() {
    let pio2 = std::f64::consts::FRAC_PI_2;
    let got = tan_strict(pio2);
    let expected = pio2.tan(); // platform reference, ~1.633e16
    assert!(got > 1e15, "tan(f64::FRAC_PI_2) should be very large positive (~1.6e16), got {got:e}");
    let d = ulps_between(got, expected);
    assert!(d <= 16, "tan(π/2 as f64): {d} ulps from platform, got {got:e}, expected {expected:e}");
}

#[test]
fn tan_near_pole_not_smooth() {
    // One ulp below f64 π/2. Must still be very large (closer to pole than tan(π/4)=1).
    let pio2 = std::f64::consts::FRAC_PI_2;
    let x = f64::from_bits(pio2.to_bits() - 1);
    let got = tan_strict(x);
    assert!(
        got > 1e14,
        "tan(π/2 - 1ulp) = {got:e}: should be very large. Silent failure if ~1.0"
    );
}

#[test]
fn tan_payne_hanek_regime() {
    for x in [1e7_f64, 1e12, 1e16] {
        let got = tan_strict(x);
        let expected = x.tan();
        if expected.abs() < 1e14 {
            // Only check non-pole region to avoid amplified pole error.
            let d = ulps_between(got, expected);
            assert!(d <= 64, "tan({x:e}): {d} ulps from platform, got {got:e}");
        }
    }
}

#[test]
fn tan_adversarial_sweep_vs_platform() {
    use tambear::recipes::libm::adversarial::sin_cos_adversarial;
    let mut worst = 0u64;
    let mut worst_x = 0.0_f64;
    for x in sin_cos_adversarial() {
        // Skip inputs near poles (odd multiples of π/2).
        let n = (x * std::f64::consts::FRAC_2_PI).round() as i64;
        if n % 2 != 0 { continue; } // near a pole

        let got = tan_strict(x);
        let expected = x.tan();
        if !got.is_finite() || !expected.is_finite() { continue; }
        let d = ulps_between(got, expected);
        if d > worst { worst = d; worst_x = x; }
    }
    assert!(
        worst <= 8,
        "tan adversarial sweep: worst {worst} ulps at x={worst_x:e}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── COT — pole at 0 ──────────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn cot_nan_is_nan() {
    assert!(cot_strict(f64::NAN).is_nan(), "cot(NaN) must be NaN");
}

#[test]
fn cot_inf_is_nan() {
    assert!(cot_strict(f64::INFINITY).is_nan(), "cot(+∞) must be NaN");
    assert!(cot_strict(f64::NEG_INFINITY).is_nan(), "cot(-∞) must be NaN");
}

#[test]
fn cot_zero_is_pos_infinity() {
    assert_eq!(cot_strict(0.0), f64::INFINITY, "cot(+0) must be +∞");
}

#[test]
fn cot_neg_zero_is_neg_infinity() {
    assert_eq!(cot_strict(-0.0), f64::NEG_INFINITY, "cot(-0) must be -∞");
}

#[test]
fn cot_pi_over_4_is_one() {
    assert_ulps!(cot_strict(std::f64::consts::FRAC_PI_4), 1.0, 1, "cot(π/4)");
}

#[test]
fn cot_is_odd() {
    for x in [0.5_f64, 1.0, 2.0, std::f64::consts::PI / 3.0] {
        let a = cot_strict(-x);
        let b = -cot_strict(x);
        assert_eq!(a.to_bits(), b.to_bits(), "cot(-{x}) != -cot({x})");
    }
}

#[test]
fn cot_agrees_with_tan_reciprocal() {
    for x in [0.5_f64, 1.0, 2.0, 3.0, -1.0, -2.5] {
        let via_cot = cot_strict(x);
        let via_tan = 1.0 / tan_strict(x);
        let d = ulps_between(via_cot, via_tan);
        assert!(d <= 4, "cot({x}) vs 1/tan({x}): {d} ulps, {via_cot:e} vs {via_tan:e}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── SEC / CSC — reciprocal trig ──────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn sec_zero_is_one() {
    assert_bits!(sec_strict(0.0), 1.0_f64, "sec(0) must be 1");
}

#[test]
fn sec_nan_is_nan() {
    assert!(sec_strict(f64::NAN).is_nan(), "sec(NaN) must be NaN");
}

#[test]
fn sec_inf_is_nan() {
    assert!(sec_strict(f64::INFINITY).is_nan(), "sec(+∞) must be NaN");
}

#[test]
fn sec_at_pi_over_2_is_large() {
    // sec(f64 π/2) is NOT ±∞ (f64 π/2 is not exactly π/2) but must be very large.
    let got = sec_strict(std::f64::consts::FRAC_PI_2);
    assert!(got.abs() > 1e15, "sec(π/2 as f64) should be very large, got {got:e}");
}

#[test]
fn sec_pi_is_neg_one() {
    assert_ulps!(sec_strict(std::f64::consts::PI), -1.0, 2, "sec(π)");
}

#[test]
fn sec_is_even() {
    for x in [0.5_f64, 1.0, 2.0, 3.0] {
        assert_eq!(sec_strict(-x).to_bits(), sec_strict(x).to_bits(), "sec(-{x}) != sec({x})");
    }
}

#[test]
fn sec_equals_one_over_cos() {
    for x in [0.5_f64, 1.0, 2.0, 3.0, -1.5] {
        let via_sec = sec_strict(x);
        let via_cos = 1.0 / cos_strict(x);
        let d = ulps_between(via_sec, via_cos);
        assert!(d <= 4, "sec({x}) vs 1/cos({x}): {d} ulps");
    }
}

#[test]
fn csc_at_zero_is_pos_infinity() {
    assert_eq!(csc_strict(0.0), f64::INFINITY, "csc(+0) must be +∞");
}

#[test]
fn csc_at_neg_zero_is_neg_infinity() {
    assert_eq!(csc_strict(-0.0), f64::NEG_INFINITY, "csc(-0) must be -∞");
}

#[test]
fn csc_nan_is_nan() {
    assert!(csc_strict(f64::NAN).is_nan(), "csc(NaN) must be NaN");
}

#[test]
fn csc_inf_is_nan() {
    assert!(csc_strict(f64::INFINITY).is_nan(), "csc(+∞) must be NaN");
}

#[test]
fn csc_half_pi_is_one() {
    assert_ulps!(csc_strict(std::f64::consts::FRAC_PI_2), 1.0, 1, "csc(π/2)");
}

#[test]
fn csc_is_odd() {
    for x in [0.5_f64, 1.0, 2.0, std::f64::consts::PI / 3.0] {
        let a = csc_strict(-x);
        let b = -csc_strict(x);
        assert_eq!(a.to_bits(), b.to_bits(), "csc(-{x}) != -csc({x})");
    }
}

#[test]
fn csc_equals_one_over_sin() {
    for x in [0.5_f64, 1.0, 2.0, 3.0, -1.5] {
        let via_csc = csc_strict(x);
        let via_sin = 1.0 / sin_strict(x);
        let d = ulps_between(via_csc, via_sin);
        assert!(d <= 4, "csc({x}) vs 1/sin({x}): {d} ulps");
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── SINCOS — fused pair ──────────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn sincos_zero() {
    let (s, c): (f64, f64) = sincos_strict(0.0);
    assert_bits!(s, 0.0_f64, "sincos(0).sin must be +0");
    assert_bits!(c, 1.0_f64, "sincos(0).cos must be 1");
}

#[test]
fn sincos_nan_produces_nan_pair() {
    let (s, c): (f64, f64) = sincos_strict(f64::NAN);
    assert!(s.is_nan(), "sincos(NaN).sin must be NaN");
    assert!(c.is_nan(), "sincos(NaN).cos must be NaN");
}

#[test]
fn sincos_inf_produces_nan_pair() {
    let (s, c): (f64, f64) = sincos_strict(f64::INFINITY);
    assert!(s.is_nan(), "sincos(+∞).sin must be NaN");
    assert!(c.is_nan(), "sincos(+∞).cos must be NaN");
}

/// CRITICAL: sincos must return bit-identical results to separate sin/cos calls.
/// If sincos shares intermediate computation to save cycles but loses a bit,
/// it silently breaks the guarantee that the pair is internally consistent.
#[test]
fn sincos_matches_separate_calls_bit_exact() {
    for x in [0.5_f64, 1.0, std::f64::consts::PI / 3.0, 10.0, 100.0, 1e6,
              std::f64::consts::FRAC_PI_4, std::f64::consts::PI, -2.5] {
        let (s, c): (f64, f64) = sincos_strict(x);
        let s_sep = sin_strict(x);
        let c_sep = cos_strict(x);
        assert_eq!(
            s.to_bits(), s_sep.to_bits(),
            "sincos({x}).sin mismatch: {s:e} vs {s_sep:e}"
        );
        assert_eq!(
            c.to_bits(), c_sep.to_bits(),
            "sincos({x}).cos mismatch: {c:e} vs {c_sep:e}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ── CROSS-FAMILY IDENTITIES ───────────────────────────────────────────────────
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn tan_equals_sin_over_cos() {
    for x in [0.5_f64, 1.0, 2.0, 3.0, -1.5, 10.0] {
        let via_tan = tan_strict(x);
        let via_ratio = sin_strict(x) / cos_strict(x);
        if !via_tan.is_finite() { continue; }
        let d = ulps_between(via_tan, via_ratio);
        assert!(d <= 4, "tan({x}) vs sin/cos: {d} ulps, {via_tan:e} vs {via_ratio:e}");
    }
}

#[test]
fn sin_cos_addition_formula() {
    let pairs = [(0.5_f64, 0.3), (1.0, 1.0), (2.0, 0.7), (0.1, std::f64::consts::PI)];
    for (a, b) in pairs {
        let lhs = sin_strict(a + b);
        let rhs = sin_strict(a) * cos_strict(b) + cos_strict(a) * sin_strict(b);
        let d = ulps_between(lhs, rhs);
        assert!(d <= 8, "sin({a}+{b}) addition formula: {d} ulps");
    }
}

/// Large-argument silent failure: a wrong range reduction can map x to a
/// completely wrong reduced argument, returning a "smooth" value near ±0.5
/// when the true result is near ±1. This is undetectable unless you check
/// against the known result.
#[test]
fn sin_does_not_collapse_near_pi_multiple() {
    // sin(10000000.5·π) should be near ±1 (the .5 gives π/2 after reduction).
    let pi = std::f64::consts::PI;
    let x = 10_000_000.5 * pi;
    let got = sin_strict(x);
    let expected = x.sin();
    let d = ulps_between(got, expected);
    assert!(d <= 64, "sin(10000000.5·π): {d} ulps from platform");
    assert!(
        got.abs() > 0.5,
        "sin(10000000.5·π) = {got:e}: should be near ±1, not near 0 (range reduction failure)"
    );
}
