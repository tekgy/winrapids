//! Adversarial tests for sinh/cosh/tanh + asinh/acosh/atanh (TRIG-17).
//! Compiles when pathmaker ships the hyperbolic and inv_hyperbolic modules.
//!
//! Key adversarial patterns:
//! - sinh/cosh overflow threshold: sinh(800) must be ±∞, not a huge-finite.
//! - Hyperbolic Pythagorean identity: cosh²-sinh²=1.
//! - tanh bounded strictly in (-1,1) for all finite inputs.
//! - ATANH CANCELLATION: atanh(x) near ±1 — identical problem to asin near ±1.
//! - ACOSH CANCELLATION: acosh(x) near 1 — √(x²-1) loses bits for x ≈ 1.
//! - asinh cancellation: log1p(x + √(1+x²)) vs correct form for small x.

use tambear::recipes::libm::hyperbolic::{sinh_strict, cosh_strict, tanh_strict};
use tambear::recipes::libm::inv_hyperbolic::{asinh_strict, acosh_strict, atanh_strict};
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

// ── sinh ──────────────────────────────────────────────────────────────────────

#[test] fn sinh_zero_is_zero() { assert_bits!(sinh_strict(0.0), 0.0_f64, "sinh(0)"); }
#[test] fn sinh_neg_zero_is_neg_zero() {
    let v = sinh_strict(-0.0_f64);
    assert!(v.is_sign_negative() && v == 0.0, "sinh(-0) must be -0");
}
#[test] fn sinh_nan_is_nan() { assert!(sinh_strict(f64::NAN).is_nan(), "sinh(NaN)"); }
#[test] fn sinh_pos_inf_is_pos_inf() { assert_eq!(sinh_strict(f64::INFINITY), f64::INFINITY, "sinh(+∞)"); }
#[test] fn sinh_neg_inf_is_neg_inf() { assert_eq!(sinh_strict(f64::NEG_INFINITY), f64::NEG_INFINITY, "sinh(-∞)"); }

#[test]
fn sinh_is_odd() {
    for x in [0.5_f64, 1.0, 10.0, 100.0] {
        assert_eq!(sinh_strict(-x).to_bits(), (-sinh_strict(x)).to_bits(), "sinh(-{x}) != -sinh({x})");
    }
}

#[test]
fn sinh_large_arg_overflows_to_infinity() {
    // sinh(800) must overflow to ±∞. A wrong implementation might return a
    // very large finite number instead of ∞.
    assert_eq!(sinh_strict(800.0), f64::INFINITY, "sinh(800) must be +∞");
    assert_eq!(sinh_strict(-800.0), f64::NEG_INFINITY, "sinh(-800) must be -∞");
}

#[test]
fn sinh_tiny_x_equals_x() {
    // For |x| < 2^-26, sinh(x) = x to full precision.
    let tiny = 1e-20_f64;
    assert_bits!(sinh_strict(tiny), tiny, "sinh(1e-20) must equal 1e-20");
}

// ── cosh ──────────────────────────────────────────────────────────────────────

#[test] fn cosh_zero_is_one() { assert_bits!(cosh_strict(0.0), 1.0_f64, "cosh(0)"); }
#[test] fn cosh_nan_is_nan() { assert!(cosh_strict(f64::NAN).is_nan(), "cosh(NaN)"); }
#[test] fn cosh_pos_inf_is_pos_inf() { assert_eq!(cosh_strict(f64::INFINITY), f64::INFINITY, "cosh(+∞)"); }
#[test] fn cosh_neg_inf_is_pos_inf() { assert_eq!(cosh_strict(f64::NEG_INFINITY), f64::INFINITY, "cosh(-∞)"); }

#[test]
fn cosh_is_even() {
    for x in [0.5_f64, 1.0, 10.0, 100.0] {
        assert_eq!(cosh_strict(-x).to_bits(), cosh_strict(x).to_bits(), "cosh(-{x}) != cosh({x})");
    }
}

#[test]
fn cosh_is_at_least_one() {
    for x in [-100.0_f64, -1.0, -0.5, 0.0, 0.5, 1.0, 100.0] {
        let v = cosh_strict(x);
        assert!(v >= 1.0, "cosh({x}) = {v}, must be ≥ 1");
    }
}

/// The hyperbolic Pythagorean identity. A wrong shared-pass implementation
/// that computes cosh and sinh using different internal representations can
/// fail this identity by more than 1 ulp.
///
/// NOTE: For large x (say x > 10), cosh(x) ≈ sinh(x) ≈ e^x/2. Their squares
/// are approximately equal, so cosh²-sinh² in f64 rounds to 0, not 1. This is
/// unavoidable when computing cosh and sinh independently — the identity becomes
/// unrepresentable in the output ULP when the values are this large. We test
/// only the range where the identity is numerically verifiable (|x| ≤ 10).
#[test]
fn hyperbolic_pythagorean_identity() {
    // For large x, cosh ≈ sinh ≈ e^x/2. Computing them independently means
    // cosh²-sinh² rounds to 0 in f64, violating the identity. This is
    // fundamental to f64 arithmetic, not an implementation bug. We test
    // only where the identity is computable to 8 ULPs.
    for x in [0.0_f64, 0.5, 1.0] {
        let c = cosh_strict(x);
        let s = sinh_strict(x);
        let diff = c * c - s * s;
        assert_ulps!(diff, 1.0, 8, "cosh²({x}) - sinh²({x}) = 1");
    }
}

// ── tanh ──────────────────────────────────────────────────────────────────────

#[test] fn tanh_zero_is_zero() { assert_bits!(tanh_strict(0.0), 0.0_f64, "tanh(0)"); }
#[test] fn tanh_nan_is_nan() { assert!(tanh_strict(f64::NAN).is_nan(), "tanh(NaN)"); }

#[test]
fn tanh_pos_inf_is_exactly_one() {
    // tanh(+∞) = 1 exactly, not 0.9999... or 1.0000...001.
    assert_bits!(tanh_strict(f64::INFINITY), 1.0_f64, "tanh(+∞) must be EXACTLY 1");
}

#[test]
fn tanh_neg_inf_is_exactly_neg_one() {
    assert_bits!(tanh_strict(f64::NEG_INFINITY), -1.0_f64, "tanh(-∞) must be EXACTLY -1");
}

#[test]
fn tanh_bounded_for_finite_input() {
    for x in [-100.0_f64, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
        let v = tanh_strict(x);
        assert!(v.abs() <= 1.0, "tanh({x}) = {v}, must be in [-1, 1]");
        assert!(v.abs() < 1.0, "tanh({x}) = {v}, must be strictly in (-1, 1) for finite x");
    }
}

#[test]
fn tanh_is_odd() {
    for x in [0.5_f64, 1.0, 10.0, 100.0] {
        assert_eq!(tanh_strict(-x).to_bits(), (-tanh_strict(x)).to_bits(), "tanh(-{x}) != -tanh({x})");
    }
}

// ── asinh ─────────────────────────────────────────────────────────────────────

#[test] fn asinh_zero_is_zero() { assert_bits!(asinh_strict(0.0), 0.0_f64, "asinh(0)"); }
#[test] fn asinh_neg_zero_is_neg_zero() {
    let v = asinh_strict(-0.0_f64);
    assert!(v.is_sign_negative() && v == 0.0, "asinh(-0) must be -0");
}
#[test] fn asinh_nan_is_nan() { assert!(asinh_strict(f64::NAN).is_nan(), "asinh(NaN)"); }
#[test] fn asinh_inf_is_inf() { assert_eq!(asinh_strict(f64::INFINITY), f64::INFINITY, "asinh(+∞)"); }
#[test] fn asinh_neg_inf_is_neg_inf() { assert_eq!(asinh_strict(f64::NEG_INFINITY), f64::NEG_INFINITY, "asinh(-∞)"); }

#[test]
fn asinh_is_odd() {
    for x in [0.5_f64, 1.0, 10.0, 100.0] {
        assert_eq!(asinh_strict(-x).to_bits(), (-asinh_strict(x)).to_bits(), "asinh(-{x}) != -asinh({x})");
    }
}

/// asinh near-zero cancellation test.
/// For small x, asinh(x) = x - x³/6 + ...
/// A naive log(x + sqrt(1+x²)) loses half the bits for tiny x because
/// sqrt(1+x²) rounds to 1 and the x term vanishes.
/// Correct formula: asinh(x) = log1p(x + x²/(1 + sqrt(1 + x²))).
#[test]
fn asinh_tiny_x_cancellation() {
    for x in [1e-8_f64, 1e-10, 1e-12, 1e-14, 1e-300] {
        let got = asinh_strict(x);
        let expected = x.asinh();
        let d = ulps_between(got, expected);
        assert!(d <= 4, "asinh({x:e}) cancellation: {d} ulps, {got:e} vs {expected:e}");
    }
}

// ── acosh ─────────────────────────────────────────────────────────────────────

#[test]
fn acosh_one_is_zero_exact() {
    assert_bits!(acosh_strict(1.0), 0.0_f64, "acosh(1) must be EXACTLY 0");
}

#[test]
fn acosh_below_domain_is_nan() {
    assert!(acosh_strict(0.0).is_nan(), "acosh(0) must be NaN");
    assert!(acosh_strict(-1.0).is_nan(), "acosh(-1) must be NaN");
    assert!(acosh_strict(1.0 - f64::EPSILON).is_nan(), "acosh(1-ε) must be NaN");
    assert!(acosh_strict(f64::NEG_INFINITY).is_nan(), "acosh(-∞) must be NaN");
    assert!(acosh_strict(f64::NAN).is_nan(), "acosh(NaN) must be NaN");
}

#[test]
fn acosh_inf_is_inf() { assert_eq!(acosh_strict(f64::INFINITY), f64::INFINITY, "acosh(+∞)"); }

/// acosh cancellation near x=1.
/// acosh(x) ≈ √(2(x-1)) for x near 1.
/// A naive log(x + sqrt(x²-1)) computes x²-1 = (x-1)(x+1) ≈ 2(x-1) and
/// loses all bits in the sqrt for x very close to 1.
/// Correct: acosh(x) = log1p(x-1 + sqrt((x-1)(x+1))).
///
/// NOTE ON ORACLE: Windows MSVC libm acosh has 34k ULP error at x = 1+1e-10
/// (verified via mpmath at 100-digit precision: correct answer is 1.4142136208675862e-5,
/// platform gives 1.4142136208733940e-5). We use mpmath-derived gold-standard values
/// for the two most extreme near-1 inputs, and platform for the rest where MSVC is
/// accurate.
#[test]
fn acosh_cancellation_near_one() {
    // mpmath-100-digit gold standard (platform MSVC acosh is wrong in the near-1 region):
    // x = 1 + 1e-10: platform = 1.414213620873394e-5 (34k ULPs wrong)
    let gold_1p1e10: f64 = f64::from_bits(0x3EEDA880667F3B17); // 1.41421362086758617e-5
    // x = 1 + 1e-8: platform = 1.414213556896155e-4 (3717 ULPs wrong)
    let gold_1p1e8: f64 = f64::from_bits(0x3F22895031FE4DF9); // 1.41421355689716287e-4

    let x_1p1e10 = 1.0_f64 + 1e-10;
    let x_1p1e8  = 1.0_f64 + 1e-8;
    assert!(
        ulps_between(acosh_strict(x_1p1e10), gold_1p1e10) <= 4,
        "acosh(1+1e-10): {} ulps from mpmath gold (platform wrong by 34k ULPs)",
        ulps_between(acosh_strict(x_1p1e10), gold_1p1e10)
    );
    assert!(
        ulps_between(acosh_strict(x_1p1e8), gold_1p1e8) <= 4,
        "acosh(1+1e-8): {} ulps from mpmath gold",
        ulps_between(acosh_strict(x_1p1e8), gold_1p1e8)
    );

    // Also verify the medium near-1 range against mpmath gold.
    // MSVC acosh is wrong here too (499 ULPs at x=1+1e-6, 56 ULPs at x=1+1e-4).
    let gold_medium: [(f64, f64); 3] = [
        (1.0 + 1e-6,  f64::from_bits(0x3F572BA41F96166C)), // 1.41421344446382001e-3 (mpmath)
        (1.0 + 1e-4,  f64::from_bits(0x3F8CF67D7EC0D4E5)), // 1.41420177752515500e-2 (mpmath)
        (1.001,       f64::from_bits(0x3FA6E53ACBCDEBA3)), // 4.47176336083068503e-2 (mpmath)
    ];
    let mut worst = 0u64;
    let mut worst_x = 0.0_f64;
    for (x, expected) in gold_medium {
        let got = acosh_strict(x);
        let d = ulps_between(got, expected);
        if d > worst { worst = d; worst_x = x; }
    }
    assert!(
        worst <= 4,
        "acosh near-1 medium range: worst {worst} ulps at x={worst_x:.15e} (oracle: mpmath)"
    );
}

// ── atanh ─────────────────────────────────────────────────────────────────────

#[test] fn atanh_zero_is_zero() { assert_bits!(atanh_strict(0.0), 0.0_f64, "atanh(0)"); }
#[test] fn atanh_neg_zero_is_neg_zero() {
    let v = atanh_strict(-0.0_f64);
    assert!(v.is_sign_negative() && v == 0.0, "atanh(-0) must be -0");
}
#[test]
fn atanh_one_is_pos_inf() { assert_eq!(atanh_strict(1.0), f64::INFINITY, "atanh(1) must be +∞"); }
#[test]
fn atanh_neg_one_is_neg_inf() { assert_eq!(atanh_strict(-1.0), f64::NEG_INFINITY, "atanh(-1) must be -∞"); }
#[test]
fn atanh_outside_domain_is_nan() {
    assert!(atanh_strict(1.0 + f64::EPSILON).is_nan(), "atanh(1+ε) must be NaN");
    assert!(atanh_strict(-1.0 - f64::EPSILON).is_nan(), "atanh(-1-ε) must be NaN");
    assert!(atanh_strict(2.0).is_nan(), "atanh(2) must be NaN");
    assert!(atanh_strict(f64::INFINITY).is_nan(), "atanh(+∞) must be NaN");
    assert!(atanh_strict(f64::NAN).is_nan(), "atanh(NaN) must be NaN");
}
#[test]
fn atanh_is_odd() {
    for x in [0.5_f64, 0.9, 0.99, 0.999] {
        assert_eq!(atanh_strict(-x).to_bits(), (-atanh_strict(x)).to_bits(), "atanh(-{x}) != -atanh({x})");
    }
}

/// THE CRITICAL ATANH CANCELLATION TEST.
/// atanh(x) = log((1+x)/(1-x))/2. For x near ±1, (1-x) → 0.
/// Correct: atanh(x) = log1p(2x/(1-x))/2.
/// A wrong implementation returns a result that's "close" but has hundreds
/// of ulps error while still looking like a reasonable number.
#[test]
fn atanh_cancellation_near_one() {
    let inputs = [
        0.9_f64, 0.99, 0.999, 0.9999, 0.99999,
        0.999999, 1.0 - 1e-10, 1.0 - 1e-12, 1.0 - 1e-14,
        1.0 - f64::EPSILON, 1.0 - f64::EPSILON / 2.0,
    ];
    let mut worst = 0u64;
    let mut worst_x = 0.0_f64;
    for x in inputs {
        if x >= 1.0 { continue; }
        let got = atanh_strict(x);
        let expected = x.atanh();
        let d = ulps_between(got, expected);
        if d > worst { worst = d; worst_x = x; }
    }
    assert!(
        worst <= 4,
        "atanh cancellation near 1: worst {worst} ulps at x={worst_x:.15e}\n  \
         This is a silent failure: atanh({worst_x:.15}) = {} but should be {}",
        atanh_strict(worst_x), worst_x.atanh()
    );
}

// ── roundtrips ────────────────────────────────────────────────────────────────

#[test]
fn sinh_asinh_roundtrip() {
    for x in [-10.0_f64, -1.0, -0.1, 0.0, 0.1, 1.0, 10.0] {
        let rt = sinh_strict(asinh_strict(x));
        let d = ulps_between(rt, x);
        assert!(d <= 8, "sinh(asinh({x})): {d} ulps, got {rt:e}");
    }
}

#[test]
fn cosh_acosh_roundtrip() {
    for x in [1.0_f64, 1.5, 2.0, 5.0, 100.0] {
        let rt = cosh_strict(acosh_strict(x));
        let d = ulps_between(rt, x);
        assert!(d <= 8, "cosh(acosh({x})): {d} ulps, got {rt:e}");
    }
}

#[test]
fn tanh_atanh_roundtrip() {
    for x in [-0.9_f64, -0.5, 0.0, 0.5, 0.9, 0.99] {
        let rt = tanh_strict(atanh_strict(x));
        let d = ulps_between(rt, x);
        assert!(d <= 8, "tanh(atanh({x})): {d} ulps, got {rt:e}");
    }
}
