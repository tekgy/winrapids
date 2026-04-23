//! Adversarial tests for sinpi/cospi/tanpi + sincospi (TRIG-17, pi-scaled family).
//! Compiles when pathmaker ships the pi_scaled and sincos_pi modules.
//!
//! Key adversarial patterns:
//! - EXACT CONTRACT: sinpi(0.5) MUST be exactly 1.0, not 0.9999...
//!   This is the entire point of sinpi. If this fails, ship nothing.
//! - tanpi(0.5) MUST be ±∞ (exact pole, unlike tan(f64::FRAC_PI_2)).
//! - sinpi/cospi at integers must be exact.
//! - sincospi must match separate calls bit-for-bit.

use tambear::recipes::libm::pi_scaled::{sinpi_strict, cospi_strict, tanpi_strict};
use tambear::recipes::libm::sincos_pi::sincospi_strict;
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

// ── sinpi — THE CONTRACT ──────────────────────────────────────────────────────

/// THE SINPI CONTRACT: sinpi(0.5) MUST return EXACTLY 1.0.
/// This is the fundamental guarantee of the pi-scaled family.
/// sin(π/2) via float multiplication can be off by 1-2 ulps. sinpi(0.5)
/// must be exact. If this test fails, the implementation is fundamentally broken.
#[test]
fn sinpi_half_is_exactly_one() {
    assert_bits!(sinpi_strict(0.5), 1.0_f64, "sinpi(0.5) must be EXACTLY 1.0 — this is the contract");
}

#[test]
fn sinpi_neg_half_is_exactly_neg_one() {
    assert_bits!(sinpi_strict(-0.5), -1.0_f64, "sinpi(-0.5) must be EXACTLY -1.0");
}

#[test]
fn sinpi_three_halves_is_exactly_neg_one() {
    // sinpi(1.5) = sin(3π/2) = -1 exactly.
    assert_bits!(sinpi_strict(1.5), -1.0_f64, "sinpi(1.5) must be EXACTLY -1.0");
}

#[test]
fn sinpi_neg_three_halves_is_exactly_one() {
    assert_bits!(sinpi_strict(-1.5), 1.0_f64, "sinpi(-1.5) must be EXACTLY 1.0");
}

#[test]
fn sinpi_integer_inputs_are_exactly_zero() {
    // sinpi(n) = sin(nπ) = 0 for all integer n. Must be EXACTLY 0 (or ±0).
    for n in [-10_i32, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 10, 100] {
        let v = sinpi_strict(n as f64);
        assert_eq!(v, 0.0, "sinpi({n}) must be exactly 0 (got {v})");
    }
}

#[test]
fn sinpi_nan_is_nan() { assert!(sinpi_strict(f64::NAN).is_nan(), "sinpi(NaN)"); }
#[test]
fn sinpi_inf_is_nan() {
    assert!(sinpi_strict(f64::INFINITY).is_nan(), "sinpi(+∞)");
    assert!(sinpi_strict(f64::NEG_INFINITY).is_nan(), "sinpi(-∞)");
}
#[test]
fn sinpi_zero_is_zero() { assert_bits!(sinpi_strict(0.0), 0.0_f64, "sinpi(0)"); }
#[test]
fn sinpi_neg_zero_is_neg_zero() {
    let v = sinpi_strict(-0.0_f64);
    assert!(v.is_sign_negative() && v == 0.0, "sinpi(-0) must be -0");
}
#[test]
fn sinpi_is_odd() {
    for x in [0.25_f64, 0.5, 0.75, 1.25, 2.3] {
        assert_eq!(sinpi_strict(-x).to_bits(), (-sinpi_strict(x)).to_bits(), "sinpi(-{x}) != -sinpi({x})");
    }
}

#[test]
fn sinpi_accuracy_general_inputs() {
    // For non-half-integer, non-integer inputs, sinpi should agree with
    // sin(π*x) within a tight budget. The VALUE of sinpi is at half-integers;
    // the accuracy for general x should be at least as good as sin.
    let pi = std::f64::consts::PI;
    for x in [0.1_f64, 0.25, 0.3, 0.7, 1.1, 2.3] {
        let got = sinpi_strict(x);
        let expected = (pi * x).sin();
        let d = ulps_between(got, expected);
        assert!(d <= 4, "sinpi({x}) vs sin(π·{x}): {d} ulps");
    }
}

// ── cospi ─────────────────────────────────────────────────────────────────────

#[test]
fn cospi_half_is_exactly_zero() {
    // cospi(0.5) = cos(π/2) = 0 exactly. cos(f64::FRAC_PI_2) is not exactly 0.
    // If cospi doesn't use exact reduction, it returns a tiny nonzero value.
    assert_eq!(cospi_strict(0.5), 0.0, "cospi(0.5) must be EXACTLY 0");
}

#[test]
fn cospi_zero_is_exactly_one() {
    assert_bits!(cospi_strict(0.0), 1.0_f64, "cospi(0) must be EXACTLY 1.0");
}

#[test]
fn cospi_one_is_exactly_neg_one() {
    assert_bits!(cospi_strict(1.0), -1.0_f64, "cospi(1.0) must be EXACTLY -1.0");
}

#[test]
fn cospi_integer_inputs_are_exactly_pm_one() {
    for n in [-4_i32, -3, -2, -1, 0, 1, 2, 3, 4, 10] {
        let v = cospi_strict(n as f64);
        let expected = if n.abs() % 2 == 0 { 1.0_f64 } else { -1.0_f64 };
        assert_eq!(v, expected, "cospi({n}) must be exactly {expected}, got {v}");
    }
}

#[test]
fn cospi_nan_is_nan() { assert!(cospi_strict(f64::NAN).is_nan(), "cospi(NaN)"); }
#[test]
fn cospi_inf_is_nan() {
    assert!(cospi_strict(f64::INFINITY).is_nan(), "cospi(+∞)");
    assert!(cospi_strict(f64::NEG_INFINITY).is_nan(), "cospi(-∞)");
}
#[test]
fn cospi_is_even() {
    for x in [0.25_f64, 0.5, 0.75, 1.25, 2.3] {
        assert_eq!(cospi_strict(-x).to_bits(), cospi_strict(x).to_bits(), "cospi(-{x}) != cospi({x})");
    }
}

// ── tanpi ─────────────────────────────────────────────────────────────────────

#[test]
fn tanpi_zero_is_zero() { assert_bits!(tanpi_strict(0.0), 0.0_f64, "tanpi(0)"); }
#[test]
fn tanpi_one_is_zero() { assert_eq!(tanpi_strict(1.0), 0.0, "tanpi(1) must be 0"); }
#[test]
fn tanpi_quarter_is_exactly_one() {
    // tanpi(0.25) = tan(π/4) = 1 exactly.
    assert_bits!(tanpi_strict(0.25), 1.0_f64, "tanpi(0.25) must be EXACTLY 1.0");
}
#[test]
fn tanpi_neg_quarter_is_exactly_neg_one() {
    assert_bits!(tanpi_strict(-0.25), -1.0_f64, "tanpi(-0.25) must be EXACTLY -1.0");
}

/// tan has period π, so tanpi(n+0.25)=+1 for ALL integers n (not just n=0).
/// Regression anchor: old code XOR'd with (n&1)!=0 ("integer_flips"), flipping
/// sign for odd n. tanpi(1.25) returned -1 instead of +1.
#[test]
fn tanpi_quarter_integer_all_n_exactly_one() {
    for n in [-4_i32, -3, -2, -1, 0, 1, 2, 3, 4] {
        let x25 = n as f64 + 0.25;
        let x75 = n as f64 + 0.75;
        assert_eq!(tanpi_strict(x25), 1.0, "tanpi({x25}) must be +1 (integer_flips regression)");
        assert_eq!(tanpi_strict(x75), -1.0, "tanpi({x75}) must be -1 (integer_flips regression)");
    }
}

/// THE TANPI POLE CONTRACT: tanpi(0.5) MUST be ±∞.
/// Unlike tan(π/2) where the f64 π/2 is slightly off and returns a large finite,
/// tanpi(0.5) has an EXACT argument of 0.5 — the pole is exact. Must be infinite.
#[test]
fn tanpi_half_is_infinite() {
    let v = tanpi_strict(0.5);
    assert!(
        v.is_infinite(),
        "tanpi(0.5) must be ±∞ (exact pole — not a large finite!), got {v:e}"
    );
}

#[test]
fn tanpi_neg_half_is_infinite() {
    let v = tanpi_strict(-0.5);
    assert!(v.is_infinite(), "tanpi(-0.5) must be ±∞ (exact pole), got {v:e}");
}

/// tanpi at all half-integer inputs must be infinite (all exact poles).
#[test]
fn tanpi_all_half_integer_poles_are_infinite() {
    for n in [-5_i32, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5] {
        let x = n as f64 + 0.5;
        let v = tanpi_strict(x);
        assert!(v.is_infinite(), "tanpi({x}) = {v:e}, should be infinite (exact pole at n+0.5)");
    }
}

#[test]
fn tanpi_nan_is_nan() { assert!(tanpi_strict(f64::NAN).is_nan(), "tanpi(NaN)"); }
#[test]
fn tanpi_inf_is_nan() { assert!(tanpi_strict(f64::INFINITY).is_nan(), "tanpi(+∞)"); }
#[test]
fn tanpi_is_odd() {
    for x in [0.1_f64, 0.25, 0.3, 0.4] {
        assert_eq!(tanpi_strict(-x).to_bits(), (-tanpi_strict(x)).to_bits(), "tanpi(-{x}) != -tanpi({x})");
    }
}

// ── sincospi — fused pair ─────────────────────────────────────────────────────

#[test]
fn sincospi_zero() {
    let (s, c): (f64, f64) = sincospi_strict(0.0);
    assert_bits!(s, 0.0_f64, "sincospi(0).sin must be +0");
    assert_bits!(c, 1.0_f64, "sincospi(0).cos must be 1");
}

#[test]
fn sincospi_half() {
    let (s, c): (f64, f64) = sincospi_strict(0.5);
    assert_bits!(s, 1.0_f64, "sincospi(0.5).sin must be EXACTLY 1.0");
    assert_eq!(c, 0.0, "sincospi(0.5).cos must be EXACTLY 0.0");
}

#[test]
fn sincospi_one() {
    let (s, c): (f64, f64) = sincospi_strict(1.0);
    assert_eq!(s, 0.0, "sincospi(1.0).sin must be exactly 0");
    assert_bits!(c, -1.0_f64, "sincospi(1.0).cos must be exactly -1.0");
}

/// sincospi must give bit-identical results to separate sinpi/cospi calls.
/// This is the same contract as sincos vs sin+cos — fused computation must
/// not sacrifice a bit for shared intermediate efficiency.
#[test]
fn sincospi_matches_separate_calls_bit_exact() {
    for x in [0.0_f64, 0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.3, -0.5, -1.0] {
        let (s, c): (f64, f64) = sincospi_strict(x);
        let s_sep = sinpi_strict(x);
        let c_sep = cospi_strict(x);
        assert_eq!(s.to_bits(), s_sep.to_bits(), "sincospi({x}).sin mismatch: {s:e} vs {s_sep:e}");
        assert_eq!(c.to_bits(), c_sep.to_bits(), "sincospi({x}).cos mismatch: {c:e} vs {c_sep:e}");
    }
}

#[test]
fn sincospi_nan_produces_nan_pair() {
    let (s, c): (f64, f64) = sincospi_strict(f64::NAN);
    assert!(s.is_nan(), "sincospi(NaN).sin must be NaN");
    assert!(c.is_nan(), "sincospi(NaN).cos must be NaN");
}
