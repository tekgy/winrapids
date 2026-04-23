//! Adversarial tests for versin / haversin / gudermannian (TRIG-17, rare trig).
//! Compiles when pathmaker ships the rare_trig module.
//!
//! Key adversarial patterns:
//! - versin(x) = 1 - cos(x): catastrophic cancellation for small x.
//!   For x=1e-8, cos(x) = 1.0 in f64, so naive 1-cos(x) returns 0.0.
//!   Correct: versin(x) = 2·sin²(x/2).
//! - haversin(x) = versin(x)/2 = sin²(x/2).
//! - gudermannian(x) = atan(sinh(x)): must match the identity.

use tambear::recipes::libm::rare_trig::{versin_strict, haversin_strict, gudermannian_strict};
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

// ── versin ────────────────────────────────────────────────────────────────────

#[test] fn versin_zero_is_zero() { assert_bits!(versin_strict(0.0), 0.0_f64, "versin(0)"); }
#[test] fn versin_nan_is_nan() { assert!(versin_strict(f64::NAN).is_nan(), "versin(NaN)"); }
#[test] fn versin_pi_is_two() { assert_ulps!(versin_strict(std::f64::consts::PI), 2.0, 1, "versin(π)"); }

#[test]
fn versin_is_nonnegative() {
    // versin(x) = 1 - cos(x) ∈ [0, 2] for all real x.
    for x in [-100.0_f64, -1.0, 0.0, 1.0, 100.0] {
        let v = versin_strict(x);
        assert!(v >= 0.0 && v <= 2.0, "versin({x}) = {v}, must be in [0, 2]");
    }
}

/// THE CRITICAL VERSIN CANCELLATION TEST.
///
/// For tiny x, versin(x) = 1 - cos(x) ≈ x²/2.
/// A naive implementation: 1.0 - cos(1e-8) = 1.0 - 1.0 = 0.0 in f64.
/// That's 100% wrong — the answer should be ~5e-17.
/// Correct: versin(x) = 2·sin²(x/2).
#[test]
fn versin_tiny_x_cancellation() {
    let tiny = 1e-8_f64;
    let got = versin_strict(tiny);
    let expected = tiny * tiny / 2.0; // x²/2 to high accuracy for this range
    let d = ulps_between(got, expected);
    assert!(
        d <= 4,
        "versin({tiny:e}) cancellation: got {got:e}, expected ≈ {expected:e} ({d} ulps)\n  \
         A naive 1-cos(x) returns 0 here — this test catches that failure"
    );
}

#[test]
fn versin_small_x_cancellation_sweep() {
    // Sweep across the cancellation zone.
    // Our formula: versin(x) = 2*sin²(x/2). This is CORRECT for all x.
    // For comparison: 1-cos(x) suffers cancellation for small x (Windows MSVC
    // cos has 63 ULP error at x=0.1 due to the 1-cos subtraction, verified vs mpmath).
    // We do NOT use `1-cos(x)` as oracle for x > 1e-8 — it's the wrong oracle.
    for k in 1..=10_u32 {
        let x = (10.0_f64).powi(-(k as i32));
        let got = versin_strict(x);
        let expected = x * x / 2.0 * (1.0 - x * x / 12.0); // Taylor to 4th order

        // For very tiny x (≤ 1e-4), the Taylor estimate is ground truth.
        if x <= 1e-4 {
            let rel_err = (got - expected).abs() / expected.abs();
            assert!(
                rel_err < 1e-10,
                "versin({x:e}) Taylor relative error: {rel_err:e} (too large, cancellation bug)"
            );
        }

        // For moderate x (1e-4 < x ≤ 0.1), verify via identity versin(x) = 2*sin²(x/2).
        // Compare against sin² with the identity as oracle.
        if x > 1e-4 && x <= 0.1 {
            use tambear::recipes::libm::sin::sin_strict;
            let s = sin_strict(x * 0.5);
            let via_identity = 2.0 * s * s;
            let d = ulps_between(got, via_identity);
            assert!(d == 0, "versin({x:e}) != 2*sin²({x:e}/2): {d} ulps (implementation bug)");
        }
    }
}

// ── haversin ──────────────────────────────────────────────────────────────────

#[test] fn haversin_zero_is_zero() { assert_bits!(haversin_strict(0.0), 0.0_f64, "haversin(0)"); }

#[test]
fn haversin_pi_is_one() {
    assert_ulps!(haversin_strict(std::f64::consts::PI), 1.0, 1, "haversin(π)");
}

#[test]
fn haversin_equals_half_versin() {
    for x in [0.5_f64, 1.0, std::f64::consts::PI / 3.0, 2.0] {
        let hv = haversin_strict(x);
        let v = versin_strict(x);
        let d = ulps_between(hv * 2.0, v);
        assert!(d <= 4, "2·haversin({x}) != versin({x}): {d} ulps");
    }
}

#[test]
fn haversin_equals_sin_squared() {
    // haversin(x) = sin²(x/2). This identity is the whole point.
    use tambear::recipes::libm::sin::sin_strict;
    for x in [0.5_f64, 1.0, 2.0, std::f64::consts::PI] {
        let hv = haversin_strict(x);
        let s = sin_strict(x / 2.0);
        let sq = s * s;
        let d = ulps_between(hv, sq);
        assert!(d <= 4, "haversin({x}) != sin²({x}/2): {d} ulps, {hv:e} vs {sq:e}");
    }
}

// ── gudermannian ──────────────────────────────────────────────────────────────

#[test] fn gudermannian_zero_is_zero() { assert_bits!(gudermannian_strict(0.0), 0.0_f64, "gd(0)"); }
#[test] fn gudermannian_nan_is_nan() { assert!(gudermannian_strict(f64::NAN).is_nan(), "gd(NaN)"); }

#[test]
fn gudermannian_pos_inf_is_pi_over_2() {
    assert_ulps!(gudermannian_strict(f64::INFINITY), std::f64::consts::FRAC_PI_2, 1, "gd(+∞)");
}

#[test]
fn gudermannian_neg_inf_is_neg_pi_over_2() {
    assert_ulps!(gudermannian_strict(f64::NEG_INFINITY), -std::f64::consts::FRAC_PI_2, 1, "gd(-∞)");
}

#[test]
fn gudermannian_is_odd() {
    for x in [0.5_f64, 1.0, 2.0, 5.0] {
        assert_eq!(
            gudermannian_strict(-x).to_bits(), (-gudermannian_strict(x)).to_bits(),
            "gd(-{x}) != -gd({x})"
        );
    }
}

#[test]
fn gudermannian_identity_atan_sinh() {
    // gd(x) = atan(sinh(x)). Both formulas should agree.
    use tambear::recipes::libm::hyperbolic::sinh_strict;
    use tambear::recipes::libm::atan::atan_strict;
    for x in [0.5_f64, 1.0, 2.0, 5.0, -1.0, -3.0] {
        let via_gd = gudermannian_strict(x);
        let via_identity = atan_strict(sinh_strict(x));
        let d = ulps_between(via_gd, via_identity);
        assert!(d <= 4, "gd({x}) vs atan(sinh({x})): {d} ulps");
    }
}

#[test]
fn gudermannian_bounded_by_pi_over_2() {
    // gd(x) ∈ (-π/2, π/2) for all finite x.
    let pio2 = std::f64::consts::FRAC_PI_2;
    for x in [-100.0_f64, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
        let v = gudermannian_strict(x);
        assert!(v.abs() < pio2, "gd({x}) = {v:e}, must be strictly inside (-π/2, π/2)");
    }
}
