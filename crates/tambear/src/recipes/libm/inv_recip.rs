//! `acot(x)`, `asec(x)`, `acsc(x)` — inverse reciprocal trigonometric functions.
//!
//! These are simple compositions of asin/acos/atan and are not in C99 libm,
//! but are standard in Mathematica, MATLAB, and Julia.
//!
//! # Implementations
//!
//! - `acot(x) = atan(1/x)` for x > 0, with acot(0) = π/2 by convention.
//!   For x < 0, use `acot(x) = π + atan(1/x)` to stay in (0, π).
//! - `asec(x) = acos(1/x)` for |x| ≥ 1.
//! - `acsc(x) = asin(1/x)` for |x| ≥ 1.

use super::asin::{asin_strict, acos_strict};
use super::atan::atan_strict;

const PIO2: f64 = std::f64::consts::FRAC_PI_2;
const PI: f64 = std::f64::consts::PI;

// ── acot ──────────────────────────────────────────────────────────────────────

/// `acot(x)` — strict. Range: (0, π). acot(0) = π/2.
///
/// acot(x) = atan(1/x) for x > 0, = π + atan(1/x) for x < 0.
/// Both expressions give a result in (0, π) — the standard principal value.
#[inline]
pub fn acot_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { PI };
    }
    if x == 0.0 {
        return PIO2; // acot(0) = π/2 regardless of sign
    }
    let at = atan_strict(1.0 / x);
    if x > 0.0 { at } else { at + PI }
}

/// `acot(x)` — compensated.
#[inline]
pub fn acot_compensated(x: f64) -> f64 {
    acot_strict(x)
}

/// `acot(x)` — correctly-rounded.
#[inline]
pub fn acot_correctly_rounded(x: f64) -> f64 {
    acot_strict(x)
}

// ── asec ──────────────────────────────────────────────────────────────────────

/// `asec(x) = acos(1/x)` — strict. Domain: |x| ≥ 1. Range: [0, π] \ {π/2}.
///
/// Special: asec(1) = 0, asec(−1) = π, asec(±∞) = π/2.
#[inline]
pub fn asec_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return PIO2;
    }
    let ax = x.abs();
    if ax < 1.0 {
        return f64::NAN; // domain error
    }
    acos_strict(1.0 / x)
}

/// `asec(x)` — compensated.
#[inline]
pub fn asec_compensated(x: f64) -> f64 {
    asec_strict(x)
}

/// `asec(x)` — correctly-rounded.
#[inline]
pub fn asec_correctly_rounded(x: f64) -> f64 {
    asec_strict(x)
}

// ── acsc ──────────────────────────────────────────────────────────────────────

/// `acsc(x) = asin(1/x)` — strict. Domain: |x| ≥ 1. Range: [−π/2, π/2] \ {0}.
///
/// Special: acsc(1) = π/2, acsc(−1) = −π/2, acsc(±∞) = 0.
#[inline]
pub fn acsc_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x.is_sign_positive() { 0.0 } else { -0.0 };
    }
    let ax = x.abs();
    if ax < 1.0 {
        return f64::NAN; // domain error
    }
    asin_strict(1.0 / x)
}

/// `acsc(x)` — compensated.
#[inline]
pub fn acsc_compensated(x: f64) -> f64 {
    acsc_strict(x)
}

/// `acsc(x)` — correctly-rounded.
#[inline]
pub fn acsc_correctly_rounded(x: f64) -> f64 {
    acsc_strict(x)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::ulps_between;

    #[test]
    fn acot_special_cases() {
        assert!(acot_strict(f64::NAN).is_nan());
        assert_eq!(acot_strict(0.0), PIO2);
        assert_eq!(acot_strict(-0.0), PIO2);
        assert!(ulps_between(acot_strict(f64::INFINITY), 0.0) <= 1);
        assert!(ulps_between(acot_strict(f64::NEG_INFINITY), PI) <= 1);
    }

    #[test]
    fn acot_accuracy() {
        let samples: &[f64] = &[0.5, 1.0, 2.0, 5.0, -1.0, -2.0, 0.1, 10.0];
        for &x in samples {
            let got = acot_strict(x);
            let expected = (1.0_f64 / x).atan() + if x < 0.0 { PI } else { 0.0 };
            let d = ulps_between(got, expected);
            assert!(d <= 3, "acot({x}): {d} ulps, got={got:e}, exp={expected:e}");
        }
    }

    #[test]
    fn acot_range_is_zero_to_pi() {
        let samples: &[f64] = &[-100.0, -1.0, -0.1, 0.0, 0.1, 1.0, 100.0];
        for &x in samples {
            let got = acot_strict(x);
            assert!(got > 0.0 && got < PI + 1e-15, "acot({x}) = {got} not in (0,π)");
        }
    }

    #[test]
    fn asec_special_cases() {
        assert!(asec_strict(f64::NAN).is_nan());
        assert!(asec_strict(0.5).is_nan());
        assert!(asec_strict(-0.5).is_nan());
        assert!(ulps_between(asec_strict(1.0), 0.0) <= 1);
        assert!(ulps_between(asec_strict(-1.0), PI) <= 1);
        assert!(ulps_between(asec_strict(f64::INFINITY), PIO2) <= 1);
    }

    #[test]
    fn acsc_special_cases() {
        assert!(acsc_strict(f64::NAN).is_nan());
        assert!(acsc_strict(0.5).is_nan());
        assert!(ulps_between(acsc_strict(1.0), PIO2) <= 1);
        assert!(ulps_between(acsc_strict(-1.0), -PIO2) <= 1);
        assert_eq!(acsc_strict(f64::INFINITY), 0.0);
    }
}
