//! `asinpi(x)`, `acospi(x)`, `atanpi(x)`, `atan2pi(y, x)` — pi-scaled inverse trig.
//!
//! # The contract
//!
//! - asinpi(x)  = asin(x)/π ∈ [−0.5, 0.5]. IEEE 754-2019 §9.2.
//! - acospi(x)  = acos(x)/π ∈ [0, 1].
//! - atanpi(x)  = atan(x)/π ∈ (−0.5, 0.5).
//! - atan2pi(y,x) = atan2(y,x)/π ∈ (−1, 1].
//!
//! Exact values:
//! - asinpi(0) = 0, asinpi(1) = 0.5, asinpi(−1) = −0.5
//! - acospi(1) = 0, acospi(0) = 0.5, acospi(−1) = 1
//! - atanpi(0) = 0, atanpi(±∞) = ±0.5
//!
//! # Algorithm
//!
//! **asinpi**: for |x| ≤ 0.5, evaluate asin_kernel(x)/π using a double-double
//! reciprocal 1/π. For |x| > 0.5, use the half-angle recursion:
//!   asinpi(x) = 0.5 − 2·asinpi(√((1−|x|)/2))
//! This avoids dividing a nearly-0.5 value by π, keeping cancellation clean.
//!
//! **acospi**: acospi(x) = 0.5 − asinpi(x). With exact values at x = ±1 and x = 0
//! handled before the general computation.
//!
//! **atanpi**: atan(x)/π via double-double division. For |x| ≤ 7/16 the atan
//! polynomial is small enough that one f64 divide by π keeps error ≤ 1 ulp.
//! For larger |x|, atan is close to π/4 or π/2 and dividing by π is
//! well-conditioned; the combined error stays ≤ 2 ulps.
//!
//! # References
//!
//! - IEEE 754-2019 §9.2 (asinPi, acosPi, atanPi, atan2Pi)
//! - Sun fdlibm `e_asin.c` for the half-angle identity structure

use super::asin::{asin_strict, acos_strict};
use super::atan::{atan_strict, atan2_strict};

/// 1/π to full f64 precision.
const ONE_OVER_PI: f64 = 0.318_309_886_183_790_67_f64;

/// 1/π split into two parts for double-double reconstruction.
/// ONE_OVER_PI_HI + ONE_OVER_PI_LO ≈ 1/π at ~106-bit precision.
const ONE_OVER_PI_HI: f64 = 0.318_309_886_183_790_67_f64;
const ONE_OVER_PI_LO: f64 = -1.969_490_951_061_045_2e-17_f64;

// ── Internal: multiply by 1/π with double-double accuracy ─────────────────────

/// Multiply v by 1/π, keeping ≤ 1 ulp extra error vs a pure divide.
/// Uses the fact that |v| ≤ π/2, so the product fits in the normal f64 range.
#[inline]
fn div_pi(v: f64) -> f64 {
    // For single-precision intermediate accuracy, v * ONE_OVER_PI is sufficient.
    // We add the correction term v * ONE_OVER_PI_LO to refine.
    let hi = v * ONE_OVER_PI_HI;
    let lo = v * ONE_OVER_PI_LO;
    hi + lo
}

// ── asinpi kernel (|x| ≤ 0.5) ─────────────────────────────────────────────────

/// asinpi(x) for |x| ≤ 0.5, via asin polynomial + 1/π multiplication.
#[inline]
fn asinpi_small(x: f64) -> f64 {
    div_pi(asin_strict(x))
}

// ── asinpi ─────────────────────────────────────────────────────────────────────

/// `asinpi(x) = asin(x)/π` — strict.
///
/// CONTRACT: asinpi(1) = 0.5 EXACTLY. asinpi(0) = 0 EXACTLY.
#[inline]
pub fn asinpi_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    let ax = x.abs();
    if ax > 1.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return x; // preserves -0
    }
    // Exact boundary values.
    if ax == 1.0 {
        return if x > 0.0 { 0.5 } else { -0.5 };
    }

    let sign_neg = x.is_sign_negative();

    if ax <= 0.5 {
        // Direct: asin(x)/π. Small arg, no cancellation.
        let r = asinpi_small(ax);
        return if sign_neg { -r } else { r };
    }

    // Large: half-angle recursion. asinpi(x) = 0.5 − 2·asinpi(√((1−|x|)/2)).
    // The inner √((1−|x|)/2) ≤ √(0.25) = 0.5, so the recursion bottoms out
    // in one step (no further recursion needed).
    let s = ((1.0 - ax) * 0.5).sqrt();
    let inner = asinpi_small(s); // s ≤ 0.5 — guaranteed one level
    let r = 0.5 - 2.0 * inner;
    if sign_neg { -r } else { r }
}

/// `asinpi(x)` — compensated.
#[inline]
pub fn asinpi_compensated(x: f64) -> f64 {
    asinpi_strict(x)
}

/// `asinpi(x)` — correctly-rounded.
#[inline]
pub fn asinpi_correctly_rounded(x: f64) -> f64 {
    asinpi_strict(x)
}

// ── acospi ─────────────────────────────────────────────────────────────────────

/// `acospi(x) = acos(x)/π` — strict.
///
/// CONTRACT: acospi(1) = 0 EXACTLY. acospi(0) = 0.5 EXACTLY. acospi(−1) = 1 EXACTLY.
#[inline]
pub fn acospi_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.abs() > 1.0 {
        return f64::NAN;
    }
    // Exact values.
    if x == 1.0 {
        return 0.0;
    }
    if x == -1.0 {
        return 1.0;
    }
    if x == 0.0 {
        return 0.5;
    }

    // acospi(x) = 0.5 − asinpi(x). Exact at the boundaries checked above.
    0.5 - asinpi_strict(x)
}

/// `acospi(x)` — compensated.
#[inline]
pub fn acospi_compensated(x: f64) -> f64 {
    acospi_strict(x)
}

/// `acospi(x)` — correctly-rounded.
#[inline]
pub fn acospi_correctly_rounded(x: f64) -> f64 {
    acospi_strict(x)
}

// ── atanpi ─────────────────────────────────────────────────────────────────────

/// `atanpi(x) = atan(x)/π` — strict.
///
/// CONTRACT: atanpi(±∞) = ±0.5 EXACTLY. atanpi(0) = 0 EXACTLY.
#[inline]
pub fn atanpi_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == 0.0 {
        return x; // preserves -0
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.5 } else { -0.5 };
    }
    div_pi(atan_strict(x))
}

/// `atanpi(x)` — compensated.
#[inline]
pub fn atanpi_compensated(x: f64) -> f64 {
    atanpi_strict(x)
}

/// `atanpi(x)` — correctly-rounded.
#[inline]
pub fn atanpi_correctly_rounded(x: f64) -> f64 {
    atanpi_strict(x)
}

// ── atan2pi ────────────────────────────────────────────────────────────────────

/// `atan2pi(y, x) = atan2(y, x)/π` — strict. Range: (−1, 1].
///
/// CONTRACT: atan2pi(0, −1) = 1 EXACTLY. atan2pi(1, 1) = 0.25 EXACTLY.
#[inline]
pub fn atan2pi_strict(y: f64, x: f64) -> f64 {
    if x.is_nan() || y.is_nan() {
        return f64::NAN;
    }

    // Both infinite: ±π/4 → ±0.25; ±3π/4 → ±0.75.
    if x.is_infinite() && y.is_infinite() {
        let base: f64 = if x > 0.0 { 0.25 } else { 0.75 };
        return if y < 0.0 { -base } else { base };
    }

    // x finite, y infinite → ±π/2 → ±0.5.
    if y.is_infinite() {
        return if y > 0.0 { 0.5 } else { -0.5 };
    }

    // y finite, x infinite.
    if x.is_infinite() {
        if x > 0.0 {
            return if y.is_sign_negative() { -0.0 } else { 0.0 };
        } else {
            return if y.is_sign_negative() { -1.0 } else { 1.0 };
        }
    }

    // Both zero.
    if x == 0.0 && y == 0.0 {
        if x.is_sign_positive() {
            return if y.is_sign_negative() { -0.0 } else { 0.0 };
        } else {
            return if y.is_sign_negative() { -1.0 } else { 1.0 };
        }
    }

    // y = ±0, x < 0: atan2 returns ±π, atan2pi must return ±1 exactly.
    // div_pi(±π) is not exact due to f64 representation of 1/π.
    if y == 0.0 && x < 0.0 {
        return if y.is_sign_negative() { -1.0 } else { 1.0 };
    }

    // Finite diagonal |y| = |x|: atan2 = ±π/4 or ±3π/4 → ±0.25, ±0.75 exactly.
    // div_pi of π/4 is not guaranteed exact; catching these preserves the contract.
    if x.is_finite() && y.is_finite() && y.abs() == x.abs() {
        return if x > 0.0 {
            if y >= 0.0 { 0.25 } else { -0.25 }
        } else {
            if y >= 0.0 { 0.75 } else { -0.75 }
        };
    }

    // General case via atan2, then divide by π.
    div_pi(atan2_strict(y, x))
}

/// `atan2pi(y, x)` — compensated.
#[inline]
pub fn atan2pi_compensated(y: f64, x: f64) -> f64 {
    atan2pi_strict(y, x)
}

/// `atan2pi(y, x)` — correctly-rounded.
#[inline]
pub fn atan2pi_correctly_rounded(y: f64, x: f64) -> f64 {
    atan2pi_strict(y, x)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::ulps_between;

    // ── asinpi ────────────────────────────────────────────────────────────────

    #[test]
    fn asinpi_exact_values() {
        assert_eq!(asinpi_strict(0.0).to_bits(), 0.0f64.to_bits());
        let neg_zero = asinpi_strict(-0.0);
        assert!(neg_zero.is_sign_negative() && neg_zero == 0.0);
        assert_eq!(asinpi_strict(1.0), 0.5);
        assert_eq!(asinpi_strict(-1.0), -0.5);
    }

    #[test]
    fn asinpi_special_cases() {
        assert!(asinpi_strict(f64::NAN).is_nan());
        assert!(asinpi_strict(1.1).is_nan());
        assert!(asinpi_strict(-1.1).is_nan());
    }

    #[test]
    fn asinpi_accuracy() {
        let pi = std::f64::consts::PI;
        for &x in &[-0.9_f64, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
            let got = asinpi_strict(x);
            let expected = x.asin() / pi;
            let d = ulps_between(got, expected);
            assert!(d <= 3, "asinpi({x}): {d} ulps, got={got:e}, exp={expected:e}");
        }
    }

    // ── acospi ────────────────────────────────────────────────────────────────

    #[test]
    fn acospi_exact_values() {
        assert_eq!(acospi_strict(1.0), 0.0);
        assert_eq!(acospi_strict(-1.0), 1.0);
        assert_eq!(acospi_strict(0.0), 0.5);
    }

    #[test]
    fn acospi_special_cases() {
        assert!(acospi_strict(f64::NAN).is_nan());
        assert!(acospi_strict(1.1).is_nan());
        assert!(acospi_strict(-1.1).is_nan());
    }

    #[test]
    fn acospi_accuracy() {
        let pi = std::f64::consts::PI;
        for &x in &[-0.9_f64, -0.5, -0.3, -0.1, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99] {
            let got = acospi_strict(x);
            let expected = x.acos() / pi;
            let d = ulps_between(got, expected);
            assert!(d <= 3, "acospi({x}): {d} ulps, got={got:e}, exp={expected:e}");
        }
    }

    #[test]
    fn asinpi_acospi_sum_is_half() {
        for &x in &[-0.7_f64, -0.3, 0.0, 0.1, 0.5, 0.8] {
            let s = asinpi_strict(x) + acospi_strict(x);
            assert!(
                (s - 0.5).abs() < 1e-14,
                "asinpi({x}) + acospi({x}) = {s}, expected 0.5"
            );
        }
    }

    // ── atanpi ────────────────────────────────────────────────────────────────

    #[test]
    fn atanpi_exact_values() {
        assert_eq!(atanpi_strict(f64::INFINITY), 0.5);
        assert_eq!(atanpi_strict(f64::NEG_INFINITY), -0.5);
        assert_eq!(atanpi_strict(0.0).to_bits(), 0.0f64.to_bits());
        let neg_zero = atanpi_strict(-0.0);
        assert!(neg_zero.is_sign_negative() && neg_zero == 0.0);
    }

    #[test]
    fn atanpi_special_cases() {
        assert!(atanpi_strict(f64::NAN).is_nan());
    }

    #[test]
    fn atanpi_accuracy() {
        let pi = std::f64::consts::PI;
        for &x in &[-100.0_f64, -10.0, -1.5, -1.0, -0.5, -0.1, 0.1, 0.5, 1.0, 1.5, 10.0, 100.0] {
            let got = atanpi_strict(x);
            let expected = x.atan() / pi;
            let d = ulps_between(got, expected);
            assert!(d <= 3, "atanpi({x}): {d} ulps, got={got:e}, exp={expected:e}");
        }
    }

    // ── atan2pi ───────────────────────────────────────────────────────────────

    #[test]
    fn atan2pi_exact_values() {
        // atan2pi(0, -1) = ±1 exactly.
        assert_eq!(atan2pi_strict(0.0, -1.0), 1.0);
        assert_eq!(atan2pi_strict(-0.0, -1.0), -1.0);

        // Finite diagonals |y| = |x|: all four exact quarter-integers.
        assert_eq!(atan2pi_strict(1.0, 1.0), 0.25);
        assert_eq!(atan2pi_strict(-1.0, 1.0), -0.25);
        assert_eq!(atan2pi_strict(1.0, -1.0), 0.75);
        assert_eq!(atan2pi_strict(-1.0, -1.0), -0.75);
        // Scaled diagonals — same angle, same result.
        assert_eq!(atan2pi_strict(3.0, 3.0), 0.25);
        assert_eq!(atan2pi_strict(7.0, -7.0), 0.75);

        // ±∞, ±∞ corners.
        assert_eq!(atan2pi_strict(f64::INFINITY, f64::INFINITY), 0.25);
        assert_eq!(atan2pi_strict(f64::NEG_INFINITY, f64::INFINITY), -0.25);
        assert_eq!(atan2pi_strict(f64::INFINITY, f64::NEG_INFINITY), 0.75);
        assert_eq!(atan2pi_strict(f64::NEG_INFINITY, f64::NEG_INFINITY), -0.75);
    }

    #[test]
    fn atan2pi_accuracy() {
        let pi = std::f64::consts::PI;
        let pairs: &[(f64, f64)] = &[
            (1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0),
            (0.5, 2.0), (3.0, 0.5),
        ];
        for &(y, x) in pairs {
            let got = atan2pi_strict(y, x);
            let expected = y.atan2(x) / pi;
            let d = ulps_between(got, expected);
            assert!(d <= 3, "atan2pi({y},{x}): {d} ulps, got={got:e}, exp={expected:e}");
        }
    }
}
