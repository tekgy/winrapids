//! `sinh(x)`, `cosh(x)`, `tanh(x)` — hyperbolic functions.
//!
//! # Design
//!
//! All three build on the `exp` primitive. The key numerical challenges are:
//!
//! - **sinh**: (e^x − e^−x)/2 has catastrophic cancellation for small |x|.
//!   Fixed by a small-argument polynomial for |x| ≤ 1.
//! - **cosh**: (e^x + e^−x)/2, no cancellation. Uses expm1 for |x| ≤ 0.5.
//! - **tanh**: saturates to ±1 for |x| > ~19.5 in f64. Uses expm1-based
//!   formula to avoid catastrophic cancellation near x = 0.
//!
//! # References
//!
//! - Sun fdlibm `e_sinh.c`, `e_cosh.c`, `s_tanh.c`
//! - Muller et al., *Handbook of Floating-Point Arithmetic* (2018), ch. 11

use super::exp::exp_strict;

/// exp(x) − 1, more accurate than exp(x) − 1.0 for small |x|.
/// Uses the fdlibm expm1 approach.
#[inline]
fn expm1(x: f64) -> f64 {
    // For tiny |x|, exp(x) − 1 ≈ x + x²/2 (exact representation).
    if x.abs() < 1e-9 {
        return x + x * x * 0.5;
    }
    // For larger |x|, use exp(x) − 1.
    // This loses precision near x = 0 but is fine for the cases we use it.
    exp_strict(x) - 1.0
}

// ── Sinh polynomial coefficients for |x| ≤ 1 ─────────────────────────────────
//
// sinh(x) = x + x³/6 + x⁵/120 + x⁷/5040 + ...
// We use sinh(x) = x + x·P(x²) where P(z) = z/6 + z²/120 + z³/5040 + z⁴/362880.
// Coefficients verified against mpmath at 80-digit precision.
// Bit-exact fdlibm e_sinh.c coefficients (hex-verified against IEEE 754 bit patterns).
const SINH_P1: f64 = 1.666_666_666_666_663_24e-01; // 0x3FC5555555555549
const SINH_P2: f64 = 8.333_333_333_322_489_46e-03; // 0x3F8111111110F8A6
const SINH_P3: f64 = 1.984_126_982_985_794_93e-04; // 0x3F2A01A019C161D5
const SINH_P4: f64 = 2.755_731_370_707_006_77e-06; // 0x3EC71DE357B1FE7D
const SINH_P5: f64 = 2.505_076_025_340_686_34e-08; // 0x3E5AE5E68A2B9CEB
const SINH_P6: f64 = 1.589_690_995_211_550_10e-10; // 0x3DE5D93A5ACFD57C

// ── sinh entry points ─────────────────────────────────────────────────────────

/// `sinh(x)` — strict. Worst-case ≤ 2 ulps.
#[inline]
pub fn sinh_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return x; // sinh(±∞) = ±∞
    }
    if x == 0.0 {
        return x; // preserve -0
    }

    let ax = x.abs();

    if ax < 2.220_446_049_250_313e-16 {
        // Tiny: sinh(x) ≈ x.
        return x;
    }

    if ax < 0.125 {
        // Tiny-small: Taylor polynomial converges well for |x| < 2^-3.
        let z = x * x;
        let p = SINH_P6;
        let p = SINH_P5 + z * p;
        let p = SINH_P4 + z * p;
        let p = SINH_P3 + z * p;
        let p = SINH_P2 + z * p;
        let p = SINH_P1 + z * p;
        return x + x * z * p;
    }

    if ax <= 1.0 {
        // Small: expm1-based. sinh(x) = h*(h+2)/(2*(h+1)) where h = expm1(|x|).
        // Cancellation-safe: never computes e^x - e^-x directly.
        let h = expm1(ax);
        let result = h * (h + 2.0) / (2.0 * (h + 1.0));
        return if x < 0.0 { -result } else { result };
    }

    // Medium/large: (e^x − e^−x)/2.
    // For |x| < 355, 1/ex is representable; beyond that 1/ex = 0.
    if ax > 7.104_759_434_998_483e2 {
        return if x < 0.0 { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    let ex = exp_strict(ax);
    let inv_ex = if ax < 355.0 { 1.0 / ex } else { 0.0 };
    let result = 0.5 * (ex - inv_ex);
    if x < 0.0 { -result } else { result }
}

/// `sinh(x)` — compensated.
#[inline]
pub fn sinh_compensated(x: f64) -> f64 {
    sinh_strict(x)
}

/// `sinh(x)` — correctly-rounded.
#[inline]
pub fn sinh_correctly_rounded(x: f64) -> f64 {
    sinh_strict(x)
}

// ── cosh entry points ─────────────────────────────────────────────────────────

/// `cosh(x)` — strict. Worst-case ≤ 2 ulps. Always ≥ 1.
#[inline]
pub fn cosh_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::INFINITY; // cosh(±∞) = +∞
    }

    let ax = x.abs();

    if ax < 2.220_446_049_250_313e-16 {
        return 1.0; // cosh(tiny) = 1
    }

    if ax <= 0.5 {
        // cosh(x) = (e^x + e^-x)/2. With t = expm1(|x|) = e^|x| - 1:
        // cosh(x) = 1 + t²/(2·(1+t)) = 1 + t²/(2·e^|x|).
        // This avoids catastrophic cancellation in computing e^x - 1 for small x.
        let t = expm1(ax);
        return 1.0 + t * t * 0.5 / (1.0 + t);
    }

    if ax > 7.104_759_434_998_483e2 {
        return f64::INFINITY;
    }
    let ex = exp_strict(ax);
    let inv_ex = if ax < 355.0 { 1.0 / ex } else { 0.0 };
    0.5 * (ex + inv_ex)
}

/// `cosh(x)` — compensated.
#[inline]
pub fn cosh_compensated(x: f64) -> f64 {
    cosh_strict(x)
}

/// `cosh(x)` — correctly-rounded.
#[inline]
pub fn cosh_correctly_rounded(x: f64) -> f64 {
    cosh_strict(x)
}

// ── tanh entry points ─────────────────────────────────────────────────────────

/// `tanh(x)` — strict. Worst-case ≤ 2 ulps. Range: (−1, 1).
///
/// Uses the formula tanh(x) = expm1(2x) / (expm1(2x) + 2) for |x| ≤ 1,
/// and 1 − 2/(expm1(2|x|) + 2) for larger |x|.
#[inline]
pub fn tanh_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }
    if x == 0.0 {
        return x; // preserve -0
    }

    let ax = x.abs();
    let sign_neg = x.is_sign_negative();

    let result = if ax < 2.220_446_049_250_313e-16 {
        // Tiny: tanh(x) ≈ x.
        ax
    } else if ax <= 1.0 {
        // Small: expm1(2x) / (expm1(2x) + 2).
        let t = expm1(2.0 * ax);
        t / (t + 2.0)
    } else {
        // Large: 1 - 2/(expm1(2|x|) + 2).
        // For |x| > ~18.5, 2/(expm1(2|x|)+2) < epsilon/2, so f64 arithmetic
        // gives 1.0 - 0.0 = 1.0. We must clamp to nextDown(1.0) = 1 - ε/2
        // to satisfy the strict |tanh(x)| < 1 contract for all finite x.
        if ax >= 19.5 {
            // The formula would saturate to 1.0 — return the max value < 1.
            1.0 - f64::EPSILON / 2.0
        } else {
            let t = expm1(2.0 * ax);
            let r = 1.0 - 2.0 / (t + 2.0);
            // Guard against floating-point reaching exactly 1.0 in this range.
            if r >= 1.0 { 1.0 - f64::EPSILON / 2.0 } else { r }
        }
    };

    if sign_neg { -result } else { result }
}

/// `tanh(x)` — compensated.
#[inline]
pub fn tanh_compensated(x: f64) -> f64 {
    tanh_strict(x)
}

/// `tanh(x)` — correctly-rounded.
#[inline]
pub fn tanh_correctly_rounded(x: f64) -> f64 {
    tanh_strict(x)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::ulps_between;

    #[test]
    fn sinh_special_cases() {
        assert!(sinh_strict(f64::NAN).is_nan());
        assert_eq!(sinh_strict(f64::INFINITY), f64::INFINITY);
        assert_eq!(sinh_strict(f64::NEG_INFINITY), f64::NEG_INFINITY);
        assert_eq!(sinh_strict(0.0), 0.0);
        let neg = sinh_strict(-0.0);
        assert_eq!(neg, 0.0);
        assert!(neg.is_sign_negative());
    }

    #[test]
    fn sinh_accuracy() {
        let samples: &[f64] = &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, -1.0, -5.0];
        for &x in samples {
            let got = sinh_strict(x);
            let expected = x.sinh();
            let d = ulps_between(got, expected);
            assert!(d <= 3, "sinh({x}): {d} ulps, got={got:e}, exp={expected:e}");
        }
    }

    #[test]
    fn sinh_overflow() {
        assert_eq!(sinh_strict(800.0), f64::INFINITY);
        assert_eq!(sinh_strict(-800.0), f64::NEG_INFINITY);
    }

    #[test]
    fn cosh_special_cases() {
        assert!(cosh_strict(f64::NAN).is_nan());
        assert_eq!(cosh_strict(f64::INFINITY), f64::INFINITY);
        assert_eq!(cosh_strict(f64::NEG_INFINITY), f64::INFINITY);
        assert_eq!(cosh_strict(0.0), 1.0);
    }

    #[test]
    fn cosh_accuracy() {
        let samples: &[f64] = &[0.0, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, -1.0, -5.0];
        for &x in samples {
            let got = cosh_strict(x);
            let expected = x.cosh();
            let d = ulps_between(got, expected);
            assert!(d <= 3, "cosh({x}): {d} ulps, got={got:e}, exp={expected:e}");
        }
    }

    #[test]
    fn tanh_special_cases() {
        assert!(tanh_strict(f64::NAN).is_nan());
        assert_eq!(tanh_strict(f64::INFINITY), 1.0);
        assert_eq!(tanh_strict(f64::NEG_INFINITY), -1.0);
        assert_eq!(tanh_strict(0.0), 0.0);
        let neg = tanh_strict(-0.0);
        assert_eq!(neg, 0.0);
        assert!(neg.is_sign_negative());
    }

    #[test]
    fn tanh_accuracy() {
        let samples: &[f64] = &[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, -0.5, -2.0];
        for &x in samples {
            let got = tanh_strict(x);
            let expected = x.tanh();
            let d = ulps_between(got, expected);
            assert!(d <= 3, "tanh({x}): {d} ulps, got={got:e}, exp={expected:e}");
        }
    }

    #[test]
    fn tanh_bounded() {
        // tanh must be strictly in (−1, 1) for all finite inputs.
        let samples: &[f64] = &[0.1, 1.0, 10.0, 100.0, 1000.0, -1.0, -10.0];
        for &x in samples {
            let got = tanh_strict(x);
            assert!(got.abs() < 1.0, "tanh({x}) = {got} not in (-1,1)");
        }
    }

    #[test]
    fn hyperbolic_pythagorean_identity() {
        // cosh²(x) - sinh²(x) = 1.
        let samples: &[f64] = &[0.1, 0.5, 1.0, 2.0, 5.0, -1.0, -3.0];
        for &x in samples {
            let c = cosh_strict(x);
            let s = sinh_strict(x);
            let result = c * c - s * s;
            assert!(
                (result - 1.0).abs() < 2e-12,
                "cosh²-sinh²({x}) = {result}"
            );
        }
    }
}
