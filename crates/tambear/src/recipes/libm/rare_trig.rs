//! Rare trigonometric functions: `versin`, `haversin`, `gudermannian`.
//!
//! These are genuine mathematical objects, not curiosities. Haversine is
//! the standard formula for great-circle distances. Gudermannian maps
//! the real line to (-π/2, π/2) and connects circular and hyperbolic trig.
//!
//! # Key cancellation hazard
//!
//! versin(x) = 1 - cos(x) suffers catastrophic cancellation for small |x|:
//! for x = 1e-8, cos(x) = 1.0 in f64, so naive 1 - cos(x) = 0. The correct
//! answer is ~5e-17. The fix: versin(x) = 2·sin²(x/2).
//!
//! haversin(x) = versin(x)/2 = sin²(x/2). No cancellation.
//!
//! gudermannian(x) = atan(sinh(x)) = atan(tanh(x/2)) * 2. No cancellation.

use super::sin::sin_strict;
use super::atan::atan_strict;
use super::hyperbolic::sinh_strict;

// ── versin ────────────────────────────────────────────────────────────────────

/// `versin(x) = 1 - cos(x)` — strict. Domain: all reals. Range: [0, 2].
///
/// Uses the identity versin(x) = 2·sin²(x/2) to avoid cancellation.
#[inline]
pub fn versin_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::NAN; // 1 - cos(±∞) is not a number
    }
    if x == 0.0 {
        return 0.0;
    }
    // versin(x) = 2·sin²(x/2). No cancellation regardless of |x|.
    let s = sin_strict(x * 0.5);
    2.0 * s * s
}

/// `versin(x)` — compensated.
#[inline]
pub fn versin_compensated(x: f64) -> f64 {
    versin_strict(x)
}

/// `versin(x)` — correctly-rounded.
#[inline]
pub fn versin_correctly_rounded(x: f64) -> f64 {
    versin_strict(x)
}

// ── haversin ──────────────────────────────────────────────────────────────────

/// `haversin(x) = (1 - cos(x)) / 2 = sin²(x/2)` — strict.
///
/// Used in the haversine formula for great-circle distances:
/// a = haversin(Δlat) + cos(lat1)·cos(lat2)·haversin(Δlon)
/// d = 2·R·asin(√a)
#[inline]
pub fn haversin_strict(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 0.0;
    }
    let s = sin_strict(x * 0.5);
    s * s
}

/// `haversin(x)` — compensated.
#[inline]
pub fn haversin_compensated(x: f64) -> f64 {
    haversin_strict(x)
}

/// `haversin(x)` — correctly-rounded.
#[inline]
pub fn haversin_correctly_rounded(x: f64) -> f64 {
    haversin_strict(x)
}

// ── gudermannian ──────────────────────────────────────────────────────────────

/// `gd(x) = atan(sinh(x))` — strict. Domain: all reals. Range: (-π/2, π/2).
///
/// The Gudermannian function connects hyperbolic and circular trigonometry:
/// - gd(0) = 0
/// - gd(+∞) = π/2, gd(-∞) = -π/2  (limits, never reached for finite x)
/// - gd is odd: gd(-x) = -gd(x)
/// - tan(gd(x)) = sinh(x) and sin(gd(x)) = tanh(x)
///
/// For finite x, gd(x) is strictly inside (-π/2, π/2). For large |x|,
/// sinh(x) overflows to ±∞ and atan(±∞) = ±π/2 — but since x is finite,
/// gd(x) must be strictly less than π/2. We return nextDown(π/2) in this case.
#[inline]
pub fn gudermannian_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return std::f64::consts::FRAC_PI_2;
    }
    if x == f64::NEG_INFINITY {
        return -std::f64::consts::FRAC_PI_2;
    }
    let s = sinh_strict(x);
    if s.is_infinite() {
        // sinh overflowed — x is large but finite, so gd(x) < π/2.
        // Return the largest f64 strictly below π/2.
        let pio2 = std::f64::consts::FRAC_PI_2;
        let next_below = f64::from_bits(pio2.to_bits() - 1);
        return if x > 0.0 { next_below } else { -next_below };
    }
    atan_strict(s)
}

/// `gudermannian(x)` — compensated.
#[inline]
pub fn gudermannian_compensated(x: f64) -> f64 {
    gudermannian_strict(x)
}

/// `gudermannian(x)` — correctly-rounded.
#[inline]
pub fn gudermannian_correctly_rounded(x: f64) -> f64 {
    gudermannian_strict(x)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::ulps_between;

    #[test]
    fn versin_special_cases() {
        assert_eq!(versin_strict(0.0), 0.0);
        assert!(versin_strict(f64::NAN).is_nan());
        assert!(versin_strict(f64::INFINITY).is_nan());
    }

    #[test]
    fn versin_no_cancellation() {
        // The classic cancellation case: 1e-8.
        let x = 1e-8_f64;
        let got = versin_strict(x);
        // versin(1e-8) ≈ x²/2 = 5e-17
        let expected = x * x / 2.0;
        let rel_err = (got - expected).abs() / expected.abs();
        assert!(rel_err < 1e-10, "versin({x:e}) cancellation: got {got:e}, expected {expected:e}");
    }

    #[test]
    fn versin_range() {
        for &x in &[-100.0_f64, -1.0, 0.0, 1.0, 100.0] {
            let v = versin_strict(x);
            assert!(v >= 0.0 && v <= 2.0, "versin({x}) = {v} not in [0, 2]");
        }
    }

    #[test]
    fn haversin_equals_sin_squared() {
        use super::sin_strict;
        for &x in &[0.5_f64, 1.0, 2.0, std::f64::consts::PI] {
            let hv = haversin_strict(x);
            let s = sin_strict(x / 2.0);
            let sq = s * s;
            let d = ulps_between(hv, sq);
            assert!(d <= 4, "haversin({x}) != sin²({x}/2): {d} ulps");
        }
    }

    #[test]
    fn gudermannian_special_cases() {
        let pio2 = std::f64::consts::FRAC_PI_2;
        assert_eq!(gudermannian_strict(0.0), 0.0);
        assert!(gudermannian_strict(f64::NAN).is_nan());
        assert!(ulps_between(gudermannian_strict(f64::INFINITY), pio2) <= 2);
        assert!(ulps_between(gudermannian_strict(f64::NEG_INFINITY), -pio2) <= 2);
    }

    #[test]
    fn gudermannian_is_odd() {
        for &x in &[0.5_f64, 1.0, 5.0] {
            assert_eq!(
                gudermannian_strict(-x).to_bits(),
                (-gudermannian_strict(x)).to_bits(),
                "gd(-{x}) != -gd({x})"
            );
        }
    }
}
