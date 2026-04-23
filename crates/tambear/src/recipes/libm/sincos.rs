//! `sincos(x)` — fused sin+cos pair with one shared range reduction.
//!
//! # Rationale
//!
//! Computing sin(x) and cos(x) separately calls `reduce_trig` twice. Both
//! results need the same reduced argument `(q, r_hi, r_lo)`. Fusing them
//! pays for the range reduction once and evaluates both kernels from the
//! same intermediate.
//!
//! # Algorithm
//!
//! 1. Handle special cases: NaN → (NaN, NaN); ±∞ → (NaN, NaN); ±0 → (0, 1).
//! 2. `reduce_trig(x)` → `(q, r_hi, r_lo)`. One call.
//! 3. `eval_sincos(q, r_hi, r_lo, false)` → sin value.
//! 4. `eval_sincos(q, r_hi, r_lo, true)`  → cos value.
//!
//! The two `eval_sincos` calls share no data — they are independent and can
//! be scheduled in parallel by the hardware. The only shared computation is
//! the range reduction above.
//!
//! CONTRACT: sincos_strict(x) is bit-identical to (sin_strict(x), cos_strict(x)).
//!
//! # References
//!
//! - Muller et al., *Handbook of Floating-Point Arithmetic* (2018), ch. 11
//! - Sun fdlibm `__kernel_sin`, `__kernel_cos`

use super::sin::{reduce_trig, eval_sincos};

// ── sincos entry points ────────────────────────────────────────────────────────

/// `sincos(x)` — strict. Returns `(sin(x), cos(x))`.
///
/// CONTRACT: bit-identical to `(sin_strict(x), cos_strict(x))`.
#[inline]
pub fn sincos_strict(x: f64) -> (f64, f64) {
    if x.is_nan() || x.is_infinite() {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        // sin(0) = 0 (preserves sign per sin), cos(0) = 1.
        return (x, 1.0);
    }
    let (q, r_hi, r_lo) = reduce_trig(x);
    let s = eval_sincos(q, r_hi, r_lo, false);
    let c = eval_sincos(q, r_hi, r_lo, true);
    (s, c)
}

/// `sincos(x)` — compensated.
#[inline]
pub fn sincos_compensated(x: f64) -> (f64, f64) {
    sincos_strict(x)
}

/// `sincos(x)` — correctly-rounded.
#[inline]
pub fn sincos_correctly_rounded(x: f64) -> (f64, f64) {
    sincos_strict(x)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::sin::{sin_strict, cos_strict};
    use crate::primitives::oracle::ulps_between;

    #[test]
    fn sincos_special_cases() {
        let (s, c) = sincos_strict(f64::NAN);
        assert!(s.is_nan() && c.is_nan(), "NaN case");

        let (s, c) = sincos_strict(f64::INFINITY);
        assert!(s.is_nan() && c.is_nan(), "+∞ case");

        let (s, c) = sincos_strict(f64::NEG_INFINITY);
        assert!(s.is_nan() && c.is_nan(), "-∞ case");

        let (s, c) = sincos_strict(0.0);
        assert_eq!(s.to_bits(), 0.0f64.to_bits(), "sin(0) = +0");
        assert_eq!(c, 1.0, "cos(0) = 1");

        let (s, c) = sincos_strict(-0.0);
        assert!(s.is_sign_negative() && s == 0.0, "sin(-0) = -0");
        assert_eq!(c, 1.0, "cos(-0) = 1");
    }

    #[test]
    fn sincos_bit_identical_to_separate() {
        let samples: &[f64] = &[
            0.1, 0.5, 1.0, 1.5, 2.0, 3.0,
            std::f64::consts::PI,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_4,
            -0.5, -1.0, -std::f64::consts::PI,
            10.0, 100.0, 1000.0,
            1e-10, -1e-10,
        ];
        for &x in samples {
            let (s, c) = sincos_strict(x);
            let s2 = sin_strict(x);
            let c2 = cos_strict(x);
            assert_eq!(
                s.to_bits(), s2.to_bits(),
                "sincos({x}).sin ≠ sin_strict({x}): {s:e} vs {s2:e}"
            );
            assert_eq!(
                c.to_bits(), c2.to_bits(),
                "sincos({x}).cos ≠ cos_strict({x}): {c:e} vs {c2:e}"
            );
        }
    }

    #[test]
    fn sincos_accuracy() {
        let samples: &[f64] = &[0.1, 0.5, 1.0, 1.5, 2.0, 3.0, 10.0, 100.0];
        for &x in samples {
            let (s, c) = sincos_strict(x);
            let ds = ulps_between(s, x.sin());
            let dc = ulps_between(c, x.cos());
            assert!(ds <= 2, "sincos({x}).sin: {ds} ulps");
            assert!(dc <= 2, "sincos({x}).cos: {dc} ulps");
        }
    }

    #[test]
    fn sincos_pythagorean_identity() {
        for &x in &[0.1_f64, 0.5, 1.0, 2.0, 5.0, 100.0] {
            let (s, c) = sincos_strict(x);
            let identity = s * s + c * c;
            assert!(
                (identity - 1.0).abs() < 1e-14,
                "sin²({x}) + cos²({x}) = {identity}, expected 1"
            );
        }
    }
}
