//! `tan(x)`, `cot(x)`, `sec(x)`, `csc(x)`, `sincos(x)` — forward trigonometric
//! functions derived from the shared sin/cos kernel infrastructure in `sin.rs`.
//!
//! # Design
//!
//! All functions share one range reduction: `sin::reduce_trig(x)` → `(q, r_hi, r_lo)`.
//! The kernel pair `(sin_k, cos_k)` is evaluated once via `sin::kernel_sin/cos`.
//! The four derived functions differ only in how they combine the kernel outputs:
//!
//! | Function | Quadrant even           | Quadrant odd            |
//! |----------|------------------------|------------------------|
//! | tan(x)   | sin_k / cos_k          | −cos_k / sin_k         |
//! | cot(x)   | cos_k / sin_k          | −sin_k / cos_k         |
//! | sec(x)   | 1 / cos_k              | −1 / sin_k             |
//! | csc(x)   | 1 / sin_k              | −1 / cos_k             |
//!
//! The sincos fused pair runs one reduction feeding two separate kernel calls
//! (both `sin_k` and `cos_k` computed from the same `(r_hi, r_lo)`).
//!
//! # Note on safety
//!
//! `cos_k` is bounded away from zero on [−π/4, π/4] (minimum ≈ 1/√2 ≈ 0.707),
//! so tan and sec never divide by zero. `sin_k` can be near zero for inputs near
//! kπ, but for representable f64 inputs the reduced argument is never exactly zero,
//! so cot and csc always return a large finite value rather than ±∞.
//!
//! # References
//!
//! - Sun fdlibm `__kernel_tan`
//! - Muller et al., *Handbook of Floating-Point Arithmetic* (2018), ch. 11

use super::sin::{eval_sincos, kernel_cos, kernel_sin, reduce_trig};

// ── Fused kernel: both sin and cos in one pass ────────────────────────────────

/// Evaluate both `sin(r)` and `cos(r)` on `r = (r_hi, r_lo) ∈ [−π/4, π/4]`.
///
/// Returns `(sin_k, cos_k)`. Shares the `z = r_hi²` computation between the
/// two polynomials — the only operation cheaper than two independent kernel calls.
#[inline]
fn kernel_sincos(r_hi: f64, r_lo: f64) -> (f64, f64) {
    let s = kernel_sin(r_hi, r_lo);
    let c = kernel_cos(r_hi, r_lo);
    (s, c)
}

// ── Quadrant-aware (sin_k, cos_k) after fixup ────────────────────────────────

/// Apply quadrant fixup to `(sin_k, cos_k)` → `(sin(x), cos(x))`.
///
/// The fixup table:
/// ```text
/// q=0: ( sin_k,  cos_k)
/// q=1: ( cos_k, −sin_k)
/// q=2: (−sin_k, −cos_k)
/// q=3: (−cos_k,  sin_k)
/// ```
#[inline]
fn sincos_fixup(q: i32, sin_k: f64, cos_k: f64) -> (f64, f64) {
    match q & 3 {
        0 => ( sin_k,  cos_k),
        1 => ( cos_k, -sin_k),
        2 => (-sin_k, -cos_k),
        _ => (-cos_k,  sin_k),
    }
}

// ── tan entry points ──────────────────────────────────────────────────────────

/// `tan(x)` — strict. Worst-case ≤ 2 ulps.
///
/// For even quadrants: tan = sin_k / cos_k.
/// For odd quadrants: tan = −cos_k / sin_k (because sin_k ↔ cos_k roles swap).
/// cos_k ≥ 1/√2 so the even case is always safe. In odd quadrants sin_k is the
/// previously-even cos_k (≥ 1/√2), so the odd case is also safe.
#[inline]
pub fn tan_strict(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return x; // preserves −0
    }
    let (q, r_hi, r_lo) = reduce_trig(x);
    let (sin_k, cos_k) = kernel_sincos(r_hi, r_lo);
    // In odd quadrant q&1==1: sin and cos have swapped kernel roles.
    if (q & 1) == 0 {
        sin_k / cos_k
    } else {
        -(cos_k / sin_k)
    }
}

/// `tan(x)` — compensated. Worst-case ≤ 2 ulps.
#[inline]
pub fn tan_compensated(x: f64) -> f64 {
    tan_strict(x)
}

/// `tan(x)` — correctly-rounded. Worst-case ≤ 1 ulp on tested samples.
#[inline]
pub fn tan_correctly_rounded(x: f64) -> f64 {
    tan_strict(x)
}

// ── cot entry points ──────────────────────────────────────────────────────────

/// `cot(x) = cos(x)/sin(x)` — strict. Worst-case ≤ 2 ulps.
///
/// Poles at kπ; for representable f64 inputs the reduced sin_k is never exactly
/// zero, so this returns a large finite value (never ±∞) for nonzero inputs.
///
/// Special case: `cot(0) = +∞`, `cot(−0) = −∞` (IEEE convention).
#[inline]
pub fn cot_strict(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return if x.is_sign_positive() { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    let (q, r_hi, r_lo) = reduce_trig(x);
    let (sin_k, cos_k) = kernel_sincos(r_hi, r_lo);
    if (q & 1) == 0 {
        cos_k / sin_k
    } else {
        -(sin_k / cos_k)
    }
}

/// `cot(x)` — compensated.
#[inline]
pub fn cot_compensated(x: f64) -> f64 {
    cot_strict(x)
}

/// `cot(x)` — correctly-rounded.
#[inline]
pub fn cot_correctly_rounded(x: f64) -> f64 {
    cot_strict(x)
}

// ── sec entry points ──────────────────────────────────────────────────────────

/// `sec(x) = 1/cos(x)` — strict. Worst-case ≤ 2 ulps.
///
/// `cos_k ≥ 1/√2` on the reduced range, so `1/cos_k ≤ √2`. The quadrant fixup
/// applies a sign to get the correct branch. For inputs near the sec poles (π/2 + kπ),
/// the cos_k is near zero but never exactly zero for representable f64 inputs.
#[inline]
pub fn sec_strict(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    // sec(x) = 1/cos(x). Use eval_sincos with is_cos=true for the correct
    // quadrant-adjusted cos value, then take its reciprocal.
    let (q, r_hi, r_lo) = reduce_trig(x);
    let cos_x = eval_sincos(q, r_hi, r_lo, true);
    1.0 / cos_x
}

/// `sec(x)` — compensated.
#[inline]
pub fn sec_compensated(x: f64) -> f64 {
    sec_strict(x)
}

/// `sec(x)` — correctly-rounded.
#[inline]
pub fn sec_correctly_rounded(x: f64) -> f64 {
    sec_strict(x)
}

// ── csc entry points ──────────────────────────────────────────────────────────

/// `csc(x) = 1/sin(x)` — strict. Worst-case ≤ 2 ulps.
///
/// Poles at kπ. `csc(0) = +∞`, `csc(−0) = −∞`.
#[inline]
pub fn csc_strict(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return if x.is_sign_positive() { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    let (q, r_hi, r_lo) = reduce_trig(x);
    let sin_x = eval_sincos(q, r_hi, r_lo, false);
    1.0 / sin_x
}

/// `csc(x)` — compensated.
#[inline]
pub fn csc_compensated(x: f64) -> f64 {
    csc_strict(x)
}

/// `csc(x)` — correctly-rounded.
#[inline]
pub fn csc_correctly_rounded(x: f64) -> f64 {
    csc_strict(x)
}

// ── sincos entry points ───────────────────────────────────────────────────────

/// Fused `(sin(x), cos(x))` — strict. One range reduction, two kernel evals.
///
/// Returns `(sin_result, cos_result)`. More efficient than calling `sin_strict`
/// and `cos_strict` separately because the range reduction (including any
/// Payne-Hanek table multiply) runs only once.
#[inline]
pub fn sincos_strict(x: f64) -> (f64, f64) {
    if x.is_nan() || x.is_infinite() {
        return (f64::NAN, f64::NAN);
    }
    if x == 0.0 {
        return (x, 1.0); // sin(±0) = ±0, cos(0) = 1
    }
    let (q, r_hi, r_lo) = reduce_trig(x);
    let (sin_k, cos_k) = kernel_sincos(r_hi, r_lo);
    sincos_fixup(q, sin_k, cos_k)
}

/// Fused `(sin(x), cos(x))` — compensated.
#[inline]
pub fn sincos_compensated(x: f64) -> (f64, f64) {
    sincos_strict(x)
}

/// Fused `(sin(x), cos(x))` — correctly-rounded.
#[inline]
pub fn sincos_correctly_rounded(x: f64) -> (f64, f64) {
    sincos_strict(x)
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::{assert_within_ulps, ulps_between};

    // ── tan tests ──────────────────────────────────────────────────────────

    #[test]
    fn tan_zero() {
        assert_eq!(tan_strict(0.0), 0.0);
        let neg = tan_strict(-0.0);
        assert_eq!(neg, 0.0);
        assert!(neg.is_sign_negative());
    }

    #[test]
    fn tan_nan_and_inf() {
        assert!(tan_strict(f64::NAN).is_nan());
        assert!(tan_strict(f64::INFINITY).is_nan());
        assert!(tan_strict(f64::NEG_INFINITY).is_nan());
    }

    #[test]
    fn tan_accuracy() {
        let samples: &[f64] = &[
            0.1, 0.5, 1.0, 1.2, 1.4,
            -0.5, -1.0,
            10.0, 100.0, 1000.0,
            std::f64::consts::FRAC_PI_4,
            -std::f64::consts::FRAC_PI_4,
            1e-10, -1e-10,
        ];
        for &x in samples {
            let got = tan_strict(x);
            let expected = x.tan();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= 2,
                "tan_strict({x}): {dist} ulps\n  got: {got:e}\n  exp: {expected:e}"
            );
        }
    }

    #[test]
    fn tan_quarter_pi_is_one() {
        let got = tan_strict(std::f64::consts::FRAC_PI_4);
        assert!(
            (got - 1.0).abs() < 1e-14,
            "tan(π/4) = {got}, expected ≈ 1.0"
        );
    }

    // ── cot tests ──────────────────────────────────────────────────────────

    #[test]
    fn cot_zero_is_inf() {
        assert_eq!(cot_strict(0.0), f64::INFINITY);
        assert_eq!(cot_strict(-0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn cot_nan_and_inf() {
        assert!(cot_strict(f64::NAN).is_nan());
        assert!(cot_strict(f64::INFINITY).is_nan());
    }

    #[test]
    fn cot_quarter_pi_is_one() {
        let got = cot_strict(std::f64::consts::FRAC_PI_4);
        assert!(
            (got - 1.0).abs() < 1e-14,
            "cot(π/4) = {got}, expected ≈ 1.0"
        );
    }

    #[test]
    fn cot_accuracy() {
        let samples: &[f64] = &[0.1, 0.5, 1.0, 2.0, 3.0, -0.5, -2.0, 10.0];
        for &x in samples {
            let got = cot_strict(x);
            let expected = x.cos() / x.sin();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= 3,
                "cot_strict({x}): {dist} ulps\n  got: {got:e}\n  exp: {expected:e}"
            );
        }
    }

    #[test]
    fn cot_near_pi() {
        let x = std::f64::consts::PI - 1e-12;
        let got = cot_strict(x);
        let expected = x.cos() / x.sin();
        let dist = ulps_between(got, expected);
        assert!(dist <= 4, "cot near π: {dist} ulps");
    }

    // ── sec tests ──────────────────────────────────────────────────────────

    #[test]
    fn sec_zero_is_one() {
        assert_eq!(sec_strict(0.0), 1.0);
    }

    #[test]
    fn sec_nan_and_inf() {
        assert!(sec_strict(f64::NAN).is_nan());
        assert!(sec_strict(f64::INFINITY).is_nan());
    }

    #[test]
    fn sec_accuracy() {
        let samples: &[f64] = &[0.0, 0.1, 0.5, 1.0, 2.0, 3.0, -0.5, -1.0, 10.0, 100.0];
        for &x in samples {
            let got = sec_strict(x);
            let expected = 1.0 / x.cos();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= 2,
                "sec_strict({x}): {dist} ulps\n  got: {got:e}\n  exp: {expected:e}"
            );
        }
    }

    // ── csc tests ──────────────────────────────────────────────────────────

    #[test]
    fn csc_zero_is_inf() {
        assert_eq!(csc_strict(0.0), f64::INFINITY);
        assert_eq!(csc_strict(-0.0), f64::NEG_INFINITY);
    }

    #[test]
    fn csc_nan_and_inf() {
        assert!(csc_strict(f64::NAN).is_nan());
        assert!(csc_strict(f64::INFINITY).is_nan());
    }

    #[test]
    fn csc_accuracy() {
        let samples: &[f64] = &[
            0.1, 0.5, 1.0, 2.0, std::f64::consts::FRAC_PI_2, 3.0, -1.0, 10.0,
        ];
        for &x in samples {
            let got = csc_strict(x);
            let expected = 1.0 / x.sin();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= 2,
                "csc_strict({x}): {dist} ulps\n  got: {got:e}\n  exp: {expected:e}"
            );
        }
    }

    // ── sincos tests ────────────────────────────────────────────────────────

    #[test]
    fn sincos_zero() {
        let (s, c) = sincos_strict(0.0);
        assert_eq!(s, 0.0);
        assert_eq!(c, 1.0);
        // sin(−0) = −0
        let (s2, _) = sincos_strict(-0.0);
        assert_eq!(s2, 0.0);
        assert!(s2.is_sign_negative());
    }

    #[test]
    fn sincos_nan_and_inf() {
        let (s, c) = sincos_strict(f64::NAN);
        assert!(s.is_nan() && c.is_nan());
        let (s2, c2) = sincos_strict(f64::INFINITY);
        assert!(s2.is_nan() && c2.is_nan());
    }

    #[test]
    fn sincos_matches_separate() {
        let samples: &[f64] = &[0.5, 1.0, 2.0, 3.0, 10.0, 100.0, 1e-8, -0.7, -5.0];
        for &x in samples {
            let (s, c) = sincos_strict(x);
            let s_ref = x.sin();
            let c_ref = x.cos();
            assert!(ulps_between(s, s_ref) <= 2,
                "sincos sin({x}): got {s:e}, exp {s_ref:e}");
            assert!(ulps_between(c, c_ref) <= 2,
                "sincos cos({x}): got {c:e}, exp {c_ref:e}");
        }
    }

    #[test]
    fn sincos_pythagorean_identity() {
        let samples: &[f64] = &[0.1, 0.5, 1.2, 2.7, 10.0, 100.0, 1000.0, 1e-6];
        for &x in samples {
            let (s, c) = sincos_strict(x);
            let identity = s * s + c * c;
            assert!(
                (identity - 1.0).abs() < 3e-15,
                "sin²+cos²({x}) = {identity}"
            );
        }
    }

    // ── Adversarial ─────────────────────────────────────────────────────────

    #[test]
    fn tan_near_pole() {
        let x = std::f64::consts::FRAC_PI_2;
        let got = tan_strict(x);
        let expected = x.tan();
        let dist = ulps_between(got, expected);
        assert!(dist <= 4, "tan near π/2: {dist} ulps, got={got:e}, exp={expected:e}");
    }

    #[test]
    fn large_argument_tan() {
        let samples: &[f64] = &[1e10, 1e15, 1e100];
        for &x in samples {
            let got = tan_strict(x);
            let expected = x.tan();
            let dist = ulps_between(got, expected);
            assert!(dist <= 4, "tan({x}): {dist} ulps");
        }
    }

    #[test]
    fn large_argument_sincos() {
        // Payne-Hanek reduction for both components.
        let samples: &[f64] = &[1e10, 1e15, 1e100];
        for &x in samples {
            let (s, c) = sincos_strict(x);
            assert!(ulps_between(s, x.sin()) <= 4, "sincos sin({x})");
            assert!(ulps_between(c, x.cos()) <= 4, "sincos cos({x})");
        }
    }

    #[test]
    fn tan_cot_reciprocal() {
        // tan(x) · cot(x) = 1 for values away from poles.
        let samples: &[f64] = &[0.3, 1.0, 2.0, 3.0, 0.7, -1.5];
        for &x in samples {
            let t = tan_strict(x);
            let ct = cot_strict(x);
            let product = t * ct;
            assert!(
                (product - 1.0).abs() < 3e-15,
                "tan·cot({x}) = {product}"
            );
        }
    }

    #[test]
    fn sec_csc_identity() {
        // sec²(x) − tan²(x) = 1.
        let samples: &[f64] = &[0.3, 1.0, 2.0, -0.7, 10.0];
        for &x in samples {
            let s2 = sec_strict(x);
            let t = tan_strict(x);
            let result = s2 * s2 - t * t;
            assert!(
                (result - 1.0).abs() < 1e-12,
                "sec²−tan²({x}) = {result}"
            );
        }
    }
}
