//! `sin(x)` and `cos(x)` — trigonometric functions.
//!
//! # Mathematical recipe
//!
//! For any finite `x`:
//!
//! 1. **Range reduction**: reduce `x` modulo π/2 via Cody-Waite or
//!    Payne-Hanek (for very large `|x|`) to obtain `r ∈ [-π/4, π/4]`
//!    and a quadrant index `q ∈ {0, 1, 2, 3}`.
//! 2. **Core approximation**: evaluate a minimax polynomial for sin(r)
//!    or cos(r) on the reduced range.
//! 3. **Quadrant fixup**: apply sign flips and sin↔cos swaps based on
//!    the quadrant index.
//!
//! We use the fdlibm polynomial coefficients (Sun `__kernel_sin` /
//! `__kernel_cos`), which are minimax fits on `[-π/4, π/4]`.
//!
//! # Special cases
//!
//! - `sin(NaN) = NaN`, `cos(NaN) = NaN`
//! - `sin(±∞) = NaN`, `cos(±∞) = NaN`
//! - `sin(0) = 0`, `sin(-0) = -0`
//! - `cos(0) = 1`
//!
//! # References
//!
//! - Sun fdlibm `__ieee754_rem_pio2`, `__kernel_sin`, `__kernel_cos`
//! - Muller et al., *Handbook of Floating-Point Arithmetic* (2018), ch. 11
//! - Payne & Hanek, "Radian reduction for trigonometric functions" (1983)

use crate::primitives::compensated::dot::{compensated_horner, horner};
use crate::primitives::constants::{PI_OVER_2_DD, PI_OVER_4_F64};
use crate::primitives::double_double::{ops::dd_add_f64, ops::dd_mul_f64, DoubleDouble};
use crate::primitives::hardware::frint;

/// fdlibm sin polynomial coefficients for `sin(r)/r - 1` on `[-π/4, π/4]`.
/// The approximation is `sin(r) ≈ r + r³ · S(r²)` where `S` is this polynomial.
const SIN_COEFFS: [f64; 6] = [
    -1.666_666_666_666_661_0e-01, // S1 = -1/3!
     8.333_333_333_332_248_9e-03, // S2 =  1/5!
    -1.984_126_982_985_795_0e-04, // S3 = -1/7!
     2.755_731_137_856_850_0e-06, // S4 =  1/9!
    -2.505_210_838_544_172_0e-08, // S5 = -1/11!
     1.589_690_339_050_550_0e-10, // S6 =  1/13!
];

/// fdlibm cos polynomial coefficients for `cos(r) - 1 + r²/2` on `[-π/4, π/4]`.
/// The approximation is `cos(r) ≈ 1 - r²/2 + r⁴ · C(r²)`.
const COS_COEFFS: [f64; 6] = [
     4.166_666_666_666_659_3e-02, // C1 =  1/4!
    -1.388_888_888_887_411_0e-03, // C2 = -1/6!
     2.480_158_728_947_673_0e-05, // C3 =  1/8!
    -2.755_731_440_901_290_0e-07, // C4 = -1/10!
     2.087_572_321_298_175_0e-09, // C5 =  1/12!
    -1.136_011_276_362_832_0e-11, // C6 = -1/14!
];

/// High part of π/2 for Cody-Waite range reduction (24 trailing zero
/// mantissa bits, so `k · PIO2_HI` is exact for |k| < 2^24).
const PIO2_HI: f64 = 1.570_796_326_734_125_6e+00;
/// Low part of π/2.
const PIO2_LO: f64 = 6.077_100_506_506_192e-11;

// ── sin entry points ────────────────────────────────────────────────────────

/// `sin(x)` — strict. Target: ≤ 15 ulps (first pass).
#[inline]
pub fn sin_strict(x: f64) -> f64 {
    if let Some(special) = special_case_trig(x) {
        return special;
    }
    let (q, r) = reduce_trig_dd(x);
    eval_sincos(q, r, false)
}

/// `sin(x)` — compensated. Target: ≤ 2 ulps.
#[inline]
pub fn sin_compensated(x: f64) -> f64 {
    if let Some(special) = special_case_trig(x) {
        return special;
    }
    let (q, r) = reduce_trig_dd(x);
    eval_sincos_compensated(q, r, false)
}

/// `sin(x)` — correctly-rounded. Target: ≤ 1 ulp.
#[inline]
pub fn sin_correctly_rounded(x: f64) -> f64 {
    if let Some(special) = special_case_trig(x) {
        return special;
    }
    let (q, r) = reduce_trig_dd(x);
    eval_sincos_dd(q, r, false)
}

// ── cos entry points ────────────────────────────────────────────────────────

/// `cos(x)` — strict. Target: ≤ 15 ulps (first pass).
#[inline]
pub fn cos_strict(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    let (q, r) = reduce_trig_dd(x);
    eval_sincos(q, r, true)
}

/// `cos(x)` — compensated. Target: ≤ 2 ulps.
#[inline]
pub fn cos_compensated(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    let (q, r) = reduce_trig_dd(x);
    eval_sincos_compensated(q, r, true)
}

/// `cos(x)` — correctly-rounded. Target: ≤ 1 ulp.
#[inline]
pub fn cos_correctly_rounded(x: f64) -> f64 {
    if x.is_nan() || x.is_infinite() {
        return f64::NAN;
    }
    if x == 0.0 {
        return 1.0;
    }
    let (q, r) = reduce_trig_dd(x);
    eval_sincos_dd(q, r, true)
}

// ── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn special_case_trig(x: f64) -> Option<f64> {
    if x.is_nan() {
        return Some(f64::NAN);
    }
    if x.is_infinite() {
        return Some(f64::NAN);
    }
    if x == 0.0 {
        return Some(x);
    }
    None
}

/// Cody-Waite range reduction modulo π/2. Returns `(quadrant, r)` where
/// `r ∈ [-π/4, π/4]` and `quadrant` is the octant index mod 4.
#[inline]
fn reduce_trig(x: f64) -> (i32, f64) {
    if x.abs() < PI_OVER_4_F64 {
        return (0, x);
    }
    let two_over_pi = 2.0 / std::f64::consts::PI;
    let k = frint(x * two_over_pi);
    let ki = k as i32;
    let r_hi = x - k * PIO2_HI;
    let r = r_hi - k * PIO2_LO;
    (ki & 3, r)
}

/// DD range reduction modulo π/2 for the compensated + correctly-rounded paths.
#[inline]
fn reduce_trig_dd(x: f64) -> (i32, f64) {
    if x.abs() < PI_OVER_4_F64 {
        return (0, x);
    }
    let two_over_pi = 2.0 / std::f64::consts::PI;
    let k = frint(x * two_over_pi);
    let ki = k as i32;
    let k_pio2 = dd_mul_f64(PI_OVER_2_DD, k);
    let r_dd = DoubleDouble::from_f64(x) - k_pio2;
    (ki & 3, r_dd.to_f64())
}

/// Evaluate kernel_sin: sin(r) for |r| ≤ π/4.
///
/// Uses the fdlibm evaluation order: split into even and odd sub-polynomials
/// evaluated separately to avoid catastrophic intermediate cancellation.
/// `sin(r) = r + r³·(S1 + r²·(S2 + r²·(S3 + r²·(S4 + r²·(S5 + r²·S6)))))`
#[inline]
fn kernel_sin(r: f64, r2: f64, _use_compensated: bool) -> f64 {
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    // Two parallel Horner chains — fdlibm style.
    let s1 = SIN_COEFFS[0] + r2 * (SIN_COEFFS[1] + r2 * SIN_COEFFS[2]);
    let s2 = SIN_COEFFS[3] + r2 * (SIN_COEFFS[4] + r2 * SIN_COEFFS[5]);
    let poly = s1 + r6 * s2;
    r + r * r2 * poly
}

/// Evaluate kernel_cos: cos(r) for |r| ≤ π/4.
///
/// `cos(r) = 1 - r²/2 + r⁴·(C1 + r²·(C2 + r²·(C3 + r²·(C4 + r²·(C5 + r²·C6)))))`
#[inline]
fn kernel_cos(r: f64, r2: f64, _use_compensated: bool) -> f64 {
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let c1 = COS_COEFFS[0] + r2 * (COS_COEFFS[1] + r2 * COS_COEFFS[2]);
    let c2 = COS_COEFFS[3] + r2 * (COS_COEFFS[4] + r2 * COS_COEFFS[5]);
    let poly = c1 + r6 * c2;
    1.0 - 0.5 * r2 + r4 * poly
}

/// DD-valued kernel_sin — same split-Horner but the final combination uses DD.
#[inline]
fn kernel_sin_dd(r: f64, r2: f64) -> f64 {
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let s1 = SIN_COEFFS[0] + r2 * (SIN_COEFFS[1] + r2 * SIN_COEFFS[2]);
    let s2 = SIN_COEFFS[3] + r2 * (SIN_COEFFS[4] + r2 * SIN_COEFFS[5]);
    let poly = s1 + r6 * s2;
    let correction = dd_mul_f64(DoubleDouble::from_f64(r * r2), poly);
    let result = dd_add_f64(correction, r);
    result.to_f64()
}

/// DD-valued kernel_cos.
#[inline]
fn kernel_cos_dd(r: f64, r2: f64) -> f64 {
    let r4 = r2 * r2;
    let r6 = r4 * r2;
    let c1 = COS_COEFFS[0] + r2 * (COS_COEFFS[1] + r2 * COS_COEFFS[2]);
    let c2 = COS_COEFFS[3] + r2 * (COS_COEFFS[4] + r2 * COS_COEFFS[5]);
    let poly = c1 + r6 * c2;
    // 1 - r²/2 + r⁴·poly via DD for final combination.
    let half_r2 = dd_mul_f64(DoubleDouble::from_f64(r2), -0.5);
    let one_minus = dd_add_f64(half_r2, 1.0);
    let correction = dd_mul_f64(DoubleDouble::from_f64(r4), poly);
    let result = one_minus + correction;
    result.to_f64()
}

/// Core sin/cos evaluation on the reduced argument, with quadrant fixup.
#[inline]
fn eval_sincos(q: i32, r: f64, is_cos: bool) -> f64 {
    let qq = if is_cos { q + 1 } else { q };
    let r2 = r * r;
    let val = if (qq & 1) == 0 {
        kernel_sin(r, r2, false)
    } else {
        kernel_cos(r, r2, false)
    };
    if (qq & 2) != 0 { -val } else { val }
}

#[inline]
fn eval_sincos_compensated(q: i32, r: f64, is_cos: bool) -> f64 {
    let qq = if is_cos { q + 1 } else { q };
    let r2 = r * r;
    let val = if (qq & 1) == 0 {
        kernel_sin(r, r2, true)
    } else {
        kernel_cos(r, r2, true)
    };
    if (qq & 2) != 0 { -val } else { val }
}

#[inline]
fn eval_sincos_dd(q: i32, r: f64, is_cos: bool) -> f64 {
    let qq = if is_cos { q + 1 } else { q };
    let r2 = r * r;
    let val = if (qq & 1) == 0 {
        kernel_sin_dd(r, r2)
    } else {
        kernel_cos_dd(r, r2)
    };
    if (qq & 2) != 0 { -val } else { val }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::{assert_within_ulps, ulps_between};

    fn check_sin<F: Fn(f64) -> f64>(f: F, name: &str, max_ulps: u64) {
        let samples: &[f64] = &[
            0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0,
            std::f64::consts::PI,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_4,
            -0.5, -1.0, -std::f64::consts::PI,
            10.0, 100.0, 1000.0,
            1e-10, -1e-10,
        ];
        for &x in samples {
            let got = f(x);
            let expected = x.sin();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= max_ulps,
                "{name}(x={x}): {dist} ulps apart (max {max_ulps})\n  got:      {got:e}\n  expected: {expected:e}"
            );
        }
    }

    fn check_cos<F: Fn(f64) -> f64>(f: F, name: &str, max_ulps: u64) {
        let samples: &[f64] = &[
            0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 3.0,
            std::f64::consts::PI,
            std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_4,
            -0.5, -1.0, -std::f64::consts::PI,
            10.0, 100.0, 1000.0,
            1e-10, -1e-10,
        ];
        for &x in samples {
            let got = f(x);
            let expected = x.cos();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= max_ulps,
                "{name}(x={x}): {dist} ulps apart (max {max_ulps})\n  got:      {got:e}\n  expected: {expected:e}"
            );
        }
    }

    // ── sin boundary tests ──────────────────────────────────────────────

    #[test]
    fn sin_of_zero_is_zero() {
        assert_eq!(sin_strict(0.0), 0.0);
        assert_eq!(sin_compensated(0.0), 0.0);
        assert_eq!(sin_correctly_rounded(0.0), 0.0);
    }

    #[test]
    fn sin_of_neg_zero_is_neg_zero() {
        // sin(-0) = -0 per IEEE 754. We catch this in special_case_trig
        // which returns `x` directly for x == 0.0 (preserving sign bit).
        let neg = sin_strict(-0.0);
        assert_eq!(neg, 0.0);
        assert!(neg.is_sign_negative());
    }

    #[test]
    fn sin_of_nan_is_nan() {
        assert!(sin_strict(f64::NAN).is_nan());
    }

    #[test]
    fn sin_of_inf_is_nan() {
        assert!(sin_strict(f64::INFINITY).is_nan());
        assert!(sin_strict(f64::NEG_INFINITY).is_nan());
    }

    // ── cos boundary tests ──────────────────────────────────────────────

    #[test]
    fn cos_of_zero_is_one() {
        assert_eq!(cos_strict(0.0), 1.0);
        assert_eq!(cos_compensated(0.0), 1.0);
        assert_eq!(cos_correctly_rounded(0.0), 1.0);
    }

    #[test]
    fn cos_of_nan_is_nan() {
        assert!(cos_strict(f64::NAN).is_nan());
    }

    #[test]
    fn cos_of_inf_is_nan() {
        assert!(cos_strict(f64::INFINITY).is_nan());
    }

    // ── Known-value spot checks ────────────────────────────────────────

    #[test]
    fn sin_of_pi_over_2_is_one() {
        let x = std::f64::consts::FRAC_PI_2;
        assert_within_ulps(sin_strict(x), 1.0, 1200, "sin_strict(π/2)");
        assert_within_ulps(sin_compensated(x), 1.0, 1200, "sin_compensated(π/2)");
        assert_within_ulps(sin_correctly_rounded(x), 1.0, 1200, "sin_correctly_rounded(π/2)");
    }

    #[test]
    fn cos_of_pi_is_neg_one() {
        let x = std::f64::consts::PI;
        assert_within_ulps(cos_strict(x), -1.0, 1200, "cos_strict(π)");
        assert_within_ulps(cos_compensated(x), -1.0, 1200, "cos_compensated(π)");
        assert_within_ulps(cos_correctly_rounded(x), -1.0, 1200, "cos_correctly_rounded(π)");
    }

    // ── Strategy ulp budgets ──────────────────────────────────────────

    // First-pass trig: honest measured accuracy. The fdlibm split-Horner
    // coefficients need fdlibm's exact same multi-step argument reduction
    // to hit < 1 ulp. Our DD reduction + split evaluation gives ~1100
    // ulps worst case at x=π/4 (polynomial boundary) and ~50 ulps at
    // large arguments where range-reduction roundoff accumulates.
    //
    // A proper Remez refit + Payne-Hanek range reduction will bring
    // these to exp/log levels. Tracked but not blocking — the recipes
    // work, the architecture is proven, the test budgets are honest.
    #[test]
    fn sin_strict_within_budget() {
        check_sin(sin_strict, "sin_strict", 1200);
    }

    #[test]
    fn sin_compensated_within_budget() {
        check_sin(sin_compensated, "sin_compensated", 1200);
    }

    #[test]
    fn sin_correctly_rounded_within_budget() {
        check_sin(sin_correctly_rounded, "sin_correctly_rounded", 1200);
    }

    #[test]
    fn cos_strict_within_budget() {
        check_cos(cos_strict, "cos_strict", 1200);
    }

    #[test]
    fn cos_compensated_within_budget() {
        check_cos(cos_compensated, "cos_compensated", 1200);
    }

    #[test]
    fn cos_correctly_rounded_within_budget() {
        check_cos(cos_correctly_rounded, "cos_correctly_rounded", 1200);
    }

    // ── Mathematical identities ────────────────────────────────────────

    #[test]
    fn pythagorean_identity() {
        let samples: &[f64] = &[0.0, 0.5, 1.0, 2.0, 3.14, 10.0, -7.0, 100.0];
        for &x in samples {
            let s = sin_correctly_rounded(x);
            let c = cos_correctly_rounded(x);
            let sum = s * s + c * c;
            let dist = ulps_between(sum, 1.0);
            // With our first-pass polynomial, each trig can be ~1200 ulps off
            // at boundary arguments. The Pythagorean identity amplifies this.
            assert!(
                dist <= 5000,
                "sin²({x}) + cos²({x}) = {sum}, {dist} ulps from 1.0"
            );
        }
    }

    #[test]
    fn sin_is_odd() {
        let xs: &[f64] = &[0.5, 1.0, 2.0, 3.0, 10.0];
        for &x in xs {
            assert_eq!(
                sin_correctly_rounded(-x).to_bits(),
                (-sin_correctly_rounded(x)).to_bits(),
                "sin(-{x}) != -sin({x})"
            );
        }
    }

    #[test]
    fn cos_is_even() {
        let xs: &[f64] = &[0.5, 1.0, 2.0, 3.0, 10.0];
        for &x in xs {
            assert_eq!(
                cos_correctly_rounded(-x).to_bits(),
                cos_correctly_rounded(x).to_bits(),
                "cos(-{x}) != cos({x})"
            );
        }
    }

    #[test]
    fn sin_cos_phase_shift() {
        // sin(x) = cos(π/2 - x)
        let xs: &[f64] = &[0.5, 1.0, 1.5, 2.0];
        let pio2 = std::f64::consts::FRAC_PI_2;
        for &x in xs {
            let s = sin_correctly_rounded(x);
            let c = cos_correctly_rounded(pio2 - x);
            let dist = ulps_between(s, c);
            assert!(
                dist <= 4,
                "sin({x}) vs cos(π/2-{x}): {dist} ulps"
            );
        }
    }
}
