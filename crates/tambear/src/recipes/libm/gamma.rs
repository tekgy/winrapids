//! `lgamma(x)` and `tgamma(x)` — log-gamma and gamma functions.
//!
//! # Mathematical recipe
//!
//! `Γ(x) = ∫₀^∞ t^(x-1) · e^(-t) dt` for x > 0.
//!
//! For computation we use Stirling's approximation with Lanczos correction:
//!
//! ```text
//! Γ(x) ≈ √(2π) · (x + g - 0.5)^(x - 0.5) · e^(-(x + g - 0.5)) · Σ aₖ/(x+k)
//! ```
//!
//! where `g` is the Lanczos parameter and `aₖ` are precomputed coefficients.
//! We use the Lanczos approximation with `g = 7` and 9 coefficients, which
//! gives ~15 digits of accuracy across the positive real line.
//!
//! `lgamma(x) = ln|Γ(x)|` is computed directly from the Lanczos formula
//! without computing Γ(x) first, avoiding overflow for large x.
//!
//! # Special cases
//!
//! - `tgamma(NaN) = NaN`, `lgamma(NaN) = NaN`
//! - `tgamma(+∞) = +∞`, `lgamma(+∞) = +∞`
//! - `tgamma(0) = ±∞` (sign matches sign of zero)
//! - `tgamma(negative integer) = NaN` (pole)
//! - `tgamma(x)` for large x overflows to ±∞
//! - `lgamma(x)` for negative non-integer x uses the reflection formula
//!
//! # References
//!
//! - Lanczos, "A Precision Approximation of the Gamma Function" (1964)
//! - Pugh, "An Analysis of the Lanczos Gamma Approximation" (2004) — our
//!   coefficient source
//! - Numerical Recipes ch. 6.1 for the Stirling/Lanczos implementation pattern

use crate::primitives::constants::PI_F64;

/// Lanczos parameter g = 7.
const LANCZOS_G: f64 = 7.0;

/// Lanczos coefficients for g=7, n=9. From Pugh (2004), table of
/// coefficients for the rational approximation. These give ~15 digits.
const LANCZOS_COEFFS: [f64; 9] = [
    0.999_999_999_999_809_93,
    676.520_368_121_885_10,
    -1259.139_216_722_402_87,
    771.323_428_777_653_08,
    -176.615_029_162_140_60,
    12.507_343_278_686_905,
    -0.138_571_095_265_720_12,
    9.984_369_578_019_572e-6,
    1.505_632_735_149_312e-7,
];

/// `lgamma(x)` — log of the absolute value of the gamma function.
///
/// Computed directly from the Lanczos approximation without computing
/// Γ(x) first, so it doesn't overflow for large x.
#[inline]
pub fn lgamma_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return f64::INFINITY;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }

    if x < 0.5 {
        // Pole check: Γ has poles at non-positive integers.
        if x < 0.0 && x == x.floor() {
            return f64::INFINITY;
        }
        // Reflection formula: Γ(x)·Γ(1-x) = π/sin(πx)
        // lgamma(x) = ln(π/sin(πx)) - lgamma(1-x)
        let sin_pi_x = (PI_F64 * x).sin().abs();
        if sin_pi_x < 1e-300 {
            return f64::INFINITY;
        }
        (PI_F64 / sin_pi_x).ln() - lgamma_positive(1.0 - x)
    } else {
        lgamma_positive(x)
    }
}

/// `lgamma(x)` — compensated (same as strict for now).
#[inline]
pub fn lgamma_compensated(x: f64) -> f64 {
    lgamma_strict(x)
}

/// `lgamma(x)` — correctly-rounded (same as strict for now).
#[inline]
pub fn lgamma_correctly_rounded(x: f64) -> f64 {
    lgamma_strict(x)
}

/// `tgamma(x)` — the gamma function itself.
///
/// Computed as `exp(lgamma(x))` with sign correction. Overflows to ±∞
/// for large |x|.
#[inline]
pub fn tgamma_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x == f64::INFINITY {
        return f64::INFINITY;
    }
    if x == f64::NEG_INFINITY {
        return f64::NAN;
    }
    if x == 0.0 {
        // Γ(+0) = +∞, Γ(-0) = -∞
        return if x.is_sign_positive() {
            f64::INFINITY
        } else {
            f64::NEG_INFINITY
        };
    }

    // Check for negative integers (poles).
    if x < 0.0 && x == x.floor() {
        return f64::NAN;
    }

    // For positive half-plane, compute directly via Lanczos.
    if x >= 0.5 {
        let lg = lgamma_positive(x);
        if lg > 709.0 {
            return f64::INFINITY;
        }
        return lg.exp();
    }

    // For x < 0.5, use reflection: Γ(x) = π / (sin(πx) · Γ(1-x))
    let sin_pi_x = (PI_F64 * x).sin();
    if sin_pi_x.abs() < 1e-300 {
        return f64::NAN; // near a pole
    }
    let gamma_1mx = {
        let lg = lgamma_positive(1.0 - x);
        if lg > 709.0 {
            return 0.0; // Γ(1-x) huge → Γ(x) tiny
        }
        lg.exp()
    };
    PI_F64 / (sin_pi_x * gamma_1mx)
}

/// `tgamma(x)` — compensated (same as strict for now).
#[inline]
pub fn tgamma_compensated(x: f64) -> f64 {
    tgamma_strict(x)
}

/// `tgamma(x)` — correctly-rounded (same as strict for now).
#[inline]
pub fn tgamma_correctly_rounded(x: f64) -> f64 {
    tgamma_strict(x)
}

// ── Internal ────────────────────────────────────────────────────────────────

/// Lanczos lgamma for x ≥ 0.5.
///
/// `lgamma(x) = 0.5·ln(2π) + (x - 0.5)·ln(x + g - 0.5) - (x + g - 0.5) + ln(Ag(x))`
///
/// where `Ag(x) = c₀ + c₁/(x) + c₂/(x+1) + ... + c₈/(x+7)`.
#[inline]
fn lgamma_positive(x: f64) -> f64 {
    let half_ln_2pi = 0.918_938_533_204_672_7; // 0.5 * ln(2π)
    let t = x + LANCZOS_G - 0.5;

    // Ag(x) = c₀ + Σ cₖ / (x + k - 1) for k = 1..8
    let mut ag = LANCZOS_COEFFS[0];
    for k in 1..9_usize {
        ag += LANCZOS_COEFFS[k] / (x + (k as f64) - 1.0);
    }

    half_ln_2pi + (x - 0.5) * t.ln() - t + ag.ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::{assert_within_ulps, ulps_between};

    // ── lgamma boundary tests ──────────────────────────────────────────

    #[test]
    fn lgamma_of_nan_is_nan() {
        assert!(lgamma_strict(f64::NAN).is_nan());
    }

    #[test]
    fn lgamma_of_inf_is_inf() {
        assert_eq!(lgamma_strict(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn lgamma_of_zero_is_inf() {
        assert_eq!(lgamma_strict(0.0), f64::INFINITY);
    }

    #[test]
    fn lgamma_at_negative_integer_is_inf() {
        // Γ has poles at non-positive integers → lgamma = +∞
        assert_eq!(lgamma_strict(-1.0), f64::INFINITY);
        assert_eq!(lgamma_strict(-2.0), f64::INFINITY);
    }

    // ── lgamma known values ────────────────────────────────────────────

    #[test]
    fn lgamma_of_one_is_zero() {
        // Γ(1) = 0! = 1, so lgamma(1) = ln(1) = 0.
        let got = lgamma_strict(1.0);
        assert!(got.abs() < 1e-13, "lgamma(1) = {got}, expected 0");
    }

    #[test]
    fn lgamma_of_two_is_zero() {
        // Γ(2) = 1! = 1, so lgamma(2) = 0.
        let got = lgamma_strict(2.0);
        assert!(got.abs() < 1e-13, "lgamma(2) = {got}, expected 0");
    }

    #[test]
    fn lgamma_at_integer_matches_factorial() {
        // Γ(n) = (n-1)!, so lgamma(n) = ln((n-1)!)
        let factorials: &[(f64, f64)] = &[
            (3.0, 2.0_f64.ln()),          // Γ(3) = 2! = 2
            (4.0, 6.0_f64.ln()),          // Γ(4) = 3! = 6
            (5.0, 24.0_f64.ln()),         // Γ(5) = 4! = 24
            (6.0, 120.0_f64.ln()),        // Γ(6) = 5! = 120
            (7.0, 720.0_f64.ln()),        // Γ(7) = 6! = 720
            (10.0, 362880.0_f64.ln()),    // Γ(10) = 9! = 362880
        ];
        for &(n, expected) in factorials {
            let got = lgamma_strict(n);
            let dist = ulps_between(got, expected);
            assert!(
                dist <= 100,
                "lgamma({n}) = {got}, expected {expected}, {dist} ulps"
            );
        }
    }

    #[test]
    fn lgamma_half_is_half_ln_pi() {
        // Γ(1/2) = √π, so lgamma(0.5) = 0.5 · ln(π)
        let expected = 0.5 * PI_F64.ln();
        let got = lgamma_strict(0.5);
        let dist = ulps_between(got, expected);
        assert!(
            dist <= 100,
            "lgamma(0.5) = {got}, expected {expected}, {dist} ulps"
        );
    }

    // ── tgamma boundary tests ──────────────────────────────────────────

    #[test]
    fn tgamma_of_nan_is_nan() {
        assert!(tgamma_strict(f64::NAN).is_nan());
    }

    #[test]
    fn tgamma_of_pos_inf_is_inf() {
        assert_eq!(tgamma_strict(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn tgamma_of_neg_inf_is_nan() {
        assert!(tgamma_strict(f64::NEG_INFINITY).is_nan());
    }

    #[test]
    fn tgamma_at_pole_is_nan() {
        assert!(tgamma_strict(-1.0).is_nan());
        assert!(tgamma_strict(-2.0).is_nan());
        assert!(tgamma_strict(-3.0).is_nan());
    }

    #[test]
    fn tgamma_of_zero_is_inf() {
        assert_eq!(tgamma_strict(0.0), f64::INFINITY);
        // Γ(-0) = -∞
        let neg = tgamma_strict(-0.0);
        assert_eq!(neg, f64::NEG_INFINITY);
    }

    // ── tgamma known values ────────────────────────────────────────────

    #[test]
    fn tgamma_at_integers_is_factorial() {
        let cases: &[(f64, f64)] = &[
            (1.0, 1.0),    // 0!
            (2.0, 1.0),    // 1!
            (3.0, 2.0),    // 2!
            (4.0, 6.0),    // 3!
            (5.0, 24.0),   // 4!
            (6.0, 120.0),  // 5!
            (7.0, 720.0),  // 6!
        ];
        for &(n, expected) in cases {
            let got = tgamma_strict(n);
            let dist = ulps_between(got, expected);
            assert!(
                dist <= 100,
                "tgamma({n}) = {got}, expected {expected}, {dist} ulps"
            );
        }
    }

    #[test]
    fn tgamma_half_is_sqrt_pi() {
        let expected = PI_F64.sqrt();
        let got = tgamma_strict(0.5);
        let dist = ulps_between(got, expected);
        assert!(
            dist <= 100,
            "tgamma(0.5) = {got}, expected {expected}, {dist} ulps"
        );
    }

    #[test]
    fn tgamma_negative_half() {
        // Γ(-0.5) = -2√π
        let expected = -2.0 * PI_F64.sqrt();
        let got = tgamma_strict(-0.5);
        let err = (got - expected).abs() / expected.abs();
        assert!(
            err < 1e-10,
            "tgamma(-0.5) = {got}, expected {expected}, rel err {err:e}"
        );
    }

    // ── Recurrence relation ────────────────────────────────────────────

    #[test]
    fn tgamma_recurrence_relation() {
        // Γ(x+1) = x · Γ(x)
        let xs: &[f64] = &[0.5, 1.5, 2.5, 3.5, 4.5, 7.3];
        for &x in xs {
            let lhs = tgamma_strict(x + 1.0);
            let rhs = x * tgamma_strict(x);
            let dist = ulps_between(lhs, rhs);
            assert!(
                dist <= 200,
                "Γ({x}+1) = {lhs}, {x}·Γ({x}) = {rhs}, {dist} ulps"
            );
        }
    }

    // ── lgamma reflection formula ──────────────────────────────────────

    #[test]
    fn lgamma_reflection_formula() {
        // lgamma(x) + lgamma(1-x) = ln(π/sin(πx)) for non-integer x
        let xs: &[f64] = &[0.25, 0.3, 0.75, -0.5, -1.5];
        for &x in xs {
            let lhs = lgamma_strict(x) + lgamma_strict(1.0 - x);
            let sin_pix = (PI_F64 * x).sin().abs();
            let rhs = (PI_F64 / sin_pix).ln();
            let err = (lhs - rhs).abs();
            assert!(
                err < 1e-8,
                "lgamma reflection at x={x}: lhs={lhs}, rhs={rhs}, err={err:e}"
            );
        }
    }
}
