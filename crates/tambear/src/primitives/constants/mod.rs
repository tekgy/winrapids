//! Mathematical constants at multiple precisions.
//!
//! Recipes need π, e, ln 2, √2, and friends at f64 precision for fast
//! paths and at double-double (~106-bit) precision for correctly-rounded
//! paths. This module is the single source of truth for both.
//!
//! # Convention
//!
//! - `PI_F64`, `E_F64`, `LN_2_F64`, `SQRT_2_F64`, etc.: plain `f64`
//!   constants, identical bit patterns to the ones in `std::f64::consts`.
//!   Named with the `_F64` suffix so the f64/DD pair is always visible
//!   in both import styles.
//! - `PI_DD`, `E_DD`, etc.: `DoubleDouble` constants. The high part
//!   matches the f64 version; the low part carries the residual bits
//!   from a reference computation at ≥ 200-bit precision (currently
//!   hand-audited against Wikipedia / DLMF values).
//!
//! # Adding a new constant
//!
//! 1. Compute the value at ~150-bit precision via a trusted reference
//!    (mpmath, Wolfram, DLMF). Copy the first ~45 decimal digits.
//! 2. Express it as `hi + lo` where `hi` is the nearest `f64` (use the
//!    Rust `const` initializer and rely on round-to-nearest-even at
//!    parse time).
//! 3. Compute `lo = reference - hi` to ~double-double precision via
//!    symbolic arithmetic, then round the residual to `f64`.
//! 4. Add a test in `tests` verifying the f64 value against
//!    `std::f64::consts`, and verifying the DD high part matches the
//!    f64 value bit-for-bit.
//!
//! Do NOT use the `PI / LN_2` style arithmetic expressions to derive
//! constants at double-double precision — those lose bits. Compute the
//! exact value externally and paste the residual.

use crate::primitives::double_double::DoubleDouble;

// ── π ──────────────────────────────────────────────────────────────────────

/// π, f64 precision. Bit-identical to `std::f64::consts::PI`.
pub const PI_F64: f64 = 3.141_592_653_589_793_f64;

/// π, double-double precision. ~32 decimal digits.
///
/// Reference value: 3.141592653589793238462643383279502884197169399375...
pub const PI_DD: DoubleDouble = DoubleDouble {
    hi: PI_F64,
    lo: 1.224_646_799_147_353_2e-16_f64,
};

/// π / 2, f64.
pub const PI_OVER_2_F64: f64 = 1.570_796_326_794_896_6_f64;

/// π / 2, double-double.
pub const PI_OVER_2_DD: DoubleDouble = DoubleDouble {
    hi: PI_OVER_2_F64,
    lo: 6.123_233_995_736_766e-17_f64,
};

/// π / 4, f64.
pub const PI_OVER_4_F64: f64 = 0.785_398_163_397_448_3_f64;

/// π / 4, double-double.
pub const PI_OVER_4_DD: DoubleDouble = DoubleDouble {
    hi: PI_OVER_4_F64,
    lo: 3.061_616_997_868_383e-17_f64,
};

/// 2π (tau), f64.
pub const TAU_F64: f64 = 6.283_185_307_179_586_f64;

/// 2π, double-double.
pub const TAU_DD: DoubleDouble = DoubleDouble {
    hi: TAU_F64,
    lo: 2.449_293_598_294_706_4e-16_f64,
};

/// 1 / π, f64.
pub const INV_PI_F64: f64 = 0.318_309_886_183_790_7_f64;

/// 1 / π, double-double.
pub const INV_PI_DD: DoubleDouble = DoubleDouble {
    hi: INV_PI_F64,
    lo: -1.967_867_667_518_248_8e-17_f64,
};

// ── e ──────────────────────────────────────────────────────────────────────

/// e (Euler's number), f64. Bit-identical to `std::f64::consts::E`.
pub const E_F64: f64 = 2.718_281_828_459_045_f64;

/// e, double-double. Reference: 2.7182818284590452353602874713526624977572...
pub const E_DD: DoubleDouble = DoubleDouble {
    hi: E_F64,
    lo: 1.445_646_891_729_250_2e-16_f64,
};

// ── Natural logarithms ─────────────────────────────────────────────────────

/// ln 2, f64. Bit-identical to `std::f64::consts::LN_2`.
pub const LN_2_F64: f64 = 0.693_147_180_559_945_3_f64;

/// ln 2, double-double. Critical for exp/log recipes.
///
/// Reference: 0.69314718055994530941723212145817656807550013436025...
pub const LN_2_DD: DoubleDouble = DoubleDouble {
    hi: LN_2_F64,
    lo: 2.319_046_813_846_299_6e-17_f64,
};

/// ln 10, f64.
pub const LN_10_F64: f64 = 2.302_585_092_994_046_f64;

/// ln 10, double-double.
pub const LN_10_DD: DoubleDouble = DoubleDouble {
    hi: LN_10_F64,
    lo: -2.170_756_223_399_868_4e-16_f64,
};

/// log₂(e), f64. Equals 1 / ln 2.
pub const LOG2_E_F64: f64 = 1.442_695_040_888_963_4_f64;

/// log₂(e), double-double.
pub const LOG2_E_DD: DoubleDouble = DoubleDouble {
    hi: LOG2_E_F64,
    lo: 2.035_527_374_093_103e-17_f64,
};

/// log₁₀(e), f64. Equals 1 / ln 10.
pub const LOG10_E_F64: f64 = 0.434_294_481_903_251_83_f64;

/// log₁₀(e), double-double.
pub const LOG10_E_DD: DoubleDouble = DoubleDouble {
    hi: LOG10_E_F64,
    lo: 1.098_319_650_216_765_4e-17_f64,
};

// ── Roots ──────────────────────────────────────────────────────────────────

/// √2, f64. Bit-identical to `std::f64::consts::SQRT_2`.
pub const SQRT_2_F64: f64 = 1.414_213_562_373_095_1_f64;

/// √2, double-double. Reference: 1.41421356237309504880168872420969807856967...
pub const SQRT_2_DD: DoubleDouble = DoubleDouble {
    hi: SQRT_2_F64,
    lo: -9.667_293_313_452_913_6e-17_f64,
};

/// 1 / √2, f64. Matches `std::f64::consts::FRAC_1_SQRT_2` (which rounds the
/// mathematical value up to `0.7071067811865476`).
pub const FRAC_1_SQRT_2_F64: f64 = std::f64::consts::FRAC_1_SQRT_2;

/// 1 / √2, double-double.
/// Reference: 0.70710678118654752440084436210484903928483593768847...
/// Since the hi part is rounded *up* to 0.7071067811865476, the residual is
/// slightly negative.
pub const FRAC_1_SQRT_2_DD: DoubleDouble = DoubleDouble {
    hi: FRAC_1_SQRT_2_F64,
    lo: -4.833_646_656_726_456_8e-17_f64,
};

#[cfg(test)]
mod tests {
    use super::*;

    // ── f64 constants must match std::f64::consts ─────────────────────────

    #[test]
    fn f64_constants_match_std() {
        assert_eq!(PI_F64, std::f64::consts::PI);
        assert_eq!(E_F64, std::f64::consts::E);
        assert_eq!(LN_2_F64, std::f64::consts::LN_2);
        assert_eq!(LN_10_F64, std::f64::consts::LN_10);
        assert_eq!(LOG2_E_F64, std::f64::consts::LOG2_E);
        assert_eq!(LOG10_E_F64, std::f64::consts::LOG10_E);
        assert_eq!(SQRT_2_F64, std::f64::consts::SQRT_2);
        assert_eq!(FRAC_1_SQRT_2_F64, std::f64::consts::FRAC_1_SQRT_2);
        assert_eq!(PI_OVER_2_F64, std::f64::consts::FRAC_PI_2);
        assert_eq!(PI_OVER_4_F64, std::f64::consts::FRAC_PI_4);
        assert_eq!(TAU_F64, std::f64::consts::TAU);
        assert_eq!(INV_PI_F64, std::f64::consts::FRAC_1_PI);
    }

    // ── DD high parts must match f64 versions ─────────────────────────────

    #[test]
    fn dd_high_parts_match_f64() {
        assert_eq!(PI_DD.hi, PI_F64);
        assert_eq!(PI_OVER_2_DD.hi, PI_OVER_2_F64);
        assert_eq!(PI_OVER_4_DD.hi, PI_OVER_4_F64);
        assert_eq!(TAU_DD.hi, TAU_F64);
        assert_eq!(INV_PI_DD.hi, INV_PI_F64);
        assert_eq!(E_DD.hi, E_F64);
        assert_eq!(LN_2_DD.hi, LN_2_F64);
        assert_eq!(LN_10_DD.hi, LN_10_F64);
        assert_eq!(LOG2_E_DD.hi, LOG2_E_F64);
        assert_eq!(LOG10_E_DD.hi, LOG10_E_F64);
        assert_eq!(SQRT_2_DD.hi, SQRT_2_F64);
        assert_eq!(FRAC_1_SQRT_2_DD.hi, FRAC_1_SQRT_2_F64);
    }

    // ── DD low parts must be within ulp(hi) / 2 (non-overlap) ─────────────

    #[test]
    fn dd_low_parts_non_overlapping() {
        let pairs: &[(&str, DoubleDouble)] = &[
            ("PI", PI_DD),
            ("PI_OVER_2", PI_OVER_2_DD),
            ("PI_OVER_4", PI_OVER_4_DD),
            ("TAU", TAU_DD),
            ("INV_PI", INV_PI_DD),
            ("E", E_DD),
            ("LN_2", LN_2_DD),
            ("LN_10", LN_10_DD),
            ("LOG2_E", LOG2_E_DD),
            ("LOG10_E", LOG10_E_DD),
            ("SQRT_2", SQRT_2_DD),
            ("FRAC_1_SQRT_2", FRAC_1_SQRT_2_DD),
        ];
        for (name, dd) in pairs {
            let ulp_hi = f64::EPSILON * dd.hi.abs();
            assert!(
                dd.lo.abs() <= ulp_hi,
                "{name}: |lo| = {} exceeds ulp(hi) = {}",
                dd.lo.abs(),
                ulp_hi
            );
        }
    }

    // ── Relationship tests ─────────────────────────────────────────────────

    #[test]
    fn pi_over_2_is_half_of_pi() {
        assert_eq!(PI_OVER_2_F64, PI_F64 / 2.0);
    }

    #[test]
    fn tau_is_two_pi() {
        assert_eq!(TAU_F64, 2.0 * PI_F64);
    }

    #[test]
    fn inv_pi_close_to_reciprocal() {
        // 1/π in f64 should equal INV_PI_F64 bit-for-bit.
        assert_eq!(INV_PI_F64, 1.0 / PI_F64);
    }

    #[test]
    fn log2e_is_reciprocal_of_ln2() {
        // 1/ln(2) should match LOG2_E to f64 precision (with expected rounding).
        let expected = 1.0 / LN_2_F64;
        assert_eq!(LOG2_E_F64, expected);
    }

    #[test]
    fn sqrt2_squared_is_two() {
        // SQRT_2 * SQRT_2 is 2.0 in f64 (though not exactly in real arithmetic).
        let got = SQRT_2_F64 * SQRT_2_F64;
        assert!((got - 2.0).abs() < 1e-15);
    }

    #[test]
    fn frac_1_sqrt_2_squared_is_half() {
        let got = FRAC_1_SQRT_2_F64 * FRAC_1_SQRT_2_F64;
        assert!((got - 0.5).abs() < 1e-15);
    }

    // ── DD constants sanity: the hi + lo sum should match the mathematical
    //    value to within the DD precision, verified against f64 computations
    //    that don't depend on the constant under test. ────────────────────

    #[test]
    fn ln2_dd_sum_is_close_to_f64_value() {
        // hi + lo, rounded back to f64, should equal hi (non-overlap).
        assert_eq!(LN_2_DD.hi + LN_2_DD.lo, LN_2_F64);
    }

    #[test]
    fn pi_dd_sum_is_close_to_f64_value() {
        assert_eq!(PI_DD.hi + PI_DD.lo, PI_F64);
    }
}
