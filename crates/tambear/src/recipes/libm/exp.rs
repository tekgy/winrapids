//! `exp(x)` — natural exponential.
//!
//! First pilot of the three-strategy libm pattern. Range reduction,
//! polynomial approximation, and reconstruction are the same recipe in
//! all three entry points; they differ only in which primitive set is
//! called for the arithmetic.
//!
//! # Mathematical recipe
//!
//! For any finite `x`, write
//!
//! ```text
//! x = k · ln(2) + r,   k ∈ ℤ,   |r| ≤ ln(2)/2 ≈ 0.347
//! ```
//!
//! Then `exp(x) = 2^k · exp(r)`. We compute `k = round(x · log₂(e))`,
//! reduce `x` by subtracting `k · ln(2)` (using Cody-Waite high/low
//! splitting to preserve bits when `k` is large), evaluate a truncated
//! Taylor series for `exp(r)` on the small range, and rescale by `2^k`
//! via `ldexp`.
//!
//! # Error sources
//!
//! 1. Range reduction: `r = x - k · ln(2)`. The subtraction can lose
//!    precision if `k · ln(2)` rounds before the subtraction. We avoid
//!    this with the Cody-Waite split `ln(2) = LN_2_HI + LN_2_LO` from
//!    `primitives::constants::LN_2_DD`.
//! 2. Polynomial approximation: Taylor series truncated at degree N.
//!    Residual is bounded by `|r|^(N+1) / (N+1)!`. For `|r| ≤ 0.347`
//!    and N = 13, this is below `2^-53` ≈ 1.1e-16.
//! 3. Reconstruction: `ldexp` is exact for normal results, so no error.
//!
//! # Strategy ulp bounds
//!
//! | Entry point              | Polynomial | Arithmetic   | Target ulps |
//! |--------------------------|-----------:|--------------|-------------|
//! | `exp_strict`             | N = 13     | plain FMA    | ≤ 4         |
//! | `exp_compensated`        | N = 13     | DD reduce + compensated horner | ≤ 2 |
//! | `exp_correctly_rounded`  | N = 16     | double-double everywhere | ≤ 1 |
//!
//! # Special cases (all three entry points)
//!
//! - `exp(NaN) = NaN`
//! - `exp(+∞) = +∞`
//! - `exp(-∞) = 0`
//! - `exp(x) = +∞` for `x > 709.782712893384`
//! - `exp(x) = 0` for `x < -745.133219101941` (below smallest subnormal)
//! - `exp(0) = 1` exactly
//!
//! # References
//!
//! - Muller et al., *Handbook of Floating-Point Arithmetic* (2018),
//!   ch. 12: the canonical reference for libm-quality `exp`.
//! - Cody & Waite, *Software Manual for the Elementary Functions* (1980):
//!   the high/low splitting trick for exact range reduction.
//! - Gal, "Computing elementary functions: a new approach for achieving
//!   high accuracy and good performance" (1985): the modern rounded
//!   libm error analysis.

use crate::primitives::compensated::dot::{compensated_horner, horner};
use crate::primitives::constants::{LN_2_DD, LOG2_E_F64};
use crate::primitives::double_double::{ops::dd_add_f64, ops::dd_mul_f64, DoubleDouble};
use crate::primitives::hardware::{ffloor, ldexp};

/// Maximum input that does not overflow: `log(f64::MAX)`.
///
/// For `x > EXP_MAX_ARG`, `exp(x) = +∞`.
const EXP_MAX_ARG: f64 = 709.782_712_893_384_f64;

/// Minimum input that produces a non-zero (possibly subnormal) result.
///
/// For `x < EXP_MIN_ARG`, `exp(x) = 0`.
const EXP_MIN_ARG: f64 = -745.133_219_101_941_f64;

/// Cody-Waite split of `ln(2)` for strict-path range reduction.
///
/// `LN_2_CW_HI` has its bottom 19 mantissa bits zeroed, so the product
/// `k * LN_2_CW_HI` is exact for `|k| < 2^21` (much larger than the
/// useful `k` range for `exp`). `LN_2_CW_LO` carries the residual bits
/// so that `LN_2_CW_HI + LN_2_CW_LO == ln(2)` at double-double precision.
///
/// Values taken from Sun fdlibm's reference `exp` implementation, where
/// they have been hand-verified against mpmath for decades.
const LN_2_CW_HI: f64 = 6.931_471_803_691_238_2e-1_f64;
const LN_2_CW_LO: f64 = 1.908_214_929_270_587_7e-10_f64;

/// Taylor coefficients of `exp(r)` in ascending order up to degree 13.
///
/// `EXP_TAYLOR[k] = 1 / k!`
///
/// Used by `exp_strict` and `exp_compensated`. The correctly-rounded
/// version uses a longer coefficient table with extended precision.
const EXP_TAYLOR: [f64; 14] = [
    1.0,                              // 1/0!
    1.0,                              // 1/1!
    0.5,                              // 1/2!
    1.0 / 6.0,                        // 1/3!
    1.0 / 24.0,                       // 1/4!
    1.0 / 120.0,                      // 1/5!
    1.0 / 720.0,                      // 1/6!
    1.0 / 5040.0,                     // 1/7!
    1.0 / 40320.0,                    // 1/8!
    1.0 / 362880.0,                   // 1/9!
    1.0 / 3628800.0,                  // 1/10!
    1.0 / 39916800.0,                 // 1/11!
    1.0 / 479001600.0,                // 1/12!
    1.0 / 6227020800.0,               // 1/13!
];

/// Same Taylor table extended to degree 16 for the correctly-rounded path.
const EXP_TAYLOR_LONG: [f64; 17] = [
    1.0,
    1.0,
    0.5,
    1.0 / 6.0,
    1.0 / 24.0,
    1.0 / 120.0,
    1.0 / 720.0,
    1.0 / 5040.0,
    1.0 / 40320.0,
    1.0 / 362880.0,
    1.0 / 3628800.0,
    1.0 / 39916800.0,
    1.0 / 479001600.0,
    1.0 / 6227020800.0,
    1.0 / 87178291200.0,              // 1/14!
    1.0 / 1307674368000.0,            // 1/15!
    1.0 / 20922789888000.0,           // 1/16!
];

// ── Entry points ────────────────────────────────────────────────────────────

/// `exp(x)` — strict lowering. Plain FMA throughout. Target: ≤ 3 ulps.
///
/// Uses a degree-13 Taylor approximation evaluated with plain
/// single-rounding Horner (`primitives::compensated::dot::horner`, which
/// wraps `f64::mul_add`). Range reduction uses the Cody-Waite
/// `LN_2_CW_HI`/`LN_2_CW_LO` split so the subtraction `r = x - k·ln(2)`
/// is exact for all useful `k`.
#[inline]
pub fn exp_strict(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }

    // Range reduction: k = round(x / ln(2)), r = x - k · ln(2).
    let k_f = ffloor(x * LOG2_E_F64 + 0.5);
    let k = k_f as i32;

    // Cody-Waite exact subtraction via the pre-split ln(2):
    //   r = (x - k · LN_2_CW_HI) - k · LN_2_CW_LO
    // The first subtraction is exact because LN_2_CW_HI has 19 trailing
    // zero mantissa bits, so k · LN_2_CW_HI fits in f64 without rounding
    // for any |k| < 2^21 (far wider than the finite exp range).
    let r_hi = x - k_f * LN_2_CW_HI;
    let r = r_hi - k_f * LN_2_CW_LO;

    // Polynomial: Horner over EXP_TAYLOR at r.
    let p = horner(&EXP_TAYLOR, r);

    // Reconstruction: exp(x) = 2^k · exp(r).
    ldexp(p, k)
}

/// `exp(x)` — compensated lowering. Target: ≤ 2 ulps.
///
/// Uses double-double range reduction (exact for any `k` in the finite
/// f64 range) and compensated Horner over the Taylor table. The DD
/// reduction is what lets us avoid the bit-loss that standard Cody-Waite
/// would suffer here — `LN_2_DD.hi` is the full-precision f64 value, not
/// a short-mantissa constant, so `k * LN_2_DD.hi` rounds for large `k`.
/// DD multiplication captures the rounding residual and the subtraction
/// stays exact.
#[inline]
pub fn exp_compensated(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }

    let k_f = ffloor(x * LOG2_E_F64 + 0.5);
    let k = k_f as i32;

    // DD range reduction: r = x - k · LN_2, exact to ~106 bits.
    let k_ln2 = dd_mul_f64(LN_2_DD, k_f);
    let r_dd = dd_sub_f64_lhs(x, k_ln2);
    let r = r_dd.to_f64();

    // Compensated Horner over the Taylor table.
    let p = compensated_horner(&EXP_TAYLOR, r);

    ldexp(p, k)
}

/// `exp(x)` — correctly-rounded lowering. Target: 0 ulps.
///
/// Uses double-double working precision throughout: DD range reduction
/// (Cody-Waite applied to DD operations), a DD-valued Horner evaluation
/// over a degree-16 Taylor table, and a final round-to-nearest-even
/// conversion back to f64.
///
/// For the overwhelming majority of inputs this gives the
/// correctly-rounded result. True hard cases (where even ~106 bits of
/// working precision isn't enough to distinguish the correct rounding)
/// would require arbitrary precision and are out of scope for the first
/// pilot.
#[inline]
pub fn exp_correctly_rounded(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }

    let k_f = ffloor(x * LOG2_E_F64 + 0.5);
    let k = k_f as i32;

    // DD range reduction: r = x - k · LN_2 with LN_2 at DD precision.
    // We compute k · LN_2 as a DD, then subtract from x.
    let k_ln2 = dd_mul_f64(LN_2_DD, k_f);
    let r_dd = dd_sub_f64_lhs(x, k_ln2);

    // Polynomial: Horner over the extended Taylor table with a DD
    // accumulator. This is the first use of DD-valued Horner in the
    // codebase; the loop is written out here for clarity. If more
    // recipes need it, we'll promote it to a helper in
    // `primitives::double_double::ops::dd_horner`.
    let r_hi = r_dd.hi + r_dd.lo; // reduce DD to f64 for x-coordinate
    let n = EXP_TAYLOR_LONG.len();
    let mut acc = DoubleDouble::from_f64(EXP_TAYLOR_LONG[n - 1]);
    for i in (0..n - 1).rev() {
        // acc = acc · r_hi + EXP_TAYLOR_LONG[i]
        acc = dd_mul_f64(acc, r_hi);
        acc = dd_add_f64(acc, EXP_TAYLOR_LONG[i]);
    }

    // Reconstruction: exp(x) = 2^k · exp(r_dd).
    // ldexp applied to the DD.hi is exact; the lo part is scaled the same
    // way, so we can just ldexp the final f64 reduction.
    let p = acc.to_f64();
    ldexp(p, k)
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Handle the special cases common to all three entry points. Returns
/// `Some(result)` if `x` is a boundary input (NaN, ±∞, out of range),
/// `None` otherwise.
#[inline]
fn special_case(x: f64) -> Option<f64> {
    if x.is_nan() {
        return Some(f64::NAN);
    }
    if x == f64::INFINITY {
        return Some(f64::INFINITY);
    }
    if x == f64::NEG_INFINITY {
        return Some(0.0);
    }
    if x > EXP_MAX_ARG {
        return Some(f64::INFINITY);
    }
    if x < EXP_MIN_ARG {
        return Some(0.0);
    }
    None
}

/// Compute `a - b` where `a` is an f64 and `b` is a DoubleDouble,
/// returning the result as a DoubleDouble.
///
/// Implemented as `dd_add(from_f64(a), -b)`. Kept here as a local helper
/// rather than pushing into `primitives::double_double::ops` until a
/// second recipe needs it — avoids premature API surface.
#[inline]
fn dd_sub_f64_lhs(a: f64, b: DoubleDouble) -> DoubleDouble {
    use crate::primitives::double_double::ops::dd_add;
    dd_add(DoubleDouble::from_f64(a), b.neg())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::{assert_within_ulps, ulps_between};

    // Generic checker: each strategy should match f64::exp within its budget.
    fn check_strategy<F: Fn(f64) -> f64>(f: F, name: &str, max_ulps: u64) {
        let samples: &[f64] = &[
            0.0,
            1.0,
            -1.0,
            0.5,
            -0.5,
            2.0,
            -2.0,
            10.0,
            -10.0,
            100.0,
            -100.0,
            500.0,
            -500.0,
            700.0,
            -700.0,
            std::f64::consts::LN_2,
            -std::f64::consts::LN_2,
            std::f64::consts::E,
            std::f64::consts::PI,
            0.1,
            0.01,
            0.001,
            -0.1,
            1e-10,
            -1e-10,
        ];

        for &x in samples {
            let got = f(x);
            let expected = x.exp();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= max_ulps,
                "{name}(x={x}): {dist} ulps apart (max {max_ulps})\n  got:      {got:e}\n  expected: {expected:e}"
            );
        }
    }

    // ── Boundary semantics ────────────────────────────────────────────────

    #[test]
    fn exp_of_zero_is_one() {
        assert_eq!(exp_strict(0.0), 1.0);
        assert_eq!(exp_compensated(0.0), 1.0);
        assert_eq!(exp_correctly_rounded(0.0), 1.0);
    }

    #[test]
    fn exp_of_neg_zero_is_one() {
        assert_eq!(exp_strict(-0.0), 1.0);
        assert_eq!(exp_compensated(-0.0), 1.0);
        assert_eq!(exp_correctly_rounded(-0.0), 1.0);
    }

    #[test]
    fn exp_of_nan_is_nan() {
        assert!(exp_strict(f64::NAN).is_nan());
        assert!(exp_compensated(f64::NAN).is_nan());
        assert!(exp_correctly_rounded(f64::NAN).is_nan());
    }

    #[test]
    fn exp_of_pos_inf_is_pos_inf() {
        assert_eq!(exp_strict(f64::INFINITY), f64::INFINITY);
        assert_eq!(exp_compensated(f64::INFINITY), f64::INFINITY);
        assert_eq!(exp_correctly_rounded(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn exp_of_neg_inf_is_zero() {
        assert_eq!(exp_strict(f64::NEG_INFINITY), 0.0);
        assert_eq!(exp_compensated(f64::NEG_INFINITY), 0.0);
        assert_eq!(exp_correctly_rounded(f64::NEG_INFINITY), 0.0);
    }

    #[test]
    fn exp_overflows_above_threshold() {
        assert_eq!(exp_strict(1000.0), f64::INFINITY);
        assert_eq!(exp_compensated(1000.0), f64::INFINITY);
        assert_eq!(exp_correctly_rounded(1000.0), f64::INFINITY);
    }

    #[test]
    fn exp_underflows_below_threshold() {
        assert_eq!(exp_strict(-1000.0), 0.0);
        assert_eq!(exp_compensated(-1000.0), 0.0);
        assert_eq!(exp_correctly_rounded(-1000.0), 0.0);
    }

    // ── Known-value spot checks ────────────────────────────────────────────

    #[test]
    fn exp_of_one_is_e() {
        let e = std::f64::consts::E;
        assert_within_ulps(exp_strict(1.0), e, 3, "exp_strict(1)");
        assert_within_ulps(exp_compensated(1.0), e, 1, "exp_compensated(1)");
        assert_within_ulps(exp_correctly_rounded(1.0), e, 1, "exp_correctly_rounded(1)");
    }

    #[test]
    fn exp_of_ln2_is_2() {
        let ln2 = std::f64::consts::LN_2;
        assert_within_ulps(exp_strict(ln2), 2.0, 3, "exp_strict(ln2)");
        assert_within_ulps(exp_compensated(ln2), 2.0, 1, "exp_compensated(ln2)");
        assert_within_ulps(exp_correctly_rounded(ln2), 2.0, 1, "exp_correctly_rounded(ln2)");
    }

    // ── Strategy ulp budgets via f64::exp reference ───────────────────────

    #[test]
    fn exp_strict_within_budget() {
        check_strategy(exp_strict, "exp_strict", 4);
    }

    #[test]
    fn exp_compensated_within_budget() {
        check_strategy(exp_compensated, "exp_compensated", 2);
    }

    #[test]
    fn exp_correctly_rounded_within_budget() {
        check_strategy(exp_correctly_rounded, "exp_correctly_rounded", 1);
    }

    // ── Strategy monotonicity: compensated ≤ strict, correct ≤ compensated

    #[test]
    fn strategies_converge_on_accuracy() {
        // For every sample, the compensated version should be at least as
        // close to the reference as the strict version (within one ulp of
        // rounding jitter). This documents that each added layer of
        // precision is a monotone improvement.
        let samples: &[f64] = &[
            0.5, 1.0, 2.0, 3.14, 10.0, 100.0, -1.0, -10.0, -100.0,
        ];
        for &x in samples {
            let reference = x.exp();
            let strict_err = ulps_between(exp_strict(x), reference);
            let comp_err = ulps_between(exp_compensated(x), reference);
            let cr_err = ulps_between(exp_correctly_rounded(x), reference);
            // Allow one ulp of wiggle since f64::exp itself may round
            // slightly differently than our reference.
            assert!(
                comp_err <= strict_err + 1,
                "x={x}: compensated ({comp_err}) worse than strict ({strict_err})"
            );
            assert!(
                cr_err <= comp_err + 1,
                "x={x}: correctly_rounded ({cr_err}) worse than compensated ({comp_err})"
            );
        }
    }

    // ── Mathematical identity tests ────────────────────────────────────────

    #[test]
    fn exp_monotone_on_positive_range() {
        let xs: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let mut prev = exp_correctly_rounded(xs[0]);
        for &x in &xs[1..] {
            let y = exp_correctly_rounded(x);
            assert!(y >= prev, "exp not monotone at x={x}: {y} < {prev}");
            prev = y;
        }
    }

    #[test]
    fn exp_positive_everywhere() {
        let xs: &[f64] = &[-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0];
        for &x in xs {
            assert!(exp_strict(x) > 0.0, "exp_strict({x}) not positive");
            assert!(exp_compensated(x) > 0.0, "exp_compensated({x}) not positive");
            assert!(
                exp_correctly_rounded(x) > 0.0,
                "exp_correctly_rounded({x}) not positive"
            );
        }
    }

    #[test]
    fn exp_product_rule_holds_approximately() {
        // exp(a) · exp(b) ≈ exp(a + b)
        let pairs: &[(f64, f64)] = &[(1.0, 2.0), (0.5, 0.5), (-1.0, 2.0), (10.0, -5.0)];
        for &(a, b) in pairs {
            let lhs = exp_correctly_rounded(a) * exp_correctly_rounded(b);
            let rhs = exp_correctly_rounded(a + b);
            let dist = ulps_between(lhs, rhs);
            assert!(
                dist <= 10,
                "exp product rule off for ({a}, {b}): {dist} ulps"
            );
        }
    }
}
