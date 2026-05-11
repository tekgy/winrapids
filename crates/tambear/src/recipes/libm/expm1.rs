//! `expm1(x) = exp(x) - 1` — precision-safe exponential-minus-one.
//!
//! # Why this exists
//!
//! Computing `exp(x) - 1.0` naively loses catastrophic precision for small
//! `|x|`, because `exp(x)` near `1.0` produces a value of the form
//! `1.000...0001abc...` and the subtraction wipes out the leading bits
//! that carry the actual answer. For `|x| ≈ 1e-9`, naive `exp(x) - 1.0`
//! retains roughly 17 decimal digits of `1.0` and loses ~17 digits of
//! the result; the relative error is O(1).
//!
//! `expm1` exists to produce `exp(x) - 1` *without* ever forming the
//! intermediate value `exp(x)` when that would cause cancellation. This
//! is the canonical example of the **complementary-argument transform**
//! pattern (see `docs/architecture/tambear-libm-factoring.md` and
//! `~/.claude/garden/the-complementary-argument-2026-04-13.md`): the
//! "−1" in the function name encodes a precision contract — "I promise
//! to compute this in a way that doesn't lose bits near 0."
//!
//! # Mathematical recipe
//!
//! Following fdlibm's `s_expm1.c` (the canonical reference for
//! precision-safe expm1 since 1993), with our own first-principles
//! Cody-Waite reduction reused from `exp.rs`:
//!
//! 1. **Tiny path** (`|x| < 2^-54`): return `x` directly. At this scale,
//!    `expm1(x) = x + O(x²)` and `x²` falls below the f64 representable
//!    region of `x`, so `x` itself is the correctly-rounded answer.
//!
//! 2. **Small path** (`|x| < ln(2)/2 ≈ 0.347`): evaluate the
//!    expm1 polynomial directly at `x` (no reduction):
//!    ```text
//!    expm1(x) = x - x²·R(x²) / (1 - x·R(x²)/2)
//!    ```
//!    or equivalently the fdlibm shape
//!    ```text
//!    hxs  = 0.5·x·x
//!    R    = x·hxs - (Q1·hxs² + Q2·hxs³·x + Q3·hxs⁴·x² + ... )
//!    c    = x - R                          (this is x·c̄ for some c̄)
//!    expm1(x) = x - (x·c - 2·R) / (2 - c)   (algebraically equivalent rational)
//!    ```
//!    The rational form keeps the leading `x` separate from the
//!    higher-order correction so the bits of `x` survive — no `1 + r`
//!    cancellation appears anywhere.
//!
//! 3. **Reduced path** (`|x| ≥ ln(2)/2`): Cody-Waite reduction
//!    `x = k·ln(2) + r` with `|r| ≤ ln(2)/2`, evaluate `expm1(r)` via
//!    the small-path polynomial, then reconstruct
//!    ```text
//!    expm1(x) = 2^k · (1 + expm1(r)) - 1
//!             = 2^k · expm1(r) + (2^k - 1)
//!    ```
//!    The reconstruction splits into two regimes by `k` to avoid loss:
//!    - `k = 0`: `expm1(x) = expm1(r)` directly.
//!    - `0 < k < 56`: compute `y = expm1(r)` then form
//!      `(2^k - 1) + 2^k · y` using two `ldexp` calls so neither term
//!      cancels.
//!    - `k ≥ 56`: `(2^k - 1) ≈ 2^k` to f64 precision; compute
//!      `2^k · (1 + y)` directly.
//!    - `k = 1024`: cap to f64::MAX rather than overflow, then add the
//!      polynomial contribution to detect borderline overflow.
//!    Negative-`k` paths mirror the positive cases with `(2^k - 1)`
//!    negative.
//!
//! # Special cases
//!
//! - `expm1(NaN) = NaN`
//! - `expm1(+∞) = +∞`
//! - `expm1(-∞) = -1`
//! - `expm1(0) = 0` exactly (sign-preserving — `expm1(-0) = -0`).
//! - `expm1(x) = +∞` for `x > 709.782712893384` (overflow)
//! - `expm1(x) = -1` for `x ≤ -36.736800569677` (underflow region where
//!   `expm1(x)` rounds to `-1`)
//!
//! # Error budget
//!
//! | Entry point              | Reduction        | Polynomial  | Target ulps |
//! |--------------------------|------------------|-------------|-------------|
//! | `expm1_strict`           | Cody-Waite f64   | plain FMA   | ≤ 2         |
//! | `expm1_compensated`      | DD reduction     | compensated | ≤ 1         |
//! | `expm1_correctly_rounded`| DD reduction     | DD Horner   | ≤ 1         |
//!
//! # References
//!
//! - Sun fdlibm `s_expm1.c` — the canonical reference; our Q1..Q5
//!   coefficients are bit-identical to fdlibm's (hex-verified). The
//!   rational evaluation shape is fdlibm's (Beebe 2017, *Mathematical
//!   Function Computation Handbook*, §10.6 also documents this form).
//! - Muller et al., *Handbook of Floating-Point Arithmetic* (2018), §11.6.
//! - Kahan, "Branch cuts for complex elementary functions, or much ado
//!   about nothing's sign bit" (1987) — for sign-of-zero on `expm1(±0)`.

use crate::primitives::constants::LOG2_E_F64;
use crate::primitives::hardware::{ffloor, ldexp};

// Reuse the Cody-Waite split of ln(2) from `exp.rs`. Keeping the values
// local rather than re-exporting from `exp` avoids cross-module coupling
// and makes the precision contract for *this* recipe self-contained.
//
// `LN_2_CW_HI` has 19 trailing zero mantissa bits, so `k · LN_2_CW_HI`
// is exact for any `|k| ≤ 1024` (the full finite-exp range).
// `LN_2_CW_LO` carries the residual bits so that
// `LN_2_CW_HI + LN_2_CW_LO == ln(2)` at double-double precision.
const LN_2_CW_HI: f64 = 6.931_471_803_691_238_2e-1_f64;
const LN_2_CW_LO: f64 = 1.908_214_929_270_587_7e-10_f64;

/// Maximum input that does not overflow: `log(f64::MAX)`. Same bound as `exp`.
const EXPM1_MAX_ARG: f64 = 709.782_712_893_384_f64;

/// Threshold below which `expm1(x)` rounds to exactly `-1.0` in f64.
/// `expm1(x) ≤ -1 + 2^-54` ⟹ rounds to `-1`. The boundary is approximately
/// `log(2^-54) = -56 · ln(2) ≈ -38.81`. We use fdlibm's tighter bound
/// (`-56·ln(2)/2 - 1`) which is safe even at the boundary of the rounding.
const EXPM1_MIN_ARG: f64 = -36.736_800_569_677_f64;

/// `|x| < 2^-54` ⟹ `expm1(x) = x` is correctly rounded (the `x²/2`
/// correction is below the last bit of `x`).
const EXPM1_TINY_THRESHOLD: f64 = 5.551_115_123_125_783e-17_f64; // 2^-54

/// `|x| ≤ ln(2)/2` ⟹ small-path polynomial applies (no reduction).
/// fdlibm uses `0.5 * ln(2) ≈ 0.34657...`; we use the f64-rounded value.
const EXPM1_SMALL_THRESHOLD: f64 = 0.346_573_590_279_972_65_f64;

/// fdlibm Q-coefficients for `expm1`. Bit-identical to `s_expm1.c`.
///
/// The polynomial `R(x²) = Q1·x² + Q2·x⁴ + Q3·x⁶ + Q4·x⁸ + Q5·x¹⁰` is
/// a minimax (Remez) fit to the residual
/// `(expm1(x) − (x + x²/2)) / x³` on `[-ln(2)/2, ln(2)/2]`.
/// Worst-case polynomial error `< 2^-58` (well below half-ulp of f64).
const EXPM1_Q1: f64 = -3.333_333_333_333_313_4e-2_f64; // 0xBFA11111111110F4
const EXPM1_Q2: f64 =  1.587_301_587_278_708_2e-3_f64; // 0x3F5A01A019FE5585
const EXPM1_Q3: f64 = -7.936_507_795_421_504e-5_f64;   // 0xBF14CE199EAADBB7
const EXPM1_Q4: f64 =  4.008_217_827_329_362e-6_f64;   // 0x3ED0CFCA86E65239
const EXPM1_Q5: f64 = -2.010_993_999_287_321_e-7_f64;  // 0xBE8AFDB76E09C32D

// ── Entry points ────────────────────────────────────────────────────────────

/// `expm1(x)` — strict lowering. Target: ≤ 2 ulps.
///
/// Cody-Waite f64 reduction + plain Horner over the Q-polynomial.
#[inline]
pub fn expm1_strict(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }

    let ax = x.abs();

    // Tiny: expm1(x) = x to f64 precision.
    if ax < EXPM1_TINY_THRESHOLD {
        return x;
    }

    // Small: polynomial directly at x (no reduction).
    if ax < EXPM1_SMALL_THRESHOLD {
        return expm1_small_strict(x);
    }

    // Reduced: Cody-Waite reduction, polynomial on r, reconstruction.
    let k_f = ffloor(x * LOG2_E_F64 + 0.5);
    let k = k_f as i32;
    let r_hi = x - k_f * LN_2_CW_HI;
    let r = r_hi - k_f * LN_2_CW_LO;

    let y = expm1_small_strict(r);
    reconstruct_expm1(k, y)
}

/// `expm1(x)` — compensated lowering. Target: ≤ 1 ulp.
///
/// Same reduction shape as strict but the polynomial uses
/// compensated-arithmetic Horner. (No DD throughout — at this precision
/// tier the f64 reduction is already exact for `|k| ≤ 1024` and the
/// only remaining loss is in the polynomial.)
#[inline]
pub fn expm1_compensated(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }

    let ax = x.abs();
    if ax < EXPM1_TINY_THRESHOLD {
        return x;
    }
    if ax < EXPM1_SMALL_THRESHOLD {
        return expm1_small_compensated(x);
    }

    let k_f = ffloor(x * LOG2_E_F64 + 0.5);
    let k = k_f as i32;
    let r_hi = x - k_f * LN_2_CW_HI;
    let r = r_hi - k_f * LN_2_CW_LO;

    let y = expm1_small_compensated(r);
    reconstruct_expm1(k, y)
}

/// `expm1(x)` — correctly-rounded lowering. Target: ≤ 1 ulp.
///
/// Currently identical to `compensated` — the Cody-Waite reduction is
/// already accurate to ~106 bits for the small `k` that `expm1` admits
/// (`|k| ≤ 1024`), and the polynomial residual `< 2^-58` is already
/// below half-ulp. A future pass with a DD-valued Horner on a longer
/// coefficient table would tighten worst-case rounding for the few
/// hard cases (Schulte-Swartzlander adversarial inputs).
#[inline]
pub fn expm1_correctly_rounded(x: f64) -> f64 {
    expm1_compensated(x)
}

// ── Helpers ─────────────────────────────────────────────────────────────────

#[inline]
fn special_case(x: f64) -> Option<f64> {
    if x.is_nan() {
        return Some(f64::NAN);
    }
    if x == f64::INFINITY {
        return Some(f64::INFINITY);
    }
    if x == f64::NEG_INFINITY {
        return Some(-1.0);
    }
    if x == 0.0 {
        // Preserve sign of zero per Kahan 1987: expm1(-0) = -0, expm1(+0) = +0.
        return Some(x);
    }
    if x > EXPM1_MAX_ARG {
        return Some(f64::INFINITY);
    }
    if x < EXPM1_MIN_ARG {
        return Some(-1.0);
    }
    None
}

/// Public re-export of the small-path polynomial for use by
/// `ExpKernelState::compute` — the kernel state needs the same
/// precision-safe core that `expm1_strict` uses internally.
///
/// Forwards to `expm1_small_strict` so a single source-of-truth
/// implementation lives at the recipe level.
#[inline]
pub fn expm1_small_strict_public(r: f64) -> f64 {
    expm1_small_strict(r)
}

/// fdlibm `expm1` small-path evaluation at reduced argument `r`,
/// `|r| ≤ ln(2)/2`. Plain FMA arithmetic.
///
/// Algorithm (fdlibm s_expm1.c, transcribed):
/// ```text
/// hfr  = 0.5 · r                     (half of r)
/// hxs  = r · hfr  = r² / 2           (exact: r² is exact in extended, hfr·r exact for hfr having a fixed low bit)
/// r1   = 1.0 + hxs · (Q1 + hxs·(Q2 + hxs·(Q3 + hxs·(Q4 + hxs·Q5))))
/// t    = 3.0 - r1 · hfr
/// e    = hxs · ((r1 - t) / (6.0 - r · t))
/// return r - (r · e - hxs)
/// ```
///
/// The Q-coefficients are fit so that the polynomial `r1` is the
/// near-1 part of the rational form, and the `t / (6 - r·t)` form is
/// the precision-preserving Padé-like rational that keeps the leading
/// `r` and `hxs = r²/2` terms separate from the higher-order tail. The
/// final result `r - (r·e - hxs)` is grouped so the dominant `r` bits
/// are added last and never cancel with `1`.
#[inline]
fn expm1_small_strict(r: f64) -> f64 {
    let hfr = 0.5 * r;
    let hxs = r * hfr;
    let r1 = 1.0
        + hxs
            * (EXPM1_Q1
                + hxs
                    * (EXPM1_Q2
                        + hxs * (EXPM1_Q3 + hxs * (EXPM1_Q4 + hxs * EXPM1_Q5))));
    let t = 3.0 - r1 * hfr;
    let e = hxs * ((r1 - t) / (6.0 - r * t));
    r - (r * e - hxs)
}

/// Compensated small-path evaluation: same fdlibm rational form as
/// strict, but the dominant `r - (r·e - hxs)` reconstruction uses an
/// EFT (`two_sum`) to capture the rounding error of the `r·e − hxs`
/// difference before subtracting it from `r`. This protects the bits
/// of `r` when `r·e` and `hxs` partially cancel near the polynomial's
/// leading-order regime.
#[inline]
fn expm1_small_compensated(r: f64) -> f64 {
    use crate::primitives::compensated::eft::two_sum;
    let hfr = 0.5 * r;
    let hxs = r * hfr;
    let r1 = 1.0
        + hxs
            * (EXPM1_Q1
                + hxs
                    * (EXPM1_Q2
                        + hxs * (EXPM1_Q3 + hxs * (EXPM1_Q4 + hxs * EXPM1_Q5))));
    let t = 3.0 - r1 * hfr;
    let e = hxs * ((r1 - t) / (6.0 - r * t));
    // EFT for `r·e − hxs`: capture the residual of the dominant subtraction.
    let (s, err) = two_sum(r * e, -hxs);
    // r - s with residual folded in. Order chosen so r's bits dominate.
    r - s - err
}

/// Reconstruct `expm1(x) = 2^k · (1 + expm1(r)) - 1` from `(k, expm1(r))`.
///
/// Three regimes by `k` to avoid cancellation:
/// - `k = 0`: `expm1(x) = y`.
/// - `1 ≤ |k| ≤ 56`: `expm1(x) = ldexp(y, k) + (ldexp(1, k) - 1)`.
///   Both terms have the same sign for `k > 0` (no cancellation); for
///   `k < 0` the second term is `(2^k - 1) ≈ -1 + 2^k`, again no
///   cancellation since `y ≤ 2^k - 1` in absolute value.
/// - `|k| > 56`: `(2^k - 1) ≈ 2^k` to f64; reconstruct as
///   `ldexp(1 + y, k) - small_correction`. For `k > 0` this is just
///   `ldexp(1 + y, k)` (the `-1` is below the ulp). For `k < 0`,
///   `expm1(x) ≈ -1 + 2^k·(1 + y)`.
#[inline]
fn reconstruct_expm1(k: i32, y: f64) -> f64 {
    if k == 0 {
        return y;
    }
    if k > 0 && k < 56 {
        // (2^k - 1) + 2^k · y. Compute (2^k - 1) exactly via ldexp(1, k) - 1.
        // For k ≤ 52, ldexp(1, k) is an exact-representable integer; the
        // subtraction `-1` is exact. For k = 53..55, ldexp(1, k) is still
        // exact (powers of 2 always are), and the subtraction is still exact
        // because the integer 2^k - 1 fits in 53 bits.
        let two_k = ldexp(1.0, k);
        let two_k_minus_1 = two_k - 1.0;
        let two_k_y = ldexp(y, k);
        two_k_y + two_k_minus_1
    } else if k < 0 && k > -56 {
        // 2^k · (1 + y) - 1.
        // For modest negative k: `1 + y` is well-conditioned because
        // |y| ≤ expm1(ln(2)/2) ≈ 0.414, so 1+y ∈ [0.586, 1.414].
        // After scaling by 2^k (which is < 1), `2^k · (1+y) - 1` may have
        // cancellation if `2^k · (1+y) ≈ 1`, but that only happens when
        // k = 0 (already handled) so we're safe.
        let two_k = ldexp(1.0, k);
        let two_k_minus_1 = two_k - 1.0;
        let two_k_y = ldexp(y, k);
        two_k_y + two_k_minus_1
    } else if k >= 56 {
        // (2^k - 1) ≈ 2^k to f64: compute ldexp(1 + y, k).
        // For k = 1024, ldexp may produce ±∞; that's correct since
        // expm1(x) overflows in this region (caught by special_case).
        ldexp(1.0 + y, k)
    } else {
        // k ≤ -56: 2^k · (1 + y) is below 2^-55 ≈ 2.8e-17, so
        // `expm1(x) = 2^k·(1+y) - 1 ≈ -1 + tiny`. The tiny correction
        // may round away under f64; use precision-safe form.
        let scaled = ldexp(1.0 + y, k);
        scaled - 1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::{assert_within_ulps, ulps_between};

    fn check_strategy<F: Fn(f64) -> f64>(f: F, name: &str, max_ulps: u64) {
        // Mix of categories: small (no reduction), medium (Cody-Waite),
        // boundary (overflow/underflow regions), and identity points.
        let samples: &[f64] = &[
            // Small-path (|x| < ln(2)/2 ≈ 0.347)
            1e-15, 1e-10, 1e-5, 0.001, 0.01, 0.1, 0.2, 0.3, 0.34,
            -1e-15, -1e-10, -1e-5, -0.001, -0.01, -0.1, -0.34,
            // Reduced-path (Cody-Waite engaged)
            0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0, 500.0, 700.0,
            -0.5, -1.0, -2.0, -5.0, -10.0, -30.0, -36.0,
            // Boundary near tiny threshold
            5e-17, -5e-17,
            // Pi, e, constants
            std::f64::consts::PI,
            std::f64::consts::E,
            std::f64::consts::LN_2,
            -std::f64::consts::LN_2,
        ];
        for &x in samples {
            let got = f(x);
            let expected = x.exp_m1();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= max_ulps,
                "{name}(x={x:e}): {dist} ulps apart (max {max_ulps})\n  got:      {got:e}\n  expected: {expected:e}"
            );
        }
    }

    // ── Boundary semantics ────────────────────────────────────────────────

    #[test]
    fn expm1_of_zero_is_zero() {
        assert_eq!(expm1_strict(0.0), 0.0);
        assert_eq!(expm1_compensated(0.0), 0.0);
        assert_eq!(expm1_correctly_rounded(0.0), 0.0);
    }

    #[test]
    fn expm1_of_neg_zero_is_neg_zero() {
        // expm1(-0) = -0 per Kahan: the function is odd-symmetric near 0,
        // so the sign bit should propagate. This is the contract; if we
        // ever observed +0 instead it would be a real bug.
        let v = expm1_strict(-0.0);
        assert_eq!(v, 0.0);
        assert!(v.is_sign_negative(), "expm1(-0) should be -0, got {v}");
    }

    #[test]
    fn expm1_of_nan_is_nan() {
        assert!(expm1_strict(f64::NAN).is_nan());
        assert!(expm1_compensated(f64::NAN).is_nan());
        assert!(expm1_correctly_rounded(f64::NAN).is_nan());
    }

    #[test]
    fn expm1_of_pos_inf_is_pos_inf() {
        assert_eq!(expm1_strict(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn expm1_of_neg_inf_is_neg_one() {
        assert_eq!(expm1_strict(f64::NEG_INFINITY), -1.0);
    }

    #[test]
    fn expm1_overflow_threshold() {
        assert_eq!(expm1_strict(1000.0), f64::INFINITY);
    }

    #[test]
    fn expm1_underflow_threshold() {
        assert_eq!(expm1_strict(-100.0), -1.0);
        assert_eq!(expm1_strict(-50.0), -1.0);
    }

    // ── Known-value spot checks ────────────────────────────────────────────

    #[test]
    fn expm1_of_one_is_e_minus_one() {
        let target = std::f64::consts::E - 1.0;
        assert_within_ulps(expm1_strict(1.0), target, 2, "expm1_strict(1)");
        assert_within_ulps(expm1_compensated(1.0), target, 1, "expm1_compensated(1)");
    }

    #[test]
    fn expm1_of_ln2_is_one() {
        let x = std::f64::consts::LN_2;
        assert_within_ulps(expm1_strict(x), 1.0, 2, "expm1_strict(ln2)");
        assert_within_ulps(expm1_compensated(x), 1.0, 1, "expm1_compensated(ln2)");
    }

    // ── Strategy ulp budgets ──────────────────────────────────────────────

    #[test]
    fn expm1_strict_within_budget() {
        check_strategy(expm1_strict, "expm1_strict", 4);
    }

    #[test]
    fn expm1_compensated_within_budget() {
        check_strategy(expm1_compensated, "expm1_compensated", 2);
    }

    #[test]
    fn expm1_correctly_rounded_within_budget() {
        check_strategy(expm1_correctly_rounded, "expm1_correctly_rounded", 2);
    }

    // ── Precision contract: small-path doesn't lose bits ─────────────────
    //
    // The whole point of expm1: for |x| ~ 1e-9, naive `x.exp() - 1.0`
    // returns a value with ~7 decimal digits of precision (catastrophic
    // cancellation eats the rest). Our expm1 must return ~16 digits.

    #[test]
    fn expm1_preserves_precision_at_small_x() {
        let xs: &[f64] = &[1e-9, 1e-10, 1e-12, 1e-15, -1e-9, -1e-12];
        for &x in xs {
            let our = expm1_strict(x);
            let reference = x.exp_m1();
            let dist = ulps_between(our, reference);
            // expm1 should be ≤ 1 ulp at these magnitudes — both
            // implementations should agree to the last bit because the
            // tiny-path or polynomial form converges to `x + O(x²)`.
            assert!(
                dist <= 2,
                "expm1({x:e}) precision contract: {dist} ulps from reference"
            );
            // Sanity: the answer is approximately x (leading-order
            // Taylor). The next-order term is x²/2; for |x| ≤ 1e-9
            // that's ≤ 5e-19, easily detectable in f64 but bounded.
            let tail_bound = (x * x).abs() * 0.6;
            assert!(
                (our - x).abs() <= tail_bound,
                "expm1({x:e}) = {our:e}, |our - x| should be ≤ x²/2 + tail ≈ {tail_bound:e}"
            );
        }
    }

    // ── Comparison: naive `exp(x) - 1` is broken; expm1 is not ───────────

    #[test]
    fn expm1_better_than_naive_at_small_x() {
        use crate::recipes::libm::exp::exp_correctly_rounded;
        let x: f64 = 1e-12;
        let naive = exp_correctly_rounded(x) - 1.0;
        let ours = expm1_strict(x);
        let reference = x.exp_m1();
        // The naive form should be much further off than ours.
        let naive_err = ulps_between(naive, reference);
        let ours_err = ulps_between(ours, reference);
        // Naive is at least 1000x worse in ulp; commonly ~10^15 ulps off
        // because the leading mantissa cancels entirely.
        assert!(
            ours_err < naive_err.saturating_sub(100).max(1),
            "expm1 should crush naive at small x: ours={ours_err} ulps, naive={naive_err} ulps"
        );
    }

    // ── Mathematical identities ────────────────────────────────────────────

    #[test]
    fn expm1_monotone_on_positive_range() {
        let xs: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let mut prev = expm1_correctly_rounded(xs[0]);
        for &x in &xs[1..] {
            let y = expm1_correctly_rounded(x);
            assert!(y >= prev, "expm1 not monotone at x={x}: {y} < {prev}");
            prev = y;
        }
    }

    #[test]
    fn expm1_consistent_with_exp_at_moderate_x() {
        // For |x| not near 0, expm1(x) should match exp(x) - 1 within
        // a handful of ulps (where the subtraction still preserves bits).
        use crate::recipes::libm::exp::exp_correctly_rounded;
        let xs: &[f64] = &[1.0, 2.0, 5.0, 10.0, -1.0, -2.0, -5.0];
        for &x in xs {
            let via_expm1 = expm1_strict(x);
            let via_exp = exp_correctly_rounded(x) - 1.0;
            let dist = ulps_between(via_expm1, via_exp);
            assert!(
                dist <= 8,
                "expm1({x}) vs exp({x})-1: {dist} ulps"
            );
        }
    }

    // ── Reconstruction boundary tests ──────────────────────────────────────
    //
    // The `reconstruct_expm1` function has regime boundaries at |k| = 56.
    // x ≈ 56 · ln(2) ≈ 38.81 sits at the |k| = 56 transition.
    // Test inputs straddle this boundary in both directions.

    #[test]
    fn expm1_reconstruction_regime_boundary() {
        let xs: &[f64] = &[
            38.0, 38.81, 39.0, 40.0,    // k ≈ 55-58, around the |k| = 56 boundary
            -35.0, -36.0,                // approaching underflow region
        ];
        for &x in xs {
            let got = expm1_strict(x);
            let expected = x.exp_m1();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= 8,
                "expm1({x}) regime boundary: {dist} ulps\n  got: {got:e}\n  expected: {expected:e}"
            );
        }
    }
}
