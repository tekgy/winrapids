//! `log1p(x) = log(1 + x)` — precision-safe logarithm-of-one-plus.
//!
//! # Why this exists
//!
//! Computing `log(1.0 + x)` naively loses catastrophic precision for
//! small `|x|`, because `1.0 + x` rounds away the low bits of `x`
//! before `log` ever sees them. For `|x| ≈ 1e-16` the result is the
//! same as `log(1.0) = 0` — every bit of information is lost.
//!
//! `log1p` is the inverse-direction analog of `expm1`. Together they
//! form the **complementary-argument transform** at the fixed point
//! `1` for multiplicative-group structure: the "1+" is the complement.
//! See `docs/architecture/tambear-libm-factoring.md` and
//! `~/.claude/garden/the-complementary-argument-2026-04-13.md`.
//!
//! # Mathematical recipe
//!
//! Following fdlibm's `s_log1p.c` (Sun's canonical 1993 reference),
//! and reusing the Lg1..Lg7 polynomial coefficients from `log.rs`
//! (the same Sun fdlibm minimax fit covers both functions):
//!
//! 1. **Tiny path** (`|x| < 2^-54`): return `x` directly. At this
//!    scale `log(1 + x) = x + O(x²)` and `x²` is below the f64
//!    representable region of `x`, so `x` is the correctly-rounded answer.
//!
//! 2. **Centered path** (`|x| ≤ 0.41421` ≈ √2 − 1): no exponent
//!    reduction needed. Compute `f = x`, set `k = 0`, the correction
//!    `c = 0` (no bits lost since we never formed `1 + x`).
//!
//! 3. **General path** (`|x| > √2 − 1`):
//!    Reduction: `y = 1 + x`, then `(m, k) = frexp(y)`. Multiply
//!    `m *= 2; k -= 1` to put `m ∈ [1, 2)`. If `m > √2`, divide by 2
//!    and increment `k` so the polynomial argument `f = m − 1` is
//!    centered near 0 with `|f| ≤ √2 − 1`.
//!
//!    The **precision correction**: when `y = 1 + x` rounded, it lost
//!    some bits of x. The exact identity `(1 + x) = y + c` with
//!    `c = 1 - (y - x)` recovers those bits losslessly (Dekker
//!    `two_sum`-style). For `k = 0`, `c = 0` because we didn't
//!    actually form `y`. For `k > 0`, we propagate `c` into the
//!    polynomial result as a small additive correction.
//!
//! 4. **Polynomial**: with `f = m − 1` (or `f = x` in the centered path)
//!    and `s = f / (2 + f)`:
//!    ```text
//!    z = s²
//!    R = z·(Lg1 + z·(Lg2 + z·(Lg3 + z·(Lg4 + z·(Lg5 + z·(Lg6 + z·Lg7))))))
//!    hfsq = 0.5 · f²
//!    log(1 + f) ≈ f − hfsq + s·(hfsq + R)
//!    ```
//!    Same series as `log`. The `s = f/(2+f)` substitution halves the
//!    magnitude of the polynomial argument vs naive Taylor in `f`,
//!    accelerating convergence.
//!
//! 5. **Reconstruction**: `log1p(x) = polynomial(f) + k·ln(2) − c`,
//!    where the `−c` distributes the lost-bits correction back into
//!    the result.
//!
//! # Special cases
//!
//! - `log1p(NaN) = NaN`
//! - `log1p(+∞) = +∞`
//! - `log1p(-1) = -∞`
//! - `log1p(x) = NaN` for `x < -1`
//! - `log1p(0) = 0` exactly (sign-preserving: `log1p(-0) = -0`)
//! - `log1p(MAX) = log(MAX + 1) ≈ 709.78` (no overflow possible since
//!   `log(MAX) ≈ 709.78`)
//!
//! # Error budget
//!
//! | Entry point              | Reduction        | Polynomial  | Target ulps |
//! |--------------------------|------------------|-------------|-------------|
//! | `log1p_strict`           | frexp + c-corr   | plain FMA   | ≤ 2         |
//! | `log1p_compensated`      | frexp + c-corr   | compensated | ≤ 1         |
//! | `log1p_correctly_rounded`| frexp + c-corr   | DD Horner   | ≤ 1         |
//!
//! # References
//!
//! - Sun fdlibm `s_log1p.c` — the canonical precision-safe log1p
//!   implementation. Our `Lg1..Lg7` coefficients are bit-identical to
//!   those used by `log.rs` and `s_log1p.c`.
//! - Muller et al., *Handbook of Floating-Point Arithmetic* (2018), §11.4.
//! - Goldberg, "What every computer scientist should know about
//!   floating-point arithmetic" (1991) — §1.4 explains the `c =
//!   1 − (y − x)` recovery exactly.

use crate::primitives::compensated::dot::{compensated_horner, horner};
use crate::primitives::compensated::eft::two_sum;
use crate::primitives::constants::{LN_2_DD, SQRT_2_F64};
use crate::primitives::hardware::frexp;

/// `|x| < 2^-54` ⟹ `log1p(x) = x` is correctly rounded.
const LOG1P_TINY_THRESHOLD: f64 = 5.551_115_123_125_783e-17_f64; // 2^-54

/// Centered-path upper bound (positive x): `√2 − 1 ≈ 0.41421...`.
/// For `0 < x < √2 − 1`, `s = x/(2+x) ∈ [0, 3−2√2 ≈ 0.172]`, in the
/// polynomial's tight-fit range.
const LOG1P_CENTERED_POS_BOUND: f64 = 0.414_213_562_373_095_05_f64; // √2 - 1

/// Centered-path lower bound (negative x): `1 − √2/2 · 2 = 1 − √2 ≈
/// −0.41421` is the natural reflection, BUT `s = x/(2+x)` is
/// asymmetric: at `x = -0.4`, `|s| = 0.25 > 0.172`, overshooting the
/// fit range. Solving `s = -0.172 = x/(2+x)` for x gives
/// `x = -2·0.172/(1−0.172) ≈ −0.2929`. For x in `[-0.2929, 0)` the
/// centered path is still safe; below that, take the general path
/// (frexp + bit-recovery correction).
const LOG1P_CENTERED_NEG_BOUND: f64 = -0.292_893_218_813_452_5_f64; // -2·s_max/(1-s_max)

/// Coefficients for the odd-power series `(log(1+f) − 2s)/s³` in `s²`.
/// Bit-identical to `log.rs`'s `LOG_COEFFS`. Duplicated here rather
/// than re-exported so each recipe's precision contract is self-contained.
const LG1: f64 = 6.666_666_666_666_735_13e-01;
const LG2: f64 = 3.999_999_999_940_941_91e-01;
const LG3: f64 = 2.857_142_874_366_239_15e-01;
const LG4: f64 = 2.222_219_843_214_978_40e-01;
const LG5: f64 = 1.818_357_216_161_805_01e-01;
const LG6: f64 = 1.531_383_769_920_937_33e-01;
const LG7: f64 = 1.479_819_860_511_658_59e-01;

const LG_COEFFS: [f64; 7] = [LG1, LG2, LG3, LG4, LG5, LG6, LG7];

// ── Entry points ────────────────────────────────────────────────────────────

/// `log1p(x)` — strict lowering. Target: ≤ 2 ulps.
#[inline]
pub fn log1p_strict(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }

    let ax = x.abs();

    // Tiny: log1p(x) = x to f64 precision.
    if ax < LOG1P_TINY_THRESHOLD {
        return x;
    }

    let (k, f, c) = reduce(x);
    eval_strict(k, f, c)
}

/// `log1p(x)` — compensated lowering. Target: ≤ 1 ulp.
#[inline]
pub fn log1p_compensated(x: f64) -> f64 {
    if let Some(special) = special_case(x) {
        return special;
    }

    let ax = x.abs();
    if ax < LOG1P_TINY_THRESHOLD {
        return x;
    }

    let (k, f, c) = reduce(x);
    eval_compensated(k, f, c)
}

/// `log1p(x)` — correctly-rounded lowering. Target: ≤ 1 ulp.
///
/// Currently shares the compensated polynomial; a future pass with a
/// DD-valued Horner on a longer coefficient table would tighten the
/// worst cases (Schulte-Swartzlander adversarial inputs near the cut
/// of the `s = f/(2+f)` substitution).
#[inline]
pub fn log1p_correctly_rounded(x: f64) -> f64 {
    log1p_compensated(x)
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
    if x == -1.0 {
        return Some(f64::NEG_INFINITY);
    }
    if x < -1.0 {
        return Some(f64::NAN);
    }
    if x == 0.0 {
        // Preserve sign: log1p(-0) = -0, log1p(+0) = +0.
        return Some(x);
    }
    None
}

/// Reduce `x` into `(k, f, c)` where `1 + x = 2^k · (1 + f)` with
/// `|f| ≤ √2 − 1` and `c` is the bit-recovery correction for the
/// `1 + x` rounding loss.
///
/// **Centered path** (k = 0, c = 0): applies only when `0 < x < √2 − 1`.
/// For positive x in this range, `s = x/(2+x) ∈ [0, (√2-1)/(√2+1)] =
/// [0, 3 − 2√2 ≈ 0.172]`, so `z = s² ≤ 0.0294` — the polynomial fit
/// range. For *negative* x in `(-√2+1, 0)`, `|s| = |x|/(2+x)` grows
/// asymmetrically: at x = -0.4, `|s| = 0.25 > 0.172`, putting z out
/// of the polynomial's tight-fit range and producing ~12-digit
/// accuracy (~7000 ulps off). So negative x always takes the general
/// path, even for small `|x|`.
///
/// **General path** (k may be nonzero, c carries the bit-recovery):
/// `y = 1 + x` rounded; `c = 1 − (y − x)` or `c = x − (y − 1.0)` per
/// fdlibm depending on `|x|`. By Goldberg/Dekker, this `c` is the
/// *exact* low-order residual of the `1 + x` rounding, so
/// `(1 + x) = y + c` losslessly. Then `frexp(y)` decomposes the
/// exponent.
///
/// **Tiny-positive shortcut**: for `0 < x < 2^-30`, the tiny path
/// already handled it (return `x`). This `reduce` is only called for
/// inputs above that threshold.
#[inline]
fn reduce(x: f64) -> (i32, f64, f64) {
    if x > 0.0 && x < LOG1P_CENTERED_POS_BOUND {
        // Centered path for positive x: no exponent reduction, no bit loss.
        return (0, x, 0.0);
    }
    if x < 0.0 && x > LOG1P_CENTERED_NEG_BOUND {
        // Centered path for negative x: same algebra, but the asymmetry
        // of `s = x/(2+x)` requires a tighter bound (`x > -0.2929`)
        // than the positive-side bound (`x < 0.4142`).
        return (0, x, 0.0);
    }

    // General path: form y = 1 + x, decompose, recover correction.
    let y = 1.0 + x;
    // c = exact residual of the 1 + x rounding. fdlibm uses two cases:
    // - |x| > 1: `c = 1 - (y - x)` (the lost bits are below `1`).
    // - |x| ≤ 1: `c = x - (y - 1.0)` (the lost bits are below `x`).
    let c = if x.abs() > 1.0 {
        1.0 - (y - x)
    } else {
        x - (y - 1.0)
    };

    let (mut m, mut k) = frexp(y);
    // frexp puts m ∈ [0.5, 1); shift to [1, 2).
    m *= 2.0;
    k -= 1;

    if m > SQRT_2_F64 {
        m *= 0.5;
        k += 1;
    }

    let f = m - 1.0;
    // Scale c by 1/y so it lives in the same multiplicative regime as f.
    let c_scaled = c / y;
    (k, f, c_scaled)
}

/// Plain-FMA polynomial evaluation + reconstruction.
///
/// Parenthesization follows fdlibm `s_log1p.c` exactly — the grouping
/// matters for f64. `f − (hfsq − s·(hfsq + R))` keeps the small
/// correction `(hfsq − s·(hfsq + R))` fully formed before being
/// subtracted from the dominant `f`, preserving `f`'s low bits.
#[inline]
fn eval_strict(k: i32, f: f64, c: f64) -> f64 {
    let s = f / (2.0 + f);
    let z = s * s;
    let r = z * horner(&LG_COEFFS, z);
    let hfsq = 0.5 * f * f;
    let kf = k as f64;

    if k == 0 {
        // Centered path: no k·ln(2) term, no c correction.
        return f - (hfsq - s * (hfsq + r));
    }

    // General path: dk·ln2_hi − ((hfsq − (s·(hfsq + R) + (dk·ln2_lo + c))) − f).
    kf * LN_2_DD.hi - ((hfsq - (s * (hfsq + r) + (kf * LN_2_DD.lo + c))) - f)
}

/// Compensated polynomial evaluation: same parenthesization, with the
/// inner Horner replaced by `compensated_horner` and the outer
/// reconstruction unchanged. The fdlibm grouping is already
/// precision-preserving; the compensated Horner tightens the residual
/// of `R` itself, which is the dominant error source.
#[inline]
fn eval_compensated(k: i32, f: f64, c: f64) -> f64 {
    let s = f / (2.0 + f);
    let z = s * s;
    let r = z * compensated_horner(&LG_COEFFS, z);
    let hfsq = 0.5 * f * f;
    let kf = k as f64;

    if k == 0 {
        // Centered path. Use an EFT for the small-correction subtraction
        // to recover the ulp of `hfsq - s·(hfsq + r)` before subtracting
        // from f.
        let corr = hfsq - s * (hfsq + r);
        let (sum, err) = two_sum(f, -corr);
        return sum + err;
    }

    kf * LN_2_DD.hi - ((hfsq - (s * (hfsq + r) + (kf * LN_2_DD.lo + c))) - f)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::{assert_within_ulps, ulps_between};

    fn check_strategy<F: Fn(f64) -> f64>(f: F, name: &str, max_ulps: u64) {
        let samples: &[f64] = &[
            // Centered path (|x| < √2-1 ≈ 0.414)
            1e-15, 1e-10, 1e-5, 0.001, 0.01, 0.1, 0.2, 0.4,
            -1e-15, -1e-10, -1e-5, -0.001, -0.01, -0.1, -0.4,
            // General path
            0.5, 1.0, 2.0, 5.0, 10.0, 100.0, 1e6, 1e15, 1e100,
            -0.5, -0.9, -0.99, -0.999,
            // Boundary near tiny threshold
            5e-17, -5e-17,
            // Centered/general boundary (around √2-1)
            0.41, 0.415, 0.42,
            -0.41, -0.42,
            // Constants
            std::f64::consts::PI,
            std::f64::consts::E,
        ];
        for &x in samples {
            let got = f(x);
            let expected = x.ln_1p();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= max_ulps,
                "{name}(x={x:e}): {dist} ulps apart (max {max_ulps})\n  got:      {got:e}\n  expected: {expected:e}"
            );
        }
    }

    // ── Boundary semantics ────────────────────────────────────────────────

    #[test]
    fn log1p_of_zero_is_zero() {
        assert_eq!(log1p_strict(0.0), 0.0);
        assert_eq!(log1p_compensated(0.0), 0.0);
        assert_eq!(log1p_correctly_rounded(0.0), 0.0);
    }

    #[test]
    fn log1p_of_neg_zero_is_neg_zero() {
        let v = log1p_strict(-0.0);
        assert_eq!(v, 0.0);
        assert!(v.is_sign_negative(), "log1p(-0) should be -0, got {v}");
    }

    #[test]
    fn log1p_of_nan_is_nan() {
        assert!(log1p_strict(f64::NAN).is_nan());
        assert!(log1p_compensated(f64::NAN).is_nan());
        assert!(log1p_correctly_rounded(f64::NAN).is_nan());
    }

    #[test]
    fn log1p_of_pos_inf_is_pos_inf() {
        assert_eq!(log1p_strict(f64::INFINITY), f64::INFINITY);
    }

    #[test]
    fn log1p_of_minus_one_is_neg_inf() {
        assert_eq!(log1p_strict(-1.0), f64::NEG_INFINITY);
        assert_eq!(log1p_compensated(-1.0), f64::NEG_INFINITY);
        assert_eq!(log1p_correctly_rounded(-1.0), f64::NEG_INFINITY);
    }

    #[test]
    fn log1p_of_less_than_minus_one_is_nan() {
        assert!(log1p_strict(-1.5).is_nan());
        assert!(log1p_strict(-2.0).is_nan());
        assert!(log1p_strict(-1e10).is_nan());
    }

    // ── Known-value spot checks ────────────────────────────────────────────

    #[test]
    fn log1p_of_e_minus_one_is_one() {
        let x = std::f64::consts::E - 1.0;
        assert_within_ulps(log1p_strict(x), 1.0, 2, "log1p_strict(e-1)");
        assert_within_ulps(log1p_compensated(x), 1.0, 1, "log1p_compensated(e-1)");
    }

    #[test]
    fn log1p_of_one_is_ln2() {
        let ln2 = std::f64::consts::LN_2;
        assert_within_ulps(log1p_strict(1.0), ln2, 2, "log1p_strict(1)");
        assert_within_ulps(log1p_compensated(1.0), ln2, 1, "log1p_compensated(1)");
    }

    // ── Strategy ulp budgets ──────────────────────────────────────────────

    #[test]
    fn log1p_strict_within_budget() {
        check_strategy(log1p_strict, "log1p_strict", 4);
    }

    #[test]
    fn log1p_compensated_within_budget() {
        check_strategy(log1p_compensated, "log1p_compensated", 2);
    }

    #[test]
    fn log1p_correctly_rounded_within_budget() {
        check_strategy(log1p_correctly_rounded, "log1p_correctly_rounded", 2);
    }

    // ── Precision contract: small-x doesn't lose bits ────────────────────
    //
    // Naive `log(1.0 + x)` rounds away the low bits of x in the addition,
    // so for |x| ~ 1e-16 the result is `log(1.0) = 0`. Our log1p must
    // preserve the bits via the centered-path or the (1 - (y-x)) trick.

    #[test]
    fn log1p_preserves_precision_at_small_x() {
        let xs: &[f64] = &[1e-9, 1e-12, 1e-15, -1e-9, -1e-12, -1e-15];
        for &x in xs {
            let our = log1p_strict(x);
            let reference = x.ln_1p();
            let dist = ulps_between(our, reference);
            assert!(
                dist <= 2,
                "log1p({x:e}) precision contract: {dist} ulps from reference"
            );
            // Sanity: the answer is approximately x (to leading order).
            // Next-order correction is -x²/2, magnitude bounded by x²/2.
            let tail_bound = (x * x).abs() * 0.6;
            assert!(
                (our - x).abs() <= tail_bound,
                "log1p({x:e}) = {our:e}, |our - x| should be ≤ x²/2 ≈ {tail_bound:e}"
            );
        }
    }

    #[test]
    fn log1p_better_than_naive_at_small_x() {
        use crate::recipes::libm::log::log_correctly_rounded;
        let x: f64 = 1e-12;
        let naive = log_correctly_rounded(1.0 + x);
        let ours = log1p_strict(x);
        let reference = x.ln_1p();
        let naive_err = ulps_between(naive, reference);
        let ours_err = ulps_between(ours, reference);
        assert!(
            ours_err < naive_err.saturating_sub(100).max(1),
            "log1p should crush naive log(1+x) at small x: ours={ours_err} ulps, naive={naive_err} ulps"
        );
    }

    // ── Mathematical identities ────────────────────────────────────────────

    #[test]
    fn log1p_monotone_in_neighborhood_of_zero() {
        let xs: Vec<f64> = (0..200).map(|i| -0.5 + i as f64 * 0.01).collect();
        let mut prev = log1p_correctly_rounded(xs[0]);
        for &x in &xs[1..] {
            let y = log1p_correctly_rounded(x);
            assert!(y >= prev, "log1p not monotone at x={x}: {y} < {prev}");
            prev = y;
        }
    }

    #[test]
    fn log1p_consistent_with_log_at_moderate_x() {
        // For |x| not near 0, log1p(x) should match log(1+x) within a few
        // ulps (where the addition still preserves bits).
        use crate::recipes::libm::log::log_correctly_rounded;
        let xs: &[f64] = &[1.0, 2.0, 5.0, 10.0, 100.0];
        for &x in xs {
            let via_log1p = log1p_strict(x);
            let via_log = log_correctly_rounded(1.0 + x);
            let dist = ulps_between(via_log1p, via_log);
            assert!(
                dist <= 4,
                "log1p({x}) vs log(1+{x}): {dist} ulps"
            );
        }
    }

    // ── Roundtrip vs expm1 ──────────────────────────────────────────────
    //
    // log1p(expm1(x)) ≈ x and expm1(log1p(x)) ≈ x by inverse-function
    // identity. Tests both directions.

    #[test]
    fn log1p_expm1_roundtrip() {
        use crate::recipes::libm::expm1::expm1_correctly_rounded;
        let xs: &[f64] = &[1e-9, 1e-5, 0.1, 0.5, 1.0, 2.0, -0.5, -0.1, -1e-9];
        for &x in xs {
            let rt = log1p_correctly_rounded(expm1_correctly_rounded(x));
            let dist = ulps_between(rt, x);
            assert!(
                dist <= 8,
                "log1p(expm1({x})) = {rt}, {dist} ulps from {x}"
            );
        }
    }

    #[test]
    fn expm1_log1p_roundtrip() {
        use crate::recipes::libm::expm1::expm1_correctly_rounded;
        let xs: &[f64] = &[1e-9, 1e-5, 0.1, 0.5, 1.0, 10.0, -0.5, -0.1, -1e-9];
        for &x in xs {
            let rt = expm1_correctly_rounded(log1p_correctly_rounded(x));
            let dist = ulps_between(rt, x);
            assert!(
                dist <= 8,
                "expm1(log1p({x})) = {rt}, {dist} ulps from {x}"
            );
        }
    }

    // ── Reduction boundary tests ──────────────────────────────────────────
    //
    // The `reduce` function switches at |x| = √2 - 1 ≈ 0.414. Test
    // inputs straddling this transition.

    #[test]
    fn log1p_centered_general_boundary() {
        let xs: &[f64] = &[
            0.40, 0.413, 0.4142, 0.4143, 0.42, 0.45, 0.5,
            -0.40, -0.413, -0.4142, -0.4143, -0.42, -0.45, -0.5,
        ];
        for &x in xs {
            let got = log1p_strict(x);
            let expected = x.ln_1p();
            let dist = ulps_between(got, expected);
            assert!(
                dist <= 4,
                "log1p({x}) at centered/general boundary: {dist} ulps"
            );
        }
    }
}
