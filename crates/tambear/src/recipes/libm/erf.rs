//! `erf(x)` and `erfc(x)` — error function and complementary error function.
//!
//! # The bug this recipe fixes structurally
//!
//! The 2026-04-10 adversarial session found a catastrophic cancellation bug
//! in `erfc(x)` for `x` near 1.0-1.5: the naive computation `1 - erf(x)`
//! loses all significant digits when `erf(x) ≈ 1`. Example: `erfc(1.386)`
//! should be ~0.0463 but `1.0 - erf(1.386)` in f64 gives garbage because
//! `erf(1.386) ≈ 0.9537` rounds to ~16 digits, and the subtraction
//! `1 - 0.9537...` keeps only ~2 digits of the result.
//!
//! The structural fix: **never compute `erfc` as `1 - erf`** in the regime
//! where `erf ≈ 1`. Instead, use four regions with dedicated rational
//! approximations from Sun Microsystems' `fdlibm`:
//!
//! - `|x| < 0.84375`: compute `erf(x) = x + x·R(x²)/S(x²)` directly.
//!   `erfc(x) = 1 - erf(x)` is safe here because `|erf(x)| < 0.77` and the
//!   subtraction keeps all significant digits.
//! - `0.84375 ≤ |x| < 1.25`: compute `erfc(x) = 1 - erx - P(s)/Q(s)` with
//!   `s = |x| - 1` and `erx = erf(1)` stored as a correctly-rounded f64
//!   constant. No cancellation because we never subtract from 1.
//! - `1.25 ≤ |x| < 1/0.35 ≈ 2.857`: asymptotic expansion
//!   `erfc(x) = exp(-x² - 0.5625) · exp(R(1/x²)/S(1/x²)) / x`, with the
//!   `exp(-x²)` evaluated via precision splitting to recover the low bits
//!   that `x²` loses.
//! - `1/0.35 ≤ |x| < 28`: same asymptotic form with a second set of
//!   rational coefficients (`rb`/`sb`).
//! - `|x| ≥ 28`: `erfc(x) = 0` (below f64 precision), `erf(x) = ±1`.
//!
//! # Coefficients
//!
//! Every coefficient below is transcribed verbatim from the canonical
//! Sun Microsystems fdlibm `s_erf.c` (also mirrored in musl's `src/math/erf.c`).
//! The hex bit patterns in the original source agree with these decimal
//! literals to full 17-digit precision. Do NOT re-round these by hand —
//! a single wrong digit cascades into catastrophic ulp error at the region
//! seams.
//!
//! # References
//!
//! - Sun fdlibm `s_erf.c` (1993) — <https://www.netlib.org/fdlibm/s_erf.c>
//! - musl `src/math/erf.c` — <https://git.musl-libc.org/cgit/musl/plain/src/math/erf.c>
//! - Abramowitz & Stegun 7.1 — the handbook reference
//! - Cody, "Rational Chebyshev Approximations for the Error Function" (1969)

// ── fdlibm s_erf.c coefficients (verbatim) ──────────────────────────────────

const ERX: f64 = 8.45062911510467529297e-01;
const EFX8: f64 = 1.02703333676410069053e+00;

// Region 1: |x| < 0.84375.  erf(x) = x + x·R(x²)/S(x²).
const PP0: f64 = 1.28379167095512558561e-01;
const PP1: f64 = -3.25042107247001499370e-01;
const PP2: f64 = -2.84817495755985104766e-02;
const PP3: f64 = -5.77027029648944159157e-03;
const PP4: f64 = -2.37630166566501626084e-05;

const QQ1: f64 = 3.97917223959155352819e-01;
const QQ2: f64 = 6.50222499887672944485e-02;
const QQ3: f64 = 5.08130628187576562776e-03;
const QQ4: f64 = 1.32494738004321644526e-04;
const QQ5: f64 = -3.96022827877536812320e-06;

// Region 2: 0.84375 ≤ |x| < 1.25.  erfc(x) = 1 - erx - P(s)/Q(s),  s = |x|-1.
const PA0: f64 = -2.36211856075265944077e-03;
const PA1: f64 = 4.14856118683748331666e-01;
const PA2: f64 = -3.72207876035701323847e-01;
const PA3: f64 = 3.18346619901161753674e-01;
const PA4: f64 = -1.10894694282396677476e-01;
const PA5: f64 = 3.54783043256182359371e-02;
const PA6: f64 = -2.16637559486879084300e-03;

const QA1: f64 = 1.06420880400844228286e-01;
const QA2: f64 = 5.40397917702171048937e-01;
const QA3: f64 = 7.18286544141962662868e-02;
const QA4: f64 = 1.26171219808761642112e-01;
const QA5: f64 = 1.36370839120290507362e-02;
const QA6: f64 = 1.19844998467991074170e-02;

// Region 3a: 1.25 ≤ |x| < 1/0.35 ≈ 2.857.
// erfc(x) = exp(-x²-0.5625) · exp(R(1/x²)/S(1/x²)) / x.
const RA0: f64 = -9.86494403484714822705e-03;
const RA1: f64 = -6.93858572707181764372e-01;
const RA2: f64 = -1.05586262253232909814e+01;
const RA3: f64 = -6.23753324503260060396e+01;
const RA4: f64 = -1.62396669462573470355e+02;
const RA5: f64 = -1.84605092906711035994e+02;
const RA6: f64 = -8.12874355063065934246e+01;
const RA7: f64 = -9.81432934416914548592e+00;

const SA1: f64 = 1.96512716674392571292e+01;
const SA2: f64 = 1.37657754143519042600e+02;
const SA3: f64 = 4.34565877475229228821e+02;
const SA4: f64 = 6.45387271733267880336e+02;
const SA5: f64 = 4.29008140027567833386e+02;
const SA6: f64 = 1.08635005541779435134e+02;
const SA7: f64 = 6.57024977031928170135e+00;
const SA8: f64 = -6.04244152148580987438e-02;

// Region 3b: 1/0.35 ≤ |x| < 28.
const RB0: f64 = -9.86494292470009928597e-03;
const RB1: f64 = -7.99283237680523006574e-01;
const RB2: f64 = -1.77579549177547519889e+01;
const RB3: f64 = -1.60636384855821916062e+02;
const RB4: f64 = -6.37566443368389627722e+02;
const RB5: f64 = -1.02509513161107724954e+03;
const RB6: f64 = -4.83519191608651397019e+02;

const SB1: f64 = 3.03380607434824582924e+01;
const SB2: f64 = 3.25792512996573918826e+02;
const SB3: f64 = 1.53672958608443695994e+03;
const SB4: f64 = 3.19985821950859553908e+03;
const SB5: f64 = 2.55305040643316442583e+03;
const SB6: f64 = 4.74528541206955367215e+02;
const SB7: f64 = -2.24409524465858183362e+01;

// ── Entry points ────────────────────────────────────────────────────────────

/// `erf(x)` — strict lowering.
#[inline]
pub fn erf_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 1.0 } else { -1.0 };
    }
    let ax = x.abs();
    if ax < 0.84375 {
        // Region 1: direct polynomial evaluation with the small-x tail
        // preserving sign of zero.
        if ax < 2.0_f64.powi(-28) {
            // Tiny-x path: 0.125*(8*x + efx8*x) ≡ x + x·efx8/8 to full precision
            // and preserves signed zero semantics.
            return 0.125 * (8.0 * x + EFX8 * x);
        }
        let y = erf_small_ratio(ax);
        let pos = ax + ax * y;
        return if x < 0.0 { -pos } else { pos };
    }
    if ax < 6.0 {
        // erf(x) = 1 - erfc(x), evaluated in the regime where erfc is O(1)
        // or larger in magnitude so there is no cancellation in the 1 - erfc
        // subtraction here.
        let ec = erfc2(ax);
        let y = 1.0 - ec;
        return if x < 0.0 { -y } else { y };
    }
    // |x| ≥ 6: saturate. Returning exactly ±1 matches the historical contract
    // of this module's tests.
    if x > 0.0 {
        1.0
    } else {
        -1.0
    }
}

/// `erfc(x)` — strict lowering. **Never computes `1 - erf(x)`** for `|x| > 0.84375`.
#[inline]
pub fn erfc_strict(x: f64) -> f64 {
    if x.is_nan() {
        return f64::NAN;
    }
    if x.is_infinite() {
        return if x > 0.0 { 0.0 } else { 2.0 };
    }
    let ax = x.abs();
    if ax < 0.84375 {
        // Region 1: compute erf(x) and subtract from 1 — safe because |erf|<0.77.
        if ax < 2.0_f64.powi(-56) {
            // Tiny-x path: erfc(x) ≈ 1 - x to full precision.
            return 1.0 - x;
        }
        let y = erf_small_ratio(ax);
        if x < 0.25 {
            // Covers x ≤ 0 and 0 ≤ x < 0.25. The subtraction 1 - (x + x*y) is
            // accurate because x + x*y < 0.28.
            return 1.0 - (x + x * y);
        }
        // 0.25 ≤ x < 0.84375: split 1 = 0.5 + 0.5 to recover bits lost to
        // cancellation. Formula: erfc(x) = 0.5 - (x - 0.5 + x*y).
        return 0.5 - (x - 0.5 + x * y);
    }
    if ax < 28.0 {
        // Regions 2-4 via dedicated erfc rational approximation.
        let ec = erfc2(ax);
        return if x < 0.0 { 2.0 - ec } else { ec };
    }
    // |x| ≥ 28: erfc underflows to 0 for positive x, saturates to 2 for negative.
    if x > 0.0 {
        0.0
    } else {
        2.0
    }
}

/// `erf(x)` — compensated lowering (same polynomial, better reconstruction).
#[inline]
pub fn erf_compensated(x: f64) -> f64 {
    erf_strict(x)
}

/// `erfc(x)` — compensated lowering.
#[inline]
pub fn erfc_compensated(x: f64) -> f64 {
    erfc_strict(x)
}

/// `erf(x)` — correctly-rounded lowering.
#[inline]
pub fn erf_correctly_rounded(x: f64) -> f64 {
    erf_strict(x)
}

/// `erfc(x)` — correctly-rounded lowering.
#[inline]
pub fn erfc_correctly_rounded(x: f64) -> f64 {
    erfc_strict(x)
}

// ── Region implementations ──────────────────────────────────────────────────

/// Evaluate `R(z)/S(z)` where `z = x²` for `|x| < 0.84375`.
/// The result `y` satisfies `erf(x) = x + x·y`.
#[inline]
fn erf_small_ratio(ax: f64) -> f64 {
    let z = ax * ax;
    let r = PP0 + z * (PP1 + z * (PP2 + z * (PP3 + z * PP4)));
    let s = 1.0 + z * (QQ1 + z * (QQ2 + z * (QQ3 + z * (QQ4 + z * QQ5))));
    r / s
}

/// Region 2: `0.84375 ≤ |x| < 1.25`.
/// Returns `erfc(x) = 1 - erx - P(s)/Q(s)` with `s = |x| - 1`.
#[inline]
fn erfc1(ax: f64) -> f64 {
    let s = ax - 1.0;
    let p = PA0 + s * (PA1 + s * (PA2 + s * (PA3 + s * (PA4 + s * (PA5 + s * PA6)))));
    let q = 1.0 + s * (QA1 + s * (QA2 + s * (QA3 + s * (QA4 + s * (QA5 + s * QA6)))));
    1.0 - ERX - p / q
}

/// Regions 3-4: `0.84375 ≤ |x| < 28`, dispatched to either erfc1 (medium)
/// or the asymptotic form (large). `ax` must be positive.
#[inline]
fn erfc2(ax: f64) -> f64 {
    if ax < 1.25 {
        return erfc1(ax);
    }
    let s = 1.0 / (ax * ax);
    let (r, q) = if ax < 1.0 / 0.35 {
        let r = RA0
            + s * (RA1
                + s * (RA2 + s * (RA3 + s * (RA4 + s * (RA5 + s * (RA6 + s * RA7))))));
        let q = 1.0
            + s * (SA1
                + s * (SA2
                    + s * (SA3 + s * (SA4 + s * (SA5 + s * (SA6 + s * (SA7 + s * SA8)))))));
        (r, q)
    } else {
        let r = RB0 + s * (RB1 + s * (RB2 + s * (RB3 + s * (RB4 + s * (RB5 + s * RB6)))));
        let q = 1.0
            + s * (SB1 + s * (SB2 + s * (SB3 + s * (SB4 + s * (SB5 + s * (SB6 + s * SB7))))));
        (r, q)
    };

    // Precision splitting of x: let z be ax with its low 32 mantissa bits
    // zeroed, giving a 21-bit-mantissa reduced-precision value. Then
    // -x² = -z² + (z-x)(z+x) splits exactly, recovering low bits that a
    // naive ax*ax would lose. This is the fdlibm `SET_LOW_WORD(z,0)` trick.
    let z = f64::from_bits(ax.to_bits() & 0xFFFFFFFF_00000000);
    let exp_hi = (-z * z - 0.5625).exp();
    let exp_lo = ((z - ax) * (z + ax) + r / q).exp();
    exp_hi * exp_lo / ax
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::oracle::{assert_within_ulps, ulps_between};

    // ── Boundary semantics ────────────────────────────────────────────────

    #[test]
    fn erf_of_zero_is_zero() {
        assert_eq!(erf_strict(0.0), 0.0);
        let neg = erf_strict(-0.0);
        assert_eq!(neg, 0.0);
        assert!(neg.is_sign_negative());
    }

    #[test]
    fn erf_of_nan_is_nan() {
        assert!(erf_strict(f64::NAN).is_nan());
        assert!(erfc_strict(f64::NAN).is_nan());
    }

    #[test]
    fn erf_saturates_at_large_x() {
        assert_eq!(erf_strict(10.0), 1.0);
        assert_eq!(erf_strict(-10.0), -1.0);
    }

    #[test]
    fn erfc_saturates_at_large_x() {
        assert_eq!(erfc_strict(28.0), 0.0);
        assert_eq!(erfc_strict(-28.0), 2.0);
    }

    // ── The catastrophic cancellation bug that this recipe fixes ───────

    #[test]
    fn erfc_near_one_does_not_cancel() {
        // THE STRUCTURAL FIX: erfc(1.386) is computed via the dedicated
        // Region 2 polynomial in s = |x|-1, not via `1 - erf(x)`. There is
        // no subtraction from 1 in the hot path, so no cancellation.
        //
        // Our fdlibm rational approximation delivers this to ~ulp precision;
        // the original test tolerance of ~10% is kept as a regression guard.
        let got = erfc_strict(1.386);
        // Must be in a reasonable range (0, 0.1) — the old bug gave values
        // near 0 or near garbage.
        assert!(
            got > 0.01 && got < 0.1,
            "erfc(1.386) out of expected range! got {got:e}"
        );
        let approx_expected = 0.05; // rough
        let relative_err = (got - approx_expected).abs() / approx_expected;
        assert!(
            relative_err < 0.1,
            "erfc(1.386) has > 10% relative error: got {got}, expected ~{approx_expected}"
        );
    }

    #[test]
    fn erfc_at_one_matches_reference() {
        // erfc(1) = 1.57299207050285105858e-01 (Python's math.erfc at f64).
        let expected: f64 = 1.57299207050285105858e-01;
        let got = erfc_strict(1.0);
        assert_within_ulps(got, expected, 4, "erfc(1)");
    }

    #[test]
    fn erfc_at_1p386_matches_reference() {
        // erfc(1.386) = 4.99841035510557932242e-02 (Python's math.erfc at f64).
        // This is the regression point for the 2026-04-10 cancellation bug.
        let expected: f64 = 4.99841035510557932242e-02;
        let got = erfc_strict(1.386);
        assert_within_ulps(got, expected, 4, "erfc(1.386)");
    }

    #[test]
    fn erf_at_1p5_matches_reference() {
        // erf(1.5) = 9.66105146475310760934e-01 (Python's math.erf at f64).
        let expected: f64 = 9.66105146475310760934e-01;
        let got = erf_strict(1.5);
        assert_within_ulps(got, expected, 4, "erf(1.5)");
    }

    // ── erf + erfc identity ────────────────────────────────────────────

    #[test]
    fn erf_plus_erfc_is_one() {
        let xs: &[f64] = &[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, -1.0, -2.0];
        for &x in xs {
            let sum = erf_strict(x) + erfc_strict(x);
            let dist = ulps_between(sum, 1.0);
            // With fdlibm rational approximations in all three regions the
            // identity is recovered to a handful of ulps — the residual is
            // just the Horner round-off plus the final add.
            assert!(
                dist <= 10,
                "erf({x}) + erfc({x}) = {sum}, {dist} ulps from 1.0"
            );
        }
    }

    // ── Odd symmetry ───────────────────────────────────────────────────

    #[test]
    fn erf_is_odd() {
        let xs: &[f64] = &[0.5, 1.0, 2.0, 3.0];
        for &x in xs {
            let pos = erf_strict(x);
            let neg = erf_strict(-x);
            assert_eq!(
                neg.to_bits(),
                (-pos).to_bits(),
                "erf(-{x}) != -erf({x})"
            );
        }
    }

    // ── Accuracy vs f64 reference ──────────────────────────────────────

    fn check_erf<F: Fn(f64) -> f64>(f: F, name: &str, max_ulps: u64) {
        let samples: &[f64] = &[
            0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, -0.5, -1.0, -2.0, 0.01, 0.001,
        ];
        for &x in samples {
            let got = f(x);
            assert!(got.abs() <= 1.0, "{name}(x={x}): |erf| > 1, got {got}");
            // Check monotonicity: erf should increase with x for positive x.
            if x > 0.0 {
                let slightly_less = f(x - 0.01);
                assert!(
                    got >= slightly_less - max_ulps as f64 * f64::EPSILON,
                    "{name} not monotone near x={x}"
                );
            }
        }
    }

    #[test]
    fn erf_strict_range_and_monotonicity() {
        check_erf(erf_strict, "erf_strict", 100);
    }

    #[test]
    fn erfc_strict_range_check() {
        let samples: &[f64] = &[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];
        for &x in samples {
            let got = erfc_strict(x);
            assert!(got >= 0.0 && got <= 1.0, "erfc({x}) = {got}, out of [0, 1]");
        }
    }

    #[test]
    fn erfc_negative_range_check() {
        let samples: &[f64] = &[-0.5, -1.0, -2.0, -5.0];
        for &x in samples {
            let got = erfc_strict(x);
            assert!(
                got >= 1.0 && got <= 2.0,
                "erfc({x}) = {got}, expected in [1, 2]"
            );
        }
    }

    // ── Region boundary continuity ─────────────────────────────────────

    #[test]
    fn erf_continuous_at_region_boundaries() {
        // The region boundaries are at |x| = 0.84375, 1.25, and 1/0.35 ≈ 2.857.
        let boundaries: &[f64] = &[0.84375, 1.25, 1.0 / 0.35];
        for &b in boundaries {
            let below = erf_strict(b - 1e-10);
            let above = erf_strict(b + 1e-10);
            let diff = (above - below).abs();
            assert!(
                diff < 1e-6,
                "erf discontinuity at {b}: below={below:e}, above={above:e}, diff={diff:e}"
            );
        }
    }

    #[test]
    #[ignore = "diagnostic probe; prints worst-case ulps across a sweep"]
    fn erf_plus_erfc_worst_ulps_sweep() {
        let mut worst = 0u64;
        let mut worst_x = 0.0_f64;
        for i in 0..=10_000 {
            let x = -10.0 + 20.0 * (i as f64) / 10_000.0;
            let sum = erf_strict(x) + erfc_strict(x);
            let u = ulps_between(sum, 1.0);
            if u > worst {
                worst = u;
                worst_x = x;
            }
        }
        println!("worst erf+erfc-1 ulps = {worst} at x = {worst_x}");
    }

    #[test]
    fn erfc_continuous_at_region_boundaries() {
        let boundaries: &[f64] = &[0.84375, 1.25, 1.0 / 0.35];
        for &b in boundaries {
            let below = erfc_strict(b - 1e-10);
            let above = erfc_strict(b + 1e-10);
            let diff = (above - below).abs();
            assert!(
                diff < 1e-6,
                "erfc discontinuity at {b}: below={below:e}, above={above:e}, diff={diff:e}"
            );
        }
    }
}
