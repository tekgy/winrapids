//! `complex_log(z, policy)` — natural logarithm of a complex number.
//!
//! First complex-transcendental recipe in tambear. Per DEC-032
//! (`docs/architecture/branch-cut-conventions.md` ratified 2026-05-09),
//! every complex-transcendental recipe takes `BranchPolicy` as a
//! **non-defaulted** parameter. Absence at the call site is a compile
//! error, not a silent wrong answer — this is the F13.C antibody for
//! branch-cut conventions.
//!
//! # Mathematical recipe
//!
//! For `z = x + iy ≠ 0`:
//!
//! ```text
//! log(z) = ln|z| + i·arg(z)
//!        = (1/2)·log(x² + y²) + i·atan2(y, x)
//! ```
//!
//! Implementation:
//! - **Real part**: `0.5 · log(x² + y²)`, computed precision-safely via
//!   `0.5 · log1p((|z|² − 1) when z is near unit circle)` or via the
//!   straight `log(hypot(x, y))` form elsewhere. We use the simpler
//!   `log(hypot(x, y))` form here — `hypot` already handles
//!   underflow/overflow safely.
//! - **Imaginary part**: `atan2(y, x)`, which by IEEE 754 and Kahan 1987
//!   already encodes the principal-value cut along the negative real axis
//!   with the correct sign-of-zero behavior. `atan2(+0, -x)` for `x > 0`
//!   returns `+π`; `atan2(-0, -x)` returns `−π`. This *is* the
//!   `Principal` policy.
//!
//! # Cut placement (Principal policy)
//!
//! Per DEC-032 sub-clause D-prime + C99 §G.6.3.4 + Kahan 1987 §3:
//!
//! - `clog(-1.0 + 0.0i) = +iπ` (cut approached from above)
//! - `clog(-1.0 - 0.0i) = -iπ` (cut approached from below)
//! - `clog(0.0 + 0.0i) = -∞ + 0.0i` (limit)
//! - `clog(0.0 - 0.0i) = -∞ - 0.0i` (limit, sign-preserving)
//!
//! The sign of zero on the input imaginary part determines the sign of
//! the output imaginary part at the cut. This is the "counter-clockwise
//! continuity" property — the unique cut placement compatible with
//! IEEE 754 signed-zero arithmetic.
//!
//! # Special cases
//!
//! - `clog(NaN + iy) = NaN + iNaN` for finite y
//! - `clog(x + iNaN) = NaN + iNaN` for finite x (some impls preserve x; we don't)
//! - `clog(+∞ + iy) = +∞ + i·atan2(y, +∞) = +∞ + i·0` for finite y > 0
//! - `clog(-∞ + iy) = +∞ + i·π` for finite y > 0
//! - `clog(z) = -∞ + i·arg(0)` when `|z| = 0`
//!
//! # Error budget
//!
//! Inherits from `log` (real-axis log of `|z|`) plus `atan2` (imaginary
//! part). Both are ≤ 2 ulps at the strict tier; total ≤ 4 ulps composed.
//!
//! # References
//!
//! - Kahan, "Branch cuts for complex elementary functions, or much ado
//!   about nothing's sign bit" (1987) — the foundational reference.
//! - C99 §G.6.3.4 / IEEE 754-2019 §9.2 — normative cut placements.
//! - `docs/architecture/branch-cut-conventions.md` — DEC-032 ratified.

use super::atan::atan2_strict;
use super::log::log_strict;
use super::hypot::hypot_strict;

/// Branch-cut convention for complex-transcendental recipes.
///
/// Per DEC-032 (ratified 2026-05-09, see
/// `docs/architecture/branch-cut-conventions.md`): every
/// complex-transcendental recipe takes a `BranchPolicy` as a
/// non-defaulted parameter. Absence at the call site is a compile
/// error.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum BranchPolicy {
    /// Kahan 1987 / C99 §G.6 / counter-clockwise-continuous (CCC).
    /// `clog(-1) = +iπ`. The unique cut placement compatible with
    /// IEEE 754 signed-zero. Recommended for all new code.
    Principal,

    /// Sign-conjugate of `Principal`: `clog(-1) = -iπ`. Same cut
    /// placement, but values on the cut have their imaginary parts
    /// negated. Provided for backward compatibility with FORTRAN-era
    /// conventions; not recommended for new code.
    AntiPrincipal,

    /// Branch picked per-call to minimize cancellation. Sacrifices
    /// identity-preservation across calls for per-call accuracy.
    /// For `complex_log` specifically this is identical to
    /// `Principal` (no cancellation issue) — defined for API
    /// uniformity across recipes that DO benefit from it
    /// (`complex_sqrt`, `complex_pow`).
    NumericallyStable,

    /// Enumerate all branches; result carries integer winding numbers.
    /// `complex_log` returns `WoundComplex { primary, windings }` —
    /// `primary` is the `Principal` value and `windings` lists
    /// `(k, primary + 2πi·k)` for `k ∈ [-max_windings, +max_windings]`.
    Discovery,
}

impl BranchPolicy {
    /// Stable tag byte for cache-key serialization. Tag `0` is
    /// reserved per DEC-032 sub-clause D to assert "byte not fed."
    /// Once shipped, these values must not change.
    #[inline]
    pub const fn tag(self) -> u8 {
        match self {
            BranchPolicy::Principal => 1,
            BranchPolicy::AntiPrincipal => 2,
            BranchPolicy::NumericallyStable => 3,
            BranchPolicy::Discovery => 4,
        }
    }
}

/// Minimal complex-number representation. Kept local to the
/// complex_log recipe rather than promoted to a crate-level
/// primitive — future complex-transcendental recipes
/// (`complex_sqrt`, `complex_pow`, `complex_arctan`, ...) will
/// motivate promotion to `primitives::complex::Complex`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    #[inline]
    pub const fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
}

/// Discovery output for `complex_log` under `BranchPolicy::Discovery`.
///
/// Per DEC-032 sub-clause E.1 (single-valued-on-cut family):
/// `complex_log` is naturally enumerable by integer winding number.
/// `primary` is the Principal branch; `windings` contains
/// `(winding_number, value)` pairs for non-zero winding indices.
#[derive(Debug, Clone, PartialEq)]
pub struct WoundComplex {
    /// Principal-branch value.
    pub primary: Complex,
    /// Non-zero winding values: `(k, primary + 2πi·k)` for `k ∈ [-max, max] \ {0}`.
    pub windings: Vec<(i64, Complex)>,
}

/// `clog(z, policy)` — complex natural logarithm.
///
/// The `policy` parameter is **non-defaulted** per DEC-032's F13.C
/// antibody. The caller must consciously pick a branch-cut
/// convention; there is no "cheap default."
///
/// Returns `Some(Complex)` for `Principal`, `AntiPrincipal`,
/// `NumericallyStable`; for `Discovery`, returns `None` — discovery
/// callers must use `clog_discovery` which returns the full
/// `WoundComplex` shape.
pub fn clog(z: Complex, policy: BranchPolicy) -> Complex {
    debug_assert!(
        !matches!(policy, BranchPolicy::Discovery),
        "use `clog_discovery` for BranchPolicy::Discovery"
    );

    let principal = clog_principal(z);
    match policy {
        BranchPolicy::Principal | BranchPolicy::NumericallyStable => principal,
        BranchPolicy::AntiPrincipal => Complex {
            re: principal.re,
            im: -principal.im,
        },
        BranchPolicy::Discovery => {
            // Caller violated the precondition. Return the primary
            // branch as a defensive default (debug_assert fired above).
            principal
        }
    }
}

/// `clog(z, Discovery, max_windings)` — discovery-policy variant
/// returning all enumerated branches.
///
/// `max_windings` controls the range: returns `(k, value)` for
/// `k ∈ [-max_windings, max_windings] \ {0}`. The principal branch
/// (k=0) lives separately in the `primary` field.
pub fn clog_discovery(z: Complex, max_windings: u32) -> WoundComplex {
    let primary = clog_principal(z);
    let two_pi = std::f64::consts::TAU;
    let mut windings = Vec::with_capacity(2 * max_windings as usize);
    for k in 1..=max_windings as i64 {
        let im_shift = two_pi * k as f64;
        windings.push((
            k,
            Complex {
                re: primary.re,
                im: primary.im + im_shift,
            },
        ));
        windings.push((
            -k,
            Complex {
                re: primary.re,
                im: primary.im - im_shift,
            },
        ));
    }
    WoundComplex { primary, windings }
}

/// Compute the principal-branch `log(z) = ln|z| + i·arg(z)`.
///
/// Uses `log(hypot(x, y))` for the real part — `hypot` is
/// already overflow/underflow-safe. Uses `atan2` for the imaginary
/// part, which encodes the principal cut along the negative real
/// axis with correct sign-of-zero behavior (Kahan 1987 §3).
#[inline]
fn clog_principal(z: Complex) -> Complex {
    // Special cases first.
    if z.re.is_nan() || z.im.is_nan() {
        return Complex::new(f64::NAN, f64::NAN);
    }
    if z.re == 0.0 && z.im == 0.0 {
        // log(0) = -∞ + i·arg(0). The atan2(±0, ±0) result preserves
        // sign of the imaginary input per IEEE 754 §9.2.1.
        return Complex::new(f64::NEG_INFINITY, atan2_strict(z.im, z.re));
    }
    if z.re.is_infinite() || z.im.is_infinite() {
        // log(±∞ + i·finite) = +∞ + i·atan2(y, x).
        // log(±∞ + i·∞) = +∞ + i·π/4 (or similar per IEEE).
        return Complex::new(f64::INFINITY, atan2_strict(z.im, z.re));
    }

    let r = hypot_strict(z.re, z.im);
    let re = log_strict(r);
    let im = atan2_strict(z.im, z.re);
    Complex::new(re, im)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Convenience: ulps_between for complex numbers, max over the two parts.
    fn complex_ulps(a: Complex, b: Complex) -> u64 {
        use crate::primitives::oracle::ulps_between;
        ulps_between(a.re, b.re).max(ulps_between(a.im, b.im))
    }

    // ── F13.C antibody — non-defaulted policy parameter ─────────────────
    //
    // This test documents that the antibody is structurally enforced via
    // a *required* parameter. Removing the policy parameter (making it
    // defaulted) would let callers skip the decision and silently get
    // Principal. The compile-time enforcement is the antibody.

    #[test]
    fn clog_requires_explicit_policy() {
        let z = Complex::new(1.0, 0.0);
        // Each policy must compile independently and produce a result.
        // (If we ever made policy defaulted, this test wouldn't catch
        // it directly — but the failing test would be "code that omits
        // the policy compiles," which the test below also wouldn't
        // catch. The antibody is at the function signature, not in tests.)
        let _ = clog(z, BranchPolicy::Principal);
        let _ = clog(z, BranchPolicy::AntiPrincipal);
        let _ = clog(z, BranchPolicy::NumericallyStable);
        // Discovery uses a different entry point:
        let _ = clog_discovery(z, 2);
    }

    // ── DEC-032 D-prime cut-placement adversarial tests ─────────────────
    //
    // From DEC-032 ratification doc § "Implementation cross-check":
    //   - clog(-1.0 + 0.0i) = +iπ   (cut approached from above)
    //   - clog(-1.0 - 0.0i) = -iπ   (cut approached from below)
    // This is the "sign-of-zero matters at the cut" property.

    #[test]
    fn clog_minus_one_plus_zero_i_is_plus_i_pi() {
        let z = Complex::new(-1.0, 0.0);
        let result = clog(z, BranchPolicy::Principal);
        assert_eq!(result.re, 0.0, "log|-1| = 0");
        let pi = std::f64::consts::PI;
        let dist = (result.im - pi).abs();
        assert!(
            dist < 1e-15,
            "clog(-1 + 0i) imaginary part should be +π, got {} (off by {})",
            result.im, dist
        );
    }

    #[test]
    fn clog_minus_one_minus_zero_i_is_minus_i_pi() {
        let z = Complex::new(-1.0, -0.0);
        let result = clog(z, BranchPolicy::Principal);
        assert_eq!(result.re, 0.0);
        let neg_pi = -std::f64::consts::PI;
        let dist = (result.im - neg_pi).abs();
        assert!(
            dist < 1e-15,
            "clog(-1 - 0i) imaginary part should be -π, got {} (off by {})",
            result.im, dist
        );
    }

    #[test]
    fn clog_sign_of_zero_at_cut_distinguishes_branches() {
        // The two limits MUST differ by 2π in their imaginary parts.
        // This is the structural identity that the F13.C antibody
        // guards against silently violating.
        let z_above = Complex::new(-1.0, 0.0);
        let z_below = Complex::new(-1.0, -0.0);
        let log_above = clog(z_above, BranchPolicy::Principal);
        let log_below = clog(z_below, BranchPolicy::Principal);
        let diff = log_above.im - log_below.im;
        let two_pi = std::f64::consts::TAU;
        let dist = (diff - two_pi).abs();
        assert!(
            dist < 1e-15,
            "clog discontinuity at cut should be exactly 2π, got {diff} (off by {dist})"
        );
    }

    // ── Cross-policy identity preservation ───────────────────────────────

    #[test]
    fn anti_principal_is_conjugate_of_principal() {
        // AntiPrincipal(z) = conj(Principal(z)) per DEC-032 D-prime.
        // The relationship holds when the imaginary part is non-zero;
        // for real-positive z both policies return the same value.
        let cases: &[Complex] = &[
            Complex::new(2.0, 3.0),
            Complex::new(-1.0, 5.0),
            Complex::new(-0.5, -0.5),
            Complex::new(1e10, 1.0),
        ];
        for &z in cases {
            let p = clog(z, BranchPolicy::Principal);
            let ap = clog(z, BranchPolicy::AntiPrincipal);
            assert_eq!(p.re, ap.re, "real parts must match for clog(z={z:?})");
            assert_eq!(p.im, -ap.im, "imaginary parts must be negated for clog(z={z:?})");
        }
    }

    #[test]
    fn numerically_stable_matches_principal_for_log() {
        // For clog specifically (no cancellation), NumericallyStable
        // and Principal produce identical results.
        let cases: &[Complex] = &[
            Complex::new(2.0, 3.0),
            Complex::new(-1.0, 0.5),
            Complex::new(0.1, 0.1),
        ];
        for &z in cases {
            let p = clog(z, BranchPolicy::Principal);
            let ns = clog(z, BranchPolicy::NumericallyStable);
            assert_eq!(p, ns, "Principal and NumericallyStable should agree for clog(z={z:?})");
        }
    }

    // ── Known-value spot checks ──────────────────────────────────────────

    #[test]
    fn clog_of_one_is_zero() {
        let result = clog(Complex::new(1.0, 0.0), BranchPolicy::Principal);
        assert_eq!(result.re, 0.0);
        assert_eq!(result.im, 0.0);
    }

    #[test]
    fn clog_of_i_is_i_pi_over_2() {
        let result = clog(Complex::new(0.0, 1.0), BranchPolicy::Principal);
        assert_eq!(result.re, 0.0);
        let pi_2 = std::f64::consts::FRAC_PI_2;
        assert!(
            (result.im - pi_2).abs() < 1e-15,
            "clog(i) should be iπ/2, got im={}",
            result.im
        );
    }

    #[test]
    fn clog_of_minus_i_is_minus_i_pi_over_2() {
        let result = clog(Complex::new(0.0, -1.0), BranchPolicy::Principal);
        assert_eq!(result.re, 0.0);
        let neg_pi_2 = -std::f64::consts::FRAC_PI_2;
        assert!(
            (result.im - neg_pi_2).abs() < 1e-15,
            "clog(-i) should be -iπ/2, got im={}",
            result.im
        );
    }

    #[test]
    fn clog_of_e_plus_zero_i_is_one() {
        // log(e + 0i) = 1 + 0i.
        let e = std::f64::consts::E;
        let result = clog(Complex::new(e, 0.0), BranchPolicy::Principal);
        assert!(
            (result.re - 1.0).abs() < 1e-15,
            "log(e + 0i) real part should be 1, got {}",
            result.re
        );
        assert_eq!(result.im, 0.0);
    }

    // ── Special-value handling ───────────────────────────────────────────

    #[test]
    fn clog_of_zero_is_neg_inf() {
        let z = Complex::new(0.0, 0.0);
        let result = clog(z, BranchPolicy::Principal);
        assert_eq!(result.re, f64::NEG_INFINITY);
    }

    #[test]
    fn clog_of_nan_returns_nan() {
        let z = Complex::new(f64::NAN, 1.0);
        let result = clog(z, BranchPolicy::Principal);
        assert!(result.re.is_nan());
        assert!(result.im.is_nan());
    }

    #[test]
    fn clog_of_infinity_returns_infinity_re() {
        let z = Complex::new(f64::INFINITY, 1.0);
        let result = clog(z, BranchPolicy::Principal);
        assert_eq!(result.re, f64::INFINITY);
    }

    // ── Identity: exp(log(z)) = z ────────────────────────────────────────
    //
    // For Principal policy, exp(clog(z)) should recover z within a few
    // ulps. Tambear's real-axis exp/log are correctly-rounded; the
    // composition through complex arithmetic introduces a few ulps.

    #[test]
    fn cexp_clog_identity_principal() {
        use super::super::exp::exp_strict;
        use super::super::sin::sin_strict;
        use super::super::sin::cos_strict;
        let cases: &[Complex] = &[
            Complex::new(2.0, 3.0),
            Complex::new(-1.5, 0.7),
            Complex::new(0.1, 0.1),
            Complex::new(10.0, 5.0),
        ];
        for &z in cases {
            let w = clog(z, BranchPolicy::Principal);
            // cexp(a + bi) = e^a · (cos b + i sin b)
            let e_re = exp_strict(w.re);
            let recovered = Complex::new(e_re * cos_strict(w.im), e_re * sin_strict(w.im));
            // Allow loose tolerance since this composes real-axis primitives.
            let re_err = (recovered.re - z.re).abs() / z.re.abs().max(1e-16);
            let im_err = (recovered.im - z.im).abs() / z.im.abs().max(1e-16);
            assert!(
                re_err < 1e-13 && im_err < 1e-13,
                "exp(log({z:?})) = {recovered:?}, relative errors: re={re_err:e}, im={im_err:e}"
            );
        }
    }

    // ── Discovery output ─────────────────────────────────────────────────

    #[test]
    fn discovery_returns_winding_numbers() {
        let z = Complex::new(1.0, 0.0); // log(1) = 0
        let wound = clog_discovery(z, 2);
        assert_eq!(wound.primary, Complex::new(0.0, 0.0));
        assert_eq!(wound.windings.len(), 4); // -2, -1, 1, 2

        // Each winding (k, v) should have v.im = primary.im + 2πk.
        let two_pi = std::f64::consts::TAU;
        for (k, v) in &wound.windings {
            assert_eq!(v.re, wound.primary.re);
            let expected_im = wound.primary.im + two_pi * (*k as f64);
            assert!(
                (v.im - expected_im).abs() < 1e-15,
                "winding k={k}: im={}, expected {expected_im}",
                v.im
            );
        }
    }

    #[test]
    fn discovery_with_zero_windings_returns_only_primary() {
        let z = Complex::new(1.0, 1.0);
        let wound = clog_discovery(z, 0);
        assert!(wound.windings.is_empty());
        assert_eq!(wound.primary, clog(z, BranchPolicy::Principal));
    }

    // ── BranchPolicy tag bytes per DEC-032 ───────────────────────────────

    #[test]
    fn branch_policy_tags_are_stable() {
        assert_eq!(BranchPolicy::Principal.tag(), 1);
        assert_eq!(BranchPolicy::AntiPrincipal.tag(), 2);
        assert_eq!(BranchPolicy::NumericallyStable.tag(), 3);
        assert_eq!(BranchPolicy::Discovery.tag(), 4);
    }

    #[test]
    fn branch_policy_tag_zero_reserved() {
        // Per DEC-032 sub-clause D: tag 0 is reserved for "byte not fed."
        // No variant should claim it. This test guards against future
        // refactors that might accidentally use 0.
        for policy in [
            BranchPolicy::Principal,
            BranchPolicy::AntiPrincipal,
            BranchPolicy::NumericallyStable,
            BranchPolicy::Discovery,
        ] {
            assert_ne!(policy.tag(), 0, "tag 0 is reserved for 'byte not fed'");
        }
    }
}
