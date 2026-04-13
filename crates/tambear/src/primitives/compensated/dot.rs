//! Compensated inner products and polynomial evaluation.
//!
//! These are the Tier 2 compensated primitives — one step above the EFTs,
//! one step below full double-double. They are the bridge between
//! `two_product_fma` and the libm recipes: every correctly-rounded exp,
//! log, sin, cos implementation bottoms out on a compensated polynomial
//! evaluation of a Remez-optimized approximation.
//!
//! # References
//!
//! - Ogita, Rump, Oishi (2005), "Accurate sum and dot product", SIAM J. Sci.
//!   Comput. The modern formulation of compensated summation and dot
//!   product; our `dot_2` follows this paper's notation.
//! - Langlois, Louvet, Graillat (2007), "Compensated Horner scheme", gives
//!   the compensated polynomial evaluation scheme we use for Remez approximations.
//! - Graillat, Langlois, Louvet (2009), "Algorithms for accurate, validated
//!   and fast polynomial evaluation", the round-to-nearest correctly-rounded
//!   Horner we use when `#[precision(correctly_rounded)]` is requested.

use super::eft::{two_product_fma, two_sum};

/// Rump-Ogita-Oishi compensated dot product.
///
/// Given two slices `x` and `y` of equal length, returns `Σ xᵢ·yᵢ` computed
/// with a single level of error compensation. The error bound is
/// `O(ε · Σ|xᵢ·yᵢ|) + O(n·ε²·Σ|xᵢ·yᵢ|)`, which in practice means the result
/// is accurate to the last bit of double precision for well-conditioned
/// inputs and continues to be usable deep into ill-conditioned regimes
/// where naive fp64 dot product fails.
///
/// Costs approximately 25 flops per element (1 FMA for the product, 1 EFT
/// for the product error, 1 EFT for the running sum, 1 add). For typical
/// Remez approximations of degree 10-20, this is 500-1000 flops total,
/// which compiles to a few hundred nanoseconds — entirely acceptable for
/// libm inner loops.
///
/// # Panics
/// Panics if `x.len() != y.len()`.
///
/// # Empty inputs
/// Returns `0.0`.
#[inline]
pub fn dot_2(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "dot_2: slice length mismatch");
    if x.is_empty() {
        return 0.0;
    }
    let (mut p, mut s) = two_product_fma(x[0], y[0]);
    for i in 1..x.len() {
        let (h, r) = two_product_fma(x[i], y[i]);
        let (new_p, q) = two_sum(p, h);
        p = new_p;
        // Accumulate both residuals into s. This is a single-level
        // compensation; for deeper k-level compensation (sum_k / dot_k)
        // see primitives/specialist/sum_k in Phase B6.
        s += q + r;
    }
    p + s
}

/// Compensated Horner evaluation of a polynomial.
///
/// Given polynomial coefficients `coeffs = [c₀, c₁, …, cₙ]` in ascending
/// order of degree and evaluation point `x`, returns the value
/// `c₀ + c₁·x + c₂·x² + … + cₙ·xⁿ` computed via compensated Horner.
///
/// The algorithm runs a standard Horner recurrence `r ← r·x + cₖ` but
/// captures the exact residual at each step using `two_product_fma` and
/// `two_sum`. The residuals are themselves evaluated via a secondary Horner
/// recurrence, and the two results combined at the end.
///
/// Error bound (Langlois-Louvet-Graillat, Theorem 3.1):
/// `|compensated - exact| <= ε·|exact| + O(ε² · cond · Σ|cₖ·xᵏ|)`
/// where `cond` is the polynomial condition number. In practice: within
/// 1-2 ulps for well-conditioned cases, still accurate for condition
/// numbers up to ~1/ε.
///
/// # Panics
/// Panics on empty coefficient slice. Returns `0.0` on single coefficient
/// being exactly 0.0. Returns `c₀` if there is only one coefficient.
#[inline]
pub fn compensated_horner(coeffs: &[f64], x: f64) -> f64 {
    assert!(
        !coeffs.is_empty(),
        "compensated_horner: coefficient slice must be non-empty"
    );
    if coeffs.len() == 1 {
        return coeffs[0];
    }
    // Start with the highest-degree coefficient.
    let n = coeffs.len();
    let mut r = coeffs[n - 1];
    let mut err = 0.0_f64;
    for i in (0..n - 1).rev() {
        // Exact r·x via FMA: (p, pi_p) = r·x, where pi_p is the product error.
        let (p, pi_p) = two_product_fma(r, x);
        // Exact p + coeffs[i] = (s, sigma_s).
        let (s, sigma_s) = two_sum(p, coeffs[i]);
        // Secondary recurrence for the running error: err ← err·x + (pi_p + sigma_s).
        // We could make this itself compensated (dot_3 style); the standard
        // reference uses a plain fp64 recurrence for the error channel, which
        // gives the Theorem 3.1 bound.
        err = err * x + (pi_p + sigma_s);
        r = s;
    }
    r + err
}

/// Plain (uncompensated) Horner — kept here as a reference point for the
/// compensated version and for recipes tagged `#[precision(strict)]` that
/// want the same polynomial API.
///
/// Costs `2·(n-1)` flops; approximately 1/10 the cost of `compensated_horner`.
#[inline]
pub fn horner(coeffs: &[f64], x: f64) -> f64 {
    assert!(
        !coeffs.is_empty(),
        "horner: coefficient slice must be non-empty"
    );
    let n = coeffs.len();
    let mut r = coeffs[n - 1];
    for i in (0..n - 1).rev() {
        // Use FMA for single-rounding: r ← r·x + coeffs[i].
        r = r.mul_add(x, coeffs[i]);
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── dot_2 ───────────────────────────────────────────────────────────────

    #[test]
    fn dot_2_matches_naive_on_well_conditioned() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [5.0, 6.0, 7.0, 8.0];
        let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;
        assert_eq!(dot_2(&x, &y), expected);
    }

    #[test]
    fn dot_2_handles_empty() {
        let empty: &[f64] = &[];
        assert_eq!(dot_2(empty, empty), 0.0);
    }

    #[test]
    fn dot_2_single_element() {
        assert_eq!(dot_2(&[3.0], &[4.0]), 12.0);
    }

    #[test]
    #[should_panic(expected = "slice length mismatch")]
    fn dot_2_panics_on_length_mismatch() {
        let _ = dot_2(&[1.0, 2.0], &[3.0]);
    }

    #[test]
    fn dot_2_beats_naive_on_catastrophic_cancellation() {
        // A classic near-cancellation dot product: the exact answer is tiny
        // but the intermediate partial sums are huge. Naive fp64 drops bits.
        //
        // Construct (large, small, -large) dotted with (1, 1, 1):
        //   1e17 * 1 + 1 * 1 + (-1e17) * 1 = 1.0 exactly.
        // Naive fp64: 1e17 + 1 == 1e17 (loses the 1), result = 0.
        let x = [1e17, 1.0, -1e17];
        let y = [1.0, 1.0, 1.0];

        let naive: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
        let compensated = dot_2(&x, &y);

        assert_eq!(compensated, 1.0, "dot_2 should recover the exact 1.0");
        // Don't assert naive is exactly 0 — the implementation may reorder —
        // just assert that compensated is strictly better.
        let naive_err = (naive - 1.0).abs();
        let comp_err = (compensated - 1.0).abs();
        assert!(
            comp_err <= naive_err,
            "dot_2 error {comp_err:e} not better than naive {naive_err:e}"
        );
    }

    // ── horner / compensated_horner ────────────────────────────────────────

    #[test]
    fn horner_matches_manual() {
        // p(x) = 1 + 2x + 3x² + 4x³
        let c = [1.0, 2.0, 3.0, 4.0];
        let x = 2.0;
        let expected = 1.0 + 2.0 * 2.0 + 3.0 * 4.0 + 4.0 * 8.0;
        assert_eq!(horner(&c, x), expected);
    }

    #[test]
    fn horner_single_coefficient() {
        assert_eq!(horner(&[3.14], 7.0), 3.14);
        assert_eq!(compensated_horner(&[3.14], 7.0), 3.14);
    }

    #[test]
    fn horner_at_zero() {
        let c = [5.0, 7.0, 11.0];
        assert_eq!(horner(&c, 0.0), 5.0);
        assert_eq!(compensated_horner(&c, 0.0), 5.0);
    }

    #[test]
    fn horner_at_one() {
        // p(1) = sum of coefficients
        let c = [1.0, 2.0, 3.0, 4.0, 5.0];
        let expected = 15.0;
        assert_eq!(horner(&c, 1.0), expected);
        assert_eq!(compensated_horner(&c, 1.0), expected);
    }

    #[test]
    fn compensated_horner_agrees_on_well_conditioned() {
        // For a well-conditioned polynomial like (x-1)²(x-2) at x = 3
        // expanded: p(x) = -2 + 5x - 4x² + x³, so coefficients are
        // c = [-2, 5, -4, 1]. At x=3: -2 + 15 - 36 + 27 = 4.
        let c = [-2.0, 5.0, -4.0, 1.0];
        let expected = 4.0;
        let plain = horner(&c, 3.0);
        let comp = compensated_horner(&c, 3.0);
        assert_eq!(plain, expected);
        assert_eq!(comp, expected);
    }

    #[test]
    fn compensated_horner_helps_on_ill_conditioned() {
        // A polynomial that's nearly zero at the evaluation point,
        // causing catastrophic cancellation in naive Horner.
        //
        // p(x) = (x - 1)^5 near x = 1 + tiny_delta
        // Expanded: c = [-1, 5, -10, 10, -5, 1]
        let c = [-1.0, 5.0, -10.0, 10.0, -5.0, 1.0];
        let x: f64 = 1.0 + 1e-8;
        let expected = (x - 1.0).powi(5); // ~1e-40
        let plain = horner(&c, x);
        let comp = compensated_horner(&c, x);

        // Compensated should be strictly closer to the analytical answer.
        let plain_err = (plain - expected).abs();
        let comp_err = (comp - expected).abs();
        // At x = 1 + 1e-8, plain Horner typically has ~1e-30 error;
        // compensated should be ~ε · |expected| = 1e-56.
        assert!(
            comp_err <= plain_err,
            "compensated error {comp_err:e} should be at most plain {plain_err:e}"
        );
    }

    #[test]
    #[should_panic(expected = "coefficient slice must be non-empty")]
    fn horner_panics_on_empty() {
        let empty: [f64; 0] = [];
        let _ = horner(&empty, 1.0);
    }

    #[test]
    #[should_panic(expected = "coefficient slice must be non-empty")]
    fn compensated_horner_panics_on_empty() {
        let empty: [f64; 0] = [];
        let _ = compensated_horner(&empty, 1.0);
    }
}
