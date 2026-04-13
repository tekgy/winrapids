//! Oracle test infrastructure: the shared machinery that compares a
//! candidate implementation's output against a trusted reference.
//!
//! This module is the mechanism by which the Tambear Contract's
//! "bit-perfect or bug-finding" rule is enforced for sum- and
//! dot-shaped primitives. A recipe or candidate primitive does not
//! claim correctness by passing a handful of unit tests — it claims
//! correctness by agreeing with the oracle across an adversarial
//! distribution of inputs, and any disagreement is either our bug
//! or a bug we'll file against someone else's implementation.
//!
//! # Current oracles
//!
//! - [`kulisch_oracle_sum`]: exact summation via the Kulisch accumulator.
//!   Any sum primitive (`kahan_sum`, `neumaier_sum`, `sum_k`,
//!   `pairwise_sum`, etc.) can be compared against this on an adversarial
//!   input to verify its error bound.
//! - [`kulisch_oracle_dot`]: exact dot product via the Kulisch accumulator
//!   composed with `two_product_fma`. The reference for `dot_2`,
//!   `compensated_horner` (when expressed as a dot product), and future
//!   mat-vec kernels.
//!
//! # Comparing a candidate against the oracle
//!
//! Use [`assert_within_ulps`] or [`assert_within_relative`] to compare a
//! candidate f64 against the oracle value with a declared tolerance. The
//! function reports which input element (if any) could plausibly explain
//! a wider-than-expected error, so test failures point at a specific
//! row of the input distribution.
//!
//! # Adversarial input generators
//!
//! Gold standard correctness tests should not just compare on "random"
//! inputs — they should compare on inputs known to stress specific
//! failure modes. [`hard_sums`] returns a canonical battery of stress
//! inputs: large-plus-small, perfect cancellation, alternating signs,
//! near-overflow, subnormal sums, and the Rump example.

use crate::primitives::compensated::eft::two_product_fma;
use crate::primitives::specialist::KulischAccumulator;

/// Compute the exact sum of `xs` using the Kulisch accumulator, rounded to
/// the nearest f64. This is the gold-standard reference for any summation
/// primitive.
pub fn kulisch_oracle_sum(xs: &[f64]) -> f64 {
    let mut acc = KulischAccumulator::new();
    acc.add_slice(xs);
    acc.to_f64()
}

/// Compute the exact dot product `Σ xᵢ·yᵢ` via Kulisch accumulation of
/// `two_product_fma` halves. This is the gold-standard reference for any
/// dot-product primitive.
///
/// # Panics
/// Panics if `x.len() != y.len()`.
pub fn kulisch_oracle_dot(x: &[f64], y: &[f64]) -> f64 {
    assert_eq!(x.len(), y.len(), "kulisch_oracle_dot: length mismatch");
    let mut acc = KulischAccumulator::new();
    for i in 0..x.len() {
        let (hi, lo) = two_product_fma(x[i], y[i]);
        acc.add_f64(hi);
        acc.add_f64(lo);
    }
    acc.to_f64()
}

/// Number of units in the last place between two f64 values, measured as
/// the magnitude of the symbolic difference `|a - b| / ulp(max(|a|, |b|))`.
///
/// Returns `0` when `a == b`, `u64::MAX` when exactly one of the inputs is
/// NaN, and the numeric distance otherwise.
pub fn ulps_between(a: f64, b: f64) -> u64 {
    if a.is_nan() || b.is_nan() {
        if a.to_bits() == b.to_bits() {
            return 0;
        }
        return u64::MAX;
    }
    if a == b {
        return 0;
    }

    // Transform to a monotone integer mapping: map signed bit pattern to
    // an unsigned total order. Signs differ → fall back to magnitude path.
    let ai = a.to_bits();
    let bi = b.to_bits();
    let transform = |bits: u64| -> i64 {
        if (bits >> 63) == 1 {
            // Negative: invert to flip the order so more-negative = smaller.
            !(bits as i64) + i64::MIN + 1
        } else {
            // Positive: shift above the negative half of the space.
            (bits as i64).wrapping_sub(i64::MIN)
        }
    };
    let at = transform(ai);
    let bt = transform(bi);
    at.abs_diff(bt)
}

/// Assert that `actual` agrees with `expected` to within `max_ulps` units
/// in the last place. On failure, the panic message includes both values
/// and the measured ulp distance.
#[track_caller]
pub fn assert_within_ulps(actual: f64, expected: f64, max_ulps: u64, context: &str) {
    let dist = ulps_between(actual, expected);
    assert!(
        dist <= max_ulps,
        "{context}: {dist} ulps apart (max allowed {max_ulps})\n  actual:   {actual:e}\n  expected: {expected:e}"
    );
}

/// Assert that `|actual - expected| / max(|expected|, 1.0) <= rel_tol`.
///
/// Use this for cases where the comparison spans multiple orders of
/// magnitude and a fixed-ulp threshold is too strict on the small side
/// and too loose on the large side. For bit-perfect comparisons, prefer
/// [`assert_within_ulps`].
#[track_caller]
pub fn assert_within_relative(actual: f64, expected: f64, rel_tol: f64, context: &str) {
    let abs_err = (actual - expected).abs();
    let scale = expected.abs().max(1.0);
    let rel_err = abs_err / scale;
    assert!(
        rel_err <= rel_tol,
        "{context}: rel err {rel_err:e} > tol {rel_tol:e}\n  actual:   {actual:e}\n  expected: {expected:e}\n  abs err:  {abs_err:e}"
    );
}

/// A canonical battery of adversarial sum inputs. Each entry is a
/// `(name, vec)` tuple where `name` is a short label that will be
/// included in test failure messages.
///
/// This is the "hard cases" suite — if a sum primitive agrees with the
/// Kulisch oracle on every input here, it is correctly implementing a
/// compensated summation algorithm up to its declared precision level.
/// Recipes are expected to run their full adversarial comparison through
/// this generator.
pub fn hard_sums() -> Vec<(&'static str, Vec<f64>)> {
    vec![
        ("empty", vec![]),
        ("single_one", vec![1.0]),
        ("powers_of_two_ascending", vec![1.0, 2.0, 4.0, 8.0, 16.0, 32.0]),
        ("powers_of_ten_ascending", vec![1.0, 10.0, 100.0, 1000.0]),
        (
            "large_plus_many_small",
            {
                let mut v = vec![1e15];
                for _ in 0..1000 {
                    v.push(1.0);
                }
                v
            },
        ),
        (
            "perfect_cancellation",
            vec![1e17, 1.0, -1e17],
        ),
        (
            "rump_example_small",
            vec![1e100, -1e100, 1.0, 1e100, -1e100, 2.0, 1e100, -1e100, 3.0],
        ),
        (
            "alternating_signs_same_magnitude",
            {
                let mut v = Vec::with_capacity(1000);
                for i in 0..1000 {
                    v.push(if i % 2 == 0 { 1.0 } else { -1.0 });
                }
                v.push(1.0); // final unpaired 1 so sum = 1.0
                v
            },
        ),
        (
            "alternating_signs_mixed_magnitude",
            vec![
                1e15, -1e15, 1.0, 2.0, 3.0, -1e10, 1e10, 0.5, 0.25, -0.75,
            ],
        ),
        (
            "subnormal_tail",
            {
                let mut v = vec![1.0];
                let tiny = f64::from_bits(1);
                for _ in 0..500 {
                    v.push(tiny);
                }
                v
            },
        ),
        (
            "near_overflow_with_tiny",
            vec![1e300, 1e-300, -1e300, 1e-300],
        ),
        (
            "sine_wave",
            (0..1000).map(|i| ((i as f64) * 0.01).sin()).collect(),
        ),
        (
            "harmonic_series",
            (1..=1000).map(|i| 1.0 / (i as f64)).collect(),
        ),
    ]
}

/// Run a summation candidate against the Kulisch oracle on every entry
/// in [`hard_sums`] (except any in `skip`) and panic on first
/// disagreement beyond tolerance.
///
/// The `skip` list lets a test suite declare known weaknesses of a
/// particular candidate. For example, `kahan_sum` and `pairwise_sum` both
/// lose the trailing `1` in the input `[1e17, 1, -1e17]` — that is a
/// documented property of their algorithms, not a bug. Passing
/// `&["perfect_cancellation"]` for these candidates skips that specific
/// case while still exercising every other adversarial input.
///
/// The candidate is given as a generic closure, so this helper works for
/// every sum primitive without modification.
#[track_caller]
pub fn assert_sum_matches_oracle<F: Fn(&[f64]) -> f64>(
    candidate: F,
    max_ulps: u64,
    name: &str,
    skip: &[&str],
) {
    for (case_name, xs) in hard_sums() {
        if skip.contains(&case_name) {
            continue;
        }
        let expected = kulisch_oracle_sum(&xs);
        let actual = candidate(&xs);
        let dist = ulps_between(actual, expected);
        assert!(
            dist <= max_ulps,
            "{name} failed on case '{case_name}':\n  \
             {dist} ulps apart (max allowed {max_ulps})\n  \
             expected: {expected:e}\n  \
             actual:   {actual:e}",
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::compensated::sums::{kahan_sum, neumaier_sum, pairwise_sum};
    use crate::primitives::specialist::{sum_2, sum_3, sum_4};

    // ── ulps_between ────────────────────────────────────────────────────────

    #[test]
    fn ulps_between_equal() {
        assert_eq!(ulps_between(1.0, 1.0), 0);
        assert_eq!(ulps_between(0.0, 0.0), 0);
        assert_eq!(ulps_between(f64::INFINITY, f64::INFINITY), 0);
    }

    #[test]
    fn ulps_between_one_and_next() {
        let one = 1.0_f64;
        let next = f64::from_bits(one.to_bits() + 1);
        assert_eq!(ulps_between(one, next), 1);
    }

    #[test]
    fn ulps_between_nans_by_bits() {
        // Same NaN bit pattern → 0 ulps.
        let nan1 = f64::from_bits(0x7FF8_0000_0000_0000);
        let nan2 = f64::from_bits(0x7FF8_0000_0000_0000);
        assert_eq!(ulps_between(nan1, nan2), 0);
    }

    #[test]
    fn ulps_between_nan_and_nonnan_is_max() {
        assert_eq!(ulps_between(f64::NAN, 1.0), u64::MAX);
        assert_eq!(ulps_between(1.0, f64::NAN), u64::MAX);
    }

    #[test]
    fn ulps_between_neg_and_pos_zero_is_zero() {
        assert_eq!(ulps_between(0.0, -0.0), 0);
    }

    // ── assert_within_ulps / assert_within_relative ─────────────────────────

    #[test]
    fn assert_within_ulps_passes_on_equal() {
        assert_within_ulps(1.0, 1.0, 0, "identical");
    }

    #[test]
    fn assert_within_ulps_passes_on_one_ulp() {
        let one = 1.0_f64;
        let next = f64::from_bits(one.to_bits() + 1);
        assert_within_ulps(one, next, 1, "adjacent");
    }

    #[test]
    #[should_panic(expected = "ulps apart")]
    fn assert_within_ulps_fails_on_too_far() {
        assert_within_ulps(1.0, 2.0, 1, "should fail");
    }

    #[test]
    fn assert_within_relative_passes_on_scale() {
        assert_within_relative(1e100, 1e100 + 1e84, 1e-15, "scale");
    }

    #[test]
    #[should_panic(expected = "rel err")]
    fn assert_within_relative_fails_when_over_tol() {
        assert_within_relative(1.0, 2.0, 0.1, "rel-fail");
    }

    // ── kulisch_oracle_sum ─────────────────────────────────────────────────

    #[test]
    fn kulisch_oracle_sum_matches_known_value() {
        assert_eq!(kulisch_oracle_sum(&[1.0, 2.0, 3.0]), 6.0);
        assert_eq!(kulisch_oracle_sum(&[1e17, 1.0, -1e17]), 1.0);
        assert_eq!(kulisch_oracle_sum(&[]), 0.0);
    }

    // ── kulisch_oracle_dot ─────────────────────────────────────────────────

    #[test]
    fn kulisch_oracle_dot_matches_known_value() {
        assert_eq!(kulisch_oracle_dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
    }

    #[test]
    fn kulisch_oracle_dot_recovers_cancellation() {
        // x·y where naive fp64 loses precision.
        let x = [1e17, 1.0, -1e17];
        let y = [1.0, 1.0, 1.0];
        assert_eq!(kulisch_oracle_dot(&x, &y), 1.0);
    }

    // ── hard_sums and the matches_oracle helper ─────────────────────────────

    #[test]
    fn hard_sums_not_empty() {
        let cases = hard_sums();
        assert!(cases.len() >= 10);
        for (name, xs) in &cases {
            // Each case should have a deterministic Kulisch oracle value
            // (we just check it doesn't panic here).
            let _ = kulisch_oracle_sum(xs);
            assert!(!name.is_empty());
        }
    }

    #[test]
    fn sum_4_passes_hard_sums_at_zero_ulps() {
        // sum_4 should match the oracle bit-for-bit on every hard case.
        assert_sum_matches_oracle(sum_4, 0, "sum_4", &[]);
    }

    #[test]
    fn neumaier_sum_passes_hard_sums_at_low_ulps() {
        // Neumaier is correct enough on all these cases to be within 4 ulps.
        assert_sum_matches_oracle(neumaier_sum, 4, "neumaier_sum", &[]);
    }

    #[test]
    fn kahan_sum_passes_hard_sums_except_big_cancellation() {
        // Kahan genuinely loses small values that are trapped between large
        // cancelling pairs — both `perfect_cancellation` and
        // `rump_example_small` lose bits because Kahan's compensation term
        // itself gets rounded away when the running sum returns near zero.
        // Neumaier handles these cases because it compares magnitudes before
        // selecting the compensation formula. This is documented algorithmic
        // weakness, not an implementation bug.
        assert_sum_matches_oracle(
            kahan_sum,
            4,
            "kahan_sum",
            &[
                "perfect_cancellation",
                "rump_example_small",
                "near_overflow_with_tiny",
            ],
        );
    }

    #[test]
    fn pairwise_sum_tolerance_is_log_n() {
        // Pairwise is O(log n · ε · Σ|xᵢ|) with no compensation at all, so it
        // loses small trailing values on any big-cancellation input. This is
        // the whole reason compensated summation exists — we skip the cases
        // that require it.
        assert_sum_matches_oracle(
            pairwise_sum,
            32,
            "pairwise_sum",
            &[
                "perfect_cancellation",
                "rump_example_small",
                "near_overflow_with_tiny",
            ],
        );
    }

    #[test]
    fn sum_2_matches_neumaier_ulps() {
        assert_sum_matches_oracle(sum_2, 4, "sum_2", &[]);
    }

    #[test]
    fn sum_3_tighter_than_sum_2() {
        assert_sum_matches_oracle(sum_3, 2, "sum_3", &[]);
    }
}
