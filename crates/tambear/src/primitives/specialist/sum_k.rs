//! Parameterized k-fold compensated summation.
//!
//! `sum_k(xs, k)` computes `Σxᵢ` with an error bound of roughly `ε^k · cond`,
//! where `ε ≈ 2^-53` is the f64 epsilon and `cond = Σ|xᵢ| / |Σxᵢ|` is the
//! condition number of the summation. Each increment of `k` divides the
//! error by another factor of `ε`, at a cost of approximately `2n`
//! additional flops.
//!
//! Calling with `k = 1` is equivalent to a naive fold (no compensation).
//! Calling with `k = 2` matches `neumaier_sum` (single-level compensation).
//! Calling with `k = 3, 4, ...` extends the accuracy further, reaching
//! Kulisch-oracle accuracy around `k = 4-6` for most realistic input
//! distributions.
//!
//! # Algorithm
//!
//! This is Algorithm SumK from Ogita-Rump-Oishi, "Accurate sum and dot
//! product" (2005), figure 3.2. The idea: repeatedly apply a vector-sum
//! transformation that produces a new array of the same length whose
//! *rounded* sum is identical to the original but whose *exact* error
//! is smaller by a factor of ε. After `k-1` such transformations, sum
//! the result naively.
//!
//! The vector-sum step is implemented via repeated `two_sum`: each
//! element is combined with a running partial sum, and the error of
//! each combination replaces the element in the output array. The last
//! position in the output holds the running partial sum.
//!
//! # When to use
//!
//! - When `neumaier_sum` (k=2) is not accurate enough.
//! - When the Kulisch accumulator's constant-factor cost is unacceptable.
//! - As a tunable trade-off where the caller can pick exactly how many
//!   levels of compensation they need.
//!
//! Recipes that want "maximum accuracy, no oracle" typically call
//! `sum_k(xs, 4)` — at ~8n flops it's about 2× the cost of Neumaier but
//! gains 2 more levels of precision.

use crate::primitives::compensated::eft::two_sum;

/// k-fold compensated sum of a slice.
///
/// `k = 1` is naive summation. `k = 2` matches Neumaier. Higher `k` adds
/// an additional level of EFT compensation per increment.
///
/// # Cost
/// Approximately `(2k - 1) · n + k` flops for a slice of length `n`.
///
/// # Panics
/// Panics if `k == 0`, since there's no sensible interpretation.
#[inline]
pub fn sum_k(xs: &[f64], k: usize) -> f64 {
    assert!(k >= 1, "sum_k: k must be at least 1");
    if xs.is_empty() {
        return 0.0;
    }

    // k = 1: plain fold.
    if k == 1 {
        let mut s = 0.0_f64;
        for &x in xs {
            s += x;
        }
        return s;
    }

    // Copy into a working buffer we can transform in place.
    let mut buf: Vec<f64> = xs.to_vec();

    // Apply k - 1 vector-sum transformations. Each pass rewrites buf so that
    // the sum of all entries equals the true sum of the original input, but
    // with the partial sum concentrated in the final element and the
    // per-element residuals distributed across the rest.
    for _ in 0..(k - 1) {
        vec_sum_in_place(&mut buf);
    }

    // Final naive sum. After k-1 transformations the elements are so small
    // that a naive fold gives error bounded by ε^k · cond.
    let mut s = 0.0_f64;
    for &x in &buf {
        s += x;
    }
    s
}

/// In-place vector-sum transformation (Ogita-Rump-Oishi, Algorithm 4.3).
///
/// After this call, `Σ buf[i]` equals the original `Σ buf[i]` exactly,
/// but the elements have been redistributed: `buf[n-1]` holds a running
/// partial sum and `buf[0..n-1]` hold the error residuals of each
/// intermediate `two_sum`.
#[inline]
fn vec_sum_in_place(buf: &mut [f64]) {
    if buf.len() < 2 {
        return;
    }
    for i in 1..buf.len() {
        let (s, e) = two_sum(buf[i], buf[i - 1]);
        buf[i] = s;
        buf[i - 1] = e;
    }
}

/// Convenience: `sum_k` with `k = 2`. This is equivalent to a
/// Kahan/Neumaier-grade result and is provided as a named entry point so
/// recipes don't have to remember that `k = 2` is the Neumaier point.
#[inline]
pub fn sum_2(xs: &[f64]) -> f64 {
    sum_k(xs, 2)
}

/// Convenience: `sum_k` with `k = 3`. Two levels of EFT compensation.
/// Gives `ε^3 · cond` accuracy — for cond up to ~10^32 the result is
/// accurate to working precision.
#[inline]
pub fn sum_3(xs: &[f64]) -> f64 {
    sum_k(xs, 3)
}

/// Convenience: `sum_k` with `k = 4`. Three levels of EFT compensation.
/// Accuracy matches the Kulisch oracle for essentially all realistic
/// input distributions.
#[inline]
pub fn sum_4(xs: &[f64]) -> f64 {
    sum_k(xs, 4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::specialist::KulischAccumulator;

    // Reference: use the Kulisch accumulator as the exact oracle.
    fn exact_sum(xs: &[f64]) -> f64 {
        let mut acc = KulischAccumulator::new();
        acc.add_slice(xs);
        acc.to_f64()
    }

    #[test]
    fn sum_k_empty_is_zero() {
        for k in 1..=5 {
            assert_eq!(sum_k(&[], k), 0.0);
        }
    }

    #[test]
    fn sum_k_single_element_is_identity() {
        for k in 1..=5 {
            assert_eq!(sum_k(&[3.14], k), 3.14);
        }
    }

    #[test]
    #[should_panic(expected = "k must be at least 1")]
    fn sum_k_zero_panics() {
        let _ = sum_k(&[1.0, 2.0], 0);
    }

    #[test]
    fn sum_k_naive_matches_fold() {
        let xs: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let naive: f64 = xs.iter().sum();
        assert_eq!(sum_k(&xs, 1), naive);
    }

    #[test]
    fn sum_2_matches_neumaier_accuracy() {
        // Classic Kahan stress: one large value, many small ones.
        let mut xs = vec![1.0];
        for _ in 0..10_000 {
            xs.push(1e-10);
        }
        let expected = exact_sum(&xs);
        let got = sum_2(&xs);
        assert!(
            (got - expected).abs() < 1e-14,
            "sum_2 err {:e}",
            (got - expected).abs()
        );
    }

    #[test]
    fn sum_k_converges_to_oracle_as_k_grows() {
        // A harder case: the "Rump example" where naive fp64 gets the sign wrong.
        // Construct by interleaving large positive, large negative, and small values.
        let xs = vec![
            1e100, -1e100, 1.0, 1e100, -1e100, 2.0, 1e100, -1e100, 3.0,
        ];
        let expected = exact_sum(&xs);
        assert_eq!(expected, 6.0);

        // As k increases, error should not increase (monotone convergence).
        let mut prev_err = f64::INFINITY;
        for k in 1..=5 {
            let got = sum_k(&xs, k);
            let err = (got - expected).abs();
            assert!(
                err <= prev_err + 1e-12,
                "sum_k({k}) err {err:e} worse than sum_k({}) err {prev_err:e}",
                k - 1
            );
            prev_err = err;
        }

        // sum_3 should nail it exactly in this small example.
        assert_eq!(sum_3(&xs), 6.0);
    }

    #[test]
    fn sum_4_matches_kulisch_on_sin_sum() {
        // Well-conditioned large input.
        let xs: Vec<f64> = (0..10_000).map(|i| (i as f64 * 0.01).sin()).collect();
        let expected = exact_sum(&xs);
        let got = sum_4(&xs);
        assert!(
            (got - expected).abs() < 1e-12,
            "sum_4 err {:e}",
            (got - expected).abs()
        );
    }

    #[test]
    fn sum_k_preserves_sum_invariant() {
        // Core property: after each vec_sum pass, the *true* sum of all
        // elements is unchanged. Verify by computing the Kulisch sum before
        // and after applying vec_sum_in_place.
        let mut buf = vec![1e100, -1e100, 1.0, 2.0, 3.0];
        let before = exact_sum(&buf);
        vec_sum_in_place(&mut buf);
        let after = exact_sum(&buf);
        assert_eq!(before, after, "vec_sum should preserve the exact sum");
    }

    #[test]
    fn sum_k_stable_on_signed_mix() {
        // Mixed signs, various magnitudes.
        let xs = vec![3.14, -2.71, 1e10, -1e10, 1e-10, -1e-10, 42.0, -42.0, 7.0];
        let expected = exact_sum(&xs);
        assert_eq!(sum_3(&xs), expected);
        assert_eq!(sum_4(&xs), expected);
    }

    #[test]
    fn sum_k_matches_fold_on_exactly_representable_sums() {
        // For a sum that's exact in fp64, every k should give the same answer.
        let xs = [1.0, 2.0, 4.0, 8.0, 16.0]; // sum = 31, exact
        for k in 1..=5 {
            assert_eq!(sum_k(&xs, k), 31.0);
        }
    }
}
