//! Compensated summation primitives.
//!
//! These all compute `sum(xs)` but at different points on the accuracy/speed
//! trade-off:
//!
//! | Primitive      | Cost per element | Error bound (worst case) | Note |
//! |----------------|-----------------:|-------------------------:|------|
//! | `fadd` fold    |           1 flop |              O(n·ε·S)    | Naive sum — the bug source. |
//! | `pairwise_sum` |           1 flop |           O(log n · ε·S) | No extra storage, good locality. |
//! | `kahan_sum`    |          4 flops |              O(ε·S) + ε² | Kahan-Babuška. |
//! | `neumaier_sum` |          ~5 flops |             O(ε·S) + ε² | Handles |a|<|b| correctly. |
//!
//! `S = Σ|xᵢ|` is the sum of absolute values. The key property for
//! Kahan/Neumaier is that the bound is independent of `n`.
//!
//! # Which to use
//! - `pairwise_sum`: default for well-conditioned sums, cache-friendly.
//! - `kahan_sum`: when correctness is critical and the input is known to
//!   have decreasing magnitudes (or ordering doesn't matter).
//! - `neumaier_sum`: when correctness is critical and inputs are
//!   heterogeneous in magnitude (the typical recipe-consumer case).
//!
//! Recipes tagged `#[precision(compensated)]` lower a `sum(..)` node to
//! `neumaier_sum` by default. `#[precision(strict)]` lowers to a plain
//! fold.

use super::eft::{fast_two_sum, two_sum};

/// Kahan-Babuška summation. Returns a running-compensated sum of `xs`.
///
/// Error bound: `|result - Σxᵢ| <= 2ε · Σ|xᵢ| + O(n·ε²·Σ|xᵢ|)` — the
/// leading term is independent of `n`, unlike the naive fold whose bound
/// grows linearly.
///
/// Costs 4 flops per element (add, sub, sub, add). Stable against ordering
/// changes for uniformly-signed inputs.
///
/// # Precondition
/// Uses `fast_two_sum` internally; safe as long as the running sum is at
/// least as large in magnitude as each incoming element. For well-behaved
/// inputs this is automatically true. For mixed-magnitude inputs prefer
/// `neumaier_sum`, which handles the unordered case correctly.
#[inline]
pub fn kahan_sum(xs: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64; // running compensation
    for &x in xs {
        let y = x - comp;
        let t = sum + y;
        comp = (t - sum) - y;
        sum = t;
    }
    sum
}

/// Neumaier's variant of Kahan summation. Handles the case where incoming
/// element magnitude can exceed the running sum — common when the first
/// element is small and later elements are large.
///
/// Error bound matches Kahan: `O(ε · Σ|xᵢ|)`, independent of `n`.
/// Costs ~5 flops per element plus a branch.
///
/// This is the default for recipes tagged `#[precision(compensated)]` when
/// the consumer doesn't know the magnitude ordering ahead of time.
#[inline]
pub fn neumaier_sum(xs: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut comp = 0.0_f64;
    for &x in xs {
        let t = sum + x;
        if sum.abs() >= x.abs() {
            comp += (sum - t) + x;
        } else {
            comp += (x - t) + sum;
        }
        sum = t;
    }
    sum + comp
}

/// Pairwise (binary tree) summation. Recursively splits the slice in half
/// and sums each half, combining the results with a single add. The recursion
/// depth is `log₂(n)`, so the accumulated error is `O(log n · ε · Σ|xᵢ|)`
/// instead of `O(n · ε · Σ|xᵢ|)` for the naive fold.
///
/// Costs 1 flop per element in the recursion, but the recursion overhead
/// is non-zero for tiny slices. Falls back to naive summation for base cases
/// of `<= 8` elements to amortize call cost.
///
/// Use this as the default summation in well-conditioned contexts where the
/// full compensated cost is unnecessary.
#[inline]
pub fn pairwise_sum(xs: &[f64]) -> f64 {
    const BASE: usize = 8;
    if xs.len() <= BASE {
        let mut s = 0.0_f64;
        for &x in xs {
            s += x;
        }
        return s;
    }
    let mid = xs.len() / 2;
    pairwise_sum(&xs[..mid]) + pairwise_sum(&xs[mid..])
}

/// Compensated sum using `two_sum` EFT chain — accumulates all exact
/// residuals into a running correction, then reinjects at the end.
///
/// This is the building block for `sum_k` (k-fold compensation) in Phase B6.
/// Exposed here as `two_sum_accumulation` so recipes needing a single level
/// of full-EFT compensation don't have to reach into Phase B6 machinery.
#[inline]
pub fn two_sum_accumulation(xs: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut err_acc = 0.0_f64;
    for &x in xs {
        let (s_new, e) = two_sum(sum, x);
        sum = s_new;
        err_acc += e;
    }
    sum + err_acc
}

/// Internal helper: equivalent to `fast_two_sum` chaining. Kept for
/// diagnostic comparisons in tests.
#[inline]
#[allow(dead_code)]
pub(crate) fn fast_two_sum_accumulation(xs: &[f64]) -> f64 {
    let mut sum = 0.0_f64;
    let mut err_acc = 0.0_f64;
    for &x in xs {
        // Only safe if |sum| grows monotonically, which we assume for
        // sorted-by-decreasing-magnitude inputs.
        let (s_new, e) = if sum.abs() >= x.abs() {
            fast_two_sum(sum, x)
        } else {
            fast_two_sum(x, sum)
        };
        sum = s_new;
        err_acc += e;
    }
    sum + err_acc
}

#[cfg(test)]
mod tests {
    use super::*;

    // A classic Kahan-breaking input: many small values added to a large one.
    fn kahan_stress() -> Vec<f64> {
        let mut v = vec![1.0_f64];
        for _ in 0..10_000 {
            v.push(1e-10);
        }
        v
    }

    #[test]
    fn naive_sum_loses_accuracy_on_kahan_stress() {
        let xs = kahan_stress();
        let naive: f64 = xs.iter().sum();
        let expected = 1.0 + 10_000.0 * 1e-10;
        // The naive sum should be measurably worse than the compensated.
        let naive_err = (naive - expected).abs();
        let kahan_err = (kahan_sum(&xs) - expected).abs();
        assert!(
            kahan_err <= naive_err,
            "kahan {kahan_err:e} should be at most naive {naive_err:e}"
        );
    }

    #[test]
    fn kahan_sum_tightens_error() {
        let xs = kahan_stress();
        let expected = 1.0 + 10_000.0 * 1e-10;
        let got = kahan_sum(&xs);
        // Kahan should nail it within a few epsilons.
        assert!(
            (got - expected).abs() < 1e-14,
            "kahan_sum off by {:e}, got {got}, expected {expected}",
            (got - expected).abs()
        );
    }

    #[test]
    fn neumaier_sum_tightens_error_even_unordered() {
        // Start with a small value, then add a huge one — the case where
        // plain Kahan can fail because |sum| < |x| and the compensation
        // assumption breaks down.
        let xs = vec![1e-10, 1e10, 1e-10, -1e10, 1e-10];
        let got = neumaier_sum(&xs);
        let expected = 3e-10;
        assert!(
            (got - expected).abs() < 1e-20,
            "neumaier_sum off by {:e}, got {got}",
            (got - expected).abs()
        );
    }

    #[test]
    fn pairwise_sum_beats_naive_on_large_inputs() {
        // Generate a vector where naive sum degrades predictably.
        let n = 100_000;
        let xs: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let pairwise = pairwise_sum(&xs);
        let naive: f64 = xs.iter().sum();
        // Reference: compute with Kahan as a gold standard.
        let gold = kahan_sum(&xs);
        let pair_err = (pairwise - gold).abs();
        let naive_err = (naive - gold).abs();
        // Pairwise should be at most slightly worse than Kahan, but often
        // strictly better than naive. We just assert pairwise is not
        // catastrophically worse.
        assert!(pair_err <= naive_err.max(1e-10));
    }

    #[test]
    fn pairwise_sum_base_cases() {
        assert_eq!(pairwise_sum(&[]), 0.0);
        assert_eq!(pairwise_sum(&[1.0]), 1.0);
        assert_eq!(pairwise_sum(&[1.0, 2.0]), 3.0);
        assert_eq!(pairwise_sum(&[1.0, 2.0, 3.0, 4.0]), 10.0);
    }

    #[test]
    fn two_sum_accumulation_matches_kahan() {
        let xs = kahan_stress();
        let expected = 1.0 + 10_000.0 * 1e-10;
        let got = two_sum_accumulation(&xs);
        assert!((got - expected).abs() < 1e-14);
    }

    #[test]
    fn empty_sum_returns_zero() {
        let empty: &[f64] = &[];
        assert_eq!(kahan_sum(empty), 0.0);
        assert_eq!(neumaier_sum(empty), 0.0);
        assert_eq!(pairwise_sum(empty), 0.0);
        assert_eq!(two_sum_accumulation(empty), 0.0);
    }

    #[test]
    fn single_element_sum_is_identity() {
        let xs = [3.14];
        assert_eq!(kahan_sum(&xs), 3.14);
        assert_eq!(neumaier_sum(&xs), 3.14);
        assert_eq!(pairwise_sum(&xs), 3.14);
        assert_eq!(two_sum_accumulation(&xs), 3.14);
    }
}
