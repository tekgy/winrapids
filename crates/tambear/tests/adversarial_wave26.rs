//! Adversarial Wave 26 — NaN sort-comparator panic in CCM (complexity.rs:1783)
//!
//! Target: `ccm` in `complexity.rs:1783`.
//!
//! ## The bug: non-total-order comparator causes runtime panic
//!
//! The k-NN step in CCM sorts distances to find nearest neighbors:
//!
//! ```rust
//! dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
//! ```
//!
//! When any embedding coordinate is NaN (from NaN input data), the distance
//! `d = sqrt(sum((a-b)²))` is NaN. The comparator maps NaN-vs-finite to
//! `Equal` (both directions), but `a.partial_cmp(&b) = Equal` while
//! `b.partial_cmp(&a) = Equal` is consistent — the problem is that
//! `NaN.partial_cmp(&NaN) = None → Equal`, and comparisons between
//! NaN and non-NaN values are inconsistent with transitivity.
//!
//! Rust's sort (pdqsort) includes debug assertions that verify the comparator
//! implements a total order. With NaN distances, these assertions fire:
//!
//! ```text
//! thread panicked at library/core/src/slice/sort/shared/smallsort.rs:860:5:
//! user-provided comparison function does not correctly implement a total order
//! ```
//!
//! This is WORSE than a silent wrong answer: CCM panics on any NaN input,
//! crashing the calling computation with no opportunity to handle the error.
//!
//! ## The severity gradient
//!
//! - Silent NaN propagation: bad (wrong answer, no signal)
//! - Silent wrong answer: bad (plausible answer, wrong)
//! - Panic: WORST for production use — crashes the computation, forces
//!   the caller to use catch_unwind or pre-filter all inputs
//!
//! The correct behavior: either (a) return a NaN CcmResult (propagate policy)
//! via an upfront NaN guard before any sorting, or (b) filter NaN distances
//! before sorting (ignore policy, with documentation). The current behavior
//! is neither — it's a panic.
//!
//! ## Fix
//!
//! Replace `unwrap_or(std::cmp::Ordering::Equal)` with `unwrap_or(std::cmp::Ordering::Greater)`:
//! NaN distances sort to the END (farthest), never selected as nearest neighbors.
//! Then add an upfront `if x.iter().any(|v| v.is_nan()) || y.iter().any(|v| v.is_nan()) { return nan; }`
//! to implement the propagate policy explicitly.
//!
//! ## Bug severity note
//!
//! CCM is used in the causal inference pipeline. A NaN in any data column
//! (missing observation, sensor dropout) causes the entire CCM computation to
//! panic. Any production caller of CCM must currently pre-filter all NaN values
//! before calling — an undocumented and unenforceable precondition.
//!
//! These tests FAIL (via panic or wrong result) with the current implementation
//! and PASS after the fix.

use tambear::complexity::ccm;

/// Simple sine-based causal pair for baseline testing.
fn causal_pair(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
    let y: Vec<f64> = (0..n).map(|i| {
        if i >= 2 { x[i - 2] * 0.8 + (i as f64 * 0.1).cos() * 0.2 }
        else { 0.0 }
    }).collect();
    (x, y)
}

// ─── Baselines ────────────────────────────────────────────────────────────────

/// CCM on clean causal pair must return finite rho values.
#[test]
fn ccm_clean_causal_pair_baseline() {
    let (x, y) = causal_pair(100);
    let result = ccm(&x, &y, 3, 1, 4);
    assert!(result.rho_xy.is_finite(),
        "CCM(clean): rho_xy should be finite, got {}", result.rho_xy);
    assert!(result.rho_yx.is_finite(),
        "CCM(clean): rho_yx should be finite, got {}", result.rho_yx);
}

/// CCM returns NaN-result when n is too small for embedding.
#[test]
fn ccm_too_small_returns_nan_baseline() {
    let x = vec![1.0, 2.0, 3.0];
    let y = vec![1.0, 2.0, 3.0];
    let result = ccm(&x, &y, 3, 1, 4);
    assert!(result.rho_xy.is_nan(),
        "CCM(too small): should return NaN result, got {}", result.rho_xy);
}

/// All-NaN x with all-NaN y: should return NaN-result (not panic).
#[test]
fn ccm_all_nan_x_returns_nan_not_panic() {
    // With all-NaN x, all distances are NaN, all comparisons invalid.
    // The sort_by comparator panics in debug mode on non-total-order.
    // This test documents that NaN input must NOT panic — it should return NaN-result.
    let x: Vec<f64> = vec![f64::NAN; 50];
    let y: Vec<f64> = (0..50).map(|i| i as f64).collect();

    // Use catch_unwind to detect the panic as a test failure (not test-harness error).
    let result = std::panic::catch_unwind(|| ccm(&x, &y, 3, 1, 4));
    match result {
        Ok(r) => {
            // Correct: returned NaN-result without panicking.
            assert!(r.rho_xy.is_nan(),
                "CCM(all-NaN x): should return NaN result, got {}", r.rho_xy);
        }
        Err(_) => {
            panic!(
                "CCM(all-NaN x) PANICKED — non-total-order comparator bug. \
                 `sort_by` with `unwrap_or(Equal)` on NaN distances violates \
                 total-order invariant. Fix: add upfront NaN guard or use \
                 `unwrap_or(Greater)` to sort NaN distances to end."
            );
        }
    }
}

// ─── Bug: NaN in x series causes panic ───────────────────────────────────────

/// NaN in the x series causes CCM to panic.
///
/// Single NaN at position 40 (mid-series). The embedding of nearby points
/// includes this NaN coordinate, making several distances NaN. The sort
/// comparator fails the total-order check, and Rust's sort panics.
///
/// Correct behavior: return NaN-result (propagate) or finite result
/// excluding the NaN point (ignore, with documentation).
/// Bug: runtime panic, crashing the caller.
#[test]
fn ccm_nan_in_x_must_not_panic() {
    let (mut x, y) = causal_pair(80);
    x[40] = f64::NAN;

    let result = std::panic::catch_unwind(|| ccm(&x, &y, 3, 1, 4));
    match result {
        Ok(r) => {
            // If no panic: result must be NaN (input had NaN → output should signal it).
            assert!(r.rho_xy.is_nan(),
                "CCM(NaN in x[40]): should return NaN, got rho_xy={:.4}", r.rho_xy);
        }
        Err(_) => {
            panic!(
                "CCM(NaN in x[40]) PANICKED — sort_by comparator bug. \
                 NaN distance at position 40 causes non-total-order violation. \
                 Fix: `unwrap_or(Ordering::Greater)` or upfront NaN guard."
            );
        }
    }
}

/// NaN at early position (x[5]) affects more embedded points.
#[test]
fn ccm_nan_in_x_early_must_not_panic() {
    let (mut x, y) = causal_pair(80);
    x[5] = f64::NAN;

    let result = std::panic::catch_unwind(|| ccm(&x, &y, 3, 1, 4));
    match result {
        Ok(r) => {
            assert!(r.rho_xy.is_nan(),
                "CCM(NaN in x[5]): should return NaN, got rho_xy={:.4}", r.rho_xy);
        }
        Err(_) => {
            panic!(
                "CCM(NaN in x[5]) PANICKED — same comparator bug, earlier position. \
                 x[5] contaminates embed_dim=3 embedded points."
            );
        }
    }
}

// ─── Bug: NaN in y series (prediction target) ────────────────────────────────

/// NaN in y series: target values for prediction contain NaN.
///
/// Even if x distances sort correctly (no panic), NaN target values
/// produce NaN predictions, which contaminate ccm_pearson.
/// The bug here: no upfront check for NaN in y, so we get NaN result
/// through an undocumented pathway. Still a panic risk if the sort
/// reaches a point where x-distances are also NaN.
#[test]
fn ccm_nan_in_y_must_not_panic() {
    let (x, mut y) = causal_pair(80);
    y[40] = f64::NAN;

    let result = std::panic::catch_unwind(|| ccm(&x, &y, 3, 1, 4));
    match result {
        Ok(r) => {
            // NaN in y target → NaN predictions → NaN rho (correct propagation).
            assert!(r.rho_xy.is_nan(),
                "CCM(NaN in y[40]): should return NaN (NaN target), got {:.4}", r.rho_xy);
        }
        Err(_) => {
            // NaN in y propagates into the ey (y-manifold) embeddings used in
            // the y-predicts-x direction: ccm_predict(&ey, x_target, ...).
            // The y-embedded distances become NaN, triggering the sort panic.
            // Both directions of CCM are affected: x→y (via NaN target) and y→x (via NaN embedding).
            panic!(
                "CCM(NaN in y[40]) PANICKED — NaN in y contaminates both: \
                 (1) y_target values used as prediction targets, and \
                 (2) ey embeddings used in the y→x direction. \
                 Both paths hit the sort_by non-total-order bug. \
                 Fix: upfront NaN guard before any embedding or sorting."
            );
        }
    }
}

// ─── The fix-direction diagnostic ────────────────────────────────────────────

/// Documents the sort-comparator bug mechanism directly.
///
/// The correct fix is `f64::total_cmp` (available since Rust 1.62), which
/// defines a total order over all f64 values including NaN: NaN sorts AFTER
/// all finite values and infinities. This makes the sort deterministic and
/// places NaN-distance points at the end, where they won't be selected as
/// nearest neighbors.
///
/// Current bug: `partial_cmp(...).unwrap_or(Equal)` — NaN comparisons treated
/// as Equal violate antisymmetry: `NaN ≤ x` and `x ≤ NaN` simultaneously.
#[test]
fn sort_comparator_nan_distance_behavior_documented() {
    let base: Vec<f64> = vec![0.5, f64::NAN, 0.2, f64::NAN, 0.8];

    // The buggy comparator panics in debug mode (non-total-order violation):
    let panicked = std::panic::catch_unwind(move || {
        let mut v = base.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    }).is_err();

    // The correct fix: use total_cmp which places NaN at end.
    let mut dists_fixed: Vec<f64> = vec![0.5, f64::NAN, 0.2, f64::NAN, 0.8];
    dists_fixed.sort_by(|a, b| a.total_cmp(b));
    // After total_cmp sort: finite values in order, then NaN values at end.
    let all_finite_before_nan = {
        let mut seen_nan = false;
        dists_fixed.iter().all(|v| {
            if seen_nan { v.is_nan() }  // once NaN seen, rest must be NaN
            else if v.is_nan() { seen_nan = true; true }
            else { true }
        })
    };
    assert!(all_finite_before_nan,
        "total_cmp must sort NaN to end, got: {:?}", dists_fixed);
    assert_eq!(dists_fixed[0], 0.2, "Smallest finite first: {:?}", dists_fixed);
    assert_eq!(dists_fixed[1], 0.5);
    assert_eq!(dists_fixed[2], 0.8);

    // Document bug status:
    if panicked {
        // Confirmed: buggy comparator panics in debug mode.
        // Fix: replace sort_by(partial_cmp + unwrap_or(Equal)) with sort_by(total_cmp).
    }
    // total_cmp is the correct fix — no unwrap_or needed, NaN goes to end deterministically.
}
