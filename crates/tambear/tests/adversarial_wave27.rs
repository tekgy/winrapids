//! Adversarial Wave 27 — sort_by(partial_cmp + unwrap_or) panic sweep
//!
//! Wave 26 found the CCM sort-comparator panic. This wave documents the same
//! bug in ALL other functions that use `sort_by` with `partial_cmp + unwrap_or`
//! on user-supplied data.
//!
//! ## The pattern
//!
//! ```rust
//! slice.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
//! ```
//!
//! IEEE 754: `NaN.partial_cmp(&x) = None` for all x (including NaN itself).
//! `None.unwrap_or(Equal)` maps all NaN comparisons to Equal.
//! This violates antisymmetry: `NaN ≤ x` and `x ≤ NaN` simultaneously.
//! Rust's pdqsort asserts total order in debug builds → **runtime panic**.
//!
//! The correct fix: `sort_by(|a, b| a.total_cmp(b))` — IEEE 754 totalOrder,
//! NaN sorts after all finite values and infinities, deterministically.
//!
//! ## Confirmed instances (all on user-data paths)
//!
//! 1. `complexity.rs:1923` — `harmonic_r_stat(levels: &[f64])` — user levels
//! 2. `time_series.rs:2423` — `rank_von_neumann_ratio(data: &[f64])` — user data
//! 3. `graph.rs:754` — kNN graph adjacency matrix (distances may be NaN)
//!
//! ## Already documented
//!
//! `complexity.rs:1783` — `ccm` — documented in wave 26.
//!
//! ## Instances that are probably safe (computed values, not raw user data)
//!
//! - `superposition.rs:685` — sorts [pearson, spearman, kendall] — computed from data,
//!   will be NaN only if the underlying correlation function returns NaN (which it
//!   does via propagation, not silent ignore — so if input is NaN, values are NaN).
//!   STILL A BUG: if input data has NaN, computed correlations are NaN → panic here.
//! - `superposition.rs:755` — slopes from regression — same reasoning.
//! - `scoring.rs:64` — drops from internal computation.
//!
//! All instances will be documented here. The fix pattern is identical across all.

use tambear::complexity::harmonic_r_stat;
use tambear::time_series::rank_von_neumann_ratio;

// ─── Baselines ────────────────────────────────────────────────────────────────

/// harmonic_r_stat on clean data returns finite result.
#[test]
fn harmonic_r_stat_clean_baseline() {
    let levels = vec![0.1, 0.3, 0.7, 1.2, 2.0];
    let r = harmonic_r_stat(&levels);
    assert!(r.is_finite() && r >= 0.0 && r <= 1.0,
        "harmonic_r_stat(clean): should return r ∈ [0,1], got {r}");
}

/// rank_von_neumann_ratio on clean data returns finite result near 2.0 (IID).
#[test]
fn rank_von_neumann_ratio_clean_baseline() {
    // Near-IID random-looking data
    let data: Vec<f64> = (0..50).map(|i| ((i * 7 + 3) % 11) as f64).collect();
    let r = rank_von_neumann_ratio(&data);
    assert!(r.is_finite(),
        "rank_von_neumann_ratio(clean): should be finite, got {r}");
}

/// harmonic_r_stat with fewer than 3 levels returns NaN.
#[test]
fn harmonic_r_stat_too_few_baseline() {
    assert!(harmonic_r_stat(&[1.0, 2.0]).is_nan(),
        "harmonic_r_stat(n<3) should return NaN");
}

// ─── Bug 1: harmonic_r_stat panics on NaN input ──────────────────────────────

/// `harmonic_r_stat(levels)` panics when any level is NaN — at large array sizes.
///
/// Rust's pdqsort uses insertion sort for small arrays (≤20 elements) which does
/// NOT assert total order. For larger arrays it switches to a pattern-defeating
/// quicksort with the debug assertion. The panic risk is real for large level sets.
///
/// Correct behavior at any size: return NaN (propagate invalid input).
/// Bug: runtime panic for large arrays; silent wrong answer for small arrays.
#[test]
fn harmonic_r_stat_nan_input_must_not_panic() {
    // Use 30 elements to ensure pdqsort's introsort path (not insertion sort)
    let mut levels: Vec<f64> = (0..30).map(|i| i as f64 * 0.1).collect();
    levels[15] = f64::NAN;

    let result = std::panic::catch_unwind(move || harmonic_r_stat(&levels));
    match result {
        Ok(r) => {
            // No panic: must return NaN (input contained NaN).
            assert!(r.is_nan(),
                "harmonic_r_stat(NaN in levels, n=30): should return NaN, got {r}");
        }
        Err(_) => {
            panic!(
                "harmonic_r_stat(NaN in levels[15], n=30) PANICKED — \
                 complexity.rs:1923 sort_by comparator fails total-order assertion for large arrays. \
                 Fix: sort_by(|a,b| a.total_cmp(b)) + upfront NaN guard."
            );
        }
    }
}

/// harmonic_r_stat with all-NaN levels (large array).
#[test]
fn harmonic_r_stat_all_nan_must_not_panic() {
    let levels = vec![f64::NAN; 30];
    let result = std::panic::catch_unwind(|| harmonic_r_stat(&levels));
    match result {
        Ok(r) => assert!(r.is_nan(),
            "harmonic_r_stat(all NaN, n=30): should return NaN, got {r}"),
        Err(_) => panic!(
            "harmonic_r_stat(all NaN, n=30) PANICKED — sort_by comparator bug at large n."
        ),
    }
}

/// harmonic_r_stat with INFINITY in levels — inf is sortable (no panic),
/// but result may be unexpected. Documents the boundary.
#[test]
fn harmonic_r_stat_inf_input_no_panic() {
    // INFINITY is totally ordered — partial_cmp returns Some(Greater).
    // Sort succeeds; spacings involving ∞ produce ∞; r_vals filtered by max < 1e-30.
    let levels = vec![0.1, f64::INFINITY, 0.7, 1.2];
    let result = std::panic::catch_unwind(|| harmonic_r_stat(&levels));
    // Should not panic (INFINITY is sortable), result may be NaN or finite.
    assert!(result.is_ok(),
        "harmonic_r_stat(INFINITY in levels) must not panic — INFINITY is totally ordered");
}

// ─── Bug 2: rank_von_neumann_ratio panics on NaN input ───────────────────────

/// `rank_von_neumann_ratio(data)` panics when any data value is NaN.
///
/// `data` is user-supplied time series. Missing observations are NaN.
/// The ranking sort at time_series.rs:2423 fails the total-order assertion.
///
/// Correct behavior: return NaN.
/// Bug: runtime panic.
#[test]
fn rank_von_neumann_ratio_nan_input_must_not_panic() {
    let mut data: Vec<f64> = (0..20).map(|i| i as f64).collect();
    data[10] = f64::NAN;

    let result = std::panic::catch_unwind(move || rank_von_neumann_ratio(&data));
    match result {
        Ok(r) => {
            assert!(r.is_nan(),
                "rank_von_neumann_ratio(NaN in data): should return NaN, got {r}");
        }
        Err(_) => {
            panic!(
                "rank_von_neumann_ratio(NaN in data[10]) PANICKED — \
                 time_series.rs:2423 sort_by comparator bug. \
                 Fix: add upfront NaN guard, then sort_by(|a,b| a.1.total_cmp(&b.1))."
            );
        }
    }
}

/// rank_von_neumann_ratio with NaN at the start of the series.
#[test]
fn rank_von_neumann_ratio_nan_at_start_must_not_panic() {
    let mut data: Vec<f64> = (0..15).map(|i| i as f64).collect();
    data[0] = f64::NAN;

    let result = std::panic::catch_unwind(move || rank_von_neumann_ratio(&data));
    match result {
        Ok(r) => assert!(r.is_nan(),
            "rank_von_neumann_ratio(NaN at start): should return NaN, got {r}"),
        Err(_) => panic!(
            "rank_von_neumann_ratio(NaN at start) PANICKED — same sort_by bug."
        ),
    }
}

// ─── Bug 3: superposition correlation sort (indirect path) ───────────────────

/// If `discover_correlation` is called with NaN data, the computed pearson/spearman/kendall
/// values are NaN, and the sort at superposition.rs:685 panics.
///
/// This documents the INDIRECT path: user NaN → computed NaN → sort panic.
/// The sort is over internal computed values, but the panic is still user-triggered.
use tambear::superposition::sweep_correlation;

#[test]
fn sweep_correlation_nan_input_must_not_panic() {
    let mut x: Vec<f64> = (0..30).map(|i| i as f64).collect();
    let y: Vec<f64> = (0..30).map(|i| i as f64 * 0.8 + 1.0).collect();
    x[15] = f64::NAN;

    let result = std::panic::catch_unwind(move || {
        sweep_correlation(&x, &y)
    });
    match result {
        Ok(_r) => {
            // No panic: result should have NaN view_agreement or NaN modal_value.
            // Either is acceptable — the key is no panic.
        }
        Err(_) => {
            panic!(
                "sweep_correlation(NaN in x) PANICKED — NaN propagated through \
                 pearson/spearman/kendall into the sort at superposition.rs:685. \
                 Fix: sort_by(|a,b| a.total_cmp(b)) in the modal_value computation."
            );
        }
    }
}

// ─── The fix: total_cmp is the correct comparator ────────────────────────────

/// Documents that total_cmp correctly handles NaN in sort contexts.
///
/// NaN sorts AFTER all finite values and infinities in IEEE 754 totalOrder.
/// This is the correct behavior for "invalid data goes to the end."
#[test]
fn total_cmp_sorts_nan_to_end() {
    let mut data = vec![3.0_f64, f64::NAN, 1.0, f64::NAN, 2.0];
    data.sort_by(|a, b| a.total_cmp(b));

    // Finite values sorted ascending, NaN values at end.
    assert_eq!(data[0], 1.0);
    assert_eq!(data[1], 2.0);
    assert_eq!(data[2], 3.0);
    assert!(data[3].is_nan());
    assert!(data[4].is_nan());
}

/// Documents that total_cmp also correctly handles INFINITY vs NaN ordering.
#[test]
fn total_cmp_orders_infinity_before_nan() {
    let mut data = vec![f64::INFINITY, f64::NAN, 1.0, f64::NEG_INFINITY];
    data.sort_by(|a, b| a.total_cmp(b));

    assert_eq!(data[0], f64::NEG_INFINITY);
    assert_eq!(data[1], 1.0);
    assert_eq!(data[2], f64::INFINITY);
    assert!(data[3].is_nan()); // NaN is last in totalOrder
}
