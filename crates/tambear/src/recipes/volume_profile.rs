//! Volume profile — fraction of total volume in each time-block.
//!
//! Locked vocabulary: this is a Tier 4 recipe — composition over a
//! `accumulate(Grouping::ByKey, Op::Add)` followed by normalization.
//! See `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Math
//!
//! Given per-event quantities and matching event timestamps, partition
//! the events into `n_blocks` equal time blocks spanning
//! `[window_start_ns, window_end_ns)`, sum quantity within each block,
//! and normalize:
//!
//! ```text
//! block_id[i]    =  floor( (timestamps[i] − window_start) / block_width )
//! block_sum[k]   =  Σ qty[i]  for events with block_id[i] == k
//! total          =  Σ block_sum[k]
//! profile[k]     =  block_sum[k] / total           if total > 0
//!                =  0                              if total == 0
//! ```
//!
//! Returns a vector of length `n_blocks`, each entry in `[0, 1]`,
//! summing to 1 (modulo the all-empty case which returns all-zeros).
//!
//! # Composition
//!
//! - **ByKey scatter** of quantity into block-id buckets — Kingdom A,
//!   Kulisch-backed via `tambear::math::sum`-style accumulation done
//!   per-block in this recipe (will lower to a single
//!   `accumulate(qty, Grouping::ByKey{block_id}, Op::Add)` atom call
//!   when the upstream pipeline carries that grouping directly).
//! - **One reduction** for the total — `tambear::math::sum` over
//!   block_sum.
//! - **Pointwise normalization** — divide each block_sum by total.
//!
//! # NaN/Inf policy
//!
//! Any tick whose `qty` is non-finite is skipped (Kulisch-style
//! is_finite gate). Ticks whose timestamp falls outside
//! `[window_start_ns, window_end_ns)` are dropped silently — those are
//! out-of-window arrivals and not part of this hour's profile.
//!
//! # Default parameters
//!
//! - `n_blocks` — caller-provided. SIP uses 12 (twelve 5-minute blocks
//!   per hour). No default; the right value depends on cadence.
//! - `window_start_ns`, `window_end_ns` — caller-provided. The full
//!   time range covered by the profile.

use crate::primitives::specialist::kulisch_accumulator::KulischAccumulator;

/// Per-block volume fractions over a time window.
///
/// # Panics
///
/// - If `n_blocks == 0`.
/// - If `qty.len() != timestamps_ns.len()`.
/// - If `window_end_ns <= window_start_ns`.
pub fn volume_profile(
    qty: &[f64],
    timestamps_ns: &[i64],
    window_start_ns: i64,
    window_end_ns: i64,
    n_blocks: usize,
) -> Vec<f64> {
    assert!(n_blocks > 0, "volume_profile: n_blocks must be > 0");
    assert_eq!(
        qty.len(),
        timestamps_ns.len(),
        "volume_profile: qty and timestamps must have same length"
    );
    assert!(
        window_end_ns > window_start_ns,
        "volume_profile: window must be non-empty"
    );

    // Per-block Kulisch accumulators — exact summation, deterministic across
    // backends and thread counts when this lowers to a `Grouping::ByKey`
    // atom call.
    let mut block_accs: Vec<KulischAccumulator> =
        (0..n_blocks).map(|_| KulischAccumulator::new()).collect();

    let window_width = (window_end_ns - window_start_ns) as f64;
    let block_width = window_width / n_blocks as f64;

    for i in 0..qty.len() {
        let q = qty[i];
        if !q.is_finite() {
            continue;
        }
        let ts = timestamps_ns[i];
        if ts < window_start_ns || ts >= window_end_ns {
            continue;
        }
        let offset = (ts - window_start_ns) as f64;
        let mut block_id = (offset / block_width) as usize;
        // Defensive: floating-point edge case might land an event exactly
        // at the boundary into n_blocks; clamp.
        if block_id >= n_blocks {
            block_id = n_blocks - 1;
        }
        block_accs[block_id].add_f64(q);
    }

    // Materialize per-block sums.
    let block_sums: Vec<f64> = block_accs.iter().map(|a| a.to_f64()).collect();

    // Total — Kulisch-exact sum of the per-block sums (themselves Kulisch-exact).
    let total = crate::math::sum(&block_sums);
    if total <= 0.0 {
        return vec![0.0; n_blocks];
    }

    block_sums.into_iter().map(|s| s / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input_returns_zeros() {
        let p = volume_profile(&[], &[], 0, 1_000_000_000, 12);
        assert_eq!(p, vec![0.0; 12]);
    }

    #[test]
    fn uniform_distribution_uniform_profile() {
        // 12 ticks spread evenly across 12 blocks, each with qty=1.
        // Profile should be [1/12, 1/12, ..., 1/12].
        let n = 12;
        let qty: Vec<f64> = (0..n).map(|_| 1.0).collect();
        let ts: Vec<i64> = (0..n).map(|i| (i * 100_000_000) as i64).collect(); // 0, 100ms, 200ms, ...
        let p = volume_profile(&qty, &ts, 0, 1_200_000_000, n);
        let expected = 1.0 / n as f64;
        for (k, &fr) in p.iter().enumerate() {
            assert!((fr - expected).abs() < 1e-12, "block {k}: got {fr}");
        }
        let total: f64 = p.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
    }

    #[test]
    fn all_in_one_block_concentrated_profile() {
        // 5 ticks all land in block 0; rest empty.
        let qty = vec![10.0, 5.0, 3.0, 2.0, 1.0];
        let ts = vec![0_i64, 50_000_000, 70_000_000, 80_000_000, 95_000_000]; // all under 100ms
        let n = 4;
        let p = volume_profile(&qty, &ts, 0, 400_000_000, n);
        assert_eq!(p[0], 1.0); // all volume in block 0
        for k in 1..n {
            assert_eq!(p[k], 0.0);
        }
    }

    #[test]
    fn profile_normalizes_to_one() {
        let qty: Vec<f64> = (1..=20).map(|i| i as f64).collect();
        let ts: Vec<i64> = (0..20).map(|i| (i * 50_000_000) as i64).collect(); // every 50ms
        let p = volume_profile(&qty, &ts, 0, 1_000_000_000, 10);
        let total: f64 = p.iter().sum();
        assert!((total - 1.0).abs() < 1e-12, "total = {total}");
    }

    #[test]
    fn skips_non_finite_qty() {
        let qty = vec![10.0, f64::NAN, 5.0, f64::INFINITY, 5.0];
        let ts = vec![0_i64, 100_000_000, 200_000_000, 300_000_000, 400_000_000];
        let p = volume_profile(&qty, &ts, 0, 500_000_000, 5);
        // Only finite entries contribute: 10, 5, 5 → total 20.
        assert_eq!(p[0], 0.5); // 10/20
        assert_eq!(p[1], 0.0); // NaN skipped
        assert_eq!(p[2], 0.25); // 5/20
        assert_eq!(p[3], 0.0); // Inf skipped
        assert_eq!(p[4], 0.25); // 5/20
    }

    #[test]
    fn drops_out_of_window_ticks() {
        // Window [100, 500). Ticks at 50 and 600 are dropped.
        let qty = vec![1.0, 1.0, 1.0, 1.0];
        let ts = vec![50_i64, 200, 400, 600];
        let p = volume_profile(&qty, &ts, 100, 500, 4);
        let total: f64 = p.iter().sum();
        assert!((total - 1.0).abs() < 1e-12);
        // 200 → block 0 ((200-100)/100 = 1 → block 1; window 100-500, width 100)
        // 400 → block 3 ((400-100)/100 = 3)
        // So block 1 = 0.5, block 3 = 0.5
        assert_eq!(p[1], 0.5);
        assert_eq!(p[3], 0.5);
    }

    #[test]
    #[should_panic(expected = "n_blocks")]
    fn panics_on_zero_blocks() {
        let _ = volume_profile(&[], &[], 0, 100, 0);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn panics_on_mismatched_lengths() {
        let _ = volume_profile(&[1.0, 2.0], &[0_i64], 0, 100, 4);
    }

    #[test]
    #[should_panic(expected = "non-empty")]
    fn panics_on_empty_window() {
        let _ = volume_profile(&[], &[], 100, 100, 4);
    }
}
