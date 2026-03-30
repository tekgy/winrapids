//! RefCenteredStatsOp — fast per-group variance without Welford's division.
//!
//! Welford's parallel merge requires division in the combine step (~100μs
//! dispatch tier). RefCentered avoids this: track sum and sum_sq separately
//! via scatter_add (both are pure addition, no division), then compute
//! variance once post-scatter.
//!
//! ## Algorithm
//!
//! For each group g with reference point ref_g (e.g., group mean estimate):
//!
//! ```text
//! for each element i in group g (parallel):
//!     centered = values[i] - ref_g
//!     atomicAdd(&group_sum[g],    centered)
//!     atomicAdd(&group_sum_sq[g], centered * centered)
//! ```
//!
//! Post-scatter (host-side, O(n_groups)):
//!   mean_g = ref_g + group_sum[g] / group_count[g]
//!   var_g  = group_sum_sq[g] / group_count[g] - (group_sum[g] / group_count[g])²
//!
//! ## Why RefCentered?
//!
//! Without centering: sum_sq accumulates large values (x²), catastrophic
//! cancellation when computing var = E[x²] - E[x]². With centering around
//! ref_g ≈ group mean: centered values are small, cancellation is bounded.
//!
//! ## Cost
//!
//! Two scatter_adds (both atomicAdd, O(n) total). No division in the scatter.
//! Dispatch cost: ~40μs (simple combine tier, same as AddOp).
//! Post-scatter: O(n_groups) — negligible.

// TODO (naturalist): implement RefCenteredStatsEngine

/// Per-group statistics computed by RefCentered scatter.
pub struct GroupStats {
    pub means: Vec<f64>,
    pub variances: Vec<f64>,
    pub counts: Vec<u32>,
}

/// RefCentered per-group mean + variance engine.
///
/// Uses two scatter_adds (sum and sum_sq) in one GPU pass, then
/// derives statistics post-scatter. No Welford division in the hot path.
pub struct RefCenteredStatsEngine {
    // TODO (naturalist): kernel cache, stream
}

impl RefCenteredStatsEngine {
    pub fn new() -> Result<Self, String> {
        Err("RefCenteredStatsEngine::new not yet implemented".to_string())
    }

    /// Compute per-group mean and variance in one pass.
    ///
    /// # Arguments
    /// - `row_to_group`: for each row, its group id (0..n_groups)
    /// - `values`: the values to aggregate
    /// - `n_groups`: number of distinct groups
    /// - `ref_values`: reference points for centering, one per group.
    ///   Pass `None` to use 0.0 (no centering — accepts cancellation risk).
    pub fn group_stats(
        &mut self,
        row_to_group: &[u32],
        values: &[f64],
        n_groups: usize,
        ref_values: Option<&[f64]>,
    ) -> Result<GroupStats, String> {
        let _ = (row_to_group, values, n_groups, ref_values);
        Err("group_stats not yet implemented".to_string())
    }
}
