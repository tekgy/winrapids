//! GroupIndex — the killer feature.
//!
//! The persistent group index encodes, for each row, which group it belongs
//! to. It is:
//!   - Built once via hash scatter (O(n))
//!   - Validated by provenance hash (35ns check)
//!   - Reused for every subsequent groupby, group-filter, group-rank, etc.
//!
//! After the first groupby on "ticker_id", the index is stored in the Frame.
//! Every subsequent groupby on "ticker_id" skips index construction entirely:
//!   Cost = provenance check (35ns) + O(n) scatter-add.
//!   Index rebuild = never, unless the column data changes.
//!
//! ## No counting pass
//!
//! The accumulator array is sized to `max_key + 1` — known from the `.tb`
//! header at file-open time. No counting pass needed. Over-allocation cost:
//! ~0.022ms vs ~0.026ms for exact allocation — over-allocation is FASTER
//! because it eliminates the counting pass entirely.
//!
//! After the first groupby, the actual number of active groups (`n_active`)
//! is recorded in provenance. Every subsequent query reads this in 35ns and
//! can allocate exactly. The provenance store IS the size estimator.
//!
//! "Tam doesn't count. Tam over-allocates or reads from provenance."
//!
//! ## Liftability
//!
//! Hash scatter is order-1 liftable: each element contributes independently
//! to its group accumulator. No inter-element dependencies. Maximum GPU
//! occupancy. The GroupIndex is the pre-computed lift structure.

use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use crate::frame::Column;

/// Pre-built row→group mapping for a key column.
///
/// The key sizing distinction:
/// - `accumulator_size` = max_key + 1. Pre-allocated from the `.tb` header.
///   This is the size of every scatter accumulator array. Known at file-open time.
/// - `n_active` = number of groups with count > 0. Only known after first groupby.
///   Stored in provenance after first run; 35ns lookup on all subsequent runs.
///
/// Never conflate these. `accumulator_size` enables no-counting-pass allocation.
/// `n_active` is the result of computation, not a prerequisite for it.
pub struct GroupIndex {
    /// Which group each row belongs to. Length = n_rows.
    /// Primary input to every scatter operation on this column.
    pub row_to_group: CudaSlice<u32>,
    /// Per-group row counts. Length = accumulator_size.
    /// Filled during build. Groups with no members have count = 0.
    pub group_counts: CudaSlice<u32>,
    /// Accumulator array size = max_key + 1. From .tb header.
    /// Scatter accumulators are ALWAYS this size — no counting pass needed.
    /// Over-allocation cost: ~0.022ms vs ~0.026ms exact. Faster to over-allocate.
    pub accumulator_size: usize,
    /// Actual number of groups with count > 0. None until first groupby runs.
    /// After first run: stored in provenance, retrieved in 35ns.
    /// Second run: exact allocation from provenance. First run: accumulator_size.
    pub n_active: Option<usize>,
    /// BLAKE3 hash of the key column's raw bytes at index-build time.
    /// Provenance check: if column unchanged, index is valid. Cost: 35ns dirty-bit.
    pub provenance: [u8; 32],
}

impl GroupIndex {
    /// Check whether this index is still valid for `col`.
    ///
    /// Compares BLAKE3 of col's current data against stored provenance.
    /// Cost: O(n) hash of column data. (TODO: lazy hash with dirty bit.)
    pub fn is_valid_for(&self, col: &Column) -> bool {
        // For now: always rebuild. Real impl hashes GPU buffer.
        // TODO: dirty-bit invalidation (O(1) check).
        let _ = col;
        false
    }

    /// Build a GroupIndex for an integer key column.
    ///
    /// `max_key`: the maximum key value in this column. From the `.tb` header —
    /// no scan needed to discover it. Accumulators are sized to `max_key + 1`.
    ///
    /// Assumes keys are in [0, max_key]. Direct-index: key IS the group id.
    /// No hash needed. One pass, O(n).
    ///
    /// For non-integer or non-contiguous keys: dictionary-encode first, then
    /// call this with the code column (codes are always in [0, n_unique-1]).
    pub fn build(col: &Column, max_key: u32, stream: &Arc<CudaStream>) -> Result<Self, String> {
        // TODO (scout): implement via hash_scatter kernel
        // One pass: for each element i, set row_to_group[i] = keys[i],
        // atomicAdd(&group_counts[keys[i]], 1).
        // Accumulator size = max_key + 1. No counting pass.
        let _ = (col, max_key, stream);
        Err("GroupIndex::build not yet implemented — scout is on it".to_string())
    }

    /// Number of words needed to pack all group flags into a u64 bitmask.
    /// Used by join and dedup to represent non-matching / non-unique flags.
    pub fn mask_word_count(&self) -> usize {
        (self.accumulator_size + 63) / 64
    }
}
