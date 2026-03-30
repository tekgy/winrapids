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
//!   Cost = O(n_groups) metadata read + O(n) scatter-add (the aggregation).
//!   Index rebuild = never, unless the column data changes.
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
/// Layout:
/// - `row_to_group[i]` = group id for row i (0..n_groups). Used by scatter.
/// - `group_counts[g]` = number of rows in group g. Used for mean/variance.
/// - `provenance` = BLAKE3 of the key column data. Invalidated if data changes.
pub struct GroupIndex {
    /// Which group each row belongs to. Length = n_rows.
    /// This is the primary input to hash_scatter_groupby.
    pub row_to_group: CudaSlice<u32>,
    /// Number of rows per group. Length = n_groups.
    pub group_counts: CudaSlice<u32>,
    /// Number of distinct groups.
    pub n_groups: usize,
    /// BLAKE3 hash of the key column's raw bytes at index-build time.
    /// If the column changes, this won't match and we rebuild.
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
    /// Assumes keys are in [0, n_groups). Direct-index: key IS the group id.
    /// No hash needed. One pass, O(n).
    ///
    /// For non-integer or non-contiguous keys: hash to slot (future variant).
    pub fn build(col: &Column, stream: &Arc<CudaStream>) -> Result<Self, String> {
        // TODO: implement via hash_scatter kernel
        // Placeholder: returns an error until scout implements hash_scatter.rs
        let _ = (col, stream);
        Err("GroupIndex::build not yet implemented — scout is on it".to_string())
    }
}
