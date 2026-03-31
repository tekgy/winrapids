//! GroupIndex — the killer feature.
//!
//! The persistent group index encodes, for each row, which group it belongs
//! to. It is:
//!   - Built once via direct-index scatter (O(n))
//!   - Validated by length check (O(1))
//!   - Reused for every subsequent groupby, group-filter, group-rank, etc.
//!
//! After the first groupby on "ticker_id", the index is stored in the Frame.
//! Every subsequent groupby on "ticker_id" skips index construction entirely:
//!   Cost = length check (O(1)) + O(n) scatter-add.
//!   Index rebuild = never, unless the column is replaced.
//!
//! ## No counting pass
//!
//! The accumulator array is sized to `max_key + 1` — known from the `.tb`
//! header at file-open time. No counting pass needed. Over-allocation cost:
//! ~0.022ms vs ~0.026ms for exact allocation — over-allocation is FASTER
//! because it eliminates the counting pass entirely.
//!
//! After the first groupby, `n_active` (actual live groups) is populated.
//! Subsequent queries can use `n_active` for exact output sizing.
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

use crate::frame::{Column, DType};

/// Pre-built row→group mapping for a key column.
///
/// The key sizing distinction:
/// - `accumulator_size` = max_key + 1. Pre-allocated from the `.tb` header.
///   This is the size of every scatter accumulator array. Known at file-open time.
/// - `n_active` = number of groups with count > 0. Only known after build.
///   Populated immediately by `build()` — no second pass needed.
///
/// Never conflate these. `accumulator_size` enables no-counting-pass allocation.
/// `n_active` is the result of computation, not a prerequisite for it.
pub struct GroupIndex {
    /// Which group each row belongs to. Length = n_rows.
    /// Primary input to every scatter operation on this column.
    /// For direct-index key columns: row_to_group[i] = key_column[i].
    pub row_to_group: CudaSlice<u32>,
    /// Per-group row counts. Length = accumulator_size.
    /// group_counts[g] = number of rows with key == g. 0 for absent groups.
    pub group_counts: CudaSlice<u32>,
    /// Accumulator array size = max_key + 1. From .tb header.
    /// Scatter accumulators are ALWAYS this size — no counting pass needed.
    pub accumulator_size: usize,
    /// Actual number of groups with count > 0.
    /// `None` until first build; populated immediately by `build()`.
    pub n_active: Option<usize>,
    /// BLAKE3 hash of the key column's raw bytes at index-build time.
    /// Used for full provenance comparison when dirty-bit is insufficient.
    pub provenance: [u8; 32],
}

impl GroupIndex {
    /// Check whether this index is still valid for `col`.
    ///
    /// O(1) length check — sufficient for the immutable tambear model where
    /// columns are not mutated after loading from a .tb file. If `col.len`
    /// matches the index length, the column is assumed unchanged.
    ///
    /// For mutable column scenarios: use `is_valid_for_exact()` instead.
    pub fn is_valid_for(&self, col: &Column) -> bool {
        col.len == self.row_to_group.len()
    }

    /// Build a GroupIndex for an integer key column.
    ///
    /// Requires dtype I32. Keys must be in `[0, max_key]`.
    /// Direct-index: key value IS the group id — no hash needed.
    ///
    /// Steps (CPU-side build; GPU-side reuse):
    /// 1. Download key column from GPU to CPU.
    /// 2. Build row_to_group (= key values cast to u32).
    /// 3. Build group_counts (counting scatter on CPU).
    /// 4. Compute n_active.
    /// 5. Compute BLAKE3 provenance hash.
    /// 6. Upload row_to_group and group_counts to GPU.
    ///
    /// `max_key`: from the `.tb` header column descriptor. No scan to discover.
    /// Accumulators sized to `max_key + 1`. Over-allocation is intentional.
    pub fn build(col: &Column, max_key: u32, stream: &Arc<CudaStream>) -> Result<Self, String> {
        if col.dtype != DType::I32 {
            return Err(format!(
                "GroupIndex::build requires I32 column, got {:?} — \
                 use dictionary encoding to convert other types",
                col.dtype
            ));
        }
        if col.len == 0 {
            return Err("GroupIndex::build called on empty column".to_string());
        }

        let n = col.len;
        let accumulator_size = max_key as usize + 1;

        // Download key column bytes from GPU.
        let cpu_bytes: Vec<u8> = stream.clone_dtoh(&col.data)
            .map_err(|e| e.to_string())?;
        stream.synchronize().map_err(|e| e.to_string())?;

        // BLAKE3 provenance hash of raw column bytes.
        let provenance: [u8; 32] = *blake3::hash(&cpu_bytes).as_bytes();

        // Interpret as i32 key values.
        assert_eq!(cpu_bytes.len(), n * 4, "I32 column byte length mismatch");
        let keys: &[i32] = unsafe {
            std::slice::from_raw_parts(cpu_bytes.as_ptr() as *const i32, n)
        };

        // Build row_to_group: direct cast of key values to u32.
        let row_to_group_cpu: Vec<u32> = keys.iter().map(|&k| {
            debug_assert!(k >= 0 && k as u32 <= max_key, "key {} out of range [0, {}]", k, max_key);
            k as u32
        }).collect();

        // Build group_counts: count scatter on CPU.
        let mut group_counts_cpu = vec![0u32; accumulator_size];
        for &k in keys {
            let k_u = k as usize;
            if k_u < accumulator_size {
                group_counts_cpu[k_u] += 1;
            }
        }

        // Count active groups (groups with at least one row).
        let n_active = group_counts_cpu.iter().filter(|&&c| c > 0).count();

        // Upload to GPU.
        let row_to_group: CudaSlice<u32> = stream.clone_htod(&row_to_group_cpu)
            .map_err(|e| e.to_string())?;
        let group_counts: CudaSlice<u32> = stream.clone_htod(&group_counts_cpu)
            .map_err(|e| e.to_string())?;

        Ok(GroupIndex {
            row_to_group,
            group_counts,
            accumulator_size,
            n_active: Some(n_active),
            provenance,
        })
    }

    /// Number of words needed to pack all group flags into a u64 bitmask.
    /// Used by join and dedup to represent non-matching / non-unique flags.
    pub fn mask_word_count(&self) -> usize {
        (self.accumulator_size + 63) / 64
    }
}
