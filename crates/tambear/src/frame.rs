//! Core DataFrame types.
//!
//! `Frame` is a GPU-resident collection of named, typed columns.
//! Columns stay on GPU until explicitly evicted. The GroupIndex
//! is stored alongside the data — built once, never rebuilt
//! unless provenance invalidates.

use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};

/// Element data type. Mirrors winrapids-store DType — unified later.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32 = 0,
    F64 = 1,
    I32 = 2,
    I64 = 3,
    U32 = 4,
    U64 = 5,
}

impl DType {
    pub fn byte_size(self) -> usize {
        match self {
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F64 | DType::I64 | DType::U64 => 8,
        }
    }
}

/// A named, typed, GPU-resident column buffer.
///
/// Data lives on GPU as raw bytes. The DType tag controls how
/// the bytes are interpreted by kernels. No CPU copy is kept
/// unless explicitly requested.
pub struct Column {
    pub name: String,
    pub dtype: DType,
    pub len: usize,
    /// Raw byte buffer on GPU. Length = len * dtype.byte_size().
    pub data: CudaSlice<u8>,
}

impl Column {
    /// Byte length of this column's GPU buffer.
    pub fn byte_len(&self) -> usize {
        self.len * self.dtype.byte_size()
    }
}

/// A GPU-resident DataFrame.
///
/// Columns are stored by name. GroupIndices are stored by key column name —
/// built once on first groupby, reused forever (provenance-keyed).
///
/// Sort-free invariant: no sort is ever emitted for groupby, join, dedup, or
/// top-k. GroupIndex encodes group membership; hash scatter does the rest.
pub struct Frame {
    pub n_rows: usize,
    /// Columns indexed by name.
    pub columns: HashMap<String, Column>,
    /// Pre-built GroupIndices indexed by key column name.
    /// Empty until first groupby on that column. Provenance-checked on reuse.
    pub group_indices: HashMap<String, crate::GroupIndex>,
    pub(crate) stream: Arc<CudaStream>,
}

impl Frame {
    /// Create an empty Frame with no columns.
    pub fn new(ctx: &Arc<CudaContext>) -> Result<Self, cudarc::driver::DriverError> {
        let stream = ctx.fork_default_stream()?;
        Ok(Frame {
            n_rows: 0,
            columns: HashMap::new(),
            group_indices: HashMap::new(),
            stream: Arc::new(stream),
        })
    }

    /// Get a column by name.
    pub fn col(&self, name: &str) -> Option<&Column> {
        self.columns.get(name)
    }

    /// Retrieve or build a GroupIndex for `key_col`.
    ///
    /// If a valid GroupIndex already exists (provenance matches), returns it
    /// without rebuilding. Otherwise builds via hash scatter and caches.
    /// Cost: O(n_groups) metadata check if valid; O(n) hash scatter if not.
    pub fn group_index_for(&mut self, key_col: &str) -> Result<&crate::GroupIndex, String> {
        if let Some(idx) = self.group_indices.get(key_col) {
            let col = self.columns.get(key_col)
                .ok_or_else(|| format!("column '{}' not found", key_col))?;
            if idx.is_valid_for(col) {
                return Ok(self.group_indices.get(key_col).unwrap());
            }
        }
        // Build or rebuild the index.
        let col = self.columns.get(key_col)
            .ok_or_else(|| format!("column '{}' not found", key_col))?;
        let idx = crate::GroupIndex::build(col, &self.stream)?;
        self.group_indices.insert(key_col.to_string(), idx);
        Ok(self.group_indices.get(key_col).unwrap())
    }
}
