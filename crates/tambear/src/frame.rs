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

/// How a column's values are encoded.
///
/// Dictionary encoding: string columns become integer codes at ingestion.
/// The dictionary (code → string) lives in the `.tb` header and in this enum.
/// GroupBy on a Dictionary column operates on the codes — hash scatter on ints.
/// Strings are decoded back to human-readable values ONLY at output.
/// "Tam doesn't do strings. Tam knows the dictionary."
#[derive(Debug)]
pub enum ColumnEncoding {
    /// Values are stored directly in `data` as their declared DType.
    Raw,
    /// Values are dictionary-encoded integer codes (u32) stored in `data`.
    /// `dict[code]` gives the original string value.
    Dictionary { dict: Vec<String> },
}

/// A named, typed, GPU-resident column buffer.
///
/// Data lives on GPU as raw bytes. The DType tag controls how
/// the bytes are interpreted by kernels. No CPU copy is kept
/// unless explicitly requested.
///
/// String columns use `ColumnEncoding::Dictionary` — the `data` buffer holds
/// u32 codes; the dictionary maps codes back to strings at output time.
pub struct Column {
    pub name: String,
    pub dtype: DType,
    pub encoding: ColumnEncoding,
    pub len: usize,
    /// Raw byte buffer on GPU. Length = len * dtype.byte_size().
    pub data: CudaSlice<u8>,
}

impl Column {
    /// Byte length of this column's GPU buffer.
    pub fn byte_len(&self) -> usize {
        self.len * self.dtype.byte_size()
    }

    /// True if this column is dictionary-encoded (was a string column at ingestion).
    pub fn is_dictionary(&self) -> bool {
        matches!(self.encoding, ColumnEncoding::Dictionary { .. })
    }
}

/// A GPU-resident DataFrame.
///
/// Upholds four invariants (see lib.rs):
/// 1. Sort-free: GroupBy/Dedup/Join/TopK never emit sort.
/// 2. Mask-not-filter: `filter()` sets `row_mask` bits; no compaction.
/// 3. Dictionary strings: string columns are int codes; decode at output only.
/// 4. Types once: `pipeline_dtype` declared at construction; no per-op casting.
pub struct Frame {
    pub n_rows: usize,
    /// Columns indexed by name.
    pub columns: HashMap<String, Column>,
    /// Pre-built GroupIndices indexed by key column name.
    /// Empty until first groupby on that column. Provenance-checked on reuse.
    pub group_indices: HashMap<String, crate::GroupIndex>,
    /// Row mask for mask-not-filter. 1 bit per row, packed into u64 words.
    /// `None` means all rows are active (no filter applied).
    /// Length (in u64 words) = ceil(n_rows / 64).
    /// "Tam doesn't filter. Tam knows which rows matter."
    pub row_mask: Option<CudaSlice<u64>>,
    /// Pipeline-wide numeric dtype. Declared once at construction.
    /// All numeric kernels JIT for this type. No per-operation type dispatch.
    /// "Tam doesn't check types. Tam knows the schema."
    pub pipeline_dtype: DType,
    pub(crate) stream: Arc<CudaStream>,
}

impl Frame {
    /// Create an empty Frame with no columns.
    ///
    /// `pipeline_dtype` is the declared numeric type for this pipeline.
    /// All JIT kernels will be generated for this type.
    pub fn new(ctx: &Arc<CudaContext>, pipeline_dtype: DType) -> Result<Self, cudarc::driver::DriverError> {
        let stream = ctx.default_stream();
        Ok(Frame {
            n_rows: 0,
            columns: HashMap::new(),
            group_indices: HashMap::new(),
            row_mask: None,
            pipeline_dtype,
            stream: Arc::new(stream),
        })
    }

    /// Number of u64 words needed to hold the row mask.
    pub fn mask_word_count(&self) -> usize {
        (self.n_rows + 63) / 64
    }

    /// True if any filter has been applied (row_mask is Some).
    pub fn is_filtered(&self) -> bool {
        self.row_mask.is_some()
    }

    /// Get a column by name.
    pub fn col(&self, name: &str) -> Option<&Column> {
        self.columns.get(name)
    }

    /// Retrieve or build a GroupIndex for `key_col`.
    ///
    /// `max_key`: maximum key value for this column. From the `.tb` header —
    /// caller provides it; the frame does not scan to discover it.
    /// Accumulators are sized to `max_key + 1`. No counting pass.
    ///
    /// Cost on cache hit: 35ns provenance check.
    /// Cost on cache miss: O(n) hash scatter kernel (one pass).
    pub fn group_index_for(&mut self, key_col: &str, max_key: u32) -> Result<&crate::GroupIndex, String> {
        if let Some(idx) = self.group_indices.get(key_col) {
            let col = self.columns.get(key_col)
                .ok_or_else(|| format!("column '{}' not found", key_col))?;
            if idx.is_valid_for(col) {
                return Ok(self.group_indices.get(key_col).unwrap());
            }
        }
        // Build or rebuild. max_key from caller (from .tb header).
        let col = self.columns.get(key_col)
            .ok_or_else(|| format!("column '{}' not found", key_col))?;
        let idx = crate::GroupIndex::build(col, max_key, &self.stream)?;
        self.group_indices.insert(key_col.to_string(), idx);
        Ok(self.group_indices.get(key_col).unwrap())
    }
}
