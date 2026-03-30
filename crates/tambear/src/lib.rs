//! # tambear
//!
//! Sort-free GPU DataFrame engine.
//!
//! Tam doesn't sort. Tam knows where everything is.
//!
//! The core insight: groupby, deduplication, join, and top-k selection
//! all traditionally use sort as an implementation convenience. On GPU,
//! hash scatter is 17x faster — O(n) with one pass versus O(n log n)
//! with random-access memory writes.
//!
//! ## Four architectural invariants
//!
//! Each eliminates an operation by carrying information forward instead of
//! recovering it later.
//!
//! **Sort-free**: GroupBy/Dedup/Join/TopK → hash ops, never sort.
//! Sort emitted only for `sort_values`, `rank`. 17x on GPU.
//! "Tam doesn't sort. Tam knows."
//!
//! **Mask-not-filter**: Filter sets bits in `Frame::row_mask` (1 bit/row,
//! packed u64). Downstream ops are mask-aware. No compaction, no new array.
//! "Tam doesn't filter. Tam knows which rows matter."
//!
//! **Dictionary strings**: String columns are int codes at ingestion.
//! Dictionary lives in the `.tb` header. GroupBy on strings = GroupBy on ints.
//! Decode only at output.
//! "Tam doesn't do strings. Tam knows the dictionary."
//!
//! **Types once**: `tb.pipeline(dtype=tb.float64)` — single declared dtype.
//! JIT kernels generated for the declared type. Zero runtime dispatch.
//! "Tam doesn't check types. Tam knows the schema."
//!
//! ## Quick Start
//!
//! ```no_run
//! use tambear::HashScatterEngine;
//!
//! let engine = HashScatterEngine::new().unwrap();
//! let keys = vec![0i32, 0, 1, 1, 2];
//! let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
//!
//! let sums = engine.scatter_sum(&keys, &values, 3).unwrap();
//! // [3.0, 7.0, 5.0] — group 0: 1+2, group 1: 3+4, group 2: 5
//! ```

pub mod dictionary;
pub mod format;
pub mod frame;
pub mod group_index;
pub mod hash_scatter;
pub mod stats;
pub mod tb_io;

pub use dictionary::Dictionary;
pub use format::{
    TileColumnStats, TbColumnDescriptor, TbFileHeader,
    global_min, global_max, global_sum, global_count, global_mean,
    tile_skip_mask_gt, tile_skip_mask_lt,
    TB_MAGIC, TB_VERSION, TB_TILE_SIZE_DEFAULT, FILE_HEADER_SIZE, TB_MAX_COLUMNS,
};
pub use frame::{Column, ColumnEncoding, DType, Frame};
pub use group_index::GroupIndex;
pub use hash_scatter::{HashScatterEngine, GroupByResult};
pub use tb_io::{TbFile, TbColumnWrite, write_tb};
