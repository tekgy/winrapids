//! .tb file format — GPU-native tiled DataFrame storage.
//!
//! ## Layout (byte offsets from file start)
//!
//! ```text
//! [0..4096)                  File header (TbFileHeader, always 4096 bytes)
//! [4096..4096+tile_hdr_sz)   Tile header section: n_tiles × n_columns × TileColumnStats
//! [data_section_offset..)    Column data: column-major, tiled
//! [provenance_offset..)      Provenance tail (appended by each execution)
//! ```
//!
//! ## The headline feature: header-only aggregation
//!
//! All tile headers are contiguous immediately after the 4096-byte file header.
//! For 1000 tiles × 5 columns: 200KB. One read gives you global stats on 40MB.
//!
//! "Tam doesn't read. Tam knows the summary."
//!
//! ## Column-major tiled data layout
//!
//! ```text
//! Column 0: [tile_0_data | tile_1_data | ... | tile_{n_tiles-1}_data]
//! Column 1: [tile_0_data | tile_1_data | ... | tile_{n_tiles-1}_data]
//! ...
//! ```
//!
//! Loading column C, tiles J through K = one contiguous range read.
//! Predicate pushdown = load only a subset of that range.
//!
//! ## Tile size
//!
//! Default: 65,536 rows (2^16).
//! - 65,536 × 8 bytes = 512KB per column per tile — fits GPU L2 cache.
//! - Granular enough for 56% predicate skip rate in practice.
//! - Store `tile_size` in the header — different datasets may want different sizes.

pub const TB_MAGIC: [u8; 8] = *b"TAMBEAR\0";
pub const TB_VERSION: u32 = 1;
/// Default tile size in rows. 2^16 = 65,536. 512KB per column per tile.
pub const TB_TILE_SIZE_DEFAULT: u32 = 65_536;
/// Exact byte size of the file header. Always 4096.
pub const FILE_HEADER_SIZE: usize = 4096;
/// Maximum number of columns in a .tb file.
pub const TB_MAX_COLUMNS: usize = 64;

// ---------------------------------------------------------------------------
// TileColumnStats — 40 bytes per (tile, column) pair
// ---------------------------------------------------------------------------

/// Per-tile, per-column statistics. 40 bytes. Contiguous in tile header section.
///
/// All tile stats live at file start → "read once, know everything."
/// The `_reserved` slot is pre-built for variance pushdown (sum_sq).
/// Anti-YAGNI: the structure guarantees we'll want variance pushdown.
/// Build the slot now; fill it later.
///
/// During write: compute `sum` via Kahan summation to avoid cancellation.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct TileColumnStats {
    /// Minimum value in this tile.
    pub min: f64,
    /// Maximum value in this tile.
    pub max: f64,
    /// Sum of all valid values in this tile (Kahan-accurate during write).
    pub sum: f64,
    /// Number of rows stored in this tile. May be < tile_size for the last tile.
    pub count: u32,
    /// Number of non-null rows. Equal to `count` for dense columns.
    pub n_valid: u32,
    /// Reserved: sum of squares for variance pushdown. Not yet populated.
    /// When populated: `sum_sq / count - (sum / count)^2` = tile variance.
    pub _reserved: u64,
}

const _: () = assert!(std::mem::size_of::<TileColumnStats>() == 40, "TileColumnStats must be 40 bytes");

impl TileColumnStats {
    /// Build stats by scanning a slice of f64 values (one tile, one column).
    ///
    /// Uses Kahan compensated summation for `sum` accuracy.
    /// Call this once per (tile, column) during the write path.
    pub fn from_slice(values: &[f64]) -> Self {
        if values.is_empty() {
            return TileColumnStats::default();
        }
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        // Kahan summation for accuracy on large prices with small variance.
        let mut sum = 0.0_f64;
        let mut kahan_c = 0.0_f64;
        for &v in values {
            if v < min { min = v; }
            if v > max { max = v; }
            let y = v - kahan_c;
            let t = sum + y;
            kahan_c = (t - sum) - y;
            sum = t;
        }
        TileColumnStats {
            min,
            max,
            sum,
            count: values.len() as u32,
            n_valid: values.len() as u32,
            _reserved: 0,
        }
    }

    /// True if this tile can be skipped for a `column > threshold` predicate.
    ///
    /// If the tile's max is ≤ threshold, no row in this tile can satisfy
    /// `column > threshold` — skip the entire tile with zero I/O.
    #[inline]
    pub fn can_skip_gt(&self, threshold: f64) -> bool {
        self.max <= threshold
    }

    /// True if this tile can be skipped for a `column < threshold` predicate.
    #[inline]
    pub fn can_skip_lt(&self, threshold: f64) -> bool {
        self.min >= threshold
    }

    /// True if this tile can be skipped for a `column >= threshold` predicate.
    #[inline]
    pub fn can_skip_gte(&self, threshold: f64) -> bool {
        self.max < threshold
    }

    /// True if this tile can be skipped for a `column <= threshold` predicate.
    #[inline]
    pub fn can_skip_lte(&self, threshold: f64) -> bool {
        self.min > threshold
    }
}

// ---------------------------------------------------------------------------
// Global aggregation from tile headers — header-only queries
// ---------------------------------------------------------------------------

/// Global minimum across all tiles for one column.
/// O(n_tiles). No data section read.
pub fn global_min(tile_stats: &[TileColumnStats]) -> f64 {
    tile_stats.iter().filter(|s| s.n_valid > 0)
        .map(|s| s.min)
        .fold(f64::INFINITY, f64::min)
}

/// Global maximum across all tiles for one column.
/// O(n_tiles). No data section read.
pub fn global_max(tile_stats: &[TileColumnStats]) -> f64 {
    tile_stats.iter().filter(|s| s.n_valid > 0)
        .map(|s| s.max)
        .fold(f64::NEG_INFINITY, f64::max)
}

/// Global sum across all tiles for one column.
/// O(n_tiles). No data section read.
pub fn global_sum(tile_stats: &[TileColumnStats]) -> f64 {
    tile_stats.iter().map(|s| s.sum).sum()
}

/// Global row count across all tiles.
pub fn global_count(tile_stats: &[TileColumnStats]) -> u64 {
    tile_stats.iter().map(|s| s.count as u64).sum()
}

/// Global mean across all tiles for one column.
/// O(n_tiles). No data section read.
pub fn global_mean(tile_stats: &[TileColumnStats]) -> f64 {
    let count = global_count(tile_stats) as f64;
    if count == 0.0 { f64::NAN } else { global_sum(tile_stats) / count }
}

/// Build a tile skip mask for a `column > threshold` predicate.
///
/// Returns a `Vec<bool>` of length `n_tiles`. Entry `i` is `true` if tile
/// `i` can be skipped (its max ≤ threshold, so no row passes the predicate).
///
/// Use the returned mask to avoid reading skipped tiles from the data section.
pub fn tile_skip_mask_gt(tile_stats: &[TileColumnStats], threshold: f64) -> Vec<bool> {
    tile_stats.iter().map(|s| s.can_skip_gt(threshold)).collect()
}

/// Build a tile skip mask for a `column < threshold` predicate.
pub fn tile_skip_mask_lt(tile_stats: &[TileColumnStats], threshold: f64) -> Vec<bool> {
    tile_stats.iter().map(|s| s.can_skip_lt(threshold)).collect()
}

// ---------------------------------------------------------------------------
// TbColumnDescriptor — 56 bytes per column, embedded in TbFileHeader
// ---------------------------------------------------------------------------

/// Per-column metadata, stored in the file header.
/// 56 bytes. Up to TB_MAX_COLUMNS (64) slots in TbFileHeader.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct TbColumnDescriptor {
    /// Column name, null-padded. Max 31 UTF-8 chars.
    pub name: [u8; 32],
    /// Element data type. Matches DType repr: F32=0, F64=1, I32=2, I64=3, U32=4, U64=5.
    pub dtype: u8,
    /// Encoding: Raw=0, Dictionary=1.
    pub encoding: u8,
    pub _pad: [u8; 2],
    /// For key/dictionary columns: maximum key value. Sizes GroupIndex accumulators.
    /// 0 for Raw columns.
    pub max_key: u32,
    pub _reserved: [u8; 16],
}

const _: () = assert!(std::mem::size_of::<TbColumnDescriptor>() == 56, "TbColumnDescriptor must be 56 bytes");

impl TbColumnDescriptor {
    /// Build a descriptor from a name string and DType byte.
    pub fn new(name: &str, dtype: u8, encoding: u8, max_key: u32) -> Self {
        let mut buf = [0u8; 32];
        let bytes = name.as_bytes();
        let len = bytes.len().min(31);
        buf[..len].copy_from_slice(&bytes[..len]);
        TbColumnDescriptor {
            name: buf,
            dtype,
            encoding,
            _pad: [0; 2],
            max_key,
            _reserved: [0; 16],
        }
    }

    /// Column name as a &str. Strips null padding.
    pub fn name_str(&self) -> &str {
        let end = self.name.iter().position(|&b| b == 0).unwrap_or(32);
        std::str::from_utf8(&self.name[..end]).unwrap_or("<invalid>")
    }
}

impl Default for TbColumnDescriptor {
    fn default() -> Self {
        TbColumnDescriptor {
            name: [0; 32],
            dtype: 0,
            encoding: 0,
            _pad: [0; 2],
            max_key: 0,
            _reserved: [0; 16],
        }
    }
}

// ---------------------------------------------------------------------------
// TbFileHeader — 4096 bytes, always at file offset 0
// ---------------------------------------------------------------------------

/// Fixed 4096-byte file header.
///
/// Layout (byte offsets within the struct):
/// ```text
/// [0..8)      magic
/// [8..12)     version
/// [12..16)    _pad0
/// [16..24)    n_rows
/// [24..28)    n_columns
/// [28..29)    pipeline_dtype
/// [29..32)    _pad1
/// [32..36)    tile_size
/// [36..40)    n_tiles
/// [40..48)    tile_header_section_offset  (always = 4096)
/// [48..56)    tile_header_section_size    (= n_tiles × n_columns × 40)
/// [56..64)    data_section_offset
/// [64..72)    provenance_offset           (0 = none yet)
/// [72..96)    _reserved_scalars
/// [96..3680)  columns[64]: TbColumnDescriptor
/// [3680..4096) _reserved_tail
/// ```
///
/// Total: 96 + 64×56 + 416 = 4096 bytes.
#[repr(C)]
pub struct TbFileHeader {
    /// Must equal TB_MAGIC = b"TAMBEAR\0".
    pub magic: [u8; 8],
    /// Format version. Currently TB_VERSION = 1.
    pub version: u32,
    pub _pad0: [u8; 4],
    /// Total number of rows in all tiles.
    pub n_rows: u64,
    /// Number of columns.
    pub n_columns: u32,
    /// Pipeline DType: F32=0, F64=1, I32=2, I64=3, U32=4, U64=5.
    pub pipeline_dtype: u8,
    pub _pad1: [u8; 3],
    /// Rows per tile. Default: TB_TILE_SIZE_DEFAULT = 65,536.
    pub tile_size: u32,
    /// Total number of tiles. `ceil(n_rows / tile_size)`.
    pub n_tiles: u32,
    /// Byte offset of tile header section from file start.
    /// Always 4096 (immediately after this header).
    pub tile_header_section_offset: u64,
    /// Byte size of tile header section.
    /// = n_tiles × n_columns × size_of::<TileColumnStats>().
    pub tile_header_section_size: u64,
    /// Byte offset of the column data section from file start.
    pub data_section_offset: u64,
    /// Byte offset of provenance tail from file start. 0 = no provenance written yet.
    pub provenance_offset: u64,
    /// Reserved for future header scalars. Zero on write.
    pub _reserved_scalars: [u8; 24],
    /// Per-column descriptors. Active columns: [0..n_columns). Rest are zero.
    pub columns: [TbColumnDescriptor; 64],
    /// Padding to fill exactly 4096 bytes.
    pub _reserved_tail: [u8; 416],
}

const _: () = assert!(
    std::mem::size_of::<TbFileHeader>() == 4096,
    "TbFileHeader must be exactly 4096 bytes"
);

impl TbFileHeader {
    /// Build a new file header.
    ///
    /// `tile_header_section_offset` is always FILE_HEADER_SIZE (4096).
    /// `data_section_offset` = 4096 + tile_header_section_size.
    pub fn new(
        n_rows: u64,
        n_columns: u32,
        pipeline_dtype: u8,
        tile_size: u32,
        columns: &[TbColumnDescriptor],
    ) -> Self {
        assert!(columns.len() == n_columns as usize);
        assert!(n_columns as usize <= TB_MAX_COLUMNS, "too many columns");

        let tile_size_u64 = tile_size as u64;
        let n_tiles = ((n_rows + tile_size_u64 - 1) / tile_size_u64) as u32;
        let tile_header_section_size =
            n_tiles as u64 * n_columns as u64 * std::mem::size_of::<TileColumnStats>() as u64;
        let tile_header_section_offset = FILE_HEADER_SIZE as u64;
        let data_section_offset = tile_header_section_offset + tile_header_section_size;

        let mut col_arr = [TbColumnDescriptor::default(); 64];
        for (i, c) in columns.iter().enumerate() {
            col_arr[i] = *c;
        }

        TbFileHeader {
            magic: TB_MAGIC,
            version: TB_VERSION,
            _pad0: [0; 4],
            n_rows,
            n_columns,
            pipeline_dtype,
            _pad1: [0; 3],
            tile_size,
            n_tiles,
            tile_header_section_offset,
            tile_header_section_size,
            data_section_offset,
            provenance_offset: 0,
            _reserved_scalars: [0; 24],
            columns: col_arr,
            _reserved_tail: [0; 416],
        }
    }

    /// Byte offset of TileColumnStats for (tile_idx, col_idx) within the tile header section.
    ///
    /// Add `tile_header_section_offset` to get the absolute offset from file start.
    pub fn tile_stats_offset_in_section(&self, tile_idx: u32, col_idx: u32) -> u64 {
        // Layout: for each tile, all columns in order.
        // tile_idx * n_columns * sizeof(TileColumnStats) + col_idx * sizeof(TileColumnStats)
        (tile_idx as u64 * self.n_columns as u64 + col_idx as u64)
            * std::mem::size_of::<TileColumnStats>() as u64
    }

    /// Absolute byte offset of TileColumnStats(tile_idx, col_idx) from file start.
    pub fn tile_stats_offset(&self, tile_idx: u32, col_idx: u32) -> u64 {
        self.tile_header_section_offset + self.tile_stats_offset_in_section(tile_idx, col_idx)
    }

    /// Absolute byte offset of tile data for (col_idx, tile_idx) from file start.
    ///
    /// Column-major layout: all tiles for column 0, then all tiles for column 1, etc.
    /// Within a column, tiles are contiguous. Loading tiles J..K for column C is one read.
    pub fn tile_data_offset(&self, col_idx: u32, tile_idx: u32) -> u64 {
        let dtype_bytes = self.dtype_byte_size() as u64;
        let tile_bytes = self.tile_size as u64 * dtype_bytes;
        self.data_section_offset
            + col_idx as u64 * self.n_tiles as u64 * tile_bytes
            + tile_idx as u64 * tile_bytes
    }

    /// Byte size of one tile for one column (may be smaller for the last tile).
    pub fn tile_byte_size(&self, tile_idx: u32) -> u64 {
        let rows = if tile_idx + 1 == self.n_tiles {
            // Last tile may be partial.
            let full_tiles = (self.n_tiles - 1) as u64;
            self.n_rows - full_tiles * self.tile_size as u64
        } else {
            self.tile_size as u64
        };
        rows * self.dtype_byte_size() as u64
    }

    /// Byte size per element for this file's pipeline dtype.
    pub fn dtype_byte_size(&self) -> usize {
        match self.pipeline_dtype {
            0 | 2 | 4 => 4, // F32, I32, U32
            1 | 3 | 5 => 8, // F64, I64, U64
            _ => panic!("unknown dtype byte {}", self.pipeline_dtype),
        }
    }

    /// Validate the magic bytes.
    pub fn is_valid(&self) -> bool {
        self.magic == TB_MAGIC && self.version == TB_VERSION
    }

    /// Byte slice view of this header for writing to disk.
    ///
    /// # Safety
    /// The repr(C) struct has no interior padding that could contain undefined bytes
    /// PROVIDED all padding fields are explicitly zeroed (which `new()` guarantees).
    pub fn as_bytes(&self) -> &[u8] {
        unsafe {
            std::slice::from_raw_parts(
                self as *const TbFileHeader as *const u8,
                FILE_HEADER_SIZE,
            )
        }
    }

    /// Interpret a 4096-byte buffer as a TbFileHeader.
    ///
    /// # Safety
    /// Buffer must be exactly 4096 bytes and contain a valid TbFileHeader.
    /// Call `is_valid()` after to check magic/version.
    pub unsafe fn from_bytes(buf: &[u8; 4096]) -> &TbFileHeader {
        &*(buf.as_ptr() as *const TbFileHeader)
    }
}
