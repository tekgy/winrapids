//! MKTF writer — crash-safe, atomic, GPU-native.
//!
//! Write protocol (identical to Python writer):
//!   1. Build complete file in memory (header + directory + data)
//!   2. Compute SHA-256 data checksum
//!   3. Patch header: data_checksum, is_complete, write_duration
//!   4. Single write to .mktf.tmp
//!   5. Optional fsync for crash safety
//!   6. Atomic rename .mktf.tmp -> .mktf
//!
//! Performance strategy: one allocation, one syscall write. Seeks only for
//! the write_duration_ms fixup (measured after the bulk write).

use std::fs;
use std::io::{self, Seek, SeekFrom, Write};
use std::path::Path;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use sha2::{Sha256, Digest};

use crate::format::*;
use crate::filter;

/// A column to write: name, dtype code, and raw bytes (little-endian).
pub struct ColumnData {
    pub name: String,
    pub dtype_code: u8,
    pub n_elements: u64,
    pub data: Vec<u8>,
    /// Pre-computed statistics (optional — computed from data if None).
    pub null_count: u64,
    pub min_value: f64,
    pub max_value: f64,
    pub mean_value: f64,
    /// Compression settings (default: none).
    pub compression_algo: u8,
    pub pre_filter: u8,
}

impl ColumnData {
    /// Create from raw bytes with no compression.
    pub fn new(name: String, dtype_code: u8, data: Vec<u8>) -> Self {
        let elem_size = dtype_size(dtype_code);
        let n_elements = if elem_size > 0 { data.len() / elem_size } else { 0 };
        Self {
            name,
            dtype_code,
            n_elements: n_elements as u64,
            data,
            null_count: 0,
            min_value: f64::NAN,
            max_value: f64::NAN,
            mean_value: f64::NAN,
            compression_algo: COMPRESS_NONE,
            pre_filter: FILTER_NONE,
        }
    }

    /// Create with shuffle + LZ4 compression.
    pub fn new_compressed(name: String, dtype_code: u8, data: Vec<u8>) -> Self {
        let mut col = Self::new(name, dtype_code, data);
        col.compression_algo = COMPRESS_LZ4;
        col.pre_filter = FILTER_SHUFFLE;
        col
    }
}

/// Write options.
pub struct WriteOptions {
    pub leaf_id: String,
    pub ticker: String,
    pub day: String,
    pub ti: u8,
    pub to: u8,
    pub leaf_version: String,
    pub alignment: u16,
    pub safe: bool,
    pub compute_duration_ms: u32,
    pub metadata: Option<Vec<u8>>,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            leaf_id: String::new(),
            ticker: String::new(),
            day: String::new(),
            ti: 0,
            to: 0,
            leaf_version: "1.0.0".into(),
            alignment: ALIGNMENT,
            safe: true,
            compute_duration_ms: 0,
            metadata: None,
        }
    }
}

/// Write columns to MKTF file with crash-safe protocol.
///
/// Builds entire file in memory, writes in one syscall, then patches
/// write_duration_ms with a single seek. Returns the finalized MktfHeader.
pub fn write_mktf(
    path: &Path,
    columns: &[ColumnData],
    opts: &WriteOptions,
) -> io::Result<MktfHeader> {
    let t0 = Instant::now();

    if columns.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "zero columns"));
    }

    let alignment = opts.alignment as u64;
    let n_col_count = columns.len();

    // ── Layout computation ───────────────────────────────────────

    let dir_offset = BLOCK_SIZE as u64;
    let dir_size = (n_col_count * ENTRY_SIZE) as u64;
    let dir_end = dir_offset + dir_size;

    let (meta_offset, meta_size) = if let Some(ref meta) = opts.metadata {
        (align(dir_end, alignment), meta.len() as u64)
    } else {
        (0u64, 0u64)
    };

    let data_region_start = if opts.metadata.is_some() {
        align(meta_offset + meta_size, alignment)
    } else {
        align(dir_end, alignment)
    };

    // ── Prepare payloads + entries ────────────────────────────────

    let mut col_entries: Vec<ByteEntry> = Vec::with_capacity(n_col_count);
    let mut col_payloads: Vec<Vec<u8>> = Vec::with_capacity(n_col_count);
    let mut current_offset = data_region_start;
    let mut total_data_bytes: u64 = 0;

    for col in columns {
        let typesize = dtype_size(col.dtype_code) as u8;
        let uncompressed_size = col.data.len() as u64;

        let (payload, compressed_size) = if col.compression_algo != COMPRESS_NONE {
            let filtered = match col.pre_filter {
                FILTER_SHUFFLE => filter::shuffle(&col.data, typesize as usize),
                _ => col.data.clone(),
            };
            let compressed = match col.compression_algo {
                COMPRESS_LZ4 => lz4_flex::compress_prepend_size(&filtered),
                _ => filtered,
            };
            let csz = compressed.len() as u64;
            (compressed, csz)
        } else {
            // Borrow path: no allocation for uncompressed
            (Vec::new(), 0u64)
        };

        let on_disk_size = if compressed_size > 0 {
            payload.len() as u64
        } else {
            uncompressed_size
        };

        col_entries.push(ByteEntry {
            name: col.name.clone(),
            dtype_code: col.dtype_code,
            n_elements: col.n_elements,
            data_offset: current_offset,
            data_nbytes: uncompressed_size,
            null_count: col.null_count,
            min_value: col.min_value,
            max_value: col.max_value,
            mean_value: col.mean_value,
            scale_factor: 1.0,
            sentinel_value: f64::NAN,
            compression_algo: col.compression_algo,
            pre_filter: col.pre_filter,
            typesize,
            filter_param: 0,
            compressed_size,
        });
        col_payloads.push(payload);

        current_offset = align(current_offset + on_disk_size, alignment);
        total_data_bytes += uncompressed_size;
    }

    let file_end = current_offset + 2;
    let header_blocks = (data_region_start / BLOCK_SIZE as u64) as u16;

    // ── Build header ─────────────────────────────────────────────

    let schema_cols: Vec<(String, u8)> = columns.iter()
        .map(|c| (c.name.clone(), c.dtype_code))
        .collect();
    let n_elements = columns[0].n_elements;

    let total_nulls: u64 = col_entries.iter().map(|e| e.null_count).sum();
    let total_elems = n_elements * n_col_count as u64;

    let mut header = MktfHeader {
        alignment: opts.alignment,
        leaf_id: opts.leaf_id.clone(),
        ticker: opts.ticker.clone(),
        day: opts.day.clone(),
        ti: opts.ti,
        to: opts.to,
        leaf_version: opts.leaf_version.clone(),
        leaf_id_hash: compute_leaf_id_hash(&opts.leaf_id),
        schema_fingerprint: compute_schema_fingerprint(&schema_cols),
        n_rows: n_elements,
        n_cols: n_col_count as u16,
        bytes_data: total_data_bytes,
        bytes_file: file_end,
        dir_offset,
        dir_entries: n_col_count as u64,
        meta_offset,
        meta_size,
        data_start: data_region_start,
        header_blocks,
        write_timestamp_ns: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as i64,
        compute_duration_ms: opts.compute_duration_ms,
        flags: FLAG_HAS_QUALITY
            | if opts.metadata.is_some() { FLAG_HAS_METADATA } else { 0 },
        total_nulls,
        null_ppm: if total_elems > 0 {
            (total_nulls as f64 / total_elems as f64 * 1_000_000.0) as u32
        } else { 0 },
        is_complete: false,
        is_dirty: false,
        columns: col_entries.clone(),
        ..MktfHeader::default()
    };

    // ── Build file buffer in one allocation ───────────────────────

    let file_size = file_end as usize;
    let mut buf = vec![0u8; file_size];

    // Block 0
    let block0 = pack_block0(&header);
    buf[..BLOCK_SIZE].copy_from_slice(&block0);

    // Column directory
    for (i, entry) in col_entries.iter().enumerate() {
        let off = BLOCK_SIZE + i * ENTRY_SIZE;
        let packed = pack_byte_entry(entry);
        buf[off..off + ENTRY_SIZE].copy_from_slice(&packed);
    }

    // Metadata
    if let Some(ref meta) = opts.metadata {
        let off = meta_offset as usize;
        buf[off..off + meta.len()].copy_from_slice(meta);
    }

    // Column data + hash
    let mut data_hasher = Sha256::new();
    for (i, (col, entry)) in columns.iter().zip(col_entries.iter()).enumerate() {
        let off = entry.data_offset as usize;
        let payload = &col_payloads[i];

        if entry.compressed_size > 0 {
            // Compressed: write payload
            buf[off..off + payload.len()].copy_from_slice(payload);
            data_hasher.update(payload);
        } else {
            // Uncompressed: write directly from column data (no clone)
            buf[off..off + col.data.len()].copy_from_slice(&col.data);
            data_hasher.update(&col.data);
        }
    }

    // Data checksum
    let hash = data_hasher.finalize();
    let data_checksum = u64::from_le_bytes(hash[..8].try_into().unwrap());
    buf[DATA_CHECKSUM_OFFSET..DATA_CHECKSUM_OFFSET + 8]
        .copy_from_slice(&data_checksum.to_le_bytes());

    // Flip is_complete at header[4094]
    buf[4094] = 1;
    buf[4095] = 0;

    // Flip is_complete at EOF[-2]
    buf[file_size - 2] = 1;
    buf[file_size - 1] = 0;

    // write_duration_ms = 0 for now (patched after write)

    // ── Single write to disk ─────────────────────────────────────

    let tmp_path = path.with_extension("mktf.tmp");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    {
        let mut file = fs::File::create(&tmp_path)?;
        file.write_all(&buf)?;

        // Patch write_duration_ms (single seek + 4 bytes)
        let write_duration_ms = t0.elapsed().as_millis() as u32;
        file.seek(SeekFrom::Start(OFF_PROVENANCE as u64 + 8))?;
        file.write_all(&write_duration_ms.to_le_bytes())?;

        if opts.safe {
            file.sync_all()?;
        }

        header.is_complete = true;
        header.write_duration_ms = write_duration_ms;
        header.data_checksum = data_checksum;
    }

    // Atomic rename
    if path.exists() {
        fs::remove_file(path)?;
    }
    fs::rename(&tmp_path, path)?;

    Ok(header)
}

/// Write columns directly to final path — no tmp file, no rename.
///
/// For derived/recomputable files (K02+, KO05) where crash safety
/// is handled by the daemon (incomplete files are re-seeded on restart).
/// Saves ~400us on NTFS by eliminating file create + rename overhead.
pub fn write_mktf_direct(
    path: &Path,
    columns: &[ColumnData],
    opts: &WriteOptions,
) -> io::Result<MktfHeader> {
    let t0 = Instant::now();

    if columns.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, "zero columns"));
    }

    let alignment = opts.alignment as u64;
    let n_col_count = columns.len();

    // ── Layout (same as write_mktf) ──────────────────────────────

    let dir_offset = BLOCK_SIZE as u64;
    let dir_size = (n_col_count * ENTRY_SIZE) as u64;
    let dir_end = dir_offset + dir_size;

    let (meta_offset, meta_size) = if let Some(ref meta) = opts.metadata {
        (align(dir_end, alignment), meta.len() as u64)
    } else {
        (0u64, 0u64)
    };

    let data_region_start = if opts.metadata.is_some() {
        align(meta_offset + meta_size, alignment)
    } else {
        align(dir_end, alignment)
    };

    let mut col_entries: Vec<ByteEntry> = Vec::with_capacity(n_col_count);
    let mut col_payloads: Vec<Vec<u8>> = Vec::with_capacity(n_col_count);
    let mut current_offset = data_region_start;
    let mut total_data_bytes: u64 = 0;

    for col in columns {
        let typesize = dtype_size(col.dtype_code) as u8;
        let uncompressed_size = col.data.len() as u64;

        let (payload, compressed_size) = if col.compression_algo != COMPRESS_NONE {
            let filtered = match col.pre_filter {
                FILTER_SHUFFLE => filter::shuffle(&col.data, typesize as usize),
                _ => col.data.clone(),
            };
            let compressed = match col.compression_algo {
                COMPRESS_LZ4 => lz4_flex::compress_prepend_size(&filtered),
                _ => filtered,
            };
            let csz = compressed.len() as u64;
            (compressed, csz)
        } else {
            (Vec::new(), 0u64)
        };

        let on_disk_size = if compressed_size > 0 {
            payload.len() as u64
        } else {
            uncompressed_size
        };

        col_entries.push(ByteEntry {
            name: col.name.clone(),
            dtype_code: col.dtype_code,
            n_elements: col.n_elements,
            data_offset: current_offset,
            data_nbytes: uncompressed_size,
            null_count: col.null_count,
            min_value: col.min_value,
            max_value: col.max_value,
            mean_value: col.mean_value,
            scale_factor: 1.0,
            sentinel_value: f64::NAN,
            compression_algo: col.compression_algo,
            pre_filter: col.pre_filter,
            typesize,
            filter_param: 0,
            compressed_size,
        });
        col_payloads.push(payload);

        current_offset = align(current_offset + on_disk_size, alignment);
        total_data_bytes += uncompressed_size;
    }

    let file_end = current_offset + 2;
    let header_blocks = (data_region_start / BLOCK_SIZE as u64) as u16;

    let schema_cols: Vec<(String, u8)> = columns.iter()
        .map(|c| (c.name.clone(), c.dtype_code))
        .collect();
    let n_elements = columns[0].n_elements;
    let total_nulls: u64 = col_entries.iter().map(|e| e.null_count).sum();
    let total_elems = n_elements * n_col_count as u64;

    let mut header = MktfHeader {
        alignment: opts.alignment,
        leaf_id: opts.leaf_id.clone(),
        ticker: opts.ticker.clone(),
        day: opts.day.clone(),
        ti: opts.ti,
        to: opts.to,
        leaf_version: opts.leaf_version.clone(),
        leaf_id_hash: compute_leaf_id_hash(&opts.leaf_id),
        schema_fingerprint: compute_schema_fingerprint(&schema_cols),
        n_rows: n_elements,
        n_cols: n_col_count as u16,
        bytes_data: total_data_bytes,
        bytes_file: file_end,
        dir_offset,
        dir_entries: n_col_count as u64,
        meta_offset,
        meta_size,
        data_start: data_region_start,
        header_blocks,
        write_timestamp_ns: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as i64,
        compute_duration_ms: opts.compute_duration_ms,
        flags: FLAG_HAS_QUALITY
            | if opts.metadata.is_some() { FLAG_HAS_METADATA } else { 0 },
        total_nulls,
        null_ppm: if total_elems > 0 {
            (total_nulls as f64 / total_elems as f64 * 1_000_000.0) as u32
        } else { 0 },
        is_complete: false,
        is_dirty: false,
        columns: col_entries.clone(),
        ..MktfHeader::default()
    };

    // ── Build buffer ─────────────────────────────────────────────

    let file_size = file_end as usize;
    let mut buf = vec![0u8; file_size];

    let block0 = pack_block0(&header);
    buf[..BLOCK_SIZE].copy_from_slice(&block0);

    for (i, entry) in col_entries.iter().enumerate() {
        let off = BLOCK_SIZE + i * ENTRY_SIZE;
        let packed = pack_byte_entry(entry);
        buf[off..off + ENTRY_SIZE].copy_from_slice(&packed);
    }

    if let Some(ref meta) = opts.metadata {
        let off = meta_offset as usize;
        buf[off..off + meta.len()].copy_from_slice(meta);
    }

    let mut data_hasher = Sha256::new();
    for (i, (col, entry)) in columns.iter().zip(col_entries.iter()).enumerate() {
        let off = entry.data_offset as usize;
        let payload = &col_payloads[i];

        if entry.compressed_size > 0 {
            buf[off..off + payload.len()].copy_from_slice(payload);
            data_hasher.update(payload);
        } else {
            buf[off..off + col.data.len()].copy_from_slice(&col.data);
            data_hasher.update(&col.data);
        }
    }

    let hash = data_hasher.finalize();
    let data_checksum = u64::from_le_bytes(hash[..8].try_into().unwrap());
    buf[DATA_CHECKSUM_OFFSET..DATA_CHECKSUM_OFFSET + 8]
        .copy_from_slice(&data_checksum.to_le_bytes());

    buf[4094] = 1; // is_complete
    buf[4095] = 0; // is_dirty
    buf[file_size - 2] = 1;
    buf[file_size - 1] = 0;

    // ── Single write, no rename ──────────────────────────────────

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = fs::File::create(path)?;
    file.write_all(&buf)?;

    let write_duration_ms = t0.elapsed().as_millis() as u32;
    file.seek(SeekFrom::Start(OFF_PROVENANCE as u64 + 8))?;
    file.write_all(&write_duration_ms.to_le_bytes())?;

    header.is_complete = true;
    header.write_duration_ms = write_duration_ms;
    header.data_checksum = data_checksum;

    Ok(header)
}

/// Flip the is_dirty byte at both positions.
pub fn flip_dirty(path: &Path, dirty: bool) -> io::Result<()> {
    let val = if dirty { 1u8 } else { 0u8 };
    let mut f = fs::OpenOptions::new().write(true).open(path)?;
    f.seek(SeekFrom::Start(4095))?;
    f.write_all(&[val])?;
    f.seek(SeekFrom::End(-1))?;
    f.write_all(&[val])?;
    Ok(())
}

/// Flip the is_complete byte at both positions.
pub fn flip_complete(path: &Path, complete: bool) -> io::Result<()> {
    let val = if complete { 1u8 } else { 0u8 };
    let mut f = fs::OpenOptions::new().write(true).open(path)?;
    f.seek(SeekFrom::Start(4094))?;
    f.write_all(&[val])?;
    f.seek(SeekFrom::End(-2))?;
    f.write_all(&[val])?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn write_and_read_back_header() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.mktf");

        let cols = vec![
            ColumnData::new("price".into(), DTYPE_FLOAT32, vec![0u8; 400]),
            ColumnData::new("volume".into(), DTYPE_FLOAT32, vec![0u8; 400]),
        ];

        let opts = WriteOptions {
            leaf_id: "K01P01.TI00TO00".into(),
            ticker: "AAPL".into(),
            day: "2026-03-28".into(),
            alignment: 64,
            safe: false,
            ..Default::default()
        };

        let header = write_mktf(&path, &cols, &opts).unwrap();
        assert!(header.is_complete);
        assert_eq!(header.alignment, 64);
        assert_eq!(header.n_cols, 2);
        assert!(header.data_checksum != 0);

        // Read back Block 0
        let data = fs::read(&path).unwrap();
        let h2 = unpack_block0(&data[..BLOCK_SIZE]).unwrap();
        assert!(h2.is_complete);
        assert_eq!(h2.leaf_id, "K01P01.TI00TO00");
        assert_eq!(h2.ticker, "AAPL");
        assert_eq!(h2.alignment, 64);
    }

    #[test]
    fn crash_safety_tmp_file() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("test.mktf");
        let tmp_path = path.with_extension("mktf.tmp");

        let cols = vec![
            ColumnData::new("x".into(), DTYPE_FLOAT32, vec![0u8; 40]),
        ];
        let opts = WriteOptions {
            safe: false,
            ..Default::default()
        };

        write_mktf(&path, &cols, &opts).unwrap();

        assert!(path.exists());
        assert!(!tmp_path.exists());
    }
}
