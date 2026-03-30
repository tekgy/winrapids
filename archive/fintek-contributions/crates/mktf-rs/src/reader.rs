//! MKTF reader — header-only, selective, full, and status fast path.
//!
//! Daemon fast path reads 1-2 bytes at EOF. Zero parsing.
//! Header-only reads Block 0 (4096 bytes) + directory.
//! Selective reads specific columns by seeking.
//! Full reads load everything.

use std::fs;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use crate::format::*;
use crate::filter;

// ── Status Fast Path (1-2 bytes, zero parsing) ──────────────────

/// Read (is_complete, is_dirty) from EOF. Zero header parsing.
pub fn read_status(path: &Path) -> io::Result<(bool, bool)> {
    let mut f = fs::File::open(path)?;
    f.seek(SeekFrom::End(-2))?;
    let mut buf = [0u8; 2];
    f.read_exact(&mut buf)?;
    Ok((buf[0] != 0, buf[1] != 0))
}

/// Check is_dirty from the last byte of the file.
pub fn is_dirty(path: &Path) -> io::Result<bool> {
    let mut f = fs::File::open(path)?;
    f.seek(SeekFrom::End(-1))?;
    let mut buf = [0u8; 1];
    f.read_exact(&mut buf)?;
    Ok(buf[0] != 0)
}

/// Check is_complete from EOF-2.
pub fn is_complete(path: &Path) -> io::Result<bool> {
    let mut f = fs::File::open(path)?;
    f.seek(SeekFrom::End(-2))?;
    let mut buf = [0u8; 1];
    f.read_exact(&mut buf)?;
    Ok(buf[0] != 0)
}

// ── Header-Only Read ─────────────────────────────────────────────

/// Read Block 0 + byte range directory. No data bytes touched.
pub fn read_header(path: &Path) -> io::Result<MktfHeader> {
    let mut f = fs::File::open(path)?;

    // Block 0
    let mut block0 = [0u8; BLOCK_SIZE];
    f.read_exact(&mut block0)?;

    let mut header = unpack_block0(&block0)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    // Byte range directory
    if header.dir_entries > 0 && header.dir_offset > 0 {
        let dir_size = header.dir_entries as usize * ENTRY_SIZE;
        f.seek(SeekFrom::Start(header.dir_offset))?;
        let mut dir_buf = vec![0u8; dir_size];
        f.read_exact(&mut dir_buf)?;

        for i in 0..header.dir_entries as usize {
            let entry = unpack_byte_entry(&dir_buf, i * ENTRY_SIZE);
            header.columns.push(entry);
        }
    }

    Ok(header)
}

// ── Full Read ────────────────────────────────────────────────────

/// Read header + all column data. Returns (header, column_name -> raw_bytes).
pub fn read_columns(path: &Path) -> io::Result<(MktfHeader, Vec<(String, Vec<u8>)>)> {
    let header = read_header(path)?;
    let mut f = fs::File::open(path)?;
    let mut columns = Vec::with_capacity(header.columns.len());

    for entry in &header.columns {
        f.seek(SeekFrom::Start(entry.data_offset))?;

        let read_size = if entry.compressed_size > 0 {
            entry.compressed_size as usize
        } else {
            entry.data_nbytes as usize
        };

        let mut raw = vec![0u8; read_size];
        f.read_exact(&mut raw)?;

        // Decompress if needed
        let data = if entry.compression_algo != COMPRESS_NONE && entry.compressed_size > 0 {
            let decompressed = match entry.compression_algo {
                COMPRESS_LZ4 => lz4_flex::decompress_size_prepended(&raw)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?,
                _ => raw,
            };
            // Reverse pre-filter
            match entry.pre_filter {
                FILTER_SHUFFLE => filter::unshuffle(&decompressed, entry.typesize as usize),
                _ => decompressed,
            }
        } else {
            raw
        };

        columns.push((entry.name.clone(), data));
    }

    Ok((header, columns))
}

/// Read header + specific columns. Minimal I/O.
pub fn read_selective(
    path: &Path,
    column_names: &[&str],
) -> io::Result<(MktfHeader, Vec<(String, Vec<u8>)>)> {
    let header = read_header(path)?;
    let mut f = fs::File::open(path)?;
    let mut columns = Vec::with_capacity(column_names.len());

    for &name in column_names {
        let entry = header.columns.iter()
            .find(|e| e.name == name)
            .ok_or_else(|| io::Error::new(
                io::ErrorKind::NotFound,
                format!("Column not found: {name}"),
            ))?;

        f.seek(SeekFrom::Start(entry.data_offset))?;

        let read_size = if entry.compressed_size > 0 {
            entry.compressed_size as usize
        } else {
            entry.data_nbytes as usize
        };

        let mut raw = vec![0u8; read_size];
        f.read_exact(&mut raw)?;

        // Decompress if needed
        let data = if entry.compression_algo != COMPRESS_NONE && entry.compressed_size > 0 {
            let decompressed = match entry.compression_algo {
                COMPRESS_LZ4 => lz4_flex::decompress_size_prepended(&raw)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.to_string()))?,
                _ => raw,
            };
            match entry.pre_filter {
                FILTER_SHUFFLE => filter::unshuffle(&decompressed, entry.typesize as usize),
                _ => decompressed,
            }
        } else {
            raw
        };

        columns.push((name.to_string(), data));
    }

    Ok((header, columns))
}

// ── Data Integrity Verification ──────────────────────────────────

/// Verify data_checksum matches actual data content.
pub fn verify_checksum(path: &Path) -> io::Result<bool> {
    use sha2::{Sha256, Digest};

    let header = read_header(path)?;
    if header.data_checksum == 0 {
        return Ok(true);
    }

    let mut f = fs::File::open(path)?;
    let mut hasher = Sha256::new();

    for entry in &header.columns {
        f.seek(SeekFrom::Start(entry.data_offset))?;

        let read_size = if entry.compressed_size > 0 {
            entry.compressed_size as usize
        } else {
            entry.data_nbytes as usize
        };

        let mut raw = vec![0u8; read_size];
        f.read_exact(&mut raw)?;
        hasher.update(&raw);
    }

    let hash = hasher.finalize();
    let actual = u64::from_le_bytes(hash[..8].try_into().unwrap());
    Ok(actual == header.data_checksum)
}

// ── Batch Status Scan ────────────────────────────────────────────

/// Scan files for is_dirty=1. Returns list of dirty paths.
/// Daemon hot path: one seek + one byte read per file.
pub fn scan_dirty(paths: &[&Path]) -> Vec<PathBuf> {
    paths.iter()
        .filter_map(|p| {
            is_dirty(p).ok().and_then(|d| if d { Some(p.to_path_buf()) } else { None })
        })
        .collect()
}

/// Scan files for is_complete=0. Returns list of incomplete paths.
pub fn scan_incomplete(paths: &[&Path]) -> Vec<PathBuf> {
    paths.iter()
        .filter_map(|p| {
            is_complete(p).ok().and_then(|c| if !c { Some(p.to_path_buf()) } else { None })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::{write_mktf, ColumnData, WriteOptions};
    use tempfile::TempDir;

    #[test]
    fn full_roundtrip() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("roundtrip.mktf");

        // Create test data
        let price_data: Vec<f32> = (0..100).map(|i| 100.0 + i as f32 * 0.5).collect();
        let volume_data: Vec<f32> = (0..100).map(|i| 1000.0 + i as f32).collect();

        let cols = vec![
            ColumnData::new("price".into(), DTYPE_FLOAT32, bytemuck_f32(&price_data)),
            ColumnData::new("volume".into(), DTYPE_FLOAT32, bytemuck_f32(&volume_data)),
        ];

        let opts = WriteOptions {
            leaf_id: "K01P01.TI00TO00".into(),
            ticker: "AAPL".into(),
            day: "2026-03-28".into(),
            alignment: 64,
            safe: false,
            ..Default::default()
        };

        write_mktf(&path, &cols, &opts).unwrap();

        // Read back
        let (header, read_cols) = read_columns(&path).unwrap();
        assert_eq!(header.leaf_id, "K01P01.TI00TO00");
        assert_eq!(header.ticker, "AAPL");
        assert_eq!(read_cols.len(), 2);
        assert_eq!(read_cols[0].0, "price");
        assert_eq!(read_cols[1].0, "volume");

        // Verify data is bit-exact
        let read_price: &[f32] = bytecast_f32(&read_cols[0].1);
        let read_volume: &[f32] = bytecast_f32(&read_cols[1].1);
        assert_eq!(read_price, &price_data[..]);
        assert_eq!(read_volume, &volume_data[..]);
    }

    #[test]
    fn status_fast_path() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("status.mktf");

        let cols = vec![
            ColumnData::new("x".into(), DTYPE_FLOAT32, vec![0u8; 40]),
        ];
        let opts = WriteOptions { safe: false, ..Default::default() };
        write_mktf(&path, &cols, &opts).unwrap();

        let (complete, dirty) = read_status(&path).unwrap();
        assert!(complete);
        assert!(!dirty);
    }

    #[test]
    fn checksum_verification() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("checksum.mktf");

        let cols = vec![
            ColumnData::new("x".into(), DTYPE_FLOAT32, vec![1, 2, 3, 4, 5, 6, 7, 8]),
        ];
        let opts = WriteOptions { safe: false, ..Default::default() };
        write_mktf(&path, &cols, &opts).unwrap();

        assert!(verify_checksum(&path).unwrap());
    }

    fn bytemuck_f32(data: &[f32]) -> Vec<u8> {
        data.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn bytecast_f32(data: &[u8]) -> &[f32] {
        unsafe {
            std::slice::from_raw_parts(
                data.as_ptr() as *const f32,
                data.len() / 4,
            )
        }
    }
}
