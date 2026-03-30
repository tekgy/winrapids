//! .tb file I/O — write and read tiled DataFrames.
//!
//! Write path: take column data → compute tile stats → write header + stats + data.
//! Read path: mmap (or read) header → load tile stats → lazy/selective column reads.
//!
//! "Tam doesn't read. Tam knows the summary."

use std::fs;
use std::io::{self, Write, Read, Seek, SeekFrom};
use std::path::Path;

use crate::format::*;

// ---------------------------------------------------------------------------
// TbWriter — write a .tb file
// ---------------------------------------------------------------------------

/// Column data for writing. All columns must have the same number of rows.
pub struct TbColumnWrite {
    pub name: String,
    pub data: Vec<f64>,
    pub dtype: u8,
    pub encoding: u8,
    pub max_key: u32,
}

impl TbColumnWrite {
    /// Build a numeric (f64) column with Raw encoding.
    pub fn f64_column(name: &str, data: Vec<f64>) -> Self {
        TbColumnWrite {
            name: name.to_string(),
            data,
            dtype: 1, // F64
            encoding: 0, // Raw
            max_key: 0,
        }
    }

    /// Build a key (i32) column — stored as f64 for uniformity, max_key for accumulators.
    pub fn key_column(name: &str, keys: &[i32]) -> Self {
        let max_key = keys.iter().cloned().max().unwrap_or(0) as u32;
        TbColumnWrite {
            name: name.to_string(),
            data: keys.iter().map(|&k| k as f64).collect(),
            dtype: 2, // I32
            encoding: 0,
            max_key,
        }
    }
}

/// Write columns to a .tb file.
///
/// 1. Build TbFileHeader (4096 bytes)
/// 2. Compute TileColumnStats per (tile, column) from the data
/// 3. Write header, tile stats, and column data in column-major tiled order
pub fn write_tb(
    path: &Path,
    columns: &[TbColumnWrite],
    tile_size: u32,
) -> io::Result<()> {
    assert!(!columns.is_empty(), "at least one column required");
    let n_rows = columns[0].data.len();
    for c in columns {
        assert_eq!(c.data.len(), n_rows, "all columns must have same row count");
    }
    let n_columns = columns.len() as u32;

    // Build column descriptors
    let descriptors: Vec<TbColumnDescriptor> = columns
        .iter()
        .map(|c| TbColumnDescriptor::new(&c.name, c.dtype, c.encoding, c.max_key))
        .collect();

    // Build file header
    let header = TbFileHeader::new(n_rows as u64, n_columns, 1, tile_size, &descriptors);
    let n_tiles = header.n_tiles as usize;
    let tile_sz = tile_size as usize;

    // Compute tile stats
    let mut tile_stats: Vec<TileColumnStats> = Vec::with_capacity(n_tiles * columns.len());
    for t in 0..n_tiles {
        let row_start = t * tile_sz;
        let row_end = (row_start + tile_sz).min(n_rows);
        for col in columns {
            let slice = &col.data[row_start..row_end];
            tile_stats.push(TileColumnStats::from_slice(slice));
        }
    }

    // Write file
    let mut f = fs::File::create(path)?;

    // 1. File header (4096 bytes)
    f.write_all(header.as_bytes())?;

    // 2. Tile header section: n_tiles × n_columns × TileColumnStats (40 bytes each)
    for stat in &tile_stats {
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                stat as *const TileColumnStats as *const u8,
                std::mem::size_of::<TileColumnStats>(),
            )
        };
        f.write_all(bytes)?;
    }

    // 3. Column data: column-major, tiled
    // For each column, write all tiles in order.
    let dtype_bytes = header.dtype_byte_size();
    for col in columns {
        for t in 0..n_tiles {
            let row_start = t * tile_sz;
            let row_end = (row_start + tile_sz).min(n_rows);
            let slice = &col.data[row_start..row_end];
            // Write f64 values as raw bytes
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    slice.as_ptr() as *const u8,
                    slice.len() * dtype_bytes,
                )
            };
            f.write_all(bytes)?;
            // Pad last tile to full tile_size if needed (so offsets stay aligned)
            if row_end - row_start < tile_sz {
                let pad = (tile_sz - (row_end - row_start)) * dtype_bytes;
                let zeros = vec![0u8; pad];
                f.write_all(&zeros)?;
            }
        }
    }

    f.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// TbReader — read a .tb file
// ---------------------------------------------------------------------------

/// Parsed .tb file: header + tile stats (loaded eagerly), data (loaded lazily).
pub struct TbFile {
    pub header: Box<TbFileHeader>,
    pub tile_stats: Vec<TileColumnStats>,
    path: std::path::PathBuf,
}

impl TbFile {
    /// Open a .tb file. Reads header (4096 bytes) + tile stats section.
    /// Column data is NOT read — call `read_column()` for selective I/O.
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut f = fs::File::open(path)?;

        // Read 4096-byte header
        let mut buf = [0u8; 4096];
        f.read_exact(&mut buf)?;
        let header = unsafe { TbFileHeader::from_bytes(&buf) };
        if !header.is_valid() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid .tb magic or version"));
        }

        // Copy header to owned Box (the unsafe ref borrows buf which will die)
        let header_owned: Box<TbFileHeader> = unsafe {
            let mut boxed = Box::new(std::mem::zeroed::<TbFileHeader>());
            std::ptr::copy_nonoverlapping(
                header as *const TbFileHeader,
                &mut *boxed as *mut TbFileHeader,
                1,
            );
            boxed
        };

        // Read tile stats section
        let n_stats = header_owned.n_tiles as usize * header_owned.n_columns as usize;
        let stats_bytes = n_stats * std::mem::size_of::<TileColumnStats>();
        let mut stats_buf = vec![0u8; stats_bytes];
        f.seek(SeekFrom::Start(header_owned.tile_header_section_offset))?;
        f.read_exact(&mut stats_buf)?;

        let tile_stats: Vec<TileColumnStats> = (0..n_stats)
            .map(|i| {
                let offset = i * std::mem::size_of::<TileColumnStats>();
                unsafe {
                    std::ptr::read(stats_buf[offset..].as_ptr() as *const TileColumnStats)
                }
            })
            .collect();

        Ok(TbFile {
            header: header_owned,
            tile_stats,
            path: path.to_path_buf(),
        })
    }

    /// Number of rows.
    pub fn n_rows(&self) -> u64 {
        self.header.n_rows
    }

    /// Number of columns.
    pub fn n_columns(&self) -> u32 {
        self.header.n_columns
    }

    /// Column names.
    pub fn column_names(&self) -> Vec<String> {
        (0..self.header.n_columns as usize)
            .map(|i| self.header.columns[i].name_str().to_string())
            .collect()
    }

    /// Find a column index by name.
    pub fn column_index(&self, name: &str) -> Option<usize> {
        (0..self.header.n_columns as usize)
            .find(|&i| self.header.columns[i].name_str() == name)
    }

    /// Get tile stats for one column (all tiles). Slice of the pre-loaded stats.
    pub fn column_tile_stats(&self, col_idx: usize) -> Vec<&TileColumnStats> {
        let n_cols = self.header.n_columns as usize;
        let n_tiles = self.header.n_tiles as usize;
        (0..n_tiles)
            .map(|t| &self.tile_stats[t * n_cols + col_idx])
            .collect()
    }

    /// Read all data for one column. Returns f64 values for all rows.
    pub fn read_column(&self, col_idx: usize) -> io::Result<Vec<f64>> {
        let mut f = fs::File::open(&self.path)?;
        let n_tiles = self.header.n_tiles as usize;
        let tile_sz = self.header.tile_size as usize;
        let dtype_bytes = self.header.dtype_byte_size();
        let n_rows = self.header.n_rows as usize;

        let mut all_data = Vec::with_capacity(n_rows);

        for t in 0..n_tiles {
            let offset = self.header.tile_data_offset(col_idx as u32, t as u32);
            f.seek(SeekFrom::Start(offset))?;

            let rows_in_tile = if t + 1 == n_tiles {
                n_rows - t * tile_sz
            } else {
                tile_sz
            };
            let bytes_to_read = rows_in_tile * dtype_bytes;
            let mut buf = vec![0u8; bytes_to_read];
            f.read_exact(&mut buf)?;

            // Interpret as f64
            let values: &[f64] = unsafe {
                std::slice::from_raw_parts(buf.as_ptr() as *const f64, rows_in_tile)
            };
            all_data.extend_from_slice(values);
        }

        Ok(all_data)
    }

    /// Read a column by name.
    pub fn read_column_by_name(&self, name: &str) -> io::Result<Vec<f64>> {
        let idx = self.column_index(name).ok_or_else(|| {
            io::Error::new(io::ErrorKind::NotFound, format!("column '{}' not found", name))
        })?;
        self.read_column(idx)
    }

    /// Header-only global stats for a column (no data read).
    pub fn global_stats(&self, col_idx: usize) -> (f64, f64, f64, u64) {
        let stats: Vec<TileColumnStats> = {
            let n_cols = self.header.n_columns as usize;
            let n_tiles = self.header.n_tiles as usize;
            (0..n_tiles)
                .map(|t| self.tile_stats[t * n_cols + col_idx])
                .collect()
        };
        (
            global_min(&stats),
            global_max(&stats),
            global_mean(&stats),
            global_count(&stats),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_simple() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_roundtrip.tb");

        let keys = vec![0i32, 0, 1, 1, 2];
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        write_tb(
            &path,
            &[
                TbColumnWrite::key_column("group", &keys),
                TbColumnWrite::f64_column("value", values.clone()),
            ],
            65536, // one tile for 5 rows
        )
        .unwrap();

        let tb = TbFile::open(&path).unwrap();
        assert_eq!(tb.n_rows(), 5);
        assert_eq!(tb.n_columns(), 2);
        assert_eq!(tb.column_names(), vec!["group", "value"]);

        // Read columns back
        let vals_back = tb.read_column_by_name("value").unwrap();
        assert_eq!(vals_back, values);

        let keys_back = tb.read_column_by_name("group").unwrap();
        let keys_i32: Vec<i32> = keys_back.iter().map(|&v| v as i32).collect();
        assert_eq!(keys_i32, keys);

        // Header-only stats
        let (min, max, mean, count) = tb.global_stats(1); // value column
        assert_eq!(min, 1.0);
        assert_eq!(max, 5.0);
        assert_eq!(count, 5);
        assert!((mean - 3.0).abs() < 1e-10);

        fs::remove_file(&path).ok();
    }

    #[test]
    fn multi_tile_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_multi_tile.tb");

        let n = 10_000;
        let values: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let keys: Vec<i32> = (0..n).map(|i| (i % 100) as i32).collect();

        write_tb(
            &path,
            &[
                TbColumnWrite::key_column("k", &keys),
                TbColumnWrite::f64_column("v", values.clone()),
            ],
            1024, // small tiles: ceil(10000/1024) = 10 tiles
        )
        .unwrap();

        let tb = TbFile::open(&path).unwrap();
        assert_eq!(tb.n_rows(), n as u64);
        assert_eq!(tb.header.n_tiles, 10); // ceil(10000/1024)

        let vals_back = tb.read_column_by_name("v").unwrap();
        assert_eq!(vals_back, values);

        // Global stats from tile headers
        let (min, max, mean, count) = tb.global_stats(1);
        assert_eq!(min, 0.0);
        assert_eq!(max, 9999.0);
        assert_eq!(count, n as u64);
        assert!((mean - 4999.5).abs() < 0.01);

        fs::remove_file(&path).ok();
    }
}
