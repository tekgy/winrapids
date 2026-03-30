//! Benchmarks — measures per-file write overhead.
//!
//! Run with: cargo test --release bench_ -- --nocapture

#[cfg(test)]
mod tests {
    use std::time::Instant;
    use tempfile::TempDir;

    use crate::format::*;
    use crate::writer::*;
    use crate::reader;

    /// Generate KO05-like data: 5 stat columns per source column, float32.
    fn make_ko05_columns(n_source_cols: usize, n_bins: usize) -> Vec<ColumnData> {
        let suffixes = ["_sum", "_sum_sq", "_min", "_max", "_count"];
        let source_names = ["price", "volume", "log_price", "spread", "imbalance"];
        let mut cols = Vec::new();

        for src in source_names.iter().take(n_source_cols) {
            for suffix in &suffixes {
                let name = format!("{src}{suffix}");
                let data: Vec<u8> = (0..n_bins)
                    .flat_map(|i| {
                        let v = 100.0f32 + i as f32 * 0.1;
                        v.to_le_bytes()
                    })
                    .collect();
                cols.push(ColumnData::new(name, DTYPE_FLOAT32, data));
            }
        }
        cols
    }

    #[test]
    fn bench_ko05_write_uncompressed() {
        let tmp = TempDir::new().unwrap();
        let n_files = 1000;
        let cols = make_ko05_columns(5, 78);

        let opts = WriteOptions {
            leaf_id: "K02P01C01.TI00TO05.KI00KO05".into(),
            ticker: "AAPL".into(),
            day: "2026-03-28".into(),
            alignment: 64,
            safe: false,
            ..Default::default()
        };

        for i in 0..10 {
            let path = tmp.path().join(format!("warmup_{i}.mktf"));
            write_mktf(&path, &cols, &opts).unwrap();
        }

        let t0 = Instant::now();
        for i in 0..n_files {
            let path = tmp.path().join(format!("bench_{i}.mktf"));
            write_mktf(&path, &cols, &opts).unwrap();
        }
        let elapsed = t0.elapsed();

        let per_file_us = elapsed.as_micros() as f64 / n_files as f64;
        println!("\n=== KO05 Uncompressed Write (25 cols x 78 bins, align=64) ===");
        println!("  Per file:    {per_file_us:.1}us ({:.3}ms)", per_file_us / 1000.0);
        println!("  Throughput:  {:.0} files/s", n_files as f64 / elapsed.as_secs_f64());
        println!("  Python ref:  870us (0.87ms)");
        println!("  Speedup:     {:.1}x", 870.0 / per_file_us);
    }

    #[test]
    fn bench_buffer_only() {
        let n_iters = 10_000;
        let cols = make_ko05_columns(5, 78);
        let alignment = 64u64;
        let n_col_count = cols.len();

        let t0 = Instant::now();
        for _ in 0..n_iters {
            let dir_offset = BLOCK_SIZE as u64;
            let dir_size = (n_col_count * ENTRY_SIZE) as u64;
            let dir_end = dir_offset + dir_size;
            let data_region_start = align(dir_end, alignment);

            let mut current_offset = data_region_start;
            let mut entries: Vec<ByteEntry> = Vec::with_capacity(n_col_count);

            for col in &cols {
                let typesize = dtype_size(col.dtype_code) as u8;
                entries.push(ByteEntry {
                    name: col.name.clone(),
                    dtype_code: col.dtype_code,
                    n_elements: col.n_elements,
                    data_offset: current_offset,
                    data_nbytes: col.data.len() as u64,
                    typesize,
                    ..ByteEntry::default()
                });
                current_offset = align(current_offset + col.data.len() as u64, alignment);
            }

            let file_end = current_offset + 2;
            let mut buf = vec![0u8; file_end as usize];

            let header = MktfHeader {
                alignment: 64,
                n_cols: n_col_count as u16,
                n_rows: cols[0].n_elements,
                bytes_file: file_end,
                dir_offset,
                dir_entries: n_col_count as u64,
                data_start: data_region_start,
                flags: FLAG_HAS_QUALITY,
                ..MktfHeader::default()
            };
            let block0 = pack_block0(&header);
            buf[..BLOCK_SIZE].copy_from_slice(&block0);

            for (i, entry) in entries.iter().enumerate() {
                let off = BLOCK_SIZE + i * ENTRY_SIZE;
                let packed = pack_byte_entry(entry);
                buf[off..off + ENTRY_SIZE].copy_from_slice(&packed);
            }

            use sha2::{Sha256, Digest};
            let mut hasher = Sha256::new();
            for (col, entry) in cols.iter().zip(entries.iter()) {
                let off = entry.data_offset as usize;
                buf[off..off + col.data.len()].copy_from_slice(&col.data);
                hasher.update(&col.data);
            }
            let hash = hasher.finalize();
            let _checksum = u64::from_le_bytes(hash[..8].try_into().unwrap());
            std::hint::black_box(&buf);
        }
        let elapsed = t0.elapsed();
        let per_iter_us = elapsed.as_micros() as f64 / n_iters as f64;

        println!("\n=== Buffer-Only (no I/O) — 25 cols x 78 bins ===");
        println!("  Per iter:    {per_iter_us:.1}us");
        println!("  This is the compute floor — everything above is NTFS I/O");
    }

    #[test]
    fn bench_ntfs_overhead() {
        let tmp = TempDir::new().unwrap();
        let n_files = 1000;
        let data = vec![0u8; 15_000];

        for i in 0..10 {
            let path = tmp.path().join(format!("warmup_{i}.bin"));
            let tmp_path = path.with_extension("bin.tmp");
            std::fs::write(&tmp_path, &data).unwrap();
            std::fs::rename(&tmp_path, &path).unwrap();
        }

        let t0 = Instant::now();
        for i in 0..n_files {
            let path = tmp.path().join(format!("bench_{i}.bin"));
            let tmp_path = path.with_extension("bin.tmp");
            std::fs::write(&tmp_path, &data).unwrap();
            std::fs::rename(&tmp_path, &path).unwrap();
        }
        let elapsed = t0.elapsed();
        let per_file_us = elapsed.as_micros() as f64 / n_files as f64;

        println!("\n=== NTFS Overhead — create + write(15KB) + rename ===");
        println!("  Per file:  {per_file_us:.1}us");
        println!("  This is the irreducible NTFS floor (new file creation)");
    }

    #[test]
    fn bench_ntfs_overwrite() {
        // Test: does writing to an EXISTING file cost less than creating new?
        let tmp = TempDir::new().unwrap();
        let n_files = 1000;
        let data = vec![0u8; 15_000];

        // Pre-create all files
        for i in 0..n_files {
            let path = tmp.path().join(format!("pre_{i}.bin"));
            std::fs::write(&path, &data).unwrap();
        }

        // Warmup: overwrite existing
        for i in 0..10 {
            let path = tmp.path().join(format!("pre_{i}.bin"));
            std::fs::write(&path, &data).unwrap();
        }

        // Timed: overwrite existing files (no rename needed)
        let t0 = Instant::now();
        for i in 0..n_files {
            let path = tmp.path().join(format!("pre_{i}.bin"));
            std::fs::write(&path, &data).unwrap();
        }
        let elapsed = t0.elapsed();
        let per_file_us = elapsed.as_micros() as f64 / n_files as f64;

        println!("\n=== NTFS Overwrite — write(15KB) to existing file ===");
        println!("  Per file:  {per_file_us:.1}us");
        println!("  Compare to new file creation above");
    }

    #[test]
    fn bench_ntfs_open_write_close() {
        // Test: explicit open -> write -> close on pre-created files
        use std::io::Write;
        let tmp = TempDir::new().unwrap();
        let n_files = 1000;
        let data = vec![0u8; 15_000];

        // Pre-create
        for i in 0..n_files {
            let path = tmp.path().join(format!("owc_{i}.bin"));
            std::fs::write(&path, &data).unwrap();
        }

        // Warmup
        for i in 0..10 {
            let path = tmp.path().join(format!("owc_{i}.bin"));
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&data).unwrap();
        }

        let t0 = Instant::now();
        for i in 0..n_files {
            let path = tmp.path().join(format!("owc_{i}.bin"));
            let mut f = std::fs::File::create(&path).unwrap();
            f.write_all(&data).unwrap();
        }
        let elapsed = t0.elapsed();
        let per_file_us = elapsed.as_micros() as f64 / n_files as f64;

        println!("\n=== NTFS Open+Write+Close — pre-created 15KB files ===");
        println!("  Per file:  {per_file_us:.1}us");
    }

    #[test]
    fn bench_direct_write_no_rename() {
        let tmp = TempDir::new().unwrap();
        let n_files = 1000;
        let cols = make_ko05_columns(5, 78);

        let opts = WriteOptions {
            leaf_id: "K02P01C01.TI00TO05.KI00KO05".into(),
            ticker: "AAPL".into(),
            day: "2026-03-28".into(),
            alignment: 64,
            safe: false,
            ..Default::default()
        };

        for i in 0..10 {
            let path = tmp.path().join(format!("warmup_{i}.mktf"));
            write_mktf_direct(&path, &cols, &opts).unwrap();
        }

        let t0 = Instant::now();
        for i in 0..n_files {
            let path = tmp.path().join(format!("bench_{i}.mktf"));
            write_mktf_direct(&path, &cols, &opts).unwrap();
        }
        let elapsed = t0.elapsed();
        let per_file_us = elapsed.as_micros() as f64 / n_files as f64;

        println!("\n=== Direct Write (no rename) — 25 cols x 78 bins ===");
        println!("  Per file:  {per_file_us:.1}us");

        let check_path = tmp.path().join("bench_0.mktf");
        let (h, _) = reader::read_columns(&check_path).unwrap();
        assert!(h.is_complete);
    }

    #[test]
    fn bench_direct_overwrite() {
        // write_mktf_direct to PRE-CREATED files
        let tmp = TempDir::new().unwrap();
        let n_files = 1000;
        let cols = make_ko05_columns(5, 78);

        let opts = WriteOptions {
            leaf_id: "K02P01C01.TI00TO05.KI00KO05".into(),
            ticker: "AAPL".into(),
            day: "2026-03-28".into(),
            alignment: 64,
            safe: false,
            ..Default::default()
        };

        // Pre-create all files
        for i in 0..n_files {
            let path = tmp.path().join(format!("bench_{i}.mktf"));
            write_mktf_direct(&path, &cols, &opts).unwrap();
        }

        // Warmup: overwrite
        for i in 0..10 {
            let path = tmp.path().join(format!("bench_{i}.mktf"));
            write_mktf_direct(&path, &cols, &opts).unwrap();
        }

        // Timed: overwrite pre-created files
        let t0 = Instant::now();
        for i in 0..n_files {
            let path = tmp.path().join(format!("bench_{i}.mktf"));
            write_mktf_direct(&path, &cols, &opts).unwrap();
        }
        let elapsed = t0.elapsed();
        let per_file_us = elapsed.as_micros() as f64 / n_files as f64;

        println!("\n=== Direct Overwrite (pre-created) — 25 cols x 78 bins ===");
        println!("  Per file:  {per_file_us:.1}us");
        println!("  This tests the daemon pre-creation optimization");

        let check_path = tmp.path().join("bench_0.mktf");
        let (h, _) = reader::read_columns(&check_path).unwrap();
        assert!(h.is_complete);
    }

    #[test]
    fn bench_ko05_write_compressed() {
        let tmp = TempDir::new().unwrap();
        let n_files = 1000;

        let suffixes = ["_sum", "_sum_sq", "_min", "_max", "_count"];
        let source_names = ["price", "volume", "log_price", "spread", "imbalance"];
        let mut cols = Vec::new();

        for src in source_names.iter() {
            for suffix in &suffixes {
                let name = format!("{src}{suffix}");
                let data: Vec<u8> = (0..78)
                    .flat_map(|i| {
                        let v = 100.0f32 + i as f32 * 0.1;
                        v.to_le_bytes()
                    })
                    .collect();
                cols.push(ColumnData::new_compressed(name, DTYPE_FLOAT32, data));
            }
        }

        let opts = WriteOptions {
            leaf_id: "K02P01C01.TI00TO05.KI00KO05".into(),
            ticker: "AAPL".into(),
            day: "2026-03-28".into(),
            alignment: 64,
            safe: false,
            ..Default::default()
        };

        for i in 0..10 {
            let path = tmp.path().join(format!("warmup_{i}.mktf"));
            write_mktf(&path, &cols, &opts).unwrap();
        }

        let t0 = Instant::now();
        for i in 0..n_files {
            let path = tmp.path().join(format!("bench_{i}.mktf"));
            write_mktf(&path, &cols, &opts).unwrap();
        }
        let elapsed = t0.elapsed();
        let per_file_us = elapsed.as_micros() as f64 / n_files as f64;

        let comp_path = tmp.path().join("bench_0.mktf");
        let comp_size = std::fs::metadata(&comp_path).unwrap().len();

        let ref_cols = make_ko05_columns(5, 78);
        let ref_path = tmp.path().join("ref_uncomp.mktf");
        write_mktf(&ref_path, &ref_cols, &opts).unwrap();
        let uncomp_size = std::fs::metadata(&ref_path).unwrap().len();

        println!("\n=== KO05 Compressed Write (shuffle+LZ4, 25 cols x 78 bins) ===");
        println!("  Per file:      {per_file_us:.1}us");
        println!("  Compressed:    {comp_size} bytes");
        println!("  Uncompressed:  {uncomp_size} bytes");
        println!("  Ratio:         {:.2}x", uncomp_size as f64 / comp_size as f64);
    }

    #[test]
    fn bench_k01_write_large() {
        let tmp = TempDir::new().unwrap();
        let n_files = 100;

        let n_rows = 598_057;
        let cols: Vec<ColumnData> = ["price", "size", "timestamp", "exchange", "is_trf"]
            .iter()
            .enumerate()
            .map(|(i, name)| {
                let dtype = match i {
                    0 | 1 => DTYPE_FLOAT32,
                    2 => DTYPE_INT64,
                    3 => DTYPE_INT32,
                    4 => DTYPE_UINT8,
                    _ => DTYPE_FLOAT32,
                };
                let elem_size = dtype_size(dtype);
                let data = vec![0u8; n_rows * elem_size];
                ColumnData::new(name.to_string(), dtype, data)
            })
            .collect();

        let opts = WriteOptions {
            leaf_id: "K01P01.TI00TO00".into(),
            ticker: "AAPL".into(),
            day: "2026-03-28".into(),
            alignment: 4096,
            safe: false,
            ..Default::default()
        };

        for i in 0..3 {
            let path = tmp.path().join(format!("warmup_{i}.mktf"));
            write_mktf(&path, &cols, &opts).unwrap();
        }

        let t0 = Instant::now();
        for i in 0..n_files {
            let path = tmp.path().join(format!("bench_{i}.mktf"));
            write_mktf(&path, &cols, &opts).unwrap();
        }
        let elapsed = t0.elapsed();

        let per_file_ms = elapsed.as_millis() as f64 / n_files as f64;
        let file_size = std::fs::metadata(tmp.path().join("bench_0.mktf")).unwrap().len();
        let throughput_mb = (file_size as f64 * n_files as f64) / elapsed.as_secs_f64() / 1e6;

        println!("\n=== K01 Large Write (5 cols x 598K rows, align=4096) ===");
        println!("  Per file:    {per_file_ms:.1}ms");
        println!("  File size:   {:.1} MB", file_size as f64 / 1e6);
        println!("  Throughput:  {throughput_mb:.0} MB/s");
    }
}
