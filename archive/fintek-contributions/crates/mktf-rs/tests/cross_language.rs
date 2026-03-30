//! Cross-language verification: Rust writer output must be readable by Python reader
//! and vice versa. Tests byte-level compatibility of the MKTF format implementation.

use std::fs;
use std::path::Path;
use tempfile::TempDir;

use mktf::format::*;
use mktf::writer::*;
use mktf::reader;

/// Write with Rust, verify the file is structurally correct.
/// The Python reader test is in tests/test_rust_mktf_compat.py.
#[test]
fn rust_write_structural_verification() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("rust_written.mktf");

    // KO05-like: 10 columns (2 source × 5 stats), 78 bins
    let mut cols = Vec::new();
    for src in &["price", "volume"] {
        for suffix in &["_sum", "_sum_sq", "_min", "_max", "_count"] {
            let name = format!("{src}{suffix}");
            let data: Vec<u8> = (0..78u32)
                .flat_map(|i| {
                    let v = 100.0f32 + i as f32 * 1.5;
                    v.to_le_bytes()
                })
                .collect();
            cols.push(ColumnData::new(name, DTYPE_FLOAT32, data));
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

    let header = write_mktf(&path, &cols, &opts).unwrap();

    // Verify structure
    assert!(header.is_complete);
    assert!(!header.is_dirty);
    assert_eq!(header.alignment, 64);
    assert_eq!(header.n_cols, 10);
    assert_eq!(header.n_rows, 78);
    assert!(header.data_checksum != 0);
    assert!(header.bytes_file > 0);

    // Read back raw bytes
    let raw = fs::read(&path).unwrap();
    assert_eq!(raw.len(), header.bytes_file as usize);

    // Magic
    assert_eq!(&raw[0..4], b"MKTF");

    // Format version
    assert_eq!(u16::from_le_bytes([raw[4], raw[5]]), 4);

    // Alignment stored correctly
    assert_eq!(u16::from_le_bytes([raw[10], raw[11]]), 64);

    // Status at header and EOF
    assert_eq!(raw[4094], 1); // is_complete
    assert_eq!(raw[4095], 0); // is_dirty
    assert_eq!(raw[raw.len() - 2], 1); // is_complete at EOF
    assert_eq!(raw[raw.len() - 1], 0); // is_dirty at EOF

    // Read back with Rust reader
    let (h2, read_cols) = reader::read_columns(&path).unwrap();
    assert_eq!(h2.leaf_id, "K02P01C01.TI00TO05.KI00KO05");
    assert_eq!(h2.ticker, "AAPL");
    assert_eq!(h2.day, "2026-03-28");
    assert_eq!(read_cols.len(), 10);

    // Verify data is bit-exact
    for (i, (name, data)) in read_cols.iter().enumerate() {
        let original = &cols[i].data;
        assert_eq!(data.len(), original.len(), "Size mismatch for column {name}");
        assert_eq!(data, original, "Data mismatch for column {name}");
    }

    // Checksum verification
    assert!(reader::verify_checksum(&path).unwrap());

    println!("PASS: Rust structural verification (10 cols, align=64)");
}

/// Write with compression, verify roundtrip.
#[test]
fn rust_compressed_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("compressed.mktf");

    let data: Vec<u8> = (0..1000u32)
        .flat_map(|i| (100.0f32 + i as f32 * 0.01).to_le_bytes())
        .collect();

    let cols = vec![
        ColumnData::new_compressed("price_sum".into(), DTYPE_FLOAT32, data.clone()),
    ];

    let opts = WriteOptions {
        leaf_id: "test.compressed".into(),
        alignment: 64,
        safe: false,
        ..Default::default()
    };

    write_mktf(&path, &cols, &opts).unwrap();

    let (h, read_cols) = reader::read_columns(&path).unwrap();
    assert!(h.is_complete);
    assert_eq!(read_cols.len(), 1);

    // Data must decompress to original
    assert_eq!(read_cols[0].1, data);

    // File should be smaller than uncompressed
    let file_size = fs::metadata(&path).unwrap().len();
    let uncomp_path = tmp.path().join("uncompressed.mktf");
    let uncomp_cols = vec![
        ColumnData::new("price_sum".into(), DTYPE_FLOAT32, data),
    ];
    write_mktf(&uncomp_path, &uncomp_cols, &opts).unwrap();
    let uncomp_size = fs::metadata(&uncomp_path).unwrap().len();

    assert!(
        file_size < uncomp_size,
        "Compressed {} >= uncompressed {}", file_size, uncomp_size,
    );

    println!(
        "PASS: Compressed roundtrip (shuffle+LZ4, {:.1}x compression)",
        uncomp_size as f64 / file_size as f64,
    );
}

/// Write existing MKTF file from the batch conversion, read with Rust reader.
#[test]
fn read_python_written_mktf() {
    // Try to read one of the batch-converted MKTF files
    let aapl_path = Path::new("W:/fintek/data/fractal/K01/2025-09-02/AAPL/K01P01.TI00TO00.mktf");
    if !aapl_path.exists() {
        println!("SKIP: AAPL MKTF not found at {:?} (batch conversion may not have run)", aapl_path);
        return;
    }

    let (h, cols) = reader::read_columns(aapl_path).unwrap();
    assert!(h.is_complete);
    assert!(!h.is_dirty);
    assert_eq!(h.ticker, "AAPL");
    assert!(h.n_rows > 0);
    assert!(!cols.is_empty());

    println!("PASS: Read Python-written MKTF (AAPL, {} rows, {} cols)", h.n_rows, cols.len());

    // Verify checksum
    assert!(reader::verify_checksum(aapl_path).unwrap());
    println!("PASS: Checksum verified on Python-written MKTF");
}

/// Multiple dtype roundtrip (f32, i64, i32, u8).
#[test]
fn multi_dtype_roundtrip() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("multi_dtype.mktf");

    let f32_data: Vec<u8> = (0..100u32).flat_map(|i| (i as f32).to_le_bytes()).collect();
    let i64_data: Vec<u8> = (0..100i64).flat_map(|i| i.to_le_bytes()).collect();
    let i32_data: Vec<u8> = (0..100i32).flat_map(|i| i.to_le_bytes()).collect();
    let u8_data: Vec<u8> = (0..100u8).collect();

    let cols = vec![
        ColumnData::new("price".into(), DTYPE_FLOAT32, f32_data.clone()),
        ColumnData::new("timestamp".into(), DTYPE_INT64, i64_data.clone()),
        ColumnData::new("exchange".into(), DTYPE_INT32, i32_data.clone()),
        ColumnData::new("is_trf".into(), DTYPE_UINT8, u8_data.clone()),
    ];

    let opts = WriteOptions {
        leaf_id: "K01P01.TI00TO00".into(),
        alignment: 4096,
        safe: false,
        ..Default::default()
    };

    write_mktf(&path, &cols, &opts).unwrap();

    let (h, read_cols) = reader::read_columns(&path).unwrap();
    assert_eq!(h.n_cols, 4);
    assert_eq!(read_cols[0].1, f32_data);
    assert_eq!(read_cols[1].1, i64_data);
    assert_eq!(read_cols[2].1, i32_data);
    assert_eq!(read_cols[3].1, u8_data);

    println!("PASS: Multi-dtype roundtrip (f32, i64, i32, u8)");
}
