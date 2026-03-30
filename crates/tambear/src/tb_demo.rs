//! Campsite 6: .tb File Format — Write, Read, Header-Only Stats, GPU Groupby
//!
//! Writes 598K AAPL ticks to .tb, reads back via header-only path and
//! selective column reads, then runs GPU scatter groupby.
//!
//! "Tam doesn't read. Tam knows the summary."
//!
//! Run with: cargo run --bin tb-demo --release

use std::path::Path;
use std::time::Instant;
use tambear::{
    write_tb, TbFile, TbColumnWrite, HashScatterEngine,
    TileColumnStats, tile_skip_mask_gt,
};

fn load_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
    let n = bytes.len() / 4;
    let mut out = vec![0f32; n];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
    }
    out
}

fn load_i64(path: &Path) -> Vec<i64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
    let n = bytes.len() / 8;
    let mut out = vec![0i64; n];
    unsafe {
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len());
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Campsite 6: .tb File Format ===\n");

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let prices_f32 = load_f32(&data_dir.join("aapl_prices_f32.bin"));
    let timestamps = load_i64(&data_dir.join("aapl_timestamps_i64.bin"));
    let n_ticks = prices_f32.len();
    println!("{} AAPL ticks loaded from raw binary", n_ticks);

    // Compute minute bins
    let min_ts = timestamps.iter().cloned().min().unwrap_or(0);
    let ns_per_minute: i64 = 60_000_000_000;
    let minute_bins: Vec<i32> = timestamps.iter()
        .map(|&ts| ((ts - min_ts) / ns_per_minute) as i32)
        .collect();

    let prices: Vec<f64> = prices_f32.iter().map(|&p| p as f64).collect();

    // -----------------------------------------------------------------------
    // WRITE: Create .tb file
    // -----------------------------------------------------------------------
    let tb_path = data_dir.join("aapl_2025-09-02.tb");
    println!("\nWriting {}", tb_path.display());

    let t0 = Instant::now();
    write_tb(
        &tb_path,
        &[
            TbColumnWrite::key_column("minute", &minute_bins),
            TbColumnWrite::f64_column("price", prices.clone()),
        ],
        65_536, // default tile size
    )?;
    let write_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let file_size = std::fs::metadata(&tb_path)?.len();
    println!("  Written: {:.1} MB in {:.1}ms", file_size as f64 / 1e6, write_ms);

    // -----------------------------------------------------------------------
    // READ: Open .tb — header + tile stats only (NO data read yet)
    // -----------------------------------------------------------------------
    let t0 = Instant::now();
    let tb = TbFile::open(&tb_path)?;
    let open_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("\nOpened .tb in {:.2}ms", open_ms);
    println!("  n_rows:    {}", tb.n_rows());
    println!("  n_columns: {}", tb.n_columns());
    println!("  n_tiles:   {}", tb.header.n_tiles);
    println!("  tile_size: {}", tb.header.tile_size);
    println!("  columns:   {:?}", tb.column_names());

    // -----------------------------------------------------------------------
    // HEADER-ONLY: Global stats with ZERO data read
    // -----------------------------------------------------------------------
    let price_col_idx = tb.column_index("price").unwrap();
    let (min, max, mean, count) = tb.global_stats(price_col_idx);

    println!("\n--- Header-Only Stats (zero data read) ---");
    println!("  Price min:  ${:.2}", min);
    println!("  Price max:  ${:.2}", max);
    println!("  Price mean: ${:.2}", mean);
    println!("  Row count:  {}", count);
    println!("  Tile stats bytes read: {} (vs {} data bytes skipped)",
        tb.header.tile_header_section_size,
        file_size - 4096 - tb.header.tile_header_section_size);

    // -----------------------------------------------------------------------
    // PREDICATE PUSHDOWN: How many tiles can we skip for "price > $240"?
    // -----------------------------------------------------------------------
    let threshold = 240.0;
    let price_stats: Vec<TileColumnStats> = {
        let n_cols = tb.header.n_columns as usize;
        let n_tiles = tb.header.n_tiles as usize;
        (0..n_tiles).map(|t| tb.tile_stats[t * n_cols + price_col_idx]).collect()
    };
    let skip_mask = tile_skip_mask_gt(&price_stats, threshold);
    let skipped = skip_mask.iter().filter(|&&s| s).count();
    let total_tiles = skip_mask.len();
    println!("\n--- Predicate Pushdown: price > ${:.0} ---", threshold);
    println!("  {}/{} tiles skippable ({:.0}% skip rate)",
        skipped, total_tiles, 100.0 * skipped as f64 / total_tiles as f64);

    // -----------------------------------------------------------------------
    // DATA READ: Selective column load → GPU groupby
    // -----------------------------------------------------------------------
    println!("\n--- Selective Read + GPU Groupby ---");

    let t0 = Instant::now();
    let keys_f64 = tb.read_column_by_name("minute")?;
    let vals = tb.read_column_by_name("price")?;
    let read_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  Column read: {:.1}ms (2 columns, {} rows)", read_ms, vals.len());

    let keys: Vec<i32> = keys_f64.iter().map(|&v| v as i32).collect();
    let n_groups = (*keys.iter().max().unwrap_or(&0) + 1) as usize;

    let engine = HashScatterEngine::new()?;
    let t0 = Instant::now();
    let result = engine.groupby(&keys, &vals, n_groups)?;
    let groupby_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  GPU groupby: {:.2}ms", groupby_ms);

    let means = result.means();
    let stds = result.stds();
    let active = result.counts.iter().filter(|&&c| c > 0.0).count();
    println!("  Active bins: {}/{}", active, n_groups);

    // Show a few bins
    let mut shown = 0;
    for g in 0..n_groups {
        if result.counts[g] > 10.0 && shown < 5 {
            println!("  bin {:>4}: {:>5} ticks, mean=${:.2}, std=${:.4}",
                g, result.counts[g] as u32, means[g], stds[g]);
            shown += 1;
        }
    }

    // Verify roundtrip: data from .tb matches original
    let max_err = vals.iter().zip(&prices)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    println!("\n--- Roundtrip Verification ---");
    println!("  max_err(price): {:.2e}  {}", max_err, if max_err == 0.0 { "BIT-IDENTICAL" } else { "DIFFERS" });

    println!("\nTam doesn't read. Tam knows the summary.");
    Ok(())
}
