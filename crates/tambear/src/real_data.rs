//! Campsite 4: Real AAPL Tick Data Through the Scatter Pipeline
//!
//! Loads actual AAPL tick data (598K ticks, 2025-09-02), creates minute bins,
//! runs hash scatter groupby on GPU. Real data, real statistics.
//!
//! Data extracted from MKTF v5 files via Python → raw binary:
//!   data/aapl_prices_f32.bin     598057 × f32 (trade price)
//!   data/aapl_sizes_f32.bin      598057 × f32 (trade size)
//!   data/aapl_timestamps_i64.bin 598057 × i64 (nanosecond epoch)
//!
//! Run with: cargo run --bin tam-real-data --release

use std::path::Path;
use std::time::Instant;
use tambear::HashScatterEngine;

fn load_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
    let n = bytes.len() / 4;
    let mut out = vec![0f32; n];
    // safe: f32 is plain-old-data, no alignment issue with copy
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
    println!("=== Campsite 4: Real AAPL Data Through Tam ===\n");

    // Data lives next to the crate root
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let price_path = data_dir.join("aapl_prices_f32.bin");
    let size_path = data_dir.join("aapl_sizes_f32.bin");
    let ts_path = data_dir.join("aapl_timestamps_i64.bin");

    if !price_path.exists() {
        println!("AAPL data not found at: {}", data_dir.display());
        println!("Run the Python extraction script first, or use synthetic fallback.\n");
        return run_synthetic();
    }

    // -----------------------------------------------------------------------
    // Load AAPL prices + timestamps from raw binary
    // -----------------------------------------------------------------------
    let t0 = Instant::now();
    let prices_f32 = load_f32(&price_path);
    let load_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let n_ticks = prices_f32.len();

    println!("{} AAPL ticks loaded in {:.1}ms", n_ticks, load_ms);
    println!("  Price range: ${:.2} — ${:.2}",
        prices_f32.iter().cloned().fold(f32::INFINITY, f32::min),
        prices_f32.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // Promote to f64 for scatter engine
    let prices: Vec<f64> = prices_f32.iter().map(|&p| p as f64).collect();

    // Load sizes if available
    if size_path.exists() {
        let sizes = load_f32(&size_path);
        let total_volume: f64 = sizes.iter().map(|&s| s as f64).sum();
        println!("  Total volume: {:.0} shares", total_volume);
    }

    // -----------------------------------------------------------------------
    // Compute minute bins from nanosecond timestamps
    // -----------------------------------------------------------------------
    let (keys, n_groups, group_label) = if ts_path.exists() {
        let timestamps = load_i64(&ts_path);
        assert_eq!(timestamps.len(), n_ticks, "timestamp/price length mismatch");

        let min_ts = timestamps.iter().cloned().min().unwrap_or(0);
        let ns_per_minute: i64 = 60_000_000_000;
        let keys: Vec<i32> = timestamps.iter()
            .map(|&ts| ((ts - min_ts) / ns_per_minute) as i32)
            .collect();
        let n_groups = (*keys.iter().max().unwrap_or(&0) + 1) as usize;
        println!("  {} minute bins (08:00–16:00 ET)", n_groups);
        (keys, n_groups, "minute")
    } else {
        // Fallback: synthetic bins (~390 bins for a 6.5h trading day)
        let n_groups = 390;
        let ticks_per_bin = n_ticks / n_groups;
        let keys: Vec<i32> = (0..n_ticks)
            .map(|i| (i / ticks_per_bin).min(n_groups - 1) as i32)
            .collect();
        println!("  No timestamps — using {} synthetic minute bins", n_groups);
        (keys, n_groups, "synthetic_minute")
    };

    // -----------------------------------------------------------------------
    // GPU scatter groupby
    // -----------------------------------------------------------------------
    println!("\nInitializing Tam...");
    let t0 = Instant::now();
    let engine = HashScatterEngine::new()?;
    let init_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  NVRTC compile: {:.1}ms", init_ms);

    println!("\nRunning scatter_stats on {} AAPL ticks, {} {} bins...",
        n_ticks, n_groups, group_label);
    let t0 = Instant::now();
    let result = engine.groupby(&keys, &prices, n_groups)?;
    let groupby_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("  groupby time: {:.2}ms (includes H2D + kernel + D2H)", groupby_ms);

    // GPU-only timing (data already resident)
    let keys_dev = engine.stream().clone_htod(&keys)?;
    let values_dev = engine.stream().clone_htod(&prices)?;
    let _ = engine.groupby_gpu(&keys_dev, &values_dev, n_ticks, n_groups)?;
    engine.stream().synchronize()?;

    let n_iters = 100;
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.groupby_gpu(&keys_dev, &values_dev, n_ticks, n_groups)?;
    }
    engine.stream().synchronize()?;
    let kernel_us = t0.elapsed().as_secs_f64() * 1e6 / n_iters as f64;
    println!("  kernel-only:   {:.0}us/op ({} iterations)", kernel_us, n_iters);

    // -----------------------------------------------------------------------
    // Real financial statistics
    // -----------------------------------------------------------------------
    let means = result.means();
    let stds = result.stds();

    println!("\n--- Per-{} Statistics (AAPL 2025-09-02) ---", group_label);
    println!("{:>6} {:>8} {:>10} {:>10} {:>10}", "Bin", "Ticks", "Mean($)", "Std($)", "Vol(bp)");
    println!("{}", "-".repeat(50));

    // Show first 10 non-empty bins
    let mut shown = 0;
    for g in 0..n_groups {
        if result.counts[g] > 1.0 && shown < 10 {
            let vol_bp = stds[g] / means[g] * 10000.0;
            println!("{:>6} {:>8} {:>10.2} {:>10.4} {:>10.1}",
                g, result.counts[g] as u32, means[g], stds[g], vol_bp);
            shown += 1;
        }
    }

    let active_bins = result.counts.iter().filter(|&&c| c > 0.0).count();
    let total_ticks: f64 = result.counts.iter().sum();
    let day_mean: f64 = result.sums.iter().sum::<f64>() / total_ticks;

    // Find the most volatile minute
    let mut max_vol_bp = 0.0f64;
    let mut max_vol_bin = 0;
    for g in 0..n_groups {
        if result.counts[g] > 10.0 {
            let vol = stds[g] / means[g] * 10000.0;
            if vol > max_vol_bp {
                max_vol_bp = vol;
                max_vol_bin = g;
            }
        }
    }

    println!("\n--- Day Summary ---");
    println!("Active {} bins: {}", group_label, active_bins);
    println!("Total ticks:   {}", total_ticks as u64);
    println!("Day VWAP:      ${:.2}", day_mean);
    println!("Hottest minute: bin {} ({:.1} bp)", max_vol_bin, max_vol_bp);
    println!("Kernel time:   {:.0}us  (sort baseline: ~1040us)", kernel_us);

    println!("\nTam doesn't sort. Tam knows where everything is.");
    Ok(())
}

fn run_synthetic() -> Result<(), Box<dyn std::error::Error>> {
    // Synthetic AAPL-like data: $230 +/- noise, 500K ticks, 390 minute bins
    let n_ticks = 500_000;
    let n_groups = 390;
    let base_price = 230.0;

    let prices: Vec<f64> = (0..n_ticks).map(|i| {
        let t = i as f64 / n_ticks as f64;
        base_price + 2.0 * (t * 3.14159).sin()
            + 0.5 * ((i * 17 + 3) as f64 * 0.618033988749).fract()
    }).collect();

    let ticks_per_bin = n_ticks / n_groups;
    let keys: Vec<i32> = (0..n_ticks)
        .map(|i| (i / ticks_per_bin).min(n_groups - 1) as i32)
        .collect();

    println!("Synthetic: {} ticks, {} minute bins, base=${:.0}\n", n_ticks, n_groups, base_price);

    let engine = HashScatterEngine::new()?;
    let result = engine.groupby(&keys, &prices, n_groups)?;
    let means = result.means();
    let stds = result.stds();

    println!("{:>6} {:>8} {:>10} {:>10}", "Bin", "Ticks", "Mean($)", "Std($)");
    for g in 0..5 {
        println!("{:>6} {:>8} {:>10.2} {:>10.4}", g, result.counts[g] as u32, means[g], stds[g]);
    }

    println!("\nTam doesn't sort. Tam knows where everything is.");
    Ok(())
}
