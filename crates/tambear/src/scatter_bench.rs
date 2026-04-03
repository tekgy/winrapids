//! Scatter Kernel Benchmark: naive vs shared-memory vs warp-aggregated
//!
//! Three kernel strategies for sort-free groupby:
//!   1. Naive:  3 global atomicAdds per element (baseline)
//!   2. Smem:   shared-memory privatized accumulators, merge to global
//!   3. Warp:   __match_any_sync + warp shuffle reduction, leader atomicAdd
//!
//! Tests on real AAPL data (598K elements, ~1200 groups) and synthetic
//! data with varying n_groups to find the crossover points.
//!
//! Run with: cargo run --bin scatter-bench --release

use std::path::Path;
use std::time::Instant;
use tambear::HashScatterEngine;

fn load_f32(path: &Path) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
    let n = bytes.len() / 4;
    let mut out = vec![0f32; n];
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len()); }
    out
}

fn load_i64(path: &Path) -> Vec<i64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {}", path.display(), e));
    let n = bytes.len() / 8;
    let mut out = vec![0i64; n];
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), out.as_mut_ptr() as *mut u8, bytes.len()); }
    out
}

/// Run GPU-resident groupby variant (avoids H2D copy in the timing loop).
fn bench_groupby_gpu(
    engine: &HashScatterEngine,
    keys: &[i32],
    values: &[f64],
    n_groups: usize,
    variant: &str, // "naive", "smem", "warp"
    n_iters: usize,
) -> f64 {
    let n = keys.len();
    let keys_dev = engine.stream().clone_htod(keys).unwrap();
    let values_dev = engine.stream().clone_htod(values).unwrap();

    // Warm up
    match variant {
        "naive" => { let _ = engine.groupby_gpu(&keys_dev, &values_dev, n, n_groups).unwrap(); }
        "smem"  => { let _ = engine.groupby_smem_gpu(&keys_dev, &values_dev, n, n_groups).unwrap(); }
        "warp"  => { let _ = engine.groupby_warp_gpu(&keys_dev, &values_dev, n, n_groups).unwrap(); }
        _ => panic!("unknown variant"),
    }
    engine.stream().synchronize().unwrap();

    let mut times = Vec::with_capacity(n_iters);
    for _ in 0..n_iters {
        let t0 = Instant::now();
        match variant {
            "naive" => { let _ = engine.groupby_gpu(&keys_dev, &values_dev, n, n_groups).unwrap(); }
            "smem"  => { let _ = engine.groupby_smem_gpu(&keys_dev, &values_dev, n, n_groups).unwrap(); }
            "warp"  => { let _ = engine.groupby_warp_gpu(&keys_dev, &values_dev, n, n_groups).unwrap(); }
            _ => {}
        }
        engine.stream().synchronize().unwrap();
        times.push(t0.elapsed().as_secs_f64() * 1e6);
    }
    times.sort_by(|a, b| a.total_cmp(b));
    times[n_iters / 2]
}

fn verify_results(a: &tambear::GroupByResult, b: &tambear::GroupByResult, label: &str) {
    let max_sum_err = a.sums.iter().zip(&b.sums)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    let max_sq_err = a.sum_sqs.iter().zip(&b.sum_sqs)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    let max_cnt_err = a.counts.iter().zip(&b.counts)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f64, f64::max);
    let ok = max_sum_err < 1e-6 && max_sq_err < 1e-3 && max_cnt_err < 0.5;
    println!("  {}: sum_err={:.2e} sq_err={:.2e} cnt_err={:.2e} {}",
        label, max_sum_err, max_sq_err, max_cnt_err,
        if ok { "OK" } else { "MISMATCH!" });
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Scatter Kernel Benchmark ===\n");

    let engine = HashScatterEngine::new()?;
    println!("NVRTC compiled (4 kernels: scatter_sum, scatter_stats, scatter_stats_smem, scatter_stats_warp)");

    // -----------------------------------------------------------------------
    // Part 1: Real AAPL data (598K elements, ~1200 groups)
    // -----------------------------------------------------------------------
    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let prices_f32 = load_f32(&data_dir.join("aapl_prices_f32.bin"));
    let timestamps = load_i64(&data_dir.join("aapl_timestamps_i64.bin"));
    let min_ts = timestamps.iter().cloned().min().unwrap_or(0);
    let ns_per_minute: i64 = 60_000_000_000;
    let keys: Vec<i32> = timestamps.iter()
        .map(|&ts| ((ts - min_ts) / ns_per_minute) as i32)
        .collect();
    let prices: Vec<f64> = prices_f32.iter().map(|&p| p as f64).collect();
    let n_groups = (*keys.iter().max().unwrap_or(&0) + 1) as usize;
    let n = prices.len();

    println!("\n--- Real AAPL: {} elements, {} groups ---", n, n_groups);

    // Correctness check
    let ref_result = engine.groupby(&keys, &prices, n_groups)?;
    let smem_result = engine.groupby_smem(&keys, &prices, n_groups)?;
    let warp_result = engine.groupby_warp(&keys, &prices, n_groups)?;
    verify_results(&ref_result, &smem_result, "smem vs naive");
    verify_results(&ref_result, &warp_result, "warp vs naive");

    // GPU-resident benchmark (no H2D copy overhead)
    let n_iters = 100;
    println!("\n  GPU-resident benchmark ({} iterations, median):", n_iters);
    let naive_us = bench_groupby_gpu(&engine, &keys, &prices, n_groups, "naive", n_iters);
    let smem_us  = bench_groupby_gpu(&engine, &keys, &prices, n_groups, "smem", n_iters);
    let warp_us  = bench_groupby_gpu(&engine, &keys, &prices, n_groups, "warp", n_iters);

    println!("  naive: {:>8.1}us", naive_us);
    println!("  smem:  {:>8.1}us  ({:.2}x vs naive)", smem_us, naive_us / smem_us);
    println!("  warp:  {:>8.1}us  ({:.2}x vs naive)", warp_us, naive_us / warp_us);

    // -----------------------------------------------------------------------
    // Part 2: Synthetic sweep — varying n_groups
    // -----------------------------------------------------------------------
    println!("\n--- Synthetic sweep: 1M elements, varying n_groups ---");
    println!("  {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "n_groups", "naive(us)", "smem(us)", "warp(us)", "smem/naiv", "warp/naiv");

    let synth_n = 1_000_000;
    let synth_values: Vec<f64> = (0..synth_n).map(|i| (i as f64) * 0.001).collect();

    for &ng in &[4, 16, 64, 256, 1024, 2048] {
        let synth_keys: Vec<i32> = (0..synth_n).map(|i| (i % ng) as i32).collect();
        let ng_sz = ng;

        let naive_us = bench_groupby_gpu(&engine, &synth_keys, &synth_values, ng_sz, "naive", 50);
        let smem_us  = bench_groupby_gpu(&engine, &synth_keys, &synth_values, ng_sz, "smem", 50);
        let warp_us  = bench_groupby_gpu(&engine, &synth_keys, &synth_values, ng_sz, "warp", 50);

        println!("  {:>10} {:>10.1} {:>10.1} {:>10.1} {:>10.2}x {:>10.2}x",
            ng_sz, naive_us, smem_us, warp_us,
            naive_us / smem_us, naive_us / warp_us);
    }

    // -----------------------------------------------------------------------
    // Part 3: High contention stress test
    // -----------------------------------------------------------------------
    println!("\n--- High contention: 1M elements, 4 groups ---");
    let stress_keys: Vec<i32> = (0..synth_n).map(|i| (i % 4) as i32).collect();
    let n_iters = 100;

    let naive_us = bench_groupby_gpu(&engine, &stress_keys, &synth_values, 4, "naive", n_iters);
    let smem_us  = bench_groupby_gpu(&engine, &stress_keys, &synth_values, 4, "smem", n_iters);
    let warp_us  = bench_groupby_gpu(&engine, &stress_keys, &synth_values, 4, "warp", n_iters);

    println!("  naive: {:>8.1}us", naive_us);
    println!("  smem:  {:>8.1}us  ({:.2}x)", smem_us, naive_us / smem_us);
    println!("  warp:  {:>8.1}us  ({:.2}x)", warp_us, naive_us / warp_us);

    // Verify high-contention correctness
    let ref_r = engine.groupby(&stress_keys, &synth_values, 4)?;
    let smem_r = engine.groupby_smem(&stress_keys, &synth_values, 4)?;
    let warp_r = engine.groupby_warp(&stress_keys, &synth_values, 4)?;
    verify_results(&ref_r, &smem_r, "smem vs naive (4 groups)");
    verify_results(&ref_r, &warp_r, "warp vs naive (4 groups)");

    println!("\nTam doesn't contend. Tam reduces locally.");
    Ok(())
}
