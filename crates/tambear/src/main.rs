//! Campsite 1: Sort-Free GroupBy via Hash Scatter
//!
//! Proves the 17x claim from manuscript 014.
//! Run with: cargo run --bin tambear-test --release

use std::time::Instant;
use tambear::HashScatterEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Campsite 1: Sort-Free GroupBy via Hash Scatter ===");
    println!("=== Tam doesn't sort. Tam knows where everything is. ===\n");

    let n: usize = 1_000_000;
    let n_groups: usize = 4_600;

    println!("Config: {} rows, {} groups (FinTek ticker universe)\n", n, n_groups);

    // -----------------------------------------------------------------------
    // Generate deterministic data
    // -----------------------------------------------------------------------
    let keys: Vec<i32> = (0..n).map(|i| (i % n_groups) as i32).collect();
    let values: Vec<f64> = (0..n).map(|i| (i + 1) as f64 * 0.001).collect();

    // CPU reference
    let mut ref_sums = vec![0.0f64; n_groups];
    let mut ref_sum_sqs = vec![0.0f64; n_groups];
    let mut ref_counts = vec![0.0f64; n_groups];
    let mut ref_mins = vec![f64::INFINITY; n_groups];
    let mut ref_maxs = vec![f64::NEG_INFINITY; n_groups];
    for i in 0..n {
        let g = keys[i] as usize;
        let v = values[i];
        ref_sums[g] += v;
        ref_sum_sqs[g] += v * v;
        ref_counts[g] += 1.0;
        if v < ref_mins[g] { ref_mins[g] = v; }
        if v > ref_maxs[g] { ref_maxs[g] = v; }
    }

    // -----------------------------------------------------------------------
    // Compile NVRTC kernels
    // -----------------------------------------------------------------------
    let t0 = Instant::now();
    let engine = HashScatterEngine::new()?;
    let compile_ms = t0.elapsed().as_secs_f64() * 1000.0;
    println!("NVRTC compile: {:.1}ms\n", compile_ms);

    // -----------------------------------------------------------------------
    // Correctness: scatter_sum
    // -----------------------------------------------------------------------
    let gpu_sums = engine.scatter_sum(&keys, &values, n_groups)?;
    let max_err_sum = ref_sums.iter().zip(&gpu_sums)
        .map(|(r, g)| (r - g).abs())
        .fold(0.0f64, f64::max);
    let sum_ok = max_err_sum < 1e-6;
    println!("--- Correctness ---");
    println!("scatter_sum:    max_err = {:.2e}  {}", max_err_sum, if sum_ok { "PASS" } else { "FAIL" });
    assert!(sum_ok, "scatter_sum error too large: {:.2e}", max_err_sum);

    // -----------------------------------------------------------------------
    // Correctness: scatter_stats (groupby)
    // -----------------------------------------------------------------------
    let result = engine.groupby(&keys, &values, n_groups)?;

    let max_err_sums = ref_sums.iter().zip(&result.sums)
        .map(|(r, g)| (r - g).abs())
        .fold(0.0f64, f64::max);
    let max_err_sq = ref_sum_sqs.iter().zip(&result.sum_sqs)
        .map(|(r, g)| (r - g).abs())
        .fold(0.0f64, f64::max);
    let max_err_cnt = ref_counts.iter().zip(&result.counts)
        .map(|(r, g)| (r - g).abs())
        .fold(0.0f64, f64::max);

    println!("scatter_stats:");
    println!("  sums:         max_err = {:.2e}  {}", max_err_sums,
        if max_err_sums < 1e-6 { "PASS" } else { "FAIL" });
    println!("  sum_sqs:      max_err = {:.2e}  {}", max_err_sq,
        if max_err_sq < 1e-4 { "PASS" } else { "FAIL" });
    println!("  counts:       max_err = {:.2e}  {}", max_err_cnt,
        if max_err_cnt < 0.5 { "PASS" } else { "FAIL" });

    // Derived statistics
    let means = result.means();
    let ref_means: Vec<f64> = ref_sums.iter().zip(&ref_counts)
        .map(|(&s, &c)| s / c)
        .collect();
    let max_err_mean = means.iter().zip(&ref_means)
        .map(|(g, r)| (g - r).abs())
        .fold(0.0f64, f64::max);
    println!("  means:        max_err = {:.2e}  {}", max_err_mean,
        if max_err_mean < 1e-10 { "PASS" } else { "FAIL" });

    let vars = result.variances();
    let ref_vars: Vec<f64> = (0..n_groups).map(|g| {
        let c = ref_counts[g];
        if c > 1.0 {
            (ref_sum_sqs[g] - ref_sums[g] * ref_sums[g] / c) / (c - 1.0)
        } else {
            f64::NAN
        }
    }).collect();
    let max_err_var = vars.iter().zip(&ref_vars)
        .filter(|(g, r)| !g.is_nan() && !r.is_nan())
        .map(|(g, r)| (g - r).abs() / r.abs().max(1e-15))
        .fold(0.0f64, f64::max);
    println!("  variances:    rel_err = {:.2e}  {}", max_err_var,
        if max_err_var < 1e-6 { "PASS" } else { "FAIL" });

    assert!(max_err_sums < 1e-6, "scatter_stats sums failed");
    assert!(max_err_cnt < 0.5, "scatter_stats counts failed");

    // -----------------------------------------------------------------------
    // Benchmark: kernel time only (data pre-copied to GPU)
    // -----------------------------------------------------------------------
    println!("\n--- Benchmark (data pre-copied to GPU) ---");

    let keys_dev = engine.stream().clone_htod(&keys)?;
    let values_dev = engine.stream().clone_htod(&values)?;

    let n_iters = 100;

    // Warm up
    let _ = engine.scatter_sum_gpu(&keys_dev, &values_dev, n, n_groups)?;
    engine.stream().synchronize()?;

    // Benchmark scatter_sum
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.scatter_sum_gpu(&keys_dev, &values_dev, n, n_groups)?;
    }
    engine.stream().synchronize()?;
    let sum_us = t0.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

    // Warm up
    let _ = engine.groupby_gpu(&keys_dev, &values_dev, n, n_groups)?;
    engine.stream().synchronize()?;

    // Benchmark scatter_stats
    let t0 = Instant::now();
    for _ in 0..n_iters {
        let _ = engine.groupby_gpu(&keys_dev, &values_dev, n, n_groups)?;
    }
    engine.stream().synchronize()?;
    let stats_us = t0.elapsed().as_secs_f64() * 1e6 / n_iters as f64;

    println!("scatter_sum:    {:.0} us/op  ({:.2} ms)", sum_us, sum_us / 1000.0);
    println!("scatter_stats:  {:.0} us/op  ({:.2} ms)  [3 aggs for 1 read]", stats_us, stats_us / 1000.0);
    println!("sort baseline:  ~1040 us  (manuscript 014, argsort alone = 490 us)");
    if sum_us > 0.0 {
        println!("speedup (sum):  {:.0}x vs sort-based", 1040.0 / sum_us);
    }

    // -----------------------------------------------------------------------
    // Sample output
    // -----------------------------------------------------------------------
    println!("\n--- Per-group sample (groups 0-4 of {}) ---", n_groups);
    let stds = result.stds();
    for g in 0..5 {
        println!("  Group {:>4}: n={:>3}, sum={:>10.3}, mean={:.6}, std={:.6}",
            g, result.counts[g] as u32, result.sums[g], means[g], stds[g]);
    }

    println!("\nTam doesn't sort. Tam knows where everything is.");
    Ok(())
}
