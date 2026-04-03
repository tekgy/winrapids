//! tb.train demo — end-to-end model training on real data.
//!
//! Pipeline:
//!   1. Load 598K AAPL ticks from raw binary
//!   2. scatter_phi: aggregate per-minute statistics (mean price, volatility, tick count)
//!   3. train.linear.fit: predict minute mean price from time-of-day features
//!   4. Show real-time training stats, model coefficients, R^2
//!
//! This chains scatter (feature engineering) -> tiled (normal equations) -> model.
//! The full pipeline uses GPU primitives end to end.
//!
//! "Tam doesn't train. Tam accumulates."
//!
//! Run with: cargo run --bin train-demo --release

use std::path::Path;
use std::time::Instant;

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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== tb.train Demo: Scatter -> Linear Regression ===\n");

    // -----------------------------------------------------------------------
    // Part 1: Synthetic proof — verify exact coefficient recovery
    // -----------------------------------------------------------------------
    println!("--- Part 1: Synthetic (exact recovery) ---");
    {
        let n = 10_000;
        let d = 3;
        let mut x = vec![0.0f64; n * d];
        let mut y = vec![0.0f64; n];
        for i in 0..n {
            let x1 = (i as f64) / 1000.0;
            let x2 = ((i as f64) * 0.1).sin();
            let x3 = ((i as f64) * 0.01).cos();
            x[i * d] = x1;
            x[i * d + 1] = x2;
            x[i * d + 2] = x3;
            y[i] = 2.5 * x1 - 1.3 * x2 + 0.7 * x3 + 4.2;
        }

        let model = tambear::train::linear::fit(&x, &y, n, d)?;
        println!("  True:      [2.5, -1.3, 0.7] + 4.2");
        println!("  Recovered: [{:.4}, {:.4}, {:.4}] + {:.4}",
            model.coefficients[0], model.coefficients[1], model.coefficients[2], model.intercept);
        println!("  R^2 = {:.10}", model.r_squared);
        println!("  RMSE = {:.2e}", model.rmse);

        assert!((model.coefficients[0] - 2.5).abs() < 1e-4);
        assert!((model.coefficients[1] + 1.3).abs() < 1e-4);
        assert!((model.coefficients[2] - 0.7).abs() < 1e-4);
        assert!((model.intercept - 4.2).abs() < 1e-4);
        println!("  EXACT RECOVERY confirmed.\n");
    }

    // -----------------------------------------------------------------------
    // Part 2: Real AAPL data — scatter features -> linear regression
    // -----------------------------------------------------------------------
    println!("--- Part 2: AAPL intraday price model ---");

    let data_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("data");
    let prices_f32 = load_f32(&data_dir.join("aapl_prices_f32.bin"));
    let timestamps = load_i64(&data_dir.join("aapl_timestamps_i64.bin"));
    let n_ticks = prices_f32.len();
    println!("  {} AAPL ticks loaded", n_ticks);

    // Compute minute bins
    let min_ts = timestamps.iter().cloned().min().unwrap_or(0);
    let ns_per_minute: i64 = 60_000_000_000;
    let keys: Vec<i32> = timestamps.iter()
        .map(|&ts| ((ts - min_ts) / ns_per_minute) as i32)
        .collect();
    let prices: Vec<f64> = prices_f32.iter().map(|&p| p as f64).collect();
    let n_groups = (*keys.iter().max().unwrap_or(&0) + 1) as usize;
    println!("  {} minute bins", n_groups);

    // -----------------------------------------------------------------------
    // Step 1: GPU scatter — per-minute statistics
    // -----------------------------------------------------------------------
    println!("\n  [Step 1] GPU scatter: per-minute statistics");
    let t0 = Instant::now();
    let engine = tambear::HashScatterEngine::new()?;
    let result = engine.groupby_warp(&keys, &prices, n_groups)?;
    let scatter_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let means = result.means();
    let _stds = result.stds();
    let counts = &result.counts;

    // Filter to active minutes (count > 0)
    let active: Vec<usize> = (0..n_groups).filter(|&g| counts[g] > 0.0).collect();
    let n_active = active.len();
    println!("  scatter_stats_warp: {:.1}ms, {} active minutes", scatter_ms, n_active);

    // -----------------------------------------------------------------------
    // Step 2: Build feature matrix from scatter statistics
    //
    // Features per active minute:
    //   x1: normalized minute (0.0 = market open, 1.0 = market close)
    //   x2: x1^2 (quadratic term for U-shape)
    //   x3: log(tick_count) — liquidity proxy
    //
    // Target: mean price for that minute
    // -----------------------------------------------------------------------
    println!("  [Step 2] Building features from scatter statistics");

    let d = 3;
    let n_train = n_active;
    let max_minute = *active.last().unwrap_or(&1) as f64;
    let mut x = vec![0.0f64; n_train * d];
    let mut y = vec![0.0f64; n_train];

    for (idx, &g) in active.iter().enumerate() {
        let t_norm = g as f64 / max_minute; // 0..1
        x[idx * d] = t_norm;
        x[idx * d + 1] = t_norm * t_norm;
        x[idx * d + 2] = (counts[g]).ln();
        y[idx] = means[g];
    }

    // -----------------------------------------------------------------------
    // Step 3: GPU linear regression
    // -----------------------------------------------------------------------
    println!("  [Step 3] train.linear.fit (GPU TiledEngine + CPU Cholesky)");
    let t0 = Instant::now();
    let model = tambear::train::linear::fit(&x, &y, n_train, d)?;
    let train_ms = t0.elapsed().as_secs_f64() * 1000.0;

    println!("\n  === Model ===");
    println!("  y = {:.4}*t + {:.4}*t^2 + {:.4}*log(n) + {:.4}",
        model.coefficients[0], model.coefficients[1], model.coefficients[2], model.intercept);
    println!("  R^2 = {:.6}", model.r_squared);
    println!("  RMSE = ${:.4}", model.rmse);
    println!("  Training time: {:.1}ms total", train_ms);

    // -----------------------------------------------------------------------
    // Interpret the model
    // -----------------------------------------------------------------------
    println!("\n  === Interpretation ===");
    let t_coeff = model.coefficients[0];
    let t2_coeff = model.coefficients[1];
    if t2_coeff.abs() > 0.01 {
        let vertex = -t_coeff / (2.0 * t2_coeff);
        let vertex_minute = vertex * max_minute;
        println!("  Quadratic vertex at t={:.2} (minute {:.0})", vertex, vertex_minute);
        if t2_coeff > 0.0 {
            println!("  U-shaped intraday curve (price dips mid-day)");
        } else {
            println!("  Inverted-U intraday curve (price peaks mid-day)");
        }
    }
    let liquidity_coeff = model.coefficients[2];
    println!("  Liquidity effect: {:.4}$/log(tick)", liquidity_coeff);
    if liquidity_coeff > 0.0 {
        println!("  Higher volume minutes -> higher prices");
    } else {
        println!("  Higher volume minutes -> lower prices");
    }

    // Show predictions for a few minutes
    println!("\n  === Sample Predictions ===");
    println!("  {:>8} {:>10} {:>10} {:>10} {:>8}", "minute", "actual", "predicted", "error", "ticks");
    let preds = model.predict(&x, n_train);
    for &sample_idx in &[0, n_active / 4, n_active / 2, 3 * n_active / 4, n_active - 1] {
        let g = active[sample_idx];
        println!("  {:>8} {:>10.2} {:>10.2} {:>10.4} {:>8}",
            g, y[sample_idx], preds[sample_idx],
            y[sample_idx] - preds[sample_idx],
            counts[g] as u32);
    }

    // -----------------------------------------------------------------------
    // Pipeline summary
    // -----------------------------------------------------------------------
    println!("\n  === Pipeline ===");
    println!("  scatter_stats_warp: {} ticks -> {} minute bins ({:.1}ms)",
        n_ticks, n_active, scatter_ms);
    println!("  train.linear.fit:   {} samples x {} features -> model ({:.1}ms)",
        n_train, d, train_ms);
    println!("  Total:              {:.1}ms end-to-end", scatter_ms + train_ms);

    println!("\nTam doesn't train. Tam accumulates.");
    Ok(())
}
