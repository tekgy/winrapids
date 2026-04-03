//! NVRTC compilation budget measurement.
//!
//! Measures: how much time does kernel compilation take vs actual GPU compute?
//! At what problem size does compilation become negligible?
//!
//! This data informs Tam's compilation strategy: when to JIT, when to fall back
//! to pre-compiled generic kernels, and whether a cross-engine kernel cache matters.
//!
//! Run with: cargo run --bin compile-budget --release

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== NVRTC Compilation Budget ===\n");

    // -----------------------------------------------------------------------
    // Measure engine construction time (= kernel compilation)
    // -----------------------------------------------------------------------
    println!("--- Engine Construction (kernel compilation) ---\n");

    // HashScatterEngine: 6 kernels (naive, smem, warp variants)
    let t0 = Instant::now();
    let scatter_engine = tambear::HashScatterEngine::new()?;
    let scatter_compile = t0.elapsed();
    println!("  HashScatterEngine: {:>8.1}ms  (scatter_stats * 3 variants + support)", scatter_compile.as_secs_f64() * 1000.0);

    // ScatterJit: lazy compilation, but let's force one
    let t0 = Instant::now();
    let mut scatter_jit = tambear::ScatterJit::new()?;
    let scatter_jit_init = t0.elapsed();
    println!("  ScatterJit (init): {:>8.1}ms  (context only, kernels lazy)", scatter_jit_init.as_secs_f64() * 1000.0);

    // Force compilation of a phi kernel
    let keys_small = vec![0i32; 10];
    let vals_small = vec![1.0f64; 10];
    let t0 = Instant::now();
    let _ = scatter_jit.scatter_phi(tambear::PHI_SUM, &keys_small, &vals_small, None, 1)?;
    let scatter_jit_first = t0.elapsed();
    println!("  ScatterJit (1st):  {:>8.1}ms  (PHI_SUM compile + 10-element compute)", scatter_jit_first.as_secs_f64() * 1000.0);

    // Cached call
    let t0 = Instant::now();
    let _ = scatter_jit.scatter_phi(tambear::PHI_SUM, &keys_small, &vals_small, None, 1)?;
    let scatter_jit_cached = t0.elapsed();
    println!("  ScatterJit (hit):  {:>8.1}ms  (cache hit + 10-element compute)", scatter_jit_cached.as_secs_f64() * 1000.0);

    // FilterJit
    let t0 = Instant::now();
    let mut filter_jit = tambear::FilterJit::new()?;
    let filter_init = t0.elapsed();
    println!("  FilterJit (init):  {:>8.1}ms  (context only, kernels lazy)", filter_init.as_secs_f64() * 1000.0);

    let vals_f64 = vec![1.0f64; 10];
    let t0 = Instant::now();
    let _ = filter_jit.filter_mask("v > 0.5", &vals_f64)?;
    let filter_first = t0.elapsed();
    println!("  FilterJit (1st):   {:>8.1}ms  (\"v > 0.5\" compile + 10-element filter)", filter_first.as_secs_f64() * 1000.0);

    let t0 = Instant::now();
    let _ = filter_jit.filter_mask("v > 0.5", &vals_f64)?;
    let filter_cached = t0.elapsed();
    println!("  FilterJit (hit):   {:>8.1}ms  (cache hit + 10-element filter)", filter_cached.as_secs_f64() * 1000.0);

    // KMeansEngine: 3 kernels (assign, update, normalize)
    let t0 = Instant::now();
    let _kmeans_engine = tambear::kmeans::KMeansEngine::new()?;
    let kmeans_compile = t0.elapsed();
    println!("  KMeansEngine:      {:>8.1}ms  (3 kernels: assign + update + normalize)", kmeans_compile.as_secs_f64() * 1000.0);

    // ClusteringEngine (TiledEngine): lazy, compiles on first op
    let t0 = Instant::now();
    let mut cluster_engine = tambear::ClusteringEngine::new()?;
    let cluster_init = t0.elapsed();
    println!("  ClusteringEngine:  {:>8.1}ms  (TiledEngine init, kernels lazy)", cluster_init.as_secs_f64() * 1000.0);

    // GatherOp
    let t0 = Instant::now();
    let _gather = tambear::GatherOp::new()?;
    let gather_compile = t0.elapsed();
    println!("  GatherOp:          {:>8.1}ms  (gather + scatter_back kernels)", gather_compile.as_secs_f64() * 1000.0);

    println!("\n--- Compilation vs Compute: Amortization ---\n");
    println!("  {:>12} {:>12} {:>12} {:>12} {:>8}", "n", "compile", "compute", "total", "ratio");
    println!("  {:>12} {:>12} {:>12} {:>12} {:>8}", "", "(ms)", "(ms)", "(ms)", "c/t");

    // Test scatter at different sizes
    let n_groups = 100;
    for &n in &[100, 1_000, 10_000, 100_000, 1_000_000] {
        let keys: Vec<i32> = (0..n).map(|i| (i % n_groups) as i32).collect();
        let vals: Vec<f64> = (0..n).map(|i| i as f64 * 0.001).collect();

        // Fresh engine = fresh compilation
        let t0 = Instant::now();
        let fresh_engine = tambear::HashScatterEngine::new()?;
        let compile = t0.elapsed();

        // Warm compute (use cached engine to avoid double-counting)
        let t0 = Instant::now();
        let _ = scatter_engine.groupby_warp(&keys, &vals, n_groups as usize)?;
        let compute = t0.elapsed();

        let compile_ms = compile.as_secs_f64() * 1000.0;
        let compute_ms = compute.as_secs_f64() * 1000.0;
        let total_ms = compile_ms + compute_ms;
        let ratio = compute_ms / total_ms;

        println!("  {:>12} {:>12.2} {:>12.3} {:>12.2} {:>7.1}%",
            format!("{}",n), compile_ms, compute_ms, total_ms, ratio * 100.0);

        drop(fresh_engine);
    }

    // -----------------------------------------------------------------------
    // Multi-phi fusion: single JIT compilation for 3 outputs vs 3 separate
    // -----------------------------------------------------------------------
    println!("\n--- JIT Fusion: 3 separate compiles vs 1 fused compile ---\n");

    let n = 100_000;
    let keys: Vec<i32> = (0..n).map(|i| (i % 100) as i32).collect();
    let vals: Vec<f64> = (0..n).map(|i| (i as f64 * 0.001).sin()).collect();

    // 3 separate phi compilations
    let mut jit_sep = tambear::ScatterJit::new()?;
    let t0 = Instant::now();
    let _ = jit_sep.scatter_phi(tambear::PHI_SUM, &keys, &vals, None, 100)?;
    let _ = jit_sep.scatter_phi(tambear::PHI_SUM_SQ, &keys, &vals, None, 100)?;
    let _ = jit_sep.scatter_phi(tambear::PHI_COUNT, &keys, &vals, None, 100)?;
    let sep_time = t0.elapsed();

    // 1 fused multi-phi compilation
    let mut jit_fused = tambear::ScatterJit::new()?;
    let t0 = Instant::now();
    let _ = jit_fused.scatter_multi_phi(
        &[tambear::PHI_SUM, tambear::PHI_SUM_SQ, tambear::PHI_COUNT],
        &keys, &vals, None, 100,
    )?;
    let fused_time = t0.elapsed();

    println!("  3 separate scatter_phi:  {:>8.1}ms  (3 compilations + 3 kernel launches)", sep_time.as_secs_f64() * 1000.0);
    println!("  1 fused scatter_multi:   {:>8.1}ms  (1 compilation + 1 kernel launch)", fused_time.as_secs_f64() * 1000.0);
    println!("  Fusion speedup:          {:>8.1}x", sep_time.as_secs_f64() / fused_time.as_secs_f64());

    // Cached calls (compilation amortized)
    let t0 = Instant::now();
    for _ in 0..10 {
        let _ = jit_sep.scatter_phi(tambear::PHI_SUM, &keys, &vals, None, 100)?;
        let _ = jit_sep.scatter_phi(tambear::PHI_SUM_SQ, &keys, &vals, None, 100)?;
        let _ = jit_sep.scatter_phi(tambear::PHI_COUNT, &keys, &vals, None, 100)?;
    }
    let sep_cached = t0.elapsed();

    let t0 = Instant::now();
    for _ in 0..10 {
        let _ = jit_fused.scatter_multi_phi(
            &[tambear::PHI_SUM, tambear::PHI_SUM_SQ, tambear::PHI_COUNT],
            &keys, &vals, None, 100,
        )?;
    }
    let fused_cached = t0.elapsed();

    println!("\n  3 separate (cached, 10x): {:>8.1}ms  ({:.2}ms/iter)", sep_cached.as_secs_f64() * 1000.0, sep_cached.as_secs_f64() * 100.0);
    println!("  1 fused (cached, 10x):    {:>8.1}ms  ({:.2}ms/iter)", fused_cached.as_secs_f64() * 1000.0, fused_cached.as_secs_f64() * 100.0);
    println!("  Compute-only speedup:     {:>8.1}x", sep_cached.as_secs_f64() / fused_cached.as_secs_f64());

    // -----------------------------------------------------------------------
    // Session overhead measurement
    // -----------------------------------------------------------------------
    println!("\n--- Session Overhead ---\n");

    let data = vec![0.0f64, 0.1, 0.2, 5.0, 5.1, 4.9, 0.0, 0.1, 0.2, 5.0, 5.1, 4.9];
    let n = 6;
    let d = 2;

    // Cold: session miss → compute + register
    let mut session = tambear::TamSession::new();
    let t0 = Instant::now();
    let _ = cluster_engine.dbscan_session(&mut session, &data, n, d, 0.5, 2)?;
    let cold = t0.elapsed();

    // Warm: session hit → skip GPU entirely
    let t0 = Instant::now();
    let _ = cluster_engine.dbscan_session(&mut session, &data, n, d, 1.0, 2)?;
    let warm = t0.elapsed();

    println!("  DBSCAN (cold, 6 pts): {:>8.3}ms  (compile + distance + DBSCAN)", cold.as_secs_f64() * 1000.0);
    println!("  DBSCAN (warm, 6 pts): {:>8.3}ms  (session hit + DBSCAN only)", warm.as_secs_f64() * 1000.0);
    println!("  Session speedup:      {:>8.0}x", cold.as_secs_f64() / warm.as_secs_f64());

    // KNN cross-algorithm
    let t0 = Instant::now();
    let _ = tambear::knn::knn_session(&mut session, &data, n, d, 2)?;
    let knn_warm = t0.elapsed();
    println!("  KNN (session hit):    {:>8.3}ms  (pure CPU, distance from session)", knn_warm.as_secs_f64() * 1000.0);

    println!("\n=== Summary ===\n");
    println!("  NVRTC compilation: ~{:.0}ms per engine", scatter_compile.as_secs_f64() * 1000.0);
    println!("  JIT compilation:   ~{:.0}ms per phi expression", scatter_jit_first.as_secs_f64() * 1000.0);
    println!("  Cache hit:         ~{:.3}ms (negligible)", scatter_jit_cached.as_secs_f64() * 1000.0);
    println!("  Session hit:       ~{:.3}ms (skip GPU entirely)", warm.as_secs_f64() * 1000.0);
    println!();
    println!("  Implication: for N < ~1K elements, compilation dominates.");
    println!("  The compiler should maintain a cross-engine kernel cache");
    println!("  and consider CPU fallback for tiny problems.");

    Ok(())
}
