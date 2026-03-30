//! Observer benchmark: winrapids-scan GPU launch overhead and throughput.
//!
//! Measures:
//! 1. NVRTC compilation cost (first call)
//! 2. Cached launch overhead (repeat calls)
//! 3. AddOp vs CuPy cumsum reference (from Entry 001)
//! 4. WelfordOp single-scan vs 5-CuPy-op reference (from Entry 001)
//!
//! Standing methodology: 3 warmup, 20 timed, p50/p99/mean.

use std::time::Instant;
use winrapids_scan::{ScanEngine, AddOp, WelfordOp};

fn main() {
    println!("{}", "=".repeat(70));
    println!("Observer Benchmark: winrapids-scan GPU Performance");
    println!("{}", "=".repeat(70));

    let mut engine = ScanEngine::new().expect("Failed to create ScanEngine");
    println!("  ScanEngine initialized on GPU 0");

    bench_jit_cost(&mut engine);
    bench_addop_cached(&mut engine);
    bench_welford_cached(&mut engine);

    println!("\n{}", "=".repeat(70));
    println!("Benchmark complete.");
    println!("{}", "=".repeat(70));
}

fn bench_jit_cost(engine: &mut ScanEngine) {
    println!("\n--- JIT Compilation Cost (First Call) ---\n");

    // AddOp: first call includes NVRTC compile
    let input: Vec<f64> = (0..1000).map(|i| i as f64).collect();
    let t0 = Instant::now();
    let _ = engine.scan_inclusive(&AddOp, &input).unwrap();
    let first_add = t0.elapsed().as_micros();
    println!("  AddOp first call (JIT + launch): {} us", first_add);

    // WelfordOp: first call
    let t0 = Instant::now();
    let _ = engine.scan_inclusive(&WelfordOp, &input).unwrap();
    let first_welford = t0.elapsed().as_micros();
    println!("  WelfordOp first call (JIT + launch): {} us", first_welford);

    // Second calls (cached)
    let t0 = Instant::now();
    let _ = engine.scan_inclusive(&AddOp, &input).unwrap();
    let second_add = t0.elapsed().as_micros();
    println!("  AddOp second call (cached): {} us", second_add);

    let t0 = Instant::now();
    let _ = engine.scan_inclusive(&WelfordOp, &input).unwrap();
    let second_welford = t0.elapsed().as_micros();
    println!("  WelfordOp second call (cached): {} us", second_welford);
}

fn bench_addop_cached(engine: &mut ScanEngine) {
    println!("\n--- AddOp (Cumsum) Cached Performance ---\n");
    println!("  CuPy cumsum baseline (Entry 001): 26-80 us at FinTek sizes");
    println!("  Target: < 85 us at 10M (bandwidth-limited, similar algorithm)\n");

    for &n in &[1_000usize, 10_000, 50_000, 100_000, 500_000, 900_000] {
        let input: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.001).sin()).collect();

        // Warmup
        for _ in 0..3 {
            let _ = engine.scan_inclusive(&AddOp, &input).unwrap();
        }

        // Timed runs
        let mut times = Vec::with_capacity(20);
        for _ in 0..20 {
            let t0 = Instant::now();
            let _ = engine.scan_inclusive(&AddOp, &input).unwrap();
            times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = times[10];
        let p99 = times[19];
        let mean = times.iter().sum::<f64>() / times.len() as f64;

        println!("  n={:>9}: p50={:7.1} us  p99={:7.1} us  mean={:7.1} us",
            n, p50, p99, mean);
    }
}

fn bench_welford_cached(engine: &mut ScanEngine) {
    println!("\n--- WelfordOp (Mean+Variance) Cached Performance ---\n");
    println!("  CuPy 5-op Welford baseline (Entry 001): 117-140 us at FinTek sizes");
    println!("  Target: < 35 us (single fused scan kernel)\n");

    for &n in &[1_000usize, 10_000, 50_000, 100_000, 500_000, 900_000] {
        let input: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.001).sin() * 50.0).collect();

        // Warmup
        for _ in 0..3 {
            let _ = engine.scan_inclusive(&WelfordOp, &input).unwrap();
        }

        // Timed runs
        let mut times = Vec::with_capacity(20);
        for _ in 0..20 {
            let t0 = Instant::now();
            let result = engine.scan_inclusive(&WelfordOp, &input).unwrap();
            // Force use of result to prevent dead code elimination
            assert!(!result.primary.is_empty());
            times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p50 = times[10];
        let p99 = times[19];
        let mean = times.iter().sum::<f64>() / times.len() as f64;

        println!("  n={:>9}: p50={:7.1} us  p99={:7.1} us  mean={:7.1} us",
            n, p50, p99, mean);
    }
}
