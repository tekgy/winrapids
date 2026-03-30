//! Test binary for winrapids-scan.
//!
//! Part 1: Kernel generation validation (no GPU needed).
//! Part 2: GPU launch + correctness verification (requires CUDA device).

use winrapids_scan::ops::*;
use winrapids_scan::engine::{generate_scan_kernel, generate_multiblock_scan};
use winrapids_scan::launch::ScanEngine;

fn main() {
    println!("{}", "=".repeat(70));
    println!("winrapids-scan validation");
    println!("{}", "=".repeat(70));

    // ── Part 1: Kernel generation (no GPU) ────────────────────
    test_kernel_generation();
    test_multiblock_kernel_generation();
    test_operator_properties();

    // ── Part 2: GPU launch ────────────────────────────────────
    test_gpu_launch();
}

fn test_kernel_generation() {
    println!("\n--- Test 1: Single-block kernel generation ---");

    let add_kernel = generate_scan_kernel(&AddOp);
    assert!(add_kernel.contains("scan_inclusive"), "Should contain kernel name");
    assert!(add_kernel.contains("(a + b)"), "Should contain AddOp combine");
    println!("  AddOp kernel generated  ({} bytes)  PASS", add_kernel.len());

    let welford_kernel = generate_scan_kernel(&WelfordOp);
    assert!(welford_kernel.contains("WelfordState"), "Should contain state struct");
    println!("  WelfordOp kernel generated  ({} bytes)  PASS", welford_kernel.len());

    let ewm_kernel = generate_scan_kernel(&EWMOp { alpha: 0.1 });
    assert!(ewm_kernel.contains("EWMState"), "Should contain state struct");
    println!("  EWMOp kernel generated  ({} bytes)  PASS", ewm_kernel.len());

    // Cache key uniqueness
    let ewm1 = EWMOp { alpha: 0.1 };
    let ewm2 = EWMOp { alpha: 0.2 };
    assert_ne!(ewm1.params_key(), ewm2.params_key());
    println!("  Cache key uniqueness  PASS");
}

fn test_multiblock_kernel_generation() {
    println!("\n--- Test 2: Multi-block kernel generation ---");

    let add_source = generate_multiblock_scan(&AddOp);
    assert!(add_source.contains("scan_per_block"), "Should have phase 1 kernel");
    assert!(add_source.contains("scan_block_totals"), "Should have phase 2 kernel");
    assert!(add_source.contains("propagate_extract"), "Should have phase 3 kernel");
    println!("  AddOp multi-block: 3 kernels  ({} bytes)  PASS", add_source.len());

    let welford_source = generate_multiblock_scan(&WelfordOp);
    assert!(welford_source.contains("extract_secondary_0"), "Should have secondary extract");
    assert!(welford_source.contains("out1[gid]"), "Should write secondary output");
    println!("  WelfordOp multi-block: 3 kernels + secondary output  ({} bytes)  PASS",
        welford_source.len());
}

fn test_operator_properties() {
    println!("\n--- Test 3: Operator properties ---");

    let ops: Vec<(&str, Box<dyn AssociativeOp>)> = vec![
        ("AddOp", Box::new(AddOp)),
        ("MulOp", Box::new(MulOp)),
        ("MaxOp", Box::new(MaxOp)),
        ("MinOp", Box::new(MinOp)),
        ("WelfordOp", Box::new(WelfordOp)),
        ("EWMOp(0.1)", Box::new(EWMOp { alpha: 0.1 })),
    ];

    for (name, op) in &ops {
        let state_kind = if op.cuda_state_type().starts_with("struct") { "struct" } else { "scalar" };
        println!("  {:15} outputs={}  state={}  state_bytes={}",
            name, op.output_width(), state_kind, op.state_byte_size());
    }

    assert_eq!(AddOp.state_byte_size(), 8);
    assert_eq!(WelfordOp.state_byte_size(), 24);
    assert_eq!(EWMOp { alpha: 0.1 }.state_byte_size(), 24);
    println!("  State byte sizes  PASS");
}

fn test_gpu_launch() {
    println!("\n--- Test 4: GPU launch ---");

    let mut engine = match ScanEngine::new() {
        Ok(e) => {
            println!("  CudaDevice acquired  PASS");
            e
        }
        Err(e) => {
            println!("  No CUDA device available: {}. Skipping GPU tests.", e);
            return;
        }
    };

    // ── AddOp: cumulative sum ─────────────────────────────────
    println!("\n  --- AddOp (cumsum) ---");

    // Small test: exact match
    let small_input: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let expected: Vec<f64> = vec![1.0, 3.0, 6.0, 10.0, 15.0];
    let result = engine.scan_inclusive(&AddOp, &small_input)
        .expect("AddOp scan failed");
    let max_err: f64 = result.primary.iter().zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("  Small (n=5): max_err={:.2e}  {}", max_err,
        if max_err < 1e-10 { "PASS" } else { "FAIL" });
    assert!(max_err < 1e-10, "Small AddOp cumsum failed");

    // CPU reference cumsum for larger tests
    fn cpu_cumsum(data: &[f64]) -> Vec<f64> {
        let mut result = Vec::with_capacity(data.len());
        let mut acc = 0.0;
        for &x in data {
            acc += x;
            result.push(acc);
        }
        result
    }

    // Medium test: multi-block
    let n = 10_000;
    let medium_input: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let expected = cpu_cumsum(&medium_input);
    let result = engine.scan_inclusive(&AddOp, &medium_input)
        .expect("AddOp medium scan failed");
    let max_err: f64 = result.primary.iter().zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let rel_err = max_err / expected.last().unwrap();
    println!("  Medium (n={}): max_err={:.2e}  rel_err={:.2e}  {}",
        n, max_err, rel_err,
        if rel_err < 1e-10 { "PASS" } else { "FAIL" });
    assert!(rel_err < 1e-10, "Medium AddOp cumsum failed");

    // FinTek sizes
    for &n in &[50_000usize, 100_000, 500_000, 900_000] {
        let input: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.001).sin()).collect();
        let expected = cpu_cumsum(&input);
        let t0 = std::time::Instant::now();
        let result = engine.scan_inclusive(&AddOp, &input)
            .expect("AddOp large scan failed");
        let elapsed_us = t0.elapsed().as_micros();
        let max_err: f64 = result.primary.iter().zip(&expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f64, f64::max);
        println!("  n={:>9}: max_err={:.2e}  time={}us  {}",
            n, max_err, elapsed_us,
            if max_err < 1e-6 { "PASS" } else { "FAIL" });
        assert!(max_err < 1e-6, "Large AddOp cumsum failed at n={}", n);
    }

    // ── WelfordOp: running mean + variance ────────────────────
    println!("\n  --- WelfordOp (mean + variance) ---");

    fn cpu_welford(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut means = Vec::with_capacity(data.len());
        let mut vars = Vec::with_capacity(data.len());
        let mut count = 0i64;
        let mut mean = 0.0;
        let mut m2 = 0.0;
        for &x in data {
            count += 1;
            let delta = x - mean;
            mean += delta / count as f64;
            let delta2 = x - mean;
            m2 += delta * delta2;
            means.push(mean);
            vars.push(if count > 1 { m2 / (count - 1) as f64 } else { 0.0 });
        }
        (means, vars)
    }

    let n = 10_000;
    let input: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.01).sin() * 100.0).collect();
    let (expected_mean, expected_var) = cpu_welford(&input);

    let result = engine.scan_inclusive(&WelfordOp, &input)
        .expect("WelfordOp scan failed");
    assert!(!result.secondary.is_empty(), "WelfordOp should produce secondary output");

    let max_mean_err: f64 = result.primary.iter().zip(&expected_mean)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let max_var_err: f64 = result.secondary[0].iter().zip(&expected_var)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    println!("  n={}: mean_err={:.2e}  var_err={:.2e}  {}",
        n, max_mean_err, max_var_err,
        if max_mean_err < 1e-6 && max_var_err < 1e-3 { "PASS" } else { "FAIL" });

    // FinTek size
    let n = 100_000;
    let input: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.001).sin() * 50.0).collect();
    let (expected_mean, expected_var) = cpu_welford(&input);
    let t0 = std::time::Instant::now();
    let result = engine.scan_inclusive(&WelfordOp, &input)
        .expect("WelfordOp large scan failed");
    let elapsed_us = t0.elapsed().as_micros();
    let max_mean_err: f64 = result.primary.iter().zip(&expected_mean)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let max_var_err: f64 = result.secondary[0].iter().zip(&expected_var)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("  n={}: mean_err={:.2e}  var_err={:.2e}  time={}us  {}",
        n, max_mean_err, max_var_err, elapsed_us,
        if max_mean_err < 1e-4 && max_var_err < 1e-1 { "PASS" } else { "FAIL" });

    println!("\n{}", "=".repeat(70));
    println!("GPU LAUNCH TESTS COMPLETE");
    println!("{}", "=".repeat(70));
}
