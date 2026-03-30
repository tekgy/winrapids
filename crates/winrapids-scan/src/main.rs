//! Test binary for winrapids-scan.
//!
//! Part 1: Kernel generation validation (no GPU needed).
//! Part 2: GPU launch + correctness verification (requires CUDA device).

use winrapids_scan::ops::*;
use winrapids_scan::{KalmanAffineOp, SarkkaOp};
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

    let ewm_kernel = generate_scan_kernel(&EWMOp::new(0.1));
    assert!(ewm_kernel.contains("EWMState"), "Should contain state struct");
    println!("  EWMOp kernel generated  ({} bytes)  PASS", ewm_kernel.len());

    // Cache key uniqueness
    let ewm1 = EWMOp::new(0.1);
    let ewm2 = EWMOp::new(0.2);
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
        ("EWMOp(0.1)", Box::new(EWMOp::new(0.1))),
    ];

    for (name, op) in &ops {
        let state_kind = if op.cuda_state_type().starts_with("struct") { "struct" } else { "scalar" };
        println!("  {:15} outputs={}  state={}  state_bytes={}",
            name, op.output_width(), state_kind, op.state_byte_size());
    }

    assert_eq!(AddOp.state_byte_size(), 8);
    assert_eq!(WelfordOp.state_byte_size(), 24);
    assert_eq!(EWMOp::new(0.1).state_byte_size(), 24);
    assert_eq!(KalmanOp { f: 1.0, h: 1.0, q: 0.01, r: 0.1 }.state_byte_size(), 32); // 3×f64 + i32 + 4B pad
    assert_eq!(KalmanAffineOp::new(0.98, 1.0, 0.01, 0.1).state_byte_size(), 16);
    assert_eq!(SarkkaOp::new(0.98, 1.0, 0.01, 0.1).state_byte_size(), 40);
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

    // ── KalmanAffineOp: steady-state Kalman filter ──────────────
    test_kalman_affine(&mut engine);

    // ── SarkkaOp: full Särkkä 5-tuple Kalman ─────────────────
    test_sarkka(&mut engine);

    println!("\n{}", "=".repeat(70));
    println!("GPU LAUNCH TESTS COMPLETE");
    println!("{}", "=".repeat(70));
}

fn test_kalman_affine(engine: &mut ScanEngine) {
    println!("\n  --- KalmanAffineOp (exact steady-state Kalman) ---");

    // Sequential reference: x[t] = A * x[t-1] + K_ss * z[t], x[-1] = 0
    fn cpu_kalman_affine(data: &[f64], a: f64, k_ss: f64) -> Vec<f64> {
        let mut result = Vec::with_capacity(data.len());
        let mut x = 0.0;
        for &z in data {
            x = a * x + k_ss * z;
            result.push(x);
        }
        result
    }

    // Test parameters: F=0.98, H=1.0, Q=0.01, R=0.1
    let op = KalmanAffineOp::new(0.98, 1.0, 0.01, 0.1);
    println!("  F=0.98, H=1.0, Q=0.01, R=0.1 -> K_ss={:.6}, A={:.6}", op.k_ss, op.a);

    // Verify Riccati convergence: A = (1 - K_ss * H) * F
    let a_check = (1.0 - op.k_ss * op.h) * op.f;
    assert!((op.a - a_check).abs() < 1e-15, "A derivation mismatch");

    // Small test: exact match against sequential
    let n = 100;
    let input: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.1).sin() * 10.0).collect();
    let expected = cpu_kalman_affine(&input, op.a, op.k_ss);

    let result = engine.scan_inclusive(&op, &input)
        .expect("KalmanAffineOp scan failed");

    let max_err: f64 = result.primary.iter().zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("  Small (n={}): max_err={:.2e}  {}", n, max_err,
        if max_err < 1e-10 { "PASS" } else { "FAIL" });
    assert!(max_err < 1e-10, "Small KalmanAffineOp failed: max_err={}", max_err);

    // Multi-block test
    let n = 10_000;
    let input: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.01).sin() * 50.0).collect();
    let expected = cpu_kalman_affine(&input, op.a, op.k_ss);
    let result = engine.scan_inclusive(&op, &input)
        .expect("KalmanAffineOp medium scan failed");
    let max_err: f64 = result.primary.iter().zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("  Medium (n={}): max_err={:.2e}  {}", n, max_err,
        if max_err < 1e-8 { "PASS" } else { "FAIL" });
    assert!(max_err < 1e-8, "Medium KalmanAffineOp failed: max_err={}", max_err);

    // FinTek size
    let n = 100_000;
    let input: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.001).sin() * 100.0).collect();
    let expected = cpu_kalman_affine(&input, op.a, op.k_ss);
    let t0 = std::time::Instant::now();
    let result = engine.scan_inclusive(&op, &input)
        .expect("KalmanAffineOp large scan failed");
    let elapsed_us = t0.elapsed().as_micros();
    let max_err: f64 = result.primary.iter().zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("  Large (n={}): max_err={:.2e}  time={}us  {}",
        n, max_err, elapsed_us,
        if max_err < 1e-6 { "PASS" } else { "FAIL" });
    assert!(max_err < 1e-6, "Large KalmanAffineOp failed: max_err={}", max_err);

    // Verify state properties
    assert_eq!(op.state_byte_size(), 16, "State should be 16 bytes (2 x f64)");
    println!("  State: 16 bytes (2x f64), combine: 2 muls + 1 add  PASS");
}

fn test_sarkka(engine: &mut ScanEngine) {
    println!("\n  --- SarkkaOp (Särkkä 5-tuple, exact from step 1) ---");

    // Sequential Kalman filter reference — the gold standard.
    // This runs the full predict-update cycle, NOT steady-state.
    // SarkkaOp must match this from step 1, not just after convergence.
    fn cpu_kalman_sequential(data: &[f64], f: f64, h: f64, q: f64, r: f64) -> (Vec<f64>, Vec<f64>) {
        let mut xs = Vec::with_capacity(data.len());
        let mut ps = Vec::with_capacity(data.len());
        let mut x = 0.0_f64; // initial state
        let mut p = 0.0_f64; // initial covariance (certain at zero)
        for &z in data {
            // Predict
            let x_pred = f * x;
            let p_pred = f * p * f + q;
            // Update
            let s = h * p_pred * h + r; // innovation covariance
            let k = p_pred * h / s;     // Kalman gain
            x = x_pred + k * (z - h * x_pred);
            p = (1.0 - k * h) * p_pred;
            xs.push(x);
            ps.push(p);
        }
        (xs, ps)
    }

    let f = 0.98;
    let h = 1.0;
    let q = 0.01;
    let r = 0.1;
    let op = SarkkaOp::new(f, h, q, r);
    println!("  F={}, H={}, Q={}, R={}", f, h, q, r);

    // Small test: exact match from step 1
    let n = 100;
    let input: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.1).sin() * 10.0).collect();
    let (expected_x, expected_p) = cpu_kalman_sequential(&input, f, h, q, r);

    let result = engine.scan_inclusive(&op, &input)
        .expect("SarkkaOp scan failed");
    assert!(!result.secondary.is_empty(), "SarkkaOp should produce secondary output (P)");

    let max_x_err: f64 = result.primary.iter().zip(&expected_x)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let max_p_err: f64 = result.secondary[0].iter().zip(&expected_p)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);

    println!("  Small (n={}): x_err={:.2e}  p_err={:.2e}  {}",
        n, max_x_err, max_p_err,
        if max_x_err < 1e-10 && max_p_err < 1e-10 { "PASS" } else { "FAIL" });

    // Print first few values for debugging
    println!("\n  Step-by-step comparison (first 10):");
    for i in 0..10.min(n) {
        println!("    [{}] gpu_x={:.10e}  cpu_x={:.10e}  x_diff={:.2e}  gpu_p={:.6e}  cpu_p={:.6e}  p_diff={:.2e}",
            i, result.primary[i], expected_x[i],
            (result.primary[i] - expected_x[i]).abs(),
            result.secondary[0][i], expected_p[i],
            (result.secondary[0][i] - expected_p[i]).abs());
    }

    assert!(max_x_err < 1e-8, "Small SarkkaOp x failed: max_err={}", max_x_err);
    assert!(max_p_err < 1e-8, "Small SarkkaOp P failed: max_err={}", max_p_err);

    // Multi-block test
    let n = 10_000;
    let input: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.01).sin() * 50.0).collect();
    let (expected_x, expected_p) = cpu_kalman_sequential(&input, f, h, q, r);
    let result = engine.scan_inclusive(&op, &input)
        .expect("SarkkaOp medium scan failed");
    let max_x_err: f64 = result.primary.iter().zip(&expected_x)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    let max_p_err: f64 = result.secondary[0].iter().zip(&expected_p)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("\n  Medium (n={}): x_err={:.2e}  p_err={:.2e}  {}",
        n, max_x_err, max_p_err,
        if max_x_err < 1e-6 && max_p_err < 1e-6 { "PASS" } else { "FAIL" });

    // Benchmark at 100K
    let n = 100_000;
    let input: Vec<f64> = (0..n).map(|i| ((i as f64) * 0.001).sin() * 100.0).collect();
    let (expected_x, _expected_p) = cpu_kalman_sequential(&input, f, h, q, r);
    let t0 = std::time::Instant::now();
    let result = engine.scan_inclusive(&op, &input)
        .expect("SarkkaOp large scan failed");
    let elapsed_us = t0.elapsed().as_micros();
    let max_x_err: f64 = result.primary.iter().zip(&expected_x)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f64, f64::max);
    println!("  Large (n={}): x_err={:.2e}  time={}us  {}",
        n, max_x_err, elapsed_us,
        if max_x_err < 1e-4 { "PASS" } else { "FAIL" });

    // Verify state properties
    assert_eq!(op.state_byte_size(), 40, "State should be 40 bytes (5 x f64)");
    println!("  State: 40 bytes (5x f64), combine: 1 division + muls/adds  PASS");
}
