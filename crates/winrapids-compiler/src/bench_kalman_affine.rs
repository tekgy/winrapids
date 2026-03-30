//! Observer benchmark: KalmanAffineOp verification and operator family gradient.
//!
//! Measurements:
//! 1. Correctness: KalmanAffineOp vs sequential reference at multiple sizes
//! 2. Dispatch time: KalmanAffineOp vs AddOp vs WelfordOp vs KalmanOp
//! 3. State complexity gradient: does state_byte_size correlate with dispatch cost?
//!
//! Standing methodology: 3 warmup, 20 timed, p50/p99/mean.

use std::time::Instant;

use winrapids_scan::{ScanEngine, AddOp, WelfordOp, DevicePtr};
use winrapids_scan::ops::{AssociativeOp, KalmanOp, KalmanAffineOp};

fn main() {
    println!("{}", "=".repeat(70));
    println!("Observer Benchmark: KalmanAffineOp + Operator Family Gradient");
    println!("{}", "=".repeat(70));

    verify_kalman_affine_correctness();
    bench_operator_family_gradient();
    bench_kalman_affine_scaling();
    compare_kalman_vs_kalman_affine();

    println!("\n{}", "=".repeat(70));
    println!("Benchmark complete.");
    println!("{}", "=".repeat(70));
}

/// Sequential Kalman filter reference: x[t] = A*x[t-1] + K_ss*z[t]
fn sequential_kalman_affine(data: &[f64], a: f64, k_ss: f64) -> Vec<f64> {
    let mut result = Vec::with_capacity(data.len());
    let mut x = 0.0f64;
    for &z in data {
        x = a * x + k_ss * z;
        result.push(x);
    }
    result
}

fn verify_kalman_affine_correctness() {
    println!("\n--- KalmanAffineOp Correctness vs Sequential Reference ---\n");

    let mut engine = ScanEngine::new().unwrap();

    // F=0.98, H=1.0, Q=0.01, R=0.1 — pathmaker's test params
    let op = KalmanAffineOp::new(0.98, 1.0, 0.01, 0.1);
    println!("  Parameters: F=0.98, H=1.0, Q=0.01, R=0.1");
    println!("  Riccati converged: K_ss={:.6}, A={:.6}", op.k_ss, op.a);

    for &n in &[100, 1_000, 10_000, 100_000] {
        // Deterministic test data: sine wave + noise-like pattern
        let data: Vec<f64> = (0..n).map(|i| {
            let x = i as f64;
            10.0 + (x * 0.05).sin() * 3.0 + (x * 0.13).cos() * 1.5
        }).collect();

        // Reference
        let reference = sequential_kalman_affine(&data, op.a, op.k_ss);

        // GPU
        let result = engine.scan_inclusive(&op, &data).unwrap();

        assert_eq!(result.primary.len(), reference.len(), "Length mismatch at n={}", n);

        let mut max_err: f64 = 0.0;
        let mut max_rel_err: f64 = 0.0;
        for i in 0..n {
            let err = (result.primary[i] - reference[i]).abs();
            if err > max_err { max_err = err; }
            if reference[i].abs() > 1e-10 {
                let rel = err / reference[i].abs();
                if rel > max_rel_err { max_rel_err = rel; }
            }
        }

        let status = if max_err < 1e-10 { "PASS" } else { "WARN" };
        println!("  n={:>7}: max_err={:.2e}  max_rel_err={:.2e}  [{}]",
            n, max_err, max_rel_err, status);
    }
}

fn bench_operator_family_gradient() {
    println!("\n--- Operator Family: State Complexity → Dispatch Time ---\n");
    println!("  Testing naturalist's prediction: does state_byte_size");
    println!("  correlate with dispatch cost, or does launch floor dominate?\n");

    let mut engine = ScanEngine::new().unwrap();
    let stream = engine.stream().clone();

    let n = 100_000usize;
    let host_data: Vec<f64> = (1..=n as u64).map(|x| x as f64 * 0.01).collect();
    let input_dev = stream.clone_htod(&host_data).unwrap();
    let input_ptr = { let (p, _g) = input_dev.device_ptr(&stream); p };

    // Operators in order of state complexity.
    // KalmanOp sizeof bug fixed (28→32), now passes sizeof validation.
    let operators: Vec<(Box<dyn AssociativeOp>, &str)> = vec![
        (Box::new(AddOp), "AddOp"),
        (Box::new(KalmanAffineOp::new(0.98, 1.0, 0.01, 0.1)), "KalmanAffineOp"),
        (Box::new(WelfordOp), "WelfordOp"),
        (Box::new(KalmanOp { f: 0.98, h: 1.0, q: 0.01, r: 0.1 }), "KalmanOp"),
    ];

    println!("  {:20} {:>10} {:>10} {:>10} {:>10}", "Operator", "state_B", "p01 us", "p50 us", "p99 us");
    println!("  {}", "-".repeat(64));

    for (op, name) in &operators {
        // Warmup
        for _ in 0..3 {
            let _ = unsafe { engine.scan_device_ptr(op.as_ref(), input_ptr, n) };
        }

        // Timed
        let mut times = Vec::with_capacity(20);
        for _ in 0..20 {
            let t0 = Instant::now();
            let _ = unsafe { engine.scan_device_ptr(op.as_ref(), input_ptr, n) };
            times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        println!("  {:20} {:>10} {:>10.1} {:>10.1} {:>10.1}",
            name, op.state_byte_size(), times[0], times[10], times[19]);
    }

    println!("\n  n={} elements (device-to-device, cached kernel)", n);
}

fn bench_kalman_affine_scaling() {
    println!("\n--- KalmanAffineOp Scaling (device-to-device) ---\n");

    let mut engine = ScanEngine::new().unwrap();
    let stream = engine.stream().clone();
    let op = KalmanAffineOp::new(0.98, 1.0, 0.01, 0.1);

    for &n in &[1_000, 10_000, 100_000, 500_000, 1_000_000] {
        let host_data: Vec<f64> = (1..=n as u64).map(|x| x as f64 * 0.01).collect();
        let input_dev = stream.clone_htod(&host_data).unwrap();
        let input_ptr = { let (p, _g) = input_dev.device_ptr(&stream); p };

        // Warmup
        for _ in 0..3 {
            let _ = unsafe { engine.scan_device_ptr(&op, input_ptr, n) };
        }

        let mut times = Vec::with_capacity(20);
        for _ in 0..20 {
            let t0 = Instant::now();
            let _ = unsafe { engine.scan_device_ptr(&op, input_ptr, n) };
            times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());

        println!("  n={:>9}: p01={:7.1} us  p50={:7.1} us  p99={:7.1} us",
            n, times[0], times[10], times[19]);
    }
}

fn compare_kalman_vs_kalman_affine() {
    println!("\n--- KalmanOp vs KalmanAffineOp: Accuracy + Speed ---\n");

    let mut engine = ScanEngine::new().unwrap();
    let stream = engine.stream().clone();

    // KalmanOp sizeof bug fixed (28→32). Now both dispatch on GPU.
    let _kalman = KalmanOp { f: 0.98, h: 1.0, q: 0.01, r: 0.1 };
    let affine = KalmanAffineOp::new(0.98, 1.0, 0.01, 0.1);

    let n = 10_000usize;
    let data: Vec<f64> = (0..n).map(|i| {
        10.0 + (i as f64 * 0.05).sin() * 3.0
    }).collect();

    // Sequential reference (exact)
    let reference = sequential_kalman_affine(&data, affine.a, affine.k_ss);

    // KalmanAffineOp result (GPU via scan_inclusive)
    let affine_result = engine.scan_inclusive(&affine, &data).unwrap();

    let mut affine_max_err: f64 = 0.0;
    for i in 0..n {
        let a_err = (affine_result.primary[i] - reference[i]).abs();
        if a_err > affine_max_err { affine_max_err = a_err; }
    }

    println!("  n={}, sequential reference: x[t] = A*x[t-1] + K_ss*z[t]", n);
    println!("  KalmanAffineOp max error: {:.2e}  (affine composition)", affine_max_err);

    // Speed: both operators device-to-device
    let n_bench = 100_000;
    let bench_data: Vec<f64> = (1..=n_bench as u64).map(|x| x as f64 * 0.01).collect();
    let dev_data = stream.clone_htod(&bench_data).unwrap();
    let dev_ptr = { let (p, _g) = dev_data.device_ptr(&stream); p };

    let ops: Vec<(&dyn AssociativeOp, &str)> = vec![
        (&_kalman, "KalmanOp"),
        (&affine, "KalmanAffineOp"),
    ];
    for (op, name) in ops {
        for _ in 0..3 {
            let _ = unsafe { engine.scan_device_ptr(op, dev_ptr, n_bench) };
        }
        let mut times = Vec::with_capacity(20);
        for _ in 0..20 {
            let t0 = Instant::now();
            let _ = unsafe { engine.scan_device_ptr(op, dev_ptr, n_bench) };
            times.push(t0.elapsed().as_nanos() as f64 / 1000.0);
        }
        times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!("  {} dispatch p50: {:.1} us (state={}B, n={})",
            name, times[10], op.state_byte_size(), n_bench);
    }
}
