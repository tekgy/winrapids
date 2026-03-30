//! Observer verification: scan_device_ptr produces identical results to scan_inclusive.
//!
//! scan_inclusive: host→GPU→host (H2D + scan + D2H)
//! scan_device_ptr: GPU→GPU (scan only, zero transfer)
//!
//! Both must produce bitwise-identical f64 output.

use winrapids_scan::{ScanEngine, AddOp, WelfordOp, DevicePtr};

fn main() {
    println!("{}", "=".repeat(70));
    println!("Observer Verification: scan_device_ptr vs scan_inclusive");
    println!("{}", "=".repeat(70));

    verify_addop();
    verify_welford();
    verify_at_block_boundary();

    println!("\n{}", "=".repeat(70));
    println!("ALL VERIFICATION PASSED");
    println!("{}", "=".repeat(70));
}

fn verify_addop() {
    println!("\n--- AddOp: cumulative sum ---\n");

    let mut engine = ScanEngine::new().unwrap();

    for &n in &[100, 1024, 1025, 10_000, 100_000] {
        let data: Vec<f64> = (1..=n as u64).map(|x| x as f64).collect();

        // Reference: scan_inclusive (host path)
        let ref_result = engine.scan_inclusive(&AddOp, &data).unwrap();

        // Test: scan_device_ptr (device path)
        let stream = engine.stream().clone();
        let dev_input = stream.clone_htod(&data).unwrap();
        let dev_ptr = { let (p, _g) = dev_input.device_ptr(&stream); p };

        let dev_output = unsafe {
            engine.scan_device_ptr(&AddOp, dev_ptr, n).unwrap()
        };
        let test_result = dev_output.primary_to_host().unwrap();

        // Bitwise comparison
        assert_eq!(ref_result.primary.len(), test_result.len(),
            "Length mismatch at n={}", n);

        let mut max_diff: f64 = 0.0;
        for i in 0..n {
            let diff = (ref_result.primary[i] - test_result[i]).abs();
            if diff > max_diff { max_diff = diff; }
            assert_eq!(ref_result.primary[i].to_bits(), test_result[i].to_bits(),
                "Bitwise mismatch at n={}, index {}: inclusive={} device={}",
                n, i, ref_result.primary[i], test_result[i]);
        }

        println!("  n={:>7}: PASS (bitwise identical, max_diff={:.0e})", n, max_diff);
    }
}

fn verify_welford() {
    println!("\n--- WelfordOp: running mean + variance ---\n");

    let mut engine = ScanEngine::new().unwrap();

    for &n in &[100, 1024, 1025, 10_000] {
        // Use realistic-ish data
        let data: Vec<f64> = (0..n).map(|i| {
            let x = i as f64;
            100.0 + (x * 0.01).sin() * 5.0
        }).collect();

        // Reference
        let ref_result = engine.scan_inclusive(&WelfordOp, &data).unwrap();

        // Device path
        let stream = engine.stream().clone();
        let dev_input = stream.clone_htod(&data).unwrap();
        let dev_ptr = { let (p, _g) = dev_input.device_ptr(&stream); p };

        let dev_output = unsafe {
            engine.scan_device_ptr(&WelfordOp, dev_ptr, n).unwrap()
        };
        let test_mean = dev_output.primary_to_host().unwrap();

        assert_eq!(ref_result.primary.len(), test_mean.len());

        let mut max_mean_diff: f64 = 0.0;
        for i in 0..n {
            let diff = (ref_result.primary[i] - test_mean[i]).abs();
            if diff > max_mean_diff { max_mean_diff = diff; }
            assert_eq!(ref_result.primary[i].to_bits(), test_mean[i].to_bits(),
                "Mean bitwise mismatch at n={}, i={}", n, i);
        }

        println!("  n={:>7}: PASS (mean bitwise identical)", n);
    }
}

fn verify_at_block_boundary() {
    println!("\n--- Block boundary: n=1024 (exactly 1 block) ---\n");

    let mut engine = ScanEngine::new().unwrap();
    let n = 1024;
    let data: Vec<f64> = (1..=n as u64).map(|x| x as f64).collect();

    let ref_result = engine.scan_inclusive(&AddOp, &data).unwrap();

    let stream = engine.stream().clone();
    let dev_input = stream.clone_htod(&data).unwrap();
    let dev_ptr = { let (p, _g) = dev_input.device_ptr(&stream); p };

    let dev_output = unsafe {
        engine.scan_device_ptr(&AddOp, dev_ptr, n).unwrap()
    };
    let test_result = dev_output.primary_to_host().unwrap();

    // Check last element: sum 1..1024 = 1024*1025/2 = 524800
    let expected_last = 524800.0f64;
    assert_eq!(ref_result.primary[n - 1], expected_last);
    assert_eq!(test_result[n - 1], expected_last);

    // Verify first element after block boundary for n=1025
    let data2: Vec<f64> = (1..=1025u64).map(|x| x as f64).collect();
    let ref2 = engine.scan_inclusive(&AddOp, &data2).unwrap();
    let dev2 = stream.clone_htod(&data2).unwrap();
    let dev2_ptr = { let (p, _g) = dev2.device_ptr(&stream); p };
    let dev2_out = unsafe { engine.scan_device_ptr(&AddOp, dev2_ptr, 1025).unwrap() };
    let test2 = dev2_out.primary_to_host().unwrap();

    // Element 1024 (index 1024, value 1025): sum = 524800 + 1025 = 525825
    assert_eq!(ref2.primary[1024], 525825.0);
    assert_eq!(test2[1024], 525825.0);

    println!("  n=1024: last={} (exact match)", test_result[n - 1]);
    println!("  n=1025: element[1024]={} (cross-block propagation correct)", test2[1024]);
    println!("  PASS");
}
