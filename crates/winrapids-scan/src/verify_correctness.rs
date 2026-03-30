//! Correctness verification: dump scan results for comparison against NumPy.
//!
//! Outputs tab-separated values to stdout:
//! 1. AddOp cumsum on 1M random doubles
//! 2. WelfordOp mean+var on 1M random doubles
//!
//! Python script reads these and compares against NumPy/Pandas.

use winrapids_scan::{ScanEngine, AddOp, WelfordOp};

fn main() {
    let mut engine = ScanEngine::new().expect("Failed to create ScanEngine");

    // Generate deterministic input: 1M doubles using simple LCG
    let n = 1_000_000;
    let mut input = Vec::with_capacity(n);
    let mut seed: u64 = 42;
    for _ in 0..n {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        // Map to [-50, 50] range (representative of price data)
        let val = (seed >> 11) as f64 / (1u64 << 53) as f64 * 100.0 - 50.0;
        input.push(val);
    }

    // AddOp (cumsum)
    let add_result = engine.scan_inclusive(&AddOp, &input).unwrap();

    // WelfordOp (mean + variance)
    let welford_result = engine.scan_inclusive(&WelfordOp, &input).unwrap();

    // Output format: one section per op, separated by markers
    println!("===INPUT===");
    println!("{}", n);
    for v in &input {
        println!("{:.17e}", v);
    }

    println!("===ADDOP===");
    for v in &add_result.primary {
        println!("{:.17e}", v);
    }

    println!("===WELFORD_MEAN===");
    for v in &welford_result.primary {
        println!("{:.17e}", v);
    }

    println!("===WELFORD_VAR===");
    if !welford_result.secondary.is_empty() {
        for v in &welford_result.secondary[0] {
            println!("{:.17e}", v);
        }
    }

    println!("===END===");
}
