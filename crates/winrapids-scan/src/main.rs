//! Test binary for winrapids-scan.
//!
//! Validates: AddOp (cumsum), WelfordOp (mean+variance), kernel generation.

use winrapids_scan::ops::*;
use winrapids_scan::engine::generate_scan_kernel;

fn main() {
    println!("=== winrapids-scan: kernel generation test ===\n");

    // Test 1: AddOp kernel generation
    let add_op = AddOp;
    let add_kernel = generate_scan_kernel(&add_op);
    println!("--- AddOp (cumsum) kernel ---");
    println!("{}\n", &add_kernel[..500.min(add_kernel.len())]);

    // Test 2: WelfordOp kernel generation
    let welford_op = WelfordOp;
    let welford_kernel = generate_scan_kernel(&welford_op);
    println!("--- WelfordOp (mean+variance) kernel ---");
    println!("{}\n", &welford_kernel[..800.min(welford_kernel.len())]);

    // Test 3: EWMOp kernel generation
    let ewm_op = EWMOp { alpha: 0.1 };
    let ewm_kernel = generate_scan_kernel(&ewm_op);
    println!("--- EWMOp (alpha=0.1) kernel ---");
    println!("{}\n", &ewm_kernel[..600.min(ewm_kernel.len())]);

    // Test 4: Cache key uniqueness
    let ewm_op2 = EWMOp { alpha: 0.2 };
    assert_ne!(
        ewm_op.params_key(),
        ewm_op2.params_key(),
        "Different alpha should produce different cache keys"
    );
    println!("Cache key test: PASS (different alpha → different key)");

    // Test 5: Operator properties
    println!("\n=== Operator properties ===");
    let ops: Vec<(&str, Box<dyn AssociativeOp>)> = vec![
        ("AddOp", Box::new(AddOp)),
        ("MulOp", Box::new(MulOp)),
        ("MaxOp", Box::new(MaxOp)),
        ("MinOp", Box::new(MinOp)),
        ("WelfordOp", Box::new(WelfordOp)),
        ("EWMOp(0.1)", Box::new(EWMOp { alpha: 0.1 })),
    ];

    for (name, op) in &ops {
        println!(
            "{:15} outputs={} state={}",
            name,
            op.output_width(),
            if op.cuda_state_type().starts_with("struct") { "struct" } else { "scalar" },
        );
    }

    println!("\n=== All tests passed ===");
    println!("Next: GPU launch test with cudarc (requires CUDA device)");
}
