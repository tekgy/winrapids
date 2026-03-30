//! Observer verification: KalmanAffineOp(F=1, H=1) == EWMOp(alpha=K_ss).
//!
//! Naturalist claims these are the same point in operator space.
//! If true, the outputs should be bitwise identical (or near-machine-epsilon).

use winrapids_scan::{ScanEngine, DevicePtr};
use winrapids_scan::ops::{AssociativeOp, EWMOp, KalmanAffineOp};

fn main() {
    println!("{}", "=".repeat(70));
    println!("Observer Verification: Kalman-EWM Family Overlap");
    println!("{}", "=".repeat(70));

    verify_overlap();
    verify_at_multiple_qr_ratios();

    println!("\n{}", "=".repeat(70));
    println!("Verification complete.");
    println!("{}", "=".repeat(70));
}

fn verify_overlap() {
    println!("\n--- KalmanAffineOp(F=1, H=1, Q, R) vs EWMOp(alpha=K_ss) ---\n");

    let mut engine = ScanEngine::new().unwrap();

    // F=1, H=1, Q=0.01, R=0.1
    let kalman = KalmanAffineOp::new(1.0, 1.0, 0.01, 0.1);
    let ewm = EWMOp { alpha: kalman.k_ss };

    println!("  F=1.0, H=1.0, Q=0.01, R=0.1");
    println!("  Kalman K_ss = {:.10}", kalman.k_ss);
    println!("  Kalman A    = {:.10}", kalman.a);
    println!("  EWM alpha   = {:.10} (set to K_ss)", ewm.alpha);
    println!("  Expected: A = 1 - K_ss = {:.10}", 1.0 - kalman.k_ss);

    // With F=1, H=1: A = (1 - K_ss * H) * F = 1 - K_ss
    // EWM decay = 1 - alpha = 1 - K_ss = A
    // So the combine operations should be equivalent.
    assert!((kalman.a - (1.0 - kalman.k_ss)).abs() < 1e-15,
        "A should equal 1 - K_ss when F=1, H=1");

    let n = 10_000usize;
    let data: Vec<f64> = (0..n).map(|i| {
        10.0 + (i as f64 * 0.05).sin() * 3.0 + (i as f64 * 0.13).cos() * 1.5
    }).collect();

    let kalman_result = engine.scan_inclusive(&kalman, &data).unwrap();
    let ewm_result = engine.scan_inclusive(&ewm, &data).unwrap();

    let mut max_abs_err: f64 = 0.0;
    let mut max_rel_err: f64 = 0.0;
    let mut bitwise_matches = 0usize;
    for i in 0..n {
        let k = kalman_result.primary[i];
        let e = ewm_result.primary[i];
        let abs_err = (k - e).abs();
        if abs_err > max_abs_err { max_abs_err = abs_err; }
        if e.abs() > 1e-10 {
            let rel = abs_err / e.abs();
            if rel > max_rel_err { max_rel_err = rel; }
        }
        if k.to_bits() == e.to_bits() { bitwise_matches += 1; }
    }

    println!("\n  n={}: max_abs_err={:.2e}  max_rel_err={:.2e}", n, max_abs_err, max_rel_err);
    println!("  Bitwise matches: {}/{} ({:.1}%)",
        bitwise_matches, n, 100.0 * bitwise_matches as f64 / n as f64);

    // Check some specific values
    println!("\n  Sample comparison:");
    for &i in &[0, 1, 10, 100, 9999] {
        println!("    [{}] kalman={:.15e}  ewm={:.15e}  diff={:.2e}",
            i, kalman_result.primary[i], ewm_result.primary[i],
            (kalman_result.primary[i] - ewm_result.primary[i]).abs());
    }

    if max_abs_err < 1e-10 {
        println!("\n  CONFIRMED: KalmanAffineOp(F=1,H=1) == EWMOp(alpha=K_ss) within machine epsilon");
    } else {
        println!("\n  DIVERGES: max_abs_err={:.2e} — operators are NOT equivalent", max_abs_err);
    }
}

fn verify_at_multiple_qr_ratios() {
    println!("\n--- Varying Q/R ratio (F=1, H=1 fixed) ---\n");

    let mut engine = ScanEngine::new().unwrap();

    let n = 1_000usize;
    let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin() * 10.0).collect();

    println!("  {:>8} {:>8} {:>10} {:>12} {:>12}", "Q", "R", "K_ss", "max_abs_err", "bitwise_%");

    for &(q, r) in &[(0.001, 1.0), (0.01, 0.1), (0.1, 0.1), (1.0, 0.01), (0.5, 0.5)] {
        let kalman = KalmanAffineOp::new(1.0, 1.0, q, r);
        let ewm = EWMOp { alpha: kalman.k_ss };

        let kr = engine.scan_inclusive(&kalman, &data).unwrap();
        let er = engine.scan_inclusive(&ewm, &data).unwrap();

        let max_err = kr.primary.iter().zip(er.primary.iter())
            .map(|(k, e)| (k - e).abs())
            .fold(0.0f64, f64::max);

        let bw = kr.primary.iter().zip(er.primary.iter())
            .filter(|(k, e)| k.to_bits() == e.to_bits())
            .count();

        println!("  {:>8.3} {:>8.3} {:>10.6} {:>12.2e} {:>11.1}%",
            q, r, kalman.k_ss, max_err, 100.0 * bw as f64 / n as f64);
    }
}
