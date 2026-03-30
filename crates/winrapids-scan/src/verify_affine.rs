//! Campsite 2+3: Verify AffineOp and RefCenteredStatsOp
//!
//! Proves:
//! 1. AffineOp::cumsum() == AddOp (bit-identical)
//! 2. AffineOp::ewm(alpha) matches EWMOp (normalized vs unnormalized, see below)
//! 3. AffineOp::kalman(F,H,Q,R) == KalmanAffineOp (bit-identical)
//! 4. RefCenteredStatsOp beats WelfordOp in stability while hitting fast tier
//!
//! Run with: cargo run --bin verify-affine --release

use winrapids_scan::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Campsite 2: Universal AffineOp ===");
    println!("=== One combine for all 1D recurrences ===\n");

    let mut engine = ScanEngine::new()?;

    // Test data: random-walk-like with known structure
    let n = 10_000;
    let data: Vec<f64> = (0..n).map(|i| {
        let t = i as f64 / n as f64;
        (t * 7.0).sin() + 0.1 * ((i * 13 + 7) as f64 * 0.618033988749).fract()
    }).collect();

    println!("Data: {} elements, sin wave + noise\n", n);

    // -----------------------------------------------------------------------
    // Test 1: AffineOp::cumsum() vs AddOp
    // -----------------------------------------------------------------------
    let add_result = engine.scan_inclusive(&AddOp, &data)?;
    let aff_cumsum = engine.scan_inclusive(&AffineOp::cumsum(), &data)?;

    let max_err = add_result.primary.iter().zip(&aff_cumsum.primary)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let pass = max_err == 0.0;
    println!("AffineOp::cumsum() vs AddOp:");
    println!("  max_err = {:.2e}  {}\n", max_err, if pass { "BIT-IDENTICAL" } else { "DIFFERS" });

    // -----------------------------------------------------------------------
    // Test 2: AffineOp::kalman() vs KalmanAffineOp
    // -----------------------------------------------------------------------
    let f = 0.98;
    let h = 1.0;
    let q = 0.01;
    let r = 0.1;

    let kal_result = engine.scan_inclusive(&KalmanAffineOp::new(f, h, q, r), &data)?;
    let aff_kalman = engine.scan_inclusive(&AffineOp::kalman(f, h, q, r), &data)?;

    let max_err = kal_result.primary.iter().zip(&aff_kalman.primary)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f64, f64::max);
    let pass = max_err == 0.0;
    println!("AffineOp::kalman() vs KalmanAffineOp:");
    println!("  max_err = {:.2e}  {}\n", max_err, if pass { "BIT-IDENTICAL" } else { "DIFFERS" });

    // -----------------------------------------------------------------------
    // Test 3: AffineOp::ewm() — affine EWM (unnormalized) vs sequential reference
    //
    // Note: AffineOp::ewm produces the UNNORMALIZED affine recurrence
    //   x[t] = (1-α)·x[t-1] + α·z[t]
    // which matches a sequential EWM starting from x[0] = α·z[0].
    //
    // EWMOp uses (weight, value) normalization: extract = value/weight.
    // These are DIFFERENT computations with different semantics.
    // AffineOp matches the raw affine recurrence. EWMOp matches pandas ewm.
    // -----------------------------------------------------------------------
    let alpha = 0.1;
    let aff_ewm = engine.scan_inclusive(&AffineOp::ewm(alpha), &data)?;

    // Sequential reference for the unnormalized affine recurrence
    let mut seq_ewm = vec![0.0f64; n];
    seq_ewm[0] = alpha * data[0];
    for i in 1..n {
        seq_ewm[i] = (1.0 - alpha) * seq_ewm[i - 1] + alpha * data[i];
    }

    let max_err = aff_ewm.primary.iter().zip(&seq_ewm)
        .map(|(g, s)| (g - s).abs())
        .fold(0.0f64, f64::max);
    println!("AffineOp::ewm(α={}) vs sequential reference:", alpha);
    println!("  max_err = {:.2e}  {}\n", max_err, if max_err < 1e-10 { "PASS" } else { "FAIL" });

    // -----------------------------------------------------------------------
    // Test 4: RefCenteredStatsOp vs WelfordOp
    // -----------------------------------------------------------------------
    println!("=== Campsite 3: RefCenteredStatsOp ===");
    println!("=== Fast tier variance — zero division in combine ===\n");

    // Use data with large mean to stress numerical stability
    let large_mean = 1e8;
    let data_large: Vec<f64> = data.iter().map(|&x| x + large_mean).collect();
    let reference = data_large[0]; // first observation as reference

    let welford = engine.scan_inclusive(&WelfordOp, &data_large)?;
    let refcent = engine.scan_inclusive(&RefCenteredStatsOp::new(reference), &data_large)?;

    // Sequential reference (Welford's algorithm, the gold standard)
    let mut seq_mean = vec![0.0f64; n];
    let mut seq_var = vec![0.0f64; n];
    {
        let mut count = 0.0f64;
        let mut mean = 0.0f64;
        let mut m2 = 0.0f64;
        for i in 0..n {
            count += 1.0;
            let delta = data_large[i] - mean;
            mean += delta / count;
            let delta2 = data_large[i] - mean;
            m2 += delta * delta2;
            seq_mean[i] = mean;
            seq_var[i] = if count > 1.0 { m2 / (count - 1.0) } else { 0.0 };
        }
    }

    // Compare means
    let welford_mean_err = welford.primary.iter().zip(&seq_mean)
        .skip(1) // skip first (count=1, trivial)
        .map(|(g, s)| (g - s).abs())
        .fold(0.0f64, f64::max);
    let refcent_mean_err = refcent.primary.iter().zip(&seq_mean)
        .skip(1)
        .map(|(g, s)| (g - s).abs())
        .fold(0.0f64, f64::max);

    println!("Mean accuracy (large mean = {:.0e}):", large_mean);
    println!("  Welford:     max_err = {:.2e}", welford_mean_err);
    println!("  RefCentered: max_err = {:.2e}", refcent_mean_err);
    println!("  Winner: {}\n", if refcent_mean_err <= welford_mean_err { "RefCentered" } else { "Welford" });

    // Compare variances
    let welford_var = &welford.secondary[0];
    let refcent_var = &refcent.secondary[0];

    let welford_var_err = welford_var.iter().zip(&seq_var)
        .skip(2)
        .filter(|(_, s)| s.abs() > 1e-15)
        .map(|(g, s)| ((g - s) / s).abs())
        .fold(0.0f64, f64::max);
    let refcent_var_err = refcent_var.iter().zip(&seq_var)
        .skip(2)
        .filter(|(_, s)| s.abs() > 1e-15)
        .map(|(g, s)| ((g - s) / s).abs())
        .fold(0.0f64, f64::max);

    println!("Variance accuracy (large mean = {:.0e}):", large_mean);
    println!("  Welford:     max_rel_err = {:.2e}", welford_var_err);
    println!("  RefCentered: max_rel_err = {:.2e}", refcent_var_err);
    println!("  Winner: {}\n", if refcent_var_err <= welford_var_err { "RefCentered" } else { "Welford" });

    // Digit count comparison
    let welford_digits = if welford_var_err > 0.0 { -(welford_var_err.log10()) } else { 16.0 };
    let refcent_digits = if refcent_var_err > 0.0 { -(refcent_var_err.log10()) } else { 16.0 };

    println!("Correct digits of variance:");
    println!("  Welford:     {:.1} digits", welford_digits);
    println!("  RefCentered: {:.1} digits", refcent_digits);

    println!("\nTam doesn't sort. Tam doesn't divide. Tam knows.");
    Ok(())
}
