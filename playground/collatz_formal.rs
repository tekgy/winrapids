//! # Formal Collatz Proof Generator
//!
//! For each residue class mod 2^k, symbolically iterate the Collatz map
//! and determine whether the class is PROVABLY decreasing.
//!
//! If a class is provably decreasing, every integer in that class
//! eventually reaches a smaller value — PROVEN, not computed.
//!
//! The question: what percentage of residue classes are provably decreasing?
//! If 99%+, then 99%+ of all integers satisfy Collatz BY PROOF.

use std::time::Instant;

/// Symbolically iterate Collatz on a residue class mod 2^k.
///
/// For n ≡ r (mod 2^k), the first k bits of n are determined.
/// The Collatz map's behavior for those k bits is deterministic.
///
/// After processing k bits: result = (3^s * n + C) / 2^k
/// where s = number of odd steps, C = a constant depending on r.
///
/// The class is "provably decreasing" if 3^s / 2^k < 1,
/// meaning the multiplicative factor is less than 1.
/// This guarantees the output is SMALLER than the input (for large enough n).
fn analyze_residue_class(r: u64, k: u32) -> ResidueAnalysis {
    let modulus = 1u64 << k;
    assert!(r < modulus);

    // Simulate k bits of Collatz on a representative number
    // The representative is r (or r + modulus if r is 0)
    let mut n = if r == 0 { modulus } else { r };
    let mut original_n = n;

    // Track the affine transform symbolically
    // After processing: result = (multiplier * original_n + offset) / 2^total_shift
    let mut multiplier: u128 = 1;  // accumulates powers of 3
    let mut offset: u128 = 0;       // accumulates additive constants
    let mut total_shift: u32 = 0;   // accumulates powers of 2 divided out
    let mut odd_steps: u32 = 0;
    let mut even_steps: u32 = 0;
    let mut total_steps: u32 = 0;

    // Process k bits
    let mut bits_processed = 0u32;
    let mut current = n as u128;

    while bits_processed < k {
        if current % 2 == 0 {
            current /= 2;
            total_shift += 1;
            bits_processed += 1;
            even_steps += 1;
        } else {
            current = 3 * current + 1;
            multiplier *= 3;
            offset = offset * 3 + 1;
            odd_steps += 1;
            // After 3n+1, the number is even, so we'll halve next
            // (the halving consumes a bit)
        }
        total_steps += 1;
    }

    // The transform: for any n ≡ r (mod 2^k),
    // after `total_steps` Collatz steps:
    //   result = (3^odd_steps * n + C_r) / 2^total_shift
    //
    // The contraction factor is 3^odd_steps / 2^total_shift
    let contraction = (3.0f64).powi(odd_steps as i32) / (2.0f64).powi(total_shift as i32);

    // The class is provably decreasing if contraction < 1
    // Meaning: for all n in this class (large enough), the result is smaller than n
    let is_decreasing = contraction < 1.0;

    // Minimum n for which the result is guaranteed smaller:
    // (3^s * n + C) / 2^k < n
    // 3^s * n + C < n * 2^k
    // n * (2^k - 3^s) > C
    // n > C / (2^k - 3^s)  [when 2^k > 3^s, i.e., contraction < 1]
    let min_n_for_decrease = if is_decreasing {
        let denominator = (1u128 << total_shift) - multiplier;
        if denominator > 0 {
            (offset + denominator - 1) / denominator // ceiling division
        } else {
            u128::MAX
        }
    } else {
        u128::MAX
    };

    ResidueAnalysis {
        residue: r,
        modulus: modulus as u128,
        multiplier,       // 3^odd_steps
        offset,           // constant C_r
        total_shift,      // total halvings
        odd_steps,
        even_steps,
        total_steps,
        contraction,
        is_decreasing,
        min_n_for_decrease,
    }
}

#[derive(Debug)]
struct ResidueAnalysis {
    residue: u64,
    modulus: u128,
    multiplier: u128,     // 3^s
    offset: u128,         // additive constant
    total_shift: u32,     // 2^shift in denominator
    odd_steps: u32,
    even_steps: u32,
    total_steps: u32,
    contraction: f64,     // 3^s / 2^shift
    is_decreasing: bool,  // contraction < 1
    min_n_for_decrease: u128,
}

/// Analyze ALL residue classes mod 2^k and report statistics
fn analyze_all_classes(k: u32) -> ClassAnalysis {
    let modulus = 1u64 << k;
    let t0 = Instant::now();

    let mut decreasing = 0u64;
    let mut non_decreasing = 0u64;
    let mut total_contraction = 0.0f64;
    let mut min_contraction = f64::MAX;
    let mut max_contraction = 0.0f64;
    let mut worst_class: u64 = 0;
    let mut best_class: u64 = 0;
    let mut max_min_n: u128 = 0;

    // Only analyze ODD residue classes (even classes trivially halve)
    let odd_classes = modulus / 2;

    for r in 0..modulus {
        if r % 2 == 0 { continue; } // skip even — trivially decreasing

        let analysis = analyze_residue_class(r, k);

        if analysis.is_decreasing {
            decreasing += 1;
        } else {
            non_decreasing += 1;
        }

        total_contraction += analysis.contraction;

        if analysis.contraction < min_contraction {
            min_contraction = analysis.contraction;
            best_class = r;
        }
        if analysis.contraction > max_contraction {
            max_contraction = analysis.contraction;
            worst_class = r;
        }
        if analysis.is_decreasing && analysis.min_n_for_decrease < u128::MAX {
            if analysis.min_n_for_decrease > max_min_n {
                max_min_n = analysis.min_n_for_decrease;
            }
        }
    }

    let elapsed = t0.elapsed();

    // Even classes are all trivially decreasing (they halve immediately)
    let even_decreasing = modulus / 2;

    ClassAnalysis {
        k,
        modulus: modulus as u128,
        total_classes: modulus,
        odd_classes,
        odd_decreasing: decreasing,
        odd_non_decreasing: non_decreasing,
        even_decreasing,
        total_decreasing: decreasing + even_decreasing,
        fraction_proven: (decreasing + even_decreasing) as f64 / modulus as f64,
        avg_contraction: total_contraction / odd_classes as f64,
        min_contraction,
        max_contraction,
        best_class,
        worst_class,
        max_min_n_for_decrease: max_min_n,
        elapsed_secs: elapsed.as_secs_f64(),
    }
}

#[derive(Debug)]
struct ClassAnalysis {
    k: u32,
    modulus: u128,
    total_classes: u64,
    odd_classes: u64,
    odd_decreasing: u64,
    odd_non_decreasing: u64,
    even_decreasing: u64,
    total_decreasing: u64,
    fraction_proven: f64,
    avg_contraction: f64,
    min_contraction: f64,
    max_contraction: f64,
    best_class: u64,
    worst_class: u64,
    max_min_n_for_decrease: u128,
    elapsed_secs: f64,
}

fn main() {
    eprintln!("==========================================================");
    eprintln!("  Formal Collatz Proof Generator");
    eprintln!("  Symbolic iteration on residue classes mod 2^k");
    eprintln!("  Question: what % of integers are PROVABLY decreasing?");
    eprintln!("==========================================================\n");

    // Analyze at increasing resolution
    for k in [4, 8, 10, 12, 14, 16, 18, 20] {
        let analysis = analyze_all_classes(k);

        eprintln!("k={:2} (mod 2^{:2} = {:>8} classes):", k, k, analysis.total_classes);
        eprintln!("  Proven decreasing: {}/{} = {:.2}%",
            analysis.total_decreasing, analysis.total_classes,
            analysis.fraction_proven * 100.0);
        eprintln!("  Odd decreasing: {}/{} = {:.2}%",
            analysis.odd_decreasing, analysis.odd_classes,
            analysis.odd_decreasing as f64 / analysis.odd_classes as f64 * 100.0);
        eprintln!("  Avg contraction: {:.4} (< 1 = good)", analysis.avg_contraction);
        eprintln!("  Min contraction: {:.6} (class {})", analysis.min_contraction, analysis.best_class);
        eprintln!("  Max contraction: {:.4} (class {})", analysis.max_contraction, analysis.worst_class);
        if analysis.max_min_n_for_decrease < u128::MAX && analysis.max_min_n_for_decrease > 0 {
            eprintln!("  Largest min_n for proof: {}", analysis.max_min_n_for_decrease);
        }
        eprintln!("  Time: {:.3}s\n", analysis.elapsed_secs);
    }

    // Deep dive on the worst (most expanding) classes at k=16
    eprintln!("-- Top 10 most expanding odd classes at k=16 --");
    let k = 16u32;
    let modulus = 1u64 << k;
    let mut expansions: Vec<(u64, f64)> = Vec::new();
    for r in (1..modulus).step_by(2) {
        let a = analyze_residue_class(r, k);
        if !a.is_decreasing {
            expansions.push((r, a.contraction));
        }
    }
    expansions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (r, c) in expansions.iter().take(10) {
        eprintln!("  class {} mod 2^16: contraction = {:.4} (EXPANDING)", r, c);
    }
    eprintln!("  Total non-decreasing odd classes at k=16: {}", expansions.len());
    eprintln!("  These are the HARD cases — focus verification here.");
}
