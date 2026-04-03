//! # Collatz Hard-Class Cascade Analysis
//!
//! Key observation from formal proof generator:
//! - Coverage peaks at k=16 (92.46%) and DECREASES at higher k
//! - Average contraction is ALWAYS 3/2 (arithmetic mean) regardless of k
//! - Geometric mean is √(3/4) ≈ 0.866 < 1 (contractive)
//! - The ~7.5% hard classes at k=16 are high-bit-density residues
//!
//! Question: when we EXTEND a hard class from k=16 to k=32,
//! what fraction of its 2^16 extensions are STILL hard?
//! If that fraction shrinks, the hard classes are "healing" — the
//! expanding first phase is being compensated by typical second phases.
//!
//! This is a multi-phase contraction analysis without requiring
//! the open normality conjecture on bit density of 3^k.

use std::time::Instant;

/// Analyze a residue class mod 2^k: return (contraction_factor, odd_steps, total_shift)
fn analyze_class(r: u64, k: u32) -> (f64, u32, u32) {
    let modulus = 1u64 << k;
    debug_assert!(r < modulus);

    let mut current = if r == 0 { modulus as u128 } else { r as u128 };
    let mut odd_steps: u32 = 0;
    let mut total_shift: u32 = 0;
    let mut bits_processed = 0u32;

    while bits_processed < k {
        if current % 2 == 0 {
            current /= 2;
            total_shift += 1;
            bits_processed += 1;
        } else {
            current = 3 * current + 1;
            odd_steps += 1;
        }
    }

    let contraction = (3.0f64).powi(odd_steps as i32) / (2.0f64).powi(total_shift as i32);
    (contraction, odd_steps, total_shift)
}

/// Find all hard (non-decreasing) odd classes at resolution k
fn find_hard_classes(k: u32) -> Vec<u64> {
    let modulus = 1u64 << k;
    let mut hard = Vec::new();
    for r in (1..modulus).step_by(2) {
        let (contraction, _, _) = analyze_class(r, k);
        if contraction >= 1.0 {
            hard.push(r);
        }
    }
    hard
}

/// For a hard class at k=base_k, extend to k=base_k+ext_k by appending
/// all possible ext_k-bit suffixes. Count how many extensions are still hard.
fn cascade_analysis(base_class: u64, base_k: u32, ext_k: u32) -> CascadeResult {
    let total_k = base_k + ext_k;
    let ext_count = 1u64 << ext_k;
    let base_modulus = 1u64 << base_k;

    let mut still_hard = 0u64;
    let mut total_contraction = 0.0f64;
    let mut worst_contraction = 0.0f64;
    let mut best_contraction = f64::MAX;
    let mut contraction_log_sum = 0.0f64;

    for ext in 0..ext_count {
        // The extended class: low base_k bits = base_class, next ext_k bits = ext
        let extended = base_class | (ext << base_k);
        let (contraction, _, _) = analyze_class(extended, total_k);

        total_contraction += contraction;
        contraction_log_sum += contraction.ln();

        if contraction >= 1.0 {
            still_hard += 1;
        }
        if contraction > worst_contraction {
            worst_contraction = contraction;
        }
        if contraction < best_contraction {
            best_contraction = contraction;
        }
    }

    CascadeResult {
        base_class,
        base_k,
        ext_k,
        total_k,
        extensions_checked: ext_count,
        still_hard,
        fraction_still_hard: still_hard as f64 / ext_count as f64,
        avg_contraction: total_contraction / ext_count as f64,
        geometric_mean_contraction: (contraction_log_sum / ext_count as f64).exp(),
        worst_contraction,
        best_contraction,
    }
}

#[derive(Debug)]
struct CascadeResult {
    base_class: u64,
    base_k: u32,
    ext_k: u32,
    total_k: u32,
    extensions_checked: u64,
    still_hard: u64,
    fraction_still_hard: f64,
    avg_contraction: f64,
    geometric_mean_contraction: f64,
    worst_contraction: f64,
    best_contraction: f64,
}

/// Two-phase contraction: compose the base-class contraction with each
/// extension's contraction to get the net 2-phase factor.
fn two_phase_analysis(hard_classes: &[u64], base_k: u32, ext_k: u32) -> TwoPhaseResult {
    let ext_count = 1u64 << ext_k;

    let mut total_net_contractive = 0u64;
    let mut total_extensions = 0u64;
    let mut worst_class: u64 = 0;
    let mut worst_net: f64 = 0.0;
    let mut log_product_sum = 0.0f64;
    let mut n_products = 0u64;

    for &base_class in hard_classes {
        let (base_contraction, _, _) = analyze_class(base_class, base_k);

        for ext in 0..ext_count {
            let extended = base_class | (ext << base_k);
            let (ext_contraction, _, _) = analyze_class(extended, base_k + ext_k);

            // Net contraction over both phases is just the extended contraction
            // (it already accounts for all base_k + ext_k bits)
            let net = ext_contraction;

            if net < 1.0 {
                total_net_contractive += 1;
            }
            if net > worst_net {
                worst_net = net;
                worst_class = extended;
            }
            log_product_sum += net.ln();
            n_products += 1;
            total_extensions += 1;
        }
    }

    TwoPhaseResult {
        hard_classes_count: hard_classes.len() as u64,
        base_k,
        ext_k,
        total_k: base_k + ext_k,
        total_extensions,
        net_contractive: total_net_contractive,
        fraction_contractive: total_net_contractive as f64 / total_extensions as f64,
        geometric_mean: (log_product_sum / n_products as f64).exp(),
        worst_net,
        worst_class,
    }
}

#[derive(Debug)]
struct TwoPhaseResult {
    hard_classes_count: u64,
    base_k: u32,
    ext_k: u32,
    total_k: u32,
    total_extensions: u64,
    net_contractive: u64,
    fraction_contractive: f64,
    geometric_mean: f64,
    worst_net: f64,
    worst_class: u64,
}

fn main() {
    eprintln!("==========================================================");
    eprintln!("  Collatz Hard-Class Cascade Analysis");
    eprintln!("  Do expanding classes heal at higher resolution?");
    eprintln!("==========================================================\n");

    // Phase 1: Find hard classes at k=16
    let base_k = 16u32;
    eprintln!("Finding hard classes at k={}...", base_k);
    let t0 = Instant::now();
    let hard = find_hard_classes(base_k);
    eprintln!("  Found {} hard odd classes (of {} total odd classes)",
        hard.len(), 1u64 << (base_k - 1));
    eprintln!("  Time: {:.3}ms\n", t0.elapsed().as_secs_f64() * 1000.0);

    // Show bit density of worst classes
    eprintln!("-- Bit density of top-10 hardest classes --");
    let mut ranked: Vec<(u64, f64)> = hard.iter().map(|&r| {
        let (c, _, _) = analyze_class(r, base_k);
        (r, c)
    }).collect();
    ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (r, c) in ranked.iter().take(10) {
        let bits_set = (*r as u32).count_ones();
        let density = bits_set as f64 / base_k as f64;
        eprintln!("  class {:5} (0b{:016b}): contraction={:.1}, bit_density={:.3} ({}/{})",
            r, r, c, density, bits_set, base_k);
    }

    // Phase 2: Cascade analysis — extend hard classes by 8 bits at a time
    for ext_k in [8u32, 12, 16] {
        eprintln!("\n-- Cascade: extend hard classes from k={} → k={} --", base_k, base_k + ext_k);
        let t0 = Instant::now();

        // Sample: analyze top-50 hardest + random sample of 200
        let sample_size = hard.len().min(250);
        let sample: Vec<u64> = ranked.iter().take(sample_size).map(|(r, _)| *r).collect();

        let mut total_still_hard = 0u64;
        let mut total_checked = 0u64;
        let mut worst_survival = 0.0f64;
        let mut worst_class = 0u64;
        let mut geo_mean_sum = 0.0f64;
        let mut geo_count = 0u64;

        for &base_class in &sample {
            let result = cascade_analysis(base_class, base_k, ext_k);
            total_still_hard += result.still_hard;
            total_checked += result.extensions_checked;
            geo_mean_sum += result.geometric_mean_contraction.ln();
            geo_count += 1;

            if result.fraction_still_hard > worst_survival {
                worst_survival = result.fraction_still_hard;
                worst_class = base_class;
            }
        }

        let fraction = total_still_hard as f64 / total_checked as f64;
        let avg_geo = (geo_mean_sum / geo_count as f64).exp();

        eprintln!("  Sampled {} hard classes × {} extensions each", sample.len(), 1u64 << ext_k);
        eprintln!("  Still hard after extension: {}/{} = {:.2}%",
            total_still_hard, total_checked, fraction * 100.0);
        eprintln!("  Avg geometric mean contraction: {:.4}", avg_geo);
        eprintln!("  Worst survival rate: {:.2}% (class {})", worst_survival * 100.0, worst_class);
        eprintln!("  Time: {:.3}s", t0.elapsed().as_secs_f64());
    }

    // Phase 3: Two-phase net contraction for ALL hard classes at k=16 → k=24
    eprintln!("\n-- Two-phase net analysis: ALL {} hard classes extended by 8 bits --", hard.len());
    let t0 = Instant::now();
    let result = two_phase_analysis(&hard, base_k, 8);
    eprintln!("  Total extensions: {}", result.total_extensions);
    eprintln!("  Net contractive: {}/{} = {:.2}%",
        result.net_contractive, result.total_extensions,
        result.fraction_contractive * 100.0);
    eprintln!("  Geometric mean net contraction: {:.6}", result.geometric_mean);
    eprintln!("  Worst net: {:.2} (class 0b{:024b})", result.worst_net, result.worst_class);
    let worst_bits = (result.worst_class as u32).count_ones();
    eprintln!("  Worst class bit density: {}/{} = {:.3}",
        worst_bits, result.total_k, worst_bits as f64 / result.total_k as f64);
    eprintln!("  Time: {:.3}s", t0.elapsed().as_secs_f64());

    // Phase 4: For the ALL-ONES class specifically, trace through multiple phases
    eprintln!("\n-- Multi-phase trace: ALL-ONES class --");
    let all_ones_16 = (1u64 << 16) - 1; // 65535
    let (c16, odd16, shift16) = analyze_class(all_ones_16, 16);
    eprintln!("  Phase 1 (k=16): class=65535, contraction={:.1}, odd_steps={}, shift={}",
        c16, odd16, shift16);

    // For all-ones extended to various k
    for total_k in [20u32, 24, 28, 32] {
        if total_k > 32 { break; } // u64 limit
        let all_ones = (1u64 << total_k) - 1;
        let (c, odd, shift) = analyze_class(all_ones, total_k);
        let bits_set = total_k; // all ones
        eprintln!("  Phase (k={}): all-ones contraction={:.2e}, odd={}, shift={}, ratio={}/{}={:.4}",
            total_k, c, odd, shift, odd, shift, odd as f64 / shift as f64);
    }

    eprintln!("\n  Key insight: odd/shift ratio tells us how many bits the 3n+1");
    eprintln!("  operation 'creates' vs 'consumes'. If ratio > log2(3)/log2(2) ≈ 0.6309,");
    eprintln!("  the class is expanding. The all-ones class always has ratio = 1.0");
    eprintln!("  (every bit triggers an odd step, each odd step adds ~0.585 bits).");
    eprintln!("  But after processing, the RESULT has typical bit density ~0.5,");
    eprintln!("  so subsequent phases contract.");

    // Phase 5: Survival rate by initial bit density
    eprintln!("\n-- Survival rate by bit density (k=16 → k=24) --");
    let mut by_density: std::collections::BTreeMap<u32, (u64, u64)> = std::collections::BTreeMap::new();
    for &r in &hard {
        let bits_set = (r as u32).count_ones();
        let ext_k = 8u32;
        let ext_count = 1u64 << ext_k;
        let mut still_hard = 0u64;
        for ext in 0..ext_count {
            let extended = r | (ext << base_k);
            let (c, _, _) = analyze_class(extended, base_k + ext_k);
            if c >= 1.0 {
                still_hard += 1;
            }
        }
        let entry = by_density.entry(bits_set).or_insert((0, 0));
        entry.0 += still_hard;
        entry.1 += ext_count;
    }

    for (bits, (hard_count, total)) in &by_density {
        let survival = *hard_count as f64 / *total as f64;
        eprintln!("  {} 1-bits ({:.1}% density): survival={:.2}% ({}/{})",
            bits, *bits as f64 / base_k as f64 * 100.0,
            survival * 100.0, hard_count, total);
    }
}
