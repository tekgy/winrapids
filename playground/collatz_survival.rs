//! # Collatz Survival Decay Rate
//!
//! For the ALL-ONES class (worst case), measure survival rate
//! at progressively larger extensions. If survival is bounded
//! below 1 at every extension, the infinite product → 0,
//! meaning the hard set has measure zero.

use std::time::Instant;

fn analyze_class(r: u64, k: u32) -> f64 {
    let mut current = if r == 0 { (1u128 << k) } else { r as u128 };
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

    (3.0f64).powi(odd_steps as i32) / (2.0f64).powi(total_shift as i32)
}

fn main() {
    eprintln!("==========================================================");
    eprintln!("  Collatz Survival Decay — All-Ones Class");
    eprintln!("  Does the hard fraction shrink at every extension?");
    eprintln!("==========================================================\n");

    // For ALL-ONES at k=16, extend by increasing amounts
    let base_k = 16u32;
    let all_ones_16 = (1u64 << base_k) - 1; // 65535

    eprintln!("Base class: {} (0b{:016b}), contraction={:.1}\n",
        all_ones_16, all_ones_16, analyze_class(all_ones_16, base_k));

    eprintln!("{:>6}  {:>10}  {:>10}  {:>10}  {:>8}  {:>10}",
        "ext_k", "total_k", "extensions", "still_hard", "survival", "time");
    eprintln!("{}", "-".repeat(68));

    let mut cumulative_survival = 1.0f64;

    for ext_k in 1..=20u32 {
        let total_k = base_k + ext_k;
        if total_k > 36 { break; } // stay within reasonable compute

        let ext_count = 1u64 << ext_k;
        let t0 = Instant::now();

        let mut still_hard = 0u64;
        for ext in 0..ext_count {
            let extended = all_ones_16 | (ext << base_k);
            let c = analyze_class(extended, total_k);
            if c >= 1.0 {
                still_hard += 1;
            }
        }

        let survival = still_hard as f64 / ext_count as f64;
        cumulative_survival *= survival;
        let elapsed = t0.elapsed().as_secs_f64();

        eprintln!("{:>6}  {:>10}  {:>10}  {:>10}  {:>7.2}%  {:>9.3}s",
            ext_k, total_k, ext_count, still_hard, survival * 100.0, elapsed);
    }

    eprintln!("\n--- Now: survival per INCREMENTAL 2-bit extension ---");
    eprintln!("(Each step extends by 2 more bits from the previous)");
    eprintln!("{:>8}  {:>10}  {:>10}  {:>10}  {:>10}  {:>8}",
        "total_k", "hard_in", "ext_count", "hard_out", "survival", "time");
    eprintln!("{}", "-".repeat(66));

    // Start from the hard classes at k=18 (all-ones extended by 2)
    // Then extend THOSE by 2 more bits, and so on
    let mut current_hard: Vec<u64> = vec![all_ones_16];
    let mut total_k = base_k;
    let step = 2u32;

    for _ in 0..10 {
        let ext_count = 1u64 << step;
        let t0 = Instant::now();

        let mut next_hard: Vec<u64> = Vec::new();

        for &base in &current_hard {
            for ext in 0..ext_count {
                let extended = base | (ext << total_k);
                let new_k = total_k + step;
                let c = analyze_class(extended, new_k);
                if c >= 1.0 {
                    next_hard.push(extended);
                }
            }
        }

        let survival = if current_hard.len() * ext_count as usize > 0 {
            next_hard.len() as f64 / (current_hard.len() as f64 * ext_count as f64)
        } else {
            0.0
        };

        total_k += step;
        let elapsed = t0.elapsed().as_secs_f64();

        eprintln!("{:>8}  {:>10}  {:>10}  {:>10}  {:>9.2}%  {:>7.3}s",
            total_k, current_hard.len(), current_hard.len() as u64 * ext_count,
            next_hard.len(), survival * 100.0, elapsed);

        current_hard = next_hard;
        if current_hard.is_empty() {
            eprintln!("\n  ALL HARD CLASSES ELIMINATED at k={}!", total_k);
            break;
        }
    }

    if !current_hard.is_empty() {
        eprintln!("\n  {} hard classes remain at k={}", current_hard.len(), total_k);
        eprintln!("  These are the MAXIMALLY persistent expanding classes.");

        // Show bit density of surviving classes
        eprintln!("\n  Bit densities of survivors:");
        let mut densities: Vec<(u64, f64, f64)> = current_hard.iter().map(|&r| {
            let bits_set = r.count_ones();
            let density = bits_set as f64 / total_k as f64;
            let c = analyze_class(r, total_k);
            (r, density, c)
        }).collect();
        densities.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());

        for (r, d, c) in densities.iter().take(20) {
            eprintln!("    contraction={:.2e}, density={:.3}, class=0b{:0width$b}",
                c, d, r, width = total_k as usize);
        }
    }

    // Phase 3: What about NEAR-all-ones? Do they heal faster?
    eprintln!("\n--- Survival of near-all-ones (k=16, 14-15 ones) ---");
    let near_all_ones: Vec<u64> = (1..1u64 << 16).step_by(2)
        .filter(|&r| {
            let bits = (r as u16).count_ones();
            bits >= 14 && analyze_class(r, 16) >= 1.0
        })
        .collect();

    eprintln!("  {} near-all-ones hard classes at k=16", near_all_ones.len());

    let ext_k = 8u32;
    let ext_count = 1u64 << ext_k;
    let mut survived = 0u64;
    let mut total = 0u64;
    for &r in &near_all_ones {
        for ext in 0..ext_count {
            let extended = r | (ext << base_k);
            let c = analyze_class(extended, base_k + ext_k);
            if c >= 1.0 { survived += 1; }
            total += 1;
        }
    }
    eprintln!("  After +8 bits: {}/{} = {:.2}% survive",
        survived, total, survived as f64 / total as f64 * 100.0);
}
