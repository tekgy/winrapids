//! # Full Superposition — Not Sweeping, SUPERPOSITIONING
//!
//! Previous experiments swept N=2, N=3, N=4... one at a time.
//! That's still choosing. This runs EVERYTHING at once:
//!
//! - All pairs of first 30 primes (435 pairs)
//! - All triples of first 15 primes (455 triples)
//! - All "first K" systems for K=2..100
//! - Continuous s (not just integers — full real line s > 1)
//! - Report: which subsets share the same s*? Which are related?
//! - The RELATIONSHIPS between subset results IS the structure.

use std::time::Instant;

fn sieve_primes(limit: usize) -> Vec<u64> {
    let mut is_prime = vec![true; limit + 1];
    is_prime[0] = false;
    if limit > 0 { is_prime[1] = false; }
    for i in 2..=((limit as f64).sqrt() as usize) {
        if is_prime[i] {
            for j in (i*i..=limit).step_by(i) { is_prime[j] = false; }
        }
    }
    is_prime.iter().enumerate().filter(|&(_, &b)| b).map(|(i, _)| i as u64).collect()
}

fn free_energy(p: f64, s: f64) -> f64 {
    let x = p.powf(-s);
    if x >= 1.0 { return f64::INFINITY; }
    -(1.0 - x).ln()
}

fn solve(primes: &[f64], target: f64) -> Option<f64> {
    if target <= 0.0 { return None; }
    let f_at = |s: f64| -> f64 { primes.iter().map(|&p| free_energy(p, s)).sum() };

    let mut prev = f64::INFINITY;
    for i in 1..10000 {
        let s = i as f64 * 0.005; // finer resolution: 0.005 steps
        let f = f_at(s);
        if !f.is_finite() { prev = f; continue; }
        if prev > target && f <= target {
            let mut lo = (i - 1) as f64 * 0.005;
            let mut hi = s;
            for _ in 0..100 {
                let mid = (lo + hi) / 2.0;
                let fm = f_at(mid);
                if fm > target { lo = mid; } else { hi = mid; }
            }
            return Some((lo + hi) / 2.0);
        }
        prev = f;
    }
    None
}

fn energy_partition(primes: &[f64], s: f64) -> Vec<f64> {
    let es: Vec<f64> = primes.iter().map(|&p| free_energy(p, s)).collect();
    let total: f64 = es.iter().sum();
    if total <= 0.0 { return vec![0.0; primes.len()]; }
    es.iter().map(|e| e / total).collect()
}

fn main() {
    let t0 = Instant::now();
    let all_p = sieve_primes(10000);
    let pf: Vec<f64> = all_p.iter().map(|&p| p as f64).collect();
    eprintln!("Sieved {} primes\n", all_p.len());

    // ═══════════════════════════════════════════════════════
    // PART 1: ALL PAIRS — the complete pairwise fold map
    // ═══════════════════════════════════════════════════════
    eprintln!("=== ALL PAIRS of first 30 primes ({} pairs) ===\n", 30*29/2);

    let n_pair = 30.min(pf.len());
    let mut pair_results: Vec<(usize, usize, f64, f64, f64)> = Vec::new(); // (i,j,s*,p_i_frac,ratio)

    for i in 0..n_pair {
        for j in (i+1)..n_pair {
            let pair = vec![pf[i], pf[j]];
            let target = 0.5 * (pf[j] / pf[i]).ln();
            if let Some(s) = solve(&pair, target) {
                let part = energy_partition(&pair, s);
                pair_results.push((i, j, s, part[0], pf[j] / pf[i]));
            }
        }
    }

    // Sort by s* — the fold hierarchy
    pair_results.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap()); // sort by first-prime fraction

    eprintln!("  {:>6} {:>6} {:>8} {:>10} {:>10} {:>10}",
        "p_a", "p_b", "ratio", "s*", "p_a frac", "contains 2?");
    for (i, j, s, frac, ratio) in pair_results.iter().take(20) {
        eprintln!("  {:>6} {:>6} {:>8.3} {:>10.4} {:>9.1}% {:>10}",
            all_p[*i], all_p[*j], ratio, s, frac * 100.0,
            if *i == 0 { "YES" } else { "no" });
    }
    eprintln!("  ...");
    eprintln!("  Bottom 10 (most equal partition):");
    let n = pair_results.len();
    for (i, j, s, frac, ratio) in pair_results.iter().skip(n.saturating_sub(10)) {
        eprintln!("  {:>6} {:>6} {:>8.3} {:>10.4} {:>9.1}% {:>10}",
            all_p[*i], all_p[*j], ratio, s, frac * 100.0,
            if *i == 0 { "YES" } else { "no" });
    }

    // KEY QUESTION: does p=2 ALWAYS dominate in pairs containing 2?
    let pairs_with_2: Vec<_> = pair_results.iter().filter(|(i,_,_,_,_)| *i == 0).collect();
    let min_p2_frac = pairs_with_2.iter().map(|r| r.3).fold(f64::INFINITY, f64::min);
    let max_p2_frac = pairs_with_2.iter().map(|r| r.3).fold(0.0f64, f64::max);
    eprintln!("\n  Pairs containing p=2: {} pairs", pairs_with_2.len());
    eprintln!("  p=2 fraction range: [{:.1}%, {:.1}%]", min_p2_frac * 100.0, max_p2_frac * 100.0);
    eprintln!("  ALL > 50%: {}", min_p2_frac > 0.5);

    // Does the SMALLER prime ALWAYS dominate?
    let smaller_dominates = pair_results.iter().all(|(_, _, _, frac, _)| *frac > 0.5);
    eprintln!("  Smaller prime ALWAYS dominates: {}", smaller_dominates);

    // ═══════════════════════════════════════════════════════
    // PART 2: ALL TRIPLES of first 15 primes
    // ═══════════════════════════════════════════════════════
    let n_trip = 15.min(pf.len());
    let mut triple_results: Vec<(usize, usize, usize, f64, f64)> = Vec::new();

    for i in 0..n_trip {
        for j in (i+1)..n_trip {
            for k in (j+1)..n_trip {
                let triple = vec![pf[i], pf[j], pf[k]];
                let target = (1.0/3.0) * (pf[k] / pf[i]).ln();
                if let Some(s) = solve(&triple, target) {
                    let part = energy_partition(&triple, s);
                    triple_results.push((i, j, k, s, part[0]));
                }
            }
        }
    }

    eprintln!("\n=== ALL TRIPLES of first {} primes ({} triples) ===\n",
        n_trip, triple_results.len());

    // Does smallest prime always dominate?
    let trip_min_frac = triple_results.iter().map(|r| r.4).fold(f64::INFINITY, f64::min);
    let trip_max_frac = triple_results.iter().map(|r| r.4).fold(0.0f64, f64::max);
    eprintln!("  Smallest prime fraction range: [{:.1}%, {:.1}%]",
        trip_min_frac * 100.0, trip_max_frac * 100.0);
    eprintln!("  Smallest ALWAYS dominates (>33.3%): {}", trip_min_frac > 1.0/3.0);
    eprintln!("  Smallest ALWAYS >50%: {}", trip_min_frac > 0.5);

    // Triples containing 2
    let trips_with_2: Vec<_> = triple_results.iter().filter(|(i,_,_,_,_)| *i == 0).collect();
    if !trips_with_2.is_empty() {
        let min_f = trips_with_2.iter().map(|r| r.4).fold(f64::INFINITY, f64::min);
        eprintln!("  Triples with p=2: {}, min p=2 fraction: {:.1}%",
            trips_with_2.len(), min_f * 100.0);
    }

    // Triples NOT containing 2
    let trips_no_2: Vec<_> = triple_results.iter().filter(|(i,_,_,_,_)| *i != 0).collect();
    if !trips_no_2.is_empty() {
        let min_f = trips_no_2.iter().map(|r| r.4).fold(f64::INFINITY, f64::min);
        eprintln!("  Triples WITHOUT p=2: {}, min smallest-prime fraction: {:.1}%",
            trips_no_2.len(), min_f * 100.0);
    }

    // ═══════════════════════════════════════════════════════
    // PART 3: Non-integer s — does the proof hold for all reals?
    // ═══════════════════════════════════════════════════════
    eprintln!("\n=== NON-INTEGER s: continuous verification ===\n");

    // For (2,3), sweep s from 1.01 to 20.0 in steps of 0.01
    // At each s, check: F(2,s) > F(3,s)?
    eprintln!("  For pair (2,3), checking F(2,s) > F(3,s) for s ∈ (1, 20]:");
    let mut f2_dominates_all = true;
    let mut min_ratio = f64::INFINITY;
    let mut min_ratio_s = 0.0;
    for i in 101..=2000 {
        let s = i as f64 * 0.01;
        let f2 = free_energy(2.0, s);
        let f3 = free_energy(3.0, s);
        if f2.is_finite() && f3.is_finite() {
            let ratio = f2 / f3;
            if ratio < min_ratio { min_ratio = ratio; min_ratio_s = s; }
            if ratio <= 1.0 { f2_dominates_all = false; }
        }
    }
    eprintln!("  F(2,s) > F(3,s) for ALL s ∈ (1,20]: {}", f2_dominates_all);
    eprintln!("  Min ratio F(2)/F(3) = {:.4} at s = {:.2}", min_ratio, min_ratio_s);

    // For ALL primes, check: F(2,s) > Σ_{p≥3} F(p,s)?
    eprintln!("\n  For ALL primes up to 10000, checking F(2,s) > Σ rest:");
    let mut p2_dom_all = true;
    let mut min_dom_ratio = f64::INFINITY;
    let mut min_dom_s = 0.0;
    for i in 200..=1000 { // s from 2.0 to 10.0
        let s = i as f64 * 0.01;
        let f2 = free_energy(2.0, s);
        let f_rest: f64 = pf[1..].iter().map(|&p| free_energy(p, s)).sum();
        if f2.is_finite() && f_rest.is_finite() && f_rest > 0.0 {
            let ratio = f2 / f_rest;
            if ratio < min_dom_ratio { min_dom_ratio = ratio; min_dom_s = s; }
            if ratio <= 1.0 { p2_dom_all = false; }
        }
    }
    eprintln!("  F(2,s) > Σ_{{p≥3}} F(p,s) for s ∈ [2,10]: {}", p2_dom_all);
    eprintln!("  Min ratio = {:.4} at s = {:.2}", min_dom_ratio, min_dom_s);

    // Check s < 2
    eprintln!("\n  What about s < 2?");
    for i in [101, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200] {
        let s = i as f64 * 0.01;
        let f2 = free_energy(2.0, s);
        let f_rest: f64 = pf[1..].iter().map(|&p| free_energy(p, s)).sum();
        let ratio = f2 / f_rest;
        let dom = if ratio > 1.0 { "✓" } else { "✗" };
        eprintln!("    s={:.2}: F(2)={:.6}, Σrest={:.6}, ratio={:.4} {}",
            s, f2, f_rest, ratio, dom);
    }

    // ═══════════════════════════════════════════════════════
    // PART 4: The crossover point — where does p=2 STOP dominating?
    // ═══════════════════════════════════════════════════════
    eprintln!("\n=== CROSSOVER: at what s does p=2 stop dominating? ===\n");

    // For each prime count, find the s where F(2) = Σ rest
    for &n_primes in &[2, 3, 5, 10, 50, 100, 500, 1000usize.min(pf.len())] {
        let primes_slice = &pf[..n_primes];
        let mut crossover_s = None;
        let mut prev_dom = true;

        for i in 100..=3000 {
            let s = i as f64 * 0.01;
            let f2 = free_energy(2.0, s);
            let f_rest: f64 = primes_slice[1..].iter().map(|&p| free_energy(p, s)).sum();
            let dom = f2 > f_rest;
            if prev_dom && !dom {
                // Refine
                let mut lo = (i-1) as f64 * 0.01;
                let mut hi = s;
                for _ in 0..100 {
                    let mid = (lo + hi) / 2.0;
                    let f2m = free_energy(2.0, mid);
                    let frm: f64 = primes_slice[1..].iter().map(|&p| free_energy(p, mid)).sum();
                    if f2m > frm { lo = mid; } else { hi = mid; }
                }
                crossover_s = Some((lo + hi) / 2.0);
                break;
            }
            prev_dom = dom;
        }

        match crossover_s {
            Some(sc) => eprintln!("  N={:>5} primes: p=2 dominates for s > {:.6}", n_primes, sc),
            None => eprintln!("  N={:>5} primes: p=2 ALWAYS dominates (s ∈ [1, 30])", n_primes),
        }
    }

    // ═══════════════════════════════════════════════════════
    // PART 5: Where does s* live relative to crossover?
    // ═══════════════════════════════════════════════════════
    eprintln!("\n=== KEY: is s* always ABOVE the crossover? ===\n");
    eprintln!("  If s*(N) > s_crossover(N) for all N, then at the equipartition");
    eprintln!("  point, p=2 ALWAYS dominates. That's the proof.\n");

    for &n_primes in &[2, 3, 5, 10, 20, 50, 100, 200, 500, 1000usize.min(pf.len())] {
        let primes_slice = &pf[..n_primes];
        let p_max = all_p[n_primes - 1];
        let target = (1.0 / n_primes as f64) * (p_max as f64 / 2.0).ln();

        let s_star = solve(&primes_slice.to_vec(), target);

        // Find crossover
        let mut crossover_s = None;
        let mut prev_dom = true;
        for i in 100..=3000 {
            let s = i as f64 * 0.01;
            let f2 = free_energy(2.0, s);
            let f_rest: f64 = primes_slice[1..].iter().map(|&p| free_energy(p, s)).sum();
            if prev_dom && f2 <= f_rest {
                let mut lo = (i-1) as f64 * 0.01;
                let mut hi = s;
                for _ in 0..100 {
                    let mid = (lo + hi) / 2.0;
                    let f2m = free_energy(2.0, mid);
                    let frm: f64 = primes_slice[1..].iter().map(|&p| free_energy(p, mid)).sum();
                    if f2m > frm { lo = mid; } else { hi = mid; }
                }
                crossover_s = Some((lo + hi) / 2.0);
                break;
            }
            prev_dom = f2 > f_rest;
        }

        let s_str = s_star.map(|s| format!("{:.4}", s)).unwrap_or("N/A".into());
        let c_str = crossover_s.map(|s| format!("{:.4}", s)).unwrap_or("never".into());
        let safe = match (s_star, crossover_s) {
            (Some(ss), Some(sc)) => if ss > sc { "✓ SAFE" } else { "✗ UNSAFE" },
            (Some(_), None) => "✓ ALWAYS SAFE",
            _ => "?"
        };

        eprintln!("  N={:>5}: s*={:>8}, crossover={:>8}, {}",
            n_primes, s_str, c_str, safe);
    }

    eprintln!("\n  Time: {:.3}s", t0.elapsed().as_secs_f64());
}
