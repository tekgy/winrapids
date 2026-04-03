//! # p=2 Dominance: Analytical Proof + Large-N Verification
//!
//! Claim: at the equipartition point s*(N), F(2,s*) > Σ_{p≥3} F(p,s*)
//! i.e., prime 2 carries more than half the total free energy.
//!
//! Proof strategy:
//! 1. F(p,s) = Σ_{k≥1} p^{-ks}/k  (power series)
//! 2. The leading term p^{-s} dominates for large s
//! 3. 2^{-s} > sum of all other primes' p^{-s} terms — this is the key inequality
//! 4. Bound the prime tail using the prime counting function
//!
//! Also: push N to 1000+ primes to see if dominance holds computationally.

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
    -((1.0 - x).ln())
}

/// Solve Σ F(pᵢ,s) = target via sweep + bisect
fn solve(primes: &[f64], target: f64) -> Option<f64> {
    if target <= 0.0 { return None; }
    let f_at = |s: f64| -> f64 {
        primes.iter().map(|&p| free_energy(p, s)).sum()
    };

    let mut prev = f64::INFINITY;
    for i in 1..5000 {
        let s = i as f64 * 0.01;
        let f = f_at(s);
        if !f.is_finite() { prev = f; continue; }
        if prev > target && f <= target {
            let mut lo = (i - 1) as f64 * 0.01;
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

fn main() {
    eprintln!("==========================================================");
    eprintln!("  p=2 DOMINANCE: Proof Attempt + Large-N Verification");
    eprintln!("==========================================================\n");

    let all_primes = sieve_primes(100_000); // 9592 primes up to 100K
    eprintln!("  Sieved {} primes up to {}\n", all_primes.len(), all_primes.last().unwrap());

    // ── Part 1: Large-N sweep ────────────────────────────
    eprintln!("=== Part 1: s* and p=2 dominance for N = 2..10000 ===\n");
    eprintln!("  {:>6} {:>8} {:>10} {:>8} {:>8} {:>8} {:>8}",
        "N", "p_max", "s*", "p2 %", "p3 %", "p5 %", "rest %");

    let mut results: Vec<(usize, f64, f64)> = Vec::new(); // (N, s*, p2_frac)

    let test_ns: Vec<usize> = {
        let mut v: Vec<usize> = (2..=20).collect();
        v.extend([25, 30, 40, 50, 75, 100, 150, 200, 300, 500, 750, 1000,
                   1500, 2000, 3000, 5000, 7500].iter());
        v.retain(|&n| n <= all_primes.len());
        v
    };

    for &n in &test_ns {
        let primes_f: Vec<f64> = all_primes[..n].iter().map(|&p| p as f64).collect();
        let p_max = all_primes[n-1];
        let target = (1.0 / n as f64) * (p_max as f64 / 2.0).ln();

        if let Some(s) = solve(&primes_f, target) {
            let total: f64 = primes_f.iter().map(|&p| free_energy(p, s)).sum();
            let f2 = free_energy(2.0, s);
            let f3 = free_energy(3.0, s);
            let f5 = free_energy(5.0, s);
            let p2_pct = f2 / total * 100.0;
            let p3_pct = f3 / total * 100.0;
            let p5_pct = f5 / total * 100.0;
            let rest_pct = 100.0 - p2_pct - p3_pct - p5_pct;

            results.push((n, s, p2_pct / 100.0));

            if n <= 20 || test_ns.contains(&n) {
                eprintln!("  {:>6} {:>8} {:>10.4} {:>7.1}% {:>7.1}% {:>7.1}% {:>7.1}%",
                    n, p_max, s, p2_pct, p3_pct, p5_pct, rest_pct);
            }
        }
    }

    // ── Part 2: Does p2 dominance converge? ──────────────
    eprintln!("\n=== Part 2: Convergence of p=2 fraction ===\n");

    if results.len() >= 3 {
        let last = &results[results.len()-3..];
        eprintln!("  Last 3 measurements:");
        for (n, s, p2) in last {
            eprintln!("    N={}: s*={:.4}, p2={:.1}%", n, s, p2 * 100.0);
        }

        let p2_min = results.iter().map(|r| r.2).fold(f64::INFINITY, f64::min);
        let p2_max = results.iter().map(|r| r.2).fold(0.0f64, f64::max);
        eprintln!("\n  p=2 fraction range: [{:.1}%, {:.1}%]", p2_min * 100.0, p2_max * 100.0);
        eprintln!("  ALWAYS > 50%: {}", p2_min > 0.5);
        eprintln!("  ALWAYS > 60%: {}", p2_min > 0.6);
    }

    // ── Part 3: Analytical bound ─────────────────────────
    eprintln!("\n=== Part 3: Analytical Proof Attempt ===\n");
    eprintln!("  CLAIM: F(2,s) > Σ_{{p≥3}} F(p,s) for all s in the equipartition range.\n");
    eprintln!("  F(p,s) = -ln(1 - p^{{-s}}) = Σ_{{k≥1}} p^{{-ks}}/k\n");
    eprintln!("  Leading term: F(p,s) ≈ p^{{-s}} for large s.\n");

    // The key inequality at s=2.5 (typical equipartition):
    let s = 2.5;
    eprintln!("  At s={}: ", s);
    let f2 = free_energy(2.0, s);
    eprintln!("    F(2,{}) = {:.10}", s, f2);

    // Sum over ALL primes ≥ 3
    let f_rest: f64 = all_primes[1..].iter().map(|&p| free_energy(p as f64, s)).sum();
    eprintln!("    Σ F(p≥3,{}) = {:.10} (over {} primes)", s, f_rest, all_primes.len() - 1);
    eprintln!("    Ratio F(2)/Σrest = {:.4}", f2 / f_rest);
    eprintln!("    p=2 fraction: {:.1}%", f2 / (f2 + f_rest) * 100.0);

    // Now the ANALYTICAL bound:
    // F(p,s) < p^{-s}/(1 - p^{-s}) for each p (geometric series bound)
    // Σ_{p≥3} F(p,s) < Σ_{p≥3} p^{-s}/(1-p^{-s})
    //                 < Σ_{n≥3} n^{-s}/(1-n^{-s})  (more terms = upper bound)
    //                 < Σ_{n≥3} n^{-s} · (1/(1-3^{-s}))  (since n^{-s} ≤ 3^{-s} for n≥3)

    // Actually: 1/(1-p^{-s}) ≤ 1/(1-3^{-s}) for p≥3
    // So Σ_{p≥3} F(p,s) < (1/(1-3^{-s})) · Σ_{p≥3} p^{-s}

    // And Σ_{p≥3} p^{-s} < Σ_{n≥3} n^{-s} = ζ(s) - 1 - 2^{-s}

    // So: Σ_{p≥3} F(p,s) < (ζ(s) - 1 - 2^{-s}) / (1 - 3^{-s})

    // For the dominance: need F(2,s) > this bound
    // F(2,s) = -ln(1-2^{-s})
    // Need: -ln(1-2^{-s}) > (ζ(s) - 1 - 2^{-s}) / (1 - 3^{-s})

    eprintln!("\n  ANALYTICAL BOUND:");
    eprintln!("    Σ_{{p≥3}} F(p,s) < (ζ(s) - 1 - 2^{{-s}}) / (1 - 3^{{-s}})");
    eprintln!("    (using Σ over primes < Σ over integers)\n");

    for &s in &[2.0, 2.3, 2.5, 2.8, 3.0, 3.5, 4.0, 5.0] {
        let f2 = free_energy(2.0, s);

        // Compute ζ(s) approximately
        let zeta_s: f64 = (1..=100000).map(|n| (n as f64).powf(-s)).sum();
        let bound = (zeta_s - 1.0 - 2.0f64.powf(-s)) / (1.0 - 3.0f64.powf(-s));

        let ratio = f2 / bound;
        let dominance = ratio > 1.0;

        eprintln!("    s={:.1}: F(2)={:.6}, bound={:.6}, ratio={:.4} {}",
            s, f2, bound, ratio,
            if dominance { "✓ p=2 DOMINATES" } else { "✗ bound too loose" });
    }

    // ── Part 4: The TIGHTER analytical bound ─────────────
    eprintln!("\n  TIGHTER BOUND (sum over primes only, not all integers):");
    eprintln!("    Use prime counting: π(x) < 1.26·x/ln(x) (Rosser-Schoenfeld)\n");

    // Σ_{p≥3} p^{-s} = Σ_{p≥3} p^{-s}
    // By partial summation with π(x):
    // Σ_{p≥3} p^{-s} = s·∫₃^∞ π(x)·x^{-s-1} dx
    // < s·∫₃^∞ (1.26·x/ln(x))·x^{-s-1} dx
    // = 1.26·s·∫₃^∞ x^{-s}/ln(x) dx
    // This integral is bounded by ∫₃^∞ x^{-s}/ln(3) dx = 3^{1-s}/((s-1)·ln(3))

    for &s in &[2.0, 2.3, 2.5, 2.8, 3.0, 4.0] {
        let f2 = free_energy(2.0, s);

        // Exact sum over first 9592 primes
        let exact_rest: f64 = all_primes[1..].iter()
            .map(|&p| free_energy(p as f64, s)).sum();

        // Rosser-Schoenfeld bound
        let rs_bound = 1.26 * s * 3.0f64.powf(1.0 - s) / ((s - 1.0) * 3.0f64.ln());
        let rs_full_bound = rs_bound / (1.0 - 3.0f64.powf(-s)); // account for higher-order terms

        eprintln!("    s={:.1}: F(2)={:.6}, exact_rest={:.6}, RS_bound={:.6}, ratio(exact)={:.3}, ratio(bound)={:.3}",
            s, f2, exact_rest, rs_full_bound, f2/exact_rest, f2/rs_full_bound);
    }

    // ── Part 5: The simplest true statement ──────────────
    eprintln!("\n=== Part 5: The Simplest True Statement ===\n");
    eprintln!("  THEOREM: For all s ≥ 2, F(2,s) > Σ_{{p prime, p≥3}} F(p,s).");
    eprintln!();
    eprintln!("  PROOF SKETCH:");
    eprintln!("    F(p,s) = Σ_{{k≥1}} p^{{-ks}}/k");
    eprintln!("    F(2,s) = 2^{{-s}} + 2^{{-2s}}/2 + 2^{{-3s}}/3 + ...");
    eprintln!("    Σ_{{p≥3}} F(p,s) < Σ_{{n≥3}} n^{{-s}} + Σ_{{n≥3}} n^{{-2s}}/2 + ...");
    eprintln!("                     = Σ_{{k≥1}} (ζ(ks) - 1 - 2^{{-ks}})/k");
    eprintln!();
    eprintln!("    The claim: 2^{{-s}} + 2^{{-2s}}/2 + ... > Σ_{{k≥1}} (ζ(ks)-1-2^{{-ks}})/k");
    eprintln!("    i.e.: Σ_{{k≥1}} 2^{{-ks}}/k > Σ_{{k≥1}} (ζ(ks)-1-2^{{-ks}})/k");
    eprintln!("    i.e.: Σ_{{k≥1}} (2·2^{{-ks}} - ζ(ks) + 1)/k > 0");
    eprintln!();

    // Verify term by term
    eprintln!("    Term-by-term check at s=2.5:");
    let s = 2.5;
    let mut total = 0.0f64;
    for k in 1..=10u32 {
        let ks = k as f64 * s;
        let zeta_ks: f64 = (1..=100000).map(|n| (n as f64).powf(-ks)).sum();
        let term = (2.0 * 2.0f64.powf(-ks) - zeta_ks + 1.0) / k as f64;
        total += term;
        eprintln!("      k={}: 2·2^{{-{}}} = {:.8}, ζ({})={:.8}, term/k = {:.8}, running sum = {:.8}",
            k, ks, 2.0 * 2.0f64.powf(-ks), ks, zeta_ks, term, total);
    }

    eprintln!("\n    The sum is {} for the first 10 terms.",
        if total > 0.0 { "POSITIVE ✓" } else { "NEGATIVE ✗" });
    eprintln!("    If positive: p=2 dominance is PROVEN for s=2.5.");
    eprintln!("    Need: prove positivity for all s in [{:.1}, {:.1}].", 2.3, 2.8);
}
