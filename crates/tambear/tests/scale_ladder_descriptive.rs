/// Scale Ladder Benchmark: Descriptive Statistics
///
/// Measures tambear's `moments_ungrouped` at exponentially increasing scales.
/// Compare with: research/gold_standard/scale_ladder_descriptive.py (numpy/scipy)
///
/// Run with: cargo test --release --test scale_ladder_descriptive -- --nocapture

use tambear::descriptive::moments_ungrouped;
use std::time::Instant;

/// Simple LCG for deterministic pseudo-random f64 generation.
/// NOT cryptographic — just fast and reproducible.
fn generate_f64_data(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        // xorshift64
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        // Convert to f64 in roughly [-4, 4] (approximate normal via central limit)
        let u1 = (state as f64) / (u64::MAX as f64); // [0, 1]
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u2 = (state as f64) / (u64::MAX as f64);
        // Box-Muller-ish: just use the sum for speed
        let v = (u1 + u2 - 1.0) * 3.46; // roughly N(0,1)
        data.push(v);
    }
    data
}

#[test]
fn scale_ladder_descriptive_stats() {
    let scales: Vec<(&str, usize)> = vec![
        ("10", 10),
        ("1K", 1_000),
        ("100K", 100_000),
        ("1M", 1_000_000),
        ("10M", 10_000_000),
        ("100M", 100_000_000),
        ("1B", 1_000_000_000),
    ];

    println!();
    println!("================================================================");
    println!("SCALE LADDER: Descriptive Statistics (tambear moments_ungrouped)");
    println!("================================================================");
    println!("{:>8}  {:>10}  {:>10}  {:>10}  {:>8}",
             "Scale", "Alloc(s)", "Compute(s)", "Total(s)", "MB");
    println!("----------------------------------------------------------------");

    for (label, n) in &scales {
        // Allocation
        let t0 = Instant::now();
        let data = generate_f64_data(*n, 42);
        let t_alloc = t0.elapsed().as_secs_f64();

        let mb = (*n * 8) as f64 / 1e6;

        // Compute all moments in one pass
        let t1 = Instant::now();
        let stats = moments_ungrouped(&data);
        let t_compute = t1.elapsed().as_secs_f64();

        let t_total = t_alloc + t_compute;

        println!("{:>8}  {:>10.4}  {:>10.4}  {:>10.4}  {:>8.1}",
                 label, t_alloc, t_compute, t_total, mb);
        println!("         mean={:.6}, std={:.6}, skew={:.6}, kurt={:.6}",
                 stats.mean(),
                 stats.variance(0).sqrt(),
                 stats.skewness(true),
                 stats.kurtosis(true, true));
    }

    println!("================================================================");
    println!();
    println!("NOTE: tambear computes mean+std+skew+kurt in a SINGLE 2-pass scan.");
    println!("      scipy requires 4 separate passes (mean, std, skew, kurtosis).");
    println!("      Expected speedup: ~4x from pass reduction alone.");
}
