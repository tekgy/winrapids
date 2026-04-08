/// Scale Ladder Benchmark: Kernel Density Estimation
///
/// Measures tambear's naive O(n*m) KDE at increasing scales to find the
/// breaking point. This motivates the FFT-KDE build item.
///
/// Current KDE: O(n*m) where n=data points, m=eval points.
/// FFT-KDE (TODO): O(n + m*log(m)) — would handle 100M+ trivially.
///
/// Run with: cargo test --release --test scale_ladder_kde -- --nocapture

use tambear::{kde, KernelType};
use std::time::Instant;

/// Deterministic pseudo-random f64 via xorshift64.
fn generate_f64_data(n: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut data = Vec::with_capacity(n);
    for _ in 0..n {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u1 = (state as f64) / (u64::MAX as f64);
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let u2 = (state as f64) / (u64::MAX as f64);
        let v = (u1 + u2 - 1.0) * 3.46; // roughly N(0,1)
        data.push(v);
    }
    data
}

#[test]
fn scale_ladder_kde_naive() {
    let m = 1000; // fixed number of evaluation points
    let eval_points: Vec<f64> = (0..m).map(|i| -4.0 + 8.0 * (i as f64) / (m as f64 - 1.0)).collect();
    let bw = Some(0.5); // fixed bandwidth for fair comparison

    let scales: Vec<(&str, usize)> = vec![
        ("100",     100),
        ("1K",      1_000),
        ("10K",     10_000),
        ("100K",    100_000),
        ("1M",      1_000_000),
        ("10M",     10_000_000),
    ];

    println!();
    println!("=========================================================================");
    println!("SCALE LADDER: KDE (tambear naive O(n*m), m={} eval points)", m);
    println!("=========================================================================");
    println!("{:>8}  {:>10}  {:>10}  {:>12}  {:>8}",
             "Scale", "Alloc(s)", "KDE(s)", "Ops(n*m)", "MB");
    println!("-------------------------------------------------------------------------");

    for (label, n) in &scales {
        // Allocation
        let t0 = Instant::now();
        let data = generate_f64_data(*n, 42);
        let t_alloc = t0.elapsed().as_secs_f64();

        let mb = (*n * 8) as f64 / 1e6;
        let ops = (*n as f64) * (m as f64);

        // KDE computation
        let t1 = Instant::now();
        let density = kde(&data, &eval_points, KernelType::Gaussian, bw);
        let t_kde = t1.elapsed().as_secs_f64();

        // Sanity: density should be positive and roughly integrate to ~1
        let dx = 8.0 / (m as f64 - 1.0);
        let integral: f64 = density.iter().sum::<f64>() * dx;

        println!("{:>8}  {:>10.4}  {:>10.4}  {:>12.0}  {:>8.1}",
                 label, t_alloc, t_kde, ops, mb);
        println!("         peak_density={:.6}, integral~={:.4}, throughput={:.0} pts/s",
                 density.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                 integral,
                 ops / t_kde);
    }

    println!("=========================================================================");
    println!();
    println!("NOTE: Current KDE is O(n*m) — each eval point scans all data.");
    println!("      At 10M data × 1K eval = 10B kernel evaluations.");
    println!("      FFT-KDE (build item) would be O(n + m*log(m)):");
    println!("        1. Bin data into m-point histogram: O(n)");
    println!("        2. FFT the histogram: O(m*log(m))");
    println!("        3. Multiply by kernel FFT: O(m)");
    println!("        4. Inverse FFT: O(m*log(m))");
    println!("      Expected: 100M data in <1s vs current ~100s.");
}
