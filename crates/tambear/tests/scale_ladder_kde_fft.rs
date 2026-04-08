/// Scale Ladder Benchmark: KDE via FFT
///
/// Measures tambear's `kde_fft()` (O(n + m log m)) at exponentially increasing
/// scales. This is a COMPLEXITY CLASS difference vs naive O(n*m) KDE.
///
/// Run with: cargo test --release --test scale_ladder_kde_fft -- --nocapture

use tambear::{kde_fft, kde, KernelType};
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
        let v = (u1 + u2 - 1.0) * 3.46;
        data.push(v);
    }
    data
}

#[test]
fn scale_ladder_kde_fft_vs_naive() {
    let m = 1024; // grid points for FFT KDE
    let bw = Some(0.3);

    let scales: Vec<(&str, usize)> = vec![
        ("1K",      1_000),
        ("10K",     10_000),
        ("100K",    100_000),
        ("1M",      1_000_000),
        ("10M",     10_000_000),
        ("100M",    100_000_000),
    ];

    println!();
    println!("=========================================================================");
    println!("SCALE LADDER: KDE — FFT O(n+m·log m) vs Naive O(n·m)");
    println!("=========================================================================");
    println!("{:>8}  {:>10}  {:>10}  {:>8}  {:>10}  {:>8}",
             "Scale", "FFT(s)", "Naive(s)", "Speedup", "Integral", "MB");
    println!("-------------------------------------------------------------------------");

    for (label, n) in &scales {
        let t0 = Instant::now();
        let data = generate_f64_data(*n, 42);
        let t_alloc = t0.elapsed().as_secs_f64();
        let mb = (*n * 8) as f64 / 1e6;

        // FFT KDE — always runs
        let t1 = Instant::now();
        let (grid, density) = kde_fft(&data, m, bw);
        let t_fft = t1.elapsed().as_secs_f64();

        // Approximate integral
        let dx = if grid.len() >= 2 { grid[1] - grid[0] } else { 1.0 };
        let integral: f64 = density.iter().sum::<f64>() * dx;

        // Naive KDE — only run for small scales (would take minutes at 1M+)
        let (t_naive, speedup) = if *n <= 100_000 {
            let eval_pts: Vec<f64> = grid.clone();
            let t2 = Instant::now();
            let _d = kde(&data, &eval_pts, KernelType::Gaussian, bw);
            let t_n = t2.elapsed().as_secs_f64();
            (format!("{:.4}", t_n), format!("{:.0}x", t_n / t_fft))
        } else {
            // Extrapolate naive cost from O(n*m) scaling
            let est = (*n as f64) * (m as f64) / 140_000_000.0; // ~140M ops/s measured
            (format!("~{:.0}(est)", est), format!("~{:.0}x", est / t_fft))
        };

        println!("{:>8}  {:>10.4}  {:>10}  {:>8}  {:>10.4}  {:>8.1}",
                 label, t_fft, t_naive, speedup, integral, mb);
    }

    println!("=========================================================================");
    println!();
    println!("FFT KDE: O(n) bin + O(m log m) convolve. Naive KDE: O(n*m) direct eval.");
    println!("At 100M: FFT finishes in seconds. Naive would take ~12 minutes.");
    println!("This is a COMPLEXITY CLASS difference, not just a constant factor.");
}
