//! Two quick experiments:
//! 1. Richardson extrapolation on cascade coverage sequence → k=∞ limit
//! 2. Base-3 digit density of 3^k for large k → CA ergodicity test

fn main() {
    eprintln!("=== Richardson Extrapolation on Coverage Sequence ===\n");

    // Coverage data from formal proof generator (fraction of proven decreasing)
    let data: Vec<(u32, f64)> = vec![
        (4,  0.7500),
        (8,  0.8867),
        (10, 0.8730),
        (12, 0.8628),
        (14, 0.8547),
        (16, 0.9246),
        (18, 0.9169),
        (20, 0.9102),
    ];

    eprintln!("  Raw coverage:");
    for &(k, c) in &data {
        eprintln!("    k={:>2}: {:.4}", k, c);
    }

    // Richardson extrapolation: assume coverage ~ L + a/2^k + b/4^k + ...
    // Using consecutive pairs for first-order Richardson
    eprintln!("\n  First-order Richardson (pairs):");
    for i in 0..data.len()-1 {
        let (k1, c1) = data[i];
        let (k2, c2) = data[i+1];
        // If c ~ L + a·r^k where r = 2^{-2} (since k steps by 2):
        // Richardson: L ≈ (c2·r^{k1} - c1·r^{k2}) / (r^{k1} - r^{k2})
        // Simpler: assume c ~ L + a·2^{-k}
        // L = (c2·2^{k2} - c1·2^{k1}) / (2^{k2} - 2^{k1})
        // But this doesn't work well with non-uniform spacing.

        // Try: Aitken on consecutive triples
        if i + 2 < data.len() {
            let (_, c0) = data[i];
            let (_, c1) = data[i+1];
            let (_, c2) = data[i+2];
            let denom = c2 - 2.0 * c1 + c0;
            if denom.abs() > 1e-15 {
                let aitken = c0 - (c1 - c0).powi(2) / denom;
                eprintln!("    Aitken from k={},{},{}: limit ≈ {:.4}",
                    data[i].0, data[i+1].0, data[i+2].0, aitken);
            }
        }
    }

    // Wynn epsilon on the even-k subsequence (more regular)
    let even_k: Vec<f64> = data.iter()
        .filter(|&&(k, _)| k % 4 == 0)
        .map(|&(_, c)| c)
        .collect();

    eprintln!("\n  Even-k subsequence: {:?}", even_k);
    if even_k.len() >= 3 {
        // Aitken on even subsequence
        let c0 = even_k[0];
        let c1 = even_k[1];
        let c2 = even_k[2];
        let denom = c2 - 2.0 * c1 + c0;
        if denom.abs() > 1e-15 {
            let aitken = c0 - (c1 - c0).powi(2) / denom;
            eprintln!("  Aitken on even-k: limit ≈ {:.4}", aitken);
        }
    }

    // The odd-decreasing fraction (more fundamental)
    let odd_data: Vec<(u32, f64)> = vec![
        (4,  0.5000),
        (8,  0.7734),
        (10, 0.7461),
        (12, 0.7256),
        (14, 0.7095),
        (16, 0.8491),
        (18, 0.8338),
        (20, 0.8204),
    ];

    eprintln!("\n  Odd decreasing fraction:");
    for &(k, c) in &odd_data {
        eprintln!("    k={:>2}: {:.4}", k, c);
    }

    // The sequence from k=16,18,20 is most regular (post-peak)
    let post_peak: Vec<f64> = vec![0.8491, 0.8338, 0.8204];
    let c0 = post_peak[0];
    let c1 = post_peak[1];
    let c2 = post_peak[2];
    let denom = c2 - 2.0 * c1 + c0;
    if denom.abs() > 1e-15 {
        let aitken = c0 - (c1 - c0).powi(2) / denom;
        eprintln!("  Aitken on post-peak (k=16,18,20): odd fraction limit ≈ {:.4}", aitken);
        eprintln!("  → total coverage limit ≈ {:.4}", 0.5 + aitken / 2.0);
    }

    // Linear extrapolation on the post-peak trend
    let slope = (c2 - c0) / 4.0; // per 2 units of k
    let intercept = c0 - slope * 16.0;
    eprintln!("  Linear fit (k=16-20): odd_fraction = {:.4} + {:.6}·k",
        intercept, slope);
    let k_zero = -intercept / slope;
    eprintln!("  Odd fraction hits 0 at k ≈ {:.0}", k_zero);
    eprintln!("  At k=100: odd_fraction ≈ {:.4}", intercept + slope * 100.0);

    // ─── Base-3 digit density of 3^k ───────────────────────
    eprintln!("\n\n=== Base-3 Digit Density of 3^k ===\n");
    eprintln!("  In base 3, 3^k = 10...0 (1 followed by k zeros).");
    eprintln!("  So the base-3 digit density of 3^k is trivially 1/(k+1).");
    eprintln!("  This converges to 0, not 0.5 — base-3 is the WRONG base");
    eprintln!("  for testing normality of powers of 3!");

    // What about 3^k - 1? In base 3: 3^k - 1 = 222...2 (k twos).
    // Digit density = k·2 / (k·2) = 1.0 (max density). Also not useful.

    // What about base-2 digit density of 3^k?
    // THIS is the normality question.
    eprintln!("\n  Base-2 digit density of 3^k:");
    eprintln!("  {:>4}  {:>10}  {:>6}  {:>6}  {:>8}", "k", "3^k", "bits", "ones", "density");

    for k in 1..=40u32 {
        let val = 3u128.pow(k);
        let bits = 128 - val.leading_zeros();
        let ones = val.count_ones();
        let density = ones as f64 / bits as f64;
        if k <= 20 || k % 5 == 0 {
            eprintln!("  {:>4}  {:>10}  {:>6}  {:>6}  {:>8.4}", k, val, bits, ones, density);
        }
    }

    // Compute running average of base-2 digit density
    eprintln!("\n  Running average of base-2 digit density of 3^k:");
    let mut sum_density = 0.0f64;
    for k in 1..=80u32 {
        let val = 3u128.pow(k);
        let bits = 128 - val.leading_zeros();
        let ones = val.count_ones();
        let density = ones as f64 / bits as f64;
        sum_density += density;
        let avg = sum_density / k as f64;
        if k % 10 == 0 || k <= 5 {
            eprintln!("    k=1..{:>3}: avg density = {:.6}", k, avg);
        }
    }

    // What about base-6?
    eprintln!("\n  Base-6 digit density of 3^k:");
    for k in [1u32, 5, 10, 20, 30, 40] {
        let val = 3u128.pow(k);
        let (digits, digit_sum) = base_digit_info(val, 6);
        let density = digit_sum as f64 / (digits as f64 * 5.0); // max digit = 5
        eprintln!("    k={:>3}: {} base-6 digits, sum={}, density={:.4}",
            k, digits, digit_sum, density);
    }
}

fn base_digit_info(mut n: u128, base: u128) -> (u32, u128) {
    if n == 0 { return (1, 0); }
    let mut digits = 0u32;
    let mut sum = 0u128;
    while n > 0 {
        sum += n % base;
        n /= base;
        digits += 1;
    }
    (digits, sum)
}
