//! # Collatz Spectral Analysis
//!
//! Compute the Collatz transition matrix for n ≤ N, find its eigenvalues
//! via power iteration, measure the spectral gap directly.
//!
//! Also: verify the Euler product connection.
//! ζ(2) = π²/6 = ∏_p 1/(1-p⁻²)
//! The {2,3} factor = (4/3)(9/8) = 3/2 = the Collatz contraction.
//!
//! And: test Collatz in different bases / representations.

use std::time::Instant;

// ── Euler Product Connection ───────────────────────────────

fn euler_product_partial(s: f64, primes: &[u64]) -> f64 {
    let mut product = 1.0f64;
    for &p in primes {
        product *= 1.0 / (1.0 - (p as f64).powf(-s));
    }
    product
}

fn verify_euler_connection() {
    eprintln!("=== Euler Product Connection ===\n");

    let pi2_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
    eprintln!("  ζ(2) = π²/6 = {:.10}", pi2_6);

    // Partial products of ζ(2) = ∏_p 1/(1-p⁻²)
    let primes: Vec<u64> = vec![2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47];

    for i in 1..=primes.len().min(10) {
        let partial = euler_product_partial(2.0, &primes[..i]);
        eprintln!("  ∏_{{p≤{}}} 1/(1-p⁻²) = {:.10}  (ratio to ζ(2): {:.6})",
            primes[i-1], partial, partial / pi2_6);
    }

    // THE KEY: {2,3} factor
    let factor_23 = euler_product_partial(2.0, &[2, 3]);
    eprintln!("\n  ** {{2,3}}-Euler factor of ζ(2) = {:.10} **", factor_23);
    eprintln!("  ** Collatz contraction = 3/2 = {:.10} **", 1.5f64);
    eprintln!("  ** Match: {} **", (factor_23 - 1.5).abs() < 1e-10);

    // √(3/2) = geometric midpoint
    let sqrt_32 = (1.5f64).sqrt();
    eprintln!("\n  √(3/2) = {:.10}", sqrt_32);
    eprintln!("  = geometric mean of one odd step (×3/2) and one even step (×1)");

    // Connection to circle of fifths
    let pythagorean_comma = (1.5f64).powi(12) / (2.0f64).powi(7);
    eprintln!("\n  Circle of fifths: (3/2)^12 / 2^7 = {:.6} (Pythagorean comma)", pythagorean_comma);
    eprintln!("  12 perfect fifths overshoot 7 octaves by {:.2}%", (pythagorean_comma - 1.0) * 100.0);

    // Deeper: does √(3/2) appear in the Euler product for other s?
    eprintln!("\n  Euler {{2,3}}-factor at other s values:");
    for s in [1.5, 2.0, 3.0, 4.0, 6.0] {
        let f = euler_product_partial(s, &[2, 3]);
        eprintln!("    s={:.1}: {{2,3}}-factor = {:.6}, √ = {:.6}", s, f, f.sqrt());
    }

    // What value of s gives {2,3}-factor = √(3/2)?
    // 1/(1-2^{-s}) × 1/(1-3^{-s}) = √(3/2) ≈ 1.2247
    // Solve numerically
    eprintln!("\n  Searching for s where {{2,3}}-factor = √(3/2)...");
    let target = sqrt_32;
    let mut s_lo = 1.01f64;
    let mut s_hi = 10.0f64;
    for _ in 0..100 {
        let s_mid = (s_lo + s_hi) / 2.0;
        let f = euler_product_partial(s_mid, &[2, 3]);
        if f > target { s_lo = s_mid; } else { s_hi = s_mid; }
    }
    let s_star = (s_lo + s_hi) / 2.0;
    eprintln!("  s* = {:.10} gives {{2,3}}-factor = {:.10}", s_star, euler_product_partial(s_star, &[2, 3]));
    eprintln!("  (For reference: s=2 gives 3/2, s*≈{:.4} gives √(3/2))", s_star);
}

// ── Collatz Transition Matrix ──────────────────────────────

/// Build the Collatz transition matrix for n = 1..N
/// T[i][j] = 1 if Collatz(j+1) = i+1, else 0
/// (using 0-indexed internally, 1-indexed for Collatz)
fn build_transition_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut t = vec![vec![0.0f64; n]; n];

    for j in 0..n {
        let num = j + 1; // 1-indexed
        let next = if num == 1 {
            1 // 1 → 4 → 2 → 1, but we treat 1 as absorbing
        } else if num % 2 == 0 {
            num / 2
        } else {
            3 * num + 1
        };

        if next >= 1 && next <= n {
            t[next - 1][j] = 1.0;
        }
        // If next > n, the column has no 1 (trajectory escapes our window)
    }

    t
}

/// Power iteration to find dominant eigenvalue/eigenvector
fn power_iteration(matrix: &[Vec<f64>], max_iter: usize) -> (f64, Vec<f64>) {
    let n = matrix.len();
    let mut v = vec![1.0 / (n as f64).sqrt(); n];
    let mut eigenvalue = 0.0f64;

    for _ in 0..max_iter {
        // Matrix-vector multiply
        let mut w = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                w[i] += matrix[i][j] * v[j];
            }
        }

        // Find magnitude
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 { break; }

        eigenvalue = norm;
        for i in 0..n { v[i] = w[i] / norm; }
    }

    (eigenvalue, v)
}

/// Deflated power iteration for second eigenvalue
fn second_eigenvalue(matrix: &[Vec<f64>], v1: &[f64], lambda1: f64, max_iter: usize) -> f64 {
    let n = matrix.len();

    // Deflate: A' = A - λ₁ v₁ v₁ᵀ
    let mut deflated = matrix.to_vec();
    for i in 0..n {
        for j in 0..n {
            deflated[i][j] -= lambda1 * v1[i] * v1[j];
        }
    }

    let (lambda2, _) = power_iteration(&deflated, max_iter);
    lambda2
}

fn spectral_analysis() {
    eprintln!("\n=== Collatz Spectral Analysis ===\n");

    for &n in &[50usize, 100, 200, 500, 1000] {
        let t0 = Instant::now();
        let matrix = build_transition_matrix(n);

        // Count non-zero entries (sparsity)
        let nnz: usize = matrix.iter().flat_map(|row| row.iter()).filter(|&&x| x != 0.0).count();
        let sparsity = 1.0 - nnz as f64 / (n * n) as f64;

        let (lambda1, v1) = power_iteration(&matrix, 1000);
        let lambda2 = second_eigenvalue(&matrix, &v1, lambda1, 1000);

        let gap = 1.0 - lambda2 / lambda1;
        let elapsed = t0.elapsed().as_secs_f64();

        eprintln!("  N={:4}: λ₁={:.4}, λ₂={:.4}, gap={:.4}, sparsity={:.3}, time={:.3}s",
            n, lambda1, lambda2, gap, sparsity, elapsed);
    }
}

// ── Collatz in Different Bases ─────────────────────────────

fn collatz_base_analysis() {
    eprintln!("\n=== Collatz in Different Bases ===\n");

    // In base 3: n mod 3 determines behavior
    // But Collatz doesn't use base 3 — it uses base 2.
    // However, we can analyze the TERNARY representation of Collatz trajectories.

    // For each n, compute Collatz trajectory and measure:
    // - Binary (base 2) digit density of trajectory values
    // - Ternary (base 3) digit density
    // - Base 6 digit density
    // See if any base gives cleaner structure.

    eprintln!("  Digit density of Collatz trajectory values in different bases:");
    eprintln!("  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}  {:>8}", "start", "steps", "b2_dens", "b3_dens", "b6_dens", "phi_est");

    for start in [27u64, 97, 871, 6171, 77031, 837799] {
        let mut n = start;
        let mut steps = 0u64;
        let mut sum_b2_density = 0.0f64;
        let mut sum_b3_density = 0.0f64;
        let mut sum_b6_density = 0.0f64;
        let mut count = 0u64;

        while n > 1 && steps < 10000 {
            // Base 2 density (fraction of 1-bits)
            let b2_bits = 64 - n.leading_zeros();
            let b2_ones = n.count_ones();
            let b2_dens = b2_ones as f64 / b2_bits as f64;

            // Base 3 density (average digit / 2, normalized to [0,1])
            let b3_dens = base_digit_density(n, 3);

            // Base 6 density
            let b6_dens = base_digit_density(n, 6);

            sum_b2_density += b2_dens;
            sum_b3_density += b3_dens;
            sum_b6_density += b6_dens;
            count += 1;

            if n % 2 == 0 { n /= 2; } else { n = 3 * n + 1; }
            steps += 1;
        }

        let avg_b2 = sum_b2_density / count as f64;
        let avg_b3 = sum_b3_density / count as f64;
        let avg_b6 = sum_b6_density / count as f64;

        // Golden ratio estimate: φ ≈ b2_dens * 2 + something?
        // Actually, just measure if any base gives density closer to 0.5
        let phi_est = avg_b2 / avg_b3; // ratio of base-2 to base-3 density

        eprintln!("  {:>8}  {:>8}  {:>8.4}  {:>8.4}  {:>8.4}  {:>8.4}",
            start, steps, avg_b2, avg_b3, avg_b6, phi_est);
    }
}

fn base_digit_density(mut n: u64, base: u64) -> f64 {
    if n == 0 { return 0.0; }
    let mut sum = 0u64;
    let mut digits = 0u64;
    while n > 0 {
        sum += n % base;
        n /= base;
        digits += 1;
    }
    // Normalize: max digit is (base-1), so density = sum / (digits * (base-1))
    sum as f64 / (digits as f64 * (base - 1) as f64)
}

// ── Convolution / Diffusion View ───────────────────────────

fn diffusion_analysis() {
    eprintln!("\n=== Collatz as Diffusion ===\n");

    // Start with uniform density on [1, N]
    // Apply Collatz transition repeatedly
    // Watch density concentrate toward {1, 2, 4}

    let n = 1000usize;
    let mut density = vec![1.0 / n as f64; n];

    let matrix = build_transition_matrix(n);

    eprintln!("  Starting with uniform density on [1, {}]", n);
    eprintln!("  {:>4}  {:>10}  {:>10}  {:>10}  {:>10}  {:>10}",
        "step", "ρ(1)", "ρ(2)", "ρ(4)", "entropy", "max_ρ");

    for step in 0..=50 {
        // Entropy
        let entropy: f64 = -density.iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| p * p.ln())
            .sum::<f64>();

        let max_rho = density.iter().cloned().fold(0.0f64, f64::max);

        if step % 5 == 0 || step <= 5 {
            eprintln!("  {:>4}  {:>10.6}  {:>10.6}  {:>10.6}  {:>10.4}  {:>10.6}",
                step, density[0], density[1], density[3], entropy, max_rho);
        }

        // Apply transition: ρ' = T · ρ
        let mut new_density = vec![0.0f64; n];
        for i in 0..n {
            for j in 0..n {
                new_density[i] += matrix[i][j] * density[j];
            }
        }

        // Re-normalize (some density escapes past N)
        let total: f64 = new_density.iter().sum();
        if total > 0.0 {
            for d in &mut new_density { *d /= total; }
        }

        density = new_density;
    }

    eprintln!("\n  If entropy decreases monotonically → diffusion is contractive (concentrating)");
    eprintln!("  The rate of entropy decrease IS the spectral gap");
}

fn main() {
    eprintln!("==========================================================");
    eprintln!("  Collatz: Spectral + Euler + Diffusion Analysis");
    eprintln!("  Connecting Collatz ↔ Riemann ↔ Music ↔ Tambear");
    eprintln!("==========================================================\n");

    verify_euler_connection();
    spectral_analysis();
    collatz_base_analysis();
    diffusion_analysis();
}
