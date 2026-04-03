//! # N-System Equipartition
//!
//! The 2-system equation: F(a,s) + F(b,s) = ½·ln(b/a)
//! What's the natural generalization to 3, 4, ... N systems?
//!
//! Candidate 1: Σ F(pᵢ,s) = (1/N)·ln(p_N/p₁)
//! Candidate 2: ∏ E(pᵢ,s) = (p_N/p₁)^{1/N}
//! Candidate 3: Σ F(pᵢ,s) = ½·Σ ln(pᵢ₊₁/pᵢ) (pairwise)
//!
//! What happens physically? Do N systems have richer structure?

fn free_energy(p: f64, s: f64) -> f64 {
    -(1.0 - p.powf(-s)).ln()
}

fn euler_factor(p: f64, s: f64) -> f64 {
    1.0 / (1.0 - p.powf(-s))
}

/// Solve Σ F(pᵢ,s) = target for s
fn solve_n_system(scales: &[f64], target: f64) -> Option<f64> {
    if target <= 0.0 { return None; }

    let mut lo = 0.001f64;
    let mut hi = 100.0f64;

    let f = |s: f64| -> f64 {
        scales.iter().map(|&p| free_energy(p, s)).sum::<f64>()
    };

    let f_lo = f(lo);
    let f_hi = f(hi);
    if f_hi > target || f_lo < target { return None; }

    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if f(mid) > target { lo = mid; } else { hi = mid; }
    }
    Some((lo + hi) / 2.0)
}

fn main() {
    eprintln!("==========================================================");
    eprintln!("  N-System Equipartition");
    eprintln!("  What happens with 3, 4, ... N coupled systems?");
    eprintln!("==========================================================\n");

    // ── 2 systems (baseline) ─────────────────────────────
    eprintln!("=== 2 systems: F(a,s)+F(b,s) = ½·ln(b/a) ===\n");
    let scales_2 = vec![2.0, 3.0];
    let target_2 = 0.5 * (3.0f64 / 2.0).ln();
    let s2 = solve_n_system(&scales_2, target_2).unwrap();
    eprintln!("  (2,3): s* = {:.6}, target = {:.6}", s2, target_2);

    // ── 3 systems ────────────────────────────────────────
    eprintln!("\n=== 3 systems: which generalization is natural? ===\n");

    let scales_3 = vec![2.0, 3.0, 5.0];

    // Candidate 1: (1/N)·ln(p_N/p₁) — equal-weight endpoint ratio
    let target_3a = (1.0/3.0) * (5.0f64 / 2.0).ln();
    let s3a = solve_n_system(&scales_3, target_3a);

    // Candidate 2: (1/N)·ln(∏ pᵢ₊₁/pᵢ) = (1/N)·ln(p_N/p₁) — same as 1!
    // (telescoping product)

    // Candidate 3: ½·Σ ln(pᵢ₊₁/pᵢ) — sum of pairwise half-log-ratios
    let target_3c = 0.5 * ((3.0f64/2.0).ln() + (5.0f64/3.0).ln());

    // Candidate 4: (1/2)·ln(geometric_mean_ratio)
    // geometric mean of consecutive ratios = (p_N/p₁)^{1/(N-1)}
    let geo_mean_ratio = (5.0f64 / 2.0).powf(1.0 / 2.0); // = √(5/2)
    let target_3d = 0.5 * geo_mean_ratio.ln();

    // Candidate 5: Nth root of product = geometric mean
    // ∏ E(pᵢ,s) = (p_N/p₁)^{1/N}
    // Taking log: Σ F(pᵢ,s) = (1/N)·ln(p_N/p₁) — same as candidate 1

    eprintln!("  Scales: {:?}", scales_3);
    eprintln!("  Candidate 1: (1/N)·ln(p_N/p₁) = {:.6} → s* = {:.6}",
        target_3a, s3a.unwrap_or(f64::NAN));
    eprintln!("  Candidate 3: ½·Σ ln(pᵢ₊₁/pᵢ) = {:.6} → s* = {:.6}",
        target_3c, solve_n_system(&scales_3, target_3c).unwrap_or(f64::NAN));
    eprintln!("  Candidate 4: ½·ln(geo_mean_ratio) = {:.6} → s* = {:.6}",
        target_3d, solve_n_system(&scales_3, target_3d).unwrap_or(f64::NAN));

    // ── The Euler product connection for N primes ────────
    eprintln!("\n=== Full Euler product: first N primes ===\n");
    eprintln!("  ζ(s) = ∏_p 1/(1-p⁻ˢ). What's the equipartition of ζ itself?\n");

    let all_primes: Vec<f64> = vec![2.0, 3.0, 5.0, 7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 29.0];

    eprintln!("  {:>3} {:>30} {:>12} {:>12} {:>12}",
        "N", "Primes", "target", "s*", "∏E at s*");

    for n in 2..=all_primes.len() {
        let scales = &all_primes[..n];
        let p1 = scales[0];
        let pn = scales[n-1];

        // Natural target: (1/N)·ln(p_N/p₁)
        let target = (1.0 / n as f64) * (pn as f64 / p1 as f64).ln();

        if let Some(s) = solve_n_system(scales, target) {
            let product: f64 = scales.iter().map(|&p| euler_factor(p, s)).product();
            let primes_str = if n <= 5 {
                format!("{:?}", scales)
            } else {
                format!("[2..{}]", pn as u64)
            };
            eprintln!("  {:>3} {:>30} {:>12.6} {:>12.6} {:>12.6}",
                n, primes_str, target, s, product);
        }
    }

    // ── Energy partition in N-system ─────────────────────
    eprintln!("\n=== Energy partition: who carries what? ===\n");

    for scales in [
        vec![2.0, 3.0],
        vec![2.0, 3.0, 5.0],
        vec![2.0, 3.0, 5.0, 7.0],
        vec![2.0, 3.0, 5.0, 7.0, 11.0],
    ] {
        let n = scales.len();
        let p1 = scales[0];
        let pn = *scales.last().unwrap();
        let target = (1.0 / n as f64) * (pn as f64 / p1 as f64).ln();

        if let Some(s) = solve_n_system(&scales, target) {
            let energies: Vec<f64> = scales.iter().map(|&p| free_energy(p, s)).collect();
            let total: f64 = energies.iter().sum();
            let fractions: Vec<f64> = energies.iter().map(|e| e / total * 100.0).collect();

            eprintln!("  N={}: s*={:.4}, scales={:?}", n, s, scales.iter().map(|x| *x as u64).collect::<Vec<_>>());
            for (i, (&p, &frac)) in scales.iter().zip(fractions.iter()).enumerate() {
                eprintln!("    p={:.0}: {:.1}% of energy", p, frac);
            }
            eprintln!();
        }
    }

    // ── The "all primes" limit: does s* → 1? ────────────
    eprintln!("=== Does s* approach a limit as N → ∞? ===\n");
    eprintln!("  If s* → 1, the equipartition of ALL primes is at ζ(1) = ∞ (pole).");
    eprintln!("  If s* → some finite value, that's a new characterization of the primes.\n");

    eprintln!("  {:>3} {:>12} {:>12}", "N", "s*", "Δs");
    let mut prev_s = 0.0f64;
    for n in 2..=10 {
        let scales = &all_primes[..n];
        let target = (1.0 / n as f64) * (scales.last().unwrap() / scales[0]).ln();
        if let Some(s) = solve_n_system(scales, target) {
            let delta = if prev_s > 0.0 { s - prev_s } else { 0.0 };
            eprintln!("  {:>3} {:>12.6} {:>12.6}", n, s, delta);
            prev_s = s;
        }
    }

    // ── The 3-system phase diagram ───────────────────────
    eprintln!("\n=== 3-system (2,3,5): phase diagram ===\n");
    eprintln!("  Below s*: all three independent (hot)");
    eprintln!("  At s*: triplet union");
    eprintln!("  Above s*: frozen triplet\n");

    let scales = vec![2.0, 3.0, 5.0];
    let target = (1.0 / 3.0) * (5.0f64 / 2.0).ln();
    let s_star = solve_n_system(&scales, target).unwrap();

    eprintln!("  {:>8} {:>12} {:>12} {:>12} {:>12}",
        "s", "F₂+F₃+F₅", "target", "state", "∏E");
    for &s in &[0.5, 1.0, 1.5, s_star, 2.0, 2.5, 3.0, 4.0] {
        let f_sum: f64 = scales.iter().map(|&p| free_energy(p, s)).sum();
        let product: f64 = scales.iter().map(|&p| euler_factor(p, s)).product();
        let state = if (f_sum - target).abs() < 0.001 {
            "=== UNION ==="
        } else if f_sum > target {
            "independent"
        } else {
            "frozen"
        };
        eprintln!("  {:>8.4} {:>12.6} {:>12.6} {:>12} {:>12.4}",
            s, f_sum, target, state, product);
    }

    // ── Pairwise vs triplet union ────────────────────────
    eprintln!("\n=== Pairwise unions vs triplet union ===\n");
    eprintln!("  Do pairs unite before the triplet?\n");

    let s_23 = solve_n_system(&[2.0, 3.0], 0.5 * (3.0f64/2.0).ln()).unwrap();
    let s_35 = solve_n_system(&[3.0, 5.0], 0.5 * (5.0f64/3.0).ln()).unwrap();
    let s_25 = solve_n_system(&[2.0, 5.0], 0.5 * (5.0f64/2.0).ln()).unwrap();
    let s_235 = solve_n_system(&[2.0, 3.0, 5.0], (1.0/3.0) * (5.0f64/2.0).ln()).unwrap();

    eprintln!("  s*(2,3)   = {:.4} — pair unites first", s_23);
    eprintln!("  s*(3,5)   = {:.4}", s_35);
    eprintln!("  s*(2,5)   = {:.4}", s_25);
    eprintln!("  s*(2,3,5) = {:.4} — triplet unites last", s_235);
    eprintln!();
    eprintln!("  ORDER of union (decreasing s = increasing temperature):");

    let mut unions = vec![
        ("(2,3)", s_23), ("(3,5)", s_35), ("(2,5)", s_25), ("(2,3,5)", s_235)
    ];
    unions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (i, (name, s)) in unions.iter().enumerate() {
        eprintln!("    {}. {} at s={:.4}", i+1, name, s);
    }

    eprintln!("\n  As temperature INCREASES (s decreases from ∞):");
    eprintln!("  First the closest pair unites, then more pairs,");
    eprintln!("  then the full triplet. Like nucleation in physics!");
    eprintln!("  Or harmonics emerging in music — the fifth first,");
    eprintln!("  then the third, then the full chord.");
}
