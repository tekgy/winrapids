//! # Equipartition Sweep
//!
//! The equation: F(a,s) + F(b,s) = ½·ln(b/a)
//! where F(p,s) = -ln(1 - p^{-s})
//!
//! What happens when we vary (a,b) beyond primes {2,3}?
//! - Sequential vs non-sequential?
//! - Primes vs composites vs irrationals?
//! - Music theory intervals?
//! - Negative? Complex?
//! - Where is the "union point" — where two systems become one?

fn free_energy(p: f64, s: f64) -> f64 {
    -(1.0 - p.powf(-s)).ln()
}

fn euler_factor(p: f64, s: f64) -> f64 {
    1.0 / (1.0 - p.powf(-s))
}

/// Solve F(a,s) + F(b,s) = ½·ln(b/a) for s
fn solve_equipartition(a: f64, b: f64) -> Option<f64> {
    if a <= 1.0 || b <= 1.0 || a >= b { return None; }
    let target = 0.5 * (b / a).ln();
    if target <= 0.0 { return None; }

    let mut lo = 0.001f64;
    let mut hi = 100.0f64;

    // Check that a solution exists (F decreases from ∞ to 0 as s goes from 0 to ∞)
    let f_lo = free_energy(a, lo) + free_energy(b, lo);
    let f_hi = free_energy(a, hi) + free_energy(b, hi);
    if f_hi > target || f_lo < target { return None; }

    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        let val = free_energy(a, mid) + free_energy(b, mid);
        if val > target { lo = mid; } else { hi = mid; }
    }
    Some((lo + hi) / 2.0)
}

/// At the equipartition point, compute diagnostic quantities
fn diagnose(a: f64, b: f64, s: f64) -> EquipDiag {
    let fa = free_energy(a, s);
    let fb = free_energy(b, s);
    let ratio = fa / fb; // how asymmetric is the energy partition?
    let product = euler_factor(a, s) * euler_factor(b, s);
    let x = a.powf(-s); // fugacity of a
    let y = b.powf(-s); // fugacity of b
    let coupling = x * y; // joint probability
    let sum_f = fa + fb;
    let half_ln_ratio = 0.5 * (b / a).ln();

    EquipDiag {
        s, fa, fb, ratio, product, x, y, coupling,
        balance_error: (sum_f - half_ln_ratio).abs(),
    }
}

struct EquipDiag {
    s: f64,
    fa: f64,
    fb: f64,
    ratio: f64,     // F_a/F_b — energy partition asymmetry
    product: f64,   // E(a,s)·E(b,s) — total Euler weight
    x: f64,         // a^{-s} — fugacity of a
    y: f64,         // b^{-s} — fugacity of b
    coupling: f64,  // x·y — joint fugacity
    balance_error: f64,
}

fn main() {
    eprintln!("==========================================================");
    eprintln!("  Equipartition Sweep");
    eprintln!("  F(a,s) + F(b,s) = ½·ln(b/a)");
    eprintln!("  Varying (a,b) across number types, intervals, scales");
    eprintln!("==========================================================\n");

    // ── Section 1: Primes vs composites vs irrationals ───
    eprintln!("=== Does it matter if a,b are prime? ===\n");
    eprintln!("  {:>12} {:>12} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "a", "b", "s*", "F_a/F_b", "product", "coupling", "type");

    let pairs: Vec<(&str, f64, f64)> = vec![
        // Primes
        ("prime", 2.0, 3.0),
        ("prime", 2.0, 5.0),
        ("prime", 3.0, 5.0),
        ("prime", 5.0, 7.0),
        ("prime", 7.0, 11.0),
        // Sequential integers
        ("sequential", 2.0, 3.0),
        ("sequential", 3.0, 4.0),
        ("sequential", 4.0, 5.0),
        ("sequential", 5.0, 6.0),
        ("sequential", 9.0, 10.0),
        ("sequential", 99.0, 100.0),
        // Non-sequential composites
        ("composite", 4.0, 6.0),
        ("composite", 4.0, 9.0),
        ("composite", 6.0, 10.0),
        ("composite", 8.0, 12.0),
        // Powers of the same prime
        ("power", 2.0, 4.0),   // 2, 2²
        ("power", 2.0, 8.0),   // 2, 2³
        ("power", 3.0, 9.0),   // 3, 3²
        ("power", 3.0, 27.0),  // 3, 3³
        // Irrationals
        ("irrational", std::f64::consts::E, std::f64::consts::PI),
        ("irrational", std::f64::consts::SQRT_2, std::f64::consts::E),
        ("irrational", 2.0, std::f64::consts::E),
        ("irrational", std::f64::consts::PI, 4.0),
        // Golden ratio
        ("golden", (1.0+5.0f64.sqrt())/2.0, (1.0+5.0f64.sqrt())/2.0 + 1.0), // φ, φ+1=φ²
        ("golden", 2.0, (1.0+5.0f64.sqrt())/2.0 + 1.0),
    ];

    for (ptype, a, b) in &pairs {
        let (a, b) = if a < b { (*a, *b) } else { (*b, *a) };
        if let Some(s) = solve_equipartition(a, b) {
            let d = diagnose(a, b, s);
            eprintln!("  {:>12.6} {:>12.6} {:>10.4} {:>10.4} {:>10.4} {:>10.6} {:>10}",
                a, b, d.s, d.ratio, d.product, d.coupling, ptype);
        }
    }

    // ── Section 2: Music theory intervals ────────────────
    eprintln!("\n=== Musical intervals as (a,b) pairs ===\n");
    eprintln!("  {:>20} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "Interval", "ratio", "a", "b", "s*", "product");

    let intervals: Vec<(&str, f64, f64)> = vec![
        // Western 12-tone (just intonation)
        ("unison", 1.0, 1.0),           // 1:1
        ("minor second", 15.0, 16.0),   // 16/15
        ("major second", 8.0, 9.0),     // 9/8
        ("minor third", 5.0, 6.0),      // 6/5
        ("major third", 4.0, 5.0),      // 5/4
        ("perfect fourth", 3.0, 4.0),   // 4/3
        ("tritone", 32.0, 45.0),        // 45/32
        ("perfect fifth", 2.0, 3.0),    // 3/2
        ("minor sixth", 5.0, 8.0),      // 8/5
        ("major sixth", 3.0, 5.0),      // 5/3
        ("minor seventh", 5.0, 9.0),    // 9/5
        ("major seventh", 8.0, 15.0),   // 15/8
        ("octave", 1.0, 2.0),           // 2/1 — but a=1 breaks our equation
        // Non-Western
        ("neutral third", 9.0, 11.0),   // 11/9 ≈ quarter-tone
        ("septimal minor 3rd", 6.0, 7.0), // 7/6
        ("harmonic seventh", 4.0, 7.0), // 7/4
        ("undecimal tritone", 8.0, 11.0), // 11/8
    ];

    for (name, a, b) in &intervals {
        let (a, b) = if a < b { (*a, *b) } else { (*b, *a) };
        if a <= 1.0 { continue; } // skip unison and octave (a=1 is degenerate)
        if let Some(s) = solve_equipartition(a, b) {
            let d = diagnose(a, b, s);
            eprintln!("  {:>20} {:>10.4} {:>10.1} {:>10.1} {:>10.4} {:>10.4}",
                name, b/a, a, b, d.s, d.product);
        }
    }

    // ── Section 3: What's special about s* at the equipartition? ──
    eprintln!("\n=== The coupling strength at equipartition ===\n");
    eprintln!("  At s*, the joint fugacity x·y = a^{{-s}}·b^{{-s}} = (ab)^{{-s}}");
    eprintln!("  This is the PROBABILITY that both systems are in their 'active' state.");
    eprintln!("  The equipartition condition says: this probability is tuned so that");
    eprintln!("  the total free energy = half the asymmetry.\n");

    // For each pair, report coupling at equipartition
    eprintln!("  {:>8} {:>8} {:>10} {:>12} {:>12}",
        "a", "b", "s*", "coupling", "√(coupling)");
    for (a, b) in [(2.0,3.0), (3.0,4.0), (4.0,5.0), (5.0,6.0), (2.0,5.0), (3.0,7.0)] {
        if let Some(s) = solve_equipartition(a, b) {
            let d = diagnose(a, b, s);
            eprintln!("  {:>8.1} {:>8.1} {:>10.4} {:>12.8} {:>12.8}",
                a, b, d.s, d.coupling, d.coupling.sqrt());
        }
    }

    // ── Section 4: The "union point" interpretation ──────
    eprintln!("\n=== The Union Point: where two become one ===\n");

    // At the equipartition, the RATIO F_a/F_b tells us how energy is shared.
    // If F_a/F_b = 1: perfect energy sharing (the two systems are "the same")
    // If F_a/F_b ≫ 1 or ≪ 1: one system dominates

    eprintln!("  Energy partition ratio F_a/F_b at equipartition:");
    eprintln!("  {:>8} {:>8} {:>10} {:>12} {:>20}",
        "a", "b", "s*", "F_a/F_b", "interpretation");

    for (a, b) in [(2.0,3.0), (2.0,4.0), (2.0,8.0), (2.0,100.0),
                   (3.0,4.0), (3.0,5.0), (5.0,6.0), (99.0,100.0)] {
        if let Some(s) = solve_equipartition(a, b) {
            let d = diagnose(a, b, s);
            let interp = if d.ratio > 0.9 && d.ratio < 1.1 {
                "near-equal (union!)"
            } else if d.ratio > 0.7 {
                "moderate asymmetry"
            } else {
                "strong asymmetry"
            };
            eprintln!("  {:>8.1} {:>8.1} {:>10.4} {:>12.6} {:>20}",
                a, b, d.s, d.ratio, interp);
        }
    }

    // ── Section 5: Does (a,b) = (2,3) have a special partition? ──
    eprintln!("\n=== Is (2,3) special? ===\n");

    // The partition ratio F_a/F_b at s* tells us if the energy is equally shared
    let d = diagnose(2.0, 3.0, solve_equipartition(2.0, 3.0).unwrap());
    eprintln!("  (2,3): F₂/F₃ = {:.6}", d.ratio);
    eprintln!("  F₂ = {:.6}, F₃ = {:.6}", d.fa, d.fb);
    eprintln!("  The '2' system carries {:.1}% of the total free energy", d.fa / (d.fa + d.fb) * 100.0);
    eprintln!("  The '3' system carries {:.1}%", d.fb / (d.fa + d.fb) * 100.0);

    // Compare to closest-to-equal
    eprintln!("\n  Partition ratios across pairs (closer to 1.0 = more 'unified'):");
    let mut ratios: Vec<(f64, f64, f64)> = Vec::new();
    for a_int in 2..=20u64 {
        for b_int in (a_int+1)..=20 {
            let a = a_int as f64;
            let b = b_int as f64;
            if let Some(s) = solve_equipartition(a, b) {
                let d = diagnose(a, b, s);
                ratios.push((a, b, d.ratio));
            }
        }
    }
    ratios.sort_by(|a, b| (a.2 - 1.0).abs().partial_cmp(&(b.2 - 1.0).abs()).unwrap());

    eprintln!("  Top 10 most equal energy partitions:");
    for (a, b, r) in ratios.iter().take(10) {
        eprintln!("    ({:.0}, {:.0}): F_a/F_b = {:.6} (deviation from 1: {:.6})",
            a, b, r, (r - 1.0).abs());
    }
    eprintln!("  ...");
    eprintln!("  Most asymmetric:");
    for (a, b, r) in ratios.iter().rev().take(5) {
        eprintln!("    ({:.0}, {:.0}): F_a/F_b = {:.6}", a, b, r);
    }

    // ── Section 6: The biphoton interpretation ──────────
    eprintln!("\n=== The Biphoton Interpretation ===\n");
    eprintln!("  A biphoton is a pair of photons that are entangled —");
    eprintln!("  they share a quantum state and behave as one entity.");
    eprintln!();
    eprintln!("  The equipartition condition F_a + F_b = ½·ln(b/a)");
    eprintln!("  defines the parameter s* where two independent geometric");
    eprintln!("  systems become 'entangled' in the sense that their");
    eprintln!("  combined free energy equals the minimal coupling energy.");
    eprintln!();
    eprintln!("  Below s*: the systems are too 'hot' — each has more");
    eprintln!("  internal energy than the coupling. They're independent.");
    eprintln!("  Above s*: the systems are too 'cold' — the coupling");
    eprintln!("  exceeds the internal energy. They're frozen.");
    eprintln!("  AT s*: internal energy = coupling. The union point.");
    eprintln!();

    // Show F_a + F_b vs ½·ln(b/a) as s varies around s*
    let s_star = solve_equipartition(2.0, 3.0).unwrap();
    eprintln!("  For (2,3), s* = {:.4}:", s_star);
    eprintln!("  {:>8} {:>12} {:>12} {:>12}",
        "s", "F₂+F₃", "½ln(3/2)", "state");
    for &s in &[1.0, 1.5, 2.0, 2.5, s_star, 3.0, 4.0, 6.0, 10.0] {
        let f_sum = free_energy(2.0, s) + free_energy(3.0, s);
        let target = 0.5 * (3.0f64 / 2.0).ln();
        let state = if (f_sum - target).abs() < 0.001 {
            "=== UNION ==="
        } else if f_sum > target {
            "independent (hot)"
        } else {
            "frozen (cold)"
        };
        eprintln!("  {:>8.4} {:>12.6} {:>12.6} {:>12}",
            s, f_sum, target, state);
    }
}
