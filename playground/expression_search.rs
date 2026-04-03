//! # Expression Search for the s* Structure
//!
//! The equation: (1-a^{-s})^{-1} · (1-b^{-s})^{-1} = (b/a)^{1/2}
//!
//! This is a MATCHING CONDITION: two local factors in balance
//! with the geometric mean of their ratio.
//!
//! Where does this structure appear?
//!
//! 1. Impedance matching: Z₁·Z₂ = √(Z_load/Z_source)
//! 2. Resonance: response₁ · response₂ = √(coupling)
//! 3. Critical phenomena: correlation₁ · correlation₂ = √(ratio)
//! 4. Partition functions: Z_local₁ · Z_local₂ = √(interaction)
//! 5. Information: capacity₁ · capacity₂ = √(SNR ratio)
//!
//! Let's: (a) generalize to other prime pairs
//!         (b) find the abstract form
//!         (c) look for it in physics/info theory/music

fn euler_factor(p: f64, s: f64) -> f64 {
    1.0 / (1.0 - p.powf(-s))
}

/// Solve: E(a,s) · E(b,s) = target for s
fn solve_s(a: f64, b: f64, target: f64) -> f64 {
    let mut lo = 0.01f64;
    let mut hi = 100.0f64;
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        let val = euler_factor(a, mid) * euler_factor(b, mid);
        if val > target { lo = mid; } else { hi = mid; }
    }
    (lo + hi) / 2.0
}

fn main() {
    eprintln!("==========================================================");
    eprintln!("  Expression Search: (1-a⁻ˢ)⁻¹·(1-b⁻ˢ)⁻¹ = (b/a)^(1/2)");
    eprintln!("  The STRUCTURE, not the number.");
    eprintln!("==========================================================\n");

    // ── Part 1: s* for every prime pair ──────────────────
    eprintln!("=== s* for prime pairs (a,b): E(a,s)·E(b,s) = √(b/a) ===\n");
    eprintln!("  {:>6} {:>6} {:>12} {:>12} {:>12}", "a", "b", "√(b/a)", "s*", "a^s* · b^s*");

    let primes = [2.0, 3.0, 5.0, 7.0, 11.0, 13.0];
    for i in 0..primes.len() {
        for j in (i+1)..primes.len() {
            let a = primes[i];
            let b = primes[j];
            let target = (b as f64 / a as f64).sqrt();
            let s = solve_s(a, b, target);
            let product = a.powf(s) * b.powf(s);
            eprintln!("  {:>6.0} {:>6.0} {:>12.6} {:>12.6} {:>12.2}", a, b, target, s, product);
        }
    }

    // ── Part 2: The abstract form ────────────────────────
    eprintln!("\n=== Abstract form: what IS this equation? ===\n");

    // Rewrite: 1/((1-x)(1-y)) = √(y/x) where x=a^{-s}, y=b^{-s}
    // Since b>a, y<x (both in (0,1) for s>0)
    //
    // 1/((1-x)(1-y)) is the product of two geometric series: (Σ x^n)(Σ y^n)
    // = Σ_{m,n} x^m · y^n = the generating function of the (m,n) lattice
    //
    // √(y/x) = √(b/a)^{-s... no. √(y/x) = √((a/b)^s) = (a/b)^{s/2}
    //
    // Wait: y/x = b^{-s}/a^{-s} = (a/b)^s
    // So √(y/x) = (a/b)^{s/2}
    //
    // The equation: Σ_{m,n≥0} x^m y^n = (a/b)^{s/2}
    //             = (Σ a^{-ms})(Σ b^{-ns}) = (a/b)^{s/2}
    //
    // LHS = partition function of a 2D lattice gas with fugacities a^{-s}, b^{-s}
    // RHS = the geometric mean ratio raised to s/2

    let a = 2.0f64;
    let b = 3.0f64;
    let s_star = solve_s(a, b, (b/a).sqrt());

    eprintln!("  For (a,b) = (2,3), s* = {:.10}", s_star);
    eprintln!("  x = 2^{{-s*}} = {:.10}", a.powf(-s_star));
    eprintln!("  y = 3^{{-s*}} = {:.10}", b.powf(-s_star));
    eprintln!("  y/x = (2/3)^{{s*}} = {:.10}", (a/b).powf(s_star));
    eprintln!("  √(y/x) = (2/3)^{{s*/2}} = {:.10}", (a/b).powf(s_star/2.0));
    eprintln!("  √(3/2) = {:.10}", (b/a).sqrt());
    eprintln!();
    eprintln!("  The equation says: the TOTAL WEIGHT of the (2,3)-lattice");
    eprintln!("  at temperature s* equals the geometric asymmetry between 2 and 3.");
    eprintln!("  This is a BALANCE point: lattice weight = asymmetry measure.");

    // ── Part 3: Rewrite as energy balance ────────────────
    eprintln!("\n=== Energy balance form ===\n");

    // Take log of both sides:
    // -ln(1-a^{-s}) - ln(1-b^{-s}) = (1/2)·ln(b/a)
    //
    // LHS = free energy of two independent systems at temperature 1/s
    // RHS = half the log-ratio of their "sizes"
    //
    // In statistical mechanics: F = -kT·ln(Z)
    // With kT = 1/s:
    // F_a(s) + F_b(s) = -(1/2s)·ln(b/a) ... not quite

    let lhs = -(1.0 - a.powf(-s_star)).ln() - (1.0 - b.powf(-s_star)).ln();
    let rhs = 0.5 * (b/a).ln();
    eprintln!("  -ln(1-2^{{-s*}}) - ln(1-3^{{-s*}}) = {:.10}", lhs);
    eprintln!("  (1/2)·ln(3/2)                       = {:.10}", rhs);
    eprintln!("  Match: {}", (lhs - rhs).abs() < 1e-10);

    eprintln!("\n  In free energy language:");
    eprintln!("  F₂(s*) + F₃(s*) = ½·ln(3/2)");
    eprintln!("  'The combined free energy of the {{2,3}} system at temperature s*");
    eprintln!("   equals half the log-ratio of the two primes.'");
    eprintln!("  This IS an equipartition condition.");

    // ── Part 4: Music theory connection ──────────────────
    eprintln!("\n=== Music theory: interval ratios ===\n");

    // In music: interval = log₂(freq ratio)
    // Perfect fifth = log₂(3/2) ≈ 0.585 octaves
    // The equation: F₂ + F₃ = ½·ln(3/2) = ½·0.585·ln(2)
    // The free energy balance IS half the perfect fifth in nats

    let fifth_nats = (3.0f64 / 2.0).ln();
    let fifth_octaves = (3.0f64 / 2.0).log2();
    eprintln!("  Perfect fifth = ln(3/2) = {:.6} nats = {:.6} octaves", fifth_nats, fifth_octaves);
    eprintln!("  Half fifth    = {:.6} nats (the RHS of our equation)", fifth_nats / 2.0);
    eprintln!("  = the NEUTRAL THIRD = √(3/2) = {:.6}", (1.5f64).sqrt());
    eprintln!("  (which is log₂(√(3/2)) = {:.6} octaves)", (1.5f64).sqrt().log2());
    eprintln!();
    eprintln!("  The matching condition says:");
    eprintln!("  'The lattice free energy at temperature s* = half a perfect fifth.'");
    eprintln!("  s* is the temperature where the {{2,3}}-lattice resonates at the neutral third.");

    // ── Part 5: Other matching conditions ────────────────
    eprintln!("\n=== Other matching conditions: E(a,s)·E(b,s) = WHAT? ===\n");

    // What if the RHS isn't √(b/a) but something else?
    let targets: Vec<(&str, f64)> = vec![
        ("1 (trivial)", 1.0),
        ("φ (golden ratio)", (1.0 + 5.0f64.sqrt()) / 2.0),
        ("√2", std::f64::consts::SQRT_2),
        ("√(3/2) (neutral third)", (1.5f64).sqrt()),
        ("3/2 (perfect fifth)", 1.5),
        ("√3", 3.0f64.sqrt()),
        ("2 (octave)", 2.0),
        ("e", std::f64::consts::E),
        ("π", std::f64::consts::PI),
    ];

    eprintln!("  {:>25}  {:>10}  {:>12}", "Target", "Value", "s* for (2,3)");
    for (name, target) in &targets {
        if *target > 1.0 {
            let s = solve_s(2.0, 3.0, *target);
            eprintln!("  {:>25}  {:>10.6}  {:>12.6}", name, target, s);
        }
    }

    // ── Part 6: The generalized matching surface ─────────
    eprintln!("\n=== Generalized: s*(a,b,r) where E(a,s)·E(b,s) = r ===\n");
    eprintln!("  For any target ratio r, there's a temperature s* where");
    eprintln!("  the (a,b)-lattice has total weight exactly r.");
    eprintln!("  The Collatz point: (a=2, b=3, r=√(3/2)).");
    eprintln!("  The perfect-fifth point: (a=2, b=3, r=3/2) → s=2 (= ζ(2)).");
    eprintln!("  The octave point: (a=2, b=3, r=2) → s = ?");

    let s_octave = solve_s(2.0, 3.0, 2.0);
    let s_phi = solve_s(2.0, 3.0, (1.0 + 5.0f64.sqrt()) / 2.0);
    eprintln!("  s*(2,3,2) = {:.10} (octave)", s_octave);
    eprintln!("  s*(2,3,φ) = {:.10} (golden ratio)", s_phi);
    eprintln!("  s*(2,3,3/2) = {:.10} (perfect fifth = ζ(2)!)", solve_s(2.0, 3.0, 1.5));

    // ── Part 7: Is (a,b,r) = (2,3,√(3/2)) special? ─────
    eprintln!("\n=== Why is (2,3,√(3/2)) special? ===\n");
    eprintln!("  r = √(b/a) means: the target is the GEOMETRIC MEAN of the ratio.");
    eprintln!("  This is the MIDPOINT (in log space) between 1 and b/a.");
    eprintln!("  It's the point of maximum uncertainty: equally far from");
    eprintln!("  'the primes are the same' (r=1) and 'full asymmetry' (r=b/a).");
    eprintln!();
    eprintln!("  In information theory: this is the CAPACITY of a binary channel");
    eprintln!("  with crossover probabilities a^{{-s}} and b^{{-s}}.");
    eprintln!("  The matching condition at r=√(b/a) is where the channel capacity");
    eprintln!("  equals the geometric mean of the error rates.");
    eprintln!();
    eprintln!("  The Collatz map operates at EXACTLY this capacity point.");
    eprintln!("  Not the raw ratio (3/2 = perfect fifth = s=2).");
    eprintln!("  Not trivial (1 = s→∞).");
    eprintln!("  The geometric midpoint. The neutral third. The maximum uncertainty.");
    eprintln!("  THAT is why s* ≈ 2.797 and not 2 or 3.");
}
