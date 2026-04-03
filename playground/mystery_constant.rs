//! # Search for s* ≈ 2.7971 in known constants
//!
//! s* is the solution of 1/((1 - 2^{-s})(1 - 3^{-s})) = √(3/2)
//!
//! Does it appear anywhere else?

fn euler_23(s: f64) -> f64 {
    1.0 / ((1.0 - 2.0_f64.powf(-s)) * (1.0 - 3.0_f64.powf(-s)))
}

fn main() {
    // Find s* precisely
    let target = (1.5_f64).sqrt();
    let mut lo = 2.5_f64;
    let mut hi = 3.0_f64;
    for _ in 0..200 {
        let mid = (lo + hi) / 2.0;
        if euler_23(mid) > target { lo = mid; } else { hi = mid; }
    }
    let s_star = (lo + hi) / 2.0;
    eprintln!("s* = {:.15}", s_star);
    eprintln!("E_{{2,3}}(s*) = {:.15}", euler_23(s_star));
    eprintln!("√(3/2)     = {:.15}", target);
    eprintln!();

    // Known constants to compare
    let constants: Vec<(&str, f64)> = vec![
        ("e (Euler's number)", std::f64::consts::E),
        ("π - 0.344", std::f64::consts::PI - 0.344),
        ("ln(2π)", (2.0 * std::f64::consts::PI).ln()),
        ("√(2π)/√e", (2.0 * std::f64::consts::PI).sqrt() / std::f64::consts::E.sqrt()),
        ("e/√e = √e", std::f64::consts::E.sqrt()),
        ("φ² (golden ratio squared)", ((1.0 + 5.0_f64.sqrt()) / 2.0).powi(2)),
        ("1 + φ", 1.0 + (1.0 + 5.0_f64.sqrt()) / 2.0),
        ("π/e + 1", std::f64::consts::PI / std::f64::consts::E + 1.0),
        ("ln(16)", 16.0_f64.ln()),
        ("4·ln(2)", 4.0 * 2.0_f64.ln()),
        ("ζ(3) + ζ(2) - 1", 1.202056903 + 1.644934068 - 1.0),
        ("Khinchin's constant", 2.6854520011),
        ("Lévy's constant", 3.275823),
        ("Ramanujan-Soldner", 1.451369),
        ("2 + Euler-Mascheroni γ", 2.0 + 0.5772156649),
        ("e^(1/e)", std::f64::consts::E.powf(1.0/std::f64::consts::E)),
        ("2^(3/2)", 2.0_f64.powf(1.5)),
        ("3^(log₂3/2)", 3.0_f64.powf(3.0_f64.log2() / 2.0)),
        ("log₂(3) + 1", 3.0_f64.log2() + 1.0),
        ("1 + log₂(3)", 1.0 + 3.0_f64.log2()),
        ("2·log₂(3)", 2.0 * 3.0_f64.log2()),
        ("log₂(7)", 7.0_f64.log2()),
        ("3·ln(2)", 3.0 * 2.0_f64.ln()),
        ("2·ln(3)", 2.0 * 3.0_f64.ln()),
        ("ln(2) + ln(3)", 2.0_f64.ln() + 3.0_f64.ln()),
        ("ln(6)", 6.0_f64.ln()),
        ("√(π²/4 + 1)", (std::f64::consts::PI.powi(2) / 4.0 + 1.0).sqrt()),
        ("cube root of 21.9", 21.9_f64.powf(1.0/3.0)),
        ("4th root of 61.2", 61.2_f64.powf(0.25)),
        ("Feigenbaum δ - 1.87", 4.669201609 - 1.87),
        ("1/Feigenbaum α", 1.0 / 2.502907875),
        ("Apéry ζ(3)", 1.2020569031),
        ("ζ(3) + 1.595", 1.2020569031 + 1.595),
        ("Catalan's constant + 1.88", 0.915965594 + 1.88),
        ("Glaisher-Kinkelin", 1.282427130),
        ("π^(1/π)", std::f64::consts::PI.powf(1.0/std::f64::consts::PI)),
        ("2^√2", 2.0_f64.powf(std::f64::consts::SQRT_2)),
        ("√2 + 1 (silver ratio)", std::f64::consts::SQRT_2 + 1.0),
        ("3/2 + log₂(3) - 1", 1.5 + 3.0_f64.log2() - 1.0),
    ];

    eprintln!("--- Comparison with known constants ---");
    eprintln!("  {:>40}  {:>15}  {:>12}", "Constant", "Value", "|s* - val|");

    let mut ranked: Vec<(&str, f64, f64)> = constants.iter()
        .map(|&(name, val)| (name, val, (s_star - val).abs()))
        .collect();
    ranked.sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap());

    for (name, val, diff) in &ranked {
        eprintln!("  {:>40}  {:>15.10}  {:>12.2e}", name, val, diff);
    }

    // Check some algebraic relationships
    eprintln!("\n--- Algebraic relationships ---");
    eprintln!("  s* = {:.15}", s_star);
    eprintln!("  s* - e = {:.15} (not clean)", s_star - std::f64::consts::E);
    eprintln!("  s* / ln(2) = {:.15}", s_star / 2.0_f64.ln());
    eprintln!("  s* / ln(3) = {:.15}", s_star / 3.0_f64.ln());
    eprintln!("  s* / ln(6) = {:.15}", s_star / 6.0_f64.ln());
    eprintln!("  s* · ln(2) = {:.15}", s_star * 2.0_f64.ln());
    eprintln!("  s* · ln(3) = {:.15}", s_star * 3.0_f64.ln());
    eprintln!("  2^s* = {:.15}", 2.0_f64.powf(s_star));
    eprintln!("  3^s* = {:.15}", 3.0_f64.powf(s_star));
    eprintln!("  6^s* = {:.15}", 6.0_f64.powf(s_star));
    eprintln!("  s* - log₂(3) - 1 = {:.15}", s_star - 3.0_f64.log2() - 1.0);
    eprintln!("  s* - 2·ln(3)/ln(2) = {:.15}", s_star - 2.0 * 3.0_f64.ln() / 2.0_f64.ln());
    eprintln!("  (s*-2)·ln(6) = {:.15}", (s_star - 2.0) * 6.0_f64.ln());

    // Physical constants
    eprintln!("\n--- Physical constant comparisons ---");
    let phys: Vec<(&str, f64)> = vec![
        ("Fine structure α⁻¹ / 50", 137.036 / 50.0),
        ("Boltzmann k_B × 10²³", 1.380649e-23 * 1e23),
        ("Proton/electron mass ratio / 652", 1836.15267 / 652.0),
        ("Weinberg angle sin²θ_W × 10", 0.23122 * 10.0),
        ("Cabibbo angle θ_C (radians)", 0.2272),
    ];
    for (name, val) in &phys {
        eprintln!("  {:>40}  {:>15.10}  {:>12.2e}", name, val, (s_star - val).abs());
    }

    // Does s* satisfy any simple polynomial?
    eprintln!("\n--- Polynomial check: does s* satisfy a low-degree polynomial? ---");
    for (a, b, c) in [(1,0,-3), (1,-3,0), (1,-2,-1), (1,0,-8), (4,-11,0), (1,-3,1)] {
        let val = a as f64 * s_star * s_star + b as f64 * s_star + c as f64;
        eprintln!("  {}s² + {}s + {} = {:.6}", a, b, c, val);
    }

    // Is s* related to the solution of x^x = something?
    eprintln!("\n--- Self-referential checks ---");
    eprintln!("  s*^s* = {:.10}", s_star.powf(s_star));
    eprintln!("  s*^(1/s*) = {:.10}", s_star.powf(1.0/s_star));
    eprintln!("  (s*/e)^(s*/e) = {:.10}", (s_star/std::f64::consts::E).powf(s_star/std::f64::consts::E));
}
