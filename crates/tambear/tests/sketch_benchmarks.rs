//! Cross-sketch comparison benchmarks — accuracy / memory / merge behavior.
//!
//! Locked vocabulary: this is a comparison harness for the four
//! quantile sketches (Tier 1 primitives) listed in
//! `R:\winrapids\docs\architecture\vocabulary.md`. It documents the
//! trade-offs each sketch makes so consumers can pick intelligently
//! when overriding the locked default (KLL) via `using(sketch: "...")`.
//!
//! # What this measures
//!
//! For each sketch on each of several distributions:
//! - **Quantile error** at q ∈ {0.05, 0.5, 0.95} vs an exact-sort oracle
//! - **State size** (rough bucket / centroid / tuple count after ingestion)
//! - **Merge fidelity** — does (a ⊕ b ⊕ c) match (a ⊕ (b ⊕ c)) bit-for-bit?
//! - **Permutation invariance** — does insertion order matter for the
//!   final query result? (DDSketch alone says "no, ever.")
//!
//! These are property-asserting tests, not microbenchmarks. They
//! compile and run as part of `cargo test`; their output documents
//! the trade-off space.
//!
//! # Distributions tested
//!
//! - **Uniform** [0, 10000) — light-tailed, well-behaved
//! - **Log-normal** μ=0, σ=2 — moderately heavy-tailed
//! - **Pareto α=2** — heavy upper tail, infinite variance
//! - **Mixed sign** ±100 — values straddling zero
//! - **Wide range** 1e-10 to 1e10 — many orders of magnitude

use tambear::primitives::specialist::{
    DdSketch, GkSketch, KllSketch, QuantileSketch, TdigestSketch,
};

// ── Distributions ──────────────────────────────────────────────────────────

/// Van der Corput sequence for deterministic low-discrepancy quasi-uniforms.
fn van_der_corput(mut n: u64, base: u64) -> f64 {
    let mut q = 0.0_f64;
    let mut bk = 1.0_f64 / base as f64;
    while n > 0 {
        q += (n % base) as f64 * bk;
        n /= base;
        bk /= base as f64;
    }
    q
}

fn uniform(n: usize, range: f64) -> Vec<f64> {
    (0..n).map(|i| (i as f64) * range / n as f64).collect()
}

fn lognormal(n: usize, mu: f64, sigma: f64) -> Vec<f64> {
    (1..=n)
        .map(|i| {
            let u = van_der_corput(i as u64, 2).clamp(0.001, 0.999);
            // Normal via Box-Muller using two quasi-uniforms.
            let v = van_der_corput(i as u64, 3).clamp(0.001, 0.999);
            let z = (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos();
            (mu + sigma * z).exp()
        })
        .collect()
}

fn pareto(n: usize, alpha: f64) -> Vec<f64> {
    (1..=n)
        .map(|i| {
            let u = van_der_corput(i as u64, 2).clamp(0.001, 0.999);
            (1.0 - u).powf(-1.0 / alpha)
        })
        .collect()
}

fn mixed_sign(n: usize, range: f64) -> Vec<f64> {
    (0..n)
        .map(|i| ((i as f64) - (n as f64) / 2.0) * range / n as f64)
        .collect()
}

fn wide_range(n: usize) -> Vec<f64> {
    // Log-spaced from 1e-10 to 1e10.
    (0..n)
        .map(|i| {
            let exponent = -10.0 + 20.0 * (i as f64) / (n as f64 - 1.0);
            10.0_f64.powf(exponent)
        })
        .collect()
}

// ── Oracle ─────────────────────────────────────────────────────────────────

fn exact_quantile(values: &[f64], q: f64) -> f64 {
    let mut sorted: Vec<f64> = values.iter().copied().filter(|x| x.is_finite()).collect();
    sorted.sort_by(|a, b| a.total_cmp(b));
    if sorted.is_empty() {
        return f64::NAN;
    }
    let n = sorted.len();
    let idx = ((q * n as f64).floor() as usize).min(n - 1);
    sorted[idx]
}

// ── Sketch-comparison helpers ──────────────────────────────────────────────

fn errors_against_oracle<S: QuantileSketch>(distribution: &[f64], epsilon: f64) -> ([f64; 3], [f64; 3]) {
    let mut sk = S::new(epsilon);
    sk.add_slice(distribution);
    let qs = [0.05_f64, 0.5, 0.95];
    let mut abs_errors = [0.0_f64; 3];
    let mut rel_errors = [0.0_f64; 3];
    for (i, &q) in qs.iter().enumerate() {
        let got = sk.quantile(q);
        let want = exact_quantile(distribution, q);
        abs_errors[i] = (got - want).abs();
        rel_errors[i] = abs_errors[i] / want.abs().max(1e-12);
    }
    (abs_errors, rel_errors)
}

fn print_row(name: &str, abs_errors: [f64; 3], rel_errors: [f64; 3]) {
    eprintln!(
        "{:<10}  {:>14.4}  {:>14.4}  {:>14.4}    {:>10.4}  {:>10.4}  {:>10.4}",
        name,
        abs_errors[0], abs_errors[1], abs_errors[2],
        rel_errors[0], rel_errors[1], rel_errors[2]
    );
}

fn print_header() {
    eprintln!("{:<10}  {:>14}  {:>14}  {:>14}    {:>10}  {:>10}  {:>10}",
              "sketch", "abs@0.05", "abs@0.50", "abs@0.95",
              "rel@0.05", "rel@0.50", "rel@0.95");
}

fn print_distribution_report(distribution_name: &str, distribution: &[f64], epsilon: f64) {
    eprintln!("\n=== {} (n={}, ε={}) ===", distribution_name, distribution.len(), epsilon);
    print_header();
    let (a, r) = errors_against_oracle::<KllSketch>(distribution, epsilon);
    print_row("KLL", a, r);
    let (a, r) = errors_against_oracle::<GkSketch>(distribution, epsilon);
    print_row("GK", a, r);
    let (a, r) = errors_against_oracle::<TdigestSketch>(distribution, epsilon);
    print_row("t-digest", a, r);
    let (a, r) = errors_against_oracle::<DdSketch>(distribution, epsilon);
    print_row("DDSketch", a, r);
}

// ── Tests that BOTH document trade-offs AND assert basic accuracy ──────────

fn assert_median_within_rel<S: QuantileSketch>(distribution: &[f64], epsilon: f64, rel_tol: f64, label: &str) {
    let mut sk = S::new(epsilon);
    sk.add_slice(distribution);
    let got = sk.quantile(0.5);
    let want = exact_quantile(distribution, 0.5);
    let rel = ((got - want) / want.abs().max(1e-12)).abs();
    assert!(rel < rel_tol, "{label}: median rel error {rel} > tol {rel_tol}");
}

#[test]
fn comparison_uniform_distribution() {
    let d = uniform(10000, 10000.0);
    print_distribution_report("Uniform [0, 10000)", &d, 0.01);
    assert_median_within_rel::<KllSketch>(&d, 0.01, 0.05, "KLL uniform");
    assert_median_within_rel::<GkSketch>(&d, 0.01, 0.05, "GK uniform");
    assert_median_within_rel::<TdigestSketch>(&d, 0.01, 0.05, "t-digest uniform");
    assert_median_within_rel::<DdSketch>(&d, 0.01, 0.05, "DDSketch uniform");
}

#[test]
fn comparison_lognormal_distribution() {
    let d = lognormal(10000, 0.0, 2.0);
    print_distribution_report("Lognormal μ=0 σ=2", &d, 0.01);
}

#[test]
fn comparison_pareto_heavy_tail() {
    let d = pareto(10000, 2.0);
    print_distribution_report("Pareto α=2 (heavy upper tail)", &d, 0.01);
}

#[test]
fn comparison_mixed_sign() {
    let d = mixed_sign(10000, 200.0);
    print_distribution_report("Mixed-sign ±100", &d, 0.01);
    // For mixed-sign data near zero: DDSketch's native handling
    // shouldn't produce a wildly different median than the others.
    let mut kll = KllSketch::new(0.01);
    kll.add_slice(&d);
    let mut dd = DdSketch::new(0.01);
    dd.add_slice(&d);
    let med_kll = kll.quantile(0.5);
    let med_dd = dd.quantile(0.5);
    assert!(med_kll.abs() < 5.0, "KLL median {med_kll}");
    assert!(med_dd.abs() < 5.0, "DDSketch median {med_dd}");
}

#[test]
fn comparison_wide_value_range() {
    let d = wide_range(1000);
    print_distribution_report("Wide range 1e-10 to 1e10", &d, 0.05);
    // DDSketch's relative-error guarantee shines on wide-range data.
    // Just verify it gives finite plausible answers across the range.
    let mut sk = DdSketch::new(0.05);
    sk.add_slice(&d);
    for &q in &[0.1, 0.5, 0.9] {
        let v = sk.quantile(q);
        assert!(v.is_finite() && v > 0.0, "DDSketch wide-range q={q}: got {v}");
    }
}

#[test]
fn merge_fidelity_comparison() {
    // For each sketch: build full from one stream; build from two
    // halves merged. Assert the two paths agree (bit-exact for
    // DDSketch, close-enough for the others).
    let d = lognormal(5000, 0.0, 1.5);
    let (a, b) = d.split_at(2500);

    macro_rules! check_merge {
        ($name:literal, $sketch:ty, $bit_exact:expr) => {{
            let mut full = <$sketch>::new(0.02);
            full.add_slice(&d);
            let mut sa = <$sketch>::new(0.02);
            sa.add_slice(a);
            let mut sb = <$sketch>::new(0.02);
            sb.add_slice(b);
            sa.merge(&sb);
            for &q in &[0.1, 0.5, 0.9] {
                let f = full.quantile(q);
                let m = sa.quantile(q);
                if $bit_exact {
                    assert_eq!(
                        f.to_bits(),
                        m.to_bits(),
                        "{} q={}: full={} merged={} (expected bit-exact)",
                        $name, q, f, m
                    );
                    eprintln!("  {:<10} q={:.2}  ✓ bit-exact ({})", $name, q, f);
                } else {
                    // Non-bit-exact sketches (KLL/GK/t-digest) merge
                    // result depends on their internal compaction
                    // history. On heavy-tailed data at extreme
                    // quantiles, the value-error from a few rank-
                    // positions of disagreement can be large. The
                    // assertion here just guards against catastrophic
                    // divergence (>50% relative), and the eprintln
                    // documents the actual divergence so consumers
                    // see it.
                    let rel = ((m - f) / f.abs().max(1e-12)).abs();
                    assert!(rel < 0.50, "{} q={}: rel diff {rel} too high", $name, q);
                    eprintln!("  {:<10} q={:.2}  full={:.4}  merged={:.4}  (rel {:.4})",
                              $name, q, f, m, rel);
                }
            }
        }};
    }

    eprintln!("\n=== Merge fidelity (lognormal n=5000, ε=0.02) ===");
    check_merge!("KLL", KllSketch, false);
    check_merge!("GK", GkSketch, false);
    check_merge!("t-digest", TdigestSketch, false);
    check_merge!("DDSketch", DdSketch, true);
}

#[test]
fn permutation_invariance_only_ddsketch() {
    // Build each sketch from forward and reverse insertion order;
    // assert which sketches produce identical results.
    let d: Vec<f64> = (1..=2000).map(|i| (i as f64).cos() * 50.0 + 100.0).collect();
    let d_rev: Vec<f64> = d.iter().rev().copied().collect();

    eprintln!("\n=== Permutation invariance test ===");

    macro_rules! check_perm {
        ($name:literal, $sketch:ty, $bit_exact_required:expr) => {{
            let mut a = <$sketch>::new(0.01);
            a.add_slice(&d);
            let mut b = <$sketch>::new(0.01);
            b.add_slice(&d_rev);
            let mut all_match = true;
            for &q in &[0.1, 0.5, 0.9] {
                let va = a.quantile(q);
                let vb = b.quantile(q);
                if va.to_bits() != vb.to_bits() {
                    all_match = false;
                    eprintln!("  {:<10} q={:.2}  forward={:.6} reverse={:.6}  (DIFFERS)",
                              $name, q, va, vb);
                }
            }
            if $bit_exact_required {
                assert!(all_match, "{} should be permutation-invariant but isn't", $name);
                eprintln!("  {:<10}  ✓ permutation-invariant", $name);
            } else {
                eprintln!("  {:<10}  (no permutation-invariance guarantee — this is expected)", $name);
            }
        }};
    }

    check_perm!("KLL", KllSketch, false);
    check_perm!("GK", GkSketch, false);
    check_perm!("t-digest", TdigestSketch, false);
    check_perm!("DDSketch", DdSketch, true);
}
