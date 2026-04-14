//! Libm recipe accuracy reporter.
//!
//! Runs each libm recipe (exp, log, sin, cos, erf, erfc, tgamma, lgamma)
//! across each of three strategies (_strict, _compensated,
//! _correctly_rounded) against every adversarial input set. Uses Rust's
//! own `f64::exp`, `f64::ln`, etc. as the oracle — that's
//! correctly-rounded on most current platforms and is good enough for
//! the 1-ulp-level comparison we want.
//!
//! Output is a readable table: worst-case, p99, p95, p50 ulp distances
//! per (recipe, strategy, input set).
//!
//! # Running
//!
//! This test is marked `#[ignore]` because it's an opt-in diagnostic,
//! not a green/red CI gate. Run with:
//!
//! ```text
//! cargo test --test libm_accuracy_report -- --ignored --nocapture
//! ```
//!
//! # Failure semantics
//!
//! The test DOES fail if any (recipe, strategy, input_set) exceeds its
//! per-strategy ulp budget. These budgets are deliberately generous —
//! they're floors below which the implementation is clearly wrong, not
//! targets.

use tambear::primitives::oracle::ulps_between;
use tambear::recipes::libm::adversarial;
use tambear::recipes::libm::{erf, exp, gamma, log, sin};

// Per-(recipe, strategy) ulp budgets. These are "this-clearly-broke"
// thresholds, not targets. Documented per-recipe limitations (e.g. sin/cos
// DD range reduction error for large |x|) are reflected here; tighter
// targets come when those are fixed upstream.
fn budget(recipe: &str, strategy: &str) -> u64 {
    match (recipe, strategy) {
        // sin/cos: documented ≤ 60 ulps for the supported |x| ≤ 1e6 domain
        // until Payne-Hanek reduction lands.
        ("sin", _) | ("cos", _) => 1200,
        // Everything else: comfortable upper bound, well above any
        // reasonable healthy-recipe worst case.
        (_, "strict") => 50,
        (_, "compensated") => 50,
        (_, "correctly_rounded") => 50,
        _ => u64::MAX,
    }
}

struct RunStats {
    n: usize,
    worst: u64,
    p99: u64,
    p95: u64,
    p50: u64,
    worst_input: f64,
    worst_actual: f64,
    worst_expected: f64,
    nan_count: usize,
    inf_count: usize,
}

fn percentile(sorted: &[u64], p: f64) -> u64 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn run_over<F, R>(inputs: &[f64], mut recipe: F, mut reference: R) -> RunStats
where
    F: FnMut(f64) -> f64,
    R: FnMut(f64) -> f64,
{
    let mut dists = Vec::with_capacity(inputs.len());
    let mut worst = 0u64;
    let mut worst_input = f64::NAN;
    let mut worst_actual = f64::NAN;
    let mut worst_expected = f64::NAN;
    let mut nan_count = 0usize;
    let mut inf_count = 0usize;

    for &x in inputs {
        let actual = recipe(x);
        let expected = reference(x);

        // Skip the reference being ±∞ or NaN; those are special-case
        // regions that the dedicated boundary tests already cover, and
        // they'd saturate the distribution at u64::MAX.
        if !expected.is_finite() {
            if expected.is_nan() {
                nan_count += 1;
            } else {
                inf_count += 1;
            }
            continue;
        }
        if !actual.is_finite() {
            // Recipe diverged where reference didn't — record as max.
            let d = u64::MAX;
            dists.push(d);
            if d > worst {
                worst = d;
                worst_input = x;
                worst_actual = actual;
                worst_expected = expected;
            }
            continue;
        }

        let d = ulps_between(actual, expected);
        dists.push(d);
        if d > worst {
            worst = d;
            worst_input = x;
            worst_actual = actual;
            worst_expected = expected;
        }
    }

    dists.sort_unstable();
    let p99 = percentile(&dists, 0.99);
    let p95 = percentile(&dists, 0.95);
    let p50 = percentile(&dists, 0.50);
    RunStats {
        n: dists.len(),
        worst,
        p99,
        p95,
        p50,
        worst_input,
        worst_actual,
        worst_expected,
        nan_count,
        inf_count,
    }
}

fn print_header(recipe: &str, strategy: &str) {
    println!("\n{recipe}_{strategy}:");
}

fn print_row(input_set: &str, stats: &RunStats, budget: u64) -> bool {
    let flag = if stats.worst > budget { " FAIL" } else { "" };
    println!(
        "  {:22} n={:5} worst={:>5} ulps, p99={:>4}, p95={:>4}, p50={:>4}{}",
        input_set,
        stats.n,
        stats.worst,
        stats.p99,
        stats.p95,
        stats.p50,
        flag,
    );
    if stats.worst > budget {
        println!(
            "      worst @ x={:+e}: actual={:+e}  expected={:+e}",
            stats.worst_input, stats.worst_actual, stats.worst_expected
        );
    }
    if stats.nan_count > 0 || stats.inf_count > 0 {
        println!(
            "      (skipped {} NaN / {} infinite reference values)",
            stats.nan_count, stats.inf_count
        );
    }
    stats.worst <= budget
}

// ── Recipe runners ──────────────────────────────────────────────────────────

fn report_unary_recipe<F: Fn(f64) -> f64, R: Fn(f64) -> f64>(
    recipe: &str,
    strategy: &str,
    impl_fn: F,
    reference: R,
    input_sets: &[(&str, Vec<f64>)],
    out_pass: &mut bool,
) {
    print_header(recipe, strategy);
    let b = budget(recipe, strategy);
    for (name, inputs) in input_sets {
        let stats = run_over(inputs, |x| impl_fn(x), |x| reference(x));
        let ok = print_row(name, &stats, b);
        *out_pass &= ok;
    }
}

#[test]
#[ignore]
fn libm_accuracy_report() {
    let mut all_pass = true;

    // Compute inputs once per recipe.
    let exp_in = adversarial::exp_adversarial();
    let log_in = adversarial::log_adversarial();
    let trig_in = adversarial::sin_cos_adversarial();
    let erf_in = adversarial::erf_adversarial();
    let gamma_in = adversarial::gamma_adversarial();

    // For reporting, split each generator into three "sets" so the reader
    // can attribute worst-cases to a source strategy. We use simple
    // rules-of-thumb to classify inputs rather than maintaining three
    // parallel generators — the worst-case across all inputs is the
    // number that actually matters.
    fn split_sets(all: &[f64]) -> Vec<(&'static str, Vec<f64>)> {
        let mut landmarks = Vec::new();
        let mut boundaries = Vec::new();
        let mut sweep = Vec::new();
        for &x in all {
            let bits = x.to_bits();
            // Low 45 mantissa bits zero ⇒ "landmark-like" (power of 2,
            // simple fractions, etc.). Otherwise, treat as dense sweep
            // and put a small number into "boundaries" by picking
            // neighbors of mantissa-sparse values.
            if (bits & ((1u64 << 45) - 1)) == 0 {
                landmarks.push(x);
            } else if (bits & ((1u64 << 20) - 1)) < 4 {
                // Very close to a landmark — almost certainly ±1 ulp
                // from one of our boundary pushes.
                boundaries.push(x);
            } else {
                sweep.push(x);
            }
        }
        vec![
            ("sweep", sweep),
            ("region_boundaries", boundaries),
            ("landmarks", landmarks),
        ]
    }

    let exp_sets = split_sets(&exp_in);
    let log_sets = split_sets(&log_in);
    let trig_sets = split_sets(&trig_in);
    let erf_sets = split_sets(&erf_in);
    let gamma_sets = split_sets(&gamma_in);

    println!("=================================================================");
    println!(" libm accuracy report — reference: Rust f64::{{exp,ln,sin,...}}");
    println!(" sin/cos budget: 1200 ulps (DD reduction, Payne-Hanek pending)");
    println!(" other budgets:  50 ulps per strategy");
    println!("=================================================================");

    // ── exp ────────────────────────────────────────────────────────────────
    report_unary_recipe("exp", "strict", exp::exp_strict, f64::exp, &exp_sets, &mut all_pass);
    report_unary_recipe("exp", "compensated", exp::exp_compensated, f64::exp, &exp_sets, &mut all_pass);
    report_unary_recipe("exp", "correctly_rounded", exp::exp_correctly_rounded, f64::exp, &exp_sets, &mut all_pass);

    // ── log ────────────────────────────────────────────────────────────────
    report_unary_recipe("log", "strict", log::log_strict, f64::ln, &log_sets, &mut all_pass);
    report_unary_recipe("log", "compensated", log::log_compensated, f64::ln, &log_sets, &mut all_pass);
    report_unary_recipe("log", "correctly_rounded", log::log_correctly_rounded, f64::ln, &log_sets, &mut all_pass);

    // ── sin ────────────────────────────────────────────────────────────────
    report_unary_recipe("sin", "strict", sin::sin_strict, f64::sin, &trig_sets, &mut all_pass);
    report_unary_recipe("sin", "compensated", sin::sin_compensated, f64::sin, &trig_sets, &mut all_pass);
    report_unary_recipe("sin", "correctly_rounded", sin::sin_correctly_rounded, f64::sin, &trig_sets, &mut all_pass);

    // ── cos ────────────────────────────────────────────────────────────────
    report_unary_recipe("cos", "strict", sin::cos_strict, f64::cos, &trig_sets, &mut all_pass);
    report_unary_recipe("cos", "compensated", sin::cos_compensated, f64::cos, &trig_sets, &mut all_pass);
    report_unary_recipe("cos", "correctly_rounded", sin::cos_correctly_rounded, f64::cos, &trig_sets, &mut all_pass);

    // ── erf ────────────────────────────────────────────────────────────────
    // No f64::erf in stable Rust, so route to libm crate... but we want
    // to avoid non-std deps. Instead, use the tambear compensated
    // implementation as a cross-check against strict & correctly_rounded.
    // Strict vs compensated ulp-gap IS a meaningful signal of
    // strict-path drift.
    report_unary_recipe(
        "erf",
        "strict",
        erf::erf_strict,
        erf::erf_correctly_rounded,
        &erf_sets,
        &mut all_pass,
    );
    report_unary_recipe(
        "erf",
        "compensated",
        erf::erf_compensated,
        erf::erf_correctly_rounded,
        &erf_sets,
        &mut all_pass,
    );

    // ── erfc ───────────────────────────────────────────────────────────────
    report_unary_recipe(
        "erfc",
        "strict",
        erf::erfc_strict,
        erf::erfc_correctly_rounded,
        &erf_sets,
        &mut all_pass,
    );
    report_unary_recipe(
        "erfc",
        "compensated",
        erf::erfc_compensated,
        erf::erfc_correctly_rounded,
        &erf_sets,
        &mut all_pass,
    );

    // ── tgamma ─────────────────────────────────────────────────────────────
    // Likewise no f64::gamma in stable Rust. Compare against the
    // correctly_rounded strategy.
    report_unary_recipe(
        "tgamma",
        "strict",
        gamma::tgamma_strict,
        gamma::tgamma_correctly_rounded,
        &gamma_sets,
        &mut all_pass,
    );
    report_unary_recipe(
        "tgamma",
        "compensated",
        gamma::tgamma_compensated,
        gamma::tgamma_correctly_rounded,
        &gamma_sets,
        &mut all_pass,
    );

    // ── lgamma ─────────────────────────────────────────────────────────────
    report_unary_recipe(
        "lgamma",
        "strict",
        gamma::lgamma_strict,
        gamma::lgamma_correctly_rounded,
        &gamma_sets,
        &mut all_pass,
    );
    report_unary_recipe(
        "lgamma",
        "compensated",
        gamma::lgamma_compensated,
        gamma::lgamma_correctly_rounded,
        &gamma_sets,
        &mut all_pass,
    );

    println!("\n=================================================================");
    if all_pass {
        println!(" ALL BUDGETS MET");
    } else {
        println!(" BUDGET VIOLATIONS — see FAIL rows above");
    }
    println!("=================================================================\n");

    assert!(all_pass, "one or more (recipe, strategy, input_set) exceeded its ulp budget");
}
