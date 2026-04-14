//! mpmath-backed oracle for tambear libm recipes.
//!
//! # Why this exists
//!
//! The existing `tests/libm_accuracy_report.rs` uses Rust's `f64::sin`,
//! `f64::ln`, etc. as the reference. On x86-64 Windows with MSVC CRT, those
//! can be up to 1 ulp off the true value — so our "≤ 1 ulp worst case"
//! measurement is relative to a potentially-1-ulp-off reference. This harness
//! calls mpmath at 100-digit precision via Python subprocess, giving us a
//! true gold-standard oracle.
//!
//! # Behavior
//!
//! - Marked `#[ignore]` by default so CI and normal `cargo test` don't run it.
//! - Opt in with: `cargo test --test mpmath_oracle -- --ignored --nocapture`
//! - Requires Python with `mpmath` installed. If Python is missing or mpmath
//!   isn't installed, the test prints a diagnostic and skips (does not fail).
//! - Produces a ulp-distance report per function comparing tambear vs mpmath.
//!
//! # What this answers
//!
//! "Is tambear's reported ≤ 1 ulp accuracy REAL, or is it an artifact of the
//! OS libm being slightly off?" After running this harness, we know.

use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::OnceLock;

use tambear::primitives::oracle::ulps_between;

/// Return true iff Python + mpmath are available on this system.
fn python_has_mpmath() -> bool {
    static RESULT: OnceLock<bool> = OnceLock::new();
    *RESULT.get_or_init(|| {
        let check = Command::new("python")
            .args(["-c", "import mpmath; print(mpmath.__version__)"])
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output();
        match check {
            Ok(out) => out.status.success(),
            Err(_) => {
                // Fallback: try python3
                let check3 = Command::new("python3")
                    .args(["-c", "import mpmath"])
                    .output();
                check3.map(|o| o.status.success()).unwrap_or(false)
            }
        }
    })
}

/// Pick the python executable that has mpmath.
fn python_exe() -> &'static str {
    static EXE: OnceLock<&'static str> = OnceLock::new();
    EXE.get_or_init(|| {
        if Command::new("python")
            .args(["-c", "import mpmath"])
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
        {
            "python"
        } else {
            "python3"
        }
    })
}

/// Evaluate `mpmath_expr` (a Python expression that returns an mpmath value)
/// for every x in `xs`, at 100-digit precision. Returns the rounded-to-f64
/// result for each input.
///
/// `mpmath_expr` is a Python expression string with a single free variable
/// `x` that takes an mpmath.mpf value. Examples:
///   "mpmath.sin(x)"        — sine
///   "mpmath.tan(x)"        — tangent
///   "mpmath.exp(x)"        — natural exponential
///   "mpmath.atan2(x, 1)"   — atan with denom fixed at 1
fn mpmath_eval(mpmath_expr: &str, xs: &[f64]) -> Vec<f64> {
    let exe = python_exe();
    let script = format!(
        r#"
import mpmath
import struct
import sys
mpmath.mp.dps = 100

def f64_to_bits_str(x):
    # mpmath float -> IEEE 754 f64 -> hex
    f = float(x)
    if f != f:  # NaN
        return "nan"
    bits = struct.unpack('<Q', struct.pack('<d', f))[0]
    return f"{{bits:016x}}"

lines = sys.stdin.read().strip().splitlines()
for line in lines:
    x_bits = int(line, 16)
    x = struct.unpack('<d', struct.pack('<Q', x_bits))[0]
    if x != x:
        print("nan")
        continue
    x_mp = mpmath.mpf(x)
    y = {expr}
    print(f64_to_bits_str(y))
"#,
        expr = mpmath_expr
    );

    let mut child = Command::new(exe)
        .args(["-c", &script])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("failed to launch python for mpmath oracle");

    // Write input bit patterns
    let stdin = child.stdin.as_mut().expect("python stdin");
    for &x in xs {
        let bits = x.to_bits();
        writeln!(stdin, "{bits:016x}").expect("write to python");
    }
    drop(child.stdin.take());

    let out = child.wait_with_output().expect("python wait");
    if !out.status.success() {
        let stderr = String::from_utf8_lossy(&out.stderr);
        panic!("mpmath oracle python failed: {stderr}");
    }

    let stdout = String::from_utf8_lossy(&out.stdout);
    let mut results = Vec::with_capacity(xs.len());
    for line in stdout.lines() {
        let line = line.trim();
        if line == "nan" {
            results.push(f64::NAN);
        } else {
            let bits = u64::from_str_radix(line, 16)
                .unwrap_or_else(|_| panic!("bad python output: {line:?}"));
            results.push(f64::from_bits(bits));
        }
    }
    assert_eq!(
        results.len(),
        xs.len(),
        "mpmath oracle returned wrong count: {} vs {}",
        results.len(),
        xs.len()
    );
    results
}

/// Core comparator: for every x in `xs`, compare `tambear_fn(x)` against
/// `mpmath(x)`. Report the worst-case ulp distance and the input that
/// produced it.
struct OracleReport {
    function: String,
    n_samples: usize,
    worst_ulps: u64,
    worst_x: f64,
    p50_ulps: u64,
    p95_ulps: u64,
    p99_ulps: u64,
    /// How often the tambear result disagreed with mpmath (in any ulp count).
    n_disagreements: usize,
}

fn run_oracle<F: Fn(f64) -> f64>(
    name: &str,
    mpmath_expr: &str,
    xs: &[f64],
    tambear_fn: F,
) -> OracleReport {
    let mpmath_results = mpmath_eval(mpmath_expr, xs);

    let mut deviations: Vec<u64> = Vec::with_capacity(xs.len());
    let mut worst_ulps = 0_u64;
    let mut worst_x = f64::NAN;
    let mut n_disagreements = 0_usize;

    for (i, &x) in xs.iter().enumerate() {
        let got = tambear_fn(x);
        let expected = mpmath_results[i];
        let dist = ulps_between(got, expected);
        deviations.push(dist);
        if dist > 0 {
            n_disagreements += 1;
        }
        if dist > worst_ulps {
            worst_ulps = dist;
            worst_x = x;
        }
    }

    deviations.sort_unstable();
    let n = deviations.len();
    let p50_ulps = deviations[n / 2];
    let p95_ulps = deviations[(n * 95) / 100];
    let p99_ulps = deviations[(n * 99) / 100];

    OracleReport {
        function: name.to_string(),
        n_samples: xs.len(),
        worst_ulps,
        worst_x,
        p50_ulps,
        p95_ulps,
        p99_ulps,
        n_disagreements,
    }
}

fn print_report(r: &OracleReport) {
    println!(
        "  {:<40} n={:<6} worst={:<8} worst_x={:<24e} p99={:<6} p95={:<6} p50={:<6} disagreements={}/{}",
        r.function,
        r.n_samples,
        r.worst_ulps,
        r.worst_x,
        r.p99_ulps,
        r.p95_ulps,
        r.p50_ulps,
        r.n_disagreements,
        r.n_samples,
    );
}

// ── Input generators ─────────────────────────────────────────────────────────

/// A compact set of inputs that stresses every regime we care about for
/// sin/cos/tan-family functions. We keep the count small (~50) because
/// each mpmath call is a Python subprocess round-trip — running thousands
/// would be slow. This is a spot-check, not exhaustive coverage.
fn trig_samples() -> Vec<f64> {
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, LN_2, PI};
    vec![
        0.0, -0.0, 0.1, -0.1, 0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0, -3.0,
        FRAC_PI_4, FRAC_PI_2, PI, -FRAC_PI_2, -PI,
        LN_2, 1.0 / LN_2,
        // Moderate — range reduction kicks in
        10.0, 100.0, 1000.0, -10.0, -1000.0,
        // Larger — should still be accurate
        1e6, 1e10, 1e15, -1e6, -1e10, -1e15,
        // Near Payne-Hanek threshold (1.647e6)
        1_647_099.0, -1_647_099.0,
        // Huge — Payne-Hanek should kick in
        1e17, -1e17,
        // Tiny
        1e-100, 1e-200, 2.0_f64.powi(-1000),
    ]
}

fn exp_samples() -> Vec<f64> {
    vec![
        0.0, -0.0, 0.5, 1.0, 2.0, -1.0, -2.0,
        std::f64::consts::LN_2, std::f64::consts::E,
        10.0, -10.0, 100.0, -100.0, 500.0, -500.0,
        // Near overflow / underflow
        709.0, -745.0,
        0.1, -0.1, 1e-10, -1e-10,
    ]
}

fn log_samples() -> Vec<f64> {
    vec![
        1.0, 0.5, 2.0, std::f64::consts::E,
        0.1, 10.0, 100.0, 1000.0, 1e10, 1e100,
        0.01, 1e-10, 1e-100,
        f64::MIN_POSITIVE, f64::MAX,
    ]
}

// ── The tests ────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn oracle_report() {
    if !python_has_mpmath() {
        eprintln!(
            "SKIPPED: Python with mpmath is required for the oracle report.\n\
             Install: pip install mpmath\n\
             Then: cargo test --test mpmath_oracle -- --ignored --nocapture"
        );
        return;
    }

    println!("\n=== tambear libm vs mpmath@100-digit oracle ===\n");
    println!("  reports: worst / p99 / p95 / p50 ulp distance");
    println!(
        "  disagreements counts any non-zero ulp distance (not just violations)\n"
    );

    let mut reports = Vec::new();

    // ── sin / cos ────────────────────────────────────────────────────────
    let trig_xs = trig_samples();

    println!("=== Sin family ===");
    let r = run_oracle(
        "sin_strict",
        "mpmath.sin(x)",
        &trig_xs,
        tambear::recipes::libm::sin::sin_strict,
    );
    print_report(&r);
    reports.push(r);
    let r = run_oracle(
        "sin_compensated",
        "mpmath.sin(x)",
        &trig_xs,
        tambear::recipes::libm::sin::sin_compensated,
    );
    print_report(&r);
    reports.push(r);
    let r = run_oracle(
        "sin_correctly_rounded",
        "mpmath.sin(x)",
        &trig_xs,
        tambear::recipes::libm::sin::sin_correctly_rounded,
    );
    print_report(&r);
    reports.push(r);

    println!("\n=== Cos family ===");
    let r = run_oracle(
        "cos_strict",
        "mpmath.cos(x)",
        &trig_xs,
        tambear::recipes::libm::sin::cos_strict,
    );
    print_report(&r);
    reports.push(r);
    let r = run_oracle(
        "cos_correctly_rounded",
        "mpmath.cos(x)",
        &trig_xs,
        tambear::recipes::libm::sin::cos_correctly_rounded,
    );
    print_report(&r);
    reports.push(r);

    // ── exp ──────────────────────────────────────────────────────────────
    let exp_xs = exp_samples();
    println!("\n=== Exp family ===");
    let r = run_oracle(
        "exp_strict",
        "mpmath.exp(x)",
        &exp_xs,
        tambear::recipes::libm::exp::exp_strict,
    );
    print_report(&r);
    reports.push(r);
    let r = run_oracle(
        "exp_correctly_rounded",
        "mpmath.exp(x)",
        &exp_xs,
        tambear::recipes::libm::exp::exp_correctly_rounded,
    );
    print_report(&r);
    reports.push(r);

    // ── log ──────────────────────────────────────────────────────────────
    let log_xs = log_samples();
    println!("\n=== Log family ===");
    let r = run_oracle(
        "log_strict",
        "mpmath.log(x)",
        &log_xs,
        tambear::recipes::libm::log::log_strict,
    );
    print_report(&r);
    reports.push(r);
    let r = run_oracle(
        "log_correctly_rounded",
        "mpmath.log(x)",
        &log_xs,
        tambear::recipes::libm::log::log_correctly_rounded,
    );
    print_report(&r);
    reports.push(r);

    // ── Summary ──────────────────────────────────────────────────────────
    println!("\n=== Summary ===");
    let max_worst = reports.iter().map(|r| r.worst_ulps).max().unwrap_or(0);
    let total_samples: usize = reports.iter().map(|r| r.n_samples).sum();
    let total_disagreements: usize = reports.iter().map(|r| r.n_disagreements).sum();

    println!("  overall worst ulps across all functions: {max_worst}");
    println!("  total samples tested: {total_samples}");
    println!("  total disagreements with mpmath: {total_disagreements}");
    println!(
        "  bit-perfect rate: {:.2}%",
        100.0 * ((total_samples - total_disagreements) as f64) / (total_samples as f64)
    );

    // Sanity assertion: if ANY recipe is 100+ ulps off mpmath, something
    // is structurally wrong — fail the test so we notice.
    for r in &reports {
        assert!(
            r.worst_ulps < 100,
            "{} is {} ulps off mpmath — something structural is broken. worst_x = {:e}",
            r.function,
            r.worst_ulps,
            r.worst_x
        );
    }
}
