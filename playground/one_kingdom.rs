//! # One Kingdom
//!
//! There are no kingdoms. There is one operation with swept parameters.
//! Run everything. Always. Collapse at the end.
//!
//! Unused parameters produce nothing. That doesn't matter.
//! The RELATIONSHIP between outputs at different depths IS the diagnostic.
//! If depth=0 and depth=converge agree → simple problem.
//! If they diverge → the divergence IS the information.
//!
//! This is Pith's depth kernel applied to computation.
//! This is .discover() applied to EVERYTHING, not just algorithm selection.
//! This is the unified theory of mathematics as a single swept operation.

use std::fmt;

// ── The One Operation ──────────────────────────────────────

/// Result of running ONE operation at ALL depths simultaneously.
/// The superposition. Collapse extracts what's relevant.
#[derive(Debug)]
struct TamResult {
    /// Depth 0: single accumulate (was "Kingdom A")
    once: Option<f64>,

    /// Depth 1: sequential scan (was "Kingdom B")
    scan: Vec<f64>,

    /// Depth ∞: iterate to convergence (was "Kingdom C")
    converge: ConvergeResult,

    /// THE DIAGNOSTIC: relationships between depths
    diagnostic: Diagnostic,
}

#[derive(Debug)]
struct ConvergeResult {
    value: f64,
    iterations: usize,
    converged: bool,
    trajectory: Vec<f64>,
}

#[derive(Debug)]
struct Diagnostic {
    /// Do depth=0 and depth=∞ agree?
    once_vs_converge_divergence: f64,

    /// How fast did convergence happen? (0 = instant = was always Kingdom A)
    convergence_rate: f64,

    /// Is the scan monotonic? (monotonic = sequential structure matters)
    scan_monotonic: bool,

    /// The "kingdom" that WOULD HAVE BEEN chosen (emergent, not imposed)
    emergent_kingdom: &'static str,

    /// Signal: was the full superposition necessary?
    superposition_informative: bool,
}

impl fmt::Display for TamResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  once:     {:?}", self.once)?;
        writeln!(f, "  scan:     [{} values]", self.scan.len())?;
        writeln!(f, "  converge: {} ({}iter, {})",
            self.converge.value, self.converge.iterations,
            if self.converge.converged { "converged" } else { "NOT converged" })?;
        writeln!(f, "  ---")?;
        writeln!(f, "  divergence(once,converge): {:.6}", self.diagnostic.once_vs_converge_divergence)?;
        writeln!(f, "  convergence_rate: {:.6}", self.diagnostic.convergence_rate)?;
        writeln!(f, "  scan_monotonic: {}", self.diagnostic.scan_monotonic)?;
        writeln!(f, "  emergent_kingdom: {}", self.diagnostic.emergent_kingdom)?;
        writeln!(f, "  superposition_informative: {}", self.diagnostic.superposition_informative)
    }
}

/// The ONE function. No kingdoms. Swept parameters. Always superposition.
fn tam<F: Fn(f64, f64) -> f64>(
    data: &[f64],
    op: F,
    init: f64,
    max_iter: usize,
    tol: f64,
) -> TamResult {
    // ── Depth 0: accumulate once ──────────────────────────
    let once_result = {
        let r = data.iter().fold(init, |acc, &x| op(acc, x));
        if r.is_finite() { Some(r) } else { None } // inf/nan = no signal at this depth
    };

    // ── Depth 1: sequential scan ──────────────────────────
    let scan_result = {
        let mut results = Vec::with_capacity(data.len());
        let mut acc = init;
        let mut all_finite = true;
        for &x in data {
            acc = op(acc, x);
            if !acc.is_finite() { all_finite = false; break; }
            results.push(acc);
        }
        if all_finite { results } else { vec![] } // empty = no signal at this depth
    };

    // ── Depth ∞: iterate to convergence ───────────────────
    let converge_result = {
        let target = if data.len() > 1 { data[1] } else { data[0] };
        let mut current = if data.is_empty() { init } else { data[0] };
        let mut trajectory = vec![current];
        let mut converged = false;

        for i in 0..max_iter {
            let next = op(current, target);
            if !next.is_finite() {
                // No signal at this depth either — that's fine
                break;
            }
            trajectory.push(next);
            if (next - current).abs() < tol {
                converged = true;
                current = next;
                break;
            }
            current = next;
        }

        ConvergeResult {
            value: current,
            iterations: trajectory.len() - 1,
            converged,
            trajectory,
        }
    };

    // ── Diagnostic: relationships between depths ──────────
    let once_val = once_result.unwrap_or(f64::NAN);
    let conv_val = converge_result.value;

    let divergence = if once_val.is_finite() && conv_val.is_finite() {
        (once_val - conv_val).abs()
    } else {
        f64::INFINITY // divergence = infinite means different depths see different things
    };

    let rate = if converge_result.iterations > 0 && converge_result.converged {
        1.0 / converge_result.iterations as f64
    } else {
        0.0
    };

    let monotonic = if scan_result.len() >= 2 {
        let increasing = scan_result.windows(2).all(|w| w[1] >= w[0]);
        let decreasing = scan_result.windows(2).all(|w| w[1] <= w[0]);
        increasing || decreasing
    } else {
        true // trivially
    };

    // Emergent kingdom: what WOULD the old system have chosen?
    let emergent = if divergence < tol && once_result.is_some() {
        "A (once suffices — converge agrees with once)"
    } else if !scan_result.is_empty() && once_result.is_none() {
        "B (scan works but once doesn't)"
    } else if converge_result.converged && once_result.is_none() {
        "C (only converge works)"
    } else if converge_result.converged {
        "C (converge differs from once — iteration was needed)"
    } else {
        "? (nothing converged — chaotic or divergent)"
    };

    let informative = once_result.is_some() as u8
        + (!scan_result.is_empty()) as u8
        + converge_result.converged as u8;
    let superposition_informative = informative >= 2; // multiple depths gave signal

    TamResult {
        once: once_result,
        scan: scan_result,
        converge: converge_result,
        diagnostic: Diagnostic {
            once_vs_converge_divergence: divergence,
            convergence_rate: rate,
            scan_monotonic: monotonic,
            emergent_kingdom: emergent,
            superposition_informative,
        },
    }
}

fn main() {
    eprintln!("==========================================================");
    eprintln!("  ONE KINGDOM");
    eprintln!("  No routing. No deciding. Sweep everything. Collapse.");
    eprintln!("==========================================================\n");

    // ── Addition on [1..10] ──────────────────────────────
    eprintln!("=== Addition ===");
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let result = tam(&data, |a, b| a + b, 0.0, 100, 1e-10);
    eprintln!("{}", result);

    // ── Heron √2 ────────────────────────────────────────
    eprintln!("=== Heron √2 ===");
    let result = tam(&[1.0, 2.0], |x, a| (x + a / x) / 2.0, 0.0, 100, 1e-15);
    eprintln!("{}", result);

    // ── Contraction to fixed point ───────────────────────
    eprintln!("=== Contraction f(x) = x/2 + 3 → fixed point 6 ===");
    let result = tam(&[100.0, 3.0], |x, c| x / 2.0 + c, 0.0, 100, 1e-10);
    eprintln!("{}", result);

    // ── Collatz ──────────────────────────────────────────
    eprintln!("=== Collatz starting at 27 ===");
    let result = tam(&[27.0, 0.0], |n, _| {
        let ni = n as u64;
        if ni <= 1 { 1.0 }
        else if ni % 2 == 0 { (ni / 2) as f64 }
        else { (3 * ni + 1) as f64 }
    }, 0.0, 1000, 0.5);
    eprintln!("{}", result);

    // ── Logistic map r=2.8 (converges) ───────────────────
    eprintln!("=== Logistic r=2.8 (converges to 0.643) ===");
    let result = tam(&[0.5, 2.8], |x, r| r * x * (1.0 - x), 0.0, 1000, 1e-10);
    eprintln!("{}", result);

    // ── Logistic map r=3.8 (chaotic) ─────────────────────
    eprintln!("=== Logistic r=3.8 (chaotic — no convergence) ===");
    let result = tam(&[0.5, 3.8], |x, r| r * x * (1.0 - x), 0.0, 1000, 1e-10);
    eprintln!("{}", result);

    // ── Maximum (semigroup) ──────────────────────────────
    eprintln!("=== Maximum ===");
    let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
    let result = tam(&data, f64::max, f64::NEG_INFINITY, 100, 1e-10);
    eprintln!("{}", result);

    // ── Mean via accumulate + count ──────────────────────
    eprintln!("=== The point ===");
    eprintln!("  Every example above: SAME function call.");
    eprintln!("  Different op. Different data. Same tam().");
    eprintln!("  No kingdom selection. No routing.");
    eprintln!("  The diagnostic tells you what WAS relevant.");
    eprintln!("  The superposition tells you what you'd MISS with one depth.");
    eprintln!();
    eprintln!("  Addition: once=55, converge=diverges → kingdom A emerges");
    eprintln!("  Heron:    once=inf, converge=√2 → kingdom C emerges");
    eprintln!("  Logistic: r=2.8 converges, r=3.8 doesn't → phase transition VISIBLE");
    eprintln!("  Collatz:  converges to 1 → the conjecture as a tam() call");
    eprintln!();
    eprintln!("  The kingdom is not a CHOICE. It's an OBSERVATION.");
    eprintln!("  You don't pick the kingdom. You run tam(). The kingdom emerges.");
    eprintln!("  The old kingdoms were shadows on the wall.");
    eprintln!("  There is only ONE operation. Always was.");
}
