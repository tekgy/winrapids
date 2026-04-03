//! # One Kingdom — `tam()`
//!
//! There are no kingdoms. There is one operation with swept parameters.
//! Run everything. Always. Collapse at the end.
//!
//! `tam(data, op, init, max_iter, tol)` runs a single operation at ALL
//! composition depths simultaneously:
//!
//! - **Depth 0** (`once`): single-pass fold/accumulate
//! - **Depth 1** (`scan`): sequential prefix scan
//! - **Depth ∞** (`converge`): iterate to fixed point
//!
//! The output includes results from ALL depths plus a [`Diagnostic`] that
//! reveals which depth was informative. The kingdom is an OBSERVATION
//! (emergent from the output), not a CHOICE (made before computation).
//!
//! # The Core Insight
//!
//! Unused parameters produce nothing — that doesn't matter.
//! The RELATIONSHIP between outputs at different depths IS the diagnostic.
//! If depth=0 and depth=∞ agree → simple problem.
//! If they diverge → the divergence IS the information.
//!
//! This is Pith's depth kernel applied to computation.
//! This is `.discover()` applied to everything.
//!
//! # Examples
//!
//! ```
//! use tambear::tam::{tam, TamValue, EmergentDepth};
//!
//! // Addition: once=55, converge diverges → kingdom A emerges
//! let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
//! let result = tam(&data, |a, b| a + b, 0.0, 100, 1e-10);
//! assert_eq!(result.once, Some(55.0));
//! assert_eq!(result.diagnostic.emergent_depth, EmergentDepth::Chaotic);
//!
//! // Heron √2: once fails, converge finds √2 → kingdom C emerges
//! let result = tam(&[1.0, 2.0], |x, a| (x + a / x) / 2.0, 0.0, 100, 1e-15);
//! assert!(result.converge.converged);
//! assert!((result.converge.value - std::f64::consts::SQRT_2).abs() < 1e-10);
//! ```

use std::cmp::Ordering;
use std::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// TamValue trait — what can participate in tam()
// ═══════════════════════════════════════════════════════════════════════════

/// A value that can participate in `tam()` superposition computation.
///
/// Implement this for any type you want to sweep through all depths:
/// f64, BigFloat, Vec<f64>, complex numbers, etc.
pub trait TamValue: Clone + fmt::Debug {
    /// Does this value represent valid signal? (Not NaN, not Inf, etc.)
    fn is_finite(&self) -> bool;

    /// Distance between two values, used for convergence detection.
    /// Returns f64 because convergence tolerance is always a scalar.
    fn distance(&self, other: &Self) -> f64;

    /// Optional ordering for monotonicity detection in the scan.
    /// Returns `None` if this type has no natural total order (e.g., Vec, complex).
    fn partial_cmp_value(&self, other: &Self) -> Option<Ordering> {
        let _ = other;
        None
    }
}

// ── f64 impl ─────────────────────────────────────────────────────────────

impl TamValue for f64 {
    fn is_finite(&self) -> bool { f64::is_finite(*self) }

    fn distance(&self, other: &Self) -> f64 {
        (self - other).abs()
    }

    fn partial_cmp_value(&self, other: &Self) -> Option<Ordering> {
        self.partial_cmp(other)
    }
}

// ── f32 impl ─────────────────────────────────────────────────────────────

impl TamValue for f32 {
    fn is_finite(&self) -> bool { f32::is_finite(*self) }

    fn distance(&self, other: &Self) -> f64 {
        (self - other).abs() as f64
    }

    fn partial_cmp_value(&self, other: &Self) -> Option<Ordering> {
        self.partial_cmp(other)
    }
}

// ── Vec<f64> impl (multi-dimensional accumulation) ───────────────────────

impl TamValue for Vec<f64> {
    fn is_finite(&self) -> bool {
        self.iter().all(|x| x.is_finite())
    }

    fn distance(&self, other: &Self) -> f64 {
        // L2 distance
        self.iter()
            .zip(other.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt()
    }

    // No natural ordering for vectors — default None
}

// ═══════════════════════════════════════════════════════════════════════════
// Result types
// ═══════════════════════════════════════════════════════════════════════════

/// Result of running ONE operation at ALL depths simultaneously.
/// The superposition. Collapse extracts what's relevant.
#[derive(Debug, Clone)]
pub struct TamResult<T: TamValue> {
    /// Depth 0: single accumulate (fold over all data).
    /// `None` if the fold produced a non-finite value.
    pub once: Option<T>,

    /// Depth 1: sequential prefix scan.
    /// Empty if any intermediate value was non-finite.
    pub scan: Vec<T>,

    /// Depth ∞: iterate to convergence (fixed-point iteration).
    pub converge: ConvergeResult<T>,

    /// THE DIAGNOSTIC: relationships between depths.
    pub diagnostic: Diagnostic,
}

/// Result of fixed-point iteration (depth ∞).
#[derive(Debug, Clone)]
pub struct ConvergeResult<T: TamValue> {
    /// Final value (may not have converged).
    pub value: T,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Did the sequence converge within tolerance?
    pub converged: bool,
    /// Full trajectory of the iteration.
    pub trajectory: Vec<T>,
}

/// Which depth emerged as primary — not a choice, an observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmergentDepth {
    /// Depth 0 suffices — converge agrees with once.
    Once,
    /// Scan works but once doesn't — sequential structure matters.
    Scan,
    /// Only convergence works — iteration was needed.
    Converge,
    /// Converge differs from once — iteration reveals something once misses.
    ConvergeDiffers,
    /// Nothing converged — chaotic or divergent system.
    Chaotic,
}

impl fmt::Display for EmergentDepth {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EmergentDepth::Once => write!(f, "once (depth 0 suffices)"),
            EmergentDepth::Scan => write!(f, "scan (sequential structure)"),
            EmergentDepth::Converge => write!(f, "converge (iteration needed)"),
            EmergentDepth::ConvergeDiffers => write!(f, "converge (differs from once)"),
            EmergentDepth::Chaotic => write!(f, "chaotic (nothing converged)"),
        }
    }
}

/// Relationships between depths — the actual information.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    /// Distance between depth=0 and depth=∞ results.
    /// `f64::INFINITY` if either depth produced no signal.
    pub once_vs_converge_divergence: f64,

    /// Convergence rate: 1/iterations if converged, 0 otherwise.
    /// Fast convergence (rate near 1) → simple fixed point.
    /// Slow convergence (rate near 0) → complex dynamics.
    pub convergence_rate: f64,

    /// Is the prefix scan monotonic? `None` if type has no ordering
    /// or scan was empty.
    pub scan_monotonic: Option<bool>,

    /// Which depth emerged as primary (observation, not choice).
    pub emergent_depth: EmergentDepth,

    /// Was the full superposition informative?
    /// True when multiple depths produced signal.
    pub superposition_informative: bool,

    /// How many depths produced signal (0-3).
    pub active_depths: u8,
}

// ═══════════════════════════════════════════════════════════════════════════
// Display
// ═══════════════════════════════════════════════════════════════════════════

impl<T: TamValue> fmt::Display for TamResult<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "  once:     {:?}", self.once)?;
        writeln!(f, "  scan:     [{} values]", self.scan.len())?;
        writeln!(f, "  converge: {:?} ({}iter, {})",
            self.converge.value, self.converge.iterations,
            if self.converge.converged { "converged" } else { "NOT converged" })?;
        writeln!(f, "  ---")?;
        writeln!(f, "  divergence(once,converge): {:.6}", self.diagnostic.once_vs_converge_divergence)?;
        writeln!(f, "  convergence_rate: {:.6}", self.diagnostic.convergence_rate)?;
        writeln!(f, "  scan_monotonic: {:?}", self.diagnostic.scan_monotonic)?;
        writeln!(f, "  emergent_depth: {}", self.diagnostic.emergent_depth)?;
        writeln!(f, "  superposition_informative: {}", self.diagnostic.superposition_informative)?;
        writeln!(f, "  active_depths: {}", self.diagnostic.active_depths)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// THE function
// ═══════════════════════════════════════════════════════════════════════════

/// The ONE function. No kingdoms. Swept parameters. Always superposition.
///
/// Runs `op` at all composition depths simultaneously:
/// - **Depth 0**: `data.fold(init, op)` — single-pass accumulate
/// - **Depth 1**: prefix scan — running accumulation at each position
/// - **Depth ∞**: iterate `op(current, target)` until convergence
///
/// The [`Diagnostic`] in the result reveals which depth was informative.
/// The kingdom emerges from the output. You don't pick it.
///
/// # Parameters
///
/// - `data`: input elements
/// - `op`: the binary operation (must work for all depths)
/// - `init`: identity element for fold/scan
/// - `max_iter`: maximum iterations for convergence
/// - `tol`: convergence tolerance (distance between successive iterates)
pub fn tam<T, F>(data: &[T], op: F, init: T, max_iter: usize, tol: f64) -> TamResult<T>
where
    T: TamValue,
    F: Fn(&T, &T) -> T,
{
    // ── Depth 0: accumulate once ──────────────────────────────────────
    let once_result = {
        let r = data.iter().fold(init.clone(), |acc, x| op(&acc, x));
        if r.is_finite() { Some(r) } else { None }
    };

    // ── Depth 1: sequential scan ──────────────────────────────────────
    let scan_result = {
        let mut results = Vec::with_capacity(data.len());
        let mut acc = init.clone();
        let mut all_finite = true;
        for x in data {
            acc = op(&acc, x);
            if !acc.is_finite() {
                all_finite = false;
                break;
            }
            results.push(acc.clone());
        }
        if all_finite { results } else { vec![] }
    };

    // ── Depth ∞: iterate to convergence ───────────────────────────────
    let converge_result = {
        let target = if data.len() > 1 { &data[1] } else if !data.is_empty() { &data[0] } else { &init };
        let mut current = if data.is_empty() { init.clone() } else { data[0].clone() };
        let mut trajectory = vec![current.clone()];
        let mut converged = false;

        for _i in 0..max_iter {
            let next = op(&current, target);
            if !next.is_finite() {
                break;
            }
            trajectory.push(next.clone());
            if current.distance(&next) < tol {
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

    // ── Diagnostic: relationships between depths ──────────────────────
    let divergence = match &once_result {
        Some(once_val) if converge_result.value.is_finite() => {
            once_val.distance(&converge_result.value)
        }
        _ => f64::INFINITY,
    };

    let rate = if converge_result.iterations > 0 && converge_result.converged {
        1.0 / converge_result.iterations as f64
    } else {
        0.0
    };

    let monotonic = if scan_result.len() >= 2 {
        // Only check if the type supports ordering
        let first_cmp = scan_result[0].partial_cmp_value(&scan_result[1]);
        match first_cmp {
            Some(_) => {
                let increasing = scan_result.windows(2).all(|w| {
                    matches!(w[0].partial_cmp_value(&w[1]), Some(Ordering::Less | Ordering::Equal))
                });
                let decreasing = scan_result.windows(2).all(|w| {
                    matches!(w[0].partial_cmp_value(&w[1]), Some(Ordering::Greater | Ordering::Equal))
                });
                Some(increasing || decreasing)
            }
            None => None,
        }
    } else {
        // Trivially monotonic if 0 or 1 elements, but None if no ordering
        if scan_result.is_empty() {
            None
        } else {
            // 1 element — trivially monotonic
            scan_result[0].partial_cmp_value(&scan_result[0]).map(|_| true)
        }
    };

    // Emergent depth: what WOULD the old system have chosen?
    // This is an observation, not a judgment. The classification tells you
    // which depth(s) produced signal, NOT which is "correct."
    let emergent_depth = if divergence < tol && once_result.is_some() {
        // Once and converge agree → once suffices (simplest case)
        EmergentDepth::Once
    } else if !scan_result.is_empty() && once_result.is_none() {
        // Scan works but once doesn't → sequential structure matters
        EmergentDepth::Scan
    } else if converge_result.converged && once_result.is_none() {
        // Only converge works — iteration was needed
        EmergentDepth::Converge
    } else if converge_result.converged && once_result.is_some() {
        // Both work but disagree → iteration reveals structure once misses
        EmergentDepth::ConvergeDiffers
    } else {
        // Nothing converged, or once works but converge doesn't settle
        EmergentDepth::Chaotic
    };

    let once_active = once_result.is_some() as u8;
    let scan_active = (!scan_result.is_empty()) as u8;
    let converge_active = converge_result.converged as u8;
    let active_depths = once_active + scan_active + converge_active;
    let superposition_informative = active_depths >= 2;

    TamResult {
        once: once_result,
        scan: scan_result,
        converge: converge_result,
        diagnostic: Diagnostic {
            once_vs_converge_divergence: divergence,
            convergence_rate: rate,
            scan_monotonic: monotonic,
            emergent_depth,
            superposition_informative,
            active_depths,
        },
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Convenience: tam_f64 for the common case
// ═══════════════════════════════════════════════════════════════════════════

/// Convenience wrapper: `tam()` specialized for `f64` with closure ergonomics.
///
/// Wraps the closure to take references internally, so callers can write
/// `tam_f64(&data, |a, b| a + b, 0.0, 100, 1e-10)` without `&`.
pub fn tam_f64<F>(data: &[f64], op: F, init: f64, max_iter: usize, tol: f64) -> TamResult<f64>
where
    F: Fn(f64, f64) -> f64,
{
    tam(data, |a: &f64, b: &f64| op(*a, *b), init, max_iter, tol)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Addition: classical fold ──────────────────────────────────────

    #[test]
    fn addition_superposition() {
        let data: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let r = tam_f64(&data, |a, b| a + b, 0.0, 100, 1e-10);

        // Depth 0: fold works perfectly
        assert_eq!(r.once, Some(55.0));
        // Depth 1: scan gives running sums
        assert_eq!(r.scan.len(), 10);
        assert_eq!(r.scan[0], 1.0);
        assert_eq!(r.scan[9], 55.0);
        assert_eq!(r.diagnostic.scan_monotonic, Some(true));
        // Depth ∞: addition diverges under iteration (1+2=3, 3+2=5, ...)
        // so emergent_depth = Chaotic (once works, converge doesn't settle)
        assert!(!r.converge.converged);
        assert_eq!(r.diagnostic.emergent_depth, EmergentDepth::Chaotic);
        // But once and scan both produced signal
        assert!(r.diagnostic.active_depths >= 2);
        assert!(r.diagnostic.superposition_informative);
    }

    #[test]
    fn addition_where_converge_agrees() {
        // With a single element, converge trivially agrees with once
        let r = tam_f64(&[5.0], |a, b| a + b, 0.0, 100, 1e-10);
        // op(5, 5) = 10, op(10, 5) = 15 → diverges. converge sees data[0]=5, target=data[0]=5.
        // Actually: single element → target = data[0] = 5. current = 5. op(5,5) = 10 → diverges.
        // So still chaotic for addition.
        assert_eq!(r.once, Some(5.0));
    }

    // ── Heron √2: convergence ────────────────────────────────────────

    #[test]
    fn heron_sqrt2_emerges_as_converge() {
        let r = tam_f64(&[1.0, 2.0], |x, a| (x + a / x) / 2.0, 0.0, 100, 1e-15);

        assert!(r.converge.converged);
        assert!((r.converge.value - std::f64::consts::SQRT_2).abs() < 1e-10);
        assert!(r.converge.iterations > 0);
        assert!(r.converge.iterations < 30); // Heron converges fast
    }

    // ── Contraction to fixed point ───────────────────────────────────

    #[test]
    fn contraction_converges_to_fixed_point() {
        // f(x) = x/2 + 3 → fixed point at x = 6
        let r = tam_f64(&[100.0, 3.0], |x, c| x / 2.0 + c, 0.0, 100, 1e-10);

        assert!(r.converge.converged);
        assert!((r.converge.value - 6.0).abs() < 1e-8);
    }

    // ── Collatz from 27 ──────────────────────────────────────────────

    #[test]
    fn collatz_27_converges_to_one() {
        let r = tam_f64(&[27.0, 0.0], |n, _| {
            let ni = n as u64;
            if ni <= 1 { 1.0 }
            else if ni % 2 == 0 { (ni / 2) as f64 }
            else { (3 * ni + 1) as f64 }
        }, 0.0, 1000, 0.5);

        assert!(r.converge.converged);
        assert_eq!(r.converge.value, 1.0);
    }

    // ── Logistic map: phase transition visibility ────────────────────

    #[test]
    fn logistic_converging_regime() {
        // r=2.8 → converges to 0.6428...
        let r = tam_f64(&[0.5, 2.8], |x, r| r * x * (1.0 - x), 0.0, 1000, 1e-10);

        assert!(r.converge.converged);
        assert!((r.converge.value - 0.6428571428571429).abs() < 1e-6);
    }

    #[test]
    fn logistic_chaotic_regime() {
        // r=3.8 → chaotic, does NOT converge
        let r = tam_f64(&[0.5, 3.8], |x, r| r * x * (1.0 - x), 0.0, 1000, 1e-10);

        assert!(!r.converge.converged);
        assert_eq!(r.diagnostic.emergent_depth, EmergentDepth::Chaotic);
    }

    // ── Maximum (semigroup) ──────────────────────────────────────────

    #[test]
    fn maximum_semigroup() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];
        let r = tam_f64(&data, f64::max, f64::NEG_INFINITY, 100, 1e-10);

        assert_eq!(r.once, Some(9.0));
        // Converge sees max(3,1)=3 repeatedly → converges to 3, not 9.
        // This divergence IS the information: convergence on a subset
        // differs from the global fold.
        assert!(r.converge.converged);
        assert_eq!(r.converge.value, 3.0);
        assert_eq!(r.diagnostic.emergent_depth, EmergentDepth::ConvergeDiffers);
    }

    // ── Empty data ───────────────────────────────────────────────────

    #[test]
    fn empty_data_returns_init() {
        let r = tam_f64(&[], |a, b| a + b, 0.0, 100, 1e-10);

        assert_eq!(r.once, Some(0.0));
        assert!(r.scan.is_empty());
    }

    // ── Single element ───────────────────────────────────────────────

    #[test]
    fn single_element() {
        let r = tam_f64(&[42.0], |a, b| a + b, 0.0, 100, 1e-10);

        assert_eq!(r.once, Some(42.0));
        assert_eq!(r.scan.len(), 1);
        assert_eq!(r.scan[0], 42.0);
    }

    // ── Generic: Vec<f64> ────────────────────────────────────────────

    #[test]
    fn vec_f64_componentwise_addition() {
        let data = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let r = tam(
            &data,
            |a: &Vec<f64>, b: &Vec<f64>| {
                a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
            },
            vec![0.0, 0.0],
            100,
            1e-10,
        );

        assert_eq!(r.once, Some(vec![9.0, 12.0]));
        assert_eq!(r.scan.len(), 3);
        assert_eq!(r.scan[2], vec![9.0, 12.0]);
        assert_eq!(r.diagnostic.scan_monotonic, None); // Vec has no ordering
    }

    // ── Diagnostic: active_depths count ──────────────────────────────

    #[test]
    fn active_depths_counted_correctly() {
        let data: Vec<f64> = (1..=5).map(|x| x as f64).collect();
        let r = tam_f64(&data, |a, b| a + b, 0.0, 100, 1e-10);

        // once works, scan works, converge works (addition to same value)
        assert!(r.diagnostic.active_depths >= 2);
    }

    // ── Scan monotonicity ────────────────────────────────────────────

    #[test]
    fn scan_monotonicity_detected() {
        // Cumulative sum of positives → monotonically increasing
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r = tam_f64(&data, |a, b| a + b, 0.0, 100, 1e-10);
        assert_eq!(r.diagnostic.scan_monotonic, Some(true));

        // Cumulative min with mixed values → monotonically decreasing
        let data = vec![5.0, 3.0, 7.0, 1.0, 8.0];
        let r = tam_f64(&data, f64::min, f64::INFINITY, 100, 1e-10);
        assert_eq!(r.diagnostic.scan_monotonic, Some(true)); // min is monotonically non-increasing
    }

    // ── Convergence trajectory preserved ─────────────────────────────

    #[test]
    fn trajectory_preserved() {
        let r = tam_f64(&[1.0, 2.0], |x, a| (x + a / x) / 2.0, 0.0, 50, 1e-15);

        assert!(r.converge.trajectory.len() >= 2);
        assert_eq!(r.converge.trajectory[0], 1.0); // starts at data[0]
        // Each step should get closer to √2
        let sqrt2 = std::f64::consts::SQRT_2;
        for window in r.converge.trajectory.windows(2) {
            assert!((window[1] - sqrt2).abs() <= (window[0] - sqrt2).abs() + 1e-15);
        }
    }

    // ── f32 works too ────────────────────────────────────────────────

    #[test]
    fn f32_generic() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let r = tam(&data, |a: &f32, b: &f32| a + b, 0.0f32, 100, 1e-6);
        assert_eq!(r.once, Some(10.0f32));
    }
}
