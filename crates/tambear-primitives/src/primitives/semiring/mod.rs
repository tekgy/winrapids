//! Semiring trait and standard instances.
//!
//! A semiring (S, ⊕, ⊗, 0, 1) is the algebraic structure underlying
//! every Kingdom A parallel scan. The `add` operation is what the scan
//! combines. The `mul` operation is how elements transform.
//!
//! Every Kingdom A computation is a prefix scan over some semiring.
//! Different semirings give different algorithms:
//!
//! | Semiring | add | mul | Algorithm |
//! |----------|-----|-----|-----------|
//! | Additive | + | × | cumsum, dot product |
//! | TropicalMinPlus | min | + | shortest path, PELT |
//! | TropicalMaxPlus | max | + | Viterbi, longest path |
//! | LogSumExp | lse | + | HMM forward, softmax |
//! | Boolean | OR | AND | reachability |
//! | MaxTimes | max | × | Viterbi (probability space) |

use crate::nan_guard::{nan_min, nan_max};
use crate::log_sum_exp::log_sum_exp_pair;

/// A semiring: two associative operations where add is commutative
/// and mul distributes over add.
pub trait Semiring: Clone + Copy + std::fmt::Debug + 'static {
    type Elem: Clone + Copy + std::fmt::Debug + PartialEq;

    /// Additive identity: add(zero, x) = x.
    fn zero() -> Self::Elem;
    /// Multiplicative identity: mul(one, x) = x.
    fn one() -> Self::Elem;
    /// Associative commutative combine (the scan operation).
    fn add(a: Self::Elem, b: Self::Elem) -> Self::Elem;
    /// Associative multiplication (distributes over add).
    fn mul(a: Self::Elem, b: Self::Elem) -> Self::Elem;
    /// Degenerate value for invalid/empty input. NOT used for padding.
    fn degenerate() -> Self::Elem;
}

// ═══════════════════════════════════════════════════════════════════
// Standard instances
// ═══════════════════════════════════════════════════════════════════

/// (ℝ, +, ×) — the default. Cumulative sums, dot products, moments.
#[derive(Debug, Clone, Copy)]
pub struct Additive;

impl Semiring for Additive {
    type Elem = f64;
    fn zero() -> f64 { 0.0 }
    fn one() -> f64 { 1.0 }
    fn add(a: f64, b: f64) -> f64 { a + b }
    fn mul(a: f64, b: f64) -> f64 { a * b }
    fn degenerate() -> f64 { f64::NAN }
}

/// (ℝ∪{∞}, min, +) — shortest path, PELT unpruned, Dijkstra.
#[derive(Debug, Clone, Copy)]
pub struct TropicalMinPlus;

impl Semiring for TropicalMinPlus {
    type Elem = f64;
    fn zero() -> f64 { f64::INFINITY }
    fn one() -> f64 { 0.0 }
    fn add(a: f64, b: f64) -> f64 { nan_min(a, b) }
    fn mul(a: f64, b: f64) -> f64 { a + b }
    fn degenerate() -> f64 { f64::NAN }
}

/// (ℝ∪{-∞}, max, +) — Viterbi, longest path.
#[derive(Debug, Clone, Copy)]
pub struct TropicalMaxPlus;

impl Semiring for TropicalMaxPlus {
    type Elem = f64;
    fn zero() -> f64 { f64::NEG_INFINITY }
    fn one() -> f64 { 0.0 }
    fn add(a: f64, b: f64) -> f64 { nan_max(a, b) }
    fn mul(a: f64, b: f64) -> f64 { a + b }
    fn degenerate() -> f64 { f64::NAN }
}

/// (ℝ, lse, +) — HMM forward, softmax, attention, Baum-Welch.
#[derive(Debug, Clone, Copy)]
pub struct LogSumExp;

impl Semiring for LogSumExp {
    type Elem = f64;
    fn zero() -> f64 { f64::NEG_INFINITY }
    fn one() -> f64 { 0.0 }
    fn add(a: f64, b: f64) -> f64 { log_sum_exp_pair(a, b) }
    fn mul(a: f64, b: f64) -> f64 { a + b }
    fn degenerate() -> f64 { f64::NAN }
}

/// ({0,1}, ∨, ∧) — reachability, connectivity, transitive closure.
#[derive(Debug, Clone, Copy)]
pub struct Boolean;

impl Semiring for Boolean {
    type Elem = bool;
    fn zero() -> bool { false }
    fn one() -> bool { true }
    fn add(a: bool, b: bool) -> bool { a || b }
    fn mul(a: bool, b: bool) -> bool { a && b }
    fn degenerate() -> bool { false }
}

/// (ℝ, max, ×) — Viterbi in probability space (not log space).
#[derive(Debug, Clone, Copy)]
pub struct MaxTimes;

impl Semiring for MaxTimes {
    type Elem = f64;
    fn zero() -> f64 { 0.0 }       // max identity: nothing beats 0 probability
    fn one() -> f64 { 1.0 }        // multiplicative identity
    fn add(a: f64, b: f64) -> f64 { nan_max(a, b) }
    fn mul(a: f64, b: f64) -> f64 { a * b }
    fn degenerate() -> f64 { f64::NAN }
}

#[cfg(test)]
mod tests;
