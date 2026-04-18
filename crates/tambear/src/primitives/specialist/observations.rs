//! `FiniteObservations` — shared bookkeeping for streaming/mergeable accumulators.
//!
//! Locked vocabulary: this is a Tier 1 primitive (low-level
//! implementation machinery). It captures the pattern that appeared in
//! all three quantile sketches (KLL / GK / t-digest) and is general
//! enough to be useful for any future streaming accumulator that
//! tracks "how many finite values have we seen, and what's the
//! min/max" alongside its own algorithm-specific state. See
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # The shared pattern
//!
//! Across KLL, GK, and t-digest the same three lines appeared in every
//! `add()`:
//!
//! ```text
//! if !x.is_finite() { return; }
//! self.n += 1;
//! if self.min.is_nan() || x < self.min { self.min = x; }
//! if self.max.is_nan() || x > self.max { self.max = x; }
//! ```
//!
//! And the same three lines appeared in every `merge()`:
//!
//! ```text
//! self.n = self.n.saturating_add(other.n);
//! if !other.min.is_nan() && (self.min.is_nan() || other.min < self.min) {
//!     self.min = other.min;
//! }
//! if !other.max.is_nan() && (self.max.is_nan() || other.max > self.max) {
//!     self.max = other.max;
//! }
//! ```
//!
//! Plus identical `count()`, `min()`, `max()` accessors.
//!
//! Extracting this into one struct: every accumulator that tracks
//! finite-input bookkeeping composes a `FiniteObservations` field
//! and delegates the four operations (`observe`, `merge`, `count`,
//! `min`, `max`).
//!
//! # NaN/Inf semantics (locked)
//!
//! `observe(x)` returns `true` if `x` is finite (and bookkeeping was
//! updated), `false` otherwise. Callers gate their algorithm-specific
//! logic on the return value:
//!
//! ```ignore
//! fn add(&mut self, x: f64) {
//!     if !self.obs.observe(x) {
//!         return;
//!     }
//!     // ... algorithm-specific add code
//! }
//! ```
//!
//! This matches the locked-vocabulary "skip non-finite by default at
//! the input gate" rule — same gate that recipes calling `using(nan_policy:
//! "skip")` expose at the top of an atom invocation.
//!
//! # Min/Max behavior on no observations
//!
//! `min()` and `max()` return `NaN` when no finite values have been
//! observed. This is intentional and locked: NaN is the honest "no
//! observation" signal (per the NaN policy in
//! `2026-04-11-op-default-deterministic-plan.md`); `±∞` would be a
//! poison sentinel that downstream code might mistake for a real
//! observation.
//!
//! # Why a struct, not a trait
//!
//! Because the bookkeeping logic is identical across consumers (no
//! per-consumer customization needed), a concrete struct that
//! consumers compose by inclusion is the simplest expression. A trait
//! would force each consumer to either inherit (which Rust doesn't
//! have) or implement the bookkeeping methods (defeating the point of
//! the abstraction).
//!
//! # Future generalizations (anticipated, not built)
//!
//! When the same pattern appears for `sum`, `sum_sq`, `count_by_bucket`,
//! or weighted-observation tracking, extend this module rather than
//! duplicating in callers. The `FiniteObservations` API is the
//! minimum-viable shared shape; richer "what to track" variants live
//! alongside it without changing this one.

/// Shared bookkeeping for streaming accumulators that need
/// finite-observation count + min + max.
///
/// Use by composition: include a `FiniteObservations` field in your
/// accumulator struct, call `observe(x)` at the input gate, and
/// delegate `count()` / `min()` / `max()` / `merge()` to it.
#[derive(Debug, Clone)]
pub struct FiniteObservations {
    /// Total count of finite values observed.
    n: u64,
    /// Min finite value observed; NaN if no finite values yet.
    min: f64,
    /// Max finite value observed; NaN if no finite values yet.
    max: f64,
}

impl FiniteObservations {
    /// Create empty bookkeeping (n=0, min/max = NaN).
    #[inline]
    pub const fn new() -> Self {
        Self {
            n: 0,
            min: f64::NAN,
            max: f64::NAN,
        }
    }

    /// Try to observe `x`. Returns `true` if `x` was finite (and
    /// bookkeeping was updated), `false` otherwise. Callers gate
    /// their algorithm-specific work on the return value.
    #[inline]
    pub fn observe(&mut self, x: f64) -> bool {
        if !x.is_finite() {
            return false;
        }
        self.n += 1;
        if self.min.is_nan() || x < self.min {
            self.min = x;
        }
        if self.max.is_nan() || x > self.max {
            self.max = x;
        }
        true
    }

    /// Merge another `FiniteObservations` into self.
    #[inline]
    pub fn merge(&mut self, other: &Self) {
        self.n = self.n.saturating_add(other.n);
        if !other.min.is_nan() && (self.min.is_nan() || other.min < self.min) {
            self.min = other.min;
        }
        if !other.max.is_nan() && (self.max.is_nan() || other.max > self.max) {
            self.max = other.max;
        }
    }

    /// Count of finite observations.
    #[inline]
    pub fn count(&self) -> u64 {
        self.n
    }

    /// Min finite value observed; NaN if no finite values.
    #[inline]
    pub fn min(&self) -> f64 {
        self.min
    }

    /// Max finite value observed; NaN if no finite values.
    #[inline]
    pub fn max(&self) -> f64 {
        self.max
    }

    /// True if no finite values have been observed yet.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

impl Default for FiniteObservations {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_state() {
        let o = FiniteObservations::new();
        assert_eq!(o.count(), 0);
        assert!(o.min().is_nan());
        assert!(o.max().is_nan());
        assert!(o.is_empty());
    }

    #[test]
    fn finite_observation_updates() {
        let mut o = FiniteObservations::new();
        assert!(o.observe(5.0));
        assert_eq!(o.count(), 1);
        assert_eq!(o.min(), 5.0);
        assert_eq!(o.max(), 5.0);
        assert!(o.observe(3.0));
        assert_eq!(o.count(), 2);
        assert_eq!(o.min(), 3.0);
        assert_eq!(o.max(), 5.0);
        assert!(o.observe(10.0));
        assert_eq!(o.count(), 3);
        assert_eq!(o.min(), 3.0);
        assert_eq!(o.max(), 10.0);
    }

    #[test]
    fn non_finite_observations_skipped() {
        let mut o = FiniteObservations::new();
        assert!(!o.observe(f64::NAN));
        assert!(!o.observe(f64::INFINITY));
        assert!(!o.observe(f64::NEG_INFINITY));
        assert_eq!(o.count(), 0);
        assert!(o.min().is_nan());
        assert!(o.max().is_nan());
    }

    #[test]
    fn merge_two_populated() {
        let mut a = FiniteObservations::new();
        a.observe(1.0);
        a.observe(2.0);
        a.observe(3.0);
        let mut b = FiniteObservations::new();
        b.observe(10.0);
        b.observe(-5.0);
        a.merge(&b);
        assert_eq!(a.count(), 5);
        assert_eq!(a.min(), -5.0);
        assert_eq!(a.max(), 10.0);
    }

    #[test]
    fn merge_with_empty_is_noop() {
        let mut a = FiniteObservations::new();
        a.observe(7.0);
        let b = FiniteObservations::new();
        a.merge(&b);
        assert_eq!(a.count(), 1);
        assert_eq!(a.min(), 7.0);
        assert_eq!(a.max(), 7.0);
    }

    #[test]
    fn merge_into_empty_takes_other() {
        let mut a = FiniteObservations::new();
        let mut b = FiniteObservations::new();
        b.observe(7.0);
        b.observe(13.0);
        a.merge(&b);
        assert_eq!(a.count(), 2);
        assert_eq!(a.min(), 7.0);
        assert_eq!(a.max(), 13.0);
    }

    #[test]
    fn merge_associative() {
        let mut a = FiniteObservations::new();
        a.observe(1.0);
        a.observe(20.0);
        let mut b = FiniteObservations::new();
        b.observe(-3.0);
        b.observe(15.0);
        let mut c = FiniteObservations::new();
        c.observe(0.5);
        c.observe(100.0);

        // (a ⊕ b) ⊕ c
        let mut left = a.clone();
        left.merge(&b);
        left.merge(&c);

        // a ⊕ (b ⊕ c)
        let mut bc = b.clone();
        bc.merge(&c);
        let mut right = a.clone();
        right.merge(&bc);

        assert_eq!(left.count(), right.count());
        assert_eq!(left.min(), right.min());
        assert_eq!(left.max(), right.max());
    }

    #[test]
    fn handles_subnormal_and_extreme_values() {
        let mut o = FiniteObservations::new();
        let tiny = f64::from_bits(1); // smallest positive subnormal
        let huge = f64::MAX;
        o.observe(tiny);
        o.observe(-huge);
        o.observe(huge);
        assert_eq!(o.count(), 3);
        assert_eq!(o.min(), -huge);
        assert_eq!(o.max(), huge);
    }
}
