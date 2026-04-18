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
//! # The full family of observation trackers
//!
//! Tambear's "all of mathematics" scope makes the appearance of more
//! tracking variants near-certain, not speculative. The four shapes
//! below cover essentially every common streaming/mergeable
//! accumulator in numerical computing:
//!
//! - **`FiniteObservations`** — n / min / max only. Cheapest. Used
//!   by quantile sketches and any bookkeeping that just needs "how
//!   many finite values, and what's their range."
//! - **`WeightedObservations`** — n / min / max + Kulisch-exact weight
//!   sum. For any aggregator over `(value, weight)` pairs that needs
//!   the same skip gate plus weight bookkeeping.
//! - **`MomentObservations`** — n / min / max + Kulisch-exact running
//!   sums of x, x², x³, x⁴. Foundation for skewness, kurtosis, and
//!   any moment-derived statistic. Finalize-time computation: produce
//!   mean/variance/skew/kurt by reading the raw sums.
//! - **`WelfordObservations`** — n / min / max + Welford-style
//!   incremental mean and M₂. For streaming queries where
//!   mean/variance must be available at every step (not just at
//!   finalize). Numerically stable for online use.
//!
//! Pick the cheapest variant that covers the consumer's needs. If a
//! consumer needs "weighted moment tracking" specifically, either
//! compose `WeightedObservations` + Kulisch sum/sum_sq directly, or
//! add a `WeightedMomentObservations` sibling here once the pattern
//! repeats across consumers.
//!
//! All four expose the same baseline API (`count`, `min`, `max`,
//! `merge`, `is_empty`) so generic code that only needs the shared
//! bookkeeping can work across them via duck-typing or, if it grows
//! to need it, a future `Observable` trait.

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

// ═══════════════════════════════════════════════════════════════════════════
// WeightedObservations — n / min / max + Kulisch-exact weight sum
// ═══════════════════════════════════════════════════════════════════════════

/// Bookkeeping for weighted streaming accumulators: finite-observation
/// count + min + max (over `value`) plus a Kulisch-exact running sum
/// of `weight`.
///
/// Use by composition for any aggregator over `(value, weight)` pairs
/// that needs the standard skip gate plus weight-total bookkeeping.
/// Examples: weighted-mean recipes, weighted-quantile sketches,
/// weighted-Welford accumulators (when those land).
///
/// The skip rule: pair `(x, w)` is observed iff BOTH `x` and `w` are
/// finite. A non-finite weight skips the pair the same way a non-finite
/// value does.
#[derive(Debug, Clone)]
pub struct WeightedObservations {
    obs: FiniteObservations,
    weight_sum: crate::primitives::specialist::KulischAccumulator,
}

impl WeightedObservations {
    /// Empty bookkeeping.
    pub fn new() -> Self {
        Self {
            obs: FiniteObservations::new(),
            weight_sum: crate::primitives::specialist::KulischAccumulator::new(),
        }
    }

    /// Observe `(x, weight)`. Returns `true` if both inputs are finite
    /// and the pair was recorded; `false` otherwise (skipped).
    pub fn observe(&mut self, x: f64, weight: f64) -> bool {
        if !x.is_finite() || !weight.is_finite() {
            return false;
        }
        self.obs.observe(x);
        self.weight_sum.add_f64(weight);
        true
    }

    /// Merge another `WeightedObservations` into self.
    pub fn merge(&mut self, other: &Self) {
        self.obs.merge(&other.obs);
        self.weight_sum.merge(&other.weight_sum);
    }

    pub fn count(&self) -> u64 {
        self.obs.count()
    }
    pub fn min(&self) -> f64 {
        self.obs.min()
    }
    pub fn max(&self) -> f64 {
        self.obs.max()
    }
    pub fn is_empty(&self) -> bool {
        self.obs.is_empty()
    }
    /// Sum of weights observed so far (Kulisch-exact).
    pub fn weight_sum(&self) -> f64 {
        self.weight_sum.to_f64()
    }
}

impl Default for WeightedObservations {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// MomentObservations — n / min / max + Kulisch sums of x, x², x³, x⁴
// ═══════════════════════════════════════════════════════════════════════════

/// Bookkeeping for moment-tracking accumulators: finite-observation
/// count + min + max plus Kulisch-exact running sums of `x`, `x²`, `x³`,
/// `x⁴`.
///
/// All four sums are tracked unconditionally. The cost is four Kulisch
/// accumulators (~544 bytes each), which is small relative to the
/// kinds of computation that use moments. Consumers that only need
/// mean (sum + n) can use `MomentObservations` and ignore the higher
/// moments — the cost is uniform and the simplification of having one
/// shape covers more callers.
///
/// Finalize-time computation: read raw sums and divide by appropriate
/// factors. This module provides `mean`, `variance_sample`,
/// `variance_population`, `skewness`, `kurtosis_excess` as
/// conveniences. For numerical stability on near-constant data, prefer
/// `WelfordObservations` (which uses centered-sum formulas internally
/// at every update); `MomentObservations`'s post-process variance is
/// exact for the accumulated sums but can suffer cancellation between
/// `sum_sq` and `n·mean²` for highly concentrated distributions.
#[derive(Debug, Clone)]
pub struct MomentObservations {
    obs: FiniteObservations,
    sum: crate::primitives::specialist::KulischAccumulator,
    sum_sq: crate::primitives::specialist::KulischAccumulator,
    sum_cu: crate::primitives::specialist::KulischAccumulator,
    sum_q4: crate::primitives::specialist::KulischAccumulator,
}

impl MomentObservations {
    /// Empty bookkeeping.
    pub fn new() -> Self {
        Self {
            obs: FiniteObservations::new(),
            sum: crate::primitives::specialist::KulischAccumulator::new(),
            sum_sq: crate::primitives::specialist::KulischAccumulator::new(),
            sum_cu: crate::primitives::specialist::KulischAccumulator::new(),
            sum_q4: crate::primitives::specialist::KulischAccumulator::new(),
        }
    }

    /// Observe `x`. Returns `true` if finite and recorded; `false` if skipped.
    pub fn observe(&mut self, x: f64) -> bool {
        if !self.obs.observe(x) {
            return false;
        }
        let x2 = x * x;
        self.sum.add_f64(x);
        self.sum_sq.add_f64(x2);
        self.sum_cu.add_f64(x2 * x);
        self.sum_q4.add_f64(x2 * x2);
        true
    }

    /// Merge another `MomentObservations` into self.
    pub fn merge(&mut self, other: &Self) {
        self.obs.merge(&other.obs);
        self.sum.merge(&other.sum);
        self.sum_sq.merge(&other.sum_sq);
        self.sum_cu.merge(&other.sum_cu);
        self.sum_q4.merge(&other.sum_q4);
    }

    pub fn count(&self) -> u64 {
        self.obs.count()
    }
    pub fn min(&self) -> f64 {
        self.obs.min()
    }
    pub fn max(&self) -> f64 {
        self.obs.max()
    }
    pub fn is_empty(&self) -> bool {
        self.obs.is_empty()
    }

    /// Σ x.
    pub fn sum(&self) -> f64 {
        self.sum.to_f64()
    }
    /// Σ x².
    pub fn sum_sq(&self) -> f64 {
        self.sum_sq.to_f64()
    }
    /// Σ x³.
    pub fn sum_cu(&self) -> f64 {
        self.sum_cu.to_f64()
    }
    /// Σ x⁴.
    pub fn sum_q4(&self) -> f64 {
        self.sum_q4.to_f64()
    }

    /// Arithmetic mean. NaN if no observations.
    pub fn mean(&self) -> f64 {
        if self.obs.is_empty() {
            return f64::NAN;
        }
        self.sum.to_f64() / self.obs.count() as f64
    }

    /// Sample variance (denominator n−1). NaN if n < 2.
    ///
    /// Computed from raw sums via `Σx² − n·μ²` formula. For
    /// numerically delicate inputs (near-constant data) prefer
    /// `WelfordObservations` for stability.
    pub fn variance_sample(&self) -> f64 {
        let n = self.obs.count();
        if n < 2 {
            return f64::NAN;
        }
        let mu = self.mean();
        let n_f = n as f64;
        (self.sum_sq.to_f64() - n_f * mu * mu) / (n_f - 1.0)
    }

    /// Population variance (denominator n).
    pub fn variance_population(&self) -> f64 {
        let n = self.obs.count();
        if n < 1 {
            return f64::NAN;
        }
        let mu = self.mean();
        let n_f = n as f64;
        (self.sum_sq.to_f64() - n_f * mu * mu) / n_f
    }

    /// Standardized skewness (third standardized central moment).
    /// Returns NaN for n < 2 or zero variance.
    pub fn skewness(&self) -> f64 {
        let n = self.obs.count();
        if n < 2 {
            return f64::NAN;
        }
        let mu = self.mean();
        let var_pop = self.variance_population();
        if var_pop <= 0.0 {
            return f64::NAN;
        }
        let n_f = n as f64;
        // Σ (x − μ)³  =  Σx³ − 3μ·Σx² + 3μ²·Σx − n·μ³
        //              =  Σx³ − 3μ·Σx² + 2n·μ³
        let m3 = self.sum_cu.to_f64() - 3.0 * mu * self.sum_sq.to_f64() + 2.0 * n_f * mu * mu * mu;
        let m3_normalized = m3 / n_f;
        m3_normalized / var_pop.powf(1.5)
    }

    /// Excess kurtosis (fourth standardized central moment minus 3).
    /// Returns NaN for n < 2 or zero variance.
    pub fn kurtosis_excess(&self) -> f64 {
        let n = self.obs.count();
        if n < 2 {
            return f64::NAN;
        }
        let mu = self.mean();
        let var_pop = self.variance_population();
        if var_pop <= 0.0 {
            return f64::NAN;
        }
        let n_f = n as f64;
        // Σ (x − μ)⁴  =  Σx⁴ − 4μ·Σx³ + 6μ²·Σx² − 4μ³·Σx + n·μ⁴
        //              =  Σx⁴ − 4μ·Σx³ + 6μ²·Σx² − 3n·μ⁴
        let m4 = self.sum_q4.to_f64() - 4.0 * mu * self.sum_cu.to_f64()
            + 6.0 * mu * mu * self.sum_sq.to_f64()
            - 3.0 * n_f * mu * mu * mu * mu;
        let m4_normalized = m4 / n_f;
        m4_normalized / (var_pop * var_pop) - 3.0
    }
}

impl Default for MomentObservations {
    fn default() -> Self {
        Self::new()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// WelfordObservations — n / min / max + Welford online mean/M2
// ═══════════════════════════════════════════════════════════════════════════

/// Bookkeeping for streaming online statistics via Welford's algorithm.
///
/// Tracks `n`, `min`, `max`, plus an incrementally-updated `mean` and
/// `M₂` (sum of squared deviations from the running mean). Mean and
/// variance are correct at every step, not just at finalize. This is
/// the right choice when:
///
/// - The consumer queries `mean` / `variance` repeatedly during
///   ingestion, not just once at the end
/// - Numerical stability matters for near-constant data (Welford
///   centers each update around the running mean, avoiding the
///   `Σx² − n·μ²` cancellation that `MomentObservations` is subject
///   to for that case)
/// - Memory is tight (one running mean + one M₂ vs four Kulisch
///   accumulators)
///
/// Trade-off vs `MomentObservations`:
/// - Welford does NOT track higher moments (no skewness or kurtosis
///   from this struct alone). For those, use `MomentObservations`.
/// - Welford's incremental updates use plain f64 arithmetic, not
///   Kulisch — the updates are cancellation-resistant by design but
///   not exact in the sense Kulisch is. For exact + post-process,
///   prefer `MomentObservations`.
///
/// # Reference
///
/// Welford, B. P. (1962). Note on a method for calculating corrected
/// sums of squares and products. *Technometrics* 4(3): 419–420.
///
/// # Merge: Chan-Welford parallel merge
///
/// The two-shard merge formula (Chan-Welford) combines `(n, mean, M₂)`
/// from each shard correctly without re-streaming. Both shards must
/// share the same finite-observation skip rule (which they do, since
/// they both use this struct).
#[derive(Debug, Clone)]
pub struct WelfordObservations {
    obs: FiniteObservations,
    /// Incrementally-updated mean of finite observations.
    mean: f64,
    /// Sum of squared deviations from the running mean (n·population_var).
    m2: f64,
}

impl WelfordObservations {
    /// Empty bookkeeping.
    pub fn new() -> Self {
        Self {
            obs: FiniteObservations::new(),
            mean: 0.0,
            m2: 0.0,
        }
    }

    /// Observe `x`. Returns `true` if finite and recorded; `false` if skipped.
    pub fn observe(&mut self, x: f64) -> bool {
        if !self.obs.observe(x) {
            return false;
        }
        // Welford update.
        let n = self.obs.count() as f64;
        let delta = x - self.mean;
        self.mean += delta / n;
        let delta2 = x - self.mean;
        self.m2 += delta * delta2;
        true
    }

    /// Merge another `WelfordObservations` into self via Chan-Welford.
    pub fn merge(&mut self, other: &Self) {
        if other.obs.is_empty() {
            return;
        }
        if self.obs.is_empty() {
            *self = other.clone();
            return;
        }
        let na = self.obs.count() as f64;
        let nb = other.obs.count() as f64;
        let n_total = na + nb;
        let delta = other.mean - self.mean;
        // Chan-Welford parallel merge.
        let new_mean = self.mean + delta * (nb / n_total);
        let new_m2 = self.m2 + other.m2 + delta * delta * (na * nb / n_total);
        self.mean = new_mean;
        self.m2 = new_m2;
        self.obs.merge(&other.obs);
    }

    pub fn count(&self) -> u64 {
        self.obs.count()
    }
    pub fn min(&self) -> f64 {
        self.obs.min()
    }
    pub fn max(&self) -> f64 {
        self.obs.max()
    }
    pub fn is_empty(&self) -> bool {
        self.obs.is_empty()
    }

    /// Running mean. NaN if no observations.
    pub fn mean(&self) -> f64 {
        if self.obs.is_empty() {
            return f64::NAN;
        }
        self.mean
    }

    /// Sample variance (denominator n−1). NaN for n < 2.
    pub fn variance_sample(&self) -> f64 {
        let n = self.obs.count();
        if n < 2 {
            return f64::NAN;
        }
        self.m2 / (n as f64 - 1.0)
    }

    /// Population variance (denominator n). NaN for n < 1.
    pub fn variance_population(&self) -> f64 {
        let n = self.obs.count();
        if n < 1 {
            return f64::NAN;
        }
        self.m2 / n as f64
    }

    /// Sample standard deviation.
    pub fn std_dev_sample(&self) -> f64 {
        self.variance_sample().sqrt()
    }

    /// Population standard deviation.
    pub fn std_dev_population(&self) -> f64 {
        self.variance_population().sqrt()
    }
}

impl Default for WelfordObservations {
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

    // ── WeightedObservations ───────────────────────────────────────────────

    #[test]
    fn weighted_basic() {
        let mut w = WeightedObservations::new();
        assert!(w.is_empty());
        assert!(w.observe(1.0, 0.5));
        assert!(w.observe(2.0, 1.5));
        assert_eq!(w.count(), 2);
        assert_eq!(w.weight_sum(), 2.0);
        assert_eq!(w.min(), 1.0);
        assert_eq!(w.max(), 2.0);
    }

    #[test]
    fn weighted_skips_non_finite_either_side() {
        let mut w = WeightedObservations::new();
        assert!(!w.observe(f64::NAN, 1.0));
        assert!(!w.observe(1.0, f64::NAN));
        assert!(!w.observe(f64::INFINITY, 1.0));
        assert!(!w.observe(1.0, f64::NEG_INFINITY));
        assert_eq!(w.count(), 0);
        assert_eq!(w.weight_sum(), 0.0);
    }

    #[test]
    fn weighted_merge() {
        let mut a = WeightedObservations::new();
        a.observe(1.0, 1.0);
        a.observe(3.0, 2.0);
        let mut b = WeightedObservations::new();
        b.observe(5.0, 0.5);
        b.observe(-1.0, 0.5);
        a.merge(&b);
        assert_eq!(a.count(), 4);
        assert_eq!(a.weight_sum(), 4.0);
        assert_eq!(a.min(), -1.0);
        assert_eq!(a.max(), 5.0);
    }

    // ── MomentObservations ──────────────────────────────────────────────────

    #[test]
    fn moment_basic_accumulation() {
        let mut m = MomentObservations::new();
        for i in 1..=10 {
            m.observe(i as f64);
        }
        assert_eq!(m.count(), 10);
        assert_eq!(m.sum(), 55.0); // 1+2+...+10
        assert_eq!(m.sum_sq(), 385.0); // 1+4+9+...+100
        assert_eq!(m.mean(), 5.5);
        // Sample variance of 1..10 is 9.166666...
        assert!((m.variance_sample() - 9.166666666666666).abs() < 1e-12);
    }

    #[test]
    fn moment_skips_non_finite() {
        let mut m = MomentObservations::new();
        m.observe(1.0);
        m.observe(f64::NAN);
        m.observe(2.0);
        m.observe(f64::INFINITY);
        m.observe(3.0);
        assert_eq!(m.count(), 3);
        assert_eq!(m.sum(), 6.0);
    }

    #[test]
    fn moment_skewness_zero_for_symmetric() {
        let mut m = MomentObservations::new();
        // Symmetric around 0.
        for v in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
            m.observe(v);
        }
        let s = m.skewness();
        assert!(s.abs() < 1e-12, "skewness of symmetric data should be ~0, got {s}");
    }

    #[test]
    fn moment_kurtosis_negative_for_uniform() {
        // Uniform discrete distribution has excess kurtosis ≈ -1.2.
        let mut m = MomentObservations::new();
        for i in 0..100 {
            m.observe(i as f64);
        }
        let k = m.kurtosis_excess();
        // Allow generous tolerance — excess kurtosis of discrete uniform is -1.2 in the limit.
        assert!(
            (k - -1.2).abs() < 0.05,
            "excess kurtosis of uniform should be ~-1.2, got {k}"
        );
    }

    #[test]
    fn moment_merge_preserves_aggregates() {
        let xs: Vec<f64> = (0..1000).map(|i| ((i as f64).sin() * 10.0)).collect();

        let mut full = MomentObservations::new();
        for &x in &xs {
            full.observe(x);
        }

        let (a, b) = xs.split_at(500);
        let mut sa = MomentObservations::new();
        for &x in a {
            sa.observe(x);
        }
        let mut sb = MomentObservations::new();
        for &x in b {
            sb.observe(x);
        }
        sa.merge(&sb);

        assert_eq!(full.count(), sa.count());
        assert!((full.mean() - sa.mean()).abs() < 1e-12);
        // Variance computed from the same exact sums must agree.
        assert!((full.variance_sample() - sa.variance_sample()).abs() < 1e-9);
    }

    // ── WelfordObservations ─────────────────────────────────────────────────

    #[test]
    fn welford_basic_streaming() {
        let mut w = WelfordObservations::new();
        for i in 1..=10 {
            w.observe(i as f64);
        }
        assert_eq!(w.count(), 10);
        assert!((w.mean() - 5.5).abs() < 1e-12);
        assert!((w.variance_sample() - 9.166666666666666).abs() < 1e-12);
    }

    #[test]
    fn welford_mean_updates_at_every_step() {
        // Welford makes mean available at every step, not just finalize.
        let mut w = WelfordObservations::new();
        w.observe(10.0);
        assert_eq!(w.mean(), 10.0);
        w.observe(20.0);
        assert_eq!(w.mean(), 15.0);
        w.observe(30.0);
        assert_eq!(w.mean(), 20.0);
    }

    #[test]
    fn welford_chan_merge() {
        // Build via streaming and via merge; results should agree.
        let xs: Vec<f64> = (0..1000).map(|i| ((i as f64).cos() * 100.0 + 50.0)).collect();

        let mut full = WelfordObservations::new();
        for &x in &xs {
            full.observe(x);
        }

        let (a, b) = xs.split_at(500);
        let mut sa = WelfordObservations::new();
        for &x in a {
            sa.observe(x);
        }
        let mut sb = WelfordObservations::new();
        for &x in b {
            sb.observe(x);
        }
        sa.merge(&sb);

        assert_eq!(full.count(), sa.count());
        // Chan-Welford merge is correct to many ULPs but not exact at f64.
        assert!(
            (full.mean() - sa.mean()).abs() < 1e-9,
            "mean: full {} vs merged {}",
            full.mean(),
            sa.mean()
        );
        assert!(
            (full.variance_sample() - sa.variance_sample()).abs() < 1e-6,
            "variance: full {} vs merged {}",
            full.variance_sample(),
            sa.variance_sample()
        );
    }

    #[test]
    fn welford_skips_non_finite() {
        let mut w = WelfordObservations::new();
        w.observe(1.0);
        w.observe(f64::NAN);
        w.observe(2.0);
        w.observe(f64::INFINITY);
        w.observe(3.0);
        assert_eq!(w.count(), 3);
        assert!((w.mean() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn welford_merge_with_empty() {
        let mut a = WelfordObservations::new();
        a.observe(5.0);
        a.observe(10.0);
        let b = WelfordObservations::new();
        a.merge(&b);
        assert_eq!(a.count(), 2);
        assert_eq!(a.mean(), 7.5);
    }

    #[test]
    fn welford_merge_into_empty() {
        let mut a = WelfordObservations::new();
        let mut b = WelfordObservations::new();
        b.observe(5.0);
        b.observe(15.0);
        a.merge(&b);
        assert_eq!(a.count(), 2);
        assert_eq!(a.mean(), 10.0);
    }
}
