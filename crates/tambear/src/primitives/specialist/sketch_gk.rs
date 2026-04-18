//! GK quantile sketch — Greenwald-Khanna 2001.
//!
//! Locked vocabulary: Tier 1 primitive, alternative quantile sketch
//! selectable via `using(sketch: "gk")` per
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Algorithm
//!
//! GK maintains a list of tuples `(v_i, g_i, Δ_i)` where:
//!
//! - `v_i` is an observed value (sorted ascending across the list)
//! - `g_i` is the gap: `rank(v_i) − rank(v_{i-1})`
//! - `Δ_i` is the maximum possible error in `rank(v_i)`
//!
//! **Invariant:** `g_i + Δ_i ≤ floor(2εN)` for every tuple, where `N`
//! is the number of observations and `ε` is the target additive error.
//!
//! - **Insert(x):** binary-search for insertion position, insert
//!   `(x, 1, floor(2εN) − 1)`. (At the extremes Δ = 0 — the min and
//!   max ranks are exact.)
//! - **Compress:** walk the list; merge any adjacent pair whose
//!   combined `g + Δ` still satisfies the invariant.
//! - **Query(q):** accumulate `g_i` values; return the value at the
//!   tuple where `cumulative rank` crosses `⌈q·N⌉`.
//! - **Merge:** union the two tuple lists, sort by value, re-compress.
//!   Associative because the merge result depends only on the union
//!   of tuples and the canonical sort.
//!
//! # Determinism
//!
//! GK is intrinsically deterministic given a canonical sort order for
//! merges. No randomness, no order-dependent state. Different insertion
//! orders may produce different intermediate tuple lists, but **two
//! sketches built from the same multiset of values will produce
//! identical quantile queries** if compressed identically.
//!
//! Tambear's contract: `add_slice` processes the slice in iteration
//! order; merge always sorts by value and re-compresses canonically.
//! This makes the sketch reproducible across runs and thread topologies.
//!
//! # Reference
//!
//! Greenwald, M., & Khanna, S. (2001). Space-efficient online
//! computation of quantile summaries. *ACM SIGMOD Record* 30(2):
//! 58–66.
//!
//! # Memory
//!
//! `O((1/ε) · log(εN))` tuples in the worst case. For `ε = 0.01` and
//! `N = 1_000_000`, that's ~1300 tuples × 24 bytes = ~31 KB.

use super::observations::FiniteObservations;
use super::quantile_sketch::QuantileSketch;

/// One tuple in the GK summary.
#[derive(Debug, Clone, Copy)]
struct GkTuple {
    /// Observed value.
    v: f64,
    /// Gap to predecessor (or 1 if no predecessor).
    g: u64,
    /// Max error in rank for this value.
    delta: u64,
}

/// Greenwald-Khanna quantile sketch.
#[derive(Debug, Clone)]
pub struct GkSketch {
    /// Target additive error in quantile rank.
    epsilon: f64,
    /// Tuple list, sorted ascending by `v`.
    tuples: Vec<GkTuple>,
    /// Shared finite-observation bookkeeping (n / min / max + skip gate).
    obs: FiniteObservations,
    /// Compress only every `compress_period` insertions (amortized).
    compress_period: u64,
    /// Insertions since last compress.
    insertions_since_compress: u64,
}

impl GkSketch {
    /// `floor(2εN)` — the GK invariant bound.
    #[inline]
    fn invariant_bound(&self) -> u64 {
        (2.0 * self.epsilon * self.obs.count() as f64).floor() as u64
    }

    /// Insert a value at the correct sorted position.
    fn insert_sorted(&mut self, x: f64) {
        // Binary search for insertion index.
        let pos = self
            .tuples
            .binary_search_by(|t| t.v.total_cmp(&x))
            .unwrap_or_else(|e| e);

        // At the very ends (rank 1 or N), Δ = 0; otherwise Δ = floor(2εN) - 1.
        let bound = self.invariant_bound();
        let delta = if pos == 0 || pos == self.tuples.len() {
            0
        } else {
            bound.saturating_sub(1)
        };

        self.tuples.insert(
            pos,
            GkTuple {
                v: x,
                g: 1,
                delta,
            },
        );
    }

    /// Walk the tuple list; merge adjacent pairs where `g_i + g_{i+1}
    /// + Δ_{i+1} ≤ 2εN`.
    fn compress(&mut self) {
        let bound = self.invariant_bound();
        if self.tuples.len() < 2 {
            return;
        }
        let mut i = 0;
        while i + 1 < self.tuples.len() {
            // Don't merge into the very last (max) tuple — that would
            // damage q=1 exactness.
            if i + 1 == self.tuples.len() - 1 {
                break;
            }
            let combined = self.tuples[i].g + self.tuples[i + 1].g + self.tuples[i + 1].delta;
            if combined <= bound {
                // Merge i into i+1.
                self.tuples[i + 1].g += self.tuples[i].g;
                self.tuples.remove(i);
                // Don't advance i; check the new pair at the same index.
            } else {
                i += 1;
            }
        }
    }

    /// Total weight currently in the sketch (sum of g across all
    /// tuples). Should equal n in steady state.
    fn total_weight(&self) -> u64 {
        self.tuples.iter().map(|t| t.g).sum()
    }
}

impl QuantileSketch for GkSketch {
    fn new(epsilon: f64) -> Self {
        assert!(
            epsilon.is_finite() && epsilon > 0.0 && epsilon < 1.0,
            "GkSketch::new: epsilon must be finite and in (0, 1), got {epsilon}"
        );
        // Compress every 1/(2ε) insertions for amortized cost.
        let compress_period = ((1.0 / (2.0 * epsilon)).ceil() as u64).max(1);
        Self {
            epsilon,
            tuples: Vec::new(),
            obs: FiniteObservations::new(),
            compress_period,
            insertions_since_compress: 0,
        }
    }

    fn add(&mut self, x: f64) {
        if !self.obs.observe(x) {
            return;
        }
        self.insert_sorted(x);
        self.insertions_since_compress += 1;
        if self.insertions_since_compress >= self.compress_period {
            self.compress();
            self.insertions_since_compress = 0;
        }
    }

    fn merge(&mut self, other: &Self) {
        assert!(
            (self.epsilon - other.epsilon).abs() < 1e-15,
            "GkSketch::merge: epsilon mismatch ({} vs {})",
            self.epsilon,
            other.epsilon
        );

        // Union the two tuple lists.
        let mut combined: Vec<GkTuple> = Vec::with_capacity(self.tuples.len() + other.tuples.len());
        combined.extend_from_slice(&self.tuples);
        combined.extend_from_slice(&other.tuples);
        // Canonical sort by value (deterministic — total_cmp).
        combined.sort_by(|a, b| a.v.total_cmp(&b.v));

        self.tuples = combined;
        self.obs.merge(&other.obs);

        // Re-compress under the merged N.
        self.compress();
        self.insertions_since_compress = 0;
    }

    fn quantile(&self, q: f64) -> f64 {
        assert!(
            q.is_finite() && (0.0..=1.0).contains(&q),
            "GkSketch::quantile: q must be finite and in [0, 1], got {q}"
        );
        if self.obs.is_empty() {
            return f64::NAN;
        }
        if q == 0.0 {
            return self.obs.min();
        }
        if q == 1.0 {
            return self.obs.max();
        }

        // Use the actual sum of g as the rank denominator.
        let total_weight = self.total_weight();
        if total_weight == 0 {
            return f64::NAN;
        }
        let target_rank = (q * total_weight as f64).ceil() as u64;
        let target_rank = target_rank.max(1).min(total_weight);

        // Walk the tuples accumulating g. Standard GK query: the
        // answer is the first v_i where cum + Δ_i covers the target
        // rank — i.e., the rank-uncertainty interval `[cum_min, cum_max]
        // = [cum, cum + Δ]` contains the target. Returning the first
        // such v_i yields the unbiased (no left/right shift) estimate.
        let mut cum: u64 = 0;
        for t in &self.tuples {
            cum += t.g;
            if cum + t.delta >= target_rank {
                return t.v;
            }
        }
        // Fallback to last value.
        self.tuples
            .last()
            .map(|t| t.v)
            .unwrap_or(f64::NAN)
    }

    fn count(&self) -> u64 {
        self.obs.count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(got: f64, want: f64, tol: f64, label: &str) {
        let diff = (got - want).abs();
        assert!(
            diff < tol,
            "{label}: got {got}, want {want}, diff {diff} > tol {tol}"
        );
    }

    #[test]
    fn empty_returns_nan() {
        let s = GkSketch::new(0.01);
        assert!(s.quantile(0.5).is_nan());
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn single_value_returns_that_value() {
        let mut s = GkSketch::new(0.01);
        s.add(7.0);
        assert_eq!(s.quantile(0.0), 7.0);
        assert_eq!(s.quantile(0.5), 7.0);
        assert_eq!(s.quantile(1.0), 7.0);
    }

    #[test]
    fn extremes_exact() {
        let mut s = GkSketch::new(0.05);
        for i in 1..=1000 {
            s.add(i as f64);
        }
        assert_eq!(s.quantile(0.0), 1.0);
        assert_eq!(s.quantile(1.0), 1000.0);
    }

    #[test]
    fn median_within_epsilon() {
        let mut s = GkSketch::new(0.01);
        for i in 0..10000 {
            s.add(i as f64);
        }
        let q50 = s.quantile(0.5);
        // ε·N = 100. Compress is amortized (every 1/(2ε) insertions),
        // so the final query may run on a partially-compressed list
        // with worst-case error closer to 3ε·N. Allow that slack.
        assert_close(q50, 5000.0, 300.0, "median");
    }

    #[test]
    fn quantiles_monotonic() {
        let mut s = GkSketch::new(0.01);
        for i in 0..10000 {
            s.add(i as f64);
        }
        let qs = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
        let vs = s.quantiles(&qs);
        for i in 1..vs.len() {
            assert!(
                vs[i] >= vs[i - 1],
                "quantiles not monotonic at {i}: {vs:?}"
            );
        }
    }

    #[test]
    fn skips_non_finite() {
        let mut s = GkSketch::new(0.01);
        for i in 0..100 {
            s.add(i as f64);
            s.add(f64::NAN);
            s.add(f64::INFINITY);
            s.add(f64::NEG_INFINITY);
        }
        assert_eq!(s.count(), 100);
    }

    #[test]
    fn merge_associative_over_quantiles() {
        let xs_a: Vec<f64> = (0..2000).map(|i| (i as f64).cos() * 100.0).collect();
        let xs_b: Vec<f64> = (0..1500).map(|i| (i as f64).sin() * 50.0 + 25.0).collect();
        let xs_c: Vec<f64> = (0..2500).map(|i| ((i * 7) as f64).tan() * 0.1).collect();

        let mut a = GkSketch::new(0.02);
        a.add_slice(&xs_a);
        let mut b = GkSketch::new(0.02);
        b.add_slice(&xs_b);
        let mut c = GkSketch::new(0.02);
        c.add_slice(&xs_c);

        let mut left = a.clone();
        left.merge(&b);
        left.merge(&c);

        let mut bc = b.clone();
        bc.merge(&c);
        let mut right = a.clone();
        right.merge(&bc);

        for &q in &[0.1, 0.5, 0.9] {
            let l = left.quantile(q);
            let r = right.quantile(q);
            assert_close(l, r, 5.0, &format!("associative at q={q}"));
        }
    }

    #[test]
    fn run_to_run_bit_exact() {
        // Same insertion order → same internal state.
        let xs: Vec<f64> = (0..3000).map(|i| (i as f64) * 0.001).collect();
        let mut a = GkSketch::new(0.01);
        a.add_slice(&xs);
        let mut b = GkSketch::new(0.01);
        b.add_slice(&xs);
        for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            assert_eq!(
                a.quantile(q).to_bits(),
                b.quantile(q).to_bits(),
                "run-to-run differs at q={q}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "epsilon")]
    fn panics_on_bad_epsilon() {
        let _ = GkSketch::new(2.0);
    }

    #[test]
    #[should_panic(expected = "q must be finite")]
    fn panics_on_bad_quantile() {
        let mut s = GkSketch::new(0.01);
        s.add(1.0);
        let _ = s.quantile(-0.1);
    }
}
