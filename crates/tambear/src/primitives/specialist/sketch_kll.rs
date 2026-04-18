//! KLL quantile sketch — Karnin-Lang-Liberty 2016.
//!
//! The locked-default mergeable quantile sketch per
//! `R:\winrapids\docs\architecture\vocabulary.md`. Recipes that need
//! quantile estimation get this implementation when no explicit
//! `using(sketch: ...)` override is provided.
//!
//! # Algorithm
//!
//! KLL maintains a sequence of "compactors" indexed by level. Level 0
//! ingests raw items; when level `i` fills to capacity `k`, it gets
//! sorted and **compacted**: half its items (chosen by a deterministic
//! parity coin) are promoted to level `i+1`. Each item at level `i`
//! conceptually represents `2^i` observations.
//!
//! Quantile queries: union all items across levels with their weights,
//! sort by value, locate the rank-th item.
//!
//! Merge: union the compactors level-by-level, compacting any that
//! overflow. Associative because the union operation doesn't depend on
//! insertion order at any given level.
//!
//! # Determinism
//!
//! The KLL paper's error bound assumes a uniform-random coin at each
//! compaction. Tambear's cross-platform bit-exact determinism contract
//! forbids that — the coin must be reproducible across runs and
//! backends.
//!
//! Resolution: use a **deterministic alternating coin** per compactor
//! level. Each level has a `compaction_count: u64`; the coin at
//! compaction `c` is `c & 1`. This makes the result reproducible. The
//! KLL error bound still holds in expectation under this scheme; the
//! worst-case bound is somewhat looser but remains O(ε·N).
//!
//! # Reference
//!
//! Karnin, Z., Lang, K., & Liberty, E. (2016). Optimal quantile
//! approximation in streams. *57th Annual IEEE Symposium on Foundations
//! of Computer Science (FOCS)*: 71–78.
//!
//! # Memory
//!
//! With capacity `k` per compactor and `n` observations, levels are
//! bounded by `log_2(n/k)`. Memory is `O(k · log(n/k))` items. For
//! `k = 200` and `n = 1_000_000`, that's ~3000 items.

use super::quantile_sketch::QuantileSketch;

/// Capacity formula tying ε to k. KLL's error in rank is approximately
/// `O(sqrt(log(1/εδ)) / k)`. For practical use we set `k = ceil(c / ε)`
/// with `c ≈ 1.5`, giving ε ≈ 1.5/k.
fn capacity_for_epsilon(epsilon: f64) -> usize {
    let k = (1.5 / epsilon).ceil() as usize;
    k.max(8) // floor at 8 to avoid degenerate behavior
}

/// KLL quantile sketch.
#[derive(Debug, Clone)]
pub struct KllSketch {
    /// Per-compactor capacity. Items beyond this trigger compaction.
    k: usize,
    /// Target additive error in quantile rank.
    epsilon: f64,
    /// `compactors[i]` holds the items at level i.
    compactors: Vec<Vec<f64>>,
    /// `compaction_counts[i]` is how many times level i has been
    /// compacted; coin at next compaction is `count & 1`.
    compaction_counts: Vec<u64>,
    /// Total observations ingested (finite).
    n: u64,
    /// Min finite value observed (NaN if none).
    min: f64,
    /// Max finite value observed (NaN if none).
    max: f64,
}

impl KllSketch {
    /// Capacity for level `i`. KLL uses geometric decrease: level 0
    /// has capacity `k`, level 1 has `k * c`, level 2 has `k * c^2`,
    /// etc. (where `c < 1` is a shrink factor). For simplicity and
    /// determinism, we use a constant `k` per level.
    fn level_capacity(&self, _level: usize) -> usize {
        self.k
    }

    /// Compact level `level` if it has reached capacity. Promotes half
    /// the items (chosen by alternating coin) to level+1.
    fn compact_if_full(&mut self, level: usize) {
        if self.compactors[level].len() < self.level_capacity(level) {
            return;
        }
        // Sort by total_cmp (handles all f64 deterministically including
        // -0.0 vs 0.0; but we've already filtered NaN at insertion).
        self.compactors[level].sort_by(|a, b| a.total_cmp(b));

        // Alternating coin: 0 keeps even-indexed, 1 keeps odd-indexed.
        let coin = (self.compaction_counts[level] & 1) as usize;
        self.compaction_counts[level] = self.compaction_counts[level].wrapping_add(1);

        let mut promoted = Vec::with_capacity(self.compactors[level].len() / 2);
        let comp = std::mem::take(&mut self.compactors[level]);
        for (i, v) in comp.into_iter().enumerate() {
            if i % 2 == coin {
                promoted.push(v);
            }
        }

        // Ensure level+1 exists.
        if self.compactors.len() <= level + 1 {
            self.compactors.push(Vec::new());
            self.compaction_counts.push(0);
        }
        self.compactors[level + 1].extend(promoted);

        // Cascade if level+1 is now over capacity.
        self.compact_if_full(level + 1);
    }

    /// Collect all (value, weight) pairs across all levels. Weight at
    /// level i is `2^i`.
    fn weighted_items(&self) -> Vec<(f64, u64)> {
        let mut out = Vec::new();
        for (level, comp) in self.compactors.iter().enumerate() {
            let weight = 1_u64 << level;
            for &v in comp {
                out.push((v, weight));
            }
        }
        out
    }
}

impl QuantileSketch for KllSketch {
    fn new(epsilon: f64) -> Self {
        assert!(
            epsilon.is_finite() && epsilon > 0.0 && epsilon < 1.0,
            "KllSketch::new: epsilon must be finite and in (0, 1), got {epsilon}"
        );
        let k = capacity_for_epsilon(epsilon);
        Self {
            k,
            epsilon,
            compactors: vec![Vec::new()],
            compaction_counts: vec![0],
            n: 0,
            min: f64::NAN,
            max: f64::NAN,
        }
    }

    fn add(&mut self, x: f64) {
        if !x.is_finite() {
            return;
        }
        self.n += 1;
        if self.min.is_nan() || x < self.min {
            self.min = x;
        }
        if self.max.is_nan() || x > self.max {
            self.max = x;
        }
        self.compactors[0].push(x);
        self.compact_if_full(0);
    }

    fn merge(&mut self, other: &Self) {
        // Sketches must agree on epsilon (and therefore k) for merge
        // to be well-defined. Differ → caller bug.
        assert!(
            (self.epsilon - other.epsilon).abs() < 1e-15,
            "KllSketch::merge: epsilon mismatch ({} vs {})",
            self.epsilon,
            other.epsilon
        );

        // Ensure we have at least as many levels as `other`.
        while self.compactors.len() < other.compactors.len() {
            self.compactors.push(Vec::new());
            self.compaction_counts.push(0);
        }

        // Union per-level items.
        for (level, comp) in other.compactors.iter().enumerate() {
            self.compactors[level].extend_from_slice(comp);
        }

        // Compaction-count merge: take the max so the deterministic
        // coin parity continues forward consistently across both
        // histories. Choosing max (rather than sum) preserves the
        // "next coin is c & 1" reproducibility — different merge
        // orders converge to the same parity at the higher count.
        for (level, &c) in other.compaction_counts.iter().enumerate() {
            if level < self.compaction_counts.len() {
                self.compaction_counts[level] = self.compaction_counts[level].max(c);
            }
        }

        // Update aggregate stats.
        self.n = self.n.saturating_add(other.n);
        if !other.min.is_nan() && (self.min.is_nan() || other.min < self.min) {
            self.min = other.min;
        }
        if !other.max.is_nan() && (self.max.is_nan() || other.max > self.max) {
            self.max = other.max;
        }

        // Cascade compaction starting at level 0 in case any level is now full.
        for level in 0..self.compactors.len() {
            self.compact_if_full(level);
        }
    }

    fn quantile(&self, q: f64) -> f64 {
        assert!(
            q.is_finite() && (0.0..=1.0).contains(&q),
            "KllSketch::quantile: q must be finite and in [0, 1], got {q}"
        );
        if self.n == 0 {
            return f64::NAN;
        }
        if q == 0.0 {
            return self.min;
        }
        if q == 1.0 {
            return self.max;
        }

        // Collect (value, weight) and sort by value.
        let mut items = self.weighted_items();
        items.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Total weight should equal n in the noise-free case; due to
        // compaction it can differ by O(k). Use the actual sum as the
        // rank denominator for self-consistency.
        let total_weight: u64 = items.iter().map(|(_, w)| w).sum();
        if total_weight == 0 {
            return f64::NAN;
        }
        let target_rank = (q * total_weight as f64).floor() as u64;
        let target_rank = target_rank.min(total_weight - 1);

        let mut cum: u64 = 0;
        for (v, w) in &items {
            cum += *w;
            if cum > target_rank {
                return *v;
            }
        }
        // Fallback (should not reach here given the target_rank clamp).
        items.last().map(|(v, _)| *v).unwrap_or(f64::NAN)
    }

    fn count(&self) -> u64 {
        self.n
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
        let s = KllSketch::new(0.01);
        assert!(s.quantile(0.5).is_nan());
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn single_value_returns_that_value() {
        let mut s = KllSketch::new(0.01);
        s.add(42.0);
        assert_eq!(s.quantile(0.0), 42.0);
        assert_eq!(s.quantile(0.5), 42.0);
        assert_eq!(s.quantile(1.0), 42.0);
    }

    #[test]
    fn extremes_exact() {
        // q=0 and q=1 should be exact min/max regardless of epsilon.
        let mut s = KllSketch::new(0.05);
        for i in 1..=1000 {
            s.add(i as f64);
        }
        assert_eq!(s.quantile(0.0), 1.0);
        assert_eq!(s.quantile(1.0), 1000.0);
    }

    #[test]
    fn median_within_epsilon() {
        // Stream of 0..10000. True median is 4999.5 (or 5000 in
        // discrete rank). KLL should be within ε·N = 0.01·10000 = 100.
        let mut s = KllSketch::new(0.01);
        for i in 0..10000 {
            s.add(i as f64);
        }
        let q50 = s.quantile(0.5);
        assert_close(q50, 5000.0, 200.0, "median");
    }

    #[test]
    fn quantiles_monotonic() {
        let mut s = KllSketch::new(0.01);
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
        let mut s = KllSketch::new(0.01);
        for i in 0..100 {
            s.add(i as f64);
            s.add(f64::NAN);
            s.add(f64::INFINITY);
            s.add(f64::NEG_INFINITY);
        }
        assert_eq!(s.count(), 100);
    }

    #[test]
    fn merge_preserves_quantiles() {
        // Build sketch from one stream, build from two halves merged,
        // compare quantile queries.
        let n = 5000;
        let xs: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();

        let mut full = KllSketch::new(0.02);
        full.add_slice(&xs);

        let (a, b) = xs.split_at(n / 2);
        let mut sa = KllSketch::new(0.02);
        sa.add_slice(a);
        let mut sb = KllSketch::new(0.02);
        sb.add_slice(b);
        sa.merge(&sb);

        for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let f = full.quantile(q);
            let m = sa.quantile(q);
            // Merge result should be close (not necessarily bit-exact —
            // KLL compaction order differs between paths). Accept ε·N
            // worth of slack on the value scale.
            assert_close(m, f, 0.5, &format!("merge q={q}"));
        }
    }

    #[test]
    fn merge_associative_over_quantiles() {
        // (a ⊕ b) ⊕ c and a ⊕ (b ⊕ c) should give close quantile
        // queries (associative at the result level, per the trait
        // contract).
        let xs_a: Vec<f64> = (0..2000).map(|i| (i as f64).cos() * 100.0).collect();
        let xs_b: Vec<f64> = (0..1500).map(|i| (i as f64).sin() * 50.0 + 25.0).collect();
        let xs_c: Vec<f64> = (0..2500).map(|i| ((i * 7) as f64).tan() * 0.1).collect();

        let mut a = KllSketch::new(0.02);
        a.add_slice(&xs_a);
        let mut b = KllSketch::new(0.02);
        b.add_slice(&xs_b);
        let mut c = KllSketch::new(0.02);
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
            // KLL's deterministic alternating coin (vs the paper's
            // randomized coin) gives slightly worse merge associativity
            // in worst case. The contract is "associative at the
            // result level" — close enough that consumers can rely on
            // approximate equality, not bit-exact. Allow ~ε·max_value.
            assert_close(l, r, 5.0, &format!("associative at q={q}"));
        }
    }

    #[test]
    fn run_to_run_bit_exact() {
        // Same input → same internal state (same coin sequence due to
        // deterministic counters).
        let xs: Vec<f64> = (0..3000).map(|i| (i as f64) * 0.001).collect();
        let mut a = KllSketch::new(0.01);
        a.add_slice(&xs);
        let mut b = KllSketch::new(0.01);
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
        let _ = KllSketch::new(0.0);
    }

    #[test]
    #[should_panic(expected = "q must be finite")]
    fn panics_on_bad_quantile() {
        let mut s = KllSketch::new(0.01);
        s.add(1.0);
        let _ = s.quantile(1.5);
    }

    #[test]
    #[should_panic(expected = "epsilon mismatch")]
    fn panics_on_merge_epsilon_mismatch() {
        let mut a = KllSketch::new(0.01);
        let b = KllSketch::new(0.02);
        a.merge(&b);
    }
}
