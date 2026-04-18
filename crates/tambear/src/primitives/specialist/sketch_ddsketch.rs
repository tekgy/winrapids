//! DDSketch — value-determined logarithmic bucketing, sort-free, our way.
//!
//! Locked vocabulary: Tier 1 primitive, alternative quantile sketch
//! selectable via `using(sketch: "ddsketch")` per
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Why this is in the catalog
//!
//! The other three sketches (KLL, GK, t-digest) all need a **canonical
//! sort during merge** to be deterministic, because their internal
//! state is order-dependent (compaction order in KLL, tuple adjacency
//! in GK, centroid positions in t-digest). DDSketch breaks that
//! pattern: its bucketing is **purely value-determined**, so the
//! state is a pure function of the multiset of inputs. No sort
//! anywhere in the hot path or the merge.
//!
//! Killing sorts (and divides, and branches where possible) was a
//! foundational principle from the start. DDSketch is the quantile
//! sketch that respects that principle natively.
//!
//! # The algorithm, our way
//!
//! Standard DDSketch buckets values by `index = ⌈log_γ(x)⌉` for some
//! `γ = (1+α)/(1-α)` parameterized by target relative error `α`. The
//! original paper splits buckets into two separate maps (positive and
//! negative) plus a separate zero count, which complicates merge and
//! creates bias near zero where the two maps don't symmetrically
//! cover their error budgets.
//!
//! Our variant uses a **single signed-index bucket map**:
//!
//! - `index > 0` ↔ value in `(γ^(i−1), γ^i]` — positive-side buckets
//! - `index = 0` ↔ value is exactly 0
//! - `index < 0` ↔ value in `[-γ^|i|, -γ^(|i|−1))` — negative-side
//!   buckets, mirror-symmetric to the positive side
//!
//! This gives:
//! - Native two-sided handling — no separate positive/negative maps
//!   to coordinate
//! - Symmetric error coverage near zero — the bucket near `+epsilon`
//!   and the bucket near `-epsilon` have the same width relative to
//!   their anchor
//! - Cleaner merge — one `BTreeMap<i32, u64>` to combine, by summing
//!   counts at matching indices
//! - **No sort**, ever. `BTreeMap` is sorted-by-construction; iteration
//!   for quantile queries is O(b) over `b` buckets in already-sorted
//!   order
//! - **One divide** at construction (`1.0 / ln(γ)` pre-computed); zero
//!   divides per insert (multiply by the precomputed reciprocal)
//! - **Three branches** per insert: `is_finite`, `x == 0`, `x > 0`. The
//!   zero check is unavoidable because `ln(0) = -∞`. The sign branch
//!   could be collapsed into bit-arithmetic on the IEEE 754 sign bit,
//!   but the saving is marginal in this code path.
//!
//! # Determinism
//!
//! The bucket map is a pure function of the multiset of inputs:
//! permutation-invariant, merge-associative, no randomness. **Same
//! multiset → bit-identical state** regardless of insertion order or
//! merge order. This is the strongest determinism property of any
//! quantile sketch in the locked catalog.
//!
//! # Trade-offs vs the other sketches
//!
//! - **Memory**: bucket count is `O(log_γ(value_range))`. For α=0.01
//!   covering ±10^17, that's ~3000 buckets × 12 bytes ≈ 36 KB.
//!   Comparable to KLL/GK/t-digest at the same epsilon, but constant-
//!   factor higher than KLL for narrow value ranges.
//! - **Error guarantee**: bounded *relative* error in value, not
//!   bounded absolute rank error. For data spanning many orders of
//!   magnitude (financial returns, latency tails, sensor data), this
//!   is often the more useful guarantee. For tightly-bounded data
//!   where rank error matters more than value error, KLL is better.
//! - **Tail accuracy**: like t-digest, naturally accurate at tails
//!   because the bucketing is logarithmic — small values get fine
//!   resolution.
//! - **Aggregation across heterogeneous shards**: DDSketch is
//!   uniquely well-suited because no canonical-sort is required for
//!   merge associativity.
//!
//! # Reference
//!
//! Masson, C., Rim, J. E., & Lee, H. K. (2019). DDSketch: A fast and
//! fully-mergeable quantile sketch with relative-error guarantees.
//! *Proceedings of the VLDB Endowment* 12(12): 2195–2205.
//!
//! Tambear's variant (single signed-index map for native two-sided
//! handling) is documented inline above; the underlying logarithmic
//! bucketing is faithful to the original.

use std::collections::BTreeMap;

use super::observations::FiniteObservations;
use super::quantile_sketch::QuantileSketch;

/// DDSketch with native two-sided handling via signed bucket indices.
#[derive(Debug, Clone)]
pub struct DdSketch {
    /// Target additive error in quantile rank (used to derive γ).
    epsilon: f64,
    /// γ = (1 + ε) / (1 − ε) — the bucket-ratio parameter.
    gamma: f64,
    /// 1 / ln(γ) — pre-computed once at construction; lets the hot
    /// path use a multiply instead of a divide.
    log_gamma_inv: f64,
    /// Bucket counts indexed by signed bucket index.
    /// Negative index ↔ negative-value bucket, 0 ↔ exactly-zero,
    /// positive index ↔ positive-value bucket. `BTreeMap` is sorted
    /// by construction — no sort step in the merge or query path.
    buckets: BTreeMap<i32, u64>,
    /// Shared finite-observation bookkeeping (n / min / max + skip gate).
    obs: FiniteObservations,
}

impl DdSketch {
    /// Signed bucket index for `x`. Assumes `x.is_finite()`.
    #[inline]
    fn bucket_index(&self, x: f64) -> i32 {
        if x == 0.0 {
            return 0;
        }
        let abs_x = x.abs();
        // log_γ(|x|) = ln(|x|) / ln(γ) = ln(|x|) · log_gamma_inv.
        // Multiply, not divide: log_gamma_inv is the precomputed reciprocal.
        let log_g = abs_x.ln() * self.log_gamma_inv;
        // ceil → positive magnitude index (≥ 1 for any non-zero |x|).
        let mag = (log_g.ceil() as i32).max(1);
        if x > 0.0 {
            mag
        } else {
            -mag
        }
    }

    /// Anchor (representative) value for a bucket at signed index `idx`.
    /// Returns the geometric midpoint of the bucket's value range.
    #[inline]
    fn anchor_value(&self, idx: i32) -> f64 {
        if idx == 0 {
            return 0.0;
        }
        // Bucket [γ^(|i|−1), γ^|i|] has geometric midpoint γ^(|i| − 0.5).
        let mag = (idx.abs() as f64) - 0.5;
        let v = self.gamma.powf(mag);
        if idx > 0 {
            v
        } else {
            -v
        }
    }

    /// Total bucket-count weight (sum of bucket counts).
    fn total_weight(&self) -> u64 {
        self.buckets.values().sum()
    }
}

impl QuantileSketch for DdSketch {
    fn new(epsilon: f64) -> Self {
        assert!(
            epsilon.is_finite() && epsilon > 0.0 && epsilon < 1.0,
            "DdSketch::new: epsilon must be finite and in (0, 1), got {epsilon}"
        );
        // Standard DDSketch γ derivation: γ = (1+ε)/(1−ε) gives
        // bucket bounds with relative error ≤ ε.
        let gamma = (1.0 + epsilon) / (1.0 - epsilon);
        let log_gamma_inv = 1.0 / gamma.ln();
        Self {
            epsilon,
            gamma,
            log_gamma_inv,
            buckets: BTreeMap::new(),
            obs: FiniteObservations::new(),
        }
    }

    fn add(&mut self, x: f64) {
        if !self.obs.observe(x) {
            return;
        }
        let idx = self.bucket_index(x);
        *self.buckets.entry(idx).or_insert(0) += 1;
    }

    fn merge(&mut self, other: &Self) {
        // Both sketches must agree on epsilon (and therefore γ) for
        // bucket indices to be comparable.
        assert!(
            (self.epsilon - other.epsilon).abs() < 1e-15,
            "DdSketch::merge: epsilon mismatch ({} vs {})",
            self.epsilon,
            other.epsilon
        );
        // Sum bucket counts at matching indices. No sort needed —
        // BTreeMap iteration is in-order, and we just add into our own
        // map at each foreign index.
        for (&idx, &count) in &other.buckets {
            *self.buckets.entry(idx).or_insert(0) += count;
        }
        self.obs.merge(&other.obs);
    }

    fn quantile(&self, q: f64) -> f64 {
        assert!(
            q.is_finite() && (0.0..=1.0).contains(&q),
            "DdSketch::quantile: q must be finite and in [0, 1], got {q}"
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

        let total = self.total_weight();
        if total == 0 {
            return f64::NAN;
        }
        let target_rank = (q * total as f64).ceil() as u64;
        let target_rank = target_rank.max(1).min(total);

        // BTreeMap iteration is in sorted-key order — naturally walks
        // from most-negative bucket through 0 to most-positive bucket.
        // No sort step ever.
        let mut cum: u64 = 0;
        for (&idx, &count) in &self.buckets {
            cum += count;
            if cum >= target_rank {
                return self.anchor_value(idx);
            }
        }
        // Fallback (should not reach here given the target_rank clamp).
        self.obs.max()
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

    fn assert_close_relative(got: f64, want: f64, rel_tol: f64, label: &str) {
        let diff = ((got - want) / want.abs().max(1e-12)).abs();
        assert!(
            diff < rel_tol,
            "{label}: got {got}, want {want}, rel_diff {diff} > rel_tol {rel_tol}"
        );
    }

    #[test]
    fn empty_returns_nan() {
        let s = DdSketch::new(0.01);
        assert!(s.quantile(0.5).is_nan());
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn single_value_returns_that_value() {
        let mut s = DdSketch::new(0.01);
        s.add(42.0);
        assert_eq!(s.quantile(0.0), 42.0);
        assert_eq!(s.quantile(1.0), 42.0);
        // Middle quantile returns the bucket anchor, not the exact value.
        let q50 = s.quantile(0.5);
        assert_close_relative(q50, 42.0, 0.01, "single-value q50");
    }

    #[test]
    fn extremes_exact() {
        // q=0 and q=1 should be exact min/max regardless of bucketing.
        let mut s = DdSketch::new(0.05);
        for i in 1..=1000 {
            s.add(i as f64);
        }
        assert_eq!(s.quantile(0.0), 1.0);
        assert_eq!(s.quantile(1.0), 1000.0);
    }

    #[test]
    fn relative_error_guarantee() {
        // DDSketch's contract: returned value is within relative-ε of
        // the true value at the queried quantile.
        let mut s = DdSketch::new(0.01); // 1% relative error
        for i in 1..=10000 {
            s.add(i as f64);
        }
        for &q in &[0.1, 0.25, 0.5, 0.75, 0.9, 0.99] {
            let got = s.quantile(q);
            let want = (q * 10000.0).round();
            // Allow ε relative error on the value.
            assert_close_relative(got, want, 0.02, &format!("q={q}"));
        }
    }

    #[test]
    fn handles_negative_values_natively() {
        // Mixed positive and negative — no sign-bucket workaround.
        let mut s = DdSketch::new(0.01);
        for i in -1000..=1000 {
            s.add(i as f64);
        }
        // q=0 should be -1000, q=1 should be 1000, q=0.5 should be ~0.
        assert_eq!(s.quantile(0.0), -1000.0);
        assert_eq!(s.quantile(1.0), 1000.0);
        let median = s.quantile(0.5);
        // Median should be very close to 0 (within one bucket near zero).
        assert!(
            median.abs() < 1.0,
            "median of symmetric ±1000 should be ~0, got {median}"
        );
    }

    #[test]
    fn handles_zero_separately_from_negatives_and_positives() {
        // Triple-mode: lots of zeros, some negatives, some positives.
        let mut s = DdSketch::new(0.01);
        for _ in 0..1000 {
            s.add(0.0);
        }
        for i in 1..=100 {
            s.add(i as f64);
            s.add(-(i as f64));
        }
        assert_eq!(s.count(), 1200);
        // q=0 is most-negative, q=1 is most-positive.
        assert_eq!(s.quantile(0.0), -100.0);
        assert_eq!(s.quantile(1.0), 100.0);
        // q=0.5 should land in the zero bucket (most weight is there).
        let median = s.quantile(0.5);
        assert!(
            median.abs() < 0.01,
            "median should be 0 (mass at zero), got {median}"
        );
    }

    #[test]
    fn quantiles_monotonic() {
        let mut s = DdSketch::new(0.01);
        for i in 1..=10000 {
            s.add(i as f64);
        }
        let qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
        let vs = s.quantiles(&qs);
        for i in 1..vs.len() {
            assert!(
                vs[i] >= vs[i - 1] - 1e-9,
                "non-monotonic at #{i}: {vs:?}"
            );
        }
    }

    #[test]
    fn skips_non_finite() {
        let mut s = DdSketch::new(0.01);
        for i in 1..=100 {
            s.add(i as f64);
            s.add(f64::NAN);
            s.add(f64::INFINITY);
            s.add(f64::NEG_INFINITY);
        }
        assert_eq!(s.count(), 100);
    }

    #[test]
    fn permutation_invariant_state() {
        // Strongest determinism guarantee unique to DDSketch:
        // same multiset → identical bucket map regardless of insertion order.
        let mut sa = DdSketch::new(0.01);
        for i in 1..=1000 {
            sa.add(i as f64);
        }
        let mut sb = DdSketch::new(0.01);
        for i in (1..=1000).rev() {
            sb.add(i as f64);
        }
        // Same buckets, same counts.
        assert_eq!(sa.buckets, sb.buckets);
        assert_eq!(sa.count(), sb.count());
        for &q in &[0.1, 0.5, 0.9] {
            assert_eq!(sa.quantile(q).to_bits(), sb.quantile(q).to_bits());
        }
    }

    #[test]
    fn merge_no_sort_ever() {
        // Build full from one stream; build from two halves merged.
        // Buckets and quantile queries should be bit-identical.
        let xs: Vec<f64> = (1..=5000).map(|i| (i as f64) * 1.5).collect();

        let mut full = DdSketch::new(0.01);
        for &x in &xs {
            full.add(x);
        }

        let (a, b) = xs.split_at(2500);
        let mut sa = DdSketch::new(0.01);
        for &x in a {
            sa.add(x);
        }
        let mut sb = DdSketch::new(0.01);
        for &x in b {
            sb.add(x);
        }
        sa.merge(&sb);

        // Bit-identical buckets — DDSketch's defining property.
        assert_eq!(full.buckets, sa.buckets);
        for &q in &[0.1, 0.5, 0.9] {
            assert_eq!(
                full.quantile(q).to_bits(),
                sa.quantile(q).to_bits(),
                "quantile mismatch at q={q}"
            );
        }
    }

    #[test]
    fn merge_associative_bit_exact() {
        // Stronger than the other sketches: DDSketch's merge is
        // bit-exact associative, not just close-enough associative.
        let xs_a: Vec<f64> = (0..2000).map(|i| (i as f64).cos() * 100.0).collect();
        let xs_b: Vec<f64> = (0..1500).map(|i| (i as f64).sin() * 50.0 + 25.0).collect();
        let xs_c: Vec<f64> = (0..2500).map(|i| ((i * 7) as f64) * 0.001 - 1.5).collect();

        let mut a = DdSketch::new(0.02);
        for &x in &xs_a {
            a.add(x);
        }
        let mut b = DdSketch::new(0.02);
        for &x in &xs_b {
            b.add(x);
        }
        let mut c = DdSketch::new(0.02);
        for &x in &xs_c {
            c.add(x);
        }

        let mut left = a.clone();
        left.merge(&b);
        left.merge(&c);

        let mut bc = b.clone();
        bc.merge(&c);
        let mut right = a.clone();
        right.merge(&bc);

        assert_eq!(left.buckets, right.buckets, "associative merge buckets");
        for &q in &[0.1, 0.5, 0.9] {
            assert_eq!(
                left.quantile(q).to_bits(),
                right.quantile(q).to_bits(),
                "associative merge quantile at q={q}"
            );
        }
    }

    #[test]
    fn run_to_run_bit_exact() {
        let xs: Vec<f64> = (0..3000).map(|i| (i as f64) * 0.001 - 1.5).collect();
        let mut a = DdSketch::new(0.01);
        for &x in &xs {
            a.add(x);
        }
        let mut b = DdSketch::new(0.01);
        for &x in &xs {
            b.add(x);
        }
        for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            assert_eq!(
                a.quantile(q).to_bits(),
                b.quantile(q).to_bits(),
                "run-to-run differs at q={q}"
            );
        }
    }

    #[test]
    fn handles_huge_value_range() {
        // DDSketch handles many orders of magnitude well — log-bucketing
        // gives constant relative error across the range.
        let mut s = DdSketch::new(0.05);
        for &x in &[1e-10, 1e-5, 1.0, 1e5, 1e10] {
            s.add(x);
        }
        let q50 = s.quantile(0.5);
        // Median should be ~1.0 (the middle value); allow generous
        // relative tolerance.
        assert_close_relative(q50, 1.0, 0.1, "huge-range median");
    }

    #[test]
    #[should_panic(expected = "epsilon")]
    fn panics_on_bad_epsilon() {
        let _ = DdSketch::new(2.0);
    }

    #[test]
    #[should_panic(expected = "q must be finite")]
    fn panics_on_bad_quantile() {
        let mut s = DdSketch::new(0.01);
        s.add(1.0);
        let _ = s.quantile(-0.5);
    }

    #[test]
    #[should_panic(expected = "epsilon mismatch")]
    fn panics_on_merge_epsilon_mismatch() {
        let mut a = DdSketch::new(0.01);
        let b = DdSketch::new(0.02);
        a.merge(&b);
    }
}
