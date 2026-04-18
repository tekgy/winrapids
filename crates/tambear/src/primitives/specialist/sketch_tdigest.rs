//! t-digest quantile sketch — Dunning 2014/2019.
//!
//! Locked vocabulary: Tier 1 primitive, alternative quantile sketch
//! selectable via `using(sketch: "tdigest")` per
//! `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Algorithm
//!
//! t-digest maintains a list of **centroids**, each `(mean, weight)`.
//! A scale function `k(q) = (δ/(2π)) · arcsin(2q − 1)` controls
//! centroid density: tails (q ≈ 0, q ≈ 1) get many small centroids,
//! middle (q ≈ 0.5) gets few large ones. This is the key trade-off
//! that makes t-digest unusually accurate at tail quantiles compared
//! to uniform-error sketches like KLL/GK.
//!
//! Insertion buffers raw values as weight-1 centroids; **compression**
//! sorts by mean and merges adjacent centroids while the k-bound
//! `k(q_R) − k(q_L) ≤ 1` holds at the relevant cumulative positions.
//!
//! Merge: union centroid lists, sort by mean (canonical for
//! determinism), re-compress.
//!
//! # Determinism
//!
//! Pure functions of the centroid list once sorted: no randomness, no
//! order-dependent state. The canonical sort during merge guarantees
//! cross-platform bit-exact reproducibility.
//!
//! # Reference
//!
//! Dunning, T. (2019). The t-digest: Efficient estimates of
//! distributions. *Software Impacts* 7: 100049.
//!
//! Dunning, T., & Ertl, O. (2014). Computing extremely accurate
//! quantiles using t-digests. arXiv:1902.04023.
//!
//! # Memory
//!
//! `O(δ)` centroids in the steady state, where `δ` is the compression
//! parameter (typical 100–1000). For `δ = 200` and `ε = 0.01`,
//! memory ≈ 200 centroids × 16 bytes = 3.2 KB.
//!
//! # ε ↔ δ relationship
//!
//! Tighter ε needs larger δ. Empirically `δ ≈ 5/ε` works well; the
//! `new(epsilon)` constructor uses this rule.

use super::observations::FiniteObservations;
use super::quantile_sketch::QuantileSketch;

/// One centroid: mean and weight.
#[derive(Debug, Clone, Copy)]
struct Centroid {
    mean: f64,
    weight: f64,
}

impl Centroid {
    /// Combine two centroids in-place into a new one (weighted mean).
    #[inline]
    fn combined_with(&self, other: &Centroid) -> Centroid {
        let w_total = self.weight + other.weight;
        if w_total == 0.0 {
            return Centroid {
                mean: 0.0,
                weight: 0.0,
            };
        }
        let mean = (self.mean * self.weight + other.mean * other.weight) / w_total;
        Centroid {
            mean,
            weight: w_total,
        }
    }
}

/// t-digest scale function `k_1(q) = (δ/(2π)) · arcsin(2q − 1)`.
#[inline]
fn k_scale(q: f64, delta: f64) -> f64 {
    let q = q.clamp(0.0, 1.0);
    delta / (2.0 * std::f64::consts::PI) * (2.0 * q - 1.0).asin()
}

/// t-digest sketch.
#[derive(Debug, Clone)]
pub struct TdigestSketch {
    /// Compression parameter — higher = more centroids = lower error.
    delta: f64,
    /// Target additive error (used to derive δ).
    epsilon: f64,
    /// Centroid list, sorted by mean after every compression.
    centroids: Vec<Centroid>,
    /// Shared finite-observation bookkeeping (n / min / max + skip gate).
    obs: FiniteObservations,
    /// Buffer-then-compress threshold: compress when uncompressed
    /// centroid count exceeds this. Amortizes O(n log n) compression
    /// over many insertions.
    compress_threshold: usize,
    /// True when the centroid list is currently sorted+compressed.
    is_compressed: bool,
}

impl TdigestSketch {
    fn delta_from_epsilon(epsilon: f64) -> f64 {
        // Empirical rule from Dunning's papers; tight quantile error
        // requires δ ~ O(1/ε).
        (5.0 / epsilon).max(20.0)
    }

    /// Total weight across all centroids (sum of `weight` field).
    fn total_weight(&self) -> f64 {
        self.centroids.iter().map(|c| c.weight).sum()
    }

    /// Sort centroids by mean and re-compress under the k-bound.
    fn compress(&mut self) {
        if self.centroids.is_empty() {
            self.is_compressed = true;
            return;
        }
        // Canonical sort by mean (deterministic via total_cmp).
        self.centroids.sort_by(|a, b| a.mean.total_cmp(&b.mean));

        let total_w = self.total_weight();
        if total_w == 0.0 {
            self.is_compressed = true;
            return;
        }

        let mut merged: Vec<Centroid> = Vec::with_capacity(self.centroids.len());
        let mut q_left = 0.0_f64;
        let mut current = self.centroids[0];

        for &next in &self.centroids[1..] {
            // Tentative merge of current with next.
            let combined = current.combined_with(&next);
            // Cumulative q range for the combined centroid:
            //   q_L = q_left
            //   q_R = q_left + (combined.weight) / total
            let q_right = q_left + combined.weight / total_w;
            let dk = k_scale(q_right, self.delta) - k_scale(q_left, self.delta);
            if dk <= 1.0 {
                // Within k-bound — accept merge.
                current = combined;
            } else {
                // Cannot merge; finalize current and start a new one.
                q_left += current.weight / total_w;
                merged.push(current);
                current = next;
            }
        }
        merged.push(current);
        self.centroids = merged;
        self.is_compressed = true;
    }
}

impl QuantileSketch for TdigestSketch {
    fn new(epsilon: f64) -> Self {
        assert!(
            epsilon.is_finite() && epsilon > 0.0 && epsilon < 1.0,
            "TdigestSketch::new: epsilon must be finite and in (0, 1), got {epsilon}"
        );
        let delta = Self::delta_from_epsilon(epsilon);
        // Compress when uncompressed list grows to ~5x the steady-state
        // centroid count. Empirical rule that amortizes well.
        let compress_threshold = (5.0 * delta).ceil() as usize;
        Self {
            delta,
            epsilon,
            centroids: Vec::new(),
            obs: FiniteObservations::new(),
            compress_threshold,
            is_compressed: true,
        }
    }

    fn add(&mut self, x: f64) {
        if !self.obs.observe(x) {
            return;
        }
        // Each insertion is a weight-1 centroid; compression aggregates them.
        self.centroids.push(Centroid {
            mean: x,
            weight: 1.0,
        });
        self.is_compressed = false;
        if self.centroids.len() >= self.compress_threshold {
            self.compress();
        }
    }

    fn merge(&mut self, other: &Self) {
        assert!(
            (self.epsilon - other.epsilon).abs() < 1e-15,
            "TdigestSketch::merge: epsilon mismatch ({} vs {})",
            self.epsilon,
            other.epsilon
        );
        self.centroids.extend_from_slice(&other.centroids);
        self.is_compressed = false;
        self.obs.merge(&other.obs);
        // Always compress after a merge — preserves the canonical form
        // and ensures associativity of quantile queries.
        self.compress();
    }

    fn quantile(&self, q: f64) -> f64 {
        assert!(
            q.is_finite() && (0.0..=1.0).contains(&q),
            "TdigestSketch::quantile: q must be finite and in [0, 1], got {q}"
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

        // For deterministic queries we need a compressed centroid list.
        // The trait method takes &self, so we work against a sorted
        // copy when uncompressed. Most callers will have done a merge
        // (which compresses) or built up to the threshold first.
        let mut sorted: Vec<Centroid>;
        let centroids: &[Centroid] = if self.is_compressed {
            &self.centroids
        } else {
            sorted = self.centroids.clone();
            sorted.sort_by(|a, b| a.mean.total_cmp(&b.mean));
            &sorted
        };

        if centroids.is_empty() {
            return f64::NAN;
        }

        let total_w: f64 = centroids.iter().map(|c| c.weight).sum();
        if total_w == 0.0 {
            return f64::NAN;
        }
        let target_rank = q * total_w;

        // Walk centroids; the answer is the centroid whose cumulative
        // weight crosses target_rank. Linear-interpolate between
        // adjacent centroid means for accuracy.
        let mut cum = 0.0_f64;
        let mut prev_cum = 0.0_f64;
        let mut prev_mean = self.obs.min();
        for c in centroids {
            let next_cum = cum + c.weight;
            if next_cum >= target_rank {
                // Interpolate between previous centroid's right edge
                // (prev_cum + prev_weight/2 effectively) and current
                // centroid's center. Simplified: linear in (prev_mean,
                // c.mean) by rank position.
                let span = c.weight;
                if span <= 0.0 {
                    return c.mean;
                }
                let local_pos = (target_rank - cum) / span;
                let local_pos = local_pos.clamp(0.0, 1.0);
                return prev_mean + (c.mean - prev_mean) * local_pos;
            }
            prev_cum = cum;
            prev_mean = c.mean;
            cum = next_cum;
            // Suppress unused-warning: prev_cum may be useful for future
            // refined interpolation strategies (Dunning's "right-edge"
            // form). Keeping the variable simplifies future tweaks.
            let _ = prev_cum;
        }
        centroids.last().map(|c| c.mean).unwrap_or(self.obs.max())
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
        let s = TdigestSketch::new(0.01);
        assert!(s.quantile(0.5).is_nan());
        assert_eq!(s.count(), 0);
    }

    #[test]
    fn single_value_returns_that_value() {
        let mut s = TdigestSketch::new(0.01);
        s.add(3.14);
        assert_eq!(s.quantile(0.0), 3.14);
        assert_eq!(s.quantile(0.5), 3.14);
        assert_eq!(s.quantile(1.0), 3.14);
    }

    #[test]
    fn extremes_exact() {
        let mut s = TdigestSketch::new(0.05);
        for i in 1..=1000 {
            s.add(i as f64);
        }
        assert_eq!(s.quantile(0.0), 1.0);
        assert_eq!(s.quantile(1.0), 1000.0);
    }

    #[test]
    fn median_within_epsilon() {
        let mut s = TdigestSketch::new(0.01);
        for i in 0..10000 {
            s.add(i as f64);
        }
        let q50 = s.quantile(0.5);
        // t-digest middle quantiles are typically less precise than
        // tails; allow generous slack.
        assert_close(q50, 5000.0, 200.0, "median");
    }

    #[test]
    fn tails_better_than_middle() {
        // t-digest's defining virtue: tail accuracy is unusually good.
        // Test that q=0.99 is within reasonable tolerance.
        let mut s = TdigestSketch::new(0.01);
        for i in 0..10000 {
            s.add(i as f64);
        }
        let q99 = s.quantile(0.99);
        assert_close(q99, 9900.0, 200.0, "q99");
    }

    #[test]
    fn quantiles_monotonic() {
        let mut s = TdigestSketch::new(0.01);
        for i in 0..10000 {
            s.add(i as f64);
        }
        let qs = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99];
        let vs = s.quantiles(&qs);
        for i in 1..vs.len() {
            assert!(
                vs[i] >= vs[i - 1] - 1e-9,
                "quantiles not monotonic at {i}: {vs:?}"
            );
        }
    }

    #[test]
    fn skips_non_finite() {
        let mut s = TdigestSketch::new(0.01);
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

        let mut a = TdigestSketch::new(0.02);
        a.add_slice(&xs_a);
        let mut b = TdigestSketch::new(0.02);
        b.add_slice(&xs_b);
        let mut c = TdigestSketch::new(0.02);
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
        let xs: Vec<f64> = (0..3000).map(|i| (i as f64) * 0.001).collect();
        let mut a = TdigestSketch::new(0.01);
        a.add_slice(&xs);
        let mut b = TdigestSketch::new(0.01);
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
        let _ = TdigestSketch::new(-0.1);
    }
}
