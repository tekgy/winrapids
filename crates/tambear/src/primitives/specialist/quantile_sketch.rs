//! Quantile-sketch trait — common interface for KLL / GK / t-digest.
//!
//! Locked vocabulary: this is a Tier 1 primitive interface (low-level
//! implementation machinery). Recipes that need quantile estimation
//! (cvar, hill_estimator_streaming, price_percentiles) compose against
//! this trait; the user picks the underlying sketch via
//! `using(sketch: "kll" | "gk" | "tdigest")` at the recipe call site.
//!
//! See `R:\winrapids\docs\architecture\vocabulary.md`.
//!
//! # Why three sketches
//!
//! All three answer the same question — "given a stream of f64 values,
//! return the value at quantile q ∈ [0,1] with bounded error" — but
//! make different trade-offs:
//!
//! - **KLL** (Karnin-Lang-Liberty 2016) — the locked default. Tight
//!   memory bound `O((1/ε) · sqrt(log(1/(εδ))))`, provably optimal up
//!   to log factors. Mergeable. Hierarchical compactor structure.
//!
//! - **GK** (Greenwald-Khanna 2001) — intrinsically mergeable (no
//!   canonical-ordering requirement). Worst-case error bound
//!   guaranteed. Memory `O((1/ε) · log(εN))`.
//!
//! - **t-digest** (Dunning 2014/2019) — best empirical accuracy on
//!   tail quantiles (better than KLL/GK at q ≈ 0 or q ≈ 1). Merge
//!   requires canonical sort order to be deterministic.
//!
//! # Determinism contract
//!
//! Every implementation must be **cross-platform bit-exact
//! deterministic**: same insertion sequence + same ε → identical
//! internal state and identical query results, regardless of thread
//! count, backend, or CPU architecture.
//!
//! Where an algorithm has internal randomness (KLL's compactor coin
//! flips), use a deterministic PRNG seeded from the data so the result
//! is reproducible. Where order matters (t-digest's merge), enforce
//! canonical sort.
//!
//! Merge must be associative at the result level: `(a ⊕ b) ⊕ c` and
//! `a ⊕ (b ⊕ c)` must yield identical quantile queries. The internal
//! state may differ as long as quantile() agrees bit-exactly.
//!
//! # NaN/Inf policy
//!
//! All implementations skip non-finite inputs at insertion (locked
//! NaN/Inf-skip semantics). `count()` reports the number of finite
//! observations. `quantile(q)` returns NaN if `count() == 0`.
//!
//! # Default parameters
//!
//! `epsilon` is the additive error in quantile rank. `ε = 0.01` means
//! `quantile(0.5)` returns a value whose true rank is in `[0.49, 0.51]`.
//! Tighter ε costs more memory; recipes should expose ε as a
//! `using()`-overridable parameter.

/// Common interface for mergeable quantile sketches.
///
/// All three locked-vocabulary sketches (KLL, GK, t-digest) implement
/// this trait. Recipes compose against it generically; the concrete
/// sketch is chosen at the recipe call site via `using(sketch: "...")`.
pub trait QuantileSketch: Sized {
    /// Construct a sketch with target additive error `epsilon` in
    /// quantile rank.
    ///
    /// # Panics
    ///
    /// Implementations panic if `epsilon` is non-finite, ≤ 0, or ≥ 1.
    fn new(epsilon: f64) -> Self;

    /// Insert a single observation. Non-finite values are skipped
    /// (NaN/Inf-skip per locked vocabulary).
    fn add(&mut self, x: f64);

    /// Insert a slice of observations. Default implementation calls
    /// `add` per element; implementations may override for speed.
    fn add_slice(&mut self, xs: &[f64]) {
        for &x in xs {
            self.add(x);
        }
    }

    /// Merge another sketch into this one in place. Must be
    /// associative at the query level — that is, for any sketches
    /// `a, b, c` of the same epsilon:
    ///
    /// ```text
    /// (a.merge(b).merge(c)).quantile(q) == a.merge(b.merge(c)).quantile(q)
    /// ```
    ///
    /// for every q ∈ (0, 1).
    fn merge(&mut self, other: &Self);

    /// Return the value at quantile `q ∈ [0, 1]`.
    ///
    /// `q = 0.0` returns the minimum observed; `q = 1.0` returns the
    /// maximum observed. Returns NaN if no finite values have been
    /// observed.
    ///
    /// # Panics
    ///
    /// Panics if `q` is non-finite or outside `[0, 1]`.
    fn quantile(&self, q: f64) -> f64;

    /// Return the count of finite observations ingested.
    fn count(&self) -> u64;

    /// Convenience: query multiple quantiles in one pass. Default
    /// implementation calls `quantile` per element; implementations
    /// may override for shared work (e.g., one sort).
    fn quantiles(&self, qs: &[f64]) -> Vec<f64> {
        qs.iter().map(|&q| self.quantile(q)).collect()
    }
}

/// Algorithm tag for sketch selection at the recipe layer. Used by
/// the `using(sketch: "...")` parameter resolution to dispatch to the
/// concrete sketch implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SketchAlgorithm {
    /// Karnin-Lang-Liberty 2016 — default per locked vocabulary.
    Kll,
    /// Greenwald-Khanna 2001 — intrinsically mergeable.
    Gk,
    /// Dunning 2014/2019 — best empirical tail accuracy.
    Tdigest,
    /// DDSketch (Masson-Rim-Lee 2019) with our native two-sided
    /// signed-index variant. Sort-free in hot path AND merge.
    /// Permutation-invariant state. Bounded relative error.
    DdSketch,
}

impl SketchAlgorithm {
    /// Parse from a string (typical `using()` value form).
    ///
    /// Accepts `"kll"`, `"gk"`, `"tdigest"`, `"ddsketch"` (case-insensitive).
    /// Returns `None` for any other value.
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "kll" => Some(Self::Kll),
            "gk" => Some(Self::Gk),
            "tdigest" | "t-digest" | "t_digest" => Some(Self::Tdigest),
            "ddsketch" | "dd-sketch" | "dd_sketch" => Some(Self::DdSketch),
            _ => None,
        }
    }

    /// The locked default per `vocabulary.md`: KLL.
    pub const DEFAULT: Self = Self::Kll;
}
