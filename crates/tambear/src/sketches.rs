//! # Streaming sketch data structures
//!
//! Sublinear-space estimators for classic streaming problems:
//!
//! - **HyperLogLog** (Flajolet, Fusy, Gandouet, Meunier 2007) — approximate
//!   distinct-count with ~1.04/√m relative error using m = 2^p registers.
//! - **Bloom filter** (Bloom 1970) — approximate set membership with no
//!   false negatives and a tunable false-positive rate.
//! - **Count-Min Sketch** (Cormode & Muthukrishnan 2005) — approximate
//!   frequency estimation with `(ε, δ)` guarantees: O(1/ε · log(1/δ)) space.
//! - **Top-K (Misra-Gries / SpaceSaving 1982, 2005)** — deterministic
//!   top-k heavy-hitters with k counters.
//!
//! ## Tambear contract
//!
//! - **Custom from first principles** — no wrapping of `hyperloglog`/`bloom`/
//!   `probabilistic-collections` crates. Every bit-twiddle is ours.
//! - **Accumulate + gather shaped** — `add` operations are pointwise scatter
//!   into registers; `merge` of two sketches is a pointwise accumulate (max
//!   for HLL, OR for Bloom, elementwise sum for CMS). All embarrassingly
//!   parallelizable.
//! - **Every parameter tunable** — precision, error bounds, hash seeds all
//!   explicit constructor arguments with documented defaults.
//! - **Kingdom**: A (pure streaming aggregation). No iterative fixed-point,
//!   no sequential recurrence beyond the natural one-pass-over-input.
//!
//! ## When to use
//!
//! When the stream is larger than memory, or when an exact count would
//! require an O(n) hash set and you want O(1) space instead. Bloom and HLL
//! trade space for small controlled error. Count-Min Sketch is biased one-
//! sided (never under-estimates). SpaceSaving is deterministic for the
//! heavy-hitter subset.
//!
//! ## Hashing
//!
//! Every sketch uses [`splitmix64_hash`], a deterministic 64-bit mixer
//! derived from Austin Appleby's SplitMix64. It is NOT cryptographic, it IS
//! fast, deterministic, and has good avalanche for the use cases here.
//! For multi-hash designs (Bloom, CMS) we derive independent hashes via
//! `splitmix64_hash(x ^ seed_i)` with distinct seeds.

// ═══════════════════════════════════════════════════════════════════════════
// Hash helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Deterministic 64-bit integer mixer (Appleby / MurmurHash finalizer).
///
/// Passes the avalanche test: a single bit flip in input mutates ~32 bits
/// of the output. Sufficient for bucket selection, NOT cryptographic.
#[inline]
pub fn splitmix64_hash(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D049BB133111EB);
    x ^ (x >> 31)
}

/// Hash a raw byte slice into a 64-bit value via a simple FNV-1a–like loop
/// with the SplitMix64 finalizer. Deterministic per build.
pub fn hash_bytes(data: &[u8]) -> u64 {
    let mut h: u64 = 0xCBF29CE484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001B3);
    }
    splitmix64_hash(h)
}

/// Hash an `f64` by its raw IEEE bits (NaN-aware — canonicalises NaN).
pub fn hash_f64(v: f64) -> u64 {
    let bits = if v.is_nan() { u64::MAX } else { v.to_bits() };
    splitmix64_hash(bits)
}

/// Hash a `u64` directly (just calls `splitmix64_hash`). Kept as a named
/// function so calling code can pick intent.
#[inline]
pub fn hash_u64(v: u64) -> u64 { splitmix64_hash(v) }

// ═══════════════════════════════════════════════════════════════════════════
// HyperLogLog
// ═══════════════════════════════════════════════════════════════════════════

/// HyperLogLog distinct-count sketch.
///
/// `p` is the precision (number of register index bits), giving
/// `m = 2^p` registers. Standard error is ≈ `1.04 / sqrt(m)`, so
/// `p = 14` → m = 16 384 → SE ≈ 0.81%.
/// Valid range: `4 ≤ p ≤ 18`.
#[derive(Debug, Clone)]
pub struct HyperLogLog {
    p: u8,
    m: usize,
    registers: Vec<u8>,
}

impl HyperLogLog {
    /// Create an empty HLL with `2^p` registers.
    pub fn new(p: u8) -> Self {
        let p = p.clamp(4, 18);
        let m = 1usize << p;
        Self { p, m, registers: vec![0; m] }
    }

    /// Number of registers `m = 2^p`.
    #[inline]
    pub fn m(&self) -> usize { self.m }

    /// Precision in bits.
    #[inline]
    pub fn precision(&self) -> u8 { self.p }

    /// Add a pre-hashed value. Callers should hash once before feeding.
    pub fn add_hash(&mut self, h: u64) {
        let idx = (h >> (64 - self.p)) as usize;
        // Count trailing zeros in the remaining bits + 1. We shift the
        // used bits away first.
        let shifted = (h << self.p) | (1u64 << (self.p.saturating_sub(1)));
        let rank = (shifted.leading_zeros() + 1) as u8;
        let cell = &mut self.registers[idx];
        if rank > *cell { *cell = rank; }
    }

    /// Add a `u64` value (hashed with `splitmix64_hash`).
    #[inline]
    pub fn add_u64(&mut self, v: u64) { self.add_hash(splitmix64_hash(v)); }

    /// Add an `f64` (by canonicalised bit pattern).
    #[inline]
    pub fn add_f64(&mut self, v: f64) { self.add_hash(hash_f64(v)); }

    /// Add a byte slice (hashed via `hash_bytes`).
    #[inline]
    pub fn add_bytes(&mut self, data: &[u8]) { self.add_hash(hash_bytes(data)); }

    /// Estimate the cardinality of the set of added values.
    /// Uses the standard HLL estimator with small-range (linear counting)
    /// and large-range bias corrections.
    pub fn estimate(&self) -> f64 {
        let m = self.m as f64;
        // Raw harmonic-mean estimate
        let mut sum = 0.0_f64;
        let mut zeros = 0_usize;
        for &r in &self.registers {
            sum += 2.0_f64.powi(-(r as i32));
            if r == 0 { zeros += 1; }
        }
        // α_m constant from Flajolet et al. 2007
        let alpha = match self.m {
            16 => 0.673,
            32 => 0.697,
            64 => 0.709,
            _ => 0.7213 / (1.0 + 1.079 / m),
        };
        let e = alpha * m * m / sum;
        // Small range: linear counting
        if e <= 2.5 * m && zeros > 0 {
            let zf = zeros as f64;
            return m * (m / zf).ln();
        }
        // Large range: the bias correction from the original paper uses
        // 2^32 as the hash-space upper bound. With 64-bit hashes the
        // correction is numerically negligible, so return raw estimate.
        e
    }

    /// Merge another HLL into self (element-wise max). Both must have the
    /// same precision `p`.
    pub fn merge(&mut self, other: &HyperLogLog) {
        assert_eq!(self.p, other.p, "HLL merge requires equal precision");
        for i in 0..self.m {
            if other.registers[i] > self.registers[i] {
                self.registers[i] = other.registers[i];
            }
        }
    }
}

/// One-shot convenience: estimate distinct count of an `f64` slice.
pub fn count_unique_approx_hll(x: &[f64], precision: u8) -> f64 {
    let mut h = HyperLogLog::new(precision);
    for &v in x { h.add_f64(v); }
    h.estimate()
}

/// Build (or retrieve) a HyperLogLog sketch for `values` via [`TamSession`]
/// so downstream consumers reuse the same sketch instead of rescanning.
///
/// Kingdom A (pure streaming aggregation): once computed, the sketch is
/// immutable and any number of downstream estimators can read from it.
pub fn hll_session(
    session: &mut crate::intermediates::TamSession,
    values: &[f64],
    precision: u8,
) -> HyperLogLog {
    use std::sync::Arc;
    use crate::intermediates::{DataId, IntermediateTag, SketchKind};

    let data_id = DataId::from_f64(values);
    let tag = IntermediateTag::Sketch {
        kind: SketchKind::HyperLogLog,
        precision: precision as u32,
        data_id,
    };
    if let Some(cached) = session.get::<HyperLogLog>(&tag) {
        return (*cached).clone();
    }
    let mut h = HyperLogLog::new(precision);
    for &v in values { h.add_f64(v); }
    session.register(tag, Arc::new(h.clone()));
    h
}

// ═══════════════════════════════════════════════════════════════════════════
// Bloom filter
// ═══════════════════════════════════════════════════════════════════════════

/// A classic Bloom filter with configurable bit capacity and number of hashes.
///
/// For a target FPR `p` and expected insertions `n`, the optimal parameters
/// are `m = -n ln p / (ln 2)²` bits and `k = (m/n) ln 2` hash functions.
#[derive(Debug, Clone)]
pub struct BloomFilter {
    bits: Vec<u64>,   // Bit array packed into u64 words
    num_bits: usize,
    k: usize,
}

impl BloomFilter {
    /// Create a filter with `num_bits` bit capacity and `k` hash functions.
    pub fn new(num_bits: usize, k: usize) -> Self {
        let words = (num_bits + 63) / 64;
        Self { bits: vec![0; words], num_bits, k }
    }

    /// Build a filter with optimal parameters for `n` expected inserts and
    /// false-positive rate `fpr` (e.g. `0.01`).
    pub fn with_fpr(n: usize, fpr: f64) -> Self {
        let fpr = fpr.clamp(1e-12, 0.5);
        let n = n.max(1) as f64;
        let ln2 = std::f64::consts::LN_2;
        let num_bits = (-n * fpr.ln() / (ln2 * ln2)).ceil().max(8.0) as usize;
        let k = ((num_bits as f64 / n) * ln2).ceil().max(1.0) as usize;
        Self::new(num_bits, k)
    }

    #[inline]
    fn bit_index(&self, h1: u64, h2: u64, i: usize) -> usize {
        // Double-hashing trick (Kirsch & Mitzenmacher 2006): h_i = h1 + i*h2
        let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
        (combined as usize) % self.num_bits.max(1)
    }

    fn set_bit(&mut self, idx: usize) {
        let word = idx / 64;
        let bit = idx % 64;
        self.bits[word] |= 1u64 << bit;
    }

    fn get_bit(&self, idx: usize) -> bool {
        let word = idx / 64;
        let bit = idx % 64;
        (self.bits[word] >> bit) & 1 == 1
    }

    /// Insert a pre-hashed value. Derives `k` positions via double hashing.
    pub fn insert_hash(&mut self, h: u64) {
        let h1 = h;
        let h2 = splitmix64_hash(h.wrapping_add(0x1234_5678_9ABC_DEF0));
        for i in 0..self.k {
            let idx = self.bit_index(h1, h2, i);
            self.set_bit(idx);
        }
    }

    /// Membership query on a pre-hashed value.
    pub fn contains_hash(&self, h: u64) -> bool {
        let h1 = h;
        let h2 = splitmix64_hash(h.wrapping_add(0x1234_5678_9ABC_DEF0));
        for i in 0..self.k {
            let idx = self.bit_index(h1, h2, i);
            if !self.get_bit(idx) { return false; }
        }
        true
    }

    /// Insert a `u64` key.
    #[inline]
    pub fn insert_u64(&mut self, v: u64) { self.insert_hash(splitmix64_hash(v)); }

    /// Test membership of a `u64` key.
    #[inline]
    pub fn contains_u64(&self, v: u64) -> bool { self.contains_hash(splitmix64_hash(v)) }

    /// Insert an `f64` key.
    #[inline]
    pub fn insert_f64(&mut self, v: f64) { self.insert_hash(hash_f64(v)); }

    /// Test membership of an `f64` key.
    #[inline]
    pub fn contains_f64(&self, v: f64) -> bool { self.contains_hash(hash_f64(v)) }

    /// Union this filter with `other` (requires identical shape).
    pub fn union(&mut self, other: &BloomFilter) {
        assert_eq!(self.num_bits, other.num_bits, "shape mismatch");
        assert_eq!(self.k, other.k, "k mismatch");
        for (a, b) in self.bits.iter_mut().zip(other.bits.iter()) {
            *a |= *b;
        }
    }

    /// Number of bits currently set. Used to estimate load factor.
    pub fn popcount(&self) -> usize {
        self.bits.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Swamidass & Baldi 2007 cardinality estimate from the popcount:
    /// `n̂ = -(m/k) · ln(1 - X/m)` where X is popcount and m is bit count.
    pub fn estimate_n(&self) -> f64 {
        let m = self.num_bits as f64;
        let x = self.popcount() as f64;
        if x >= m { return f64::INFINITY; }
        -(m / self.k as f64) * (1.0 - x / m).ln()
    }
}

/// Count how many of `queries` pass the Bloom membership test — a
/// convenience helper for the "count bloom filter members" use case
/// (intersection-size estimation between a pre-built filter and a probe set).
pub fn count_bloom_filter_members(bf: &BloomFilter, queries: &[f64]) -> usize {
    queries.iter().filter(|&&q| bf.contains_f64(q)).count()
}

// ═══════════════════════════════════════════════════════════════════════════
// Count-Min Sketch
// ═══════════════════════════════════════════════════════════════════════════

/// Count-Min Sketch for approximate frequency estimation.
///
/// For `width = w`, `depth = d`, frequency estimates have additive error
/// at most `ε · N` with probability `1 − δ`, where
/// `ε = e/w` and `δ = exp(−d)`, and `N` is the stream size.
#[derive(Debug, Clone)]
pub struct CountMinSketch {
    width: usize,
    depth: usize,
    table: Vec<u64>, // flat depth×width row-major
    seeds: Vec<u64>,
}

impl CountMinSketch {
    /// Create an empty sketch with explicit `depth × width`.
    pub fn new(width: usize, depth: usize) -> Self {
        let width = width.max(1);
        let depth = depth.max(1);
        let mut seeds = Vec::with_capacity(depth);
        let mut s = 0xA5A5A5A5A5A5A5A5_u64;
        for _ in 0..depth {
            s = splitmix64_hash(s);
            seeds.push(s);
        }
        Self { width, depth, table: vec![0; width * depth], seeds }
    }

    /// Build with `(ε, δ)` guarantees. Uses
    /// `width = ceil(e / ε)`, `depth = ceil(ln(1/δ))`.
    pub fn with_epsilon_delta(epsilon: f64, delta: f64) -> Self {
        let epsilon = epsilon.clamp(1e-12, 1.0);
        let delta = delta.clamp(1e-12, 1.0);
        let w = (std::f64::consts::E / epsilon).ceil() as usize;
        let d = (1.0_f64 / delta).ln().ceil().max(1.0) as usize;
        Self::new(w, d)
    }

    #[inline]
    fn col(&self, row: usize, h: u64) -> usize {
        (splitmix64_hash(h ^ self.seeds[row]) as usize) % self.width
    }

    /// Increment the frequency of a pre-hashed key by `count`.
    pub fn add_hash(&mut self, h: u64, count: u64) {
        for r in 0..self.depth {
            let c = self.col(r, h);
            self.table[r * self.width + c] =
                self.table[r * self.width + c].saturating_add(count);
        }
    }

    /// Query the estimated frequency of a pre-hashed key.
    /// Returns the minimum over rows (the Count-Min estimator).
    pub fn estimate_hash(&self, h: u64) -> u64 {
        let mut best = u64::MAX;
        for r in 0..self.depth {
            let c = self.col(r, h);
            let v = self.table[r * self.width + c];
            if v < best { best = v; }
        }
        best
    }

    /// Increment a `u64` key.
    #[inline]
    pub fn add_u64(&mut self, v: u64, count: u64) { self.add_hash(splitmix64_hash(v), count); }

    /// Query a `u64` key.
    #[inline]
    pub fn estimate_u64(&self, v: u64) -> u64 { self.estimate_hash(splitmix64_hash(v)) }

    /// Increment an `f64` key.
    #[inline]
    pub fn add_f64(&mut self, v: f64, count: u64) { self.add_hash(hash_f64(v), count); }

    /// Query an `f64` key.
    #[inline]
    pub fn estimate_f64(&self, v: f64) -> u64 { self.estimate_hash(hash_f64(v)) }

    /// Merge another sketch of the same shape into self (elementwise add).
    pub fn merge(&mut self, other: &CountMinSketch) {
        assert_eq!(self.width, other.width);
        assert_eq!(self.depth, other.depth);
        assert_eq!(self.seeds, other.seeds, "CMS merge requires identical seeds");
        for (a, b) in self.table.iter_mut().zip(other.table.iter()) {
            *a = a.saturating_add(*b);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// SpaceSaving / Misra-Gries top-k heavy hitters
// ═══════════════════════════════════════════════════════════════════════════

/// SpaceSaving algorithm (Metwally, Agrawal, El Abbadi 2005) for deterministic
/// top-k heavy-hitter tracking. Guarantees: items with frequency greater than
/// `N/k` are guaranteed to be in the output; the estimated count is always ≥
/// the true count and differs by at most `N/k`.
///
/// Keys are `u64`; hash your `f64`/bytes before insertion.
#[derive(Debug, Clone)]
pub struct TopKCounter {
    k: usize,
    entries: Vec<(u64, u64, u64)>, // (key, count, error_bound)
}

impl TopKCounter {
    pub fn new(k: usize) -> Self {
        Self { k: k.max(1), entries: Vec::with_capacity(k.max(1)) }
    }

    pub fn add(&mut self, key: u64, count: u64) {
        // Hit on existing key
        if let Some(pos) = self.entries.iter().position(|&(k, _, _)| k == key) {
            self.entries[pos].1 = self.entries[pos].1.saturating_add(count);
            return;
        }
        // Room to grow
        if self.entries.len() < self.k {
            self.entries.push((key, count, 0));
            return;
        }
        // Replace the minimum-count entry (overcount by its count)
        let (min_idx, &(_, min_count, _)) = self.entries.iter()
            .enumerate()
            .min_by_key(|&(_, &(_, c, _))| c)
            .unwrap();
        self.entries[min_idx] = (key, min_count.saturating_add(count), min_count);
    }

    /// Top k entries sorted by estimated count (descending).
    /// Each tuple is `(key, estimated_count, max_overcount)`.
    pub fn top(&self) -> Vec<(u64, u64, u64)> {
        let mut v = self.entries.clone();
        v.sort_by(|a, b| b.1.cmp(&a.1));
        v
    }

    /// Return just the estimated frequency of a key (0 if untracked).
    pub fn estimate(&self, key: u64) -> u64 {
        self.entries.iter().find(|e| e.0 == key).map(|e| e.1).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Hash helpers ────────────────────────────────────────────────────

    #[test]
    fn splitmix_deterministic() {
        assert_eq!(splitmix64_hash(0), splitmix64_hash(0));
        assert_ne!(splitmix64_hash(0), splitmix64_hash(1));
    }

    #[test]
    fn hash_f64_nan_canonical() {
        let a = hash_f64(f64::NAN);
        let b = hash_f64(f64::from_bits(0x7ff8000000000001));
        assert_eq!(a, b, "all NaNs should hash to same value");
    }

    // ── HyperLogLog ─────────────────────────────────────────────────────

    #[test]
    fn hll_empty_is_zero() {
        let h = HyperLogLog::new(12);
        let e = h.estimate();
        assert!(e.abs() < 1e-6);
    }

    #[test]
    fn hll_distinct_small_exact() {
        // Small cardinality → linear counting regime is exact-ish
        let mut h = HyperLogLog::new(14);
        for i in 0..100u64 { h.add_u64(i); }
        let e = h.estimate();
        assert!((e - 100.0).abs() < 10.0, "HLL(100) estimate = {}", e);
    }

    #[test]
    fn hll_distinct_medium() {
        // 10_000 distinct values, p=14 → SE ≈ 0.81% → 1σ ≈ 81
        let mut h = HyperLogLog::new(14);
        for i in 0..10_000u64 { h.add_u64(i); }
        let e = h.estimate();
        let err = (e - 10_000.0).abs() / 10_000.0;
        assert!(err < 0.05, "HLL(10k) error = {}, estimate = {}", err, e);
    }

    #[test]
    fn hll_ignores_duplicates() {
        let mut h = HyperLogLog::new(12);
        for _ in 0..100_000 { h.add_u64(42); }
        let e = h.estimate();
        assert!((e - 1.0).abs() < 1.0, "duplicates should count as 1, got {}", e);
    }

    #[test]
    fn hll_merge_is_union_cardinality() {
        let mut a = HyperLogLog::new(12);
        let mut b = HyperLogLog::new(12);
        for i in 0..1_000u64 { a.add_u64(i); }
        for i in 500..1_500u64 { b.add_u64(i); }
        a.merge(&b);
        let e = a.estimate();
        // |A ∪ B| = 1500
        let err = (e - 1500.0).abs() / 1500.0;
        assert!(err < 0.1, "merged HLL error = {}, estimate = {}", err, e);
    }

    #[test]
    fn hll_precision_clamped() {
        assert_eq!(HyperLogLog::new(2).precision(), 4); // clamped up
        assert_eq!(HyperLogLog::new(20).precision(), 18); // clamped down
    }

    #[test]
    fn count_unique_approx_hll_floats() {
        let x: Vec<f64> = (0..5000).map(|i| i as f64 * 0.5).collect();
        let e = count_unique_approx_hll(&x, 14);
        let err = (e - 5000.0).abs() / 5000.0;
        assert!(err < 0.05, "float HLL error = {}", err);
    }

    // ── Bloom filter ────────────────────────────────────────────────────

    #[test]
    fn bloom_no_false_negatives() {
        let mut bf = BloomFilter::with_fpr(1000, 0.01);
        for i in 0..1000u64 { bf.insert_u64(i); }
        for i in 0..1000u64 {
            assert!(bf.contains_u64(i), "false negative at {}", i);
        }
    }

    #[test]
    fn bloom_false_positive_rate_reasonable() {
        let mut bf = BloomFilter::with_fpr(1000, 0.01);
        for i in 0..1000u64 { bf.insert_u64(i); }
        let mut fp = 0;
        for i in 10_000..20_000u64 {
            if bf.contains_u64(i) { fp += 1; }
        }
        let rate = fp as f64 / 10_000.0;
        assert!(rate < 0.05, "Bloom FPR = {} (target 0.01)", rate);
    }

    #[test]
    fn bloom_union_monotone() {
        let mut a = BloomFilter::new(1024, 4);
        let mut b = BloomFilter::new(1024, 4);
        a.insert_u64(7);
        b.insert_u64(42);
        let popcount_before = a.popcount();
        a.union(&b);
        assert!(a.popcount() >= popcount_before);
        assert!(a.contains_u64(7));
        assert!(a.contains_u64(42));
    }

    #[test]
    fn bloom_estimate_n() {
        let mut bf = BloomFilter::with_fpr(5000, 0.01);
        for i in 0..5000u64 { bf.insert_u64(i); }
        let est = bf.estimate_n();
        assert!((est - 5000.0).abs() / 5000.0 < 0.1,
            "Bloom cardinality estimate = {}", est);
    }

    #[test]
    fn count_bloom_filter_members_helper() {
        let mut bf = BloomFilter::new(4096, 4);
        for v in [1.0_f64, 2.0, 3.0, 4.0] { bf.insert_f64(v); }
        let probe = vec![1.0_f64, 2.0, 99.0];
        let c = count_bloom_filter_members(&bf, &probe);
        assert!(c >= 2, "expected ≥2, got {}", c);
    }

    // ── Count-Min Sketch ────────────────────────────────────────────────

    #[test]
    fn cms_never_under_estimates() {
        let mut cms = CountMinSketch::with_epsilon_delta(0.01, 0.01);
        for i in 0..1000u64 {
            let freq = (i % 10) + 1; // heavy hitters
            cms.add_u64(i, freq);
        }
        for i in 0..1000u64 {
            let true_freq = (i % 10) + 1;
            let est = cms.estimate_u64(i);
            assert!(est >= true_freq, "CMS under-estimated at {}", i);
        }
    }

    #[test]
    fn cms_heavy_hitter_accurate() {
        let mut cms = CountMinSketch::with_epsilon_delta(0.001, 0.001);
        // Insert key 42 many times, surrounded by noise
        for _ in 0..10_000 { cms.add_u64(42, 1); }
        for i in 100..200u64 { cms.add_u64(i, 1); }
        let est = cms.estimate_u64(42);
        // Should be close to 10_000 with small additive noise.
        assert!(est >= 10_000);
        assert!(est < 11_000, "CMS heavy-hitter estimate drifted too high: {}", est);
    }

    #[test]
    fn cms_merge_preserves_counts() {
        // Build two sketches with identical seeds via `clone()` so `merge`
        // sees the same hash functions.
        let mut a = CountMinSketch::new(256, 4);
        let mut b = a.clone();
        a.add_u64(1, 5);
        b.add_u64(1, 3);
        a.merge(&b);
        assert_eq!(a.estimate_u64(1), 8);
    }

    // ── SpaceSaving top-k ───────────────────────────────────────────────

    #[test]
    fn topk_contains_heavy_hitters() {
        // SpaceSaving guarantees: items with count > N/k are in the output.
        // Here N = 500, k = 3, so N/k ≈ 167. Keys 1 (300) and 2 (200) exceed it;
        // key 3 (100) does not, so the guarantee covers only 1 and 2.
        let mut tk = TopKCounter::new(3);
        for _ in 0..300 { tk.add(1, 1); }
        for _ in 0..200 { tk.add(2, 1); }
        for _ in 0..100 { tk.add(3, 1); }
        for _ in 0..50 { tk.add(4, 1); }
        for _ in 0..50 { tk.add(5, 1); }
        let top = tk.top();
        let keys: Vec<u64> = top.iter().map(|&(k, _, _)| k).collect();
        assert!(keys.contains(&1), "heavy hitter 1 must be in top-k");
        assert!(keys.contains(&2), "heavy hitter 2 must be in top-k");
    }

    #[test]
    fn topk_estimate_is_overestimate() {
        let mut tk = TopKCounter::new(2);
        for _ in 0..10 { tk.add(1, 1); }
        for _ in 0..8 { tk.add(2, 1); }
        // A third key evicts something — estimate will overcount but never undercount
        tk.add(3, 1);
        assert!(tk.estimate(1) >= 10 || tk.estimate(1) == 0);
        assert!(tk.estimate(2) >= 8 || tk.estimate(2) == 0);
    }

    #[test]
    fn topk_small_stream() {
        let mut tk = TopKCounter::new(5);
        for v in [1u64, 2, 3, 1, 2, 1] { tk.add(v, 1); }
        assert_eq!(tk.estimate(1), 3);
        assert_eq!(tk.estimate(2), 2);
        assert_eq!(tk.estimate(3), 1);
    }
}
