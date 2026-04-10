# Findings 6-10: Adversarial Test Suites

**Author**: Adversarial Mathematician
**Date**: 2026-04-01

---

## Finding 6: DBSCAN Border-Point Order Dependence

**Code**: `crates/tambear/src/clustering.rs:394-402`
**Bug**: Border point assignment uses `break` at first core neighbor found by index order, not nearest core neighbor.

### The Problem

```rust
for j in 0..n {
    if is_core[j] && row_i[j] <= epsilon_threshold {
        labels[i] = labels[j];
        break;  // ← FIRST core neighbor, not NEAREST
    }
}
```

If a border point is within epsilon of two core points from different clusters, it gets assigned to whichever has the lower index. Reordering the input data changes cluster assignments.

### Proof Construction

```
Cluster A: points P0=(0,0), P1=(1,0)     [indices 0,1]
Cluster B: points P2=(10,0), P3=(11,0)    [indices 2,3]
Border:    P4=(5,0)                        [index 4]

eps = 6.0, min_pts = 2

P4 is within eps of both P1 (dist=4) and P2 (dist=5).
With this ordering: P4 assigned to cluster A (P0 has lower index).

Reverse the data: P0'=(11,0), P1'=(10,0), P2'=(1,0), P3'=(0,0), P4'=(5,0)
Now P4' is within eps of P1' (dist=5) and P2' (dist=4).
With reversed ordering: P4' assigned to cluster B (P1' has lower index).
```

### Test Vectors

**TV-DBSCAN-ORDER-01: Two equidistant core clusters**
```
points = [(0,0), (1,0), (10,0), (11,0), (5.5,0)]
eps = 6.0, min_pts = 2
Point 4 is equidistant from both clusters.
Run with original and reversed order — labels MUST be same.
```

**TV-DBSCAN-ORDER-02: Border point at intersection**
```
Create two clusters that overlap at one border point.
Verify assignment is to NEAREST core point, not FIRST.
```

**TV-DBSCAN-ORDER-03: Random permutation stability**
```
100 points in 3 clusters. Run DBSCAN on 10 random permutations.
All must produce identical cluster assignments (up to label permutation).
```

### Fix
Replace `break` with distance tracking:
```rust
let mut best_j = None;
let mut best_dist = f64::INFINITY;
for j in 0..n {
    if is_core[j] && row_i[j] <= epsilon_threshold && row_i[j] < best_dist {
        best_dist = row_i[j];
        best_j = Some(j);
    }
}
if let Some(j) = best_j { labels[i] = labels[j]; }
```

---

## Finding 7: GPU atomicAdd Non-Determinism

**Code**: `crates/tambear/src/hash_scatter.rs` (all CUDA kernels)
**Bug**: f64 atomicAdd is non-associative. Different thread scheduling → different bit-level results.

### The Problem

Floating-point addition is not associative: `(a + b) + c != a + (b + c)` in general.
GPU atomicAdd processes threads in arbitrary order. Two runs of the same kernel on the same data can produce different results at the bit level.

### Proof Table

This cannot be proven in Python — it requires running the actual CUDA kernel twice and comparing results. But the MATHEMATICAL proof is:

```
a = 1.0000000000000002  (1 + 1 ULP)
b = 1e-16
c = 1e-16

(a + b) + c = 1.0000000000000004  (accumulate b first)
a + (b + c) = 1.0000000000000002  (b+c rounds to 1e-16, added to a)

Difference: 2 ULP
```

For hash_scatter with 1000 values being atomically added, the accumulated order-dependent error can be 100+ ULPs.

### Test Vectors

**TV-ATOMIC-01: Determinism check**
```
Run scatter_sum on [1.0 + i*1e-15 for i in range(1000)] twice on GPU.
Compare results bit-for-bit.
Expected: MAY differ (this is the bug documentation, not a fix)
```

**TV-ATOMIC-02: Kahan vs naive accumulation**
```
Same data, compare atomicAdd result vs CPU Kahan summation.
Document the maximum discrepancy.
```

### Recommended Fix
- **Document**: This is inherent to GPU atomicAdd. Not a code bug, but users must know.
- **Deterministic mode**: Optional sorted-reduce path for bit-reproducible results (slower).
- **V-column**: Accumulation confidence based on element count and magnitude spread.

---

## Finding 8: SufficientStatistics::mean() Zero-Count Panic

**Code**: `crates/tambear/src/intermediates.rs:274`
**Bug**: `self.sums[g] / self.counts[g]` — if `counts[g]` is 0.0, this divides by zero.

### The Problem

```rust
pub fn mean(&self, g: usize) -> f64 {
    self.sums[g] / self.counts[g]  // counts[g] = 0.0 → 0.0/0.0 = NaN
                                    // sums[g] = x, counts[g] = 0.0 → x/0.0 = Inf
}
```

In f64, `0.0 / 0.0 = NaN` and `x / 0.0 = Inf`. Neither panics in Rust. But:
- NaN propagates silently through all downstream computation
- Inf in mean → variance becomes NaN → all derived statistics are NaN
- The user gets NaN with no indication of WHERE it came from

Also: `variance()` calls `self.sums[g] / n` where n = `self.counts[g]`. If n=0: NaN.

### Test Vectors

**TV-STATS-EMPTY-01: Empty group**
```
Create SufficientStatistics with counts=[10.0, 0.0, 5.0]
mean(1) should be NaN (currently is NaN — acceptable if documented)
variance(1) should be NaN
std(1) should be NaN
MUST NOT: panic or return Inf
```

**TV-STATS-EMPTY-02: All groups empty**
```
counts = [0.0, 0.0, 0.0]
All operations should return NaN without panic.
```

### Recommended Fix
- Return `NaN` explicitly when `counts[g] < 1.0` (or == 0.0)
- Add `#[inline] pub fn is_valid(&self, g: usize) -> bool { self.counts[g] > 0.0 }`

---

## Finding 9: KMeans f32/f64 Mismatch

**Code**: `crates/tambear/src/kmeans.rs`
**Bug**: KMeans uses f32 (GPU scatter kernels) while the rest of tambear uses f64.

### The Problem

All KMeans GPU kernels operate in f32:
- Distance computation: f32
- Centroid accumulation: f32 atomicAdd
- Convergence check: comparison of f32 centroids

But input data comes from f64 pipelines (intermediates.rs, TamSession).
The f32 conversion silently truncates precision.

### Impact

For data with values > 1e7, f32 has only ~1 digit of fractional precision.
For data with values > 1e38, f32 overflows to Inf.

Financial data (prices in hundreds of thousands, volumes in billions) routinely exceeds f32 safe range for accumulation.

### Test Vectors

**TV-KMEANS-F32-01: Large values**
```
points = [[1e7 + i*0.01, 0] for i in range(100)]
k = 2
f64 should separate into two clusters; f32 may merge them (0.01 is below f32 ULP at 1e7)
```

**TV-KMEANS-F32-02: Mixed magnitude**
```
Cluster A: values near 1e-3
Cluster B: values near 1e6
f32 may lose cluster A's internal structure
```

### Recommended Fix
- Use f64 for KMeans computation (matching rest of codebase)
- Or: normalize data to [0,1] range before f32 conversion

---

## Finding 10: DataId Birthday Paradox

**Code**: `crates/tambear/src/intermediates.rs` (DataId)
**Bug**: blake3 truncated to u64. Birthday paradox collision at ~2^32 entries (~4 billion).

### The Problem

`DataId` is `blake3(content)` truncated to u64 (8 bytes). Used as the key for TamSession's sharing registry.

Birthday paradox: for k random values in a space of size N, collision probability > 50% when k > sqrt(N).
- N = 2^64 → collision at k = 2^32 ≈ 4.3 billion entries
- N = 2^128 → collision at k = 2^64 ≈ 1.8 × 10^19

4 billion intermediates is a lot, but with thousands of leaves × hundreds of tickers × 365 days × multiple cadences, it's reachable in a year of production.

### Collision Probability Table

| # intermediates | P(collision) with u64 | P(collision) with u128 |
|-----------------|----------------------|------------------------|
| 1,000 | 2.7e-14 | 1.5e-33 |
| 1,000,000 | 2.7e-8 | 1.5e-27 |
| 1,000,000,000 | 0.026 (2.6%) | 1.5e-21 |
| 4,000,000,000 | **0.58 (58%)** | 1.5e-20 |
| 10,000,000,000 | 0.999 | 1.5e-19 |

### Test Vectors

**TV-DATAID-01: Collision search**
```
Generate random DataIds until collision. Average: 2^32 attempts.
Not practical as a unit test, but the MATH is the proof.
```

**TV-DATAID-02: Near-identical inputs**
```
Verify DataId("data_v1") != DataId("data_v2") — basic correctness.
```

### Recommended Fix
- Expand DataId to u128 (16 bytes). Birthday paradox now at 2^64 entries — unreachable.
- Alternatively: use full blake3 (32 bytes) as HashMap key.
- Cost: 8 extra bytes per intermediate entry. Negligible.

---

## Priority Summary

| Finding | Severity | Impact | Fix Complexity |
|---------|----------|--------|---------------|
| F06: DBSCAN order | MEDIUM | Wrong cluster labels | 5-line change |
| F07: atomicAdd | LOW | Bit-level non-determinism | Document + optional mode |
| F08: zero-count | MEDIUM | Silent NaN propagation | 2-line guard |
| F09: f32/f64 | HIGH | Silent precision loss on financial data | Change dtype or normalize |
| F10: DataId | LOW now, HIGH at scale | Hash collision in sharing | Widen to u128 |
