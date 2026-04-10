# TamSession Extension: MomentStats and ExtremaStats Design

Created: 2026-04-01T05:20:36-05:00
By: navigator

---

## What Already Exists (from reading the codebase)

`intermediates.rs` is further along than I expected:

- `TamSession`: working HashMap-based intermediate registry
- `SufficientStatistics`: `(sums, sum_sqs, counts)` — this IS MomentStats(order=2)
- `DistanceMatrix` + `Metric`: fully working with compatibility checks
- `IntermediateTag`: enum with 7 variants already

`accumulate.rs`: Grouping enum dispatches to real primitives — All, ByKey, Masked, Tiled work. Prefix and Segmented are `todo!()` stubs.

`stats.rs`: `RefCenteredStatsEngine` is documented but empty. `new()` returns Err.

---

## Design Decisions

### D1: Extend, Don't Replace SufficientStatistics

`SufficientStatistics` is `(sum, sum_sq, count)` — MomentStats(order=2) with different field names. We have two options:

**Option A: Rename SufficientStatistics → MomentStats, add higher-order fields**

```rust
pub struct MomentStats {
    pub order: u8,          // highest order accumulated (2, 3, 4, ..., 8)
    pub n_groups: usize,
    pub counts: Arc<Vec<f64>>,
    pub sum1: Arc<Vec<f64>>,   // Σx   (was: sums)
    pub sum2: Arc<Vec<f64>>,   // Σx²  (was: sum_sqs)
    pub sum3: Option<Arc<Vec<f64>>>,  // Σx³  (order ≥ 3)
    pub sum4: Option<Arc<Vec<f64>>>,  // Σx⁴  (order ≥ 4)
    // ... sum5..sum8 similarly
}
```

This breaks backward compat for `SufficientStatistics` users.

**Option B: New MomentStats type, SufficientStatistics unchanged**

Keep `SufficientStatistics` as-is (it's used in 3 places). Add `MomentStats` as a new type. The compiler treats them as separate intermediates. When code asks for MomentStats(order=2), the compiler can satisfy it from either a cached MomentStats(order=2+) OR from SufficientStatistics + DataId match.

**Decision: Option B.** No backward-compat hacks, no renaming. Per the project principle. Code that uses `SufficientStatistics` keeps working. New code uses `MomentStats`. Eventually we may deprecate `SufficientStatistics` but that's not our problem now.

### D2: MomentStats is Always Centered

Per RefCenteredStatsOp design: accumulated values are centered around group means to avoid catastrophic cancellation. This means:

```rust
pub struct MomentStats {
    pub order: u8,
    pub n_groups: usize,
    /// Reference values used for centering. ref_g ≈ group mean (or 0 if unknown).
    pub ref_values: Arc<Vec<f64>>,
    pub counts: Arc<Vec<f64>>,
    /// Σ(x - ref)  for each group
    pub sum1_centered: Arc<Vec<f64>>,
    /// Σ(x - ref)²  for each group
    pub sum2_centered: Arc<Vec<f64>>,
    /// Σ(x - ref)³  (if order ≥ 3)
    pub sum3_centered: Option<Arc<Vec<f64>>>,
    /// Σ(x - ref)⁴  (if order ≥ 4)
    pub sum4_centered: Option<Arc<Vec<f64>>>,
}
```

Derived statistics use uncentered formulas but computed via re-centering:

```
mean_g = ref_g + sum1_centered[g] / counts[g]
var_g  = sum2_centered[g] / counts[g] - (sum1_centered[g] / counts[g])²
```

For higher moments: the centered moments relate to raw moments via binomial expansion.
For skewness (third central moment μ₃):
```
μ₃ = E[(x - μ)³]
   = sum3_centered/n - 3*(sum2_centered/n)*(sum1_centered/n) + 2*(sum1_centered/n)³
```

This is more complex than the direct formula but numerically stable for all magnitudes.

### D3: ExtremaStats is Separate

Min/max don't benefit from centering and use a different operator (Min/Max, not Add). Keep them as a separate type to maintain the accumulate-per-operator invariant:

```rust
pub struct ExtremaStats {
    pub n_groups: usize,
    pub maxima: Arc<Vec<f64>>,  // max per group
    pub minima: Arc<Vec<f64>>,  // min per group
}
```

`IntermediateTag::ExtremaStats { data_id, grouping_id }` — new variant.

The MinMax fused operator (proposed in MSR doc) is a good optimization but can wait. For now: two separate `accumulate(ByKey, identity, Max)` and `accumulate(ByKey, identity, Min)` passes = two ~100μs dispatches. Not ideal but correct. The fused MinMax operator (single 100μs dispatch for both) is a follow-on optimization.

### D4: IntermediateTag Extension

Add to `IntermediateTag` enum in `intermediates.rs`:

```rust
/// Per-group moment statistics (sum, sum_sq, ..., sum^order).
/// The `order` field indicates highest power accumulated.
/// MomentStats(order=2) subsumes SufficientStatistics for matching purposes.
MomentStats {
    order: u8,
    data_id: DataId,
    grouping_id: DataId,
},

/// Per-group extrema (min, max).
ExtremaStats {
    data_id: DataId,
    grouping_id: DataId,
},
```

Tag matching rule for MomentStats: `MomentStats { order: 4, .. }` satisfies a request for `MomentStats { order: 2, .. }` if data_id and grouping_id match. The session needs a `get_at_least_order` query:

```rust
pub fn get_moment_stats_at_order(
    &self,
    min_order: u8,
    data_id: DataId,
    grouping_id: DataId
) -> Option<Arc<MomentStats>> {
    // Try to find any MomentStats(order >= min_order) for this data
    for order in min_order..=8 {
        let tag = IntermediateTag::MomentStats { order, data_id, grouping_id };
        if let Some(stats) = self.get::<MomentStats>(&tag) {
            return Some(stats);
        }
    }
    None
}
```

This is the "subsumption" query that lets algorithms asking for order=2 find a cached order=4. The key insight: if we computed higher moments, those automatically satisfy lower-order requests.

---

## Implementation Plan for pathmaker

### Step 1: Implement RefCenteredStatsEngine in stats.rs

Wire to `ScatterJit::scatter_multi_phi` (exists in `scatter_jit.rs`):
- Scatter `centered_value = value - ref_g` and `centered_sq = centered * centered` atomically
- Post-scatter: derive mean and variance
- This unlocks MomentStats(order=2) from the group-stats use case

### Step 2: Add MomentStats struct + derive methods to intermediates.rs

```rust
pub struct MomentStats {
    pub order: u8,
    pub n_groups: usize,
    pub ref_values: Arc<Vec<f64>>,
    pub counts: Arc<Vec<f64>>,
    pub sum1: Arc<Vec<f64>>,
    pub sum2: Arc<Vec<f64>>,
    pub sum3: Option<Arc<Vec<f64>>>,  // order >= 3
    pub sum4: Option<Arc<Vec<f64>>>,  // order >= 4
}
```

Derive methods:
```rust
impl MomentStats {
    pub fn mean(&self, g: usize) -> f64 { ... }
    pub fn variance(&self, g: usize) -> f64 { ... }
    pub fn std(&self, g: usize) -> f64 { ... }
    pub fn skewness_fisher(&self, g: usize) -> Option<f64> { ... } // requires order >= 3
    pub fn kurtosis_excess(&self, g: usize) -> Option<f64> { ... } // requires order >= 4
    pub fn cv(&self, g: usize) -> f64 { std / mean }
    pub fn se_mean(&self, g: usize) -> f64 { std / sqrt(n) }
}
```

### Step 3: Add ExtremaStats to intermediates.rs

Simple: `(maxima, minima)` per group, derive methods for range, midrange.

### Step 4: Add MomentStats + ExtremaStats to IntermediateTag

Extend the enum. Add `get_moment_stats_at_order()` to TamSession.

### Step 5: Wire MomentStats accumulation through the accumulate dispatch layer

In `accumulate.rs`: new `AggregateExpr` enum variant for multi-moment:

```rust
pub enum AggregateExpr {
    Sum,
    SumOfSquares,
    MultiMoment { order: u8 },  // accumulates x, x², ..., x^order simultaneously
    MinMax,
}
```

---

## Open Questions

1. Should MomentStats use centered or raw moments internally? Centered is numerically stable but the derive methods are more complex (binomial expansion). Raw is simpler to derive from but loses precision for large-magnitude data. **My recommendation: centered internally, centring is transparent to consumers.**

2. For the `All` grouping (global stats), grouping_id = DataId::from_i32(&[0]) or a special sentinel? Need a consistent convention.

3. When pathmaker implements bias-corrected skewness (G₁ vs g₁), should the derive method return both? Or should there be separate `skewness_biased()` and `skewness_unbiased()` methods? The answer affects how scientist validates against R (which uses different corrections in different packages).

4. At what order should we STOP accumulating? sum8 is needed for 8th moment. But sum8 for large values is astronomically large. At f64, x=1000 gives x⁸ = 10²⁴ — within f64 range (max ~1.8×10³⁰⁸) but losing precision from underrepresentation of mantissa. Centering helps but doesn't fully solve for sum8. Does this matter in practice? (Moments 5-8 are rarely used in practice — but the methodology says "all flavors".)
