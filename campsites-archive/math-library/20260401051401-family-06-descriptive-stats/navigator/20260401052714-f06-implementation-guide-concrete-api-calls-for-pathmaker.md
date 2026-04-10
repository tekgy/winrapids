# F06 Implementation Guide: Concrete API Calls

Created: 2026-04-01T05:27:14-05:00
By: navigator

---

## What We're Building

Family 06: Descriptive Statistics. Full accumulate decomposition, from tambear primitives.
No calls to anything external. Every formula from first principles.

**Status of underlying infrastructure** (from codebase read):
- `ScatterJit::scatter_multi_phi()` — working, JIT-compiled, fused multi-output ✓
- `RefCenteredStatsEngine` — stub, implementation is the first task
- `TamSession` + `IntermediateTag` — working, needs MomentStats + ExtremaStats variants added
- `SufficientStatistics(sum, sum_sq, count)` — exists as MomentStats(order=2)

---

## Step 1: Implement RefCenteredStatsEngine

**File**: `crates/tambear/src/stats.rs`

The design doc is already there (read it carefully — it explains the centering math).

**The key call** — this is what the implementation wires to:

```rust
// In RefCenteredStatsEngine::group_stats():
let results = jit.scatter_multi_phi(
    &["v - r", "(v - r) * (v - r)", "1.0"],
    row_to_group,   // keys: which group each row belongs to
    values,         // the data values
    ref_values,     // reference points per group (group mean estimate, or all zeros)
    n_groups,
).unwrap();

// results[0] = centered_sum[g] = Σ(x - ref) per group
// results[1] = centered_sum_sq[g] = Σ(x - ref)² per group
// results[2] = count[g] = n per group

// Post-scatter (host-side):
let mean_g = ref_g + centered_sum[g] / count[g];
let var_g  = centered_sum_sq[g] / count[g] - (centered_sum[g] / count[g]).powi(2);
```

Where to get `ref_values`:
- First call: pass `None` → ScatterJit uses `r = 0.0` for all groups
- Iterative improvement: first pass gives approximate means, second pass uses those as refs
- For GroupStats API: one pass (ref=0) is fine for typical data; two passes needed for large-magnitude data

**Expected output**: `GroupStats { means: Vec<f64>, variances: Vec<f64>, counts: Vec<u32> }`

**NaN handling** (consistent with existing `ReduceOp` convention):
- Filter NaN values before scatter, or pass a masked version
- Document that NaN rows are excluded from all statistics

---

## Step 2: Add MomentStats to intermediates.rs

**File**: `crates/tambear/src/intermediates.rs`

Add AFTER the existing `SufficientStatistics` struct:

```rust
/// Per-group moment statistics: (sum_centered, sum_sq_centered, ..., count, ref_values)
/// up to a specified order. Numerically stable via reference centering.
///
/// `order` indicates the highest power accumulated:
/// - order=2: mean + variance (subsumes SufficientStatistics)
/// - order=3: + skewness (Fisher g₁)
/// - order=4: + kurtosis (excess g₂)
/// - order=8: + moments 5-8
///
/// All sums are centered around ref_values to avoid catastrophic cancellation.
#[derive(Debug, Clone)]
pub struct MomentStats {
    pub order: u8,
    pub n_groups: usize,
    /// Reference values used for centering. ref[g] ≈ group mean.
    pub ref_values: Arc<Vec<f64>>,
    pub counts: Arc<Vec<f64>>,
    pub sum1: Arc<Vec<f64>>,    // Σ(x - ref)
    pub sum2: Arc<Vec<f64>>,    // Σ(x - ref)²
    pub sum3: Option<Arc<Vec<f64>>>,  // Σ(x - ref)³, if order >= 3
    pub sum4: Option<Arc<Vec<f64>>>,  // Σ(x - ref)⁴, if order >= 4
    // sum5..sum8 to be added when needed
}

impl MomentStats {
    /// Mean for group g.
    pub fn mean(&self, g: usize) -> f64 {
        self.ref_values[g] + self.sum1[g] / self.counts[g]
    }

    /// Sample variance (n-1 denominator) for group g.
    pub fn variance(&self, g: usize) -> f64 {
        let n = self.counts[g];
        let m = self.sum1[g] / n;  // centered mean
        self.sum2[g] / (n - 1.0) - m * m * n / (n - 1.0)
        // = (Σ(x-ref)² - (Σ(x-ref))²/n) / (n-1)
    }

    /// Population variance (n denominator) for group g.
    pub fn variance_pop(&self, g: usize) -> f64 {
        let n = self.counts[g];
        let m = self.sum1[g] / n;
        self.sum2[g] / n - m * m
    }

    /// Sample standard deviation for group g.
    pub fn std(&self, g: usize) -> f64 { self.variance(g).sqrt() }

    /// Coefficient of variation (std/mean) for group g.
    pub fn cv(&self, g: usize) -> f64 { self.std(g) / self.mean(g).abs() }

    /// Standard error of the mean for group g.
    pub fn se_mean(&self, g: usize) -> f64 { self.std(g) / self.counts[g].sqrt() }

    /// Fisher's g₁ skewness (bias-uncorrected) for group g.
    /// Returns None if order < 3.
    pub fn skewness_fisher_g1(&self, g: usize) -> Option<f64> {
        let sum3 = self.sum3.as_ref()?;
        let n = self.counts[g];
        let m1 = self.sum1[g] / n;
        let m2 = self.sum2[g] / n;
        let m3 = sum3[g] / n;
        // Third central moment from centered sums (binomial expansion):
        // μ₃ = E[(x-μ)³] where μ is the true mean (not ref)
        // = m3 - 3*m2*m1 + 2*m1³
        let mu3 = m3 - 3.0 * m2 * m1 + 2.0 * m1.powi(3);
        let sigma = self.std(g);
        if sigma == 0.0 { return Some(f64::NAN); }
        Some(mu3 / sigma.powi(3))
    }

    /// Excess kurtosis (Fisher's g₂, bias-uncorrected) for group g.
    /// Returns None if order < 4.
    pub fn kurtosis_excess(&self, g: usize) -> Option<f64> {
        let sum3 = self.sum3.as_ref()?;
        let sum4 = self.sum4.as_ref()?;
        let n = self.counts[g];
        let m1 = self.sum1[g] / n;
        let m2 = self.sum2[g] / n;
        let m3 = sum3[g] / n;
        let m4 = sum4[g] / n;
        // Fourth central moment from centered sums:
        // μ₄ = m4 - 4*m3*m1 + 6*m2*m1² - 3*m1⁴
        let mu4 = m4 - 4.0*m3*m1 + 6.0*m2*m1.powi(2) - 3.0*m1.powi(4);
        let sigma2 = self.variance_pop(g);
        if sigma2 == 0.0 { return Some(f64::NAN); }
        Some(mu4 / sigma2.powi(2) - 3.0)
    }
}
```

**Note on variance formula**: the formula above uses the binomial-expansion approach to compute sample variance from centered sums. Verify with adversarial test cases A and B (large-magnitude data).

---

## Step 3: Add ExtremaStats to intermediates.rs

```rust
/// Per-group extrema: min and max values.
#[derive(Debug, Clone)]
pub struct ExtremaStats {
    pub n_groups: usize,
    pub maxima: Arc<Vec<f64>>,
    pub minima: Arc<Vec<f64>>,
}

impl ExtremaStats {
    pub fn range(&self, g: usize) -> f64 { self.maxima[g] - self.minima[g] }
    pub fn midrange(&self, g: usize) -> f64 { (self.maxima[g] + self.minima[g]) / 2.0 }
}
```

For accumulation: two separate `scatter_single_phi` calls with `phi = "v"` using Max and Min operators. OR (phase 2 optimization): a custom `MinMaxOp` that tracks both in one pass.

---

## Step 4: Extend IntermediateTag enum

In `intermediates.rs`, add to the `IntermediateTag` enum:

```rust
/// Per-group moment statistics up to specified order.
/// MomentStats(order=2) subsumes SufficientStatistics for new code.
MomentStats {
    order: u8,
    data_id: DataId,
    grouping_id: DataId,  // hash of the keys array, or DataId::from_bytes(&[]) for All grouping
},

/// Per-group extrema (min + max).
ExtremaStats {
    data_id: DataId,
    grouping_id: DataId,
},
```

Add `get_moment_stats_min_order()` to `TamSession`:
```rust
pub fn get_moment_stats(&self, min_order: u8, data_id: DataId, grouping_id: DataId)
    -> Option<Arc<MomentStats>>
{
    for order in min_order..=8 {
        let tag = IntermediateTag::MomentStats { order, data_id, grouping_id };
        if let Some(s) = self.get::<MomentStats>(&tag) { return Some(s); }
    }
    None
}
```

---

## Step 5: Wire Multi-Moment Accumulation

For order=4 (skewness + kurtosis), extend the scatter_multi_phi call:

```rust
let phi_exprs = [
    "v - r",                                  // sum1: Σ(x-ref)
    "(v - r) * (v - r)",                      // sum2: Σ(x-ref)²
    "(v - r) * (v - r) * (v - r)",            // sum3: Σ(x-ref)³
    "(v - r) * (v - r) * (v - r) * (v - r)", // sum4: Σ(x-ref)⁴
    "1.0",                                    // count
];
// One GPU pass → all moment sums, order 1-4
```

This is the crucial insight: `scatter_multi_phi` takes a SLICE of expressions — add more expressions, same GPU pass. No overhead per extra moment (data is read once, compute cost is tiny multiplies).

---

## Step 6: Additional F06 Accumulate Expressions

For geometric/harmonic/log-based stats, additional scatter_phi calls (separate passes):

```rust
// Geometric mean prep: Σlog(x) → exp(sum/n)
let log_sum = jit.scatter_phi("log(v)", keys, values, None, n_groups)?;
let geo_mean_g = (log_sum[g] / count[g]).exp();

// Harmonic mean: n / Σ(1/x)
let recip_sum = jit.scatter_phi("1.0 / v", keys, values, None, n_groups)?;
let harm_mean_g = count[g] / recip_sum[g];
```

**Fusion opportunity** (Phase 2): these can all be fused into one multi_phi call:
`["v - r", "(v-r)*(v-r)", "(v-r)*(v-r)*(v-r)", "(v-r)*(v-r)*(v-r)*(v-r)", "log(v)", "1.0/v", "1.0"]`
= ONE GPU pass for ALL F06 statistics except quantiles.

---

## Step 7: Descriptive Stats API Surface

New file: `crates/tambear/src/descriptive_stats.rs`

Public functions, each taking MomentStats/ExtremaStats/QuantileSketch as input:

```rust
pub fn mean(stats: &MomentStats, g: usize) -> f64
pub fn variance(stats: &MomentStats, g: usize) -> f64
pub fn std(stats: &MomentStats, g: usize) -> f64
pub fn skewness_fisher(stats: &MomentStats, g: usize) -> Option<f64>
pub fn skewness_pearson(stats: &MomentStats, extrema: &ExtremaStats, g: usize) -> Option<f64>
pub fn skewness_bowley(quantiles: &QuantileSketch, g: usize) -> f64
pub fn kurtosis_excess(stats: &MomentStats, g: usize) -> Option<f64>
pub fn cv(stats: &MomentStats, g: usize) -> f64
pub fn gini_coeff(stats: &MomentStats, g: usize) -> f64  // approx from moments
pub fn geometric_mean(log_sum: f64, n: f64) -> f64       // from log_sum accumulate
pub fn harmonic_mean(recip_sum: f64, n: f64) -> f64       // from recip_sum accumulate
pub fn mad(values: &[f64], median: f64) -> f64            // median absolute deviation
```

---

## Validation Targets (from scientist)

For `x = [3.1, 1.4, 1.5, 9.2, 6.5, 3.5, 8.9]`:
- mean: 4.871428...
- var (sample): 9.447619...
- std: 3.073860...
- R `moments::skewness(x)` = g₁ (Fisher, biased)
- R `e1071::skewness(x, type=2)` = G₁ (bias-corrected)

Wait for scientist to compute exact values before coding the validation tests.

---

## Edge Cases to Handle

From adversarial brief:
- n=0: all stats return NaN (not panic)
- n=1: mean = x, variance = NaN (0/0 by convention), skewness = NaN
- n=2: variance = (x₂-x₁)²/2, skewness = NaN (n=2: 0 df for 3rd moment)
- all-same: variance=0, skewness=NaN (0/0), kurtosis=NaN
- [1e8, 1e8+1, ...]: variance should be ~833.5 — test this explicitly
- Inf in input: document behavior, don't panic

---

## Build Order

1. `RefCenteredStatsEngine` (stats.rs) — unblocks everything
2. `MomentStats` + `ExtremaStats` + `IntermediateTag` extension (intermediates.rs)
3. `scatter_multi_phi` call for order=4 (in a stats_accumulate.rs or similar)
4. `descriptive_stats.rs` — derive methods as thin wrappers over MomentStats
5. Tests (unit + adversarial vectors)
6. Lab notebook (wait for scientist gold standard values)
