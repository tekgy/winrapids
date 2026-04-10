# Compound Primitives — Scout Analysis
Written: 2026-04-10

## What "Compound" Actually Means Here

The 102-violation audit and the hardcoded-constants audit together describe two different
failure modes that look related but are structurally distinct:

**Type A: Embedded primitives** — a math function is implemented INSIDE another function
when it should be extracted to the global catalog. The consumer reinvented a wheel.

**Type B: Parameter burial** — a primitive IS correctly implemented and global, but
the auto-detection layer above it hardcodes the parameters instead of flowing them
through `using()`. The wheel exists but the steering is disconnected.

These require different fixes. Type A needs extraction. Type B needs `using()` wiring.

## The OLS Convergence: The Best Example of Type A

Seven different OLS implementations across the codebase — tracked in the extraction audit.
But there's a subtlety the audit flags as "compound primitive":

`linear_fit_segment` in `complexity.rs:340` is NOT just a duplicate of `ols_slope`.
It's specialized: x-coordinates are implicit (0, 1, 2, ..., n-1). This means:
- `mean_x = (n-1)/2` — closed form, no sum needed
- `sxx = n*(n-1)*(2n-1)/6 - n*(n-1)^2/4` — also computable in O(1)

This is a legitimate specialization that could be extracted as:
```rust
pub fn ols_slope_indexed(y: &[f64]) -> f64  // x = 0..n-1
pub fn ols_intercept_indexed(y: &[f64]) -> f64
```
...which could then be called by `linear_fit_segment`, DFA, Higuchi, and any other
"OLS on time-indexed data" consumer. That's the compound primitive to extract.

## The Most Critical Type A: `median_from_sorted`

The audit found 6 independent inline copies of:
```rust
if n % 2 == 0 { (sorted[n/2-1] + sorted[n/2]) / 2.0 } else { sorted[n/2] }
```

This is 3 lines, written 6 times. It's the simplest possible extraction. Priority 1.

## The Most Interesting Type A: `binarize_at_median`

`complexity.rs:lempel_ziv_complexity:403–406` binarizes data at the median before
computing LZ complexity. This is a standalone operation:

```rust
pub fn binarize_at_median(data: &[f64]) -> Vec<bool>
pub fn binarize_at_threshold(data: &[f64], threshold: f64) -> Vec<bool>
pub fn symbolize_quantile(data: &[f64], n_symbols: usize) -> Vec<usize>
```

These three form a family of "symbolization primitives" — continuous data → discrete symbols.
They're used by: LZ complexity, permutation entropy (ordinal patterns), transfer entropy,
quantile symbolize (nonparametric), and any information-theoretic computation on continuous data.

The pattern:  `continuous → symbols → entropy/complexity/information`

The symbolization step is a genuine primitive family that doesn't exist as a named module yet.

## The Most Important Type B: tbs_executor hardcoded decisions

The hardcoded-constants audit found that `tbs_executor` makes method-selection decisions
using hardcoded thresholds. The auto-detection layer IS the thing that makes tambear
feel like "a $10M/year quant sitting next to the user." Getting these thresholds
right — and making them tunable — is what makes the auto-detection scientifically sound.

The critical cluster (all in `tbs_executor.rs`):
- `normality_alpha = 0.05` (4 different decision points)
- `vif_threshold = 10.0`  
- `kmo_threshold = 0.5`
- `hopkins_threshold = 0.5`
- `normality_test_n_threshold = 5000`
- `arch_alpha = 0.05`

Fix pattern is uniform:
```rust
// Before (hardcoded):
let x_norm = px > 0.05;

// After (using() wired):
let alpha = using_bag.get_f64("normality_alpha").unwrap_or(0.05);
let x_norm = px > alpha;
```

The `using_bag` is already threaded through `execute()`. It just needs to be consulted.

## The Pattern That Connects Distant Parts: Symbolization

I notice a rhyme between:
- `complexity.rs` symbolizing at median for LZ complexity
- `nonparametric.rs:quantile_symbolize` for entropy
- `information_theory.rs:joint_histogram` for transfer entropy
- `family12_causality_info.rs:quantize_ranks` for Granger in discrete space

All four are doing the same thing: continuous time series → discrete symbol sequence.
They're all one missing primitive family away from sharing a common substrate.

The family would be:
- `symbolize_median(data)` → Vec<bool>  (2 symbols)
- `symbolize_quantile(data, k)` → Vec<usize>  (k symbols, equal-frequency bins)
- `symbolize_uniform(data, k)` → Vec<usize>  (k symbols, equal-width bins)
- `symbolize_ordinal(data, m)` → Vec<usize>  (ordinal patterns, used by perm entropy)

Once this family exists as `pub` in `nonparametric.rs` or a new `symbolize.rs`, every
complexity and information-theory consumer can share the same symbolization step,
and TamSession can cache the symbolized representation across methods.

## What I'd Tell the Pathmaker

The highest ROI extractions, ordered:

1. `median_from_sorted` — 3 lines, 6 consumers, zero complexity. Do it first.

2. `ols_slope_indexed` — 10 lines, 3 consumers (DFA, Higuchi, linear_fit_segment).
   Captures the indexed-x specialization cleanly.

3. Symbolization family — new `symbolize.rs` module with 4 functions. Unblocks
   TamSession sharing across complexity and information theory.

4. `using()` wiring in `tbs_executor` auto-detection decisions — not primitive extraction
   but critical for the using() passthrough principle. The `using_bag` is already there;
   just need to consult it.

5. `numerical_gradient` — extracted from `arma_fit`, moved to `numerical.rs`. Currently
   it's the only numerical differentiation of a black-box function in the codebase and it
   would be useful for optimization, sensitivity analysis, and any MCMC gradient step.

## The Surprising Finding

The fintek bridges (family15, family22, family24) are actually BETTER composed than the
tambear core in some ways:
- `family15` delegates to `tambear::delay_embed`, `tambear::graph::pairwise_dists` etc.
- `family22` and `family24` have their own reimplementations (different API shapes)

The bridges that DON'T delegate are the newer ones (22, 24). The older bridge (15) has
already been through the decomposition discipline. This suggests the extraction work
is converging — new code is learning the pattern.

The shape mismatch between tambear::complexity::CcmResult and family24::CcmResult
is the real structural tension. Tambear returns full spectra; fintek needs fixed columns.
The right architecture: fintek calls tambear then extracts what it needs. Currently it
avoids the call by reimplementing. Once tambear's CCM is on the pub surface (missing
pub use — found in phantom scan), the bridge can delegate.
