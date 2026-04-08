# F26 Complexity & Chaos — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 26 (Complexity & Chaos).
F26 is cross-kingdom: Kingdom A (distances, histograms), Kingdom B (prefix scan),
Kingdom C (multiscale sweep). The naturalist identified a new template: **multiscale sweep**.
This document verifies which algorithms use it, and how to avoid reimplementing OLS.

---

## The Multiscale Template (new pattern)

Appears in ~6 of ~12 F26 algorithms:

```
Template:
1. Define log-spaced scale range: s = [s_min, s_max, n_scales]
2. For each scale sₖ: compute statistic f(sₖ) using existing primitives
3. Log-log regression: β = slope of log(f(s)) vs log(s)  ← F10 OLS!
4. Return β as the scaling exponent
```

**The log-log regression IS ordinary OLS on log-transformed values.**
`y = β·x + c` where `y = log(f(s))`, `x = log(s)`.
`β` = the scaling exponent (Hurst, fractal dimension, etc.).

Every implementation of this template should call `linear_regression(log(s), log(f(s)))` from F10 — not reimplement least-squares slope. This is the connection the naturalist identified.

### Python: Gold Standard Libraries for F26

```python
# nolds — numerical nonlinear dynamics
import nolds
nolds.hurst_rs(x)           # Hurst exponent (R/S analysis)
nolds.dfa(x)                # DFA (Detrended Fluctuation Analysis)
nolds.corr_dim(x, emb_dim=2)  # Correlation dimension
nolds.lyap_r(x, emb_dim=2)  # Largest Lyapunov exponent (Rosenstein method)
nolds.sampen(x)              # Sample Entropy

# EntroPy
import entropy
entropy.sample_entropy(x, order=2, metric='chebyshev')
entropy.perm_entropy(x, order=3, delay=1, normalize=True)
entropy.spectral_entropy(x, sf=100, method='welch')
entropy.app_entropy(x, order=2)

# antropy
import antropy as ant
ant.sample_entropy(x)
ant.perm_entropy(x, order=3, delay=1, normalize=True)
ant.app_entropy(x, order=2)
ant.spectral_entropy(x, sf=100, method='welch')
```

---

## Algorithm-by-Algorithm Breakdown

### 1. Hurst Exponent (R/S Analysis)

```
Template: multiscale sweep

For each window size n ∈ log-spaced range:
  R/S(n) = mean over non-overlapping windows of:
    (max(cumsum(x - mean(x))) - min(cumsum(x - mean(x)))) / std(x)
H = slope of log(R/S) vs log(n)  (Hurst exponent)
```

```python
import nolds
H = nolds.hurst_rs(x)  # typical range [0,1]; H=0.5 = Brownian, H>0.5 = trending
```

**Tambear decomposition**:
- `cumsum = accumulate(Prefix, Identity, Add)` — prefix scan
- `max/min = accumulate(Windowed(n), Max/Min)` — windowed extrema
- `std = MomentStats(order=2, Windowed(n))` — windowed variance
- `log-log regression = OLS(log_n, log_rs)` — F10 reuse

**Kingdom**: B (prefix scan) + A (windowed stats) + C (scale sweep).

```r
library(pracma)
hurstexp(x)              # Hurst via R/S analysis
library(longmemo)
# Multiple methods available
```

---

### 2. DFA (Detrended Fluctuation Analysis)

```
Template: multiscale sweep

For each window size n:
  1. Integrate: y(t) = cumsum(x - mean(x))           [prefix scan]
  2. For each non-overlapping window of size n:
       Fit OLS trend to y in window                   [F10 OLS]
       Compute RMS of detrended residuals
  3. F(n) = sqrt(mean of all squared residuals)       [MomentStats]
H = slope of log(F(n)) vs log(n)
```

```python
import nolds
H_dfa = nolds.dfa(x)
# order=1 (default): linear detrending. order=2: quadratic detrending.
```

**Tambear decomposition**:
- Prefix scan: existing `accumulate(Prefix, Identity, Add)`
- Per-window OLS: `accumulate(Segmented(window_boundaries), ...)` — uses Segmented grouping!
  This is the second concrete use case for `Grouping::Segmented` (first was RQA)
- F(n) = RMS = MomentStats(order=2) on residuals

**Critical**: DFA IS the use case that makes Segmented grouping valuable beyond RQA.

```r
library(fractal)
DFA(x, detrend='poly1')
```

---

### 3. Correlation Dimension (Grassberger-Procaccia)

```
Template: multiscale sweep (count pairs at varying radii)

1. Embed: M[i] = [x[i], x[i+1], ..., x[i+m-1]] (delay embedding)
2. For each radius r ∈ log-spaced range:
   C(r) = #{(i,j): ||M[i] - M[j]|| < r} / n²   [threshold count on DistancePairs]
3. d = slope of log(C(r)) vs log(r)              [log-log regression, F10]
```

```python
import nolds
d = nolds.corr_dim(x, emb_dim=2)
```

**Tambear decomposition**:
- Pairwise L2 distances: `TiledEngine` on embedded matrix → DistancePairs
- Threshold count at each r: `accumulate(All, Masked(dist < r), Count)` — FilterJit!
- Log-log OLS: F10

**This is the same DistancePairs computation as DBSCAN.** The fintek chaos scout notes confirmed this — correlation dimension IS DBSCAN's distance matrix computation, with a different extraction (pair count at threshold vs neighborhood density).

```r
library(nonlinearTseries)
rqa <- rqa(time.series=x, embedding.dim=2, lag=1, radius=0.5)
```

---

### 4. Sample Entropy (SampEn)

```
1. Embed: M[i] = [x[i], ..., x[i+m-1]] (m-length templates)
2. B = #{(i,j): L∞(M[i], M[j]) ≤ r} / (n*(n-1))
3. A = #{(i,j): L∞(M[i+1], M[j+1]) ≤ r} / (n*(n-1))
   (same as B but for m+1 templates)
4. SampEn = -log(A/B)
```

```python
import nolds
s = nolds.sampen(x, emb_dim=2)    # default r = 0.2*std

import antropy as ant
s = ant.sample_entropy(x, order=2)  # same algorithm

import entropy
s = entropy.sample_entropy(x, order=2, metric='chebyshev')
```

**Critical**: uses **L∞ (Chebyshev) distance**, not L2. This requires `max|aᵢ - bⱼ|` combine in TiledEngine — the one missing primitive from F01 scout notes.

**Tambear path**:
- L∞ distance matrix: TiledOp with `max|aᵢ - bⱼ|` combine (NOT yet in TiledEngine)
- Threshold count: FilterJit on distance matrix
- Ratio and log: trivial

OR: compute L∞ as element-wise max of per-dimension absolute differences — one GPU kernel.

**If L∞ TiledOp is deferred**: SampEn can be computed via `max(|x-y|, dim)` as a custom WGSL kernel until the general max-combine TiledOp is added.

```r
library(pracma)
sample_entropy(x, edim=2, r=0.2*sd(x))
```

---

### 5. Approximate Entropy (ApEn)

Similar to SampEn but counts self-matches and uses `log` not `-log`. Less rigorous statistically.

```python
import antropy as ant
ap = ant.app_entropy(x, order=2)

import nolds
# nolds doesn't have ApEn — use antropy or entropy package
```

```r
library(pracma)
approx_entropy(x, edim=2, r=0.2*sd(x))
```

**Note**: `entropy.app_entropy` and `antropy.app_entropy` may give slightly different values due to boundary handling. Use antropy as primary oracle (more actively maintained).

---

### 6. Permutation Entropy

```
1. For each window of length m starting at position t:
   pattern[t] = rank order of [x[t], ..., x[t+m-1]]
2. Count frequency of each ordinal pattern (m! possible patterns)
3. H = -Σ p(π) * log(p(π))  [Shannon entropy of pattern distribution]
```

```python
import antropy as ant
h = ant.perm_entropy(x, order=3, delay=1, normalize=True)
# normalize=True: divide by log(m!) to get H ∈ [0,1]

import entropy
h = entropy.perm_entropy(x, order=3, delay=1, normalize=True)
```

**Tambear decomposition**:
- Rank within each window: `argsort` on each window → ordinal patterns
- Pattern to integer (bijection): encode rank permutation as integer key
- Count per pattern: `scatter(ByKey{pattern_id}, Count)` — same as Shannon entropy histogram!
- Shannon entropy: `-Σ p*log(p)` — F25 reuse

**This is histogram entropy (F25) applied to ordinal patterns.** Kingdom A: scatter(ByKey, Count).

```r
library(statcomp)
ordinal_pattern_entropy(x, m=3)
```

---

### 7. Spectral Entropy

```
1. Compute PSD: P(f) via Welch's method (F19 family)
2. Normalize: p(f) = P(f) / Σ P(f)  (probability over frequencies)
3. H = -Σ p(f) * log(p(f))  [Shannon entropy of PSD]
```

```python
import antropy as ant
h = ant.spectral_entropy(x, sf=sampling_freq, method='welch', normalize=True)
```

**Tambear**: this is F19 (FFT/PSD) → F25 (Shannon entropy). Pure composition. No new primitives.

---

### 8. Largest Lyapunov Exponent (Rosenstein method)

```
Template: multiscale sweep

1. Embed with delay: M[i] = [x[i], x[i+τ], ..., x[i+(m-1)τ]]
2. For each point i, find nearest neighbor j (KNN, F01)
3. Divergence: d(t) = ||M[i+t] - M[j+t]||  for t = 0..T
4. Mean divergence: Y(t) = mean(log(d(t)))
5. λ = slope of Y(t) vs t  [log-log regression, F10]
```

```python
import nolds
lmax = nolds.lyap_r(x, emb_dim=2, lag=1)   # Rosenstein's method
```

**Tambear decomposition**:
- KNN on embedded matrix: DistancePairs + row argmin (F01/F20)
- Distance at later times: gather from distance matrix at offset indices
- Mean log divergence: MomentStats(order=1) on log-distances
- Slope regression: F10 OLS

---

### 9. Lempel-Ziv Complexity

```
1. Convert x to binary sequence: b[i] = 1 if x[i] > median(x), else 0
2. Count distinct subsequences via sequential parsing
3. C_LZ = c(n) · log(n) / n  (normalized)
```

Sequential parsing IS a Kingdom B computation — sequential scan maintaining a dictionary.
Not easily parallelizable. CPU-side is appropriate.

```python
import antropy as ant
lz = ant.lziv_complexity(x, normalize=True)

# Or with explicit binarization:
median = np.median(x)
binary = (x > median).astype(int)
lz = ant.lziv_complexity(binary, normalize=True)
```

---

## Library Comparison: antropy vs entropy vs nolds

| Metric | antropy | entropy | nolds |
|--------|---------|---------|-------|
| Sample Entropy | ✓ fast C | ✓ | ✓ |
| Approx Entropy | ✓ | ✓ | — |
| Permutation Entropy | ✓ | ✓ | — |
| Spectral Entropy | ✓ | ✓ | — |
| Hurst R/S | — | — | ✓ |
| DFA | — | — | ✓ |
| Correlation Dimension | — | — | ✓ |
| Lyapunov Exponent | — | — | ✓ |
| Lempel-Ziv | ✓ | — | — |

**Primary oracle**: `nolds` for scaling exponents (Hurst, DFA, corr dim, Lyapunov).
**Primary oracle**: `antropy` for entropy measures (SampEn, PermEn, ApEn, spectral).

Both use scipy/numpy and are reproducible with `random_state`.

---

## The Multiscale Template (reusable chain)

Worth formalizing in tambear's API:

```rust
pub struct MultiscaleSweep {
    pub scales: Vec<f64>,          // log-spaced range [s_min, s_max]
    pub statistic: Box<dyn Fn(f64) -> f64>,  // f(scale) → statistic value
}

impl MultiscaleSweep {
    pub fn scaling_exponent(&self) -> f64 {
        let log_scales = self.scales.iter().map(|s| s.ln()).collect::<Vec<_>>();
        let log_stats = self.scales.iter().map(|s| (self.statistic)(*s).ln()).collect::<Vec<_>>();
        // F10 OLS: slope of log_stats vs log_scales
        ols_slope(&log_scales, &log_stats)
    }
}
```

This captures the pattern explicitly. When the pathmaker sees the structure in F26, they should reach for this rather than reimplementing OLS inside each algorithm.

---

## Infrastructure Gaps for F26

| Gap | Needed by | Severity |
|-----|----------|---------|
| L∞ TiledOp | SampEn, ApEn | Medium — workaround with custom kernel |
| `Grouping::Segmented` | DFA, RQA | Medium — currently `todo!()` |
| FFT primitive (F19) | Spectral entropy | High for spectral — deferred to F19 |
| Delay embedding (CPU) | All chaotic algorithms | Trivial — CPU argsort |

The Segmented grouping gap affects both DFA (F26) and RQA (F26). This is now the THIRD use case for Segmented (after RQA line detection). It should be unblocked before F26 implementation begins.

---

## Validation Targets

```python
import numpy as np
import nolds
import antropy as ant

np.random.seed(42)
n = 1000
# Brownian motion (H ≈ 0.5 for geometric Brownian):
x = np.cumsum(np.random.randn(n))

print("Hurst (R/S):", nolds.hurst_rs(x))          # expect ≈ 0.5 for random walk
print("DFA H:", nolds.dfa(x))                       # expect ≈ 1.5 (DFA gives H+1 for integrated)
print("SampEn:", ant.sample_entropy(x, order=2))
print("PermEn:", ant.perm_entropy(x, order=3, normalize=True))
```
