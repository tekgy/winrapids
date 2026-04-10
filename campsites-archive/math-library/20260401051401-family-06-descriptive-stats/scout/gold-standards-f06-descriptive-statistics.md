# F06 Descriptive Statistics — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load context for the pathmaker on Family 06 (Descriptive Statistics).
Documents: which R/Python library implements which formula, the exact formula used,
the traps to avoid, and numerical validation targets.

---

## Critical Trap: moments::kurtosis() Does NOT Return Excess Kurtosis

```r
library(moments)
moments::kurtosis(x)    # returns m4/m2² (Pearson kurtosis = 3 for normal)
                        # NOT excess kurtosis (= Pearson - 3)
```

This is the most common silent failure in gold standard comparisons.
If you expect excess kurtosis ≈ 0 for normal data and get ≈ 3, this is why.

**Correct excess kurtosis sources**:
- `e1071::kurtosis(x, type=2)` — excess kurtosis, SAS/SPSS convention
- `scipy.stats.kurtosis(x)` — excess kurtosis by default (fisher=True)
- `scipy.stats.kurtosis(x, fisher=False)` — Pearson kurtosis (no -3)

---

## Shape Statistics: Skewness

### The Three Standard Types (Joanes & Gill 1998)

All three are in e1071. The R function `skewness(x, type=...)`.

| Type | Name | Formula | Libraries |
|------|------|---------|-----------|
| 1 | g₁ (biased, Fisher) | m₃ / m₂^(3/2) | `moments::skewness(x)`, `e1071::skewness(x, type=1)` |
| 2 | G₁ (SAS/SPSS/Excel) | g₁ · √(n(n-1)) / (n-2) | `e1071::skewness(x, type=2)`, `scipy.stats.skew(x)` |
| 3 | b₁ (Minitab) | m₃ / m₂^(3/2) · (1 - 1/n)^(3/2) | `e1071::skewness(x, type=3)` (default) |

Where m_k = (1/n) · Σ(xᵢ - x̄)^k (k-th central moment, population convention).

**Answer to navigator's question**: `moments::skewness()` and `e1071::skewness(type=2)` give DIFFERENT values.
- `moments::skewness()` = Type 1 (g₁, biased)
- `e1071::skewness(type=2)` = G₁ (bias-corrected)
- At n=7: G₁ = g₁ · √(7·6)/5 = g₁ · √42/5 ≈ g₁ · 1.2961...
- Difference ≈ 30% at n=7. Converges to 0 as n→∞.

**Default behavior trap**:
- `e1071::skewness()` default is type=3 (b₁, Minitab) — NOT the most common type
- `scipy.stats.skew()` default is Type 2 (G₁, bias=False by default)
- With `scipy.stats.skew(x, bias=True)` you get Type 1 (g₁)

### Pearson's Skewness (non-moment based)

Two variants, both approximations for unimodal distributions:

```
Pearson 1st: (mean - mode) / std
Pearson 2nd: 3 · (mean - median) / std
```

- **NOT in standard R packages** as a named function
- Available via manual computation
- Pearson 2nd is more commonly used (mode estimation is unstable)
- Neither is in `moments` or `e1071` — must compute manually from mean/median/std

### Bowley (Galton) Skewness

Quartile-based, robust to outliers:

```
Bowley = (Q₁ - 2Q₂ + Q₃) / (Q₃ - Q₁)
```

- Range: [-1, +1]
- Exactly 0 for symmetric distributions around median
- Available in Wolfram: `QuartileSkewness[data]`
- R: manual computation from `quantile(x, c(0.25, 0.5, 0.75))`
- Python: manual from `np.quantile(x, [0.25, 0.5, 0.75])`
- Uses order statistics only — outside polynomial MSR, requires QuantileSketch

### L-Moment Ratio Skewness (τ₃)

```
τ₃ = λ₃ / λ₂
```

Where L-moments λ_k are derived from probability-weighted moments.
λ₂ = L-scale (like std but based on expected absolute differences).
λ₃ = third L-moment = E[X(3:3) - 2·X(2:3) + X(1:3)] / 3 (using order statistics).

- R: `lmomco::lmom.ub(x)$ratios[3]` or `lmom::samlmu(x)`
- Python: `lmoments3` package
- More robust than g₁ for heavy-tailed distributions
- Requires full sort of data — outside polynomial MSR

### Medcouple (Brys, Hubert & Struyf 2004)

Most robust skewness measure (breakdown value = 25%):

```
MC = median { h(xᵢ, xⱼ) } over all pairs (xᵢ, xⱼ) where xᵢ ≤ M ≤ xⱼ

h(xᵢ, xⱼ) = [(xⱼ - M) - (M - xᵢ)] / (xⱼ - xᵢ)
           = [xᵢ + xⱼ - 2M] / (xⱼ - xᵢ)

Special cases:
  h(xᵢ, xⱼ) = +1 if xⱼ = M (right ties with median)
  h(xᵢ, xⱼ) = -1 if xᵢ = M (left ties with median)
```

- Range: [-1, +1]
- R: `robustbase::mc(x)` — correct, fast O(n log n) implementation
- Python: `statsmodels.stats.stattools.medcouple_1d(x)` — KNOWN BUG (issue #5395 in statsmodels)
  - The statsmodels implementation has incorrect handling of ties at median
  - Prefer R or verify against R for any test values
- Tambear: requires full sort + O(n) scan; outside polynomial MSR

---

## Shape Statistics: Kurtosis

### The Three Standard Types

| Type | Name | Formula | Library |
|------|------|---------|---------|
| 1 | g₂ (biased excess) | m₄/m₂² - 3 | `e1071::kurtosis(x, type=1)` |
| 2 | G₂ (SAS/SPSS) | (n+1)·g₂·n(n-1)/[(n-2)(n-3)] + 6/(n-3) approx | `e1071::kurtosis(x, type=2)`, `scipy.stats.kurtosis(x)` |
| 3 | b₂-3 (Minitab) | ((1-1/n)·g₂ + ...) | `e1071::kurtosis(x, type=3)` |

**`moments::kurtosis(x)` = m₄/m₂² (NO -3)** ← Pearson kurtosis, NOT excess. See trap above.

The exact bias-correction formula for Type 2 (from Joanes & Gill 1998):
```
G₂ = [(n+1)·g₂ + 6] · (n-1) / [(n-2)·(n-3)]
```

This is what Excel's KURT() function computes.

---

## Quantile Types: R's 9 Methods

R's `quantile(x, probs, type=...)` implements all 9 Hyndman & Fan (1996) methods:

| Type | h | Quantile definition | Used by |
|------|---|--------------------|---------|
| 1 | floor(np) | Empirical CDF inverse (discontinuous) | — |
| 2 | (floor(np)+ceil(np))/2 | Average at discontinuities | — |
| 3 | closest even order stat | Nearest even order | SAS (PROCtile) |
| 4 | np | Linear interp of empirical CDF | — |
| 5 | np + 0.5 | Hazen plotting positions | Maple |
| 6 | (n+1)p | Linear interp, Weibull plotting | Excel PERCENTILE.EXC, Minitab, SPSS |
| 7 | 1 + (n-1)p | Linear interp (default in R) | R default, S-Plus, SAS UNIVARIATE |
| 8 | (n+1/3)p + 1/3 | Median unbiased (recommended) | — |
| 9 | (n+1/4)p + 3/8 | Normal distribution unbiased | — |

**Hyndman's recommendation**: Type 8 (median-unbiased). Best for continuous distributions.
**R default**: Type 7.
**numpy default**: `np.quantile(x, q)` uses linear interpolation (Type 7 equivalent).
**scipy default**: `scipy.stats.scoreatpercentile` also Type 7 equivalent.

For tambear: IQR computation must document which type. Default should match R's Type 7 for compatibility, but expose `quantile_type` parameter.

---

## Package-by-Package Reference

### R: moments package

```r
library(moments)
moments::mean(x)           # plain mean (identical to base mean())
moments::skewness(x)       # g₁ — Type 1, biased
moments::kurtosis(x)       # m₄/m₂² — PEARSON (not excess!) ← TRAP
moments::moment(x, order=3, central=TRUE)  # raw central moment
moments::geary(x)          # Geary's kurtosis test stat
moments::anscombe.test(x)  # d'Agostino kurtosis test
moments::agostino.test(x)  # d'Agostino skewness test
```

### R: e1071 package

```r
library(e1071)
e1071::skewness(x)           # type=3 (b₁, Minitab) by default — surprise!
e1071::skewness(x, type=1)   # g₁ (biased, matches moments::skewness)
e1071::skewness(x, type=2)   # G₁ (SAS/SPSS, bias-corrected) ← USE THIS
e1071::kurtosis(x)           # type=3 by default — excess kurtosis
e1071::kurtosis(x, type=2)   # G₂ (SAS/SPSS, matches scipy.stats.kurtosis)
```

### Python: scipy.stats

```python
from scipy import stats
stats.describe(x)          # returns namedtuple: nobs, minmax, mean, variance, skewness, kurtosis
stats.skew(x)              # G₁ by default (bias=False ≡ type=2)
stats.skew(x, bias=True)   # g₁ (biased, type=1)
stats.kurtosis(x)          # excess kurtosis G₂ by default (fisher=True)
stats.kurtosis(x, fisher=False)  # Pearson kurtosis m₄/m₂² (like moments::kurtosis)
stats.moment(x, moment=3)  # 3rd central moment (population)
```

### Python: numpy

```python
import numpy as np
np.mean(x)           # arithmetic mean
np.std(x)            # population std (ddof=0 default)
np.std(x, ddof=1)    # sample std
np.var(x, ddof=1)    # sample variance
np.quantile(x, 0.25) # first quartile (Type 7 equivalent)
# NO skewness/kurtosis in numpy — use scipy.stats
```

---

## NaN Handling Conventions

| Library | NaN behavior |
|---------|-------------|
| numpy | propagates NaN (returns NaN) unless `np.nanmean()` etc. |
| scipy.stats | propagates NaN by default; `nan_policy='omit'` to skip NaN rows |
| R moments | `na.rm=TRUE` parameter available for most functions |
| R e1071 | same convention as base R: `na.rm=TRUE` |

**Tambear convention**: filter NaN rows before scatter (exclude them from all statistics). Document that NaN rows are excluded from count, consistent with `na.rm=TRUE` R convention.

---

## Numerical Validation Targets

Test vector: `x = [3.1, 1.4, 1.5, 9.2, 6.5, 3.5, 8.9]` (n=7)

```
mean = 34.1/7 = 4.8714285714...
```

**Verify in R before hardcoding in tests:**
```r
x <- c(3.1, 1.4, 1.5, 9.2, 6.5, 3.5, 8.9)
cat("mean:", mean(x), "\n")
cat("var_sample:", var(x), "\n")
cat("sd:", sd(x), "\n")
cat("skew_g1_moments:", moments::skewness(x), "\n")
cat("skew_G1_e1071:", e1071::skewness(x, type=2), "\n")
cat("kurtosis_excess:", e1071::kurtosis(x, type=2), "\n")
cat("kurtosis_pearson_trap:", moments::kurtosis(x), "\n")  # off by 3
```

**Known relationship**: G₁ = g₁ · √(n(n-1))/(n-2). At n=7: multiplier = √42/5 ≈ 1.2961...

**Expected approximate values** (need R confirmation):
- var (population, ddof=0): ≈ 9.436
- var (sample, ddof=1): ≈ 11.009
- The navigator's document says var_sample ≈ 9.447619 — discrepancy; verify in R before using.

For large-offset adversarial test (centering canary):
```r
x_big <- c(1e4, 1e4+1, 1e4+2, 1e4+3, 1e4+4, 1e4+5)
# Expected: var ≈ 3.5, skew = 0, kurtosis_excess = -1.2 (uniform)
# Naive formula (sum_sq/n - mean²) breaks at ~1e8 for f64
# But breaks at ~1e4 for kurtosis via naive 4th moment accumulation
```

---

## Tambear Decomposition Summary

### One GPU Pass → All Polynomial Statistics

```rust
let phi_exprs = [
    "v - r",                                         // sum1: Σ(x-ref)
    "(v - r) * (v - r)",                             // sum2: Σ(x-ref)²
    "(v - r) * (v - r) * (v - r)",                   // sum3: Σ(x-ref)³
    "(v - r) * (v - r) * (v - r) * (v - r)",         // sum4: Σ(x-ref)⁴
    "log(v)",                                        // for geometric mean
    "1.0 / v",                                       // for harmonic mean
    "1.0",                                           // count
];
// 7 phi expressions, one GPU pass, all F06 stats except quantiles
```

### Statistics Outside Polynomial MSR (Require Additional Passes/Structures)

| Statistic | Method | Reason |
|-----------|--------|--------|
| Median | QuantileSketch | Not a power sum |
| IQR, Bowley skewness | QuantileSketch | Order statistics |
| L-moment ratio | Full sort | Requires order statistics |
| Medcouple | Full sort + O(n) scan | Non-linear kernel median |
| Mode | Histogram or full sort | Not a power sum |
| MAD (median absolute deviation) | Two passes: median, then |·-median| scatter | Non-polynomial |
