# Descriptive Statistics — Gold Standard Scout Report

**What this covers**: Central tendency, dispersion, shape (skewness/kurtosis), quantiles.
Family 06 from the landscape. Foundational — used by EVERY other family for validation.
**The headline**: Moments 1-8 all come from ONE accumulate pass. All 6 skewness/kurtosis
flavors (3 skewness types × 2 kurtosis types) derive from the same 5 sufficient stats.

---

## What Gold Standard Says EXISTS

### R's stats package

- `mean()`, `var()`, `sd()`, `median()`, `range()`, `IQR()`, `quantile(type=1..9)`
- No built-in skewness/kurtosis — delegated to `e1071`, `moments`, `psych`

### Python scipy/numpy

- `scipy.stats.describe()` — gives n, min, max, mean, variance, skewness, kurtosis in one call
- `scipy.stats.skew()`, `scipy.stats.kurtosis()` — bias-corrected (Type 2) by default
- `numpy.percentile()`, `numpy.quantile()` — 4 interpolation methods (linear/lower/upper/midpoint/nearest)

### R e1071 package

- `skewness(x, type=1|2|3)` — all three types
- `kurtosis(x, type=1|2|3)` — all three types

### R moments package

- `skewness()` — Type 1 only (classical)
- `kurtosis()` — Pearson's (g2 + 3, NOT excess kurtosis)
- `all.moments(x, order.max=k)` — moments 1..k in one call

---

## The Three Types of Skewness

All three come from the SAME four sufficient statistics: `{n, Σx, Σx², Σx³}`.

Let:
```
n = count
mean = Σx / n
m2 = Σ(x - mean)² / n   (2nd central moment, population)
m3 = Σ(x - mean)³ / n   (3rd central moment, population)
s² = m2 * n/(n-1)        (sample variance = unbiased)
s = sqrt(s²)             (sample std dev)
```

Then:
```
Type 1 (classical, older textbooks, "moments" package):
  g1 = m3 / m2^(3/2)

Type 2 (Fisher's, SAS, SPSS, scipy default):
  G1 = g1 * sqrt(n*(n-1)) / (n-2)
     = g1 * sqrt(n*(n-1)) / (n-2)

Type 3 (Minitab, BMDP, e1071 default):
  b1 = g1 * ((n-1)/n)^(3/2)
     = m3 / s^3                  ← uses sample std dev instead of population std dev
```

**Key insight**: `G1 = f(g1, n)` and `b1 = f(g1, n)`. Compute g1 once, apply scalar
adjustment for the desired type. The sufficient stats are identical.

**Central moment computation from raw moments**:
```
m3 = (Σx³ - 3·mean·Σx² + 2·mean³·n) / n
   = Σx³/n - 3·mean·(Σx²/n) + 2·mean³
```
→ computable from `{Σx, Σx², Σx³, n}` without centering the data first.

---

## The Three Types of Kurtosis

All from `{n, Σx, Σx², Σx³, Σx⁴}` (5 sufficient stats).

Let:
```
m4 = Σ(x - mean)⁴ / n   (4th central moment, population)
```

Then:
```
Type 1 (classical, excess kurtosis, "moments" package: returns m4/m2² NOT excess):
  g2 = m4 / m2² - 3

Type 2 (SAS, SPSS, scipy default):
  G2 = ((n+1)*g2 + 6) * (n-1) / ((n-2)*(n-3))
     ← only unbiased type under normality

Type 3 (Minitab, BMDP):
  b2 = m4/s⁴ - 3
     = (g2 + 3) * (1 - 1/n)² - 3
```

**TRAP**: The `moments` package `kurtosis()` function returns Pearson's kurtosis = `m4/m2²`
(NO subtraction of 3). This is NOT excess kurtosis. Different from all three types above.
When comparing against `moments::kurtosis`, don't subtract 3.

**Central 4th moment from raw moments**:
```
m4 = (Σx⁴ - 4·mean·Σx³ + 6·mean²·Σx² - 3·mean⁴·n) / n
```
→ computable from `{Σx, Σx², Σx³, Σx⁴, n}`.

---

## The 9 Quantile Methods (Hyndman & Fan 1996)

Gold standard: R `quantile(x, p, type=1..9)` and `quantile` R package.

All nine methods compute Q(p) from order statistics x_(1) ≤ x_(2) ≤ ... ≤ x_(n).
Methods 1-3 are discontinuous (discrete); methods 4-9 are continuous (interpolating).

For methods 4-9: `Q(p) = (1 - γ)·x_(j) + γ·x_(j+1)`

where j and γ depend on the method:

```
Type 1: Inverse CDF (no interp). j = ceil(n·p). Lower empirical quantile.
Type 2: Average of Type 1 and Type 3. Midpoint at steps.
Type 3: Nearest even order statistic. j = round(n·p) (rounds to even on ties).
Type 4: p(k) = k/n.            γ = np - j
Type 5: p(k) = (k-0.5)/n.      γ = np + 0.5 - j       ← hydrologists' choice
Type 6: p(k) = k/(n+1).        γ = (n+1)p - j          ← Minitab, SPSS
Type 7: p(k) = (k-1)/(n-1).    γ = (n-1)p - j + 1      ← R default, S, Excel
Type 8: p(k) = (k-1/3)/(n+1/3). γ = (n+1/3)p + 1/3 - j ← Hyndman's recommendation
Type 9: p(k) = (k-3/8)/(n+1/4). γ = (n+1/4)p + 3/8 - j ← approximately unbiased for normal
```

**R default is Type 7. scipy/numpy default is linear (= Type 7). Excel is Type 7.**
Hyndman recommends Type 8 for most uses.

### Tambear quantile strategy

Quantiles require sorted order statistics — the ONE place tambear needs to sort.
This is `sort_values` semantics, not `groupby` semantics. Legitimate sort.

Two approaches:
1. **Sort the column first, then index**: `O(n log n)` + `O(1)` per quantile
2. **Selection/introselect**: `O(n)` per quantile, `O(kn)` for k quantiles

For the full family (percentiles, quartiles, deciles, all-types), sorting once and
indexing is cheaper than selection for large k.

**GPU sort strategy**: CUB `DeviceRadixSort` for GPU sort → then GPU gather at indices.
The gather pattern fits naturally into `GatherOp`.

---

## Measures of Central Tendency

### Geometric mean

`exp(Σlog(x) / n)` — needs `{Σlog(x), n}`. New sufficient stat: `Σlog(x)`.
**Edge case**: any non-positive value → undefined or NaN.

### Harmonic mean

`n / Σ(1/x)` — needs `{Σ(1/x), n}`. New sufficient stat: `Σ(1/x)`.
**Edge case**: any zero → undefined.

### Trimmed mean (α-trimmed)

`mean(x without bottom α and top α fraction)` — needs sorted order stats.
Different from winsorized mean (clamps extremes instead of removing).

### Weighted mean

`Σ(w·x) / Σw` — needs `{Σwx, Σw}`.

---

## Measures of Dispersion

### Coefficient of Variation (CV)

`std / mean` — from `{Σx, Σx², n}`. Dimensionless.

### Mean Absolute Deviation (MAD)

Two variants — easy to confuse:
1. Mean of `|x - mean|`: `Σ|x - mean| / n` — not a standard sufficient stat
2. Median of `|x - median|`: more robust, widely used in robust stats
   → requires two sort/median passes

Sklearn uses definition (1) sometimes, R uses definition (2) in `mad()` by default
(with scale factor `1/0.6745` to make it consistent with σ under normality).

**R's `mad()` default**:
```r
mad(x) = 1.4826 * median(|x - median(x)|)
```
The factor 1.4826 ≈ 1/Φ⁻¹(0.75) is the consistency constant for normal distributions.

### IQR (Interquartile Range)

`Q(0.75) - Q(0.25)` — uses Q type 7 by default in R.

### Gini Coefficient

`(2/n²) · Σ_i Σ_j |x_i - x_j| / (2·mean)` = `1/(n·mean) · Σ_i (2i - n - 1)·x_(i)`
→ requires sorted order. `O(n log n)` sort + `O(n)` pass.

---

## The Sufficient Stats Master Table

For descriptive statistics, these sufficient stats cover everything except quantiles/sort:

```
n                          — count
Σx                         — for mean, CV, skewness correction
Σx²                        — for variance, std, all shape stats
Σx³                        — for skewness
Σx⁴                        — for kurtosis
max(x)                     — range, min/max
min(x)                     — range
Σ|x - median|              — MAD variant 1 (needs median too)
Σlog(x)                    — geometric mean
Σ(1/x)                     — harmonic mean
Σw                         — weighted stats normalization
Σwx                        — weighted mean
```

These 12 sufficient stats cover: mean, variance, std, CV, skewness (all 3 types), kurtosis
(all 3 types), geometric mean, harmonic mean, weighted mean, range, IQR (sort needed).

One `accumulate(All, [1, x, x², x³, x⁴, max, min, log(x), 1/x], ...)` pass.
This is an 9-expression phi — `scatter_multi_phi` directly.

**Connection to the 11-field fintek MSR**: The fintek MSR has `{n, Σp, Σp², max, min, Σsz,
Σ(p·sz), Σr, Σr², Σr³, Σr⁴}`. The shape stats (skewness, kurtosis of returns) are
ALREADY embedded in `{Σr, Σr², Σr³, Σr⁴}`. Descriptive statistics of returns is free.

---

## Edge Cases Gold Standard Handles

From `scipy.stats.describe()` and R `base::summary()`:

- **n=1**: variance = NaN (can't compute with n-1 denominator), skewness = NaN
- **n=2**: kurtosis = NaN (requires n≥4 for Type 2 G2)
- **All identical**: variance=0, CV=0, skewness=0, kurtosis=0 (or NaN for some types)
- **NaN values**: propagate by default in most functions; `na.rm=TRUE` in R, `nan_policy` in scipy
- **Inf values**: propagate; explicitly documented as undefined for skewness/kurtosis
- **Empty input**: return NaN (scipy) or error (R)

Minimum n requirements by metric:
```
mean, median:     n ≥ 1
variance:         n ≥ 2 (sample); n ≥ 1 (population)
skewness Type 1:  n ≥ 3 (m2=0 undefined for n<3)
skewness Type 2:  n ≥ 3 (n-2 in denominator)
kurtosis Type 1:  n ≥ 4 (m2=0 undefined)
kurtosis Type 2:  n ≥ 4 (n-2, n-3 in denominator)
quantiles:        n ≥ 1 (Type 1); n ≥ 2 (Types 4-9 interpolation)
```

---

## What Tambear Can Do That Gold Standard Cannot

### 1. Streaming / one-pass sufficient stats

R/Python/scipy all materialize the entire array before computing anything.
`scipy.stats.describe()` makes multiple passes (sort for median, pass for moments).

Tambear's accumulate infrastructure enables TRUE one-pass streaming:
- Single GPU pass → {n, Σx, Σx², Σx³, Σx⁴} accumulated
- CPU post-processing → ALL moment-based stats

For datasets that don't fit in memory, this is the only viable approach.

### 2. Grouped descriptive statistics for FREE

`accumulate(data, ByKey{labels, K}, [1, x, x², x³, x⁴], Add)`
→ per-group {n_k, Σx_k, Σx_k², Σx_k³, Σx_k⁴} simultaneously
→ compute skewness/kurtosis for EVERY group in ONE GPU pass

R's `tapply(x, groups, skewness)` makes K separate passes. This is K× speedup.

### 3. Cross-column correlation via sufficient stats

The 11-field MSR includes `Σ(p·sz)` — a cross-column product. The MSR infrastructure
naturally extends to `{Σx·y, Σx·y²}` etc. for cross-moment computation (Pearson r,
partial correlations) as part of the same accumulate pass.

---

## Build Order Recommendation for Family 06

1. **Core moments** (mean, variance, std, skewness g1, kurtosis g2): one accumulate pass
   → demonstrates `accumulate(All, multi_expr)` as the production pattern

2. **All six shape variants** (skewness types 1-3, kurtosis types 1-3): scalar adjustments
   on the same sufficient stats → free after step 1

3. **Grouped moments**: same pass with `ByKey` grouping → validates groupby path

4. **Quantiles types 7 and 8** (the two most useful): implement sort → gather → interpolate
   → this is the one new primitive needed (GPU sort, or CPU sort for small n)

5. **MAD, Gini, trimmed mean**: require sort or two-pass median, implement after quantiles

6. **Geometric mean, harmonic mean**: easy single-pass, special phi expressions

---

## Gold Standard Parity Tests (to run against tambear)

```python
import numpy as np
from scipy import stats

x = np.array([2, 8, 0, 4, 1, 9, 9, 0])

# Ground truth for tambear validation:
print(stats.describe(x))
# DescribeResult(nobs=8, minmax=(0, 9), mean=4.125, variance=14.982...,
#                skewness=0.2650..., kurtosis=-1.4286...)
# Note: scipy.stats.kurtosis default is excess (g2) with bias correction (Type 2)

print(np.percentile(x, [25, 50, 75], method='hazen'))  # Type 5
print(np.percentile(x, [25, 50, 75], method='weibull'))  # Type 6
print(np.percentile(x, [25, 50, 75]))  # default linear = Type 7
```

R equivalents:
```r
x <- c(2, 8, 0, 4, 1, 9, 9, 0)
library(e1071)
skewness(x, type=1)  # g1
skewness(x, type=2)  # G1 (Fisher, bias corrected)
skewness(x, type=3)  # b1 (Minitab)
kurtosis(x, type=1)  # g2 (excess)
kurtosis(x, type=2)  # G2 (SAS/SPSS)
kurtosis(x, type=3)  # b2 (Minitab)
quantile(x, type=7)  # R default (= scipy default)
quantile(x, type=8)  # Hyndman recommended
```
