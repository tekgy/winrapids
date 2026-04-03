# Family 06: Descriptive Statistics — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.

---

## Overview

Family 06 covers: central tendency, dispersion, shape, and quantiles. Every major statistical package (R, Python scipy/numpy/pandas, SAS, SPSS, Stata, Excel) computes these differently in subtle ways. This document specifies EVERY variant, EVERY edge case, and EVERY cross-package discrepancy so the implementation is provably correct.

### MSR Connection

The 11-field MSR {n, Σp, Σp², max, min, Σsz, Σ(p·sz), Σr, Σr², Σr³, Σr⁴} already covers most of what's needed:
- Central moments from raw power sums: μ₂, μ₃, μ₄ all derivable
- All moment-based skewness/kurtosis are scalar transforms of {n, M₂, M₃, M₄}
- max, min give range directly
- Quantile-based statistics (median, MAD, IQR, Bowley, Moors) require **sorted data** — not derivable from MSR

### Two Computation Paths

1. **From MSR** (sufficient statistics — free when Tam has stats): moments, variance, std, skewness, kurtosis, CV, range
2. **From sorted data** (requires a sort pass): median, quantiles, IQR, MAD, trimmed mean, winsorized mean, mode, Gini, Bowley skewness, Moors kurtosis, L-moments, medcouple

---

## 1. Central Tendency

### 1.1 Arithmetic Mean

```
x̄ = (1/n) Σxᵢ
```

**From MSR**: x̄ = S₁/n where S₁ = Σx.

**Edge cases**: n=0 → NaN. n=1 → x₁. All NaN → NaN. Contains ±Inf → ±Inf (or NaN if both).

**Numerical stability**: For GPU parallel reduction, use Kahan-compensated summation or pairwise summation (O(log n) error growth). Naive summation has O(n) error growth. The fixed-reference centering approach from manuscript 001 applies here.

### 1.2 Weighted Mean

```
x̄_w = Σ(wᵢ · xᵢ) / Σwᵢ
```

**Assumptions**: Σwᵢ ≠ 0. Weights non-negative (by convention; negative weights are mathematically valid but change interpretation).

**Edge cases**: All weights zero → NaN. Negative weights → computed but interpretation changes.

### 1.3 Geometric Mean

```
GM = exp((1/n) Σ ln(xᵢ))
```

**Assumptions**: ALL xᵢ > 0. Not defined for negative values (ln undefined in ℝ).

**Edge cases**:
- Contains zero → GM = 0 (ln(0) = -∞, exp(-∞) = 0)
- Contains negative → NaN
- Overflow: direct product overflows; ALWAYS use log-space
- Underflow: direct product underflows; log-space handles correctly

**Weighted**: GM_w = exp(Σ(wᵢ · ln(xᵢ)) / Σwᵢ)

### 1.4 Harmonic Mean

```
HM = n / Σ(1/xᵢ)
```

**Assumptions**: ALL xᵢ ≠ 0. Conventionally requires all xᵢ > 0.

**Edge cases**:
- Contains zero → HM = 0 (1/0 = ∞, n/∞ = 0)
- Mixed sign → undefined (reciprocals may cancel to zero → HM = ∞)
- All negative → HM is negative (valid but unusual)

**Ordering guarantee**: For all positive data: HM ≤ GM ≤ AM, equality iff all values identical.

### 1.5 Median

```
median = x_(⌈n/2⌉)           if n is odd
median = (x_(n/2) + x_(n/2+1))/2   if n is even
```

**Requires sorted data.** This is the standard definition used by all major packages.

**Edge cases**: n=1 → x₁. n=2 → (x₁+x₂)/2.

### 1.6 Mode

**Definition**: Most frequent value. For continuous data, mode is the value with highest kernel density estimate.

**CAUTION**: Mode is ill-defined for continuous data without binning/KDE. Not commonly used in automated pipelines. Multiple modes possible (bimodal, multimodal).

**Edge cases**: All unique values → no mode (or all values are modes). All same → that value.

### 1.7 Trimmed Mean

```
m = ⌊α·n⌋
trimmed_mean = (1/(n-2m)) Σᵢ₌ₘ₊₁ⁿ⁻ᵐ x_(i)
```

**Both R and scipy floor α·n.** No standard package does fractional/partial trimming.

**Special values**: α=0 → arithmetic mean. α→0.5 → median.

**Edge cases**: α·n ≥ n/2 → no data remains → undefined.

### 1.8 Winsorized Mean

```
m = ⌊α·n⌋
Replace x_(1)...x_(m) with x_(m+1)
Replace x_(n-m+1)...x_(n) with x_(n-m)
winsorized_mean = mean of modified array
```

**Key difference from trimming**: denominator is always n (all observations used, with replacements).

---

## 2. Dispersion

### 2.1 Variance and Standard Deviation

**Population variance** (biased): σ² = m₂ = (1/n) Σ(xᵢ - x̄)²

**Sample variance** (unbiased): s² = (1/(n-1)) Σ(xᵢ - x̄)² = n·m₂/(n-1)

**From MSR power sums**:
```
μ₂ = S₂/n - (S₁/n)²  =  (n·S₂ - S₁²) / n²
```

**CRITICAL NUMERICAL WARNING**: This formula suffers from catastrophic cancellation when the mean is large relative to the standard deviation. For financial prices (~100) with returns (~0.01), computing variance of prices loses ~8 decimal digits in f32.

**Mitigation**: Use Pebay's parallel combining algorithm or fixed-reference centering (manuscript 001).

**Pebay parallel merge** (for combining partitions A and B):
```
δ = x̄_B - x̄_A
M₂_X = M₂_A + M₂_B + δ² · nₐ·n_B/n_X
```

**Edge cases**: n=0 → NaN. n=1 → population var = 0, sample var = NaN (0/(1-1)). All same → 0.

### 2.2 MAD (Median Absolute Deviation)

```
MAD = median(|xᵢ - median(x)|)
```

**Consistency constant**: 1/Φ⁻¹(3/4) ≈ 1.4826

**CRITICAL CROSS-PACKAGE DIFFERENCE**:
- **R** `mad()`: returns 1.4826 × MAD by default
- **Python** `median_abs_deviation()`: returns raw MAD by default

**We must support both**: raw MAD and scaled MAD (with configurable constant).

**Edge cases**: All same → MAD = 0. n=1 → MAD = 0. n=2 → MAD = |b-a|/2.

### 2.3 Range

```
range = max - min
```

**From MSR**: directly available.

**Edge cases**: n=1 → 0. Contains ±Inf → Inf.

### 2.4 IQR (Interquartile Range)

```
IQR = Q(0.75) - Q(0.25)
```

**CRITICAL**: IQR depends on quantile method. R/NumPy default (type 7) gives DIFFERENT IQR than SPSS default (type 6).

**Example**: data 1..20: Type 7 IQR = 9.5, Type 6 IQR = 10.5.

Must specify which quantile type is used.

### 2.5 Coefficient of Variation (CV)

```
CV = s / x̄     (sample version, ddof=1)
CV = σ / μ     (population version, ddof=0)
```

**Bias correction** (Haldane 1955): CV* = (1 + 1/(4n)) · CV

**CROSS-PACKAGE DIFFERENCE**: scipy `variation()` defaults to ddof=0. R uses ddof=1. Must support both.

**Assumptions**: Ratio-scale data. x̄ ≠ 0. Same-sign data preferred.

**Edge cases**: mean=0, std>0 → ±Inf. mean=0, std=0 → NaN. n=1 with ddof=1 → NaN.

### 2.6 Gini Coefficient

**Sorting-based formula** (population):
```
G = (2 / (n·S)) Σᵢ₌₁ⁿ i·x_(i) - (n+1)/n
```
where S = Σx_(i) and data is sorted ascending.

**Sample correction** (Parzen): G_sample = (n/(n-1)) · G_pop

**Assumptions**: All xᵢ ≥ 0. Mean ≠ 0.

**Edge cases**:
- All same → G = 0
- All zeros → NaN (mean = 0)
- Contains negatives → computable but loses [0,1] bound and Lorenz interpretation
- n=1 → G = 0 (sample correction undefined; return 0)

**Computation**: O(n log n) via sorting. Do NOT use O(n²) all-pairs.

---

## 3. Shape Statistics

### 3.1 Raw Moments to Central Moments

Given GPU accumulators {n, S₁=Σx, S₂=Σx², S₃=Σx³, S₄=Σx⁴}, let μ = S₁/n:

```
μ₂ = S₂/n - μ²
μ₃ = S₃/n - 3μ·(S₂/n) + 2μ³
μ₄ = S₄/n - 4μ·(S₃/n) + 6μ²·(S₂/n) - 3μ⁴
```

Or equivalently:
```
μ₂ = (n·S₂ - S₁²) / n²
μ₃ = (n²·S₃ - 3n·S₁·S₂ + 2·S₁³) / n³
μ₄ = (n³·S₄ - 4n²·S₁·S₃ + 6n·S₁²·S₂ - 3·S₁⁴) / n⁴
```

**NUMERICAL STABILITY**: These formulas involve catastrophic cancellation for large means. Prefer Pebay's parallel combining algorithm:

```
Pebay merge for M₃:
M₃_X = M₃_A + M₃_B + δ³·nₐ·n_B·(nₐ-n_B)/n_X² + 3δ·(nₐ·M₂_B - n_B·M₂_A)/n_X

Pebay merge for M₄:
M₄_X = M₄_A + M₄_B + δ⁴·nₐ·n_B·(nₐ²-nₐ·n_B+n_B²)/n_X³
      + 6δ²·(nₐ²·M₂_B + n_B²·M₂_A)/n_X² + 4δ·(nₐ·M₃_B - n_B·M₃_A)/n_X
```

### 3.2 Skewness — Three Standard Variants

All three are scalar transforms of g₁. Compute g₁ once, derive the rest.

**Type 1: g₁ (Fisher's skewness, biased/population)**
```
g₁ = m₃ / m₂^(3/2) = √n · M₃ / M₂^(3/2)
```
Default in: scipy.stats.skew(bias=True), R e1071::skewness(type=1), Stata

**Type 2: G₁ (adjusted Fisher-Pearson, bias-corrected)**
```
G₁ = √(n(n-1)) / (n-2) · g₁
```
Default in: Excel SKEW(), SAS, SPSS, pandas .skew(), scipy.stats.skew(bias=False), R e1071::skewness(type=2)

**Type 3: b₁ (MINITAB/BMDP)**
```
b₁ = g₁ · ((n-1)/n)^(3/2)
```
Default in: R e1071::skewness(type=3), MINITAB

**Minimum n**: 3 for all types. G₁ has (n-2) in denominator → undefined at n=2.

**When variance = 0**: ALL return NaN (0/0).

### 3.3 Kurtosis — Four Standard Variants

All are scalar transforms of g₂. Compute g₂ once, derive the rest.

**Type 1: g₂ (excess kurtosis, biased/population)**
```
g₂ = m₄/m₂² - 3 = n·M₄/M₂² - 3
```
Normal = 0. Default in: scipy.stats.kurtosis(fisher=True, bias=True), R e1071::kurtosis(type=1)

**Type 2: G₂ (adjusted excess kurtosis, bias-corrected)**
```
G₂ = ((n-1) / ((n-2)(n-3))) · ((n+1)·g₂ + 6)
```
Normal = 0. Default in: Excel KURT(), SAS, SPSS, pandas .kurtosis(), scipy.stats.kurtosis(bias=False), R e1071::kurtosis(type=2)

**Type 3: b₂ (MINITAB/BMDP)**
```
b₂ = (g₂ + 3) · ((n-1)/n)² - 3
```
Normal = 0. Default in: R e1071::kurtosis(type=3), MINITAB

**Pearson's kurtosis (non-excess, Stata)**
```
β₂ = g₂ + 3
```
Normal = 3. Default in: Stata summarize.

**Minimum n**: 4 for all types. G₂ has (n-2)(n-3) in denominator → undefined at n≤3.

**When variance = 0**: ALL return NaN.

### 3.4 Robust/Quantile-Based Shape Measures

**Bowley skewness** (quartile skewness):
```
SK_B = (Q₃ + Q₁ - 2·Q₂) / (Q₃ - Q₁)
```
Range: [-1, +1]. Undefined when IQR = 0. Requires sorted data + quantile method choice.

**Pearson's 2nd skewness** (median-based):
```
Sk₂ = 3·(x̄ - median) / s
```
Typically in [-3, +3] for unimodal distributions.

**Moors' kurtosis** (octile-based):
```
K_M = [(Q(7/8) - Q(5/8)) + (Q(3/8) - Q(1/8))] / (Q(6/8) - Q(2/8))
```
Normal ≈ 1.2330.

**L-moment skewness (τ₃) and kurtosis (τ₄)**: From probability weighted moments (PWMs):
```
b_r = (1/n) Σⱼ₌ᵣ₊₁ⁿ [C(j-1,r) / C(n-1,r)] · x_(j)

λ₁ = b₀
λ₂ = 2b₁ - b₀
λ₃ = 6b₂ - 6b₁ + b₀
λ₄ = 20b₃ - 30b₂ + 12b₁ - b₀

τ₃ = λ₃/λ₂,  τ₄ = λ₄/λ₂
```
Ranges: τ₃ ∈ (-1,1), τ₄ ∈ [-1/4, 1).

**Medcouple** (robust kernel skewness):
```
MC = median{ h(xᵢ, xⱼ) : xᵢ ≥ median, xⱼ ≤ median }
where h(xᵢ, xⱼ) = ((xᵢ - med) - (med - xⱼ)) / (xᵢ - xⱼ)
```
Range: [-1, 1]. Breakdown point: 25%. O(n log n) fast algorithm exists.

### 3.5 Raw Moments (m'₁ through m'₈)

```
m'_k = (1/n) Σxᵢᵏ
```

From MSR: m'_k = S_k/n. For k > 4, need additional accumulators Σx⁵...Σx⁸.

**Central moments via binomial transform**:
```
μ_k = Σⱼ₌₀ᵏ C(k,j)·(-1)^(k-j)·m'_j·μ^(k-j)
```

**Numerical stability degrades rapidly with k**. For k ≥ 5, the cancellation in the binomial transform makes raw-power-sum accumulation essentially unusable in f32. Use f64 or two-pass centering.

---

## 4. Quantiles — The 9 Hyndman-Fan Methods

### General Framework

For sorted data x_(1) ≤ ... ≤ x_(n) and probability p ∈ [0,1]:

**Types 1-3** (discontinuous, return observed values):
- Type 1: j = ⌈np⌉. Q = x_(j).
- Type 2: if np is integer k: Q = (x_(k) + x_(k+1))/2. Otherwise: Q = x_(⌈np⌉).
- Type 3: j = round(np) with round-half-to-even. Q = x_(j).

**Types 4-9** (continuous, linear interpolation):
```
h = (n + 1 - α - β)·p + α
j = ⌊h⌋
g = h - j
Q(p) = (1-g)·x_(j) + g·x_(j+1)
```
with j clamped to [1, n] and x_(j+1) = x_(n) when j = n.

| Type | α | β | p_k = | Default in |
|------|---|---|-------|------------|
| 4 | 0 | 1 | k/n | SAS PCTLDEF=1 |
| 5 | 1/2 | 1/2 | (k-1/2)/n | Hydrology (Hazen) |
| 6 | 0 | 0 | k/(n+1) | SPSS, Excel PERCENTILE.EXC, Minitab |
| 7 | 1 | 1 | (k-1)/(n-1) | **R, NumPy, SciPy, Julia, Excel PERCENTILE.INC** |
| 8 | 1/3 | 1/3 | (k-1/3)/(n+1/3) | Recommended by Hyndman & Fan 1996 |
| 9 | 3/8 | 3/8 | (k-3/8)/(n+1/4) | Optimal under normality (Blom) |

### Properties Satisfied

| Property | T1 | T2 | T3 | T4 | T5 | T6 | T7 | T8 | T9 |
|----------|----|----|----|----|----|----|----|----|-----|
| Continuous | No | No | No | Yes | Yes | Yes | Yes | Yes | Yes |
| Q(0.5) = median | Yes | Yes | **No** | Yes | Yes | Yes | Yes | Yes | Yes |
| Q(0) = min, Q(1) = max | Yes | Yes | Yes | No | No | No | **Yes** | No | No |

**Type 7 is the only continuous method where Q(0) = min and Q(1) = max exactly.** This is why it's the default in R/NumPy.

### SAS Mapping

| SAS PCTLDEF | H-F Type |
|-------------|----------|
| 1 | Type 4 |
| 2 | Type 3 |
| 3 | Type 1 |
| 4 | Type 6 |
| 5 (default) | Type 2 |

SAS does NOT natively support types 5, 7, 8, or 9.

### Edge Cases (all types)

- **n = 1**: Q(p) = x_(1) for all p.
- **n = 2**: Types 1-3 return x_(1) or x_(2) (or average for type 2). Types 4-9 interpolate.
- **p = 0**: All types return x_(1).
- **p = 1**: All types return x_(n).
- **All same values**: Q(p) = c for all p and all types.
- **NaN in data**: R/NumPy propagate by default; na.rm=TRUE/nanquantile skip NaN.

### Implementation Note

**Multiple quantiles of the same data**: Sort once, compute all quantiles in parallel. The sort dominates cost (O(n log n)); each quantile computation is O(1).

---

## 5. Mathematical Genealogy (What Shares What)

### From MSR {n, S₁, S₂, S₃, S₄}:
- mean (S₁/n)
- variance, std (from μ₂)
- CV (std/mean)
- range (from max, min — separate accumulators)
- all 3 skewness types (from g₁, which needs μ₂, μ₃)
- all 4 kurtosis types (from g₂, which needs μ₂, μ₄)
- raw moments 1-4

### From sorted data:
- median
- all 9 quantile types
- IQR (from Q₁, Q₃)
- MAD (from median, then sort of deviations — **two sorts!**)
- trimmed mean, winsorized mean
- Gini coefficient
- Bowley skewness (from Q₁, Q₂, Q₃)
- Moors kurtosis (from octiles)
- L-moments (from order statistics)
- medcouple

### Sharing opportunities:
1. **Sort once** → feed median, all quantiles, IQR, trimmed mean, winsorized mean, Gini, Bowley, Moors, L-moments
2. **MSR accumulate once** → feed mean, variance, std, CV, all moment skewness, all moment kurtosis
3. **MAD is special**: needs median (from sort 1), then absolute deviations, then another sort (sort 2). Can we avoid the second sort? Only if we use a selection algorithm (O(n) median-of-medians) instead of full sort for the outer median.

---

## 6. Failure Modes to Test

| Statistic | Failure mode | Expected behavior |
|-----------|-------------|-------------------|
| Variance (raw power sums) | Large mean, small std → cancellation | Use Pebay or centering |
| Skewness | n < 3 | NaN for all types |
| Kurtosis G₂ | n ≤ 3 | NaN (division by zero) |
| CV | mean ≈ 0 | ±Inf or NaN |
| Gini | negative values | Computable but meaningless (>1 or <0) |
| Gini | all zeros | NaN (mean = 0) |
| MAD | all same values | 0 (with or without constant) |
| Geometric mean | contains zero | 0 |
| Geometric mean | contains negative | NaN |
| Harmonic mean | contains zero | 0 |
| Quantile type 3 | n=2, p=0.5 | Does NOT equal median |
| IQR | quantile type matters | Different types give different IQR |
| Medcouple | ties at median | Special case formula needed |
| L-moments | λ₂ = 0 | τ₃, τ₄ undefined |
| Moments k ≥ 5 | f32 cancellation | Essentially unusable; require f64 |

---

## 7. Implementation Priority

**Phase 1 — MSR-derived (no sort needed)**:
1. mean (trivial)
2. variance / std (population + sample)
3. skewness g₁, G₁, b₁
4. kurtosis g₂, G₂, b₂, β₂
5. CV (+ bias correction)
6. range

**Phase 2 — Sort-derived**:
7. median
8. quantiles (all 9 types)
9. IQR
10. trimmed mean
11. winsorized mean
12. Gini

**Phase 3 — Two-pass / special**:
13. MAD (two sorts or sort + selection)
14. weighted mean
15. geometric mean, harmonic mean
16. Bowley skewness, Pearson 2nd skewness
17. Moors kurtosis
18. L-moments (τ₃, τ₄)
19. medcouple
20. raw moments 5-8

---

## 8. Composability Contract Template

```toml
[algorithm]
name = "skewness_fisher"
family = "descriptive.shape"

[inputs]
required = ["numeric_array"]
optional = ["weights"]

[outputs]
primary = "skewness_g1"
secondary = ["skewness_G1", "skewness_b1"]  # all derivable from g1

[sufficient_stats]
consumes = ["n", "M2", "M3"]       # Pebay central moment accumulators
produces = ["n", "M2", "M3"]

[sharing]
provides_to_session = ["MomentStats(order=3)"]
consumes_from_session = ["MomentStats(order>=3)"]
auto_insert_if_missing = "accumulate(All, [pebay_M2_M3], PebayCombine)"

[assumptions]
requires_sorted = false
requires_positive = false
requires_no_nan = false
minimum_n = 3
variance_must_be_nonzero = true
```
