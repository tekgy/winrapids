# F08 Non-parametric Statistics — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 08 (Non-parametric Statistics).
Key division: tests that use MomentStats(order=4) directly (buildable now)
vs. tests that need SortedPermutation (blocked on rank infrastructure).

---

## Infrastructure Dependency Map

### Buildable NOW (MomentStats already exists):

| Test | MSR needed | Status |
|------|-----------|--------|
| Jarque-Bera normality test | MomentStats(order=4) | Immediately buildable |
| D'Agostino-Pearson omnibus | MomentStats(order=4) | Immediately buildable |
| Shapiro-Francia approximation | MomentStats(order=4) | Approximately buildable |

### Blocked on SortedPermutation:

| Test | Blocking dependency | Notes |
|------|-------------------|-------|
| Shapiro-Wilk | Sort → W statistic | Full O(n²) algorithm |
| Mann-Whitney U (Wilcoxon rank-sum) | Ranks | Two-sample nonparametric |
| Wilcoxon signed-rank | Signed ranks | Paired nonparametric |
| Kruskal-Wallis | Ranks | k-sample ANOVA nonparametric |
| Spearman rank correlation | Ranks of both variables | Correlation nonparametric |
| Kolmogorov-Smirnov | ECDF (sorted) | Goodness of fit |
| Anderson-Darling | ECDF (sorted) | More powerful than KS |

---

## Immediately Buildable: Normality Tests from Moments

### Jarque-Bera Test

**What it tests**: H₀: data is normally distributed (via skewness and kurtosis).

```
JB = n/6 · [g₁² + (g₂_excess)²/4]
```

Where:
- `g₁` = Fisher skewness (Type 1, biased) = m₃/m₂^(3/2) from MomentStats(order=3)
- `g₂_excess` = excess kurtosis = m₄/m₂² - 3 from MomentStats(order=4)

Under H₀ (normal), JB ~ χ²(2) asymptotically. p = P(χ²(2) > JB).

```python
from scipy import stats
stats.jarque_bera(x)
# Returns: JarqueBeraResult(statistic, pvalue)
# Uses the chi-square(2) p-value

# Manual verification:
n = len(x)
g1 = stats.skew(x, bias=True)     # Type 1 (biased)
g2 = stats.kurtosis(x, bias=True)  # excess kurtosis
JB = n/6 * (g1**2 + g2**2/4)
p = 1 - stats.chi2.cdf(JB, df=2)
```

```r
# R: tseries or moments package
library(tseries)
jarque.bera.test(x)  # returns X-squared, df=2, p-value

# Or manually (matches scipy):
library(moments)
n <- length(x)
g1 <- moments::skewness(x)   # Type 1 (biased)
g2 <- moments::kurtosis(x) - 3  # excess kurtosis (moments::kurtosis is Pearson)
JB <- n/6 * (g1^2 + g2^2/4)
p <- pchisq(JB, df=2, lower.tail=FALSE)
```

**MSR formula** (from MomentStats(order=4), single group, ref=0 or ref=mean):
```
g₁ = m₃ / m₂^(3/2)           [from sum3, sum2, count]
g₂_excess = m₄/m₂² - 3       [from sum4, sum2, count — same as kurtosis extraction]
JB = n/6 * (g₁² + g₂_excess²/4)
p = chisq2_survival(JB)        [chi-square(2) survival function]
```

Zero new GPU compute. Pure extraction from MomentStats(order=4).

---

### D'Agostino-Pearson Omnibus Test (K²)

**More powerful than Jarque-Bera** for detecting departures from normality.
Uses bias-corrected skewness (G₁, Type 2) and kurtosis with the K² statistic.

```python
from scipy import stats
stats.normaltest(x)
# Returns NormaltestResult(statistic, pvalue)
# Uses D'Agostino-Pearson K² = Z_skew² + Z_kurt²

# The test applies normal approximations to the skewness and kurtosis:
# Z_skew ~ N(0,1) under H₀ (skewness-to-normal transform)
# Z_kurt ~ N(0,1) under H₀ (kurtosis-to-normal transform)
# K² = Z_skew² + Z_kurt² ~ χ²(2)
```

```r
# R: fBasics::dagoTest or manual
library(fBasics)
fBasics::dagoTest(x)  # returns statistic, p-value

# Or via moments:
moments::agostino.test(x)    # D'Agostino skewness component only
moments::anscombe.test(x)    # Anscombe-Glynn kurtosis component only
```

**Important**: `scipy.stats.normaltest` implements D'Agostino-Pearson (both components).
`scipy.stats.skewtest` tests skewness alone. `scipy.stats.kurtosistest` tests kurtosis alone.

**MSR dependency**: needs G₁ (Type 2, bias-corrected skewness) and corresponding kurtosis.
The normal transform for skewness uses a complex formula from D'Agostino (1970).
Involves sqrt((n+1)(n+3)/(6(n-2))) and a cube-root transform. Still pure extraction from
MomentStats(order=4) — no new GPU compute.

For the pathmaker: document that `scipy.stats.normaltest` is the oracle, not a manual formula.
The bias correction transform is complex — validate against scipy, don't rederive.

---

### Shapiro-Francia (Approximation, Large n)

For n > 50, the Royston (1992) approximation to Shapiro-Wilk can be computed from moments.
Full Shapiro-Wilk needs sorted order statistics (F08 blocked path).

Shapiro-Francia approximation is less common — use D'Agostino-Pearson for large-n normality.

---

## Blocked Path: Rank-Based Tests (Needs SortedPermutation)

These are documented here for completeness. NOT buildable until sort infrastructure lands.

### Shapiro-Wilk Normality Test (gold standard for small n)

```python
scipy.stats.shapiro(x)  # returns (W_statistic, p_value)
```

**Algorithm**: W = (Σ aᵢ x_(i))² / Σ(xᵢ - x̄)²
where x_(i) are order statistics, aᵢ = expected normal order statistics coefficients.

Requires sorting. O(n log n) sort + O(n²) coefficient lookup.

For tambear: requires `SortedPermutation` MSR type. After sort is available, W numerator
is a weighted gather + sum. Denominator is MomentStats(order=2, centered sum_sq).

```r
shapiro.test(x)    # R gold standard. Exact for n ≤ 5000 via Royston.
```

### Mann-Whitney U / Wilcoxon Rank-Sum

```python
scipy.stats.mannwhitneyu(x, y)           # two-sided by default
scipy.stats.mannwhitneyu(x, y, alternative='greater')
```

```r
wilcox.test(x, y)                       # Mann-Whitney in R
wilcox.test(x, y, paired=TRUE)          # Wilcoxon signed-rank
```

**Algorithm**: U = Σ rank(xᵢ in combined xy) - n₁(n₁+1)/2.
Equivalent: count all pairs (xᵢ, yⱼ) where xᵢ > yⱼ.
For large n: U ~ N(μ_U, σ_U) with tie-corrected variance.

Requires ranks. Blocked on SortedPermutation.

### Kruskal-Wallis (non-parametric one-way ANOVA)

```python
scipy.stats.kruskal(*groups)  # arbitrary number of groups
```

```r
kruskal.test(y ~ group)
```

H statistic = 12/(N(N+1)) · Σ_k nk · (R̄_k - (N+1)/2)² where R̄_k = mean rank in group k.
Under H₀: H ~ χ²(k-1). Requires joint ranking = SortedPermutation across all groups.

### Kolmogorov-Smirnov

```python
scipy.stats.ks_1samp(x, cdf)           # one-sample vs theoretical CDF
scipy.stats.ks_2samp(x, y)             # two-sample
scipy.stats.kstest(x, 'norm')          # against normal distribution
```

```r
ks.test(x, "pnorm")                    # one-sample
ks.test(x, y)                          # two-sample
```

D = max_x |F_n(x) - F(x)| where F_n = empirical CDF = step function on sorted data.
Requires sort. Blocked.

### Spearman Rank Correlation

```python
scipy.stats.spearmanr(x, y)           # returns (correlation, p_value)
```

```r
cor.test(x, y, method="spearman")
```

ρ = Pearson r applied to ranks. Requires ranking both x and y.
Pearson r of ranks = 1 - 6·Σd_i²/(n(n²-1)) where d_i = rank_x_i - rank_y_i.
Blocked on SortedPermutation.

---

## Package Reference Summary

### scipy.stats normality tests

```python
from scipy import stats

# Available NOW (can be validated):
stats.jarque_bera(x)            # JB statistic + chi2(2) p-value
stats.normaltest(x)             # D'Agostino-Pearson K² + chi2(2) p-value
stats.skewtest(x)               # D'Agostino skewness component alone
stats.kurtosistest(x)           # Anscombe-Glynn kurtosis component alone

# Blocked on sort:
stats.shapiro(x)                # Shapiro-Wilk (gold standard for n < 5000)
stats.kstest(x, 'norm')         # KS test vs normal
stats.anderson(x)               # Anderson-Darling (more powerful than KS)

# Rank-based tests (blocked):
stats.mannwhitneyu(a, b)
stats.wilcoxon(d)               # signed-rank
stats.kruskal(*groups)
stats.spearmanr(x, y)
stats.kendalltau(x, y)
```

### R gold standards

```r
# Available NOW:
shapiro.test(x)               # gold standard normality test for n ≤ 5000
jarque.bera.test(x)           # (tseries package)
moments::agostino.test(x)     # D'Agostino skewness test
moments::anscombe.test(x)     # Anscombe-Glynn kurtosis test

# Rank-based (blocked):
wilcox.test(x, y)             # Mann-Whitney / Wilcoxon
kruskal.test(y ~ group)       # Kruskal-Wallis
ks.test(x, "pnorm")           # KS test
cor.test(x, y, method="spearman")
```

---

## Validation Targets for Jarque-Bera

Test vector: `x = [3.1, 1.4, 1.5, 9.2, 6.5, 3.5, 8.9]` (n=7, same as F06)

```python
from scipy import stats
import numpy as np
x = np.array([3.1, 1.4, 1.5, 9.2, 6.5, 3.5, 8.9])
result = stats.jarque_bera(x)
print(f"JB={result.statistic:.6f}, p={result.pvalue:.6f}")
```

```r
library(tseries)
x <- c(3.1, 1.4, 1.5, 9.2, 6.5, 3.5, 8.9)
jarque.bera.test(x)
```

**Note**: for n=7, the chi-square(2) approximation is poor (JB is only asymptotically chi-square).
The p-value should be treated as approximate for small n. Shapiro-Wilk is preferred for n < 50.

For D'Agostino-Pearson:
```python
stats.normaltest(x)  # same vector — compare statistic and p-value vs scipy oracle
```

---

## Architectural Note: SortedPermutation as Shared MSR Type

Once SortedPermutation is built (required for F08 rank tests and F09 robust estimators), it will be the gateway to:
- Shapiro-Wilk W statistic (F08)
- Spearman correlation (F08)
- KS/Anderson-Darling tests (F08)
- Trimmed mean, Winsorized statistics (F09)
- L-statistics in general (F09)
- Quantile-based skewness/kurtosis (F06 Bowley/Bowley-medcouple)
- Percentile-based confidence intervals (F07)

SortedPermutation is a Kingdom A Level-1 MSR type missing from the current list.
The sorted indices array is itself a reusable intermediate — once computed for Shapiro-Wilk,
all rank-based tests share it via TamSession.
