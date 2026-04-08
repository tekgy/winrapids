# Scout Report: Descriptive Statistics (Family 06) (2026-04-01)

## descriptive_statistics_gold_standards.md

Gold standard (R e1071, scipy, numpy, R stats) for the full descriptive statistics family.
Central tendency, dispersion, shape, quantiles. All flavors, all edge cases.

**Headline**: All 6 shape stat variants (skewness types 1-3, kurtosis types 1-3) derive
from the same 5 sufficient stats {n, Σx, Σx², Σx³, Σx⁴}. One accumulate pass.
The "types" are just scalar correction factors applied post-hoc.

**Surprise**: `moments::kurtosis()` in R returns Pearson's kurtosis (m4/m2²) WITHOUT
subtracting 3. NOT excess kurtosis. Looks right but is off by 3. Common trap.

**Quantile note**: All 9 types need sorted order statistics. This is the one legitimate
place tambear sorts. The two most useful: Type 7 (R/scipy default) and Type 8 (Hyndman's
recommendation — approximately median-unbiased).

**Connection to fintek MSR**: The 11-field MSR has {Σr, Σr², Σr³, Σr⁴}. All shape stats
for financial returns are ALREADY embedded in the existing MSR. Skewness/kurtosis of
returns requires zero additional sufficient stats beyond what's already accumulated.

## Build order

1. Core moments (mean, var, std, skewness g1, kurtosis g2): accumulate(All, 5-expr)
2. All six type variants: scalar post-processing, same sufficient stats
3. Grouped versions: ByKey grouping, same phi
4. Quantiles Type 7 and 8: needs GPU sort + gather + linear interpolation
5. MAD (both flavors), Gini, trimmed mean: require sort/median, after quantiles
6. Geometric mean, harmonic mean: new phi expressions (log(x), 1/x)

## Edge cases to validate against gold standard

See document for exact n-requirements per statistic. Key:
- Kurtosis Type 2 (G2): requires n ≥ 4 (n-3 in denominator)
- moments::kurtosis returns m4/m2² (Pearson, NOT excess) — adjust by -3 for comparison
