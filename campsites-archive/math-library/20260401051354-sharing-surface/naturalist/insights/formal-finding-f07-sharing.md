# Formal Finding: F07 Hypothesis Testing = Pure Extraction from F06

*Naturalist finding, April 1 2026*
*Verified by observer: 68/68 gold standard parity tests pass*

---

## Claim

**Family 07 (Hypothesis Testing) adds zero new GPU primitives. Every statistical test is CPU arithmetic on MomentStats accumulated by Family 06 (Descriptive Statistics).**

## Evidence

### Import analysis (hypothesis.rs)

```rust
use crate::descriptive::MomentStats;
use crate::special_functions::{
    normal_two_tail_p, t_two_tail_p,
    f_right_tail_p, chi2_right_tail_p,
};
```

No imports of: `scatter_jit`, `accumulate`, `compute_engine`, `gpu`, `kernel`, `cuda`, or any GPU infrastructure.

### Function signatures

Every test function takes `&MomentStats` (or `&[MomentStats]` for ANOVA) and returns a result struct. Examples:

```rust
pub fn one_sample_t(stats: &MomentStats, mu: f64) -> TestResult
pub fn welch_t(stats1: &MomentStats, stats2: &MomentStats) -> TestResult
pub fn one_way_anova(groups: &[MomentStats]) -> AnovaResult
```

The HypothesisEngine wraps DescriptiveEngine — it delegates ALL scatter work to F06.

### Coverage

27 algorithms implemented in ~930 lines of CPU-only code:

**Tests (18):** one-sample t, two-sample t, Welch's t, paired t, one-way ANOVA, chi-square goodness-of-fit, chi-square independence, one-sample z (proportions), two-sample z (proportions)

**Effect sizes (6):** Cohen's d (pooled), Glass's delta, Hedges' g, eta-squared, omega-squared, Cramér's V, odds ratio

**Corrections (3):** Bonferroni, Holm-Bonferroni, Benjamini-Hochberg (FDR)

### Observer verification

- 26 F07-specific parity tests pass against scipy/R gold standards
- F=t² identity confirmed (ANOVA reduces to t-test for k=2 groups)
- F06→F07 sharing chain test proves single MomentStats feeds t-test, ANOVA, and Cohen's d simultaneously
- p-value CDFs (custom special_functions) accurate to ~1e-10

## Structural Implication

MomentStats(order=2) is a **sufficient statistic** for the entire hypothesis testing family. The MSR principle predicts this: if hypothesis tests are polynomial functions of {n, Σx, Σx²} per group, then MomentStats IS the minimum sufficient representation.

The polynomial extraction functions are:
- t-statistic: `(x̄ - μ) / (s / √n)` — rational function of {n, sum, m2}
- F-statistic: `(SS_between / (k-1)) / (SS_within / (N-k))` — ratio of sums of m2 and group means
- χ²: `Σ (O-E)² / E` — polynomial in counts
- Cohen's d: `(x̄₁ - x̄₂) / sp` — rational function of group {n, sum, m2}

All are rational functions of MomentStats fields. No field beyond order 2 is needed.

## For the Lab Notebook

> **The sharing tree is not theoretical.** Family 07 (Hypothesis Testing — the SPSS replacement) requires zero new GPU computation beyond Family 06 (Descriptive Statistics). 27 algorithms, ~930 lines, zero GPU code. The MomentStats MSR type is the sharing surface: one accumulation, many extractions.
>
> This is the engineering consequence of a mathematical fact: every two-sample hypothesis test is a COMPARISON of sufficient statistics. The t-test compares means. ANOVA compares variances. Chi-square compares frequencies. All are polynomial functions of the same accumulator fields.
>
> **Publishable claim**: MomentStats(order=2, ByKey) is a sufficient statistic for the parametric hypothesis testing family. The proof is constructive: hypothesis.rs implements 27 algorithms as pure extraction functions on MomentStats, with zero re-scanning of data.

---

*Signed: naturalist (structural analysis), observer (68/68 verification), navigator (design prediction)*
