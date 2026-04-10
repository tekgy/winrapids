# F07 Sharing Surface: All Hypothesis Tests from MomentStats

Created: 2026-04-01T05:35:41-05:00
By: navigator

Prerequisite: F06 complete (MomentStats implemented, RefCenteredStats working).

---

## The Core Insight

Every hypothesis test in F07 is an EXTRACTION from MomentStats — not a new accumulation.

The t-test IS descriptive statistics with a null-hypothesis comparison. ANOVA IS the regression F-test, extracted from the same GramMatrix. Once MomentStats(order=2, ByKey) exists in TamSession, F07 costs essentially nothing to compute — it's last-mile arithmetic on existing data.

---

## What Each Test Needs

### T-Tests (all variants)

**One-sample t-test** (H₀: μ = μ₀):
```
MSR needed: MomentStats(order=2, group=All)
t = (x̄ - μ₀) / (s / √n)
p = 2 * pt(-|t|, df=n-1)  // t-distribution CDF
```
Fields: n, sum1 (→ x̄), sum2 (→ s²). All from MomentStats(order=2, All).

**Two-sample t-test, equal variance** (Welch's when unequal):
```
MSR needed: MomentStats(order=2, group=ByKey(group_col))
t = (x̄₁ - x̄₂) / (sp * √(1/n₁ + 1/n₂))
sp² = ((n₁-1)s₁² + (n₂-1)s₂²) / (n₁+n₂-2)
```
Fields per group: n_g, sum1_g (→ x̄_g), sum2_g (→ s²_g). Pure extraction from ByKey MomentStats.

**Welch's t-test** (unequal variance, standard in R and Python):
```
t = (x̄₁ - x̄₂) / √(s₁²/n₁ + s₂²/n₂)
df = Welch-Satterthwaite equation  // uses s²/n per group
```
Same fields, different extraction formula.

**Paired t-test**: one-sample t on d = x₁ - x₂ column. MomentStats on difference column.

**Key**: all four t-test variants use EXACTLY the same accumulator fields: (n_g, Σx_g, Σx²_g) per group. Build ONE accumulate, four extraction functions.

---

### ANOVA (all variants)

**One-way ANOVA** F-test:
```
MSR needed: MomentStats(order=2, group=ByKey(group_col))

Grand mean: x̄ = Σ(n_g * x̄_g) / N  // derived from group stats, no second pass
SS_between = Σ n_g * (x̄_g - x̄)²
SS_within  = Σ (n_g - 1) * s²_g    // from sum2_g
F = (SS_between / (k-1)) / (SS_within / (N-k))
```

Same ByKey MomentStats as two-sample t-test. The F-statistic is derived from the same (n_g, x̄_g, s²_g) triples that the t-test uses.

**The ANOVA = Regression F-test rhyme** (naturalist):
- ANOVA F-test with k groups = regression of y on k-1 group dummies
- Same F-statistic: F = (R²/(k-1)) / ((1-R²)/(N-k))
- Same GramMatrix: X'X for group dummy matrix = block-diagonal of group sizes
- This means: if GramMatrix is available (e.g., from F10 regression work), ANOVA F is free

For F07 Phase 1: implement via ByKey MomentStats (simpler, self-contained)
For F07 Phase 2: wire to GramMatrix for the structural rhyme (sharing with F10)

**Two-way ANOVA**: needs cross-group MomentStats → ByKey((group1, group2)) double-key
**Repeated measures ANOVA**: needs per-subject, per-time MomentStats → ByKey((subject, time))

---

### Multiple Comparison Corrections

These are PURE ARITHMETIC on an array of p-values — no accumulate needed.

- **Bonferroni**: multiply each p by m (number of tests)
- **Holm**: sort p-values, multiply in rank order
- **Benjamini-Hochberg (FDR)**: sort, apply k/m × α threshold sequence
- **Tukey HSD**: needs studentized range distribution lookup

These are all O(k) CPU operations after the tests compute their p-values. No GPU needed. No MSR.

---

### Effect Sizes

**Cohen's d**:
```
d = (x̄₁ - x̄₂) / sp
```
Same MomentStats as t-test. Pure extraction.

**η² (eta-squared)**:
```
η² = SS_between / SS_total
```
From same ByKey MomentStats as ANOVA.

**Cramér's V**: from CrosstabStats (chi-square based). Different MSR type.

---

### Chi-square Tests

**Goodness of fit** (H₀: observed = expected):
```
MSR needed: CrosstabStats (counts per category)
χ² = Σ (O_i - E_i)² / E_i
```

**Independence** (H₀: row and column variables independent):
```
MSR needed: CrosstabStats(col_a, col_b)
χ² = Σᵢⱼ (n_ij - n_i·n_j/n)² / (n_i·n_j/n)
```

These need CrosstabStats, not MomentStats. Different accumulate (ByKey((a,b)) with Add on indicator).

---

### Proportion Tests

**Z-test for proportions** (H₀: p = p₀):
```
z = (p̂ - p₀) / √(p₀(1-p₀)/n)
```
p̂ = mean of binary (0/1) variable. MomentStats(order=1, All) suffices.

**Fisher exact test**: needs the full 2×2 contingency table → CrosstabStats.

---

## Accumulate Plan for F07

F07 needs ZERO new GPU compute beyond F06 and CrosstabStats. Every test is an extraction.

**Inputs** from TamSession:
1. `MomentStats(order=2, ByKey(group_col))` — all t-tests, ANOVA, effect sizes
2. `MomentStats(order=4, ByKey(group_col))` — normality tests (D'Agostino-Pearson skewness+kurtosis)
3. `CrosstabStats(col_a, col_b)` — chi-square, Fisher exact, McNemar
4. `MomentStats(order=2, All)` — one-sample t, proportion z-test

**No new primitives needed.** F07 is a library of extraction functions + statistical distribution lookups.

---

## Statistical Distribution Tables

Every test produces a test statistic and a p-value. The p-value requires evaluating a CDF:
- t-distribution: regularized incomplete beta function
- F-distribution: regularized incomplete beta function
- Chi-square: regularized incomplete gamma function
- Normal: error function (erf)
- Binomial: incomplete beta function

These are special functions, not accumulates. Options:
1. Taylor series approximation (fast, ~1% error for most cases)
2. Lookup table with interpolation (fast, accurate for common df values)
3. Full implementation via Lanczos/continued fractions (exact but complex)

**Decision**: use the approximation approach for Phase 1. The test statistic computation is the structural core; the p-value approximation can be improved later. This is consistent with the "implement, validate, iterate" methodology.

For validation: R's `pt()`, `pf()`, `pchisq()` are the gold standard. Match them within 1e-6.

---

## F07 Implementation Order

1. **t-test extraction functions** — from existing MomentStats(order=2, ByKey)
   - one_sample_t, two_sample_t_equal_var, welch_t, paired_t

2. **ANOVA F-test** — from MomentStats(order=2, ByKey), pure extraction
   - one_way_anova (note: same data as t-test, different extraction)

3. **Chi-square tests** — from CrosstabStats (need to implement CrosstabStats first)
   - goodness_of_fit, independence, homogeneity, mcnemar

4. **Effect sizes** — from same MomentStats as tests
   - cohen_d, eta_squared, omega_squared, cramer_v

5. **Multiple comparison corrections** — pure arithmetic, no GPU
   - bonferroni, holm, benjamini_hochberg

6. **Power analysis** — inverse functions of test statistics, pure math

---

## The Structural Insight to Surface in Lab Notebook

> The one-way ANOVA F-test and the two-sample t-test are the same extraction with different denominators. For k=2 groups, F = t². This is not a coincidence — both measure "signal (between-group difference) / noise (within-group variance)" using the same MomentStats(ByKey).

Document this explicitly. It's the kind of structural insight that makes tambear's design philosophy visible.
