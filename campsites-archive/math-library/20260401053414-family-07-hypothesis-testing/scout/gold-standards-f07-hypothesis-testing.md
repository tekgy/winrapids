# F07 Hypothesis Testing — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load context for the pathmaker on Family 07 (Hypothesis Testing).
Documents: exact scipy/R function signatures, default behavior traps,
p-value computation formulas, and validation targets.

Navigator's structural analysis is already in the campsite — this document
covers the gold standard *implementation* layer, not the architecture.

---

## Verification: "A t-test IS a descriptive statistic with different extraction"

**CONFIRMED — literally true in scipy source.**

`scipy.stats.ttest_1samp` (simplified):
```python
def ttest_1samp(a, popmean, axis=0, nan_policy='propagate', alternative='two-sided'):
    n = a.shape[axis]
    d = np.mean(a, axis) - popmean       # ← np.mean(). Same as descriptive stats.
    v = np.var(a, axis, ddof=1)           # ← np.var(ddof=1). Same as descriptive stats.
    denom = np.sqrt(v / n)                # ← standard error = std/√n
    t = np.divide(d, denom)
    # p-value from t-distribution:
    p = distributions.t.sf(np.abs(t), n-1) * 2  # two-tailed
    return t, p
```

Proof: the ONLY quantities used are `mean(a)` and `var(a, ddof=1)` — both are MomentStats(order=2).
The accumulate kernel is IDENTICAL to F06. The t-test is pure extraction from the same MSR.

`scipy.stats.ttest_ind` (Welch's, simplified):
```python
def ttest_ind(a, b, equal_var=False):
    na, nb = a.shape[0], b.shape[0]
    ma, mb = np.mean(a), np.mean(b)
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    if equal_var:
        # Student's t-test: pooled variance
        df = na + nb - 2
        sp2 = ((na-1)*va + (nb-1)*vb) / df
        se = np.sqrt(sp2 * (1/na + 1/nb))
    else:
        # Welch's t-test (default): separate variances
        vna, vnb = va/na, vb/nb
        # Welch-Satterthwaite degrees of freedom:
        df = (vna + vnb)**2 / (vna**2/(na-1) + vnb**2/(nb-1))
        se = np.sqrt(vna + vnb)
    t = (ma - mb) / se
    # p-value: two-tailed by default
    p = distributions.t.sf(np.abs(t), df) * 2
    return t, p
```

Again, only `mean` and `var` — MomentStats(order=2, ByKey). The extraction formulas differ
between Welch's and Student's, but the accumulate is identical.

---

## Critical Traps

### Trap 1: Welch's t-test is the DEFAULT in BOTH R and scipy

```r
t.test(a, b)              # Welch's (var.equal=FALSE) — DEFAULT
t.test(a, b, var.equal=TRUE)  # Student's (pooled variance)
```

```python
scipy.stats.ttest_ind(a, b)                  # Welch's (equal_var=False) — DEFAULT
scipy.stats.ttest_ind(a, b, equal_var=True)  # Student's
```

**Impact**: Welch's produces fractional degrees of freedom via Welch-Satterthwaite.
Student's always produces integer df = n₁ + n₂ - 2. When comparing against gold standard,
must match this default or the p-values will differ.

### Trap 2: p-values are TWO-TAILED by default

```r
t.test(a, mu=0)                            # two-tailed (default)
t.test(a, mu=0, alternative="greater")    # one-tailed (right)
t.test(a, mu=0, alternative="less")       # one-tailed (left)
```

```python
scipy.stats.ttest_1samp(a, 0)                          # two-tailed
scipy.stats.ttest_1samp(a, 0, alternative='greater')   # one-tailed
```

For two-tailed: `p = 2 * P(T > |t| | df)` = `2 * t_cdf(-|t|, df)`.
For one-tailed (greater): `p = P(T > t | df)`.

### Trap 3: R's t.test vs oneway.test for ANOVA

```r
aov(y ~ group)         # classic F-test, assumes equal variance
oneway.test(y ~ group) # Welch ANOVA, robust to unequal variance
```

`oneway.test` uses adjusted df (Welch-style). R's `aov` uses the classic F with integer df.
Same relationship as Student's vs Welch's t-test.

### Trap 4: scipy.stats.chi2_contingency applies Yates' correction by default for 2×2

```python
scipy.stats.chi2_contingency(table)                  # correction=True for 2×2 ← default
scipy.stats.chi2_contingency(table, correction=False) # no correction
```

Yates' correction reduces χ² to be more conservative:
`χ²_Yates = Σ (|O_i - E_i| - 0.5)² / E_i`

For large samples, the correction is negligible. For small 2×2 tables, it can change the
p-value substantially. Match R's behavior:
- R `chisq.test(table)` applies Yates' correction for 2×2 by default too.
- R `chisq.test(table, correct=FALSE)` disables it.

### Trap 5: R's chisq.test warns on small expected frequencies

R warns "Chi-squared approximation may be incorrect" when any expected frequency < 5.
`fisher.test()` is preferred for small-n 2×2 tables. Document this threshold in tests.

---

## Package Reference: T-Tests

### R: stats package (base R)

```r
# One-sample:
t.test(x, mu=0)                    # H₀: μ = 0
t.test(x, mu=μ₀, conf.level=0.95) # custom null, 95% CI

# Two-sample independent:
t.test(x, y)                       # Welch's (default)
t.test(x, y, var.equal=TRUE)       # Student's pooled
t.test(x, y, paired=TRUE)          # paired t-test

# Returns:
# $statistic  : t value
# $parameter  : degrees of freedom (fractional for Welch's)
# $p.value    : two-tailed p-value
# $conf.int   : confidence interval for mean (or mean difference)
# $estimate   : sample mean(s)
# $stderr     : standard error of the mean
```

### Python: scipy.stats

```python
from scipy import stats

# One-sample:
stats.ttest_1samp(x, popmean=0)
stats.ttest_1samp(x, 0, alternative='two-sided')  # explicit

# Two-sample:
stats.ttest_ind(x, y)                    # Welch's
stats.ttest_ind(x, y, equal_var=True)   # Student's
stats.ttest_rel(x, y)                   # paired

# Returns TtestResult(statistic, pvalue, df)
# Note: df field added in scipy 1.11; older versions return (t, p) tuple

# NaN handling:
stats.ttest_1samp(x, 0, nan_policy='omit')  # skip NaN rows
```

---

## Package Reference: ANOVA

### R

```r
# Classic one-way ANOVA:
result <- aov(y ~ factor(group), data=df)
summary(result)               # F statistic, p-value, df
TukeyHSD(result)              # post-hoc pairwise comparisons

# Welch ANOVA (unequal variances):
oneway.test(y ~ group)

# Kruskal-Wallis (non-parametric alternative, F08):
kruskal.test(y ~ group)

# Two-way ANOVA:
aov(y ~ A + B + A:B)         # with interaction term
```

### Python

```python
from scipy import stats

# One-way:
stats.f_oneway(group1, group2, group3)  # arbitrary number of groups
# Returns (F_statistic, p_value) — uses equal-variance assumption
# No Welch ANOVA in scipy — use pingouin or statsmodels

import pingouin as pg
pg.welch_anova(data=df, dv='y', between='group')  # Welch ANOVA

# Post-hoc:
pg.pairwise_tukey(data=df, dv='y', between='group')  # Tukey HSD
pg.pairwise_tests(data=df, dv='y', between='group', padjust='bonf')
```

**Note**: `scipy.stats.f_oneway` assumes equal variances (no Welch adjustment).
For unequal variance ANOVA, use `pingouin.welch_anova` or `statsmodels`.

---

## Package Reference: Chi-Square Tests

### R

```r
# Goodness of fit:
chisq.test(x=observed, p=expected_proportions)
chisq.test(x=observed)   # assumes uniform expected

# Independence (contingency table):
chisq.test(table(A, B))
chisq.test(table(A, B), correct=FALSE)  # disable Yates' correction

# Fisher exact (small n):
fisher.test(table(A, B))
fisher.test(table(A, B), alternative="greater")  # one-sided

# McNemar (paired proportions):
mcnemar.test(matrix(c(a,b,c,d), 2, 2))
```

### Python

```python
from scipy import stats

# Goodness of fit:
stats.chisquare(f_obs, f_exp)  # f_exp defaults to uniform

# Independence:
chi2, p, dof, expected = stats.chi2_contingency(table)
chi2, p, dof, expected = stats.chi2_contingency(table, correction=False)

# Fisher exact:
odds_ratio, p = stats.fisher_exact(table_2x2)
odds_ratio, p = stats.fisher_exact(table_2x2, alternative='greater')

# Power divergence (generalized chi-square family):
stats.power_divergence(f_obs, f_exp, lambda_="log-likelihood")  # G-test
```

---

## p-Value Computation: Special Functions

Every test produces a statistic; the p-value requires evaluating a survival function.

### t-distribution

```
Two-tailed p = 2 · P(T > |t| | df)
             = 2 · (1 - CDF_t(|t|, df))
             = 2 · I_{df/(df+t²)}(df/2, 1/2) / 2   [incomplete beta form]
```

```python
# scipy:
from scipy.special import stdtr  # t-distribution CDF
p = 2 * stdtr(df, -abs(t))      # two-tailed

# Equivalently:
from scipy.stats import t as t_dist
p = 2 * t_dist.sf(abs(t_val), df)
```

```r
p = 2 * pt(-abs(t), df=df)  # pt = t-distribution CDF
```

### F-distribution

```
p = P(F > f_obs | df1, df2)
  = 1 - CDF_F(f_obs, df1, df2)
  = I_{df2/(df2+df1·f)}(df2/2, df1/2)  [incomplete beta form]
```

```python
from scipy.stats import f as f_dist
p = f_dist.sf(f_obs, df1, df2)
```

```r
p = pf(f_obs, df1, df2, lower.tail=FALSE)
```

### Chi-square distribution

```
p = P(χ² > χ²_obs | df)
  = 1 - CDF_χ²(χ²_obs, df)
  = Γ(df/2, χ²_obs/2) / Γ(df/2)   [regularized incomplete gamma]
```

```python
from scipy.stats import chi2
p = chi2.sf(chi2_obs, df)
```

```r
p = pchisq(chi2_obs, df, lower.tail=FALSE)
```

### Tambear Phase 1 Approximation Strategy

These special functions can be approximated:
- **Normal approximation to t**: for df > 30, t ≈ z and use `erfc(-|t|/√2)/2` for p-value
- **χ² CDF**: use regularized incomplete gamma via series expansion
- **F CDF**: reduce to incomplete beta via `x = df1*F/(df1*F + df2)`

Exact reference values for validation: match R's `pt()`, `pf()`, `pchisq()` within 1e-8.

---

## Welch-Satterthwaite Degrees of Freedom

For two-sample Welch's t-test:
```
df = (v₁/n₁ + v₂/n₂)² / ((v₁/n₁)²/(n₁-1) + (v₂/n₂)²/(n₂-1))

where v_i = sample variance (ddof=1) for group i
```

This is a **floating-point** df. R returns it directly; scipy stores it in `TtestResult.df`.
The p-value uses this fractional df in the t-distribution CDF.

**Lower bound**: df ≥ min(n₁-1, n₂-1).
**Upper bound**: df ≤ n₁+n₂-2 (= Student's df).

For implementation: compute df as f64, pass to t-distribution CDF. No rounding to integer.

---

## Key Structural Identity: F = t² for k=2

For k=2 groups, one-way ANOVA F-statistic = t²:
```
F = t²   (exactly, when using Student's pooled variance, not Welch's)
p_F(df1=1, df2) = p_t(df2) two-tailed
```

Test this identity in validation:
```python
from scipy import stats
a = [1.2, 2.3, 1.8, 2.9, 1.1]
b = [3.1, 4.2, 3.7, 4.8, 3.5]
t_result = stats.ttest_ind(a, b, equal_var=True)
f_result = stats.f_oneway(a, b)
assert abs(f_result.statistic - t_result.statistic**2) < 1e-10
assert abs(f_result.pvalue - t_result.pvalue) < 1e-10
```

This identity is the proof that ANOVA generalizes t-test — same MSR, different extraction.

---

## Validation Targets

Use `x = [3.1, 1.4, 1.5, 9.2, 6.5, 3.5, 8.9]` (n=7) for one-sample tests vs μ₀=0.

**Confirm in R:**
```r
x <- c(3.1, 1.4, 1.5, 9.2, 6.5, 3.5, 8.9)
result <- t.test(x, mu=0)
cat("t:", result$statistic, "\n")
cat("df:", result$parameter, "\n")
cat("p:", result$p.value, "\n")
cat("ci_lo:", result$conf.int[1], "\n")
cat("ci_hi:", result$conf.int[2], "\n")
```

**Confirm in Python:**
```python
from scipy import stats
import numpy as np
x = np.array([3.1, 1.4, 1.5, 9.2, 6.5, 3.5, 8.9])
result = stats.ttest_1samp(x, 0)
print(f"t={result.statistic:.6f}, p={result.pvalue:.6f}, df={result.df}")
```

**For F07 parity tests**: run BOTH and verify they match to < 1e-10 before using as tambear oracle.

**For two-sample Welch's t-test:**
```r
a <- c(3.1, 1.4, 1.5, 9.2)
b <- c(6.5, 3.5, 8.9)
t.test(a, b)  # Welch's, capture t, df (fractional), p
t.test(a, b, var.equal=TRUE)  # Student's, df = n1+n2-2 = 5
```

Note: the df values will differ between Welch's and Student's — this is expected.

---

## CrosstabStats: What Chi-Square Tests Need Beyond MomentStats

Chi-square tests require counts per (category_a, category_b) pair.
This is NOT in MomentStats — it's a ByKey((a,b)) accumulation.

For a k×m contingency table:
```
counts[i,j] = scatter_phi("1.0", ByKey((row_cat, col_cat)), n_groups=k*m, Add)
```

Where the key = `row_index * m + col_index` (linearized 2D index).
Same pattern as the entropy family's multi-dimensional histogram.

The F07 navigator document calls this `CrosstabStats`. It needs to be added to `intermediates.rs`
alongside `MomentStats` and `ExtremaStats`.

Expected structure:
```rust
pub struct CrosstabStats {
    pub n_rows: usize,        // categories in first variable
    pub n_cols: usize,        // categories in second variable
    pub counts: Arc<Vec<f64>>, // n_rows × n_cols, row-major
    pub row_totals: Arc<Vec<f64>>,
    pub col_totals: Arc<Vec<f64>>,
    pub grand_total: f64,
}
```

Row/column marginals can be computed from counts via CPU reduce — no second GPU pass.
