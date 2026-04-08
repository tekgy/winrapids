# Family 07: Hypothesis Testing — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: PROVEN numerically
**Code**: `crates/tambear/src/hypothesis.rs`, `crates/tambear/src/special_functions.rs`
**Proof script**: `docs/research/notebooks/f07-hypothesis-adversarial-proof.py`

---

## Operations Tested

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| one_sample_t | hypothesis.rs:118-139 | OK (NaN for zero-var, correct) |
| two_sample_t | hypothesis.rs:148-172 | OK |
| welch_t | hypothesis.rs:181-214 | OK (Welch-Satterthwaite correct) |
| paired_t | hypothesis.rs:223-227 | OK (delegates to one_sample_t) |
| one_way_anova | hypothesis.rs:241-287 | **MEDIUM** (empty groups inflate k) |
| chi2_goodness_of_fit | hypothesis.rs:299-308 | **MEDIUM** (O>0, E=0 -> silent 0) |
| chi2_independence | hypothesis.rs:318-364 | OK |
| proportion z-tests | hypothesis.rs:374-422 | OK (p0 boundary guarded) |
| cohens_d | hypothesis.rs:432-438 | OK |
| hedges_g | hypothesis.rs:453-459 | **LOW** (n<4 guard wrong) |
| bonferroni | hypothesis.rs:503-506 | **MEDIUM** (NaN -> 1.0 via .min()) |
| holm | hypothesis.rs:512-526 | **MEDIUM** (NaN sort undefined) |
| benjamini_hochberg | hypothesis.rs:532-547 | **MEDIUM** (NaN sort undefined) |
| erf/erfc | special_functions.rs:32-59 | OK (A&S 7.1.26, error < 1.5e-7) |
| log_gamma | special_functions.rs:69-98 | OK (Lanczos g=7, 9-term) |
| reg_inc_beta | special_functions.rs:121-173 | OK (Lentz CF, handles extremes) |
| t_cdf / f_cdf / chi2_cdf | special_functions.rs:273-329 | OK |

---

## Finding F07-1: ANOVA Empty Groups Inflate k

**Bug**: `one_way_anova(groups)` uses `groups.len()` for k, not the count of non-empty groups. If the input contains empty `MomentStats` (count=0), they inflate `df_between = k - 1` and deflate `df_within = N - k`.

**Impact**: F-statistic is wrong. The error is conservative (inflated p-value, harder to reject H0), so it won't produce false positives. But it can produce false negatives.

**Example**:
```
groups = [real_A, real_B, MomentStats::empty()]
k = 3 (should be 2)
df_between = 2 (should be 1)
df_within = N - 3 (should be N - 2)
```

**Fix**: `let k = groups.iter().filter(|g| g.count > 0.0).count() as f64;`

---

## Finding F07-2: Chi-square Silently Ignores Impossible Events

**Bug**: `chi2_goodness_of_fit` computes `(O-E)^2/E`, guarding with `if e > 0.0 { ... } else { 0.0 }`. When O > 0 and E = 0, the term should be +INFINITY (observed an event with zero expected probability = impossible event), but the code returns 0.0.

**Example**:
```
observed = [10, 20, 5, 25]
expected = [15, 15, 0, 30]

Code returns: chi2 = 4.17  (drops the 5-in-zero-expected term)
Correct:      chi2 = +Inf  (impossible event observed)
```

**Fix**: `if e <= 0.0 && o > 0.0 { return ChiSquareResult { statistic: f64::INFINITY, ... }; }`

---

## Finding F07-3: Hedges' g Small-n Guard

**Bug**: For `n1 + n2 < 4`, code returns uncorrected Cohen's d. The correct Hedges' J correction factor for n=3 (df=1) is `J = 1 - 3/(4*1 - 1) = 0`, so `g = d * 0 = 0`.

**Impact**: Very small (only affects n1+n2=3, which is n1=1, n2=2 or similar).

**Fix**: Remove the `if n < 4.0 { return d; }` guard. The formula works correctly for all n >= 3.

---

## Finding F07-4: Multiple Comparison NaN Propagation

**Bug**: In Rust, `NaN.min(1.0)` returns `1.0` (Rust's `f64::min` returns the non-NaN argument). So Bonferroni turns `NaN * m = NaN` into `NaN.min(1.0) = 1.0`. A NaN p-value silently becomes "non-significant".

For Holm and BH: `partial_cmp` returns `Equal` for NaN, causing undefined sort position. The NaN p-value ends up at an arbitrary rank, corrupting the adjusted p-values for ALL tests.

**Fix**: Pre-filter NaN p-values. Or: use `f64::min` that propagates NaN (custom helper).

---

## Finding F07-5: P-value Underflow for Moderate t

**Observation**: `t_two_tail_p(10, 100)` returns 0.0. The true p-value is ~2e-22, which is representable in f64. The underflow occurs in the incomplete beta function: `I_x(50, 0.5)` where `x=0.5`. The front factor `exp(-33.3)/50 ~ 7e-17` multiplied by the CF result may lose precision.

**Impact**: For t > ~8 with df > ~50, p-values underflow to 0.0. This is acceptable for hypothesis testing (p < 1e-15 is "significant" regardless) but would break `log(p)` or p-value combination methods.

**Severity**: LOW for hypothesis testing, MEDIUM for meta-analysis methods that use log-p.

---

## Test Vectors

### TV-F07-T1: One-sample t, known exact
```
data = [3.0, 4.0, 5.0, 6.0, 7.0]
mu = 0.0
expected_t > 0, p < 0.01, df = 4.0, effect_size > 0
```

### TV-F07-T2: Zero variance, same mean
```
data = [5.0; 10], mu = 5.0
expected: t = NaN, p = NaN
```

### TV-F07-T3: Zero variance, different mean
```
data = [5.0; 10], mu = 3.0
expected: t = +Inf, p = 0.0
```

### TV-F07-T4: Welch unequal variance
```
x = [5.0, 5.1, 4.9, 5.0, 5.0]
y = [4.0, 6.0, 3.0, 7.0, 5.0]
expected: p > 0.5 (same mean), Welch df < 8
```

### TV-F07-T5: ANOVA F = t^2 for k=2
```
g1 = MomentStats::from([2,3,4,5])
g2 = MomentStats::from([6,7,8,9])
assert: F == t_stat^2 (within 1e-10)
assert: anova.p_value == t_test.p_value (within 0.01)
```

### TV-F07-T6: Chi-square O>0 E=0 (BUG CHECK)
```
observed = [10, 20, 5, 25]
expected = [15, 15, 0, 30]
expected_chi2 = +Inf (currently returns 4.17 -- WRONG)
```

### TV-F07-T7: Bonferroni NaN propagation (BUG CHECK)
```
p_values = [0.01, NaN, 0.04, 0.5]
bonferroni should return [0.04, NaN, 0.16, 1.0]
currently returns [0.04, 1.0, 0.16, 1.0] -- NaN silently becomes 1.0
```

### TV-F07-T8: BH monotonicity
```
p_values = [0.01, 0.04, 0.03, 0.5]
adjusted must satisfy: adj[i] >= raw[i] for all i
adjusted must satisfy: adj_bh[i] <= adj_bonf[i] for all i (BH less conservative)
```

### TV-F07-T9: Cauchy (df=1) heavy tails
```
t = 100, df = 1
expected_p = 2/pi * arctan(1/100) ~ 0.00637
tolerance: 0.001
```

### TV-F07-T10: Fractional Welch df
```
Construct groups where Welch-Satterthwaite gives df < 1
result must not be NaN, must be in [0, 1]
```

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F07-1: ANOVA empty groups | **MEDIUM** | Wrong df, conservative error | Filter count > 0 for k |
| F07-2: Chi-square O>0 E=0 | **MEDIUM** | Silent wrong chi2 | Return Inf for impossible events |
| F07-4: NaN propagation | **MEDIUM** | Silent wrong corrections | Pre-filter or propagate NaN |
| F07-5: P-value underflow | **LOW** | 0.0 for very small p | Acceptable for testing, fix for meta-analysis |
| F07-3: Hedges' g n<4 | **LOW** | Wrong g for tiny samples | Remove guard |
