# Families 08+09: Non-parametric + Robust — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: PROVEN numerically
**Code**: `crates/tambear/src/nonparametric.rs`, `crates/tambear/src/robust.rs`
**Proof script**: `docs/research/notebooks/f08-f09-adversarial-proof.py`

---

## Operations Tested

### F08 (nonparametric.rs)

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| rank | nonparametric.rs:41-80 | OK (NaN-safe, average ties) |
| spearman | nonparametric.rs:86-91 | OK (rank-based = offset-immune) |
| kendall_tau | nonparametric.rs:97-132 | **MEDIUM** (NaN -> discordant) |
| mann_whitney_u | nonparametric.rs:173-206 | **MEDIUM** (no tie correction) |
| wilcoxon_signed_rank | nonparametric.rs:220-253 | OK |
| kruskal_wallis | nonparametric.rs:267-297 | OK |
| ks_test_normal | nonparametric.rs:307-332 | OK |
| ks_test_two_sample | nonparametric.rs:337-375 | **LOW** (small n p-value) |
| bootstrap_percentile | nonparametric.rs:427-468 | OK |
| permutation_test | nonparametric.rs:473-509 | OK |
| kde | nonparametric.rs:539-555 | OK |
| runs_test | nonparametric.rs:608-642 | OK |
| sign_test | nonparametric.rs:667-686 | OK |

### F09 (robust.rs)

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| huber_m_estimate | robust.rs:109-111 | OK (IRLS converges) |
| bisquare_m_estimate | robust.rs:117-119 | OK |
| hampel_m_estimate | robust.rs:124-126 | OK |
| qn_scale | robust.rs:194-227 | OK |
| sn_scale | robust.rs:234-249 | **LOW** (includes self-distance) |
| tau_scale | robust.rs:257-275 | OK |
| lts_simple | robust.rs:313-349 | **HIGH** (naive OLS) |
| mcd_2d | robust.rs:417-489 | OK (centered covariance) |
| medcouple | robust.rs:522-549 | OK |

---

## Finding F09-1: LTS ols_subset Naive OLS (HIGH)

**Bug**: `ols_subset` at robust.rs:375-388 computes:
```rust
let denom = n * sxx - sx * sx;
let b = (n * sxy - sx * sy) / denom;
```
This is the naive formula for OLS regression. `n * sxx - sx * sx = n^2 * Var(x)` computed via the naive variance formula. Catastrophic cancellation at large offset.

**Proof table**:
```
offset    naive_b   naive_a     status
0         2.000000  1.000000    OK
1e4       2.000000  1.000000    OK
1e6       2.000000  1.000000    OK
1e8       SINGULAR              BROKEN
1e10      0.000000  2e10        BROKEN
1e12      0.000000  2e12        BROKEN
1e14      SINGULAR              BROKEN
```

**Centered fix produces exact results at ALL offsets through 1e14.**

This is the SAME bug class as:
- F06: naive variance (hash_scatter.rs, intermediates.rs)
- F10: GramMatrix det(X'X) = N^2 * Var(x)

**Fix**: Center x by mean before computing sums. Replace the inner loop with:
```rust
let mx = sx / n;
let my = sy / n;
let sxy_c = indices.iter().map(|&i| (x[i] - mx) * (y[i] - my)).sum();
let sxx_c = indices.iter().map(|&i| (x[i] - mx).powi(2)).sum();
let b = sxy_c / sxx_c;
let a = my - b * mx;
```

---

## Finding F08-1: Kendall Tau NaN as Discordant (MEDIUM)

**Bug**: `kendall_tau` at nonparametric.rs:107-132 computes `dx = x[i] - x[j]` and `dy = y[i] - y[j]`. When either value is NaN:
- `dx = NaN`, `dy = NaN`, `product = NaN`
- `dx == 0.0` is false, `dy == 0.0` is false, `product > 0.0` is false
- Falls through to `discordant += 1`

**Impact**: NaN values are silently counted as discordant pairs. A single NaN in perfectly concordant data drops tau from 1.0 to 0.2.

**Proof**:
```
x = [1, 2, NaN, 4, 5],  y = [1, 2, 3, 4, 5]
With NaN (buggy):   concordant=6, discordant=4, tau=0.20
Without NaN:        concordant=6, discordant=0, tau=1.00
```

**Fix**: Skip pairs where either x[i], x[j], y[i], or y[j] is NaN:
```rust
if x[i].is_nan() || x[j].is_nan() || y[i].is_nan() || y[j].is_nan() { continue; }
```

---

## Finding F08-2: Mann-Whitney No Tie Correction (MEDIUM)

**Bug**: Mann-Whitney uses `sigma = sqrt(n1*n2*(n1+n2+1)/12)` which assumes no ties. The tie-corrected formula:
```
sigma = sqrt(n1*n2/12 * ((N+1) - sum(t^3-t) / (N*(N-1))))
```
where t = tie group sizes.

**Impact**: For heavily tied data (Likert scales), uncorrected sigma is ~4% too large, z is ~4% too small, p-value is too large. Conservative error.

**Proof**: With Likert data (values 1-5, many ties):
```
sigma_uncorrected = 13.23
sigma_corrected   = 12.74
|z_corrected / z_uncorrected| = 1.039  (3.9% difference)
```

---

## Finding F09-3: Sn Includes Self-Distance (LOW)

**Bug**: `sn_scale` includes `|xi - xi| = 0` in each inner median computation. Rousseeuw & Croux (1993) define Sn with `j != i`. Including the zero biases the inner median downward.

**Proof** (n=5, data=[1,2,3,4,5]):
```
i=0: median_with_self=2.0, median_without=3.0  (50% error!)
i=1: median_with_self=1.0, median_without=2.0  (50% error!)
...
```

**Fix**: Add `if j == i { continue; }` in the inner loop, or compute `diffs` excluding the self-distance.

---

## Test Vectors

### TV-F09-LTS-01: OLS at offset (BUG CHECK)
```
x = [1e8 + i*0.1 for i in 0..10]
y = [1.0 + 2.0*xi for xi in x]
Expected: slope=2.0, intercept=1.0
Currently: SINGULAR (denom rounds to 0)
```

### TV-F08-TAU-01: Kendall NaN (BUG CHECK)
```
x = [1, 2, NaN, 4, 5], y = [1, 2, 3, 4, 5]
Expected: tau = NaN (or tau = 1.0 with NaN excluded)
Currently: tau = 0.20 (WRONG)
```

### TV-F08-RANK-01: Rank offset invariance
```
x = [1e12 + i for i in 0..10]
Expected ranks: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
(Rank-based methods are immune to offset cancellation)
```

### TV-F08-MW-01: Mann-Whitney tied data
```
x = [1, 2, 2, 3, 3, 3, 4, 4, 5, 5]
y = [3, 3, 4, 4, 4, 5, 5, 5, 5, 5]
Verify: tie correction reduces sigma by ~4%
```

### TV-F08-KS-01: KS identical samples
```
x = y = [1, 2, 3, 4, 5]
Expected D = 0.0
```

### TV-F09-SN-01: Sn self-distance (BUG CHECK)
```
data = [1, 2, 3, 4, 5]
Current: Sn includes 0 in inner medians
Expected: Sn excludes self (higher value)
```

---

## Positive Findings

**Rank-based methods are offset-immune**: Spearman rho = 1.0 at offset 1e12. The ranking step eliminates the raw data, so catastrophic cancellation cannot occur. This is a fundamental structural advantage.

**MCD uses centered covariance**: `subset_stats_2d` computes `dx = x[i] - cx` before summing. Unlike LTS, MCD does NOT have the naive formula bug.

**IRLS M-estimators**: The convergence is well-behaved. MAD=0 fallback to scale=1.0 is acceptable because the offset cancels in `(x - mu)`. The LCG for bootstrap/permutation has excellent uniformity (chi-square = 9.64 vs critical 16.92).

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F09-1: LTS naive OLS | **HIGH** | BROKEN at offset >= 1e8 | Center x,y before OLS |
| F08-1: Kendall NaN | **MEDIUM** | tau=0.2 instead of 1.0 | Skip NaN pairs |
| F08-2: MW no tie correction | **MEDIUM** | Conservative 4% error | Tie-corrected sigma |
| F08-3: KS small n | **LOW** | Inaccurate p for n<20 | Exact tables or warn |
| F09-2: MAD=0 fallback | **LOW** | Acceptable behavior | Document only |
| F09-3: Sn self-distance | **LOW** | Biased for small n | Exclude j=i |
