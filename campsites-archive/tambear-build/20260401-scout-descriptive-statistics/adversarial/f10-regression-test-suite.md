# Family 10: Regression — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: PROVEN by code review + naive formula analysis
**Code**: `crates/tambear/src/train/linear.rs`, `cholesky.rs`, `logistic.rs`

---

## Operations Tested

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| fit (normal equations) | linear.rs:60-159 | **MEDIUM** (ill-conditioned X'X at offset) |
| column_stats | linear.rs:183-196 | **HIGH** (naive Σv² accumulation) |
| fit_session (z-score) | linear.rs:226-289 | **HIGH** (z-score uses naive std) |
| Cholesky decomposition | cholesky.rs:12-37 | OK (correctly detects non-PD) |
| forward_solve | cholesky.rs:40-50 | OK |
| back_solve | cholesky.rs:53-63 | OK |
| logistic fit (GD) | logistic.rs:76-173 | **MEDIUM** (no feature normalization) |
| sigmoid | logistic.rs:60-62 | OK (numerically stable) |
| predict/predict_proba | logistic.rs:40-57 | OK |
| R² computation | linear.rs:131-144 | OK (correctly centered) |
| coeff back-transform | linear.rs:273-279 | OK (correct algebra) |

---

## Finding F10-1: column_stats Naive Accumulation (HIGH)

**Bug**: `column_stats` at linear.rs:183-196 accumulates `sum_sqs[j] += v * v`. The `SufficientStatistics::variance()` at intermediates.rs:279-283 then computes:

```rust
pub fn variance(&self, g: usize) -> f64 {
    let n = self.counts[g];
    let m = self.sums[g] / n;
    self.sum_sqs[g] / n - m * m  // <-- naive E[x²] - E[x]²
}
```

This is the SAME naive formula bug found in 5 other locations.

**Impact**: `fit_session` uses `stats.std(j)` to z-score features. If std is wrong due to catastrophic cancellation, z-scoring is wrong, and the ENTIRE PURPOSE of `fit_session` is defeated.

**Example**: Feature values x = [1e8, 1e8+1, ..., 1e8+99]:
```
true variance = 833.25
naive E[x²] = 1.0000000165e16
naive E[x]² = 1.0000000165e16  (same to 10+ digits!)
naive variance = GARBAGE (cancellation destroys all significant digits)
```

**Fix**: `column_stats` should accumulate centered statistics:
```rust
// Two-pass: first compute means, then centered sums
let means: Vec<f64> = (0..d).map(|j| sums[j] / n as f64).collect();
let mut m2s = vec![0.0f64; d];
for i in 0..n {
    for j in 0..d {
        let dev = x[i * d + j] - means[j];
        m2s[j] += dev * dev;
    }
}
```

Or better: `SufficientStatistics` should store `m2` (centered sum of squares) instead of `sum_sqs`.

---

## Finding F10-2: Normal Equations Ill-Conditioning (MEDIUM)

**Bug**: `fit()` computes X'X directly without centering features. For features with large offsets, X'X has condition number ~ O(offset²/variance), which causes Cholesky to fail or give inaccurate results.

**Impact**: `fit()` will return errors or wrong coefficients for financial data (prices in millions + intercept column). `fit_session` is designed to fix this via z-scoring, but F10-1 undermines that fix.

**Example**: Feature x = [1e8, 1e8+0.1, ..., 1e8+9.9]:
```
X'X ≈ [[1e18, 1e10], [1e10, 100]]
cond(X'X) ≈ 1e16 → Cholesky loses 16 digits → all precision gone
```

**Fix**: Already handled by `fit_session`... IF the std computation is correct (see F10-1).

---

## Finding F10-3: Logistic Regression No Normalization (MEDIUM)

**Bug**: `logistic::fit` at logistic.rs:76-173 has no feature normalization. Gradient descent on un-normalized features with different scales converges slowly or not at all (learning rate needs to be tuned per-feature).

**Impact**: Poor convergence for mixed-scale features. Not a correctness bug but a usability/reliability issue.

**Fix**: Add a `logistic::fit_session` analogous to `linear::fit_session` that z-scores first.

---

## Positive Findings

**Sigmoid is numerically stable.** `1/(1+exp(-z))` handles extremes correctly: returns 0 for z << -700, returns 1 for z >> 700. The cross-entropy clamping at 1e-15 prevents log(0).

**Cholesky correctly detects non-positive-definite.** Returns `None` when diagonal element goes ≤ 0. This is the right behavior.

**R² is correctly centered.** `ss_tot = Σ(y - y_mean)²` uses the centered formula, not the naive formula.

**Coefficient back-transformation is algebraically correct.** The mapping from normalized-space β to original-space β is derived correctly.

---

## Test Vectors

### TV-F10-OLS-01: Large offset regression
```
x = [1e8 + i*0.1 for i in 0..100], y = 2.0 * x + 1.0
fit() expected: slope≈2.0, intercept≈1.0
Currently: Cholesky likely fails or gives garbage
```

### TV-F10-OLS-02: fit_session with large offset (BUG CHECK)
```
Same as TV-F10-OLS-01, but via fit_session
Expected: z-scoring should fix conditioning
Currently: z-scoring uses naive std → GARBAGE
```

### TV-F10-OLS-03: Multi-feature, mixed scale
```
x1 = [1e6 + i*0.01], x2 = [0.001*i]
y = 3*x1 + 500*x2 + 7
Expected: recover true coefficients
Test: compare fit() vs fit_session()
```

### TV-F10-CHO-01: Singular matrix
```
X has collinear features: x2 = 2*x1
X'X is rank-deficient → Cholesky returns None
```

### TV-F10-LOG-01: Linearly separable, mixed scale
```
x1 in [1e6, 1e6+1], x2 in [-1, 1]
Decision boundary exists.
Expected: converges with appropriate lr
Currently: may not converge with fixed lr
```

### TV-F10-SIG-01: Sigmoid extremes
```
sigmoid(-1000) ≈ 0.0 (not NaN, not -Inf)
sigmoid(+1000) ≈ 1.0 (not NaN, not +Inf)
sigmoid(0) = 0.5
```

---

## Connection to Naive Formula Bug Class

This is instance #7 (via `SufficientStatistics::variance`) of the naive formula bug class. The complete list is in `naive-formula-codebase-sweep.md`. The `SufficientStatistics` type stores `sum_sqs` (raw Σv²) rather than `m2` (centered Σ(v-mean)²), making EVERY downstream consumer vulnerable.

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F10-1: column_stats naive | **HIGH** | z-score safety net broken | Store m2 in SufficientStatistics |
| F10-2: Normal equations cond | **MEDIUM** | Expected limitation | Already mitigated by fit_session (once F10-1 fixed) |
| F10-3: Logistic no normalization | **MEDIUM** | Slow/non-convergence | Add logistic fit_session |
