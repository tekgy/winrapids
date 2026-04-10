# F07 Hypothesis Testing — Adversarial Test Suite (Phase 2 Update)

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**File**: `src/hypothesis.rs` (~37KB)

**Positive**: Uses `MomentStats.m2` (centered second moment) — NOT affected by naive formula Bug Class 1.

---

## New Findings

### H1 [MEDIUM]: chi2_goodness_of_fit skips zero expected counts

**Location**: hypothesis.rs:303

`if e > 0.0 { ... } else { 0.0 }` — when expected=0 but observed>0, the chi-square contribution should be +∞ (impossible observation). Skipping makes the test accept data that contradicts the null.

**Test**: `chi2_goodness_of_fit(&[5.0, 0.0], &[5.0, 0.0])` → statistic=0 (correct).
`chi2_goodness_of_fit(&[5.0, 3.0], &[8.0, 0.0])` → statistic=0 (WRONG — should be ∞, observation in impossible bin).

### H2 [MEDIUM]: odds_ratio returns +∞ for 0/0 case

**Location**: hypothesis.rs:479

`[0, 0, 5, 5]`: `b*c = 0*5 = 0`, returns inf. But `a*d = 0*5 = 0`, so OR = 0/0 = NaN.

### H3 [MEDIUM]: one_proportion_z doesn't validate successes <= n

**Location**: hypothesis.rs:374-387

`one_proportion_z(10.0, 5.0, 0.5)`: `p_hat=2.0`, `asin(sqrt(2.0))=NaN`. Cohen's h is NaN, z-statistic is computed but meaningless.

### H4 [LOW]: chi2_independence panics on n_rows=0

**Location**: hypothesis.rs:319

Division by zero. Also `n_cols - 1` underflows for empty table.

### H5 [LOW]: log_odds_ratio_se infinite for zero cells

**Location**: hypothesis.rs:493

Known statistical limitation. Should use Haldane correction (add 0.5 to cells).

---

## Summary: 0 HIGH, 3 MEDIUM (new), 2 LOW (new)
