# Special Functions — Adversarial Test Suite (Phase 2 Update)

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**File**: `src/special_functions.rs` (~20KB)

---

## Previously reported:
- SF1 [LOW]: erf Abramowitz & Stegun approximation has max error 1.5e-7 (sufficient for p-values)

## New findings:

### SF2 [MEDIUM]: t_two_tail_p catastrophic cancellation for extreme test statistics

**Location**: special_functions.rs:317-318

**Code**:
```rust
pub fn t_two_tail_p(t: f64, df: f64) -> f64 {
    2.0 * (1.0 - t_cdf(t.abs(), df))
}
```

**Bug**: `t_cdf(t.abs(), df)` for large t computes `1.0 - 0.5 * I_x(df/2, 0.5)` where `I_x` is near 0. When I_x is smaller than ~2e-16, `1.0 - tiny` rounds to exactly `1.0`, then `1.0 - 1.0 = 0.0`. The two-tailed p-value is reported as exactly 0.0 when it should be a very small positive number.

**Test**: `t_two_tail_p(100.0, 10.0)` — the true p-value is ~1e-16, but the function returns 0.0.

**Fix**: Compute directly: `regularized_incomplete_beta(df / (df + t*t), df/2.0, 0.5)`. This avoids the double cancellation through CDF and back. Equivalently, add a `t_sf()` function that returns `0.5 * I_x(...)` for t >= 0.

### SF3 [MEDIUM]: f_right_tail_p same cancellation

**Location**: special_functions.rs:322-324

**Code**:
```rust
pub fn f_right_tail_p(x: f64, d1: f64, d2: f64) -> f64 {
    1.0 - f_cdf(x, d1, d2)
}
```

Same issue. For very large F-statistics, `f_cdf ≈ 1.0`, and `1.0 - 1.0 = 0.0`.

**Fix**: Use symmetry: `f_right_tail_p(x, d1, d2) = regularized_incomplete_beta(d2 / (d1*x + d2), d2/2, d1/2)`.

### Positive findings:
- `normal_sf` correctly uses `erfc` directly (no cancellation)
- `chi2_sf` correctly uses `regularized_gamma_q` directly (no cancellation)
- `regularized_incomplete_beta` uses proper Lentz CF with symmetry relation
- `log_gamma` handles x ≤ 0, x < 0.5 (reflection), and large x correctly
- All CDFs have proper boundary guards

---

## Summary: 0 HIGH, 2 MEDIUM (new), 1 LOW (confirmed)
