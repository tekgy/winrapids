# pinv rcond Bug — Absolute vs Relative Threshold

## The Bug

`linear_algebra.rs:753`:
```rust
let rcond = rcond.unwrap_or(1e-12);
```

The default rcond is absolute (1e-12), but it should be relative to the matrix.

## Why This Is Wrong

For a 100×100 matrix with singular values [1e6, 5e5, ..., 1.0, 1e-3, 1e-10]:
- **Current behavior**: treats σ < 1e-12 as zero. All singular values survive. CORRECT for this matrix.
- But for a matrix with singular values [1e-6, 5e-7, ..., 1e-12, 1e-15, 1e-22]:
  - Current: treats σ < 1e-12 as zero. Keeps 1e-12, drops 1e-15. 
  - This means noise at 1e-15 is treated as zero, but signal at 1e-12 (which is 1e-6 of max σ) is kept.
  - **This is the wrong cutoff**: 1e-12 relative to max σ = 1e-6 is a ratio of 1e-6 — that's real signal, not noise. But 1e-15 relative to 1e-6 is a ratio of 1e-9 — also possibly signal.

The real problem: for a matrix with max σ = 1e6, the correct cutoff is:
```
100 * 1e6 * 2.2e-16 ≈ 2.2e-8
```
But the code uses 1e-12. So singular values between 1e-12 and 2.2e-8 are treated as nonzero when they're actually numerical noise. The pseudoinverse will amplify this noise by factors of 1/σ up to 1/1e-12 = 1e12.

## The Fix

```rust
let rcond = rcond.unwrap_or_else(|| {
    let max_dim = a.rows.max(a.cols) as f64;
    let max_sv = svd_res.sigma.first().copied().unwrap_or(0.0);
    max_dim * max_sv * f64::EPSILON
});
```

Note: `svd_res.sigma` must be sorted in descending order for `sigma[0]` to be the largest. Verify this is the case.

## Reference

numpy.linalg.pinv uses:
```python
rcond = len(a) * amax(s) * finfo(a.dtype).eps
```
where `s` is the singular value array, `len(a)` is max(m,n).

LAPACK's dgelss uses:
```
rcond = max(m,n) * ||A||_2 * eps  (when rcond < 0, meaning "use default")
```

## Impact

This affects any downstream method that uses `pinv` on matrices that aren't unit-scaled:
- `lstsq` (least squares via pinv path)
- Any covariance-based method where covariances are in squared units
- Financial data where prices are in thousands (covariance ~ 1e6)

## Severity

MEDIUM. Most uses in tambear work on standardized data (correlation matrices, normalized features) where max σ ≈ n and the absolute threshold is adequate. But the principle is wrong, and any user passing raw (non-standardized) data will get incorrect results for ill-conditioned systems.

## Note

The `rcond: Option<f64>` parameter already exists, so users CAN override. The issue is solely the default. This is a one-line fix with no API change.
