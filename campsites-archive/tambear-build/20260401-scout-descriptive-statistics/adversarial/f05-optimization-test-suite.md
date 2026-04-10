# Family 05: Optimization — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: REVIEWED
**Code**: `crates/tambear/src/optimization.rs`

---

## Operations Tested

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| Backtracking line search | optimization.rs:44-67 | OK |
| Golden section | optimization.rs:72-98 | OK |
| Gradient descent + momentum | optimization.rs:112-136 | OK |
| Adam | optimization.rs:144-175 | OK |
| AdaGrad | optimization.rs:182-206 | OK |
| RMSProp | optimization.rs:213-237 | OK |
| L-BFGS | optimization.rs:246-330 | **MEDIUM** (absolute curvature) |
| Nelder-Mead | optimization.rs:342-442 | **MEDIUM** (NaN panic) |
| Coordinate descent | optimization.rs:450-474 | OK |
| Projected gradient | optimization.rs:481-506 | OK |

---

## Finding F05-1: Nelder-Mead Final Min NaN Panic (MEDIUM)

**Bug**: At line 437, `f_vals.iter().enumerate().min_by(|a, b| a.1.partial_cmp(b.1).unwrap())` will panic if any function value is NaN.

This is inconsistent with line 366, which correctly uses `unwrap_or(std::cmp::Ordering::Equal)` in the sort. The sort is NaN-safe but the final min selection is not.

**Impact**: If the objective function returns NaN for any simplex vertex, the optimizer panics instead of returning the best non-NaN result.

**Fix**: `.unwrap_or(std::cmp::Ordering::Equal)` as already done at line 366.

---

## Finding F05-2: L-BFGS Absolute Curvature Threshold (MEDIUM)

**Bug**: At line 316, `if sy > 1e-14` uses an absolute threshold to decide whether to accept a curvature pair (s, y).

**Impact**: For small-scale optimization (parameters of order 1e-10), legitimate s·y values will be smaller than 1e-14, causing all curvature pairs to be rejected. The L-BFGS degrades to steepest descent with a unit Hessian approximation.

**Fix**: Use a relative threshold: `sy > tol * s_norm * y_norm` or `sy > eps * max(1.0, s_norm * y_norm)`.

---

## Positive Findings

**Adam is textbook correct.** Bias-corrected moments, standard hyperparameters.

**L-BFGS two-loop recursion is correct.** The algorithm structure, γ scaling, and history management are all standard. Only the curvature threshold is problematic.

**Nelder-Mead reflection/expansion/contraction/shrinkage logic is correct.** Standard coefficients α=1, γ=2, ρ=0.5, σ=0.5.

**All gradient convergence tests use gradient norm** — appropriate for unconstrained optimization.

---

## Test Vectors

### TV-F05-NM-01: Nelder-Mead with NaN objective (BUG CHECK)
```
f(x) = x[0]² + x[1]² if norm < 10 else NaN
x0 = [5, 5], step = 1.0
Expected: finds minimum or reports non-convergence
Currently: PANIC when any vertex lands in NaN region
```

### TV-F05-LBFGS-01: L-BFGS small-scale optimization
```
f(x) = 1e-20 * (x[0] - 1e10)² + 1e-20 * (x[1] - 1e10)²
x0 = [0, 0]
Expected: converges to [1e10, 1e10]
Test: s·y values will be ~1e-20, below 1e-14 threshold
Currently: may degrade to steepest descent
```

### TV-F05-ADAM-01: Rosenbrock convergence
```
f = Rosenbrock, x0 = [0, 0]
lr=0.01, max_iter=50000
Expected: |x - [1,1]| < 0.01
```

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F05-1: Nelder-Mead NaN panic | **MEDIUM** | Thread panic | unwrap_or(Equal) |
| F05-2: L-BFGS curvature threshold | **MEDIUM** | Steepest descent fallback | Relative threshold |
