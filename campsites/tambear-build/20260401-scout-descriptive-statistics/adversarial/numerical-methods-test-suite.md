# Numerical Methods (numerical.rs) — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01 (Phase 2)
**File**: `src/numerical.rs` (~27KB)

---

## N1 [HIGH]: Bisection — no bracket sign-change validation

**Location**: numerical.rs:39-61

**Bug**: Unlike `brent()` (line 131), `bisection()` never checks `f(a) * f(b) < 0`. Silently returns garbage root.

**Test vectors**:
```rust
// No real root exists — but bisection returns "converged"
bisection(|x| x*x + 1.0, -1.0, 1.0, 1e-10, 100)
// Expected: converged=false
// Actual: converged=true, root≈0.0, function_value=1.0

// Same-sign bracket — silently collapses to endpoint
bisection(|x| x - 10.0, 20.0, 30.0, 1e-10, 100)
// Expected: converged=false (no root in [20,30])
// Actual: converged=true, root≈20.0 (garbage)
```

**Fix**: Add `if fa * f(b) > 0.0 { return RootResult { converged: false, ... }; }` like Brent does.

---

## N2 [HIGH]: Newton — hardcoded 1e-15 derivative guard

**Location**: numerical.rs:84

**Bug**: `if dfx.abs() < 1e-15` blocks convergence for well-scaled small-valued functions.

**Test vector**:
```rust
// f'(x) ≈ 1e-20 everywhere — blocks Newton even though it would work
newton(|x| 1e-20 * (x*x - 2.0), |x| 1e-20 * 2.0 * x, 1.0, 1e-30, 50)
// Expected: root ≈ √2
// Actual: converged=false on first iteration (|f'(1)| = 2e-20 < 1e-15)
```

**Fix**: Test step magnitude `|f(x)/f'(x)| > bound` instead of absolute `|f'(x)| < eps`.

---

## N3 [HIGH]: Secant — hardcoded 1e-15 denominator guard

**Location**: numerical.rs:110

Same failure mode as Newton. `secant(|x| 1e-20 * (x*x - 2.0), 1.0, 2.0, 1e-30, 50)` bails immediately.

---

## N4 [HIGH]: Brent — missing mflag bookkeeping

**Location**: numerical.rs:150-180

**Bug**: Standard Brent tracks whether the previous step was bisection (`mflag`) and uses this to decide whether to force bisection on the next iteration. This implementation omits `mflag`, which can cause the algorithm to accept interpolation steps when it should be forcing bisection, potentially causing the bracket to stall.

**Test vector**: Construct a function where IQI/secant oscillates without making progress. Standard Brent detects this via `mflag` and forces bisection; this implementation doesn't.

---

## N5 [HIGH]: RK45 — claims Dormand-Prince, implements RKF45

**Location**: numerical.rs:451-464

**Bug**: Docstring says "The standard ODE45 from MATLAB/scipy" — MATLAB uses Dormand-Prince (7 stages, FSAL property). This code has 6 stages with RKF45 coefficients:
- Nodes: `[0, 1/4, 3/8, 12/13, 1, 1/2]` — RKF45 ✓, not Dormand-Prince
- 4th-order weights: `[25/216, 0, 1408/2565, 2197/4104, -1/5, 0]` — RKF45 ✓

Anyone relying on MATLAB-compatible behavior gets different error characteristics.

**Fix**: Either fix the docstring to say "RKF45" or implement actual Dormand-Prince.

---

## N6 [MEDIUM]: Brent secant branch — fb==fa division by zero

**Location**: numerical.rs:169

`s = b - fb * (b - a) / (fb - fa)` — if `fb == fa`, produces NaN.

**Test**: Function returning near-zero at both bracket points.

---

## N7-N8 [MEDIUM]: RK45 hardcoded absolute thresholds

- Line 474: `t >= t_end - 1e-15` — fails for `t_end < 1e-14`
- Line 495: `h <= 1e-15` — forces step acceptance for tiny-scale problems

---

## N9-N10 [MEDIUM]: No divergence detection in ODE solvers

All three solvers (euler, rk4, rk45) silently fill output with NaN/Inf when the ODE blows up.

**Test**: `rk45(|_t, y| y * y, 1.0, 0.0, 2.0, 1e-8, 0.01)` — blows up at t=1.

---

## N11 [MEDIUM]: RK4 system — no dimension check

**Location**: numerical.rs:518-553

`rk4_system(|_t, _y| vec![1.0], &[1.0, 2.0], ...)` — `f` returns wrong-length vec, panics at `k1[1]`.

---

## N12 [MEDIUM]: Adaptive Simpson — NaN exponential blowup

**Location**: numerical.rs:314-349

If `f` returns NaN at any point, `NaN.abs() < tol` is false, causing subdivision until `depth=0`. For `max_depth=50`, that's `2^50 ≈ 10^15` function evaluations.

**Test**: `adaptive_simpson(|x| if x > 0.5 { f64::NAN } else { x }, 0.0, 1.0, 1e-10, 50)` — hangs.

**Fix**: Early return NaN when `whole.is_nan()`.

---

## N13 [MEDIUM]: Bug Class 5 — all convergence checks are absolute

Lines 47, 80, 106, 566. Unified fix: `|value| < tol * (1 + |reference|)`.

---

## N14 [LOW]: RK45 error threshold 1e-30 for step growth

Line 503: `if error > 1e-30` — for tolerance 1e-40, error of 1e-35 is huge but treated as "zero".

## N15 [LOW]: Simpson n=0

Line 259: `n=0` → `h = (b-a)/0 = Inf` → NaN result.
