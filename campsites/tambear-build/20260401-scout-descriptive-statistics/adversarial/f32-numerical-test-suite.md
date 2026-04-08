# Family 32: Numerical Methods — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: REVIEWED
**Code**: `crates/tambear/src/numerical.rs`

---

## Operations Tested

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| Bisection | numerical.rs:39-62 | OK |
| Newton's method | numerical.rs:69-92 | OK (LOW: abs threshold) |
| Secant method | numerical.rs:99-121 | OK (LOW: abs threshold) |
| Brent's method | numerical.rs:127-201 | OK |
| Central difference | numerical.rs:210-216 | OK |
| Second derivative | numerical.rs:219-225 | OK |
| Richardson extrapolation | numerical.rs:231-250 | OK |
| Simpson's rule | numerical.rs:259-274 | OK |
| Gauss-Legendre 5-pt | numerical.rs:282-308 | OK |
| Adaptive Simpson | numerical.rs:314-349 | OK |
| Trapezoidal rule | numerical.rs:354-362 | OK |
| Forward Euler | numerical.rs:378-402 | OK |
| RK4 | numerical.rs:409-437 | OK |
| RK45 (adaptive) | numerical.rs:443-515 | OK (LOW: misleading comment) |
| RK4 system | numerical.rs:518-553 | OK |
| Fixed-point iteration | numerical.rs:562-572 | OK |

---

## Finding F32-1: Newton Absolute Derivative Threshold (LOW)

**Note**: Newton's method at line 84 uses `dfx.abs() < 1e-15` as the zero-derivative guard. For functions with very small derivatives at the root (e.g., f(x) = x^3 near x=0, where f'(0) = 0), this is appropriate. But for functions where f'(x) is legitimately ~1e-16 (e.g., very flat functions), Newton would stop prematurely.

Same pattern in secant method (line 110).

---

## Finding F32-2: RK45 Coefficient Comment Mismatch (LOW)

**Note**: Line 451 comment says "Dormand-Prince coefficients" but the actual constants are the **Runge-Kutta-Fehlberg** (RKF45) coefficients. Both are valid 4th/5th order embedded pairs. Dormand-Prince (used by MATLAB's ode45 and scipy's solve_ivp) is slightly more accurate for the 5th-order solution, but RKF45 works correctly.

---

## Positive Findings

**Brent's method is correctly implemented.** IQI fallback to bisection, proper bracketing, acceptance conditions all match the standard algorithm.

**Adaptive Simpson is correct and efficient.** Richardson correction at the leaf level, tolerance halving per recursion, max depth guard. Standard implementation.

**Gauss-Legendre 5-point nodes and weights are correct.** Tabulated values match DLMF reference to full f64 precision. Exact for degree ≤ 9 polynomials.

**RK4 is textbook correct.** Standard k1-k4 computation with the classic 1/6 weighting.

**RK45 adaptive stepping is well-designed.** Safety factor 0.9, different exponents for grow (0.2) and shrink (0.25), max step count guard. Uses local extrapolation (5th-order solution). Relative/absolute tolerance blend.

**Richardson extrapolation uses correct factor 4^j.** Central differences have O(h²) error, so Richardson with ratio 2 requires factor 4.

---

## Test Vectors

### TV-F32-NEWTON-01: Flat function
```
f(x) = x^10 near x = 0.1
f'(0.1) = 10 * 0.1^9 = 1e-9
Expected: converges (f' is not near 1e-15)
```

### TV-F32-BRENT-01: No sign change
```
f(x) = x², a = 1, b = 2
Expected: no convergence reported (no sign change)
Currently: returns NaN (correct)
```

### TV-F32-RK45-01: Exponential growth
```
dy/dt = y, y(0) = 1, t ∈ [0, 1]
Expected: y(1) = e within tol
```

### TV-F32-GL5-01: Polynomial exactness
```
∫₀¹ x^9 dx = 1/10
GL5 is exact for degree ≤ 9, so error should be < 1e-14
```

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F32-1: Newton abs threshold | **LOW** | Premature stop for flat f | Relative threshold |
| F32-2: RK45 comment mismatch | **LOW** | Misleading only | Fix comment |
