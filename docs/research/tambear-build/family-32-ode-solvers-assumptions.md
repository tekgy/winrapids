# Family 32: ODE Solvers — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.

---

## Overview

Family 32 covers numerical solution of ordinary differential equations (ODEs):

```
dy/dt = f(t, y),    y(t₀) = y₀
```

This is "adaptive Kingdom B" — the Prefix grouping pattern with adaptive step control. The key insight: an ODE solver is a **prefix scan** over time steps, where the associative operation is function composition of flow maps. Each Runge-Kutta stage is an accumulate, the Butcher tableau specifies the expression, and adaptive stepping is a control loop over the scan.

### Accumulate Decomposition

```
y_{n+1} = accumulate(
    data = [t_n, y_n, h_n],
    grouping = Prefix(forward),
    expr = butcher_tableau_stages(f, t_n, y_n, h_n),
    op = Add  // stages are weighted sums
)
```

The "scan" over time is Prefix(forward). Each step composes flow maps. Adaptive control adjusts h_n per step — this is the "carry-augmented" prefix pattern.

### MSR Insight

The MSR for an ODE trajectory is {y_n, t_n, h_n, error_estimate}. Once these are accumulated, the solver state is fully described. For dense output (interpolation within a step), the MSR also includes the stage values k₁...k_s.

---

## 1. Explicit Runge-Kutta Methods

### 1.1 General Framework

An s-stage explicit Runge-Kutta method:

```
k₁ = f(tₙ, yₙ)
k₂ = f(tₙ + c₂h, yₙ + h(a₂₁k₁))
k₃ = f(tₙ + c₃h, yₙ + h(a₃₁k₁ + a₃₂k₂))
...
kₛ = f(tₙ + cₛh, yₙ + h(aₛ₁k₁ + ... + aₛ,ₛ₋₁kₛ₋₁))

yₙ₊₁ = yₙ + h(b₁k₁ + ... + bₛkₛ)
```

The Butcher tableau encodes the method:

```
c₁ |
c₂ | a₂₁
c₃ | a₃₁  a₃₂
...
cₛ | aₛ₁  aₛ₂  ...  aₛ,ₛ₋₁
----|-------------------------
    | b₁   b₂   ...  bₛ
```

**Consistency condition**: c_i = Σⱼ a_ij for each row.

### 1.2 Classical RK4

The workhorse. 4 stages, 4th-order accurate.

```
0   |
1/2 | 1/2
1/2 | 0    1/2
1   | 0    0    1
----|------------------
    | 1/6  1/3  1/3  1/6
```

```
k₁ = f(tₙ, yₙ)
k₂ = f(tₙ + h/2, yₙ + h·k₁/2)
k₃ = f(tₙ + h/2, yₙ + h·k₂/2)
k₄ = f(tₙ + h, yₙ + h·k₃)
yₙ₊₁ = yₙ + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)
```

**Properties**:
- Local truncation error: O(h⁵)
- Global error: O(h⁴)
- 4 function evaluations per step
- No error estimate (fixed step only)

### 1.3 Forward Euler

1 stage, 1st-order. Baseline for testing.

```
yₙ₊₁ = yₙ + h·f(tₙ, yₙ)
```

**Properties**:
- Local error: O(h²), global error: O(h)
- Stability region: |1 + hλ| < 1 (circle of radius 1 centered at -1)
- Unstable for stiff systems (need implicit methods)

### 1.4 Midpoint Method (RK2)

```
k₁ = f(tₙ, yₙ)
k₂ = f(tₙ + h/2, yₙ + h·k₁/2)
yₙ₊₁ = yₙ + h·k₂
```

**Properties**: 2nd-order, 2 evaluations/step.

### 1.5 Heun's Method (Improved Euler / Trapezoidal)

```
k₁ = f(tₙ, yₙ)
k₂ = f(tₙ + h, yₙ + h·k₁)
yₙ₊₁ = yₙ + (h/2)(k₁ + k₂)
```

**Properties**: 2nd-order, predictor-corrector structure.

---

## 2. Adaptive Step Control (RK45 Dormand-Prince)

### 2.1 Embedded Methods

An embedded Runge-Kutta pair uses the SAME stages to compute two solutions of different orders. The difference is the error estimate.

**Dormand-Prince 4(5)** — the standard RK45 (used by MATLAB `ode45`, scipy `solve_ivp(method='RK45')`):

7 stages, 5th-order solution with embedded 4th-order error estimate.

```
Butcher tableau (Dormand-Prince):

0       |
1/5     | 1/5
3/10    | 3/40       9/40
4/5     | 44/45      -56/15      32/9
8/9     | 19372/6561 -25360/2187 64448/6561  -212/729
1       | 9017/3168  -355/33     46732/5247  49/176    -5103/18656
1       | 35/384     0           500/1113    125/192   -2187/6784   11/84
--------|---------------------------------------------------------------
5th     | 35/384     0           500/1113    125/192   -2187/6784   11/84     0
4th     | 5179/57600 0           7571/16695  393/640   -92097/339200 187/2100 1/40
```

**Error estimate**: err = |ŷ₅ - ŷ₄| (difference between 5th and 4th order solutions).

### 2.2 Step Size Control

**Standard PI controller** (Hairer-Nørsett-Wanner):

```
h_new = h · min(fac_max, max(fac_min, fac · (tol/err)^(1/q)))
```

where:
- `q` = min(order_low, order_high) = 4 for RK45
- `fac` = safety factor = 0.9 (or 0.8)
- `fac_max` = maximum step increase = 5.0
- `fac_min` = minimum step decrease = 0.2
- `tol` = user tolerance

**Error norm** (mixed absolute-relative):

```
err = √((1/d) Σᵢ (errᵢ / (atol + rtol·max(|yₙ,ᵢ|, |yₙ₊₁,ᵢ|)))²)
```

where d = dimension of the ODE system.

**Accept/reject**: If err ≤ 1.0, accept step. Otherwise reject and retry with smaller h.

### 2.3 Dense Output (Continuous Extension)

For output at arbitrary times within [tₙ, tₙ₊₁], the Dormand-Prince method provides a 4th-order interpolant:

```
y(tₙ + θh) = yₙ + h Σᵢ bᵢ*(θ) kᵢ
```

where bᵢ*(θ) are polynomials in θ ∈ [0,1]. This avoids storing intermediate states — the MSR is the stage values k₁...k₇.

---

## 3. Stiff Systems

### 3.1 Stiffness Detection

A system is stiff when eigenvalues of the Jacobian ∂f/∂y span a wide range (large ratio λ_max/λ_min). Explicit methods require h proportional to 1/|λ_max|, making them impractical.

**Detection heuristic** (Shampine):
```
If the ratio (rejected steps / total steps) exceeds 0.5 after initial transient,
the system is likely stiff. Switch to implicit method.
```

**Practical test**: If error controller repeatedly shrinks h below a threshold relative to the time scale of interest, the problem is likely stiff.

### 3.2 Implicit Methods (for stiff systems)

**Backward Euler**:
```
yₙ₊₁ = yₙ + h·f(tₙ₊₁, yₙ₊₁)
```
Requires solving a nonlinear system at each step (Newton iteration).

**BDF (Backward Differentiation Formulas)**:
MATLAB's `ode15s`, scipy's `BDF` method. Multistep methods using previous solution values.

**Radau IIA** (5th-order implicit RK): scipy's `Radau` method. 3-stage implicit RK with excellent stability.

### 3.3 Implicit Method Architecture

For implicit methods, each step requires:
1. Newton iteration: solve `G(yₙ₊₁) = yₙ₊₁ - yₙ - h·f(tₙ₊₁, yₙ₊₁) = 0`
2. Jacobian computation: `J = ∂f/∂y` (analytical or finite-difference)
3. Linear solve: `(I - h·J)·Δy = -G` at each Newton step

This connects to F02 (linear algebra) — the linear solve uses LU factorization, and the Jacobian may be dense (O(d²) storage) or sparse (banded/block structure).

---

## 4. Systems of ODEs

### 4.1 Vector ODE

A system of d first-order ODEs:

```
dy₁/dt = f₁(t, y₁, y₂, ..., y_d)
dy₂/dt = f₂(t, y₁, y₂, ..., y_d)
...
dy_d/dt = f_d(t, y₁, y₂, ..., y_d)
```

All RK methods apply componentwise — the Butcher tableau is the same, but k₁...kₛ are now vectors in ℝᵈ.

### 4.2 Higher-Order ODEs → First-Order System

An nth-order ODE y^(n) = g(t, y, y', ..., y^(n-1)) converts to a first-order system:

```
u₁ = y,  u₂ = y',  ...,  uₙ = y^(n-1)
du₁/dt = u₂
du₂/dt = u₃
...
duₙ/dt = g(t, u₁, ..., uₙ)
```

Dimension d = n.

---

## 5. Numerical Stability

### 5.1 Stability Region

The stability function of an explicit RK method for the test equation y' = λy:

```
R(z) = 1 + z + z²/2! + ... + z^s/s!    (for an s-stage method)
```

where z = hλ. Stability requires |R(z)| < 1.

| Method | Stability Boundary on Real Axis | Max Stable |hλ| |
|--------|-------------------------------|------------|
| Euler | -2 ≤ hλ_real ≤ 0 | 2.0 |
| RK2 | wider | ~2.0 |
| RK4 | much wider | ~2.8 |
| RK45 | similar to RK4 | ~3.3 |
| Backward Euler | entire left half-plane | ∞ (A-stable) |

### 5.2 Error Accumulation

Global error grows as O(e^{Lt} · h^p) where L is the Lipschitz constant of f. For long integrations:
- Stable systems (Re(λ) < 0): errors bounded
- Neutral systems (Re(λ) = 0): errors grow linearly — chaotic systems (Lyapunov > 0) grow exponentially regardless of method order

### 5.3 f32 vs f64 Implications

For ODE solvers, f32 is problematic because:
1. **Accumulated roundoff**: Each step adds O(ε_mach) error. Over N steps, total ≈ N·ε_mach.
   - f64: 1e6 steps → ~1e-10 accumulated error
   - f32: 1e6 steps → ~1e-1 accumulated error (**UNUSABLE**)
2. **Step rejection**: Error estimate needs precision to distinguish tol from err
3. **Stiff Jacobian**: J eigenvalue computation needs precision

**Design rule**: ODE solvers MUST use f64. f32 is unsuitable for any multi-step integration.

---

## 6. Edge Cases and Failure Modes

| Scenario | Behavior | Mitigation |
|----------|----------|------------|
| f evaluates to NaN | Propagates → solution poisoned | Check f output, reject step |
| h → 0 (stiffness) | Infinite loop or exceeds max_steps | Stiffness detector, switch to implicit |
| y → ±∞ (blowup) | ODE genuinely diverges | Report "solution escaped" with last good y |
| Discontinuity in f | Accuracy loss at discontinuity | Event detection, restart integrator at event |
| Very long integration (T >> 1/h) | Error accumulation | Compensated summation for t + h |
| Highly oscillatory f | Needs many steps per period | Specialized methods (Gautschi, trigonometric RK) |
| Symplectic systems (Hamiltonian) | Standard RK drifts energy | Use symplectic integrators (Störmer-Verlet) |
| t₀ + h == t₀ (f64 stagnation) | h so small that t + h rounds to t | Raise error: "step size too small at t=..." |

### Compensated Time Accumulation

After many steps, `t += h` accumulates roundoff. When t is large and h is small:
```
// BAD:
t += h;  // loses low bits of h

// GOOD (compensated):
let temp = t + h;
let error = (temp - t) - h;
t = temp;
// carry error forward to next step
```

This is the same Kahan summation principle as in F06 descriptive statistics.

---

## 7. Implementation Priority

**Phase 1 — Core explicit methods**:
1. Forward Euler (testing baseline)
2. Classical RK4 (fixed step)
3. Dormand-Prince RK45 with adaptive stepping
4. Systems of ODEs (vector RK)

**Phase 2 — Dense output and events**:
5. Continuous extension (Dormand-Prince interpolant)
6. Event detection (root finding during integration)
7. Higher-order ODE → system conversion

**Phase 3 — Stiff solvers**:
8. Backward Euler (simplest implicit)
9. BDF methods (multistep)
10. Stiffness detection heuristic

---

## 8. Composability Contract Template

```toml
[algorithm]
name = "rk45_dormand_prince"
family = "ode.adaptive"

[inputs]
required = ["f: (f64, &[f64]) -> Vec<f64>", "t_span: (f64, f64)", "y0: Vec<f64>"]
optional = ["rtol: f64 = 1e-6", "atol: f64 = 1e-9", "max_step: f64", "max_steps: usize = 100000"]

[outputs]
primary = "OdeSolution { t: Vec<f64>, y: Vec<Vec<f64>>, n_eval: usize, n_reject: usize }"
secondary = ["dense_output: ContinuousExtension"]

[sufficient_stats]
consumes = ["f", "y0", "t_span"]
produces = ["y_final", "t_history", "y_history", "stage_values"]

[sharing]
provides_to_session = ["JacobianApprox(f, y)"]
consumes_from_session = ["JacobianApprox"]

[assumptions]
requires_sorted = false
requires_positive = false
requires_no_nan = true  # f must not return NaN
minimum_d = 1
f_must_be_lipschitz = true  # for convergence guarantee
f64_only = true  # f32 unsuitable for multi-step integration
```

---

## 9. Connection to Other Families

- **F02 (Linear Algebra)**: Implicit methods need linear solves (LU, Jacobian)
- **F05 (Optimization)**: Newton iteration for implicit step = optimization subproblem
- **F17 (Time Series)**: ARIMA state-space form is a linear ODE; EWM is a discretized exponential decay ODE
- **F26 (Complexity/Chaos)**: Lyapunov exponents computed by integrating variational equations alongside the ODE
- **F23/F24 (Neural Networks/Training)**: Neural ODEs (Chen et al. 2018) use ODE solvers as layers; adjoint method for gradients

---

## 10. Key References

- Hairer, Nørsett, Wanner (1993). "Solving Ordinary Differential Equations I: Nonstiff Problems." Springer.
- Hairer, Wanner (1996). "Solving Ordinary Differential Equations II: Stiff and Differential-Algebraic Problems." Springer.
- Dormand, Prince (1980). "A family of embedded Runge-Kutta formulae." J. Comp. Appl. Math.
- Shampine, Reichelt (1997). "The MATLAB ODE Suite." SIAM J. Sci. Comput. (documents ode45/ode15s design).
