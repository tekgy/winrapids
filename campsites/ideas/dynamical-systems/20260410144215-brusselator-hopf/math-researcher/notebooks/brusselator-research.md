# Brusselator-Hopf — Mathematical Research Notes

## What Exists

`tambear::numerical` already has:
- `brusselator_rhs(state, a, b)` — the ODE
- `brusselator_jacobian(state, b)` — 2x2 Jacobian
- `brusselator_bifurcation(a, b)` — fixed point analysis
- `brusselator_simulate(a, b, x0, y0, t_end, n_steps)` — RK4 trajectory

The doc comment already notes the structural rhyme: "bifurcation distance is analogous
to GARCH persistence (α+β - 1)."

## The Hopf Bifurcation — Original Paper

**Prigogine & Lefever (1968)**: "Symmetry-Breaking Instabilities in Dissipative Systems II"

System: dx/dt = a - (b+1)x + x²y, dy/dt = bx - x²y

Fixed point: (x*, y*) = (a, b/a)

Jacobian at fixed point:
```
J = [b-1   a²]
    [-b   -a²]
```

Eigenvalues: λ = ½(tr ± √(tr² - 4 det))
- tr(J) = b - 1 - a²
- det(J) = a²

Hopf bifurcation at tr = 0 → **b_c = 1 + a²**

Below b_c: stable spiral (complex eigenvalues with negative real part)
Above b_c: unstable spiral → limit cycle (supercritical Hopf)

## Why This Matters for Tambear

### 1. Bifurcation detection as a primitive

The Brusselator is the simplest system that exhibits Hopf bifurcation. The detection
algorithm is:
1. Compute Jacobian at equilibrium
2. Track eigenvalues as parameter varies
3. Bifurcation when: real part crosses zero, imaginary part nonzero

This decomposes into: `eigendecomposition(jacobian)` → `track_eigenvalue_crossing(params)`

For financial markets: the "parameter" could be GARCH persistence, and the "bifurcation"
is the regime transition between mean-reverting and explosive volatility.

### 2. Critical slowing down as a precursor signal

Near Hopf bifurcation, the system exhibits:
- Increasing autocorrelation (ACF decay rate → 0)
- Increasing variance
- Spectral narrowing (power concentrates at critical frequency)

These are exactly the DFA/ACF/spectral features that fintek already computes.
The insight: these features aren't arbitrary — they're manifestations of proximity
to a phase boundary in the dynamical system sense.

### 3. Amplitude equation (normal form)

Near bifurcation, the Brusselator reduces to the Stuart-Landau equation:
dA/dt = μA - |A|²A (after appropriate coordinate transform)

where μ = (b - b_c) / (2a²) is the reduced bifurcation parameter.

This is the UNIVERSAL normal form for supercritical Hopf. Any system near Hopf
bifurcation looks like this. This means:
- Amplitude grows as √μ (supercritical)
- Frequency at bifurcation: ω = a (the determinant's square root)
- Transient decay rate: proportional to |μ|

### Primitives to extract

1. `hopf_bifurcation_analysis(jacobian_fn, param_range)` — sweep parameter, track eigenvalues
2. `critical_slowing_down(time_series)` — ACF increase + variance increase + spectral narrowing
3. `stuart_landau_fit(oscillation_amplitude, param_distance)` — fit normal form

## Connection to Gray-Scott

Gray-Scott has TWO coupled Hopf + Turing instability → spatial patterns.
The Brusselator is the non-spatial version. Gray-Scott adds diffusion:
∂u/∂t = D_u ∇²u + F(1-u) - uv²
∂v/∂t = D_v ∇²v + uv² - (F+k)v

Turing instability: homogeneous steady state unstable to spatial perturbations
when D_u ≫ D_v ("activator-inhibitor").

For tambear: Gray-Scott needs PDE solvers (finite difference on grid).
This is a natural GPU kernel — each grid point updates independently in the
reaction step; diffusion is a stencil operation.
