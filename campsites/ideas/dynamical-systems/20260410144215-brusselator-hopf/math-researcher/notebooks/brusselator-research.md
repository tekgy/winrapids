<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

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


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

