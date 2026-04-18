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

# Gray-Scott Reaction-Diffusion — Mathematical Research

## The System

Pearson (1993), "Complex patterns in a simple system", Science 261(5118):189-192.

Two coupled reaction-diffusion PDEs:

```
∂u/∂t = D_u ∇²u - uv² + F(1-u)
∂v/∂t = D_v ∇²v + uv² - (F+k)v
```

Where:
- u, v: concentrations of two chemical species
- D_u, D_v: diffusion coefficients (D_u > D_v for Turing patterns)
- F: feed rate (controls inflow of u)
- k: kill rate (controls removal of v)
- uv²: autocatalytic reaction term

## Why This Matters for Tambear

### 1. PDE Solver Primitives

Gray-Scott is the "Hello World" of PDE simulation. Implementing it properly requires:

**Laplacian (∇²)**: 2D finite difference stencil
```
∇²u(i,j) ≈ (u(i+1,j) + u(i-1,j) + u(i,j+1) + u(i,j-1) - 4u(i,j)) / h²
```
This is a stencil operation — the quintessential GPU kernel pattern. The Laplacian stencil
composes as: `accumulate(Stencil2D(3×3), weighted_sum)`. Kingdom A on the interior;
boundary handling is the only complication.

**Time stepping**: Forward Euler for simplicity, but the diffusion term is stiff:
- Explicit: stable only if Δt < h²/(4D_u) (CFL condition)
- Semi-implicit (IMEX): treat diffusion implicitly, reaction explicitly
- The implicit diffusion step is a tridiagonal solve per dimension (ADI)
  → uses `solve_tridiagonal` which is already a Kingdom A scan primitive!

### 2. Turing Instability — Bifurcation Analysis

The homogeneous steady state (u*, v*) satisfies:
- F(1-u*) = u*v*² → u* is the pre-pattern concentration
- (F+k)v* = u*v*² → v* relates to activation level

Linear stability analysis: the Jacobian of the reaction terms is:
```
J = [-v*² - F      -2u*v*   ]
    [ v*²       2u*v* - (F+k)]
```

Without diffusion: Hopf bifurcation (like Brusselator) when tr(J) changes sign.

WITH diffusion: Turing instability when:
1. Homogeneous state is stable (tr(J) < 0, det(J) > 0)
2. But adding diffusion with D_u ≫ D_v makes some spatial mode unstable

Turing condition: there exists wavenumber q such that:
det(J - D·q²·I) < 0 where D = diag(D_u, D_v)

This gives: D_v·j₁₁ + D_u·j₂₂ > 2·√(D_u·D_v·det(J))

### 3. Pattern Zoo

The (F, k) parameter space produces an extraordinary pattern zoo:

| Pattern | F range | k range | Description |
|---|---|---|---|
| Spots | 0.025-0.035 | 0.055-0.060 | Isolated self-replicating dots |
| Stripes | 0.025-0.040 | 0.055-0.065 | Labyrinthine patterns |
| Worms | 0.040-0.050 | 0.060-0.065 | Moving filaments |
| Mitosis | 0.030-0.035 | 0.057-0.060 | Splitting spots |
| Coral | 0.050-0.055 | 0.062-0.064 | Branching structures |
| Chaos | 0.015-0.020 | 0.050-0.055 | Turbulent patterns |

Each pattern type has its own TDA signature (persistent homology), its own spectral
fingerprint (2D FFT), and its own complexity measures (permutation entropy on the
spatial field).

### 4. Primitives This Generates

**Level 0 — Pure math primitives**:
1. `laplacian_2d(field, dx)` — 5-point finite difference stencil
2. `reaction_gray_scott(u, v, f, k)` — pointwise reaction terms
3. `euler_forward_2d(field, rhs, dt)` — explicit time step

**Level 1 — Methods**:
4. `gray_scott_simulate(u0, v0, params, dt, n_steps)` — full simulation
5. `turing_analysis(J, D_u, D_v)` — linear stability for Turing modes
6. `pattern_classify(field)` — classify pattern type from steady state

**Compositions with existing primitives**:
- `fft_2d(field)` → spectral analysis of patterns
- `persistent_homology(sublevel_sets(field))` → TDA of pattern topology
- `permutation_entropy(field_slice)` → spatial complexity
- `mfdfa(time_series_of_global_mean)` → temporal multifractal behavior

### 5. The GPU Kernel Structure

```
// Reaction step: pointwise (embarrassingly parallel)
for each (i,j):
    du = -u[i,j]*v[i,j]*v[i,j] + F*(1.0 - u[i,j])
    dv =  u[i,j]*v[i,j]*v[i,j] - (F+k)*v[i,j]

// Diffusion step: stencil (neighbor access pattern)
for each (i,j):
    lap_u = (u[i+1,j] + u[i-1,j] + u[i,j+1] + u[i,j-1] - 4*u[i,j]) / dx²
    lap_v = (v[i+1,j] + v[i-1,j] + v[i,j+1] + v[i,j-1] - 4*v[i,j]) / dx²
    u_new = u + dt * (D_u * lap_u + du)
    v_new = v + dt * (D_v * lap_v + dv)
```

This is accumulate(Grid2D, reaction_diffusion_rhs, Add) — a single fused kernel.
The stencil access pattern is a 5-point cross, perfectly regular.
On GPU: each thread handles one grid point, reads 4 neighbors from shared memory.

### 6. Connection to Market Patterns

**Structural rhyme**: the (F, k) parameter space of Gray-Scott mirrors the
(alpha+beta, leverage) parameter space of GARCH variants. Both have:
- Stability regions (mean-reverting vol / stable homogeneous state)
- Bifurcation boundaries (unit root / Turing instability)
- Pattern diversity beyond the boundary (vol clustering types / spatial patterns)

The question is whether financial "patterns" (vol regimes, correlation structure changes)
are better described as reaction-diffusion on a cross-asset spatial graph than as
independent 1D processes. If yes, the GARCH-per-asset model is the wrong dimensional
projection — and Gray-Scott-on-graph would be the correct K04+ model.

## Implementation Priority

For tambear proper: LOW. This is an ideas campsite.

But the PRIMITIVES it would exercise (stencil operations, 2D FFT, Turing analysis)
are independently important:
- `laplacian_2d` is needed for: image processing, spatial statistics, mesh processing
- 2D FFT is needed for: spatial spectral analysis, convolution
- Turing analysis is needed for: any spatially-coupled dynamical system

The playground value is HIGH — Gray-Scott simulations are visually stunning and
would make an excellent interactive demonstration of tambear's GPU capabilities.


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

