# Trig Family — Master Synthesis

**Author**: Aristotle (tambear-trig)
**Date**: 2026-04-14
**Status**: Deliverable for TRIG-20. Synthesis of the trig-family expedition.

**For cold readers**: this directory holds the output of the tambear-trig team's expedition to design and implement the complete trigonometric library from first principles. There are 22 docs (~7000 lines). This README tells you what's in them, what decisions have been made, and where to look for what.

**For active team members**: the decisions section below captures the consensus that emerged from the expedition. If you disagree with a decision, propose a revision — don't silently work around it.

---

## The one-paragraph summary

The trig family covers ~30 named functions (sin, cos, tan, their inverses, hyperbolics, pi-scaled variants). Rather than write 30 separate recipes, the expedition concluded that the mathematically honest decomposition is **~8 hand-written kernels** (5 range-reductions by angle unit + 3 core kernels `sincos`/`tan`/`atan2` + hyperbolic kernels via `exp`/`log`) with **~45 user-visible views** sitting on top as 1-5 line compositions. Range reduction is the most expensive step and the natural shared intermediate — we invert the conventional "each function owns its reduction" ownership so that reduction is a first-class recipe and forward-trig functions are its consumers. Angle unit is not a pre-multiply; it selects the reduction modulus, which makes degrees / gradians / turns / pi-scaled **faster and more precise** than radians because their moduli are dyadic-friendly. Precision is orthogonal to algorithm: every recipe has strict / compensated / correctly_rounded lowerings driven by a compiler pass over the same recipe tree.

---

## The doc map — what's where

Organize the ~22 docs into four layers matching tambear's architecture doc:

### Layer 0 — research & foundations

| Doc | Purpose | Owner |
|---|---|---|
| [`references.md`](references.md) | Citations (Cody & Waite, Payne & Hanek, Muller, fdlibm, etc.) | researcher |
| [`catalog.md`](catalog.md) | Master enumeration: every function, domain, codomain, edge cases, algorithmic flavor | researcher |
| [`lab-notebook.md`](lab-notebook.md) | Empirical notes during implementation | researcher |
| [`expedition-log.md`](expedition-log.md) | Naturalist's running journal — what's surprising, patterns spotted | naturalist |

### Layer 1 — first-principles deconstruction

| Doc | Purpose | Owner |
|---|---|---|
| [`first-principles.md`](first-principles.md) | 8-phase Aristotle deconstruction: 12 assumptions → 12 irreducible truths → 10 reconstructions → Aristotelian move (reduction-as-primitive) | aristotle |
| [`atoms_gaps.md`](atoms_gaps.md) | Primitive/expr/op/tag gaps the reconstruction structurally demands | aristotle |
| [`notation.md`](notation.md) | Three notation styles (publication / recipe-tree / TBS) for 30+ functions | aristotle |

### Layer 2 — architecture & algorithms

| Doc | Purpose | Owner |
|---|---|---|
| [`hardware-mapping.md`](hardware-mapping.md) | Cycle budgets per kernel across x86/ARM/PTX/SPIR-V; atom decomposition per function | aristotle |
| [`compilation.md`](compilation.md) | Hardware mapping + per-precision-strategy compilation differences (DD needed? what primitives?) | researcher |
| [`angle_units.md`](angle_units.md) | Per-unit reduction algorithms; why degrees/turns/pi-scaled are faster than radians | researcher |
| [`shared_pass.md`](shared_pass.md) / [`shared-pass-analysis.md`](shared-pass-analysis.md) | Flop-count analysis of which intermediates justify TamSession sharing tags | researcher / naturalist |
| [`variants.md`](variants.md) | Tiny/small/medium/large range variants per function (TRIG-2) | researcher |

### Layer 3 — user-facing surface & defaults

| Doc | Purpose | Owner |
|---|---|---|
| [`scientist-defaults.md`](scientist-defaults.md) | What a $1M/yr quant scientist would choose as the default per function | scientist |
| [`defaults-rationale.md`](defaults-rationale.md) | Why each default is what it is | scientist |
| [`tbs-syntax.md`](tbs-syntax.md) / [`tbs-grammar.md`](tbs-grammar.md) | TBS verb catalog: ~45 user-facing verbs wired to the executor | aristotle / researcher |
| [`autodiscover-probes.md`](autodiscover-probes.md) | 22 probes for `.discover()`: input characterization, algorithm dispatch, identity checks, scientific insight, cross-method superposition | aristotle |
| [`parity-table.md`](parity-table.md) | Bench vs scipy / R / CUDA / Julia / fdlibm per function | scientist |
| [`convergence-evidence.md`](convergence-evidence.md) | Cross-cutting rhymes / patterns spotted across the expedition | naturalist |

### Layer 4 — validation

| Doc | Purpose | Owner |
|---|---|---|
| [`validation/sin.md`](validation/sin.md) | Per-function validation notes | various |
| [`validation/exp_log.md`](validation/exp_log.md) | exp/log validation (trig dependencies) | various |
| [`validation/erf_gamma.md`](validation/erf_gamma.md) | Adjacent libm validation | various |

---

## Decisions (consensus reached during the expedition)

The following are **decisions**, not open proposals. Each carries its originating doc and a one-line rationale.

### D1. Catalog collapses to ~8 kernels + ~45 views.

Per [`first-principles.md`](first-principles.md) Phase 8 and [`hardware-mapping.md`](hardware-mapping.md), the hand-written surface is 8 kernel-level recipes:

1. `rem_pio2` (radians reduction, Cody-Waite + Payne-Hanek)
2. `rem_half_turn` (pi-scaled — exact reduction mod 0.5)
3. `rem_degrees_90` (degrees — exact-for-practical-inputs reduction mod 90)
4. `rem_gradians_100` (gradians — exact mod 50)
5. `rem_turns_quarter` (turns — exact mod 0.25)
6. `sincos_kernel` (forward polynomial kernel on reduced input)
7. `tan_kernel` (fused tangent polynomial)
8. `atan2_kernel` (inverse)

Plus three hyperbolic kernels (`sinh_kernel`, `cosh_kernel`, `tanh_kernel`) that are regime-dispatched wrappers around `exp`.

Everything else is a 1-5 line view. This cuts implementation work by ~75% vs the naive 30-separate-recipes approach.

**Rationale**: tan = sincos().s / sincos().c, asin = atan2(x, sqrt(1-x²)), asinh = log(x + sqrt(x²+1)), etc. These compositions are **identities**, not approximations. Giving each its own recipe is duplication.

### D2. Range reduction is the shared intermediate.

Per [`shared_pass.md`](shared_pass.md) and [`first-principles.md`](first-principles.md) Phase 5. A single `TrigReduce::{Unit}(col_key)` tag feeds all forward-trig consumers on the same column. Computed once per column per session; retrieved by downstream `sin`, `cos`, `tan`, `sincos`, `sec`, `csc`, `cot`.

**Rationale**: the reduction is ~45 cycles for radians, dominating the ~100-cycle kernel cost. Sharing it across N forward functions on the same column divides the cost by N.

### D3. Angle unit is a reduction-modulus selector, not a pre-multiply.

Per [`angle_units.md`](angle_units.md) and [`first-principles.md`](first-principles.md) A9. Degrees reduces mod 90 (exact integer arithmetic for practical inputs). Gradians reduces mod 50. Turns reduces mod 0.25 (exact, dyadic). Pi-scaled reduces mod 0.5 (exact, dyadic). Radians reduces mod π/2 via Cody-Waite + Payne-Hanek (nontrivially expensive).

**Result**: non-radian units are **faster AND more accurate** than radians. This inverts a naive expectation — document it prominently in user-facing material.

**Rationale**: pre-multiplying `x` by π/180 (to convert degrees to radians before reduction) introduces ~1 ulp error proportional to `|x|`. Reducing mod 90 first and multiplying the small residual by π/180 keeps error bounded by 1 ulp independent of `|x|`.

### D4. Scientist default: compensated precision, radians, auto reduction.

Per [`scientist-defaults.md`](scientist-defaults.md). Compensated gives 1-ulp worst-case error for ~10% overhead over strict; worth it universally.

**Rationale**: correctly_rounded costs 3-4× and is for publishing, not production. Strict saves 10% over compensated and costs 1 extra ulp — bad trade when the 10% doesn't matter but the extra ulp might.

### D5. The one structural gap we still need is tuple-valued `accumulate` exprs.

Per [`atoms_gaps.md`](atoms_gaps.md) S7. Every reduction produces `(q, r_hi, r_lo)` and `sincos_kernel` produces `(c, s)`. The current `accumulate` atom expects scalar-valued exprs. Without tuple support, we either pack into f128-layout tricks or do two passes wasting half the kernel work.

**Status**: proposed; navigator to route to the accumulate-atom maintainers. This is load-bearing beyond trig (Givens rotations, modf, frexp, complex multiply, quaternions all want it).

### D6. Hyperbolics decompose to `exp` + regime dispatch.

Per [`hardware-mapping.md`](hardware-mapping.md) K8. Three regimes per function: Taylor near 0 (for cancellation), `(exp(x) ± exp(-x))/2` in the middle, overflow-safe scaled form at the top. Regime boundaries at ~1, ~22, ~709.

**Rationale**: the naive formula overflows above 709 and cancels badly near 0. Regime dispatch costs nothing at runtime (mostly predicated arithmetic).

### D7. Three precision strategies = one recipe tree, three compiler lowerings.

Per [`compilation.md`](compilation.md). `strict` emits hardware primitives directly; `compensated` lifts to DoubleDouble; `correctly_rounded` adds final rounding correction via DoubleDouble throughout.

**Status**: architecturally correct; current implementation stubs compensated and correctly_rounded as aliases for strict. True compiler-pass lowering blocks on the `.tam` IR (Peak 1). Acceptable — strict is within 1.5 ulp on tested domain for sin/cos, and the stubs can be replaced without user-visible changes.

### D8. The `sincos_kernel` fundamental domain is `[-π/4, π/4]`.

Per existing sin.rs + [`compilation.md`](compilation.md). Smallest minimax polynomial error. Kernel expects this domain; reduction produces this domain; they are matched.

**Alternative domains** (CORDIC on `[0, π/2)`, table lookup on `[0, 1)`-turns) are future kernels under `sincos_kernel.using(method=...)`.

### D9. `atan2` is the inverse primitive; `asin`/`acos`/`atan` are one-liners.

Per [`first-principles.md`](first-principles.md) Phase 6.4 and [`notation.md`](notation.md). Complex log is the deeper mathematical primitive, but atan2 is kept as a distinct recipe for real-input branch-cut-correctness.

### D10. Non-standard verbs (`versin`, `haversin`, `gudermannian`, `haversine_distance`) ship as top-level verbs.

Per [`tbs-syntax.md`](tbs-syntax.md). Rationale: they have literature names and real use cases (navigation, great-circle distance); users shouldn't have to re-derive `1-cos(x)` every time. Cost to ship them is ~5 lines each.

---

## Open questions (not yet decided)

These carry over — if you find a resolution, update this README + the specific doc.

### OQ1. Tuple-valued accumulate expr (D5 dependency).

Status: proposed in `atoms_gaps.md`, not yet confirmed with accumulate-atom maintainers. Blocks clean implementation of reductions that produce `(q, r_hi, r_lo)` and kernels that produce `(c, s)`.

### OQ2. CORDIC and table-lookup kernels as future `sincos_kernel.using(method=...)`.

Status: deferred. Polynomial kernel ships first. CORDIC attractive for FPGAs/embedded; table-lookup attractive for GPU texture units. Would unlock `autodiscover-probes.md` P5.1 (kernel superposition).

### OQ3. Complex-number-native forward trig.

Status: deferred. `cis(θ) = cos θ + i·sin θ` as the primitive is mathematically cleaner (unifies trig with rotations + complex exp). But tambear doesn't yet have a first-class complex type. Park at the `tambear::geometry` milestone.

### OQ4. Compensated libm dispatch implementation.

Status: blocked on `.tam` IR (Peak 1). Until then, hand-write compensated variants for critical functions (sin, exp — the ones other libm recipes depend on).

### OQ5. Special-value fast path for dyadic inputs.

`sinpi(0.5) = 1.0` exactly; `sinpi(1/3)` does not simplify exactly. But `sinpi(k/2^n)` for small n has closed-form DoubleDouble values. Worth a fast-path check? Defer — low-frequency optimization.

---

## Cross-cutting patterns

These are structural observations spotted across multiple docs, worth naming because they show up again in future families.

### P1. The three-stage shape: reduce → kernel → view.

Per [`hardware-mapping.md`](hardware-mapping.md) and [`expedition-log.md`](expedition-log.md). Every forward trig operation on a column is:

```
Stage 1 (reduce):  accumulate(Pointwise, Rem{Unit}, Concat) → (q, r_hi, r_lo)
Stage 2 (kernel):  accumulate(Pointwise, SinCosKernel, Concat) → (c, s)
Stage 3 (view):    elementwise primitives (fdiv, fmul, fsqrt)
```

**This rhymes with:**
- **FFT family**: fft_reduce → butterfly kernel → spectral view (PSD, spectrum, MTM)
- **Moment stats**: welford accumulation → moment extraction → descriptive views (mean, var, skew, kurt)
- **Covariance family**: centering reduce → gram kernel → projection view (PCA, LDA, factor)

**Hypothesis**: this is a library-wide architectural shape, not trig-specific. Convergence-check candidate.

### P2. Angle unit as reduction-modulus selector generalizes.

The pattern "unit selects the reduction modulus, not a pre-multiply" applies anywhere a function is periodic or scale-invariant. For example:
- Frequency unit (Hz vs rad/s) in spectral functions.
- Log base (ln vs log10 vs log2) in logarithms.
- Temperature scale (K vs C vs F) in thermodynamic functions.

Each has a "pre-multiply loses precision" trap; each has a "select the native modulus" escape.

### P3. Aliases dispatch to one body via the (name, method) tuple.

Per [`tbs-syntax.md`](tbs-syntax.md). The TBS executor uses `match (step.name, step.method)` with or-patterns to route aliases (`test.t` | `t_test`). Trig's 45 verbs collapse to ~6 dispatch groups. This pattern is already established in the codebase — trig reuses it.

### P4. Identity checks are probes; probes are adversarial tests.

Per [`autodiscover-probes.md`](autodiscover-probes.md). The pythagorean probe (`sin² + cos² = 1`) is a cheap CI check, a runtime sanity probe, AND an adversarial test. Same code; differs only in whether it runs on user data, synthetic data, or curated evil data.

---

## How this rolls up to pathmaker / the ship list

Forward implementation (pathmaker and researcher are handling these):
- [x] TRIG-12: spec.tomls for every function (32 shipped)
- [x] TRIG-13: forward Rust recipes (sin/cos/tan/cot/sec/csc/sincos)
- [ ] TRIG-14: inverse trig (asin/acos/atan/atan2/acot/asec/acsc) — in flight
- [ ] TRIG-15: hyperbolics
- [ ] TRIG-16: pi-scaled
- [ ] TBS executor wiring — mechanical once the Rust recipes expose `*_with(precision, angle_unit, col)` entry points (~500 lines of match arms + shared helper)

Validation (adversarial + scientist):
- [x] TRIG-17: adversarial test battery
- [ ] TRIG-18: gold-standard parity (R/Python/CUDA) — in flight

Not yet started:
- [ ] TRIG-21: scout recon (Julia/CUDA/Arm/fdlibm catalog) — literature survey

---

## If you're new to the expedition — where to start reading

**Scenario A: I need to implement a trig recipe.**
1. Read [`catalog.md`](catalog.md) for the function you're building.
2. Read [`first-principles.md`](first-principles.md) Phase 8 for the view/kernel split.
3. Read the function's `.spec.toml` (under `crates/tambear/src/recipes/libm/`).
4. Read [`compilation.md`](compilation.md) or [`hardware-mapping.md`](hardware-mapping.md) for the primitives you'll call.
5. Read [`validation/<function>.md`](validation/) for what correctness looks like.

**Scenario B: I need to use trig from TBS.**
1. Read [`tbs-syntax.md`](tbs-syntax.md) for the verb you want.
2. Read [`scientist-defaults.md`](scientist-defaults.md) for what happens if you don't pass `.using()`.
3. Use [`autodiscover-probes.md`](autodiscover-probes.md) via `.discover()` if unsure about your input.

**Scenario C: I'm reviewing a PR.**
1. [`first-principles.md`](first-principles.md) Phase 8 — does the PR match the decomposition?
2. [`atoms_gaps.md`](atoms_gaps.md) — does it introduce new atoms without justification?
3. [`scientist-defaults.md`](scientist-defaults.md) — does it respect the default?
4. [`parity-table.md`](parity-table.md) — how does it measure vs reference implementations?

**Scenario D: I'm writing a related family (exp-family extensions, spectral, geometry).**
1. [`expedition-log.md`](expedition-log.md) for the naturalist's running journal — patterns found.
2. Cross-cutting patterns section above — apply the three-stage shape (P1) if it fits.
3. [`angle_units.md`](angle_units.md) — if your family has periodicity or scale-invariance, apply P2.

---

## Expedition metrics

| Metric | Count |
|---|---|
| Research docs in this directory | 22 |
| Total lines of research | ~7000 |
| Hand-written kernel recipes (target) | 8 (+ 3 hyperbolic regime dispatches) |
| User-visible TBS verbs | ~45 |
| `.spec.toml` files shipped | 32 |
| Rust recipes shipped | TRIG-13 scope (forward) |
| Adversarial tests | TRIG-17 scope |

---

## Acknowledgements

The expedition was a team effort:
- **Researcher (math)**: catalog, variants, angle units, shared-pass, compilation, hardware mapping, TBS grammar — the mathematical truth-keeper.
- **Pathmaker**: `.spec.toml` for every function, Rust recipes for forward trig — the build-order executor.
- **Scientist**: scientist-defaults, parity tables — the $1M/yr taste-checker.
- **Adversarial**: adversarial test battery — the hostile-input harness.
- **Naturalist**: expedition log, convergence evidence — the pattern-spotter.
- **Aristotle (me)**: first-principles deconstruction, atom gaps, notation, hardware mapping, autodiscover probes, TBS syntax, this synthesis — the assumption-questioner.
- **Navigator**: team coordination — the route-keeper.

Each role was distinct and necessary. The deconstruction would have been empty without the catalog's math; the math would have been disconnected without the deconstruction's structure; the spec.tomls would have been premature without both; the defaults would have been arbitrary without the scientist's judgment; the tests would have been toothless without the adversarial harness; the patterns would have been invisible without the naturalist's patience. Journey before destination.

---

## Status

- [x] Doc map (4 layers, 22 docs)
- [x] 10 expedition decisions (D1–D10) with rationale
- [x] 5 open questions (OQ1–OQ5)
- [x] 4 cross-cutting patterns (P1–P4)
- [x] Reading guide for 4 scenarios
- [x] Expedition metrics
- [x] Acknowledgements

This README IS the TRIG-20 master synthesis. Not a summary of every doc — an orientation tool that tells a cold reader where to find what and what the team decided.
