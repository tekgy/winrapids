# What Is Trig Really? — First-Principles Deconstruction

**Author**: Aristotle (tambear-trig)
**Date**: 2026-04-13
**Status**: Deliverable for TRIG-19. Feeds TRIG-10 (atoms/exprs/ops gaps) and TRIG-8 (notation).

**Reading instructions for the team**: Phases 1-5 are the deconstruction proper; Phase 6 is the recursive rejection loop; Phase 7 fixes the stable truths; Phase 8 is the reconstruction recommendation with concrete architectural consequences that pathmaker should build against. Navigator: Phase 5 and Phase 8 are where your decisions live. Math-researcher: Phase 3 reconstructions connect to your HOW-level decomposition.

---

## Phase 1 — Assumption autopsy

Every bullet below is an assumption I found embedded in the existing sin.rs, sin.spec.toml, and task list. For each: what the assumption says, where it comes from (historically), and whether it survives scrutiny.

### A1. "Trig is about angles and triangles."

*Source*: etymology (*tri-gonon* = three-angled). Encoded in every elementary textbook and in the `angle_unit` parameter of the existing spec.

*Challenge*: The circle definition is already a generalization beyond triangles. Euler's identity `e^(iθ) = cos θ + i sin θ` drops triangles entirely: trig functions are projections of the point `e^(iθ)` onto the real and imaginary axes. Angles don't have primacy — **phase** does. Phase is a coordinate on the circle group S¹, which can be parameterized by radians, turns, degrees, or by its natural embedding in ℂ (no scalar coordinate at all).

*Verdict*: Angles are a convention for coordinatizing S¹. Rotation/phase is the real object.

### A2. "Radians are the natural unit."

*Source*: calculus (d/dθ sin θ = cos θ only in radians). Adopted as "natural" because derivatives are clean.

*Challenge*: Radians are natural **only for calculus**. For signal processing, `turns` (cycles) are natural: frequencies are cycles per second, phase wraps naturally at integer turns, FFT bins are indexed by fractional turns. For GPU shaders and graphics, `turns` or `[0,1)` parameterizations are natural. For CORDIC, binary-fraction angles are natural. For `sinpi(x)`, the input is already in half-turns. The word "natural" hides a choice of calculus-centric reference frame.

*Verdict*: No angle unit is inherently natural. The natural representation depends on the consumer's math. `radians` is a default, not a primitive.

### A3. "We need separate functions for sin/cos/tan."

*Source*: trigonometry pedagogy, most libms, IEEE 754 recommended functions.

*Challenge*: On any circle point, the pair `(cos θ, sin θ)` is one object — a unit vector. Splitting it wastes range reduction (the expensive step). `tan = sin/cos` is an exact algebraic consequence. `sec = 1/cos`, `csc = 1/sin`, `cot = cos/sin` — all derived. The literature even has a name for the fused pair: `sincos`. Splitting the object into separate functions is a C89 API artifact, not a mathematical choice.

*Verdict*: The pair (c, s) is the object. Individual components are projections. `tan/sec/csc/cot` are derived ratios.

### A4. "We need polynomial approximations."

*Source*: fdlibm, Muller ch. 11. Load-bearing for software libms on IEEE-754 hardware without dedicated trig instructions.

*Challenge*: Polynomials are one approach. Alternatives:
- **CORDIC** — iterative rotate-by-known-angles; no multiplies, all shifts+adds; great on FPGAs/embedded; ~1 bit per iteration.
- **Table lookup + small correction** — large precomputed table, small polynomial fixup; fast on GPUs with texture units.
- **Bhaskara approximations** — rational functions, lower order than polynomials for the same accuracy; older literature.
- **Newton iteration on cos² + sin² = 1** — compute s approximately, refine c = sqrt(1 - s²) and renormalize.
- **Direct AGM / theta function** — unusual but exists (high-precision libraries).
- **Half-angle doubling** — compute on a shrunken interval, double repeatedly (CORDIC-adjacent).
- **Bit manipulation** — `sinpi(p/2^k)` for dyadic inputs has closed-form double-double values; exploit when detectable.

Polynomials win in f64-on-scalar-CPU because FMA is one cycle and tables cost cache. That's a hardware fact, not a mathematical necessity. On a different architecture (SRAM-rich TPU, GPU with texture cache, FPGA) a different method wins.

*Verdict*: Polynomials are the current-hardware winner, not the definition. A general library should carry the algorithm as a parameter.

### A5. "Range reduction is essential."

*Source*: finite-precision polynomials diverge outside a narrow interval; must fold `x` into `[-π/4, π/4]`.

*Challenge*: Range reduction exists **because we chose polynomials**. If we pick CORDIC, the reduction is baked into the iteration count. If we pick table lookup keyed on phase bits, reduction is free (phase is already in `[0, 1)` after masking). If we pick complex-exponential `e^(iθ)` via `exp(ix)`, range reduction piggybacks on exp's reduction (`mod ln 2`) plus a phase correction. Reduction is a **consequence of polynomial approximation on a bounded interval**, not a universal requirement.

*Verdict*: Range reduction is a property of one algorithm family. Other families do it differently or not at all.

### A6. "Sin, cos, tan, cot, sec, csc, their inverses, hyperbolics, pi-scaled, and inverse hyperbolics are 26 distinct functions."

*Source*: the TRIG task list. Standard libm catalog.

*Challenge*: The 26-function cardinality inflates dramatically:
- **Six forward circular functions** collapse to **one fused sincos** + algebraic ratios (4 divisions).
- **Six inverse circular functions** collapse to **atan2(y, x)** + algebraic manipulations.
- **Six hyperbolic functions** collapse to **one fused sinhcosh** which collapses further to `exp(x)` and `exp(-x)`.
- **Six inverse hyperbolic functions** collapse to **log** of algebraic expressions.
- **Pi-scaled variants** are the same functions with a multiplication by π folded into reduction.

So the true primitive count is closer to:
1. `sincos(x)` (forward, fused)
2. `atan2(y, x)` (inverse, two-arg)
3. `exp(x)` (already a recipe)
4. `log(x)` (already a recipe)
5. Range reduction primitives (Cody-Waite, Payne-Hanek) — shared

Everything else is composition.

*Verdict*: The catalog has ~2-4 real mathematical primitives; the other 20+ are compositions. A well-factored library should reflect this.

### A7. "Each precision strategy is a separate function — `sin_strict`, `sin_compensated`, `sin_correctly_rounded`."

*Source*: visible in current sin.rs; `sin_compensated` and `sin_correctly_rounded` just call `sin_strict`.

*Challenge*: This is a temporary implementation stub; the architecture doc says the lowering is a **compiler pass over the recipe tree**, not three hand-written functions. The fact that we wrote three wrappers is a smell that the compiler isn't doing its job yet. The right structure is **one recipe tree** + **three lowerings**.

*Verdict*: Three entry points is a migration artifact. The recipe should be one tree with a lowering tag.

### A8. "Inputs are scalar f64."

*Source*: libm convention.

*Challenge*: In tambear, inputs are columns. A trig recipe that takes `f64` and returns `f64` is leaving the accumulate-over-a-column structure implicit. The natural signature is `sin(col)` returning a column. The scalar version is a degenerate grouping (`All` with `n=1`, or unrolled Single-element). Scalar-first implies lots of boilerplate lifting to vector; vector-first degrades cleanly to scalar.

*Verdict*: Column-first. Scalar is a degenerate case.

### A9. "Angle unit is a pre-multiply."

*Source*: the `angle_unit` parameter in sin.spec.toml ("pre-multiplied to radians before reduction").

*Challenge*: Pre-multiplying by π/180 (degrees) loses precision — 180 isn't a dyadic rational, so the multiplication rounds. For high-accuracy trig in degrees, the reduction should be modulo 90 (degrees), not modulo π/2 (radians after conversion). Likewise, `sinpi(x)` is most accurately computed via reduction modulo 2 (directly on x), not by multiplying by π and reducing modulo 2π. The pre-multiply trick **loses ulps** in the conversion step.

*Verdict*: Angle unit should influence **which reduction modulus is used**, not be a pre-multiply. The reduction algorithm is parameterized by the unit. This is actually structurally demanded — see Phase 3 below.

### A10. "The input to trig is one real number."

*Source*: elementary treatment.

*Challenge*: For many workflows the natural input is the **pair (x, y)** where `θ = atan2(y, x)`. Users computing a 2D rotation or a phase from a complex number never form `θ` — they work with `(x, y)` or with complex numbers directly. Forcing them through a scalar angle loses information (the magnitude) and loses precision (one `atan2` + one `sin` = two evaluations where one `(x, y) -> (cx, sx)` would have sufficed). Many real pipelines never want an angle at all.

*Verdict*: Angles are a choice of representation. `(x, y)`-native trig is a legitimate surface the library should expose.

### A11. "sin(NaN) = NaN, sin(±∞) = NaN."

*Source*: IEEE 754, fdlibm.

*Challenge*: These are correct and not challenged, but they **do** hide an assumption: that trig is defined on the extended reals the way exp and log are. Mathematically, sin has no limit at ∞ — the value oscillates. Returning NaN says "we refuse to give an answer in a regime where the mathematical function has no single answer." This is correct. But it's a **policy choice**, not forced by the math; some applications want sin of very large x to return the best-effort reduction even when the input has lost precision. This should be a knob.

*Verdict*: NaN-at-infinity is a policy, not a theorem. Policy knob candidate: `large_input_policy = {nan, best_effort, error}`.

### A12. "Range reduction is an internal detail."

*Source*: fdlibm structure, most libm implementations; even in sin.spec.toml, the `range_reduction` parameter is marked `advanced = true`.

*Challenge*: In tambear, range reduction is **the most expensive part** of trig and is **shared** with the sincos pair, sincospi, tan, etc. It isn't an internal detail — it's the first-class intermediate that everything else consumes. The spec acknowledges this under `sharing.writes = ["trig_reduce"]`, but then hides the reduction behind a parameter named `advanced`. That's misdirection. Reduction is the main event; the polynomial kernel is the garnish.

*Verdict*: Reduction is a first-class primitive, not an implementation detail. It deserves its own recipe name, its own tests, its own oracle, and its own sharing tag.

---

## Phase 2 — Irreducible truths

What remains when all 12 assumptions are stripped? The truths below are the ones I could not deconstruct further; if the team finds a deeper layer, we strike these and add the deeper layer.

**T1. The circle group S¹ is a one-parameter abelian group.** This is the primitive mathematical object. Trig functions are coordinate projections of points on S¹. Every subsequent structure (angles, sin, cos, phase, rotation) is a view of S¹.

**T2. Phase is a coordinate on S¹.** Radians, turns, degrees, gradians, π-scaled, complex-exponential parameter — all are equivalent phase representations. None is privileged by the mathematics; privilege comes from the consumer's local math (calculus prefers radians, signal processing prefers turns, graphics prefers [0,1)).

**T3. The forward map `θ → (c, s)` is one operation.** Computing either projection alone is strictly wasteful when the other is even slightly likely to be needed. The paired output is the primitive.

**T4. The inverse map `(x, y) → θ` is one operation.** This is atan2. All single-argument inverses (asin, acos, atan) collapse to the two-argument form with specific (x, y) choices.

**T5. Range reduction is a geometric operation on S¹, not an arithmetic trick.** It expresses "pick the representative of the coset θ + 2πℤ that lives in [-π/4, π/4]." The reduction **must** be exact (or near-exact) because any error becomes a rotation error downstream. The output is a pair (quadrant, residual) — quadrant is a coset index mod 4, residual is a phase within the fundamental domain.

**T6. The kernel approximation `(c, s) ≈ f(residual)` is the only operation that has any arithmetic choice.** CORDIC, polynomial, table — these all live here. Reduction is canonical; the kernel is algorithmic.

**T7. Every other trig function is algebraic on `(c, s)` or composed with exp/log.** This is not a convenience — it is an identity. The hyperbolic family is exp-based, the inverse hyperbolic family is log-based, the rational six (tan, cot, sec, csc) are division-based, the pi-scaled family is a reduction-modulus substitution.

**T8. Precision is a separate axis from algorithm.** A polynomial-based trig at strict, compensated, or correctly_rounded precision is the **same recipe tree** evaluated under three different lowering strategies. A CORDIC-based trig at strict, compensated, or correctly_rounded precision is a **different recipe tree** (different orchestration) but still three lowerings. Algorithm × precision is a 2D grid, not a 1D list.

**T9. The input representation (scalar, column, tile, complex, (x,y)-pair) is orthogonal to everything above.** Tambear is column-first; the mathematics above is index-free. The accumulate atom lifts the pointwise recipe to columns for free, provided the recipe itself is expressed purely in terms of arithmetic primitives.

---

## Phase 3 — Reconstruction from zero (gradient of 10 approaches)

Ordered from "smallest delta from current implementation" to "maximally structural." Each has a one-sentence pitch, concrete API, and the argument for/against.

### R1 — Status quo ante (minimal): 26 named recipes, shared reduction, shared kernels.

Keep `sin`, `cos`, `tan`, ... as 26 separate recipes. Share `trig_reduce::<x>` via TamSession. Each recipe knows the reduction result exists and pulls it if another recipe produced it first.

*For*: smallest diff from current work; fits existing atoms-primitives-recipes doc cleanly; matches libm convention.
*Against*: 26 recipes for ~2 mathematical primitives is ~12× inflation; every `tan` call makes a `sin` call + `cos` call + `fdiv` — why isn't `tan` just that composition explicitly?

### R2 — Fused primitive + derived: `sincos` is the primitive, others are one-liners.

`sincos(x)` is the only hand-written forward recipe. `sin(x) := sincos(x).1`, `cos(x) := sincos(x).0`, `tan(x) := { let (c, s) = sincos(x); fdiv(s, c) }`, etc. Each "function" is 1-3 lines.

*For*: encodes T3; eliminates duplicate reduction; matches Julia's `sincos`, Rust's `f64::sin_cos`; derived functions are trivial.
*Against*: users calling just `sin` pay for the `cos` evaluation even if they didn't want it. Worth it: the reduction dominates cost, and the kernel for cos runs in parallel.

### R3 — Two-primitive architecture: `sincos` forward + `atan2` inverse.

Extends R2 with inverse collapse. `asin(x) := atan2(x, fsqrt(fsub(1.0, fmul(x, x))))`, `acos`, `atan` all reduce to `atan2`. Hyperbolics wrap `exp`/`log`.

*For*: encodes T3 and T4; whole library is 4 primitives (`sincos`, `atan2`, `exp`, `log`) plus compositions.
*Against*: `asin` at precision correctly_rounded through this formula accumulates error at `x → ±1` (cancellation in `1 - x²`). Needs a near-boundary compensated variant. Fixable.

### R4 — Reduction as first-class recipe.

Extend R3 by elevating the reduction step. `rem_pio2_e(x) -> (q, r_hi, r_lo)` is a named recipe with Cody-Waite and Payne-Hanek variants selectable via `using(algorithm=…)`. `sincos_kernel(r_hi, r_lo) -> (c, s)` is the kernel recipe. `sincos(x) := sincos_kernel(rem_pio2_e(x).residual) folded with quadrant fixup`.

*For*: encodes T5 and T6; reduction becomes testable and oraclable in isolation; sharing tag becomes self-evident (the reduction recipe IS the shared intermediate).
*Against*: more files, more recipe names. Worth it: exposes the actual complexity where it lives.

### R5 — Angle unit as reduction modulus, not pre-multiply.

Encode A9's fix. Define a generic `rem_fundamental_domain<U: AngleUnit>(x) -> (q, r_hi, r_lo)` parameterized by the unit. Radians reduce modulo π/2 via Payne-Hanek; degrees reduce modulo 90 via integer arithmetic; pi-scaled reduces modulo 0.5 via `frint`; turns reduce modulo 0.25 via `frint`. Each unit gets its own reduction primitive, all feeding the same `sincos_kernel`.

*For*: preserves precision near the degrees/pi-scaled boundary (current pre-multiply loses ~1 ulp per conversion); makes `sinpi` structurally simple instead of a workaround.
*Against*: one reduction primitive per unit instead of one. Worth it: the math says they **are** different reductions.

### R6 — `(x, y)`-native API alongside angle-native.

Encode A10. Expose `unit_vector(θ) -> (c, s)` AND `normalize(x, y) -> (c, s)`. The first is `sincos`; the second is `(x/r, y/r)` with `r = hypot(x, y)`. Rotation on a pair `(a, b) → (c·a - s·b, s·a + c·b)` can skip `θ` entirely if the user has `(c, s)` already, and can produce `(c, s)` from `(x, y)` with one hypot+divide.

*For*: many workflows never form angles; exposing the `(x, y)` surface avoids round-tripping through θ.
*Against*: another dimension in the API surface. Fine — it's a compile-time thin wrapper.

### R7 — Complex-exponential as the primitive.

`cis(θ) -> Complex<f64>` is the primitive: `cis(θ) := (cos θ, sin θ)` as a complex number. All forward trig derives from `cis`. Multiplication of unit complexes is rotation. Exponentiation of `cis(θ)` gives `cis(nθ)`. `sincos(x) := cis(x).into()`. This fuses trig and the complex library.

*For*: most unified mathematical view; makes `exp(ix) = cis(x)` a literal identity; rotation algebra becomes complex multiplication; matches Euler's formula as the governing theorem.
*Against*: current tambear doesn't have a first-class complex type in the primitive catalog. Adding one is a separate expansion.

### R8 — Unified `trig_at(x, unit, output_mask)`.

One function to rule all forward trig. `trig_at(x, radians, {SIN | COS})` returns `(sin(x), cos(x))`. Bitmask lets caller pick any subset of {sin, cos, tan, sec, csc, cot}. One shared reduction, one kernel, one output per request. Hyperbolics become `trig_at(x, hyperbolic, ...)` with exp-based kernel swap.

*For*: direct expression of T3+T7 — trig is ONE function parameterized on input unit and output shape.
*Against*: violates "one operation per name" (A3 in reverse). The bitmask is unergonomic at call sites. Better kept as an internal fused form, with named surface APIs as views.

### R9 — Angles don't exist in the API.

The user never writes `θ`. They write `(x, y)` pairs for locations, `(c, s)` pairs for rotations, and the library composes. `sin` and `cos` disappear from the public surface; `rotate(vec, by_rotation)`, `angle_between(u, v)`, `polar_to_cartesian(r, rotation)` replace them. Angles become a representation detail of `Rotation` type.

*For*: matches how 3D graphics, robotics, and signal-processing actually work; hides the coordinate choice; makes it impossible to write unit-confused code.
*Against*: too far from convention for a general numerical library; breaks compatibility with every user's expectations; probably right for a **second-layer** API (`tambear::geometry`) built on top, not for the libm surface itself.

### R10 — Trig as a Lie group (S¹) representation, unified with rotation groups.

S¹ is U(1); 3D rotations are SO(3)/SU(2); general Lie groups have exp/log maps. Build a generic `LieGroup` trait with `exp_map`, `log_map`, `compose`, `inverse`. S¹'s exp_map is `θ → cis(θ)`; its log_map is `cis(θ) → θ` (i.e. atan2). Get SO(3) rotations, quaternion slerp, and matrix exponentials from the same abstraction.

*For*: maximally general; encodes T1 completely; unifies trig, rotations, screw theory, differential geometry.
*Against*: for a libm layer this is overreach — we need sin/cos for users today. But the structure should exist in tambear's geometry layer, and trig should be the S¹ specialization. Journey-before-destination: park this at the tambear::geometry milestone.

---

## Phase 4 — Assumption vs truth map

| Conventional view (assumption) | First-principles view (truth) |
|---|---|
| Trig = six functions on angles (A1, A3, A6) | Trig = two primitives: forward `sincos` on phase, inverse `atan2` on `(x, y)` (T3, T4) |
| Angles in radians (A2) | Phase on S¹; radians are one chart (T2) |
| Range reduction is an implementation detail (A12) | Reduction is the expensive, shared, first-class recipe (T5) |
| Each precision strategy is its own function (A7) | One recipe tree, three lowerings — same source, different compiler pass (T8) |
| Polynomial kernel is required (A4) | Polynomial is one choice; kernel is the only algorithmic freedom (T6) |
| Angle unit is pre-multiplied (A9) | Angle unit selects the reduction modulus (fix in Phase 3.R5) |
| Scalar f64 → f64 (A8) | Column → column; scalar is a degenerate grouping (T9) |
| Input is one real number (A10) | `(x, y)` pair or complex number is also legitimate (Phase 3.R6) |
| 26 separate functions in the catalog (A6) | 4 real primitives, ~20 compositions |
| sin / cos / tan split (A3) | `sincos` fused; tan/sec/csc/cot are algebraic (T3, T7) |
| Infinite inputs → NaN (A11) | Policy knob, not theorem |

---

## Phase 5 — The Aristotelian move

The highest-leverage move — one that conventional libm thinking would never surface — is this:

**Range reduction is not an implementation detail of sin; range reduction is a recipe that produces a shared, cached, S¹-coset-representative, and sin is the *consumer* that reads that intermediate.**

Invert the dependency. Today the recipe tree is:

```
sin(x) ─┬─ reduce(x) ─→ (q, r_hi, r_lo)
        └─ eval_kernel(q, r_hi, r_lo)
```

Sin owns reduction. Cos owns reduction. Tan owns reduction. Shared via TamSession tag, but still conceptually owned downstream.

The Aristotelian move inverts this:

```
s1_representative(x) ─→ (q, r_hi, r_lo)     // first-class recipe, shared intermediate
  │
  ├── forward consumers:
  │     sincos_kernel → (c, s)
  │     sinhcosh_kernel → (ch, sh)  // via exp, not a direct kernel
  │     ...
  │
  └── downstream views:
        sin(x) := sincos(x).1
        cos(x) := sincos(x).0
        tan(x) := sincos(x).s / sincos(x).c
        ...
```

**What changes structurally when you make this move:**

1. **The catalog shrinks from 26 to ~6.** `rem_pio2`, `rem_half_turn`, `rem_degrees_90`, `sincos_kernel`, `sinhcosh_via_exp`, `atan2_kernel`. Everything else is a 1-line view.
2. **Sharing becomes automatic, not annotated.** The reduction recipe's output IS the intermediate; consumers query by name. No `sharing.reads = ["trig_reduce"]` annotations to maintain.
3. **Precision strategy applies once, per layer.** Reduce at `correctly_rounded`, kernel at `correctly_rounded`, combined via lowering. Today each of the 26 recipes gets a precision tag; tomorrow two recipes (reduction + kernel) carry them.
4. **Testing concentrates.** The oracle for reduction is `mpmath.frac(x / (pi/2))`. The oracle for kernel is `mpmath.sin(r) / mpmath.cos(r)` on `[-pi/4, pi/4]`. Two oracles replace 26.
5. **Angle-unit support becomes a reduction-modulus selector, not a pre-multiply.** A single `rem_to_fundamental<unit>(x)` dispatches to the correct reduction for the unit. Degrees reduces mod 90 via integer arithmetic, pi-scaled reduces mod 0.5 via frint, radians reduces via Cody-Waite/Payne-Hanek. No precision loss in the conversion.
6. **The sincospi "trick" disappears.** `sinpi(x) := { let r = frint(2.0 * x); let residual = 2.0 * x - r; sincos_kernel(residual, 0.0, r as i32) }` — it's just a different reduction feeding the same kernel. The separate sinpi function becomes a view over reduction + kernel with a different unit.
7. **The `advanced = true` marker on `range_reduction` is wrong.** The parameter should be first-class — it's the main event.

**The philosophical shift**: sin is not a function that happens to share its reduction; sin is a **view** on a shared phase-space computation. The recipe that produces `(q, r_hi, r_lo)` is the real work. Everything else is a projection.

This move is why we don't need CORDIC or table-lookup variants in the catalog: they are kernel alternatives, living under `sincos_kernel`'s `using(method=…)`. The reduction is independent of the kernel algorithm. The kernel is independent of the reduction algorithm. They meet at the interface `(q, r_hi, r_lo) → (c, s)`.

---

## Phase 6 — Recursive rejection

I gave Phase 5 back to the hypothetical critic: *"Is this stable? What assumption is still embedded?"*

### Round 1 rejection: "Why is `(q, r_hi, r_lo)` the right intermediate shape?"

Two slots for the residual (hi, lo) is a double-double artifact. What if the kernel consumed a full `DoubleDouble` type instead? Or a triple-double? What if it consumed a complex number `cis(r_hi + r_lo)` directly in complex form, and the kernel was just "mul by a small polynomial correction"?

*Response*: accepted. The shape `(q, r_hi, r_lo)` is pragmatic — it matches fdlibm, current primitives, hardware FMA patterns. But the **right** shape for the shared intermediate is whatever the precision demand requires: `DoubleDouble` for compensated, `f64 + f64` for strict, `TripleDouble` for correctly_rounded at extreme arguments. The intermediate type should be a lowering-strategy-aware variant. Add this as a specific question for math-researcher.

### Round 2 rejection: "Why is the fundamental domain `[-π/4, π/4]` rather than `[0, π/2)` or `[0, 2π)`?"

The `[-π/4, π/4]` choice is because polynomial kernels on that domain have the smallest minimax error. If we used CORDIC, `[0, π/2)` is natural (positive iteration). If we used table lookup keyed by phase bits, `[0, 1)` in turns is natural (bitfield-clean).

*Response*: accepted. The fundamental-domain choice is **coupled to the kernel**. So the reduction recipe is parameterized by the kernel it feeds, or equivalently, kernel + reduction come as matched pairs. This matches a standard libm organization: `__kernel_sin` expects input in `[-π/4, π/4]`; a CORDIC kernel would expect `[0, π/2)`. Document the (reduction-type, kernel-type) compatibility pairs.

### Round 3 rejection: "Why compute `(c, s)` at all? Why not just the bits we need?"

For `tan(x)`, the user needs `s/c`, not the pair. For `sec(x)`, they need `1/c`. Computing both `c` and `s` and then dividing discards work. The `(c, s)` split happens because we lack a fused "s/c" kernel.

*Response*: partially accepted. A fused `tan` kernel exists in fdlibm (`__kernel_tan`) — it computes tan directly on the reduced interval, no division. For `sec`, `csc`, `cot`, the literature doesn't have common fused kernels; they are division-based. So: `tan` gets its own kernel (composed atop reduction); `sec/csc/cot` are 1/c, 1/s, c/s compositions. The Aristotelian move accommodates this — kernels are a family under the reduction layer.

### Round 4 rejection: "Why is `atan2` the inverse primitive and not `arg` (complex argument) or `phase_unwrap`?"

`atan2(y, x)` is the library-conventional inverse. But the **mathematical** inverse of `cis(θ) = (c, s)` is `θ = ln(c + is) / i`. That's the complex log, which tambear has. So the inverse trig primitive is **literally a specialization of complex log** restricted to the unit circle.

*Response*: accepted. This matches Phase 3.R7/R10. The "inverse primitive" family collapses into complex log. `atan2` is a specialization that avoids forming the complex number explicitly. At the **mathematical** level, all inverse trig + inverse hyperbolic + argument calculations are one primitive: complex log. At the **implementation** level, we might keep `atan2` as a named recipe for the common real-pair input because it avoids branch cuts.

### Round 5 rejection: "Why do the inverse and hyperbolic families stay as separate recipes if they all reduce to exp/log + algebra?"

In Phase 3.R3, I said "hyperbolics wrap exp/log." Pushed further: `asinh(x) := log(x + sqrt(x² + 1))` — that's it. One line. `atanh(x) := 0.5 * log((1 + x) / (1 - x))` — one line. These don't need recipes at all; they are **formulas applied pointwise**. The entire inverse hyperbolic family (6 functions) is 6 one-line compositions.

*Response*: accepted. Phase 5 proposed this implicitly; making it explicit: the library's hand-written surface is:
- `rem_pio2`, `rem_half_turn`, `rem_degrees_90` (reductions)
- `sincos_kernel`, `tan_kernel`, `atan2_kernel` (kernels)
- `exp`, `log` (existing)
- Everything else is ≤5-line compositions.

Total new hand-written code: ~6 recipes for the forward+inverse+hyperbolic trig surface of ~26 functions.

### Round 6 — Stability.

After 5 rounds, the structure is stable. The essential recipes are:
1. Range reduction (family of recipes, parameterized by angle unit and by target kernel)
2. Forward kernels on reduced input (family: sincos polynomial, tan polynomial, maybe CORDIC)
3. Inverse kernel (atan2)
4. Complex exponential and complex log (if we go the R7/R10 route eventually)

Everything else is view/composition.

---

## Phase 7 — Consolidated truths (post-rejection)

The truths T1-T9 from Phase 2 survive. Add three refinements from Phase 6:

**T10. The shared intermediate shape is lowering-strategy-aware.** `(q, r_hi, r_lo)` at strict; `(q, DoubleDouble)` at compensated; `(q, TripleDouble)` at correctly_rounded. Not a separate type — a precision-parameterized type.

**T11. Reduction fundamental domain is coupled to kernel choice.** A kernel expects a specific domain; a reduction produces a specific domain. The (reduction, kernel) pair is matched.

**T12. Inverse trig is a specialization of complex log.** Real-input specializations (atan2) exist for performance/branch-cut reasons, but the mathematical primitive is `clog(cis(θ)) = iθ`.

---

## Phase 8 — Reconstruction recommendation

Given the full deconstruction, the recommended architecture for the trig family is:

### Primitives to add to Layer 2

These are true primitives (terminal operations, hardware-level or error-free transformations) that trig needs and doesn't currently have:

1. **`frem_py2(x) -> (i32, f64, f64)`** — compensated reduction. Not a single hardware op; a composed primitive that packages Cody-Waite+Payne-Hanek as one atomic operation at the recipe layer.
2. **`ldexp_f64(x, k) -> f64`** — bit-level `x · 2^k`. Already a near-primitive in scale.rs; elevate status in the catalog (matches architecture doc's open question about ldexp).

### Recipes to write at Layer 3

1. **`rem_pio2`** — radians reduction. Cody-Waite + Payne-Hanek dispatch. Lowering-strategy-parameterized residual type.
2. **`rem_half_turn`** — reduction mod 0.5 for π-scaled input. Uses `frint`, no π constant involved — exact.
3. **`rem_degrees_90`** — reduction mod 90 for degree input. Integer arithmetic for the integer quadrant; residual stays in degrees; conversion to radians happens inside the kernel (mul by π/180, but only the small residual).
4. **`rem_gradians_100`** — same structure for gradians (mod 100).
5. **`rem_turns_quarter`** — same for turns (mod 0.25).
6. **`sincos_kernel(q, r_hi, r_lo, unit) -> (c, s)`** — evaluates the polynomial on the reduced input, applies quadrant fixup, applies residual folding. Unit tag flags whether `r_hi, r_lo` are in radians already or in another unit (in which case the kernel multiplies by the unit's radian-conversion factor after reduction, on the small residual only).
7. **`tan_kernel(q, r_hi, r_lo) -> f64`** — fused tan polynomial (fdlibm `__kernel_tan`), avoids the division.
8. **`atan2_kernel(y, x) -> f64`** — the inverse.

### Recipes as thin views

All of these become 1-5 line compositions:

```rust
pub fn sin(x: f64) -> f64 { sincos(x).1 }
pub fn cos(x: f64) -> f64 { sincos(x).0 }
pub fn tan(x: f64) -> f64 { tan_kernel(rem_pio2(x)) }
pub fn sec(x: f64) -> f64 { fdiv(1.0, cos(x)) }
pub fn csc(x: f64) -> f64 { fdiv(1.0, sin(x)) }
pub fn cot(x: f64) -> f64 { let (c, s) = sincos(x); fdiv(c, s) }

pub fn sinpi(x: f64) -> f64 { sincospi(x).1 }
pub fn cospi(x: f64) -> f64 { sincospi(x).0 }
pub fn sind(x: f64) -> f64 { sincos_degrees(x).1 }  // etc

pub fn sinh(x: f64) -> f64 { 0.5 * (exp(x) - exp(-x)) }
pub fn cosh(x: f64) -> f64 { 0.5 * (exp(x) + exp(-x)) }
pub fn tanh(x: f64) -> f64 { sinh(x) / cosh(x) }  // numerically sound variant exists; see note

pub fn asin(x: f64) -> f64 { atan2(x, fsqrt(fsub(1.0, fmul(x, x)))) }
pub fn acos(x: f64) -> f64 { atan2(fsqrt(fsub(1.0, fmul(x, x))), x) }
pub fn atan(x: f64) -> f64 { atan2(x, 1.0) }

pub fn asinh(x: f64) -> f64 { log(x + fsqrt(fmadd(x, x, 1.0))) }
pub fn acosh(x: f64) -> f64 { log(x + fsqrt(fsub(fmul(x, x), 1.0))) }
pub fn atanh(x: f64) -> f64 { 0.5 * log((1.0 + x) / (1.0 - x)) }
```

Total forward + inverse + hyperbolic + pi/degrees/gradians = ~30-40 named surface functions, ~8 actual hand-written recipes behind them.

Note: `tanh` by the naive formula overflows for large |x|; use `tanh(x) = 1 - 2/(exp(2x) + 1)` for numerical safety. This is a **compensated-variant** concern, not a fundamental change — the view function dispatches to a numerically safe body.

### What this means for pathmaker (TRIG-12, TRIG-13, TRIG-14, TRIG-15, TRIG-16)

**If the team accepts this reconstruction:** the .spec.toml files should describe the **views** with a link to their **kernel recipe**. The kernel recipes get their own spec.toml files. TRIG-13/14/15/16 collapse from "implement 20+ Rust functions" to "implement 8 kernels + wire 20+ one-liners."

**If the team rejects this reconstruction:** proceed with the current 26-recipe plan, but the naturalist should tag every function's sharing contract with the fact that reduction is the shared intermediate, and the campsite graveyard should hold this deconstruction as a reference.

### What this means for naturalist

The pattern surfacing here — "one expensive shared computation, many thin views" — rhymes with:
- `MomentStats` → mean/var/skew/kurt/cumulants as views
- `CovarianceMatrix` → PCA / LDA / factor analysis / Mahalanobis as views
- `FFT` → spectral density / periodogram / Welch / MTM as views

**This is the same structure three layers up.** If naturalist confirms the rhyme, it strengthens the reconstruction — the library's architecture is already in this shape for other families; trig is an outlier that conventional-libm convention forces into per-function recipes.

### What this means for math-researcher

The HOW of each kernel (CORDIC vs polynomial vs table) lives under `sincos_kernel`'s `using(method=...)`. The reduction HOW (Cody-Waite vs Payne-Hanek) lives under `rem_pio2`'s `using(algorithm=...)`. Two independent method families, cleanly separated.

---

## Deliverable status

- [x] Phase 1 — Assumption autopsy (12 assumptions)
- [x] Phase 2 — Irreducible truths (T1–T9)
- [x] Phase 3 — 10 reconstruction approaches (R1–R10)
- [x] Phase 4 — Assumption-vs-truth map
- [x] Phase 5 — The Aristotelian move (range-reduction-as-primitive inversion)
- [x] Phase 6 — Recursive rejection (5 rounds, stable at Round 6)
- [x] Phase 7 — Consolidated truths (T10–T12 added)
- [x] Phase 8 — Reconstruction recommendation

**Companion deliverables**: `atoms_gaps.md` (primitives the reconstruction structurally demands) and `notation.md` (three notation styles per function, for math-researcher's catalog).
