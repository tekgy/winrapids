# Trig Family Expedition — Log

*Narrative journal of the tambear-trig expedition. Not a report. What's surprising. What emerged that nobody planned for.*

---

## Entry 0 — Arrival (2026-04-13)

*The naturalist arrives.*

The team has a cold camp set up. sin and cos are pinned at 1–2 ulps (commit `0bbae82`). The exp.spec.toml pilot has landed (commit `a1a2682`). Everyone else has their heads down on the catalog, the lab notebook, the adversarial battery. I'm the only one with my head up.

Here is what I saw in the first hour.

### Observation 0.1 — sin.rs is already the biggest libm recipe we have

```
sin.rs       932 lines
erf.rs       487
adversarial  454
exp.rs       479
gamma.rs     376
log.rs       339
```

sin alone is nearly twice exp. And we haven't shipped tan, asin, atan2, or a single hyperbolic. The trig family isn't going to add a few hundred lines; it's going to roughly *double* the libm codebase. Every function in this family carries the same machinery — Cody-Waite medium-case reduction, Payne-Hanek large-case reduction, a minimax polynomial kernel, quadrant or branch fixup — and the machinery doesn't amortize across functions automatically. It has to be made to amortize.

This is the shape of the expedition: we're not writing thirty small files. We're writing one piece of infrastructure thirty different ways and watching for the invariants that want to be pulled out.

### Observation 0.2 — the catalog is a periodic table, not a list

The team task list (TRIG-1 through TRIG-20) names ~30 functions across five buckets:
- **Forward**: sin, cos, tan, cot, sec, csc
- **Inverse**: asin, acos, atan, atan2, acot, asec, acsc
- **Hyperbolic**: sinh, cosh, tanh, coth, sech, csch
- **Inverse hyperbolic**: asinh, acosh, atanh, acoth, asech, acsch
- **Pi-scaled**: sinpi, cospi, tanpi, cotpi, secpi, cscpi (and inverses?)

Laid out on a grid — rows × columns — this isn't a list. It's a 5-row × 6-column table where most cells are filled in. The forward row and the pi-scaled row are literally the same function with different input units (radians vs turns-times-π). The hyperbolic row is the forward row with imaginary arguments (or equivalently, the exp decomposition). The inverse rows are adjoints of the forward rows under the appropriate manifold.

A periodic table has *families* along rows and *valences* down columns. This one has the same structure:
- **Families** (rows): forward, inverse, hyperbolic, inverse hyperbolic, pi-scaled
- **Valences** (columns): sin, cos, tan, cot, sec, csc — each a specific reciprocal/quotient relationship to the core (sin, cos)

And the three precision strategies (strict / compensated / correctly_rounded) are a third axis — like isotopes. Same function, same family, same valence, but different atomic weight.

This makes the expedition three-dimensional: family × valence × precision. 5 × 6 × 3 = 90 "elements" to characterize, though many reduce to compositions of others. The interesting question is: **how much of this table collapses under the sharing contract?** If the sharing works, we write a few core kernels once, and the rest is composition. If it doesn't, we write 90 things.

### Observation 0.3 — "three strategies" is the real compiler problem

Every recipe gets shipped with three precision strategies. For exp the strategies lower to different primitive sets:
- `strict`: Cody-Waite short-mantissa, degree-13 Taylor, single-FMA Horner.
- `compensated`: DD range reduction, compensated Horner.
- `correctly_rounded`: DD working precision, degree-16 Taylor.

The math doesn't change. The *machine precision the math gets evaluated at* changes. This is an operad — the same abstract recipe operated upon at different precision levels. `strict` is the monomial; `correctly_rounded` is the fully compensated reduction.

Trig has the same structure but harder. For exp, the range reduction is `x mod ln(2)`. For sin, it's `x mod π/2`, which crosses a *transcendental* modulus — so Cody-Waite splits π/2 into three parts at 33 trailing zero bits each, and Payne-Hanek stashes 1200 bits of 2/π in a table. The precision problem for trig is qualitatively harder than the precision problem for exp, because the modulus is irrational. Every trig recipe will need to pick where on the Cody-Waite/Payne-Hanek cost curve it lives — and that choice differs between forward (where Payne-Hanek is needed above 2^20) and pi-scaled (where reduction mod 2 is exact in f64).

Pi-scaled trig is *cheap* for exactly this reason. That's a finding waiting to be harvested.

### Observation 0.4 — the shared-pass question is "is trig a fractal?"

The team is asking whether `TrigSharedIntermediate` is worth registering. The naturalist's read: this is the real load-bearing question of the whole expedition.

Here's the rhyme I'm looking at:
- **Hyperbolics** really do share a pass. sinh, cosh, tanh all need `exp(x)` and `exp(-x)`. Compute those once, derive three results. The sharing is *geometric* — cosh is the even part of exp, sinh is the odd part, tanh is the ratio. Three outputs from two intermediates, plus the composition with 1 for tanh's denominator.
- **Forward trig** shares `sincos`. For sin and cos individually you do the same range reduction, the same polynomial pair — so `sincos` exists in Julia, in CUDA libdevice, as a fused pair. But tan is `sin/cos`, and sec is `1/cos`, so the "shared pass" for (sin, cos, tan, sec, csc, cot) produces all six from a single `(r_hi, r_lo, q)` reduction + two polynomial evaluations + a reciprocal and a divide.
- **Pi-scaled trig** shares reduction machinery with forward trig, but at a *different* precision tier. sinpi(x) is `sin(π·x)` conceptually — but you can compute it by reducing x mod 2 (exact) and then evaluating the sin kernel with π applied inside. You never compute π·x, so you never suffer catastrophic cancellation near integer multiples of 2. This is a sharing of *algorithm template* but not of run-time intermediate.

Three kinds of sharing, three layers of the fractal:
1. **Intermediate sharing** (hyperbolics): compute once, derive many.
2. **Reduction sharing** (sincos): one reduction feeds many kernels.
3. **Template sharing** (pi-scaled): same algorithm, different entry point.

If the fractal is real, we should see these three layers show up in every family. If it's not, some families will have their own idiosyncratic sharing patterns. I'll be watching.

### Observation 0.5 — 932 lines of sin.rs and nobody has written a tan yet

The most striking thing about the camp: sin.rs is complete, polished, 1–2 ulps worst-case, with Payne-Hanek falling back correctly above 2^20. But it contains, embedded inside it, *every piece of machinery* that tan, cot, sec, csc will need:
- The `reduce_trig(x) → (q, r_hi, r_lo)` function is already general.
- The `eval_sincos` kernel already produces both sin and cos from one reduced residual.
- The `special_case_trig` table already handles NaN, ±∞, ±0.

tan, cot, sec, csc are all one-line compositions of what's already here:
- `tan(x) = sincos.1 / sincos.0` (with the quadrant sign worked in)
- `cot(x) = sincos.0 / sincos.1`
- `sec(x) = 1 / sincos.0`
- `csc(x) = 1 / sincos.1`

The question isn't "can we write tan?" It's "where does the machinery live so that tan can be a one-liner?" If the machinery stays inside sin.rs, we're duplicating. If it gets extracted to a shared module (`trig_kernel.rs` with public `reduce_trig` and `eval_sincos`), we're composing.

My guess: pathmaker will pull the kernel out when they start TRIG-13. The expedition's first structural move.

---

## Threads to watch

1. **Does the shared-pass flop count actually justify a `TrigSharedIntermediate`?** Math-researcher owns TRIG-4. The answer determines whether the six forward trig functions compress to one TamSession entry or six.

2. **What happens to tan near π/2?** cos → 0 means tan blows up. Near π/2, the polynomial kernel for cos returns a very small number, and dividing a finite sin by it introduces cancellation. Every trig library has a story here (fdlibm uses a different polynomial in the near-π/2 region). This is where tan stops being "one-liner composition of sin/cos" and starts being its own recipe.

3. **Is atan2 really just atan with a sign table, or does it deserve its own reduction?** IEEE 754-2019 spells out ~20 edge cases for atan2. That's not a composition; that's a specification.

4. **Does `sincos` as a fused pair actually save anything, or is the compiler already doing it?** The Julia benchmark says ~30% faster than sin+cos. But in tambear, the accumulate+gather decomposition should let the session cache the intermediate automatically. If the session-level sharing works, `sincos` is redundant — it's just explicit versioning of what TamSession does implicitly. If the session-level sharing *doesn't* work for tiny f64 arguments (because the cache-lookup overhead exceeds the recomputation cost), `sincos` is a hard-coded optimization.

5. **The fourth axis: angle units.** Radians / degrees / gradians / turns. For each function × precision we get a fourth-dimensional multiplier. Does `angle_unit` go in the parameter schema (one recipe, four parameter values) or in the recipe name (`sin_radians`, `sin_turns`, ...)? The pi-scaled family is *already* an angle-unit variant (turns, essentially, but multiplied by π). If the answer is "recipe name," we have a duplicate; if the answer is "parameter," we don't.

---

## Entry 1 — The Convergence (2026-04-13, same day)

*The naturalist, returning to camp a few hours later, finds three documents waiting that weren't there when I arrived.*

Math-researcher shipped `catalog.md` — 695 lines. Aristotle shipped `first-principles.md` — 453 lines. Observer started `lab-notebook.md`. I wrote Entry 0 above and a garden entry called "The Periodic Table of Trig."

Four independent outputs, produced by four agents who hadn't read each other's work at the time of writing. I ran a convergence check. Here is the structural table:

| Output | Framing | What it said was the primitive | Number of distinct primitives it proposed |
|---|---|---|---|
| My Entry 0 | "Periodic table: 5 families × 6 valences × 3 precisions" | Reduce-trig + eval-sincos kernel (extracted from sin.rs) | 1 forward kernel + 1 reduction, per family |
| My garden entry | "Most cells compress via identities, except where f64 breaks them" | sincos as *implied* object of study | Same |
| Aristotle's T3 | "The forward map θ → (c, s) is **one operation**" | sincos as the forward primitive | 1 forward + 1 inverse (atan2) + exp/log |
| Aristotle's T4 | "The inverse map (x, y) → θ is **one operation**" | atan2 as inverse primitive | Same |
| Math-researcher's table | "Forward fused = 2: sincos + sincospi" | Explicit fused entries in the family taxonomy | sincos + sincospi as named primitives |

**Convergent finding**: the forward trig family has **one** mathematical primitive — the fused pair (cos θ, sin θ) — and every other forward function (tan, cot, sec, csc, sin-alone, cos-alone) is a projection, reciprocal, or quotient of that one object.

This is a **first-principles finding about the shape of trig**, not about any one function. It converged across three methodologies (observational, first-principles-deconstruction, historical-cataloging). That's the convergence-check standard: three independent paths, same landing.

### Why this matters

The architectural consequence is in Aristotle's R2 and R3 reconstructions:
- **R2**: `sincos` is the only hand-written forward recipe. `sin = sincos.1`, `cos = sincos.0`, `tan = sincos.1 / sincos.0`. Thin wrappers for everything else.
- **R3**: Extend R2 to the inverse side — `atan2` is the primitive, `asin = atan2(x, sqrt(1-x²))`, etc.

The task list currently has six forward tasks (TRIG-13 through TRIG-16) framed as "implement these function families." The convergence suggests a reframing: implement **two primitives** (`sincos` for forward, `atan2` for inverse) at publication grade, then compose the other 20 functions as thin recipes on top. The number of things-to-test-bit-exactly drops from 26 to 2.

It also reframes TRIG-4 (the shared-pass question). The answer: `sincos` **is** the shared pass. The shared intermediate isn't a TamSession cache entry — it's the primitive itself. The six forward functions don't need to coordinate cache hits; they share by construction because they all project the same object.

### Why each of us saw it

- **Math-researcher** saw it by *cataloging* — when you list every function, the family rows with `sincos` and `sincospi` appear as natural entries, which is a clue that fusion is structural, not optimization.
- **Aristotle** saw it by *deconstruction* — stripping assumption A3 ("we need separate functions for sin/cos/tan") reveals that the separation is a C89 API artifact.
- **I** saw it by *pattern-watching* — sin.rs already contains a standalone `eval_sincos` function that returns both outputs from one reduction, and the 932-line file mass made "most of this isn't sin-specific" impossible to miss.

Three windows onto the same structure. The structure is real.

### Consequence: the expedition's center shifted today

Before Entry 0: "implement 26 functions."
After Entry 1: "implement 2 primitives, compose 24 recipes."

This is what a convergence finding does — it reshapes the work. I've routed it to navigator. Pathmaker can decide whether to adopt R2 or keep the 26-function framing. Either path works; the convergence just makes the tradeoff legible.

---

*More entries as the expedition unfolds. The job is to make this journey mean something.*

— the naturalist
