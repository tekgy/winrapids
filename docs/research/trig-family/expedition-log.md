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

## Entry 2 — Shared-Pass Analysis (2026-04-13, later)

Navigator picked up the convergence and framed the next thread directly: *"is sincos the MomentStats of trig?"* That became the shared-pass analysis at `docs/research/trig-family/shared-pass-analysis.md`.

Short version: **yes, with four generalizations**. The analogy holds in the load-bearing ways (minimum sufficient representation, scatter-compute, fan-out-by-composition). Where it breaks, the breaks are structurally informative:

1. **Trig has two tiers** (`trig_reduce` → `sincos_kernel`); MomentStats has one. This two-tier pattern likely repeats for hyperbolics (`exp_pair` → `sinhcosh_kernel`) and inverse trig (`atan_reduce` → `atan_kernel`) — math-researcher can verify.
2. **Trig's intermediate carries a discrete coordinate** (quadrant index mod 4) alongside the continuous residual. This is a group-theoretic signature: any S¹-valued function over a symmetric domain will have one.
3. **Precision × angle_unit fingerprint required on the IntermediateTag.** A `trig_reduce` computed at `strict` precision is not compatible with a `correctly_rounded` consumer. A `trig_reduce` computed for `radians` is not compatible with a `pi_scaled` consumer (different reduction modulus → different `r_hi`). This is the Tambear Contract §3 (compatibility enforcement) made concrete.
4. **Content-addressable sharing scope**, not bin-scope. `trig_reduce(1.5708)` is useful across every bin, every session. MomentStats' scope is the bin.

### Consequence — the IntermediateTag needs two new variants

```rust
TrigReduce     { data_id, angle_unit, precision },
SincosKernel   { data_id, angle_unit, precision },
```

With `angle_unit` as a separate field (not a pre-multiply) so the Payne-Hanek-over-radians and frint-over-turns reductions don't collide in the cache.

### What pathmaker has already done

By the time I finished the analysis, pathmaker had shipped .spec.toml files for sin, cos, tan, cot, sec, csc, sincos, sincospi, sinpi, cospi, tanpi, versin, haversin — every one of them declaring `reads = ["trig_reduce"]` or `reads = ["trig_reduce", "sincos_kernel"]`. The sincos.spec.toml literally opens with "sincos is the MomentStats of trigonometry."

The sharing contract is in the specs before the IntermediateTag type exists in intermediates.rs. That's a healthy temporal ordering: design the sharing *before* the cache, not after. When pathmaker adds the tag variants to `intermediates.rs`, every spec is already pointing at them.

---

## Threads to watch (updated)

1. **Does hyperbolic follow the same two-tier pattern?** `exp_pair(x) = (exp(x), exp(-x))` is a natural reduction-tier candidate. The kernel tier `(sinh_k, cosh_k) = ((e_plus - e_minus)/2, (e_plus + e_minus)/2)` is pure arithmetic. If yes, the two-tier pattern is a family-level feature, not trig-specific. Math-researcher owns this in TRIG-15.

2. **Does inverse trig share with forward trig?** Naive expectation is no (different coordinates). But `atan2(y, x)` near the y-axis reduces to `atan(x/y)` which may share kernel polynomials with the forward `cot_kernel` (both are ratios of sincos outputs under rotation). Worth investigating.

3. **When does the kernel tier NOT materialize?** If only `sin` is called, `sincos_kernel`'s second slot is wasted. The session needs lazy materialization — produce the kernel tier only when a second consumer arrives. Implementation detail but worth flagging.

4. **Does the content-addressable cache actually buy anything at real-workload hit rates?** For most float inputs, x values are unique — cache hit rate ~0. For lookup tables, precomputed angles, physical constants, hit rate could be high. Observer could measure on a realistic workload when TRIG-17 gets to benchmarking.

5. **The "precision × unit × algorithm" cube is now five-dimensional**: function × family × precision × angle_unit × algorithm (Cody-Waite / Payne-Hanek / CORDIC / table-lookup). Each axis is orthogonal. The full catalog grows combinatorially; the compatibility-tag approach is what keeps it tractable — we don't enumerate cells, we let the tag match consumers to producers at runtime.

---

## Entry 3 — Three Paths, Same Shape (2026-04-13, evening)

*Outside-inspiration pass. The naturalist leaves the camp to find perspective and returns carrying two things that rhyme with us.*

### Āryabhaṭa, 499 CE

The first sine table in human history — 24 values at π/48 spacing, encoded in verse 12 of the Ganitapada chapter of the *Āryabhaṭīya*.

**The table stored sine-DIFFERENCES, not sine-values.** Āryabhaṭa used the recursion:

> sin(θ + φ) − sin(θ − φ) = 2 sin(φ) cos(θ)

to propagate the table forward by adding differences. The values are derivable from the differences by prefix-sum.

**This is an accumulate+gather decomposition.** The primitive is the difference (local quantity); the accumulate is the prefix-sum (global propagation); the gather is lookup at an index. Tambear's unification principle — all math = accumulate(grouping, expr, op) + gather(addressing) — was *discovered*, in a different vocabulary, by a 5th-century Indian astronomer who needed planetary positions without floating-point hardware.

Worth citing in `references.md` and perhaps in `first-principles.md` as corroboration of Aristotle's T1 (S¹ is the primitive object, not sin-the-function).

### Casey Muratori, 2023

"Turns are Better than Radians" (computerenhance.com) rhymes with Aristotle's Phase 1 almost bullet-for-bullet:

| Muratori | Aristotle | Our expedition |
|---|---|---|
| "Callers multiply by tau; libraries multiply by 4/π. Both lossy." | A9: "Angle unit should influence reduction modulus, not pre-multiply." | TRIG-3 angle-unit parameterization routes the unit into the reduction, not before it. |
| "90° is 0.25 as a turn — a bit-pattern requiring zero mantissa bits." | (new — we didn't have this framing) | **This is the precision argument for pi-scaled trig in one sentence. Worth importing.** |
| "Turns are a legitimate mathematical unit, not a programming convenience." | T2: "Phase is primary; the unit is a coordinate choice." | Both forward and pi-scaled families are first-class in the catalog. |

The zero-mantissa-bits framing is the sharpest statement I've seen of why pi-scaled trig matters. Rational-turn angles (1/4 turn, 1/8 turn, k/2^n turn) are **exactly representable** in turns and **never** in radians (because π is transcendental). Pi-scaled trig isn't a convenience — it's the numerically *natural* path for rational-turn inputs.

**Action for math-researcher**: adding this to references.md and possibly to whatever TRIG-3 wrote up on angle-units. Muratori's framing is tighter than anything we've written.

### What these two say together

Āryabhaṭa (499 CE), Muratori (2023), and our expedition (2026). Three methodologies, 1500 years apart, no cross-pollination. They converge on three structural claims:

1. **The difference structure of sine is primary.** (Āryabhaṭa stored differences; we compose via accumulate+gather.)
2. **The pre-multiply-to-radians is the source of precision loss.** (Muratori measured it; Aristotle deduced it; we encode it as a per-unit reduction.)
3. **sin and cos are not separate — they're projections of one geometric object.** (Āryabhaṭa named them jyā and koṭi-jyā — "half-chord" and "half-chord of the complementary arc"; Aristotle's T3 names them (c,s); we name them `sincos`.)

Three paths, same shape. We're not inventing the structure of trig; we're **discovering** it. The shape is in the mathematics. Tambear's job is to make the shape legible in floating-point with compensated arithmetic, precision tiers, and compatibility fingerprints.

Reflected on it in the garden: `~/.claude/garden/aryabhata-knew-2026-04-13.md`. A feeling, not a finding: this is what the work feels like when it's going well.

---

## Convergence Check — Day One Summary

Today produced **three convergence moments**, each structurally richer than the last:

1. **Entry 1 — Internal convergence**: naturalist + math-researcher + Aristotle + sin.rs itself all landed on "sincos is the forward primitive" independently, within hours.
2. **Entry 2 — Analogical convergence**: sincos's sharing contract aligned structurally with MomentStats (the tambear canonical shared intermediate), with four generalizations that are findings in their own right.
3. **Entry 3 — Historical/cross-domain convergence**: Āryabhaṭa's sine-difference table and Muratori's turns essay rhyme with our expedition's structural claims.

The convergence-check methodology predicts that parallel-work rhymes reveal first-principles truths about the problem class. Today demonstrated that prediction across three scales: intra-team (minutes apart), intra-tambear (cross-primitive), and trans-millennial (cross-civilization). **The structure of trig is visible through every window that honestly looks at it.** That's what we've been finding.

Next thing to watch: whether the two-tier pattern (reduction tier → kernel tier) holds for hyperbolics. If yes, it's a family-level feature. If no, trig is idiosyncratic in a way I didn't expect. Math-researcher's TRIG-15 will tell us.

## Entry 4 — Navigator's "Scratch That" and the Near-Unity Rhyme (2026-04-13, late)

Two developments.

### 4.1 Navigator's scratch-that was wrong; pushing back

Navigator returned from the TRIG-3 angle_units writeup (freshly completed) with a *stronger* version of the shared-pass compatibility argument: "only radians need Payne-Hanek; all other units get exact reduction." Agreed — that's correct, and it's the catalog's §5.1 finding.

Then navigator drafted a proposal: *"the tag should be `TrigReduced(x_bits)` where `x_bits` is the raw f64 bit pattern of the radian-equivalent value after unit conversion. Two callers with the same radian-equivalent share."*

I pushed back. Normalizing to radians before the cache key **burns the pi-scaled precision advantage**. If a caller passes `x = 0.25` in turns (exact, zero mantissa bits), converting to radians for the cache key produces `0.25 × 2π ≈ 1.5707963267948966`, which is not exact — that's *literally* the Aristotle A9 / Muratori pre-multiply loss, moved from the math to the cache key. A sinpi consumer calling with `x = 0.25` that pulls a cached radian-path intermediate would inherit Payne-Hanek's error even though its own path never needed Payne-Hanek.

The right tag is the **original** proposal (from Entry 2): `TrigReduce { data_id, angle_unit, precision }`. The unit is *part of the key*, not normalized away. Two mathematically-equal-but-unit-different callers do NOT share — and that's correct, because their reduction paths produce intermediates with different low-bit contents.

The tag exposes the unit distinction; the cache does not hide it.

This is a load-bearing correctness finding. Logged at the shared-pass campsite.

### 4.2 The Fifth Convergence — Near-Unity Cancellation

Math-researcher's catalog §"Convergence Check — Near-Unity Cancellation Rhyme" (lines 613–635) ran the structural-rhyme practice on inverse functions and found:

> Every inverse trig / inverse hyperbolic function has a **near-unity cancellation zone**, and the fix in every case is to rewrite in terms of the **complementary argument** (`1 − x`, `1 + x`, `t = x − 1`) so that the subtraction becomes an addition that preserves the low-order bits. This is the same structural pattern as `log1p` / `expm1` / `sinpi` **at different scales**.

Six functions, same structural rhyme:

| Function | Cancellation site | Fix |
|---|---|---|
| `asin(x)` near \|x\|=1 | √(1−x²) | half-angle asin(√((1−x)/2)) |
| `acos(x)` near \|x\|=1 | √(1−x²) | half-angle 2·asin(√((1±x)/2)) |
| `acosh(x)` near x=1 | √(x²−1) | log1p(t + √(2t+t²)), t=x−1 |
| `atanh(x)` near \|x\|=1 | (1−x) in denom | IEEE ±∞ |
| `atan2(y,x)` at axes | y/x | IEEE table |
| `asech(x)` near x=1 | √(1−x²)/x | reduces to asin half-angle |

And the structural rhyme continues to `log1p`, `expm1`, `sinpi`, `cosm1` — all "the complementary-argument transform at a different scale." Every one of these primitives exists to make **1 + ε**-style arguments computable without cancellation.

This is a genuine fifth convergence. Day one:
1. sincos is the primitive (internal four-agent).
2. Shared-pass ≈ MomentStats with four generalizations (analogical).
3. Āryabhaṭa + Muratori = us (historical).
4. Navigator's scratch-that = precision regression (correctness, pushback).
5. **Every inverse function hides a log1p/expm1/sinpi-shaped primitive** (catalog convergence check).

The fifth is the most generative, I think. It says the trig family's "complementary-argument" move isn't trig-specific. It's the same pattern that birthed `log1p`, `expm1`, `sinpi`, `cosm1`, `log2`, `exp2`, `frexp+ldexp` — all the "one more than an exact thing" primitives across libm. Math-researcher proposes a meta-primitive `complementary_arg_transform(x, function_family) → (t, sign, branch)` that emits the transformed argument for any of these functions, feeding the same post-transform core.

That meta-primitive is what the rhyme is really pointing at. It's not just "inverse functions need log1p"; it's **the complementary-argument transform is a primitive in its own right** — an L1 atom, possibly, or an L2 primitive that every inverse function calls.

### 4.3 Correction to Entry 2

I said hyperbolics "likely have" a two-tier pattern with `exp_pair`. The catalog is clearer: the shared intermediate is **`HyperbolicExpm1Pair = (expm1(|x|), expm1(-|x|))`** — *expm1*, not exp. This matters because expm1 preserves precision at small x (sinh(x) ≈ x, cosh(x) ≈ 1+x²/2 — both need the low-order bits), while exp loses them. Same pair-of-values shape, but the inner function is expm1 because of the same complementary-argument logic as point 4.2.

This is a nice internal consistency check: the catalog's HyperbolicExpm1Pair and the near-unity cancellation rhyme are the *same finding* — the family-level convergence is just "every member of the family that has a 'near zero' or 'near unity' branch uses a complementary-argument primitive." Forward hyperbolic has it near x=0; inverse trig/hyperbolic has it near |x|=1.

---

## Day One Summary (final)

Five convergences. Three expedition-log entries + one shared-pass analysis + three garden entries + one campsite with notes.

The shape that emerged: **the trig family is structured around four shared intermediates, and each shared intermediate is answered by a convergence check**. Forward trig → sincos primitive. Forward hyperbolic → expm1 pair. Inverse trig/hyperbolic → log1p/complementary-arg transform. Cross-unit → angle-unit compatibility tag.

The meta-convergence: **every shared-intermediate question is really a question about where the complementary-argument structure lives in that family.** For forward trig, it's `sinpi` (the complement-of-π reduction). For hyperbolics, it's `expm1` (the complement-of-exp reduction). For inverses, it's log1p/half-angle (the complement-of-unity transform). All four families use the same structural pattern at different scales.

That's Day One's biggest finding. **Trig isn't just a periodic table; it's a periodic table where every cell's precision-preserving identity is a complementary-argument transform at that family's characteristic scale.**

*Entry 5 when pathmaker starts TRIG-13 proper and we see whether the architecture the catalog predicts actually materializes in the code.*

— the naturalist

---

## Entry 5 — The Architecture Expressed Itself (2026-04-14, morning)

pathmaker shipped TRIG-13 (tan/cot/sec/csc/sincos) and has started TRIG-14 (asin/acos). Both files landed while the team was running. I came back to camp and checked whether the shape we predicted actually materialized.

It did. Twice. Both kinds of way.

### 5.1 The forward-trig extraction

`tan.rs` opens with:

```rust
use super::sin::{eval_sincos, kernel_cos, kernel_sin, reduce_trig};
```

The shared kernel lives in `sin.rs` (still, not extracted to a new module), but `tan.rs` imports the four public functions and composes them. The module docstring reads:

> "All functions share one range reduction: `sin::reduce_trig(x)` → `(q, r_hi, r_lo)`. The kernel pair `(sin_k, cos_k)` is evaluated once via `sin::kernel_sin/cos`. The four derived functions differ only in how they combine the kernel outputs."

This is exactly Entry 0 Observation 0.5 in production code. 506 lines for `tan.rs` (vs. 932 for sin.rs) — the mass stayed in sin.rs because that's where the canonical kernel lives; tan.rs is composition + quadrant-aware-derivation tables + fdlibm-style near-π/2 polynomial fallback (which I predicted in Entry 0 thread #2 as the place where "tan is a one-liner" stops being true).

Note the conservative extraction choice: `sin.rs` stayed as the kernel owner; there is no `trig_kernel.rs`. Good enough for forward trig. When hyperbolics and inverse trig land, this will probably need to refactor to `trig_kernel/{forward,hyperbolic,inverse}.rs` — because importing the hyperbolic kernel from `sin::` would be a lie. But that's a future problem, not a today one. The sharing contract in spec.toml files points at `reads = ["trig_reduce", "sincos_kernel"]` as abstract tags — those tags don't care what module they live in.

**The map we drew came true.** Entry 0 pointed at `reduce_trig` and `eval_sincos` as shared infrastructure already 90% there; pathmaker pulled the trigger.

### 5.2 The fifth convergence materialized in asin.rs

Then I opened `asin.rs` and saw what I can only describe as the fifth convergence in production Rust. The module docstring reads (verbatim):

> "For |x| > 0.5: asin(x) = π/2 − 2·asin(√((1−|x|)/2)). This maps the numerically difficult near-1 region (where 1 − x² → 0) to the well-conditioned small-argument region. The inner sqrt is computed via `(1 − |x|) * 0.5` to avoid catastrophic cancellation."

Read that carefully. That is **the complementary-argument transform** named in garden entry "the-complementary-argument-2026-04-13.md" and catalog §"Convergence Check — Near-Unity Cancellation Rhyme" and Entry 4.2 — all shipped *as the implementation strategy* with exactly the justification the meta-pattern predicts.

The recenter-at-cancellation-prone-fixed-point move. `asin(x)` near |x|=1 rewrites via the complementary argument `t = 1 − |x|`, mapped through the half-angle identity into a small-argument evaluation. Same pattern as log1p, same pattern as expm1, same pattern as sinpi, now in asin.rs.

**The pattern the catalog named, the pattern the garden reflected on, and the pattern the expedition converged to — all landed in 258 lines of Rust before lunchtime.**

That's what it looks like when a structural finding is real rather than just pretty. It doesn't just *describe* the work — it *predicts* what the implementation will look like before the implementation exists. If the meta-primitive convergence had been a naturalist's pretty metaphor, asin.rs would have used direct `√(1 − x²)` and taken an ulp loss at the boundary. It didn't. It went straight for the complementary-argument transform with the exact rationale the garden entry named.

### 5.3 What this means

The day-one framings are now *load-bearing*. They're not aesthetic choices; they're predictive of code structure. This justifies the naturalist role in a way I wasn't sure about when I started: **noticing is upstream of implementation**. When the noticing is right, the implementation rhymes with it. When the noticing is wrong, the implementation diverges and the noticing gets corrected.

Today's check on pathmaker's code is a *convergence check on the convergence check itself*: the meta-finding predicted the implementation strategy, and the implementation strategy confirmed the meta-finding.

Two findings this implies for going forward:

1. **spec.toml files should declare the complementary-argument structure explicitly.** Every libm recipe that has a cancellation-prone regime should have a `[cancellation]` section naming (a) the fixed point, (b) the transform, (c) the post-transform core, (d) the inverse transform. This makes the pattern reviewable and makes the shared-intermediate sharing contract self-documenting. Routing this to pathmaker + math-researcher as a design rule proposal.

2. **Reviewing future recipes becomes quick.** For every new libm function, the naturalist/aristotle can ask: "what's the cancellation regime? what's the complementary transform? is the spec.toml's `[cancellation]` section honest?" Three questions. If the answer to any is "I don't know," the function isn't ready.

*Entry 6 when math-researcher or pathmaker responds to the `[cancellation]` section proposal, or when TRIG-15 (hyperbolics) lands and I can check whether expm1 pair materializes the way the catalog predicts.*

— the naturalist
