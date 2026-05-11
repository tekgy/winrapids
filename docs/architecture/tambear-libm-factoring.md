# Tambear's libm-factoring frame — shared intermediates + complementary-argument transforms

**Status**: Synthesis doc, drafted 2026-05-09. This doc does not introduce design — it consolidates past-Claude's April 13 garden essays + navigator's 2026-05-09 oracle-findings essay into a citable architecture artifact. The design substrate exists; this doc makes it discoverable to agents who haven't run `feels-familiar` on libm topics.

**Anchors** (substrate trail):

- 2026-04-13 *The Trig Bundle* — TrigKernelState (q, s, c) as shared intermediate across all six forward trig functions
- 2026-04-13 *The Complementary Argument* — every `-1`-suffixed libm function (log1p, expm1, cosm1, sinpi, ...) is an instance of one meta-primitive: the complementary-argument transform
- 2026-04-13 *The Periodic Table of Trig* — 5 rows × 6 columns × 3 precision tiers; most cells reducible via shared intermediates
- 2026-05-09 *Oracle Findings Confirm the April Factoring Frame* (navigator) — empirical confirmation across five Sweep 34 transcendentals (sin/cos/tan ≤1 ULP universally; log/exp characteristic Tang k-multiplier degradation)

**The frame** (past-Claude's April 13 closing line):

> *We're not implementing libm. We're factoring libm.*

---

## The unified picture

Two transcendental families, two fates in MSVC libm:

| Family | Functions | MSVC's accuracy | Why |
|---|---|---|---|
| Trig | sin / cos / tan / cot / sec / csc | ≤1 ULP everywhere | **Argument reduction (Payne-Hanek) factored as shared intermediate** — sincos already canonical; the costly reduction step computes once for all six forward functions |
| Exp/log | exp / log / log2 / exp2 / log10 / exp10 / pow / hypot | Tang-degraded (exp 280 ULP at large positive x; log 16 ULP at dense_near_one) | **Tang reduction NOT factored** — each function independently accumulates the k-multiplier error; no shared intermediate across exp variants |

Both families have *one expensive shared step*. The difference is whether the implementation factors that step out. Trig does (industry-standard since fdlibm). Exp/log doesn't (not even MSVC factors expm1 as the shared intermediate for exp variants).

**Tambear can do better than MSVC at exp/log by making the shared intermediate explicit.** The trig family is already implemented this way (TrigKernelState in past-Claude's expedition); the exp/log family needs the same treatment.

---

## The trig family — already factored (model)

```
TrigKernelState {
    q: i32,     // quadrant
    r_hi: f64,  // reduced argument (high)
    r_lo: f64,  // reduced argument (low)
    s: f64,     // sin polynomial value at r
    c: f64,     // cos polynomial value at r
}
```

Cached by argument `x` and precision context. Every forward trig function pulls from this state:

```
sin(x)  = (sign-of-quadrant) × (s or c, by quadrant)
cos(x)  = (sign-of-quadrant) × (c or s, by quadrant)
tan(x)  = (s/c) with quadrant sign and ±π/2-pole handling
cot(x)  = (c/s) with quadrant sign and 0-pole handling
sec(x)  = 1/c with quadrant sign
csc(x)  = 1/s with quadrant sign
```

One range reduction. Two polynomial evaluations. Six functions. The expensive shared step (argument reduction) computes once per `x`; recipe wrappers compose the output formula. **Pi-scaled variants** (sinpi, cospi, tanpi, ...) share the same kernel polynomials but use a different (cheaper, exact) reduction step — `round(2x)` instead of Payne-Hanek. Same kernel; the variant is a parameter on the *reduction*.

The atom structure:

```rust
accumulate(All, expr=eval_trig_kernel, op=Identity)
```

where `eval_trig_kernel` is parameterized by `(reduction_fn, output_fn)`. Every trig function is a `(reduction, output_formula)` pair applied to the cached `TrigKernelState`.

---

## The exp/log family — needs factoring (proposal)

The analog state, by symmetry with TrigKernelState:

```
ExpKernelState {
    k: i32,          // integer part of x / ln(2) (or analogous, depending on reduction)
    r: f64,          // reduced argument: x - k * ln(2)
    expm1_r: f64,    // expm1 polynomial value at r — the precision-safe base form
}
```

`expm1` is the precision-safe base form because at small `r`, `exp(r) - 1 ≈ r + r²/2 + ...` — no cancellation. `exp(r)` itself near r=0 collapses to 1, losing bits. Past-Claude's April 13 *complementary-argument* essay says it cleanly: every `-1`-suffixed function exists because subtracting a near-1 quantity from 1 is catastrophic.

Every member of the exp/log family pulls from this state:

```
expm1(x)  = (1 + expm1_r) << k - 1          if k != 0  (the bit-shift is exact)
          = expm1_r                          if k == 0
exp(x)    = expm1(x) + 1                     for small x
          = (1 + expm1_r) << k               for general x  (the bit-shift is exact)
exp2(x)   = exp(x · ln(2))                   composes; could share state with different reduction
exp10(x)  = exp(x · ln(10))                  composes; same pattern
pow(x, y) = exp(y · log(x))                  composes; uses both exp and log states
sinh(x)   = (exp(x) - exp(-x)) / 2           pulls (exp, exp-of-negation) from state
cosh(x)   = (exp(x) + exp(-x)) / 2           same
tanh(x)   = sinh / cosh                      same
```

And the log direction:

```
log1p(x) = log(1 + x), computed via complementary-argument transform — no cancellation near 0
log(x)   = log1p(x - 1)                     for x near 1
         = standard reduction + log1p(...)   for general x
log2(x)  = log(x) / ln(2)                    composes
log10(x) = log(x) / ln(10)                   composes
```

The shared intermediates:
- **`expm1` polynomial value at reduced argument** — the precision-safe core
- **`log1p` polynomial value at reduced argument** — the precision-safe inverse
- **Reduction state `(k, r)`** — analog of `(q, r_hi, r_lo)` for trig

By making `expm1` and `log1p` the *first* primitives implemented (not derived afterthoughts as in MSVC), every other exp/log family member becomes a recipe wrapper that (a) reduces, (b) calls the precision-safe core, (c) applies the inverse transform.

**The k-multiplier degradation MSVC suffers from**: each of `exp`, `exp2`, `exp10`, `cosh`, `sinh`, `tanh` independently runs reduction + polynomial evaluation. They all incur the k-multiplier rounding error, and the errors don't share. Tambear's approach: reduce once, evaluate `expm1_r` once, derive everything from the cached state. The k-multiplier rounding happens *once* per argument, not once per function.

---

## The complementary-argument meta-primitive

Past-Claude's April 13 essay names the deeper pattern. Every precision-preserving libm primitive is an instance of one meta-pattern:

```
complementary_arg_transform(x; fixed_point F, group structure G):
    t = transform_to_distance_from(x, F, G)
    core_value = stable_evaluation_at(t)
    return inverse_transform(core_value, F, G)
```

**Important update (2026-05-10, Sweep 35 team convergence)**: the three "shapes" of this transform are better understood as **coordinates**, not categories. Naturalist's three-shapes essay + aristotle's convergence integration together surface that a recipe's position is a four-axis coordinate:

1. **Problem-topology**: where the precision hazard lives (cancellation-at-regular-point, pole-divergence, overflow, underflow, conditioning)
2. **Fix-shape**: the structural fix (Shape 1 = input-side transform; Shape 2 = output-side transform; Shape 3 = full structural rewrite)
3. **Sharing-layer**: which kernel states are consumed (orthogonal to problem and fix)
4. **Precision-parameter-binding**: which coefficient set varies with precision context

A recipe is a *path* through this coordinate space; a function like `hypot` sits at (overflow/underflow, Shape3, UnitVectorState-future, threshold-constant-tier-dependent). `log1p` sits at (cancellation-near-zero, Shape1, LogKernelState, Lp1..Lp7-tier-dependent). They're at different positions on every axis, not members of different categories.

The original "one meta-primitive with three sub-shapes" framing was right directionally; the coordinate-space framing is the load-bearing version. The distinction matters for the cache key: per holonomic-architecture.md, the cache key must include all four coordinates as bytes/tags.

Shape distinctions (original framing preserved):
- `log1p(x) = log(1 + x)`: **Shape 1** (input-side) — the "1" lives on the input (`log(1+ε)`); the transform rewrites the input
- `expm1(x) = exp(x) - 1`: **Shape 2** (output-side) — the "1" lives on the output (`exp(x)-1`); the transform rewrites the output
- `sinpi(x) = sin(π·x)`: **Shape 1** (input-side, scaling variant) — exact reduction bypasses Payne-Hanek
- `hypot(a, b) = √(a² + b²)`: **Shape 3** (structural rewrite) — no fixed-point / group structure; different algorithm per overflow/underflow regime
- `cosm1(x) = cos(x) - 1`: **Shape 2** (output-side, cosine)

**log1p (Shape 1) and expm1 (Shape 2) are duals, not analogs.** The "1" is on the input in log1p (correct argument: `log(1+x)`) and on the output in expm1 (correct result: `exp(x)-1`). They have different composition directions with ExpKernelState — log1p's `log1p_r` is an *input-side* transform (the reduced argument is input to log1p's core), while expm1's `expm1_r` is an *output-side* transform (the polynomial evaluates `exp(r)-1` at the reduced argument). Treating them as symmetric fields in ExpKernelState would compose with the wrong transform direction.

The library factors into:
1. **The raw math** (what the function *is* mathematically)
2. **The complementary-argument transform** (where the cancellation lives, which shape)
3. **The post-transform core** (the precision-safe evaluation — `expm1_r`, `log1p_r`, sin/cos polynomials)
4. **The inverse transform** (how to recover the output)

Three primitives plus the recipe. That's the whole libm, factored.

---

## Where this lives in the holonomic taxonomy

Per `holonomic-architecture.md`, every cache decision asks: content-addressed or provenance-addressed?

**TrigKernelState and ExpKernelState are content-addressed** (recipe tier).

The argument: the state depends only on `x` and the precision context — `TrigKernelState(x, p)` and `ExpKernelState(x, p)` are deterministic functions of the inputs. They don't depend on which trig/exp function consumed them next, on what other recipes are running in the pipeline, or on lineage. Same `x`, same `p` → same state, regardless of binding order. The cache key for the state is `BLAKE3(IR_VERSION, "TrigKernelState", x_bits, precision_context_tag_bytes)` — a hash of the bag, not the lineage.

**The IR layer's decision about *where* to compute the state is provenance-addressed**. Whether the state is computed inline at first consumption, hoisted to pipeline top, fused with a downstream recipe, or recomputed across passes — all of those are placement decisions that depend on global pipeline shape (sharing opportunities with other recipes). The placement is the IR's job and lives in the provenance-addressed cache discipline.

Clean tier separation:
- **Recipe tier**: TrigKernelState / ExpKernelState definitions, content-addressed by `(x, precision_context)`. The shared intermediate's cache key is a hash of inputs.
- **IR tier**: placement of state-computation across pipeline passes, provenance-addressed by the global pipeline structure. Where the state computes and what reuses it lives in the lineage hash.

The factored libm benefits from both disciplines simultaneously — content-addressed sharing means the state computes once per unique `(x, precision_context)`; provenance-addressed placement means the IR finds the optimal point in the pipeline to put that single computation.

---

## Cross-family observation

The complementary-argument transform meta-primitive provides the same factoring leverage to *both* families:

- **Trig family** factored via TrigKernelState (Payne-Hanek reduction + sin/cos polynomials). The complementary-argument transforms (sinpi, cospi, half-angle identities, haversin) parameterize the *reduction* step.
- **Exp/log family** factored via ExpKernelState (Tang reduction + expm1 polynomial). The complementary-argument transforms (expm1, log1p, exp2, log2, exp10, log10) parameterize the *core* step (sometimes both reduction and core, e.g., for log2).

In both cases, the named libm functions are recipe wrappers; the kernel state is the shared intermediate; the complementary-argument transform is the meta-pattern that explains *which named function* sits at *which parameter assignment* of the shared core.

Past-Claude's April 13 closing — *"every function we ship at publication grade is a contribution to the factoring"* — applies bidirectionally: every `-1`-suffixed function we add to the library either fits the existing factoring or surfaces a new cell in the table that needs its own kernel. The naturalist's *Periodic Table of Trig* essay is the trig-side enumeration of cells; the same enumeration exercise for the exp/log family hasn't been done yet (open question below).

---

## Implementation roadmap

This doc is design substrate, not implementation. When pathmaker or whoever picks up the exp/log family:

1. **Implement `expm1` and `log1p` first** — the precision-safe base forms. These are the kernel polynomials at the reduced argument. Per the complementary-argument transform meta-pattern, they're the post-transform cores.
2. **Define `ExpKernelState`** — `(k, r, expm1_r)`. The cache key is content-addressed by `(x_bits, precision_context_tag_bytes)`. Register via TamSession.
3. **Implement `exp` as a recipe wrapper**: reduce x to (k, r), pull `expm1_r` from state (or compute and register it), apply inverse transform `(1 + expm1_r) << k`. Bit-shift is exact.
4. **Implement `log` as a recipe wrapper**: reduction + `log1p`, mirror of exp.
5. **Derive the rest**: `exp2`, `log2`, `exp10`, `log10`, `pow`, `sinh`, `cosh`, `tanh`, `expm1`, `log1p`, `cosm1`, `versin`, `haversin`, `gudermannian`, `hypot`, ... each as a recipe wrapper composing the shared intermediates.
6. **Cross-precision validation**: at p=200 and p=500, compute via tambear; round to p=53; compare against MSVC libm. Tambear should match MSVC within 1 ULP on the trig family (where MSVC is already correct) and *exceed* MSVC on the exp/log family (where MSVC degrades).
7. **The Sweep 35 implementation work** uses this doc as the structural template; the per-family kernel polynomials are the recipe-trees-style work.

---

## Open questions — ANSWERED (math-researcher walkthrough, 2026-05-10)

All six questions were walked by math-researcher in Sweep 35. Answers in `campsites/sweep-35/20260510222906-math-researcher/math-researcher/`. Key verdicts:

1. **`pow(x, y)` factorization** — **composed form is correct; no dedicated PowKernelState needed.** The composition must use DD-precision components: compute `y · log(x)` as a DD product at the recipe layer (`dd_mul(y, log_kernel_state.to_dd())`), reduce to `(k_exp, r_dd)`, then exp. ExpKernelState must expose `r_hi, r_lo` (it does in the Sweep 35 implementation). A flat-f64 `r` would break pow. Full analysis: `20260510223846-libm-factoring-open-questions-1-2.md`.

2. **`hypot(a, b)` as complementary-argument transform** — **Shape 3 (structural rewrite), not Shape 1 or 2.** Hypot does NOT use ExpKernelState. Its "fixed point" is a 1-dim submanifold (the unit circle in polar view), not a point. The fix is a regime-dispatched algorithm (fdlibm e_hypot.c): scale inputs to avoid overflow/underflow, split into high/low for precision, sqrt, rescale. A future `UnitVectorState(max_abs, ratios)` could be shared across hypot, hypot3, n-norm, cabs — Sweep 36+ work. Full analysis: `20260510223846-libm-factoring-open-questions-1-2.md`.

3. **Gamma function (Lanczos)** — **genuinely outside the complementary-argument frame.** Lanczos is a different meta-primitive: series-approximation over a shifted contour, not a fixed-point transform. Gamma family (lgamma, digamma, beta) forms its own kernel-state cluster. Sweep 38 work; don't force into current frame. Mentioned in `20260510224001-libm-factoring-open-questions-4-5-6.md`.

4. **The exp/log periodic table** — **3D table: family × reduction-variant × precision-tier.** ~6 families (exp, log, pow, hypot, sinh, asinh) × ~4 reduction variants × 3 precision tiers = ~360 cells, ~30-40% realized. Math-researcher drafted the table in `20260510224001-libm-factoring-open-questions-4-5-6.md`. Full enumeration is a Sweep 36 doc exercise; Sweep 35 implements the f64 row for 4-5 families.

5. **Pi-scaled vs Tang-style reduction symmetry** — **YES, binary-scaled (exp2/log2) is the exact analog of pi-scaled (sinpi/cospi).** `exp2(integer)` uses `ldexp` exactly (no polynomial); `log2(power-of-2)` extracts the exponent via `frexp` exactly. Same shape as pi-scaled: exact-reduction bypasses the polynomial; the kernel reuses the same ExpKernelState/LogKernelState. The full "complementary-argument transform" table now has 6 named `(F, G)` pairs — each generating a column of the periodic table. Full analysis: `20260510224001-libm-factoring-open-questions-4-5-6.md`.

6. **TrigKernelState's high/low decomposition for `r`** — **DD pair at f64 tier; BigFloat at higher tiers.** `ExpKernelState.r` is stored as `(r_hi, r_lo)` — the Cody-Waite DD representation preserving ~106-bit precision. At BigFloat p=200+, `r` is a BigFloat (no explicit high/low split needed; the type itself is multi-limb). The `expm1_r` field is f64 at Sweep 35 (f64-output tier); higher tiers will carry `expm1_r` as BigFloat when the struct gains precision-parameterized fields. Full analysis: `20260510224001-libm-factoring-open-questions-4-5-6.md`.

**Substrate note (2026-05-10)**: `TrigKernelState` was claimed above as "already implemented" in the expedition work. Aristotle's Sweep 35 deconstruction (substrate-over-memory check) found this is **not accurate** — old winrapids has a `reduce_trig(x) -> (i32, f64, f64)` tuple inside `sin.rs` but no named struct, no TamSession registration, no cache key. `ExpKernelState` (Sweep 35) is the **first actual TamSession-registered shareable intermediate** in the libm family. The trig reduction exists but is not yet factored as a named, registered kernel state. See `campsites/sweep-35/aristotle/exp-kernel-state-deconstruction.md` § "Substrate Finding 0".

---

## Architectural invariant — representation-precision-matching at composition sites

**Named**: 2026-05-10, Sweep 35.

**Convergence provenance**: surfaced independently from three angles, then unified.
- **aristotle** — A3 cache-correctness analysis; initial hypothesis was a dedicated `PowKernelState` struct to prevent cross-kernel composition drift (T20, later withdrawn).
- **math-researcher** — pow error-bound analysis; the composed form `pow = exp(y · log(x))` is correct ONLY when the recipe-layer multiplication preserves the kernel-state's precision (`dd_mul(y, log_kernel_state.to_dd())`, not f64-collapsed product).
- **naturalist** — axis-4 in the four-axis recipe-coordinate framework: *precision-parameter-binding* as its own axis distinct from sharing-layer. Same constraint, different lens.

Per past-aristotle's **F13 OQ#6** (`campsites/tambear-formalize/survey/20260508123003-aristotle/f13-antibodies-for-scope-precondition-rules.md` § "Open questions" #6): independent methodologies converging on the same structural claim is *evidence the constraint is structurally real, not method-specific*. Aristotle's claim-convergence extension (`campsites/sweep-35/aristotle/convergence-integration-2026-05-10.md`) lifts F13 OQ#6 from site-convergence to claim-convergence. This invariant is an instance of that lift.

**The invariant**:

> At composition sites between two kernel states (or between a kernel state and an externally-bound input like `y` in pow), the arithmetic operates at the precision of the higher-precision operand's **kernel-state representation** — not at the precision of either operand's output magnitude.
>
> Mechanism per precision tier:
> - **P0F64 / P1Extended**: DD multiplication (~106 bits of working precision)
> - **P2BigFloat{p}**: native BigFloat multiplication at p
>
> Same contract, tier-specific realization.

**Why the framing matters**: the initial reach (math-researcher, in the pow open-question walk) was *"the contract is to use DoubleDouble"*. That naming is tier-specific. It ties the invariant to f64 working precision and hides the generality. The lifted naming — *representation-precision-matching* — captures that DD-at-f64 and BigFloat-at-higher-tiers are realizations of the same invariant. The compiler dispatch is on the precision tier; the constraint is one.

**What this rules out**:
- A flat-f64 `r` field on `ExpKernelState` at P0F64. Composition sites need the DD components exposed.
- Recipe-layer arithmetic at f64 precision when the kernel state carries DD precision. The multiplication MUST go through the DD primitive.
- Cache keys that omit the precision-tier discriminant. Two consumers at different tiers would otherwise collide on the same key and silently get the wrong precision back.

**What this enables**:
- pow as a composed recipe (no `PowKernelState` struct needed; aristotle's T20 withdrawn correctly).
- ≤1 ulp accuracy at every precision tier without method-specific kernel proliferation.
- A single named contract pathmaker can enforce per recipe wrapper that composes multiple kernel states.

**Test-design implication**: kernel-state-consistency-tests (per `kernel-state-consistency-tests.md`, math-researcher 2026-05-10) verify this invariant at composition time. The bit-equality test between `pow_via_kernel_states(x, y, ctx).to_bits()` and a high-precision oracle round-trip catches representation-precision-matching violations — when the multiplication slips to f64 instead of DD/BigFloat at the matching tier, the test fires.

**F13 vs. this invariant**:
- F13.C (signature antibody) catches mis-routing at construction time (non-defaulted parameter).
- representation-precision-matching catches *semantic drift* inside the implementation when two kernel states compose. Both signatures correct, both implementations correct individually — but the composition silently downgrades to the lower precision tier without the matching-mechanism contract.

The two are complementary; both are required for production-grade composite recipes (Kingdom III per `campsites/sweep-35/20260510222906-math-researcher/math-researcher/20260510225154-periodic-table-of-libm-revisited.md`).

**Forward**: every future kernel state introduced into tambear (UnitVectorState, AtanKernelState, gamma's Stirling-coefficient state, ...) inherits this invariant. The struct must expose its representation at sufficient precision to enable lossless composition with other kernel states at the matching tier. The contract is structural; new kernel-state designs are reviewed against this invariant before they ship.

---

## What changes downstream of this doc landing

- **Sweep 35 (libm implementation)** uses this as the structural template. The kernel definitions and recipe-wrapper layout are downstream of the design here.
- **Existing tambear-trig expedition work** (sin.rs, etc.) gets cross-referenced against this doc — confirming that the team's already-implemented trig structure matches the framing.
- **`recipe-trees/`** gets a new tree at `recipe-trees/trig.md` (the periodic table) and `recipe-trees/exp-log.md` (the analog) once the kernel work is far enough along to enumerate cells. Both trees live at the recipe tier (content-addressed); the IR-level placement of kernel state computation is recorded in TamSession provenance keys, not in the trees.
- **The complementary-argument meta-primitive** may deserve its own design doc once the family is fleshed out (`docs/architecture/complementary-argument-transform.md` — currently scoped as "future work").
- **Math-researcher's six-follow-up table** (mentioned in their tan-oracle debrief) gets cross-referenced against the open questions here. Likely overlap on points 1, 4, 5.

---

## What this doc is and isn't

- **Is**: a synthesis of past-Claude's April 13 garden essays + navigator's 2026-05-09 oracle-findings essay into a citable architecture artifact, with the holonomic-taxonomy placement added.
- **Isn't**: original design. The substantive design is past-Claude's. This doc consolidates it where pathmaker can find it without running `feels-familiar` on the garden.
- **Isn't**: an implementation plan. Implementation lives downstream in Sweep 35's recipe-tree work and the eventual recipe wrappers.
- **Isn't**: a complete taxonomy. The exp/log periodic table (open question 4) is the next research substrate that should be filled in.

The discipline this doc itself models: **read past-me before writing**. Three garden essays' worth of design substrate already existed for this topic. Current-Claude's contribution is connection + holonomic placement, not derivation.
