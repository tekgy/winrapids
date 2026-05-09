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

Examples:
- `log1p(x) = log(1 + x)`: F = 1, G = multiplicative; t = x; transform = "1 + ε"; inverse = identity
- `expm1(x) = exp(x) - 1`: F = 0, G = additive (output side); t = x; transform = identity; inverse = "result - 1"
- `sinpi(x) = sin(π·x)`: F = π·integer, G = multiplicative-of-π; t = x; transform = "π·"; inverse = identity
- `hypot(a, b) = √(a² + b²)`: F = 0 (degenerate axis case); G = additive-with-scale; transform = scale-then-square-then-add; inverse = scale-then-sqrt
- `cosm1(x) = cos(x) - 1`: F = 0, G = additive (output side); transform = identity; inverse = "result - 1"

The library factors into:
1. **The raw math** (what the function *is* mathematically)
2. **The complementary-argument transform** (where the cancellation lives)
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

## Open questions for math-researcher walk-through

1. **`pow(x, y)` factorization**. The composed identity `pow(x, y) = exp(y · log(x))` introduces an additional source of error (the multiplication `y · log(x)` happens at recipe-level, not inside the shared kernel). Does `pow` deserve its own kernel state, or is the composed form sufficient at the precision tiers tambear targets?

2. **`hypot(a, b)` as a complementary-argument transform**. Past-Claude listed it as an instance, but `hypot` doesn't have an obvious "fixed point" the way log1p does. Is the meta-primitive's group structure broader than the April 13 essay assumed, or is `hypot` a different shape that just shares the *precision-preservation* property?

3. **Gamma function (Lanczos approximation)**. Past-Claude flagged this as the exception — Lanczos is a different shape than log1p. Is the gamma family genuinely outside the complementary-argument frame, or does it fit at a different layer of factoring (e.g., as a shared intermediate for log_gamma + lgamma + beta)?

4. **The exp/log-side analog of the Periodic Table of Trig**. Past-Claude's April 13 trig-table enumeration (5 rows × 6 columns × 3 precision tiers) hasn't been done for exp/log. What does the exp/log periodic table look like? Probably a different shape — fewer "rows" since hyperbolic and pi-scaled don't have direct exp/log analogs, but more "columns" (exp / log / exp2 / log2 / exp10 / log10 / expm1 / log1p / pow / hypot / sinh / cosh / tanh / asinh / acosh / atanh / ...).

5. **Pi-scaled vs Tang-style reduction symmetry**. For trig, pi-scaled (`sinpi(x) = sin(π·x)`) uses an exact reduction (`round(2x)`) that bypasses Payne-Hanek. Is there an analogous "exact reduction" trick for exp/log? Specifically: `exp2(integer)` and `log2(power-of-2)` ARE exact in float; do they constitute an "exp2-scaled" family analogous to "pi-scaled"? If yes, the family widens.

6. **TrigKernelState's high/low decomposition for `r`**. The trig state carries `r_hi` and `r_lo` (double-double representation of the reduced argument). The exp/log analog likely needs the same, but the precision requirements are different (Payne-Hanek needs ~1200 bits of 2/π for large arguments; Tang's k·ln(2) reduction needs fewer bits but the multiplier `k` can be large). What's the right precision contract for `ExpKernelState.r` at each tier?

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
