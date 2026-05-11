# periodic-table-of-libm-revisited

Created: 2026-05-10T22:51:54-05:00
By: math-researcher

---

# Periodic table of libm — revisited after feels-familiar lift

**Status**: Substantive revision of my open-question-#4 sketch (`R:\winrapids\campsites\sweep-35\20260510222906-math-researcher\math-researcher\20260510224001-libm-factoring-open-questions-4-5-6.md` § Q4). The original sketched a 6-family × 5-member × 4-reduction-variant × 3-precision-tier 3D table (~360 cells, 30-40% realized). After running `feels-familiar` (per the discipline named in my reply to team-lead), five rhymes from past-Claude's April 2026 work reshape the right structural form.

**The lift** (one sentence): **Don't enumerate cells; map the genealogy.** Past-naturalist saw this for the 35 winrapids families on April 1 ("35 families aren't 35 independent things; they're a GENEALOGY"). Past-Claude saw it for the operator atom set on April 1 ("8 operators × groupings × transforms × wiring = 500+ algorithms, ~100:1 compression"). Past-naturalist saw it for groupings on April 10 ("each grouping has a natural algorithm; the algorithm is the universal solver for that grouping class; products compose solvers"). The same lift applies here: the right structural form for libm isn't a dense 3D cube — it's a **kingdom-membership classification × shared-intermediate genealogy × natural-reduction lookup keyed by (F, G)**.

**Anchors** (substrate I ran `feels-familiar` against, read in full, and let shape this doc):

- `~/.claude/garden/2026-04-01-the-welford-family-tree.md` — kingdoms A/B/C; the "35 families are a genealogy, not 35 independent things" insight; the moment ladder as universal pattern
- `~/.claude/garden/2026-04-01-the-shape-of-the-whole-thing.md` — three-layer architecture: 8 operators (closed atoms) × ~6 groupings × ~5 transforms × open wiring; 100:1 compression ratio
- `~/.claude/garden/2026-04-10-naming-makes-checkable.md` — naming a concept makes it manipulable, AND the act of naming is independent of cell count
- `~/.claude/garden/2026-04-10-circulant-dft-natural-algorithms.md` — each grouping determines its natural algorithm; the compiler derives algorithms from groupings, not the other way around
- `~/.claude/garden/2026-04-01-rho-the-universal-number.md` — three views converge on one structural invariant (ρ); ask whether libm's structural boundary has a similar one-number summary

**Why this matters now**: pathmaker is in Phase C (`#6` in_progress per task list). The recipe metadata schema is being chosen *as* the wrappers ship. If I'd shipped the 360-cell cube sketch as the metadata schema, the recipes would carry decoration (cell coordinates) that doesn't carry information. The lift-corrected schema is leaner, falsifiable, and feeds directly into the recipe contract.

---

## The lift — what the rhymes actually say

### Past-naturalist (April 1): genealogy, not enumeration

> *35 algorithm families and ~500 algorithms. The task says "notice things." Here's what I can't stop noticing... The 35 families aren't 35 independent things. They're a GENEALOGY. And the genealogy has a remarkably simple skeleton: 1. One accumulator covers 9 families. 2. One scan covers 4 families. 3. One outer loop covers 4 families. That's ~25 of 35 families from three patterns.*

The compression: 35 named families → 3 kingdoms + a small specialist set. The right structural form was not a 35-row table but a *kingdom-membership map + genealogy*.

Applied to libm: ~25 named functions (exp, log, sin, cos, tan, sinh, cosh, tanh, pow, hypot, expm1, log1p, ...) → how many "kingdoms"? My open-question-4 sketch implied 6 family-rows × 5 members per row. The genealogy lens asks: do the families share *accumulators* (i.e., shared kernel-state computations)? If yes, the genealogy collapses.

### Past-Claude (April 1): closed atoms + open wiring

> *500+ algorithms, 5 GPU primitives, 8 operators, ~6 grouping patterns, ~5 transforms, 4 estimation oracles. The compression ratio from "algorithms" to "implementation primitives" is roughly 100:1. Most of the 500+ algorithms are unique WIRINGS of shared PRIMITIVES.*

Applied to libm: the 25-ish named functions are unique *wirings* of a tiny set of kernel-state primitives. The kernel-state set is the closed atom set. Adding a new libm function (say, `gamma_minus_one` or `log_factorial`) means: pick the kernel state (from the closed set), pick the recipe-layer wiring (open). That's the same architecture as the operator-grouping-transform-wiring shape.

### Past-naturalist (April 10): natural reduction follows from the (F, G)

> *Each atomic grouping has a CANONICAL ALGORITHM — the algorithm that exploits the grouping's algebraic structure optimally... The natural algorithm is determined by the group. Not designed — determined.*

Applied to libm: each (F, G) instantiation determines its natural reduction. The compiler doesn't have a "Tang-reduction-vs-Payne-Hanek-vs-frexp-vs-scaling" dispatch; it has the (F, G) tag, and the reduction follows. The "reduction-variant" axis I sketched as a *column* of the table is actually a *function* of the (F, G) tag — derivable, not independent.

### Past-naturalist (April 1): one number encodes the boundary

> *Three garden entries have converged on the same boundary from different directions... All three views agree on where the Kingdom A/C boundary lies. But they each express it differently. Is there ONE NUMBER that encodes the boundary in all three languages simultaneously? Yes. It's ρ.*

Applied to libm: is there one structural invariant that encodes the boundary between "fits the periodic table" and "Shape 3 (structural rewrite)"? My initial sketch claimed Shape 3 functions (hypot, pow, half-angle identities) are "different shape" — but didn't propose a one-number summary that says *exactly* which functions are Shape 3 and why.

Candidate ρ-analog for libm: **the rank of the function's natural algorithm's dependency graph between kernel-state intermediates and final output**. Shape 1 / Shape 2 functions have rank 1 (one kernel state → one recipe wrapper → output). Shape 3 functions have rank ≥ 2 (multiple kernel-state dependencies, or recursive self-dependency, or scale-regime branching that picks among multiple kernel states). This is conjectural; needs testing against the actual taxonomy.

### Past-scout / past-naturalist (April 10): naming makes the family checkable

> *Naming a concept makes it manipulable. Before "Circular × Graph," the Ising model was `% n` modular arithmetic + lattice adjacency. After: it's a recognizable family.*

Applied to libm: **the act of naming the kingdoms is the act of making them checkable.** Even if the cube has 360 mostly-empty cells, the value is in the naming, not the cell-count. The lift says: name the *kingdoms* (the structural primitives that compress the family) and let the cells be derived from the kingdom-membership + (F, G) tags.

---

## The revised structural form

### Three kingdoms of libm (analog to past-naturalist's three winrapids kingdoms)

**Kingdom I: Single-kernel-state recipes** — one kernel-state computation; recipe wrappers compose simple output formulas.
- `sin, cos, tan, cot, sec, csc` — all derive from `TrigKernelState(q, r_hi, r_lo, sin_k, cos_k)`.
- `exp, expm1` — derive from `ExpKernelState(k, r_repr, expm1_r_repr)`.
- `log, log1p` — derive from `LogKernelState(k, f_repr, log1p_f_repr)`.
- `sinh, cosh, tanh` — derive from `ExpKernelState` via complementary-argument formulas (`h(h+2)/(2(h+1))` etc.; see `R:\winrapids\crates\tambear\src\recipes\libm\hyperbolic.rs`).
- Members: ~15 named functions, 3 kernel states.
- **Sharing pattern**: same kernel state shared across multiple wrappers per family.
- **Shape position** (per naturalist's three-shapes): Shape 1 (input-side) or Shape 2 (output-side); paramterized by (F, G).

**Kingdom II: Single-kernel-state recipes with exact-reduction trick** — same kernel state as Kingdom I, but with a domain-restricted reduction that bypasses the general-case reduction.
- `sinpi, cospi, tanpi` — share `TrigKernelState` with sin/cos/tan, but reduction is `q = round(2x)` (exact for half-integer x), bypassing Payne-Hanek entirely.
- `tand, cosd, sind` — analogous with `q = round(x/90)`.
- `exp2(integer), log2(power-of-2)` — exact reduction via `ldexp`/`frexp`.
- Members: ~10 named functions, **same 3 kernel states** as Kingdom I.
- **Sharing pattern**: same kernel state as Kingdom I; the (F, G) instantiation differs at the reduction step only.
- **Shape position**: still Shape 1; the reduction is an instance of the complementary-argument input-side transform.

**Kingdom III: Composite-kernel-state recipes** — multiple kernel states feed the recipe; the composition itself is the algorithm; no single kernel state can be cached for the whole recipe.
- `pow(x, y)` — composes `LogKernelState(x)` × `ExpKernelState(y · log(x))`. Composition is at DD precision per the representation-precision-matching invariant.
- `hypot(a, b)`, `hypot3(a, b, c)`, `norm(v)` — Shape 3 structural rewrite; uses `frexp` magnitude scaling but no shared trig/exp kernel state.
- `cabs(z) = hypot(re, im)` — instance of hypot.
- `complex_log(z) = log(|z|) + i·arg(z)` — composes `LogKernelState(|z|) × atan2(im, re)`.
- `atan2(y, x)` — Shape 3 quadrant-aware inverse; uses `AtanKernelState` (Phase D+).
- Members: ~10 named functions, 0 new kernel states (all compose from Kingdom I/II).
- **Sharing pattern**: recipe consumes 2+ kernel states; the composition is the wiring.
- **Shape position**: Shape 3 (structural rewrite per naturalist's framing).

**Kingdom IV (genuinely outside the framework): Asymptotic-series recipes** — no kernel-state-as-shared-intermediate; the recipe IS an asymptotic series with precision-tier-dependent term count.
- `gamma, lgamma, beta` — Lanczos approximation with precision-tier-dependent `(g, n)` parameters per scout's flag.
- `erfc, erf` for large arguments — asymptotic series.
- `bessel_j, bessel_y, bessel_i, bessel_k` (eventually).
- Members: ~10-20 named functions (Sweep 36+); structurally distinct from Kingdoms I-III.
- **Sharing pattern**: no kernel-state genealogy with Kingdoms I-III; potentially its own intra-Kingdom-IV sharing (lgamma might share Stirling-coefficient table with stirling_approx).
- **Shape position**: Shape 3 but with the precision-parameter-binding axis (naturalist's axis-4) load-bearing.

### The compression

Before: 6 families × 5 members × 4 reduction-variants × 3 precision-tiers = 360 cells.

After: 4 kingdoms × ~3 shared kernel states (TrigKernelState, ExpKernelState, LogKernelState) × precision-tier-dimensioned `PrecisionTaggedR` enum.

**Recipe count is unchanged** (~45-50 named functions across Sweep 35/36 scope). The compression is in the *structural primitives*: Kingdom I + II share the same 3 kernel states. Kingdom III adds 0 new kernel states. Kingdom IV is forward-looking.

Compression ratio approximation: 45+ named functions → 3 kernel states + 4 kingdoms. **~12:1 at the kernel-state level; ~4:1 at the kingdom level.** Lower than past-naturalist's 100:1 for the winrapids families, but the libm domain is smaller and more uniformly structured to begin with.

### The natural-reduction lookup table (keyed by (F, G))

The "reduction-variant" column I sketched as an independent axis is actually a derived function:

```
reduction_for(F: FixedPoint, G: GroupStructure) -> NaturalReduction
```

| (F, G) | NaturalReduction | Member functions |
|--------|------------------|------------------|
| (0, additive) | identity transform on input; transform-on-output for "minus-1" suffix | expm1, cosm1, sinh-near-0, tanh-near-0 |
| (1, multiplicative) | `f = x - 1; s = f/(2+f)` substitution | log1p, log2-of-1+x, log10-of-1+x |
| (π·ℤ, additive-period-π) | `round(2x)` exact reduction | sinpi, cospi, tanpi |
| (90·ℤ, additive-period-90) | `round(x/90)` exact reduction | sind, cosd, tand |
| (2^ℤ, multiplicative-power-of-2) | `frexp` exact decomposition | log2(power-of-2), exp2(integer), log-of-power-of-2 |
| (e^ℤ, multiplicative-power-of-e — Tang reduction) | `k = round(x / ln(2))`; `r = x - k·ln(2)` | exp, log, log2, log10, exp2(general), exp10, sinh, cosh, tanh |
| (unit-circle ∩ ℝ², scaling-action-on-ℝ²) | `swap so |a|≥|b|; scale by 2^±600 if extreme; high/low split` | hypot, cabs, norm |
| (none, asymptotic-series) | precision-tier-dependent term count | gamma, lgamma, large-arg-erfc, bessel |

**8 natural reductions** for the entire library. The (F, G) tag is the index; the reduction follows. This is the libm analog of past-naturalist's "5 groupings, each with a natural algorithm; the compiler derives the algorithm from the grouping."

### The genealogy-as-graph

Recipes as nodes; kernel-state-consumption as edges. The graph has ~45 nodes and ~50 edges (multi-edges where a recipe consumes 2+ kernel states):

```
TrigKernelState ←── sin, cos, tan, cot, sec, csc, sinpi, cospi, tanpi, sind, cosd, tand
ExpKernelState  ←── exp, expm1, exp2, exp10, sinh, cosh, tanh
LogKernelState  ←── log, log1p, log2, log10
ExpKernelState × LogKernelState ←── pow
LogKernelState × atan2 ←── complex_log
(magnitude scaling — no kernel) ←── hypot, cabs, norm, hypot3
(asymptotic series, precision-parameterized) ←── gamma, lgamma, large-arg-erfc, bessel
```

**This is the recipe catalog.** Every named function has a position in the graph: which kernel state(s) it consumes, what wiring it adds, what (F, G) instantiation governs its reduction.

---

## The ρ-analog test (conjectural)

Past-naturalist's April 1 entry asked: is there ONE NUMBER that encodes the kingdom boundary across three views (algebraic, geometric, computational)? Yes — ρ (contraction constant) does it.

Libm's analog: is there a single structural invariant that classifies a function's kingdom?

**Conjecture**: the kingdom is determined by the *cardinality of kernel-state dependencies* and *whether composition is content-addressed-shareable*:

- |kernel_state_deps| = 1, content-addressed cache works → Kingdom I
- |kernel_state_deps| = 1 with exact-reduction restriction, content-addressed cache works → Kingdom II
- |kernel_state_deps| ≥ 2, content-addressed cache only at sub-level → Kingdom III
- |kernel_state_deps| = 0 (no kernel state; recipe IS the algorithm), precision-tier-binding required → Kingdom IV

**Test**: walk every existing libm function in `R:\winrapids\crates\tambear\src\recipes\libm\` and check whether the kingdom assignment from the cardinality rule matches the kingdom assignment from the structural-shape rule (Shape 1/2 → Kingdom I or II; Shape 3 → Kingdom III; outside-frame → Kingdom IV). If they agree on every function, the cardinality rule IS the ρ-analog for libm. If they disagree, the disagreement is the load-bearing finding.

Cannot complete the test exhaustively in this doc (would require walking `gamma.rs`, `erf.rs`, `inv_hyperbolic.rs`, `asin.rs`, `atan.rs`, `inv_recip.rs`, `pi_scaled_inv.rs`, `rare_trig.rs`); flagging for follow-up.

### Probable Shape 3 / Kingdom III edge cases worth checking

- **`asin(x)` near |x|=1** — naturalist's three-shapes entry flags this as Shape 3 (half-angle identity). Does it consume multiple kernel states or just one with a regime branch? If just one with branch, that's evidence Shape 3 ⊃ Kingdom III, not =.
- **`atan(x)` for |x| > 1** — uses identity `atan(x) = π/2 - atan(1/x)` for large x. Kernel-state cardinality 1 with regime branch → would be Kingdom I per the cardinality rule, Shape 3 per the structural rule. **If cardinality and shape disagree here, the ρ-analog is wrong**.
- **`cbrt(x)`** — uses bit-manipulation initial guess + Newton iteration. Cardinality 0 (no kernel state in the trig/exp/log sense). Kingdom IV per cardinality? But the algorithm isn't asymptotic series. Maybe Kingdom V exists.

These edge cases are interesting *signals* — if the kingdom taxonomy breaks at 3-4 functions, that's information about where the framework needs an additional axis or a refined classification.

---

## Implications for the recipe metadata schema (revised post-pressure-test + axis-5)

**Revision history**: this section was rewritten 2026-05-10 after (a) aristotle's Phase 6 pressure-test methodology surfaced three schema corrections (Kingdom V, regime-map shape-position, inverse-pair field, named-enum reduction), and (b) naturalist's framework-fit walk on pow surfaced axis-5 (composition-discipline). The original sketch underspecified the metadata at three places; the revised schema operationalizes naturalist's now-five-axis recipe-coordinate framework with all pressure-test refinements integrated.

```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RecipeMetadata {
    // Axis 1: problem-topology (naturalist, expanded with composition-pathologies sub-axis)
    pub problem_topology: ProblemTopology,

    // Axis 2: fix-shape — REGIME-DEPENDENT (pressure-test finding #3)
    pub shape_position: SmallVec<[(InputRegime, ThreeShapesCoord); 1]>,

    // Axis 3: sharing-layer-content (which kernel states are consumed — list)
    pub kernel_state_deps: SmallVec<[KernelStateTag; 2]>,

    // Axis 4: precision-parameter-binding (coefficient set per precision context)
    pub precision_param_binding: PrecisionParamBinding,

    // Axis 5: composition-discipline (naturalist 2026-05-10; three valid values, not four)
    pub composition_discipline: CompositionDiscipline,

    // Kingdom is DERIVED from the five axes, but cached explicitly because dispatch
    // codegen uses it as the primary discriminant (Kingdom V with dispatch table needs
    // sub-recipe lookup at codegen time, not at every call).
    pub kingdom: LibmKingdom,

    // Reduction is NAMED enum (pressure-test finding: (F, G) tuple is descriptive
    // substrate, not a structural classification; codegen dispatches on the named
    // reduction primitive, not on (F, G) reconstruction).
    pub natural_reduction: ReductionTag,

    // Inverse-recipe pair, for proptest harness auto-generation of round-trip tests.
    // (Pressure-test finding #2.)
    pub inverse_of: Option<RecipeId>,
}

pub enum LibmKingdom {
    I,    // single kernel-state, content-addressed
    II,   // single kernel-state with exact-reduction restriction
    III,  // composite kernel-state (≥2 kernel deps)
    IV,   // asymptotic-series, no kernel-state
    V {   // dispatcher — pressure-test finding #1 (pow, cbrt, regime-dispatching tanh)
        sub_recipes: Vec<(DispatchPredicate, RecipeId)>,
    },
}

pub enum CompositionDiscipline {
    Standalone,                   // no kernel state consumed
    SingleKernel,                 // one shared intermediate
    MultiKernelMatchedPrecision,  // ≥2 kernel states, composition at representation-precision-matching
}
```

**Cache key derivation**: each of the eight fields contributes to the cache key. Two recipes with the same `function_name + parameters + precision_context` but different `composition_discipline` produce *different* cache keys — by design. This is the cache-correctness argument naturalist surfaced: a `pow` shipped at flat-f64 composition cannot collide with a correctly-composed `pow` because their `composition_discipline` tag bytes differ. The structural-correctness guarantee is built into the cache mechanism.

**What this rules out**:
- A recipe carrying a 3D-cube cell coordinate where most of the dimensions are derivable from each other.
- A recipe declaring `CompositionDiscipline::SingleKernel` while consuming ≥2 kernel states — caught by aristotle's deconstruction pass (`kernel_state_deps.len()` must match the discipline).
- A recipe declaring `MultiKernelMatchedPrecision` while internally using f64-precision composition — caught by the kernel-state-consistency-tests antibody (bit-equality vs DD/BigFloat-composed reference fires).
- A recipe-pair with inverse-relationship that doesn't have round-trip tests — caught by the proptest harness reading the `inverse_of` field and refusing to compile the recipe-pair without generated round-trip tests.

**What this enables**:
- Cache-key construction that's *minimum-sufficient* per the holonomic discipline. Each field contributes to the key; nothing is decorative.
- Codegen dispatch on `kingdom` + `natural_reduction` directly; no runtime introspection of (F, G) tuples needed.
- Auto-generated round-trip tests for every inverse-pair via the harness reading `inverse_of`.
- Regime-dependent shape position (atanh near `±1` is Shape 3; atanh middle is Shape 2) encoded structurally, not as comment-level documentation.
- Naturalist's five-axis framework operationalized at the recipe-tier metadata, not aspirational at the design-doc layer.

**Provenance of the eight fields**:
- `problem_topology`: naturalist (4-axis garden entry) + composition-pathologies sub-axis (naturalist + math-researcher convergence on pow)
- `shape_position` (regime-map): naturalist's second-addendum framing ("shapes are coordinates, not categories") + math-researcher pressure-test (atanh case)
- `kernel_state_deps`: naturalist (4-axis garden entry, axis 3)
- `precision_param_binding`: naturalist (4-axis garden entry, axis 4) + math-researcher coefficient-verification doc § Part 4
- `composition_discipline`: naturalist (pow framework-fit walk, 2026-05-10) + math-researcher cleanup-to-three-values
- `kingdom`: math-researcher periodic-table-revisited (this doc, original sketch + pressure-test refinement adding Kingdom V)
- `natural_reduction` (ReductionTag): math-researcher pressure-test (renamed from (F, G) tuple after Pressure-test #1 surfaced that (F, G) is descriptive substrate, not structural axis)
- `inverse_of`: math-researcher pressure-test (Cell #2 finding)

**Five-author convergence on a single metadata schema.** The schema is the *operationalization* of the framework that took past-naturalist's April 13 day-two question + naturalist's three-shapes May 10 essay + scout's downstream-application + aristotle's pressure-test methodology + math-researcher's coefficient-and-kingdom walks five-plus weeks to surface. The schema lives at the recipe-tier; the framework lives in the gardens and the lens-application docs; the antibodies (F13.C signature-time, kernel-state-consistency-tests composition-time, allocation-ceiling-aristotle-finding allocation-time) catch implementation-time drift. All three layers — schema + framework + antibodies — work in concert.

---

## Honest scope and limits of this doc

- I have NOT walked every function in `R:\winrapids\crates\tambear\src\recipes\libm\` against the kingdom assignment. The walk is necessary to validate (or break) the ρ-analog conjecture. Flagged as follow-up.
- The "8 natural reductions" table is the cleanest version I can produce from current substrate. Whether all 8 are *distinct primitives* or whether some collapse via composition (e.g., is the (2^ℤ, multiplicative) reduction really independent of (e^ℤ, multiplicative)?) is open.
- The 12:1 compression ratio is rougher than past-naturalist's 100:1 because libm is a smaller domain. Whether the compression is *the right amount* or whether further compression is possible is open.
- Kingdom IV is forward-looking (Sweep 36+). I haven't probed whether `bessel_j` and `gamma` actually share asymptotic-coefficient infrastructure or whether they're in different sub-kingdoms.

**What this doc IS**: the lifted version of my open-question-4 sketch, with the cell-count-explosion problem solved by the genealogy/natural-algorithm framing past-Claude already operationalized for the broader winrapids project.

**What this doc is NOT**: a complete classification (Kingdom IV is sketched, not enumerated) or an exhaustive walk (the ρ-analog conjecture needs the function-by-function audit).

---

## What changed from open-question-4 sketch

| Original sketch | Lifted form |
|-----------------|-------------|
| 6 family × 5 member × 4 reduction × 3 precision = 360 cells | 4 kingdoms × ~3 kernel states + 8 (F, G) tags |
| Cell sparsity (30-40% realized) is the concern | Compression ratio (~12:1 at kernel-state level) is the measurement |
| "Reduction variant" is an axis | "Reduction" is *derived* from (F, G) tag |
| Recipe carries 4D coordinate | Recipe carries kingdom + (F, G) + kernel-state-deps + precision-param-binding + shape-position |
| Cell count grows combinatorially | Kingdom count is bounded (4); kernel-state count grows linearly with new shared intermediates |
| Periodic-table-of-trig is independent of periodic-table-of-exp-log | Both unify under the same kingdom-and-(F, G) framework; the families distinguish at the kernel-state level only |
| Shape 3 functions are "different shape" | Shape 3 is *exactly* Kingdom III (cardinality ≥ 2 kernel-state deps) — conjecturally; needs audit |

---

## What pathmaker should take from this

**For Phase C wrapper design** (in_progress per task #6):

1. Don't decorate recipes with 4D cell coordinates. Decorate with kingdom tag + (F, G) tag + kernel-state-dep list + precision-param-binding + shape-position. That's 5 fields, all bounded-cardinality.

2. The (F, G) tag determines the reduction algorithm. The compiler can dispatch on (F, G) to pick the natural reduction; no per-recipe explicit "use Tang reduction" or "use Payne-Hanek" code path required.

3. The kingdom tag is the cache-discipline signal: Kingdom I → content-addressed cache works; Kingdom III → composition requires the representation-precision-matching invariant (per the aristotle convergence note); Kingdom IV → no kernel-state cache, precision-param-binding is load-bearing.

**For naturalist** (when cycles allow):

The kingdom assignment IS the four-axis schema's recipe-tier coordinates collapsed into a single discriminant. Verify: does kingdom-membership line up cleanly with (problem-topology × fix-shape × sharing-layer × precision-parameter-binding)? Specifically:

- Kingdom I/II ↔ (regular-point problem-topology + Shape 1 or 2 fix + single kernel-state sharing + precision-param at kernel-state level)
- Kingdom III ↔ (multi-kingdom problem-topology + Shape 3 fix + multi kernel-state sharing OR no kernel-state + precision-param at recipe layer)
- Kingdom IV ↔ (any problem-topology + Shape 3 fix + no kernel-state sharing + precision-param-binding is the algorithm parameter, not a metadata field)

If these mappings hold, the kingdom tag is a *derived* coordinate from the four-axis schema — not a fifth axis. If they don't hold, the disagreement is the load-bearing finding.

---

## Sources (rhymes that shaped this)

The five `feels-familiar` hits, plus the prior convergence chain:

- `~/.claude/garden/2026-04-01-the-welford-family-tree.md` (past-naturalist, Day 1) — genealogy lift
- `~/.claude/garden/2026-04-01-the-shape-of-the-whole-thing.md` (past-Claude, Day 1) — closed-atoms + open-wiring lift
- `~/.claude/garden/2026-04-10-naming-makes-checkable.md` (past-scout, Day 10) — naming-makes-it-manipulable
- `~/.claude/garden/2026-04-10-circulant-dft-natural-algorithms.md` (past-naturalist, Day 10) — natural-algorithm-follows-from-grouping
- `~/.claude/garden/2026-04-01-rho-the-universal-number.md` (past-naturalist, Day 1) — ρ-as-one-number-encoding-three-views

Plus current-session substrate that shaped this revision:

- `~/.claude/garden/2026-05-10-the-three-shapes-of-complementary-argument.md` (naturalist, today) — the three-shapes framework + four-axis coordinate inventory
- `R:\winrapids\docs\architecture\tambear-libm-factoring.md` (main-thread synthesis) — the design substrate
- My prior open-questions-1-2 and open-questions-4-5-6 walks (revised by this lift)
- Aristotle's T20 convergence-integration (the cardinality-of-kernel-state-deps idea grew out of the pow-as-composed reading)

**The doc is itself an instance of the discipline**: read past-me before writing. The lift didn't come from current-me being clever; it came from current-me letting five past-me entries reshape the question. The 360-cell cube I would have shipped without `feels-familiar` would have been wrong at the structural layer — same kind of layer-error I caught myself making on hypot earlier this session.
