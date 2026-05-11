# Phase 1-8 — `DimHint::Adaptive` + `UpTo` Extensions

**Sweep 8 / Task 8A (ongoing)** · Author: aristotle · Date: 2026-04-22

**Trigger:** Tekgy addendum — `DimHint` gains `Adaptive` (DEFAULT) and
`UpTo(N)` variants. Fuzzy size-boundary specialization via TAM-observed
data distribution, not hard bucketing.

```rust
enum DimHint {
    Static(usize),               // hard specialize at exactly n
    UpTo(usize),                 // specialized for n ≤ bound
    Bounded { multiple_of: N },  // SIMD-friendly
    SymbolicEqual(axis_ref),     // cross-size sharing
    Dynamic,                     // generic, all sizes
    Adaptive,                    // DEFAULT — TAM picks based on data
}
```

**Lens applied:** the using-annotation principle (see
`principle-decisions-as-using-annotations.md`) now rides on every
design decision. Every TAM choice inside `Adaptive` resolution must
surface as a typed `using()` annotation.

---

## Phase 1 — Assumption Autopsy

- **A1 — `Dynamic` is the right default.** REJECTED by Tekgy. The
  right default is `Adaptive` — the compiler observes data and picks.
  Dynamic becomes "I explicitly don't want specialization."
- **A2 — Size boundaries are user-specified.** REJECTED. TAM picks
  a boundary that covers observed data distribution.
- **A3 — Hard size buckets (small/medium/large, or 1024 vs 1025).**
  REJECTED. Cache-miss dominance from strict boundaries; fuzzy
  coverage wins. `UpTo(N)` serves a range, not a point.
- **A4 — Outlier sizes force fresh compile per outlier.** REJECTED.
  Outliers fall back to the generic `Dynamic` kernel as a one-time
  cost. If outliers accumulate past threshold, TAM re-specializes
  and promotes.
- **A5 — Adaptive resolution is once-per-pipeline.** REJECTED. Adaptive
  can RE-EVALUATE as data shifts across sessions. The annotation in
  the pipeline reflects the current resolution; a future run with
  different data may re-resolve and rewrite the annotation.
- **A6 — The adaptive decision has no user-visible form.** REJECTED
  per the using-annotation principle. `Adaptive` resolved to
  `UpTo(1024)` MUST surface as
  `using(dim_1 = adaptive[picked: upto(1024), covers: 94%])`.
- **A7 — `Adaptive` is a runtime concept.** Partly true, partly
  wrong. TAM's *resolution* is compile-time (with possible warm-up
  dispatch for observation). The *observation that informs the
  resolution* is runtime. So: `Adaptive` is a compile-time DECISION
  informed by runtime data, which is exactly the annotation-carries-
  rationale pattern.
- **A8 — All axes of a multi-dim Shape can be Adaptive.** Reasonable,
  but: if every axis is Adaptive, TAM must coordinate the
  resolution across axes (a square matrix with two Adaptive axes
  should end up with a coupled resolution via `symbolic_groups`).
  Validator rule: Adaptive + symbolic_groups must co-resolve.
- **A9 — Adaptive resolution is deterministic.** Partly — given the
  same observation history, TAM produces the same resolution. Across
  different data histories, the resolution can differ. The
  annotation preserves determinism *within a pipeline's annotated
  state*; dropping and re-resolving from fresh observation may
  produce different outcomes. This is the whole point: fuzzy, not
  fragile.
- **A10 — `UpTo(N)` is a distinct variant from `Bounded {
  multiple_of: N }`.** They serve different purposes. `UpTo(N)`:
  specialized for ANY size ≤ N (kernel handles variable-length
  loop up to N). `Bounded`: specialized for sizes that are
  multiples of N (SIMD-friendly chunking with no remainder loop).
  Not overlapping; both needed.
- **A11 — Cache-key handling for Adaptive is obvious.** NOT obvious.
  Adaptive is a compile-time decision; by the time a kernel compiles,
  Adaptive has already resolved to a concrete variant
  (Static/UpTo/Bounded/Dynamic). The cache key reflects the
  RESOLVED form, not `Adaptive` itself. Two pipelines authored with
  `Adaptive` that resolve to the same `UpTo(1024)` hit the same
  cache entry.
- **A12 — Outlier promotion (one-time Dynamic fallback) is
  transparent to the user.** REJECTED per the principle. The
  outlier dispatch should produce a summary annotation or log:
  "step 7 dispatch at size 1025 fell to Dynamic (outside current
  UpTo(1024) specialization; consider re-adaptive-resolve if this
  grows past 5% of dispatches)". Not per-dispatch; summarized.

---

## Phase 2 — Irreducible Truths

- **T1 — `Adaptive` is a DECISION-DEFERRAL.** It says "TAM will
  pick at compile time; pick gets recorded in the annotation." Not
  a runtime-varying kernel.
- **T2 — The resolved form is one of the concrete DimHint variants
  (Static/UpTo/Bounded/Dynamic).** Adaptive never survives into the
  cache key. `Adaptive` is an author-intent marker; the resolution
  is the compile-time state.
- **T3 — Resolution is driven by observation.** TAM may use:
  (a) warm-up dispatches on a sample of data, (b) prior runs'
  logged dispatch sizes, (c) user-provided dataset size
  distribution metadata, (d) heuristic defaults when nothing
  observed.
- **T4 — Every Adaptive resolution is TYPED, ANNOTATED, and
  ROUND-TRIPPABLE via `using()`.** Per principle.
- **T5 — The rationale clause carries the OBSERVATION EVIDENCE
  that informed the choice.** "covers: 94%" — coverage rate. "alt:
  upto(2048)" — what the runner-up was. The user can re-examine
  the choice.
- **T6 — Outlier handling is a compile-time INVARIANT of the
  resolution:** "This resolved form has a documented
  out-of-bound strategy — fall through to Dynamic, recompile after
  N outliers exceed threshold, re-resolve at next session start."
- **T7 — `UpTo(N)` kernels handle variable length up to N.** Kernel
  contains a loop with a runtime-bound iteration count, plus
  codegen optimizations (unroll outer loop up to a sub-bound, etc.)
  that exploit the N-ceiling.
- **T8 — `Bounded { multiple_of: N }` kernels handle ONLY sizes
  satisfying the mod constraint.** Sizes outside fall to Dynamic.
  The mod constraint is a CONTRACT the kernel asserts.
- **T9 — `Adaptive` axes that are in the same `symbolic_group` MUST
  resolve to compatible forms.** If axis 0 and axis 2 are in the
  same equivalence class AND both are Adaptive, TAM resolves them
  together (e.g., both `UpTo(1024)`). Resolving them
  differently would violate the symbolic equality.
- **T10 — Adaptive resolution is RE-RUNNABLE without destroying
  existing user overrides.** User may have already overridden
  axis 0 to `Static(512)`; TAM's Adaptive resolution for axis 2
  must still run (possibly resolving to `UpTo(1024)` for axis 2).
  Overrides win per-axis.

---

## Phase 3 — Reconstruction

### R1 — Literal Tekgy draft (add two variants)

```rust
enum DimHint {
    Static(usize),
    UpTo(usize),
    Bounded { multiple_of: usize },
    // SymbolicEqual LIFTED OUT per R5' in phase-1-8-multi-dim-shape.md
    Dynamic,
    Adaptive,
}
```

Problem: how does the cache key handle Adaptive? (T2 says it
never survives; but if we hash DimHint directly into
shape_fingerprint, the SOURCE `Adaptive` produces a different hash
than the RESOLVED `UpTo(1024)`.)

### R2 — Split author-intent from resolved-form

```rust
pub enum DimHintAuthored {
    Static(usize),
    UpTo(usize),
    Bounded { multiple_of: usize },
    Dynamic,
    Adaptive,  // marks "TAM resolves"
}

pub enum DimHintResolved {
    Static(usize),
    UpTo(usize),
    Bounded { multiple_of: usize },
    Dynamic,
    // NO Adaptive variant — can't exist after resolution
}
```

And Shape carries `DimHintResolved` (used for cache-key), while
pipeline source carries `DimHintAuthored` (what the user wrote).
TAM's compile pass consumes `DimHintAuthored`, emits
`DimHintResolved` + an annotation.

**This is clean but doubles the enum.** May be overkill. Evaluate
R3.

### R3 — Single enum + resolution state flag

```rust
pub enum DimHint {
    Static(usize),
    UpTo(usize),
    Bounded { multiple_of: usize },
    Dynamic,
    Adaptive,  // only valid pre-resolution
}

pub struct Shape {
    // ... existing ...
    pub axes: Vec<DimHint>,
    pub resolution_state: ResolutionState,
}

pub enum ResolutionState {
    /// Shape is pre-resolution — `Adaptive` axes haven't been
    /// picked yet. Cannot be passed to cache-key computation.
    Authored,
    /// Shape is post-resolution — no Adaptive axes remain. Safe
    /// for cache-key computation.
    Resolved,
}
```

Invariant: `Shape::fingerprint()` panics (or returns Err) if
`resolution_state == Authored`. Compile pipeline: user writes
`Authored` shape; TAM resolves to `Resolved` shape; kernel compile
consumes `Resolved`.

Problem: state flag is a type-level concept leaking into runtime
validation. More idiomatic: **phantom types** (but that's a bigger
refactor for dubious win).

### R4 — Runtime-validated single enum (keep it simple)

Same as R3 but WITHOUT the resolution_state field. Instead:
`Shape::fingerprint()` returns `Result<[u8; 32], ShapeError>` and
errors if any axis is `Adaptive`. Calling `fingerprint()` on an
authored shape is a runtime error caught during compile, not at
construction.

Simpler than R3; same safety properties; no PhantomData gymnastics.

### R5 — Separate resolution step as a first-class method

```rust
impl Shape {
    /// Resolve `Adaptive` axes by consulting the TAM observation
    /// bag. Returns a fully-resolved Shape plus the set of
    /// annotations that record TAM's picks.
    pub fn resolve_adaptive(
        &self,
        observations: &ObservationBag,
    ) -> Result<(Shape, Vec<DimHintAnnotation>), ShapeError>;

    /// Fingerprint for cache-key computation. Panics if any axis
    /// is `Adaptive` (author-intent hasn't been resolved).
    pub fn fingerprint(&self) -> [u8; 32];
}

pub struct DimHintAnnotation {
    pub axis: usize,
    pub resolved: DimHint,
    pub rationale: ResolutionRationale,
}

pub struct ResolutionRationale {
    pub coverage_fraction: f64,
    pub observation_count: u64,
    pub alternative: Option<DimHint>,
    pub outlier_fallback: Option<DimHint>,  // usually Dynamic
}
```

R5 is the structural winner. `resolve_adaptive()` is the
compile-time hook; `ObservationBag` is TAM's state; the
annotations feed into the pipeline's `using()` surface per the
principle.

**Winner: R5.** Matches the using-annotation principle by
construction — resolution PRODUCES the annotations.

---

## Phase 4 — Assumption → Truth Map

| Assumption | Replacing truth |
|---|---|
| A1 Dynamic is default | T1+T3 Adaptive is default; resolves to concrete |
| A2 boundaries user-specified | T3 TAM picks from observations |
| A3 hard buckets | T7 UpTo is range-serving |
| A4 outliers force recompile | T6 outliers fall to Dynamic; promote on threshold |
| A5 adaptive once-per-pipeline | T3 re-resolvable on data shift |
| A6 no user surface | T4 typed using() annotation |
| A7 adaptive is runtime | T1 compile-time decision-deferral |
| A8 all-adaptive is fine | T9 symbolic_groups co-resolve |
| A9 adaptive non-deterministic | T4 annotation locks determinism per pipeline state |
| A10 UpTo == Bounded | T7+T8 different contracts |
| A11 cache-key obvious | T2 cache key uses resolved form only |
| A12 outlier promotion opaque | T6 summarized to user |

---

## Phase 5 — The Move

**MOVE: ship R5 with the UsingAnnotation integration.**

Concrete deliverables for Sweep 8:

1. **`crates/tambear/src/jit/shape.rs`**:
   - Add `Adaptive` and `UpTo(usize)` to `DimHint`. (`Bounded` is
     already in R5' of the multi-dim Phase 1-8.)
   - `Shape::fingerprint() -> Result<[u8; 32], ShapeError>` errors
     on any Adaptive axis.
   - `Shape::resolve_adaptive(&self, observations: &ObservationBag)
     -> Result<(Shape, Vec<DimHintAnnotation>), ShapeError>` is the
     resolution hook.
   - `ObservationBag` is a new type (stub form for Sweep 8 —
     essentially `HashMap<CallSite, Vec<usize>>` of historical axis
     sizes; real implementation lands in 8.5 with the observation-
     logger).

2. **`crates/tambear/src/jit/using_annotation.rs`** (new):
   - `UsingAnnotation` trait with `display_as_using_annotation`,
     `parse_annotation`, `rationale` methods.
   - `DimHintAnnotation` struct with Display/FromStr matching the
     `adaptive[picked: upto(1024), covers: 94%]` syntax.
   - Implement for `DimHint` and `ExecutionStrategy`.

3. **Round-trip property tests**:
   - `parse(display(x)) == x` for every DimHint variant
   - `parse(display(x)) == x` for every ExecutionStrategy variant
   - `parse(display(annotation)) == annotation` including rationale

4. **Default DimHint constructor changes**:
   - `Shape::new(grouping, validity)` defaults to `axes:
     vec![DimHint::Adaptive]`. Old default was `Dynamic`; new
     default is `Adaptive` per Tekgy.
   - Explicit `Shape::new_dynamic()` for callers who want the
     old behavior (rare).

5. **Documentation**:
   - `trait-spec-locked.md` addendum: `DimHint::Adaptive` is the
     default; every compile-time TAM decision surfaces as a
     using() annotation (cite the principle file).
   - DO-NOT.md entry: "Never call `Shape::fingerprint()` on an
     authored shape; call `resolve_adaptive()` first."

---

## Phase 6/7 — Recursive challenge

- **Q-rec-1.** What happens when there's NO observation data?
  (Cold start.) TAM must still resolve Adaptive. Answer: fall back
  to a heuristic default — likely `UpTo(4096)` for most axes, or
  `Dynamic` for axes flagged "size unknown." Rationale:
  `adaptive[picked: upto(4096), covers: heuristic default, no
  observations yet]`. On next session with observations, re-resolve.

- **Q-rec-2.** What observation count is "enough" to pick? My
  proposal: `resolve_adaptive` takes a `min_observations: usize`
  parameter (default 100). Below that, heuristic default. Above,
  empirical distribution drives the pick.

- **Q-rec-3.** What coverage fraction is "good enough" to pick
  `UpTo(N)` vs falling back to `Dynamic`? If the 99th percentile is
  far from the 90th percentile (long-tail distribution), `UpTo` may
  not be worth it. Heuristic: pick `UpTo(N)` where
  `N = ceil(95th_percentile)` if the 99.9th percentile is within
  2×N; else pick `Dynamic`. Configurable via `using()`:
  `using(adaptive_upto_percentile = 0.95, adaptive_upto_outlier_ratio
  = 2.0)`.

- **Q-rec-4.** Does `Adaptive` belong on EVERY Shape axis by
  default, or only when the author writes `Adaptive`? The addendum
  says default, so: `Shape::new(grouping, validity)` constructs with
  `axes: vec![DimHint::Adaptive]`. Authors who want explicit
  Dynamic write it.

- **Q-rec-5.** Do we ever need to serialize `Adaptive` to disk
  (persisted pipeline that hasn't been resolved yet)? Yes — pipeline
  source carries authored intent, including `Adaptive`. Compile
  cache does NOT (only resolved). So TBS/pipeline serialization
  supports Adaptive; kernel cache serialization never sees it.

- **Q-rec-6.** Observation collection — is it a SWEEP 8 concern?
  My answer: NO, only the *interface* is. `ObservationBag` is a
  stub type with `record_dispatch(call_site, axis_sizes)` and
  `sizes_for(call_site) -> &[usize]` methods. The observation
  logger (wired into dispatch) lands in Sweep 8.5 or 8.H. Sweep 8
  ships the interface + an in-memory empty ObservationBag that
  causes `resolve_adaptive` to use heuristic defaults.

- **Q-rec-7.** Does `ResolutionRationale` need to carry MORE than
  coverage_fraction + alternative + fallback? Candidate additions:
  observation_time_range (when was the data collected),
  outlier_count (raw). Defer; add if needed during 8.5 observation
  wiring.

---

## Phase 8 — Forced Rejection

- **What if `Adaptive` were ITSELF a cache-keyable variant (no
  resolution step)?** Then every pipeline with an Adaptive axis
  produces a DIFFERENT cache key than the same pipeline with an
  explicit variant, even if the observations would have resolved to
  the explicit form. Cache fragmentation. Resolution-first is
  essential. Confirmed R5.

- **What if resolution were LAZY (on first dispatch, not at
  compile)?** Then the kernel cache would need to handle "shape is
  being resolved right now" — which pushes the decision into the
  hot path. Resolution must complete BEFORE cache-key computation.
  Confirmed.

- **What if `ObservationBag` were a WEAK INPUT (hints only, TAM
  may ignore)?** Then the annotation's rationale claim ("covers
  94%") has no referent — where did 94% come from? Observations
  must be the authoritative source of rationale. Confirmed.

- **What if `UpTo(N)` and `Bounded { multiple_of: N }` collapsed
  into one variant?** Different contracts: UpTo is range-ceiling;
  Bounded is mod-constraint. A kernel for `UpTo(1024)` can handle
  size 7; a kernel for `Bounded { multiple_of: 8 }` cannot.
  Separate contracts; separate variants. Confirmed.

- **What if outlier handling were USER-CONFIGURABLE rather than
  defaulted?** Some users may want strict (Error on outlier); some
  want tolerant (fall to Dynamic silently). Default is
  fall-to-Dynamic-with-summary; strict mode is `using(adaptive_
  outlier_policy = "error")`. Add to the using() surface in Sweep
  8.5; for Sweep 8, fall-to-Dynamic is hardcoded.

- **What if the parent Shape had a RESOLVED-ON-CONSTRUCTION
  invariant enforced via a type distinction (R2)?** PhantomData or
  two separate types. Cost: double the struct; every Shape-
  consuming function needs two versions or a trait. Win: no
  runtime error on missed resolution. Evaluated: not worth the
  refactor cost for Sweep 8. R4's runtime validation (error on
  fingerprint() with Adaptive present) is clear enough. **Flag
  for Sweep 9+ reconsideration** if the runtime error becomes a
  recurring footgun.

- **What MUST ALSO EXIST that I haven't named?** Ghost-hunting.
  Candidate: **re-resolution triggers**. Under what conditions does
  TAM decide the existing resolution is stale? (a) outlier rate
  exceeds threshold, (b) user explicitly requests re-resolve,
  (c) shape fingerprint drift beyond a similarity threshold. None
  of these are Sweep 8 scope — but noting that `ResolutionRationale`
  should carry a `resolved_at: Timestamp` field so staleness is
  inspectable. Add to Sweep 8 spec; populate in 8.5.

  Second candidate: **annotation stability under re-resolution**.
  If TAM re-resolves and picks the same value, does the annotation
  STRING change (different observation count)? If yes, the
  annotation's rationale clause adds noise to diffs. Mitigation:
  normalize rationale (round coverage to 0.1%, bucket observation
  count to 10×). Stable diffs. Add to UsingAnnotation trait
  design.

---

## Cache-bake audit on R5 extension (per Tekgy Note 2)

- `DimHint::Adaptive` — does NOT appear in cache key (resolution
  elides it; post-resolution cache key uses concrete variant)
- `DimHint::UpTo(N)` — BAKES (different N → different kernel
  specialized for that ceiling)
- `DimHint::Static(N)` — BAKES (unrolling)
- `DimHint::Bounded { multiple_of: N }` — BAKES (different N →
  different loop tiling)
- `DimHint::Dynamic` — BAKES the variant tag (one kernel for all
  Dynamic; separate from UpTo kernel)
- The resolution rationale (coverage fraction, observation count)
  does NOT bake (it's metadata on how TAM chose, not a property
  of the kernel)

Cache-bake principle honored.

---

## Conclusion

Ship R5. Adds `Adaptive` + `UpTo` per Tekgy's draft; surfaces
resolution via `Shape::resolve_adaptive()` + `DimHintAnnotation`;
wires into the using-annotation principle from day one. About 150
lines of new code + 15 new tests. Pathmaker can land in 2-3 hours
alongside the R5' multi-dim extension.

Sequence: R5' multi-dim Shape lands first (adds `axes: Vec<DimHint>`
and rename); then Adaptive/UpTo variants extend DimHint; then
UsingAnnotation trait + property tests.
