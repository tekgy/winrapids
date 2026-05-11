# Principle — Compiler Decisions Surface as `using()` Annotations

**Sweep 8 / Task 8A** · Author: aristotle · Date: 2026-04-22

Tekgy: *"Every non-trivial decision TAM makes at compile time gets
written BACK INTO THE PIPELINE via `using()` annotations. The
annotation IS the UI for the compiler's reasoning. Pipeline source +
annotations is the single shared representation between human and
compiler. No translation layer."*

This is **the deepest principle yet stated for tambear's compile-time
behavior.** It reshapes how I deconstruct everything — the question
"can the user see and override this?" must accompany every design
decision.

Phase 1-8 on the principle itself.

---

## Phase 1 — Assumption Autopsy

What design assumptions does this principle reject?

- **A1 — Compiler decisions are private.** The conventional
  compiler is a black box: input → output, internals opaque. REJECTED.
  Every TAM decision is *visible* to the user.
- **A2 — Optimization is invisible by default, surfaced only on
  request.** REJECTED. The annotation is the *default* surface.
  Visibility is structural; opt-out (suppress annotations) is the
  exception.
- **A3 — User override is via flags or compiler directives that
  live OUTSIDE the program text.** REJECTED. Override lives IN the
  program text via typed `using()` values that *replace* the compiler's
  choice.
- **A4 — There are two distinct concepts: "user code" and "compiler
  metadata."** REJECTED. There is only "pipeline + annotations"; the
  annotations are co-equal authorial artifacts written by both human
  and compiler. The compiler's writes are no less authorial than the
  human's.
- **A5 — Decisions are binary (TAM picked X; user can override
  with Y).** REJECTED. Annotations carry RATIONALE
  (`adaptive[picked: upto(1024) covers 94%]`) so the user can
  evaluate the choice and consider counterfactuals.
- **A6 — Semantic-rewriting reuse (e.g. "step 35 reuses step 7's
  clustering even though step 35's default would differ") is silent
  optimization.** REJECTED. This is the most dangerous case — the
  USER would expect HDBSCAN, gets DBSCAN, doesn't know why. The
  annotation makes the substitution VISIBLE so the user retains
  agency.
- **A7 — Per-user-session decisions (kernel cache hits, adaptive
  boundary tuning) live in TAM's internal state, not in the
  pipeline.** REJECTED. The annotation is part of the pipeline. If
  TAM's choice changes session-to-session because observation data
  shifted, the annotation in the persisted pipeline records the
  *reason* so future-user can compare against current TAM behavior.
- **A8 — `using()` values are user-authored, period.** REJECTED.
  `using()` accepts both user-authored values AND compiler-authored
  annotations; they coexist in the same namespace, distinguished by
  marker syntax (e.g. typed value vs `adaptive[...]` notation).
- **A9 — Annotations are markdown commentary, not structured data.**
  REJECTED. Annotations are TYPED (each decision domain has a known
  shape: adaptive boundary, fusion link, lift strategy, semantic-share
  source, etc.) so tools can parse them, the compiler can re-read
  them on next run, the IDE can render them inline.
- **A10 — Round-tripping annotations is hard.** REJECTED. If the
  pipeline source + annotations is the only representation, round-
  trip is structural — it's the same syntactic surface for write
  and read.

Ten assumptions, all about hidden vs visible, opaque vs co-authored.

---

## Phase 2 — Irreducible Truths

- **T1 — The pipeline source is the SHARED REPRESENTATION between
  human and compiler.** Both write to it; both read from it; neither
  has a private channel. Co-native by construction.
- **T2 — Decisions are TYPED.** Each decision domain (boundary,
  strategy, fusion, share-source, etc.) has a stable schema in the
  `using()` namespace. Tools can introspect.
- **T3 — Annotations carry RATIONALE.** Not just "TAM picked X" —
  "TAM picked X because Y, where Y is the observed empirical fact."
  Rationale enables informed override.
- **T4 — User override TYPED-VALUE WINS over annotation.** When a
  `using()` key has a user-authored value AND TAM's annotation, the
  user value wins. The annotation becomes a comment on what TAM
  *would have* picked, kept for diff visibility.
- **T5 — Annotation persistence is part of the pipeline persistence
  contract.** Save pipeline → save annotations. Load pipeline →
  re-load annotations. Re-compile pipeline → either reuse annotations
  (cached schedule still valid) or regenerate (data shape changed
  enough to invalidate; new annotation reflects new choice).
- **T6 — Every TAM decision must be EXPRESSIBLE as a `using()`
  annotation.** A decision that has no `using()` form is a decision
  the user can't see or override — that's the failure mode this
  principle is built against.
- **T7 — `using()` is the SINGLE compile-time configuration channel.**
  No flags, no environment variables, no separate config files for
  things TAM decides. Everything that affects the kernel goes
  through `using()` either as user value or TAM annotation.
- **T8 — Annotations are CO-AUTHORED ARTIFACTS, not metadata.** The
  compiler writes back into the source-of-truth document. There is
  no second-class "metadata" tier that could be lost in a save/load
  round-trip.
- **T9 — The annotation surface must support marker syntax that
  distinguishes user-typed values from TAM-generated ones.**
  Otherwise users can't tell what they wrote vs what TAM filled in.
- **T10 — Annotations are LOCAL to their decision site.** Each atom
  call's TAM-chosen strategy is annotated AT THAT CALL — not in a
  global "scheduling notes" appendix. The decision lives where it
  was made.

---

## Phase 3 — Reconstruction (the principle's structural form)

What the system must look like to satisfy the principle:

```
pipeline_source.tbs:
    col(0).normalize.using(
        normalize_strategy = adaptive[picked: zscore covers 96%, alt: minmax],
        unroll_boundary    = adaptive[picked: upto(1024) covers 94%],
    ) | mean.using(
        strategy = lifted,           # user authored
        nan_skip = adaptive[picked: false (no NaN observed)],
    )
```

User reads, sees both their choices and TAM's. Edit any one to
override. Re-run; TAM re-evaluates the un-edited annotations against
fresh data.

The `adaptive[...]` syntax is the marker: bracketed `picked:` clause
identifies a TAM-authored choice. Plain typed values like
`strategy = lifted` are user-authored.

Three structural constraints on TAM:

1. **Every decision TAM makes must produce a typed annotation
   payload.** Strategy choice → `lifted`/`lifted_conjugated`/`sequential`.
   Boundary choice → `upto(N)`/`exact(N)`/`bounded(mod_N)`/`dynamic`.
   Fusion link → `fused_with(step_id)`. Share-source →
   `shared_from(step_id, was_default: alt_method)`.

2. **The annotation must include enough rationale that the user can
   judge the override cost.** "94% coverage" tells the user the
   outlier rate; "step_7 was DBSCAN; default here is HDBSCAN" tells
   the user the semantic substitution.

3. **The annotation persistence is a property of the pipeline file
   format.** Not a sidecar `.tam.cache.json`. If pipeline source
   files don't natively carry annotations, the principle is broken
   by design.

---

## Phase 4 — Map: assumption → truth

| Rejected assumption | Replacing truth |
|---|---|
| A1 compiler is opaque | T1 shared source-of-truth |
| A2 optimization invisible default | T6+T1 visible by default |
| A3 override outside program text | T4 override IN program text |
| A4 user vs metadata distinction | T8 co-authored artifacts |
| A5 decisions binary | T3 rationale enables informed override |
| A6 semantic substitution silent | T6 every decision expressible |
| A7 per-session state hidden | T5 annotation persists with pipeline |
| A8 using() user-only | T8 using() carries both |
| A9 annotations markdown | T2 annotations typed schemas |
| A10 round-trip hard | T1+T8 same surface read/write |

---

## Phase 5 — The principle's implications for Sweep 8 (and the design anchor)

Re-reading the design anchor (`design-anchor-pipeline-lift-substrate-
requirements.md`) through the principle:

- **S4 (`#[non_exhaustive]` on ExecutionStrategy):** still right,
  but now the new `Fused { with: AtomCallId }` variant must SURFACE
  as `using(strategy = fused_with(step_4))`. The variant exists in
  TAM's IR; the user-visible form is a using() annotation. Both must
  match.

- **NEW REQUIREMENT (S10):** every `ExecutionStrategy` variant must
  have a known display form that maps to a `using()` annotation
  syntax. Add `ExecutionStrategy::display_as_using_annotation(
  &self) -> String` (or trait impl) producing the round-trippable
  textual form. Reciprocally, parse: `parse_strategy_from_using(
  s: &str) -> Result<ExecutionStrategy, ...>`. Round-trip property
  test (parse(display(s)) == s).

- **NEW REQUIREMENT (S11):** every TAM decision domain enumerated
  as a typed `enum`. The list as of today:
  - `ExecutionStrategy` (atom-tier; landed)
  - `DimHint` (per-axis tier; landed; gains Adaptive variant per
    Tekgy addendum)
  - `BoundaryClass` (the resolved Adaptive boundary — `UpTo(N)`,
    `Static(N)`, `Bounded { multiple_of }`)
  - `ShareSource` (post-9; `Fresh | SharedFrom(step_id) |
    SharedFromWithRewrite(step_id, original_method)`)
  - `FusionLink` (post-9; `NotFused | FusedWith(step_id) |
    FusionRoot(absorbs: Vec<step_id>)`)

  Each must support the (display, parse) round-trip pair. The trait
  is something like:

  ```rust
  pub trait UsingAnnotation: Sized {
      fn write_annotation(&self, w: &mut dyn AnnotationWriter);
      fn parse_annotation(s: &str) -> Result<Self, AnnotationParseError>;
      fn rationale(&self) -> Option<String>;  // the [picked: ...] clause
  }
  ```

- **NEW REQUIREMENT (S12):** the persistent kernel cache (Sweep 8G)
  and the future schedule cache (post-9) cannot be the SOURCE OF
  TRUTH for TAM's decisions. The pipeline source + its annotations
  are the source of truth; the caches are content-addressed
  *materializations* that can be reconstructed from the annotated
  pipeline + observed data. If the cache is lost, the pipeline still
  reproduces the same kernel because the annotation is in the
  pipeline.

- **NEW REQUIREMENT (S13):** `IntermediateTag` consumers (per S9
  in the design anchor) — when TAM discovers a CSE opportunity, the
  `using(shared_from = step_N)` annotation surfaces the discovery
  so the user can opt out. Without surfacing, the share is silent
  optimization (rejects A6).

---

## Phase 6/7 — Recursive challenge

- **Q-rec-1.** What about decisions that are PURE OPTIMIZATION with
  no user-observable consequence? E.g., loop unroll factor for an
  arithmetic kernel. Should EVERY such decision surface? My answer:
  YES, because (a) it costs nothing to surface, (b) someday the user
  will care (debugging, performance investigation, reproducibility),
  (c) the principle is about consistency — once we make exceptions,
  the user can never trust that a hidden decision isn't elsewhere.
  Surface everything; let users filter their view.

- **Q-rec-2.** Doesn't this make pipeline files giant? Annotation
  bloat. My answer: annotations are concise; the rationale clause is
  a few words ("covers 94%"). And if the file gets large, the
  RENDERING (IDE / TBS UI) can fold annotations to a summary by
  default. The on-disk representation is verbose; the visible
  representation is configurable. This is a presentation concern,
  not a structural one.

- **Q-rec-3.** What if the user EDITS an annotation TAM wrote?
  Conceptually: that's just an override. The user-edited version
  takes precedence on next compile. TAM's `[picked: X]` notation
  becomes irrelevant; TAM still computes "what I would have picked"
  but defers to the user. Tracked as a *standing override*
  (annotation diff against TAM's would-pick).

- **Q-rec-4.** What if TAM's would-pick diverges from the user's
  override later (data shifted)? TAM CAN annotate *next to* the
  user override: `using(strategy = lifted, # TAM_would_now_pick:
  lifted_conjugated [reason: shape changed to ByKey])`. The user
  reads, decides whether to update. Co-design continues across
  sessions.

- **Q-rec-5.** Does this principle apply to RUNTIME decisions too
  (per-dispatch entry-point selection)? My answer: NO — runtime
  decisions are too fast/frequent to annotate per-dispatch. They
  could be SUMMARIZED (e.g., "98% of dispatches used reduce_warp
  entry; 2% fell to reduce_block") but the per-dispatch decision
  isn't a using() annotation. The principle is about COMPILE-TIME
  decisions; dispatch-time observations are summary statistics.

- **Q-rec-6.** What about decisions made by the proof engine (e.g.,
  "this Op is associative, so this scan can lift")? My answer:
  proof-engine outputs are FACTS, not DECISIONS. Facts feed
  decisions. Facts don't need user-override surface (overriding a
  fact would mean lying to the compiler, which can produce wrong
  results). Decisions DO need override surface. Distinction
  matters — surface decisions; cite facts.

- **Q-rec-7.** Annotations as CO-AUTHORED ARTIFACTS — does this
  mean version control sees TAM's writes as commits? My answer:
  YES at the conceptual level, NO at the literal git-commit level.
  Annotations live in pipeline files; pipeline files are version-
  controlled; TAM's writes appear as diffs. The user reviews TAM's
  changes the way they'd review a coworker's PR. *That's* what
  co-design with a compiler looks like. (Massive implication for
  how diffs are visualized in the IDE — defer to playground/IDE
  sweep.)

---

## Phase 8 — Forced Rejection

- **What if there were NO `using()` annotations at all — TAM kept
  state internal?** Then every TAM decision is invisible; user
  cannot debug; user cannot reproduce; user cannot share a
  pipeline+state with a colleague who doesn't have TAM's local
  cache. The principle's necessity is structural. Confirmed.

- **What if annotations were OPTIONAL (TAM writes them only on
  request)?** Then "request" becomes a compiler flag, which IS the
  out-of-program-text override path the principle rejects. Some
  decisions surface; others don't; user can't tell what's hidden.
  Annotations MUST be the default. Confirmed.

- **What if annotations were UNTYPED (free-form text)?** Then the
  compiler can't re-read its own writes; user can't tooling-edit
  them; round-trip is broken. Typed schemas are essential.
  Confirmed.

- **What if rationale were OPTIONAL (the `[picked: ...]` clause)?**
  Then the user knows TAM picked X but not why; informed override
  is impossible. Always include rationale. Confirmed.

- **What if user override DIDN'T win?** Then the principle is
  vacuous — user can edit but compiler ignores. Override-wins is
  load-bearing. Confirmed.

- **What if there were a MORE FUNDAMENTAL principle this is a
  consequence of?** Forcing myself to find the parent. Candidate:
  **"Every system state that affects observable behavior MUST be
  representable in the system's authoritative source-of-truth
  document."** Tambear's pipeline files are the authoritative
  source-of-truth; therefore every compiler decision lives there.
  This is a CONSERVATION LAW for state. **Yes, this is the deeper
  principle.** Reframe: tambear obeys *state conservation in the
  source-of-truth*. The using-annotation pattern is a consequence
  of state conservation applied to compile-time decisions. Filed
  for the post-9 sweep design.

- **What if the IDE rendering didn't make this apparent?** The
  principle would still hold structurally but be invisible in
  practice. The IDE/playground-sweep MUST render annotations
  prominently — folding for compactness, expansion for
  inspection, edit-in-place for override. UX implementation
  detail; out of Sweep 8 scope; in scope for the IDE/playground
  sweep.

---

## Action items

**For Sweep 8 (immediate):**

1. Add `Adaptive` to DimHint with rationale-carrying form (see
   companion file `phase-1-8-adaptive-dimhint.md`).
2. Add `UsingAnnotation` trait (or equivalent) so each decision-
   domain enum (`ExecutionStrategy`, `DimHint`, future) implements
   `display_as_using_annotation`/`parse_annotation`/`rationale`
   PLUS provenance accessor — see extension below.
3. Property tests: parse(display(x)) == x for every implementing
   enum (lossless round-trip). 5-10 tests. Plus provenance round-
   trip: parse(display(value, Default)) → (value, Default);
   parse(display(value, TamOverride{"ev"})) → (value, TamOverride{"ev"});
   parse(display(value, UserOverride)) → (value, UserOverride).
4. `ExecutionStrategy` display forms per provenance:
   - (Lifted, Default) → invisible (no `using()` emitted)
   - (Lifted, UserOverride) → `using(strategy = lifted)`
   - (Lifted, TamOverride{"ev"}) → `using(strategy = lifted[tam: ev])`
   - (LiftedConjugated{perm}, P) → `using(strategy = lifted_conjugated(perm)<P>)`
   - (Sequential{reason}, P) → `using(strategy = sequential[reason: r]<P>)`
5. Document the principle in trait-spec-locked.md as a top-level
   constraint on every type that participates in TAM decisions,
   cross-linking to `docs/LIVE_COMPILER.md` (§ State conservation +
   provenance per binding) as the canonical source.

### Provenance extension (team-lead ratification, 2026-04-22)

Per DEC-020's sub-clause (`docs/LIVE_COMPILER.md` lines 178-192
capture the canonical form): every binding carries (value,
provenance). Provenance is one of:

```rust
pub enum Provenance {
    /// Recipe-defined default. Absence of a user `using(key=...)`
    /// auto-applies this. Renders INVISIBLY in TBS source — no
    /// `using()` printed for Default bindings.
    Default,
    /// TAM (compile-time analysis) changed this from default.
    /// `evidence` is short human-readable reason:
    /// `"kurtosis 7.4 rejects normality"`, `"upto(1024) covers 94%"`,
    /// `"fused with step_4"`.
    TamOverride { evidence: String },
    /// User typed this explicitly in TBS or set via IDE. Renders
    /// as plain `using(key=value)` — no `[tam: ...]` marker.
    /// TAM does NOT override UserOverride values; respects them
    /// and may warn (as a separate annotation) if questionable.
    UserOverride,
}
```

The `UsingAnnotation` trait extends to:

```rust
pub trait UsingAnnotation: Sized {
    fn write_annotation(&self, provenance: &Provenance,
                        w: &mut dyn AnnotationWriter);
    fn parse_annotation(s: &str)
        -> Result<(Self, Provenance), AnnotationParseError>;
    fn rationale(&self) -> Option<String>;
}
```

Provenance flows on write, returns on parse. Round-trip property:
`parse(write(value, prov)) == (value, prov)` for every provenance
variant.

**Why provenance is observable-behavior-affecting (and therefore
conserved per DEC-020):** the kernel produces the same compute
regardless of who set a value. But:

- **IDE warnings**: "you've overridden 12 defaults; 7 are TAM-
  evidence-based, 5 are your explicit choices." Requires
  provenance to partition.
- **Debug messages**: "kernel compiled with method=spearman
  chosen because kurtosis 7.4." Requires TamOverride carrying
  evidence.
- **Writeup prose** (Sweep 25): methodology cites TAM's evidence
  for overrides; leaves defaults uncited (they are the recipe's
  recommended usage). Requires partition.
- **Pipeline diff review**: user inspects what changed between
  commits; "TAM picked X because Y" reads differently from "user
  changed X to Y" at review time. Requires provenance.

All four are observable behaviors of tambear-as-experienced-by-
the-user (not behaviors of the kernel). Per DEC-020, provenance is
state and must live in the source-of-truth. The `using()`
annotation is the source; provenance rides on every binding.

**CONSEQUENCE for CacheKey / kernel-compute equivalence:**
bindings with the same VALUE but different PROVENANCE hit the same
kernel cache entry — provenance does NOT affect codegen. Only the
value matters to the kernel; provenance matters to tambear-
surrounding-the-kernel. This is consistent with cache-bake
principle: provenance is not in the instruction stream.

**For pipeline-lift sweep (post-9, deferred):**

6. New decision domains (`ShareSource`, `FusionLink`,
   `BoundaryClass`) all implement `UsingAnnotation`.
7. Annotation persistence integrates with pipeline file format
   (TBS-IDE sweep territory).
8. State-conservation principle (the parent) gets its own
   architectural-decision record.

**For aristotle (ongoing):**

9. Add to deconstruction-pass checklist: *"For every design
   decision, can the user see and override it via a `using()`
   annotation?"* If no, surface why.

---

## What this changes about my deconstructor lens

This principle is now part of my Phase 1 assumption autopsy
checklist:

> **Has this design hidden a TAM decision from the user?**
> If yes, surface it as a using() annotation. If the
> annotation has no obvious form, the design is wrong.

Every future deconstruction pass will run this check. Filed in my
working notes; carrying forward.
