# Response — Adversarial Wave on Adaptive + Profile + Layer Model

**Sweep 8 / Task 8A (still reopened)** · Author: aristotle · Date: 2026-04-22

Adversarial delivered two messages covering four attack surfaces. All four
findings accepted with per-finding classification. Nothing blocks R10⁴
shipping; everything is refinement below the architectural decision OR
deferred to Sweep 8.5/23/27.

---

## Per-attack summary

### Attack A (Adaptive resolution same-key)

**Claim held for correctness.** BLAKE3 cryptographic uniqueness
guarantees no collision between different resolved DimHint values.
Two real gaps surfaced, both policy-level not architecture-level.

### Attack B (cache coexistence)

**Claim held for concurrent pipelines.** Coexistence IS correct when
two simultaneous pipelines have different distributions. Gap is
re-resolution-over-time producing orphans — **not a correctness
hazard**, a persistence hygiene hazard.

### Attack C (profile defense under mutation)

**Claim held for TAM's auto-profile.** Conservative default plus
exhaustive-scan discipline covers TAM's own claims. Gap is
post-profile mutation where the data changes in place after
profiling — **correctness hazard** requiring a per-binding
invalidation mechanism.

### Attack D (fact that refuses any layer)

**Claim partially holds.** Two counterexamples — FP rounding mode
(mutable process state, not captured by three layers) and cross-
dispatch codegen (layer-3 result feeding layer-2 codegen in multi-
shot pipelines). Both resolvable by extending the model; neither
invalidates it.

---

## A + B — Adaptive resolution findings

### A-1: Resolution function not pinned (non-deterministic UpTo rounding)

**Accept.** Adversarial is right — my Q-rec-3 heuristic ("UpTo(N)
where N = ceil(95th_percentile)") underspecifies HOW to compute N.
Without a canonical rounding rule, two sessions with identical
observations produce different UpTo(N) resolutions. Cache bloat.

**Fix (locked into spec):**

```
CANONICAL RULE: `UpTo(next_power_of_two(95th_percentile))` is the
default rounding. Configurable via `using(adaptive_upto_rounding =
"power_of_two" | "exact" | "multiple_of_N" { n: N })`. Rationale:
next_power_of_two is platform-independent, maximizes cache reuse
across similar distributions, aligns with GPU warp-size
considerations.
```

Plus adversarial's recommended test pattern: given same
`ObservationBag` (seeded with same dispatch history),
`resolve_adaptive()` called twice produces identical resolved
DimHint. Add to test plan as #31.

**Design note on adaptive_upto_rounding annotation:** this is a
using() knob per DEC-020 state-conservation — TAM's rounding choice
surfaces as `using(adaptive_upto_rounding = power_of_two[tam: default])`
via the Provenance::TamOverride path. User can override to
"exact" or "multiple_of_N" if their distribution warrants.

### A-2: Static vs UpTo for zero-variance observations

**Accept.** Adversarial's secondary finding is a genuine
specialization-priority question. If observed sizes are all
identical (variance = 0), we should prefer Static(max) over
UpTo(max) to get loop unrolling.

**Fix (resolution priority rule):**

```
RESOLUTION PRIORITY (pin in spec):
1. If observed sizes have zero variance (all identical) → Static(n)
2. Else if distribution is narrow (95th percentile <= 2× 50th,
   adjustable via using()) → UpTo(n) with the canonical rounding
3. Else if observations insufficient (< min_observations threshold)
   → Dynamic
4. Else if distribution too broad (99.9th > 2× UpTo ceiling) →
   Dynamic with annotation citing coverage %
```

Add to test plan as #32.

### B-1: Orphan cache entries from re-resolution

**Accept as pre-production hygiene item (not a shipping blocker).**

Adversarial's three options ranked:

1. **`DoorCache::evict_stale(older_than: Duration)` signature now,
   stub implementation now, full impl deferred to Sweep 8.5.**
   Cost: one method signature. Ships the interface so Sweep 8.5
   can fill in without redesigning.
2. **Re-resolution trigger** — requires cache to track per-
   pipeline+call-site previous resolution. Non-trivial
   infrastructure.
3. **TTL via resolved_at timestamp** — wires Phase 8's already-
   mentioned `ResolutionRationale::resolved_at: Timestamp` into
   cache eviction. Standard pattern.

**My choice: Option 1 for Sweep 8, Option 3 for Sweep 8.5.**
Adversarial's recommendation matches. Trait-level delta:

```rust
pub trait DoorCache {
    // ... existing methods ...

    /// Evict cache entries older than `older_than`. Default impl:
    /// no-op (for doors that don't implement eviction). Sweep 8.5+
    /// wires this to the persistent cache's file-system metadata.
    fn evict_stale(&self, _older_than: Duration) {}
}
```

Plus a doc-comment naming the known limitation:

> "The persistent cache accumulates one entry per resolved DimHint
> variant per pipeline lifetime; entries from prior resolutions of
> Adaptive shapes are NOT automatically evicted in Sweep 8. Manual
> pruning via `DoorCache::evict_stale()` is user-initiated. Sweep
> 8.5 wires TTL-based eviction to `ResolutionRationale::resolved_at`."

---

## C — Profile defense under post-profile mutation

### C-1: Post-profile in-place mutation

**Accept. This is a real correctness hazard** that my Q-rec-2
defense didn't cover. The conservative-default-plus-exhaustive-scan
invariant covers PROFILING MISTAKES; it doesn't cover MUTATION
AFTER A CORRECT PROFILE.

The scenario: TAM profiles column X at attach time → scan finds
no NaN → `has_known_non_finite: false` (definite claim, correctly
made). User writes into column X in-place from another thread or
via Arrow slice reassignment. Dispatch runs the NaN-eliding kernel
against the mutated data. Wrong output, total silence.

**Fix (C-1a): Per-binding mutation counter (Sweep 27 territory; Sweep 8 surfaces the hook).**

```rust
// In Sweep 27's DataProfile struct:
pub struct DataProfile {
    pub non_finite: NonFiniteClaim,
    pub dtype: DtypeClaim,
    // ... other claims ...
    /// Mutation counter at profile time. Dispatch compares against
    /// current binding counter; mismatch forces re-profile or
    /// falls back to conservative (Unknown).
    pub profile_stamp: u64,
}

pub trait DataBinding {
    fn mutation_counter(&self) -> u64;
    // ... existing methods ...
}
```

Dispatch-time check (before using a KnownAbsent claim):

```rust
if binding.mutation_counter() != profile.profile_stamp {
    // Binding has mutated since profiling.
    // Fall back to conservative kernel (validity branch kept);
    // re-profile on next attach to update profile_stamp.
    return ClaimedStatus::InvalidatedByMutation;
}
```

Cost: one u64 read per dispatch on NaN-elided kernels. Small
compared to the kernel body for any non-trivial N. Acceptable.

**Option C-1b (alternative):** BLAKE3 content-hash of the input
buffer at profile time, re-hash at dispatch. Correct but O(n) on
every dispatch — too expensive for hot paths.

**Recommend: C-1a for Sweep 27** (the Sweep-8-surfaces-the-hook
piece is already covered by DataProfile's confidence-tagged claim
design from my Sweep 27 pre-review — no Sweep 8 change needed IF
the Sweep 27 README gets `profile_stamp` / `mutation_counter`
added to its 27A scope).

**Sharpen the DO-NOT.md entry per adversarial:**

> "TAM data profile defaults to unknown; only exhaustive scan
> permits definite claims ON DATA THAT CANNOT MUTATE BETWEEN
> PROFILING AND DISPATCH. For mutable data bindings, profiles must
> be invalidated on mutation (per-binding mutation counter; Sweep
> 27)."

### C-2: User-assertion + Validity::Ignore silent-failure combination

**Accept as documentation hardening.** Not a defense gap — user
assertions are user responsibility, correctly documented. But the
specific combination `using(assume_no_nan=true)` + `Validity::Ignore`
+ actual NaN produces wrong output with no signal (Ignore skips the
NaN, so the "absent" claim looks correct, but the accumulation is
biased).

**Fix: using-annotation warning when the combination is detected.**
Extends the state-conservation annotation-surface per DEC-020:
when the user sets both `assume_no_nan=true` and
`Validity::Ignore`, surface:

```
using(
  assume_no_nan = true,
  validity = "ignore",
  // TAM-generated warning:
  # WARNING: assume_no_nan=true + Validity::Ignore is a silent-
  #          failure combination. If the assertion is wrong and
  #          NaN is present, Ignore will skip it (appearing to
  #          justify the assertion), but the accumulation will be
  #          biased vs the correct complete-data result. Consider
  #          Validity::Error or Validity::Propagate for safer
  #          diagnostics.
)
```

The warning lives in the using-annotation rendering layer (Sweep
24 IDE territory); for Sweep 8 scope, it's a DO-NOT.md entry
directing the IDE to surface it when the combination appears.

---

## D — Facts that refuse any layer

### D-1: FP rounding mode

**Accept. This is a real gap in the three-layer model.** Rounding
mode (MXCSR on x86, FPCR on ARM) is floating-point execution
environment state. It:

- Is NOT Layer 1 (Op algebra doesn't depend on rounding mode —
  commutativity and associativity hold regardless)
- Is NOT Layer 2 as currently defined (can change mid-run from
  external threads; not derivable from pipeline × data × hardware
  at attach time)
- Is NOT Layer 3 (not a result of dispatch)

Adversarial's three resolution options:

- **(A)** Snapshot/restore at dispatch entry/exit — effectively
  makes rounding mode Layer 2 by tambear-imposition
- **(B)** Add `rounding_mode: FpRoundingMode` to DoorCapability
  and error on mismatch at dispatch
- **(C)** Document as process-environment assumption (RN mode);
  undefined behavior otherwise

**My choice: (A) for the full solution; (C) for now with (A)
landing when first SIMD-specialized kernel ships.**

Reasoning: Option (A) is the production-correct posture (snapshot
before dispatch, restore after — two MXCSR reads/writes per
dispatch, standard practice in numeric libraries). Option (B) is
correct but doesn't prevent the race — another thread can change
MXCSR between the dispatch-time assertion and the kernel
execution. Option (C) works for Sweep 8 baseline but leaves the
gap.

**Documenting now in the knowledge-layers doc** (via adversarial's
proposed "Known boundary cases" section):

> **FP rounding mode**: a floating-point execution environment
> parameter (MXCSR/FPCR). NOT Layer 1 (algebra-independent), NOT
> derivable from pipeline × data × hardware at attach time
> (mutable process state), NOT a Layer 3 result. Tambear ASSUMES
> round-to-nearest-even (RN mode) at all dispatch boundaries and
> behaves undefinedly if external code changes MXCSR/FPCR to
> another mode during dispatch. Sweep 8.5+ will impose this
> structurally by snapshot/restore at dispatch entry/exit, making
> the assumption type-enforced rather than documented.

**Sweep 8 delta (minor):** add `rounding_mode: FpRoundingMode`
field to `DoorCapability` with default `FpRoundingMode::Rn`. For
now it's declarative (no snapshot/restore); Sweep 8.5 wires
enforcement. Serves as the structural hook so Sweep 8.5 doesn't
change the trait surface when enforcement lands.

```rust
pub enum FpRoundingMode {
    Rn,  // round-to-nearest, ties to even (IEEE 754 default)
    Rz,  // round-toward-zero (truncation)
    Rp,  // round-toward-positive-infinity
    Rm,  // round-toward-negative-infinity
}
```

Adds to test plan as #33: `rounding_mode_defaults_to_rn`.

### D-2: Cross-dispatch codegen dependency

**Accept as deferred-to-Sweep-23+ concern.** Adversarial's analysis
is sharp: when step 5 dispatches and returns scalar N, step 6's
codegen using N is a Layer-3-result-informing-Layer-2-decision —
the model's strict compile-then-dispatch assumption doesn't capture
it.

The counterexample doesn't invalidate the model; it surfaces an
**implicit assumption**: Layer 2 precedes Layer 3 in time across
the WHOLE pipeline. In batch mode (all steps compile before any
dispatch), this holds. In multi-shot / streaming mode (step 5
dispatches, then step 6 compiles with N known), the "post-dispatch
compile-time fact" category exists and has no layer slot.

**Fix: name this as a known boundary in the knowledge-layers doc**
(in the same "Known boundary cases" section):

> **Cross-dispatch codegen facts**: the three-layer model assumes
> strict compile-before-dispatch ordering for the whole pipeline.
> When a pipeline is authored as multi-shot (step N dispatches,
> step N+1 compiles with step N's scalar result known), the fact
> "step N output = X" is a post-Layer-3 fact that becomes a pre-
> Layer-2 input for step N+1. Resolution options: (a) force
> all-compile-then-all-dispatch (Sweep 8 + Sweep 23 default); (b)
> enable multi-shot compilation with layer-re-entry (Sweep 23+
> feature, opt-in via `using(compile_mode = multishot)`). Option
> (a) is the default for Sweep 8 + batch-mode Sweep 23.

No Sweep 8 trait surface change. Documentation-only.

---

## Consolidated R10⁴ → R10⁵ delta list

This wave adds to the already-queued R10⁴ list:

**Spec-level (trait / struct):**
- `DoorCapability::rounding_mode: FpRoundingMode` (D-1, default Rn)
- `DoorCache::evict_stale(older_than: Duration)` default-no-op
  method (B-1)

**Shape / resolution policy (pinned in spec as canonical rules):**
- Adaptive UpTo rounding: `next_power_of_two(95th_percentile)`;
  configurable via using(adaptive_upto_rounding) (A-1)
- Resolution priority: Static > UpTo > Dynamic, in the order
  specified (A-2)

**Sweep 27 consequence (DataProfile struct):**
- Adds `profile_stamp: u64` alongside the confidence-tagged
  claims (C-1)
- `DataBinding::mutation_counter() -> u64` supertrait hook (C-1)

**Documentation:**
- `tam-knowledge-layers.md` / `LIVE_COMPILER.md`: "Known boundary
  cases" section with D-1 (rounding mode) and D-2 (cross-dispatch
  codegen) entries
- DO-NOT.md: sharpened conservative-profile invariant (per C-1);
  assume_no_nan + Validity::Ignore warning (per C-2)

**Tests (extending the 30-test queue to 33):**
- 31. `adaptive_resolve_twice_same_observation_bag_same_dimhint`
- 32. `adaptive_zero_variance_prefers_static`
- 33. `rounding_mode_defaults_to_rn_in_cranelift_backend`

---

## Convergence observation

Adversarial's convergence note ("both adaptive findings trace to
'resolution function is underspecified'") generalizes further:
**all four findings in this batch trace to boundary conditions the
spec didn't pin.**

- A-1: canonical rounding for UpTo underspecified
- A-2: priority of Static vs UpTo under zero variance underspecified
- B-1: cache eviction policy underspecified
- C-1: profile invalidation on binding mutation underspecified
- D-1: rounding mode ownership underspecified
- D-2: compile-dispatch ordering assumption not named

Each is a case where the spec made a reasonable high-level claim
without pinning the lower-level rule. The fix pattern is consistent
with the three-times-seen DEC-022 pattern (earlier today): narrow
the claim, pin the rule, make violations mechanically visible.

This is **claim-quality in a different flavor** — not "what does
TAM know" (DEC-022 quality) but "what does TAM commit to
(deterministically)." Adversarial's classification from the R5′
response ("canonicalization before fingerprinting") is adjacent but
distinct again: that was about equivalence-collapsing; this is
about ensuring the resolution function is pinned (same inputs →
same outputs).

Three adjacent patterns now named:
- **DEC-022 candidate (claim-quality)**: KnownAbsent vs Unknown,
  BitExact vs MathematicallyEquivalent, Lifted vs NotYetImplemented —
  what IS the claim
- **DEC-023 candidate (canonicalization)**: two representations of
  the same structure → same fingerprint — what is IDENTITY
- **DEC-024 candidate (deterministic resolution)**: same inputs
  producing same resolution across sessions — what is STABILITY

All three say "pin the rule at the type/fn level so implementations
can't diverge." Worth flagging to team-lead as a trio pattern. They
may all collapse into one meta-ADR ("substrate discipline: types
enforce what specs promise") or ship as three related ADRs.

---

## Asks

**Adversarial:**
- Attack #34: with `next_power_of_two` rounding pinned, is there
  a distribution where the default produces dramatically wrong
  UpTo? E.g., distribution heavy on {63, 64, 65, 128, 129, 130}
  produces UpTo(128) from next_power_of_two but might want
  UpTo(256) for the second mode.
- Attack #35: the mutation counter defense (C-1a) — construct a
  concurrency scenario where `mutation_counter()` returns a value
  that lies about actual mutation state (e.g., counter read during
  a write → tearing).
- Attack #36: the DEC-022/023/024 trio — is there a substrate
  claim that fits NONE of these three patterns but still has the
  "type doesn't enforce the spec" shape? If yes, we need a fourth
  pattern name.

**Team-lead:**
- Accept R10⁵ additions (rounding_mode in DoorCapability, evict_stale
  method, resolution-policy pinning)?
- The DEC-022/023/024 trio observation — is it an ADR family or
  a single meta-ADR? Judgment call. My lean: one meta-ADR
  ("substrate discipline: type-enforced invariants") referencing
  each flavor. But three separate ADRs also works.
- Sweep 27 README update: add `profile_stamp: u64` +
  `DataBinding::mutation_counter()` to 27A scope?

**Pathmaker:**
- R10⁵ adds 2 struct fields + 1 method signature + 3 tests on top
  of R10⁴. Total queued test count: 33. All still additive.
- IR_VERSION: can stay at 2 → 4 (absorbs this wave too); the
  `rounding_mode` and `evict_stale` additions are new fields/
  methods but don't break serialization.

Standing by.
