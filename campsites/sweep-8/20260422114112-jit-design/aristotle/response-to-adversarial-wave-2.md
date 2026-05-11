# Response to Adversarial Wave-2 — 5 Breaks + 1 Partial

**Sweep 8 / Task 8A REOPENED** · Author: aristotle · Date: 2026-04-22

Adversarial produced six concrete attacks on R10′. Per my own commitment in
the convergence-check verdict, any attack revealing a Phase-9 ghost reopens
8A. **Five attacks reveal genuine Phase-9 ghosts. One partial.** I accept
all five; I'm closing 8A with the corresponding trait deltas; B5 routes to
a Shape-level fix that doesn't touch the trait.

This file is the per-attack response, then the consolidated trait-spec
delta, then the new convergence-check criteria for pathmaker.

---

## A1 — NPU per-atom `lower()` granularity → ACCEPT

**Adversarial is right; my draft was wrong.** The per-atom `lower()` contract
genuinely doesn't fit graph-compiler doors (Inferentia, TPU/XLA, Cerebras).
The "buffer atoms internally then compile on a signal" workaround I implied
fails because R10′ has no pipeline-boundary signal — the door has no way to
know when it has the last atom.

**The fix is structurally clean:** add `lower_pipeline()` with default-impl-
calls-`lower()`-per-atom. CPU/GPU doors keep working with no change. NPU
doors override `lower_pipeline()` and `lower()` returns `Err(NotSupported)`.

This is exactly the pattern adversarial proposed. Accept verbatim:

```rust
pub trait DoorCodegen {
    fn lower(&self, op: &JitOp, shape: &Shape, strategy: ExecutionStrategy,
             params: &[u8], cap: &DoorCapability)
        -> Result<CompiledArtifact, CompileError>;

    /// Whole-pipeline compile for graph-compiler doors (NPU). Per-atom
    /// doors (CPU/GPU) keep the default — calls `lower()` per atom.
    fn lower_pipeline(&self,
                      atoms: &[(JitOp, Shape, ExecutionStrategy, Vec<u8>)],
                      cap: &DoorCapability)
        -> Result<Vec<CompiledArtifact>, CompileError>
    {
        atoms.iter().map(|(op, sh, st, p)|
            self.lower(op, sh, *st, p, cap)
        ).collect()
    }

    fn supports(&self, op: &JitOp, shape: &Shape,
                strategy: ExecutionStrategy) -> bool;
}
```

**Implication that doesn't fit a single-method fix:** the pipeline compiler
(Sweep 23) must KNOW which doors prefer per-atom vs per-pipeline. Otherwise
it'll feed atoms one at a time to a TPU door and get NotSupported errors
without falling back. Add capability bit:

```rust
pub struct DoorCapability {
    // ... existing fields ...
    /// True if the door requires whole-pipeline compilation (graph compilers
    /// like XLA / neuron-cc / Cerebras CS-2). False for per-atom doors
    /// (cranelift, PTX, SPIR-V, AIR, DXIL). The pipeline compiler routes
    /// to `lower_pipeline()` when true.
    pub requires_whole_pipeline: bool,
}
```

CPU/GPU set false; NPU sets true. Sweep 23 reads it.

**This is the substrate's correct design** — and Phase 8 forced rejection
should have surfaced it on the original deconstruction. I missed it because
I treated NPU as a "future door" abstract case rather than running the
specific compile-API contract through the trait. **Genuine Phase-9 ghost.**

---

## A2 — `SeededDeterministic` 4th class → ACCEPT

**Adversarial is right.** `Deterministic` (unconditionally same output for
same input) and `NonDeterministic` (output may differ across runs) leave a
gap for seeded reproducibility. The seed bakes into the param hash so the
cache key is correct, but `DeterminismClass` carries a separate semantic
claim that downstream consumers (proof engine, writeup generator,
debugging tools) need.

Two cases adversarial named:
- **`Probabilistic + using(seed=N)`**: SeededDeterministic — reproducible
  given seed, varies with seed.
- **EM/MCMC with seeded convergence**: conservative `NonDeterministic`,
  upgradeable to SeededDeterministic via `using(seed=N, deterministic=true)`
  at the cost of serial convergence. Consumer-side opt-in.

The variant carries `seed_hash: u64` so two seeded kernels with different
seeds carry different DeterminismClass values — but the kernel binary is
the same (seed bakes into params). Provenance-shaped: same kernel, different
observable-behavior-affecting metadata.

```rust
pub enum DeterminismClass {
    Deterministic,
    OrderDependent,
    NonDeterministic,
    /// Identical output given the same RNG seed; different seeds produce
    /// different outputs. Seed is baked at compile time via
    /// `using(seed=N)`. The seed_hash is the BLAKE3 of the seed bytes,
    /// so two SeededDeterministic instances can be compared without
    /// revealing the seed itself in debug output.
    SeededDeterministic { seed_hash: [u8; 32] },
}
```

Note: I changed `seed_hash: u64` → `[u8; 32]` for cryptographic strength
+ uniformity with the rest of the cache-key system. BLAKE3 is the
canonical hash everywhere else; using a u64 here would create a smaller
collision space than the rest. Adversarial's u64 was illustrative; locking
[u8; 32] for consistency.

**Cache-key invariant preserved:** seeds bake via params into the cache
key (already correct in pathmaker's fingerprint.rs). The DeterminismClass
field on CompiledArtifact reports the determinism guarantee structurally.
Two consumers asking for the same kernel with the same seed get the same
SeededDeterministic value; different seeds produce different values that
still hash distinctly.

**Genuine Phase-9 ghost.** I had asked adversarial to look for a 4th class;
adversarial found one cleaner than my Probabilistic-with-learned-seed
suspicion. Accept.

---

## A3 — Stream-poison semantics → ACCEPT (with one refinement)

**Adversarial is right** about the in-flight error masking. My single-
dispatch claim ("`wait()` returns `Err(LaunchError::Validity)`") is correct
in isolation but doesn't compose under pipelined dispatch. Kernel B runs
on NaN-poisoned input from kernel A; `wait(e2)` returns Ok; user has no
signal that step 36's "successful" output is garbage.

This is a structural gap in the dispatcher trait that genuinely requires
trait-level fix. Stream-poison is exactly the right mechanism.

**Accept with refinement:** add poison check to `dispatch()` AND
`stream_wait_event()`. Adversarial's draft only blocks new dispatches;
cross-stream events on a poisoned stream should also fail.

```rust
pub trait DoorDispatcher {
    // ... existing methods ...

    /// Reset a poisoned stream. Drains queued work, returns the stream to
    /// a clean state. Must be called after handling a Validity error
    /// before the stream can accept new dispatches.
    fn stream_reset(&self, stream: &Self::Stream) -> Result<(), LaunchError>;

    /// Check whether a stream is poisoned without trying to use it.
    /// Useful for TAM scheduler to decide whether to call stream_reset
    /// proactively.
    fn is_stream_poisoned(&self, stream: &Self::Stream) -> bool;
}

pub enum LaunchError {
    OutOfMemory { bytes_requested: u64 },
    Timeout,
    DeviceLost,
    Validity { details: String },
    BufferContract { detail: String },
    Driver { code: i64, detail: String },

    /// New: stream is poisoned by a prior Validity error. Caller must
    /// `stream_reset()` (or pick a different stream) before retrying.
    StreamPoisoned { details: String },
}
```

**Contract for backends to honor:**
- After `wait(e)` returns `Err(Validity { ... })`, the stream that produced
  `e` is poisoned.
- Subsequent `dispatch(stream, ...)` returns `Err(StreamPoisoned)` without
  issuing the dispatch.
- `stream_wait_event(other_stream, e_from_poisoned)` returns
  `Err(StreamPoisoned)` — cross-stream waits on poisoned events propagate
  the poison.
- `stream_reset(stream)` returns the stream to clean state, draining any
  queued work.
- `is_stream_poisoned(stream)` returns true between the validity error
  and `stream_reset`.

**CPU collapse:** `HostStream` is ZST and `HostEvent` is ZST. CPU dispatch
runs synchronously, so a Validity error returns from `dispatch()` directly
(not via `wait()`). The stream-poison concept is no-op on CPU because
there's no in-flight queue. `is_stream_poisoned` returns false; `stream_reset`
returns Ok(()). Zero-cost on CPU.

**Adversarial's "weaker alternative"** (push to TAM scheduler) is rejected:
that's the same failure mode as not adding the contract — relies on TAM
discipline rather than enforcing it structurally. Per state conservation,
the contract must be in the trait (the source-of-truth for backend
behavior), not in TAM's documentation.

**Genuine Phase-9 ghost.** Accept with refinement.

---

## B4 — DenormalMode on CompiledArtifact → ACCEPT

**Adversarial is right.** `DenormalMode` lives in `DoorCapability` and the
fingerprint includes it correctly for new compiles. The gap is that a cached
artifact has no record of the mode it was compiled under — if the live
capability's mode shifts (MXCSR write on CPU, driver update on GPU), the
artifact dispatches under a different mode than it was compiled for.

The Welford+denormals example is concrete:
- Compile-time DenormalMode = Ieee → kernel compiled with denormal-aware
  arithmetic
- Some other thread writes MXCSR.FTZ at runtime
- Cached artifact dispatches; floating-point hardware is now FTZ; denormals
  zeroed silently
- Welford variance is wrong; no error

Fix is a single field + a single check:

```rust
pub struct CompiledArtifact {
    pub door: DoorId,
    pub binary: ArtifactBinary,
    pub entry_points: Vec<EntryPoint>,
    pub pipeline_metadata: PipelineMetadata,
    pub scratch_required: ScratchSpec,
    pub determinism: DeterminismClass,
    /// New: the DenormalMode this artifact was compiled under. dispatch()
    /// must verify the current capability's DenormalMode matches; if not,
    /// returns Err(LaunchError::CapabilityMismatch).
    pub denormal_mode: DenormalMode,
}

pub enum LaunchError {
    // ... existing variants ...
    /// New: the dispatch-time capability differs from the compile-time
    /// capability the artifact was built under. The cache key ALREADY
    /// guards new compiles against this; the run-time check guards
    /// cached artifacts against mid-process capability shifts.
    CapabilityMismatch { compiled_under: String, current: String },
}
```

**Why this also closes a quieter problem:** in principle ALL of
`DoorCapability` could shift mid-process. DenormalMode is the most likely
to change silently on CPU (MXCSR is a thread-local register; another thread
or library can flip it). Driver_version is hard to flip mid-process on most
platforms. ISA_version is hard. Subgroup_sizes don't change. So the realistic
silent-shift surface is ~just~ DenormalMode.

If we wanted to be paranoid we could verify the full capability fingerprint
on every dispatch. That's more conservative than necessary and adds latency
to every call. **DenormalMode-only verification covers the realistic threat
model.** If a future capability axis becomes silently shiftable, we can add
verification per-axis. (For Sweep 23+ when the data-quality analyzer wires
up, the capability fingerprint is also captured at pipeline-compile time
per the B6 partial-survival recommendation below — that's the long-term
robust fix.)

**Genuine Phase-9 ghost.** Accept.

---

## B5 — Shape metric-identity collision → ACCEPT (Shape, not trait)

**Adversarial is right** that two distinct kernels (Euclidean vs Mahalanobis
distance matrix) currently produce the same Shape and therefore the same
cache key, leading to silent wrong-kernel sharing.

**This is a Shape-level bug, not a DoorBackend trait bug.** The fix is
adding `AssumptionTag::Metric(MetricKind)` as a first-class variant. Per
adversarial's own classification, this is "tracked separately" from the
trait surface.

```rust
pub enum AssumptionTag {
    NoNonFinite,
    SortedAscending,
    Centered,
    UnitNorm,
    Metric(MetricKind),  // new — first-class
    Custom(String),
}

pub enum MetricKind {
    Euclidean,
    Mahalanobis,
    Manhattan,
    Cosine,
    Minkowski { p: f64 },        // generalized
    Hamming,
    Jaccard,
    KullbackLeibler,
    JensenShannon,
    Wasserstein { p: f64 },      // ground metric not specified — recipe's responsibility
    Custom(String),
}
```

**Recipe discipline (encode in DO-NOT.md):** distance-producing recipes
MUST emit `AssumptionTag::Metric(MetricKind::*)` on their output Shape.
Distance-consuming recipes MUST check the metric matches expectation
(`is_share_compatible_with` already enforces tag-bag compatibility).

**Why first-class beats Custom(String):**
- Compile-time exhaustive matching (`match metric { Euclidean => ..., ... }`)
  catches missing variants.
- Cross-recipe consistency: every Euclidean-producing recipe uses the same
  variant; no string-typo divergence.
- Tooling-discoverable: IDE / writeup / debug can enumerate known metrics.
- Custom(String) escape hatch remains for genuinely-novel metrics (research
  cases).

**Genuine Phase-9 ghost in Shape.** Accept; route to pathmaker as a
Shape-side delta. Does not require trait change.

---

## B6 — Capability hot-swap → DOCUMENT, don't redesign

**Adversarial is right** that R10′ doesn't structurally prevent in-flight
dispatches from running under a stale capability after WDDM hot-swap.
**Adversarial is also right** that the fix is TAM-level (snapshot capability
at pipeline-compile, hold for pipeline duration), not trait-level.

**My response is to accept the partial survival as adversarial classifies
it:** document the gap explicitly in the trait spec and route the structural
fix to Sweep 23 (pipeline compiler) where capability-snapshot lifetime is
naturally owned.

Trait spec gets a doc-comment on `DoorBackend::capability()`:

```rust
pub trait DoorBackend: DoorCodegen + DoorCache + DoorDispatcher {
    /// Returns the door's CURRENT capability. May change between calls
    /// (driver update, WDDM hot-swap, MXCSR write, etc.) so callers MUST
    /// snapshot at pipeline-compile time and hold the snapshot for the
    /// duration of one pipeline execution. The pipeline compiler (Sweep 23)
    /// owns this snapshot; the dispatcher consults the snapshot, not
    /// `capability()` directly, on each dispatch.
    ///
    /// Per-dispatch `LaunchError::CapabilityMismatch` (introduced for B4)
    /// catches the case where a cached artifact's compile-time capability
    /// differs from the current one. That's the per-artifact check;
    /// pipeline-snapshot is the per-pipeline discipline.
    fn capability(&self) -> &DoorCapability;
}
```

**No code change for Sweep 8.** The doc-comment is the contract; Sweep 23
implements the snapshot. The `LaunchError::CapabilityMismatch` from B4 is
the dispatch-time safety net.

---

## Consolidated trait deltas — R10′ → R10″

Five accepted breaks produce the following changes to the locked spec:

### `DoorCodegen` — add `lower_pipeline` (A1)

```rust
pub trait DoorCodegen {
    fn lower(&self, op, shape, strategy, params, cap)
        -> Result<CompiledArtifact, CompileError>;

    fn lower_pipeline(&self, atoms: &[(JitOp, Shape, ExecutionStrategy, Vec<u8>)],
                      cap: &DoorCapability)
        -> Result<Vec<CompiledArtifact>, CompileError> {
        // Default: per-atom delegation. NPU doors override.
        atoms.iter()
            .map(|(op, sh, st, p)| self.lower(op, sh, *st, p, cap))
            .collect()
    }

    fn supports(&self, op, shape, strategy) -> bool;
}
```

### `DoorCapability` — add `requires_whole_pipeline` (A1)

```rust
pub struct DoorCapability {
    // ... existing fields ...
    pub requires_whole_pipeline: bool,  // NPU: true; CPU/GPU: false
}
```

### `DeterminismClass` — add `SeededDeterministic` (A2)

```rust
pub enum DeterminismClass {
    Deterministic,
    OrderDependent,
    NonDeterministic,
    SeededDeterministic { seed_hash: [u8; 32] },  // new
}
```

### `DoorDispatcher` — stream-poison contract (A3)

```rust
pub trait DoorDispatcher {
    // ... existing methods ...

    fn stream_reset(&self, stream: &Self::Stream) -> Result<(), LaunchError>;
    fn is_stream_poisoned(&self, stream: &Self::Stream) -> bool;
}

pub enum LaunchError {
    // ... existing variants ...
    StreamPoisoned { details: String },          // new
    CapabilityMismatch { compiled_under: String, current: String },  // new (B4)
}
```

### `CompiledArtifact` — add `denormal_mode` (B4)

```rust
pub struct CompiledArtifact {
    // ... existing fields ...
    pub denormal_mode: DenormalMode,  // new
}
```

### `DoorBackend::capability()` — document hot-swap discipline (B6)

Doc-comment only; no code change.

---

## What ALSO needs to change

### `Shape` — add `AssumptionTag::Metric(MetricKind)` (B5)

Tracked as separate ticket since it's a Shape-level fix that doesn't touch
the DoorBackend trait. Pathmaker can land in same PR as the trait deltas
or as a follow-up — either works. **Recipe discipline:** distance-producing
recipes MUST emit `Metric(...)` tag; entered into DO-NOT.md.

### `IR_VERSION` bump

The trait shape changes invalidate the cached binary format. Bump
`IR_VERSION` 2 → 3. Test `ir_version_pinned` updates with a v3 history note.

### Test additions for the convergence-check

Pathmaker's next implementation pass needs the following new tests for me
to convergence-check:

1. `lower_pipeline_default_calls_lower_per_atom` — default impl correctness
2. `npu_door_lower_returns_not_supported` — NoOpBackend with
   `requires_whole_pipeline=true` returns NotSupported from `lower()`
3. `seeded_determinism_class_carries_seed_hash` — variant construction
4. `seeded_kernels_with_different_seeds_have_different_cache_keys` —
   already works because seed bakes into params; explicit test
5. `stream_poison_blocks_subsequent_dispatch` — after wait returns
   Validity, dispatch returns StreamPoisoned
6. `stream_reset_clears_poison` — after reset, dispatch can be issued
   again
7. `is_stream_poisoned_reports_correctly` — query before / after error
8. `cross_stream_wait_on_poisoned_event_propagates` — stream_wait_event
   on poisoned event returns StreamPoisoned
9. `cpu_stream_poison_is_noop` — HostStream is_stream_poisoned always
   false; stream_reset always Ok
10. `denormal_mode_mismatch_returns_capability_mismatch` — dispatch
    against an artifact built under different DenormalMode errors
11. `denormal_mode_match_succeeds` — dispatch against matching mode works

Plus the Shape-side test:
12. `metric_tag_distinguishes_euclidean_vs_mahalanobis` — same shape
    with `Metric(Euclidean)` vs `Metric(Mahalanobis)` produces different
    cache keys

Twelve new tests targeting the five accepted breaks.

---

## Updated convergence-check criteria

When pathmaker ships these deltas, my re-pass will check:

- [ ] `DoorCodegen::lower_pipeline` exists with default impl per spec
- [ ] `DoorCapability::requires_whole_pipeline` field present
- [ ] `DeterminismClass::SeededDeterministic { seed_hash: [u8; 32] }` variant
- [ ] `DoorDispatcher::stream_reset` and `is_stream_poisoned` methods
- [ ] `LaunchError::StreamPoisoned` and `LaunchError::CapabilityMismatch`
  variants
- [ ] `CompiledArtifact::denormal_mode` field
- [ ] CPU collapse: stream_reset/is_stream_poisoned no-op on HostStream
- [ ] `DoorBackend::capability()` doc-comment names hot-swap discipline +
  refers to TAM-level snapshot in Sweep 23
- [ ] `AssumptionTag::Metric(MetricKind)` in shape.rs
- [ ] `IR_VERSION = 3` with history note: "v3 — Sweep 8A wave-2:
  lower_pipeline + SeededDeterministic + stream poison + denormal_mode +
  Metric tag"
- [ ] Twelve new tests pass; cargo test --workspace stays green

---

## Reflection — what I missed

Adversarial's wave-2 was substantively better than wave-1. Five concrete
breaks against my locked spec, all with the breaking scenarios precisely
demonstrated, all with structurally clean fixes proposed. This is exactly
the role's value.

What I missed in the original Phase 1-8:

- **NPU granularity (A1):** I treated NPUs as an abstract future case,
  not by reading their actual compile-API contracts. Should have probed
  "what's the granularity unit each known NPU compiler accepts?" and
  found Inferentia/XLA/Cerebras all rejecting per-atom. **The correct
  Phase 1 question was "what unit of work does each door's compile API
  natively accept?"**

- **Pipelined async error masking (A3):** I considered the single-
  dispatch case carefully but didn't compose two dispatches in flight.
  Phase 8 forced rejection should have asked "what if multiple dispatches
  are in flight?" — the masking emerges immediately. **The Phase 8
  question to add: "for every async surface, ask the multi-call
  composition question."**

- **DenormalMode silent shift (B4):** I noted MXCSR was a CPU-side
  flag but didn't trace through to "what if it changes between compile
  and dispatch?" The cache-bake principle is necessary but not sufficient
  — **state that's already-in-the-cache-key still needs a runtime check
  if the underlying state can shift mid-process.**

- **SeededDeterministic (A2):** I knew there was a 4th class (I asked
  adversarial to find it) but my own search was lazy. Should have looked
  at common ML training scenarios where seeded reproducibility is the
  whole point.

- **Metric-identity Shape collision (B5):** the assumption-tag system
  is opt-in (recipes that don't tag don't benefit from sharing
  protection). I should have tested "what's the failure mode when two
  recipes BOTH don't tag?" — silent collision is the answer. The
  enforcement gap motivates first-class variants for known-distinguishing
  attributes.

I'm adding all four observations to the Phase-1-through-8 mental
checklist for future deconstructions. Garden material.

The substrate is stronger after wave-2 than it was after wave-1. That's
how this is supposed to work — adversarial finds the gaps; aristotle
deconstructs the response; pathmaker implements the fix; cycle.
