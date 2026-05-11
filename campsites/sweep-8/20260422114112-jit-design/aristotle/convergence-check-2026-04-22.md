# Convergence-Check — pathmaker's 8A implementation pass

**Date:** 2026-04-22 · **Author:** aristotle
**Subject:** verdict on whether pathmaker's landed code matches the
locked spec, and whether any Phase-9 ghosts surfaced.

**TL;DR: SPEC HOLDS. No Phase-9 ghosts. 8A converges.**

`cargo test --workspace`: **1081 passed, 0 failed, 86 ignored**
(73 unit-suites compiled and ran clean; ignored = adversarial #[ignore]
soaks scoped to 8C/8G/future doors per design).

---

## File-by-file verdict against the locked spec

### `door.rs` — TRAIT TRIAD ✓

Every spec'd type, every spec'd method. Subtle wins:
- `WorkgroupShape::SINGLE` and `ScratchSpec::NONE` constants — small
  ergonomic add I didn't spec; saves boilerplate at every CPU call
  site. **Free quality-of-life. Accept.**
- `select_entry()` has a default impl returning `entry_points[0]` —
  matches my Q-amend-3 recommendation (free function default,
  per-backend override). **Accept.**
- `DoorCache::persist()` and `record_timing()` default to no-ops.
  Pathmaker promoted from "every backend implements" to "default
  impl is no-op; backends opt in." This is a **small spec
  divergence** worth naming: it means a CPU backend without persist
  silently drops the call. Per the using-annotation principle, that's
  a hidden behavior. **Recommendation: leave the default impl, but
  the `Loaded` doc-comment should note that calling `persist()`
  before Sweep 8G simply doesn't persist.** Not a blocker.
- `DoorBuffer`/`DoorStream`/`DoorEvent` supertraits with `Send + Sync`
  bounds — exactly as spec'd.
- Object-safe `ErasedDoorBackend` lives alongside the typed
  `DoorBackend` — Q1 resolved cleanly via type-erased
  `Arc<dyn Any + Send + Sync>` on `Loaded.loaded`.
- `IntermediateTag { kind: &'static str, fingerprint: [u8; 32] }` —
  pathmaker added a typed `kind` field separate from the raw
  fingerprint. **Better than the bare `[u8; 32]` in my spec.** The
  `kind` lets the cache surface human-readable telemetry without
  needing a reverse lookup. Accept.

**Phase-9 ghost probe:** searched for hidden TAM decisions in
`door.rs`. None found. The trait surface is decision-free; all
decisions live one level up (in `dispatcher.rs`, in
`JitOp::default_strategy`, in `Shape::is_share_compatible_with`).
Conservation is preserved.

### `strategy.rs` — STRATEGY ENUMS ✓

Locked spec matched exactly. `tag()` method for cache-key
serialization is the exact discipline I asked for — short stable
strings, `strategy_tags_unique` test enforces uniqueness. NOTE:
pathmaker didn't add `#[non_exhaustive]` (substrate-hygiene item
from my design-anchor). Not blocking — adding a fourth variant for
pipeline-lift later still bumps `IR_VERSION` by spec. **Convergence-
check verdict: accept as landed; the IR version bump catches any
future variant addition.**

### `dispatcher.rs` — STRATEGY DISPATCHER ✓

10 tests covering the test-plan item-8 flowchart. Every one of my
8 default-strategy decision-tree assertions is present (Add-All,
Welford-All, Add-Prefix, Welford-Prefix, AffineCompose-Prefix,
AffineCompose-ByKey, MatMul-ByKey, ArgMax-All) plus the three
StrategyOverride variants (Auto, ForceSequential, ForceLifted).
`force_lifted_panics_when_algebra_blocks` is `#[should_panic]`-
asserted with the right error string.

`#[track_caller]` on `dispatch_strategy` and
`lifted_strategy_or_panic` — small win for diagnostics that I
hadn't spec'd. **Accept.**

### `cpu_cranelift.rs` — CPU COLLAPSE ✓

The CPU collapse is exactly what the spec asked for.
`HostStream`/`HostEvent` are ZSTs; `wait` and `stream_wait_event`
are no-ops. `host_event_wait_is_noop` and the existence of
`HostBuffer { bytes: Vec<u8>, dtype }` validate the collapse.

`capability_fingerprint()` BLAKE3s door_id + isa + driver + feature
flags. `capability_fingerprint_is_stable` proves determinism;
`capability_fingerprint_distinct_from_arbitrary_bytes` is a sanity
check on collision avoidance. **Accept.**

`dispatch` errors with `LaunchError::Driver { code: 0, detail }`
until 8C lands real codegen. `cranelift_lower_is_not_yet_implemented`
asserts. **Accept** — clean stub shape; 8C drops in cleanly.

The two `_assert_any_send_sync` and `_empty_metadata` helpers exist
to silence unused-import warnings until 8C; they will get
deleted. **Acceptable transient hygiene.**

### `fingerprint.rs` — CACHE-KEY ✓

`feed_strategy()` lives at `0x18` tag, after every other shape
component but before door_id and params. Per my locked spec,
`ExecutionStrategy` IS in the cache key, and pathmaker's discipline
(tag byte + variant code + nested perm/reason code) keeps it
distinct from neighbouring fields.

`IR_VERSION = 2`. The pinning test `ir_version_pinned` has the
history note: "v1 — Sweep 8B initial. v2 — Sweep 8A liftability-
default: ExecutionStrategy folded into the cache key." **Spec
discipline observed.**

`cache_key()` defaults to `Lifted`; `cache_key_with_strategy()` is
the explicit version. Smart pre-Sweep-8H ergonomic — recipes that
haven't decided a strategy don't have to invent one. **Accept.**

The 18 hash-distinguishability tests cover every axis I'd want to
test (different ops, groupings, validity, doors, params, dim hints,
assumption tags, dtype, matmul-n, semiring kind, known-non-finite,
no-split-collision, negative-zero, NaN-bit-pattern). **More
discipline than I spec'd.** The `keys_no_split_collision` test in
particular is the right adversarial probe.

**One spec gap caught and resolved by pathmaker:** the
`dim_hint_affects_cache_key` test confirms that `Static(N)` produces
a different cache entry than `Static(M)` than `Dynamic` — the
size-specialization-bakes-in story. This was implicit in my spec
and pathmaker made it explicit.

---

## Pathmaker's three open questions — answered

### Q(a) — "Did I miss any shape-of-state distinction that should live on the trait?"

**No.** The state-of-shape lives in three places:
- `JitOp::state_repr() -> StateRepr` for codegen alloc (was already
  in jit_op.rs)
- `Shape` (dtype + dim + grouping + validity + tags + non-finite
  marker) for the input-side
- `CompiledArtifact { entry_points + scratch_required + determinism +
  pipeline_metadata }` for the output-side

That's the full trio. Pipeline-lift will add a `FusedWith` annotation
in a future sweep but it's a property of an atom call, not of state
shape, so it lives on the dispatcher's annotated graph rather than
the trait surface.

### Q(b) — "`Any + Send + Sync` on `Loaded.loaded` and `ArtifactBinary::ModuleHandle` — right object-safety boundary?"

**Yes.** Three reasons:
1. The hot per-recipe path goes through the concrete backend type
   (CraneliftBackend, future CudaBackend), not through `dyn
   ErasedDoorBackend`, so monomorphization preserves zero-cost.
2. The cross-door scheduler (TAM, post-Sweep-23) goes through
   `ErasedDoorBackend` and pays one vtable hop per call; perfectly
   acceptable for cross-door coordination.
3. The `Any` lives at the leaf (loaded handle / module handle), not
   at the trait shape. Backends downcast in their own dispatch
   methods using their concrete type. CudaBackend will downcast
   `loaded.loaded` to `Arc<CUfunction>` in its `dispatch()`; CPU
   will downcast to `Arc<JITFunction>` (when 8C lands).

The `Send + Sync` bounds are correct: every backend's loaded handle
needs to cross thread boundaries (TAM scheduler is multi-threaded;
work-stealing can hand off a `Loaded` between threads).

### Q(c) — "CPU collapse via ZST: compatible with thread-pool-pending-Q3 answer?"

**Yes.** When Sweep 8.5 adds thread-pool dispatch on the CPU, the
options are:
- (A) Replace `HostStream` with a wrapper around a thread-pool
  handle. Trait surface unchanged; ZST → small struct. Costs one
  word per stream; no per-call overhead change.
- (B) Keep `HostStream` as ZST and add a separate
  `HostThreadPool` parameter to `CraneliftBackend`. Stream is
  identity; thread-pool is the actual scheduler.
- (C) Split into `CraneliftBackend` (single-threaded) and
  `CraneliftPoolBackend` (multi-threaded), both implementing
  `DoorBackend`. User picks at construction.

Path (A) is least disruptive; (B) is cleanest separation; (C) is
most explicit. I'd lean (B) when 8.5 lands. The ZST decision today
doesn't preclude any of them. **No constraint introduced.**

---

## Substrate-hygiene items from my prior design anchor

Five items I flagged after 8A landed. Status:

1. **`#[non_exhaustive]` on `ExecutionStrategy`** + tag-based cache
   serialization. **Tag-based: ✓ (via `tag()` method).
   `#[non_exhaustive]`: NOT applied.** Acceptable because adding a
   fourth variant (`Fused { with: AtomCallId }` for pipeline-lift)
   bumps `IR_VERSION`, which invalidates persisted caches by
   construction.

2. **`JitOp::canonical_structure()` delegating to OpKind.** Not
   verified yet — pathmaker may add in 8C. Flag for follow-up.

3. **`DeterminismClass::weakest_of`.** Not landed. Follow-up; not
   blocking until pipeline-lift sweep.

4. **`Validity::can_fuse_with`.** Not landed. Follow-up; same
   reason.

5. **Doc-comment on `is_share_compatible_with` asserting transitivity
   + reflexivity, with property test.** Need to verify in shape.rs.

Items 2-5 are post-8A polish for the pipeline-lift sweep substrate.
Not 8A blockers; flagging in a separate quick-pass note for
pathmaker to fold in alongside the (Add,Prefix) + (Welford,Prefix) +
explicit `preserves_order` test additions he already mentioned.

---

## Convergence verdict

**The trait spec holds in code.** Pathmaker's implementation matches
the locked spec including: liftability default, three sequential
reasons, cache-key strategy folding, CPU collapse via ZST, three-
trait split, ErasedDoorBackend object-safe facade, NoOpBackend smoke
suite, type-erased `Any` for backend handles, `IR_VERSION` discipline.

**No Phase-9 ghosts surfaced.** The convergence check looked for
anything pathmaker had to add that the spec didn't anticipate.
Findings:
- `WorkgroupShape::SINGLE` / `ScratchSpec::NONE` constants:
  ergonomic, not a ghost.
- `IntermediateTag::kind` field: minor enrichment, not a ghost.
- Default no-op `persist()` / `record_timing()`: small spec
  divergence (see § door.rs notes). Worth a doc-comment update;
  not a structural ghost.
- `#[track_caller]` on dispatcher: ergonomic, not a ghost.
- `cache_key()` convenience defaulting to `Lifted`: pre-Sweep-8H
  ergonomic, not a ghost.

All five surface differences are quality-of-life additions, not
constraint violations. **Spec ≅ implementation.**

---

## What follows

- Pathmaker's two-test follow-up ((Add,Prefix), (Welford,Prefix),
  explicit preserves_order coverage) is fine to land whenever.
- The five substrate-hygiene items remain pending for the post-8A
  polish or pipeline-lift sweep prep.
- Adversarial's wave-1 conversion of `#[ignore]` tests can proceed
  against the landed surface; no changes needed.
- Sweep 8C/8D/8E/8F/8G/8H/8I/8J/8K queued; the substrate is ready.

8A is closed. Convergence-check passes.
