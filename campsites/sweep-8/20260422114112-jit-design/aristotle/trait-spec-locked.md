# `DoorBackend` Trait Spec — LOCKED

**Sweep 8 / Task 8A** · Author: aristotle · Date: 2026-04-22

This is the consolidated, normative specification for the `DoorBackend`
trait surface. It supersedes the placeholder stub in
`crates/tambear/src/jit/door.rs`. Pathmaker may paste this verbatim as
the doc-comment header on `door.rs` and implement the corresponding
types.

The reasoning behind every line is in two companion documents in this
campsite:

- `phase-1-8-deconstruction.md` — the original 21 assumptions / 19
  truths / 10 reconstructions / Phase-8 forced rejection that
  produced R10′
- `addendum-liftability-default.md` — the lift-as-default
  re-deconstruction that added `ExecutionStrategy` + `supports()` +
  `default_strategy()` + `Grouping::preserves_order()`

If anything in code drifts from this spec without an aristotle re-pass,
that's the failure mode the campsite was opened to prevent.

---

## Design contract — what the trait MUST achieve

1. **Survives every door in DEC-019** — CPU/Cranelift, NVIDIA/PTX,
   Vulkan/SPIR-V, Metal/AIR-MSL, DX12/DXIL, AMD/ROCm, Intel/oneAPI,
   future NPUs. Adding any door is implement-the-trait work, never
   redesign-the-trait work.
2. **CPU collapses by construction** — associated types degenerate to
   `()` / `HostBuffer` / `HostEvent` so the Cranelift path pays zero
   for the GPU shape. Trait absorbs CPU as a degenerate case of GPU,
   not the other way around.
3. **Lifting is the default** per Tekgy's design note. Sequential is
   fallback codegen. Three exit conditions only: user override,
   algebra blocks, Fock boundary. The decision lives on `JitOp`, not
   on the backend.
4. **Bit-exact determinism is structural.** `DeterminismClass` rides
   on every `CompiledArtifact`; consumers reason about reproducibility
   from the type system, not from documentation.
5. **Three responsibilities, one trait.** Codegen / cache / dispatcher
   are orthogonal concerns; the trait surface separates them so each
   can evolve independently. The marker trait `DoorBackend` simply
   requires all three.
6. **Nothing in the trait names a vendor.** No `cuda::`, no `vk::`,
   no `metal::`. Vendor-specific FFI lives in sealed sub-crates
   (`tambear-door-cuda`, etc., per Sweep 17+).

---

## The trait surface

```rust
// ============================================================
// Door identity (already landed by pathmaker; spec lock confirms)
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DoorId(pub u32);

impl DoorId {
    pub const CPU:    DoorId = DoorId(0);
    pub const CUDA:   DoorId = DoorId(1);
    pub const VULKAN: DoorId = DoorId(2);
    pub const METAL:  DoorId = DoorId(3);
    pub const DX12:   DoorId = DoorId(4);
    pub const AMD:    DoorId = DoorId(5);
    pub const INTEL:  DoorId = DoorId(6);
    pub const NOOP:   DoorId = DoorId(255);
}

// ============================================================
// Capability — door tells us what it supports
// ============================================================

#[derive(Debug, Clone)]
pub struct DoorCapability {
    pub door_id: DoorId,
    /// Vendor-specific ISA / shader-model / Cranelift-ISA-flags
    /// version. Folded into the cache key.
    pub isa_version: IsaVersion,
    /// Driver version (vk driver build, CUDA driver, macOS build,
    /// Cranelift crate version). Folded into the cache key so a
    /// driver update invalidates stale binaries.
    pub driver_version: Version,
    /// Subgroup / warp / SIMD-lane width (32 NVIDIA, 64 AMD-GCN,
    /// 4-16 CPU SIMD via Cranelift, 1 if no parallelism).
    pub subgroup_sizes: SmallVec<[u32; 4]>,
    pub supports_fma: bool,
    pub supports_f16: bool,
    pub supports_bf16: bool,
    pub denormal_mode: DenormalMode,
    /// Per-invocation shared / threadgroup / scratch memory ceiling.
    /// CPU: stack size budget. GPU: shared-memory bytes per workgroup.
    pub max_scratch_bytes: u64,
    /// Maximum workgroup / thread count per dispatch.
    pub max_workgroup: WorkgroupShape,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DenormalMode {
    /// IEEE 754 strict — denormals preserved.
    Ieee,
    /// Flush-to-zero on input.
    Ftz,
    /// Denormals-are-zero on input.
    Daz,
    /// Both FTZ + DAZ.
    FtzDaz,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct IsaVersion(pub String);   // e.g. "ptx_7.5", "spirv_1.6", "cranelift_x86_64_avx512"

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Version(pub String);

// ============================================================
// Cache key — BLAKE3 of (ir, params, shape, validity, strategy,
// door, capability)
// ============================================================

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub ir_hash: [u8; 32],
    pub param_hash: [u8; 32],
    pub shape_fingerprint: [u8; 32],
    pub validity_policy: Validity,
    pub strategy: ExecutionStrategy,   // <- ADDED per liftability addendum
    pub door_id: DoorId,
    pub capability_fingerprint: [u8; 32],
}

// ============================================================
// Execution strategy — liftability is default; sequential is fallback
// ============================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExecutionStrategy {
    /// Parallel-prefix / scan / fused-pass form. **DEFAULT for any
    /// Op whose algebra admits it.** Lowered to whatever parallelism
    /// primitives the door exposes (warp shuffle on NVIDIA, subgroup
    /// ops on Vulkan, threadgroup on Metal, SIMD lanes + thread pool
    /// on CPU). Bit-exact via fixed deterministic associativity tree.
    Lifted,

    /// Lifted with permutation envelope (P ∘ T ∘ P⁻¹). For
    /// non-commutative Ops where the grouping doesn't preserve
    /// order. Permute-pass + lifted scan + inverse-permute.
    LiftedConjugated { perm_kind: PermutationKind },

    /// Sequential single-stream codegen. **Fallback only.** Reasons:
    Sequential { reason: SequentialReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SequentialReason {
    /// User explicitly forced via `using(strategy="sequential")`.
    UserOverride,
    /// Op's algebra blocks lift (no Associativity, OR non-commutative
    /// with no fitting conjugation pattern).
    AlgebraBlocks,
    /// TAM determined the recipe sits past the Fock boundary.
    FockBoundary,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PermutationKind {
    /// Sort by some key column before the scan.
    SortAscending,
    SortDescending,
    /// Reverse order (e.g. backward pass).
    Reverse,
    /// Hilbert / Z-order curve for spatial locality.
    SpaceFilling,
    /// User-supplied permutation index (the dispatcher carries it).
    Custom,
}

#[derive(Debug, Clone, Copy)]
pub enum StrategyOverride {
    /// Default: dispatcher calls `JitOp::default_strategy(shape)`.
    Auto,
    /// Force sequential. Honoured even when lifted is available.
    ForceSequential,
    /// Force lifted. Panics if `default_strategy()` would not return
    /// Lifted/LiftedConjugated for this (op, shape).
    ForceLifted,
}

// ============================================================
// Compiled artifact
// ============================================================

#[derive(Debug)]
pub struct CompiledArtifact {
    pub door: DoorId,
    /// Door-opaque binary. CPU: cranelift `JITModule` finalized
    /// pointer wrapped in `Arc`. PTX: text or bytecode. SPIR-V:
    /// 4-byte words. DXIL: bytes. Tambear above the trait does NOT
    /// interpret this.
    pub binary: ArtifactBinary,
    /// One per sub-kernel produced by lowering. Lifted reductions
    /// often produce three: per-warp / per-block / device-wide.
    pub entry_points: SmallVec<[EntryPoint; 4]>,
    /// Vendor-specific PSO / pipeline state / descriptor layout.
    /// Door-opaque.
    pub pipeline_metadata: PipelineMetadata,
    /// Per-invocation scratch the dispatcher must allocate.
    pub scratch_required: ScratchSpec,
    /// Determinism guarantee carried structurally.
    pub determinism: DeterminismClass,
}

#[derive(Debug)]
pub enum ArtifactBinary {
    /// Cranelift case — finalized function pointer. Type-erased
    /// behind `Arc<dyn Any>` so the trait is door-agnostic.
    /// Resolves Q2 from the original Phase 1-8 file.
    ModuleHandle(Arc<dyn Any + Send + Sync>),
    /// Vendor-IR case — bytes the driver consumes (PTX text, SPIR-V
    /// words, DXIL, etc.).
    RawBinary(Vec<u8>),
}

#[derive(Debug, Clone, Copy)]
pub struct EntryPoint {
    /// Stable, door-agnostic name: "main_lifted", "warp_reduce",
    /// "block_reduce", "device_reduce", "main_sequential", etc.
    pub name: &'static str,
    /// Workgroup-shape constraint. None = "any shape".
    pub workgroup_constraint: Option<WorkgroupShape>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkgroupShape {
    /// Threads per workgroup along x / y / z. CPU degenerates to
    /// (1, 1, 1) and uses thread-pool over outer dispatch.
    pub xyz: [u32; 3],
}

#[derive(Debug, Clone, Copy)]
pub struct ScratchSpec {
    /// Per-workgroup scratch in bytes. 0 means "no scratch needed."
    pub bytes: u64,
    /// Alignment requirement.
    pub align: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeterminismClass {
    /// Bit-exact across runs and across all valid (door, capability)
    /// combinations. The default for tambear math.
    Deterministic,
    /// Same multiset of inputs gives the same output bit pattern, but
    /// order-of-arrival within a group affects output. ArgMax with
    /// lowest-index tiebreak is the worked example.
    OrderDependent,
    /// Output bits depend on RNG / thread-scheduling. Opt-in only via
    /// `using(sum_strategy="nondet")` or equivalent.
    NonDeterministic,
}

// ============================================================
// Errors — three structural classes (per Phase 2 truth T10)
// ============================================================

#[derive(Debug, Clone)]
pub enum CompileError {
    /// IR contains a variant this door doesn't support (yet).
    UnsupportedIr { op: String, door: DoorId },
    /// Shape combination this door can't codegen (e.g. f16 on a CPU
    /// without f16 intrinsics).
    UnsupportedShape { detail: String },
    /// User-forced strategy that the algebra doesn't admit
    /// (StrategyOverride::ForceLifted on a non-associative Op).
    /// Honest error rather than silent degradation.
    StrategyNotApplicable { strategy: ExecutionStrategy, reason: String },
    /// Capability check failed — door doesn't meet a requirement
    /// the IR demands.
    CapabilityMissing { detail: String },
    /// Code generation itself failed (cranelift internal error,
    /// driver rejected our PTX, etc.).
    Codegen { detail: String },
    /// Sweep-8 stub — backend doesn't implement this combination yet.
    NotYetImplemented,
}

#[derive(Debug, Clone)]
pub enum LaunchError {
    OutOfMemory { bytes_requested: u64 },
    Timeout,
    DeviceLost,
    /// `Validity::Error` was set and the dispatch hit a non-finite
    /// at runtime. `wait()` is the surface that returns this.
    Validity { details: String },
    /// Buffer dtype / size mismatch with kernel signature.
    BufferContract { detail: String },
    /// Driver-level error (CUresult/VkResult/HRESULT after we map it).
    Driver { code: i64, detail: String },
}

// ============================================================
// Buffer / Stream / Event supertraits — the door-opaque memory
// + scheduling surface. All three associated types collapse to
// trivial unit-shaped values on CPU; GPU doors realize them.
// ============================================================

pub trait DoorBuffer: Send + Sync {
    fn len_bytes(&self) -> u64;
    fn dtype(&self) -> ScalarTy;
}

pub trait DoorStream: Send + Sync {}

pub trait DoorEvent: Send + Sync {}

#[derive(Debug, Clone, Copy)]
pub enum AllocHint {
    /// One-shot use; deallocate immediately after readback.
    OneShot,
    /// Persistent across many dispatches (long-lived intermediates).
    Persistent,
    /// Zero-copy host alias if the door supports unified memory
    /// (CUDA UM, Apple Silicon shared memory). Backend may ignore.
    SharedHost,
}

// ============================================================
// The trait triad
// ============================================================

/// IR -> binary (door-specific lowering)
pub trait DoorCodegen {
    fn lower(
        &self,
        op: &JitOp,
        shape: &Shape,
        strategy: ExecutionStrategy,
        params: &[u8],
        cap: &DoorCapability,
    ) -> Result<CompiledArtifact, CompileError>;

    /// Feature-detection: can this door compile (op, shape, strategy)?
    /// Lets the dispatcher avoid a failed compile.
    fn supports(
        &self,
        op: &JitOp,
        shape: &Shape,
        strategy: ExecutionStrategy,
    ) -> bool;
}

/// Persistent + in-memory kernel cache + intermediate-sharing hooks
pub trait DoorCache {
    /// Whole-kernel cache lookup keyed on `CacheKey`.
    fn get(&self, key: &CacheKey) -> Option<Arc<Loaded>>;
    fn put(&self, key: CacheKey, art: CompiledArtifact) -> Arc<Loaded>;

    /// TamSession sharing — one entry per shareable intermediate.
    fn get_intermediate(&self, tag: &IntermediateTag) -> Option<Arc<Loaded>>;
    fn put_intermediate(&self, tag: IntermediateTag, art: CompiledArtifact)
        -> Arc<Loaded>;

    /// Flush in-memory entries to disk
    /// (`$XDG_CACHE_HOME/tambear/kernels/` or platform equivalent).
    fn persist(&self);

    /// Profile-feedback channel (per Phase 8 forced rejection insight).
    /// The cache may use timing data to pick among multiple entry
    /// points on subsequent calls.
    fn record_timing(&self, k: &Loaded, entry: EntryPoint, wall_ns: u64);
}

#[derive(Debug)]
pub struct Loaded {
    pub artifact: Arc<CompiledArtifact>,
    /// Door-specific loaded handle (CUfunction / VkPipeline /
    /// MTLFunction / cranelift function pointer).
    pub loaded: Arc<dyn Any + Send + Sync>,
}

/// Memory + scheduling
pub trait DoorDispatcher {
    type Buffer: DoorBuffer;
    type Stream: DoorStream;
    type Event: DoorEvent;

    fn alloc(
        &self,
        bytes: u64,
        dtype: ScalarTy,
        hint: AllocHint,
    ) -> Result<Self::Buffer, LaunchError>;

    fn import_host(&self, bytes: &[u8], dtype: ScalarTy)
        -> Result<Self::Buffer, LaunchError>;

    fn export_host(
        &self,
        b: &Self::Buffer,
        out: &mut [u8],
        e: &Self::Event,
    ) -> Result<(), LaunchError>;

    fn dispatch(
        &self,
        stream: &Self::Stream,
        k: &Loaded,
        entry: EntryPoint,
        inputs: &[&Self::Buffer],
        outputs: &mut [&mut Self::Buffer],
        workgroups: WorkgroupShape,
        scratch: ScratchSpec,
    ) -> Result<Self::Event, LaunchError>;

    /// Pick the right entry point for a given shape (warp vs block
    /// vs device kernel). Backend-specific heuristic; default impl
    /// returns the first entry point.
    fn select_entry(&self, k: &Loaded, shape: &Shape) -> EntryPoint;

    fn wait(&self, e: Self::Event) -> Result<(), LaunchError>;

    fn default_stream(&self) -> Self::Stream;
    fn new_stream(&self) -> Result<Self::Stream, LaunchError>;
    fn stream_wait_event(
        &self,
        stream: &Self::Stream,
        event: &Self::Event,
    ) -> Result<(), LaunchError>;
}

/// Marker — every backend implements all three.
pub trait DoorBackend: DoorCodegen + DoorCache + DoorDispatcher {
    fn capability(&self) -> &DoorCapability;
}

/// Object-safe facade — the type-erased entry the multi-door scheduler
/// uses without monomorphizing on associated types. Backends provide
/// it via blanket impl over their concrete `DoorBackend`.
pub trait ErasedDoorBackend: Send + Sync {
    fn door_id(&self) -> DoorId;
    fn capability_fingerprint(&self) -> [u8; 32];
    // Compile/dispatch surfaces use type-erased Buffer/Stream/Event
    // (boxed), at the cost of one vtable hop per call. Used by TAM
    // for cross-door scheduling; not used on the hot per-recipe path.
    // Full sketch in dispatcher.rs (sweep 8H scope).
}
```

---

## CPU Cranelift collapse — the proof the trait works

```rust
pub struct CraneliftBackend { /* ... */ }

#[derive(Debug)]
pub struct HostBuffer {
    pub bytes: Vec<u8>,
    pub dtype: ScalarTy,
}
impl DoorBuffer for HostBuffer {
    fn len_bytes(&self) -> u64 { self.bytes.len() as u64 }
    fn dtype(&self) -> ScalarTy { self.dtype }
}

#[derive(Debug, Clone, Copy)] pub struct HostStream;
impl DoorStream for HostStream {}

#[derive(Debug, Clone, Copy)] pub struct HostEvent;
impl DoorEvent for HostEvent {}

impl DoorDispatcher for CraneliftBackend {
    type Buffer = HostBuffer;
    type Stream = HostStream;
    type Event  = HostEvent;

    fn default_stream(&self) -> HostStream { HostStream }
    fn new_stream(&self) -> Result<HostStream, LaunchError> { Ok(HostStream) }
    fn wait(&self, _e: HostEvent) -> Result<(), LaunchError> { Ok(()) }
    fn stream_wait_event(&self, _: &HostStream, _: &HostEvent)
        -> Result<(), LaunchError> { Ok(()) }
    // ... alloc/import/export/dispatch/select_entry are real ...
}
```

Zero runtime cost: `HostStream` and `HostEvent` are zero-sized types
that monomorphize away. `wait` and `stream_wait_event` collapse to no-ops
in release builds.

---

## What lives WHERE

| Concern | Lives in |
|---|---|
| IR enum `JitOp` | `jit_op.rs` (already landed) |
| Scalar dtype `ScalarTy` | `jit_op.rs` (already landed) |
| Shape struct + AssumptionTag | `shape.rs` (already landed) |
| `CacheKey`, `FingerprintHasher` | `fingerprint.rs` (already landed; needs `strategy` field) |
| `ExecutionStrategy`, `SequentialReason`, `PermutationKind`, `StrategyOverride` | `strategy.rs` (NEW, ~100 lines) |
| `JitOp::default_strategy(shape)`, `JitOp::lifted_strategy_or_panic(shape)` | `jit_op.rs` (additions) |
| `Grouping::preserves_order()` | `accumulate.rs` (one method on the existing Grouping enum) |
| `DoorBackend` triad + supertraits + errors + capability + compiled artifact | `door.rs` (rewrite per this spec) |
| `CraneliftBackend` impl | `cpu_cranelift.rs` (NEW; Sweep 8C+ fills in real codegen) |
| Dispatcher (consumes `using_override`, picks strategy, picks backend, hits cache) | `dispatcher.rs` (NEW; thin skeleton in 8A, real wiring in 8H) |
| Object-safe facade `ErasedDoorBackend` | `door.rs` (declaration); blanket impl alongside `CraneliftBackend` |

---

## Test plan — what 8A's tests must assert

(These are the stub-level assertions. Real codegen tests land in 8C+.)

1. **Trait-shape compiles.** `DoorBackend`, `DoorCodegen`, `DoorCache`,
   `DoorDispatcher` declarations type-check together; `Loaded`,
   `CompiledArtifact`, `CacheKey`, the error enums, and the capability
   struct all exist with the spec'd fields.
2. **NoOp backend.** A `NoOpBackend` impl that returns
   `Err(CompileError::NotYetImplemented)` for every `lower()` call,
   `false` from `supports()` (or `true` for a trivial smoke variant —
   pathmaker's call), `()` for streams/events, returns Ok from
   `wait`/`stream_wait_event`, errors on real `dispatch`. Lets us test
   the trait surface in isolation.
3. **`DoorId` constants distinct.** Already covered in pathmaker's
   landed test; preserved.
4. **`ExecutionStrategy` cache-key participation.** Same `(op, shape,
   params, validity, door, capability)` with different `strategy`
   produces different `CacheKey`. Folded into the BLAKE3 hash.
5. **`JitOp::default_strategy` decision tree.** 8 assertions:
   - `(Add, All)` → `Lifted`
   - `(Add, Prefix)` → `Lifted`
   - `(Welford, All)` → `Lifted`
   - `(Welford, Prefix)` → `Lifted`
   - `(AffineCompose, Prefix)` → `Lifted` (assoc + order-preserving)
   - `(AffineCompose, ByKey)` → `Sequential { AlgebraBlocks }`
   - `(ArgMax, All)` → `Lifted` (assoc + order-preserving via index
     tiebreak — note ArgMax declares `OrderDependent` determinism so
     codegen knows to fix the tree shape)
   - `(MatMulPrefix { n: 3 }, ByKey)` → `Sequential { AlgebraBlocks }`
6. **`Grouping::preserves_order()`.** Returns:
   - `All` → false, `ByKey` → false, `Prefix` → true, `Segmented` → true,
   - `Windowed` → true, `Tiled` → true, `Graph` → false (conservative),
   - `Probabilistic` → false.
7. **`JitOp::lifted_strategy_or_panic` panics** when forced lifted on
   `(MatMulPrefix { n: 3 }, ByKey)` (no conjugation pattern fits).
8. **`StrategyOverride::Auto` calls `default_strategy`** in the
   dispatcher skeleton; `ForceSequential` produces `Sequential {
   UserOverride }`; `ForceLifted` panics on AlgebraBlocks Ops.

Target: ~12-15 new tests, all green at every commit per the standing
constraint.

---

## Open questions (still mine to chase)

- **Q1** (originally for pathmaker): `Loaded.loaded: Arc<dyn Any +
  Send + Sync>` — type-erased. **My recommendation: yes, keep it
  type-erased** at the trait level. Performance-critical paths
  monomorphize via the concrete backend type; the type-erased path is
  only for the multi-door scheduler. If pathmaker disagrees, the
  associated-type path on `DoorCache` is the alternative — speak up.

- **Q2**: `ArtifactBinary { ModuleHandle, RawBinary }` — **resolved**
  in this spec. Cranelift uses `ModuleHandle`; vendor doors use
  `RawBinary`.

- **Q3**: Real CPU streams (thread-pool) for Sweep 8 vs `Stream = ()`
  — **my recommendation: Stream = HostStream (zero-sized type), wait
  is no-op** for Sweep 8. Thread-pool dispatch lands as a Sweep-8.5
  optimization once the codegen ships and we have a benchmark to tune
  against. Don't pre-optimize the parallelism layer before we have
  measurable kernels.

---

## Sweep-17 readiness check (NVIDIA PTX via CUDA Driver)

Walking the spec against Sweep-17's needs as a sanity check:

| Sweep 17 need | This spec provides |
|---|---|
| Raw CUDA Driver FFI (cuModuleLoadData / cuLaunchKernel) | `DoorCodegen::lower` produces `ArtifactBinary::RawBinary(Vec<u8>)` containing PTX text/bytecode; `Loaded.loaded` wraps the `CUfunction` |
| Per-device PTX cache (sm_75 ≠ sm_90) | `DoorCapability::isa_version` participates in `CacheKey` |
| Async dispatch | `Stream = CUstream`, `Event = CUevent` (real types on CUDA backend) |
| Memory mgmt (cuMemAlloc) | `DoorDispatcher::alloc` returns `CudaBuffer`; `import_host` does `cuMemcpyHtoD`; `export_host` does `cuMemcpyDtoHAsync` + fence |
| Bit-exact CPU/GPU parity test | `DeterminismClass::Deterministic` rides on both backends' `CompiledArtifact`; recipe-level test asserts byte equality of CPU and CUDA outputs |
| Persistent kernel cache | `DoorCache::persist` flushes; `get_intermediate` enables TamSession sharing |
| Bit-exact reductions across warp/block/device | `EntryPoint` machinery lets codegen ship 3 kernels in one artifact; `select_entry` picks per shape |
| Validity::Error on GPU | `LaunchError::Validity` surfaces at `wait()`; the kernel writes an abort-flag to a status buffer that wait() checks |

All Sweep-17 needs land on this trait without redesign. Confirms the
spec's primary success criterion.

---

## Acceptance

When pathmaker has implemented this spec in `door.rs` + `strategy.rs` +
the additions to `jit_op.rs` and `accumulate.rs`, with the test plan
above all green, task 8A is complete. Aristotle re-passes once the
code lands to confirm no Phase-8 ghost emerged in implementation that
the spec missed at the type level.

Adversarial's open attacks (NPU graph compilers, RNG-seeded
determinism, async Validity::Error race, plus the five lift-default
counterexamples in the addendum) remain standing as soak-tests; any
that break this spec re-opens 8A.

---

*Spec locked. Companion docs: phase-1-8-deconstruction.md and
addendum-liftability-default.md remain authoritative for the reasoning;
this file is the implementation surface.*
