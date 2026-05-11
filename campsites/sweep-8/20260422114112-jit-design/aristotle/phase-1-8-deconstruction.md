# Phase 1-8 Deconstruction — `DoorBackend` trait shape

**Campsite:** `sweep-8/jit-design` · **Author:** aristotle · **Date:** 2026-04-22

The mandate: a trait shape that fits **every** DEC-019 door (CPU/Cranelift,
NVIDIA/PTX, Vulkan/SPIR-V, Metal/AIR, DX12/DXIL, AMD/ROCm, Intel/oneAPI,
future NPUs) without redesign. A trait that fits CPU and forces redesign
for PTX is the exact failure mode the sweep is structured to prevent.

---

## Phase 1 — Assumption Autopsy

Every assumption I can find embedded in the "CPU-first" framing and the
Sweep 8 README. Each is a candidate for removal.

1. **A1 — synchronous dispatch.** The word `dispatch` in Sweep 8 implies
   call-site blocking until result. Cranelift-JIT'd CPU code IS
   synchronous. Every GPU door is fundamentally async (command buffer
   submission → fence/event → completion).
2. **A2 — single host-addressable buffer space.** CPU code takes `&[f64]`
   and returns `Vec<f64>`. Pointers are first-class. Every GPU door has
   a **device address space** distinct from the host — the trait cannot
   assume buffer identity is a raw Rust pointer.
3. **A3 — one kernel per compile.** `compile(ir, shape) -> Kernel`
   suggests a 1:1 map. But on GPU, a single dispatch bundles multiple
   kernels (pre-kernel, main kernel, epilogue kernel) into one
   command-buffer record. TamSession sharing further means one
   compile produces **fused kernels** that share intermediates across
   recipe boundaries.
3b. **A3-corollary — compile returns one artifact.** Implied by A3.
   A GPU compile produces (a) an IR blob (PTX/SPIR-V/DXIL) AND (b) a
   *pipeline-state-object*-equivalent (compute pipeline, root signature,
   descriptor layout). Two artifacts, not one.
4. **A4 — reads are synchronous memory loads.** CPU recipes read from
   `&[f64]` directly. GPU doors require an explicit **readback path**
   (`cuMemcpyDtoH`, `vkCmdCopyBuffer`+barriers+fence, MTLBuffer contents
   after fence). Result is *not available at call-return*.
5. **A5 — float scalars are the only return type.** `O::State` is
   `Clone + Send + Sync + 'static`. GPU scans produce device-resident
   states; downstream recipes may consume them WITHOUT a readback. The
   trait must admit "state lives on-device."
6. **A6 — ownership is host-exclusive.** CPU code owns its buffers.
   GPU doors share buffers across multiple dispatches, across
   TamSession consumers, across pipeline steps. Buffer *lifetime* is
   a runtime concern the trait must expose.
7. **A7 — cache key = shape + params.** Sweep 8 README says the key
   is `(tambear_ir_hash, param_bag_hash, shape_fingerprint,
   validity_policy, door_id)`. **Missing from this list:**
   - *device compute capability* (PTX for sm_75 ≠ sm_90; DXIL for
     SM 6.6 ≠ SM 6.8; Metal GPU family ≠ Apple M4)
   - *driver version* (Vulkan pipeline cache invalidates across
     driver updates; CUDA PTX JIT can behave differently)
   - *workgroup/warp size* (codegen specializes on these)
8. **A8 — single-device dispatch.** Sweep 17 reserves multi-GPU as its
   own later sweep. But the trait decides now whether `compile` +
   `dispatch` take a device handle or not. If we bake the device
   implicitly ("backend owns one device") we will redesign later.
9. **A9 — host allocates, backend consumes.** CPU: Rust's allocator.
   GPU: the driver allocates device memory with specific alignment
   (CUDA 512-byte, Vulkan NonCoherentAtomSize). The *allocator* is a
   per-door concern.
10. **A10 — no scratch memory concept.** Welford's parallel merge fits
    in a register. A large `Probabilistic` scatter needs **shared
    memory / groupshared / threadgroup memory** — a per-invocation
    scratch the kernel declares at compile time and the dispatch
    allocates. CPU has stack; GPU has explicit scratch. The trait
    must carry the scratch-requirement contract.
11. **A11 — determinism is free.** CPU Cranelift with IEEE round-to-
    nearest is bit-exact across runs. GPU: thread-scheduling order is
    non-deterministic in the general case; deterministic reductions
    require explicit tree-reduction kernels with fixed associativity
    order. The **determinism contract** belongs on the trait, not on
    the backend impl's goodwill.
12. **A12 — NaN/Inf propagation is uniform.** IEEE 754 says what each
    hardware op does with NaN, but: CUDA compute shaders default to
    fast-math (flush-to-zero on denormals), DX12/DXIL defaults are
    shader-model-dependent, Vulkan default is relaxed (spec allows
    FTZ). Our Validity contract needs a *compiler flag surface*
    exposed through the trait.
13. **A13 — Kingdom A forever.** `accumulate` decomposes to scan +
    reduce when the Op is associative. But `Op::ArgMax` with lowest-
    index tiebreak is associative yet *order-sensitive at ties*;
    parallel tree reduction may break this. The trait must surface
    the Op's "parallel-safe-under-tiebreak" status so codegen picks
    the correct reduction kernel (deterministic tree vs warp-shuffle).
14. **A14 — one execution stream.** Sweep 8 single-backend-CPU does
    whole dispatch synchronously. GPU doors benefit from **queue /
    stream / command-buffer parallelism** — submit multiple
    independent dispatches simultaneously. The trait must not
    serialize what the substrate can parallelize.
15. **A15 — error = panic.** Cranelift JIT errors can panic. CUDA
    `cuLaunchKernel` returns `CUresult`. Vulkan returns `VkResult`.
    DX12 returns `HRESULT`. The trait MUST distinguish **compile
    errors** (bad IR) from **dispatch errors** (OOM, device lost,
    timeout) — these are structurally different.
16. **A16 — inputs are f64.** The entire Op substrate assumes f64.
    GPU hardware has f32, f16, bf16, i8, tensor-core types. When we
    need f32 for performance (Sweep N), does the trait carry a
    *precision-tag* so the Shape fingerprint is honest?
17. **A17 — no host/device round-trip bookkeeping.** A recipe that
    goes (CPU stats) → (GPU correlation matrix) → (CPU regression)
    pays a transfer each hop. The trait-level cost model must see
    those edges; otherwise TamSession cannot make placement
    decisions.
18. **A18 — Op is hermetic.** `OpKind` requires `State: Send + Sync +
    'static`. But a GPU `WelfordState` is a device-side 24-byte
    struct; it Cannot be `Send` in Rust's sense (it's a GPU address).
    The Op IR must be **representation-independent**, with per-door
    representations chosen by the backend.
19. **A19 — one IR per Op.** Sweep 8 implies "lower Op to Cranelift
    IR." A GPU Op needs at least three lowerings: per-thread
    (lift+combine in registers), per-warp (shuffle reduction),
    per-workgroup (shared-memory tree reduction). The trait's
    `compile` cannot be "one Op → one kernel."
20. **A20 — no shape *specialization cascade*.** Same Op, different
    shape, different kernel: contiguous-stride vs strided; aligned
    vs unaligned; power-of-two vs odd N; small-N (register tile) vs
    large-N (block tile). Each is a separate compile under JIT. The
    cache key must cover all of these.
21. **A21 — no `using()` parameter flow at the trait level.** Sweep
    0/2 established `using()` as the override channel. Some
    `using()` values affect codegen (sum-strategy Kulisch vs fast,
    block-size, unroll factor). Others only affect dispatch
    (timeout, priority). The trait must partition these two classes.

---

## Phase 2 — Irreducible Truths

Strip the assumptions. What remains when every door-specific framing is
removed. These are the things that would be true of a 22nd-century door
we have not yet imagined.

- **T1 — IR must be representation-independent of the door.** Anything
  the trait exposes about "what the computation is" must be expressible
  *before* choosing a door. Tambear IR (atoms + Op/Expr + grouping +
  validity) already satisfies this; the trait must accept that IR
  *as data*, never as door-specific types.
- **T2 — The boundary between `tambear math` and `the ALU` is the only
  hardware-bound concept.** Memory layouts, descriptor sets, warp
  sizes, fences — these live on one side of the boundary. IR,
  shapes, structures, proofs live on the other.
- **T3 — Every door lowers the same IR to some binary form the driver
  accepts, and dispatches that binary against some buffers.** The
  *shape* of this is: `ir -> binary`, then `(binary, inputs) ->
  outputs`. That's 2 steps minimum.
- **T4 — Shape determines what binary is needed.** Two recipes, same
  IR, different shapes (N=10 vs N=10^9, dtype f64 vs f32, ByKey with
  K=4 vs K=4096) compile to different binaries. Shape is the cache key.
- **T5 — Binaries live in a cache.** Every door has a natural binary
  granularity (kernel object / pipeline object / module). The cache
  is BLAKE3-keyed on (IR × params × shape × validity × door ×
  target-capability).
- **T6 — Inputs and outputs are typed byte regions.** On CPU these are
  `&[T]` slices; on GPU they are driver handles to device memory.
  The trait sees them as **opaque handles with a byte size and a
  dtype tag**, not as slices.
- **T7 — Dispatch produces a completion condition.** Sync or async.
  CPU: returns when done. GPU: returns a fence/event/semaphore.
  The trait must admit both futures.
- **T8 — Sharing is a property of the cache, not the backend.** If
  backend A's cache sees a compatible entry, it uses it. TamSession's
  `IntermediateTag` machinery already expresses compatibility. The
  trait exposes a hook for "given this IR and shape, does your cache
  have an entry?" and "here's a computed intermediate, stash it
  under this tag."
- **T9 — Capability is a door property.** Every door has a version
  string (PTX ISA version, SPIR-V version, DXIL shader model, Metal
  GPU family, Cranelift ISA flags). The trait MUST expose
  capability so the cache key is honest and so codegen can feature-
  detect.
- **T10 — Errors fall into three bins that persist across all doors:**
  - *Compile errors* (bad IR, shape not supported, capability not
    met) — surface at compile time.
  - *Launch errors* (OOM, timeout, driver lost) — surface at dispatch.
  - *Validity errors* (NaN under `Validity::Error`) — surface at
    dispatch; same structural class on every door.
- **T11 — Buffers have ownership.** Allocation, deallocation, and
  cross-dispatch lifetime are first-class. CPU Rust owns via `Vec`;
  GPU doors own via driver handles. The trait exposes allocate,
  release, import (wrap host buffer), export (extract to host).
- **T12 — The compiler, the cache, and the dispatcher are THREE
  responsibilities.** CPU Cranelift + `Vec` makes them feel like one
  thing because the ALU and the allocator and the compiler all live
  in the same process. GPU makes them three. The trait must NOT
  conflate them.

Twelve truths. Everything else in Phase 1 was assumption.

---

## Phase 3 — Reconstruction from Zero (ten gradient approaches)

Building up from T1–T12 only. Gradient runs simple→structural. I have
NOT pre-screened for "feasibility" — that's Phase 5's job.

### R1 — Single object, blocking synchronous (**simple, CPU-shaped**)

```rust
trait DoorBackend {
    fn compile_and_run(&self, ir: TambearIr, shape: Shape,
                       inputs: &[&[f64]]) -> Vec<Vec<f64>>;
}
```
Fits CPU. Fails A1/A2/A3/A3b/A4/A5/A14. **Rejected as the trait, but
useful as a *convenience layer atop the real trait*.**

### R2 — Two-method compile+dispatch

```rust
trait DoorBackend {
    type Kernel;
    fn compile(&self, ir: TambearIr, shape: Shape) -> Self::Kernel;
    fn dispatch(&self, k: &Self::Kernel, inputs: &[&[f64]]) -> Vec<Vec<f64>>;
}
```
Sweep 8 README shape. Still fails A1 (sync), A2 (host buffers), A14
(no streams). CPU-adequate; GPU forces redesign.

### R3 — R2 + opaque buffer handles

```rust
trait DoorBackend {
    type Kernel; type Buffer; type Event;
    fn alloc(&self, bytes: usize, dtype: DType) -> Self::Buffer;
    fn write(&self, b: &mut Self::Buffer, data: &[u8]);
    fn read(&self, b: &Self::Buffer, out: &mut [u8]);
    fn compile(&self, ir: TambearIr, shape: Shape) -> Self::Kernel;
    fn dispatch(&self, k: &Self::Kernel, inputs: &[&Self::Buffer],
                outputs: &mut [&mut Self::Buffer]) -> Self::Event;
    fn wait(&self, e: Self::Event);
}
```
Fixes A1/A2/A4/A14. Still fails A3b (one-kernel-per-compile), A7
(cache key), A10 (scratch), A18 (per-door state reps).

### R4 — R3 + cache-hosted compile

```rust
trait DoorBackend {
    type Kernel; type Buffer; type Event;
    fn compile(&self, ir: &TambearIr, shape: &Shape, cache: &mut KernelCache)
        -> Arc<Self::Kernel>;
    // cache handles hit/miss; compile is always safe to call
    ...
}
```
`Arc<Kernel>` so multiple recipes reuse the compiled artifact.
`cache.compatible_entry(...)` is where TamSession sharing lives.

### R5 — R4 + capability queries + target triple

```rust
trait DoorBackend {
    fn capability(&self) -> DoorCapability;  // ISA version, FMA support,
                                              // denormal behavior, etc.
    ...
}
```
Capability participates in the cache key. Now PTX-for-sm_75 and
PTX-for-sm_90 are separate cache entries.

### R6 — R5 + the three-responsibility split

```rust
trait DoorCodegen {         // turns IR → binary
    fn lower(&self, ir: &TambearIr, shape: &Shape, cap: &DoorCapability)
        -> Result<Binary, CompileError>;
}

trait DoorCache {            // persists and retrieves binaries
    fn get(&self, key: &CacheKey) -> Option<Arc<Binary>>;
    fn put(&self, key: CacheKey, bin: Arc<Binary>);
}

trait DoorDispatcher {       // actually runs a binary against buffers
    type Buffer; type Event;
    fn alloc(&self, bytes: usize, dtype: DType) -> Self::Buffer;
    fn dispatch(&self, b: &Binary, inputs: &[&Self::Buffer],
                outputs: &mut [&mut Self::Buffer], scratch: ScratchSpec)
        -> Result<Self::Event, LaunchError>;
    fn wait(&self, e: Self::Event) -> Result<(), LaunchError>;
}

trait DoorBackend: DoorCodegen + DoorDispatcher {
    fn capability(&self) -> DoorCapability;
}
```
Matches T12. Each vendor sweep implements three traits, not one.
Cranelift CPU door implements all three trivially in one struct.

### R7 — R6 + parameterized over IR+State representations

```rust
trait DoorBackend: DoorCodegen + DoorDispatcher {
    type ScalarRepr: OpScalarRepr;   // how f64 state lives (register vs device)
    type StateRepr<S>: OpStateRepr<S>; // per-Op state rep
    fn capability(&self) -> DoorCapability;
}
```
Fixes A18. `WelfordState` has three representations: CPU-native
struct, SPIR-V `struct { uint n; float mean; float m2; }`, PTX
`.b32/.f32/.f32` reg triplet. The Op IR stays door-independent.

### R8 — R7 + scan-family hierarchy of lowerings

```rust
// An Op is not one kernel — it's a family of kernels, each for a
// different scan fan-out.
trait DoorCodegen {
    fn lower_scalar(&self, op: &OpIr, shape: &Shape, cap: &DoorCapability)
        -> Result<Binary, CompileError>;
    fn lower_warp_reduce(&self, op: &OpIr, shape: &Shape, cap: &DoorCapability)
        -> Result<Binary, CompileError>;
    fn lower_block_reduce(&self, op: &OpIr, shape: &Shape, cap: &DoorCapability)
        -> Result<Binary, CompileError>;
    fn lower_device_reduce(&self, op: &OpIr, shape: &Shape, cap: &DoorCapability)
        -> Result<Binary, CompileError>;
    // ... the dispatcher picks the right lowering and may compose them
}
```
Fixes A19. CPU implements only `lower_scalar`; GPU implements all
four and the dispatcher chains them across `cuLaunchKernel` edges.

### R9 — R8 + explicit stream/queue surface (multi-dispatch in flight)

```rust
trait DoorDispatcher {
    type Stream; type Event;
    fn default_stream(&self) -> Self::Stream;
    fn new_stream(&self) -> Self::Stream;     // optional; default impl forwards
    fn dispatch(&self, stream: &Self::Stream, b: &Binary, ...)
        -> Result<Self::Event, LaunchError>;
    fn stream_wait(&self, stream: &Self::Stream, e: Self::Event);
}
```
Fixes A14. CPU backend returns a trivial single-threaded `Stream`;
GPU backends expose real streams. TAM schedules across streams.

### R10 — The fully-structural reconstruction (**"structurally ambitious"**)

A single unified picture where every Phase-1 assumption has an explicit
trait axis.

```rust
// ========== IR layer (door-independent) ==========
struct TambearIr { atoms: Vec<AtomIr>, validity: Validity, ... }
enum AtomIr { Accumulate { grouping, expr, op }, Gather { addressing } }
struct Shape {
    dtype: DType,              // f64, f32, f16, bf16, i32, ...
    dim: SmallVec<[usize; 4]>, // logical extents
    strides: SmallVec<[isize; 4]>, // per-dim stride in elements
    grouping_shape: GroupingShape, // K for ByKey, window size, etc.
    assumption_tags: AssumptionBag, // monotone-keys, sorted, unique, etc.
    validity_tags: ValidityShape,   // known-no-NaN, has-inf, all-finite, ...
}

// ========== Capability (door tells us what it supports) ==========
struct DoorCapability {
    door_id: DoorId,
    isa_version: IsaVersion,
    driver_version: Version,
    subgroup_sizes: &'static [u32],  // 32 (NVIDIA), 64 (AMD GCN), 32 (CPU-SIMD)
    supports_fma: bool,
    denormal_mode: DenormalMode,
    max_shared_memory: u64,
    // ...
}

// ========== Cache key ==========
struct CacheKey {
    ir_hash: Blake3,
    param_hash: Blake3,
    shape_fingerprint: Blake3,
    validity_policy: Validity,
    door_id: DoorId,
    capability_fingerprint: Blake3, // isa + driver + feature flags
}

// ========== The trait triad ==========
trait DoorCodegen {
    fn lower(&self, ir: &TambearIr, shape: &Shape,
             cap: &DoorCapability) -> Result<CompiledArtifact, CompileError>;
}
struct CompiledArtifact {
    binary: Vec<u8>,             // PTX text, SPIR-V bytes, DXIL bytes, ELF, ...
    entry_points: Vec<EntryPoint>, // one per sub-kernel in a multi-kernel dispatch
    pipeline_metadata: PipelineMetadata, // descriptor sets, root sig equivalent
    scratch_required: ScratchSpec,      // per-invocation scratch bytes
    determinism: DeterminismClass,       // Deterministic | OrderDependent | NonDet
}

trait DoorCache {
    fn get(&self, key: &CacheKey) -> Option<Arc<Loaded>>;
    fn put(&self, key: CacheKey, art: CompiledArtifact) -> Arc<Loaded>;
    fn persist(&self); // flush to disk
}
struct Loaded { // door-specific loaded-kernel handle
    artifact: Arc<CompiledArtifact>,
    loaded: Arc<dyn Any + Send + Sync>, // CUfunction / VkPipeline / MTLFunction / ...
}

trait DoorDispatcher {
    type Buffer: DoorBuffer;
    type Stream: DoorStream;
    type Event: DoorEvent;

    fn alloc(&self, bytes: usize, dtype: DType, hint: AllocHint)
        -> Result<Self::Buffer, LaunchError>;
    fn import_host(&self, bytes: &[u8]) -> Result<Self::Buffer, LaunchError>;
    fn export_host(&self, b: &Self::Buffer, out: &mut [u8], e: &Self::Event)
        -> Result<(), LaunchError>;
    fn dispatch(&self, stream: &Self::Stream, k: &Loaded,
                inputs: &[&Self::Buffer], outputs: &mut [&mut Self::Buffer],
                workgroups: WorkgroupShape)
        -> Result<Self::Event, LaunchError>;
    fn wait(&self, e: Self::Event) -> Result<(), LaunchError>;

    fn default_stream(&self) -> Self::Stream;
    fn new_stream(&self) -> Result<Self::Stream, LaunchError>;
}

trait DoorBackend: DoorCodegen + DoorCache + DoorDispatcher {
    fn capability(&self) -> &DoorCapability;
}
```

This is what every Phase 1 assumption removal asks for. Scary shape.
But: CPU Cranelift door implements all three traits in one struct,
with `Stream = ()`, `Event = ()`, `Buffer = Vec<u8>`, `wait = no-op`,
`scratch = stack`. The trait absorbs CPU as a degenerate case of the
GPU case — not the other way around.

---

## Phase 4 — Assumption vs Truth Map

| # | Assumption (Phase 1) | Replacing truth (Phase 2) |
|---|---|---|
| A1 | Dispatch is blocking | T7: dispatch produces a completion condition (sync or async) |
| A2 | Pointers are buffers | T6: inputs/outputs are opaque typed byte regions |
| A3 | One kernel per compile | T3+T5: a compile produces a binary (possibly multi-kernel); cache stores binaries |
| A3b | `compile` returns one artifact | T3: a compile produces a CompiledArtifact — binary + entry points + pipeline metadata |
| A4 | Reads are memory loads | T7: completion condition gates readback |
| A5 | Float scalars are return type | T6: typed byte regions (on-device or on-host) |
| A6 | Host owns buffers | T11: buffers have first-class ownership exposed by the trait |
| A7 | Cache key = shape+params | T5+T9: cache key includes capability fingerprint + validity policy + door |
| A8 | Single-device implicit | T11+T2: dispatcher exposes device via streams/buffers; multi-device is stream composition, not new trait |
| A9 | Host allocates | T11: `alloc` is a door method |
| A10 | No scratch concept | T3: `CompiledArtifact::scratch_required: ScratchSpec` |
| A11 | Determinism is free | T3: `CompiledArtifact::determinism: DeterminismClass` is compile output |
| A12 | NaN/Inf uniform | T1: validity is part of IR; codegen enforces denormal-mode via compile flags |
| A13 | Kingdom A forever | T1: Op's structural facts flow into IR; tiebreak semantics are IR-declared |
| A14 | One execution stream | T7: streams are explicit; CPU is degenerate-single-stream |
| A15 | Error = panic | T10: three error classes (compile / launch / validity) |
| A16 | Inputs are f64 | T6: typed byte regions with DType tag (f16/f32/f64/bf16/i8/…) |
| A17 | No host/device edges | T11: buffers are door-local; cross-door transfer is an explicit recipe |
| A18 | Op state hermetic | T1: Op IR is representation-independent; per-door rep is codegen output |
| A19 | One IR per Op | T3: a CompiledArtifact can have multiple entry points (scalar + warp + block lowerings) |
| A20 | No shape cascade | T4: shape is in the cache key; specialization is automatic |
| A21 | No using() partition | T4: codegen-affecting `using()` values go in ParamHash; dispatch-affecting go in a separate channel (stream priority, timeout) |

All 21 assumptions have a truth that replaces them without adding new axioms.

---

## Phase 5 — The Aristotelian Move

Conventional analysis would read Sweep 8's `compile + dispatch` shape,
ship R2, and redesign when GPU lands. **That is the failure mode this
sweep exists to prevent.**

The Aristotelian move is **R10, with a CPU-collapse pattern**. Specifically:

**MOVE: ship R10's trait triad now (DoorCodegen + DoorCache +
DoorDispatcher + DoorBackend), with associated types that collapse to
trivial unit values on CPU.** The CPU Cranelift backend sets
`Stream = ()`, `Event = CompletionToken(())`, `Buffer = Vec<u8>` with
a dtype tag, `new_stream = Ok(())`, `wait = Ok(())`. The GPU backends
set them to real driver handles. Zero runtime cost on CPU (monomorphized
away), zero redesign cost when PTX lands.

Three concrete artifacts on commit:

1. `crates/tambear/src/jit/door.rs` with the trait triad, associated
   types, `CompiledArtifact`, `CompileError`, `LaunchError`,
   `DoorCapability`, `CacheKey`, and a `TambearIr` enum.
2. `crates/tambear/src/jit/cpu_cranelift.rs` with the full
   `DoorBackend` impl (stub codegen for Sweep 8B+; compiles but
   returns `Err(CompileError::NotYetImplemented)` until Sweep 8C
   lands real lowering).
3. `crates/tambear/src/jit/stub.rs` with a `StubBackend` that panics
   on codegen — for unit tests of the trait surface itself.

This is not a grand refactor. It's ~400 lines of trait declarations +
structs + error enums, and a CPU-collapse backend that degenerates to
`Vec<u8>` and `()`. Every door-specific sweep (17, 18, 19, 20) is
implement-the-trait work, not redesign-the-trait work.

---

## Phase 6 — Challenge Yourself (re-running with Phase 5's output in the assumption set)

Adding the Phase 5 move to the list of assumptions.

- **A22 — CPU collapse is free.** Is it? Monomorphizing associated
  types to `()` is a compile-time concept, but the trait object
  path (`Box<dyn DoorBackend>`) cannot collapse — it pays for
  vtable dispatch on every hop. Truth (T13): **the trait must be
  usable monomorphized AND as a trait object**. R10 uses associated
  types, which blocks trait-object use. We need a *companion
  object-safe trait* or type-erase the associated types.
- **A23 — CompiledArtifact is a blob.** Is it? PTX is text. DXIL is
  bytes. Cranelift produces raw machine code. Truth (T14): the
  CompiledArtifact must be **opaque to tambear** above the trait —
  only the backend interprets it. Keep it as `Vec<u8>` in the trait
  and let each backend cast it back.
- **A24 — Capability fingerprint is static.** CUDA Driver API can
  change behavior mid-program (e.g., CUDA lazy module loading, MPS
  multi-tenant). Capability may be per-device and per-context.
  Truth (T15): capability is a property of a *DoorBackend instance*,
  not of a *door family*. The CPU Cranelift instance on x86_64
  with AVX-512 has different capability than the same struct on
  aarch64 — and tambear must see both.
- **A25 — Stream is implicitly ordered.** CUDA streams are in-order;
  Vulkan command buffers plus queue submit are in-order per queue
  but multiple queues run in parallel. Truth (T16): stream = "an
  in-order sequence of dispatches." Independent streams are
  independent. This matches GPU reality and CPU thread-pool reality
  (a CPU stream is a thread).
- **A26 — Async event = fence.** CUDA events ≠ Vulkan fences ≠
  Metal fences exactly. Truth (T17): event = "an opaque handle the
  host can wait on that signals when a particular dispatch's
  outputs are ready." That abstraction works on every door.
- **A27 — Scratch is per-kernel.** But TamSession-fused kernels may
  reuse scratch across sub-kernel edges within one dispatch. Truth
  (T18): scratch is per-dispatch, declared by the compile, allocated
  by the dispatch, sized by the shape.
- **A28 — The IR hash is enough.** The IR hash covers the structural
  shape, but TamSession sharing happens at *intermediate* level,
  not whole-pipeline level. Truth (T19): cache keying operates at
  two levels — whole-pipeline (one compile per pipeline instance)
  AND per-intermediate (one entry per shareable `IntermediateTag`).
  `DoorCache` must expose both.

---

## Phase 7 — Recursive Process

Re-running Phase 6 with the seven new truths (T13–T19). After one more
pass:

- T13 resolves via two-trait pattern: `DoorBackend` (generic, monomorphized
  for performance) + `ErasedDoorBackend` (object-safe, for multi-door
  scheduling in TAM). Both delegate to the same `DoorCodegen` +
  `DoorCache` + `DoorDispatcher`.
- T14 is a refinement, not a structural change.
- T15 means `capability()` returns a reference that can be revalidated
  if the backend supports hot-reconfiguration. Most don't. Leave it
  as `&DoorCapability`.
- T16/T17 resolve the stream/event shape.
- T18 confirms `scratch_required` on `CompiledArtifact`.
- T19 adds a `DoorCache::{get_intermediate, put_intermediate}` method
  pair keyed by `IntermediateTag` from tambear-tam.

No new insights emerge on the next iteration. The trait shape stabilizes.

---

## Phase 8 — Forced Rejection

Forcibly reject R10. What would each absent piece imply?

- **What if CompiledArtifact had NO `determinism` field?** The
  dispatcher would have to ask every compile "did you produce a
  deterministic kernel?" out-of-band. Consumers couldn't reason
  about reproducibility. **Determinism MUST be carried structurally,
  not documented.** Confirms R10.
- **What if Shape didn't carry `assumption_tags`?** Then TamSession
  sharing couldn't check compatibility, and two consumers with
  different assumption fingerprints (say, Mahalanobis vs Euclidean
  distance) would collide. **Assumption tags MUST participate in
  the shape fingerprint.** Confirms R10.
- **What if ScratchSpec didn't exist?** Scratch would be sized
  conservatively (per-thread max) → massive over-allocation on GPU,
  and the kernel would have to runtime-check bounds. **Scratch MUST
  be compile-time declared.** Confirms R10.
- **What if `alloc` didn't take an `AllocHint`?** All allocations go
  to the default pool, but some recipes (streaming, one-shot,
  persistent) have different lifetime profiles. Without a hint,
  the dispatcher over-allocates or fails. **`AllocHint` IS needed**
  even though R10 barely mentioned it. ← *This was not explicit in
  R10; Phase 8 promotes it.*
- **What if streams didn't support "wait on another stream's event"?**
  Multi-stream pipelines where stream B starts after stream A
  finishes would require host CPU sync for every hand-off. **Cross-
  stream events MUST be supported.** Adds: `fn stream_wait_event(
  &self, stream: &Self::Stream, event: &Self::Event)`. ← *Added by
  Phase 8.*
- **What would it mean IF every Op had multiple equally-good
  lowerings?** (I.e., reject A19's single-kernel model even within
  one Op on one shape.) Then *which kernel to dispatch* is a
  runtime decision — the codegen produces a family, the dispatcher
  picks. This is R8's picture. **Policy for kernel-family selection
  MUST be part of `Loaded`**, not rediscovered at dispatch time.
  Adds: `fn select_kernel(&self, loaded: &Loaded, shape: &Shape)
  -> EntryPoint`. ← *Added by Phase 8.*
- **What if capability changes under the backend?** Driver update,
  hot-reconfiguration, runtime-lost device. Then cached binaries
  may be stale. The cache MUST key on capability fingerprint and
  revalidate on load. R10 has this.
- **What if the IR didn't fully constrain the output?** E.g.,
  `Op::ArgMax` on equal values has two valid answers. If the IR
  doesn't pin the tiebreak, parallelization breaks determinism.
  **The IR MUST be total** in the sense that every semantic choice
  is encoded in it. This is T1 restated.
- **What would it mean if the trait had MORE than three sub-traits?**
  E.g., pull out a DoorAlloc. The split at 3 (codegen / cache /
  dispatcher) matches the 3 responsibilities of T12. Splitting
  further fragments the impl without adding structure. Stays at 3.
- **What would it mean if the trait had FEWER than three sub-traits?**
  E.g., merge codegen into cache (cache calls codegen on miss). That
  couples eviction policy to codegen; cache grows dependencies.
  Stays at 3.
- **What must exist that we haven't named?** The ghost from Phase 8's
  void: **a profile-feedback channel.** Binaries run, produce
  timing, and feed back into kernel-family selection. CPU JIT has
  this (PGO). GPU drivers have this (persistent PSO cache with
  hotness scoring). Without it, we recompile the same binary every
  run, never improving. Adds: `fn record_timing(&self, k: &Loaded,
  entry: EntryPoint, wall_ns: u64)`. ← *Added by Phase 8; formerly
  invisible in R10.*

After Phase 8, the proposal becomes **R10 + AllocHint + cross-stream
waits + EntryPoint-selection policy + profile feedback**. Call this
R10′.

---

## Summary — the trait triad R10′

```rust
// IR (door-independent)
struct TambearIr { atoms: Vec<AtomIr>, validity: Validity, ... }
struct Shape { dtype, dims, strides, grouping_shape, assumption_tags, ... }
struct DoorCapability { door_id, isa_version, ... }
struct CacheKey { ir_hash, param_hash, shape_fingerprint,
                  validity_policy, door_id, capability_fingerprint }
struct CompiledArtifact { binary, entry_points, pipeline_metadata,
                          scratch_required, determinism }
enum CompileError { UnsupportedIr, UnsupportedShape, Capability, Codegen, ... }
enum LaunchError { OutOfMemory, Timeout, DeviceLost, Validity, ... }

trait DoorCodegen {
    fn lower(&self, ir: &TambearIr, shape: &Shape, cap: &DoorCapability)
        -> Result<CompiledArtifact, CompileError>;
}

trait DoorCache {
    fn get(&self, key: &CacheKey) -> Option<Arc<Loaded>>;
    fn put(&self, key: CacheKey, art: CompiledArtifact) -> Arc<Loaded>;
    fn get_intermediate(&self, tag: &IntermediateTag) -> Option<Arc<Loaded>>;
    fn put_intermediate(&self, tag: IntermediateTag, art: CompiledArtifact)
        -> Arc<Loaded>;
    fn persist(&self);
    fn record_timing(&self, k: &Loaded, entry: EntryPoint, wall_ns: u64);
}

trait DoorDispatcher {
    type Buffer: DoorBuffer;
    type Stream: DoorStream;
    type Event: DoorEvent;

    fn alloc(&self, bytes: usize, dtype: DType, hint: AllocHint)
        -> Result<Self::Buffer, LaunchError>;
    fn import_host(&self, bytes: &[u8]) -> Result<Self::Buffer, LaunchError>;
    fn export_host(&self, b: &Self::Buffer, out: &mut [u8], e: &Self::Event)
        -> Result<(), LaunchError>;

    fn dispatch(&self, stream: &Self::Stream, k: &Loaded,
                entry: EntryPoint,
                inputs: &[&Self::Buffer], outputs: &mut [&mut Self::Buffer],
                workgroups: WorkgroupShape, scratch: ScratchSpec)
        -> Result<Self::Event, LaunchError>;
    fn select_entry(&self, k: &Loaded, shape: &Shape) -> EntryPoint;

    fn wait(&self, e: Self::Event) -> Result<(), LaunchError>;
    fn default_stream(&self) -> Self::Stream;
    fn new_stream(&self) -> Result<Self::Stream, LaunchError>;
    fn stream_wait_event(&self, stream: &Self::Stream, event: &Self::Event)
        -> Result<(), LaunchError>;
}

trait DoorBackend: DoorCodegen + DoorCache + DoorDispatcher {
    fn capability(&self) -> &DoorCapability;
}

// Object-safe facade for TAM (multi-door scheduling):
trait ErasedDoorBackend: Send + Sync { /* type-erased */ ... }
```

CPU-collapse on Cranelift backend:
- `Buffer = HostBuffer { data: Vec<u8>, dtype: DType }`
- `Stream = ()`, `Event = HostEvent(())`
- `new_stream = Ok(())`, `wait = Ok(())`, `stream_wait_event = Ok(())`
- `scratch` = stack alloca at codegen time; `ScratchSpec` unused

Every existing `Send + Sync + 'static` bound on tambear OpKind state is
*at the IR level*, not the per-door state-rep level.

---

## Counter-cases to invite (for adversarial)

Asking adversarial to attack this shape with these specific scenarios:

1. **Multi-GPU fabric (NVIDIA NVLink, CXL).** Does `new_stream` on one
   device suffice, or does the trait need an explicit `Device`
   parameter? Draft answer: `Device` = a `DoorBackend` instance; one
   backend per device; cross-device transfer via explicit recipe.
2. **Apple MTLSharedEvent across processes.** Does the `Event`
   abstraction survive process boundaries? Probably no; cross-process
   sharing is out of scope for Sweep 8 but we should NOT design the
   trait to forbid it structurally.
3. **NPU with async hardware scheduler (e.g., AWS Inferentia, Google
   TPU).** NPUs may not expose streams — they compile graphs
   holistically. Does `dispatch(binary, inputs)` even fit? Draft
   answer: NPU graph compiler IS the codegen; the whole pipeline
   becomes one binary; `dispatch` is one call. Survives R10′.
4. **Shared virtual memory (CUDA UM, Metal unified memory on Apple
   Silicon).** Buffer handles become cheap on some doors. Does R10′
   force unnecessary copies? Draft answer: `import_host` can return
   a zero-copy wrapper on UM hardware; the trait doesn't forbid it.
5. **Denormals-are-zero as a CPU flag too.** x86 `MXCSR.DAZ/FTZ`,
   aarch64 `FPCR.FZ`. Does `DenormalMode` in capability make this
   clean on CPU? Yes — Cranelift ISA flag.
6. **Op::ArgMax with parallel reduction breaking lowest-index
   tiebreak.** Does the `DeterminismClass` in CompiledArtifact force
   codegen to produce a deterministic-tree kernel for this Op,
   rejecting a warp-shuffle kernel? Yes — codegen reads Op's
   `canonical_structure()`; missing Commutativity forces the tree
   path.

---

## Next moves

Invite pathmaker + adversarial review. Open questions for pathmaker:

- **Q1 for pathmaker** — is `Arc<Loaded>` acceptable with
  `Arc<dyn Any>` inside, or do you want an associated `Loaded`
  type too? Type-erased is simpler; typed is faster. Preference?
- **Q2 for pathmaker** — Cranelift's `JITModule` produces
  `*const u8` function pointers. Is wrapping that in a
  `CompiledArtifact::binary: Vec<u8>` awkward, or should
  `CompiledArtifact` be `enum { ModuleHandle, RawBinary }`?
- **Q3 for pathmaker** — does the CPU backend want real streams
  (thread-pool) for Sweep 8, or `Stream = ()` with a TODO for
  parallelization in a later sweep?

Open attacks for adversarial:

- **A1 for adversarial** — find a door where R10′ forces an
  unnatural mapping. NPU graph compilers are my top suspect;
  please test that scenario.
- **A2 for adversarial** — find an adversarial `Op` + Shape where
  `DeterminismClass` cannot be reduced to `{Deterministic,
  OrderDependent, NonDet}`. I suspect `Probabilistic` grouping with
  a learned RNG seed falls in a 4th bucket.
- **A3 for adversarial** — Validity::Error with async dispatch:
  where does the panic surface? At `wait()`? At readback? I say
  `wait()` returns `Err(LaunchError::Validity(details))`. Attack
  that.

---

*Phase 1-8 complete. Awaiting teammate feedback before pathmaker writes
the trait code.*
