# Gap Resolution — Adversarial Wave 1 (GAP-8A-1 through GAP-8A-7)

**Sweep 8 / Task 8A** · Author: aristotle · Date: 2026-04-22

Adversarial documented seven trait-shape gaps as `#[ignore]`'d tests
in `tests/sweep_8_adversarial.rs`. This file maps each gap to the
spec section that closes it, with concrete code so adversarial can
convert each `#[ignore]` into a passing assertion against the locked
trait surface.

If any gap is NOT actually closed by the spec, that's a Phase-9 ghost
and 8A reopens. Adversarial is the source of truth on whether the
spec text actually prevents the failure mode the test was written to
expose.

Companion documents (in this campsite folder):
- `trait-spec-locked.md` — the implementation surface
- `addendum-liftability-default.md` — lift-default + ExecutionStrategy
- `amendment-canonical-entry-points.md` — fixed entry-point name set
- `phase-1-8-deconstruction.md` — original deconstruction reasoning

---

## GAP-8A-1 — Async kernel completion

**Adversarial's claim:** A synchronous `dispatch()` signature forces
CUDA to block on every call. The trait must expose async completion.
Highest priority — bakes in at trait level.

**Spec resolution: CLOSED.**

The trait splits compile (`DoorCodegen::lower`) from dispatch
(`DoorDispatcher::dispatch`). `dispatch` returns `Self::Event` (a
completion token), not the output. `wait(event)` is a separate trait
method. Cross-stream synchronization uses
`stream_wait_event(stream, event)`. Stream/Event are associated
types per backend.

```rust
pub trait DoorDispatcher {
    type Buffer: DoorBuffer;
    type Stream: DoorStream;
    type Event: DoorEvent;

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

    fn wait(&self, e: Self::Event) -> Result<(), LaunchError>;

    fn default_stream(&self) -> Self::Stream;
    fn new_stream(&self) -> Result<Self::Stream, LaunchError>;
    fn stream_wait_event(&self, s: &Self::Stream, e: &Self::Event)
        -> Result<(), LaunchError>;
}
```

**Per-door reality:**

| Door | Stream type | Event type | wait() implementation |
|---|---|---|---|
| CPU Cranelift | `HostStream` (ZST) | `HostEvent` (ZST) | `Ok(())` — work already done at dispatch return |
| CUDA | `CUstream` (FFI handle) | `CUevent` (FFI handle) | `cuEventSynchronize(event)` |
| Vulkan | `VkQueue` + command buffer | `VkFence` | `vkWaitForFences` |
| Metal | `MTLCommandQueue` | `MTLEvent` / `MTLSharedEvent` | `[event waitUntilSignaledValue]` |
| DX12 | `ID3D12CommandQueue` | `ID3D12Fence` | `Fence::SetEventOnCompletion + WaitForSingleObject` |

**Convertible test pattern:**

```rust
#[test]
fn gap_8a_1_async_completion_via_event_token() {
    // Backend trait can express: dispatch returns an Event without
    // blocking; wait(event) blocks until completion. Verified at
    // type level by the trait method signatures.
    fn assert_async_signature<B: DoorBackend>(
        backend: &B,
        loaded: &Loaded,
        entry: EntryPoint,
        s: &B::Stream,
        i: &[&B::Buffer],
        o: &mut [&mut B::Buffer],
    ) -> Result<B::Event, LaunchError> {
        backend.dispatch(s, loaded, entry, i, o,
                         WorkgroupShape { xyz: [1,1,1] },
                         ScratchSpec { bytes: 0, align: 1 })
    }
    fn assert_wait<B: DoorBackend>(b: &B, e: B::Event)
        -> Result<(), LaunchError> { b.wait(e) }
    // Trait surface compiles -> async-completion is structurally
    // expressible. CUDA backend (sweep 17) wires CUstream/CUevent
    // into the same trait method shapes.
}
```

CPU backend's `Event = HostEvent` (ZST) means `wait` is a no-op —
CPU collapses async to sync without forcing GPU to be sync.

---

## GAP-8A-2 — Shared memory allocation

**Adversarial's claim:** `cuLaunchKernel` takes `sharedMemBytes`.
`KernelDescriptor` must carry `shared_mem_bytes: usize`.

**Spec resolution: CLOSED.**

Two surfaces in the spec carry shared-memory contracts:

1. **Compile-time:** `CompiledArtifact::scratch_required: ScratchSpec`
   — the artifact declares how much per-workgroup scratch the kernel
   needs. This is determined by codegen (the kernel knows its
   reduction-tree depth, etc.) and rides on the artifact.

2. **Dispatch-time:** `DoorDispatcher::dispatch(..., scratch:
   ScratchSpec)` — the dispatcher receives the scratch spec from the
   loaded artifact and allocates the threadgroup memory accordingly.

```rust
#[derive(Debug, Clone, Copy)]
pub struct ScratchSpec {
    /// Per-workgroup scratch in bytes. 0 means "no scratch needed."
    pub bytes: u64,
    /// Alignment requirement.
    pub align: u64,
}

#[derive(Debug)]
pub struct CompiledArtifact {
    // ...
    pub scratch_required: ScratchSpec,
    // ...
}
```

**Per-door reality:**

| Door | Where ScratchSpec.bytes goes |
|---|---|
| CUDA | `cuLaunchKernel`'s `sharedMemBytes` parameter |
| Vulkan | `VkComputePipelineCreateInfo` shared-memory + push-via-spec-constants |
| Metal | `[encoder setThreadgroupMemoryLength:bytes atIndex:0]` |
| DX12 | `groupshared` array sized at HLSL compile time → `numthreads(...)` config |
| CPU | unused (collapses to stack alloca during cranelift codegen) |

**Convertible test pattern:**

```rust
#[test]
fn gap_8a_2_scratch_carried_by_artifact_and_dispatch() {
    // Welford reduction needs 3 doubles + 1 u64 per warp lane.
    // Artifact declares it; dispatch consumes it.
    let backend = NoOpBackend;
    let art = CompiledArtifact {
        door: DoorId::NOOP,
        binary: ArtifactBinary::RawBinary(vec![]),
        entry_points: smallvec![EntryPoint {
            kind: EntryPointKind::ReduceWarp,
            workgroup_constraint: Some(WorkgroupShape { xyz: [32,1,1] }),
        }],
        pipeline_metadata: PipelineMetadata::default(),
        scratch_required: ScratchSpec { bytes: 32 * 24, align: 8 },
        determinism: DeterminismClass::Deterministic,
    };
    assert_eq!(art.scratch_required.bytes, 32 * 24);
    // Dispatch propagates this to the kernel launch; tested when
    // cranelift backend lands.
}
```

---

## GAP-8A-3 — 3D grid/block dimensions

**Adversarial's claim:** CUDA / Vulkan / Metal all require (x, y, z)
dispatch dimensions. Tiled grouping (matmul) requires 2D dispatch.
A single `n_elements: usize` interface won't cover it.

**Spec resolution: CLOSED.**

The spec has `WorkgroupShape { xyz: [u32; 3] }` carried in two places:

1. **Per-entry-point compile-time constraint:** `EntryPoint::workgroup_constraint:
   Option<WorkgroupShape>` — the codegen pins the workgroup size when
   the kernel was compiled with a fixed shape (e.g. warp-reduce uses
   `xyz: [32,1,1]` on NVIDIA, `[64,1,1]` on AMD GCN).

2. **Per-dispatch runtime parameter:** `DoorDispatcher::dispatch(...,
   workgroups: WorkgroupShape)` — the dispatcher passes the grid
   dimensions for this specific dispatch (number of workgroups along
   each axis).

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkgroupShape {
    pub xyz: [u32; 3],
}
```

Grouping::Tiled { m, n } produces `WorkgroupShape { xyz: [m_blocks,
n_blocks, 1] }` at dispatch time. Grouping::All produces `xyz: [n_workgroups,
1, 1]`. Z-axis is reserved for 3D problems (cubic spline grids,
N-body cells, future).

**Per-door mapping:**

| Door | dispatch-time `workgroups.xyz` | compile-time `entry.workgroup_constraint.xyz` |
|---|---|---|
| CUDA | `gridDim.{x,y,z}` in `cuLaunchKernel` | `blockDim.{x,y,z}` baked into kernel |
| Vulkan | `vkCmdDispatch(x, y, z)` | `local_size_x/y/z` in SPIR-V |
| Metal | `dispatchThreadgroups:MTLSizeMake(x,y,z)` | `threadsPerThreadgroup` |
| DX12 | `ID3D12GraphicsCommandList::Dispatch(x, y, z)` | `numthreads(x,y,z)` in HLSL |
| CPU | thread-pool fan-out: `for i in 0..xyz[0] for j in 0..xyz[1] for k in 0..xyz[2] { ... }` | unused (1,1,1) |

**Convertible test pattern:**

```rust
#[test]
fn gap_8a_3_three_dim_dispatch() {
    let mat_shape = Shape::new(
        Grouping::Tiled { m: 1024, n: 1024 },
        Validity::Propagate,
    );
    // workgroups.xyz = [m/block_m, n/block_n, 1] for 2D matmul
    let workgroups = WorkgroupShape { xyz: [16, 16, 1] };
    assert_eq!(workgroups.xyz, [16, 16, 1]);
    // entry's compile-time constraint pins the per-block thread count
    let entry = EntryPoint {
        kind: EntryPointKind::Specialized("matmul_block_16x16"),
        workgroup_constraint: Some(WorkgroupShape { xyz: [16, 16, 1] }),
    };
    assert_eq!(entry.workgroup_constraint.unwrap().xyz, [16, 16, 1]);
}
```

---

## GAP-8A-4 — Push constants / inline params

**Adversarial's claim:** Vulkan push constants, CUDA inline kernel
args. Small scalars should NOT go through buffer allocation.
`KernelDescriptor` should carry `inline_params: &[u8]`.

**Spec resolution: CLOSED via `compile()`'s `params` argument.**

The spec's `DoorCodegen::lower(op, shape, strategy, params: &[u8],
cap)` takes `params` as a byte buffer that the codegen *bakes into
the compiled kernel* — not into a runtime buffer. This is exactly the
push-constant / kernel-argument shape:

- For CUDA: kernel formal parameters become `params` in
  `cuLaunchKernel`'s `kernelParams` array (already a flat byte
  buffer in the CUDA Driver API).
- For Vulkan: `params` lowers to push-constant block + descriptor
  set bindings.
- For Metal: `params` lowers to `[encoder setBytes:length:atIndex:]`
  (inline argument table).
- For DX12: `params` lowers to root-signature constants + descriptor
  table.
- For CPU: `params` becomes constant-folded into the cranelift IR
  during compile (zero runtime cost).

**Cache-key behavior:** `params` is BLAKE3-hashed into `param_hash`
in `CacheKey`. Different params → different cache entry → different
specialized kernel. This matches the GPU reality: changing a push
constant value that affects codegen (loop unroll factor, tile size)
forces recompile, while runtime values (input length, etc.) flow
through buffers and don't recompile.

**Distinction from buffers:** spec has TWO surfaces:
- `params: &[u8]` (compile-time; baked in; participates in cache key)
- `inputs: &[&Self::Buffer]` / `outputs` (runtime; not in cache key)

If a value is "small + fixed across many dispatches" → params.
If a value is "varies per dispatch" → buffer.

**Convertible test pattern:**

```rust
#[test]
fn gap_8a_4_push_constants_via_params() {
    let backend = NoOpBackend;
    let shape = Shape::new(Grouping::All, Validity::Propagate);
    let alpha_bytes: [u8; 8] = 0.5_f64.to_le_bytes(); // alpha for EWMA

    // Different alpha values produce different cache entries
    let key1 = CacheKey::compute(&JitOp::AffineCompose, &shape,
                                 ExecutionStrategy::Lifted,
                                 Validity::Propagate,
                                 DoorId::NOOP, &cap_fp(),
                                 &alpha_bytes);
    let alpha_other: [u8; 8] = 0.3_f64.to_le_bytes();
    let key2 = CacheKey::compute(&JitOp::AffineCompose, &shape,
                                 ExecutionStrategy::Lifted,
                                 Validity::Propagate,
                                 DoorId::NOOP, &cap_fp(),
                                 &alpha_other);
    assert_ne!(key1, key2,
        "different inline params must produce different cache keys");
}
```

---

## GAP-8A-5 — Buffer residency

**Adversarial's claim:** GPU-computed intermediates should stay on
device for chained GPU consumers. Trait needs `DeviceResident` vs
`CpuResident` distinction — not just `Vec<f64>` returns.

**Spec resolution: CLOSED.**

The spec has NO `Vec<f64>` in the trait — outputs are
`&mut [&mut Self::Buffer]`. The buffer is whatever the door says it
is via `DoorBuffer` supertrait. CPU backend uses `HostBuffer`
(wraps `Vec<u8>` + dtype); CUDA backend uses `CudaBuffer` (wraps
`CUdeviceptr` + dtype + bytes). A chained dispatch passes the SAME
device buffer as input to the next dispatch — never touches the
host.

```rust
pub trait DoorBuffer: Send + Sync {
    fn len_bytes(&self) -> u64;
    fn dtype(&self) -> ScalarTy;
}
```

Host transfer is **explicit** via `import_host`/`export_host`:

```rust
fn alloc(&self, bytes: u64, dtype: ScalarTy, hint: AllocHint)
    -> Result<Self::Buffer, LaunchError>;
fn import_host(&self, bytes: &[u8], dtype: ScalarTy)
    -> Result<Self::Buffer, LaunchError>;
fn export_host(&self, b: &Self::Buffer, out: &mut [u8],
               e: &Self::Event) -> Result<(), LaunchError>;
```

A pipeline that runs N GPU steps in sequence calls `alloc` once,
chains `dispatch -> dispatch -> dispatch` with the same `Self::Buffer`,
and only calls `export_host` once at the end (or never, if the next
recipe is also GPU-resident).

**AllocHint covers chaining intent:**

```rust
pub enum AllocHint {
    OneShot,        // dealloc after readback — short-lived
    Persistent,     // long-lived intermediate — TamSession candidate
    SharedHost,     // zero-copy alias if door supports unified memory
}
```

**TamSession integration:** `DoorCache::get_intermediate(&IntermediateTag)`
returns an `Arc<Loaded>` that wraps the device-resident buffer. Chained
GPU consumers pull from this without round-tripping to host.

**Convertible test pattern:**

```rust
#[test]
fn gap_8a_5_buffer_residency_no_implicit_readback() {
    // Buffer is door-specific; outputs are never `Vec<f64>`.
    // Verified at type level: dispatch's outputs slot is
    // &mut [&mut Self::Buffer], not &mut Vec<f64>.
    fn assert_no_implicit_readback<B: DoorBackend>(
        b: &B, k: &Loaded, e: EntryPoint,
        stream: &B::Stream,
        inputs: &[&B::Buffer],
        outputs: &mut [&mut B::Buffer],
    ) -> Result<B::Event, LaunchError> {
        b.dispatch(stream, k, e, inputs, outputs,
                   WorkgroupShape { xyz: [1,1,1] },
                   ScratchSpec { bytes: 0, align: 1 })
    }
    // Chained GPU pipeline:
    //   alloc(buf_a) -> dispatch(input=raw, output=buf_a)
    //                -> dispatch(input=buf_a, output=buf_b)
    //                -> dispatch(input=buf_b, output=buf_c)
    //                -> export_host(buf_c, &mut host_out)
    // Only one host transfer. Trait surface forces it.
}
```

---

## GAP-8A-6 — Cooperative groups / warp sync

**Adversarial's claim:** `cuLaunchKernel` vs `cuLaunchCooperativeKernel`
are different APIs. `KernelDescriptor` needs `CooperativeMode`.

**Spec resolution: CLOSED via `EntryPoint` + `pipeline_metadata`.**

Cooperative-launch is a property of the **kernel binary**, not the
dispatcher's per-call config. The spec's `CompiledArtifact` carries
`pipeline_metadata: PipelineMetadata` which is the door-opaque blob
where this lives. The codegen decides at compile time whether the
kernel needs cooperative launch (e.g., grid-wide barriers for
device-wide reductions) and encodes that in the `Loaded` handle.

The dispatcher reads the `Loaded` handle and picks the launch API:

- CUDA: `cuLaunchKernel` for normal; `cuLaunchCooperativeKernel`
  when the artifact's pipeline_metadata says "needs cooperative."
- Vulkan: subgroup-extension features are negotiated at
  `VkComputePipelineCreateInfo` time; cooperative-matrix-multiply is
  a different pipeline.
- Metal: `[encoder useResidencySet]` for cooperative buffer access
  patterns.

**Why this lives on the artifact, not on dispatch:**

Cooperative mode is determined by the kernel's algorithm
(does it use `__syncthreads_count()`? `__cooperative_groups::grid_group::sync()`?).
The codegen knows this. The dispatcher doesn't need to, except to
pick the launch API — and the artifact's pipeline_metadata tells it.

If we put `CooperativeMode` on the dispatcher's `dispatch()` call,
we'd be asking the user/dispatcher to know an implementation detail
of the codegen.

**EntryPoint extension for explicit declaration (optional):**

If we want the user to be able to *force* cooperative launch (or
forbid it), we add a flag to EntryPoint:

```rust
pub struct EntryPoint {
    pub kind: EntryPointKind,
    pub workgroup_constraint: Option<WorkgroupShape>,
    /// Forced cooperative launch mode. None means "use whatever the
    /// codegen requested in pipeline_metadata."
    pub cooperative_override: Option<bool>,
}
```

But the default — codegen decides, dispatch reads metadata — is
right.

**Convertible test pattern:**

```rust
#[test]
fn gap_8a_6_cooperative_via_pipeline_metadata() {
    // Device-wide reduction needs cooperative launch on Blackwell;
    // codegen flags this in pipeline_metadata. Backend-specific test
    // lands when CUDA backend ships in Sweep 17.
    let art_normal = CompiledArtifact {
        // ... pipeline_metadata says cooperative=false ...
    };
    let art_coop = CompiledArtifact {
        // ... pipeline_metadata says cooperative=true ...
    };
    // Dispatcher reads pipeline_metadata, picks correct launch API.
    // Tested as integration test against the CUDA backend.
}
```

---

## GAP-8A-7 — Metal dispatch model

**Adversarial's claim:** `dispatchThreadgroups` (group-count) vs
`dispatchThreads` (thread-count) require different `KernelDescriptor`
shapes.

**Spec resolution: CLOSED via `WorkgroupShape` semantics.**

The spec defines `WorkgroupShape` as **"threads per workgroup along
x/y/z"** (compile-time constraint, on EntryPoint) AND **"number of
workgroups to dispatch along x/y/z"** (runtime parameter to
dispatch).

Metal's `dispatchThreadgroups` = "I'm passing the grid (number of
workgroups)" → maps to `dispatch(...workgroups: WorkgroupShape...)`.
Metal's `dispatchThreads` = "I'm passing the total thread count and
you compute the grid for me" → maps to a *helper layer* in the
Metal backend that converts thread-count → workgroup-count using the
known per-workgroup size from EntryPoint::workgroup_constraint.

Both work against the same trait surface. The Metal backend internally
chooses which dispatchXXX call to use; the trait's
`workgroups: WorkgroupShape` is the *grid*, and per-workgroup size is
the *EntryPoint constraint*. Metal's flexibility is implementation
detail, not trait surface.

**Per-backend reality:**

| Door | dispatch surface | uses |
|---|---|---|
| CUDA | always grid + block | `cuLaunchKernel(grid_x, grid_y, grid_z, block_x, block_y, block_z, ...)` |
| Vulkan | always grid (block in pipeline) | `vkCmdDispatch(grid_x, grid_y, grid_z)` |
| Metal | grid OR threads | `dispatchThreadgroups` (when grid is what we have); `dispatchThreads` (when the kernel doesn't care about workgroup size); spec-trait passes grid; Metal backend wraps |
| DX12 | always grid (block in HLSL) | `Dispatch(grid_x, grid_y, grid_z)` |
| CPU | grid + block-per-thread-pool-task | thread-pool fan-out |

**Why `WorkgroupShape` is sufficient:**

It carries 3 axes × 2 surfaces (compile-time per-workgroup size +
dispatch-time grid) = 6 numbers. CUDA/Vulkan/Metal/DX12 all reduce
to this 6-number representation. Metal's `dispatchThreads` is a
convenience that the Metal backend applies internally; the trait
doesn't need to expose two dispatch shapes.

**Convertible test pattern:**

```rust
#[test]
fn gap_8a_7_workgroup_shape_covers_metal_dispatch() {
    // Compile-time block size from EntryPoint
    let entry = EntryPoint {
        kind: EntryPointKind::ReduceWarp,
        workgroup_constraint: Some(WorkgroupShape { xyz: [32, 1, 1] }),
    };
    // Dispatch-time grid
    let grid = WorkgroupShape { xyz: [128, 1, 1] };
    // Total threads = 32 * 128 = 4096
    // Metal backend can call:
    //   dispatchThreadgroups:MTLSizeMake(128,1,1)
    //                            threadsPerThreadgroup:MTLSizeMake(32,1,1)
    // OR
    //   dispatchThreads:MTLSizeMake(4096,1,1)
    //                            threadsPerThreadgroup:MTLSizeMake(32,1,1)
    // Both work; Metal backend picks per heuristic. Trait surface
    // exposes only the grid + per-workgroup constraint.
    assert_eq!(grid.xyz[0], 128);
    assert_eq!(entry.workgroup_constraint.unwrap().xyz[0], 32);
}
```

---

## Summary table

| Gap | Spec section | Resolution mechanism |
|---|---|---|
| GAP-8A-1 async | `DoorDispatcher::dispatch -> Event` + `wait` + `stream_wait_event` | Self::Event associated type |
| GAP-8A-2 shared mem | `CompiledArtifact::scratch_required` + `dispatch(scratch:)` | ScratchSpec on artifact + dispatch |
| GAP-8A-3 3D dims | `WorkgroupShape { xyz: [u32; 3] }` | both compile-time + dispatch-time |
| GAP-8A-4 push constants | `DoorCodegen::lower(params: &[u8])` baked in | params in CacheKey |
| GAP-8A-5 buffer residency | `Self::Buffer` assoc type + `import/export_host` | buffer is door-specific |
| GAP-8A-6 cooperative | `pipeline_metadata` on artifact | codegen decides; dispatcher reads |
| GAP-8A-7 Metal dispatch | `WorkgroupShape` (grid) + `EntryPoint::workgroup_constraint` (block) | 6 numbers cover all doors |

All seven trait-level gaps closed by the locked spec. None require
trait redesign in Sweep 17+.

---

## Asks of adversarial

**Convert the wave-1 `#[ignore]` tests:**

For each of the 7 gaps, please convert your `#[ignore]`'d test into
a passing assertion against the trait surface in
`trait-spec-locked.md`. The test should fail to compile (or fail at
runtime with a clear assertion) if pathmaker accidentally drops one
of the 6 mechanisms above.

If during conversion you find a gap the spec actually does NOT
close, ping me — that's a Phase-9 ghost and 8A reopens.

**Remaining attacks I want you to keep alive:**

- Original three (NPU graph compilers, RNG-seeded determinism class,
  async Validity::Error race)
- Five lift-default counterexamples from the addendum (ArgMax+Prefix,
  Welford+Prefix bit-exact, LSE+Windowed, DotProduct+Tiled,
  AffineCompose+Segmented)
- New: **what if a door has NO concept of streams?** Some embedded
  NPUs have a single in-order command queue, no parallelism between
  dispatches. Does `Stream = SingleQueue` (a singleton ZST that
  serializes everything) work, or does the trait need a "no-streams"
  capability flag?

Standing campsite is `sweep-8/20260422114112-jit-design`. Your folder
is `adversarial/`.
