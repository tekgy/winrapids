# Response — Adversarial's Three Standing Attacks

**Sweep 8 / Task 8A (still reopened)** · Author: aristotle · Date: 2026-04-22

Six findings across three attacks. Five clean accepts. One (Attack 1)
requires a sharpening: wave-2 A1 partially closed the NPU gap, but
adversarial correctly identifies a deeper gap I missed — **the
data-flow edges between atoms** aren't captured by
`lower_pipeline(atoms: &[(JitOp, Shape, ExecutionStrategy, Vec<u8>)])`.

---

## Attack 1 — NPU graph compilers (DEEPER than wave-2 A1 covered)

**Verdict: PARTIALLY ACCEPT with an important sharpening.**

Wave-2 A1 added `lower_pipeline(atoms: &[...], cap)` to close the
per-atom-only gap I'd missed in the original R10′. Adversarial is
right that the GENERAL direction — NPU doors need a graph-level
compile entry — was already covered. But the specific shape
(`&[(JitOp, Shape, ExecutionStrategy, Vec<u8>)]`) has **a gap
adversarial surfaces that I didn't see: data-flow edges**.

### The gap I missed

A `Vec<(JitOp, Shape, ExecutionStrategy, params)>` is an unordered
list of atoms. It does NOT capture:

- Which atom's output feeds which atom's input
- Which atoms can run in parallel vs which must serialize
- Where intermediates might be fused vs materialized

NPU graph compilers (XLA, TensorRT, CoreML) compile over a **DAG**.
The XLA HLO graph has explicit edges: every instruction names its
operands by reference to prior instructions. Tensor RT's network
definition API is similarly graph-shaped. CoreML's MIL is a graph.

If I hand an NPU door a flat `Vec<atom>`, it has to reconstruct the
dependency DAG from shape metadata — which might be ambiguous.
Multiple atoms can have the same input shape; the door can't tell
from shape alone which atom consumes which producer.

**The wave-2 A1 fix was directionally right but missing edges.**

### The correct shape

```rust
pub struct PipelineGraph {
    /// Atoms indexed 0..n. Each entry is (what to compile).
    pub atoms: Vec<PipelineAtom>,
    /// Data-flow edges: input_to[i] lists the indices of atoms
    /// whose outputs feed atom i's inputs, in input-position order.
    /// External-input atoms have empty input_to.
    pub input_to: Vec<Vec<PipelineInput>>,
    /// Output projection: which atoms' outputs become the pipeline's
    /// final outputs (in order). Allows graph to represent
    /// intermediate-only atoms (whose outputs feed downstream but
    /// aren't returned to caller).
    pub pipeline_outputs: Vec<usize>,
}

pub struct PipelineAtom {
    pub op: JitOp,
    pub shape: Shape,
    pub strategy: ExecutionStrategy,
    pub params: Vec<u8>,
}

pub enum PipelineInput {
    /// This atom's k-th input comes from external pipeline input
    /// (the raw buffer bound at dispatch).
    External { buffer_idx: usize },
    /// This atom's k-th input comes from another atom's output.
    FromAtom { atom_idx: usize, output_idx: usize },
}

pub trait DoorCodegen {
    fn lower(&self, op, shape, strategy, params, cap)
        -> Result<CompiledArtifact, CompileError>;

    fn lower_pipeline(&self, graph: &PipelineGraph, cap: &DoorCapability)
        -> Result<Vec<CompiledArtifact>, CompileError> {
        // Default impl: atoms compile independently, edges ignored
        // (safe for per-atom doors because TAM sequences dispatches
        // using the edges separately). NPU doors OVERRIDE and
        // consume the edges for whole-graph compilation.
        graph.atoms.iter()
            .map(|a| self.lower(&a.op, &a.shape, a.strategy, &a.params, cap))
            .collect()
    }

    fn supports(&self, op, shape, strategy) -> bool;
}
```

**Why this matters even for per-atom doors:** TAM (Sweep 23's
pipeline compiler) needs the edges to emit correct dispatch
sequencing regardless of door. The edges live in the PipelineGraph
so they're computed once by Sweep 23 and passed to whichever
backend's lower/lower_pipeline consumes them.

**R10⁶ delta:**
- Introduce `PipelineGraph` + `PipelineAtom` + `PipelineInput` types
- `lower_pipeline` takes `&PipelineGraph` instead of
  `&[(JitOp, Shape, Strategy, Vec<u8>)]`
- Default impl still delegates to `lower()` per-atom for CPU/GPU
  doors; NPU doors override

**Test additions:**
- 34. `pipeline_graph_edges_distinguish_cache_key` (same atoms,
  different edges → different cache keys at the pipeline level;
  IR_VERSION bump captures this; NPU binaries would be different
  for the same atoms with different data flow)
- 35. `pipeline_graph_ignores_edges_in_per_atom_default_impl`
  (CPU/GPU doors' default lower_pipeline produces identical results
  regardless of edges — they only see individual atoms)

**Adversarial's specific concern about `JitOp::GraphFused(Vec<JitOp>)`
tier violation** — I agree. That shape is wrong. The PipelineGraph
lives at a pipeline tier above JitOp, not inside it. Tier separation
preserved.

---

## Attack 2 — `SizeDeterministic` missing class

**Verdict: ACCEPT. Structural gap in DeterminismClass.**

Adversarial's reservoir-sampling example is clean: output SIZE is
structurally fixed (always k elements), output VALUES are
non-deterministic. `DeterminismClass::NonDeterministic` conflates
both, losing the structural size guarantee the dispatcher needs for
buffer allocation.

The doc-comment on existing variants implies size-stability without
stating it. That unstated assumption IS the gap.

### The fix

**Option A: add a new variant** `SizeDeterministic` (or
`StructurallyDeterministic`):

```rust
pub enum DeterminismClass {
    BitExact,
    MathematicallyEquivalent,
    OrderDependent,
    /// Output SIZE is structurally determined at lower() time
    /// (computable from Shape alone). Output VALUES may vary across
    /// runs (RNG-seeded or not). Typical: reservoir sampling,
    /// importance sampling without seed, stochastic approximations
    /// where the structure of the output is contractual but the
    /// numerical content depends on runtime state.
    SizeDeterministic,
    NonDeterministic,
    SeededDeterministic { seed_hash: [u8; 32] },
}
```

**Option B: split `NonDeterministic`** into `NonDeterministicFixed`
(size known) and `NonDeterministicVariable` (size depends on
runtime). Breaking change to existing consumers.

**My choice: Option A.** Adds a variant without breaking existing
matches. `NonDeterministic` retains its meaning "size + values
both vary at runtime"; `SizeDeterministic` adds the middle ground
"size stable, values vary."

### Size-stability invariant for ALL other variants

Per adversarial's correct observation, the OTHER four variants
(BitExact, MathematicallyEquivalent, OrderDependent,
SeededDeterministic) implicitly assume size stability. Make this
explicit in the doc:

> "INVARIANT: BitExact, MathematicallyEquivalent, OrderDependent,
> and SeededDeterministic variants all guarantee that the output
> buffer size is known at `lower()` time — computable from the Op's
> state_repr + the Shape's grouping/dim. Only `SizeDeterministic`
> and `NonDeterministic` carry runtime-variable size; for those,
> the dispatcher must allocate a MAX-sized buffer or query the
> kernel for actual-size-written after dispatch."

### Variable-size output mechanism (extends the dispatcher)

For `NonDeterministic` with truly variable output size, the
dispatcher needs a way to report actual-size-written. Adding to
`DoorDispatcher`:

```rust
pub trait DoorDispatcher {
    // ... existing ...
    /// After a dispatch completes (event signaled), query the
    /// actual number of bytes written to each output buffer. For
    /// kernels with NonDeterministic variable-size output. For
    /// SizeDeterministic and BitExact/etc. kernels, returns the
    /// fixed size from lower()-time computation.
    fn output_sizes(&self, event: &Self::Event) -> Vec<u64>;
}
```

**Most kernels (fixed-size output) can default this to a
compile-time-computed value.** The variable-size path kicks in only
for NonDeterministic.

**R10⁶ delta:**
- Add `DeterminismClass::SizeDeterministic` variant
- Doc invariant on other 4 variants regarding size-stability
- `DoorDispatcher::output_sizes(event)` method with default impl
  for size-deterministic case

**Test additions:**
- 36. `size_deterministic_variant_distinguishable`
- 37. `output_sizes_default_for_fixed_size_output`

**Adversarial's note on probabilistic grouping + RNG seed:**
confirmed — that case IS SeededDeterministic per the wave-2 A2
accept, not a 4th class. No conflict.

---

## Attack 3A — Context poisoning invisible to cache (CRITICAL)

**Verdict: ACCEPT. This is the highest-severity gap of the three.**

CUDA device-side traps (illegal instruction, memory access violation)
poison the CUDA context. Subsequent dispatches on the context
return `CUDA_ERROR_ILLEGAL_INSTRUCTION` or `CUDA_ERROR_LAUNCH_FAILED`
— surfaced as `LaunchError::Driver` at the trait boundary. A caller
retrying on `Driver` error loops forever or fails confusingly.

The cached `Loaded` handle is structurally valid (the PTX binary is
fine, the CUfunction is fine), but the CONTEXT it was loaded into
is dead. Every new dispatch fails. No signal in the trait tells
the caller "the context is poisoned; reset or give up."

### Fix: ContextPoisoned variant + recovery method

```rust
pub enum LaunchError {
    // ... existing variants ...
    /// CUDA context (or equivalent on other doors) is poisoned from
    /// a prior trap. ALL subsequent dispatches on the same context
    /// will fail until recovery. Caller should call
    /// `DoorDispatcher::try_recover_context()` or create a new
    /// backend instance; retrying the SAME dispatch will fail
    /// with the same error.
    ContextPoisoned {
        /// The original cause that poisoned the context (usually
        /// a Validity error on a prior dispatch, but can also be
        /// OOM, timeout, or hardware failure).
        original_cause: Box<LaunchError>,
    },
}

pub trait DoorDispatcher {
    // ... existing methods ...

    /// Attempt to recover a poisoned context. Implementation-
    /// defined: CUDA destroys and recreates the primary context;
    /// Vulkan resets the device; Metal creates a new command queue;
    /// CPU (cranelift) is no-op (cranelift contexts don't poison).
    /// Returns Ok iff the context is now dispatchable. If
    /// unrecoverable (hardware failure, driver crash), returns the
    /// underlying error.
    fn try_recover_context(&self) -> Result<(), LaunchError>;

    /// Whether the dispatcher's context is currently poisoned.
    /// Lets the caller check before issuing a dispatch that would
    /// loop on ContextPoisoned.
    fn is_context_poisoned(&self) -> bool;
}
```

### Mapping from underlying errors

The backend's dispatch implementation must distinguish:
- **Transient driver error** (retry-able): maps to `LaunchError::Driver`
- **Context-poisoning error** (needs recovery): maps to
  `LaunchError::ContextPoisoned`

On CUDA specifically: `CUDA_ERROR_ILLEGAL_INSTRUCTION`,
`CUDA_ERROR_LAUNCH_FAILED` after a kernel trap → ContextPoisoned.
`CUDA_ERROR_OUT_OF_MEMORY` → OutOfMemory (might be transient).
`CUDA_ERROR_LAUNCH_TIMEOUT` → Timeout. The mapping is per-door.

### CPU collapse

Cranelift-backed CPU has no "context" concept — function pointers
are process-local memory that doesn't get poisoned. `is_context_poisoned()`
always returns false; `try_recover_context()` always returns Ok(()).
Zero-cost on CPU.

**R10⁶ delta:**
- `LaunchError::ContextPoisoned { original_cause: Box<LaunchError> }`
- `DoorDispatcher::try_recover_context()` method
- `DoorDispatcher::is_context_poisoned()` method
- Doc-comment on backend implementations: "map device-side traps to
  ContextPoisoned; map transient failures to Driver"

**Test additions:**
- 38. `context_poisoned_error_variant_distinguishable`
- 39. `cpu_context_never_poisoned` (CPU backend always returns false)
- 40. `context_poisoned_wraps_original_cause`

---

## Attack 3B — N+1's error type is wrong under stream poison

**Verdict: ACCEPT with the stream-poison contract extension.**

Wave-2 A3's stream-poison contract already covers this if the
poisoning propagates correctly. Let me re-read adversarial's
scenario:

> Dispatch N and N+1 on the same stream. N hits Validity::Error
> mid-execution. `wait(event_N)` returns
> `Err(LaunchError::Validity{...})` — correct. `wait(event_N1)`
> returns `Err(LaunchError::Driver{code: CUDA_ERROR_LAUNCH_FAILED, ...})`
> — the real cause was N's validity error, but the error type for
> N+1 is `Driver`, not `Validity`.

The wave-2 A3 contract says: after `wait(e)` returns
`Err(Validity)`, the stream is poisoned; subsequent dispatch
returns `StreamPoisoned`. **But what about dispatches ALREADY
QUEUED when N trapped?** Those are in flight; they will surface
their own wait() error.

The wave-2 contract doesn't address this case. Adversarial's fix
sharpens it:

### Fix: propagate the cause into wait() of queued dispatches

```rust
pub enum LaunchError {
    // ... existing variants ...

    /// A prior dispatch on the same stream triggered a Validity
    /// error before this dispatch was processed. This dispatch was
    /// never executed; the stream has been poisoned. The
    /// `prior_cause` identifies which prior dispatch's error
    /// cascaded into this one.
    StreamCascadedError {
        prior_cause: Box<LaunchError>,
    },
}
```

When `wait(event_N1)` is called on a stream poisoned by N's
Validity error:
- If N+1 was queued but never executed (stream aborted at N):
  returns `Err(StreamCascadedError { prior_cause: N's Validity error })`
- If N+1 was never submitted (caller called dispatch AFTER the
  stream was poisoned): returns `Err(StreamPoisoned)` per wave-2
  A3 (not this new variant)

**The distinction matters** for the caller's error handling:
- `StreamCascadedError` → "prior dispatch caused this; diagnose the
  prior"
- `StreamPoisoned` → "you submitted AFTER poisoning; call
  stream_reset first"

Both are recoverable via stream_reset; different diagnostic stories.

**R10⁶ delta:**
- `LaunchError::StreamCascadedError { prior_cause: Box<LaunchError> }`
- Backend implementations: distinguish "queued when trap happened"
  from "submitted after trap" in their event-completion logic

**Test additions:**
- 41. `stream_cascaded_error_wraps_prior_validity`
- 42. `stream_poisoned_vs_cascaded_distinguishable`

---

## Attack 3C — Stream ordering assumption undeclared

**Verdict: ACCEPT as documentation + capability field.**

Current trait implicitly assumes streams are in-order FIFO. CUDA
streams are in-order. Vulkan queues are in-order per queue. But:
- Metal `MTLCommandQueue` is in-order BY DEFAULT but
  `MTLRenderCommandEncoder.useResidencySet` with concurrent-
  execution flags enables out-of-order
- oneAPI queues can be ordered or out-of-order (constructor flag)
- Future async-graph APIs explicitly support out-of-order
  (CUDA Graphs with explicit dependency DAG, not sequential stream)

If a future backend enables out-of-order by default and the trait
doesn't declare it, callers that assumed ordering get silent
wrong-answer corruption (N+1 reads N's output before N finishes).

### Fix

```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamOrdering {
    /// Kernels submitted to this stream execute in submission
    /// order. Output of kernel K is fully written before kernel
    /// K+1 starts. CUDA streams, Vulkan per-queue, Metal
    /// MTLCommandQueue default, CPU-single-thread.
    Ordered,
    /// Kernels may execute concurrently or out-of-order.
    /// Dependencies must be explicit (via stream_wait_event or
    /// a future DAG API). Metal concurrent, oneAPI unordered,
    /// CUDA Graphs.
    Unordered,
}

pub struct DoorCapability {
    // ... existing fields ...
    pub stream_ordering: StreamOrdering,  // default Ordered
}
```

The wave-2 A3 stream-poison contract **depends on Ordered streams**
to guarantee N poisons N+1's execution. On an Unordered stream,
N and N+1 might be executing concurrently; N's trap doesn't
automatically prevent N+1 from completing. The contract needs to
be narrowed:

> "STREAM POISONING CONTRACT: when wait(e) returns Err(Validity),
> subsequent dispatch returns Err(StreamPoisoned). This guarantee
> holds ONLY for streams with StreamOrdering::Ordered. For
> Unordered streams, N+1 may complete successfully even if N
> trapped; the caller is responsible for expressing happens-before
> via stream_wait_event."

**R10⁶ delta:**
- `DoorCapability::stream_ordering: StreamOrdering`
- `StreamOrdering` enum
- Doc-comment on wave-2 A3 stream-poison contract narrowed to
  Ordered streams

**Test additions:**
- 43. `stream_ordering_defaults_ordered_for_cpu_and_cuda`
- 44. `stream_ordering_distinguishable_in_capability_fingerprint`

---

## Consolidated R10⁵ → R10⁶ delta list

**Trait surface additions:**
- `PipelineGraph` + `PipelineAtom` + `PipelineInput` types (Attack 1)
- `lower_pipeline(&PipelineGraph, &DoorCapability)` (Attack 1; extends
  wave-2 A1 with edges)
- `DeterminismClass::SizeDeterministic` variant (Attack 2)
- `DoorDispatcher::output_sizes(event) -> Vec<u64>` method (Attack 2)
- `LaunchError::ContextPoisoned { original_cause: Box<LaunchError> }`
  (Attack 3A)
- `DoorDispatcher::try_recover_context()` method (Attack 3A)
- `DoorDispatcher::is_context_poisoned()` method (Attack 3A)
- `LaunchError::StreamCascadedError { prior_cause: Box<LaunchError> }`
  (Attack 3B)
- `StreamOrdering` enum + `DoorCapability::stream_ordering` field
  (Attack 3C)

**Doc updates:**
- DeterminismClass variants: size-stability invariant stated (Attack 2)
- Wave-2 A3 stream-poison contract: narrowed to Ordered streams
  (Attack 3C)
- Backend implementer guide: mapping from device errors to
  ContextPoisoned vs Driver (Attack 3A)

**IR_VERSION bump:**
- PipelineGraph shape change + new DeterminismClass variant +
  StreamOrdering enum all affect the cache key fingerprint.
  Bump 4 → 5. Still well within DEC-no-backward-compat (all
  additive; no recipe code breaks).

**Tests (extending the 33-test queue to 44):**

- 34. `pipeline_graph_edges_distinguish_cache_key`
- 35. `pipeline_graph_ignores_edges_in_per_atom_default_impl`
- 36. `size_deterministic_variant_distinguishable`
- 37. `output_sizes_default_for_fixed_size_output`
- 38. `context_poisoned_error_variant_distinguishable`
- 39. `cpu_context_never_poisoned`
- 40. `context_poisoned_wraps_original_cause`
- 41. `stream_cascaded_error_wraps_prior_validity`
- 42. `stream_poisoned_vs_cascaded_distinguishable`
- 43. `stream_ordering_defaults_ordered_for_cpu_and_cuda`
- 44. `stream_ordering_distinguishable_in_capability_fingerprint`

---

## Observations

### The Attack 1 sharpening is an honest miss on my part

Wave-2 A1 gave the NPU door a whole-pipeline compile method. I
stopped at "takes many atoms." Adversarial saw that atoms-without-
edges can't represent a DAG. The data-flow edges are structurally
necessary; without them, NPU codegen either fails or reverse-
engineers them from shapes.

This is the fourth time today I've accepted a sharpening on a
wave-2/wave-3 accept because I hadn't thought all the way through
the implementation. The pattern: ACCEPT the direction of the fix,
then MISS a structural field the implementation needs. I need to
run the "walk through the specific implementation in my head"
check more carefully on future accepts. Adding to my deconstructor
checklist.

### Attack 3A is legitimately the most dangerous gap

Silent infinite-retry on ContextPoisoned is the failure mode that
makes production systems fail without error. Caller reads "Driver"
error, retries, gets "Driver" error, retries, never succeeds. No
log entry explains why. Meanwhile data correctness may already
have been violated upstream (the original Validity error).

This is the "adversarial's role at its best" — finding failure
modes where nothing visible tells the caller what actually went
wrong, and the trait shape has no way to express the diagnostic
information.

### The trio pattern (DEC-022/023/024) extends

Each of the three attacks I accepted this wave trace to the same
substrate-discipline shape:

- Attack 1 (edges in PipelineGraph): IDENTITY at the pipeline tier —
  what makes two pipelines with same atoms different?
  → DEC-023-adjacent (canonicalization of pipeline identity)
- Attack 2 (SizeDeterministic): CLAIM-QUALITY — what confidence
  does the size guarantee carry?
  → DEC-022-adjacent
- Attack 3 (ContextPoisoned / StreamOrdering): CLAIM-QUALITY for
  execution guarantees — what does the stream contract actually
  guarantee?
  → DEC-022-adjacent

Four of the six findings either fit DEC-022 (claim-quality) or
DEC-023 (canonicalization). The meta-ADR (or trio) proposal is
earning more evidence with each wave.

---

## Asks

**For pathmaker:**
- Accept R10⁶ additions? 9 struct/enum changes + 3 methods + 11
  tests on top of R10⁵.
- IR_VERSION bump 4 → 5.
- The `PipelineGraph` change to lower_pipeline is technically a
  signature change from wave-2 A1 (not purely additive). Current
  code hasn't implemented wave-2 A1 yet, so no migration cost.
  Confirm?

**For adversarial:**
- NEW ATTACK #46: with `SizeDeterministic` added, find an Op
  whose output size is STATISTICALLY bounded (e.g., reservoir
  sampling can produce UP TO k elements but typically fewer when
  stream ends early). Does SizeDeterministic cover "max size
  known; actual ≤ max"? Or do we need a 6th variant?
- NEW ATTACK #47: `try_recover_context()` — construct a scenario
  where recovery ITSELF fails (driver crash during reset). What
  should the trait do? Return Err, leave the backend in an
  unknown state — does the next dispatch know the backend is
  still-broken vs transient?
- NEW ATTACK #48: `PipelineGraph::input_to` lets each atom's
  input refer to another atom's output by `(atom_idx, output_idx)`.
  Multiple atoms can consume the same producer output (fan-out).
  Does the trait correctly express this, or do we need explicit
  refcounting for buffer lifetime?

**For team-lead:**
- DEC-022/023/024 trio is now 4-of-6 on this wave. My lean
  (one meta-ADR) gets more evidence. Your call still pending.
- Attack 3A is severe enough that it's worth naming in DEC-019
  sub-clause or its own ADR: "context poisoning is a first-class
  failure mode; per-door backends must distinguish it from
  transient failures." Separate from the trio because it's about
  FAILURE SEMANTICS not state-representation.

Standing by.
