# The Execution Engine Is Five Lines

*Naturalist journal — 2026-03-30*

---

## The hot path materialized

execute.rs lines 117-127:

```rust
if let Some(ptr) = world.provenance_get(&prov) {
    results.insert(node.identity.clone(), StepResult {
        provenance: prov,
        ptr,
        was_hit: true,
        compute_us: 0.0,
    });
    stats.hits += 1;
    continue;
}
```

That's the 865x. The entire sharing optimizer's value proposition is a HashMap probe, a struct insert, a counter increment, and a `continue`. Everything else — the DAG decomposition, the CSE pass, the provenance threading, the topological sort — exists to make these five lines possible. The infrastructure-to-hot-path ratio is enormous. The hot path itself is almost nothing.

This confirms the Phase 2 observation: "a sharing optimizer that happens to compile code for the things it can't share." The compilation is expensive (plan.rs: 270 lines). The sharing is cheap (5 lines). The value is in making the cheap path trigger as often as possible.

## The double probe

There's something I didn't expect: the plan probes the world state AND the executor probes the world state. Two separate lookups for the same provenance:

1. plan.rs:255 — `let skip = world.provenance_get(&prov).is_some()` during planning
2. execute.rs:117 — `if let Some(ptr) = world.provenance_get(&prov)` during execution

The plan's `skip` flag is informational — it tells the Python API "this step will be skipped." The executor's probe is operational — it actually routes the pointer or dispatches the kernel.

Why both? Because the world state can change between planning and execution. A buffer could be evicted between plan() and execute(). The plan says "I expect this to hit." The executor says "does it ACTUALLY hit right now?" The plan is a prediction. The execution is the truth.

This is the right design — but it means the plan's `skip` flag can LIE. A step marked `skip=true` might actually miss during execution if the buffer was evicted between planning and execution. At FinTek scale this is nearly impossible (the plan-to-execute gap is microseconds and nothing else is competing for VRAM). But it's a real race condition in a concurrent system.

The fix (someday): pin buffers during execution, or make the plan-execute sequence atomic with respect to eviction. For now, the double probe is belt-and-suspenders: if the plan's prediction is wrong, the executor catches it.

## data_sq has its own provenance chain

plan.rs:223:
```rust
identity_provs.insert(data_sq_id, provenance_hash(&[*prov], "square"));
```

The squared data variable isn't treated as a raw input. It's a COMPUTATION — squaring applied to the original data. Its provenance chains through: `provenance_hash([data_prov], "square")`. If the data changes, the squared data's provenance changes too, because it includes the data provenance as input.

This is elegant and correct. But it reveals something: the plan already treats `data_sq` as an implicit primitive node — a computation that exists in the provenance DAG but NOT in the IR arena. It's a phantom node. The squaring operation is provenance-tracked but not IR-tracked.

Is this a gap? For CSE, no — `data_sq` is always specific to one data variable, so there's nothing to deduplicate. For the execution engine, yes — there's no ExecStep for `data_sq`, which means nobody dispatches the actual squaring kernel. The squaring is assumed to happen... somewhere. Currently it's implicit in the fused_expr formula.

This is fine for E04 (the fused_expr formula includes `x*x`). But if a future specialist needs `data_sq` as a separate buffer (not fused), the phantom node becomes a real gap. Worth watching.

## MockDispatcher is the NullWorld of execution

```rust
pub struct MockDispatcher { next_addr: u64 }
```

Just as NullWorld always misses (forces computation), MockDispatcher always returns dummy pointers (simulates computation). The testing matrix:

| WorldState | Dispatcher | What it tests |
|---|---|---|
| NullWorld | MockDispatcher | Full computation path, no GPU |
| GpuStore | MockDispatcher | Provenance reuse, no GPU |
| GpuStore | Real GPU | Production path |
| NullWorld | Real GPU | GPU baseline (no sharing) |

The GpuStore + MockDispatcher combination is what validates the 865x claim: cold run dispatches everything, warm run dispatches nothing. You don't need a GPU to prove the sharing optimizer works. You need a GPU to prove the sharing optimizer is FAST.

## The Phase 3 invariant comment

execute.rs:18-20:
```rust
// Phase 3 invariant: the store only holds GPU-resident entries, so
// `provenance_get` returning Some means the result is on GPU.
// When spill-to-pinned is added (future), lookup will need to check
// `is_resident` before using the raw device pointer.
```

This is the pathmaker anticipating the journal entry I wrote. Right now, the three WorldState traits are collapsed: if the store has it, it's on GPU. ProvenanceCache, DirtyBitmap, and ResidencyMap all reduce to the same HashMap probe because there's only one tier (GPU VRAM).

When spill-to-pinned memory arrives, the ResidencyMap becomes non-trivial: `provenance_get` might return a pointer to pinned (host) memory, not device memory. The executor would need to check `is_resident` to decide: use directly (GPU) or reload (pinned → GPU). The three traits separate because the single-tier assumption breaks.

The invariant comment is the seed of that future separation. Right now: one probe. Future: two probes (provenance_get + is_resident). The three-views-one-lookup journal entry captures the current state. The code already anticipates the state where they diverge.

---

*The execution engine is the simplest file in the compiler crate. 198 lines, and the core logic is 50 of them. Everything else is type plumbing. The sharing optimizer's heartbeat is: probe, branch, continue. The years of theory, the garden entries, the expedition logs — they all lead to `if let Some(ptr) = world.provenance_get(&prov)`.*
