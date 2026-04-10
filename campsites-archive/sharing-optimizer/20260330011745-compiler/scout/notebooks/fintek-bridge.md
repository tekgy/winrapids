# FinTek Integration Bridge

*Scout analysis, 2026-03-30. Navigator provocation: is the bridge thin or does it require rethinking the leaf interface?*

---

## The Current Leaf Contract

A K02 leaf in FinTek declares two strings:

```python
# trunk/cli/templates/branch/leaf/compute/core_fused.py
kernel_type = "rolling_stats"
kernel_key = "mean"
```

That's it. The leaf doesn't write math. It doesn't know about CuPy. It doesn't know about windows or channels. It just names what it wants: "I am a rolling_mean."

The runner (`cupy_lagged_fused/runner.py`) picks this up:
1. Discovers all channels needed across all leaves in the batch
2. Stacks them into a `(C, N)` matrix — multiple channels batched together
3. Computes all kernels for all leaves using that matrix
4. Dispatches based on `kernel_type` → `KernelType` enum → ops module

Within the runner, there's already intermediate sharing:

```python
sums = rolling_sum(data_matrix)  # computed once
cache[KernelType.ROLLING_STATS] = {
    "mean": rolling_mean(data_matrix, sums=sums),
    "std": rolling_std(data_matrix, sums=sums),
}
```

`rolling_sum` is not exposed to the leaf. The runner knows that mean and std both need it. This is manual, runner-local CSE — someone had to think about it and write it. It works, but it only shares within this one dispatch batch.

---

## The WinRapids Leaf Contract

A WinRapids-native leaf would return a `PipelineSpec`:

```python
def pipeline_spec(data_var: str, windows: list[int]) -> PipelineSpec:
    return PipelineSpec(calls=[
        SpecialistCall(specialist="rolling_mean", data_var=data_var, window=w)
        for w in windows
    ])
```

Or, keeping it declarative (no method, just attributes):

```python
specialist = "rolling_mean"
# data_var and window come from the leaf's channel + window config
```

The compiler handles the rest: decompose the specialist into primitives, run CSE, topo-sort, probe the store.

---

## The Bridge

The translation is mechanical:

```
("rolling_stats", "mean") → SpecialistCall("rolling_mean", ...)
("rolling_stats", "std")  → SpecialistCall("rolling_std", ...)
```

A dictionary lookup. Five lines of Python. The leaf's _semantics_ don't change — it still declares "I am a rolling_mean." Only the declaration format changes: from a two-string enum lookup to a named specialist call.

The runner changes more substantially. Instead of:
```python
result = compute_kernel_dispatch(kernel_type, kernel_key, data_matrix, windows)
```

It becomes:
```python
spec = build_pipeline_spec(leaves, channels)
plan = winrapids.plan(spec, registry, world)
results = winrapids.execute(plan, world, dispatcher)
```

The runner stops owning the execution graph. The compiler owns it.

**The bridge is thin on the leaf side. It's more interesting on the runner side.**

---

## What Changes, What Stays the Same

**Stays the same:**
- Leaf declarations remain declarative (no math in the leaf)
- The leaf doesn't know about CuPy or CUDA or primitives
- The node===node structural invariant still holds
- Windows configuration lives at the leaf level
- Channel routing lives above the leaf

**Changes:**
- The runner is replaced by a compiler + executor
- Intermediate sharing moves from manual (runner writes `sums = rolling_sum(...)`) to structural (compiler CSE)
- Sharing scope expands: from within-batch to across the full pipeline

---

## The Scope Expansion Is the Interesting Part

The current runner shares `sums` within a single leaf batch. This is local sharing — someone looked at mean and std, noticed they both need sums, and made them share.

WinRapids CSE shares `rolling_sum` primitives across ALL leaves in the pipeline. If leaf A (rolling_mean) and leaf B (rolling_std) both decompose to the same `rolling_sum` primitive with the same window on the same data, the compiler deduplicates them automatically. No one has to notice the sharing and write it down.

More importantly: provenance sharing extends this across TIME. If the rolling_sum for SPY over window=20 was computed yesterday and the data hasn't changed, the executor skips it entirely — pointer handoff, zero computation. The runner has no equivalent mechanism.

---

## The Batching Gap

There's one structural gap worth noting. The current runner batches channels: `_discover_shared_channels()` gathers all channels needed across all leaves, stacks a `(C, N)` matrix, and processes them together. WinRapids currently operates on 1D arrays — one `data_var` per `SpecialistCall`.

This means the multi-channel batching (C tickers in one kernel launch) doesn't directly translate. For N channels, you'd have N `SpecialistCall`s rather than one `(C, N)` matrix call.

But this is arguably a better design:
- No upfront channel discovery phase needed
- CSE deduplicates automatically (same data_var + window = same computation)
- Provenance handles cross-day reuse without explicit channel tracking
- The GPU kernel still runs on a single channel; batching becomes loop-level parallelism at the executor

The current explicit discovery step (`_discover_shared_channels`) was working around the absence of structural sharing. With structural sharing, you don't need to discover it — you just declare your needs and let the compiler find the duplicates.

---

## Verdict

**The bridge is thin.** The leaf interface barely changes — two strings become one named specialist. The runner is replaced by a different execution substrate, but the runner was already doing the same conceptual work (build an execution graph, share intermediates, dispatch). It was just doing it manually, locally, without persistence.

The WinRapids version does the same things structurally and automatically, at global scope, with provenance persistence across days.

The FinTek leaf is already thinking in the right shape. `kernel_type = "rolling_stats"` is already a specialist name with one level of indirection. The indirection is what makes the bridge possible — the leaf never coupled to CuPy directly.

---

## One Non-Obvious Observation

The current runner is doing CSE manually — someone saw that mean and std both need sums and wrote the sharing by hand. That's a structural debt. Every time a new kernel is added, someone has to look at all existing kernels and find the sharing.

`Arena::add_or_dedup` replaces that entire class of work. You never again have to audit the sharing manually. New specialist? Add it to the registry. The compiler finds all the sharing automatically.

The leaf interface isn't the interesting part of this bridge. The elimination of manual CSE is.
