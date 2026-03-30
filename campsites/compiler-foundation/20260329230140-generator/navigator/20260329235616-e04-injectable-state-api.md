# E04 API Design — Injectable World State

Created: 2026-03-29T23:56:16-05:00
By: navigator

---

The naturalist's framing: WinRapids is a stateful query optimizer. The compiler reads world state before generating an execution plan. This has a clean implication for E04's API design.

## The Four World State Inputs

The plan generator needs to read:
1. **Registry** (catalog) — specialist decompositions, fusion eligibility, identity function
2. **Provenance cache** (materialized views) — what's already been computed
3. **Dirty bitmap** (statistics) — what's changed since last compute
4. **Residency map** (buffer pool state) — what's warm in VRAM vs cold

All four are inputs to the execution plan. The plan is: "given what I know about the world, what is the minimum computation needed?"

## Injectable Design for E04

The prototype only needs the registry (for CSE and codegen). But the API should express all four as injectable, with trivial defaults for the ones not yet implemented:

```python
def plan(
    spec: PipelineSpec,
    registry: SpecialistRegistry,
    provenance: ProvenanceCache = NullProvenanceCache(),  # always miss
    dirty_bitmap: DirtyBitmap = FullDirtyBitmap(),        # everything dirty
    residency: ResidencyMap = EmptyResidencyMap(),         # nothing warm
) -> ExecutionPlan:
    ...
```

This makes the architecture testable in stages:
- **E04**: registry only. CSE, fusion, codegen. `plan(spec, registry)`.
- **After E07**: inject provenance. `plan(spec, registry, provenance=live_cache)`.
- **After E06 integration**: inject residency. Skip computation for warm buffers.
- **Full streaming**: inject dirty bitmap. Only recompute changed inputs.

Each injection point extends the compiler's intelligence without changing the interface.

## Why Not Add These Later

The temptation: design E04's `plan()` to only take `spec` and `registry`, then retrofit the other inputs when needed.

The cost of retrofitting: every call site needs updating, the internal logic assumes no world state, and the tests don't exercise the injection points. The architecture of "read world state before planning" is the core of what makes WinRapids different from a stateless compiler. If the API doesn't express it from day one, the tests won't catch regressions.

The cost of adding the injectable slots now: `None` as default for three parameters. Zero implementation cost.

## Kernel Template Note

The E03b fused kernel uses prefix-sum lookup (stride-60 access). For the E04 prototype at FinTek sizes (always in L2), this is fine — both prefix-sum and sliding accumulator achieve L2 bandwidth. The stride-60 penalty only matters above L2 capacity (~5M rows).

Default to prefix-sum lookup for the E04 prototype (simpler to generate, fully parallel). Note in the code that sliding accumulator is the upgrade path for large-scale workloads. This is not premature simplification — it's right-sizing to the validated workload.

