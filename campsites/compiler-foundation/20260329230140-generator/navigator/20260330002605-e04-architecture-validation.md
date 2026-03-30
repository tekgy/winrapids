# E04 Architecture Validation

Created: 2026-03-30T00:26:05-05:00
By: navigator

---

## What E04 Proves

The core compiler loop works: spec -> primitive DAG -> CSE -> codegen -> execute.
CSE finds 33% node reduction automatically (6->4). Generated kernels correct to 2e-7.
2x vs naive at all sizes. 1.1-1.3x vs manually-shared CuPy.

## Two Sources of Speedup (not one)

E04 benchmarking reveals the compiler produces two distinct speedups simultaneously:

**1. CSE: shared scan primitives**
Rolling_zscore + rolling_std both need scan(price, add) and scan(price_sq, add).
CSE deduplicates: 4 scans -> 2 scans. This is what Path B (manual) also achieves.

**2. Pipeline compilation: fused element-wise ops**
Path C (compiler): 2 scans + 2 fused kernels = 4 GPU launches
Path B (manual):   2 scans + ~15 CuPy element-wise calls = ~17 GPU launches

The generated fused kernels eliminate the intermediate launches between mean/std/zscore
computations. This is separate from CSE -- it's the "expression fusion" and "intermediate
elision" optimizations from the 6-type list.

At 100K: Path B=286us, Path C=216us. The 24% gap vs manual sharing is the fusion value.
At 10M: Path B=1938us, Path C=1792us. Gap narrows as scans dominate (both bandwidth-limited).

The benchmark measured BOTH effects. "2x vs naive" = CSE + fusion combined.

## The Executor IS the Pointer Routing Graph

E04's executor is exactly the architecture the naturalist described:

    buffers: dict[str, cp.ndarray]  # identity_hash -> result array

The execution loop:
    for node, binding in plan.steps:
        if node.identity in buffers: continue  # hit: pointer handoff
        # miss: compute and store
        buffers[node.identity] = compute(node, binding, buffers)

This IS "a for loop over identity checks." The pointer routing graph is instantiated
as a dict. The identity hash IS the pointer key. Zero copy on hit.

E04 validates the naturalist's pointer routing graph description concretely,
not just theoretically.

## Binding Carries Sharing Transparently

After CSE, the fused_expr node's `binding` dict already points to the SHARED
scan identity hashes:

    binding = {"data": "data:price", "cs": "a4a43a69", "cs2": "51925aa0"}

These are the CSE-merged identities. The executor doesn't search for shared nodes --
it just looks up binding["cs"] in the buffer pool. The sharing was baked into the
plan during CSE. Execution is pure lookup. This is the clean design.

## Registry is Complete Enough for CSE

The 5-field registry entry (primitive_dag, fusion_eligible, fusion_crossover_rows,
independent, identity) is sufficient for:
- Decomposition (primitive_dag)
- CSE correctness (identity parameters)
- Fusion decisions (fusion_eligible, crossover)
- Ordering constraints (independent)

The injectable world state slots are wired but null -- provenance, dirty_bitmap,
residency all pass through without effect. The API is ready for E07 integration.

## What Remains for Full Registry (E10)

E04 demonstrates the loop with 3 specialists. E10 needs ~135.
The work is filling in the registry, not changing the architecture.
Each specialist entry is 5 fields. The CSE and codegen infrastructure is reusable.

Priority for E10 registry build-out:
1. Scan-heavy specialists (all rolling_* variants) -- most sharing value
2. Sort-heavy specialists (groupby, merge, rank) -- second most
3. DataFrame core (filter, fillna, concat) -- high volume
4. ML preprocessing (StandardScaler, PCA) -- cross-algorithm sharing with FinTek
