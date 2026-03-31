# The Unified Structure: Everything Is Decomposable Accumulation

## The theorem (manuscript 003)

**Decomposable Accumulation.** A computation C over a collection S is decomposable if:
1. `lift: S → M` maps each element to a monoid M
2. `⊕: M × M → M` is associative with identity e
3. `extract: M → Result`

such that `C(S) = extract(⊕ᵢ lift(sᵢ))`.

The liftability test (rendering) and the scannability test (parallel prefix) are the SAME test: does the computation admit decomposable accumulation?

**This is the mathematical foundation of the WinRapids compiler.**

## Where the tambear work sits in this structure

**Hash scatter** (tambear, HashScatterEngine):
- S = {(key_i, value_i)} for all rows
- lift = value → identity (trivial lift)
- ⊕ = + (atomicAdd)
- extract = identity
- Result = per-group accumulated sums

This is decomposable accumulation with the SIMPLEST possible lift (identity).

**ScatterJit** (tambear, new):
- S = {(key_i, value_i)} for all rows
- lift = φ_expr(v) — ANY monoid-compatible expression
- ⊕ = + (atomicAdd — always, for additive scatter)
- extract = identity
- Result = per-group accumulated φ-values

ScatterJit implements decomposable accumulation with a USER-SPECIFIED lift function.
The `phi_expr` parameter is the `lift` function, expressed as a CUDA string.

The monoid is always (ℝ, +). The freedom is in the lift.

**AffineOp / SarkkaOp** (winrapids-scan):
- S = sequence of observations (x_t)
- lift = per-element state initialization: (a, b) ← (a_coeff, b_coeff * x_t)
- ⊕ = affine combine: (a₂, b₂) ⊕ (a₁, b₁) = (a₂·a₁, a₂·b₁ + b₂)
- extract = state projection: b (the accumulated state)

This is decomposable accumulation with the ASSOCIATIVE COMBINE over the 2×1 matrix monoid.

## The two primitives are two orders of the same structure

| Primitive | Order | Lift | Monoid | Parallelism |
|---|---|---|---|---|
| Scatter | 0 | φ(v): value → ℝ | (ℝ, +) | O(n) per element |
| Scan | 1 | (a, b) from (x_t) | (2×2 matrices, ·) | O(log n) depth |

Order 0: no state. The accumulation IS the result. Full parallelism.
Order 1: state propagates forward. Tree-structured parallelism.

Order 2 would be: state depends on accumulated STATE (not just current element).
This is the Fock boundary — where decomposability breaks.

## The compiler's job, formally stated

Given a pipeline description, the compiler must:

1. **Identify** all sub-computations that admit decomposable accumulation
2. **Extract** their (lift, ⊕, extract) triples
3. **Fuse** adjacent decomposable accumulations when possible:
   - Same monoid: fuse into one kernel (multiple φ expressions, single pass)
   - Different monoids: separate kernels, but share the input scan

The "9 primitives" in the compiler (manuscript 004) are the 9 canonical forms
of decomposable accumulation available on GPU. Everything else is either:
- A composition of primitives (handled by the compiler)
- Non-decomposable (must be sequential or O(n²))

## The fusion insight

Two scatter operations with the same key and monoid = one scatter, two φ functions:

```
scatter(keys, values, phi="v")       // sum
scatter(keys, values, phi="v * v")   // sum_sq
scatter(keys, values, phi="1.0")     // count
```

These three scatter operations are ONE decomposable accumulation with lift:
```
lift(key, v) = (v, v*v, 1.0)    // 3D lift into ℝ³
⊕ = (a₁+b₁, a₂+b₂, a₃+b₃)    // pointwise addition
```

This is exactly what HashScatterEngine::scatter_stats does: a 3D lift with pointwise sum.
The compiler should detect three adjacent scatter_phi calls with the same key
and fuse them into scatter_stats.

The fusion rule: if (key, ⊕) are the same, multiple (lift_1, lift_2, ..., lift_k) can
be fused into one lift: (lift_1, ..., lift_k) with pointwise ⊕.

## What the ScatterJit needs to become the full compiler primitive

Currently: ScatterJit compiles one φ function, runs one scatter.

For the compiler: ScatterJit needs a multi-phi interface:
```rust
jit.scatter_multi_phi(
    &[PHI_SUM, PHI_SUM_SQ, PHI_COUNT],  // multiple lift functions
    keys, values, refs,
    n_groups,
)
// → (sums, sum_sqs, counts) in one kernel pass
```

This is exactly scatter_stats, generalized. The compiler generates the multi-phi
template by fusing adjacent scatter_phi operations.

The generated CUDA:
```cuda
__global__ void scatter_multi(keys, values, refs, out0, out1, out2, n) {
    int gid = ...
    if (gid < n) {
        int g = keys[gid]; double v = values[gid]; double r = refs[g];
        atomicAdd(&out0[g], (PHI_0));   // v
        atomicAdd(&out1[g], (PHI_1));   // v * v
        atomicAdd(&out2[g], (PHI_2));   // 1.0
    }
}
```

One memory pass, three accumulations. The fusion is free.

## The segmented scan connection

The natural composition: per-ticker time series (scan) + cross-ticker aggregation (scatter).

For a DataFrame with ticker_id and prices columns:
1. Segmented scan: AffineOp within each ticker segment → per-tick smoothed prices
2. Scatter: groupby ticker → aggregate smoothed prices

The segmented scan is a scan with "reset" at group boundaries. It's decomposable
accumulation where the lift knows about segment boundaries.

```
lift(key, x) = {
    if key != prev_key: fresh_state(x)
    else: state_update(accumulated_state, x)
}
```

But this lift depends on `accumulated_state` — it's order 1, not order 0.
A segmented scan IS a scan (same structure), just with reset at segment boundaries.

The compiler's fusion rule for scan ∘ scatter:
- Detect a scan followed by a scatter on the same key
- If the scan is segmented on that key: the scan already handles segment boundaries
- The scatter post-processes the scan outputs

No sort needed: the segmented scan knows which segments it's in (from the key column)
without requiring them to be contiguous. The implementation uses per-segment state
initialization via the GroupIndex.

## Why this matters for the roadmap

The compiler is:
1. A decomposable accumulation detector
2. A (lift, ⊕, extract) triple extractor
3. A fusion engine for adjacent accumulations with compatible monoids
4. A JIT generator for fused kernels

The ScatterJit is step 2+4 for the scatter primitive.
The AffineOp templates are step 2+4 for the scan primitive.
The fusion engine (step 3) is the missing piece that enables "10 lines → compiled pipeline."

Fusion engine = step 3: detect adjacent accumulations with compatible monoids → generate merged (lift, ⊕, extract) triple → JIT compile.

This is the architectural vision. The foundations are laid.
