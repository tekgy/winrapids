# Scout Report: Peak 6 — RFA Terrain

*Scout: claude-sonnet-4-6 | Date: 2026-04-11*

Pre-reading for the Peak 6 implementer. The navigator updated the campsites.md with a
significant correction: Peak 6 now targets **gpu_to_gpu determinism** using the
**Reproducible Floating-point Accumulator (RFA)** algorithm from Demmel-Nguyen.

This report maps the algorithm, its I7 compliance, its relation to the existing
accumulate+gather structure, and the two key issues the camp team needs to resolve.

---

## The problem statement (clarified)

There are two distinct determinism problems in the recipe catalog:

**Problem A: run_to_run determinism** (what fixed-tree reduce solves)
- Same GPU, same kernel, different runs → different bit results
- Cause: `atomicAdd` reduction order is non-deterministic
- Fix: fixed-order tree reduce with deterministic launch config
- This is the weaker requirement

**Problem B: gpu_to_gpu determinism** (what RFA solves)
- Different GPUs, different architectures, Vulkan vs CUDA → same bit results
- Cause: tree shape depends on block count, warp size, hardware topology
- Fix: an order-independent algorithm that produces the same answer regardless
  of how the work is partitioned
- This is what Peak 7's summit test (`cpu.to_bits() == cuda.to_bits() == vulkan.to_bits()`) requires

The navigator's correction: Peak 6 must achieve Problem B, not just Problem A. The
fixed-tree approach from the original plan only solves A.

---

## The RFA algorithm (Demmel & Nguyen, 2013)

**Mandatory reading (from navigator's campsite 6.1):**
1. Demmel & Nguyen, "Fast Reproducible Floating-Point Summation," ARITH 2013
2. Demmel & Nguyen, "Parallel Reproducible Summation," IEEE TC 2015
3. Ahrens, Demmel, Nguyen, ReproBLAS tech report EECS-2016-121

**The algorithm in one paragraph:**

Partition the fp64 exponent range into K bins (K = 39 is sufficient for exact rounding).
Each bin covers an exponent range of width 1. For each input element `x`, compute its
exponent-aligned bin index `b = floor(log2(|x|))` (using integer bit manipulation, not
a transcendental). Accumulate `x` into bin `b`. After all elements are processed, fold
the K bins in a fixed order (say, highest exponent first) to produce the final sum.

The fold order is independent of the input data, thread count, or GPU architecture.
Two different hardware configurations that process the same elements will produce the same
K bin vectors, which when folded produce the same scalar.

**Why this works numerically:**

Two elements in the same bin have exponents within 1 of each other. When you add two
fp64 numbers whose exponents differ by at most 1, there is no catastrophic cancellation —
you're adding numbers of comparable magnitude. The within-bin sum accumulates error at
most `K * eps * |bin_max|` which is bounded by the bin width.

The key invariant: elements stay in their bins. No element migrates to a different bin
during accumulation (because the bin is chosen by the element's exponent, not the
running sum's exponent). This differs from naive Kahan summation where the error
compensation depends on the running sum.

---

## I7 compliance: accumulate + gather decomposition

The navigator notes: "bin-accumulate step is `accumulate(exponent_bin_grouping, identity, add)`
+ `gather(fixed_bin_order, add)`."

Let's make this concrete:

**Accumulate step:**
```
grouping: ExponentBin(K)    -- new Grouping variant needed
expr:     identity (val)
op:       Add
```
This groups input elements by their exponent bin and adds them into K accumulators.
It's exactly the accumulate+gather pattern, just with a K-dimensional grouping.

**Gather step:**
```
for b in (K-1)..=0:
    final_sum += bins[b]
```
This is a scalar fold over the K bins in fixed order. CPU-side, not GPU-side. Same
pattern as the existing gather stage (but over K intermediate values instead of
just computing a formula from the accumulated scalars).

**I7 verdict:** Fully compliant. The RFA is `accumulate(ExponentBin(K), identity, add)`
followed by `gather(fixed_order_fold, add)`. It's a new Grouping variant, not a new
paradigm.

---

## The two issues for the IR Architect (before 1.2)

**Issue 1: `Grouping::ExponentBin(K)` in the Grouping enum**

The current `Grouping` enum in `accumulates/mod.rs`:
```rust
pub enum Grouping {
    All, ByKey, Prefix, Segmented, Windowed, Tiled, Graph,
}
```

For RFA, we need:
```rust
ExponentBin(usize),   // K bins, elements grouped by floor(log2(|x|))
```

This is the most natural structural fit. `ExponentBin(39)` would be the Phase 1
default. The `fuse_passes` machinery would treat this as a separate pass (it shares
no fuse group with `All` or `ByKey`).

The IR Architect needs to decide before writing the Rust AST types whether
`ExponentBin` goes in `Grouping` now or is deferred to Peak 6. Recommendation:
add it to the `Grouping` enum now (anti-YAGNI: the structure guarantees we'll need it),
but don't implement it yet. The IR can declare it unimplemented and panic if a pass
tries to execute it before Peak 6 lands.

**Issue 2: vector-valued accumulator output**

An `ExponentBin(K)` pass produces K output values (one per bin), not one scalar.
The current `AccumulateSlot.output: String` names ONE output. For RFA we need K named
outputs or one named output that's a K-vector.

Options:
(a) K separate `AccumulateSlot`s with outputs `"bin_0"`, `"bin_1"`, ..., `"bin_K"`
    — fits current structure, noisy
(b) One slot with `output: "bins"` and `K: usize` metadata — cleaner but the slot
    type needs a new field
(c) A separate `ReduceSpec` type for the fold operation that carries K

Recommendation: option (a) for Phase 1 — generate K slots at recipe construction time.
A `rfa_slots(K, input_expr, name_prefix)` helper returns `Vec<AccumulateSlot>` with
outputs `"name_prefix_0"` through `"name_prefix_{K-1}"`. The recipe's gather formula
then has access to each bin by name.

This is verbose but doesn't require any structural changes to `AccumulateSlot`.

---

## The variance problem (separate from RFA)

RFA solves reproducibility. It does NOT solve catastrophic cancellation.

The current `variance()` recipe uses the one-pass formula:
`(Σx² - (Σx)²/n) / (n-1)`

**Confirmed failure case** (verified in Python):

| Data | True variance | One-pass result | Relative error |
|---|---|---|---|
| [1e9, 1e9+1, ..., 1e9+9] | 9.1667 | 0.0 (exact zero) | 100% |
| [1e9 + x/1e6 for x in 0..100] | 8.4e-10 | 165.5 | 2e+11 × |

For financial data (prices ~100-500, daily changes ~0.01-1%), this formula WILL fail.

**The correct algorithm:** Welford's online variance:
```
for each x:
    n    += 1
    delta = x - mean
    mean += delta / n
    delta2 = x - mean
    M2   += delta * delta2
variance = M2 / (n - 1)
```

Welford accumulates `(n, mean, M2)` and is numerically stable because it always
subtracts the running mean, not a precomputed `Σx²`.

**Welford is order-dependent but stable.** RFA applied to the Welford mean update would
give both stability AND reproducibility, but this is a more complex composition.

For Peak 6: the variance recipe should be flagged as needing Welford before the
bit-exact target is meaningful. An unstable formula that happens to match on uniform
small integers is not a valid acceptance criterion.

---

## The accumulate+gather view of RFA + Welford

The full "correct variance" decomposition:

**Stage 1 (accumulate):** RFA over `ExponentBin(K)` with `identity` expr
  → K bin sums for Σx, K bins for Σ(x - running_mean)², etc.
  → Actually: Welford is sequential-dependent, so the GPU stage accumulates per-block
    Welford statistics, and the host folds them using the parallel Welford merge formula.

**Parallel Welford merge** (Chan et al., 1979):
```
// Merge two (n, mean, M2) summaries into one:
n_combined = na + nb
delta = mean_b - mean_a
mean_combined = mean_a + delta * nb / n_combined
M2_combined = M2_a + M2_b + delta^2 * na * nb / n_combined
```

This merge formula is exact (up to fp rounding) and can be applied in any order —
which means it's compatible with a fixed-order fold from a block-partial scheme.

**The right structure for variance in Peak 6:**
- Each GPU block computes a local Welford `(n, mean, M2)` via the sequential update
- The host folds the per-block `(n, mean, M2)` triples using the parallel merge formula
  in a fixed order (block 0, 1, 2, ..., in order)
- Result: numerically stable AND run-deterministic (order of merge is fixed by block index)

This is NOT yet gpu_to_gpu deterministic (the parallel merge is order-dependent). For
full gpu_to_gpu: apply RFA to the delta accumulations. That's Phase 2 territory.

---

## Summary: what Peak 6 needs from earlier peaks

| Peak | Needed for Peak 6 |
|---|---|
| Peak 1 (IR) | `Grouping::ExponentBin(K)` in the Grouping enum (even as stub) |
| Peak 1 (IR) | `reduce_rfa.f64` as a new `.tam` IR op |
| Peak 2 (libm) | `tam_floor`, `tam_log2_exponent_extract` (integer bit op, not a transcendental) |
| Peak 3 (PTX) | PTX emission for RFA bin extraction and per-bin atomicAdd |
| Peak 5 (CPU) | CPU interpreter for `reduce_rfa.f64` (sequential, trivially correct) |

The exponent extraction is a bit manipulation, not a transcendental: `(f64_bits >> 52) & 0x7FF`
gives the biased exponent. This can be expressed in the IR as integer ops on the bit
representation of the float. It does NOT need libm.

---

## Papers summary for the math-researcher

The three Demmel-Nguyen papers establish:
1. **ARITH 2013**: The basic bin-accumulate algorithm. K=40 for fp64. Faithful rounding
   (the result is within 1 ULP of the true sum). Sequential version.
2. **IEEE TC 2015**: Parallel version for multi-core and GPU. Shows that per-block
   bin vectors can be summed in any order and the result is still reproducible.
   The key: bin vectors commute under vector addition, so any tree shape works.
3. **ReproBLAS tech report**: Full implementation details. K=39 chosen specifically
   so that the within-bin accumulation never overflows fp64 for N < 2^53 elements.
   Covers the subnormal and inf special cases.

Scout recommendation: the math-researcher should read all three before campsite 6.1,
then write the 6.1 decision doc arguing for RFA + the specific K value.
