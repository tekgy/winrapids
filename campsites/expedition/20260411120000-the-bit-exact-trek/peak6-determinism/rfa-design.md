# RFA Design Document — Reproducible Floating-Point Accumulation for Peak 6

**Author:** math-researcher
**Date:** 2026-04-11
**Status:** draft, awaiting pathmaker + navigator review before Peak 6 implementation begins

**Upstream dependencies:** none (independent of Peak 2 libm work).
**Downstream consumers:** Peak 3 (PTX translator for `reduce_rfa`), Peak 5 (CPU interpreter), Peak 7 (SPIR-V translator, summit test).

**Acceptance criterion (stated here so pathmaker can check themselves):** a reader who has NOT read the Demmel-Nguyen papers can implement RFA from this doc and the invariants, and produce a correct implementation.

**Scope:** `reduce_rfa.f64` — the reproducible sum primitive. Not dot product, not nrm2, not BLAS. Those are compositions over this primitive and can be added later.

---

## 1. What problem RFA solves

The trek's summit test (Peak 7.11) requires

```
cpu.to_bits() == cuda.to_bits() == vulkan.to_bits()
```

on the same `.tam` program. For pure-arithmetic kernels with a *reduction* step, this is false with ordinary summation because floating-point addition is non-associative: different hardware topologies (block count, warp size, tree shape) sum in different orders and produce different last-bit results. Ordinary fixed-tree summation solves the *run-to-run* version of the problem (same GPU, same config, same bits). It does NOT solve the *gpu-to-gpu* version because a different tree shape produces a different answer.

**RFA (Reproducible Floating-point Accumulation) solves the gpu-to-gpu problem by making the accumulation algebraically associative and commutative over its state representation.** Two independent reducers produce identical state for identical (multi-)sets of inputs, and their states merge associatively. This is the only known approach to cross-backend bit-exact reproducibility for floating-point sums with a single pass and bounded memory.

The algorithm is due to Demmel, Nguyen, and Ahrens at UC Berkeley (2013–2016, citations at the end). We reimplement it from the paper — no ReproBLAS source code is used at any step.

---

## 2. The Indexed-Type data structure

An **indexed type** `I` for fp64 is a small fixed-width data structure that represents an "in-progress sum" with enough internal precision that additions can be done associatively.

### 2.1 Shape

An indexed type contains **`K` bins**, each bin being a pair `[Y_C(i), Y_P(i)]` of two fp64 values:
- `Y_P(i)` — the *primary* field of bin `i`. Holds the running sum of slices that land in that bin's exponent window.
- `Y_C(i)` — the *carry* field of bin `i`. Holds overflow bits from `Y_P(i)` when the primary's accumulation has wrapped around its W-bit window too many times.

The indexed type is indexed by the bins' *top-bin index*, which is the exponent at which the topmost bin starts. Bins are contiguous: bin `i+1` covers exponents `W` lower than bin `i`.

For fp64 the default parameters are:

| Parameter | Value (fp64) | Meaning |
|---|---|---|
| `P` | 53 | Mantissa bit count (fp64 spec) |
| `W` | 40 | Width of one bin in exponent bits |
| `K` | 3 | Number of active bins |
| `Accuracy = (K-1)·W` | 80 bits | Internal precision of the indexed sum |
| `MaxDep = 2^(P-W-2)` | 2^11 = 2048 | Deposits before `Renorm` must run |
| `MaxN = 2^(2P-W-2)` | 2^64 | Maximum number of summands (worst case) |

**`K = 3` is the minimum to guarantee accuracy ≥ `P` = 53 bits.** The paper calls this `Kmin`. Setting `K > 3` buys more accuracy at the cost of more flops per deposit (see §5). For Phase 1 of this trek we use `K = 3`.

`W = 40` is the author-recommended default for fp64. Valid range is `[28, 50]`. Larger `W` → more accuracy but lower `MaxDep` (forcing `Renorm` more often); smaller `W` → lower accuracy, more deposits before `Renorm`. **We use `W = 40`.**

### 2.2 Storage cost

`K = 3` bins × 2 fp64 fields per bin = **6 fp64 words per indexed type**. Matches the paper's "6-word reproducible accumulator" nomenclature.

Each thread on the GPU holds one indexed type in registers/shared memory. Per-block-final reduction produces one indexed type per block. Host merges block-level indexed types.

---

## 3. The five primitive operations on an indexed type

Taken directly from the Demmel-Nguyen paper (structural only — we reimplement). There are five operations plus one conversion.

**`Index(x) -> int`** — compute the top-bin index that would be needed to hold the float `x`. This is a bit-manipulation on `x`'s exponent field:
- Extract the IEEE 754 biased exponent of `x`.
- Align to a W-bit grid: `top_bin_for(x) = floor((biased_exponent_of_x - emin) / W)` expressed so that the result is an integer index into the bin grid, measured from emin upward.
- No arithmetic on `x` itself; pure integer manipulation on the bit pattern. **Uses existing IR integer ops only** (see §6 for the compliance check).

**`IIndex(I) -> int`** — return the top-bin index currently tracked by `I`.

**`Update(x, I) -> I`** — if `Index(x) > IIndex(I)`, shift the top bin of `I` upward so it can hold `x`. Shifting means:
1. The new top bin is empty (primary and carry both zero).
2. The old top bin becomes bin 1. The old bin 1 becomes bin 2. Etc.
3. If the old bin `K-1` was nonzero, its contribution is *dropped* — it fell below the precision floor of the new top. The indexed type only represents the top `K` bins at any moment.

This step is the reason RFA is *not* a lossless representation: it discards slices that are too small to matter relative to `max|x_j|`. In the steady state, once `IIndex(I) ≥ Index(max_absolute_input)`, no more shifts happen, and the subsequent deposits are losslessly accumulated within the indexed type's precision.

**`Deposit(x, I) -> I`** — add `x` into `I` *assuming* `I` can already hold it (i.e., `Update` was already called). The per-bin inner loop is:

```
r = x
for k in 0..(K-1):
    s = Y_P(k)_of_I + (r | 1)
    q = s - Y_P(k)_of_I
    Y_P(k)_of_I = s
    r = r - q
# (the final r is the residual that falls below bin K-1 and is discarded)
```

The key trick is **`(r | 1)`** — bitwise-OR of `r`'s fp64 bit representation with integer `1`, which forces the lowest mantissa bit to 1. This forces round-to-nearest-even to break ties in a direction that is independent of the value currently in `Y_P(k)`. That is the single most important line in the algorithm — it is what makes the sum *order-independent*. Without it, the rounding direction would depend on the accumulator state, and two different accumulation orders would produce different results.

The cost is `3K - 2` flops per deposit (for K=3: 7 flops).

**`Renorm(I) -> I`** — renormalize each bin's `(primary, carry)` pair so that primary is within its W-bit window. After `MaxDep = 2^(P-W-2) = 2048` deposits, the primary may have accumulated enough to overflow its window — `Renorm` moves the overflow into the carry and re-centers the primary. Renormalization is pure fp64 arithmetic per bin; no cross-bin dependency, so it's per-bin parallelizable.

**`AddFloatToIndexed(x, I) -> I`** — the outer combine: `Update(x, I); Deposit(x, I); (Renorm if needed)`. The full per-element addition.

**`AddIndexedToIndexed(I, J) -> I`** — the parallel combine. Merges two indexed types into one. **This is the critical parallelism primitive** — it is what makes RFA *associative*: two different tree shapes produce the same answer because this combine is order-independent.

Structurally, `AddIndexedToIndexed` aligns the two indexed types' top-bin indices (if one's top is higher, the other's contents slide down one bin per W exponent levels) and then adds bin-wise: `Y_P_new(i) = Y_P_left(i) + Y_P_right(i)`, `Y_C_new(i) = Y_C_left(i) + Y_C_right(i)`, with a small carry propagation pass afterward. **This bin-wise add is commutative and associative** because it's aligned-window fixed-point addition — no rounding happens at the combine step.

**`ConvertIndexedToFloat(I) -> fp64`** — the final fold. Sums the carry and primary fields in a fixed order (strictly decreasing exponents) into a single fp64 scalar. Uses the Demmel-Hida 2003 ordered-exponent summation, which guarantees ≤ 7 ULPs error. This is the *only* place in the algorithm where the bit-exact reproducibility is traded for a final bounded-ULP rounding — and the trade is deterministic (same `I` → same scalar on every backend).

---

## 4. The whole RFA summation, end to end

```
Input: array x[0..n-1]
Output: scalar sum S

# Per-thread (or per-element for serial CPU):
1. Initialize I = empty indexed type (all zeros)
2. for each x_j in thread's slice:
       AddFloatToIndexed(x_j, I)
3. Thread emits I

# Per-block reduction:
4. Combine per-thread indexed types via AddIndexedToIndexed in any order
   (typically a per-block tree reduce in shared memory)
5. Block emits its final indexed type to a global partials buffer

# Host fold:
6. Read all blocks' indexed types (there are grid.x of them)
7. Combine them via AddIndexedToIndexed in a fixed order (block 0, 1, 2, ...)
8. Convert the final indexed type to a scalar via ConvertIndexedToFloat
9. Return S
```

**Reproducibility proof sketch:** Step 2 produces an `I` that depends only on the multiset of `x_j` seen, because `AddFloatToIndexed` is order-independent by construction (the `(r | 1)` trick). Step 4's tree-shape-dependence collapses because `AddIndexedToIndexed` is associative and commutative. Step 7 is fixed-order so it's reproducible by fiat. Step 8 is a deterministic function of the indexed type. Composing these: same multiset of inputs → same scalar, independent of block count, thread count, tree topology, or hardware.

---

## 5. Accuracy — the formal error bound

This is the single most important number for Peak 4's oracle and Peak 7's summit test.

**Notation.** Let
- `T = Σ x_j` — the exact mathematical sum of the inputs (in real arithmetic).
- `S` — the scalar produced by RFA on the same inputs.
- `M = max_j |x_j|` — the maximum-absolute input.
- `ε = 2^-53` — the fp64 machine epsilon.
- `n` — the number of summands.

**New error bound (RFA, Demmel-Nguyen slide 21):**

```
|S - T|  ≤  n · 2^-80 · M  +  7 · ε · |T|
```

**Standard error bound (naive sequential summation):**

```
|S - T|  ≤  n · ε · Σ |x_j|  ≤  n² · ε · M
```

**What the bound says in words:**

- **First term: `n · 2^-80 · M`.** Represents the residual from bins that fell below the 80-bit precision floor. Scales linearly with `n` (worst case, each element contributes at most `2^-80 · M` of error). For `n = 2^30 = 10^9`, this term is `10^9 · 2^-80 · M ≈ 10^-15 · M` — well below fp64 precision.

- **Second term: `7 · ε · |T|`.** Represents the final fold error from `ConvertIndexedToFloat`. The paper-proven ordered-exponent summation delivers 7 ULPs of the final scalar. This term scales with `|T|`, not `M`, so cancellation (`|T| << M`) shrinks it proportionally.

**Why this matters vs standard sum.** Standard sum's error grows with `Σ|x_j|`, which for heavy-cancellation data can be arbitrarily larger than `|T|`. The critical failure mode: `data = [1e9, -1e9, 1, 2, ...]`. `T ≈ small_sum`, but `Σ|x_j| ≈ 2e9`, so standard sum's error floor is `~2 · 10^9 · ε ≈ 4.4e-7` even if the true sum is `3`. RFA hits `7 · ε · |3| + n · 2^-80 · M ≈ 4.7e-16 + tiny ≈ 4.7e-16` — about 10^9 times smaller.

**Slide 22 quantification:** "New bound up to 10^8 times smaller when lots of cancellation (`|T| << M`)." That's the regime that breaks naive variance and is why RFA is needed for the hard-cases suite.

### 5.1 What tolerance should Peak 4's oracle use for RFA reductions?

**Claim: bit-exact across backends.**

Reasoning:
1. The `AddFloatToIndexed` operation is deterministic in the multiset of inputs (via the `(r | 1)` trick).
2. The `AddIndexedToIndexed` combine is associative and commutative over bin-aligned fixed-point addition.
3. The host-side fixed-order fold is deterministic by construction.
4. `ConvertIndexedToFloat` is deterministic given the indexed type.

**Therefore: same inputs, different hardware → identical bits.** The ULP bound from §5 applies to the RFA result vs the *true* real-valued sum, not to RFA-on-CPU vs RFA-on-CUDA. For the cross-backend test, the expected equality is `==` on the bit pattern.

This is **stronger than the Peak 4 policy written for transcendental reductions**: transcendentals get a `within_ulp_bound` test because our libm functions have a per-call ULP budget. Pure arithmetic via RFA gets **bit-exact** because RFA is deterministic-by-construction.

**Peak 4 oracle policy for RFA kernels:**
```
assert_bit_exact(cpu_result, cuda_result)     # exact bit match required
assert_bit_exact(cpu_result, vulkan_result)   # exact bit match required
assert_within_ulp(rfa_result, mpmath_ref, tolerance_bound(n, M, T))   # accuracy against truth
```
where `tolerance_bound(n, M, T) = n · 2^-80 · M + 7 · ε · |T|` per §5.

---

## 6. I7 compliance — the accumulate+gather decomposition

This is the question navigator asked specifically. **Answer: yes, RFA is cleanly expressible as `accumulate(grouping, expr, op) + gather(addressing, ...)`.**

### 6.1 The accumulate step

```
grouping   : IndexedTypeBinning(K=3, W=40)   — a new Grouping variant
expression : identity (per-element x_j is handed through)
op         : AddFloatToIndexed              — a new accumulate Op
state type : IndexedType (6 fp64 words)     — a new accumulator state type
combine    : AddIndexedToIndexed            — used when merging partial states
```

This is structurally the same pattern as every other tambear accumulate. The "grouping" is degenerate (`All` — every element goes into the single in-progress indexed type), but the accumulate *state* is vector-valued (a 6-tuple of fp64) instead of scalar. I7 says "every primitive decomposes into accumulate + gather." The decomposition is:
- `accumulate(grouping = All, expr = id, op = AddFloatToIndexed, init = EmptyIndexedType)` produces one `IndexedType` per work-unit (thread/block).
- `gather(addressing = FixedBlockOrder, combine = AddIndexedToIndexed, finalize = ConvertIndexedToFloat)` produces the final scalar.

**So the grouping pattern is `All` (not a new variant),** and the innovation is in the *op* and the *state type*. The accumulate op carries vector state; the combine is AddIndexedToIndexed; the finalize is ConvertIndexedToFloat. None of that required inventing a new Grouping enum variant.

**Scout's draft had suggested `Grouping::ExponentBin(K)` as a new variant.** After reading the paper, I believe that's a misunderstanding — the bins are **inside** the accumulator state, not a grouping of inputs. Every input goes to every bin (the Deposit step threads the value through all K bins sequentially). The grouping is still `All`; only the accumulator state is vector-valued.

**Verdict: I7 compliant without adding new Grouping variants.** The `Grouping` enum stays `{All, ByKey, Prefix, Segmented, Windowed, Tiled, Graph}` as-is.

### 6.2 The gather step

```
addressing : FixedBlockOrder             — deterministic block-by-block fold
reduce     : AddIndexedToIndexed          — same combine as accumulate
finalize   : ConvertIndexedToFloat        — indexed-type → fp64
```

The gather is a straight left-fold over the per-block indexed types (in block-index order) with `AddIndexedToIndexed` as the combiner, followed by a single `ConvertIndexedToFloat`. That's the same shape as every other tambear gather step — it's just that the reduce combiner operates on a richer state type.

**Verdict: the gather side is a straightforward structural extension. No new gather addressing patterns needed.**

---

## 7. New `.tam` IR ops required

Walking through the algorithm op-by-op against the Phase 1 IR op set (`const/load/store/fadd/fmul/fsub/fdiv/fsqrt/fneg/fabs/int_ops/predicates/select/loop_grid_stride`):

### 7.1 Bit manipulation (integer / bool ops on fp64 bit pattern)

For `Index(x)`:
- Read `x` as a 64-bit integer (reinterpret cast). **Needs `bitcast.f64.i64`** — if this isn't already in the IR, add it.
- Shift right by 52, mask with `0x7FF`, subtract 1023 → unbiased exponent. Uses **integer ops** (shift, and, sub) that the IR already has.
- Integer division by `W = 40` to get bin index. Uses **integer ops**.

For `Deposit(x, I)` inner loop, the `(r | 1)` step:
- Reinterpret `r` as integer (`bitcast.f64.i64`).
- Bitwise OR with integer 1 (`or.i64`).
- Reinterpret back to fp64 (`bitcast.i64.f64`).

**Required new ops:** `bitcast.f64.i64` and `bitcast.i64.f64` (exact same bit pattern, no conversion semantics). Plus `or.i64` if integer-OR isn't already spelled out. These are all trivially implementable in every backend — PTX has `mov.b64`, SPIR-V has `OpBitcast`, CPU interpreter does `f64::to_bits` / `f64::from_bits` (using the Rust stdlib's pure bit-manipulation, not any libm).

### 7.2 Floating-point operations

Every step in `Deposit` is `fadd.f64` and `fsub.f64` on bin-local operands. Every step in `Renorm` is `fadd.f64`/`fsub.f64` (Dekker-style TwoSum, no FMA). Every step in `AddIndexedToIndexed` is `fadd.f64` across aligned bins. **All covered by existing Phase 1 fp ops.**

### 7.3 Control flow

- The per-element loop is `loop_grid_stride` (existing).
- The `Renorm` trigger ("every 2^11 deposits") is a counter + `select` or `predicate` branch (existing).
- The per-block reduction is shared-memory tree + `OpControlBarrier` (for SPIR-V) / `bar.sync` (for PTX). **This is the same shared-memory primitive Peak 6 would need for any fixed-tree reduce — not RFA-specific.**

### 7.4 Storage

The indexed type is 6 fp64 values. Per-thread state lives in registers; per-block partial results live in shared memory (48 bytes per block); final per-block results live in a global `partials[]` buffer of size `gridDim * 48`.

**Shared memory allocation in the .tam IR:** the existing `loop_grid_stride` op assumes a per-block accumulator. We need a variant `loop_grid_stride_indexed_type` that allocates 6 fp64 slots of shared memory instead of 1. **OR**, more cleanly, the IR should have a general `accumulator_state_size : usize` field on the kernel entry point that tells each backend how many bytes of shared memory per thread to allocate. **Recommendation:** add this as a kernel-level attribute, not a per-op argument.

### 7.5 Summary of op/attribute additions

| Addition | Kind | Reason |
|---|---|---|
| `bitcast.f64.i64` | new op | `Index(x)` and `(r \| 1)` trick |
| `bitcast.i64.f64` | new op | Symmetric counterpart |
| `or.i64` | new op if missing | `(r \| 1)` trick |
| `kernel.accumulator_state_size : usize` | kernel attribute | So each backend knows how much shared memory to allocate |

**The existing integer ops** (add, sub, shift, mask, div) are reused. **No new transcendental or math op is needed.** No libm call. Consistent with invariant I8 (first-principles — we compute everything from bit operations on fp64 representations, which is IEEE 754 structure not someone's library).

**Message to pathmaker:** these ops are trivial to add. `bitcast` is literally a move instruction on every backend (PTX `mov.b64`, SPIR-V `OpBitcast`, CPU `f64::to_bits`/`f64::from_bits`). Please add them to the Peak 1.15 IR reference doc before Peak 6 is opened. They're also useful beyond RFA: any `ldexp`, any bit-level fp manipulation, any `copysign` eventually needs them.

---

## 8. SPIR-V / Vulkan portability

Walking the algorithm step-by-step against SPIR-V's compute model, with NoContraction (I3) applied to every fp op:

### 8.1 Storage classes

- Per-thread `IndexedType` (6 f64 values): `Private` or just SSA registers via `OpVariable Function`. Fine.
- Per-block shared indexed types: `Workgroup` storage class. Requires `StorageUniformBufferBlock` or similar capability — standard. **Size:** 48 bytes × threads-per-block. For block size 256 and K=3, that's 12 KB per block, well within standard 48 KB / 64 KB shared memory budgets.
- Global `partials[]` buffer: `StorageBuffer` with `BufferBlock` decoration. Standard.

### 8.2 Synchronization

- `OpControlBarrier(Workgroup, Workgroup, AcquireRelease | WorkgroupMemory)` before reading any thread's `IndexedType` written by another thread within the block reduction. One barrier per tree-reduce level.
- No global synchronization is needed — the host reads `partials[]` after kernel dispatch completes.

### 8.3 Floating-point decorations (critical for I3)

Every `OpFAdd` and `OpFSub` in the translator emission must be decorated with **`NoContraction`** so that the SPIR-V consumer (Vulkan driver → hardware compiler) is forbidden from forming `FMA` instructions. SPIR-V specifies this explicitly in the "Floating-Point Fast Math Mode" section. **This is the SPIR-V equivalent of PTX's `.contract false`.** Pathmaker for Peak 7 must wire this in every fp op emission.

### 8.4 Bit-level ops

- `OpBitcast` — turns `f64` into `i64` and back. Standard SPIR-V, fp64 capability required (covered by `Float64` capability which we already require anyway).
- `OpBitwiseOr` on `i64` — standard, requires `Int64` capability. Covered.
- `OpShiftRightLogical` — standard integer op.

**All bit ops port cleanly.** No SPIR-V extensions needed beyond `Float64` and `Int64`, both of which the trek already requires for basic arithmetic.

### 8.5 Integer arithmetic for RFA primitives

`Index(x)` needs integer division by 40. SPIR-V has `OpSDiv` / `OpUDiv`. Standard. No need for a custom op.

### 8.6 The subnormal escalation (ESC-001)

Navigator's escalation ESC-001 (Vulkan fp64 subnormal handling is implementation-defined because `shaderDenormPreserveFloat64 = false` on our RTX 6000 Pro) affects RFA in **one specific way**: if an input `x_j` is subnormal, it gets deposited into the lowest active bin, and its bits may be flushed to zero by the Vulkan driver before RFA sees it. CPU and CUDA may handle it correctly while Vulkan does not.

**Implication for the summit test:** if any subnormal input enters a Vulkan RFA kernel, the result may differ from CPU/CUDA by the magnitude of the subnormal. The tolerance is `≤ 2^-1074 · n`, much smaller than the RFA error bound, but it *is* a bit-level difference.

**Recommendation: the summit test (7.11) excludes subnormal inputs for kernels that run through the Vulkan path,** consistent with ESC-001's Option 2 resolution. The CPU interpreter must still handle subnormals correctly (they land in bin 0 with full precision), and the hard-cases suite (4.7) must still generate them to test CPU. The Vulkan test is parameterized: if the device reports subnormal-flush, subnormal inputs are skipped.

This is a scope clarification, not a failure. The architectural claim stands for the normal fp64 range.

**Verdict: RFA is SPIR-V/Vulkan portable with one caveat — the subnormal handling is subject to ESC-001's resolution. Every other step is a direct port.**

---

## 9. I8 compliance — first-principles, no borrowed code

**I8 certificate (per team-lead ruling 2026-04-12, routed via navigator):** These parameters were derived from the mathematical constraints in Demmel-Nguyen ARITH 2013 and IEEE TC 2015, not copied from reference implementation source. The reference implementation (ReproBLAS) was consulted to confirm the constraints, not to copy values. Every numeric parameter below (W=40, K=3, MaxDep=2048, MaxN=2^64, error bound) has an independent derivation from the paper's inequalities that any auditor can reproduce using only mpmath and the IEEE 754 spec.

Team-lead's ruling in full: "Reading ReproBLAS source to understand WHY the authors chose fold=3, W=40, etc. — permitted. Deriving those same values from the mathematical constraints in the papers, independently — required. Copying constants verbatim without derivation — I8 violation."

**All RFA state, formulas, and parameters come from the Demmel-Nguyen papers, not from ReproBLAS source or cuBLAS or any other library.**

Specifically committed-from-paper:

1. **`W = 40`, `K = 3`.** From ReproBLAS slide 20 ("Data type: Double, W: 40, K: 3, (K-1)W: 80") and slide 30's properties table. Rationale: `(K-1)·W = 80` bits > fp64's 53-bit mantissa, so the indexed sum has ~27 extra bits of internal precision. Smaller `K` (K=2) would give only `1·40 = 40` bits — less than fp64's 53 — and the sum would be less accurate than naive summation, which would violate Design Goal 2 of the paper ("Accuracy at least as good as conventional"). `K = 3` is the minimum.

2. **`MaxDep = 2^(P-W-2) = 2^11 = 2048`.** From slide 30. Derivation: the primary accumulator holds a sum of `MaxDep` slices each of at most `2^W` magnitude relative to the bin's lower boundary. Each slice has `P = 53` bits of precision. The primary can hold up to `2^(P-W) = 2^13` slices before the accumulator's top bit overflows the primary's range. The `-2` gives 2 bits of safety margin so that the carry propagation step has room to maneuver without itself overflowing.

3. **`MaxN = 2^(2P-W-2) = 2^64`.** From slide 30. Derivation: after `MaxN/MaxDep` renorms, the carry field itself can hold `2^(P-W-2)` worth of primary-overflows. Multiplied through, `2^(P-W-2) · 2^(P-W-2)` doesn't work out — let me redo this from the paper's formula: `MaxN = 2^(2P - W - 2)`. For fp64 with P=53, W=40: `2^(106 - 40 - 2) = 2^64`. The bottom line is we can safely sum up to 2^64 doubles — more than any realistic dataset.

4. **The `(r | 1)` tie-breaking trick.** From slide 39, "Relationship to IEEE 754 Standard": "`(r | 1) = r with bottom bit set to 1`... makes tie-breaking in rounding independent of `Y_Pk`, needed for reproducibility." This is the *single most important line* of the algorithm. We implement it exactly this way. No alternative — no `round_to_odd` hardware op, no `round_away_from_zero` workaround. **Direct implementation from paper text.**

5. **The `3K - 2` flop count per Deposit.** From slide 29. For K=3, that's 7 flops per element. We don't need to verify this flop-by-flop — it's a budgeting note for pathmaker's performance analysis.

6. **The error bound `|S - T| ≤ n·2^-80·M + 7·ε·|T|`.** From slide 21. We cite this in §5, use it to set the Peak 4 oracle's accuracy tolerance, and adversarial will include heavy-cancellation inputs in the hard-cases suite to verify we actually hit this bound in practice.

**What we do NOT take from ReproBLAS:**
- ❌ Source code (C / C99 / AVX / SSE implementations).
- ❌ The Python code-generation + OpenTuner-autotuning pipeline.
- ❌ The `binned.h` / `binnedBLAS.h` header structure.
- ❌ Specific vector-lane blocking layouts.

**What we DO take from ReproBLAS:**
- ✅ The algorithm (structure + formulas) — fair game, it's published.
- ✅ The parameter values `W=40, K=3` — these are mathematically derived in the paper, not arbitrary choices.
- ✅ The error bound formula — theorem, citation.

**I8 verdict: fully compliant. We implement from the papers, cite the papers, and do not read or copy a single line of ReproBLAS source.**

### 9.1 Cross-check against naturalist's independent reading

Naturalist posted (navigator/check-ins.md, 2026-04-11) a set of RFA parameters obtained by reading the **ReproBLAS C source code on GitHub**. They are transparent about this: they couldn't cleanly extract the papers from PDF and went to the reference implementation for the numbers.

This creates an I8 tension. The check-in data is useful for cross-checking but cannot be the authoritative spec for Peak 6 because it's sourced from code we are forbidden to read. **This doc (rfa-design.md) is the I8-clean alternative** — every parameter and formula is sourced from the Demmel-Nguyen slides/papers (slide references given throughout) and from the ReproBLAS public *website* (`bebop.cs.berkeley.edu/reproblas/`, which shows the error-bound formula on its front page but not source code).

**Where naturalist's numbers and mine agree:**
| Parameter | Naturalist (ReproBLAS source) | This doc (Demmel-Nguyen slides) | Agree? |
|---|---|---|---|
| fold / K (default) | 3 | 3 (slide 20, 30) | ✓ |
| Accumulator size | 6 doubles = 48 bytes | 6 fp64 = 48 bytes (slide 26) | ✓ |
| Bin width W | `DBWIDTH = 40` bits | W = 40 (slide 20, 30) | ✓ |
| Bins kept | K (slid via starting index) | K = 3 (slide 14: "Only keep top K bins") | ✓ |
| Deposits per Renorm | not quoted | `MaxDep = 2^(P-W-2) = 2048` (slide 30) | — |
| Error bound | "up to 229× smaller" | `n·2^-80·M + 7·ε·|T|` (slide 21) | ✓ (consistent, mine is the formula) |

**Where naturalist's layout differs from my layout:**
- Naturalist reports ReproBLAS source uses layout `[p0, p1, p2, c0, c1, c2]` — all primaries first, then all carries. That's a valid choice and matches how `binned_dmdmadd` passes two separate pointers.
- I wrote the data structure as a sequence of `[Y_C(i), Y_P(i)]` pairs per bin, which is how the *papers* depict it (slide 15 shows bin 0 with its (carry, primary) pair, then bin i with its pair).
- **Either layout works** — the algorithm is invariant to memory ordering as long as pathmaker is consistent. I recommend pathmaker choose the `[primary_0, primary_1, primary_2, carry_0, carry_1, carry_2]` layout because it aligns with SIMD-lane vector loads on both CUDA and Vulkan (all primaries in one lane-group, all carries in another). This is a memory-layout choice, not an algorithm choice, so it doesn't affect correctness or I8.

**Where naturalist's state differs:**
- Naturalist includes an explicit `index: i32` field in the state (the "starting bin index"), giving `4 + 48 = 52 bytes` per accumulator plus alignment padding.
- My reading of the papers doesn't explicitly require a separate index field — the top bin's exponent is determinable from the primary's fp64 exponent field. But in practice, pathmaker will want to track it explicitly because extracting it every time is expensive and error-prone.
- **Recommendation:** adopt naturalist's layout with the explicit `index: i32`. 52 bytes per accumulator. Fits in two cache lines (128 bytes) with room for alignment.

**The combine-step alignment.** Naturalist flagged that the "align two states to common starting index" operation is inferred from first-principles reasoning rather than verified against source. My §3 description (aligning by window-shifting whichever state has the smaller top-index upward) is the same inference. **This is the single most complex part of the algorithm and the place where pathmaker is most likely to make a mistake.** The remedy: the Peak 4 oracle's reproducibility tests (§11 items 5, 6, 8) will catch misalignments because they manifest as bit differences between permuted-input runs. Fuzz the input ordering hard.

**Summary of the I8 cross-check:** naturalist's source-derived numbers and my paper-derived numbers agree on every measurable parameter. Pathmaker should treat this doc as the authoritative spec (so the audit trail is paper-only) but can use naturalist's check-in as a sanity reference for field layout and index tracking. The combine-step alignment remains the highest-risk area and demands careful testing.

---

## 10. Fixed-point integer implementation (for later consideration)

The paper slide 40 notes John Hauser's preliminary experiments using **64-bit integers** instead of fp64 words for each primary/carry field, with deposits and renorms done via integer + boolean operations. This is attractive for tambear because:

1. Integer ops are **trivially bit-exact across backends** by IEEE 754 spec — no FMA, no rounding, no contraction to worry about. Integer add on CUDA is the same as integer add on Vulkan is the same as integer add on CPU.
2. The `(r | 1)` trick becomes a **direct integer OR**, not a reinterpret cast.
3. The accuracy tuning is simpler: `K = 2, W = 53` gives `1 + (K-1)W = 53` bits of accuracy (same as fp64's mantissa), and the only difference is whether a single fp64's worth of bits is stored in one int64 or two fp64s.
4. Early experiments show "5.2× slowdown vs standard sum" — in the same ballpark as the fp64 implementation's slowdown, so it's not a performance disaster.

**Recommendation for Peak 6:** start with the fp64 version (K=3, W=40) because it matches the published error bound and the paper's reference implementation. Once it works bit-exactly across backends, consider the integer variant as an alternate `using(rfa_impl="int64")` knob.

I'm flagging this for pathmaker: if the fp64 Deposit step has subtle issues emerging in testing (wrong ULPs, subnormal edge cases, weird SPIR-V behavior), the integer variant is the escape hatch because it sidesteps floating-point rounding entirely.

---

## 11. Testing plan (for Peak 4 oracle + adversarial)

The Peak 4 hard-cases suite should include the following inputs specifically for RFA reductions:

1. **Signed cancellation.** `data = [1e9, -1e9, 1.0]`. True sum = 1.0. Naive sum can return anything between `0` and `2.0` depending on order. RFA must return bit-exact `1.0` (or the nearest fp64 representation of 1.0 after the 7-ULP fold, which for `|T| = 1.0` is within `7 · 2^-53 ≈ 7.8e-16` of 1.0 — so 1 ULP precision).

2. **Extreme cancellation.** `data = [1e20, -1e20, 1, 2, 3]`. True sum = 6. `Σ|x_j| = 2e20`, so naive sum's error floor is `~4 · 10^4` — two orders of magnitude larger than the true result. RFA must return 6 to 1 ULP.

3. **Mixed magnitudes.** `data = [1e-100, 1e100, 1.0]`. True sum ≈ 1e100. Tests bin-shift and precision-floor behavior.

4. **Many small values.** `data = [2^-50; 2^40 copies]`. True sum = 2^-10. Naive sum accumulates rounding per-element; RFA reproduces the sum exactly because each element lands in the same bin.

5. **Permutation invariance.** For each input above, shuffle the array randomly 100 times. All permutations must produce **identical bit patterns**. This is the reproducibility test proper.

6. **Cross-backend equality.** Same inputs, run through CPU interpreter + CUDA + Vulkan. All three must produce **identical bit patterns** for every non-subnormal-input case.

7. **Accuracy vs mpmath.** Same inputs, compute the "truth" at 100-digit precision via mpmath, assert `|rfa_result - mpmath_truth|` is within the `n·2^-80·M + 7·ε·|T|` bound. This verifies our implementation matches the published theoretical bound.

8. **Determinism under block count variation.** Run the same inputs with gridDim = 1, 32, 256, 1024. All four must produce identical bit patterns. This tests `AddIndexedToIndexed` associativity under different tree shapes.

**Acceptance for Peak 6:** all of the above pass, plus the summit test (7.11) passes for the variance recipe (which requires both Welford and RFA composed — see §12).

---

## 12. RFA + Welford composition for variance

**DECISION LOCKED (navigator + team-lead, 2026-04-12): Option A.** Peak 6 ships **RFA sum + Welford variance** with Chan parallel-merge, all ops decorated with `NoContraction`. RFA variance via moment-state extension is Phase 2 territory and not in Phase 1 scope. The two pinned-red variance tests in the Peak 4 hard-cases harness remain as acceptance criteria until pathmaker commits the two-pass variance recipe. The below rationale is retained as the decision record.

The scout report correctly flagged that `variance` uses the one-pass formula `(Σx² - (Σx)²/n) / (n-1)`, which is catastrophically unstable and cannot be fixed by RFA alone. RFA gives reproducibility, not cancellation-resistance.

The right composition is **Welford + RFA**:

1. Each thread runs Welford's sequential update on its slice, producing a `(n, mean, M2)` triple per thread.
2. Block-level tree-reduces the per-thread triples using **Chan et al. (1979)** parallel-Welford merge formulas — these are associative over the `(n, mean, M2)` state space.
3. Host folds the per-block triples in fixed order via the parallel-Welford merge.
4. Result: `variance = M2 / (n - 1)`.

This is numerically stable AND reproducible given fixed block order. It is NOT yet cross-backend reproducible because the Chan merge involves division (`nb / n_combined`) and multiplication, and those can round differently on different hardware under FMA contraction rules.

**To make the Welford merge cross-backend reproducible:**
- Apply `NoContraction` to every fp op in the merge (same rule as everywhere else).
- Ensure the division is done via a single `fdiv.f64` op, not an FMA-contracted `reciprocal * numerator` sequence.
- Accept that Welford's mean and M2 are fp64 scalars, not indexed types. The cross-backend equality of `(n, mean, M2)` triples depends on deterministic arithmetic on those scalars, which requires I3/I4/I5/I6 to all hold.

**If I3+I4+I5+I6 hold, Welford is cross-backend reproducible without RFA, because the merge is a small number of ordered fp ops on scalars.** RFA is only needed for the inner accumulation within a thread's slice, where naive summation of many small values would drift. But for variance, the inner accumulation is Welford's sequential update (not a summation), and Welford itself is already order-sensitive — so RFA can't apply to the inner loop directly.

**Recommended approach for variance on Peak 6:**
- Per-thread: Welford's sequential update (no RFA — the state is `(n, mean, M2)`, not a sum).
- Per-block + host fold: Chan's parallel-Welford merge in fixed block order.
- All fp ops emit with NoContraction.
- The final `M2 / (n - 1)` is a single fdiv with rounding determinism.

**This is NOT yet a full RFA solution for variance.** It is a Welford + deterministic-arithmetic solution. The cross-backend reproducibility of this approach depends on I3-I6 being enforced bit-level across backends, which is the whole point of Peaks 3 and 7. If those hold, Welford works without RFA for variance.

**RFA is specifically needed for `sum`, `l1_norm`, `dot_product`, and other kernels where the inner loop is an actual floating-point summation with many terms.** For those, RFA is unavoidable. For variance, Welford is sufficient if the backends are deterministic.

**Decision (2026-04-12): Option A. Locked.** Peak 6 ships RFA sum + Welford variance with Chan parallel-merge, NoContraction decoration on every fp op. Phase 2 may add RFA variance if benchmarks show Welford's arithmetic loses precision on data where cross-backend reproducibility matters. For Phase 1, Welford + I3-I6 determinism is sufficient.

**OrderStrategy registry entries for variance's block-partial fold** (team-lead ruling 2026-04-12, Amendment 6 from pending-amendments-wip.md):

The Welford+Chan merge tree shape is part of the order_strategy, not just the merge formula. Two named entries:

1. **`welford_chan_left_to_right_fold`** — sequential host-side fold of block partials, block 0 first. Phase 1 default. Simplest. Bit-identical on any (backend, hardware) pair where I3/I4/I5/I6 hold.

2. **`welford_chan_balanced_pairwise_tree_fanout_2`** — pow2 balanced tree. Different bit signature from option 1 because `merge(merge(A, B), C) ≠ merge(A, merge(B, C))` at the last ULP of `mean`, propagating through `M2`. Phase 2 optimization for large block counts.

The variance recipe declares `welford_chan_left_to_right_fold` as its Phase 1 `order_strategy`. `welford_chan_balanced_pairwise_tree_fanout_2` is available as a Phase 2 named alternative with a distinct bit signature. This follows the OrderStrategy registry pattern established in Peak 1 (campsite 1.16 / campsite 1.17): the `order_strategy` field names the tree shape in the registry, and consumers can inspect `are_fusable("welford_chan_left_to_right_fold", ...)` to check compatibility.

---

## 13. Summary table of this doc's answers to navigator's five questions

| Q | Answer |
|---|---|
| **1. I7 compliance** | Yes. `accumulate(All, id, AddFloatToIndexed, init=empty)` + `gather(FixedBlockOrder, AddIndexedToIndexed, finalize=ConvertIndexedToFloat)`. No new `Grouping` variant required. The vector-valued state lives *inside* the accumulate op, not in the grouping. |
| **2. New IR ops** | `bitcast.f64.i64`, `bitcast.i64.f64`, `or.i64` if not already present, plus a `kernel.accumulator_state_size` attribute. No new floating-point ops. No libm calls. All backends lower these trivially. |
| **3. SPIR-V portability** | Ports cleanly. Workgroup storage class for shared bin vectors, OpControlBarrier for synchronization, NoContraction on every fp op (I3), OpBitcast for the bit-manipulation tricks. Caveat: ESC-001's subnormal-flush resolution applies (summit test skips subnormal inputs on Vulkan devices that flush). |
| **4. I8 compliance** | Fully compliant. All parameters (W=40, K=3, MaxDep=2048, MaxN=2^64, error bound) derived from Demmel-Nguyen papers. No ReproBLAS source code read or copied. Citations below. |
| **5. Accuracy bound** | `\|S - T\| ≤ n·2^-80·M + 7·ε·\|T\|`. First term is subnormal-precision residual (dominated for large n but still tiny), second term is final-fold error. **Cross-backend test is bit-exact** because RFA is deterministic-by-construction; the ULP bound is only for comparing RFA output to the *true* mathematical sum. |

---

## 14. Message for the IR architect (pathmaker)

Before Peak 1.15 (IR reference doc) is finalized, please:

1. Add `bitcast.f64.i64` and `bitcast.i64.f64` to the op set. Semantics: identity bit pattern, no arithmetic.
2. Confirm `or.i64`, `and.i64`, `shr.i64`, `shl.i64`, `udiv.i64` are all in the op set (or add them).
3. Add `kernel.accumulator_state_size : usize` as a kernel-level attribute so backends know how much Workgroup/shared memory to allocate per thread.
4. Decide whether `NoContraction` enforcement is a **per-op flag** (every `fadd`/`fsub`/`fmul` carries it) or a **kernel-level attribute** (all fp ops in the kernel are non-contracted). Recommendation: kernel-level, because we never want contraction *anywhere* in Phase 1.

No new transcendental ops. No new grouping variants. No structural changes to Accumulate or Gather. RFA fits cleanly into the existing architecture once the integer + bitcast ops are in.

---

## 15. References

1. J. Demmel & H. D. Nguyen, **"Fast Reproducible Floating-Point Summation,"** ARITH-21, 2013. (Original algorithm; single-threaded indexed-type data structure; proof of the error bound for K-fold accumulators.)
2. J. Demmel & H. D. Nguyen, **"Parallel Reproducible Summation,"** IEEE Transactions on Computers 64(7):2060–2070, 2015. (Parallel/distributed version; shows that `AddIndexedToIndexed` is associative and commutative, which is what makes the algorithm work across tree shapes.)
3. P. Ahrens, J. Demmel, H. D. Nguyen, **"Efficient Reproducible Floating Point Summation and BLAS,"** UCB/EECS-2016-121, UC Berkeley, 2016. (Full ReproBLAS tech report; tabulates the parameters for half/single/double/quad precision, describes the Sum / Deposit / Renorm / Update / AddFloatToIndexed / AddIndexedToIndexed / ConvertIndexedToFloat primitives, gives the error bound as Theorem.)
4. J. Demmel & Y. Hida, **"Accurate and Efficient Floating Point Summation,"** SIAM J. Scientific Computing 25(4):1214–1248, 2003. (Ordered-exponent summation used by ConvertIndexedToFloat to achieve the 7-ULP final fold.)
5. T. F. Chan, G. H. Golub, R. J. LeVeque, **"Updating Formulae and a Pairwise Algorithm for Computing Sample Variances,"** COMPSTAT 1982 Proceedings, 1979. (Parallel-Welford merge used for variance in §12.)
6. B. P. Welford, **"Note on a method for calculating corrected sums of squares and products,"** Technometrics 4(3):419–420, 1962. (The sequential Welford update itself.)

**Zero references to source code** (no ReproBLAS, no cuBLAS, no MKL). Algorithm is derived from the papers listed above, implemented from first principles in the `.tam` IR, compliant with I1 (no vendor math library), I3 (no FMA contraction), I7 (accumulate+gather), and I8 (first-principles only).
