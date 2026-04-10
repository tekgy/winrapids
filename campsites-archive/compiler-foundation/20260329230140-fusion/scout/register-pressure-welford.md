# Register Pressure: 5-Accumulator Welford Reduce for E01

*Scout: 2026-03-29*

---

## Welford Algorithm State

Per-thread running accumulators for sum+mean+std+min+max in one pass:

```
// For each element x processed by this thread:
count   += 1                     // u32: 1 register
sum     += x                     // f32: 1 register
delta    = x - wf_mean
wf_mean += delta / count         // f32: 1 register (Welford mean)
delta2   = x - wf_mean
wf_M2   += delta * delta2        // f32: 1 register (Welford M2 → variance)
minimum  = min(minimum, x)       // f32: 1 register
maximum  = max(maximum, x)       // f32: 1 register
```

**Per-thread accumulator state: 6 registers (f32) = 24 bytes**

Plus loop variables and input register: ~3-4 more. Total per-thread overhead: **~10 registers**.

---

## Register File Budget

Blackwell register file: 64K 32-bit registers per SM.

At 256 threads per block, 10 extra accumulator registers = 2,560 extra registers per block. The total registers per block including the reduction kernel scaffolding will be ~30-40 registers per thread × 256 threads = 7,680-10,240 registers per block. Well within budget.

**Occupancy impact**: a well-written reduction kernel targeting 256 threads per block with 30-40 total registers per thread should achieve 4-6 active warps per SM, which is sufficient for the memory latency hiding needed.

**Verdict: register pressure is NOT a concern for E01.** The Welford accumulators add ~10 registers to a kernel that would use ~25 anyway.

---

## Warp-Level Reduction Pattern

After each thread accumulates over its stride-assigned elements, the per-warp reduction uses warp shuffle:

```cuda
// Warp-level Welford merge (from OneFlow LayerNorm pattern):
for (int offset = WARP_SIZE/2; offset > 0; offset /= 2) {
    float other_mean = __shfl_down_sync(0xffffffff, wf_mean, offset);
    float other_M2   = __shfl_down_sync(0xffffffff, wf_M2, offset);
    int   other_count = __shfl_down_sync(0xffffffff, count, offset);
    // parallel Welford merge: Chan et al. formula
    float delta = other_mean - wf_mean;
    wf_M2   += other_M2 + delta * delta * count * other_count / (count + other_count);
    wf_mean  = (wf_mean * count + other_mean * other_count) / (count + other_count);
    count   += other_count;
}
// sum and min/max use standard __shfl_down_sync patterns
```

This is the established LayerNorm warp-reduce pattern (OneFlow, PyTorch custom ops). **No shared memory needed for the warp reduction** — pure register-to-register shuffle.

---

## Practical Implementation Note

The Welford merge formula (Chan et al.) requires care: divide-by-zero check when merging empty counts. In practice: initialize `count = 0`, skip merge if `count == 0`. First element sets mean directly.

For the E01 kernel:
- Grid-stride loop: each thread processes `n/num_threads` elements
- Warp shuffle: reduce 32 threads → 1 warp result
- Shared memory: reduce warp results within block (32 warps max → 32×6 scalars = 192×4 bytes = 768 bytes per block — trivial)
- Block-level output: atomic update to global accumulator, or write per-block partial results and do a second pass

**Second-pass recommendation**: for N > 10M, a two-pass approach (per-block partials → final reduction) is cleaner and avoids global atomic contention. First pass writes 256 partial results; second pass (single block) reduces them. Total kernel launches: 2 vs 5 for the naive approach.
