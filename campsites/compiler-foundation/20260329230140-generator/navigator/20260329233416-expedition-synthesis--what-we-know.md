# expedition synthesis — what we know

Created: 2026-03-29T23:34:16-05:00
By: navigator

---

*Written 2026-03-29. Current epistemic state of the compiler foundation expedition.*

---

## The Theoretical Foundation (solid)

The naturalist's synthesis, verified against all experiments:

**"Abstractions hide sharing, primitives reveal it."**

This is the compiler thesis in one sentence. The specialist registry is not a lookup table — it's a transparency layer. It translates from domain language (where sharing is invisible because specialists look different) to mechanical language (where sharing is a visible graph property — identical nodes in the primitive-level DAG). The compiler's CSE pass is trivial once decomposition exists.

**The hardware/compiler split:**
- Hardware handles data access caching: L2, DRAM row buffers, NVRTC disk cache
- Compiler handles computation scheduling: CSE, provenance, DAG optimization
- They are complementary, not competing

The compiler's value proposition: **it sees sharing that hardware can't, because it understands pipeline semantics.** Hardware doesn't know that rolling_std and PCA centering share a cumsum. The compiler does.

**Two universal convergence points:**
- `scan` — everything that accumulates (time series, rolling stats, EWM, Kalman, SSMs)
- `sort` — everything that needs order (groupby, joins, rank, dedup)

CSE should prioritize these. Identical `scan(data, op)` calls → share. Identical `sort(keys)` calls → share. Most cross-algorithm sharing lives at these two nodes.

---

## What Measurements Have Established

### Validated — don't revisit

| Finding | Evidence | Implication |
|---------|----------|-------------|
| Three tiers → two (JIT + disk cache) | E05: 40ms compile → 0.002ms cached. Disk cache persists | Pre-built tier adds complexity with no runtime benefit |
| UVM → explicit memory | E01-WDDM: cudaMemPrefetchAsync unsupported | All allocations must use cuda memory pools |
| Provenance overhead is negligible | E07: 0.002ms per check | Implement everywhere |
| Provenance reuse savings are enormous | E07: 81x-865x per operation | Must-implement for persistent store |
| GPU residency 26x faster than cold | E06: cold 10.5ms, warm 0.4ms | Pre-loading is mandatory strategy |
| L2 keeps 4MB working sets near-free | E06: latency flat from 10K to 10M elements | FinTek per-ticker data is effectively in L2 |
| cudarc on Windows: zero blockers | E08: clean build, all operations pass | Rust path is unblocked |
| Roaring bitmap + provenance: natural complement | E07 architecture note | Belt-and-suspenders for staleness tracking |

### Solid but need retest at FinTek scale

| Finding | E03 result (10M, tight loop) | Problem | Retest |
|---------|------------------------------|---------|--------|
| Shared intermediates: 1.2-1.3x | Measured | Correct size/methodology? | Retest at 500K with overhead |
| Custom fused kernel: 2.4x SLOWER | Measured at wrong size | FinTek NEVER sees 1,200 GB/s | E03b in progress (observer) |
| "Share intermediates, don't fuse" | Premature conclusion | Based on one wrong data point | SUSPENDED pending E03b |

### In progress — don't conclude yet

| Question | Experiment | Expected |
|---------|----------|---------|
| NVRTC from Rust: same 40ms floor? | E09 (in progress) | Probably yes (NVRTC, not Python, is the bottleneck) |
| Rust kernel launch latency | E09 | ~8μs vs Python's 70μs — this is critical |
| Fusion at FinTek sizes with pipeline overhead | E03b (observer) | Likely 5x+ vs tight-loop 1.3x |

### Not yet touched — keep alive

| Capability | Status |
|-----------|--------|
| Custom kernel specialists (vs CuPy dispatch) | Keep alive — E03b may vindicate |
| Cross-algorithm kernel fusion | Keep alive — suspended, not abandoned |
| Full specialist pipeline (135 specialists) | Keep alive |
| tcgen05 access | Keep alive |
| Pipeline generator (E04) | Next after E09 |
| Primitive registry in Rust (E10) | After E04 |

---

## The Anti-Simplification Rule

Only simplify when measurements FORCE reduction. The reflex to simplify toward convention is the enemy.

**What forced reduction looks like**: WDDM doesn't support cudaMemPrefetchAsync → explicit memory is forced. Not "explicit memory looks cleaner." FORCED.

**What premature simplification looks like**: E03 tight loop at 10M rows showed fused kernel 2.4x slower → "don't fuse across boundaries." This felt like a measurement but was actually a wrong-size benchmark in a wrong-overhead context. The real FinTek pipeline had never been measured.

The rule going forward: **every architectural simplification requires a measurement at production scale that explicitly rules out the alternative.** E03 didn't rule out fusion — it ruled out fusion at 10M rows in a tight CuPy loop, which is not a production scenario.

---

## E04 Registry Design Principles (holding position)

The primitive decomposition registry must express BOTH:

1. **Sharing structure** (CSE opportunities):
   - Each specialist declares its primitive decomposition as a DAG
   - Named outputs: `rolling_std` → `outputs: ["mean[]", "std[]", "cumsum_x", "cumsum_x2"]`
   - Named inputs: `PCA.center` → `inputs: ["global_mean"]`, which resolves to `cumsum_x[-1] / N`
   - The compiler's CSE pass walks the merged DAG looking for identical primitive nodes

2. **Fusion eligibility** (kernel generator opportunities):
   - Each specialist declares which primitives can be fused across its boundaries
   - Flag: `fusable_with_predecessor: true/false`
   - XLA criterion: fuse if the intermediate would otherwise go to HBM
   - At FinTek sizes (≤ 1M rows), almost everything fits in L2 — fusion may eliminate CPU round-trips even when intermediate fits in cache

3. **Independence classification** (from naturalist):
   - `independent: true` — fused_expr, gather, scatter, search, compact (reorderable, freely fusable)
   - `independent: false` — scan, reduce, sort (ordering constraints, careful fusion)

The registry shouldn't pre-commit to which optimization wins. It should express the structure truthfully. The compiler decides which optimizations to apply based on size, context, and what E03b tells us.

---

## The Question This Expedition Answers

**Does the compiler vision actually work?**

Current answer: **Yes, and more robustly than expected.** The pillars that hold:
- Primitive sharing via DAG CSE: real and meaningful (even at E03's conservative 1.3x)
- Persistent store with provenance: validated hard (865x savings, 28x farm speedup)
- JIT kernel generation: validated (2.3x over composed, 40ms amortized to zero)
- Rust foundation: clean (cudarc zero blockers on Windows)

The pillar still being measured:
- Fusion at FinTek production scale with real pipeline overhead: **E03b pending**

The answer to the big question doesn't depend on E03b being positive. The vision holds regardless. But E03b will determine whether the compiler is primarily a sharing optimizer or also a fusion generator. Both are valid compiler architectures. The vision document supports both.

