# expedition orientation

Created: 2026-03-29T23:05:09-05:00
By: navigator

---

## What Phase 1 Already Proved (Don't Re-Prove)

E011 demonstrated **sort-once-derive-many** at the groupby scale: one argsort feeds sum, mean, count simultaneously. E02's question is answered in principle. The experiment should confirm it cleanly and measure the bandwidth math — 1 pass vs N — to establish the quantitative basis for the compiler's fusion decisions.

E010b demonstrated **Python codegen via CuPy RawKernel**: expression tree → CUDA source → NVRTC compile → cached execute. JIT overhead ~0ms. The generator campsite's E05 is 80% built already. What's missing is the registry abstraction.

E006 demonstrated **GPU residency pays off**: all-GPU 0.33ms (87x over pandas) for repeated queries. The persistent campsite needs to formalize this as a store with provenance tracking, not just ad-hoc retention.

## The Genuinely Open Questions

**E01 — Multi-output reduce**: Does sum+mean+std+min+max in ONE kernel beat 5 separate kernels? Theoretically yes (1x data reads vs 5x). But the tradeoff: register pressure, occupancy, L2 reuse. Need the number.

**E03 — Cross-algorithm sharing**: The *true* compiler thesis test. Rolling_std feeds PCA centering. When algorithm A outputs intermediates that algorithm B needs, does the compiler *see* this and eliminate redundant computation? No prior experiment touched this. The only genuinely unknown thing in this campsite.

**E08-E10 (rust campsite)**: Entirely unexplored. cudarc on Windows, PyO3 + CUDA roundtrip, NVRTC from Rust. The production path has never been touched.

## Recommended Attack Order

1. **E01** (multi-output reduce): ~2 hours, pure Python/CuPy, establishes quantitative basis
2. **E03** (cross-algorithm sharing): The core thesis. Design carefully — what intermediate is shared, how does the graph represent it?
3. **E02** (sort-once-use-many): Confirmation of E011 at the primitive level. Should be fast.

## A Note on E03's Design

The setup: `rolling(window=5).std(A) → PCA.center(with=mean_from_rolling)`

The shared intermediate is the **running mean** computed during rolling_std. PCA centering needs the same mean. If the compiler detects this, PCA centering costs zero extra — it reuses rolling_std's mean.

The deeper question isn't the benchmark number — it's architectural: *can we represent this sharing in the primitive graph?* The registry needs to express "rolling_std produces (mean, std) as named outputs" and "PCA centering consumes (mean) as input, which may come from a prior computation." That representation question matters more than the speedup.

