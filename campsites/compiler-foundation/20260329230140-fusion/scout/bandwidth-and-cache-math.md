# Bandwidth and Cache Math for E01 (Multi-Output Reduce)

*Scout: 2026-03-29*

---

## Hardware Baseline: RTX PRO 6000 Blackwell

- HBM bandwidth: 1,677 GB/s
- L2 cache: ~96-128 MB (RTX 5090 = 96 MB; GB200 = 126 MB; RTX PRO 6000 similar range)
- L1/shared memory per SM: 228 KB (Blackwell compute capability 10.0)
- Register file per SM: 64K 32-bit registers

## Working Set for E01

- 10M float32s = 40 MB
- 40 MB < L2 cache capacity (96-128 MB) → **the entire working set fits in L2**

---

## Single-Pass Multi-Output Kernel

One kernel reads 40 MB once from HBM, accumulates sum+count+mean+min+max+M2 (Welford) per thread, warp-shuffles to produce final values.

- HBM read: 40 MB / 1677 GB/s = **24 μs**
- Kernel launch (WDDM): ~7 μs
- **Total: ~31 μs**

---

## Five Separate Kernels: L2 Reuse Analysis

After kernel 1 (sum) loads 40 MB from HBM, the data sits in L2. Kernels 2-5 may read from L2 instead of HBM.

**L2 bandwidth estimate**: Ada/Blackwell L2 typically 4-8 TB/s effective. Using conservative 5 TB/s.

- Kernel 1 (sum): 7μs launch + 40MB HBM read @ 1677 GB/s = **31 μs**
- Kernel 2 (mean): 7μs launch + 40MB L2 read @ 5000 GB/s = **14.8 μs**
- Kernel 3 (std): 7μs launch + 40MB L2 read = **14.8 μs**
- Kernel 4 (min): 7μs launch + 40MB L2 read = **14.8 μs**
- Kernel 5 (max): 7μs launch + 40MB L2 read = **14.8 μs**
- **Total: ~90 μs** (assuming hot L2 between launches)

Single-pass wins: **~2.9x** under ideal L2 retention.

---

## The Cache Crossover Point

Question: at what data size does the L2-reuse scenario close the gap enough to matter?

Answer: **there is no crossover point where 5 separate kernels win**. Single-pass is strictly better because:

1. **Kernel launch overhead dominates small sizes.** At 1M floats (4 MB, definitely in L2): single-pass = 7μs + 2.4μs = 9.4μs. Five-pass = 5×7μs + 2.4μs + 4×0.8μs = 42.2μs. L2 reuse narrows the HBM gap but launch overhead still makes 5-pass worse.

2. **L2 retention is not guaranteed in production.** In the FinTek pipeline, 8+ leaves are running concurrently. The L2 is shared across all SMs and all concurrent work. Between kernel launches, other leaves will evict the 40 MB. In production, the 5-pass case approaches the full 5× HBM cost: 5 × 31μs = 155μs. Single-pass still 31μs. **5× advantage in production.**

3. **Algorithmic floor**: even with perfect L2 retention, 5 kernel launches vs 1 = 4 extra × 7μs = 28μs wasted on scheduling overhead. The single-pass kernel pays this once.

**Recommendation for E01 experiment:** benchmark at 10M elements (40 MB, fits in L2) AND at 100M elements (400 MB, exceeds L2). The 100M case will show full 5× HBM advantage. The 10M case will show ~3× (still significant). Both numbers tell the story clearly.

---

## Where the 5× Number Comes From (and Why It's Honest)

The 5× claim is the HBM bandwidth ratio: 5 passes vs 1. This is correct when:
- Data does not fit in L2, OR
- Other work runs between kernel launches (evicts L2)

In a production FinTek pipeline: both conditions are typically true. 5× is the honest production estimate.

In a microbenchmark with no concurrent work and data that fits in L2: ~2.9×. Still compelling. Report both.
