# Integer Encoding Research — Scout Notes
## 2026-03-27

### The Core Question
Does avoiding float entirely (or reducing to float32) speed up GPU computation AND reduce I/O?

Short answer: **Yes, significantly.** 2x smaller files, 1.6-1.9x faster writes, and on GPU the
dequantize cost is essentially zero.

---

### Precision Analysis

**Price → int32 at 10^-4 scale ($0.0001 precision)**
- int32 max at this scale: $214,748.36 (covers all US equities)
- Zero round-trip error for prices with ≤4 decimal places
- Example: $230.88 → 2308800 → $230.8800, err=0.00
- This IS the standard tick encoding used in HFT firmware/FPGA ring buffers

**Size → int32 (natural)**
- Shares are already integers. No precision loss.
- float64→int32: 8B→4B, 2x compression, zero error.

**Timestamps → int64 absolute (or int32 delta)**
- Already int64. No conversion.
- Delta encoding: avg gap 50-100ms = 50-100M ns. int32 max = 2.147B ns = 2.1 seconds.
- int32 delta covers ~100% of intraday ticks (p99.9 < 0.7s for normal trading).
- Caveat: pre/post-market gaps can be hours. Use int64 sentinel (e.g., 0xFFFFFFFF = absolute follows).

**Derived features (log, sqrt, recip) → float32**
- float32 vs float64 error for log(price): max 3.36e-07, mean 1.20e-07
- Relative precision: 2.39e-08 (i.e., 8 significant digits → fine for signal computation)
- 8B→4B, zero semantic loss at signal level

**Delta prices → int16 at 10^-4**
- Typical delta: $0.01-$0.50. At 10^-4 scale: 100-5000. int16 max: 32767 = $3.2767.
- Use int32 if expecting larger moves; int16 for normalized/bounded deltas.

**bfloat16 verdict: NO**
- bfloat16 max error for log(price): 3.12e-02 (3 cents on a $230 price). Unacceptable.
- float32 is the minimum for derived features. bfloat16 only for embeddings/ML weights.

**uint16 quantized (per-column min/max):**
- For tight range (AAPL $220-$240): LSB = $0.00030 — better than tick size!
- Requires min/max known before write. Suitable for batch (whole-day known range).
- Not suitable for streaming/append.

---

### File Size

Full schema (18 columns, 598K rows):

| Scheme | Bytes/row | File size | vs baseline |
|--------|-----------|-----------|-------------|
| All float64 | 129 | 81.9 MB | 1.0x |
| Integer/float32 mix | 67 | 40.1 MB | 2.04x smaller |

Per year (4,604 tickers × 250 days):
- float64: 88.8 TB
- integer/float32: 46.1 TB
- **Savings: 42.7 TB (48%)**

---

### Benchmark Results

CPU proxy (includes dequantize on read):
```
float64 scheme: write=27.6ms  read=18.6ms  size=81.9MB
integer scheme: write=16.8ms  read=9.9ms   size=40.1MB
Write speedup: 1.64x
Read speedup:  1.87x (includes CPU dequant)
```

On GPU, dequantize is ~0 cost:
- int32→float32 conversion: 1 CUDA instruction (`__int2float_rn`)
- Scale multiply: 1 FMA instruction
- Fused with first memory access in kernel → no extra kernel launch needed

---

### GPU Integer Arithmetic

RTX 6000 Pro Blackwell theoretical:
- FP64: ~3.9 TFLOPS (**1/32 of FP32 — huge penalty**)
- FP32: ~125 TFLOPS
- INT32: ~125 TOPS (same CUDA cores)
- Memory bandwidth: ~896 GB/s

For memory-bandwidth-bound element-wise ops:
- float64: 598K × 8B = 4.8MB → 5.3ms GPU transfer
- float32/int32: 598K × 4B = 2.4MB → 2.7ms GPU transfer
- int16: 598K × 2B = 1.2MB → 1.3ms GPU transfer

**The pipeline is I/O bound. Halving bytes = halving wall time.**

---

### Recommended Encoding Scheme per Column

| Column | Storage | Notes |
|--------|---------|-------|
| price | int32 | 10^-4 scale |
| size | int32 | natural integer |
| timestamp | int64 | absolute (or first + int32 deltas) |
| notional | int64 | 10^-4 scale |
| ln_price | float32 | 7 sig digits, fine |
| sqrt_price | float32 | same |
| recip_price | float32 | same |
| delta[1-5] | int16 | 10^-4 scale (clip at ±$3.27) or int32 |
| rolling_mean | float32 | aggregated, precision fine |
| rolling_std | float32 | same |
| gap_ns | int32 | ns gap (int64 sentinel for >2.1s) |
| exchange | int32 | unchanged |
| cond_bitmask | uint32 | unchanged |
| is_trf | uint8 | unchanged |

---

### The GPU-Native Dream

If the file format IS the GPU memory layout:
1. Disk: int32 prices, float32 derived, int64 timestamps
2. DirectStorage: DMA directly into CUDA pinned memory (no CPU copy)
3. Kernel: first instruction per lane is `float p = raw[i] * 1e-4f` — this IS the dequantize
4. No separate conversion pass. No extra kernel. No extra bandwidth.

The dequantize IS the kernel's first computation. Zero overhead.

The only kernel that needs changing: replace `float p = prices[i]` with
`float p = __int2float_rn(iprices[i]) * 1e-4f`. One extra multiply per element per kernel.
At 598K elements on a Blackwell chip, this costs ~5 microseconds.

---

### Open Questions for Pathmaker

1. What's the actual write speed to NVMe vs tmpdir? (tmpdir benchmark above may be RAM-backed)
2. Does the fused-stats kernel need float64 accumulator for std calculation?
   (float32 accumulator may lose precision for large n — Kahan summation?)
3. For the delta timestamp scheme: is int32 sufficient or do we need the sentinel approach?
4. Should notional be recomputed from price×size on GPU rather than stored? (saves 8B/row)
