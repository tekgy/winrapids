# Navigator Synthesis — MKTF Research Complete
Created: 2026-03-27
By: navigator

## What We Established Today

### The Numbers

| Format | Disk→GPU | Universe (4604 tickers) |
|--------|----------|-------------------------|
| Parquet→numpy→GPU | 66.65ms | 306.8s (5.1 min) |
| MKTF float32 full | 5.90ms | 27.2s (0.5 min) |
| MKTF float32 selective (2 cols) | 0.82ms | 3.8s |
| MKTF float32 vs Parquet | **11.3x faster** | **11x faster** |

### The Key Decisions (All Validated by Benchmark)

1. **float32 for price/size in the hot path** — same size as int32, wins on reads, competitive on GPU
2. **Absolute int64 timestamps** — 72K negative deltas in real AAPL data kill delta encoding
3. **No file-level compression** — float data is incompressible (LZ4 ratio 1.000x on price/size)
4. **uint32 bitmask for conditions** — 84% savings vs 26×uint8, zero decode overhead
5. **Ticker as file metadata, not column** — eliminates 7.18MB of "AAPL" × 598K waste
6. **64-byte aligned columns** — confirmed by observer benchmark (10% better GPU pipeline)

### Where Integer Encoding Wins (Nuanced)

GPU compute is largely a wash (within 10-15% for most ops):
- int32 wins **sort** (117μs vs 128μs float32)
- int32/int64 win **product** (notional p*s: ~27μs vs 31μs float32)
- float32 wins **mean/std**
- Everything else is tied

int64 is worth it for: exact arithmetic audit trails. Not worth it for: file size, I/O, general GPU compute.

### The Precision Cliff (bfloat16)

bfloat16 KILLS small return values:
- delta_ln_price at 1bp → 100bp: rounds to exactly 0.0 in bfloat16 at ALL magnitudes
- bfloat16 is ONLY safe for stored levels (ln_price, sqrt_price, recip_price, sin/cos)
- All delta/return columns must be float32

K01P02 potential with mixed precision (estimated):
- All float64: 71.8MB
- Mixed (bf16 for levels, f32 for deltas, int8 for flags): 27.5MB (62% smaller)
- Not yet benchmarked

### What Drives File Size

K01P01 at 35MB → 15MB (56% reduction) breaks down as:
- Ticker column removal: 7.18MB (21%)
- Sequence string → uint32: 5.49MB (16%)
- Conditions string → uint32 bitmask: 4.75MB (14%)
- Exchange int32 → uint8: 1.87MB (5%)
- Type narrowing (is_cancel etc.): small

None of this requires compression. Just correct types.

## Remaining Open Questions

1. **MKTF v3 "self-describing"**: file IS its own catalog entry. Fixed header should carry ticker/date/schema_version as first-class fields (not just in JSON metadata) for fast directory indexing.

2. **K01P02 bfloat16 prototype**: mixed precision for derived columns. float32 for deltas, bfloat16 for levels. Estimate: 27.5MB vs 71.8MB all-float64.

3. **DirectStorage path**: NVMe → GPU VRAM. Requires 4096-byte column alignment (NVMe sector). Currently using 64-byte. May be worth testing.

4. **Multi-ticker batch read**: read 100 tickers concurrently to saturate NVMe. Async I/O + CuPy stream pipelining.

5. **Rust/C reader**: for maximum throughput, a native reader vs Python overhead.

## Architecture Note

The pipeline at this point:
- GPU compute: 1.5ms per ticker
- MKTF file I/O: 5.9ms per ticker (disk→GPU)
- **I/O is the bottleneck** (80% of total per-ticker time)
- To get below 2ms per ticker: need concurrent reads OR DirectStorage OR smaller files

The real lever for sub-2ms is selective reads — if K01P02 only needs prices+sizes for some passes, 0.82ms disk→GPU for 2 columns is already sub-millisecond. The format supports this. The pipeline just needs to use it.
