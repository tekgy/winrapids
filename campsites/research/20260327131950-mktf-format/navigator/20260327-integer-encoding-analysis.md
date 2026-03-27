# Integer / Reduced-Precision Encoding Analysis
Created: 2026-03-27
By: navigator
Context: team-lead research direction — explore avoiding float entirely

## The Critical Bfloat16 Finding

**Bfloat16 kills delta_ln_price (returns).** At every realistic return magnitude:

| Return | True value | BF16 | F32 |
|--------|-----------|------|-----|
| 1 bp | 1.00e-04 | 0.00 (100% err) | 1.00e-04 (0.14%) |
| 5 bp | 5.00e-04 | 0.00 (100% err) | 5.00e-04 (0.04%) |
| 100 bp | 1.00e-02 | 0.00 (100% err) | 1.00e-02 (~0%) |

Why: bfloat16 has 7 mantissa bits = ~0.78% relative precision. Two adjacent ln_price values
(e.g., 5.4419 and 5.4420) round to the same bfloat16 bit pattern. The delta vanishes.

**Rule**: bfloat16 ONLY for stored level features. NEVER for delta/return/difference columns.

## Per-Column Precision Matrix

| Column | Proposed dtype | Bytes | Rationale |
|--------|---------------|-------|-----------|
| price | float32 | 4 | K01P01 raw — cents precision, keep f32 |
| size | float32 | 4 | K01P01 raw — fractional shares |
| timestamp | int64 | 8 | Absolute ns, no deltas possible (72K out-of-order) |
| notional | float32 | 4 | Product price×size |
| ln_price (level) | bfloat16 | 2 | ML level feature — 0.08% err fine |
| ln_size (level) | bfloat16 | 2 | ML level feature |
| ln_notional (level) | bfloat16 | 2 | ML level feature |
| sqrt_price (level) | bfloat16 | 2 | ML level feature |
| recip_price (level) | bfloat16 | 2 | ML level feature |
| delta_ln_price ×5 | float32 | 4 | RETURN — tiny values, must be f32 |
| direction | int8 | 1 | -1/0/+1 |
| sin_time, cos_time | bfloat16 | 2 | Cyclical level features |
| round_lot, odd_lot | uint8_flags | 1 | Pack 8 bool flags per byte |
| rolling_mean/std | float32 | 4 | Statistical features |
| conditions | uint32 bitmask | 4 | 26 flags in one uint32 |
| exchange_id | uint8 | 1 | 17 unique venues |
| sequence_num | uint32 | 4 | Was string, max 265,794 |

## Sizing

598K ticks, 15 representative columns:
- All float64: 71.8 MB
- Mixed precision (above): 27.5 MB (62% smaller)

More practically, K01P01 only (7 cols):
- K01P01 current parquet: 34.5 MB
- K01P01 MKTF correct types: ~15 MB (56% smaller)

## Fixed-Point Integer for Prices

int32 fixed-point at 0.0001 dollar precision:
- $230.88 → 2,308,800 (4 bytes — same as float32)
- Exact representation: no floating-point error
- Max at this precision: $214,748.36 (int32 max / 10000)
- GPU decode: `float p = (float)fixed_int * 0.0001f;` — one multiply
- **Does NOT save bytes vs float32** but enables exact price comparison

Worth it for K01P01 if downstream needs exact price equality checks.
Probably not worth the complexity for v1 — float32 has 23 mantissa bits
which is ~0.000001 precision at $230, which is 0.0001 cents. More than adequate.

## The DirectStorage Architecture

For "file IS GPU memory layout," alignment must be 4096 bytes (NVMe sector):
- 64-byte alignment (GPU cache line): insufficient for DirectStorage
- 4096-byte alignment: supports DirectStorage AND GPU cache alignment
- Waste per column: max 4096 bytes × 7 cols = 28KB. Negligible.

DirectStorage flow:
```
NVMe → [async I/O request to col_offset] → GPU VRAM (column array)
                                              ↓
                                         CUDA kernel reads float* directly
```

No CPU memcpy. No deserialization. The bytes on disk ARE the bytes the kernel reads.
Requires: Win11 + DirectX 12 + RTX 6000 Pro Blackwell (all satisfied).

**Column alignment for DirectStorage**: 4096 bytes. Update from previous 64-byte spec.

## What Stays Float

- price, size: float32 (raw tick data — precision matters)
- delta_ln_price: float32 (NEVER bfloat16 — tested empirically)
- Any difference/return column: float32

## What Can Go Smaller

- Level features (ln, sqrt, recip, sin/cos): bfloat16 (2 bytes each)
- Flags: pack into uint8 (8 bools per byte)
- Enum columns (exchange): uint8
- String → integer mappings: uint32 or uint16
- String conditions → uint32 bitmask

## Recommendations for Pathmaker

Build in two prototype passes:

**Pass 1 — MKTF-v1 (K01P01 correct types, no bfloat16)**:
- Focus on string → integer encoding wins
- Target: 15MB, benchmark reads vs 9.2ms raw binary
- Header: 64 bytes + JSON metadata + column directory
- Alignment: 4096 bytes per column (DirectStorage ready)

**Pass 2 — MKTF-v2 (K01P02 mixed precision)**:
- Add bfloat16 for level features
- Add float32 for delta/return features
- Target: 27MB for full 15-column K01P02 (vs 120MB float64)
- Benchmark: K01P02 read time + GPU decode overhead
