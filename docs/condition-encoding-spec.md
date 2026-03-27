# Condition Code Encoding — Eliminating Strings from the Pipeline

**Date**: 2026-03-27
**Context**: The conditions column (DO07) is currently a comma-separated string. This forces string parsing in K01P02 and prevents GPU acceleration of the trade_flags leaf. This spec proposes replacing the string with a packed uint32 bitmask.

## The Encoding

26 condition flags pack into a single uint32 (26 of 32 bits used):

```
Bit  0: DO01 is_regular (code 0)
Bit  1: DO02 is_form_t (code 12)
Bit  2: DO03 is_extended_hours (codes 12, 29)
Bit  3: DO04 is_opening_print (codes 6, 15, 53)
Bit  4: DO05 is_closing_print (codes 7, 16, 54)
Bit  5: DO06 is_reopening_print (codes 17, 21)
Bit  6: DO07 is_bunched_trade (codes 4, 25)
Bit  7: DO08 is_intermarket_sweep (codes 14, 33)
Bit  8: DO09 is_trade_thru_exempt (code 41)
Bit  9: DO10 is_sellers_option (code 23)
Bit 10: DO11 is_bunched_sold (code 5)
Bit 11: DO12 is_cross_trade (code 9)
Bit 12: DO13 is_acquisition (code 1)
Bit 13: DO14 is_next_day (code 20)
Bit 14: DO15 is_odd_lot (code 37)
Bit 15: DO16 is_average_price (codes 2, 38, 52)
Bit 16: DO17 is_market_center_closing (codes 19, 30)
Bit 17: DO18 is_closing_prints (code 8)
Bit 18: DO19 is_yellow_flag (code 36)
Bit 19: DO20 is_automatic_execution (code 3)
Bit 20: DO21 is_stopped_stock (code 27)
Bit 21: DO22 is_halt_related (codes 15, 17, 21)
Bit 22: DO23 is_reopening_prints (code 28)
Bit 23: DO24 is_sold (codes 13, 14)
Bit 24: DO25 is_derivatively_priced (code 10)
Bit 25: DO26 has_unknown_conditions (any unrecognized code)
Bits 26-31: reserved
```

## Storage

- Current: `large_string` ("14,37,41") = 8+ bytes per trade, variable length
- Proposed: `uint32` bitmask = 4 bytes per trade, fixed length
- For 598K AAPL trades: 2.3 MB (bitmask) vs ~4 MB (strings) = 43% smaller

## GPU Operations on Bitmask

```cuda
// Extract any flag: one bitwise AND
bool is_odd_lot = (bitmask & (1u << 14)) != 0;

// Count active flags: popcount (single hardware instruction)
int n_flags = __popc(bitmask);

// Per-bin flag statistics: just OR all bitmasks in the bin
// then popcount the OR result = number of DISTINCT flags in the bin
uint32_t bin_flags = 0;
for (int i = lo; i < hi; i++) bin_flags |= bitmasks[i];
int n_distinct = __popc(bin_flags);

// Per-bin flag fractions: popcount of AND-masked bitmasks
int n_odd_lot = 0;
for (int i = lo; i < hi; i++) n_odd_lot += (bitmasks[i] >> 14) & 1;
float odd_lot_fraction = (float)n_odd_lot / (float)(hi - lo);
```

All single-instruction operations. No string parsing. No conditionals. Pure bit manipulation.

## The Fused Bin Flags Kernel

```cuda
extern "C" __global__
void fused_bin_flag_stats(
    const unsigned int* bitmasks,
    const long long* boundaries,
    unsigned int* out_union,        // OR of all flags in bin
    unsigned int* out_intersection, // AND of all flags in bin
    int* out_popcount,              // total flag activations in bin
    float* out_fractions,           // per-flag fraction (26 floats per bin)
    int n_bins
) {
    int bin_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bin_idx >= n_bins) return;

    long long lo = boundaries[bin_idx];
    long long hi = boundaries[bin_idx + 1];
    int count = (int)(hi - lo);

    if (count == 0) { /* NaN outputs */ return; }

    unsigned int union_mask = 0;
    unsigned int inter_mask = 0xFFFFFFFF;
    int total_flags = 0;
    int per_flag_count[26] = {0};

    for (long long i = lo; i < hi; i++) {
        unsigned int m = bitmasks[i];
        union_mask |= m;
        inter_mask &= m;
        total_flags += __popc(m);
        for (int f = 0; f < 26; f++) {
            per_flag_count[f] += (m >> f) & 1;
        }
    }

    out_union[bin_idx] = union_mask;
    out_intersection[bin_idx] = inter_mask;
    out_popcount[bin_idx] = total_flags;
    for (int f = 0; f < 26; f++) {
        out_fractions[bin_idx * 26 + f] = (float)per_flag_count[f] / (float)count;
    }
}
```

This replaces the ENTIRE trade_flags leaf + eigenvalue analysis pipeline's
first stage. One kernel: bitmask → per-bin flag fractions. Then eigenvalue
decomposition operates on the 26-column fraction matrix.

## Implementation Path

1. Add `condition_bitmask` column to K01P01 during ingest (one uint32 per trade)
2. Keep the string column for debugging but mark it as non-compute
3. The trade_flags leaf becomes: read bitmask → extract individual flags (bitwise AND)
4. The bin-level flag analysis becomes: fused_bin_flag_stats kernel
5. Eventually: drop the string column from K01P01 entirely

## Compatibility

- Crypto: codes 1 and 2 pack into bits 12 and 15 respectively
- Stocks: all 26 bits potentially active
- The bitmask is asset-class-agnostic — same encoding, same bits, same GPU operations
- Unknown codes set bit 25 (has_unknown) — no information lost
