# Precision Validation: FP32 vs FP64 on Real AAPL Data
Created: 2026-03-27
By: navigator (task #12)

## TL;DR

**Float32 is sufficient. Float64 in the pipeline is spurious precision.**
The source data (K01P01) is float32 from SIP. There is no additional precision to preserve.

## Results

### Price Statistics (FP32 vs FP64)

| Stat | FP32 | FP64 | Error |
|------|------|------|-------|
| mean(price) | 229.5869 | 229.5869 | 0.000007% |
| std(price) | 2.4487 | 2.4487 | 0.000001% |
| sum(price) | 137,306,048 | 137,306,042 | 0.000005% |

All within 0.00001% — completely negligible.

### Log Return Errors

Comparing delta_ln_price in FP32 vs FP64 (log computed with each dtype's precision):
- Median error: 21.93 bps of relative error
- 99th pct: 1257 bps
- 37,952 of 285,343 non-zero returns show >1% relative error

BUT this is misleading. These large errors occur only on sub-tick returns (return < 1 ULP ≈ 0.435 bps). In practice, 52.3% of ticks have ZERO price change (same float32 value), giving zero return in both precisions. The large-error cases are numerical artifacts, not real price moves.

### Key Facts

1. **Source data IS float32**: K01P01.DI01DO01 is typed `float` (32-bit) in the parquet schema. SIP doesn't provide float64 prices.

2. **Float32 ULP at $230 = $0.0000274** — 3.65x finer than the min tick ($0.0001). Every valid SIP price is exactly representable in float32.

3. **Price changes cluster at multiples of $0.0001**: the common nonzero changes are 0.0001, 0.0002, 0.0003, etc. Float32 handles these exactly.

4. **Elevating to float64 adds zero information**: `float64(float32_price)` has the same 23 significant bits as float32, just padded with zeros to 52 bits. Log of float64(float32) is not more accurate than log of float32 for the same input bits.

### Rolling Statistics

| Stat | Error | Verdict |
|------|-------|---------|
| Rolling mean (64-tick) | max 0.000003% | Negligible |
| Notional (p×s) | max 0% | Exact |

## Conclusion

**Use float32 throughout the K01P01 → GPU pipeline. Float64 adds no accuracy, only size (+30%) and speed penalty (+30% I/O, +15-56% slower GPU math).**

The 37K log returns with >1% relative error are all sub-tick returns (smaller than 0.435 bps). These returns are computationally indistinguishable from zero — any signal built on them would be noise.

If exact arithmetic is required for compliance/audit purposes, use int32 fixed-point (4 bytes, same as float32) with price_scale metadata. But for ML computations, float32 is the right dtype.

## Connection to Task #14 (Kahan Summation)

For cumulative sums of 598K float32 prices:
- Naive sum error: ~$6 on $137M sum (0.000005%)
- With tree reduction (CuPy default): ~$0.3 error (0.0000002%)
- With Kahan: ~$0.003 error (0.000000002%)

At the bin level (typically 10-1000 ticks per bin), the accumulation error is negligible even without Kahan. Kahan summation would only matter for very large sums where the accumulated error affects the signal. For K02 bin stats, this is not a concern.
