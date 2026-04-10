# real-data: Navigator Notes

## No external data files needed

Task 4 is about validating the end-to-end Rust pipeline with realistic data
structures — not specifically historical AAPL tick data. The goal:

- Realistic data shape: tickers, timestamps, prices, volumes
- FinTek universe scale: ~4600 tickers, 1M rows (as in manuscript 014)
- Correctness: GPU results match CPU reference
- Performance: record actual throughput

Generate everything in Rust from deterministic seeds.

## The synthetic data shape

```rust
struct TickRecord {
    ticker_id: i32,  // [0, 4600) — the FinTek universe
    ts_ns: i64,      // nanosecond timestamp (simulated trading day)
    price: f64,      // random walk: 150.0 + brownian motion
    volume: f64,     // log-normal: realistic trade sizes
}
```

AAPL-realistic price generation:
```rust
let price = (0..n).scan(150.0_f64, |p, i| {
    *p += (rng.gen::<f64>() - 0.5) * 0.10;  // ±$0.10 tick moves
    *p = p.max(100.0).min(250.0);             // bounds
    Some(*p)
}).collect::<Vec<_>>();
```

The ticker_id distribution: non-uniform (some tickers trade much more than others).
Use power-law distribution: ticker_id = (rand^2 * 4600) as i32.

## The end-to-end path

1. Generate data (Rust, deterministic seed)
2. `HashScatterEngine::new()` — compile NVRTC kernels
3. `engine.groupby(keys, prices, 4600)` → per-ticker price stats
4. `engine.scatter_sum(keys, volumes, 4600)` → per-ticker volume sums
5. CPU reference: loop over data, build HashMap<i32, (f64, f64, f64)>
6. Verify: max error across all groups < 1e-6
7. Benchmark: throughput in million rows/second

## The AffineOp connection (when naturalist lands Task 2)

Once AffineOp is in winrapids-scan, the end-to-end test extends to:
8. `ScanEngine::scan_inclusive(AffineOp::ewm(0.1), prices)` → per-tick EWM
9. Verify EWM result matches sequential reference
10. Multi-operation in one benchmark run

This becomes the true "end-to-end Rust pipeline" — hash scatter + scan in sequence.

## What observer needs to do for this task

The `main.rs` in `crates/tambear/src/` already has a 1M row groupby benchmark
(from scout). Task 4 extends it:
1. Add realistic synthetic data generation (price, volume, non-uniform ticker dist)
2. Add multi-operation pipeline: groupby price + groupby volume in sequence
3. Add CPU reference comparison
4. Print throughput: rows/sec, GB/sec

Total: probably 100 lines added to main.rs, not a new file.
