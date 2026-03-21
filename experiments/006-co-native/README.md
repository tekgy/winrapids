# Experiment 006: Co-Native Data Structures for GPU Computing

## Hypothesis

A tiered memory architecture with explicit location tagging and co-native metadata enables intelligent query planning where both human and AI agents can reason about data placement without GPU access.

## Method

Built `TieredColumn` and `TieredFrame` classes implementing:
- Four memory tiers: Device (GPU), Pinned, Pageable, Storage
- Explicit promotion between tiers with measured costs
- Memory map showing per-column tier, size, promotion cost, and access count
- Query planner that estimates transfer costs before execution
- Arrow device type mapping for interop

Validated with filtered sum across different tier configurations, compared against pandas.

## Results

### Tier Transfer Costs (8 MB column)

| Transfer | Estimated | Actual |
|----------|----------|--------|
| GPU -> GPU | 0.0 ms | 0.0 ms |
| Pinned -> GPU | 0.1 ms | ~0.2 ms |
| CPU -> GPU | 0.3 ms | 0.77 ms |

Estimates are conservative lower bounds based on experiment 002 bandwidth measurements.

### Filtered Sum: sum(value) where flag == 1 (10M rows)

| Scenario | Promotion | Compute | Total | vs pandas |
|----------|----------|---------|-------|-----------|
| All on GPU | 0.0 ms | 0.33 ms | 0.33 ms | 87x |
| Value GPU, flag pinned | 1.1 ms | 0.5 ms | 1.6 ms | 18x |
| All on CPU | 6.5 ms | 0.5 ms | 7.0 ms | 4x |
| Re-run (data now on GPU) | 0.0 ms | 0.33 ms | 0.33 ms | 87x |
| pandas baseline | - | - | 29.1 ms | 1x |

**Key insight:** Even the worst case (all data on CPU, first access) is 4x faster than pandas. After initial promotion, subsequent operations are 87x faster.

### Memory Map Output (Co-Native)

```
TieredFrame: 10,000,000 rows x 7 columns

  Column               Type           Size Tier   GPU Cost   Accesses
  -------------------- ---------- -------- ---- ---------- ----------
  timestamp            int64         80.0M  CPU     3.2 ms          0
  open                 float64       80.0M  CPU     3.2 ms          0
  high                 float64       80.0M  GPU   0 (here)          0
  low                  float64       80.0M  GPU   0 (here)          0
  close                float64       80.0M  GPU   0 (here)          0
  volume               int64         80.0M  PIN     1.4 ms          0
  symbol               int32         40.0M  CPU     1.6 ms          0

  GPU: 240.0 MB
  PIN: 80.0 MB
  CPU: 200.0 MB
```

**This is readable by both humans and AI agents.** An agent can:
1. Parse the memory map to understand data layout
2. Query the plan for a specific operation
3. See which columns need promotion and the cost
4. Decide whether to promote proactively or compute on CPU
5. All without touching GPU memory

### Query Planner

```python
plan = frame.query_plan(["high", "low"])
# -> {'needs_promotion': [], 'total_transfer_ms': 0.0}

plan = frame.query_plan(["open", "high", "low", "close", "volume"])
# -> {'needs_promotion': ['open', 'volume'], 'total_transfer_ms': 4.6, 'total_transfer_mb': 160.0}
```

## Conclusions

1. **The co-native split works.** CPU metadata is always accessible. GPU data is only touched during compute. An agent can reason about data placement without any GPU access.

2. **Explicit tier management beats transparent spilling.** The memory map makes costs visible. The query planner shows what a computation will cost before it runs. No hidden transfers, no surprise OOM.

3. **Promotion is amortized.** The first access pays the transfer cost. All subsequent accesses are free (data stays on GPU). For iterative workloads, the promotion cost becomes negligible.

4. **Even cold-start is 4x faster than pandas.** All-CPU to all-GPU transfer + compute (7 ms) still beats pandas (29 ms). The GPU's compute speed more than compensates for the transfer cost at 10M rows.

5. **The architecture scales to the Sirius model.** With 93 GB VRAM, we can keep many datasets fully on GPU. The tier system extends naturally to disk-backed storage for out-of-core processing.

6. **Arrow device type mapping** connects the tier system to the broader Arrow ecosystem. Each tier maps to an Arrow device type, enabling zero-copy handoffs to other Arrow-aware libraries.

## Files

- `co_native.py` — TieredColumn, TieredFrame, query planner, and demos
