# Experiment 007: Polars GPU Acceleration on Windows

## Hypothesis

Polars GPU engine (based on cuDF) does not work on Windows. The integration path is through Arrow zero-copy interop.

## Results

### Polars GPU Engine: NOT AVAILABLE

Confirmed: `engine="gpu"` requires `cudf_polars` package, which depends on RAPIDS cuDF (Linux-only). Error message directs to `pip install cudf-polars-cu12`.

Polars has a `GPUEngine` class in its API, but it's just a wrapper for the cuDF integration. The backend is NOT pluggable — it's a hardcoded dependency on `cudf_polars`.

### Polars CPU vs GPU Comparison (10M rows)

| Operation | Polars CPU (ms) | GPU CuPy (ms) | GPU Speedup |
|-----------|----------------|---------------|-------------|
| sum | 0.807 | 0.096 | 8.4x |
| mean | 1.482 | - | - |
| filtered sum | 7.203 | 0.311 | 23.2x |
| arithmetic | 14.208 | - | - |

**Polars CPU is already fast** — notably faster than pandas (sum: 0.8 ms vs 8.3 ms). But GPU still wins by 8-23x.

### Arrow Interop

| Direction | Time (ms) |
|-----------|----------|
| Polars -> Arrow | 0.324 |
| Arrow -> Polars | 0.316 |

Essentially zero-copy for numeric columns.

### Available Engines

- `cpu`: Available (default)
- `gpu`: Requires cudf_polars (Linux-only)
- `streaming`: Available (for out-of-core processing)

## Conclusions

1. **Polars GPU is Linux-only.** The `cudf_polars` dependency is not available on Windows.

2. **The GPU backend is NOT pluggable.** It's a hardcoded cuDF integration, not a generic interface that WinRapids could plug into. A custom backend would require modifying Polars itself.

3. **Polars CPU is already 4-10x faster than pandas.** The gap between Polars CPU and GPU is smaller (8-23x) than between pandas and GPU (50-200x).

4. **Arrow is the interop path.** Polars <-> Arrow is near-zero-cost. The pipeline: Polars for I/O + query planning, then Arrow -> GPU for heavy compute. This is pragmatic and works today.

5. **Long-term vision:** If Polars exposes a backend plugin interface (which Polars has discussed), WinRapids could provide a Windows GPU backend via the Arrow Device Data Interface. But this requires upstream changes to Polars, not just WinRapids work.

## Files

- `polars_gpu_test.py` — GPU engine test, benchmarks, and architecture investigation
