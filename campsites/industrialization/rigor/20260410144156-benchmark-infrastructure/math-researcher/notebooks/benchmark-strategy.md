# Benchmark Infrastructure Strategy

## Three Dimensions of Benchmarking

Every primitive needs three kinds of benchmarks:

### 1. Correctness Benchmarks (Oracle Parity)

**Goal**: Prove our answer matches ground truth to numerical precision.

**Infrastructure needed**:
- Pre-computed oracle values at 50+ digit precision (mpmath)
- Stored as JSON: `benches/oracles/<primitive>.json`
- Format: `{ "cases": [{ "name": "...", "inputs": {...}, "expected": "0.94868329805051..." }] }`
- Test harness loads oracle, runs our impl, checks tolerance

**Oracle computation strategy**:
1. Python script using mpmath for arbitrary-precision ground truth
2. scipy/R for peer comparison (document exact version)
3. SymPy for closed-form verification where possible
4. Store oracle values IN the test file as constants (see workup_kendall_tau.rs pattern)

**Tolerance tiers**:
| Primitive class | Expected precision | Tolerance |
|---|---|---|
| Exact integer (inversion count) | Exact | 0 |
| Simple formula (Pearson r, mean) | 15+ digits | 1e-14 |
| Iterative (SVD, eigenvalue) | 12-14 digits | 1e-12 |
| Numerical integration | 8-12 digits | 1e-8 |
| Optimizer-based (GARCH, MLE) | 6-10 digits | 1e-6 |
| Stochastic (bootstrap) | Statistical | CI overlap test |

### 2. Performance Benchmarks (Scale Ladder)

**Goal**: Verify complexity class and measure absolute throughput.

**Infrastructure needed**:
- Scale ladder test harness (exists: `tests/scale_ladder*.rs`)
- Standard scales: `n ∈ [10, 100, 1K, 10K, 100K, 1M, 10M]`
- Timing: median of 5 runs, exclude first (warmup)
- Memory: peak RSS tracking

**Scale ladder test pattern**:
```rust
#[test]
#[ignore] // Run with --ignored for benchmark
fn scale_ladder_kendall_tau() {
    for &n in &[10, 100, 1_000, 10_000, 100_000, 1_000_000] {
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (i * 7 + 3) as f64 % 100.0).collect();
        let start = std::time::Instant::now();
        let _ = kendall_tau(&x, &y);
        let elapsed = start.elapsed();
        println!("kendall_tau n={}: {:?}", n, elapsed);
    }
}
```

**Complexity verification**: fit log(time) vs log(n)
- slope ≈ 1.0 → O(n)
- slope ≈ 1.0-1.3 → O(n log n)
- slope ≈ 2.0 → O(n²)
- slope ≈ 3.0 → O(n³)

If measured slope > expected: investigate (cache misses? memory allocation?)

### 3. Competitive Benchmarks (Peer Comparison)

**Goal**: Know exactly how we compare to scipy, R, Julia, MATLAB.

**Infrastructure needed**:
- Pre-computed peer timings at standard scales
- Python benchmarking script: `benches/peers/bench_<primitive>.py`
- Results stored as JSON: `benches/peers/results/<primitive>_scipy.json`

**Peer timing format**:
```json
{
  "primitive": "kendall_tau",
  "peer": "scipy",
  "version": "1.17.1",
  "platform": "x86_64-linux, Python 3.12",
  "results": [
    {"n": 10, "median_ns": 45000},
    {"n": 100, "median_ns": 120000},
    {"n": 1000, "median_ns": 850000}
  ]
}
```

---

## Adversarial Coverage Matrix

For each primitive, we need adversarial tests across these dimensions:

### Dimension 1: Numerical Stability
| Test | Purpose |
|---|---|
| Extreme magnitudes (1e300, 1e-300) | Overflow/underflow |
| Near-cancellation (a ≈ b, compute a-b) | Catastrophic cancellation |
| Subnormal values (1e-320) | Flush-to-zero behavior |
| Mixed scales (some large, some tiny) | Precision loss |
| Accumulation order (sum large then small) | Associativity |

### Dimension 2: Degeneracy
| Test | Purpose |
|---|---|
| Empty input | Graceful NaN/error |
| Single element | Minimal valid input |
| All identical values | Zero variance/range |
| All zeros | Special case |
| Contains NaN | Propagation behavior |
| Contains ±Inf | Infinity arithmetic |

### Dimension 3: Statistical Edge Cases
| Test | Purpose |
|---|---|
| Perfect correlation/anticorrelation | Boundary of range |
| Zero correlation (independent uniform) | Null case |
| Bimodal distribution | Violates unimodality |
| Heavy tails (Cauchy) | Undefined moments |
| Skewed (log-normal) | Asymmetry |
| Ties (50%+ identical values) | Tied-rank handling |
| Outliers (99% normal + 1% extreme) | Breakdown resistance |

### Dimension 4: Scale
| Test | Purpose |
|---|---|
| n = 2 (minimum) | Minimum viable |
| n = 10⁶ (production) | Real workload |
| n = 10⁸ (stress) | Memory/time limits |
| p >> n (high-dimensional) | Curse of dimensionality |
| p = 1 (univariate) | Degenerate dimension |

---

## Benchmark Results Storage

```
benches/
├── oracles/
│   ├── kendall_tau.json           # mpmath ground truth
│   ├── pearson_r.json
│   └── ...
├── peers/
│   ├── bench_kendall_tau.py       # Python benchmark script
│   ├── bench_pearson_r.py
│   └── results/
│       ├── kendall_tau_scipy.json  # Peer timing results
│       └── pearson_r_scipy.json
└── reports/
    └── scale_ladder_results.json   # Our timing results
```

---

## Priority: What to Benchmark First

### Tier 1 — Core primitives (used by 10+ downstream methods):
1. sort (via std, but verify scale behavior)
2. moments_ungrouped (mean/var/skew/kurt)
3. pearson_r / spearman / kendall_tau
4. covariance_matrix
5. svd / sym_eigen / cholesky / lu / qr
6. ols_slope / ols_normal_equations
7. fft (via signal_processing)
8. shannon_entropy / mutual_information
9. acf / pacf
10. sample_entropy / dfa / hurst

### Tier 2 — Methods that are most likely to have precision issues:
11. regularized_incomplete_beta (tricky numerics)
12. adf_test (critical values from asymptotic tables)
13. garch11_fit (optimizer convergence)
14. stl_decompose (LOESS internal)
15. shapiro_wilk (coefficients + approximation)
16. logistic_regression (Newton convergence)

### Tier 3 — Everything else in the catalog

---

## Existing Scale Ladder Tests

Already exist (verify these still compile and pass):
- `tests/scale_ladder.rs`
- `tests/scale_ladder_dbscan_knn.rs`
- `tests/scale_ladder_descriptive.rs`
- `tests/scale_ladder_kde.rs`
- `tests/scale_ladder_kde_fft.rs`

Already exist as workups:
- `tests/workup_kendall_tau.rs`
- `tests/workup_pearson_r.rs`
- `tests/workup_inversion_count.rs`
- `tests/workup_incomplete_beta.rs`

## Current Compilation Issues Observed

The codebase has 7 compilation errors in tambear (lib test):
1. `rank` name collision (likely from re-export clash after nonparametric changes)
2. `sigmoid` name collision (same pattern)
3. `mat_approx_eq` called with 4 args but now takes 3 (signature changed in linear_algebra.rs)

These are task #3 (phantom references / fixes) territory, but they block running the test suite.
