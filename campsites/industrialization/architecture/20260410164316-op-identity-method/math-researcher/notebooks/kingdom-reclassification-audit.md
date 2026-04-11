# Kingdom Reclassification Audit — Final Table

Joint work: scout + math-researcher. Observer-verified.
Date: 2026-04-10

## The Two-Step Test

1. Is the transition map data-determined or state-dependent?
2. If data-determined, what's the associative algebra?

**Data-determined → Kingdom A. State-dependent → Kingdom B.**

---

## Volatility Module (volatility.rs)

| Function | Recurrence | Old label | New label | Algebra | Proof |
|---|---|---|---|---|---|
| garch11_fit (filter) | σ²=ω+αr²+βσ² | B | **A** | Affine(3×3) | r² is data, not state |
| egarch11_fit (filter) | log σ²=ω+α\|z\|+γz+β log σ² | B | **B** | Non-associative | z=r/σ couples state |
| gjr_garch11_fit (filter) | σ²=ω+αr²+γr²I(r<0)+βσ² | B | **A** | Affine(3×3, piecewise) | I(r<0) depends on r (data) |
| tgarch11_fit (filter) | σ=ω+α\|r\|+γmax(0,-r)+βσ | B | **A** | Affine(3×3) | \|r\|, max(0,-r) are data |
| ewma_variance | σ²=λσ²+(1-λ)r² | B | **A** | Affine(2×2) | EMA structure |
| garch11_log_likelihood | Σ log p(r_t\|σ²_t) | - | **A** | Standard Add | Independent across t |
| garch11_fit (optimizer) | argmin -LL | - | **C** | Iterative | Parameter search |

## Time Series Module (time_series.rs)

| Function | Recurrence | Old label | New label | Algebra | Proof |
|---|---|---|---|---|---|
| ar_fit (filter) | x_t = Σ φ_i x_{t-i} + ε_t | B | **A** | Affine(companion p×p) | Companion matrix, data offset |
| arma_fit (filter) | innovations form | B | **A** | Affine((p+q)×(p+q)) | A=F-KHF constant, b=Ky_t data |
| arima_fit (filter) | difference + ARMA | B | **A** | Same as ARMA | Differencing is pointwise (A) |
| simple_exp_smoothing | s=αy+(1-α)s | B | **A** | Affine(2×2) | EMA |
| holt_linear | (level, trend) = affine | B | **A** | Affine(3×3) | Constant 2×2 matrix, data offset |
| adf_test | OLS on lagged diffs | - | **A** | Standard accumulate | Independent regression |
| kpss_test | Partial sums | - | **A** | Prefix(Add) | Cumulative sum |
| bocpd | P(r_t\|y_{1:t}) | B | **B** | Growing dimension | State dim grows with t |
| cusum_mean | S_t = max(0, S_{t-1}+x_t-k) | B | **B** | Non-associative | max(0,...) is state-dependent |
| pelt | F(t)=min_τ[F(τ)+C(τ,t)+β] | B | **A (tropical)** | TropicalMinPlus | min-plus is associative |
| stl_decompose | iterative LOESS | - | **C** | Iterative | Repeated smoothing |

## Graph Module (graph.rs) — already correct

| Function | Old label | Verified | Notes |
|---|---|---|---|
| dijkstra | A (tropical) | ✓ | Scout annotated this session |
| bellman_ford | A (tropical) | ✓ | Scout annotated this session |
| floyd_warshall | A (tropical) | ✓ | Scout annotated this session |

## Complexity Module (complexity.rs)

| Function | Old label | New label | Notes |
|---|---|---|---|
| sample_entropy | A | A | Pairwise distance counting, parallel |
| dfa | A | A | Segment detrending, independent per segment |
| hurst_rs | A | A | R/S per block, independent |
| mfdfa | A | A | DFA at multiple q, independent |
| lyapunov_spectrum | B | **B** | QR iteration is state-dependent |

## Other Modules

| Function | Module | Old label | New label | Algebra |
|---|---|---|---|---|
| kalman_filter | state_space | B | **A** | Sarkka 5-tuple prefix scan |
| hmm_forward | hmm | unlabeled | **A** | Matrix prefix product |
| hmm_viterbi | hmm | unlabeled | **A (tropical)** | Max-plus prefix product |
| kaplan_meier | survival | A | A | Prefix product (confirmed) |

---

## Summary

| Reclassification | Count |
|---|---|
| B → A (standard affine) | 9 |
| B → A (tropical) | 1 (PELT) |
| Unlabeled → A | 2 (HMM forward, Viterbi) |
| Confirmed B | 4 (EGARCH, BOCPD, CUSUM, Lyapunov spectrum) |
| Confirmed A | 8 (no change needed) |
| Confirmed C | 2 (optimizers, STL) |

**9 out of 13 "Kingdom B" labels were wrong.** The sequential implementation was debt, not a model constraint.

## Genuine Kingdom B Survivors

1. **EGARCH**: z_t = r_t/σ_t couples state nonlinearly (standardized residual depends on volatility)
2. **BOCPD**: state dimension grows with t (unbounded run-length posterior)
3. **CUSUM**: max(0, ...) clips to zero, making the map state-dependent (can't go negative)
4. **Lyapunov spectrum**: QR iteration on the tangent space is state-dependent (each QR depends on previous Q)

## Criterion for Future Classification

When adding a new recurrence to tambear:
```
Write: x_{t+1} = M(data_t) · x_t + b(data_t)

If M depends only on data_t (not on x_t): Kingdom A.
  → Implement as affine prefix scan over (dim+1)×(dim+1) matrices.

If M depends on x_t: Kingdom B.
  → Implement sequentially. Document WHY it's genuinely sequential.
```
