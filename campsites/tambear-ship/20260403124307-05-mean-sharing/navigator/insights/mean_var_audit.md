# Mean/Var Audit — Campsite 05

## Summary

Full grep audit found **120+ independent mean computations** across **27 source modules**.
The naturalist's count of 20 was conservative — the actual scope is larger.

## Priority: Mean+Var Pairs (Stability Risk)

These compute BOTH mean AND variance manually. Highest risk of catastrophic cancellation.

| Module | Lines | Pattern |
|--------|-------|---------|
| bayesian.rs | 183-184 | samples mean+var |
| bigfloat.rs | 1815-1816 | steps mean+var |
| complexity.rs | 286+ | data mean (var nearby) |
| series_accel.rs | 816-817 | tail mean+var |
| time_series.rs | 273-274, 289-293 | data mean+var (twice!) |
| volatility.rs | 43-44 | returns mean+var |
| tbs_lint.rs | 305-306, 350-351 | column mean+var (twice!) |
| rng.rs | 453-454, 590-591 | samples mean+var (tests) |
| nonparametric.rs | 503, 603 | bootstrap mean+var |
| multi_adic.rs | 329-330 | vals mean+var |

## Do Not Change

- descriptive.rs — canonical implementation
- neural.rs — loss functions (MSE etc.), not statistical moments
- pipeline.rs — internal, already uses moments where needed
- interpolation.rs — Chebyshev coefficients
- causal.rs — domain-specific treatment/control means
- Test functions — they check behavior, not production stats

## LockBag Integration Gap

The LockBag is wired in tbs_executor.rs (accumulated by lock steps, cleared after non-lock steps) but **no step actually queries it**. The bag is populated and discarded without effect.

Steps that SHOULD query it when wired:
- precision: describe, pca, fa, cluster, etc. (GPU compute)  
- nan: all data-processing steps
- rotation: fa/paf steps
- method: cluster steps
- sweep/superposition: discover steps

This is campsite 08 territory, not campsite 05.
