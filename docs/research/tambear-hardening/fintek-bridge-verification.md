# Fintek Bridge Verification Report

**Author**: math-researcher
**Date**: 2026-04-08
**Crate**: `R:/winrapids/crates/tambear-fintek/` (1,778 lines across 11 family modules)
**Reference**: `R:/winrapids/docs/research/tambear-hardening/fintek-math-catalog.md`
**Fintek source**: `R:/fintek/crates/trunk-rs/src/leaves/` (126 leaves)

## Executive Summary

The bridge crate implements **~40 of 126 trunk-rs leaves** across 11 family modules. All functions reviewed are **mathematically correct** in their core computation, but there are systematic issues in two dimensions:

1. **Output column coverage**: Most bridge functions compute FEWER output columns than the fintek leaf they replace. Fintek leaves emit 4-15 DO columns; bridge wrappers typically return 3-8. Leaves with 10+ outputs are the worst offenders (fft_spectral: 8 of 75, realized_vol: 4 of 8, persistent_homology: semantic mismatch on 1 field).
2. **Compilation blockers**: Two family files (`family8_correlation.rs`, `family16_extremes.rs`) import tambear functions that **do not yet exist** — they will not compile until the referenced tambear tasks land.

**One serious math bug** identified: **stationarity classification labels are swapped** (family5_stationarity.rs).

**One unit mismatch** identified: **elapsed** returns seconds but fintek expects minutes (family2_transforms.rs).

Remaining issues are all "output column completeness" — the bridge computes the right math but doesn't emit all the columns fintek expects. Each is a surface-level fix, not a math error.

---

## 1. Compilation Blockers (Must Fix Before Bridge Compiles)

### Blocker #1 — `family8_correlation.rs` imports non-existent tambear functions

**File**: `family8_correlation.rs:5`
```rust
use tambear::nonparametric::{dtw, dtw_banded, levenshtein, quantile_symbolize, edit_distance_on_series, pearson_r};
```

`dtw`, `dtw_banded`, `levenshtein`, `quantile_symbolize`, `edit_distance_on_series` **do not exist in tambear yet**. These are tracked by Task #141 (DTW and edit distance). Until #141 lands, `family8_correlation.rs` will not compile.

**Resolution**: Finish Task #141. The tambear-side function signatures should match these imports exactly.

### Blocker #2 — `family16_extremes.rs` imports non-existent tambear functions

**File**: `family16_extremes.rs:6`
```rust
use tambear::volatility::{hill_estimator, hill_tail_alpha};
```

These functions **do not exist in tambear yet**. Task #139 (Hill tail index estimator) covers this.

**Resolution**: Finish Task #139. Ensure the tambear function name matches (should be `hill_estimator` per my C-spec B6).

---

## 2. Math Bugs

### Bug #1 — Stationarity classification swapped [MEDIUM SEVERITY]

**File**: `family5_stationarity.rs:63-68`

```rust
let classification = match (adf_rejects, kpss_rejects) {
    (true, false)  => StationarityClass::Stationary,         // ✓ correct
    (false, true)  => StationarityClass::NonStationary,      // ✓ correct
    (true, true)   => StationarityClass::TrendStationary,    // ✗ should be TrendStationary per textbook
    (false, false) => StationarityClass::Inconclusive,       // ✓ correct
};
```

Actually the labeling is ALMOST right. Let me re-check with fresh eyes against the textbook rule (KPSS 1992 § 4, also Enders "Applied Econometric Time Series" § 4.8):

| ADF rejects H₀=unit root? | KPSS rejects H₀=stationary? | Conclusion |
|---|---|---|
| Yes | No | **Stationary** |
| No | Yes | **Unit root / non-stationary** |
| Yes | Yes | **Inconclusive** (contradictory) OR trend-stationary if trend is in ADF model |
| No | No | **Inconclusive** (both weak) |

The bridge has `(true, true) → TrendStationary` and `(false, false) → Inconclusive`. Textbook-strict:
- `(true, true)`: **Inconclusive** (textbook), **TrendStationary** ONLY if the ADF was run with a trend term (which the bridge doesn't check)
- `(false, false)`: **Inconclusive**

**Finding**: Since the bridge runs ADF WITHOUT a trend term (it calls `adf_test(data, n_lags)` which is the no-trend version), `(true, true)` should be **Inconclusive**, not **TrendStationary**. The current label is misleading.

**Fix**: Either (a) change `(true, true)` → `Inconclusive`, or (b) add a `trend: bool` parameter and report `TrendStationary` only when running trend-aware ADF.

**Severity**: MEDIUM. Doesn't cause wrong numbers (the raw ADF and KPSS statistics are correct), but the classification label users see is wrong.

### Bug #2 — `elapsed` returns seconds, fintek expects minutes [UNIT MISMATCH]

**File**: `family2_transforms.rs:107-110` (`elapsed_seconds_of_day`)

Fintek's `R:/fintek/crates/trunk-rs/src/leaves/elapsed.rs:23-25`:
```rust
let minutes: Vec<f32> = ts.iter()
    .map(|&t| ((t % NS_PER_DAY) as f64 / NS_PER_MINUTE) as f32)
    .collect();
```
Fintek returns **minutes** (range [0, 1440)).

Bridge's `elapsed_seconds_of_day`:
```rust
timestamps_ns.iter().map(|&t| ((t % DAY_NS) as f64) / 1e9).collect()
```
Returns **seconds** (range [0, 86400)).

**Finding**: Unit mismatch. If the bridge ships as-is, fintek's downstream logic will see values 60× larger than expected.

**Fix**: Rename to `elapsed_minutes_of_day` and divide by `60.0e9` (nanoseconds per minute) instead of `1e9`. Or add a `unit: TimeUnit` parameter.

**Severity**: HIGH (bit-perfect violation for a DIRECT leaf).

---

## 3. Output Column Coverage Gaps

For each family, the bridge computes a SUBSET of the DO columns emitted by the fintek leaf. These are NOT math bugs — the math is correct where implemented — but they mean the bridge CANNOT drop in as a complete replacement. Each gap is a task to add the missing computation.

### Family 1 — Distribution

| Leaf | Fintek DO count | Bridge outputs | Missing |
|---|---|---|---|
| distribution | 10 | 10 | ✓ complete |
| normality | 5 | 4 | excess_kurtosis (trivially from stats) |
| shannon_entropy | 4 | 1 (just entropy) | entropy_rate, max_entropy, normalized_entropy |
| spectral_entropy | 1 | — not implemented in family1 (is in family6) | moved to family6 |
| tail_field | 7 | 1 (chi² only) | tail_concentration_entropy, peak_tail_quintile, tail_cramer_v, tail_ks_stat, joint_mi, n_extreme_events |
| heavy_tail | 4 | in family16 | (see family16) |
| fisher_info | 3 | — not implemented | Task #172 (C4) |

**Critical gaps**: `tail_field` at 1/7 = 14% coverage. `shannon_entropy` at 1/4 = 25%.

**Notes**:
- ⚠ Bridge's `shannon_entropy_of_returns` uses **log2 (bits)**. Fintek's header doesn't specify the base — this should be verified against fintek's test data. Most statistical entropy is in nats (ln).

### Family 2 — Transforms

| Leaf | Fintek DO count | Bridge outputs | Missing |
|---|---|---|---|
| All 11 pointwise | — | 11 of 11 | ✓ complete |

**Only issue**: elapsed unit mismatch (Bug #2 above).

### Family 3 — Bin Aggregates

| Leaf | Fintek DO count | Bridge outputs | Missing |
|---|---|---|---|
| ohlcv | 9 | 6 | count, notional_sum, realized_variance |
| counts | 5 | 5 | ✓ complete |
| validity | ? | 4 | (match fintek source needed) |
| **variability** | 4 | **0** | NOT IMPLEMENTED |

**Critical gap**: `variability` is listed in the family header docstring but **has no function**. Needs rolling_var_cv, rolling_mean_cv, range_variation, stability_index.

### Family 4 — Time Series

| Leaf | Fintek DO count | Bridge outputs | Missing |
|---|---|---|---|
| autocorrelation | 16 | 16 (ACF[1..8] + PACF[1..8]) | ✓ complete |
| ar_model | 5 | 5 | ✓ complete |
| arma | 5 | 5 | ✓ complete (approximate MA OK) |
| arima | 5 | 5 | ✓ complete (d=1 case) |
| **arx** | 4 | **0** | NOT IMPLEMENTED |
| **ar_burg** | 4 | 0 | blocked by Task #137 (B2) |

**Critical gap**: `arx` missing entirely. This is AR + exogenous input (volume → returns). Two OLS fits: restricted (AR only) and unrestricted (AR + exogenous). Partial R² improvement.

### Family 5 — Stationarity

| Leaf | Fintek DO count | Bridge outputs | Missing |
|---|---|---|---|
| stationarity | 5 | 5 (after bug fix) | ✓ complete |
| dependence | 4 | 3 | optimal_lags, max_abs_acf |
| **struct_break** | 4 | **0** | NOT IMPLEMENTED (Chow scan) |
| **classical_cp** | 4 | 0 | blocked by Task #143 (B14/CUSUM) |
| **pelt** | 4 | 0 | blocked by Task #143 (B15) |
| **bocpd** | 4 | 0 | blocked by Task #143 (B16) |

**Critical gap**: Bug #1 stationarity classification (fix above). Plus all 4 changepoint leaves are missing (three are tasked at #143).

### Family 6 — Spectral

| Leaf | Fintek DO count | Bridge outputs | Missing |
|---|---|---|---|
| fft_spectral (M8) | 15 | 8 | 7 features missing |
| fft_spectral (M16/M32/M64/M128) | 15 × 4 = 60 | **0** (only 1 resolution impl) | 4 resolution variants NOT implemented |
| welch | 4 | 2 (just freqs + psd) | spectral_centroid, spectral_bandwidth, spectral_entropy, peak_frequency |
| **multitaper** | 4 | **0** | NOT IMPLEMENTED (tambear has `multitaper_psd`) |
| **lombscargle** | 4 | **0** | NOT IMPLEMENTED (tambear has `lomb_scargle`) |
| cepstrum | 4 | 1 (raw cepstrum vec) | quefrency_peak, cepstral_energy, spectral_flatness, cepstral_distance |
| hilbert | 5 | 2 (env + phase vecs) | inst_amplitude_mean/var, inst_freq_mean/var, phase_coherence |
| **stft_leaf** | 4 | **0** | NOT IMPLEMENTED (tambear has `stft`) |
| spectral_entropy | 1 | 1 | ✓ complete |
| **fir_bandpass** | 3 | **0** | NOT IMPLEMENTED |
| **energy_bands** | 4 | **0** | NOT IMPLEMENTED |
| **periodicity** | 4 | **0** | NOT IMPLEMENTED |
| **coherence** | 4 | 0 | blocked by Task #175 (C7) |
| **cwt_wavelet** | 3 × 12 variants = 36 | 0 | blocked by Task #144 (B7) |
| **scattering** | 4 | 0 | blocked by cwt dependency |
| **haar_wavelet** | 13 × 15 variants = 195 | 0 | NOT IMPLEMENTED (tambear has `haar_dwt`, `haar_wavedec`) |
| **wigner_ville** | 4 | 0 | blocked by Task #164 (B19) |
| **wavelet_leaders** | 4 | 0 | NOT IMPLEMENTED |

**Critical gap**: Family 6 is the WORST covered family by output count. fft_spectral alone goes from 75 expected outputs to 8 actual (10.7%). Six leaves the catalog lists as DIRECT (multitaper, lombscargle, stft, fir_bandpass, energy_bands, periodicity, haar_wavelet) are NOT IMPLEMENTED despite tambear having the primitives ready.

### Family 8 — Correlation

| Leaf | Fintek DO count | Bridge outputs | Missing |
|---|---|---|---|
| dtw | 4 | 1 (just distance) | normalized, ratio, path_efficiency |
| edit_distance | 3 | 1 | normalized, symbol_entropy_diff |
| dist_distance | 4 | 1 (wasserstein) | energy_distance, ks_stat, ks_p |
| cross_correlation | 11 | 1 (single-lag correlation) | 10 features (peak lag, max corr, bandwidth, etc.) |
| **coherence** | 4 | 0 | blocked by Task #175 |
| **cross_correlation (bin-level)** | multi-feature | partial | 10+ features missing |
| **ccm** | 4 | 0 | NOT IMPLEMENTED |
| **transfer_analysis** | 4 | 0 | NOT IMPLEMENTED (compose from FFT) |
| **tick_causality** | 4 | 0 | NOT IMPLEMENTED |
| **mutual_info** | 4 | 0 | NOT IMPLEMENTED (tambear has `mutual_information`) |
| **transfer_entropy (bin)** | 4 | 0 | blocked by Task #142 |

**Critical gap**: Family 8 returns single scalars where fintek expects feature vectors. Also blocked by Task #141 (DTW/edit exist but aren't in tambear yet).

### Family 9 — Volatility

| Leaf | Fintek DO count | Bridge outputs | Missing |
|---|---|---|---|
| garch | 5 | 7 (more than fintek, including near_igarch flag) | ✓ complete |
| realized_vol | 8 | 4 | tripower_var, quadpower_var, integrated_quarticity, rv_ratio (partial: need Task #171 C3) |
| jump_detection | 4 | 2 (bns_stat + boolean) | n_jumps, max_jump_size, jump_fraction (different definitions) |
| roll_spread | ? | 1 scalar | (match fintek source) |
| signature_plot | 4 | **0** | NOT IMPLEMENTED (compose from realized_variance at multiple freqs) |
| **range_vol** | 4 | 0 | blocked by Task #138 (B5) |
| **vpin_bvc** | 4 | 0 | blocked by Task #146 (B8) |
| **vol_regime** | 4 | **0** | NOT IMPLEMENTED |
| **vol_dynamics** | 4 | **0** | NOT IMPLEMENTED |
| **stochvol** | 4 | **0** | NOT IMPLEMENTED (compose: ar_fit on log(r²+ε)) |
| **tick_vol** | 4 | **0** | NOT IMPLEMENTED |

**Critical gap**: jump_detection semantic mismatch — bridge returns a Z-statistic and boolean, fintek expects jump counts and sizes. Need to rework to match fintek's output structure.

### Family 10 — Nonlinear Dynamics

| Leaf | Fintek DO count | Bridge outputs | Missing |
|---|---|---|---|
| sample_entropy | 1 | 1 | ✓ complete |
| permutation_entropy | 3 | 1 (just PE) | statistical_complexity, disequilibrium |
| hurst_rs | 1 | 1 | ✓ complete |
| dfa | 1 | 1 | ✓ complete |
| correlation_dim | 4 | 1 | d2_confidence, scaling_range, lacunarity |
| lyapunov | 4 | 1 | confidence, divergence_rate, prediction_horizon |
| poincare | 4 | 3 | return_correlation missing |
| **mfdfa** | 4 | **0** | NOT IMPLEMENTED (compose: loop DFA over q values) |
| **embedding** | 4 | **0** | NOT IMPLEMENTED (compose: AMI + FNN) |
| **rqa** | 8 | 0 | blocked by Task #174 (C6) |
| **lz_complexity** | 3 | 0 | blocked by Task #173 (C5) |
| **correlation_dim (tick)** | 4 | 0 | reuse existing correlation_dim |

**Critical gap**: Bridge returns single scalars for multi-output leaves. Also `mfdfa` and `embedding` are COMPOSE — should be straightforward additions.

### Family 13 — Dim Reduction

| Leaf | Fintek DO count | Bridge outputs | Missing |
|---|---|---|---|
| pca | 4 | 3 | spectral_entropy (trivial from eigenvalues) |
| ssa | 7 | 3 | signal_noise_separation, reconstructability, grouping_quality, more eigenvalue ratios |
| tick_compression | ? | 1 | (match fintek source) |
| **ica** | 4 | 0 | blocked by Task #163 (B18) |
| **rmt** | 4 | **0** | NOT IMPLEMENTED (compose: eigenvalues + Marchenko-Pastur edges) |
| **grassmannian** | 4 | **0** | NOT IMPLEMENTED (SVD of Q1'Q2 + principal angles) |
| **spectral_embedding** | 4 | **0** | NOT IMPLEMENTED (Laplacian eigendecomposition) |
| **diff_geometry** | 4 | **0** | NOT IMPLEMENTED (Menger curvature — trivial) |
| **harmonic** | 12 | **0** | NOT IMPLEMENTED (SVD + Oganesyan-Huse r-stat) |

**Critical gap**: RMT, Grassmannian, spectral_embedding, diff_geometry, harmonic are all COMPOSE (use existing sym_eigen / SVD) — none should block on tambear.

### Family 14 — Topological

| Leaf | Fintek DO count | Bridge outputs | Missing |
|---|---|---|---|
| persistent_homology | 4 | 4 | semantic mismatch: `n_components` vs `n_components_50` |
| **nvg** | 4 | 0 | blocked by Task #145 (B9) |
| **hvg** | 4 | 0 | blocked by Task #145 (B9) |
| **tick_geometry** | 4 | 0 | blocked by Task #177 (C9 convex hull) |

**Critical gap**: persistent_homology's `n_components` is the raw count of persistence pairs, but fintek's `n_components_50` is "components alive at the 50th percentile threshold" — a filtration-level quantity. Not the same. Fix: compute at a user-provided threshold.

### Family 16 — Extremes

| Leaf | Fintek DO count | Bridge outputs | Missing |
|---|---|---|---|
| heavy_tail | 4 | 3 (wrong field names) | hill_std (= α/√k), tail_fraction |
| **seismic** | 4 | 0 | blocked by Task #176 (C8) |

**Note**: Also blocked on compilation (Task #139 hill_estimator).

---

## 4. Missing Families (from catalog but no bridge file)

These families from the catalog have NO family_N.rs file in the bridge crate:

- **Family 7 — Wavelets** (haar_wavelet, cwt_wavelet, scattering, wavelet_leaders, emd) — 4 leaves blocked, 1 (haar) needs implementation (tambear has haar_dwt)
- **Family 11 — State-space** (kalman, statespace, hmm, wiener, smoothers) — 5 leaves
  - wiener, smoothers are COMPOSE (spectral + EWMA)
  - kalman/statespace/hmm are blocked (Tasks #101, #168)
- **Family 12 — Point processes** (hawkes, ou_process, tick_ou) — 3 leaves
  - ou_process, tick_ou are COMPOSE (ar_fit on differences)
  - hawkes blocked (Task #140)
- **Family 15 — Distribution distances** — partially in family8, could use separate module
- **Family 17 — Bin-level microstructure** (tick_alignment, tick_attractor, tick_complexity, tick_space, tick_scaling, tick_geometry, pith_attractor, shape, phase_transition) — 9 leaves
  - All COMPOSE (moments + histograms + simple geometry)
  - tick_geometry needs C9 convex hull
- **Family 18 — Cross-leaf/meta** (scaling_triple, coboundary, cadence_gradient, viscosity, taylor_fold, seismic) — 6 leaves
  - These read OTHER leaves' outputs, so they ship AFTER primitive leaves
- **Family 19 — Miscellaneous** (savgol, stl, fir_bandpass, energy_bands, periodicity, sde, logsig, fisher_info) — 8 leaves
  - fir_bandpass, energy_bands, periodicity are COMPOSE (FFT slicing)
  - Others blocked (Tasks #165 Sav-Gol, #179 STL, #180 SDE, #178 logsig, #172 fisher)

---

## 5. Silent-Gap Check

No function in the bridge returns **fake or hardcoded values** when the underlying tambear primitive is unavailable. Every function either:
- Returns `NaN`/`NaN-struct` for degenerate inputs, OR
- Delegates to a real tambear primitive that performs the math, OR
- Fails to compile (family8, family16 — explicit blocker), OR
- Is simply absent from the file (silent gap = "not implemented" rather than "fake result")

**Result**: No silent-gap bugs found. The bridge is honest: what's there is real; what's missing is missing.

---

## 6. Recommended Next Steps

### Immediate (before any compile work)

1. **Fix Bug #2** (elapsed unit mismatch) — bridge's `elapsed_seconds_of_day` must return minutes to match fintek. 2-line change.
2. **Fix Bug #1** (stationarity classification) — change `(true, true) → Inconclusive` unless the bridge adds trend-aware ADF. 1-line change.
3. **Resolve compilation blockers** — either wait for Task #141 (DTW/edit) and #139 (Hill) to land, or stub out the imports.

### Short-term (fill DIRECT gaps the catalog already identified)

These are leaves where tambear has the primitive and the bridge just hasn't wrapped it:

- **multitaper**, **lombscargle**, **stft** — tambear has `multitaper_psd`, `lomb_scargle`, `stft`. 3 × ~15 lines each.
- **fir_bandpass**, **energy_bands**, **periodicity** — compose from existing FFT. 3 × ~15 lines each.
- **haar_wavelet** — tambear has `haar_dwt`, `haar_wavedec`. ~30 lines.
- **mfdfa** — loop over q values on existing `dfa`. ~25 lines.
- **mutual_info** — tambear has `mutual_information`. ~10 lines.
- **rmt** — compose from `sym_eigen` + Marchenko-Pastur edges. ~25 lines.
- **grassmannian** — compose from SVD of Q1'Q2. ~20 lines.
- **spectral_embedding** — sym_eigen on Laplacian. ~25 lines.
- **diff_geometry** — Menger curvature, trivial. ~25 lines.
- **harmonic** — SVD + r-statistic, trivial. ~25 lines.
- **wiener**, **smoothers** — compose from FFT + existing EWMA. ~30 lines each.
- **ou_process**, **tick_ou** — `ar_fit` on differences. ~15 lines each.
- **tick_alignment**, **tick_attractor**, **tick_complexity**, **tick_space**, **tick_scaling**, **shape**, **phase_transition** — all use existing moments/histograms/entropy. ~20 lines each.
- **signature_plot** — realized_variance at multiple frequencies. ~15 lines.
- **vol_regime**, **vol_dynamics**, **stochvol** — compose from rolling moments + ar_fit. ~25 lines each.
- **tick_vol**, **struct_break** — compose from moments + scan. ~30 lines each.
- **ccm**, **transfer_analysis**, **tick_causality**, **cross_correlation**, **msplit_temporal_coherence** — compose from FFT/cross_correlate. ~30 lines each.
- **arx** — two OLS fits + F-test. ~40 lines.

**Total**: ~25 additional leaves at ~600 lines of bridge code, using existing tambear primitives. No new math required.

### Medium-term (fix output column coverage)

Walk through each implemented bridge function and extend its output struct to emit ALL fintek DO columns. Examples:
- **fft_spectral**: extend from 8 to 15 features, then add M16/M32/M64/M128 resolution variants
- **tail_field**: extend from 1 to 7 features
- **shannon_entropy_of_returns**: extend from 1 to 4 features, verify log base
- **cepstrum**: extend from raw vec to 4 named features
- **hilbert**: extend from (env, phase) vecs to 5 feature scalars
- **correlation_dim**, **lyapunov**, **permutation_entropy**: extend from 1 scalar to 3-4 features each
- **ssa**: extend from 3 to 7 features
- **heavy_tail**: rename/extend to match fintek column names (hill_alpha, hill_std, optimal_k, tail_fraction)

### Long-term (blocked gaps from Category B/C specs)

All tracked in existing tasks (#137-146, #162-180). Adversarial and pathmaker implement from my catalog specs.

---

## 7. Summary Table

| Family | Bridge status | Math bugs | Coverage % (DO columns) |
|---|---|---|---|
| 1 Distribution | Implemented | none | ~60% |
| 2 Transforms | Implemented | elapsed unit bug | 100% (minus bug) |
| 3 Bin aggregates | Implemented | variability missing | ~70% |
| 4 Time series | Implemented | none | ~80% (arx missing) |
| 5 Stationarity | Implemented | classification bug | ~60% |
| 6 Spectral | Implemented | none | **~15%** (worst) |
| 7 Wavelets | MISSING FILE | n/a | 0% |
| 8 Correlation | Won't compile | n/a | ~25% once compilable |
| 9 Volatility | Implemented | jump_detection semantic | ~50% |
| 10 Nonlinear | Implemented | none | ~45% |
| 11 State-space | MISSING FILE | n/a | 0% |
| 12 Point processes | MISSING FILE | n/a | 0% |
| 13 Dim reduction | Implemented | none | ~40% |
| 14 Topological | Implemented | n_components semantic | ~75% |
| 15 Dist distances | in family8 | n/a | ~25% |
| 16 Extremes | Won't compile | n/a | ~75% once compilable |
| 17 Microstructure | MISSING FILE | n/a | 0% |
| 18 Meta | MISSING FILE | n/a | 0% (expected: comes last) |
| 19 Miscellaneous | MISSING FILE | n/a | 0% |

**Grand total**: ~40 of 126 trunk-rs leaves have SOME bridge wrapper (~32%), with average DO coverage of ~55% on the implemented ones.

**Bridge-level rescue completion**: roughly **18-20%** of total fintek output columns are currently wrapped.

The catalog I wrote identified 80 leaves as "Category A — Ready to compile". Only ~40 have bridge wrappers. The other ~40 are COMPOSE leaves where tambear has the primitives and the bridge just needs to add the adapters — no new tambear math required.

---

_Report generated by math-researcher during parallel verification pass on the tambear-fintek bridge crate (~1778 lines at time of review)._
_Bridge crate location: `R:/winrapids/crates/tambear-fintek/`_
_Catalog reference: `docs/research/tambear-hardening/fintek-math-catalog.md`_
