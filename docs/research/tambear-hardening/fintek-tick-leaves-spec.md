# Fintek tick_* Leaves Spec

**Author**: math-researcher
**Date**: 2026-04-09
**Scope**: 10 tick-level leaves in `R:/fintek/crates/trunk-rs/src/leaves/tick_*.rs` (1,512 lines total)
**Purpose**: Spec for tambear primitives + bridge wrappers to replace these leaves.

**Clarification from team lead**: "binned" means cadences, but tambear math is cadence-agnostic — it takes slices. Every tick_* leaf here is a pure `fn(&[f64]) -> [f64; 4]` plus a few E-column predicates. No cadence logic in tambear.

---

## Summary Table

| Leaf | ID | Outputs | Inputs | Tambear primitives needed | Status |
|------|-----|---------|--------|--------------------------|--------|
| tick_alignment | K02P16C04R01 | arrival_regularity, clustering_index, gap_ratio, uniformity_score | timestamps (i64 ns) | All COMPOSE from moments + sort + KS | COMPOSE |
| tick_attractor | K02P14C05R01 | phase_asymmetry, tick_persistence, reversal_rate, phase_spread | price (f64) | COMPOSE from moments, pairwise lag, sign counting | COMPOSE |
| tick_causality | K02P17C03R01 | lead_lag_corr, lead_lag_offset, coupling_strength, impulse_ratio | price, size | COMPOSE from lagged cross-correlation, abs_returns, quantile | COMPOSE |
| tick_complexity | K02P13C04R01 | inter_arrival_entropy, size_entropy, joint_entropy, normalized_complexity | timestamps, size | COMPOSE from binned Shannon entropy (already in info_theory) | COMPOSE |
| tick_compression | K02P25C01 | real_effective_rank, shuffled_effective_rank, compression_ratio, n_active_features | price, size | COMPOSE from SVD, standardization, shuffle RNG, effective rank from eigenvalues | COMPOSE |
| tick_geometry | K02P15C6R1 | hull_area, angular_entropy, radial_kurtosis, aspect_ratio | price (f64) | COMPOSE from convex_hull_2d (#177 DONE), Shoelace, 2×2 sym_eigen | COMPOSE |
| tick_ou | K02P11C05R01 | tick_theta, tick_half_life, tick_sigma, mr_strength | price (f64) | COMPOSE from OLS on differences (AR(1) form) | COMPOSE |
| tick_scaling | K02P12C03R01 | scaling_exponent, scaling_r2, xmin, burstiness | timestamps (i64 ns) | COMPOSE from log-log OLS on CCDF + Hill-like power law | COMPOSE |
| tick_space | K02P18C03R03F01 | tick_entropy, mode_concentration, tick_clustering, regime_persistence | price (f64) | COMPOSE from binned entropy, mode counting, lag-1 autocorr, run-length | COMPOSE |
| tick_vol | K02P10C02R03F01 | realized_var, bipower_var, tick_frequency_var, microstructure_noise | price → returns | COMPOSE — all DIRECT (realized_variance, bipower_variation exist) | COMPOSE |

**Every single tick leaf is Category A (COMPOSE)** — no new tambear math is needed. They all compose existing primitives. Total new bridge code: ~500 lines across the 10 wrappers.

---

## Input Conventions

Three input types across the 10 leaves:

1. **`price: &[f64]`** — raw prices (tick_attractor, tick_geometry, tick_ou, tick_space, tick_vol, tick_compression also takes volume)
2. **`timestamps_ns: &[i64]`** — tick timestamps in nanoseconds (tick_alignment, tick_complexity, tick_scaling)
3. **`price: &[f64], size: &[f64]`** — price and size together (tick_causality, tick_compression, tick_complexity mixes timestamps+size)

The leaf pattern is always: receive a slice, return 4 scalar outputs + NaN on degenerate input.

---

## Per-Leaf Specs

### 1. `tick_alignment` — arrival regularity

**Algorithm** (on `&[i64]` timestamps):
1. Compute inter-tick differences `dts[i] = ts[i+1] - ts[i]`, drop zero/negative, require ≥ 5 valid diffs.
2. **DO01 arrival_regularity** = `(1 - std(dts)/mean(dts)).max(0)` — inverse coefficient of variation.
3. **DO02 clustering_index** = fraction of gaps < median(dts)/2.
4. **DO03 gap_ratio** = `max(dts) / mean(dts)`.
5. **DO04 uniformity_score** = `1 - max KS deviation` between empirical CDF of ts and uniform on `[ts[0], ts[n-1]]`.

**Min samples**: 20 ticks (MIN_TICKS), 5 nonzero dts.

**Bridge signature**:
```rust
pub struct TickAlignmentResult {
    pub arrival_regularity: f64,
    pub clustering_index: f64,
    pub gap_ratio: f64,
    pub uniformity_score: f64,
}
pub fn tick_alignment(timestamps_ns: &[i64]) -> TickAlignmentResult;
```

**E columns**: `e_irregular = regularity < 0.3`, `e_gapped = gap_ratio > 10`.

**Tambear primitives used**: none new — just `descriptive::moments_ungrouped` on f64-converted dts, sort, KS statistic (already in `nonparametric::ks_test_*`).

**Gotcha**: Fintek converts `i64` ns to `f64` inside the loop. Bridge should accept `&[i64]` directly so fintek can pass its column unchanged.

---

### 2. `tick_attractor` — phase portrait dynamics

**Algorithm** (on `&[f64]` price):
1. Compute raw price changes `dp[i] = price[i+1] - price[i]` (NOT log returns — this is deliberate).
2. Require n_dp ≥ 2 and var(dp) > 1e-30.
3. **DO01 phase_asymmetry** = `sqrt(Σ(p_q - 0.25)²)` where p_q is the proportion of `(dp[t], dp[t-1])` pairs in each of 4 quadrants.
4. **DO02 tick_persistence** = lag-1 autocorrelation of dp = `Σ centered[t]·centered[t+1] / Σ centered²`.
5. **DO03 reversal_rate** = fraction of sign changes in dp (ignoring zeros).
6. **DO04 phase_spread** = std of radial distances `sqrt(dp[t]² + dp[t-1]²)` (ddof=1).

**Min samples**: 3 price ticks (gives 2 dp, 1 pair).

**Bridge signature**:
```rust
pub struct TickAttractorResult {
    pub phase_asymmetry: f64,
    pub tick_persistence: f64,
    pub reversal_rate: f64,
    pub phase_spread: f64,
}
pub fn tick_attractor(prices: &[f64]) -> TickAttractorResult;
```

**E columns**: `e_reversal = reversal_rate > 0.6`, `e_persistent = |persistence| > 0.3`.

**Tambear primitives used**: trivially composed from `descriptive::moments_ungrouped` and sign counting. No new math.

**Gotcha**: lag-1 autocorrelation here is ddof=0 in numerator (not Bessel-corrected) — matches fintek exactly.

---

### 3. `tick_causality` — lead-lag price/volume

**Algorithm** (on `&[f64]` price, `&[f64]` size):
1. Compute log returns `r[i] = ln(price[i+1]/price[i])` and align volume `v[i] = size[i]` for i in 0..n-1.
2. Min ≥ MIN_TICKS = 20 valid points.
3. **DO01 lead_lag_corr** = `|best_corr|` where best_corr is the max-absolute-value Pearson correlation between `r[t+lag]` and `v[t]` (positive lag = volume leads) or `r[t]` and `v[t+lag]` (negative lag = returns lead), for lag in 1..=MAX_LAG=5.
4. **DO02 lead_lag_offset** = `best_lag` (signed, `+` = volume leads).
5. **DO03 coupling_strength** = `|corr(|r|, v)|` at lag 0.
6. **DO04 impulse_ratio** = fraction of high-volume ticks (vol > 90th pct) followed by high |return| ticks (|r[t+1]| > 75th pct).

**Bridge signature**:
```rust
pub struct TickCausalityResult {
    pub lead_lag_corr: f64,
    pub lead_lag_offset: f64,  // signed lag
    pub coupling_strength: f64,
    pub impulse_ratio: f64,
}
pub fn tick_causality(prices: &[f64], sizes: &[f64]) -> TickCausalityResult;
```

**E columns**: `e_lead_lag = corr > 0.2`, `e_impulse = impulse > 0.5`.

**Tambear primitives used**: Pearson correlation at lag, quantile (both exist). Could also use `signal_processing::cross_correlate` but fintek's inline formula avoids the FFT overhead for small windows.

---

### 4. `tick_complexity` — inter-arrival + size entropy

**Algorithm** (on `&[i64]` timestamps, `&[f64]` size):
1. Compute `iat[i] = |ts[i+1] - ts[i]|.max(1e-30)` as f64.
2. Min MIN_TICKS = 10 ticks.
3. Bin each series into N_HIST_BINS = 16 linear bins over `[min, max]`.
4. **DO01 inter_arrival_entropy** = Shannon entropy of iat histogram (nats, natural log).
5. **DO02 size_entropy** = Shannon entropy of size histogram.
6. **DO03 joint_entropy** = Shannon entropy of joint (iat[t], size[t]) histogram (16×16 = 256 cells).
7. **DO04 normalized_complexity** = `(H_iat + H_size) / (2·ln(N_HIST_BINS))`.

**Bridge signature**:
```rust
pub struct TickComplexityResult {
    pub inter_arrival_entropy: f64,
    pub size_entropy: f64,
    pub joint_entropy: f64,
    pub normalized_complexity: f64,
}
pub fn tick_complexity(timestamps_ns: &[i64], sizes: &[f64], n_bins: usize) -> TickComplexityResult;
```

**Note**: `n_bins` should default to 16 but be tunable (Tekgy directive — every parameter tunable).

**E columns**: `e_high_complex = norm_c > 0.7`, `e_low_complex = norm_c < 0.3`.

**Tambear primitives used**: `information_theory::shannon_entropy_from_counts` (already exists). Just need a small helper for `bin_and_entropy(data, n_bins)` that does linear bucketing.

---

### 5. `tick_compression` — effective rank of feature matrix

**Algorithm** (on `&[f64]` price, `&[f64]` size):
1. Build 7-feature matrix (m × 7) where m = n - 1:
   - col 0: log_ret = `ln(p[i+1]/p[i])`
   - col 1: |log_ret|
   - col 2: log_ret²
   - col 3: log_vol = `ln(v[i+1])`
   - col 4: d_log_vol = `ln(v[i+1]/v[i])`
   - col 5: |d_log_vol|
   - col 6: log_ret · d_log_vol (interaction)
2. Subsample to MAX_PTS = 2000 if larger (memory cap).
3. Compute column means and stds; mark columns with std > 1e-15 as active.
4. Standardize active columns: `z[i,j] = (mat[i,j] - mean[j]) / std[j]`.
5. Compute SVD singular values of (m × n_active) matrix.
6. **Effective rank** = `exp(Shannon entropy of (sv²/Σsv²))`.
7. **DO01 real_effective_rank** = effective rank of real data.
8. **DO02 shuffled_effective_rank** = mean effective rank over N_SHUFFLES = 5 column-shuffled versions (using xorshift64 with seed 42 for determinism).
9. **DO03 compression_ratio** = real / shuffled.
10. **DO04 n_active_features** = count of active columns.

**Bridge signature**:
```rust
pub struct TickCompressionResult {
    pub real_effective_rank: f64,
    pub shuffled_effective_rank: f64,
    pub compression_ratio: f64,
    pub n_active_features: usize,
}
pub fn tick_compression(
    prices: &[f64],
    sizes: &[f64],
    max_pts: usize,
    n_shuffles: usize,
    seed: u64,
) -> TickCompressionResult;
```

**Params**: `max_pts` default 2000, `n_shuffles` default 5, `seed` default 42.

**E columns**: `e_compressed = ratio < 0.7`, `e_noise = ratio > 0.95`, `e_low_rank = real_rank < 2.0`.

**Tambear primitives used**: 
- `dim_reduction::svd_singular_values` (exists)
- Effective rank helper: `fn effective_rank(sv: &[f64]) -> f64` (from fintek code, ~10 lines — should be promoted to `spectral.rs` as a reusable primitive, it's the same thing the bridge `delay_pca` computes)
- Fisher-Yates shuffle with deterministic RNG (use `rng::Xoshiro256` instead of xorshift — task #63 directive: no LCG-family RNGs)

**Gotcha**: fintek uses `Xorshift64` with seed 42 for determinism. Tambear must use `Xoshiro256::new(42)` to match — this will NOT be bit-perfect with fintek's xorshift but shuffled_effective_rank has a different numerical value each shuffle anyway, and the ratio is statistically robust. Document the tolerance ~1e-3 for this specific output.

---

### 6. `tick_geometry` — phase-plane geometry

**Algorithm** (on `&[f64]` price):
1. Compute log returns; subsample to MAX_PTS = 5000 if larger.
2. Phase portrait points: `(x[t] = r[t], y[t] = r[t-1])` for t in 1..n.
3. **DO01 hull_area** = convex hull area via gift wrapping + Shoelace formula.
4. **DO02 angular_entropy** = normalized Shannon entropy over 32 angular bins in `[-π, π]`, using `angle = atan2(y, x)`.
5. **DO03 radial_kurtosis** = excess kurtosis of radial distances `r_t = sqrt(x² + y²)`.
6. **DO04 aspect_ratio** = ratio `λ_min/λ_max` of 2×2 scatter matrix eigenvalues (closed form: `(sxx+syy)/2 ± sqrt(((sxx-syy)/2)² + sxy²)`).

**Min samples**: MIN_RETURNS + 1 = 21 ticks.

**Bridge signature**:
```rust
pub struct TickGeometryResult {
    pub hull_area: f64,
    pub angular_entropy: f64,
    pub radial_kurtosis: f64,
    pub aspect_ratio: f64,
}
pub fn tick_geometry(prices: &[f64], n_angle_bins: usize, max_pts: usize) -> TickGeometryResult;
```

**Params**: `n_angle_bins` default 32, `max_pts` default 5000.

**E columns**: `e_elongated = aspect < 0.3`, `e_isotropic = angular_entropy > 0.9`.

**Tambear primitives used**: 
- `graph::convex_hull_2d` (Task #177 DONE, wave 15)
- `descriptive::moments_ungrouped` for radial_kurtosis (excess kurtosis via m2, m4)
- Closed-form 2×2 eigenvalues (5 lines inline — or we add `linear_algebra::eig_2x2_sym(sxx, syy, sxy) -> (λ_large, λ_small)` as a utility)

**Gotcha**: Fintek's convex hull is gift wrapping (Jarvis march). Tambear's `convex_hull_2d` uses Graham scan. Both produce the same hull. Shoelace area is identical. Bit-perfect or ~1e-12 tolerance.

---

### 7. `tick_ou` — Ornstein-Uhlenbeck on log-prices

**Algorithm** (on `&[f64]` price):
1. Log prices: `x[i] = ln(max(price[i], 1e-300))`.
2. Fit OLS on first differences: `Δx = a + b·x[i] + ε`.
3. **DO01 tick_theta** = `-b` (mean-reversion speed).
4. **DO02 tick_half_life** = `ln(2) / theta` if theta > 1e-10, else Inf.
5. **DO03 tick_sigma** = residual std with ddof=1.
6. **DO04 mr_strength** = `theta · std(log_prices) / sigma` (dimensionless).

**Min samples**: 5 price ticks.

**Bridge signature**:
```rust
pub struct TickOuResult {
    pub tick_theta: f64,
    pub tick_half_life: f64,
    pub tick_sigma: f64,
    pub mr_strength: f64,
}
pub fn tick_ou(prices: &[f64]) -> TickOuResult;
```

**E columns**: `e_reverting = theta > 0`, `e_strong_mr = mr_strength > 0.5`.

**Tambear primitives used**: Inline OLS on the AR(1)-form differences (5 running sums: sum_x, sum_dx, sum_x², sum_xdx, plus ss_res loop). The existing `time_series::ar_fit` could be used but the scalar case is cleaner inline.

**Opportunity**: There's an existing `stochastic::ornstein_uhlenbeck` simulator but no FIT function. Worth adding `pub fn ou_fit_ols(data: &[f64]) -> OuFitResult` to stochastic.rs — used by tick_ou, ou_process (non-tick variant), any future OU-based leaf.

---

### 8. `tick_scaling` — power-law scaling of inter-arrivals

**Algorithm** (on `&[i64]` timestamps):
1. Compute absolute inter-arrival times `iat`, drop zeros, min 10 valid.
2. Sort ascending.
3. **DO04 burstiness** = `(σ - μ) / (σ + μ)` where μ, σ are mean and std of iat (Goh-Barabási burstiness).
4. Power-law fit on the tail:
   - xmin = iat[n/10] (10th percentile)
   - Log-log OLS: `log(CCDF)` vs `log(x)` over `x >= xmin`, where `CCDF[j] = (n_tail - j) / n_tail`
   - slope = `(n·Σxy - Σx·Σy) / (n·Σx² - (Σx)²)`
   - intercept = `(Σy - slope·Σx) / n`
   - **DO01 scaling_exponent** = `-slope + 1` (α from CCDF ~ x^{-(α-1)})
   - **DO02 scaling_r2** = `1 - ss_res/ss_tot`
5. **DO03 xmin** = the xmin value used.

**Min samples**: 20 ticks, 10 positive iat, 5 log points in tail.

**Bridge signature**:
```rust
pub struct TickScalingResult {
    pub scaling_exponent: f64,
    pub scaling_r2: f64,
    pub xmin: f64,
    pub burstiness: f64,
}
pub fn tick_scaling(timestamps_ns: &[i64], xmin_percentile: f64) -> TickScalingResult;
```

**Params**: `xmin_percentile` default 0.1 (10th percentile).

**E columns**: `e_power_law = r2 > 0.8`, `e_bursty = burstiness > 0.3`.

**Tambear primitives used**: Inline log-log OLS (same 5 running sums). Could use existing `time_series::ar_fit`-style helper but inline is cleaner.

**Opportunity**: Promote a generic `pub fn loglog_ols(x: &[f64], y: &[f64]) -> LoglogFit { slope, intercept, r2 }` to numerical.rs or time_series.rs. It's used by tick_scaling, DFA, Hurst R/S, mfdfa, any scaling exponent computation — easily 5+ consumers.

---

### 9. `tick_space` — tick-size distribution & regime persistence

**Algorithm** (on `&[f64]` price):
1. Price differences `dp[i] = price[i+1] - price[i]`, tick sizes `|dp[i]|`.
2. Filter nonzero ticks, require ≥ 10.
3. **DO01 tick_entropy** = Shannon entropy of nonzero tick sizes binned into N_ENTROPY_BINS = 50 linear bins.
4. **DO02 mode_concentration** = fraction of ticks at the mode when rounded to precision `median(ticks)/100`.
5. **DO03 tick_clustering** = lag-1 autocorrelation of `|dp|` clamped to `[-1, 1]`.
6. **DO04 regime_persistence** = mean run length of same-sign `dp` sequences (excluding zero-sign runs in interior).

**Min samples**: 11 price ticks.

**Bridge signature**:
```rust
pub struct TickSpaceResult {
    pub tick_entropy: f64,
    pub mode_concentration: f64,
    pub tick_clustering: f64,
    pub regime_persistence: f64,
}
pub fn tick_space(prices: &[f64], n_entropy_bins: usize) -> TickSpaceResult;
```

**Params**: `n_entropy_bins` default 50.

**E columns**: `e_concentrated = mode_conc > 0.5`, `e_clustered = clustering > 0.3`.

**Tambear primitives used**: Binned entropy (same pattern as tick_complexity). Lag-1 autocorrelation (moments-based). Run-length encoding (inline, trivial).

**Opportunity**: `pub fn run_length_encode<T: Eq>(signs: &[T]) -> Vec<usize>` (general primitive, useful for regime analysis, LZ complexity, streaks). Promote to `numerical.rs` or `nonparametric.rs`.

---

### 10. `tick_vol` — realized variance at tick level

**Algorithm** (on log returns computed from `&[f64]` price):
1. Log returns: `r[i] = ln(price[i+1]/price[i])`.
2. Min n ≥ 10 returns.
3. **DO01 realized_var** = `Σ r²` — **DIRECT** via `volatility::realized_variance`.
4. **DO02 bipower_var** = `(π/2) · Σ |r[i]|·|r[i-1]|` for i ≥ 1 — **DIRECT** via `volatility::bipower_variation` (after checking the normalization matches fintek's (π/2) convention).
5. **DO03 tick_frequency_var** = `realized_var` (fintek's code just computes `(RV/n) · n = RV` — this appears to be a placeholder; maybe should be standardized differently). FLAG FOR FINTEK REVIEW.
6. **DO04 microstructure_noise** = `RV - mean_over_5_offset_grids(subsampled_RV)`, where each sub-sampled grid uses block returns of width 5.

**Bridge signature**:
```rust
pub struct TickVolResult {
    pub realized_var: f64,
    pub bipower_var: f64,
    pub tick_frequency_var: f64,
    pub microstructure_noise: f64,
}
pub fn tick_vol(prices: &[f64], subsample_k: usize) -> TickVolResult;
```

**Params**: `subsample_k` default 5.

**E columns**: `e_jump = RV > 1.2·BV`, `e_noisy = noise/RV > 0.2`, `e_high_rv = RV > 2·mean_RV`.

**Note on E03 `e_high_rv`**: this predicate requires cross-bin aggregation (mean over the day's bins) and is NOT a per-call predicate. The bridge function can't compute it from a single slice — fintek must compute e_high_rv in a second pass after collecting all per-bin RVs. Document this in the bridge API.

**Tambear primitives used**: `volatility::realized_variance` and `volatility::bipower_variation` — both DIRECT. Subsampling is a new inline helper.

**Tambear addition**: `pub fn subsampled_rv(returns: &[f64], k: usize) -> f64` — averages RV over k offset grids with block-sum returns. ~20 lines to add to volatility.rs. Useful beyond tick_vol: any microstructure noise estimation (Zhang et al. 2005 "Two-scale realized volatility").

---

## Bridge Module Plan

All 10 tick leaves should go into a new bridge module:

```rust
// R:/winrapids/crates/tambear-fintek/src/family17_microstructure.rs
```

**Family 17 — Tick-Level Microstructure**. This matches the "tick*" grouping in the fintek-math-catalog.md Family 17.

The module exposes 10 `tick_*` functions, each returning a `*Result` struct. Callers (fintek) use these inside their leaf `execute()` methods per-bin.

**Estimated bridge code**: ~500 lines (50 lines per leaf wrapper on average).

**Estimated tambear additions**: 
- `spectral::effective_rank(sv: &[f64]) -> f64` — already computed in the bridge's `delay_pca`, promote to shared helper (~10 lines)
- `numerical::loglog_ols(x: &[f64], y: &[f64]) -> LoglogFit` — reusable scaling-exponent fit (~25 lines)
- `numerical::run_length_encode<T: Eq>(seq: &[T]) -> Vec<usize>` — reusable (~10 lines)
- `stochastic::ou_fit_ols(data: &[f64]) -> OuFitResult` — OU parameter estimation (~25 lines)
- `volatility::subsampled_rv(returns: &[f64], k: usize) -> f64` — Zhang et al. 2005 two-scale RV subsampling (~20 lines)
- `linear_algebra::eig_2x2_sym(sxx, syy, sxy) -> (λ_large, λ_small)` — closed-form 2×2 eigenvalues (~8 lines)

**Total tambear additions**: ~100 lines of new math, all COMPOSE from existing primitives.

---

## Why All 10 Are Easy

The pattern across all 10 leaves is:
1. Take a slice (price, timestamps, size, or combination).
2. Compute a small feature vector (n-1 diffs, sorted array, binned histogram, phase-plane points, OLS regression, SVD).
3. Extract 4 scalar summaries.
4. Apply 2-3 predicates to produce E columns.

None of this is new mathematics. It's all standard descriptive/dynamical/spectral analysis that tambear already supports. The tick_* leaves are Category A (COMPOSE) in the catalog — they unlock immediately once the bridge adapters are written.

**Priority ordering for adversarial**:
1. **tick_vol** — 80% DIRECT (realized_var, bipower_var exist). 15 lines.
2. **tick_attractor** — pure moments and sign counting. 40 lines.
3. **tick_ou** — inline OLS, 5 running sums. 40 lines. Promotes `ou_fit_ols` to stochastic.rs.
4. **tick_alignment** — moments + sort + KS. 50 lines.
5. **tick_space** — binned entropy + lag autocorr + run-length. 60 lines. Promotes `run_length_encode` to numerical.rs.
6. **tick_complexity** — binned entropy with i64 timestamps. 40 lines.
7. **tick_scaling** — log-log OLS on CCDF + burstiness. 50 lines. Promotes `loglog_ols` to numerical.rs.
8. **tick_causality** — lead-lag cross-correlation + quantile thresholds. 60 lines.
9. **tick_geometry** — uses existing convex_hull_2d + 2×2 eigenvalues. 70 lines. Promotes `eig_2x2_sym` to linear_algebra.rs.
10. **tick_compression** — SVD + effective rank + Fisher-Yates shuffle. 80 lines. Promotes `effective_rank(sv)` to spectral.rs. Note: shuffle uses Xoshiro256 not xorshift64, so `shuffled_effective_rank` output will differ from fintek's exact values but ratio is robust.

**Total: ~500 bridge lines + ~100 tambear helper lines = ~600 lines**. Zero new math research needed.

---

_Document is the formal spec for the 10 fintek tick_* leaves._
_Adversarial can implement directly from this spec without consulting the fintek source._
