# Fintek Leaf Signatures — Shim Contract Reference

*Generated: 2026-04-08 — tambear hardening expedition, fintek rescue*

**Purpose**: Exact struct names, IDs, output column specs, and context inputs for every fintek trunk-rs leaf. Adversarial uses this to write shims without re-reading source. Math researcher uses the catalog (`fintek-math-catalog.md`) for algorithm mapping.

---

## The Contract

Every leaf is a zero-field (or const-parameterized) struct implementing:

```rust
pub trait Leaf: Send + Sync {
    fn id(&self) -> &str;
    fn execute(&self, ctx: &dyn ExecutionContext, cadence_id: u8) -> Result<LeafOutput, Box<dyn std::error::Error>>;
    fn outputs(&self) -> &[(&str, Dtype)];
}

pub type LeafOutput = HashMap<String, ColumnData>;

pub enum ColumnData {
    F64(Vec<f64>),
    F32(Vec<f32>),
    I64(Vec<i64>),
    I32(Vec<i32>),
    U32(Vec<u32>),
    U8(Vec<u8>),
}
```

**Context inputs** (what leaves pull from `ctx`):
- `ctx.column_f64("price")` — tick-level price (all K01/K02 leaves)
- `ctx.column_f64("size")` — tick-level trade size
- `ctx.column_i64("K01P01.DI01DO03")` — nanosecond timestamps
- `ctx.column_f64("K01P02C01.DI01DO01")` — notional (downstream dep)
- `ctx.bin_boundaries(cadence_id)` — `(&[u32] starts, &[u32] ends)` for K02 bins
- `ctx.n_bins(cadence_id)` — bin count

**Output keys**: `"DI01DO01"`, `"DI01DO02"` ... (DO columns) and `"E01"`, `"E02"` ... (emission columns, always F32).

**Shim pattern**: The shim struct has the same name, same `id()`, same `outputs()`. The `execute()` body calls tambear instead of local computation.

---

## Phylum K01P02 — Tick-Level Pointwise (Priority 1: all DIRECT)

All 10 leaves use `_cadence_id` (ignored — tick-level, not binned). No bin boundaries needed.

| Struct | ID | Inputs from ctx | Outputs | Output dtype | Notes |
|--------|-----|----------------|---------|-------------|-------|
| `Notional` | `K01P02C01` | `price`, `size` | `DI01DO01` (notional) | F64 | price × size per tick |
| `LogTransform` | `K01P02C02R01` | `price`, `size`, `K01P02C01.DI01DO01` | `DI01DO01` (ln_price), `DI02DO01` (ln_size), `DI03DO01` (ln_notional) | F64×3 | ln of each |
| `SqrtTransform` | `K01P02C02R02` | `price`, `size`, `K01P02C01.DI01DO01` | `DI01DO01`, `DI02DO01`, `DI03DO01` | F64×3 | sqrt of each |
| `Reciprocal` | `K01P02C02R03` | `price`, `size`, `K01P02C01.DI01DO01` | `DI01DO01`, `DI02DO01`, `DI03DO01` | F64×3 | 1/x for each |
| `Elapsed` | `K01P02C02R04` | `K01P01.DI01DO03` (i64 ns timestamps) | `DI01DO01` (minutes since midnight) | **F32** | `(ts % DAY_NS) / MINUTE_NS` |
| `Cyclical` | `K01P02C02R05` | `K01P01.DI01DO03` (i64 ns timestamps) | `DI01DO01` (sin_tod), `DI01DO02` (cos_tod) | **F32×2** | 24h unit circle |
| `DeltaValue` | `K01P02C03R02F01` | `price`, `size`, `K01P02C01.DI01DO01`, `K01P01.DI01DO03` | 35 cols: DI01-DI07 × DO01-DO05 | F64×35 | x_t − x_{t−n} for lags {1,2,3,5,10}, channels {ts,price,size,notional,ln_price,ln_size,ln_notional} |
| `DeltaPercent` | `K01P02C03R02F02` | `price`, `size`, `K01P02C01.DI01DO01` | 15 cols: DI01-DI03 × DO01-DO05 | F64×15 | pct delta at lags {1,2,3,5,10}, channels {price,size,notional} |
| `DeltaDirection` | `K01P02C03R02F03` | `price`, `size`, `K01P02C01.DI01DO01` | 15 cols: DI01-DI03 × DO01-DO05 | **I32×15** | sign(delta) ∈ {-1,0,+1}, same 3 channels |
| `DeltaLog` | `K01P02C03R02F04` | `price` | 5 cols: DI01DO01..DO05 | F64×5 | ln(p_t/p_{t-n}) at lags {1,2,3,5,10} |

**DeltaValue channel mapping**:
- DI01 = timestamp deltas (i64→f64 cast)
- DI02 = price deltas
- DI03 = size deltas
- DI04 = notional deltas
- DI05 = ln_price deltas
- DI06 = ln_size deltas
- DI07 = ln_notional deltas

**Shim note for K01P02**: All are pointwise maps over tick arrays. No bin grouping. Output length = n_ticks. The shim calls `tambear::compute_engine::map_phi` or direct iterator.

---

## Phylum K02P01 — Bin-Level Statistics (Priority 2)

All use `ctx.bin_boundaries(cadence_id)`. Output length = n_bins.

| Struct | ID | Inputs | Outputs (all F64) | E cols | Notes |
|--------|-----|--------|---------|--------|-------|
| `Ohlcv` | `K02P01C01` | `price`, `size`, `K01P01.DI01DO03` | DO01=open, DO02=high, DO03=low, DO04=close, DO05=volume, DO06=count, DO07=vwap, DO08=notional, DO09=realized_var | E01-E10 (F32) | E01=price_changed, E02-E04=gap pct, E05-E07=vol thresholds, E08=count>100, E09=range>0.5%, E10=rv>0.001 |
| `Distribution` | `K02P01C03` | `price` | DO01=mean, DO02=std, DO03=skewness, DO04=kurtosis, DO05=q10, DO06=q25, DO07=q50, DO08=q75, DO09=q90, DO10=realized_var, DO11=bipower_var, DO12=jump_test | none | 12 F64 outputs |
| `Validity` | `K02P01C04` | `price`, `K01P01.DI01DO03` | DO01=null_count, DO02=zero_count, DO03=negative_count, DO04=range, DO05=gap_max, DO06=density, DO07=coverage | none | 7 F64 outputs |
| `Returns` | `K02P01C05` | `price`, `size` | DO01=open_return, DO02=close_return, DO03=high_low_range, DO04=log_return, DO05=abs_return, DO06=signed_volume | none | 6 F64 outputs. Uses prev_close cross-bin state |
| `Counts` | `K02P01C06` | `price` | DO01=tick_count, DO02=unique_prices, DO03=price_changes, DO04=upticks, DO05=downticks | none | 5 F64 outputs |

**Shim note for `Ohlcv`**: Needs FirstOp/LastOp for open (first price) and close (last price) — task #136/#148. The `realized_var` and `bipower_var` are already in `tambear::volatility`. The 10 E columns are fintek-specific threshold triggers that the shim must also compute.

**Shim note for `Returns`**: Has cross-bin state (`prev_close`). The shim must carry this across bins within a single `execute()` call — the loop iterates bins sequentially.

---

## Phylum K02P02 — Spectral Analysis (Priority 3)

All use `price` → log-returns → FFT. Share FFT intermediate.

| Struct | ID | Inputs | Outputs (all F64) | Notes |
|--------|-----|--------|---------|-------|
| `FftSpectral { m: usize, id: &'static str }` | varies per M | `price` | DO01=dominant_freq, DO02=dominant_power, DO03=spectral_slope, DO04=slope_r2, DO05=spectral_entropy, DO06=edge_50, DO07=edge_90, DO08=centroid, DO09=bandwidth, DO10=total_power, DO11=low_band, DO12=mid_band, DO13=high_band, DO14=harmonic_ratio, DO15=inharmonicity | **15 F64 outputs**. Parameterized: M8=R05, M16=R01, M32=R02, M64=R03, M128=R04. Needs price regularized to M points before FFT |
| `Welch` | `K02P02C04R01` | `price` | DO01=spectral_centroid, DO02=spectral_bandwidth, DO03=spectral_entropy, DO04=peak_frequency | 4 F64. Hann window, 50% overlap, 64-pt segments |
| `Multitaper` | `K02P02C04R02` | `price` | DO01=spectral_centroid, DO02=spectral_bandwidth, DO03=spectral_entropy, DO04=f_test_significance | 4 F64. K=5 sine tapers |
| `LombScargle` | `K02P02C03R01` | `price`, `K01P01.DI01DO03` (i64) | DO01=peak_frequency, DO02=peak_power, DO03=spectral_slope, DO04=false_alarm_probability | 4 F64. Irregular timestamps → 64 test frequencies |
| `ArBurg` | `K02P02C05R01` | `price` | DO01=spectral_centroid, DO02=spectral_bandwidth, DO03=spectral_rolloff, DO04=ar_order | 4 F64. AR PSD via Burg lattice; needs tambear B2 |
| `Cepstrum` | `K02P02C06R01` | `price` | DO01=quefrency_peak, DO02=cepstral_energy, DO03=spectral_flatness, DO04=cepstral_distance | 4 F64. Real cepstrum = IFFT(log|FFT|²) |

**Shim note for `FftSpectral`**: The struct has fields `m` and `id`. The shim constructor takes these same args: `FftSpectralShim::new(m: usize, id: &'static str)`. The 5 instances are: `FftSpectral::new(8, "K02P02C02R05")`, `new(16, "K02P02C02R01")`, `new(32, "K02P02C02R02")`, `new(64, "K02P02C02R03")`, `new(128, "K02P02C02R04")`.

**Price regularization**: Before FFT, price is regularized to M equispaced points via linear interpolation. The shim must replicate this — it's not just raw FFT of prices.

---

## Phylum K02P06 — Statistical Tests

| Struct | ID | Inputs | Outputs (all F64) | Notes |
|--------|-----|--------|---------|-------|
| `Normality` | `K02P06C03R01` | `price` | DO01=jb_stat, DO02=jb_p, DO03=shapiro_w, DO04=shapiro_p, DO05=excess_kurtosis, DO06=skewness | 6 F64. jb from tambear::hypothesis, shapiro_w is D'Agostino-Pearson proxy |
| `Dependence` | `K02P06C02R01` | `price` | DO01=lb_stat, DO02=lb_p, DO03=optimal_lags, DO04=max_abs_acf | 4 F64. Ljung-Box from tambear::hypothesis |
| `Stationarity` | `K02P06C01R01` | `price` | DO01=adf_stat, DO02=adf_p, DO03=kpss_stat, DO04=kpss_p, DO05=stationarity_class, DO06=adf_lags, DO07=kpss_lags | 7 F64. Both tests from tambear::time_series |
| `HeavyTail` | `K02P06C04R01` | `price` | DO01=hill_alpha, DO02=hill_std | 2 F64. Hill estimator — needs tambear B6/task #139 |

---

## Phylum K02P07 — Correlation

| Struct | ID | Inputs | Outputs (all F64) | Notes |
|--------|-----|--------|---------|-------|
| `Autocorrelation` | `K02P07C01R01` | `price` | 16 F64: DO01-DO08 = ACF lags 1-8, DO09-DO16 = PACF lags 1-8 | tambear::time_series::acf + pacf |
| `CrossCorrelation` | `K02P07C02R01` | `price`, `size` | 5 F64: DO01-DO05 = CCF at lags (likely -2..+2 or peak features) | tambear::signal_processing::cross_correlate |
| `MsplitTemporalCoherence { m: usize, strategy: ..., id: &'static str }` | varies | `price` | DO01=mean_autocorr, DO02=split_half_corr, DO03=temporal_coherence, DO04=decorrelation_scale | 4 F64. 12 variants: M∈{4,8,16,32} × strategy∈{interp,bin_mean,subsample} |

**Shim note for `CrossCorrelation`**: Need to confirm exact lag structure — the 5 outputs likely include max_corr, lag_of_max, and summary stats. Read full leaf if exact mapping needed.

---

## Phylum K02P09 — Time Series Models

| Struct | ID | Inputs | Outputs (all F64) | Notes |
|--------|-----|--------|---------|-------|
| `ArModel` | `K02P09C01R01` | `price` | DO01=order, DO02=ar1_coeff, DO03=coeff_norm, DO04=residual_var, DO05=aic, DO06=root_modulus | 6 F64. Yule-Walker + BIC order selection |
| `Arima` | `K02P09C01R02` | `price` | DO01=p, DO02=d, DO03=ar1_coeff | 3 F64. Auto-d via ADF, then AR(p) |
| `Arma` | `K02P09C02R01` | `price` | DO01=p, DO02=q | 2 F64. AR(p) + residual ACF for MA order |
| `Statespace` | `K02P09C03R01` | `price` | DO01=signal_noise_ratio, DO02=state_var, DO03=obs_var, DO04=smoothness, DO05=loglikelihood | 5 F64. Local level Kalman — needs tambear B3 |

---

## Phylum K02P10 — Volatility (10 leaves)

| Struct | ID | Inputs | Outputs (all F64) | Notes |
|--------|-----|--------|---------|-------|
| `Garch` | `K02P10C01R01F01` | `price` | DO01=omega, DO02=alpha, DO03=beta, DO04=persistence, DO05=unconditional_var | 5 F64. GARCH(1,1) grid search |
| `StochVol` | `K02P10C01R02F01` | `price` | DO01=vol_of_vol, DO02=mean_log_vol, DO03=persistence, DO04=leverage_corr | 4 F64. AR(1) on log(r²) |
| `RealizedVol` | `K02P10C02R01` | `price` | DO01=rv, DO02=realized_vol, DO03=bipower_var, DO04=jump_component, DO05=relative_jump, DO06=log_rv, DO07=tripower_var, DO08=quadpower_var | 8 F64. tripower/quadpower = TQ — partially in tambear |
| `RangeVol` | `K02P10C02R02F01` | `price` | DO01=parkinson_vol, DO02=garman_klass_vol, DO03=rogers_satchell_vol, DO04=yang_zhang_vol | 4 F64. Needs OHLC subwindows + range estimators — task #138 |
| `VolDynamics` | `K02P10C03R01F01` | `price` | DO01=vol_trend, DO02=vol_of_vol, DO03=mean_reversion_speed, DO04=vol_autocorr | 4 F64. Rolling std + AR(1) |
| `SignaturePlot` | `K02P10C02R01F02` | `price` | DO01=rv_tick, DO02=rv_sparse, DO03=rv_ratio, DO04=convergence_rate | 4 F64. RV at multiple sampling frequencies |
| `TickVol` | `K02P10C02R03F01` | `price` | DO01=realized_var, DO02=bipower_var, DO03=tick_frequency_var, DO04=microstructure_noise | 4 F64 |
| `JumpDetection` | `K02P10C03R02F01` | `price` | DO01=jump_flag, DO02=jump_size, DO03=jump_ratio, DO04=bns_stat | 4 F64. BNS test RV/BV |
| `RollSpread` | `K02P10C03R03F01` | `price` | DO01=roll_spread, DO02=serial_cov, DO03=spread_volatility, DO04=spread_to_price_ratio | 4 F64. Roll (1984) spread |
| `VpinBvc` | `K02P10C04R01F01` | `price`, `size` | DO01-DO04 (VPIN metrics), E01-E05 | 4 F64 + 5 E cols (F32). BVC volume classification — task #146 |
| `VolRegime` | `K02P18C03R01F02` | `price` | DO01=n_vol_regimes, DO02=vol_ratio_range, DO03=high_vol_fraction, DO04=regime_switching_rate | 4 F64 |

---

## Phylum K02P11 — Continuous Time / SDE

| Struct | ID | Inputs | Outputs (all F64) | Notes |
|--------|-----|--------|---------|-------|
| `OuProcess` | `K02P11C01R01` | `price` | DO01=theta, DO02=mu, DO03=sigma, DO04=half_life, DO05=r_squared | 5 F64. OLS on price differences |
| `Sde` | `K02P11C02R01` | `price` | DO01=drift_mean, DO02=drift_slope, DO03=diffusion_mean, DO04=diffusion_slope, DO05=drift_diffusion_corr | 5 F64. Nadaraya-Watson — partial |
| `TransferAnalysis` | `K02P11C03R01` | `price`, `size` | DO01=gain, DO02=peak_gain, DO03=bandwidth, DO04=peak_impulse, DO05=impulse_decay, DO06=impulse_energy | 6 F64. FFT cross-spectrum |
| `Arx` | `K02P11C04R01` | `price`, `size` | DO01=ar1_coeff, DO02=exog_coeff, DO03=exog_gain, DO04=residual_var, DO05=r2_improvement | 5 F64 |
| `TickOu` | `K02P11C05R01` | `price` | DO01=tick_theta, DO02=tick_half_life, DO03=tick_sigma, DO04=mr_strength | 4 F64. OU on log-prices |

---

## Phylum K02P12 — Fractal/Scaling

| Struct | ID | Inputs | Outputs (all F64) | Notes |
|--------|-----|--------|---------|-------|
| `Dfa` | `K02P12C01R02` | `price` | DO01=alpha, DO02=alpha_se, DO03=r_squared, DO04=n_scales, DO05=intercept, DO06=alpha_z | 6 F64 |
| `HurstRs` | `K02P12C01R01` | `price` | DO01=H, DO02=H_se, DO03=r_squared, DO04=n_scales | 4 F64 |
| `Mfdfa` | `K02P12C02R01` | `price` | 10 F64: h(q) at 6 q values + width + confidence + n_scales + intercept | Multifractal — extends DFA |
| `WaveletLeaders` | `K02P12C02R02` | `price` | DO01=cp1, DO02=cp2, DO03=cp3, DO04=multifractal_index | 4 F64 |
| `SpectralEntropy` | `K02P12C03R01` | `price` | DO01=spectral_entropy, DO02=spectral_entropy_raw, DO03=spectral_concentration, DO04=peak_freq_ratio, DO05=n_fft_bins | 5 F64 |
| `TickScaling` | `K02P12C03R01` | `price`, `K01P01.DI01DO03` | DO01=scaling_exponent, DO02=scaling_r2, DO03=xmin, DO04=burstiness | 4 F64. Inter-trade times |
| `ScaleFreeness` | `K02P12C03R02` | `price` | DO01=b_value, DO02=b_std, DO03=scaling_range, DO04=deviation_from_gr | 4 F64. GR power law |

---

## Phylum K02P13 — Entropy/Complexity

| Struct | ID | Inputs | Outputs (all F64) | Notes |
|--------|-----|--------|---------|-------|
| `ShannonEntropy` | `K02P13C01R01` | `price` | DO01=shannon_entropy, DO02=entropy_rate, DO03=max_entropy, DO04=normalized_entropy | 4 F64 |
| `SampleEntropy` | `K02P13C02R01` | `price` | DO01=sample_entropy, DO02=sample_entropy_std (NaN), DO03=optimal_m (=2), DO04=optimal_r (=0.2σ) | 4 F64. r = 0.2×std |
| `PermutationEntropy` | `K02P13C02R02` | `price` | 7 F64: entropy, statistical_complexity, entropy_rate, JS_divergence, normalized_complexity, L (order), n_patterns | Needs B17 |
| `LzComplexity` | `K02P13C02R03` | `price` | DO01=lz_complexity, DO02=normalized_lz, DO03=compression_ratio | 3 F64 |
| `MutualInfo` | `K02P13C03R01` | `price`, `size` | DO01=mutual_info, DO02=mi_corrected, DO03=mi_nonlinear_excess, DO04=mi_normalized | 4 F64. MI(returns, vol_change) |
| `TransferEntropyBin` | `K02P13C03R02` | `price`, `size` | DO01=te_xy, DO02=te_yx, DO03=net_te, DO04=significance_ratio | 4 F64. Needs B13 |
| `TickComplexity` | `K02P13C04R01` | `price`, `K01P01.DI01DO03` | DO01=inter_arrival_entropy, DO02=size_entropy, DO03=joint_entropy, DO04=normalized_complexity | 4 F64 |
| `FisherInfo` | `K02P13C05R01` | `price` | DO01=fisher_info, DO02=fisher_distance, DO03=gradient_norm | 3 F64 |

---

## Phylum K02P14 — Chaos/Nonlinear

| Struct | ID | Inputs | Outputs (all F64) | Notes |
|--------|-----|--------|---------|-------|
| `Embedding` | `K02P14C01R01` | `price` | DO01=optimal_delay, DO02=embedding_dim, DO03=fnn_fraction | 3 F64. AMI + FNN |
| `Lyapunov` | `K02P14C01R02` | `price` | DO01=lambda_max, DO02=confidence, DO03=divergence_rate, DO04=prediction_horizon | 4 F64. Rosenstein 1993 |
| `CorrelationDim` | `K02P14C01R03` | `price` | DO01=d2, DO02=d2_confidence, DO03=scaling_range | 3 F64. Needs B |
| `Rqa` | `K02P14C02R01` | `price` | DO01=recurrence_rate, DO02=determinism, DO03=laminarity, DO04=trapping_time, DO05=diagonal_entropy, DO06=divergence | 6 F64 |
| `Poincare` | `K02P14C03R01` | `price` | DO01=sd1, DO02=sd2, DO03=sd_ratio, DO04=return_correlation | 4 F64 |
| `PithAttractor` | `K02P14C04R01` | `price` | DO01=basin_extent, DO02=scr, DO03=reconstruction_quality, DO04=local_lyapunov | 4 F64 |
| `TickAttractor` | `K02P14C05R01` | `price` | DO01=phase_asymmetry, DO02=tick_persistence, DO03=reversal_rate, DO04=phase_spread | 4 F64 |

---

## Phylum K02P15 — Manifold/Topology

| Struct | ID | Inputs | Outputs (all F64) | Notes |
|--------|-----|--------|---------|-------|
| `Pca` | `K02P15C1R1` | `price` | DO01=explained_var_ratio_1, DO02=explained_var_ratio_2, DO03=effective_dim, DO04=spectral_entropy | 4 F64 |
| `Ica` | `K02P15C1R2` | `price` | DO01=max_negentropy, DO02=mean_negentropy, DO03=kurtosis_range | 3 F64. FastICA — task #163 |
| `DiffGeometry` | `K02P15C3R1` | `price` | DO01=mean_curvature, DO02=curvature_std, DO03=torsion_proxy | 3 F64 |
| `SpectralEmbedding` | `K02P15C2R1` | `price` | DO01=fiedler_value, DO02=spectral_gap, DO03=cheeger_proxy, DO04=effective_resistance | 4 F64 |
| `PersistentHomology` | `K02P15C4R1` | `price` | DO01=n_components_50, DO02=max_persistence, DO03=mean_persistence, DO04=entropy | 4 F64 |
| `Grassmannian` | `K02P15C5R1` | `price` | DO01=max_principal_angle, DO02=mean_principal_angle, DO03=chordal_distance | 3 F64 |
| `TickGeometry` | `K02P15C6R1` | `price` | DO01=hull_area, DO02=angular_entropy, DO03=radial_kurtosis, DO04=aspect_ratio | 4 F64 |
| `Rmt` | `K02P15C7R1` | `price` | DO01=n_signal_eigenvalues, DO02=tracy_widom_stat, DO03=mp_ratio, DO04=spectral_rigidity | 4 F64 |

---

## Phylum K02P16 — Distance Metrics

| Struct | ID | Inputs | Outputs (all F64) | Notes |
|--------|-----|--------|---------|-------|
| `DistDistance` | `K02P16C02R01` | `price` | DO01=wasserstein, DO02=energy_distance, DO03=ks_stat | 3 F64. First vs second half of bin |
| `Dtw` | `K02P16C01R01` | `price` | DO01=dtw_distance, DO02=dtw_normalized, DO03=dtw_ratio | 3 F64. First vs second half — task #156 |
| `EditDistance` | `K02P16C03R01` | `price` | DO01=edit_distance, DO02=edit_normalized, DO03=symbol_entropy_diff | 3 F64 — task #157 |
| `TickAlignment` | `K02P16C04R01` | `K01P01.DI01DO03` | DO01=arrival_regularity, DO02=clustering_index, DO03=gap_ratio, DO04=uniformity_score | 4 F64 |

---

## Phylum K02P17 — Causality

| Struct | ID | Inputs | Outputs (all F64) | Notes |
|--------|-----|--------|---------|-------|
| `Coherence` | `K02P17C01R02` | `price`, `size` | DO01=mean_coherence, DO02=max_coherence, DO03=peak_coherence_freq | 3 F64. Cross-spectral coherence |
| `Granger` | `K02P17C01R01` | `price`, `size` | DO01=f_stat_xy, DO02=p_value_xy, DO03=f_stat_yx, DO04=p_value_yx | 4 F64. Bidirectional Granger |
| `Ccm` | `K02P17C02R01` | `price`, `size` | DO01=ccm_corr_xy, DO02=ccm_corr_yx, DO03=ccm_converges_xy | 3 F64. Needs B |
| `TickCausality` | `K02P17C03R01` | `price`, `size` | DO01=lead_lag_corr, DO02=lead_lag_offset, DO03=coupling_strength, DO04=impulse_ratio | 4 F64 |

---

## Phylum K02P18 — Regime Detection

| Struct | ID | Inputs | Outputs (all F64) | Notes |
|--------|-----|--------|---------|-------|
| `ClassicalCp` | `K02P18C01R01F01` | `price` | DO01=n_changepoints, DO02=mean_segment_length, DO03=max_cusum_stat, DO04=break_location | 4 F64. CUSUM+binseg — task #159 |
| `Pelt` | `K02P18C01R02F01` | `price` | DO01=n_changepoints, DO02=mean_segment_length, DO03=max_cost_reduction, DO04=penalty_ratio | 4 F64 — task #160 |
| `Bocpd` | `K02P18C01R03F01` | `price` | DO01=max_run_length, DO02=cp_probability, DO03=mean_run_length, DO04=hazard | 4 F64 — task #161 |
| `PhaseTransition` | `K02P18C03R01F01` | `price` | DO01=order_parameter, DO02=susceptibility, DO03=binder_cumulant, DO04=critical_exponent | 4 F64 |
| `VolRegime` | `K02P18C03R01F02` | `price` | DO01=n_vol_regimes, DO02=vol_ratio_range, DO03=high_vol_fraction, DO04=regime_switching_rate | 4 F64 |
| `StructBreak` | `K02P18C03R02F01` | `price` | DO01=max_f_stat, DO02=break_location, DO03=pre_post_var_ratio, DO04=pre_post_mean_shift | 4 F64 |
| `TickSpace` | `K02P18C03R03F01` | `price` | DO01=tick_entropy, DO02=mode_concentration, DO03=tick_clustering, DO04=regime_persistence | 4 F64 |
| `Hmm` | `K02P18C02R01F01` | `price` | DO01=state_persistence, DO02=state_separation, DO03=entropy_rate, DO04=switching_rate, DO05=viterbi_agreement | 5 F64 — task Baum-Welch |
| `Hawkes` | `K02P18C02R02F01` | `price` | DO01=mu, DO02=alpha, DO03=beta, DO04=branching_ratio | 4 F64 — task #155 |

---

## Phylum K02P19 — Cross-Scale

| Struct | ID | Inputs | Outputs | Notes |
|--------|-----|--------|---------|-------|
| `EnergyBands` | `K02P19C3R1` | `price` | DO01=total_energy, DO02=low_freq_ratio, DO03=high_freq_ratio, DO04=mid_freq_ratio, DO05=frequency_balance | 5 F64 |
| `Periodicity` | `K02P19C4R1` | `price` | DO01=acf_peak, DO02=acf_peak_lag, DO03=periodicity_strength, DO04=spectral_peak_ratio | 4 F64 |
| `Variability` | `K02P19C5R1` | `price` | DO01=rolling_var_cv, DO02=rolling_mean_cv, DO03=range_variation, DO04=stability_index | 4 F64 |
| `Hilbert` | `K02P19C2R1` | `price` | DO01=inst_amplitude_mean, DO02=inst_amplitude_var, DO03=inst_freq_mean, DO04=inst_freq_var | 4 F64 |
| `Seismic` | `K02P19C1R1` | `price` | DO01=gr_b_value, DO02=omori_p, DO03=bath_ratio, DO04=n_extreme | 4 F64 |

---

## Phylum K02P20 — Graph Theory (GAP leaves)

| Struct | ID | Inputs | Outputs | Notes |
|--------|-----|--------|---------|-------|
| `Nvg` | `K02P20C01R01` | `price` | DO01=degree_exponent, DO02=mean_degree, DO03=degree_entropy, DO04=clustering_coefficient, DO05=assortativity | 5 F64 — task #145 |
| `Hvg` | `K02P20C01R02` | `price` | DO01=degree_exponent, DO02=mean_degree, DO03=degree_entropy, DO04=irreversibility | 4 F64 — task #145 |

---

## Remaining Phyla (Specialist)

| Struct | ID | Inputs | Outputs | Notes |
|--------|-----|--------|---------|-------|
| `Shape` | `K02P08C01R01` | `price` | DO01-DO21 (21 F64): monotonicity×6, extrema×6, gradient×6, curvature×3 | Full list in leaf doc |
| `Logsig` | `K02P21C01R01` | `price`, `size` | DO01=levy_area_price, DO02=levy_area_vol, DO03=levy_area_cross, DO04=total_increment_price, DO05=total_increment_vol, DO06=logsig_l2_norm, DO07=depth2_energy_frac | 7 F64 |
| `TailField` | `K02P22C01R01` | `price`, `size` | DO01=tail_concentration_entropy, DO02=peak_tail_quintile, DO03=tail_chi2_stat, DO04=tail_cramer_v, DO05=tail_ks_stat, DO06=joint_mi, DO07=n_extreme_events | 7 F64 |
| `TaylorFold` | `K02P23C01` | `price` | DO01=correction_ratio_01, DO02=correction_ratio_12, DO03=correction_ratio_23, DO04=divergence_onset, DO05=regime_flag, DO06=fit_residual_0, DO07=max_correction_ratio, DO08=regime_strength | 8 F64 |
| `Harmonic` | `K02P24C01` | `price` | varies (r-statistic on SVD spacings) | Hankel SVD + consecutive ratio |
| `TickCompression` | `K02P25C01` | `price`, `size` | DO01=real_effective_rank, DO02=shuffled_effective_rank, DO03=compression_ratio, DO04=n_active_features | 4 F64 |
| `VpinBvc` | `K02P10C04R01F01` | `price`, `size` | DO01-DO04 + E01-E05 | VPIN — task #146 |

---

## K02P03 — Wavelet (multi-variant)

| Struct | ID | Inputs | Outputs | Notes |
|--------|-----|--------|---------|-------|
| `HaarWavelet { m: usize, strategy: RegStrategy, id: &'static str }` | varies | `price` | 13 F64 (DO01-DO13) | 15 variants: M∈{8,16,32,64,128} × strategy∈{interp,bin_mean,subsample} |
| `CwtWavelet` | `K02P03C01` | `price` | varies | Morlet CWT — task #144 |
| `Scattering` | `K02P03C03R01` | `price` | DO01=first_order_energy, DO02=second_order_energy, DO03=scattering_ratio, DO04=invariance_measure | 4 F64 |
| `StftLeaf` | `K02P03C04R01` | `price` | DO01=spectral_centroid_var, DO02=spectral_flux, DO03=onset_strength, DO04=chromagram_entropy | 4 F64 |
| `WignerVille` | `K02P03C05R01` | `price` | DO01=time_freq_concentration, DO02=instantaneous_freq_var, DO03=cross_term_energy, DO04=marginal_entropy | 4 F64 — task #164 |

---

## K02P05 — Filtering/Smoothing

| Struct | ID | Inputs | Outputs | Notes |
|--------|-----|--------|---------|-------|
| `Smoothers` | `K02P05C01R01` | `price` | DO01=ma_residual_energy, DO02=ewma_residual_energy, DO03=ewma_halflife, DO04=smoothing_snr | 4 F64 |
| `Savgol` | `K02P05C02R01` | `price` | DO01=savgol_residual_energy, DO02=derivative_mean, DO03=derivative_var, DO04=curvature_mean | 4 F64 — task #165 |
| `FirBandpass` | `K02P05C03R01` | `price` | DO01=low_band_energy, DO02=mid_band_energy, DO03=high_band_energy | 3 F64 |
| `Wiener` | `K02P05C03R02` | `price` | DO01=noise_estimate, DO02=signal_estimate, DO03=wiener_snr, DO04=noise_fraction | 4 F64 |
| `Stl` | `K02P05C05R01` | `price` | DO01=trend_strength, DO02=seasonal_strength, DO03=residual_acf1, DO04=trend_slope | 4 F64 |
| `Kalman` | `K02P05C04R01` | `price` | DO01=innovation_var, DO02=gain_mean, DO03=smoothing_improvement | 3 F64 — task #101 |

---

## K03/K05 — Cross-Cadence

| Struct | ID | Inputs | Outputs | Notes |
|--------|-----|--------|---------|-------|
| `TransferEntropy` | `K03P01C01` | reads K02 spectral entropy outputs via ctx | DO01=te_fast_to_slow, DO02=te_slow_to_fast | 2 F64. K03 leaf — ctx provides upstream columns |
| `Viscosity` | `K03P02C01` | reads Taylor max_correction_ratio at 5 cadences | 6 F64 | Cross-cadence fold propagation |
| `ScalingTriple` | `K03P03C01` | reads DFA + R/S outputs at all cadences | 10 F64 | Cross-leaf consistency |
| `CadenceGradient` | `K03P04C01` | reads K02P01C03 + K02P01C01 nanmean across cadences | varies | Log-linear regression |
| `CoherenceMatrix` | `K03P04C02` | reads K02P01C03.DI01DO10 (realized_var) at 31 cadences | varies | 31×31 correlation matrix + eigendecomp |
| `Coboundary` | `K05P01C01` | reads DFA + R/S scaling_regime E columns | varies | Coboundary classification |

---

## Shim Template

Every shim follows this pattern:

```rust
// In fintek-tambear-prelim/src/leaves/leaf_name.rs
use fintek_trunk_rs::leaf::{ColumnData, ExecutionContext, Leaf, LeafOutput};
use mkt::types::Dtype;

pub struct StructName; // zero-field; or { m: usize, id: &'static str } if parameterized

impl Leaf for StructName {
    fn id(&self) -> &str { "KxxPxxCxx" }

    fn outputs(&self) -> &[(&str, Dtype)] {
        &[
            ("DI01DO01", Dtype::Float64),
            // ... exact same list as trunk-rs leaf
        ]
    }

    fn execute(&self, ctx: &dyn ExecutionContext, cadence_id: u8) -> Result<LeafOutput, Box<dyn std::error::Error>> {
        // Pull inputs via ctx
        let price = ctx.column_f64("price").ok_or("missing price")?;
        let (starts, ends) = ctx.bin_boundaries(cadence_id).ok_or("missing boundaries")?;
        let n_bins = starts.len();

        // Call tambear
        let results: Vec<f64> = (0..n_bins).map(|i| {
            let bin = &price[starts[i] as usize..ends[i] as usize];
            tambear::some_module::some_function(bin)
        }).collect();

        // Pack output
        let mut out = LeafOutput::new();
        out.insert("DI01DO01".into(), ColumnData::F64(results));
        Ok(out)
    }
}
```

---

*Updated as reconnaissance progresses. Adversarial: pull from this file for exact struct names and output specs. Flag any discrepancy between this file and the trunk-rs source.*
