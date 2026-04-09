# Fintek → Tambear Math Catalog

**Author**: math-researcher  
**Date**: 2026-04-08  
**Purpose**: Map every fintek leaf in `R:/fintek/crates/trunk-rs/src/leaves/` to its corresponding tambear function. For each leaf: algorithm, paper, tambear mapping, tolerance, gaps.

**SCOPE**: 100% `R:/fintek/crates/trunk-rs/src/leaves/` — the 126 active Rust leaves. The Python `R:/fintek/trunk/leaves/` is NOT the rescue target (fintek has moved past it). Python trunk is reference only for broader expansion ideas, not for this rescue mission.

**Campsite**: tambear-hardening/20260408214502-fintek-rescue  
**Task**: #134 (catalog), #135 (bridge crate)

**Status legend**:
- **DIRECT**: Tambear has the exact function; fintek can call it with identical semantics (tolerance ~1e-10).
- **ADAPT**: Tambear has the primitive; needs thin adapter (feature extraction, windowing, shape-change).
- **COMPOSE**: Tambear has building blocks; requires composition (e.g., Burg AR = Levinson-Durbin, which we have via PACF).
- **GAP**: No tambear equivalent — needs new implementation.

**Scope**: 126 leaves in `leaves/` directory. Grouped by mathematical family rather than alphabetically.

---

## Family 1: Distribution / Moments (7 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `distribution.rs` | Mean, std, skew, kurtosis, quantiles, realized_var, bipower_var, jump_test | `descriptive::moments_ungrouped` + `volatility::realized_variance` + `volatility::bipower_variation` + `volatility::jump_test_bns` | DIRECT | All components exist. Quantiles: `descriptive::quantile`. |
| `shannon_entropy.rs` | Shannon entropy of binned returns | `information_theory::shannon_entropy_from_counts` | DIRECT | Bin returns → counts → entropy. |
| `spectral_entropy.rs` | FFT → normalized PSD → Shannon entropy | `signal_processing::rfft` + `information_theory::shannon_entropy` | COMPOSE | 5 lines of composition. |
| `normality.rs` | Jarque-Bera + Shapiro-Wilk | `nonparametric::jarque_bera` + `nonparametric::shapiro_wilk` | DIRECT | Both exist. |
| `heavy_tail.rs` | Hill estimator on tail | GAP | GAP | Need Hill estimator. Spec: sort |r|, top-k ratio of ln(r_i/r_k). |
| `fisher_info.rs` | Fisher information of return distribution | GAP | GAP | Need histogram-based Fisher info. Not a standard function. |
| `tail_field.rs` | Tail concentration by quintile | `descriptive::quantile` + chi-square | COMPOSE | Quintile binning + `hypothesis::chi2_goodness_of_fit`. |

---

## Family 2: Returns & Transforms (11 leaves)

These are almost entirely pointwise `map` operations — ideal for Kingdom A (scatter_phi).

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `returns.rs` | log_return = ln(p_t / p_{t-1}) | `compute_engine::map_phi2` | DIRECT | Pure map. |
| `log_transform.rs` | ln(x) | `compute_engine::map_phi` | DIRECT | Pointwise. |
| `sqrt_transform.rs` | √x | `compute_engine::map_phi` | DIRECT | Pointwise. |
| `reciprocal.rs` | 1/x | `compute_engine::map_phi` | DIRECT | Pointwise. |
| `notional.rs` | price × size | `compute_engine::map_phi2` | DIRECT | Pointwise product. |
| `delta_value.rs` | x_t - x_{t-n} | `compute_engine::map_phi2` (with lag gather) | ADAPT | Need lag-gather pattern. |
| `delta_log.rs` | ln(x_t) - ln(x_{t-n}) | Same as delta_value on ln(x) | ADAPT | Lag-gather. |
| `delta_percent.rs` | (x_t - x_{t-n})/x_{t-n} | Same pattern | ADAPT | Lag-gather. |
| `delta_direction.rs` | sign(x_t - x_{t-n}) | Same pattern + sign | ADAPT | Lag-gather + sign. |
| `elapsed.rs` | (ts % day_ns) / minute_ns | Pure map | DIRECT | Pointwise modular arithmetic. |
| `cyclical.rs` | sin/cos on 24h circle | Pure map | DIRECT | Pointwise. |

**Key insight**: All 11 are embarrassingly parallel pointwise operations. Can be JIT-compiled via `scatter_jit::ScatterJit` directly from a small expression DSL.

---

## Family 3: Bin Aggregates (5 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `ohlcv.rs` | open/high/low/close/volume/vwap per bin | `descriptive::GroupedMomentStats` + first/last (needs addition) | COMPOSE | GroupedMomentStats gives mean/var/min/max. Need first/last accumulators. |
| `counts.rs` | tick_count, unique_prices, upticks, downticks | Count-based reductions | COMPOSE | Simple scatter_phi with boolean conditions. |
| `validity.rs` | Data quality metrics | Composition | COMPOSE | Count NaN, range checks. |
| `variability.rs` | Rolling var/mean CV, stability index | Windowed moments | COMPOSE | Need rolling/windowed grouping. |
| `returns.rs` (bin-level) | open_return, close_return, high_low_range | From OHLCV | COMPOSE | Arithmetic on OHLCV. |

**Gap to add**: `FirstOp` and `LastOp` as reduce operators — not yet in tambear. ~15 lines.

---

## Family 4: Time Series / ARMA (6 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `autocorrelation.rs` | ACF + PACF per bin with 16 features | `time_series::acf` + `time_series::pacf` | DIRECT | Levinson-Durbin in pacf. Feature extraction on top. |
| `ar_model.rs` | Yule-Walker AR fit, BIC order selection | `time_series::ar_fit` + BIC wrapper | ADAPT | `ar_fit` exists; BIC selection is a ~10-line loop. |
| `ar_burg.rs` | Burg's method AR + PSD evaluation | GAP | GAP | Have Yule-Walker AR but NOT Burg. Need `ar_burg_fit`. ~50 lines: forward/backward prediction errors + Levinson update. |
| `arma.rs` | ARMA(p,q) via Yule-Walker AR + residual ACF for MA | `time_series::ar_fit` + residual ACF | COMPOSE | Residual MA estimation is a simple loop on residual ACF. |
| `arima.rs` | Auto-select d via differencing, then AR | `time_series::difference` + `time_series::ar_fit` | COMPOSE | Exists piecewise. Task #85 has full ARIMA. |
| `arx.rs` | AR + exogenous input (volume → returns) | OLS regression with lagged features | COMPOSE | Build lag matrix + OLS. |

**Paper refs**:
- Burg 1968, "A new analysis technique for time series data" — for ar_burg
- Box & Jenkins 1970 — for ARIMA framework

---

## Family 5: Stationarity / Structural Tests (5 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `stationarity.rs` | ADF + KPSS, classify stationary/trend/difference | `time_series::adf_test` + `time_series::kpss_test` + logic | DIRECT | Both exist. Classification is a 5-line decision tree. |
| `dependence.rs` | Ljung-Box Q on returns | `time_series::ljung_box` | DIRECT | Already implemented. |
| `struct_break.rs` | Chow-like max F-stat scan for break point | OLS at each candidate + F-stat | COMPOSE | Brute force scan. Each candidate is an OLS fit. |
| `classical_cp.rs` | CUSUM + binary segmentation | GAP | GAP | Have nothing for CUSUM. ~40 lines. |
| `pelt.rs` | PELT changepoint detection with BIC | GAP | GAP | Killick et al. 2012. Dynamic programming with pruning. ~60 lines. |
| `bocpd.rs` | Bayesian Online Changepoint Detection | GAP | GAP | Adams & MacKay 2007. Run-length message passing. ~80 lines. |

---

## Family 6: Spectral Analysis (10 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `fft_spectral.rs` | FFT → PSD → 15 features (5 resolution variants) | `signal_processing::rfft` + feature extraction | DIRECT | Core FFT exists. Feature extraction is spectral centroid/bandwidth/rolloff/entropy/flatness. |
| `welch.rs` | Welch's method PSD | `signal_processing::welch` | DIRECT | Exists! |
| `multitaper.rs` | Multitaper PSD with sine tapers | `signal_processing::multitaper_psd` | DIRECT | Exists! |
| `lombscargle.rs` | Lomb-Scargle periodogram (irregular sampling) | `signal_processing::lomb_scargle` | DIRECT | Exists! |
| `coherence.rs` | Cross-spectral coherence between price/volume | Welch on each + cross PSD | COMPOSE | Need cross-PSD: FFT(x)·conj(FFT(y)). ~20 lines. |
| `coherence_matrix.rs` | 31×31 Pearson on realized_var across cadences | Multi-column Pearson | COMPOSE | Straight correlation matrix. |
| `cepstrum.rs` | Real cepstrum = IFFT(log|FFT|²) | `signal_processing::real_cepstrum` | DIRECT | Exists! |
| `hilbert.rs` | Hilbert transform → instantaneous amp/phase | `signal_processing::hilbert` | DIRECT | Exists! |
| `stft_leaf.rs` | STFT spectrogram features | `signal_processing::stft` | DIRECT | Exists! |
| `wigner_ville.rs` | Discrete Wigner-Ville distribution | GAP | GAP | W(t,f) = 2 Re Σ_τ x(t+τ)x*(t-τ)exp(-j4πfτ). ~40 lines using existing FFT. |

---

## Family 7: Wavelets & Time-Frequency (5 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `haar_wavelet.rs` | Haar DWT with M-regularization | `signal_processing::haar_dwt` / `haar_wavedec` | DIRECT | Have both single-level and multi-level. |
| `cwt_wavelet.rs` | Morlet CWT via FFT | GAP | GAP | Have FFT; need Morlet kernel + frequency scaling. ~40 lines. |
| `scattering.rs` | Cascaded wavelet scattering (Mallat) | GAP | GAP | Needs CWT first. ~80 lines on top of CWT. |
| `wavelet_leaders.rs` | Wavelet leader multifractal (Holder regularity cumulants) | GAP | GAP | Extension of DWT. Cumulant estimation from leaders. ~60 lines. |
| `emd.rs` | Empirical Mode Decomposition (sifting) | GAP | GAP | Huang et al. 1998. Iterative extrema-based sifting. ~80 lines. Genuinely hard (extrema detection is sequential). |

---

## Family 8: Correlation & Dependence (11 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `cross_correlation.rs` | CCF between returns/volume at lags ±5 | `signal_processing::cross_correlate` | DIRECT | Exists. Feature extraction on top. |
| `mutual_info.rs` | MI with Miller-Madow correction | `information_theory::mutual_information` + bias correction | ADAPT | Have MI from contingency. Miller-Madow = +½(R-1)(C-1)/n correction. 5-line addition. |
| `transfer_entropy.rs` | Schreiber 2000 TE between fast/slow spectral entropy | GAP | GAP | TE(X→Y) = H(Y_{t+1}|Y_t) - H(Y_{t+1}|Y_t,X_t). Computable from conditional entropies via binning. ~40 lines. |
| `transfer_entropy_bin.rs` | Same but on returns/volume within a bin | Same as transfer_entropy | GAP | Shares implementation. |
| `granger.rs` | Granger causality F-tests | Restricted vs unrestricted OLS + F-test | COMPOSE | Two OLS fits + F-statistic. ~40 lines. |
| `ccm.rs` | Convergent Cross Mapping (Sugihara 2012) | GAP | GAP | Takens embedding + NN-based manifold prediction. ~60 lines. |
| `tick_causality.rs` | Cross-correlation at lags ±5 | `signal_processing::cross_correlate` | DIRECT | Same as cross_correlation but tick-level. |
| `transfer_analysis.rs` | H(f) = S_yx / S_xx, impulse via IFFT | FFT + complex division | COMPOSE | ~30 lines. |
| `msplit_temporal_coherence.rs` | Split-half correlation, lag-1 autocorrelation | Pearson on slices | COMPOSE | Trivial composition. |
| `dist_distance.rs` | Wasserstein, KS between halves | `tda::wasserstein_distance` + `nonparametric::ks_test_two_sample` | ADAPT | Have KS. Wasserstein is currently for persistence pairs — need 1D EMD version. ~15 lines. |
| `edit_distance.rs` | Levenshtein on symbolized returns | GAP | GAP | DP on (n+1)×(m+1) matrix. ~30 lines. |

---

## Family 9: Volatility (11 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `garch.rs` | GARCH(1,1) grid search + local refinement | `volatility::garch11_fit` | DIRECT | Exists! Tambear version may use different optimizer. |
| `stochvol.rs` | AR(1) on log(r²) | `time_series::ar_fit` on ln(r²+eps) | COMPOSE | 5-line wrapper. |
| `vol_dynamics.rs` | Rolling vol + trend/vol-of-vol/MR speed | Rolling std + AR(1) | COMPOSE | Compose rolling window + ar_fit. |
| `vol_regime.rs` | Short/long window variance ratio | Rolling moments + classification | COMPOSE | Trivial. |
| `realized_vol.rs` | RV + BV + TQ (multipower) | `volatility::realized_variance` + `bipower_variation` + TQ | DIRECT+GAP | RV and BV exist. TQ (tripower quarticity) is missing. ~15 lines. |
| `range_vol.rs` | Parkinson/Garman-Klass/Rogers-Satchell/Yang-Zhang from OHLC | GAP | GAP | All closed-form from OHLC. ~25 lines total for all 4. |
| `roll_spread.rs` | Roll 1984 effective spread from serial covariance | `volatility::roll_spread` | DIRECT | Exists! |
| `vpin_bvc.rs` | Easley-Lopez de Prado-O'Hara VPIN via BVC | GAP | GAP | Needs volume bucketing + normal CDF classification. ~50 lines. |
| `signature_plot.rs` | RV at multiple sampling frequencies | Loop over frequencies + realized_variance | COMPOSE | ~15 lines. |
| `jump_detection.rs` | BNS jump test RV vs BV | `volatility::jump_test_bns` | DIRECT | Exists! |
| `tick_vol.rs` | Realized var, bipower, frequency-normalized, microstructure noise | RV + BV + sub-sampling | COMPOSE | All primitives exist. |

---

## Family 10: Nonlinear Dynamics / Chaos (11 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `sample_entropy.rs` | Richman & Moorman 2000 SampEn | `complexity::sample_entropy` | DIRECT | Exists! |
| `permutation_entropy.rs` | Bandt-Pompe + Lopez-Ruiz statistical complexity | `complexity::permutation_entropy` + composition | DIRECT+COMPOSE | PE exists; statistical complexity is JS divergence from uniform. 10 lines. |
| `lz_complexity.rs` | Lempel-Ziv 76 parsing | GAP | GAP | Sequential parsing. ~30 lines. |
| `hurst_rs.rs` | R/S Hurst | `complexity::hurst_rs` | DIRECT | Exists! |
| `dfa.rs` | Detrended Fluctuation Analysis | `complexity::dfa` | DIRECT | Exists! |
| `mfdfa.rs` | Multifractal DFA (multiple q moments) | Extension of DFA | COMPOSE | Loop over q values on existing DFA. ~25 lines. |
| `correlation_dim.rs` | Grassberger-Procaccia | `complexity::correlation_dimension` | DIRECT | Exists! |
| `lyapunov.rs` | Rosenstein 1993 largest Lyapunov | `complexity::largest_lyapunov` | DIRECT | Exists! |
| `embedding.rs` | Optimal tau (AMI) + dim (FNN) | AMI + false nearest neighbor | COMPOSE | AMI needs histogram-based MI (have via contingency). FNN is ~30 lines. |
| `rqa.rs` | Recurrence quantification analysis | GAP | GAP | Recurrence matrix (from distance matrix) + diagonal/vertical line statistics. ~50 lines. |
| `poincare.rs` | SD1/SD2 from Poincaré plot | Pairwise returns + eigendecomposition of 2×2 | COMPOSE | Trivial. |

**Key shared intermediate**: Phase-space distance matrix. If multiple leaves use the same (m, tau), distance matrix should be shared via `IntermediateTag::DistanceMatrix`.

---

## Family 11: State-Space / Filtering (5 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `kalman.rs` | Kalman filter/smoother | GAP (Task #101) | GAP | Need Kalman via Särkkä 5-tuple matrix prefix scan. |
| `statespace.rs` | Local level model via Kalman with SNR grid search | Kalman + grid | GAP | Depends on Kalman. |
| `hmm.rs` | 2-state HMM Baum-Welch + Viterbi | GAP | GAP | Forward-backward is a matrix prefix scan (like Kalman). Baum-Welch is EM. Viterbi is max-plus prefix scan. ~100 lines. |
| `wiener.rs` | Wiener filter SNR estimation | FFT-based | COMPOSE | Spectral noise estimation from PSD tail. |
| `smoothers.rs` | MA + EWMA with optimal half-life | Composition | COMPOSE | MA is `accumulate(Windowed, Mean)`. EWMA is a prefix scan with decay. |

---

## Family 12: Point Processes (3 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `hawkes.rs` | Self-exciting Hawkes with exponential kernel | GAP | GAP | Ogata 1981. Log-likelihood requires recursive kernel evaluation (sequential). Grid search + CD refinement. ~80 lines. |
| `ou_process.rs` | OU fit via OLS on price differences | `time_series::ar_fit` on differences | COMPOSE | OU discretized is AR(1). Parameter transformation is closed-form. ~15 lines. |
| `tick_ou.rs` | OU on log prices | Same as ou_process on ln(p) | COMPOSE | Same infrastructure. |

---

## Family 13: Dimension Reduction / Multivariate (8 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `pca.rs` | PCA via delay embedding | `dim_reduction::pca` + `copa::copa_pca` | DIRECT | Have PCA. Delay embedding is a reshape. |
| `ica.rs` | FastICA with negentropy | GAP | GAP | Hyvärinen & Oja 1997. Fixed-point iteration with g(u)=u³ or tanh. ~80 lines. |
| `ssa.rs` | Singular Spectrum Analysis | SVD on trajectory matrix | COMPOSE | Have SVD. Trajectory matrix construction is a reshape. ~25 lines. |
| `rmt.rs` | Marchenko-Pastur null comparison | `copa::covariance` + eigenvalues + MP edges | COMPOSE | MP bulk edges: (1 ± √(p/n))² × σ². Closed-form. ~25 lines. |
| `grassmannian.rs` | Principal angles between subspaces | SVD of Q1'Q2 | COMPOSE | Have QR. Principal angles = arccos(singular values). ~20 lines. |
| `spectral_embedding.rs` | Graph Laplacian eigendecomposition | Sym_eigen on Laplacian | COMPOSE | Have sym_eigen. Build Laplacian from distance matrix. ~25 lines. |
| `diff_geometry.rs` | Menger curvature along trajectory | Pairwise triangle circumradius | COMPOSE | Trivial geometry. ~25 lines. |
| `tick_compression.rs` | Effective rank via SVD | `dim_reduction::pca` eigenvalues | COMPOSE | Effective rank = exp(Shannon entropy of normalized eigenvalues). 10 lines. |

---

## Family 14: Topological / Geometric (4 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `persistent_homology.rs` | H₀ via union-find on filtered edges | `tda::rips_h0` | DIRECT | Exists! |
| `nvg.rs` | Natural Visibility Graph | GAP | GAP | Lacasa 2008. Geometric visibility test between time points. ~40 lines + graph metrics. |
| `hvg.rs` | Horizontal Visibility Graph | GAP | GAP | Luque 2009. Simpler than NVG. ~25 lines. |
| `tick_geometry.rs` | Convex hull area + angular entropy | GAP | GAP | Need 2D convex hull (not in tambear). Graham scan = ~30 lines. |

---

## Family 15: Distribution Distances (3 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `dtw.rs` | Dynamic Time Warping | GAP | GAP | DP on (n+1)×(m+1) matrix. ~30 lines. Similar to edit_distance. |
| `dist_distance.rs` | Wasserstein-1 + energy distance + KS | `nonparametric::ks_test_two_sample` + W1 + energy | ADAPT | Need 1D W1 (= |F⁻¹_X - F⁻¹_Y| integral = mean abs diff of sorted samples). ~10 lines. |
| `edit_distance.rs` | Levenshtein on symbolized returns | GAP | GAP | Standard DP. ~30 lines. |

---

## Family 16: Extreme Events / Tails (4 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `jump_detection.rs` | BNS RV vs BV | `volatility::jump_test_bns` | DIRECT | Exists. |
| `heavy_tail.rs` | Hill estimator | GAP | GAP | Hill 1975: α̂ = k / Σ ln(X_{(n-i+1)}/X_{(n-k)}). ~20 lines. |
| `seismic.rs` | Gutenberg-Richter b-value, Omori p, Bath ratio | MLE on power law | COMPOSE | All standard MLEs. ~30 lines. |
| `phase_transition.rs` | SOC indicators (magnetization, susceptibility, Binder) | Moments + scaling | COMPOSE | Sign transforms + moments. ~25 lines. |

---

## Family 17: Bin-level Microstructure (9 leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `tick_alignment.rs` | Inter-tick time regularity | Moments on diffs | COMPOSE | Trivial. |
| `tick_attractor.rs` | Phase portrait features | 2D scatter moments | COMPOSE | Trivial. |
| `tick_complexity.rs` | Entropy of inter-arrival + size | `information_theory::shannon_entropy` | DIRECT | Histogram + entropy. |
| `tick_space.rs` | Tick size entropy, modal concentration | Entropy + histogram stats | COMPOSE | Trivial. |
| `tick_scaling.rs` | Power-law fit to inter-trade times | Linear regression on log-log | COMPOSE | OLS on log-transformed. |
| `vpin_bvc.rs` | VPIN via BVC | GAP | GAP | See family 9. |
| `pith_attractor.rs` | Basin extent, SCR, local Lyapunov | Composition of complexity | COMPOSE | Uses existing Lyapunov. |
| `harmonic.rs` | Oganesyan-Huse r-statistic on SVD spacings | SVD + consecutive ratios | COMPOSE | Have SVD. r-stat is ~10 lines. |
| `shape.rs` | Monotonicity, extrema, gradient stats | Composition | COMPOSE | Pure statistical summaries. |

---

## Family 18: Cross-Leaf / Derived (5 leaves)

These read OTHER leaves' outputs — they're meta-computations at the pipeline level, not raw math leaves.

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `scaling_triple.rs` | Cross-leaf fractal consistency (DFA α vs R/S H) | Reads DFA + Hurst outputs | META | Post-processing on computed leaves. |
| `coboundary.rs` | DFA + R/S scaling regime agreement | Reads DFA + Hurst | META | Post-processing. |
| `cadence_gradient.rs` | Cross-cadence feature scaling | Reads all cadence outputs | META | Log-linear regression on outputs. |
| `viscosity.rs` | Cross-cadence fold propagation (Taylor) | Reads Taylor outputs | META | Gradient across cadences. |
| `taylor_fold.rs` | Polynomial fit correction ratios | Polynomial fits + ratios | COMPOSE | OLS polynomial fit at orders 0-3. |

---

## Family 19: Miscellaneous / Specialized (15+ leaves)

| Fintek leaf | Algorithm | Tambear mapping | Status | Notes |
|-------------|-----------|-----------------|--------|-------|
| `savgol.rs` | Savitzky-Golay smoothing + derivatives | GAP | GAP | Polynomial filter. ~40 lines. |
| `stl.rs` | Seasonal-Trend decomposition via LOESS (or moving avg proxy) | GAP | GAP | ~60 lines. |
| `fir_bandpass.rs` | FFT-based band energy partitioning | `signal_processing::rfft` + slicing | COMPOSE | ~10 lines. |
| `energy_bands.rs` | Total energy, low/high freq ratios | Same as FIR bandpass | COMPOSE | ~10 lines. |
| `periodicity.rs` | ACF peak + PSD peak | `time_series::acf` + FFT | COMPOSE | ~15 lines. |
| `sde.rs` | Non-parametric Nadaraya-Watson drift/diffusion | Kernel regression | COMPOSE | KDE exists. ~30 lines. |
| `logsig.rs` | Log-signature (iterated integrals / Levy area) | GAP | GAP | Signature methods are specialized. ~40 lines for levels 1-2. |
| `fisher_info.rs` | Fisher info from histogram | GAP | GAP | ~25 lines. |
| `cadence_gradient.rs` | (See Family 18) | META | META | |
| `ar_burg.rs` | (See Family 4) | GAP | GAP | |
| `distribution.rs` | (See Family 1) | DIRECT | DIRECT | |

---

## Summary Statistics

**Breakdown by status**:
- **DIRECT** (thin wrapper): ~38 leaves
- **COMPOSE** (2-5 tambear calls): ~42 leaves  
- **ADAPT** (add lag/window pattern): ~10 leaves
- **GAP** (new implementation): ~30 leaves
- **META** (reads other leaves): ~6 leaves

**Total**: ~126 leaves mapped.

**Tambear coverage by family**:
| Family | Direct+Compose | Gaps | Coverage |
|--------|----------------|------|----------|
| Distribution/Moments | 6 | 1 | 86% |
| Returns/Transforms | 11 | 0 | 100% |
| Bin aggregates | 5 | 0 | 100% (with FirstOp/LastOp) |
| Time series/ARMA | 5 | 1 (Burg) | 83% |
| Stationarity | 2 | 3 (CUSUM, PELT, BOCPD) | 40% |
| Spectral | 9 | 1 (Wigner-Ville) | 90% |
| Wavelets | 1 | 4 (CWT, scattering, leaders, EMD) | 20% |
| Correlation/dependence | 6 | 5 (TE, CCM, edit, DTW, MI correction) | 55% |
| Volatility | 7 | 4 (range vol, VPIN, TQ, stochvol pieces) | 64% |
| Nonlinear dynamics | 9 | 2 (LZ, RQA) | 82% |
| State-space | 1 | 4 (Kalman, HMM, statespace, Wiener) | 20% |
| Point processes | 2 | 1 (Hawkes) | 67% |
| Dimension reduction | 7 | 1 (ICA) | 88% |
| Topological | 1 | 3 (NVG, HVG, convex hull) | 25% |
| Distribution distances | 0 | 3 (DTW, edit, W1) | 0% |
| Extremes/tails | 2 | 2 (Hill, full seismic) | 50% |
| Microstructure | 7 | 2 (VPIN, Hawkes dependency) | 78% |
| Cross-leaf | 5 | 0 | 100% (meta) |
| Miscellaneous | 4 | 5 (SavGol, STL, LogSig, SDE, Burg) | 44% |

**Overall tambear coverage**: ~76% (96 of 126 leaves have existing tambear primitives).

---

## Top-20 Priority Gaps (for tambear implementation)

Ranked by: (1) number of leaves unlocked, (2) implementation complexity, (3) missing from the universal taxonomy too.

| # | Missing function | Lines | Unlocks leaves | Priority |
|---|-----------------|-------|----------------|----------|
| 1 | **FirstOp, LastOp reducers** | 15 | ohlcv, counts, others | HIGH |
| 2 | **Burg AR** | 50 | ar_burg | HIGH (high frequency leaf) |
| 3 | **Kalman filter (Särkkä prefix scan)** | 80 | kalman, statespace | HIGH (Task #101) |
| 4 | **HMM (Baum-Welch + Viterbi)** | 100 | hmm | HIGH |
| 5 | **Range volatility (Parkinson/GK/RS/YZ)** | 25 | range_vol | HIGH |
| 6 | **Hill tail estimator** | 20 | heavy_tail | HIGH |
| 7 | **Morlet CWT (via FFT)** | 40 | cwt_wavelet, scattering dependency | HIGH |
| 8 | **VPIN (Easley-Lopez-O'Hara)** | 50 | vpin_bvc | MEDIUM |
| 9 | **NVG / HVG (visibility graphs)** | 65 | nvg, hvg | MEDIUM |
| 10 | **Hawkes process (exp kernel MLE)** | 80 | hawkes | MEDIUM |
| 11 | **DTW** | 30 | dtw | MEDIUM |
| 12 | **Edit distance (Levenshtein)** | 30 | edit_distance | MEDIUM |
| 13 | **Transfer entropy (Schreiber 2000)** | 40 | transfer_entropy, transfer_entropy_bin | MEDIUM |
| 14 | **CUSUM + binary segmentation** | 40 | classical_cp | MEDIUM |
| 15 | **PELT changepoint** | 60 | pelt | MEDIUM |
| 16 | **BOCPD** | 80 | bocpd | MEDIUM |
| 17 | **EMD sifting** | 80 | emd | MEDIUM |
| 18 | **FastICA** | 80 | ica | MEDIUM |
| 19 | **Wigner-Ville distribution** | 40 | wigner_ville | LOW |
| 20 | **Savitzky-Golay filter** | 40 | savgol | LOW |

**Total for top-20**: ~1,060 lines. Unlocks ~28 fintek leaves directly, plus dependencies.

---

## Architectural Observations

### Shared intermediates within fintek

Many leaves within a bin share computation. The bridge crate should exploit:

1. **Log returns** — computed once per bin, used by ~60 leaves. Single scatter pass.
2. **FFT of log returns** — computed once, used by: fft_spectral (5 resolutions!), welch, multitaper, cepstrum, coherence, fir_bandpass, energy_bands, wiener, periodicity, cwt_wavelet (if FFT-based), hilbert, spectral_entropy, hvg (optional). ~15 leaves share this.
3. **ACF** — used by autocorrelation, ar_model, dependence (Ljung-Box), periodicity, arma. 5 leaves.
4. **PACF** — autocorrelation, ar_model, arma. 3 leaves.
5. **Moments (mean/var/skew/kurt)** — distribution, normality, shape, heavy_tail, variability. 5+ leaves.
6. **Distance matrix (phase space)** — sample_entropy, correlation_dim, lyapunov, rqa, poincare, embedding (FNN). 6 leaves.
7. **GARCH σ²_t series** — garch, vol_dynamics, stochvol. 3 leaves.

**Implication for the bridge**: Compute each shared intermediate ONCE per bin, cache in TamSession, reuse across all consuming leaves. This is exactly the sharing infrastructure I documented in `sharing-rules.md`.

### The fintek "per bin" pattern

Every fintek leaf operates on a "bin" — a variable-length array of tick-level data. The bin is the natural grouping unit. This maps to:

```
accumulate(
    data: bin_data,
    grouping: GroupBy(bin_id),
    expr: leaf_computation,
    op: leaf_specific_reduce,
)
```

This is EXACTLY the tambear accumulate primitive. The bridge crate can be implemented as:
1. Group input ticks by bin_id (scatter into bin-shaped groups)
2. For each bin, run the leaf-specific computation
3. Emit V×DO output columns in MKTF format

For embarrassingly parallel leaves (all of Family 2), steps 1-3 fuse into a single kernel launch.

### GPU vs CPU split

Based on the decomposition analysis:

**GPU-friendly (A+G)**: Families 1, 2, 3, 6 (most), 8 (cross-correlation), 13 (PCA). ~60 leaves.

**CPU-sequential (SEQ/ITER)**: Families 4 (ARMA fitting), 5 (tests with iteration), 9 (GARCH fitting), 10 (some nonlinear), 11 (Kalman, HMM), 12 (Hawkes). ~40 leaves.

**Hybrid**: Families 7 (wavelets), 13 (ICA is iterative), 14 (topology). ~20 leaves.

The pathmaker's task #135 (bridge crate) should start with the GPU-friendly families — they're the highest throughput and unblock most leaves.

---

_Document is the formal mapping specification for the tambear-fintek bridge (Task #135)._
_Updated as new tambear primitives land and fintek leaves are migrated._
