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

## Work Plan: Three-Category Classification

Per the architectural clarification: **math stays in tambear; only compiled artifacts ship to fintek.** Every fintek rescue is simultaneously a tambear math entry being checked off.

Each fintek trunk-rs leaf falls into exactly one of:

- **Category A — READY TO COMPILE**: Tambear has a verified function that can be called with identical semantics. Pathmaker wires up the compile target; no new tambear math needed.
- **Category B — TAMBEAR IMPLEMENTATION NEEDED**: The math is well-understood and standard, but tambear doesn't have it yet. File as a tambear task, implement using accumulate+gather where possible, then compile.
- **Category C — FUNDAMENTALLY NEW MATH**: Requires novel tambear primitive or substantial research. File as a tambear task, derive from first principles, implement, then compile.

### Category A — Ready to Compile (80 leaves)

All DIRECT + COMPOSE + ADAPT entries from the family tables above. Tambear functions listed in the "Tambear mapping" column. Pathmaker's work is:
1. Build the leaf-shaped wrapper around the tambear call
2. Map leaf inputs (bin data) to tambear function signatures
3. Extract the leaf's DO/V columns from tambear return structs
4. Compile to fintek's output format

**Sub-categories by implementation complexity** (pathmaker ordering):

**A1 — Pure pointwise (11 leaves, fuse into one kernel class)**: returns, log_transform, sqrt_transform, reciprocal, notional, delta_value, delta_log, delta_percent, delta_direction, elapsed, cyclical.

**A2 — Moment-based (6 leaves, one scatter pass each)**: distribution, shannon_entropy, normality, counts, variability, shape.

**A3 — Spectral (9 leaves, share FFT)**: fft_spectral, welch, multitaper, lombscargle, cepstrum, hilbert, stft_leaf, spectral_entropy, fir_bandpass, energy_bands, periodicity.

**A4 — Time series compositions (5 leaves)**: autocorrelation, ar_model, arma, arima, dependence.

**A5 — Volatility compositions (7 leaves)**: garch, stochvol, vol_dynamics, vol_regime, roll_spread, jump_detection, signature_plot, realized_vol (minus TQ), tick_vol.

**A6 — Stationarity (2 leaves)**: stationarity, struct_break.

**A7 — Nonlinear dynamics (9 leaves)**: sample_entropy, permutation_entropy, hurst_rs, dfa, mfdfa, correlation_dim, lyapunov, embedding, poincare.

**A8 — Dimension reduction (8 leaves)**: pca, ssa, rmt, grassmannian, spectral_embedding, diff_geometry, tick_compression, harmonic.

**A9 — Cross-correlation (5 leaves)**: cross_correlation, tick_causality, msplit_temporal_coherence, coherence_matrix, transfer_analysis.

**A10 — Topology (1 leaf)**: persistent_homology.

**A11 — Bin-level microstructure (9 leaves)**: tick_alignment, tick_attractor, tick_complexity, tick_space, tick_scaling, tick_geometry (needs convex hull — actually B5), pith_attractor, shape, phase_transition.

**A12 — Meta / cross-leaf (6 leaves, depend on A1-A11 being ready)**: scaling_triple, coboundary, cadence_gradient, viscosity, taylor_fold, seismic.

**A13 — Miscellaneous compositions (7 leaves)**: ohlcv (depends on B1 FirstOp/LastOp), smoothers, ou_process, tick_ou, mutual_info, transfer_entropy (meta-style: depends on B14 transfer_entropy), coherence, sde.

### Category B — Tambear Implementation Needed (28 leaves → 20 new tambear functions)

Each gap below has a full tambear-side spec that the pathmaker or adversarial can implement directly. Specs include: algorithm name + paper, core formula, accumulate+gather decomposition, parameters with defaults, tambear module placement, and the fintek leaf(s) unlocked.

Specs are in the next section ("Tambear-Side Specs for Category B Gaps").

### Category C — Fundamentally New Math (0 leaves)

**None.** All 30 gaps identified in fintek trunk-rs are well-understood standard algorithms from the literature. None require novel primitives. This is because fintek is a signal-engineering project, not a research project — every leaf corresponds to a published method.

This means the rescue is a pure engineering exercise: implement 20 well-defined algorithms in tambear, wire up all 126 leaves, compile. No mathematical research required.

---

## Tambear-Side Specs for Category B Gaps

Each spec is written so the pathmaker or adversarial can implement it without consulting papers directly. Papers are cited for cross-reference, but the spec contains the full formula and decomposition.

### B1 — FirstOp, LastOp Reducers

**Paper**: None (standard aggregation primitives)
**Module**: `tambear/src/reduce_op.rs`
**Fintek leaves unlocked**: ohlcv, counts (partial)

**Purpose**: Add `First` and `Last` to the `ReduceOp` enum so that grouped accumulation can extract the first or last element per group.

**accumulate+gather**: The combine function for `First` is non-commutative (identity on left, discard right). Same for `Last` (discard left, keep right). These are associative monoids with identity = None.

```rust
pub enum ReduceOp {
    // ... existing variants ...
    First,  // identity: None; combine: (a, b) → a if a.is_some() else b
    Last,   // identity: None; combine: (a, b) → b if b.is_some() else a
}
```

For tambear's accumulate framework: the state is `Option<f64>`. First takes the left if defined, right otherwise. Last takes the right if defined, left otherwise. Both are associative (trivially).

**Parameters**: None.

**Lines**: ~15 (enum variant + combine logic + dispatch).

---

### B2 — Burg AR Method

**Paper**: Burg 1968, "A new analysis technique for time series data", Modern Spectrum Analysis, IEEE Press.
**Module**: `tambear/src/time_series.rs`
**Fintek leaves unlocked**: ar_burg

**Purpose**: Estimate AR coefficients via forward/backward prediction error minimization. More stable than Yule-Walker for short series.

**Algorithm**:
```
Input: x[0..n], max_order p
Output: ar[0..p], variance, effective_order

1. Initialize forward error ef = x, backward error eb = x.
2. Initial variance = mean((x - mean(x))²).
3. For k = 0..p:
     a. num = Σ_{j=k+1..n} ef[j] * eb[j-1]
     b. den = Σ_{j=k+1..n} (ef[j]² + eb[j-1]²)
     c. reflection coefficient: rc = -2·num/den, clamped to (-0.9999, 0.9999)
     d. Update AR coefficients via Levinson recursion:
        new_ar[k] = rc
        for j in 0..k:
          new_ar[j] = old_ar[j] + rc * old_ar[k-1-j]
     e. Update errors:
        for j in k+1..n:
          new_ef[j] = ef[j] + rc * eb[j-1]
          new_eb[j] = eb[j-1] + rc * ef[j]
     f. variance *= (1 - rc²)
4. Return (ar, variance, p)
```

**accumulate+gather**: The inner sums for num and den are `accumulate(All, Add)` — pure parallel reduction. The Levinson update is SEQ but O(p) per iteration with p ≤ 12, so trivial CPU. The outer loop over k is inherently sequential.

**Parameters**: `max_order: usize` (default min(12, n/4)).

**Signature**:
```rust
pub struct BurgResult {
    pub ar: Vec<f64>,          // length = effective_order
    pub variance: f64,
    pub effective_order: usize,
    pub reflection_coeffs: Vec<f64>,  // useful for diagnostics
}

pub fn ar_burg(data: &[f64], max_order: usize) -> BurgResult;
```

**Lines**: ~50.

---

### B3 — Kalman Filter via Särkkä Matrix Prefix Scan

**Paper**: Särkkä 2013, "Bayesian Filtering and Smoothing", Cambridge University Press. Särkkä & García-Fernández 2021, "Parallel Cubature Kalman Filter".
**Module**: `tambear/src/time_series.rs` or new `tambear/src/state_space.rs`
**Fintek leaves unlocked**: kalman, statespace
**Task**: #101 (already tracked)

**Purpose**: Linear Gaussian state-space filtering. Produces filtered means, covariances, innovation variances, and Kalman gains.

**Algorithm (standard form)**:
```
State model: x_t = F·x_{t-1} + w_t, w_t ~ N(0, Q)
Observation:  y_t = H·x_t + v_t, v_t ~ N(0, R)

Predict:
  x_pred = F · x_{t-1}
  P_pred = F · P_{t-1} · F' + Q

Update:
  innovation: ν_t = y_t - H · x_pred
  S_t = H · P_pred · H' + R
  K_t = P_pred · H' · S_t⁻¹
  x_t = x_pred + K_t · ν_t
  P_t = (I - K_t · H) · P_pred
```

**accumulate+gather (Särkkä 5-tuple)**: Express each Kalman step as an associative binary operation on a 5-tuple state (A, b, C, η, J). The prefix scan of these tuples computes the entire filter in O(log n) parallel depth.

The 5-tuple combine rule (Särkkä 2021, Eq. 21):
```
(A₁, b₁, C₁, η₁, J₁) ⊕ (A₂, b₂, C₂, η₂, J₂) = (A, b, C, η, J)
where:
  M = I + C₁ · J₂
  A = A₂ · M⁻¹ · A₁
  b = A₂ · M⁻¹ · (b₁ + C₁ · η₂) + b₂
  C = A₂ · M⁻¹ · C₁ · A₂' + C₂
  η = A₁' · (I - J₂ · M⁻¹ · C₁)' · η₂ + η₁
  J = A₁' · J₂ · M⁻¹ · A₁ + J₁
```

This is the canonical Särkkä prefix scan element. Each element is a matrix whose combine is associative — exactly the pattern for `accumulate(Prefix, MatrixPrefixCombine)`.

**Parameters**:
- `f: &[f64]` — state transition matrix F (d×d, row-major)
- `h: &[f64]` — observation matrix H (m×d)
- `q: &[f64]` — process noise Q (d×d)
- `r: &[f64]` — observation noise R (m×m)
- `x0: &[f64]` — initial state mean
- `p0: &[f64]` — initial state covariance
- `observations: &[f64]` — sequence of y_t (n×m)
- `n: usize`, `d: usize`, `m: usize` — dimensions

**Signature**:
```rust
pub struct KalmanResult {
    pub filtered_means: Vec<f64>,       // n × d
    pub filtered_covariances: Vec<f64>, // n × d × d
    pub innovations: Vec<f64>,           // n × m
    pub innovation_variances: Vec<f64>,  // n × m × m
    pub kalman_gains: Vec<f64>,          // n × d × m
    pub log_likelihood: f64,
}

pub fn kalman_filter(
    f: &[f64], h: &[f64], q: &[f64], r: &[f64],
    x0: &[f64], p0: &[f64],
    observations: &[f64],
    n: usize, d: usize, m: usize,
) -> KalmanResult;
```

**Lines**: ~100 (sequential form first, then prefix-scan form as Task #104's tridiagonal solver lands).

---

### B4 — HMM (Forward-Backward + Viterbi + Baum-Welch)

**Paper**: Rabiner 1989, "A tutorial on hidden Markov models", Proceedings of the IEEE 77(2):257-286.
**Module**: `tambear/src/hmm.rs` (new)
**Fintek leaves unlocked**: hmm

**Purpose**: Discrete HMM with K states. Three operations needed:
1. Forward-backward: compute α_t(i) = P(o_1..o_t, x_t=i) and β_t(i)
2. Viterbi: most likely state sequence
3. Baum-Welch: EM for parameter learning

**Algorithm**:

Forward recurrence:
```
α_1(i) = π_i · b_i(o_1)
α_{t+1}(j) = [Σ_i α_t(i) · a_ij] · b_j(o_{t+1})
```

Backward recurrence:
```
β_T(i) = 1
β_t(i) = Σ_j a_ij · b_j(o_{t+1}) · β_{t+1}(j)
```

Viterbi (max instead of sum):
```
δ_1(i) = π_i · b_i(o_1)
δ_{t+1}(j) = max_i [δ_t(i) · a_ij] · b_j(o_{t+1})
ψ_{t+1}(j) = argmax_i [δ_t(i) · a_ij]
```

Baum-Welch E-step: γ_t(i) = α_t(i)·β_t(i) / Σ_j α_t(j)·β_t(j), ξ_t(i,j) ∝ α_t(i)·a_ij·b_j(o_{t+1})·β_{t+1}(j)
Baum-Welch M-step: update π, A, B from γ, ξ.

**Critical numerical note**: Use log-space throughout to avoid underflow. α_t(i) → log α_t(i), sum → log-sum-exp.

**accumulate+gather**: Forward-backward is a matrix prefix scan in log space. The combine operation is `LogSumExpMerge` (max, sum of exp-differences). Viterbi uses `MaxMerge` instead. Both are associative → O(log n) parallel depth.

**Parameters**:
- `n_states: usize` — K
- `n_observations: usize` — alphabet size (for discrete HMM)
- `pi: &[f64]` — initial distribution (length K)
- `a: &[f64]` — transition matrix (K × K, row-major)
- `b: &[f64]` — emission matrix (K × n_observations)
- `observations: &[usize]` — discrete observation sequence

**Signature**:
```rust
pub struct HmmResult {
    pub log_likelihood: f64,
    pub filtered_probs: Vec<f64>,   // n × K (γ_t)
    pub viterbi_path: Vec<usize>,    // length n
    pub viterbi_log_prob: f64,
}

pub fn hmm_forward_backward(
    pi: &[f64], a: &[f64], b: &[f64],
    observations: &[usize], n_states: usize, n_obs_symbols: usize,
) -> HmmResult;

pub fn hmm_baum_welch(
    pi: &mut [f64], a: &mut [f64], b: &mut [f64],
    observations: &[usize], n_states: usize, n_obs_symbols: usize,
    max_iter: usize, tol: f64,
) -> BaumWelchResult;
```

**Lines**: ~120.

---

### B5 — Range Volatility Estimators (Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang)

**Papers**:
- Parkinson 1980, "The Extreme Value Method for Estimating the Variance of the Rate of Return", JBus 53:61-65
- Garman & Klass 1980, "On the Estimation of Security Price Volatilities from Historical Data", JBus 53:67-78
- Rogers & Satchell 1991, "Estimating Variance From High, Low and Closing Prices", Ann. Appl. Prob. 1:504-512
- Yang & Zhang 2000, "Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices", JBus 73:477-491

**Module**: `tambear/src/volatility.rs`
**Fintek leaves unlocked**: range_vol

**Purpose**: Estimate volatility from OHLC data. More efficient than close-to-close because it uses intraday extremes.

**Formulas** (all per-period σ² estimators):

Parkinson (uses H, L):
```
σ²_P = (1/(4·ln(2))) · (ln(H/L))²
```

Garman-Klass (uses O, H, L, C):
```
σ²_GK = 0.5·(ln(H/L))² - (2·ln(2) - 1)·(ln(C/O))²
```

Rogers-Satchell (uses O, H, L, C; drift-independent):
```
σ²_RS = ln(H/C)·ln(H/O) + ln(L/C)·ln(L/O)
```

Yang-Zhang (uses O, H, L, C from consecutive periods; drift-independent and minimum-variance):
```
σ²_YZ = σ²_overnight + k·σ²_open-to-close + (1-k)·σ²_RS
where k = 0.34 / (1.34 + (n+1)/(n-1))
```

**accumulate+gather**: All four are `accumulate(ByKey=period, Mean)` over log-ratio expressions — pure Kingdom A. Each sub-window's σ² is a scatter, then averaged.

**Parameters**:
- `open: &[f64]`, `high: &[f64]`, `low: &[f64]`, `close: &[f64]` — length n
- `window_size: usize` — subwindow over which to average

**Signature**:
```rust
pub struct RangeVolResult {
    pub parkinson: f64,
    pub garman_klass: f64,
    pub rogers_satchell: f64,
    pub yang_zhang: f64,
}

pub fn range_volatility(
    open: &[f64], high: &[f64], low: &[f64], close: &[f64],
    window_size: Option<usize>,  // None = use full series
) -> RangeVolResult;
```

**Lines**: ~40 (10 per estimator + dispatch).

---

### B6 — Hill Tail Index Estimator

**Paper**: Hill 1975, "A simple general approach to inference about the tail of a distribution", Annals of Statistics 3(5):1163-1174.
**Module**: `tambear/src/volatility.rs` or new `tambear/src/extremes.rs`
**Fintek leaves unlocked**: heavy_tail

**Purpose**: Estimate the tail index α of a Pareto-like distribution from the top-k order statistics. Lower α = heavier tail.

**Formula**:
```
Sort |r_i| in descending order: X_(1) ≥ X_(2) ≥ ... ≥ X_(n)
Hill estimator at threshold k:
  α̂_k = [(1/k) · Σ_{i=1}^k ln(X_(i) / X_(k+1))]⁻¹
Standard error:
  se(α̂_k) = α̂_k / √k
```

**k selection**: The main trick. Two common choices:
- Fixed: k = 0.1 · n (top 10%)
- Optimal (Danielsson et al. 2001): minimize MSE via bootstrap — too expensive
- Automatic (Drees et al. 2000): k = argmin over plot of α̂ vs k (Hill plot stability)

For fintek: use fixed k = max(10, 0.1·n) as default, allow user override.

**accumulate+gather**:
1. Sort |r| descending: `accumulate(abs, All, SortDescending)` (gather via sort)
2. Take top k+1 elements: gather
3. Compute sum of logs: `accumulate(top_k, All, Add)` on ln(X_(i) / X_(k+1))
4. Invert and divide.

**Parameters**:
- `data: &[f64]` — returns (we take absolute values internally)
- `k: Option<usize>` — number of tail observations (default max(10, n/10))

**Signature**:
```rust
pub struct HillResult {
    pub alpha: f64,
    pub se: f64,
    pub k_used: usize,
    pub tail_fraction: f64,
}

pub fn hill_tail_index(data: &[f64], k: Option<usize>) -> HillResult;
```

**Lines**: ~25.

---

### B7 — Morlet Continuous Wavelet Transform (CWT)

**Paper**: Torrence & Compo 1998, "A Practical Guide to Wavelet Analysis", BAMS 79(1):61-78.
**Module**: `tambear/src/signal_processing.rs`
**Fintek leaves unlocked**: cwt_wavelet, scattering (dependency)

**Purpose**: Time-frequency analysis via Morlet wavelet at multiple scales. Implemented efficiently via FFT.

**Algorithm**:
```
1. FFT the signal: X[k] = FFT(x[n]), k = 0..N-1
2. For each scale s_j = s_0 · 2^(j·δj):
     a. Angular frequency ω_k for each k
     b. Morlet wavelet in frequency domain:
        Ψ̂(s·ω) = π^(-1/4) · H(ω) · exp(-(s·ω - ω_0)²/2)
        where ω_0 = 6 (default central frequency for Morlet)
        and H(ω) = 1 if ω > 0, 0 otherwise (one-sided)
     c. Pointwise multiply: Y_j[k] = X[k] · conj(Ψ̂(s_j·ω_k)) · sqrt(2π·s_j/dt)
     d. Inverse FFT: W_j[n] = IFFT(Y_j)
3. Return W[j][n] — wavelet coefficients at scale j, time n
```

**Scale grid**: s_j = s_0 · 2^(j·δj), j = 0, 1, ..., J where:
- s_0 = 2·dt (smallest resolvable scale)
- δj = 0.125 or 0.25 (scale resolution)
- J = log₂(N·dt / s_0) / δj (largest scale)

**accumulate+gather**:
- Step 1: FFT (existing primitive)
- Step 2: Pointwise map per scale (embarrassingly parallel over scales AND frequencies)
- Step 3: IFFT per scale (existing primitive)

The per-scale computation is independent — all J scales can be computed in parallel. Total work: O(J·N·log N).

**Parameters**:
- `data: &[f64]`
- `dt: f64` — sample interval
- `s0: Option<f64>` — smallest scale (default 2·dt)
- `dj: Option<f64>` — scale resolution (default 0.125)
- `n_scales: Option<usize>` — override J (default automatic)
- `omega_0: f64` — Morlet central frequency (default 6.0)

**Signature**:
```rust
pub struct CwtResult {
    pub coefficients: Vec<Complex>,  // J × N complex values
    pub scales: Vec<f64>,             // length J
    pub frequencies: Vec<f64>,        // length J (Fourier-equivalent)
    pub n_scales: usize,
    pub n_samples: usize,
}

pub fn morlet_cwt(
    data: &[f64], dt: f64,
    s0: Option<f64>, dj: Option<f64>, n_scales: Option<usize>,
    omega_0: f64,
) -> CwtResult;
```

**Lines**: ~60.

---

### B8 — VPIN (Volume-Synchronized Probability of Informed Trading)

**Paper**: Easley, López de Prado, O'Hara 2012, "Flow Toxicity and Liquidity in a High-Frequency World", Review of Financial Studies 25(5):1457-1493.
**Module**: `tambear/src/volatility.rs` or new `tambear/src/microstructure.rs`
**Fintek leaves unlocked**: vpin_bvc

**Purpose**: Estimate order flow toxicity using Bulk Volume Classification (BVC).

**Algorithm**:
```
1. Compute price changes: Δp_i = p_i - p_{i-1}
2. Normalize by rolling std: z_i = Δp_i / σ_window
3. Classify each trade's volume as buy/sell via CDF:
     V_buy_i = V_i · Φ(z_i)
     V_sell_i = V_i · (1 - Φ(z_i))
   where Φ is the standard normal CDF
4. Aggregate into equal-volume buckets of size V_bucket:
     For each bucket: sum V_buy, V_sell, total volume
5. VPIN = mean(|V_buy - V_sell| / V_bucket) over the last n_buckets
```

**accumulate+gather**:
- Price changes: `accumulate(Prefix, Diff)` — O(log n) prefix scan
- Rolling std: windowed moments (need windowed primitive, or compute per-bucket)
- Classification: pointwise map with normal_cdf
- Bucket aggregation: `accumulate(ByKey=bucket_id, Add)` — pure scatter
- VPIN: mean over last k buckets

**Parameters**:
- `prices: &[f64]`, `volumes: &[f64]` — tick data
- `bucket_volume: f64` — volume per bucket
- `window_size: usize` — rolling std window (for z normalization)
- `n_buckets: usize` — number of buckets to average VPIN over

**Signature**:
```rust
pub struct VpinResult {
    pub vpin: f64,
    pub n_buckets_used: usize,
    pub bucket_imbalances: Vec<f64>,  // |V_buy - V_sell| / V_bucket per bucket
    pub mean_bucket_volume: f64,
}

pub fn vpin_bvc(
    prices: &[f64], volumes: &[f64],
    bucket_volume: f64, window_size: usize, n_buckets: usize,
) -> VpinResult;
```

**Lines**: ~60.

---

### B9 — Natural/Horizontal Visibility Graphs

**Papers**:
- Lacasa et al. 2008, "From time series to complex networks: The visibility graph", PNAS 105(13):4972-4975 (NVG)
- Luque et al. 2009, "Horizontal visibility graphs: Exact results for random time series", Phys. Rev. E 80:046103 (HVG)

**Module**: `tambear/src/graph.rs` or `tambear/src/time_series.rs`
**Fintek leaves unlocked**: nvg, hvg

**Purpose**: Convert a time series to a graph where each data point is a node, and edges connect points that can "see" each other geometrically. Extract graph metrics.

**Algorithm (NVG)**:
```
Two points (t_i, y_i) and (t_j, y_j) with i < j are connected iff
for every intermediate k with i < k < j:
  y_k < y_i + (y_j - y_i) · (t_k - t_i) / (t_j - t_i)
```
(They can "see" each other over the convex hull above intermediate bars.)

**Algorithm (HVG)**: Simpler variant — connection iff every intermediate y_k < min(y_i, y_j) (horizontal visibility).

**Output metrics**:
- Degree distribution: histogram of node degrees
- Degree exponent γ: power-law fit via MLE (Clauset et al. 2009)
- Mean degree
- Degree entropy: Shannon entropy of normalized degree distribution
- Clustering coefficient: mean local clustering

**accumulate+gather**: NVG construction is O(n²) naive. Optimal O(n log n) algorithms exist (Lan et al. 2015). For initial implementation, go O(n²):
- For each pair (i,j), check visibility: pure pairwise check (A+G on pairs).
- Degree of node i = count of visible neighbors: `accumulate(ByKey=i, Add)`.

HVG has linear-time O(n) algorithm via stack-based scan — more efficient.

**Parameters**:
- `data: &[f64]` — time series values (times assumed uniform)
- `horizontal: bool` — true for HVG, false for NVG

**Signature**:
```rust
pub struct VisibilityGraphResult {
    pub n_nodes: usize,
    pub n_edges: usize,
    pub degrees: Vec<usize>,
    pub degree_exponent: f64,    // power-law fit
    pub mean_degree: f64,
    pub degree_entropy: f64,
    pub clustering_coefficient: f64,
}

pub fn visibility_graph(data: &[f64], horizontal: bool) -> VisibilityGraphResult;
```

**Lines**: ~80 (NVG O(n²) + HVG stack + metrics).

---

### B10 — Hawkes Process (Exponential Kernel MLE)

**Paper**: Hawkes 1971, "Spectra of some self-exciting and mutually exciting point processes", Biometrika 58(1):83-90. Ogata 1981 for recursive likelihood.
**Module**: `tambear/src/stochastic.rs`
**Fintek leaves unlocked**: hawkes

**Purpose**: Fit self-exciting point process with intensity λ(t) = μ + Σ_{t_i < t} α·exp(-β·(t - t_i)).

**Algorithm (Ogata 1981 recursive log-likelihood)**:
```
Given event times t_1, ..., t_n on [0, T]:

ℓ(μ, α, β) = Σ_i ln(λ(t_i)) - ∫_0^T λ(s) ds

Intensity at event i (using recursion to avoid O(n²)):
  A_1 = 0
  A_i = exp(-β·(t_i - t_{i-1})) · (1 + A_{i-1})   for i >= 2
  λ(t_i) = μ + α·A_i

Integral:
  ∫_0^T λ(s) ds = μ·T + (α/β) · Σ_i [1 - exp(-β·(T - t_i))]
```

**Fit**: Grid search over (α, β) then coordinate descent refinement. μ solved analytically from α, β.

**accumulate+gather**: The recursion A_i depends on A_{i-1} — inherently sequential (SEQ). Can be expressed as a prefix scan with combine:
```
(A₁, δ₁) ⊕ (A₂, δ₂) = (exp(-β·δ₂)·(1 + A₁·exp(-β·δ₁... ∏ δ's)) + A₂, δ₁+δ₂)
```
Complicated but possible. For initial implementation, keep sequential (O(n) per likelihood evaluation, fast enough for grid search).

**Parameters**:
- `event_times: &[f64]` — sorted ascending
- `t_end: f64` — observation window end
- `threshold_percentile: f64` — (optional) percentile for event detection from returns

**Signature**:
```rust
pub struct HawkesResult {
    pub mu: f64,             // baseline intensity
    pub alpha: f64,          // excitation strength
    pub beta: f64,           // decay rate
    pub branching_ratio: f64,  // α/β (stability: must be < 1)
    pub log_likelihood: f64,
}

pub fn hawkes_fit(event_times: &[f64], t_end: f64, max_iter: usize) -> HawkesResult;
```

**Lines**: ~80.

---

### B11 — Dynamic Time Warping (DTW)

**Paper**: Sakoe & Chiba 1978, "Dynamic programming algorithm optimization for spoken word recognition", IEEE Trans. ASSP 26(1):43-49.
**Module**: `tambear/src/nonparametric.rs`
**Fintek leaves unlocked**: dtw

**Purpose**: Minimum-cost alignment between two sequences allowing local time stretching.

**Algorithm**:
```
Input: x[0..n], y[0..m]
DP matrix D[0..n, 0..m]:
  D[0][0] = |x[0] - y[0]|
  D[i][0] = D[i-1][0] + |x[i] - y[0]|
  D[0][j] = D[0][j-1] + |x[0] - y[j]|
  D[i][j] = |x[i] - y[j]| + min(D[i-1][j], D[i][j-1], D[i-1][j-1])
Return D[n-1][m-1] — minimum alignment cost
```

**Variants**:
- Sakoe-Chiba band (constraint |i-j| ≤ w) — reduces O(nm) to O(max(n,m)·w)
- Itakura parallelogram — slope constraint
- Multiple distance metrics (L1, L2, Lp)

**accumulate+gather**: DTW is inherently sequential (anti-diagonal dependency). However, anti-diagonal parallelism exists: cells on the same anti-diagonal can be computed in parallel, then the next anti-diagonal, etc. O(n+m) parallel depth with O(nm) total work. This is a wavefront pattern.

**Parameters**:
- `x: &[f64]`, `y: &[f64]`
- `window: Option<usize>` — Sakoe-Chiba band (None = unconstrained)
- `metric: fn(f64, f64) -> f64` — local distance (default L1)

**Signature**:
```rust
pub struct DtwResult {
    pub distance: f64,
    pub normalized_distance: f64,  // distance / path_length
    pub path_length: usize,
}

pub fn dtw(x: &[f64], y: &[f64], window: Option<usize>) -> DtwResult;
```

**Lines**: ~30.

---

### B12 — Edit Distance (Levenshtein)

**Paper**: Levenshtein 1966, "Binary codes capable of correcting deletions, insertions, and reversals", Soviet Physics Doklady 10(8):707-710.
**Module**: `tambear/src/nonparametric.rs`
**Fintek leaves unlocked**: edit_distance

**Purpose**: Minimum number of insertions, deletions, and substitutions to transform one sequence into another.

**Algorithm**: Same DP structure as DTW but on discrete symbols:
```
D[0][0] = 0
D[i][0] = i, D[0][j] = j
D[i][j] = min(D[i-1][j] + 1,                  // deletion
              D[i][j-1] + 1,                   // insertion
              D[i-1][j-1] + (x[i] != y[j]))   // substitution
```

**Symbolization**: For return series, bin into k symbols first (e.g., quartiles: {down, down_mild, up_mild, up}). Compute Levenshtein on symbol sequences.

**accumulate+gather**: Same anti-diagonal parallelism as DTW.

**Parameters**:
- `x: &[usize]`, `y: &[usize]` — symbol sequences

**Signature**:
```rust
pub struct EditDistanceResult {
    pub distance: usize,
    pub normalized_distance: f64,  // distance / max(n, m)
}

pub fn edit_distance(x: &[usize], y: &[usize]) -> EditDistanceResult;

// Helper: symbolize continuous series by quantiles
pub fn symbolize_by_quantiles(data: &[f64], n_symbols: usize) -> Vec<usize>;
```

**Lines**: ~30 + 10 for symbolization helper.

---

### B13 — Transfer Entropy (Schreiber 2000)

**Paper**: Schreiber 2000, "Measuring Information Transfer", Phys. Rev. Lett. 85(2):461-464.
**Module**: `tambear/src/information_theory.rs`
**Fintek leaves unlocked**: transfer_entropy, transfer_entropy_bin

**Purpose**: Information flow from X to Y beyond Y's own past. TE is asymmetric and detects causal influence at the information-theoretic level.

**Formula**:
```
TE(X → Y) = Σ p(y_{t+1}, y_t, x_t) · log [p(y_{t+1} | y_t, x_t) / p(y_{t+1} | y_t)]
```

Equivalently, in terms of joint entropies:
```
TE(X → Y) = H(Y_{t+1}, Y_t) + H(X_t, Y_t) - H(Y_t) - H(Y_{t+1}, X_t, Y_t)
```

**Binning**: Discretize X and Y into q quantile bins (default q = 4). Then joint entropies reduce to histogram counts.

**Algorithm**:
1. Quantile-bin X and Y into q symbols each.
2. Build 3D contingency tables: C_{i,j,k} = #{(x_t=i, y_t=j, y_{t+1}=k)}.
3. Compute four joint entropies via `shannon_entropy_from_counts`.
4. TE = H(Y_{t+1}, Y_t) + H(X_t, Y_t) - H(Y_t) - H(Y_{t+1}, X_t, Y_t).

**accumulate+gather**: 
- Binning: pointwise map (map_phi).
- Contingency table: `accumulate(ByKey=(i,j,k), Add)` — 3D scatter.
- Entropies: O(q³) CPU — trivial.

**Bias correction (Miller-Madow)**: H_corrected = H + (R-1)/(2n), where R = number of occupied bins. Apply to each entropy.

**Parameters**:
- `x: &[f64]`, `y: &[f64]` — time series (same length)
- `n_bins: usize` — quantile bins per variable (default 4)
- `lag: usize` — prediction lag (default 1)
- `bias_correct: bool` — Miller-Madow correction (default true)

**Signature**:
```rust
pub struct TransferEntropyResult {
    pub te_xy: f64,          // TE(X → Y)
    pub te_yx: f64,          // TE(Y → X)
    pub net_te: f64,         // te_xy - te_yx
    pub significance_ratio: f64,  // vs shuffled baseline (optional)
}

pub fn transfer_entropy(
    x: &[f64], y: &[f64],
    n_bins: usize, lag: usize, bias_correct: bool,
) -> TransferEntropyResult;
```

**Lines**: ~50.

---

### B14 — CUSUM + Binary Segmentation Changepoint

**Papers**:
- Page 1954, "Continuous inspection schemes", Biometrika 41:100-115 (CUSUM)
- Vostrikova 1981, "Detecting 'disorder' in multidimensional random processes" (binary segmentation)

**Module**: `tambear/src/time_series.rs`
**Fintek leaves unlocked**: classical_cp

**Purpose**: Detect structural breaks in mean (or other statistics) via cumulative sum tests.

**CUSUM Algorithm**:
```
Given x[0..n]:
  mean = mean(x)
  S_0 = 0
  S_i = S_{i-1} + (x_i - mean)
  CUSUM_max = max |S_i|
  break_location = argmax |S_i|
```

**Binary segmentation**: Recursively apply CUSUM to the segments before and after the detected break, until no significant break is found (threshold on CUSUM_max).

**accumulate+gather**:
- Mean: `accumulate(All, Mean)` — pure reduction.
- S_i: `accumulate(Prefix, Add)` on (x_i - mean) — prefix scan.
- max |S_i| and argmax: `accumulate(All, ArgMaxAbs)` — reduction.

**Parameters**:
- `data: &[f64]`
- `threshold: f64` — minimum CUSUM for significance
- `max_segments: usize` — maximum recursion depth

**Signature**:
```rust
pub struct ChangepointResult {
    pub n_changepoints: usize,
    pub changepoints: Vec<usize>,     // locations sorted
    pub max_cusum: f64,
    pub segment_means: Vec<f64>,
}

pub fn cusum_binary_segmentation(
    data: &[f64], threshold: f64, max_segments: usize,
) -> ChangepointResult;
```

**Lines**: ~40.

---

### B15 — PELT Changepoint Detection

**Paper**: Killick, Fearnhead & Eckley 2012, "Optimal detection of changepoints with a linear computational cost", JASA 107(500):1590-1598.
**Module**: `tambear/src/time_series.rs`
**Fintek leaves unlocked**: pelt

**Purpose**: Exact changepoint detection via dynamic programming with pruning. O(n) amortized.

**Algorithm**:
```
F(t) = min_{s < t} [F(s) + C(y_{s+1..t}) + β]
```
where C(segment) is the segment cost (negative log-likelihood for Gaussian: n·log(var)) and β is the BIC penalty.

**Pruning**: If F(s) + C(y_{s+1..t}) > F(t) + K for some constant K, then s cannot be a changepoint for any future t. Remove it from the candidate set.

**accumulate+gather**: The DP recursion is sequential. But each segment cost can be computed from running sums via `accumulate(Prefix, Sum)` for mean and `accumulate(Prefix, SumSq)` for variance. So C(y_{s+1..t}) reduces to O(1) per evaluation given prefix sums.

**Parameters**:
- `data: &[f64]`
- `penalty: Option<f64>` — β (default BIC: 2·log(n)·σ²)
- `min_segment_length: usize` — minimum segment size (default 1)

**Signature**:
```rust
pub struct PeltResult {
    pub n_changepoints: usize,
    pub changepoints: Vec<usize>,
    pub segment_costs: Vec<f64>,
    pub total_cost: f64,
}

pub fn pelt_changepoint(
    data: &[f64], penalty: Option<f64>, min_segment_length: usize,
) -> PeltResult;
```

**Lines**: ~60.

---

### B16 — Bayesian Online Changepoint Detection (BOCPD)

**Paper**: Adams & MacKay 2007, "Bayesian Online Changepoint Detection", arXiv:0710.3742.
**Module**: `tambear/src/time_series.rs` or `tambear/src/bayesian.rs`
**Fintek leaves unlocked**: bocpd

**Purpose**: Online (sequential) changepoint detection. Tracks posterior over "run length" r_t — time since last changepoint.

**Algorithm** (Gaussian predictive, constant hazard H = 1/λ):
```
Message passing:
  P(r_t = r+1 | x_{1:t}) ∝ P(x_t | r_{t-1}=r) · P(r_{t-1}=r | x_{1:t-1}) · (1 - H)
  P(r_t = 0 | x_{1:t}) ∝ Σ_r P(x_t | r_{t-1}=r) · P(r_{t-1}=r | x_{1:t-1}) · H

Normalize: P(r_t | x_{1:t}) = above / Σ

Gaussian predictive (conjugate Normal-Gamma):
  Maintain (n_r, μ_r, κ_r, α_r, β_r) for each run length r
  p(x | r) = t-distribution with updated params
```

**accumulate+gather**: Inherently sequential (online algorithm). Each time step: update all run-length hypotheses (A+G over r). Then advance.

**Parameters**:
- `data: &[f64]`
- `hazard_rate: f64` — H (default 1/100 = expected run length of 100)
- `prior_mu: f64`, `prior_kappa: f64`, `prior_alpha: f64`, `prior_beta: f64` — Normal-Gamma prior

**Signature**:
```rust
pub struct BocpdResult {
    pub run_length_distribution: Vec<Vec<f64>>,  // n × (n+1)
    pub max_a_posteriori: Vec<usize>,  // most likely run length at each t
    pub changepoint_probs: Vec<f64>,    // P(r_t = 0 | x_{1:t})
}

pub fn bocpd(
    data: &[f64], hazard_rate: f64,
    prior_mu: f64, prior_kappa: f64, prior_alpha: f64, prior_beta: f64,
) -> BocpdResult;
```

**Lines**: ~80.

---

### B17 — Empirical Mode Decomposition (EMD)

**Paper**: Huang et al. 1998, "The empirical mode decomposition and the Hilbert spectrum for nonlinear and non-stationary time series analysis", Proc. R. Soc. A 454:903-995.
**Module**: `tambear/src/signal_processing.rs`
**Fintek leaves unlocked**: emd

**Purpose**: Decompose signal into intrinsic mode functions (IMFs) via iterative sifting. Each IMF captures one oscillatory mode.

**Algorithm**:
```
1. Find all local maxima and minima.
2. Interpolate max envelope (cubic spline on maxima) and min envelope.
3. Mean envelope m = (max_env + min_env) / 2.
4. Candidate IMF: h = x - m.
5. Check IMF criterion: # extrema ≈ # zero crossings AND mean envelope ≈ 0.
6. If yes, store h as IMF, subtract from x, repeat from step 1.
7. If no, set x = h, repeat sifting from step 1.
8. Stop when residual is monotonic.
```

**Difficulty**: Extrema detection is sequential. Spline interpolation per envelope. The sifting criterion is heuristic. Not a clean A+G algorithm.

**accumulate+gather**: 
- Extrema detection: can be parallelized as local comparisons (window size 3).
- Spline interpolation: mostly sequential (tridiagonal solve). Use existing `interpolation::natural_cubic_spline`.
- Sifting loop: inherently iterative (ITER).

**Parameters**:
- `data: &[f64]`
- `max_imfs: usize` — maximum IMFs to extract (default 10)
- `sift_tol: f64` — stopping criterion for sifting (default 0.2, Huang's recommendation)
- `max_sift_iter: usize` — maximum sifting iterations per IMF (default 100)

**Signature**:
```rust
pub struct EmdResult {
    pub imfs: Vec<Vec<f64>>,   // list of IMFs, each length n
    pub residual: Vec<f64>,     // trend (monotonic residual)
    pub n_imfs: usize,
}

pub fn emd(
    data: &[f64], max_imfs: usize, sift_tol: f64, max_sift_iter: usize,
) -> EmdResult;
```

**Lines**: ~100 (extrema + spline + sifting loop + stopping criteria).

---

### B18 — FastICA

**Paper**: Hyvärinen & Oja 1997, "A fast fixed-point algorithm for independent component analysis", Neural Computation 9(7):1483-1492.
**Module**: `tambear/src/dim_reduction.rs`
**Fintek leaves unlocked**: ica

**Purpose**: Find linear transformation that makes output components maximally non-Gaussian (independent).

**Algorithm**:
```
1. Center data: X = X - mean(X)
2. Whiten via PCA: X_white = D^(-1/2) · V' · X, where V, D = eigendecomp of cov(X)
3. For each component:
     a. Initialize w randomly (unit norm)
     b. Loop:
        w_new = E[x · g(w'x)] - E[g'(w'x)] · w    (fixed-point update)
        where g(u) = tanh(u) or u³ (contrast functions)
        Orthogonalize against previously found components: w -= Σ_j (w'w_j) · w_j
        Normalize: w /= ||w||
        Check convergence: |1 - |w_new' · w_old|| < tol
4. Return unmixing matrix W (stacked w's)
```

**accumulate+gather**: 
- Whitening: SVD (existing).
- Expectations E[x·g(w'x)] and E[g'(w'x)]: `accumulate(All, Mean)` over pointwise functions.
- Orthogonalization: matrix-vector operations.

Each iteration is A+G. Outer loop is ITER.

**Parameters**:
- `data: &[f64]` (n × d, row-major)
- `n_components: usize`
- `contrast: ContrastFunction` — enum { LogCosh, Cube, Gaussian }
- `max_iter: usize` (default 200)
- `tol: f64` (default 1e-4)

**Signature**:
```rust
pub enum IcaContrast { LogCosh, Cube, Gaussian }

pub struct IcaResult {
    pub components: Vec<f64>,     // n_components × d (unmixing matrix)
    pub sources: Vec<f64>,        // n × n_components (estimated independent sources)
    pub mean: Vec<f64>,           // column means (for transform)
    pub whitening: Vec<f64>,      // whitening matrix
    pub iterations: Vec<usize>,   // iterations per component
    pub negentropies: Vec<f64>,   // negentropy per component
}

pub fn fast_ica(
    data: &[f64], n: usize, d: usize, n_components: usize,
    contrast: IcaContrast, max_iter: usize, tol: f64,
) -> IcaResult;
```

**Lines**: ~100.

---

### B19 — Wigner-Ville Distribution

**Paper**: Wigner 1932, "On the quantum correction for thermodynamic equilibrium", Phys. Rev. 40:749. Ville 1948, "Théorie et applications de la notion de signal analytique".
**Module**: `tambear/src/signal_processing.rs`
**Fintek leaves unlocked**: wigner_ville

**Purpose**: Quadratic time-frequency distribution. Higher resolution than spectrogram but has cross-term artifacts.

**Discrete formula**:
```
W(t, f) = 2 · Re[Σ_τ x(t+τ) · x*(t-τ) · exp(-j·4π·f·τ)]
```

**Algorithm**:
1. Compute analytic signal: z(t) = x(t) + j·H(x(t))  (Hilbert transform)
2. For each time t, form r(τ) = z(t+τ) · z*(t-τ) for τ in a window.
3. FFT r(τ) in τ to get W(t, f).
4. Aggregate features: time-frequency concentration (Rényi entropy of |W|), instantaneous frequency variance, cross-term energy, marginal entropy.

**accumulate+gather**:
- Hilbert transform: existing primitive.
- Pointwise products r(τ): map.
- FFT: existing primitive.
- Per time t independent: all t can be parallelized.

**Parameters**:
- `data: &[f64]`
- `window_length: usize` — τ range (default min(n/4, 256))

**Signature**:
```rust
pub struct WignerVilleResult {
    pub distribution: Vec<f64>,  // n_time × n_freq
    pub n_time: usize,
    pub n_freq: usize,
    pub time_freq_concentration: f64,  // Rényi entropy (lower = more concentrated)
    pub instantaneous_freq_var: f64,
    pub cross_term_energy: f64,        // energy in negative regions
    pub marginal_entropy: f64,
}

pub fn wigner_ville(data: &[f64], window_length: usize) -> WignerVilleResult;
```

**Lines**: ~60.

---

### B20 — Savitzky-Golay Filter

**Paper**: Savitzky & Golay 1964, "Smoothing and Differentiation of Data by Simplified Least Squares Procedures", Anal. Chem. 36(8):1627-1639.
**Module**: `tambear/src/signal_processing.rs`
**Fintek leaves unlocked**: savgol

**Purpose**: Polynomial least-squares smoothing in a sliding window. Preserves higher moments (peaks, widths) better than moving average.

**Algorithm**:
```
Pre-compute convolution coefficients h[k] by fitting polynomial of degree p
to window of length w:
  Let A be the Vandermonde-like matrix of window offsets [-m..m] with columns [1, k, k², ..., k^p]
  h = (A' A)⁻¹ A' e₀
  where e₀ picks the center value (for smoothing) or derivative value (for differentiation)

Apply as convolution:
  y[i] = Σ_{k=-m..m} h[k] · x[i+k]
```

**For derivatives**: Replace e₀ with e_d where d is the derivative order. The same matrix inverse gives coefficients for the d-th derivative.

**accumulate+gather**:
- Coefficient computation: one-time matrix solve (small, p+1 × p+1).
- Convolution: `accumulate(ByKey=output_index, Add)` — pure scatter with stencil pattern.

**Parameters**:
- `data: &[f64]`
- `window_length: usize` — must be odd (default 11)
- `polyorder: usize` — polynomial degree (default 3)
- `deriv: usize` — derivative order (default 0 = smoothing, 1 = first derivative, etc.)

**Signature**:
```rust
pub struct SavGolResult {
    pub smoothed: Vec<f64>,
    pub derivatives: Option<Vec<f64>>,  // only if deriv > 0
    pub coefficients: Vec<f64>,  // for reference
}

pub fn savitzky_golay(
    data: &[f64], window_length: usize, polyorder: usize, deriv: usize,
) -> SavGolResult;
```

**Lines**: ~50.

---

## Category B Summary

**20 tambear implementations totaling ~1,260 lines** unlock 28 fintek trunk-rs leaves:

| Spec | Name | Lines | Fintek leaves |
|------|------|-------|---------------|
| B1 | FirstOp/LastOp | 15 | ohlcv, counts |
| B2 | Burg AR | 50 | ar_burg |
| B3 | Kalman filter | 100 | kalman, statespace |
| B4 | HMM | 120 | hmm |
| B5 | Range volatility | 40 | range_vol |
| B6 | Hill tail index | 25 | heavy_tail |
| B7 | Morlet CWT | 60 | cwt_wavelet, scattering |
| B8 | VPIN | 60 | vpin_bvc |
| B9 | Visibility graphs | 80 | nvg, hvg |
| B10 | Hawkes process | 80 | hawkes |
| B11 | DTW | 30 | dtw |
| B12 | Edit distance | 40 | edit_distance |
| B13 | Transfer entropy | 50 | transfer_entropy, transfer_entropy_bin |
| B14 | CUSUM + binseg | 40 | classical_cp |
| B15 | PELT | 60 | pelt |
| B16 | BOCPD | 80 | bocpd |
| B17 | EMD | 100 | emd |
| B18 | FastICA | 100 | ica |
| B19 | Wigner-Ville | 60 | wigner_ville |
| B20 | Savitzky-Golay | 50 | savgol |

**Additional smaller gaps** (already noted in family tables, not detailed above — each <15 lines):
- Miller-Madow bias correction for MI (5 lines, goes into information_theory.rs)
- 1D Wasserstein-1 distance (15 lines, goes into nonparametric.rs)
- Tripower quarticity for realized vol (15 lines, goes into volatility.rs)
- Fisher information from histogram (25 lines, goes into information_theory.rs)
- LZ76 complexity parser (30 lines, goes into complexity.rs)
- Recurrence quantification analysis (50 lines, goes into complexity.rs — uses existing distance matrix)
- Cross-PSD for coherence (20 lines, goes into signal_processing.rs — uses existing FFT)
- Seismic MLEs (Gutenberg-Richter b, Omori p) (30 lines, new extremes.rs module)
- 2D convex hull (30 lines, goes into graph.rs or new geometry.rs)
- Logsig level-1/2 Levy area (40 lines, new rough_paths.rs or signal_processing.rs)
- STL decomposition (60 lines, goes into time_series.rs)
- SDE Nadaraya-Watson drift/diffusion (30 lines, goes into stochastic.rs — uses existing KDE)

**Grand total**: ~1,640 lines of new tambear code. Unlocks all 30 GAP leaves plus dependencies. Pathmaker can work on these in priority order (B1 unblocks ohlcv which is Family 3 foundation; B3 + B4 unblock state-space).

---

## Tambear-Side Specs — Batch C (Small Gaps)

The C-series specs cover the ~12 smaller gaps (each <40 lines) that I mentioned in the catalog but didn't detail. Each is a thin addition to an existing module.

### C1 — Miller-Madow Bias Correction for Mutual Information

**Paper**: Miller 1955, "Note on the bias of information estimates", Information Theory in Psychology, 95-100. Madow correction: Paninski 2003, "Estimation of entropy and mutual information", Neural Computation 15(6):1191-1253.
**Module**: `tambear/src/information_theory.rs`
**Unlocks**: mutual_info, transfer_entropy (quality improvement)

**Purpose**: Plug-in MI estimators have negative bias that decays as -(R-1)/(2n) where R is the number of occupied contingency cells. Miller-Madow correction subtracts this.

**Formula**:
```
MI_corrected = MI_plugin + (R_x + R_y - R_xy - 1) / (2n)
```
where:
- R_x = number of non-empty row marginals
- R_y = number of non-empty column marginals
- R_xy = number of non-empty joint cells
- n = sample size

**Signature**:
```rust
pub fn mutual_information_miller_madow(
    contingency: &[f64], nx: usize, ny: usize,
) -> f64;
```

**Lines**: ~10 on top of existing `mutual_information`.

---

### C2 — 1D Wasserstein-1 Distance

**Paper**: Villani 2009, "Optimal Transport: Old and New", Springer. For 1D closed form: Kantorovich 1942.
**Module**: `tambear/src/nonparametric.rs`
**Unlocks**: dist_distance leaf

**Purpose**: Earth Mover's Distance between two 1D distributions. In 1D, it has a closed form via sorted samples.

**Formula**:
```
W₁(P, Q) = ∫ |F_P(x) - F_Q(x)| dx
       = (1/n) · Σ_i |X_(i) - Y_(i)|    (for equal sample sizes, both sorted ascending)
```

For unequal sizes: interpolate empirical CDFs and integrate via trapezoidal rule.

**Signature**:
```rust
pub fn wasserstein_1d(x: &[f64], y: &[f64]) -> f64;
```

**Algorithm**:
```
1. Sort copies of x and y ascending.
2. If |x| == |y|: return mean(|X_(i) - Y_(i)|).
3. Else: build empirical CDFs on the merged support, integrate |F_x - F_y|.
```

**accumulate+gather**: Sort (gather), then pointwise subtraction + mean (accumulate).

**Lines**: ~15.

---

### C3 — Tripower and Quadpower Quarticity

**Paper**: Barndorff-Nielsen & Shephard 2004, "Power and Bipower Variation with Stochastic Volatility and Jumps", J. Financial Econometrics 2(1):1-37.
**Module**: `tambear/src/volatility.rs`
**Unlocks**: realized_vol (TQ and QQ outputs)

**Purpose**: Higher-order multipower variation used for standardizing the BNS jump test and estimating integrated quarticity.

**Formulas**:
```
μ_p = E[|Z|^p] = 2^(p/2) · Γ((p+1)/2) / Γ(1/2)

Tripower Quarticity:
  TQ = N · μ_{4/3}^{-3} · Σ_{i=3..N} |r_i|^{4/3} · |r_{i-1}|^{4/3} · |r_{i-2}|^{4/3}

Quadpower Quarticity:
  QQ = N · μ_1^{-4} · Σ_{i=4..N} |r_i| · |r_{i-1}| · |r_{i-2}| · |r_{i-3}|
```

**Signature**:
```rust
pub fn tripower_quarticity(returns: &[f64]) -> f64;
pub fn quadpower_quarticity(returns: &[f64]) -> f64;
```

**accumulate+gather**: `accumulate(Windowed(3), ProductOfAbs)` then sum. Pure Kingdom A.

**Lines**: ~20.

---

### C4 — Fisher Information from Histogram

**Paper**: Cover & Thomas 2006, "Elements of Information Theory", Chapter 17.
**Module**: `tambear/src/information_theory.rs`
**Unlocks**: fisher_info leaf

**Purpose**: Non-parametric estimate of Fisher information: I(θ) = ∫ (f'(x))² / f(x) dx. Uses finite differences on histogram.

**Formula** (histogram estimator):
```
Bin data into k bins with centers x_i and frequencies p_i.
Let h = bin width.
Finite difference: f'(x_i) ≈ (p_{i+1} - p_{i-1}) / (2h)
Fisher info:
  I ≈ Σ_i (f'(x_i))² / p_i · h
    = Σ_i ((p_{i+1} - p_{i-1}) / (2h))² / p_i · h
```

**Fisher-Rao distance from Gaussian**: For a Gaussian with the same variance σ², I_gauss = 1/σ². The distance:
```
D_FR = 2 · |arctan(√(I_data·σ²)) - π/4|
```

**Signature**:
```rust
pub struct FisherInfoResult {
    pub fisher_info: f64,
    pub fisher_distance_gauss: f64,
    pub gradient_norm: f64,  // mean |score| = mean |f'/f|
}

pub fn fisher_information_histogram(
    data: &[f64], n_bins: usize,
) -> FisherInfoResult;
```

**Lines**: ~30.

---

### C5 — Lempel-Ziv 76 Complexity

**Paper**: Lempel & Ziv 1976, "On the Complexity of Finite Sequences", IEEE Trans. Information Theory 22(1):75-81.
**Module**: `tambear/src/complexity.rs`
**Unlocks**: lz_complexity leaf

**Purpose**: Count the number of distinct phrases in the LZ76 parsing. Normalized LZ complexity measures algorithmic randomness of a symbolic sequence.

**Algorithm**:
```
Input: binary (or k-ary) sequence s[0..n]
Output: c(s) = number of distinct phrases in parsing

i = 0
c = 0
while i < n:
    // Find longest prefix of s[i..] that also occurs as substring of s[0..i+k]
    // where k = length of the new prefix
    k = 1
    while i + k <= n:
        if s[i..i+k] is NOT a substring of s[0..i+k-1]:
            c += 1
            i += k
            break
        k += 1
    else:
        c += 1
        break
```

**Normalized LZ**: 
```
LZ_norm = c(s) · ln(n) / n     (theoretical upper bound)
```

**Symbolization** (for continuous returns): Binary — `1` if x > median, `0` otherwise. Or 4-ary quantile binning.

**Signature**:
```rust
pub struct LzComplexityResult {
    pub lz_complexity: usize,       // raw count c(s)
    pub normalized_lz: f64,          // c·ln(n)/n
    pub compression_ratio: f64,      // c·log₂(c)/n (bit savings)
}

pub fn lz76_complexity(symbols: &[u8]) -> LzComplexityResult;
pub fn lz76_from_returns(data: &[f64], binary: bool) -> LzComplexityResult;
```

**accumulate+gather**: Inherently sequential. LZ76 is a state-dependent parser. O(n²) naive, O(n log n) with suffix trees, O(n) with suffix automaton. ~30 lines for the O(n²) version.

**Lines**: ~30.

---

### C6 — Recurrence Quantification Analysis (RQA)

**Paper**: Marwan et al. 2007, "Recurrence plots for the analysis of complex systems", Physics Reports 438(5-6):237-329.
**Module**: `tambear/src/complexity.rs`
**Unlocks**: rqa leaf

**Purpose**: Quantify recurrence structure of a dynamical system via diagonal and vertical line statistics in the recurrence plot.

**Algorithm**:
```
1. Phase-space reconstruction via delay embedding (m, tau) — reuse tambear embedding code.
2. Distance matrix D[i,j] = ||x_i - x_j|| (reuse tambear distance matrix).
3. Recurrence matrix R[i,j] = 1 if D[i,j] < epsilon, else 0.
4. Extract line statistics:
     - Recurrence rate: RR = (1/N²) · Σ R[i,j]
     - Determinism: DET = sum of diagonal line lengths ≥ l_min / Σ R[i,j]
     - Average diagonal length: mean diagonal line length ≥ l_min
     - Laminarity: LAM = sum of vertical line lengths ≥ v_min / Σ R[i,j]
     - Trapping time: mean vertical line length ≥ v_min
     - Entropy of diagonal line distribution
```

**accumulate+gather**:
- Distance matrix: existing TiledEngine (SHARE with sample_entropy, correlation_dim)
- Threshold: pointwise map
- Diagonal line lengths: scan diagonals of R — each diagonal is a reduction over runs (Kingdom B per diagonal, A+G across diagonals)

**Parameters**:
- `data: &[f64]`
- `m: usize` — embedding dimension (default 3)
- `tau: usize` — delay (default 1)
- `epsilon: f64` — recurrence threshold (default 10% of max distance)
- `l_min: usize` — minimum diagonal length (default 2)
- `v_min: usize` — minimum vertical length (default 2)

**Signature**:
```rust
pub struct RqaResult {
    pub recurrence_rate: f64,
    pub determinism: f64,
    pub mean_diagonal_length: f64,
    pub max_diagonal_length: usize,
    pub diagonal_entropy: f64,
    pub laminarity: f64,
    pub trapping_time: f64,
    pub max_vertical_length: usize,
}

pub fn rqa(
    data: &[f64], m: usize, tau: usize,
    epsilon: f64, l_min: usize, v_min: usize,
) -> RqaResult;
```

**Lines**: ~50.

---

### C7 — Cross Power Spectral Density (for coherence)

**Paper**: Welch 1967, "The use of fast Fourier transform for the estimation of power spectra", IEEE Trans. AE 15(2):70-73. Carter et al. 1973 for coherence.
**Module**: `tambear/src/signal_processing.rs`
**Unlocks**: coherence leaf

**Purpose**: Cross-spectral density S_xy(f) = E[X(f)·Y*(f)]. Squared coherence C²_xy(f) = |S_xy(f)|² / (S_xx(f)·S_yy(f)).

**Algorithm** (Welch-style):
```
1. Split x and y into K overlapping segments of length L.
2. For each segment: window, FFT → X_k, Y_k.
3. Compute:
     S_xx(f) = (1/K) Σ_k |X_k(f)|²
     S_yy(f) = (1/K) Σ_k |Y_k(f)|²
     S_xy(f) = (1/K) Σ_k X_k(f) · conj(Y_k(f))
4. Coherence: C²_xy(f) = |S_xy(f)|² / (S_xx(f) · S_yy(f))
```

**accumulate+gather**: Each segment is independent → A+G over segments. Within each segment: FFT + pointwise products. Cross-spectrum accumulation: `accumulate(ByKey=freq_bin, ComplexMean)`.

**Signature**:
```rust
pub struct CrossSpectralResult {
    pub frequencies: Vec<f64>,
    pub s_xx: Vec<f64>,
    pub s_yy: Vec<f64>,
    pub s_xy_real: Vec<f64>,
    pub s_xy_imag: Vec<f64>,
    pub coherence_sq: Vec<f64>,
    pub phase: Vec<f64>,        // phase of S_xy
}

pub fn cross_spectral_density(
    x: &[f64], y: &[f64], segment_len: usize, overlap: usize, fs: f64,
) -> CrossSpectralResult;
```

**Lines**: ~40 (reuses existing `welch` internals).

---

### C8 — Gutenberg-Richter and Omori Law MLEs

**Papers**:
- Gutenberg & Richter 1944, "Frequency of earthquakes in California", Bull. Seism. Soc. Am. 34(4):185-188
- Aki 1965, "Maximum likelihood estimate of b in the formula log N = a - bM", Bull. ERI 43:237-239 (MLE for b-value)
- Utsu 1961, "A statistical study on the occurrence of aftershocks", Geophys. Mag. 30:521-605 (Omori)

**Module**: `tambear/src/volatility.rs` or new `tambear/src/extremes.rs`
**Unlocks**: seismic leaf

**Purpose**: Fit power-law magnitude-frequency distribution and Omori temporal decay for extreme events.

**Formulas**:

Gutenberg-Richter b-value (Aki MLE):
```
b̂ = log₁₀(e) / (M̄ - M_c)
where M̄ = mean magnitude ≥ cutoff M_c
```

Omori law (aftershock decay):
```
n(t) = K / (t + c)^p

Log-linearize and OLS: log(n(t)) = log(K) - p·log(t + c)
(Or iterative MLE over (K, c, p).)
```

Bath's law ratio:
```
ΔM = M_main - M_largest_aftershock
Typical ΔM ≈ 1.2
```

**Signature**:
```rust
pub struct SeismicResult {
    pub gr_b_value: f64,
    pub gr_a_value: f64,
    pub omori_p: f64,
    pub omori_k: f64,
    pub bath_ratio: f64,
    pub n_extreme: usize,
}

pub fn seismic_laws(
    returns: &[f64],
    magnitude_cutoff: Option<f64>,  // default: 3 × std
) -> SeismicResult;
```

**Lines**: ~40.

---

### C9 — 2D Convex Hull (Graham Scan)

**Paper**: Graham 1972, "An efficient algorithm for determining the convex hull of a finite planar set", Information Processing Letters 1(4):132-133.
**Module**: `tambear/src/graph.rs` or new `tambear/src/geometry.rs`
**Unlocks**: tick_geometry leaf

**Purpose**: Compute convex hull of 2D points. Used for phase portrait area estimation.

**Algorithm (Graham scan)**:
```
1. Find pivot p₀ = point with lowest y (break ties by lowest x).
2. Sort remaining points by polar angle from p₀.
3. Stack-based scan: for each point p:
     while len(stack) >= 2 and cross(stack[-2], stack[-1], p) <= 0:
         stack.pop()
     stack.push(p)
4. Return stack as hull vertices in CCW order.
```

**Cross product** (sign determines turn direction):
```
cross(O, A, B) = (A.x - O.x)(B.y - O.y) - (A.y - O.y)(B.x - O.x)
```

**Area of convex polygon** (shoelace formula):
```
Area = |Σ_i (x_i · y_{i+1} - x_{i+1} · y_i)| / 2
```

**accumulate+gather**: Sorting is gather. The scan is inherently sequential (Kingdom B). For n < 10^5, pure CPU is fast enough.

**Signature**:
```rust
pub struct ConvexHullResult {
    pub hull_indices: Vec<usize>,    // indices into input points in CCW order
    pub area: f64,
    pub perimeter: f64,
}

pub fn convex_hull_2d(points: &[(f64, f64)]) -> ConvexHullResult;
```

**Lines**: ~50.

---

### C10 — Log-Signature (Level-1 and Level-2 Lévy Area)

**Papers**:
- Chen 1958, "Integration of paths — a faithful representation of paths by noncommutative formal power series", Trans. AMS 89(2):395-407
- Lyons 1998, "Differential equations driven by rough signals", Revista Matematica Iberoamericana 14(2):215-310

**Module**: new `tambear/src/rough_paths.rs` or `tambear/src/signal_processing.rs`
**Unlocks**: logsig leaf

**Purpose**: Encode sequential structure of a multidimensional path via iterated integrals. Level-1 = increments, Level-2 = Lévy area.

**Formulas** (for path (x, y) = price, volume):

Level-1 increments:
```
I_x = x_N - x_0
I_y = y_N - y_0
```

Level-2 iterated integrals:
```
I_xx = ∫_0^T (x_t - x_0) dx_t = 0.5 · (x_T - x_0)²   (for smooth paths)
I_yy = 0.5 · (y_T - y_0)²
I_xy = ∫_0^T (x_t - x_0) dy_t   (path-dependent, NOT just 0.5·(x_T - x_0)(y_T - y_0))
I_yx = ∫_0^T (y_t - y_0) dx_t
```

Lévy area:
```
A(x, y) = 0.5 · (I_xy - I_yx)
```
(Signed area enclosed by the path and its chord.)

**Discrete computation** (Riemann sum):
```
I_xy ≈ Σ_{i=0..N-1} (x_i - x_0) · (y_{i+1} - y_i)
I_yx ≈ Σ_{i=0..N-1} (y_i - y_0) · (x_{i+1} - x_i)
```

**accumulate+gather**: Both iterated integrals are `accumulate(All, Add)` over pointwise products of centered paths and increments. Pure Kingdom A.

**Signature**:
```rust
pub struct LogSigResult {
    pub increment_x: f64,        // level-1
    pub increment_y: f64,
    pub iterated_xx: f64,         // level-2
    pub iterated_yy: f64,
    pub iterated_xy: f64,
    pub iterated_yx: f64,
    pub levy_area: f64,           // 0.5 · (I_xy - I_yx)
    pub l2_norm: f64,             // sqrt(sum of squares)
    pub depth2_energy_fraction: f64,  // level-2 energy / total energy
}

pub fn log_signature_2d(x: &[f64], y: &[f64]) -> LogSigResult;
```

**Lines**: ~40.

---

### C11 — STL Decomposition (Seasonal-Trend-LOESS)

**Paper**: Cleveland et al. 1990, "STL: A Seasonal-Trend Decomposition Procedure Based on Loess", J. Official Statistics 6(1):3-73.
**Module**: `tambear/src/time_series.rs`
**Unlocks**: stl leaf

**Purpose**: Decompose y_t = T_t + S_t + R_t where T = trend, S = seasonal, R = residual.

**Algorithm** (simplified; full STL is iterative LOESS):
```
Outer loop (robust weights):
  Inner loop:
    1. Detrend: y_t - T_t
    2. Seasonal subseries smoothing (LOESS on each period's values)
    3. Low-pass filter on smoothed seasonal → extract smoothed trend component
    4. Update trend: LOESS on (y_t - S_t)
    5. Update seasonal: (y_t - T_t) minus low-pass
  Update robustness weights based on residuals
```

**Simplified version** (for fintek's "STL-like"): Use moving averages instead of LOESS:
```
1. Compute trend: centered moving average with window = period (e.g., 24 for hourly data)
2. Detrended = y - trend
3. Seasonal: mean of detrended at each phase of period
4. Residual: y - trend - seasonal
5. Features:
     trend_strength = 1 - var(residual)/var(y - seasonal)
     seasonal_strength = 1 - var(residual)/var(y - trend)
     residual_acf1 = lag-1 ACF of residual
     trend_slope = OLS slope on trend
```

**accumulate+gather**: Moving average = `accumulate(Windowed, Mean)`. Phase-mean = `accumulate(ByKey=phase, Mean)`. Pure Kingdom A for the simplified version.

**Signature**:
```rust
pub struct StlResult {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
    pub trend_strength: f64,
    pub seasonal_strength: f64,
    pub residual_acf1: f64,
    pub trend_slope: f64,
}

pub fn stl_decompose(
    data: &[f64], period: usize, use_loess: bool,
) -> StlResult;
```

**Lines**: ~60 (simplified version) or ~150 (full LOESS).

---

### C12 — SDE Drift/Diffusion via Nadaraya-Watson

**Paper**: Stanton 1997, "A nonparametric model of term structure dynamics and the market price of interest rate risk", J. Finance 52(5):1973-2002. Based on Nadaraya 1964 and Watson 1964 kernel regression.
**Module**: `tambear/src/stochastic.rs`
**Unlocks**: sde leaf

**Purpose**: Non-parametric estimation of drift μ(x) and diffusion σ²(x) from observed increments of an SDE dx = μ(x)dt + σ(x)dW.

**Formulas** (Stanton 1997, first-order approximations):
```
μ̂(x) = E[(x_{t+Δ} - x_t)/Δ | x_t = x]
σ̂²(x) = E[(x_{t+Δ} - x_t)²/Δ | x_t = x]
```

**Nadaraya-Watson estimator** with bandwidth h:
```
μ̂(x) = Σ_i K((x - x_i)/h) · (Δx_i / Δt) / Σ_i K((x - x_i)/h)
σ̂²(x) = Σ_i K((x - x_i)/h) · (Δx_i)²/Δt / Σ_i K((x - x_i)/h)
```

**Bandwidth** (Silverman's rule): h = 1.06 · σ_x · n^(-1/5).

**Features extracted**:
```
drift_mean = mean(μ̂(x)) over state space
drift_slope = OLS slope of μ̂(x) vs x (mean reversion if negative)
diffusion_mean = mean(σ̂²(x))
diffusion_slope = OLS slope of σ̂²(x) vs x (leverage if negative)
drift_diffusion_corr = correlation(μ̂, σ̂²) (leverage-like)
```

**accumulate+gather**: Kernel regression at each query point is a weighted sum — `accumulate(ByKey=query_point, KernelWeightedMean)`. Each query point independent → A+G.

**Signature**:
```rust
pub struct SdeResult {
    pub drift_values: Vec<f64>,      // μ̂ at query points
    pub diffusion_values: Vec<f64>,   // σ̂² at query points
    pub query_points: Vec<f64>,
    pub drift_mean: f64,
    pub drift_slope: f64,
    pub diffusion_mean: f64,
    pub diffusion_slope: f64,
    pub drift_diffusion_corr: f64,
}

pub fn nadaraya_watson_sde(
    data: &[f64], dt: f64,
    n_query: usize, bandwidth: Option<f64>,
) -> SdeResult;
```

**Lines**: ~40.

---

## Phyla: Shared Math Kernels Across Leaf Groups

A "phylum" is a group of fintek leaves that share the SAME core math kernel. Compiling ONE shader per phylum and parameterizing it covers multiple leaves at once. This is the highest-leverage compilation strategy.

### Phylum Φ1 — FFT of Log-Returns (15+ leaves)

**Kernel**: `fft(log_returns(prices))` — single rfft on a bin of log-transformed returns.

**Consumers**:
- fft_spectral (5 resolution variants: M8, M16, M32, M64, M128)
- welch (segmented + averaged FFTs)
- multitaper (FFT × tapers)
- cepstrum (IFFT of log |FFT|²)
- coherence (cross-FFT with a second series)
- fir_bandpass (FFT → band energies)
- energy_bands (same as fir_bandpass)
- wiener (noise from PSD tail)
- periodicity (peak of PSD)
- cwt_wavelet (FFT × Morlet in frequency domain)
- hilbert (FFT → zero negative freqs → IFFT)
- spectral_entropy (Shannon on normalized PSD)
- harmonic (SVD on Hankel, but FFT of |returns| is used for vol substrate)

**Compile strategy**: ONE shader emits `rfft(log_returns)` with the bin's padding/windowing. Every downstream leaf takes the complex FFT buffer as input. Per-bin FFT is cached in TamSession.

**Sharing tag**: `SpectralRepresentation { data_id: bin_id, n_points }`.

**Expected savings**: 15× reduction in FFT calls (ignoring the 5 fft_spectral variants which use different sizes).

---

### Phylum Φ2 — Raw Moments / MomentStats (15+ leaves)

**Kernel**: `moments_ungrouped(returns)` → `MomentStats { count, sum, min, max, m2, m3, m4 }`.

**Consumers**:
- distribution (mean, std, skew, kurt, realized_var)
- normality (Jarque-Bera, Shapiro-Wilk on raw)
- heavy_tail (needs tail statistics — partial consumer)
- shape (gradient stats from returns)
- variability (rolling CV of moments)
- phase_transition (magnetization = mean sign)
- seismic (n_extreme via std threshold)
- tail_field (via quantile binning)
- fisher_info (variance input)
- tick_complexity (partial — uses moments of inter-arrival)
- jump_test_bns (uses variance)
- scaling_triple (reads DFA/R-S outputs but uses moments for validation)

**Compile strategy**: ONE scatter-phi shader computes all 7 moment accumulators for each bin in a single pass. Every downstream leaf reads from the shared MomentStats.

**Sharing tag**: `MomentStats { data_id: bin_id }` (already exists).

**Expected savings**: 15× reduction in scatter passes.

---

### Phylum Φ3 — Phase-Space Distance Matrix (6 leaves)

**Kernel**: Delay-embedded distance matrix D[i,j] = ‖(x_i, x_{i+τ}, ..., x_{i+(m-1)τ}) - (x_j, ...)‖.

**Consumers**:
- sample_entropy (Chebyshev distance, template matching)
- correlation_dim (L2 distance, Grassberger-Procaccia)
- lyapunov (Rosenstein nearest neighbors)
- rqa (thresholded distance → recurrence matrix)
- poincare (return map from distance structure)
- embedding (FNN uses distance ratios at different m)

**Compile strategy**: ONE tiled distance shader parameterized by (m, tau, metric). All 6 leaves share the same n×n distance matrix per bin when they use the same (m, tau).

**Sharing tag**: `PhaseSpaceDistance { data_id: bin_id, m, tau, metric }` (new variant).

**Expected savings**: 6× reduction in distance matrix computations.

---

### Phylum Φ4 — ACF / Autocorrelation (6 leaves)

**Kernel**: `acf(returns, max_lag)` — autocorrelation up to lag k.

**Consumers**:
- autocorrelation (extracts 16 features from ACF + PACF)
- ar_model (Yule-Walker uses ACF)
- arma (Yule-Walker + residual ACF)
- arima (after differencing)
- dependence (Ljung-Box Q from ACF)
- periodicity (peak of ACF)

**Compile strategy**: ONE shader computes ACF via FFT (Wiener-Khinchin: FFT → |X|² → IFFT → ACF). This also shares with Φ1.

**Sharing tag**: `AutocorrelationFunction { data_id: bin_id, max_lag }` (already proposed in sharing-rules.md).

---

### Phylum Φ5 — OLS on Log-Log (3 leaves)

**Kernel**: OLS slope of log(F(s)) vs log(s) for varying scale s.

**Consumers**:
- dfa (log(F(s)) vs log(s), slope = α)
- hurst_rs (log(R/S) vs log(n), slope = H)
- mfdfa (multiple q values, each needs a slope)

**Compile strategy**: ONE OLS-on-log-log shader parameterized by the inner F computation. The slope extraction is the same across leaves.

**Sharing tag**: N/A (different F computations preclude direct sharing, but the OLS step is the same).

---

### Phylum Φ6 — GARCH σ² Recurrence (3 leaves)

**Kernel**: σ²_t = ω + α·r²_{t-1} + β·σ²_{t-1} (Särkkä Op prefix scan).

**Consumers**:
- garch (full GARCH fit with parameter estimation)
- vol_dynamics (rolling vol = √σ²)
- stochvol (AR(1) on log(σ²))

**Compile strategy**: Shared σ² series per bin. Tag `GarchVolatilitySeries { data_id, params }`.

---

### Phylum Φ7 — Sorted Returns / Order Statistics (5 leaves)

**Kernel**: Sorted copy of returns (or |returns|) for order-statistic-based computation.

**Consumers**:
- heavy_tail (Hill estimator on top-k)
- distribution (quantiles)
- tail_field (quintile binning)
- dist_distance (KS and Wasserstein need sorted samples)
- shapiro_wilk (ordered samples in nonparametric.rs)

**Compile strategy**: ONE sort shader per bin. Share sorted array via `SortedData { data_id, ascending }`.

---

### Phylum Φ8 — Delay Embedding (2 leaves)

**Kernel**: Reshape time series into `n-(m-1)τ` vectors of length m.

**Consumers**:
- ssa (trajectory matrix for SSA)
- pca (delay-embedded PCA)
- harmonic (Hankel matrix SVD)
- All Phase-Space consumers from Φ3

**Compile strategy**: ONE reshape kernel parameterized by (m, tau). Shared pointer.

---

### Phylum Φ9 — Binning / Histogram (6 leaves)

**Kernel**: Quantile-bin a series into k equal-mass bins.

**Consumers**:
- shannon_entropy (Shannon on bin counts)
- mutual_info (joint histogram)
- transfer_entropy (joint histogram)
- tick_complexity (entropy of binned inter-arrivals)
- tail_field (quintile of returns)
- edit_distance (symbolization)

**Compile strategy**: ONE quantile-binning shader. Share counts via `HistogramCounts { data_id, n_bins }`.

---

### Phylum Φ10 — Windowed Moments (4 leaves)

**Kernel**: Rolling mean and variance over sliding window.

**Consumers**:
- vol_dynamics (rolling std for vol-of-vol)
- vol_regime (short/long window variance ratio)
- variability (rolling var and mean CVs)
- vpin_bvc (rolling std for z-normalization)

**Compile strategy**: ONE sliding-window moment shader. Uses tambear's `accumulate(Windowed, Mean)`.

---

### Phyla Summary

| Phylum | Kernel | Leaves | Est. savings |
|--------|--------|--------|--------------|
| Φ1 | FFT of log-returns | 15 | 15× FFT calls |
| Φ2 | MomentStats | 15 | 15× scatter passes |
| Φ3 | Phase-space distance | 6 | 6× tiled distance |
| Φ4 | ACF | 6 | 6× (shares with Φ1 via FFT) |
| Φ5 | OLS log-log | 3 | 3× OLS fits |
| Φ6 | GARCH σ² | 3 | 3× GARCH recurrence |
| Φ7 | Sorted returns | 5 | 5× sort calls |
| Φ8 | Delay embedding | 2+ | Shared memory |
| Φ9 | Quantile binning | 6 | 6× histogram |
| Φ10 | Windowed moments | 4 | 4× rolling passes |

**Total coverage**: 65+ leaves (with overlap — many leaves consume multiple phyla). These 10 phyla cover the majority of Category A work.

**Implication for pathmaker**: Instead of compiling 126 separate shaders, compile 10 phylum kernels + 80 thin consumer adapters. This is a ~5-10× reduction in shader count.

**Sharing tag registry needed**:
- `PhaseSpaceDistance { data_id, m, tau, metric }` — new
- `HistogramCounts { data_id, n_bins }` — new
- `SortedData { data_id, ascending }` — new
- `GarchVolatilitySeries { data_id, omega, alpha, beta }` — new
- `WindowedMoments { data_id, window_size }` — new

Existing tags that support phyla:
- `SpectralRepresentation { data_id, n_points }` — Φ1
- `MomentStats { data_id }` — Φ2
- `AutocorrelationFunction { data_id, max_lag }` — Φ4

---

## Final Summary

**Specs written**: 20 detailed B-specs (B1-B20) + 12 detailed C-specs (C1-C12) = **32 tambear implementations specified**.

**Total lines of tambear code needed**: ~2,000 lines to unlock all 30 GAP leaves + all small-gap quality improvements.

**Phyla identified**: 10 shared-kernel groups covering 65+ of the 126 leaves.

**Category A + B + C coverage**:
- Category A (ready to compile): 80 leaves
- Category B (with full spec, 20 functions): 28 leaves → 20 tambear functions
- Category C (small gaps, with spec, 12 helpers): ~20 leaves improved/unlocked
- Meta leaves (read other outputs): 6 leaves

**Total**: 126 of 126 trunk-rs leaves have a path to compilation.

---

_Document is the formal mapping specification for the tambear-fintek bridge (Task #135)._
_Every fintek rescue is simultaneously a tambear math entry being checked off._
_Updated as new tambear primitives land and fintek leaves are migrated._
