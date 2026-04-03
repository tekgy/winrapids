# Family 19: Spectral Time Series — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: A (all methods reduce to FFT + accumulate) with transform preprocessing

---

## Core Insight: Every Spectral Method Is FFT + Accumulate

The entire spectral time series family reduces to:
1. **Window** the data (element-wise multiply)
2. **FFT** (F03 infrastructure)
3. **Accumulate** power/cross-spectral quantities

The power spectral density is `|FFT(x)|² / N` — squared magnitude of the Fourier transform. Every method in this family is a variation on how to estimate this quantity more stably, with higher resolution, or for irregular sampling.

**Structural rhyme**: Welch PSD = windowed scatter of periodograms (same `scatter_multi_phi` + `Add` as F06 mean of windowed statistics).

---

## 1. Periodogram

### Raw Periodogram
```
I(f_k) = (1/N) · |Σ_{t=0}^{N-1} x_t · exp(-2πi·k·t/N)|²
       = (1/N) · |DFT(x)[k]|²
```

at frequencies f_k = k/N for k = 0, 1, ..., N/2.

### Properties
- **Inconsistent estimator**: Var(I(f)) does NOT decrease with N. The periodogram is a NOISY estimate of the true PSD.
- **Expected value**: E[I(f)] ≈ S(f) (asymptotically unbiased)
- **Spectral leakage**: sharp features in S(f) spread to adjacent frequencies due to rectangular window

### Frequency Resolution
```
Δf = 1/(N·Δt)     where Δt = sampling interval
```
Nyquist frequency: f_max = 1/(2·Δt).

### GPU decomposition
- FFT: F03 (single call)
- Squared magnitude: `accumulate(Contiguous, |fft_k|², Identity)` — elementwise parallel
- Normalization: scalar divide

---

## 2. Welch's Method

### Algorithm (Welch 1967)
1. Divide signal into overlapping segments of length L with overlap D
2. Apply window to each segment
3. Compute periodogram of each segment
4. **Average** the periodograms

```
Ŝ_Welch(f) = (1/K) · Σ_{j=1}^K I_j(f)
```

where K = number of segments.

### Parameters

| Parameter | Typical | Effect |
|-----------|---------|--------|
| Segment length L | N/8 to N/2 | Longer = better resolution, fewer averages |
| Overlap D | L/2 (50%) | More overlap = more segments, diminishing returns >67% |
| Window | Hann | See window table below |

### Variance Reduction
```
Var(Ŝ_Welch) ≈ Var(I) / K_eff
```
where K_eff = effective number of independent segments (depends on overlap and window).

**Trade-off**: more segments (shorter L) → lower variance, worse frequency resolution.

### GPU decomposition
- Segment: gather with stride (parallel)
- Window: elementwise multiply (parallel)
- FFT per segment: batch FFT (F03, parallel across segments)
- Squared magnitude: elementwise (parallel)
- Average: `accumulate(All, per_segment_periodogram, Add)` / K — reduce

**Key parallelism**: ALL segments can be FFT'd simultaneously. This is embarrassingly parallel.

---

## 3. Multitaper Method (Thomson 1982)

### Algorithm
Instead of one window, use K orthogonal tapers (DPSS — Discrete Prolate Spheroidal Sequences):
```
Ŝ_MT(f) = Σ_{k=0}^{K-1} w_k · |Σ_t h_k(t) · x_t · exp(-2πift)|²
```

### DPSS Tapers (Slepian Sequences)
Solve the eigenvalue problem:
```
Σ_s D(t,s) · v_k(s) = λ_k · v_k(t)
```
where D(t,s) = sin(2πW(t-s)) / (π(t-s)), W = half-bandwidth.

First K = ⌊2NW⌋ - 1 tapers have eigenvalue λ_k ≈ 1. These are the useful tapers.

### Half-Bandwidth (NW)
- **NW = 2**: 3 tapers, low smoothing (for sharp peaks)
- **NW = 4**: 7 tapers, moderate smoothing (default, general purpose)
- **NW = 8**: 15 tapers, heavy smoothing (for broadband estimation)

### CRITICAL: DPSS Computation
Computing DPSS tapers requires solving a symmetric tridiagonal eigenvalue problem (F02). For fixed N and NW, precompute once and reuse.

### Adaptive Weighting (Thomson 1982)
Optimal weights w_k minimize broadband bias:
```
w_k(f) = √λ_k · Ŝ(f) / (λ_k · Ŝ(f) + (1-λ_k) · σ²)
```
Iterate until convergence (typically 2-3 iterations).

### Advantages over Welch
- Uses the FULL data (no segmentation → no resolution loss)
- Optimal bias-variance trade-off (provably)
- Better for short series
- Can compute confidence intervals analytically

### GPU decomposition
- DPSS: precompute (F02 eigensolve, done once)
- Tapered FFTs: K independent FFTs (batch, parallel)
- Eigenspectra: `accumulate(Contiguous, |tapered_fft_k|², Identity)` per taper
- Weighted average: `accumulate(All, w_k · eigenspectrum_k, Add)` — reduce across tapers

---

## 4. AR Spectral Estimation (Burg PSD)

### Idea
Fit an AR(p) model (F17), then compute the theoretical PSD:
```
S_AR(f) = σ²_ε / |1 - Σ_{k=1}^p φ_k · exp(-2πifk)|²
```

### Burg's Method
Estimates AR coefficients by minimizing forward + backward prediction error simultaneously. Lattice filter — each order adds one reflection coefficient.

### Advantages
- Very high frequency resolution for short series
- Smooth spectrum (no leakage)
- Can extrapolate beyond observed frequency range

### Disadvantages
- Model-dependent (assumes AR is correct)
- AR order selection affects results (use AIC/BIC from F17)
- Can produce spurious peaks if order too high

### GPU decomposition
- AR estimation: F17 Burg method
- PSD evaluation: evaluate |1 - Σφ_k·z^(-k)|² at frequency grid points — parallel per frequency
- Same as polynomial evaluation on unit circle (F36 Horner on complex numbers)

---

## 5. Lomb-Scargle Periodogram (Irregular Sampling)

### Problem
Standard FFT requires uniform sampling. Astronomical, geophysical, and some financial data are irregularly sampled.

### Formula (Lomb 1976, Scargle 1982)
```
P_LS(f) = (1/2) · [(Σ x_i cos(ω(t_i-τ)))² / Σ cos²(ω(t_i-τ)) + (Σ x_i sin(ω(t_i-τ)))² / Σ sin²(ω(t_i-τ))]
```

where τ is a time offset that makes the formula equivalent to least-squares fit of sinusoid.

### τ Computation
```
tan(2ωτ) = Σ sin(2ωt_i) / Σ cos(2ωt_i)
```

### Fast Lomb-Scargle (Press & Rybicki 1989)
Use NUFFT (Non-Uniform FFT) to compute all frequencies at once. O(N log N) instead of O(N·N_freq).

### Significance (False Alarm Probability)
```
FAP ≈ 1 - (1 - exp(-P_LS))^M
```
where M = number of independent frequencies ≈ N/2.

**Baluev (2008) analytic FAP** is more accurate for large P_LS.

### GPU decomposition
- Per frequency: trig evaluations + accumulate. Parallel across frequencies.
- NUFFT: F03 infrastructure (Finufft-style spreading + FFT)

---

## 6. Cross-Spectral Analysis

### Cross-Spectral Density
```
S_xy(f) = (1/N) · DFT(x)[f] · conj(DFT(y)[f])
```

Complex-valued: amplitude tells co-variation strength, phase tells time lag.

### Coherence
```
C_xy(f) = |S_xy(f)|² / (S_xx(f) · S_yy(f))
```

Ranges [0, 1]. Analogous to R² at each frequency. C = 1 means perfectly linearly related at frequency f.

### Phase Spectrum
```
φ_xy(f) = arg(S_xy(f)) = atan2(Im(S_xy), Re(S_xy))
```

Time delay at frequency f: Δt(f) = φ_xy(f) / (2πf).

### Transfer Function (Frequency Response)
```
H(f) = S_xy(f) / S_xx(f)
```

Gain: |H(f)|. Phase: arg(H(f)).

### CRITICAL: Raw cross-spectra are noisy (same problem as raw periodogram). Smooth using Welch or multitaper before computing coherence. Without smoothing, coherence ≡ 1 at every frequency (trivially).

### GPU decomposition
- Two FFTs: parallel (batch FFT of x and y)
- Cross-product: `accumulate(Contiguous, fft_x * conj(fft_y), Identity)` — elementwise
- Smoothing: Welch segments or multitaper (same as univariate case)
- Coherence: elementwise division of smoothed spectra

---

## 7. Spectral Entropy

### Definition
```
H_spectral = -Σ_k p_k · log(p_k)
```
where p_k = S(f_k) / Σ_j S(f_j) (normalized PSD as probability distribution).

High entropy = flat spectrum (white noise). Low entropy = concentrated spectrum (periodic signal).

### GPU: F25 Shannon entropy applied to normalized PSD. No new primitives.

---

## 8. Energy Band Decomposition

### Definition
Power in frequency band [f_a, f_b]:
```
P_band = Σ_{f_a ≤ f_k ≤ f_b} S(f_k) · Δf
```

### Standard Bands (EEG example)

| Band | Range | Use |
|------|-------|-----|
| Delta | 0.5-4 Hz | Sleep |
| Theta | 4-8 Hz | Memory |
| Alpha | 8-13 Hz | Relaxation |
| Beta | 13-30 Hz | Active thinking |
| Gamma | 30-100 Hz | Perception |

### Financial Bands
```
Ultra-low: < 1/day
Low: 1/day to 1/hour
Mid: 1/hour to 1/minute
High: 1/minute to 1/second
Ultra-high: > 1/second
```

### GPU decomposition
- PSD: computed above
- Band power: `accumulate(Segmented(band_boundaries), S_k · Δf, Add)` — segmented reduce
- Band ratio: elementwise division of band powers

---

## 9. Window Functions

| Window | Sidelobe Level | Main Lobe Width | Use case |
|--------|---------------|----------------|----------|
| Rectangular | -13 dB | Narrowest | Maximum resolution |
| Hann | -32 dB | 1.5× rect | General purpose |
| Hamming | -43 dB | 1.4× rect | Good sidelobe suppression |
| Blackman | -58 dB | 1.7× rect | Low leakage |
| Kaiser(β) | tunable | tunable | Adjustable trade-off |
| DPSS | optimal | set by NW | Multitaper |

**Decision**: Default to Hann for Welch, DPSS for multitaper. Provide all standard windows.

### Window Correction Factors
```
Amplitude correction: S₁ = Σ w_t             (compensate for amplitude reduction)
Power correction:     S₂ = Σ w_t²            (compensate for power reduction)
ENBW = N · S₂ / S₁²                           (effective noise bandwidth)
```

---

## 10. Numerical Stability

### FFT Precision
- f32 FFT: sufficient for most spectral estimation (PSD is already an estimate)
- f64 FFT: needed when dynamic range > 120 dB or for calibration
- **Welch averaging reduces noise, so f32 segments averaged in f64 is a good strategy**

### Log-Spectrum
When PSD spans many orders of magnitude, work in log-space:
```
log S(f) = 2 · log|FFT(x)| - log(N)
```

### Leakage at DC
f=0 component contains mean of signal. **Remove mean before FFT** unless DC component is of interest.

### Aliasing
Frequencies above Nyquist fold back. **Low-pass filter before spectral estimation** if high-frequency content exists above f_max.

---

## 11. Edge Cases

| Algorithm | Edge Case | Expected |
|-----------|----------|----------|
| Periodogram | N = 1 | Single frequency bin. Not meaningful. |
| Periodogram | Constant signal | S(f) = 0 for f > 0, S(0) = mean². |
| Welch | L > N | Only one segment. Degenerate to raw periodogram. |
| Welch | Overlap > L | Invalid. Error. |
| Multitaper | NW < 1 | Only 1 usable taper. Degenerate to windowed periodogram. |
| Lomb-Scargle | All times equal | Cannot estimate. Error. |
| Lomb-Scargle | 2 data points | 0 independent frequencies. Insufficient data. |
| Coherence | Unsmoothed | Coherence ≡ 1 trivially. Warn user. |
| Cross-spectrum | Different length x,y | Truncate to shorter. Warn. |
| Spectral entropy | Flat spectrum (white noise) | H = log(N/2) (maximum). Correct. |
| Band power | Band outside [0, f_Nyquist] | Clamp to valid range. Warn. |

---

## Sharing Surface

### Reuses from Other Families
- **F03 (Signal Processing)**: FFT (the core computation), window functions, NUFFT
- **F17 (Time Series)**: AR models for Burg PSD estimation
- **F06 (Descriptive)**: mean removal, variance for normalization
- **F25 (Information Theory)**: Shannon entropy for spectral entropy
- **F01 (Distance)**: spectral distance metrics (Itakura-Saito, log-spectral)

### Provides to Other Families
- **F17 (Time Series)**: spectral domain diagnostics (check for periodicity, frequency content)
- **F18 (Volatility)**: spectral analysis of returns (power-law scaling = long memory)
- **F26 (Complexity)**: spectral entropy as complexity measure, DFA spectral interpretation
- **F03 (Signal)**: coherence for multi-channel signal analysis

### Structural Rhymes
- **Welch = windowed mean of periodograms**: same as F06 rolling_mean but in frequency domain
- **Multitaper = weighted average of orthogonal views**: same philosophy as F37 superposition (multiple views, one cost)
- **AR PSD = polynomial evaluation on unit circle**: same as F36 polynomial evaluation
- **Coherence = R² per frequency**: same as F07/F10 coefficient of determination

---

## Implementation Priority

**Phase 1** — Core PSD estimation (~100 lines):
1. Raw periodogram (FFT + |·|²)
2. Welch's method (segmented, windowed, averaged)
3. Standard window functions (Hann, Hamming, Blackman, Kaiser)
4. Frequency axis computation (Hz, normalized, angular)

**Phase 2** — Advanced estimation (~120 lines):
5. Multitaper (DPSS taper computation + eigenspectral average)
6. AR PSD (Burg method + spectral evaluation, wraps F17)
7. Lomb-Scargle periodogram (for irregular data)
8. Window correction factors (ENBW, amplitude/power correction)

**Phase 3** — Cross-spectral (~100 lines):
9. Cross-spectral density
10. Coherence (magnitude-squared)
11. Phase spectrum + group delay
12. Transfer function estimation

**Phase 4** — Derived quantities (~80 lines):
13. Spectral entropy (wraps F25)
14. Energy band decomposition (segmented reduce)
15. Spectral peaks detection (local maxima + significance)
16. Cepstrum (log spectrum → IFFT)

---

## Composability Contract

```toml
[family_19]
name = "Spectral Time Series"
kingdom = "A (FFT + accumulate — all methods reduce to frequency-domain operations)"

[family_19.shared_primitives]
periodogram = "PSD via |FFT(x)|² / N"
welch = "Averaged windowed periodograms"
multitaper = "DPSS-weighted eigenspectral average"
cross_spectral = "FFT(x) · conj(FFT(y)) → coherence, phase"
lomb_scargle = "Least-squares spectral estimation for irregular sampling"

[family_19.reuses]
f03_signal_processing = "FFT, NUFFT, window functions"
f17_time_series = "AR model fitting for Burg PSD"
f06_descriptive = "Mean removal, variance normalization"
f25_information_theory = "Shannon entropy for spectral entropy"

[family_19.provides]
psd = "Power spectral density estimate (Welch, multitaper, AR, Lomb-Scargle)"
coherence = "Frequency-domain correlation between two signals"
phase = "Phase relationship and time delay between signals"
band_power = "Integrated power in frequency bands"
spectral_entropy = "Entropy of normalized PSD"

[family_19.consumers]
f17_time_series = "Spectral diagnostics for time series models"
f18_volatility = "Spectral power-law for long-memory detection"
f26_complexity = "Spectral entropy as complexity measure"
fintek = "Financial signal frequency content analysis"
```
