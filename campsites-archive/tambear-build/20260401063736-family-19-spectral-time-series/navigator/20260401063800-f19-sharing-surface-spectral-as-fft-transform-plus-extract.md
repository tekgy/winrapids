# F19 Sharing Surface: Spectral Time Series as FFT Transform + Extraction

Created: 2026-04-01T06:38:00-05:00
By: navigator

Prerequisites: F17 complete (Affine scan for EWM/ARIMA baseline).

---

## Core Insight: FFT is a Transform, Not an Accumulate

F19 is the first family that genuinely breaks the `accumulate(grouping, expr, op)` model.
FFT is a **butterfly network** — O(n log n), recursive, specific memory access pattern
that cannot be expressed as a scatter/gather over a grouping.

**Classification: 4th Transform Category**

Alongside Sort (F08), EigenDecomp (F22), and Dijkstra (F28), FFT joins as a **transform**:
a structured O(n log n) or O(n²) pass that produces a fundamentally different representation.

| Transform | Input | Output | Cost |
|-----------|-------|--------|------|
| Sort | values | sorted permutation | O(n log n) |
| FFT | time domain | frequency domain | O(n log n) |
| EigenDecomp | symmetric matrix | eigenvalues + eigenvectors | O(n³) |
| Dijkstra | adjacency + source | shortest path distances | O(n² log n) |

The 8-operator model (accumulate primitives) covers ~33 of 35 families.
FFT is an honest exception, alongside TDA H₁ boundary matrix reduction (GF(2)).
Both are structural transforms with specific algorithmic structure — not new accumulators.

**Decision**: wrap cuFFT (CUDA) + rustfft (CPU).
Do NOT re-implement FFT. These libraries are battle-tested, plan-caching optimized.

---

## FFT Plan Cache in TamSession

FFT plans are expensive to compute (O(n) to O(n log n) depending on n's factorization)
but reusable for all transforms of the same size and type.

The plan cache IS the primary sharing surface for F19:

```rust
/// Key into the FFT plan cache
#[derive(Hash, Eq, PartialEq, Clone, Debug)]
pub struct FftPlanKey {
    pub n: usize,
    pub dtype: FftDtype,       // F32, F64, C32, C64
    pub direction: FftDirection, // Forward, Inverse
    pub normalized: FftNorm,   // None, Forward, Backward, Ortho
}

pub enum FftDtype { F32, F64, C32, C64 }
pub enum FftDirection { Forward, Inverse }
pub enum FftNorm { None, Forward, Backward, Ortho }

/// TamSession key for FFT plan cache
pub struct FftPlanCache;  // IntermediateTag

/// Stored in TamSession:
/// FftPlanCache → Arc<HashMap<FftPlanKey, Arc<dyn FftPlan>>>
```

**Every F19 algorithm** checks TamSession for an existing plan before creating one.
First call pays the plan creation cost. All subsequent calls on same-size data are free.

For financial time series (typical N = 250 trading days, 6.5h × 390 1-min bars = 97,500):
the plan for N=97500 is created once per session and reused across all spectral analyses.

---

## How All Spectral Algorithms Compose from FFT

### Core FFT Outputs

```
forward_fft(x) → X[k]      (complex, N/2+1 unique coefficients for real input)
|X[k]|²        → power[k]  (real, N/2+1)
angle(X[k])    → phase[k]  (real, N/2+1)
```

These two extractions (power and phase) produce the raw material for ALL spectral statistics.

### Periodogram (Raw PSD)

```
PSD[k] = |FFT(x)[k]|² / N
```

```
1. FFT(x) — plan cache lookup or create
2. |·|² element-wise — scatter_phi("x_re*x_re + x_im*x_im", ByAll) [Kingdom A]
3. / N — scalar divide
```

Total new code: ~15 lines (FFT call + extraction).
Gold standard: `scipy.signal.periodogram`.

### Welch PSD (Windowed Average)

```
1. Segment x into overlapping windows of length L (stride S)
2. Apply window function (Hann, Hamming) element-wise to each segment
3. FFT each segment
4. Average |FFT(segment)|² across segments
```

Windowing = **sliding window accumulate** — the same grouping as convolution (see F23 windowed accumulate), but Kingdom A (no sequential dependency).

```
1. Segment_indices: ByWindow(L, S) grouping — existing pattern from sliding stats
2. Window function: scatter_phi("x * hann(i)", ByWindow) — element-wise
3. FFT per window: batch FFT (cuFFT handles batched transforms natively)
4. |·|² → average across segments: scatter_phi("mag²", ByFrequency) with MeanOp
```

Gold standard: `scipy.signal.welch`. Default L=256, 50% overlap, Hann window.

### STFT / Spectrogram

STFT = sliding batch FFT without the averaging step.

```
output[t, k] = FFT(x[t*S : t*S+L] * window)[k]
```

Same as Welch Phase 1-3 but return the time×frequency matrix instead of averaging.
Output is complex — both power and phase are accessible.

New code beyond Welch: remove the averaging step (~5 lines).

### Autocorrelation via FFT (Wiener-Khinchin)

```
ACF[lag] = IFFT(|FFT(x)|²) / variance(x)
```

The FFT-based path is O(N log N) vs O(N²) for direct correlation.

```
1. FFT(x) → X[k]
2. |X[k]|² → power spectrum
3. IFFT(power) → autocorrelation (unnormalized)
4. / (N × variance) → normalized ACF
```

For large lag analysis (N > 10K), this path is 1000× faster than lag-by-lag accumulate.
For small lag analysis (lag ≤ 50), direct accumulate may be faster (fewer function calls).

TamSession: if `DistancePairs` already in cache (F01 ran), reuse for lag-0 ACF.

Gold standard: `statsmodels.tsa.stattools.acf`.

### Hilbert Transform / Instantaneous Amplitude

```
1. FFT(x) → X[k]
2. Phase mask: zero negative frequencies, double positive frequencies
   (or: H[k] = 0 if k>N/2, 2×X[k] if 0<k<N/2, X[k] if k=0 or k=N/2)
3. IFFT(H) → analytic signal z = x + i·Hilbert(x)
4. |z[t]| → instantaneous amplitude (envelope)
5. angle(z[t]) → instantaneous phase
```

New code: phase mask construction (~10 lines) + amplitude/phase extraction (~5 lines).
Gold standard: `scipy.signal.hilbert`.

### Cross-Spectrum / Coherence (Two Series)

```
cross_spectrum(x, y)[k] = FFT(x)[k] * conj(FFT(y)[k])
coherence[k] = |cross_spectrum|² / (PSD(x)[k] × PSD(y)[k])
```

Coherence = normalized cross-power, range [0,1]. For financial time series:
coherence between two stocks = frequency-resolved correlation (complement to cross-correlation F16/F17).

New code: ~20 lines (element-wise complex multiply + normalize).
Gold standard: `scipy.signal.coherence`.

### Wavelet Transform (CWT)

```
CWT[scale, time] = IFFT(FFT(x) × scale_factor × FFT(morlet_wavelet(scale)))
```

= batch FFT × wavelet filter bank × batch IFFT.
The wavelet filter bank can be precomputed and cached in TamSession alongside FFT plans.

```rust
pub struct WaveletFilterKey {
    pub n: usize,
    pub wavelet: WaveletType,   // Morlet, Ricker, Daubechies
    pub scales: Vec<f64>,       // requested scale range
}
/// WaveletFilterCache → Arc<HashMap<WaveletFilterKey, Arc<Vec<f64>>>>
```

Gold standard: `scipy.signal.cwt`, `pywt.cwt`.

### DWT (Discrete Wavelet Transform)

DWT uses a **filter bank** (low-pass + high-pass), NOT FFT.
Computation: convolution + downsampling at each level = F23's windowed accumulate (ConvOp).

```
level 0: approx = lowpass(x)[::2],  detail = highpass(x)[::2]
level 1: approx = lowpass(approx)[::2], ...
```

This is F23's Conv1d with stride 2, applied recursively.
DWT does NOT use FFT plans — it shares F23's infrastructure instead.

**TamSession bridge**: if F23's ConvFilterCache already populated (via F23 conv layers),
DWT uses the same cache. Cross-family sharing.

Gold standard: `pywt.dwt`, `pywt.wavedec`.

---

## Normalization Convention Trap (Scout Warning)

The biggest source of discrepancy with gold standards is FFT normalization:

| Library | FFT norm | IFFT norm | Product |
|---------|----------|-----------|---------|
| numpy (default) | 1 (none) | 1/N | 1/N |
| scipy.fft (ortho) | 1/√N | 1/√N | 1/N |
| MATLAB | 1 (none) | 1/N | 1/N |
| cuFFT | 1 (none) | 1 (none) | 1 |

**Rule**: store FFT normalization convention in `FftPlanKey.normalized`.
When comparing with scipy/numpy: always match their convention explicitly.
Default for F19 implementations: `FftNorm::None` (no normalization in forward pass,
apply 1/N in inverse pass). Document this explicitly in every gold standard test.

Failure mode: tests pass at N=1024 but fail at N=1025 (non-power-of-2 changes behavior
in some FFT implementations). Always test with N that is NOT a power of 2.

---

## MSR Types F19 Produces

```rust
pub struct PowerSpectrum {
    pub n_obs: usize,
    pub n_freq: usize,          // N/2+1 for real input
    pub freqs: Arc<Vec<f64>>,   // Hz (if sample_rate given) or normalized [0, 0.5]
    pub power: Arc<Vec<f64>>,   // PSD values, shape (n_freq,)
    pub method: PsdMethod,      // Periodogram, Welch { window_size, overlap, window_fn }
}

pub struct Spectrogram {
    pub n_time: usize,
    pub n_freq: usize,
    pub times: Arc<Vec<f64>>,   // center time of each window
    pub freqs: Arc<Vec<f64>>,
    pub power: Arc<Vec<f64>>,   // shape (n_time × n_freq), row-major time-first
}

pub struct WaveletTransform {
    pub n_obs: usize,
    pub n_scales: usize,
    pub scales: Arc<Vec<f64>>,
    pub coefficients: Arc<Vec<f64>>,  // shape (n_scales × n_obs), complex as (re, im) interleaved
    pub wavelet: WaveletType,
}

/// MSR for cross-spectral analysis (two series)
pub struct CrossSpectrum {
    pub n_freq: usize,
    pub freqs: Arc<Vec<f64>>,
    pub cross_power_re: Arc<Vec<f64>>,   // real part
    pub cross_power_im: Arc<Vec<f64>>,   // imaginary part
    pub coherence: Arc<Vec<f64>>,        // magnitude-squared coherence, [0,1]
    pub phase_angle: Arc<Vec<f64>>,      // cross-spectrum phase angle, radians
}
```

---

## Build Order

**Phase 1 (Periodogram + Welch + ACF)**:
1. `FftPlanKey` and `FftPlanCache` TamSession infrastructure (~30 lines)
2. `forward_fft()` and `inverse_fft()` wrapping rustfft (CPU) + cuFFT (CUDA) (~50 lines)
3. Periodogram: FFT → |·|² → scale (~20 lines)
4. Welch: batched FFT → average |·|² (~40 lines, reuse ByWindow grouping from F17)
5. ACF via Wiener-Khinchin: FFT → power → IFFT → normalize (~25 lines)
6. `PowerSpectrum` MSR type + normalization convention tests
7. Gold standards: `scipy.signal.periodogram`, `scipy.signal.welch`, `statsmodels.tsa.stattools.acf`

**Phase 2 (STFT + Hilbert)**:
1. STFT: Welch without averaging → `Spectrogram` type (~25 lines)
2. Hilbert: FFT → phase mask → IFFT → extract amplitude/phase (~20 lines)
3. Gold standards: `scipy.signal.stft`, `scipy.signal.hilbert`

**Phase 3 (CWT + Cross-spectrum)**:
1. CWT: morlet wavelet filter bank + batch FFT convolution (~60 lines)
2. `WaveletFilterCache` TamSession key
3. Cross-spectrum + coherence: two FFTs + element-wise complex multiply (~30 lines)
4. Gold standards: `scipy.signal.cwt`, `scipy.signal.coherence`

**Phase 4 (DWT)**:
1. DWT: recursive ConvOp (F23 prerequisite) + downsampling (~40 lines)
2. This phase depends on F23 ConvOp being implemented first

**Gold standards for normalization testing**:
- Always test with N = 100 (non-power-of-2) alongside N = 128 (power-of-2)
- Check: periodogram sums to variance (Parseval's theorem) — this catches normalization bugs

---

## Sharing Into and Out of TamSession

**Consumes from TamSession**:
- `MomentStats` (F06): if ACF requested AND variance already computed, reuse variance for normalization
- `DistancePairs` (F01): never directly, but temporal distance structure can inform window sizing

**Produces into TamSession**:
- `FftPlanCache`: all F19 algorithms populate and consume
- `WaveletFilterCache`: CWT filter bank, reusable across scales
- `PowerSpectrum`: can be produced once and consumed by spectral entropy (F08), frequency-domain KNN (F21), coherence (F19 cross-spectrum)

**Cross-family sharing**:
- F19 ACF → F16 (cross-correlation via spectral methods for large N)
- F19 STFT power → F06 spectral moments (MomentStats on frequency axis)
- F19 DWT → F23 ConvOp filter bank (both use sliding conv infrastructure)

---

## The Lab Notebook Claim

> FFT breaks the `accumulate` model and that's fine — it joins Sort, EigenDecomp, and Dijkstra as a **transform**: a structured pass with specific algorithmic requirements that don't fit the scatter/gather framework. The right answer is to wrap cuFFT (CUDA) and rustfft (CPU), not to invent a new accumulate operator for butterfly networks. The key TamSession contribution from F19 is the `FftPlanCache` — FFT plans are expensive to create but reusable for same-size inputs, making the cache the primary sharing surface. All spectral algorithms compose from FFT + element-wise extraction: Periodogram = |FFT|²/N, Welch = batched |FFT|² averaged, ACF = IFFT(|FFT|²), Hilbert = FFT + phase mask + IFFT, CWT = FFT × wavelet mask × IFFT. DWT is the exception — it uses F23's ConvOp filter bank, not FFT. The normalization convention trap (numpy vs scipy vs cuFFT) is the primary source of test failures; always anchor to an explicit norm convention and test with non-power-of-2 N.
