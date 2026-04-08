# F19 Spectral Time Series — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 19 (Spectral Time Series: FFT, PSD, spectrogram, wavelet, Hilbert).
Central architectural question: FFT is NOT a scan/accumulate — it's a butterfly network.
This is the first primitive in tambear's taxonomy that requires a different execution model.

---

## FFT: The Structural Break

All algorithms in families F01–F18, F20–F26 (with the exception of DFA's internal OLS) reduce
to the `accumulate(grouping, expr, op)` template. FFT does NOT.

FFT = Cooley-Tukey butterfly network: recursive, non-commutative in structure, O(n log n)
with a specific memory access pattern. It is NOT:
- A reduction (accumulate over all elements)
- A scan (running accumulate)
- A scatter (group-wise accumulate)

**Tambear architectural decision**: should tambear implement its own FFT, or wrap cuFFT?

**Recommendation**: wrap cuFFT (and fallback to RustFFT for CPU). Reasons:
1. cuFFT is mature, handles edge cases (non-power-of-2 via Bluestein, multi-dimensional)
2. FFT plans are reusable across calls on same-size input (major optimization)
3. Custom FFT would be ~2000 lines for a production-quality implementation
4. The custom-over-wrap principle from CLAUDE.md applies when translation layers exist —
   but cuFFT maps cleanly to `fft(x)` with no translation. Not the same as "just use what exists."

**For CPU backend**: `rustfft` crate (pure Rust, comparable performance to FFTW on modern CPUs).

**The key insight**: FFT computes the GramMatrix in frequency space.
`|X(f)|²` = power spectral density = how much variance lives at frequency f.
FFT and GramMatrix are two views of the same covariance structure — one in time domain,
one in frequency domain. F19 is not architecturally isolated from F10/F22.

---

## Python: scipy.fft — The Oracle

```python
from scipy.fft import fft, ifft, rfft, irfft, fftfreq, rfftfreq

# Forward FFT (complex output):
X = fft(x)               # x: length n → X: length n (complex)
X = rfft(x)              # real input → n//2+1 complex (symmetric, saves memory)

# Inverse:
x_back = ifft(X)
x_back = irfft(X, n=len(x))   # need original n for odd-length inputs

# Frequencies:
freqs = fftfreq(n, d=1/sampling_rate)   # returns both positive and negative freqs
freqs = rfftfreq(n, d=1/sampling_rate)  # only positive freqs (for rfft output)

# 2D FFT:
from scipy.fft import fft2, ifft2
X2d = fft2(image)

# Normalization options:
# norm="backward" (default): forward unscaled, inverse scaled by 1/n
# norm="ortho": both scaled by 1/sqrt(n) (unitary, symmetric)
# norm="forward": forward scaled by 1/n, inverse unscaled
```

```python
# numpy.fft (also valid oracle, identical results to scipy.fft):
import numpy.fft as np_fft
X = np_fft.rfft(x)
freqs = np_fft.rfftfreq(n, d=1/fs)
```

**Trap: normalization convention**. Default scipy is backward — the forward FFT is NOT normalized
(no 1/n factor), the inverse IS normalized. R uses the symmetric convention by default.

---

## Power Spectral Density (PSD)

### Periodogram (naive PSD)

```python
from scipy.signal import periodogram

f, Pxx = periodogram(x, fs=sampling_rate, window='boxcar', scaling='density')
# f: frequencies (0 to fs/2), Pxx: PSD in units/Hz
# scaling='density': PSD (power per Hz) — use for physical signals
# scaling='spectrum': power spectrum (sum over frequencies = total power)

# Equivalent manually:
X = np.fft.rfft(x)
Pxx = (np.abs(X)**2) / (fs * n)    # density
Pxx = (np.abs(X)**2) / n           # spectrum
```

**Trap**: periodogram is HIGH VARIANCE — wildly noisy estimate of true PSD. Never use
without windowing or smoothing. Welch's method is the standard.

### Welch's Method (gold standard for PSD)

```python
from scipy.signal import welch

f, Pxx = welch(x, fs=sampling_rate, window='hann', nperseg=256, noverlap=128,
               scaling='density', detrend='constant')

# Parameters:
# nperseg: segment length (controls freq resolution: Δf = fs/nperseg)
# noverlap: overlap between segments (typically 50% = nperseg//2)
# window: taper to reduce spectral leakage ('hann', 'hamming', 'blackman', etc.)
# detrend: remove trend per segment ('constant'=demean, 'linear'=detrend, False=none)
```

**Trade-off**: longer nperseg → better frequency resolution, fewer segments → higher variance.
This is the fundamental uncertainty principle of spectral analysis.

```r
# R: spec.welch doesn't exist in base; use signal package or manual:
library(signal)
Pxx <- pwelch(x, window=hanning(256), overlap=128, nfft=256, Fs=fs)
# OR:
spectrum(x, method="pgram", taper=0.1, spans=c(3,5))  # base R
```

### Multitaper (highest quality PSD)

```python
from spectrum import pmtm    # spectrum package
# OR:
from nitime.algorithms import multi_taper_psd    # nitime package
Pxx, nu, jackknife_var = pmtm(x, NW=4, k=7, NFFT=512)
# NW = time-bandwidth product (controls resolution-leakage tradeoff)
# k = number of tapers (usually 2*NW - 1)
```

```r
library(multitaper)
spec.mtm(ts(x, frequency=fs), nw=4, k=7)
```

---

## Spectrogram (Time-Frequency)

```python
from scipy.signal import spectrogram, stft, istft

# Spectrogram (squared magnitude of STFT):
f, t, Sxx = spectrogram(x, fs=sampling_rate, window='hann', nperseg=256,
                         noverlap=192, scaling='density')
# f: frequency bins, t: time bins, Sxx: 2D array (n_freq, n_time)

# STFT (Short-Time Fourier Transform, preserves phase):
f, t, Zxx = stft(x, fs=sampling_rate, window='hann', nperseg=256, noverlap=192)
# Zxx is complex: |Zxx|² = Sxx

# Inverse STFT:
_, x_rec = istft(Zxx, fs=sampling_rate, window='hann', nperseg=256, noverlap=192)
```

**Tambear path**: STFT = sliding window FFT = repeated FFT on overlapping windows.
GPU implementation: batch FFT on windowed segments. cuFFT supports batch FFT natively.
Same as `accumulate(Windowed(nperseg, stride=nperseg-noverlap), FFT)` — but FFT is not
a simple combine op. This is the only case where the accumulate framework doesn't apply.

---

## Autocorrelation via FFT

The cross-correlation theorem: autocorrelation is efficiently computed via FFT.

```python
# Autocorrelation via FFT (Wiener-Khinchin):
def autocorr_fft(x, max_lag=None):
    n = len(x)
    x_centered = x - x.mean()
    # Zero-pad to avoid circular correlation:
    f = np.fft.rfft(x_centered, n=2*n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]     # normalize by variance
    if max_lag: acf = acf[:max_lag]
    return acf

# Alternative: statsmodels
from statsmodels.tsa.stattools import acf, pacf
acf_vals = acf(x, nlags=40, fft=True)   # use FFT for efficiency
pacf_vals = pacf(x, nlags=40)           # via Yule-Walker
```

**Tambear key**: autocorrelation and cross-correlation are free given FFT.
`acf(lag k) = E[x_t · x_{t+k}]` = the off-diagonal of the GramMatrix in lag space.
FFT-based ACF is O(n log n) vs O(n²) for direct computation.

---

## Hilbert Transform

```python
from scipy.signal import hilbert

# Analytic signal = x + i·H(x) where H = Hilbert transform:
z = hilbert(x)
amplitude = np.abs(z)    # instantaneous amplitude (envelope)
phase = np.angle(z)      # instantaneous phase
freq = np.diff(np.unwrap(phase)) / (2*np.pi) * fs  # instantaneous frequency

# Hilbert transform via FFT:
# H(x) = IFFT(X(f) · (-i·sign(f)))
# i.e., zero negative frequencies, double positive, back-transform → real part
```

**Tambear**: Hilbert = FFT → multiply by phase mask → IFFT. Pure FFT composition.
No new primitives beyond FFT.

---

## Wavelet Transform

### Continuous Wavelet Transform (CWT)

```python
import pywt    # PyWavelets

# CWT (time-frequency at log-spaced scales):
scales = np.arange(1, 128)
cwtmatr, freqs = pywt.cwt(x, scales, 'morl', sampling_period=1/fs)
# cwtmatr: (n_scales, n_time) complex array
# 'morl' = Morlet wavelet, 'mexh' = Mexican hat, 'cmor' = complex Morlet

# Convert scales to frequencies:
freqs = pywt.scale2frequency('morl', scales, sampling_period=1/fs)
```

### Discrete Wavelet Transform (DWT)

```python
# Single-level DWT:
cA, cD = pywt.dwt(x, 'db4')     # 'db4' = Daubechies 4
x_rec = pywt.idwt(cA, cD, 'db4')

# Multi-level DWT decomposition:
coeffs = pywt.wavedec(x, 'db4', level=5)
# coeffs = [cA5, cD5, cD4, cD3, cD2, cD1]

# Reconstruction:
x_rec = pywt.waverec(coeffs, 'db4')
```

```r
library(wavelets)
wt <- modwt(x, filter="haar", n.levels=4)  # maximal overlap DWT
wt@W  # wavelet coefficients per level
wt@V  # scaling coefficients
```

**Wavelet families**: Haar (discontinuous), Daubechies (db2-db38, compact support),
Coiflets, Symlets, Mexican Hat, Morlet (complex). Each is a tradeoff in
smoothness vs compact support vs time-frequency resolution.

**Tambear note**: CWT at each scale is a convolution = multiplication in FFT domain.
CWT = batch FFT × wavelet filter mask × batch IFFT. Can reuse FFT primitive.
DWT (subband coding) is NOT naturally expressed as accumulate — it's a recursive
filter bank. CPU-side DWT is appropriate for most use cases.

---

## Spectral Coherence (Cross-Spectrum)

```python
from scipy.signal import coherence, csd

# Coherence between two signals:
f, Cxy = coherence(x, y, fs=sampling_rate, window='hann', nperseg=256)
# Cxy: values in [0,1], 1 = perfectly coherent

# Cross-spectral density:
f, Pxy = csd(x, y, fs=sampling_rate, window='hann', nperseg=256)
# Pxy: complex, magnitude = cross-PSD, phase = phase lag between x and y
```

---

## Key Spectral Algorithms for Market Data

### Power Spectral Density (market cycles)

Standard Welch PSD on price returns identifies dominant cycles.
IAT spectral analysis (from the fintek scout's notes) uses periodogram approach.

```python
# For OHLCV data at tick resolution:
returns = np.diff(np.log(prices))  # log returns
f, Pxx = welch(returns, fs=1.0, nperseg=min(len(returns), 512))
# fs=1.0 means frequency is in units of ticks
```

### Instantaneous Frequency (cycle detection)

```python
z = hilbert(returns)
inst_freq = np.diff(np.unwrap(np.angle(z))) / (2*np.pi)
# inst_freq: cycles per tick (rolling dominant frequency)
```

### Hurst from PSD (different from R/S method)

```python
# Hurst from spectral slope: H = (1 - β) / 2 where β = PSD slope in log-log
f, Pxx = welch(returns, fs=1.0, nperseg=256)
log_f = np.log(f[1:])   # skip DC component
log_Pxx = np.log(Pxx[1:])
beta = np.polyfit(log_f, log_Pxx, 1)[0]   # slope
H_spectral = (1 - beta) / 2
# = F10 OLS on log-transformed PSD (same as F26 multiscale template)
```

---

## Validation Targets

```python
import numpy as np
from scipy.signal import welch

np.random.seed(42)
fs = 100.0   # 100 Hz
t = np.arange(0, 10, 1/fs)   # 10 seconds

# Known signal: 10 Hz + 20 Hz sinusoids + noise
x = np.sin(2*np.pi*10*t) + 0.5*np.sin(2*np.pi*20*t) + 0.3*np.random.randn(len(t))

f, Pxx = welch(x, fs=fs, nperseg=256)
# Peak at f=10 Hz: Pxx ≈ 0.5 (half the variance of unit-amplitude sinusoid)
# Peak at f=20 Hz: Pxx ≈ 0.125 (half the variance of 0.5-amplitude sinusoid)

print("Max PSD frequency:", f[np.argmax(Pxx)])    # should be ~10 Hz
print("PSD at 10 Hz:", Pxx[np.argmin(np.abs(f - 10))])   # should be ~0.5
```

```python
# Phase consistency test:
X = np.fft.rfft(x)
x_back = np.fft.irfft(X, n=len(x))
assert np.allclose(x, x_back, atol=1e-10)   # FFT round-trip
```

---

## Tambear Decomposition Summary

| Algorithm | Primitive | Backend |
|-----------|----------|---------|
| FFT | cuFFT (CUDA) / rustfft (CPU) | External |
| Periodogram | FFT → \|·\|² → scale | F19 |
| Welch PSD | Windowed batch FFT → avg \|·\|² | F19 |
| STFT/spectrogram | Sliding batch FFT | F19 |
| Autocorrelation | FFT → multiply conj → IFFT | F19 |
| Hilbert | FFT → phase mask → IFFT | F19 |
| CWT | Batch FFT × wavelet mask × IFFT | F19 |
| DWT | Recursive filter bank | CPU |
| Spectral entropy | Welch PSD → Shannon entropy (F25) | F19 + F25 |
| Hurst (spectral) | Welch PSD → log-log OLS (F10) | F19 + F10 |

**Key insight**: FFT is the one primitive that doesn't fit accumulate(grouping, expr, op).
It deserves its own entry in tambear's primitive taxonomy alongside accumulate and gather.
Everything in F19 builds on top of FFT — none require new primitives beyond FFT itself.

**cuFFT integration path**:
- Add `fft(x: &TamBuffer, plan: &FftPlan) -> TamBuffer` to TamGpu trait
- CudaBackend: wraps cuFFT plan (cached by (n, dtype))
- CpuBackend: wraps rustfft
- `rfft`, `irfft`, `fft2` as surface methods
- Batch FFT: `fft_batch(segments: &[TamBuffer], plan: &FftPlan)` for Welch/STFT

---

## Infrastructure Gap: FFT Plan Cache

FFT plans are expensive to compute (O(n) precomputation of twiddle factors).
For market data, we repeatedly FFT the same segment length (e.g., always n=512).
The plan cache should be keyed by (n, dtype, direction) and live in TamSession.

```rust
// Proposed API sketch:
pub struct FftPlan {
    n: usize,
    dtype: DType,
    direction: FftDirection,
    // backend-specific plan handle (cuFFT cufftHandle or RustFFT Arc<dyn Fft<f32>>)
}

impl TamSession {
    pub fn get_or_create_fft_plan(&mut self, n: usize, dtype: DType, dir: FftDirection)
        -> Arc<FftPlan> { ... }
}
```

The plan cache is the main reason to wrap rather than reimplement — plan reuse is the
dominant optimization for repeated FFT on same-size inputs.
