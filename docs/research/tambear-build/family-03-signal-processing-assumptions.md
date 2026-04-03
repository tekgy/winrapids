# Family 03: Signal Processing — Mathematical Assumptions Document

**Author**: Math Researcher
**Date**: 2026-04-01
**Status**: Pre-implementation reference. Read this BEFORE coding.
**Kingdom**: Mixed — A (FFT butterfly, spectral accumulates), B (IIR/Kalman = Affine scan), embarrassingly parallel (FIR, CWT, STFT evaluation)

---

## Core Insight: FFT is the Shared Primitive

Nearly everything in this family touches the FFT at some point:
- Convolution → FFT multiply IFFT
- FIR filtering → FFT-based for long filters
- Power spectrum → |FFT|²
- DCT/DST → reindexed FFT
- Hilbert transform → FFT → zero negative frequencies → IFFT
- Cepstrum → log(|FFT|²) → IFFT
- CWT at fixed scale → convolution → FFT
- STFT → windowed FFT

**Build FFT once → feed 15+ algorithms.**

---

## 1. Discrete Fourier Transform (DFT) / FFT

### Definition
```
X[k] = Σ_{n=0}^{N-1} x[n] · e^{-j2πkn/N},    k = 0, 1, ..., N-1
```

Inverse:
```
x[n] = (1/N) Σ_{k=0}^{N-1} X[k] · e^{j2πkn/N}
```

### FFT Algorithms

**Cooley-Tukey Radix-2** (N = 2^m):
Split into even/odd indices, recurse:
```
X[k] = E[k] + W_N^k · O[k]
X[k + N/2] = E[k] - W_N^k · O[k]
```
where E[k] = DFT of even samples, O[k] = DFT of odd samples, W_N = e^{-j2π/N}.

This is the **butterfly** operation. Each stage has N/2 butterflies, log₂(N) stages. Total: O(N log N).

**Cooley-Tukey Radix-4** (N = 4^m):
Four-way split. Fewer multiplications (3/4 of radix-2). Same O(N log N) but better constant.

**Split-Radix** (N = 2^m):
Combines radix-2 for even indices and radix-4 for odd indices. Achieves 4N log₂N - 6N + 8 real multiplications — lowest known for power-of-2 DFT.

**Bluestein's Algorithm** (arbitrary N):
Converts arbitrary-N DFT to a convolution of length ≥ 2N-1 (round up to power of 2):
```
X[k] = W_N^{k²/2} · Σ_n [x[n]·W_N^{n²/2}] · W_N^{-(k-n)²/2}
```
This is a correlation (≈ convolution) which can be computed via 3 FFTs of size M ≥ 2N-1 (power of 2). Supports ANY N without zero-padding to power of 2.

**RFFT** (Real-Valued Input):
Exploit conjugate symmetry: X[N-k] = X[k]*. Pack two real transforms into one complex transform, or use specialized real-valued butterflies. Output: N/2+1 complex values.

### GPU Strategy
- **Per-stage parallelism**: Each butterfly stage is embarrassingly parallel (N/2 independent operations)
- **Memory access**: Bit-reversal permutation is the bottleneck (non-coalesced). Use shared memory for each stage.
- **Stockham auto-sort FFT**: Avoids explicit bit reversal by alternating between two buffers. Better GPU memory access pattern than Cooley-Tukey.

### Edge Cases
- N = 0 or N = 1: trivial (identity)
- N not power of 2: Bluestein or mixed-radix (factor N, apply Cooley-Tukey per factor)
- Very large N: out-of-core FFT (split into blocks, FFT each, twiddle, FFT columns)
- Numerical precision: twiddle factors W_N^k accumulate error. Precompute from sin/cos tables, not recursive multiplication.

### Kingdom: A (each butterfly stage is a parallel map; the composition of stages is a specific DAG, not a sequential scan)

---

## 2. Short-Time Fourier Transform (STFT)

### Definition
```
STFT{x}(m, k) = Σ_{n=0}^{L-1} x[n + m·H] · w[n] · e^{-j2πkn/L}
```
where w[n] = window of length L, H = hop size, m = frame index.

### Parameters
- **Window**: Hann, Hamming, Blackman, Kaiser — trade-off between main lobe width and side lobe suppression
- **FFT size** (NFFT): ≥ L, zero-pad if NFFT > L for frequency interpolation
- **Hop size** H: overlap = L - H. For perfect reconstruction: H ≤ L/4 (75% overlap) with Hann window
- **Output**: complex spectrogram, shape (NFFT/2+1) × num_frames

### Inverse STFT (Griffin-Lim for magnitude-only)
With proper overlap-add:
```
x[n] = Σ_m ISTFT_frame[m, n] · w[n - mH] / Σ_m w²[n - mH]
```

### GPU: Embarrassingly parallel across frames. Each frame = one FFT.

### Edge Cases
- Short signal (< L): zero-pad
- Non-COLA windows: reconstruction not perfect
- Phase unwrapping for instantaneous frequency: Δφ = angle(X[m+1,k]) - angle(X[m,k]) - 2πkH/N, wrapped to [-π,π]

---

## 3. Non-Uniform FFT (NUFFT)

### Problem
Evaluate DFT at non-uniformly spaced points (Type 1: non-uniform input, uniform output; Type 2: uniform input, non-uniform output; Type 3: both non-uniform).

### Algorithm (Greengard-Lee)
1. Spread non-uniform points to oversampled uniform grid via Gaussian/Kaiser-Bessel kernel
2. FFT on oversampled grid
3. Deconvolve (divide by kernel's Fourier transform)

### Complexity: O(N log N + M·w) where w = kernel width (typically 6-12)

### Relevance: Financial data has non-uniform timestamps. NUFFT avoids resampling artifacts.

---

## 4. DCT (Discrete Cosine Transform)

### DCT-II (the "DCT" — used in JPEG, MPEG, ML)
```
X[k] = Σ_{n=0}^{N-1} x[n] · cos(π(2n+1)k / (2N))
```

### Relation to FFT
DCT-II of length N = real part of a DFT of length 4N applied to a symmetrically extended sequence. In practice: reorder x, compute RFFT, extract.

### All 8 Types
| Type | Boundary | Use |
|------|----------|-----|
| DCT-I | Symmetric at both endpoints | Chebyshev polynomials |
| **DCT-II** | Symmetric at n=-½, N-½ | Standard "DCT", compression |
| DCT-III | Inverse of DCT-II | Synthesis |
| DCT-IV | Symmetric at n=-½ | MDCT basis |

### CRITICAL: DCT-II via FFT saves 2x over direct computation. Reindex: y[n] = x[2n] for n<N/2, y[n] = x[2(N-n)-1] for n≥N/2, then FFT(y) with twiddle.

### DST: Same but with sine. DST-II: X[k] = Σ x[n]·sin(π(2n+1)(k+1)/(2N))

### NTT (Number Theoretic Transform)
FFT over finite field Z_p (p prime, p = k·2^m + 1). Used for: exact integer convolution, polynomial multiplication, cryptography. Same butterfly structure as FFT.

---

## 5. Wavelets

### 5a. DWT (Discrete Wavelet Transform)

**Filter bank implementation** (Mallat's algorithm):
```
Approximation: a[j+1][n] = Σ_k h[k-2n] · a[j][k]    (lowpass + downsample by 2)
Detail:        d[j+1][n] = Σ_k g[k-2n] · a[j][k]    (highpass + downsample by 2)
```
where h = lowpass filter, g = highpass filter (QMF: g[n] = (-1)^n h[1-n]).

Repeat for J levels. Output: {d[1], d[2], ..., d[J], a[J]}.

**Computational complexity**: O(NL) per level where L = filter length. Total O(NLJ) ≈ O(N) for fixed J, L.

### Filter Coefficients (Selected)
| Wavelet | h coefficients | Vanishing moments |
|---------|---------------|-------------------|
| Haar (db1) | [1/√2, 1/√2] | 1 |
| db2 | [0.4830, 0.8365, 0.2241, -0.1294] | 2 |
| db4 | 8 coefficients | 4 |
| db20 | 40 coefficients | 20 |

**CRITICAL**: Daubechies coefficients are irrational. Use published tables to full precision (at least float64). Do NOT recompute — the spectral factorization is numerically sensitive.

### 5b. CWT (Continuous Wavelet Transform)
```
W(a, b) = (1/√a) ∫ x(t) · ψ*((t-b)/a) dt
```
Discretized: for each scale a and position b, inner product with scaled/shifted wavelet.

**GPU**: Embarrassingly parallel across (a, b) pairs. For each scale, this is a convolution → use FFT.

**Morlet wavelet**: ψ(t) = π^(-1/4) · e^{jω₀t} · e^{-t²/2}, ω₀ = 6 (standard)
**Mexican hat**: ψ(t) = (2/√3) · π^(-1/4) · (1-t²) · e^{-t²/2}

### 5c. SWT (Stationary/Undecimated Wavelet Transform)
Same as DWT but WITHOUT downsampling. Instead, upsample the filters at each level (insert zeros between coefficients). Output has same length as input at every level.

**Advantage**: Translation-invariant (DWT is not)
**Disadvantage**: O(N·J) storage vs O(N) for DWT

### 5d. Lifting Scheme (Sweldens 1996)
Factorize wavelet filter bank into predict/update/normalize steps:
```
1. Split: even = x[2n], odd = x[2n+1]
2. Predict: d[n] = odd[n] - P(even)      (detail = residual from prediction)
3. Update: a[n] = even[n] + U(detail)     (smooth = even + correction from detail)
```

**Advantage**: In-place computation (no extra memory), easier to design custom wavelets, invertible by construction.

### Kingdom: DWT = B (sequential across levels, parallel within level). CWT at fixed scale = A (convolution). Lifting = B.

---

## 6. Digital Filters

### 6a. FIR (Finite Impulse Response)
```
y[n] = Σ_{k=0}^{M} b[k] · x[n-k]
```

This is convolution → O(NM) direct, O(N log N) via FFT overlap-save/overlap-add.

**Design Methods**:
- **Window method**: Ideal impulse response h_d[n] = sin(ωc·n)/(πn) windowed by w[n]
- **Parks-McClellan (Remez)**: Minimax optimal equiripple. Iterative algorithm (Kingdom C). Produces filter of minimum order for given specs.
- **Least squares**: Minimize ∫|H(ω) - H_d(ω)|² (closed form via sinc integrals)

**GPU**: Direct FIR = parallel across output samples (each is a dot product). FIR = tiled accumulate of lagged products.

### 6b. IIR (Infinite Impulse Response)
```
y[n] = Σ_{k=0}^{M} b[k]·x[n-k] - Σ_{k=1}^{P} a[k]·y[n-k]
```

**CRITICAL**: IIR is INHERENTLY SEQUENTIAL — y[n] depends on y[n-1], ..., y[n-P].

This is an **Affine scan** (Kingdom B):
State = [y[n-1], ..., y[n-P]], input = x[n]:
```
y[n] = b₀·x[n] + Σb_k·x[n-k] - Σa_k·y[n-k]
```
The y-dependency makes it a P-dimensional affine recurrence.

### IIR Design (Analog Prototype → Digital)

**Butterworth**: Maximally flat magnitude. Poles on unit circle equally spaced:
```
|H(jΩ)|² = 1 / (1 + (Ω/Ωc)^{2N})
```

**Chebyshev Type I**: Equiripple in passband, monotone in stopband:
```
|H(jΩ)|² = 1 / (1 + ε²·T_N²(Ω/Ωc))
```
where T_N = Chebyshev polynomial of degree N.

**Chebyshev Type II**: Monotone in passband, equiripple in stopband.

**Elliptic (Cauer)**: Equiripple in both bands. Minimum order for given specs. Uses Jacobi elliptic functions.

**Bilinear Transform**: Maps analog s-plane to digital z-plane:
```
s = 2/T · (z-1)/(z+1)
```
with frequency warping: ω_d = 2·arctan(ω_a·T/2). Must pre-warp critical frequencies.

### 6c. Savitzky-Golay Filter
Fit polynomial of degree p to window of 2m+1 points, evaluate at center. Coefficients from least-squares: C = (A'A)⁻¹A' where A is Vandermonde matrix.

**Key**: Coefficients depend ONLY on window size and polynomial degree, not data. Precompute once. Then it's FIR: y[n] = Σ c[k]·x[n+k].

### Kingdom: FIR = A (parallel dot products). IIR = B (affine scan). Savgol = A after precomputing coefficients.

---

## 7. Kalman Filter

### State-Space Model
```
State:       x_{t+1} = F·x_t + B·u_t + w_t,    w_t ~ N(0, Q)
Observation: z_t = H·x_t + v_t,                  v_t ~ N(0, R)
```

### Predict Step
```
x̂_{t|t-1} = F·x̂_{t-1|t-1} + B·u_t
P_{t|t-1} = F·P_{t-1|t-1}·F' + Q
```

### Update Step
```
ỹ_t = z_t - H·x̂_{t|t-1}                        (innovation)
S_t = H·P_{t|t-1}·H' + R                        (innovation covariance)
K_t = P_{t|t-1}·H'·S_t⁻¹                        (Kalman gain)
x̂_{t|t} = x̂_{t|t-1} + K_t·ỹ_t
P_{t|t} = (I - K_t·H)·P_{t|t-1}
```

### As Affine Scan
State = [x̂, P] (vectorized). The predict/update cycle for constant F, H, Q, R is:
```
[x̂_new, P_new] = f(x̂_old, P_old, z_t)
```
This is NOT a simple affine map because K depends on P which depends on previous P. However, for constant system matrices, P converges to steady-state P∞ and K converges to K∞. In steady-state: the x̂ update IS Affine.

### Extended Kalman Filter (EKF)
Linearize nonlinear dynamics: F_t = ∂f/∂x|_{x̂_t}, H_t = ∂h/∂x|_{x̂_t}. Same equations with time-varying Jacobians.

### Unscented Kalman Filter (UKF)
Generate 2p+1 sigma points, propagate through nonlinear function, reconstruct statistics:
```
χ₀ = x̂,  χᵢ = x̂ + √((p+λ)P)_i,  χ_{i+p} = x̂ - √((p+λ)P)_i
```
where λ = α²(p+κ) - p (typically α=10⁻³, κ=0, β=2).

### Edge Cases
- P becomes non-positive-definite: use Joseph form P = (I-KH)P(I-KH)' + KRK' or square-root filter (Cholesky of P)
- Singular S: observation is perfectly predicted → K = 0 → skip update
- Large state dimension: ensemble Kalman filter (EnKF) for p > ~100

### Kingdom: B (sequential scan — each step depends on previous state). Steady-state Kalman = Affine(p,p) scan of observations.

---

## 8. Hilbert Transform & Analytic Signal

### Hilbert Transform
```
H{x}(t) = (1/π) PV ∫ x(τ)/(t-τ) dτ
```

In frequency domain: H{x}[k] = -j·sgn(k)·X[k] (multiply by -j for positive frequencies, +j for negative).

### Analytic Signal
```
z(t) = x(t) + j·H{x}(t)
```

Instantaneous amplitude: A(t) = |z(t)|
Instantaneous phase: φ(t) = arg(z(t))
Instantaneous frequency: f(t) = (1/2π)·dφ/dt

### Implementation via FFT
1. X = FFT(x)
2. X[0] unchanged, X[1..N/2-1] *= 2, X[N/2] unchanged, X[N/2+1..N-1] = 0
3. z = IFFT(X)

### Edge Cases
- DC component: Hilbert transform of constant = 0 (correct)
- Endpoint effects: circular convolution assumption → wrap-around artifacts. Zero-pad.
- Instantaneous frequency can be negative (non-physical for narrowband signals)

---

## 9. Cepstrum

### Real Cepstrum
```
c[n] = IFFT(log|FFT(x)|²)
```

### Complex Cepstrum
```
ĉ[n] = IFFT(log(FFT(x)))     ← requires phase unwrapping
```

### Power Cepstrum
```
p[n] = |IFFT(log|FFT(x)|²)|²
```

### Application: Deconvolution, pitch detection (fundamental frequency = peak in cepstrum)

### CRITICAL: log(0) = -∞. Clamp |X[k]|² to max(|X[k]|², ε) where ε ≈ 10⁻¹⁰.

### Phase unwrapping for complex cepstrum is HARD and sometimes ambiguous. If not needed, use real cepstrum.

---

## 10. Wigner-Ville Distribution

### Definition
```
W_x(t, f) = ∫ x(t+τ/2)·x*(t-τ/2)·e^{-j2πfτ} dτ
```

### Properties
- Real-valued (even for complex x)
- Marginals: ∫W_x df = |x(t)|², ∫W_x dt = |X(f)|²
- Perfect time-frequency resolution (no uncertainty principle trade-off)

### CRITICAL: Cross-terms (interference)
For x = x₁ + x₂:
```
W_x = W_{x₁} + W_{x₂} + 2·Re(W_{x₁,x₂})    ← cross-term!
```
Cross-terms oscillate and can dominate the auto-terms. Solutions:
- **Pseudo-WVD**: Window in time → smoothing reduces cross-terms
- **Smoothed Pseudo-WVD**: Window in both time and frequency
- **Cohen's class**: General TFD = WVD * kernel (different kernels suppress cross-terms differently)

### Discrete Implementation
```
W[n, k] = 2·Σ_m x[n+m]·x*[n-m]·e^{-j4πkm/N}
```
For each time n: form instantaneous autocorrelation r[m] = x[n+m]·x*[n-m], then FFT(r).

### GPU: Parallel across time indices. Each time slice = one FFT.

---

## Sharing Surface

### The FFT IS the Family
```
FFT → STFT (windowed FFT)
    → Power spectrum (|FFT|²)
    → Convolution (FFT multiply IFFT)
    → FIR filtering (long filters via overlap-save)
    → Hilbert transform (FFT → mask → IFFT)
    → Cepstrum (log(|FFT|²) → IFFT)
    → DCT/DST (reindexed FFT)
    → CWT at fixed scale (convolution via FFT)
    → Wigner-Ville (instantaneous autocorrelation → FFT)
    → NUFFT (spread → FFT → deconvolve)
```

### Independent of FFT
```
DWT → filter bank (FIR convolution per level + downsample)
IIR → Affine scan (sequential)
Kalman → Affine scan (sequential)
Parks-McClellan → Remez exchange (iterative, Kingdom C)
```

### Reuse from Other Families
- **F06 (Descriptive)**: Window functions = weighted means
- **F10 (Regression)**: Savitzky-Golay = polynomial regression; spectral estimation via autoregressive models
- **F31 (Interpolation)**: Chebyshev approximation via DCT
- **F32 (Numerical)**: Root-finding for analog filter design (pole placement)
- **F05 (Optimization)**: Parks-McClellan (Remez exchange = iterative optimization)

### Consumers of F03
- **F19 (Spectral Time Series)**: ALL spectral analysis builds on FFT
- **F26 (Complexity)**: Phase space reconstruction filtering
- **F17 (Time Series)**: Spectral density estimation, seasonal decomposition
- **Fintek pipeline**: IAT spectral analysis (FFT of inter-arrival times)

---

## Kingdom Classification Summary

| Algorithm | Kingdom | Why |
|-----------|---------|-----|
| FFT (Cooley-Tukey) | A | Butterfly DAG — parallel within each stage |
| STFT | A | Independent FFTs per frame |
| RFFT | A | FFT with symmetry exploitation |
| Bluestein | A | Three FFTs + pointwise multiply |
| NUFFT | A | Spread + FFT + deconvolve |
| DCT/DST | A | Reindexed FFT |
| DWT (per level) | A (conv) | FIR convolution + downsample |
| DWT (across levels) | B | Each level depends on previous |
| CWT | A | Independent convolutions per scale |
| SWT | A/B | Same as DWT but without downsample |
| Lifting scheme | B | Sequential predict/update |
| FIR filter | A | Parallel dot products |
| IIR filter | **B** | Affine scan (y depends on previous y) |
| Butterworth/Chebyshev design | CPU | Analog prototype → bilinear transform |
| Parks-McClellan | **C** | Remez exchange algorithm (iterative) |
| Savitzky-Golay | A | FIR after coefficient precomputation |
| Kalman filter | **B** | Sequential predict/update scan |
| EKF/UKF | **B/C** | B (scan) with C (Jacobian/sigma point computation) |
| Hilbert transform | A | FFT → mask → IFFT |
| Cepstrum | A | FFT → log → IFFT |
| Wigner-Ville | A | Per-time-slice FFT |

---

## Implementation Priority

**Phase 1** — FFT core (~300 lines):
1. Radix-2 Cooley-Tukey (GPU butterfly kernel)
2. Stockham auto-sort variant (better GPU memory access)
3. RFFT (real-valued specialization)
4. Bluestein (arbitrary N support)
5. Inverse FFT

**Phase 2** — Spectral analysis (~200 lines):
6. STFT (windowed FFT + overlap-add ISTFT)
7. Power spectrum, cross-spectrum, coherence
8. Window functions (Hann, Hamming, Blackman, Kaiser, Blackman-Harris)
9. Hilbert transform / analytic signal
10. Cepstrum (real + complex)

**Phase 3** — Wavelets (~250 lines):
11. DWT (Mallat filter bank, Haar + Daubechies + Symlet + Coiflet coefficients)
12. IDWT (reconstruction)
13. SWT (undecimated)
14. CWT (via FFT convolution)
15. Lifting scheme

**Phase 4** — Filters (~200 lines):
16. FIR (direct + overlap-save FFT convolution)
17. IIR (direct form II transposed — Affine scan kernel)
18. Butterworth/Chebyshev/Elliptic design (CPU: analog → bilinear)
19. Savitzky-Golay (Vandermonde precomputation)
20. Kalman filter (predict/update scan)

**Phase 5** — Advanced (~150 lines):
21. DCT-II/III (via FFT)
22. NUFFT (Type 1 + Type 2)
23. Wigner-Ville / Pseudo-WVD
24. Sparse FFT (sublinear for sparse spectra)

---

## Composability Contract

```toml
[family_03]
name = "Signal Processing"
kingdom = "Mixed: A (FFT, spectral), B (IIR, Kalman, DWT levels)"

[family_03.shared_primitives]
fft = "Stockham radix-2 + Bluestein for arbitrary N"
rfft = "Real-valued FFT (N/2+1 complex output)"
stft = "Windowed FFT with overlap"
fir = "Parallel dot products or overlap-save FFT"
iir = "Affine scan kernel (Kingdom B)"
dwt = "Mallat filter bank (FIR per level)"
kalman = "Predict/update Affine scan"

[family_03.reuses]
f10_regression = "Savitzky-Golay = polynomial regression"
f31_chebyshev = "Chebyshev approximation via DCT"
f32_roots = "Pole placement for analog filter design"
f05_optimization = "Remez exchange for Parks-McClellan"

[family_03.provides]
fft = "Core FFT for all spectral analysis"
stft = "Time-frequency representation"
dwt = "Multi-resolution analysis"
kalman = "State estimation for dynamic systems"
hilbert = "Analytic signal / instantaneous frequency"

[family_03.consumers]
f19_spectral_ts = "All spectral time series analysis"
f26_complexity = "Filtering for phase space reconstruction"
f17_time_series = "Spectral density, seasonal decomposition"
fintek = "IAT spectral analysis"

[family_03.session_intermediates]
fft_result = "FFTResult(data_id, n_fft) — cached complex spectrum"
stft_result = "STFTResult(data_id, window, hop, nfft) — spectrogram"
dwt_coefficients = "DWTCoeffs(data_id, wavelet, levels)"
kalman_state = "KalmanState(model_id) — [x̂, P] for online update"
twiddle_factors = "TwiddleTable(N) — precomputed e^{-j2πk/N}, reused across FFTs"
```
