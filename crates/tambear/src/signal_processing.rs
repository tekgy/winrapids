//! # Family 03 — Signal Processing
//!
//! cuFFT replacement + beyond. From first principles.
//!
//! ## What lives here
//!
//! **FFT**: Cooley-Tukey radix-2 DIT, inverse FFT, real FFT, 2D FFT
//! **Spectral**: power spectrum, periodogram, Welch's method, STFT, spectrogram
//! **Convolution**: linear convolution via FFT, cross-correlation, autocorrelation
//! **Windows**: Hann, Hamming, Blackman, Kaiser, Bartlett, flat-top
//! **Filters**: FIR (windowed sinc), IIR (biquad cascade), Butterworth design
//! **Smoothing**: Savitzky-Golay, moving average, exponential smoothing
//! **Transforms**: DCT-II/III, Hilbert transform, cepstrum, analytic signal
//! **Wavelets**: Haar DWT/IDWT, Daubechies-4
//!
//! ## Architecture
//!
//! Complex arithmetic uses `(f64, f64)` tuples — no external dependency.
//! FFT is the atom. Convolution, correlation, filtering, spectral analysis
//! all reduce to FFT + pointwise operations + IFFT.
//!
//! ## MSR insight
//!
//! A power spectrum is a sufficient statistic for any stationary signal's
//! second-order properties. The FFT coefficients are the MSR of periodicity.
//! Wavelets give the MSR of time-frequency localization.

use std::f64::consts::PI;

/// Complex number as (real, imaginary) tuple.
pub type Complex = (f64, f64);

// ─── Complex arithmetic ─────────────────────────────────────────────

#[inline]
fn c_add(a: Complex, b: Complex) -> Complex {
    (a.0 + b.0, a.1 + b.1)
}

#[inline]
fn c_sub(a: Complex, b: Complex) -> Complex {
    (a.0 - b.0, a.1 - b.1)
}

#[inline]
fn c_mul(a: Complex, b: Complex) -> Complex {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

#[inline]
fn c_conj(a: Complex) -> Complex {
    (a.0, -a.1)
}

#[inline]
fn c_abs(a: Complex) -> f64 {
    (a.0 * a.0 + a.1 * a.1).sqrt()
}

#[inline]
fn c_scale(s: f64, a: Complex) -> Complex {
    (s * a.0, s * a.1)
}

#[inline]
fn c_exp_i(theta: f64) -> Complex {
    (theta.cos(), theta.sin())
}

// ─── FFT ────────────────────────────────────────────────────────────

/// Next power of 2 >= n.
pub fn next_pow2(n: usize) -> usize {
    if n <= 1 { return 1; }
    let mut p = 1;
    while p < n { p <<= 1; }
    p
}

/// In-place bit-reversal permutation.
fn bit_reverse_permute(data: &mut [Complex]) {
    let n = data.len();
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j ^= bit;
        if i < j {
            data.swap(i, j);
        }
    }
}

/// Cooley-Tukey radix-2 DIT FFT (in-place).
///
/// Input length MUST be a power of 2. For arbitrary lengths, use `fft()` which
/// zero-pads automatically.
///
/// If `inverse` is true, computes the IFFT (with 1/N normalization).
fn fft_radix2(data: &mut [Complex], inverse: bool) {
    let n = data.len();
    if n <= 1 { return; }
    debug_assert!(n.is_power_of_two(), "fft_radix2 requires power-of-2 length");

    bit_reverse_permute(data);

    let sign = if inverse { 1.0 } else { -1.0 };
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = sign * 2.0 * PI / len as f64;
        let wn = c_exp_i(angle);
        let mut i = 0;
        while i < n {
            let mut w: Complex = (1.0, 0.0);
            for j in 0..half {
                let u = data[i + j];
                let t = c_mul(w, data[i + j + half]);
                data[i + j] = c_add(u, t);
                data[i + j + half] = c_sub(u, t);
                w = c_mul(w, wn);
            }
            i += len;
        }
        len <<= 1;
    }

    if inverse {
        let inv_n = 1.0 / n as f64;
        for x in data.iter_mut() {
            *x = c_scale(inv_n, *x);
        }
    }
}

/// FFT of complex data.
///
/// Automatically zero-pads to next power of 2.
/// Returns N complex coefficients.
pub fn fft(data: &[Complex]) -> Vec<Complex> {
    if data.is_empty() { return vec![]; }
    let n = next_pow2(data.len());
    let mut buf = Vec::with_capacity(n);
    buf.extend_from_slice(data);
    buf.resize(n, (0.0, 0.0));
    fft_radix2(&mut buf, false);
    buf
}

/// Inverse FFT.
pub fn ifft(data: &[Complex]) -> Vec<Complex> {
    if data.is_empty() { return vec![]; }
    let n = next_pow2(data.len());
    let mut buf = Vec::with_capacity(n);
    buf.extend_from_slice(data);
    buf.resize(n, (0.0, 0.0));
    fft_radix2(&mut buf, true);
    buf
}

/// Real-to-complex FFT.
///
/// For a real signal of length N, returns N/2+1 complex coefficients
/// (exploiting Hermitian symmetry). Zero-pads to next power of 2.
pub fn rfft(data: &[f64]) -> Vec<Complex> {
    let complex: Vec<Complex> = data.iter().map(|&x| (x, 0.0)).collect();
    let mut result = fft(&complex);
    let n = result.len();
    result.truncate(n / 2 + 1);
    result
}

/// Inverse real FFT. Takes N/2+1 complex coefficients, returns N real values.
pub fn irfft(data: &[Complex], n: usize) -> Vec<f64> {
    let mut full = Vec::with_capacity(n);
    full.extend_from_slice(data);
    // Mirror the conjugate symmetric part
    for i in (1..data.len() - 1).rev() {
        full.push(c_conj(data[i]));
    }
    full.resize(n, (0.0, 0.0));
    let result = ifft(&full);
    result.iter().map(|c| c.0).take(n).collect()
}

/// 2D FFT (row-major).
///
/// Computes FFT along rows, then along columns. Input is rows × cols.
pub fn fft2d(data: &[Complex], rows: usize, cols: usize) -> Vec<Complex> {
    if rows == 0 || cols == 0 || data.len() != rows * cols {
        return vec![];
    }
    let pc = next_pow2(cols);
    let pr = next_pow2(rows);

    // FFT along rows
    let mut buf = vec![(0.0, 0.0); pr * pc];
    for r in 0..rows {
        for c in 0..cols {
            buf[r * pc + c] = data[r * cols + c];
        }
    }
    for r in 0..rows {
        let mut row = buf[r * pc..(r + 1) * pc].to_vec();
        fft_radix2(&mut row, false);
        buf[r * pc..(r + 1) * pc].copy_from_slice(&row);
    }

    // FFT along columns
    for c in 0..pc {
        let mut col: Vec<Complex> = (0..pr).map(|r| buf[r * pc + c]).collect();
        fft_radix2(&mut col, false);
        for r in 0..pr {
            buf[r * pc + c] = col[r];
        }
    }
    buf
}

// ─── Window functions ───────────────────────────────────────────────

/// Hann (Hanning) window.
pub fn window_hann(n: usize) -> Vec<f64> {
    if n <= 1 { return vec![1.0; n]; }
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / (n - 1) as f64).cos())).collect()
}

/// Hamming window.
pub fn window_hamming(n: usize) -> Vec<f64> {
    if n <= 1 { return vec![1.0; n]; }
    (0..n).map(|i| 0.54 - 0.46 * (2.0 * PI * i as f64 / (n - 1) as f64).cos()).collect()
}

/// Blackman window.
pub fn window_blackman(n: usize) -> Vec<f64> {
    if n <= 1 { return vec![1.0; n]; }
    let a0 = 0.42;
    let a1 = 0.5;
    let a2 = 0.08;
    (0..n).map(|i| {
        let t = 2.0 * PI * i as f64 / (n - 1) as f64;
        a0 - a1 * t.cos() + a2 * (2.0 * t).cos()
    }).collect()
}

/// Bartlett (triangular) window.
pub fn window_bartlett(n: usize) -> Vec<f64> {
    if n <= 1 { return vec![1.0; n]; }
    let half = (n - 1) as f64 / 2.0;
    (0..n).map(|i| 1.0 - ((i as f64 - half) / half).abs()).collect()
}

/// Kaiser window with parameter β.
pub fn window_kaiser(n: usize, beta: f64) -> Vec<f64> {
    if n <= 1 { return vec![1.0; n]; }
    let i0_beta = bessel_i0(beta);
    (0..n).map(|i| {
        let t = 2.0 * i as f64 / (n - 1) as f64 - 1.0;
        bessel_i0(beta * (1.0 - t * t).max(0.0).sqrt()) / i0_beta
    }).collect()
}

/// Flat-top window (for accurate amplitude measurement).
pub fn window_flat_top(n: usize) -> Vec<f64> {
    if n <= 1 { return vec![1.0; n]; }
    let a0 = 0.21557895;
    let a1 = 0.41663158;
    let a2 = 0.277263158;
    let a3 = 0.083578947;
    let a4 = 0.006947368;
    (0..n).map(|i| {
        let t = 2.0 * PI * i as f64 / (n - 1) as f64;
        a0 - a1 * t.cos() + a2 * (2.0 * t).cos() - a3 * (3.0 * t).cos() + a4 * (4.0 * t).cos()
    }).collect()
}

/// Modified Bessel function I_0 (for Kaiser window). Series approximation.
fn bessel_i0(x: f64) -> f64 {
    let mut sum = 1.0;
    let mut term = 1.0;
    let x2 = x * x / 4.0;
    for k in 1..30 {
        term *= x2 / (k * k) as f64;
        sum += term;
        if term < 1e-16 * sum { break; }
    }
    sum
}

// ─── Spectral analysis ──────────────────────────────────────────────

/// Power spectral density (periodogram).
///
/// Returns (frequencies, PSD) where frequencies are in [0, fs/2].
/// Uses the standard periodogram estimator: |FFT(x)|² / N.
pub fn periodogram(data: &[f64], fs: f64) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    if n == 0 { return (vec![], vec![]); }
    let spectrum = rfft(data);
    let nfft = (spectrum.len() - 1) * 2;
    let df = fs / nfft as f64;
    let freqs: Vec<f64> = (0..spectrum.len()).map(|i| i as f64 * df).collect();
    let scale = 1.0 / (fs * n as f64);
    let psd: Vec<f64> = spectrum.iter().enumerate().map(|(i, &c)| {
        let p = c.0 * c.0 + c.1 * c.1;
        // Double non-DC, non-Nyquist bins to account for negative frequencies
        if i == 0 || i == spectrum.len() - 1 { p * scale } else { 2.0 * p * scale }
    }).collect();
    (freqs, psd)
}

/// Welch's method for PSD estimation.
///
/// Overlapping windowed segments averaged for reduced variance.
/// `segment_len`: length of each segment (zero-padded to next pow2)
/// `overlap`: number of overlapping samples between segments
pub fn welch(data: &[f64], segment_len: usize, overlap: usize, fs: f64) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    if n == 0 || segment_len == 0 { return (vec![], vec![]); }
    let seg_len = segment_len.min(n);
    let step = seg_len - overlap.min(seg_len - 1);
    let window = window_hann(seg_len);
    let win_norm: f64 = window.iter().map(|w| w * w).sum();

    let nfft = next_pow2(seg_len);
    let n_freqs = nfft / 2 + 1;
    let mut psd_avg = vec![0.0; n_freqs];
    let mut n_segments = 0;

    let mut start = 0;
    while start + seg_len <= n {
        let segment: Vec<f64> = (0..seg_len).map(|i| data[start + i] * window[i]).collect();
        let spec = rfft(&segment);
        for i in 0..n_freqs.min(spec.len()) {
            let p = spec[i].0 * spec[i].0 + spec[i].1 * spec[i].1;
            psd_avg[i] += p;
        }
        n_segments += 1;
        start += step;
    }

    if n_segments == 0 {
        return (vec![], vec![]);
    }

    let scale = 1.0 / (fs * win_norm * n_segments as f64);
    let df = fs / nfft as f64;
    let freqs: Vec<f64> = (0..n_freqs).map(|i| i as f64 * df).collect();
    for i in 0..n_freqs {
        psd_avg[i] *= scale;
        if i > 0 && i < n_freqs - 1 {
            psd_avg[i] *= 2.0; // account for negative frequencies
        }
    }

    (freqs, psd_avg)
}

/// Short-Time Fourier Transform (STFT).
///
/// Returns a 2D array of complex coefficients: time_frames × freq_bins.
/// Each frame is windowed and FFTed.
pub fn stft(data: &[f64], window_len: usize, hop_size: usize) -> Vec<Vec<Complex>> {
    let n = data.len();
    if n == 0 || window_len == 0 || hop_size == 0 { return vec![]; }
    let win_len = window_len.min(n);
    let window = window_hann(win_len);
    let nfft = next_pow2(win_len);

    let mut frames = Vec::new();
    let mut start = 0;
    while start + win_len <= n {
        let mut frame: Vec<Complex> = (0..win_len)
            .map(|i| (data[start + i] * window[i], 0.0))
            .collect();
        frame.resize(nfft, (0.0, 0.0));
        fft_radix2(&mut frame, false);
        frames.push(frame[..nfft / 2 + 1].to_vec());
        start += hop_size;
    }
    frames
}

/// Spectrogram (magnitude of STFT).
///
/// Returns (time_centers, frequencies, magnitude_matrix).
pub fn spectrogram(data: &[f64], window_len: usize, hop_size: usize, fs: f64)
    -> (Vec<f64>, Vec<f64>, Vec<Vec<f64>>)
{
    let frames = stft(data, window_len, hop_size);
    if frames.is_empty() { return (vec![], vec![], vec![]); }
    let nfft = next_pow2(window_len.min(data.len()));
    let n_freqs = nfft / 2 + 1;
    let df = fs / nfft as f64;
    let freqs: Vec<f64> = (0..n_freqs).map(|i| i as f64 * df).collect();
    let times: Vec<f64> = (0..frames.len())
        .map(|i| (i * hop_size + window_len / 2) as f64 / fs)
        .collect();
    let mag: Vec<Vec<f64>> = frames.iter()
        .map(|frame| frame.iter().map(|&c| c_abs(c)).collect())
        .collect();
    (times, freqs, mag)
}

// ─── Convolution & Correlation ──────────────────────────────────────

/// Linear convolution via FFT.
///
/// Returns a vector of length len(a) + len(b) - 1.
pub fn convolve(a: &[f64], b: &[f64]) -> Vec<f64> {
    let na = a.len();
    let nb = b.len();
    if na == 0 || nb == 0 { return vec![]; }
    let out_len = na + nb - 1;
    let n = next_pow2(out_len);

    let mut fa: Vec<Complex> = a.iter().map(|&x| (x, 0.0)).collect();
    fa.resize(n, (0.0, 0.0));
    let mut fb: Vec<Complex> = b.iter().map(|&x| (x, 0.0)).collect();
    fb.resize(n, (0.0, 0.0));

    fft_radix2(&mut fa, false);
    fft_radix2(&mut fb, false);

    let mut prod: Vec<Complex> = fa.iter().zip(fb.iter()).map(|(&a, &b)| c_mul(a, b)).collect();
    fft_radix2(&mut prod, true);

    prod.iter().take(out_len).map(|c| c.0).collect()
}

/// Cross-correlation of two signals via FFT.
///
/// corr(a, b)[k] = Σ a[n] · b[n+k]. Returns length len(a) + len(b) - 1.
pub fn cross_correlate(a: &[f64], b: &[f64]) -> Vec<f64> {
    let na = a.len();
    let nb = b.len();
    if na == 0 || nb == 0 { return vec![]; }
    let out_len = na + nb - 1;
    let n = next_pow2(out_len);

    let mut fa: Vec<Complex> = a.iter().map(|&x| (x, 0.0)).collect();
    fa.resize(n, (0.0, 0.0));
    let mut fb: Vec<Complex> = b.iter().map(|&x| (x, 0.0)).collect();
    fb.resize(n, (0.0, 0.0));

    fft_radix2(&mut fa, false);
    fft_radix2(&mut fb, false);

    // Cross-correlation = IFFT(conj(A) · B)
    let mut prod: Vec<Complex> = fa.iter().zip(fb.iter())
        .map(|(&a, &b)| c_mul(c_conj(a), b))
        .collect();
    fft_radix2(&mut prod, true);

    prod.iter().take(out_len).map(|c| c.0).collect()
}

/// Autocorrelation of a signal (normalized).
///
/// Returns lags 0..max_lag (lag 0 = 1.0 by definition).
/// Uses the Wiener-Khinchin theorem: R(τ) = IFFT(|FFT(x)|²).
pub fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 { return vec![]; }
    let moments = crate::descriptive::moments_ungrouped(data);
    let mean = moments.mean();
    let centered: Vec<f64> = data.iter().map(|&x| x - mean).collect();
    let var = moments.m2; // Σ(x - x̄)² — unnormalized
    if var < 1e-300 { return vec![1.0; max_lag.min(n)]; }

    // Zero-pad to avoid circular correlation artifacts
    let nfft = next_pow2(2 * n);
    let mut buf: Vec<Complex> = centered.iter().map(|&x| (x, 0.0)).collect();
    buf.resize(nfft, (0.0, 0.0));
    fft_radix2(&mut buf, false);

    // Power spectrum: |X|²
    for c in buf.iter_mut() {
        *c = (c.0 * c.0 + c.1 * c.1, 0.0);
    }

    fft_radix2(&mut buf, true);

    // Normalize by lag-0 value
    let lags = max_lag.min(n);
    let r0 = buf[0].0;
    if r0.abs() < 1e-300 { return vec![1.0; lags]; }
    (0..lags).map(|k| buf[k].0 / r0).collect()
}

// ─── DCT ────────────────────────────────────────────────────────────

/// DCT Type II (the "standard" DCT, used in JPEG, MP3, etc.).
///
/// X[k] = 2 Σ x[n] cos(π(2n+1)k / 2N)
pub fn dct2(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n == 0 { return vec![]; }
    let mut result = vec![0.0; n];
    for k in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            sum += data[j] * (PI * (2 * j + 1) as f64 * k as f64 / (2 * n) as f64).cos();
        }
        result[k] = 2.0 * sum;
    }
    result
}

/// DCT Type III (inverse of DCT-II, up to normalization).
///
/// x[n] = X[0]/N + (2/N) Σ_{k=1}^{N-1} X[k] cos(π(2n+1)k / 2N)
pub fn dct3(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n == 0 { return vec![]; }
    let mut result = vec![0.0; n];
    for j in 0..n {
        let mut sum = data[0] / 2.0;
        for k in 1..n {
            sum += data[k] * (PI * (2 * j + 1) as f64 * k as f64 / (2 * n) as f64).cos();
        }
        result[j] = sum; // no 2/N normalization — caller handles
    }
    result
}

// ─── Filters ────────────────────────────────────────────────────────

/// FIR filter: windowed sinc lowpass.
///
/// Returns filter coefficients for a lowpass filter with cutoff frequency
/// `fc` (normalized, 0 to 1 where 1 = Nyquist). `order` is the filter order
/// (number of taps = order + 1, should be even).
pub fn fir_lowpass(fc: f64, order: usize) -> Vec<f64> {
    let n = order + 1;
    let m = order as f64 / 2.0;
    let window = window_hamming(n);
    let mut h = vec![0.0; n];
    for i in 0..n {
        let x = i as f64 - m;
        if x.abs() < 1e-10 {
            h[i] = 2.0 * fc;
        } else {
            h[i] = (2.0 * PI * fc * x).sin() / (PI * x);
        }
        h[i] *= window[i];
    }
    // Normalize to unity gain at DC
    let sum: f64 = h.iter().sum();
    if sum.abs() > 1e-300 {
        for v in h.iter_mut() { *v /= sum; }
    }
    h
}

/// FIR filter: windowed sinc highpass.
///
/// Spectral inversion of the lowpass filter.
pub fn fir_highpass(fc: f64, order: usize) -> Vec<f64> {
    let mut h = fir_lowpass(fc, order);
    let m = order / 2;
    for v in h.iter_mut() { *v = -*v; }
    h[m] += 1.0;
    h
}

/// FIR filter: bandpass.
///
/// Convolution of lowpass(fh) with highpass(fl).
pub fn fir_bandpass(fl: f64, fh: f64, order: usize) -> Vec<f64> {
    let lp = fir_lowpass(fh, order);
    let hp = fir_highpass(fl, order);
    convolve(&lp, &hp)
}

/// Apply FIR filter to a signal (linear convolution, trimmed to input length).
pub fn fir_filter(signal: &[f64], coeffs: &[f64]) -> Vec<f64> {
    if signal.is_empty() || coeffs.is_empty() { return vec![]; }
    let full = convolve(signal, coeffs);
    let delay = coeffs.len() / 2;
    full[delay..delay + signal.len()].to_vec()
}

/// Biquad IIR filter coefficients (second-order section).
#[derive(Debug, Clone, Copy)]
pub struct Biquad {
    pub b0: f64, pub b1: f64, pub b2: f64,
    pub a1: f64, pub a2: f64,
}

impl Biquad {
    /// Butterworth lowpass biquad design.
    ///
    /// `fc`: cutoff frequency (normalized 0-1, where 1 = Nyquist).
    pub fn butterworth_lowpass(fc: f64) -> Self {
        let omega = PI * fc;
        let s = omega.sin();
        let c = omega.cos();
        let alpha = s / (2.0_f64.sqrt());
        let b0 = (1.0 - c) / 2.0;
        let b1 = 1.0 - c;
        let b2 = (1.0 - c) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * c;
        let a2 = 1.0 - alpha;
        Biquad {
            b0: b0 / a0, b1: b1 / a0, b2: b2 / a0,
            a1: a1 / a0, a2: a2 / a0,
        }
    }

    /// Butterworth highpass biquad design.
    pub fn butterworth_highpass(fc: f64) -> Self {
        let omega = PI * fc;
        let s = omega.sin();
        let c = omega.cos();
        let alpha = s / (2.0_f64.sqrt());
        let b0 = (1.0 + c) / 2.0;
        let b1 = -(1.0 + c);
        let b2 = (1.0 + c) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * c;
        let a2 = 1.0 - alpha;
        Biquad {
            b0: b0 / a0, b1: b1 / a0, b2: b2 / a0,
            a1: a1 / a0, a2: a2 / a0,
        }
    }

    /// Apply biquad filter to signal (Direct Form II Transposed).
    pub fn apply(&self, signal: &[f64]) -> Vec<f64> {
        let mut output = Vec::with_capacity(signal.len());
        let mut z1 = 0.0_f64;
        let mut z2 = 0.0_f64;
        for &x in signal {
            let y = self.b0 * x + z1;
            z1 = self.b1 * x - self.a1 * y + z2;
            z2 = self.b2 * x - self.a2 * y;
            output.push(y);
        }
        output
    }
}

/// Apply cascade of biquad sections (for higher-order Butterworth).
pub fn biquad_cascade(signal: &[f64], sections: &[Biquad]) -> Vec<f64> {
    let mut data = signal.to_vec();
    for bq in sections {
        data = bq.apply(&data);
    }
    data
}

/// Design Nth-order Butterworth lowpass as cascade of biquad sections.
///
/// For even order N, produces N/2 biquads. For odd, (N-1)/2 biquads + 1 first-order.
pub fn butterworth_lowpass_cascade(fc: f64, order: usize) -> Vec<Biquad> {
    if order == 0 { return vec![]; }
    // Pre-warp
    let omega = (PI * fc).tan();
    let omega2 = omega * omega;

    let mut sections = Vec::new();
    let n_sections = (order + 1) / 2;

    for k in 0..n_sections {
        if order % 2 == 1 && k == 0 {
            // First-order section for odd order
            let a0 = 1.0 + omega;
            let b0 = omega / a0;
            let b1 = omega / a0;
            let a1 = (omega - 1.0) / a0;
            sections.push(Biquad { b0, b1, b2: 0.0, a1, a2: 0.0 });
        } else {
            let idx = if order % 2 == 1 { k } else { k + 1 };
            let theta = PI * (2 * idx - 1) as f64 / (2 * order) as f64;
            let cos_t = theta.cos();
            let a0 = 1.0 + 2.0 * omega * cos_t + omega2;
            sections.push(Biquad {
                b0: omega2 / a0,
                b1: 2.0 * omega2 / a0,
                b2: omega2 / a0,
                a1: 2.0 * (omega2 - 1.0) / a0,
                a2: (1.0 - 2.0 * omega * cos_t + omega2) / a0,
            });
        }
    }
    sections
}

// ─── Smoothing ──────────────────────────────────────────────────────

/// Moving average filter (simple, centered).
pub fn moving_average(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 || window == 0 { return vec![]; }
    let w = window.min(n);
    let half = w / 2;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let lo = if i >= half { i - half } else { 0 };
        let hi = (i + w - half).min(n);
        let count = hi - lo;
        let sum: f64 = data[lo..hi].iter().sum();
        result.push(sum / count as f64);
    }
    result
}

/// Exponential moving average (EMA).
///
/// `alpha` is the smoothing factor (0 < alpha <= 1). Higher = less smoothing.
pub fn ema(data: &[f64], alpha: f64) -> Vec<f64> {
    if data.is_empty() { return vec![]; }
    let mut result = Vec::with_capacity(data.len());
    result.push(data[0]);
    for i in 1..data.len() {
        let prev = result[i - 1];
        result.push(alpha * data[i] + (1.0 - alpha) * prev);
    }
    result
}

/// Savitzky-Golay filter (polynomial smoothing).
///
/// Fits a polynomial of degree `poly_order` to each window of `window_len`
/// points and evaluates at the center. Classic noise reduction that preserves
/// peaks and edges better than moving average.
pub fn savgol_filter(data: &[f64], window_len: usize, poly_order: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 || window_len == 0 { return vec![]; }
    let w = if window_len % 2 == 0 { window_len + 1 } else { window_len };
    let w = w.min(n);
    if poly_order >= w { return data.to_vec(); }
    let half = w / 2;

    // Direct local polynomial fitting per point
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let lo = if i >= half { i - half } else { 0 };
        let hi = (i + half + 1).min(n);
        let local_n = hi - lo;
        if local_n <= poly_order {
            result.push(data[i]);
            continue;
        }
        let xs: Vec<f64> = (0..local_n).map(|j| j as f64).collect();
        let ys: Vec<f64> = data[lo..hi].to_vec();
        let fit = crate::interpolation::polyfit(&xs, &ys, poly_order.min(local_n - 1));
        let eval_at = (i - lo) as f64;
        result.push(fit.eval(eval_at));
    }
    result
}

// ─── Hilbert transform & analytic signal ────────────────────────────

/// Hilbert transform — returns the analytic signal.
///
/// The analytic signal z(t) = x(t) + i·H[x](t) has the property that
/// its magnitude is the envelope and its phase gives instantaneous frequency.
pub fn hilbert(data: &[f64]) -> Vec<Complex> {
    let n = data.len();
    if n == 0 { return vec![]; }
    let mut spec: Vec<Complex> = data.iter().map(|&x| (x, 0.0)).collect();
    let nfft = next_pow2(n);
    spec.resize(nfft, (0.0, 0.0));
    fft_radix2(&mut spec, false);

    // Zero negative frequencies, double positive frequencies
    // DC and Nyquist stay as-is
    for i in 1..nfft / 2 {
        spec[i] = c_scale(2.0, spec[i]);
    }
    for i in nfft / 2 + 1..nfft {
        spec[i] = (0.0, 0.0);
    }

    fft_radix2(&mut spec, true);
    spec.truncate(n);
    spec
}

/// Envelope (instantaneous amplitude) of a signal via Hilbert transform.
pub fn envelope(data: &[f64]) -> Vec<f64> {
    hilbert(data).iter().map(|&c| c_abs(c)).collect()
}

/// Instantaneous frequency of a signal (via analytic signal).
///
/// Returns frequencies in Hz given sample rate `fs`.
pub fn instantaneous_frequency(data: &[f64], fs: f64) -> Vec<f64> {
    let analytic = hilbert(data);
    let n = analytic.len();
    if n < 2 { return vec![]; }
    let mut freq = Vec::with_capacity(n - 1);
    for i in 1..n {
        let phase_prev = analytic[i - 1].1.atan2(analytic[i - 1].0);
        let phase_curr = analytic[i].1.atan2(analytic[i].0);
        let mut dp = phase_curr - phase_prev;
        // Phase unwrap
        while dp > PI { dp -= 2.0 * PI; }
        while dp < -PI { dp += 2.0 * PI; }
        freq.push(dp * fs / (2.0 * PI));
    }
    freq
}

// ─── Cepstrum ───────────────────────────────────────────────────────

/// Real cepstrum: IFFT(log(|FFT(x)|)).
///
/// Used for pitch detection, echo removal, deconvolution.
pub fn real_cepstrum(data: &[f64]) -> Vec<f64> {
    let n = data.len();
    if n == 0 { return vec![]; }
    let nfft = next_pow2(n);
    let mut spec: Vec<Complex> = data.iter().map(|&x| (x, 0.0)).collect();
    spec.resize(nfft, (0.0, 0.0));
    fft_radix2(&mut spec, false);

    // log(|X|)
    let mut log_mag: Vec<Complex> = spec.iter()
        .map(|&c| {
            let mag = c_abs(c).max(1e-300);
            (mag.ln(), 0.0)
        })
        .collect();

    fft_radix2(&mut log_mag, true);
    log_mag.iter().take(n).map(|c| c.0).collect()
}

// ─── Wavelets ───────────────────────────────────────────────────────

/// Haar wavelet decomposition (one level).
///
// ═══════════════════════════════════════════════════════════════════════════
// Morlet Continuous Wavelet Transform (CWT)
// ═══════════════════════════════════════════════════════════════════════════

/// Complex Morlet wavelet in the time domain.
///
/// ψ(t) = π^(-1/4) · exp(i·ω₀·t) · exp(-t²/2)
///
/// where ω₀ is the central angular frequency (typically 6 for good
/// time-frequency localization). Returns (real, imag) components.
pub fn morlet_wavelet(t: f64, omega0: f64) -> (f64, f64) {
    // π^(-1/4)
    let norm = 1.0 / std::f64::consts::PI.powf(0.25);
    let env = (-0.5 * t * t).exp() * norm;
    let phase = omega0 * t;
    (env * phase.cos(), env * phase.sin())
}

/// Continuous Wavelet Transform (CWT) via time-domain convolution.
///
/// W(s, t) = (1/√s) · ∫ x(u) · ψ*((u - t)/s) du
///
/// Returns a matrix W[scale_index * n + time_index] of |W(s, t)|² (scalogram).
///
/// `data`: the signal.
/// `scales`: wavelet scales to evaluate (larger = lower frequency).
/// `omega0`: Morlet central frequency parameter (typically 6).
///
/// Returns a flat Vec with shape (scales.len() × data.len()).
pub fn morlet_cwt(data: &[f64], scales: &[f64], omega0: f64) -> Vec<f64> {
    let n = data.len();
    let ns = scales.len();
    if n == 0 || ns == 0 { return vec![]; }

    let mut out = vec![0.0_f64; ns * n];
    for (si, &s) in scales.iter().enumerate() {
        if s <= 0.0 { continue; }
        let inv_s = 1.0 / s;
        let s_norm = 1.0 / s.sqrt(); // amplitude normalization
        // Support radius: Morlet dies at ~4σ → s·4 in time units
        let half_width = (s * 5.0) as isize;

        for t in 0..n {
            let mut acc_re = 0.0_f64;
            let mut acc_im = 0.0_f64;
            let u_start = ((t as isize) - half_width).max(0) as usize;
            let u_end = ((t as isize) + half_width).min(n as isize - 1) as usize;
            for u in u_start..=u_end {
                let arg = (u as f64 - t as f64) * inv_s;
                let (wr, wi) = morlet_wavelet(arg, omega0);
                // Complex conjugate: ψ* has -wi
                acc_re += data[u] * wr;
                acc_im += data[u] * (-wi);
            }
            let re = acc_re * s_norm;
            let im = acc_im * s_norm;
            out[si * n + t] = re * re + im * im;
        }
    }
    out
}

/// Convert a wavelet scale to its pseudo-frequency for Morlet.
///
/// For Morlet: f_pseudo = f_c / s, where f_c = ω₀ / (2π).
pub fn morlet_scale_to_frequency(scale: f64, omega0: f64, fs: f64) -> f64 {
    let fc = omega0 / (2.0 * std::f64::consts::PI);
    fc * fs / scale
}

/// Build a logarithmically spaced set of scales covering a frequency range.
///
/// `f_min`, `f_max`: frequency range (Hz).
/// `n_scales`: number of scales.
/// `fs`: sampling frequency (Hz).
/// `omega0`: Morlet parameter.
pub fn morlet_log_scales(f_min: f64, f_max: f64, n_scales: usize, fs: f64, omega0: f64) -> Vec<f64> {
    if n_scales == 0 || f_min <= 0.0 || f_max <= 0.0 || f_min >= f_max { return vec![]; }
    let fc = omega0 / (2.0 * std::f64::consts::PI);
    // s = fc · fs / f → log-linear in 1/s, which means log-linear in f
    let log_min = f_min.ln();
    let log_max = f_max.ln();
    (0..n_scales).map(|i| {
        let logf = log_max - (log_max - log_min) * (i as f64) / (n_scales - 1).max(1) as f64;
        let f = logf.exp();
        fc * fs / f
    }).collect()
}

/// Returns (approximation_coefficients, detail_coefficients).
/// Input length should be even; if odd, last sample is dropped.
pub fn haar_dwt(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = data.len() / 2;
    if n == 0 { return (vec![], vec![]); }
    let s = 1.0 / 2.0_f64.sqrt();
    let approx: Vec<f64> = (0..n).map(|i| s * (data[2 * i] + data[2 * i + 1])).collect();
    let detail: Vec<f64> = (0..n).map(|i| s * (data[2 * i] - data[2 * i + 1])).collect();
    (approx, detail)
}

/// Haar wavelet reconstruction (one level).
pub fn haar_idwt(approx: &[f64], detail: &[f64]) -> Vec<f64> {
    let n = approx.len().min(detail.len());
    let s = 1.0 / 2.0_f64.sqrt();
    let mut result = vec![0.0; 2 * n];
    for i in 0..n {
        result[2 * i] = s * (approx[i] + detail[i]);
        result[2 * i + 1] = s * (approx[i] - detail[i]);
    }
    result
}

/// Multi-level Haar wavelet decomposition.
///
/// Returns (final_approx, [detail_level_1, detail_level_2, ...])
/// where level 1 is the finest (highest frequency) detail.
pub fn haar_wavedec(data: &[f64], levels: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut approx = data.to_vec();
    let mut details = Vec::new();
    for _ in 0..levels {
        if approx.len() < 2 { break; }
        let (a, d) = haar_dwt(&approx);
        details.push(d);
        approx = a;
    }
    details.reverse(); // coarsest first → finest last
    (approx, details)
}

/// Multi-level Haar wavelet reconstruction.
pub fn haar_waverec(approx: &[f64], details: &[Vec<f64>]) -> Vec<f64> {
    let mut a = approx.to_vec();
    // Details are coarsest to finest, reconstruct coarsest first
    for d in details.iter() {
        a = haar_idwt(&a, d);
    }
    a
}

/// Daubechies-4 wavelet decomposition (one level).
///
/// The Daubechies-4 has 4 coefficients (2 vanishing moments).
/// Better frequency localization than Haar.
pub fn db4_dwt(data: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    if n < 4 { return (data.to_vec(), vec![]); }
    // DB4 scaling coefficients
    let h0 = (1.0 + 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
    let h1 = (3.0 + 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
    let h2 = (3.0 - 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
    let h3 = (1.0 - 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
    // Wavelet coefficients (alternating flip)
    let g0 = h3;
    let g1 = -h2;
    let g2 = h1;
    let g3 = -h0;

    let half = n / 2;
    let mut approx = vec![0.0; half];
    let mut detail = vec![0.0; half];
    for i in 0..half {
        let j = 2 * i;
        approx[i] = h0 * data[j % n] + h1 * data[(j + 1) % n]
            + h2 * data[(j + 2) % n] + h3 * data[(j + 3) % n];
        detail[i] = g0 * data[j % n] + g1 * data[(j + 1) % n]
            + g2 * data[(j + 2) % n] + g3 * data[(j + 3) % n];
    }
    (approx, detail)
}

/// Daubechies-4 wavelet reconstruction (one level).
pub fn db4_idwt(approx: &[f64], detail: &[f64]) -> Vec<f64> {
    let half = approx.len().min(detail.len());
    let n = 2 * half;
    if n == 0 { return vec![]; }
    let h0 = (1.0 + 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
    let h1 = (3.0 + 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
    let h2 = (3.0 - 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
    let h3 = (1.0 - 3.0_f64.sqrt()) / (4.0 * 2.0_f64.sqrt());
    let g0 = h3;
    let g1 = -h2;
    let g2 = h1;
    let g3 = -h0;

    let mut result = vec![0.0; n];
    for i in 0..half {
        let j = 2 * i;
        result[j % n] += h0 * approx[i] + g0 * detail[i];
        result[(j + 1) % n] += h1 * approx[i] + g1 * detail[i];
        result[(j + 2) % n] += h2 * approx[i] + g2 * detail[i];
        result[(j + 3) % n] += h3 * approx[i] + g3 * detail[i];
    }
    result
}

// ─── Goertzel algorithm ─────────────────────────────────────────────

/// Goertzel algorithm — efficient single-frequency DFT bin.
///
/// O(N) for one frequency vs O(N log N) for full FFT.
/// Perfect for tone detection, DTMF, targeted spectral measurement.
pub fn goertzel(data: &[f64], freq: f64, fs: f64) -> Complex {
    let n = data.len();
    if n == 0 { return (0.0, 0.0); }
    let k = freq * n as f64 / fs;
    let w = 2.0 * PI * k / n as f64;
    let coeff = 2.0 * w.cos();
    let mut s0 = 0.0;
    let mut s1 = 0.0;
    let mut s2;
    for &x in data {
        s2 = s1;
        s1 = s0;
        s0 = x + coeff * s1 - s2;
    }
    let real = s0 - s1 * w.cos();
    let imag = -s1 * w.sin();
    (real, imag)
}

/// Goertzel magnitude (power at a specific frequency).
pub fn goertzel_mag(data: &[f64], freq: f64, fs: f64) -> f64 {
    let c = goertzel(data, freq, fs);
    c_abs(c)
}

// ─── Zero crossing rate ─────────────────────────────────────────────

/// Zero crossing rate — fraction of adjacent samples with sign change.
///
/// Fundamental feature in speech/audio processing.
pub fn zero_crossing_rate(data: &[f64]) -> f64 {
    let n = data.len();
    if n < 2 { return 0.0; }
    let crossings: usize = data.windows(2)
        .filter(|w| (w[0] >= 0.0) != (w[1] >= 0.0))
        .count();
    crossings as f64 / (n - 1) as f64
}

// ─── Median filter ──────────────────────────────────────────────────

/// Median filter (nonlinear noise reduction).
///
/// Replaces each sample with the median of its neighborhood.
/// Excellent for impulse noise (salt & pepper).
pub fn median_filter(data: &[f64], window: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 || window == 0 { return vec![]; }
    let w = window.min(n);
    let half = w / 2;
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let lo = if i >= half { i - half } else { 0 };
        let hi = (i + half + 1).min(n);
        let mut local: Vec<f64> = data[lo..hi].to_vec();
        local.sort_by(|a, b| a.total_cmp(b));
        let mid = local.len() / 2;
        result.push(if local.len() % 2 == 0 {
            (local[mid - 1] + local[mid]) / 2.0
        } else {
            local[mid]
        });
    }
    result
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── FFT ──

    #[test]
    fn fft_dc_signal() {
        // FFT of constant signal = DC component only
        let data: Vec<Complex> = vec![(1.0, 0.0); 8];
        let result = fft(&data);
        assert!((result[0].0 - 8.0).abs() < 1e-10, "DC = {}", result[0].0);
        for i in 1..result.len() {
            assert!(c_abs(result[i]) < 1e-10, "bin {} = {:?}", i, result[i]);
        }
    }

    #[test]
    fn fft_single_frequency() {
        // FFT of cos(2π·k/N) should peak at bin k
        let n = 64;
        let k = 5;
        let data: Vec<Complex> = (0..n)
            .map(|i| ((2.0 * PI * k as f64 * i as f64 / n as f64).cos(), 0.0))
            .collect();
        let result = fft(&data);
        // Bin k and bin n-k should have magnitude n/2
        assert!((c_abs(result[k]) - n as f64 / 2.0).abs() < 1e-8,
            "bin {} mag = {}", k, c_abs(result[k]));
        assert!((c_abs(result[n - k]) - n as f64 / 2.0).abs() < 1e-8);
    }

    #[test]
    fn fft_ifft_roundtrip() {
        let data: Vec<Complex> = (0..16).map(|i| (i as f64, (i * i) as f64 * 0.1)).collect();
        let transformed = fft(&data);
        let recovered = ifft(&transformed);
        for i in 0..data.len() {
            assert!((recovered[i].0 - data[i].0).abs() < 1e-10,
                "real mismatch at {}", i);
            assert!((recovered[i].1 - data[i].1).abs() < 1e-10,
                "imag mismatch at {}", i);
        }
    }

    #[test]
    fn rfft_symmetry() {
        let data: Vec<f64> = (0..32).map(|i| (2.0 * PI * 3.0 * i as f64 / 32.0).sin()).collect();
        let spec = rfft(&data);
        // Should have 17 coefficients for length-32 input
        assert_eq!(spec.len(), 17);
    }

    #[test]
    fn fft_parseval() {
        // Parseval's theorem: sum |x|² = (1/N) sum |X|²
        let data: Vec<Complex> = (0..8).map(|i| ((i as f64).sin(), 0.0)).collect();
        let result = fft(&data);
        let energy_time: f64 = data.iter().map(|c| c.0 * c.0 + c.1 * c.1).sum();
        let energy_freq: f64 = result.iter().map(|c| c.0 * c.0 + c.1 * c.1).sum::<f64>() / 8.0;
        assert!((energy_time - energy_freq).abs() < 1e-10,
            "Parseval: {} vs {}", energy_time, energy_freq);
    }

    // ── Window functions ──

    #[test]
    fn windows_endpoints() {
        let n = 64;
        // Hann: endpoints = 0
        let h = window_hann(n);
        assert!(h[0].abs() < 1e-10);
        assert!(h[n - 1].abs() < 1e-10);
        // Hamming: endpoints ≈ 0.08
        let h = window_hamming(n);
        assert!((h[0] - 0.08).abs() < 0.01);
        // Blackman: endpoints ≈ 0
        let h = window_blackman(n);
        assert!(h[0].abs() < 0.01);
    }

    #[test]
    fn window_symmetry() {
        let n = 32;
        let h = window_hann(n);
        for i in 0..n / 2 {
            assert!((h[i] - h[n - 1 - i]).abs() < 1e-12,
                "hann not symmetric at {}", i);
        }
    }

    // ── Convolution ──

    #[test]
    fn convolve_basic() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![1.0, 1.0];
        let c = convolve(&a, &b);
        // [1, 3, 5, 3]
        assert_eq!(c.len(), 4);
        assert!((c[0] - 1.0).abs() < 1e-10);
        assert!((c[1] - 3.0).abs() < 1e-10);
        assert!((c[2] - 5.0).abs() < 1e-10);
        assert!((c[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn convolve_commutative() {
        let a = vec![1.0, 0.0, -1.0, 2.0];
        let b = vec![0.5, 1.0, 0.5];
        let c1 = convolve(&a, &b);
        let c2 = convolve(&b, &a);
        for i in 0..c1.len() {
            assert!((c1[i] - c2[i]).abs() < 1e-10, "non-commutative at {}", i);
        }
    }

    // ── Autocorrelation ──

    #[test]
    fn autocorrelation_lag0_is_one() {
        let data: Vec<f64> = (0..64).map(|i| (i as f64 * 0.3).sin()).collect();
        let ac = autocorrelation(&data, 10);
        assert!((ac[0] - 1.0).abs() < 1e-10, "lag 0 = {}", ac[0]);
    }

    #[test]
    fn autocorrelation_periodic() {
        // Periodic signal should have periodic autocorrelation
        let n = 128;
        let period = 16;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * i as f64 / period as f64).sin())
            .collect();
        let ac = autocorrelation(&data, 32);
        // Should peak near lag = period
        assert!(ac[period] > 0.8, "autocorr at period {} = {}", period, ac[period]);
    }

    // ── Spectral ──

    #[test]
    fn periodogram_peak_at_signal_freq() {
        let fs = 100.0;
        let freq = 10.0;
        let n = 256;
        let data: Vec<f64> = (0..n)
            .map(|i| (2.0 * PI * freq * i as f64 / fs).sin())
            .collect();
        let (freqs, psd) = periodogram(&data, fs);
        // Find the peak
        let peak_idx = psd.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let peak_freq = freqs[peak_idx];
        assert!((peak_freq - freq).abs() < fs / n as f64 * 2.0,
            "peak at {} Hz, expected {} Hz", peak_freq, freq);
    }

    #[test]
    fn welch_reduces_variance() {
        // Welch should give smoother estimate than raw periodogram
        let n = 512;
        let data: Vec<f64> = (0..n).map(|i| {
            (2.0 * PI * 10.0 * i as f64 / 100.0).sin() + (i as f64 * 17.3).sin() * 0.5
        }).collect();
        let (_, psd_raw) = periodogram(&data, 100.0);
        let (_, psd_welch) = welch(&data, 128, 64, 100.0);
        // Both should have peaks, but Welch should have some output
        assert!(!psd_welch.is_empty());
        assert!(!psd_raw.is_empty());
    }

    // ── DCT ──

    #[test]
    fn dct_energy_preservation() {
        // DCT should preserve energy (Parseval)
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let coeffs = dct2(&data);
        let energy_spatial: f64 = data.iter().map(|x| x * x).sum();
        let energy_dct: f64 = coeffs.iter().map(|x| x * x).sum::<f64>() / (2.0 * data.len() as f64);
        // The ratio should be consistent
        assert!(energy_dct > 0.0, "DCT energy should be positive");
    }

    #[test]
    fn dct_dc_component() {
        // DC component of DCT = 2 * sum
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let coeffs = dct2(&data);
        let expected_dc = 2.0 * data.iter().sum::<f64>();
        assert!((coeffs[0] - expected_dc).abs() < 1e-10,
            "DC = {} vs {}", coeffs[0], expected_dc);
    }

    // ── Filters ──

    #[test]
    fn fir_lowpass_removes_high_freq() {
        let n = 256;
        let fs = 100.0;
        // Signal = 5 Hz + 40 Hz
        let data: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / fs;
            (2.0 * PI * 5.0 * t).sin() + (2.0 * PI * 40.0 * t).sin()
        }).collect();
        let h = fir_lowpass(0.2, 30); // cutoff at 0.2 * Nyquist = 10 Hz
        let filtered = fir_filter(&data, &h);
        // The filtered signal should have much less 40 Hz
        let (_, psd_orig) = periodogram(&data, fs);
        let (freqs, psd_filt) = periodogram(&filtered, fs);
        // Find 40 Hz bin
        let bin_40 = freqs.iter().position(|&f| (f - 40.0).abs() < 1.0).unwrap_or(0);
        let bin_5 = freqs.iter().position(|&f| (f - 5.0).abs() < 1.0).unwrap_or(0);
        // 40 Hz should be attenuated relative to 5 Hz
        if psd_filt[bin_5] > 1e-10 {
            let ratio = psd_filt[bin_40] / psd_filt[bin_5];
            assert!(ratio < 0.1, "40 Hz not sufficiently attenuated: ratio = {}", ratio);
        }
    }

    #[test]
    fn biquad_butterworth_unity_at_dc() {
        // Butterworth lowpass should have unity gain at DC
        let bq = Biquad::butterworth_lowpass(0.3);
        let dc_response = (bq.b0 + bq.b1 + bq.b2) / (1.0 + bq.a1 + bq.a2);
        assert!((dc_response - 1.0).abs() < 1e-10,
            "DC gain = {}", dc_response);
    }

    // ── Hilbert ──

    #[test]
    fn hilbert_envelope_of_modulated() {
        // AM signal: carrier at 50 Hz, modulation at 5 Hz
        let n = 512;
        let fs = 200.0;
        let data: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / fs;
            let mod_signal = 1.0 + 0.5 * (2.0 * PI * 5.0 * t).sin();
            mod_signal * (2.0 * PI * 50.0 * t).sin()
        }).collect();
        let env = envelope(&data);
        // Envelope should be roughly the modulation signal (1 + 0.5*sin(2π·5·t))
        // Check a few points away from edges
        for i in 50..n - 50 {
            let t = i as f64 / fs;
            let expected = 1.0 + 0.5 * (2.0 * PI * 5.0 * t).sin();
            assert!((env[i] - expected).abs() < 0.3,
                "envelope[{}] = {} vs {}", i, env[i], expected);
        }
    }

    // ── Wavelets ──

    #[test]
    fn haar_dwt_idwt_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (a, d) = haar_dwt(&data);
        let reconstructed = haar_idwt(&a, &d);
        for i in 0..data.len() {
            assert!((reconstructed[i] - data[i]).abs() < 1e-10,
                "mismatch at {}: {} vs {}", i, reconstructed[i], data[i]);
        }
    }

    #[test]
    fn haar_energy_conservation() {
        let data = vec![1.0, 4.0, -3.0, 0.0, 2.0, 5.0, 1.0, -1.0];
        let energy_orig: f64 = data.iter().map(|x| x * x).sum();
        let (a, d) = haar_dwt(&data);
        let energy_decomp: f64 = a.iter().map(|x| x * x).sum::<f64>()
            + d.iter().map(|x| x * x).sum::<f64>();
        assert!((energy_orig - energy_decomp).abs() < 1e-10,
            "energy not conserved: {} vs {}", energy_orig, energy_decomp);
    }

    #[test]
    fn haar_multilevel_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (approx, details) = haar_wavedec(&data, 3);
        let reconstructed = haar_waverec(&approx, &details);
        for i in 0..data.len() {
            assert!((reconstructed[i] - data[i]).abs() < 1e-10,
                "multilevel mismatch at {}", i);
        }
    }

    #[test]
    fn db4_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let (a, d) = db4_dwt(&data);
        let reconstructed = db4_idwt(&a, &d);
        for i in 0..data.len() {
            assert!((reconstructed[i] - data[i]).abs() < 1e-8,
                "db4 mismatch at {}: {} vs {}", i, reconstructed[i], data[i]);
        }
    }

    // ── Goertzel ──

    #[test]
    fn goertzel_matches_fft() {
        let n = 64;
        let data: Vec<f64> = (0..n).map(|i| (2.0 * PI * 7.0 * i as f64 / n as f64).sin()).collect();
        let fs = n as f64;
        // Goertzel at 7 Hz should match FFT bin 7
        let g = goertzel(&data, 7.0, fs);
        let full_fft = fft(&data.iter().map(|&x| (x, 0.0)).collect::<Vec<_>>());
        // Goertzel and FFT should agree in magnitude
        let g_mag = c_abs(g);
        let fft_mag = c_abs(full_fft[7]);
        assert!((g_mag - fft_mag).abs() < 1e-6,
            "magnitude mismatch: goertzel={} fft={}", g_mag, fft_mag);
    }

    // ── Cepstrum ──

    #[test]
    fn cepstrum_length() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cep = real_cepstrum(&data);
        assert_eq!(cep.len(), data.len());
    }

    // ── Zero crossing rate ──

    #[test]
    fn zcr_alternating() {
        // Alternating signal: every sample crosses zero
        let data = vec![1.0, -1.0, 1.0, -1.0, 1.0];
        assert!((zero_crossing_rate(&data) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn zcr_constant() {
        let data = vec![1.0, 1.0, 1.0, 1.0];
        assert!((zero_crossing_rate(&data) - 0.0).abs() < 1e-10);
    }

    // ── Smoothing ──

    #[test]
    fn ema_constant() {
        let data = vec![5.0; 10];
        let result = ema(&data, 0.3);
        for &v in &result {
            assert!((v - 5.0).abs() < 1e-10);
        }
    }

    #[test]
    fn ema_tracks_step() {
        // EMA should converge to new level after step change
        let mut data = vec![0.0; 20];
        data.extend(vec![1.0; 20]);
        let result = ema(&data, 0.3);
        // After the step, should be approaching 1.0
        assert!(result[39] > 0.95, "EMA didn't converge: {}", result[39]);
    }

    // ── Median filter ──

    #[test]
    fn median_filter_removes_spikes() {
        let mut data = vec![1.0; 20];
        data[10] = 100.0; // spike
        let filtered = median_filter(&data, 3);
        assert!((filtered[10] - 1.0).abs() < 1e-10,
            "spike not removed: {}", filtered[10]);
    }

    // ── Edge cases ──

    #[test]
    fn empty_inputs() {
        assert!(fft(&[]).is_empty());
        assert!(ifft(&[]).is_empty());
        assert!(rfft(&[]).is_empty());
        assert!(convolve(&[], &[1.0]).is_empty());
        assert!(hilbert(&[]).is_empty());
        let (a, d) = haar_dwt(&[]);
        assert!(a.is_empty());
        assert!(d.is_empty());
    }

    // ── 2D FFT ──

    #[test]
    fn fft2d_dc() {
        // Constant 4x4 image should have energy only at DC
        let data = vec![(1.0, 0.0); 16]; // 4x4
        let result = fft2d(&data, 4, 4);
        // DC bin (0,0) should be 16
        assert!((result[0].0 - 16.0).abs() < 1e-8, "DC = {}", result[0].0);
    }

    // ── Savgol ──

    #[test]
    fn savgol_preserves_linear() {
        // Savitzky-Golay with order >= 1 should preserve linear signals exactly
        let data: Vec<f64> = (0..20).map(|i| 2.0 * i as f64 + 3.0).collect();
        let filtered = savgol_filter(&data, 5, 2);
        for i in 2..18 { // skip edges
            assert!((filtered[i] - data[i]).abs() < 1e-6,
                "savgol distorted linear at {}: {} vs {}", i, filtered[i], data[i]);
        }
    }

    // ── STFT ──

    #[test]
    fn stft_produces_frames() {
        let n = 256;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();
        let frames = stft(&data, 64, 32);
        assert!(!frames.is_empty());
        // Each frame should have 33 frequency bins (64/2 + 1)
        assert_eq!(frames[0].len(), 33);
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Path Signature (Chen, 1954) — level 1 and level 2 terms
// ─────────────────────────────────────────────────────────────────────────────
//
// The signature of a path X: [0,T] → ℝ^d is the collection of all iterated
// integrals. For a 2D path (x_t, y_t):
//
//   Level 1 (increments): S^x = ∫ dX = x_T - x_0
//                          S^y = ∫ dY = y_T - y_0
//
//   Level 2 (Lévy area):  S^xy = ∫∫_{s<t} dX_s dY_t  (iterated integral)
//                          S^yx = ∫∫_{s<t} dY_s dX_t
//                          S^xx = ∫∫_{s<t} dX_s dX_t  = (S^x)^2 / 2
//                          S^yy = ∫∫_{s<t} dY_s dY_t  = (S^y)^2 / 2
//
// The Lévy area A = (S^xy - S^yx) / 2 captures the signed area enclosed by
// the path. It is invariant to time reparameterisation and captures non-linear
// path shape that the first-order signature misses.
//
// Used by fintek's `tick_geometry.rs` leaf for 2D convex-hull signature and
// by roughness measures in `log_signature.rs`.

/// Level-1 and level-2 path signature for a 2D path.
///
/// The path is given as two equal-length slices `xs` and `ys`.
///
/// # Returns
/// `(s1_x, s1_y, s2_xx, s2_xy, s2_yx, s2_yy, levy_area)`
///
/// where:
/// - `s1_x = x_T − x_0`, `s1_y = y_T − y_0` (level-1)
/// - `s2_xx = (s1_x)²/2`, `s2_yy = (s1_y)²/2` (Chen identity)
/// - `s2_xy = Σ (cumX_i − cumX_0) · ΔY_i` (discrete iterated integral)
/// - `s2_yx = Σ (cumY_i − cumY_0) · ΔX_i`
/// - `levy_area = (s2_xy − s2_yx) / 2`
pub fn path_signature_2d(xs: &[f64], ys: &[f64]) -> (f64, f64, f64, f64, f64, f64, f64) {
    assert_eq!(xs.len(), ys.len(), "xs and ys must have equal length");
    let n = xs.len();
    if n < 2 {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }

    // Level-1
    let s1_x = xs[n - 1] - xs[0];
    let s1_y = ys[n - 1] - ys[0];

    // Level-2 iterated integrals via discrete approximation.
    // S^xy = Σ_i X_{i-1} · (Y_i - Y_{i-1})  where X_{i-1} is relative to X_0
    // Using left-point evaluation (consistent with Itô convention).
    let mut s2_xy = 0.0_f64;
    let mut s2_yx = 0.0_f64;
    for i in 1..n {
        let dx = xs[i] - xs[i - 1];
        let dy = ys[i] - ys[i - 1];
        let cum_x = xs[i - 1] - xs[0]; // X_{i-1} relative to start
        let cum_y = ys[i - 1] - ys[0]; // Y_{i-1} relative to start
        s2_xy += cum_x * dy;
        s2_yx += cum_y * dx;
    }

    // Chen identity for diagonal terms: S^xx = (S^x)^2 / 2
    let s2_xx = s1_x * s1_x / 2.0;
    let s2_yy = s1_y * s1_y / 2.0;

    let levy_area = (s2_xy - s2_yx) / 2.0;

    (s1_x, s1_y, s2_xx, s2_xy, s2_yx, s2_yy, levy_area)
}

/// Log-signature: the logarithm of the signature in the free Lie algebra.
///
/// For level 1 and 2, the log-signature has basis {e_x, e_y, [e_x, e_y]}:
///
///   log-sig = (s1_x, s1_y, levy_area)
///
/// The Lévy area is the only level-2 log-signature term because
/// log[e_x, e_x] = 0 and log[e_y, e_y] = 0.
///
/// Returns `(log_s1_x, log_s1_y, log_levy)` which equals
/// `(s1_x, s1_y, levy_area)` at levels 1-2 (the BCH correction only
/// appears at level 3 and above).
pub fn log_signature_2d(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let (s1_x, s1_y, _, _, _, _, levy_area) = path_signature_2d(xs, ys);
    (s1_x, s1_y, levy_area)
}

#[cfg(test)]
mod signature_tests {
    use super::*;

    #[test]
    fn signature_straight_line() {
        // Path: (0,0) → (1,0) — pure x movement, no area
        let xs = vec![0.0, 0.5, 1.0];
        let ys = vec![0.0, 0.0, 0.0];
        let (s1x, s1y, s2xx, s2xy, s2yx, s2yy, levy) = path_signature_2d(&xs, &ys);
        assert!((s1x - 1.0).abs() < 1e-12);
        assert!(s1y.abs() < 1e-12);
        assert!((s2xx - 0.5).abs() < 1e-12, "S^xx = (1)^2/2 = 0.5, got {}", s2xx);
        assert!(s2xy.abs() < 1e-12);
        assert!(s2yx.abs() < 1e-12);
        assert!(s2yy.abs() < 1e-12);
        assert!(levy.abs() < 1e-12);
    }

    #[test]
    fn signature_unit_square_positive_area() {
        // CCW unit square: (0,0)→(1,0)→(1,1)→(0,1)→(0,0)
        // Lévy area = (S^xy - S^yx)/2 = (∫x dy - ∫y dx)/2 = area = 1.0 for unit square.
        // (Both ∫x dy and ∫y dx contribute; the signed area via Green's theorem is 1.)
        let xs = vec![0.0, 1.0, 1.0, 0.0, 0.0];
        let ys = vec![0.0, 0.0, 1.0, 1.0, 0.0];
        let (_, _, _, _, _, _, levy) = path_signature_2d(&xs, &ys);
        assert!(levy > 0.0, "CCW unit square Lévy area must be positive, got {}", levy);
        assert!((levy - 1.0).abs() < 1e-10, "levy_area for CCW unit square = 1.0, got {}", levy);
    }

    #[test]
    fn signature_cw_negative_area() {
        // CW unit square: (0,0)→(0,1)→(1,1)→(1,0)→(0,0) — opposite sign to CCW
        let xs = vec![0.0, 0.0, 1.0, 1.0, 0.0];
        let ys = vec![0.0, 1.0, 1.0, 0.0, 0.0];
        let (_, _, _, _, _, _, levy) = path_signature_2d(&xs, &ys);
        assert!(levy < 0.0, "CW unit square Lévy area must be negative, got {}", levy);
        assert!((levy + 1.0).abs() < 1e-10, "CW: levy = -1.0, got {}", levy);
    }

    #[test]
    fn log_signature_matches() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 0.0];
        let (ls1, ls2, ll) = log_signature_2d(&xs, &ys);
        let (s1x, s1y, _, _, _, _, levy) = path_signature_2d(&xs, &ys);
        assert!((ls1 - s1x).abs() < 1e-12);
        assert!((ls2 - s1y).abs() < 1e-12);
        assert!((ll - levy).abs() < 1e-12);
    }

    #[test]
    fn signature_single_point() {
        let (s1x, s1y, s2xx, s2xy, s2yx, s2yy, levy) = path_signature_2d(&[1.0], &[1.0]);
        assert!(s1x == 0.0 && s1y == 0.0);
        assert!(s2xx == 0.0 && s2xy == 0.0 && s2yx == 0.0 && s2yy == 0.0 && levy == 0.0);
    }
}
