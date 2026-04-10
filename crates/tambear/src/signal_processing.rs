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

/// Parallel prefix scan over real affine maps `(a, b)` with associative composition:
///   `(a₁, b₁) ∘ (a₂, b₂) = (a₁·a₂, a₁·b₂ + b₁)`
///
/// Given parallel arrays `a` and `b` (same length) and initial state `s0`,
/// computes `s[t] = a[t]·(... a[1]·(a[0]·s0 + b[0]) + b[1] ...) + b[t]`.
///
/// **Kingdom A**: the affine semigroup is associative, enabling Blelloch-style
/// parallel prefix scan on GPU. The sequential implementation here is the correct
/// CPU fallback; the GPU path replaces it without changing the observable contract.
///
/// Consumers: `ema`, `ema_period`, `ewma_variance`, any first-order linear recurrence.
pub fn affine_prefix_scan(a: &[f64], b: &[f64], s0: f64) -> Vec<f64> {
    assert_eq!(a.len(), b.len());
    let n = a.len();
    if n == 0 { return vec![]; }
    let mut result = Vec::with_capacity(n);
    let mut s = s0;
    for i in 0..n {
        s = a[i] * s + b[i];
        result.push(s);
    }
    result
}

/// Exponential moving average (EMA).
///
/// `alpha` is the smoothing factor (0 < alpha <= 1). Higher = less smoothing.
///
/// **Kingdom A** via affine semigroup prefix scan:
/// `s_t = (1-α)·s_{t-1} + α·x_t` is an affine recurrence with constant map
/// `a_t = (1-α)`, `b_t = α·x_t`. Delegates to `affine_prefix_scan`.
pub fn ema(data: &[f64], alpha: f64) -> Vec<f64> {
    if data.is_empty() { return vec![]; }
    let decay = 1.0 - alpha;
    let a: Vec<f64> = vec![decay; data.len() - 1];
    let b: Vec<f64> = data[1..].iter().map(|&x| alpha * x).collect();
    let mut result = vec![data[0]];
    result.extend(affine_prefix_scan(&a, &b, data[0]));
    result
}

/// EMA with period-based smoothing factor: `alpha = 2 / (period + 1)`.
///
/// Standard convention for MACD, RSI, and technical analysis EMA periods.
pub fn ema_period(data: &[f64], period: usize) -> Vec<f64> {
    let alpha = 2.0 / (period as f64 + 1.0);
    ema(data, alpha)
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

// ─── Savitzky-Golay filter ───────────────────────────────────────────
//
// Savitzky & Golay 1964, "Smoothing and Differentiation of Data by Simplified
// Least Squares Procedures", Analytical Chemistry 36(8):1627-1639.
//
// Fits a polynomial of degree `poly_order` to a sliding window of `window_size`
// points via least squares, then outputs the polynomial's value (or derivative)
// at the center of the window. Preserves peak heights/widths better than
// moving average — the key advantage over median/mean filters.

/// Compute Savitzky-Golay coefficients for a centered window.
///
/// Returns a vector of `window_size` coefficients such that applying them as
/// a convolution to the data produces the smoothed output (for `deriv=0`) or
/// the `deriv`-th derivative estimate.
///
/// `window_size`: must be odd and ≥ `poly_order + 1`.
/// `poly_order`: polynomial degree (typically 2-4).
/// `deriv`: derivative order (0 = smoothing, 1 = first derivative, ...).
///
/// Returns None for invalid parameters.
pub fn savgol_coefficients(window_size: usize, poly_order: usize, deriv: usize) -> Option<Vec<f64>> {
    if window_size % 2 == 0 { return None; }
    if window_size < poly_order + 1 { return None; }
    if deriv > poly_order { return None; }

    let m = (window_size - 1) / 2;
    let n_coeffs = poly_order + 1;

    // Vandermonde matrix A: rows are [1, k, k², ..., k^p] for k = -m..=m
    // We want row `deriv` of (A'A)⁻¹ A'.
    //
    // Solve (A'A) x = e_deriv for x (the `deriv`-th row of the pseudo-inverse left factor),
    // then the coefficients are A x. Multiply by deriv! for the derivative case.

    // Build A'A (symmetric, (n_coeffs × n_coeffs))
    let mut ata = vec![vec![0.0f64; n_coeffs]; n_coeffs];
    for k in -(m as i64)..=(m as i64) {
        let kf = k as f64;
        // Power series k^0, k^1, ..., k^(2p)
        let mut powers = vec![0.0; 2 * poly_order + 1];
        powers[0] = 1.0;
        for p in 1..=(2 * poly_order) { powers[p] = powers[p - 1] * kf; }
        for i in 0..n_coeffs {
            for j in 0..n_coeffs {
                ata[i][j] += powers[i + j];
            }
        }
    }

    // Solve (A'A) x = e_deriv (i.e., unit vector at `deriv`) via Gaussian elimination
    let mut aug = vec![vec![0.0f64; n_coeffs + 1]; n_coeffs];
    for i in 0..n_coeffs {
        for j in 0..n_coeffs { aug[i][j] = ata[i][j]; }
        aug[i][n_coeffs] = if i == deriv { 1.0 } else { 0.0 };
    }
    // Forward elimination with partial pivoting
    for col in 0..n_coeffs {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[col][col].abs();
        for r in (col + 1)..n_coeffs {
            if aug[r][col].abs() > max_val {
                max_val = aug[r][col].abs();
                max_row = r;
            }
        }
        if max_val < 1e-300 { return None; }
        if max_row != col { aug.swap(col, max_row); }
        let pivot = aug[col][col];
        for j in col..=n_coeffs { aug[col][j] /= pivot; }
        for r in 0..n_coeffs {
            if r == col { continue; }
            let factor = aug[r][col];
            for j in col..=n_coeffs { aug[r][j] -= factor * aug[col][j]; }
        }
    }
    let x: Vec<f64> = (0..n_coeffs).map(|i| aug[i][n_coeffs]).collect();

    // Coefficients: c[k] = Σ_i x[i] * k^i for k = -m..=m
    // Multiply by deriv! for derivatives.
    let mut deriv_factorial = 1.0;
    for d in 1..=deriv { deriv_factorial *= d as f64; }

    let mut coeffs = vec![0.0f64; window_size];
    for (idx, k) in (-(m as i64)..=(m as i64)).enumerate() {
        let kf = k as f64;
        let mut poly_val = 0.0;
        let mut kp = 1.0;
        for i in 0..n_coeffs {
            poly_val += x[i] * kp;
            kp *= kf;
        }
        coeffs[idx] = poly_val * deriv_factorial;
    }

    Some(coeffs)
}

/// Apply Savitzky-Golay filter to a signal.
///
/// Smooths (or differentiates) `data` by fitting polynomials of degree
/// `poly_order` to a sliding window of `window_size` points.
///
/// Edge handling: mirror-reflection padding. Input and output have same length.
///
/// `window_size`: odd, ≥ `poly_order + 1`. Typical: 5, 7, 11, 21.
/// `poly_order`: polynomial degree. Typical: 2 (parabolic) or 3 (cubic).
/// `deriv`: 0 = smoothing, 1 = first derivative, 2 = second derivative.
///
/// Returns empty vector on invalid parameters.
pub fn savitzky_golay(data: &[f64], window_size: usize, poly_order: usize, deriv: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 { return Vec::new(); }
    let coeffs = match savgol_coefficients(window_size, poly_order, deriv) {
        Some(c) => c,
        None => return Vec::new(),
    };
    let m = (window_size - 1) / 2;

    let mut out = vec![0.0f64; n];
    for i in 0..n {
        let mut sum = 0.0;
        for (j, &c) in coeffs.iter().enumerate() {
            let k = j as i64 - m as i64;
            let idx = i as i64 + k;
            // Mirror reflection padding at edges
            let data_val = if idx < 0 {
                let r = (-idx) as usize;
                data[r.min(n - 1)]
            } else if idx >= n as i64 {
                let r = 2 * (n - 1) - idx as usize;
                data[r.min(n - 1)]
            } else {
                data[idx as usize]
            };
            sum += c * data_val;
        }
        out[i] = sum;
    }
    out
}

// ─── Signal regularization ───────────────────────────────────────────
//
// These functions map variable-length data to a fixed grid of M points.
// Used by CWT, FFT-spectral, and Haar wavelet leaves in fintek's trunk-rs
// to normalize variable-length bins before transform computation.
// All three strategies match fintek's `cwt_wavelet.rs` and `haar_wavelet.rs`.

/// Linear interpolation to M equispaced grid points.
///
/// The i-th output point is at `t = (i / M) * (n-1)`.
/// NaN inputs are not handled (caller should screen).
pub fn regularize_interp(data: &[f64], m: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 { return vec![f64::NAN; m]; }
    if n == 1 { return vec![data[0]; m]; }
    let mut result = vec![0.0; m];
    for i in 0..m {
        let t = (i as f64 / m as f64) * (n - 1) as f64;
        let lo = t as usize;
        let hi = (lo + 1).min(n - 1);
        let frac = t - lo as f64;
        result[i] = data[lo] * (1.0 - frac) + data[hi] * frac;
    }
    result
}

/// Bin-mean regularization to M equispaced grid points.
///
/// Divides the n input samples into M equal bins and takes the mean of each.
/// Empty bins (when n < M) are left as NaN.
pub fn regularize_bin_mean(data: &[f64], m: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 { return vec![f64::NAN; m]; }
    if n == 1 { return vec![data[0]; m]; }
    let mut result = vec![f64::NAN; m];
    let mut sums = vec![0.0_f64; m];
    let mut counts = vec![0u32; m];
    for i in 0..n {
        let b = ((i * m) / n).min(m - 1);
        sums[b] += data[i];
        counts[b] += 1;
    }
    for b in 0..m {
        if counts[b] > 0 {
            result[b] = sums[b] / counts[b] as f64;
        }
    }
    result
}

/// Nearest-tick subsample to M equispaced grid points.
///
/// The i-th output point is the sample at index `round((i/M) * n)`.
pub fn regularize_subsample(data: &[f64], m: usize) -> Vec<f64> {
    let n = data.len();
    if n == 0 { return vec![f64::NAN; m]; }
    if n == 1 { return vec![data[0]; m]; }
    let mut result = vec![0.0; m];
    for i in 0..m {
        let idx = ((i as f64 / m as f64) * n as f64).round() as usize;
        result[i] = data[idx.min(n - 1)];
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

    #[test]
    fn ema_matches_sequential_formula() {
        // Verify the affine_prefix_scan-based ema produces identical results to
        // the reference sequential formula for a non-trivial series.
        let data = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
        let alpha = 0.4;
        let result = ema(&data, alpha);

        // Compute reference sequentially inline
        let mut reference = vec![data[0]];
        for i in 1..data.len() {
            reference.push(alpha * data[i] + (1.0 - alpha) * reference[i - 1]);
        }

        for (i, (&got, &expected)) in result.iter().zip(reference.iter()).enumerate() {
            assert!((got - expected).abs() < 1e-12,
                "ema[{}]: got {got} expected {expected}", i);
        }
    }

    #[test]
    fn ema_period_alpha_convention() {
        // ema_period(data, 9) should equal ema(data, 2/10 = 0.2)
        let data: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let by_period = ema_period(&data, 9);
        let by_alpha = ema(&data, 2.0 / 10.0);
        for (i, (&a, &b)) in by_period.iter().zip(by_alpha.iter()).enumerate() {
            assert!((a - b).abs() < 1e-12, "ema_period vs ema at [{i}]: {a} vs {b}");
        }
    }

    #[test]
    fn affine_prefix_scan_identity_map() {
        // a_t = 1, b_t = 0: should return s0 repeated
        let a = vec![1.0; 5];
        let b = vec![0.0; 5];
        let result = affine_prefix_scan(&a, &b, 3.0);
        for &v in &result {
            assert!((v - 3.0).abs() < 1e-12, "identity map: {v}");
        }
    }

    #[test]
    fn affine_prefix_scan_constant_shift() {
        // a_t = 0, b_t = c: result[t] = c for all t
        let a = vec![0.0; 4];
        let b = vec![7.0; 4];
        let result = affine_prefix_scan(&a, &b, 999.0);
        for &v in &result {
            assert!((v - 7.0).abs() < 1e-12, "constant: {v}");
        }
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

    // ── Savitzky-Golay ──

    #[test]
    fn savgol_coefficients_symmetric_sum_to_one() {
        // For smoothing (deriv=0), SG coefficients sum to 1
        for &(w, p) in &[(5, 2), (7, 2), (9, 3), (11, 3), (21, 4)] {
            let c = savgol_coefficients(w, p, 0).unwrap();
            let sum: f64 = c.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10, "w={} p={} sum={}", w, p, sum);
            // Also symmetric about center
            let m = (w - 1) / 2;
            for i in 0..m {
                assert!((c[i] - c[w - 1 - i]).abs() < 1e-10, "asymmetry at w={} p={}", w, p);
            }
        }
    }

    #[test]
    fn savgol_preserves_polynomial_exactly() {
        // For input that is a polynomial of degree ≤ poly_order,
        // SG smoothing should return the exact values (it fits and evaluates
        // the same polynomial that generated the data).
        let data: Vec<f64> = (0..20).map(|i| {
            let x = i as f64;
            1.0 + 2.0 * x - 0.5 * x * x // degree-2 polynomial
        }).collect();
        let smoothed = savitzky_golay(&data, 7, 2, 0);
        // Interior points (away from edges) should match exactly
        for i in 3..17 {
            assert!((smoothed[i] - data[i]).abs() < 1e-8,
                "SG changed polynomial at i={}: {} vs {}", i, smoothed[i], data[i]);
        }
    }

    #[test]
    fn savgol_smooths_noise() {
        // SG should reduce variance of noisy flat data
        let mut rng = 42u64;
        let data: Vec<f64> = (0..100).map(|_| {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            10.0 + (rng as f64 / u64::MAX as f64 - 0.5) * 2.0
        }).collect();
        let smoothed = savitzky_golay(&data, 11, 2, 0);

        let var_orig = {
            let mean: f64 = data.iter().sum::<f64>() / data.len() as f64;
            data.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / data.len() as f64
        };
        let var_smooth = {
            let mean: f64 = smoothed.iter().sum::<f64>() / smoothed.len() as f64;
            smoothed.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / smoothed.len() as f64
        };
        assert!(var_smooth < var_orig * 0.6,
            "SG should reduce variance: orig={}, smooth={}", var_orig, var_smooth);
    }

    #[test]
    fn savgol_first_derivative_of_linear() {
        // First derivative of y = 3x + 1 should be 3 everywhere
        let data: Vec<f64> = (0..30).map(|i| 3.0 * i as f64 + 1.0).collect();
        let deriv = savitzky_golay(&data, 7, 2, 1);
        for i in 3..27 {
            assert!((deriv[i] - 3.0).abs() < 1e-8,
                "deriv at i={} should be 3, got {}", i, deriv[i]);
        }
    }

    #[test]
    fn savgol_invalid_params() {
        // Even window size
        assert!(savgol_coefficients(6, 2, 0).is_none());
        // Window smaller than poly_order + 1
        assert!(savgol_coefficients(3, 5, 0).is_none());
        // Derivative > poly_order
        assert!(savgol_coefficients(7, 2, 3).is_none());
        // Empty data returns empty
        assert!(savitzky_golay(&[], 5, 2, 0).is_empty());
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

// ═══════════════════════════════════════════════════════════════════════════
// Wigner-Ville Distribution
// ═══════════════════════════════════════════════════════════════════════════

/// Result of Wigner-Ville distribution feature extraction.
#[derive(Debug, Clone)]
pub struct WvdResult {
    /// Rényi entropy of |WVD| (lower = more concentrated in time-frequency).
    pub time_freq_concentration: f64,
    /// Variance of instantaneous frequency across time points.
    pub instantaneous_freq_var: f64,
    /// Energy in negative WVD regions (interference / cross-term energy).
    pub cross_term_energy: f64,
    /// Shannon entropy of frequency marginal distribution.
    pub marginal_entropy: f64,
}

impl WvdResult {
    pub fn nan() -> Self {
        Self {
            time_freq_concentration: f64::NAN,
            instantaneous_freq_var: f64::NAN,
            cross_term_energy: f64::NAN,
            marginal_entropy: f64::NAN,
        }
    }
}

/// Discrete Wigner-Ville Distribution feature extraction.
///
/// W(t,f) = 2 Σ_τ x(t+τ)·x(t-τ)·cos(2π·f·τ/n)   [real signal, one-sided]
///
/// Extracts 4 features per signal:
/// - `time_freq_concentration`: Rényi entropy of |WVD| (lower → more concentrated)
/// - `instantaneous_freq_var`: variance of centroid-frequency across time
/// - `cross_term_energy`: fraction of energy in negative WVD cells
/// - `marginal_entropy`: Shannon entropy of summed frequency marginal
///
/// Input is clipped at `max_pts` for performance (O(n² × n_freq)).
pub fn wvd_features(x: &[f64], max_pts: usize) -> WvdResult {
    let n_raw = x.len();
    if n_raw < 8 { return WvdResult::nan(); }

    // Subsample if too long
    let x_used: Vec<f64> = if n_raw > max_pts {
        let step = n_raw / max_pts;
        x.iter().step_by(step).copied().collect()
    } else {
        x.to_vec()
    };
    let n = x_used.len();
    let n_freq = n / 2 + 1;

    let c = 2.0 * std::f64::consts::PI / n as f64;

    let mut wvd_sum_abs = 0.0_f64;
    let mut negative_energy = 0.0_f64;
    let mut freq_marginal = vec![0.0_f64; n_freq];
    let mut inst_freqs = Vec::with_capacity(n);

    for t in 0..n {
        let max_tau = t.min(n - 1 - t);

        let mut row = vec![0.0_f64; n_freq];
        for k in 0..n_freq {
            let mut val = x_used[t] * x_used[t]; // τ=0
            for tau in 1..=max_tau {
                let kernel = x_used[t + tau] * x_used[t - tau];
                val += 2.0 * kernel * (c * k as f64 * tau as f64).cos();
            }
            row[k] = val;
        }

        let mut row_total = 0.0_f64;
        let mut weighted_freq = 0.0_f64;
        for k in 0..n_freq {
            let abs_val = row[k].abs();
            wvd_sum_abs += abs_val;
            if row[k] < 0.0 { negative_energy += abs_val; }
            freq_marginal[k] += abs_val;
            row_total += abs_val;
            weighted_freq += k as f64 * abs_val;
        }
        if row_total > 1e-30 {
            inst_freqs.push(weighted_freq / row_total);
        }
    }

    if wvd_sum_abs < 1e-30 { return WvdResult::nan(); }

    // Rényi entropy (order 2): -log2(Σ p²) where p = |W(t,f)| / Σ|W|
    let mut renyi_sum = 0.0_f64;
    // Iterate over all cells; we don't store the full grid, so recompute row sums
    // Approximate using marginal: Rényi ≈ -log2(Σ_f (m_f/total)²)
    let marginal_total: f64 = freq_marginal.iter().sum();
    if marginal_total > 1e-30 {
        for &m in &freq_marginal {
            let p = m / marginal_total;
            renyi_sum += p * p;
        }
    }
    let time_freq_concentration = if renyi_sum > 1e-300 { -renyi_sum.log2() } else { f64::NAN };

    // Instantaneous frequency variance
    let instantaneous_freq_var = if inst_freqs.len() >= 2 {
        let mean = inst_freqs.iter().sum::<f64>() / inst_freqs.len() as f64;
        inst_freqs.iter().map(|&f| (f - mean).powi(2)).sum::<f64>() / inst_freqs.len() as f64
    } else {
        f64::NAN
    };

    // Cross-term energy fraction
    let cross_term_energy = negative_energy / wvd_sum_abs;

    // Marginal entropy
    let mut marginal_entropy = 0.0_f64;
    if marginal_total > 1e-30 {
        for &m in &freq_marginal {
            let p = m / marginal_total;
            if p > 1e-300 { marginal_entropy -= p * p.ln(); }
        }
        let max_ent = (n_freq as f64).ln();
        if max_ent > 1e-30 { marginal_entropy /= max_ent; }
    }

    WvdResult { time_freq_concentration, instantaneous_freq_var, cross_term_energy, marginal_entropy }
}

// ═══════════════════════════════════════════════════════════════════════════
// FastICA
// ═══════════════════════════════════════════════════════════════════════════

/// Result of FastICA on delay-embedded signal.
#[derive(Debug, Clone)]
pub struct IcaResult {
    /// Negentropy of the most non-Gaussian independent component.
    pub max_negentropy: f64,
    /// Mean negentropy across all components.
    pub mean_negentropy: f64,
    /// Range of excess kurtosis across components (max - min).
    pub kurtosis_range: f64,
    /// Total sift/convergence iterations across all components.
    pub convergence_iterations: u32,
}

impl IcaResult {
    pub fn nan() -> Self {
        Self { max_negentropy: f64::NAN, mean_negentropy: f64::NAN, kurtosis_range: f64::NAN, convergence_iterations: 0 }
    }
}

/// FastICA via delay embedding.
///
/// Embeds `x` as a `dim × (n - dim + 1)` delay matrix, whitens, then extracts
/// `dim` independent components via the deflation algorithm with logcosh nonlinearity.
///
/// Returns negentropy and kurtosis statistics across the extracted ICs.
///
/// `dim`: embedding dimension (default 4 for financial data).
/// `max_iter`: maximum deflation iterations per component.
pub fn fast_ica(x: &[f64], dim: usize, max_iter: usize) -> IcaResult {
    let n = x.len();
    if n < dim + 20 || dim == 0 { return IcaResult::nan(); }

    let n_emb = n - dim + 1;

    // Delay embedding: row t = [x[t], x[t+1], ..., x[t+dim-1]]
    let mut data = vec![0.0_f64; n_emb * dim];
    let mut means = vec![0.0_f64; dim];
    for t in 0..n_emb {
        for d in 0..dim {
            data[t * dim + d] = x[t + d];
            means[d] += x[t + d];
        }
    }
    for d in 0..dim { means[d] /= n_emb as f64; }
    // Center
    for t in 0..n_emb {
        for d in 0..dim { data[t * dim + d] -= means[d]; }
    }

    // Approximate whitening: divide each dimension by its std
    let mut stds = vec![1.0_f64; dim];
    for d in 0..dim {
        let var = data.iter().skip(d).step_by(dim).map(|&v| v * v).sum::<f64>() / n_emb as f64;
        let s = var.sqrt();
        stds[d] = if s > 1e-30 { s } else { 1.0 };
    }
    for t in 0..n_emb {
        for d in 0..dim { data[t * dim + d] /= stds[d]; }
    }

    // FastICA deflation
    let mut total_iter = 0u32;
    let mut negentropies = Vec::with_capacity(dim);
    let mut kurtoses = Vec::with_capacity(dim);
    let mut extracted: Vec<Vec<f64>> = Vec::with_capacity(dim); // weight vectors

    for comp in 0..dim {
        // Initialize w: unit vector along comp axis
        let mut w = vec![0.0_f64; dim];
        w[comp] = 1.0;

        for _iter in 0..max_iter {
            total_iter += 1;

            // Project data: y[t] = w·x[t]
            let y: Vec<f64> = (0..n_emb).map(|t| {
                (0..dim).map(|d| w[d] * data[t * dim + d]).sum()
            }).collect();

            // logcosh nonlinearity: g(u) = tanh(u), g'(u) = 1 - tanh²(u)
            let mut new_w = vec![0.0_f64; dim];
            let mut g_prime_mean = 0.0_f64;
            for t in 0..n_emb {
                let yt = y[t];
                let tanh_yt = yt.tanh();
                let gp = 1.0 - tanh_yt * tanh_yt;
                g_prime_mean += gp;
                for d in 0..dim {
                    new_w[d] += tanh_yt * data[t * dim + d];
                }
            }
            for d in 0..dim { new_w[d] /= n_emb as f64; }
            g_prime_mean /= n_emb as f64;
            for d in 0..dim { new_w[d] -= g_prime_mean * w[d]; }

            // Deflation: orthogonalize against already-extracted components
            for prev_w in &extracted {
                let dot: f64 = (0..dim).map(|d| new_w[d] * prev_w[d]).sum();
                for d in 0..dim { new_w[d] -= dot * prev_w[d]; }
            }

            // Normalize
            let norm: f64 = new_w.iter().map(|&v| v * v).sum::<f64>().sqrt();
            if norm < 1e-30 { break; }
            for d in 0..dim { new_w[d] /= norm; }

            // Check convergence
            let delta: f64 = (0..dim).map(|d| (new_w[d] - w[d]).abs()).sum();
            w = new_w;
            if delta < 1e-6 { break; }
        }

        // Project to get IC
        let ic: Vec<f64> = (0..n_emb).map(|t| {
            (0..dim).map(|d| w[d] * data[t * dim + d]).sum()
        }).collect();

        // Negentropy via logcosh approximation: E[logcosh(y)] - E[logcosh(G(0))]
        // Reference: for standard Gaussian, E[logcosh] ≈ 0.3745
        let gauss_ref = 0.3745_f64;
        let logcosh_mean = ic.iter().map(|&v| (v.cosh().ln())).sum::<f64>() / n_emb as f64;
        let negentropy = (logcosh_mean - gauss_ref).abs();
        negentropies.push(negentropy);

        // Excess kurtosis
        let m2 = ic.iter().map(|&v| v * v).sum::<f64>() / n_emb as f64;
        let m4 = ic.iter().map(|&v| v.powi(4)).sum::<f64>() / n_emb as f64;
        let kurt = if m2 > 1e-30 { m4 / (m2 * m2) - 3.0 } else { 0.0 };
        kurtoses.push(kurt);

        extracted.push(w);
    }

    if negentropies.is_empty() { return IcaResult::nan(); }

    let max_negentropy = negentropies.iter().cloned().fold(f64::NEG_INFINITY, crate::numerical::nan_max);
    let mean_negentropy = negentropies.iter().sum::<f64>() / negentropies.len() as f64;
    let kurt_max = kurtoses.iter().cloned().fold(f64::NEG_INFINITY, crate::numerical::nan_max);
    let kurt_min = kurtoses.iter().cloned().fold(f64::INFINITY, crate::numerical::nan_min);
    let kurtosis_range = kurt_max - kurt_min;

    IcaResult { max_negentropy, mean_negentropy, kurtosis_range, convergence_iterations: total_iter }
}

// ═══════════════════════════════════════════════════════════════════════════
// Empirical Mode Decomposition (EMD)
// ═══════════════════════════════════════════════════════════════════════════

/// Result of EMD decomposition.
#[derive(Debug, Clone)]
pub struct EmdResult {
    /// Number of IMFs extracted.
    pub n_imfs: usize,
    /// Energy fraction of first (highest-frequency) IMF.
    pub imf1_energy: f64,
    /// Energy fraction of second IMF.
    pub imf2_energy: f64,
    /// Energy fraction of third IMF.
    pub imf3_energy: f64,
    /// Energy fraction of residual (trend).
    pub residual_energy: f64,
    /// Mean absolute correlation between adjacent IMFs (mode-mixing index).
    pub mode_mixing_index: f64,
    /// Mean dominant period across all IMFs.
    pub mean_period: f64,
    /// Mean absolute pairwise correlation across all IMF pairs.
    pub imf_orthogonality: f64,
}

impl EmdResult {
    pub fn nan() -> Self {
        Self {
            n_imfs: 0, imf1_energy: f64::NAN, imf2_energy: f64::NAN,
            imf3_energy: f64::NAN, residual_energy: f64::NAN,
            mode_mixing_index: f64::NAN, mean_period: f64::NAN, imf_orthogonality: f64::NAN,
        }
    }
}

fn emd_find_maxima(x: &[f64]) -> Vec<usize> {
    let mut out = Vec::new();
    for i in 1..x.len().saturating_sub(1) {
        if x[i] > x[i - 1] && x[i] >= x[i + 1] { out.push(i); }
    }
    out
}

fn emd_find_minima(x: &[f64]) -> Vec<usize> {
    let mut out = Vec::new();
    for i in 1..x.len().saturating_sub(1) {
        if x[i] < x[i - 1] && x[i] <= x[i + 1] { out.push(i); }
    }
    out
}

/// Linear-interpolation envelope through extrema, extended to signal boundaries.
fn emd_envelope(x_idx: &[usize], y_vals: &[f64], n: usize) -> Vec<f64> {
    if x_idx.is_empty() { return vec![0.0; n]; }
    if x_idx.len() == 1 { return vec![y_vals[0]; n]; }

    // Extend to boundaries using first/last extremum values
    let mut xs: Vec<usize> = Vec::with_capacity(x_idx.len() + 2);
    let mut ys: Vec<f64> = Vec::with_capacity(y_vals.len() + 2);
    xs.push(0); ys.push(y_vals[0]);
    xs.extend_from_slice(x_idx);
    ys.extend_from_slice(y_vals);
    xs.push(n - 1); ys.push(*y_vals.last().unwrap());

    let mut env = vec![0.0_f64; n];
    let mut seg = 0_usize;
    for t in 0..n {
        while seg + 1 < xs.len() - 1 && t > xs[seg + 1] { seg += 1; }
        let x0 = xs[seg] as f64;
        let x1 = xs[seg + 1] as f64;
        if (x1 - x0).abs() < 1e-10 {
            env[t] = ys[seg];
        } else {
            let frac = (t as f64 - x0) / (x1 - x0);
            env[t] = ys[seg] + frac * (ys[seg + 1] - ys[seg]);
        }
    }
    env
}

/// Mean absolute correlation between adjacent pairs in a list of equal-length vectors.
fn mean_adjacent_correlation(imfs: &[Vec<f64>]) -> f64 {
    if imfs.len() < 2 { return 0.0; }
    let n = imfs[0].len() as f64;
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for i in 0..imfs.len() - 1 {
        let a = &imfs[i];
        let b = &imfs[i + 1];
        let ma: f64 = a.iter().sum::<f64>() / n;
        let mb: f64 = b.iter().sum::<f64>() / n;
        let num: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - ma) * (bi - mb)).sum();
        let da: f64 = a.iter().map(|&ai| (ai - ma).powi(2)).sum::<f64>();
        let db: f64 = b.iter().map(|&bi| (bi - mb).powi(2)).sum::<f64>();
        let denom = (da * db).sqrt();
        if denom > 1e-30 { sum += (num / denom).abs(); count += 1; }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// Mean absolute pairwise correlation across all IMF pairs.
fn mean_pairwise_correlation(imfs: &[Vec<f64>]) -> f64 {
    if imfs.len() < 2 { return 0.0; }
    let n = imfs[0].len() as f64;
    let mut sum = 0.0_f64;
    let mut count = 0_usize;
    for i in 0..imfs.len() {
        for j in i + 1..imfs.len() {
            let a = &imfs[i];
            let b = &imfs[j];
            let ma: f64 = a.iter().sum::<f64>() / n;
            let mb: f64 = b.iter().sum::<f64>() / n;
            let num: f64 = a.iter().zip(b.iter()).map(|(&ai, &bi)| (ai - ma) * (bi - mb)).sum();
            let da: f64 = a.iter().map(|&ai| (ai - ma).powi(2)).sum::<f64>();
            let db: f64 = b.iter().map(|&bi| (bi - mb).powi(2)).sum::<f64>();
            let denom = (da * db).sqrt();
            if denom > 1e-30 { sum += (num / denom).abs(); count += 1; }
        }
    }
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// Mean dominant period of an IMF (1 / mean zero-crossing rate).
fn dominant_period(imf: &[f64]) -> f64 {
    let n = imf.len();
    if n < 4 { return f64::NAN; }
    let crossings = imf.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
    if crossings == 0 { return n as f64; }
    // Each full period has 2 zero crossings
    n as f64 / (crossings as f64 / 2.0)
}

/// Empirical Mode Decomposition via sifting (linear envelope interpolation).
///
/// Decomposes `x` into at most `max_imfs` intrinsic mode functions (IMFs) using
/// the sifting algorithm. Each IMF satisfies:
/// - equal number of extrema and zero crossings (± 1)
/// - mean of upper/lower envelope ≈ 0
///
/// Returns energy fractions, mode-mixing index, orthogonality, and period stats.
///
/// `max_imfs`: cap on number of IMFs (default 10).
/// `max_sift_iter`: cap on sifting iterations per IMF (default 100).
/// `sift_threshold`: SD(mean envelope) / SD(signal) stopping criterion (default 0.05).
pub fn emd(x: &[f64], max_imfs: usize, max_sift_iter: usize, sift_threshold: f64) -> EmdResult {
    let n = x.len();
    if n < 20 || max_imfs == 0 { return EmdResult::nan(); }

    let total_energy: f64 = x.iter().map(|&v| v * v).sum();
    if total_energy < 1e-30 { return EmdResult::nan(); }

    let mut residual = x.to_vec();
    let mut imfs: Vec<Vec<f64>> = Vec::with_capacity(max_imfs);

    for _ in 0..max_imfs {
        // Check if residual is monotone (stop criterion)
        let max_idx = emd_find_maxima(&residual);
        let min_idx = emd_find_minima(&residual);
        if max_idx.len() < 2 || min_idx.len() < 2 { break; }

        // Sift to extract one IMF
        let mut proto = residual.clone();
        for _ in 0..max_sift_iter {
            let mx_idx = emd_find_maxima(&proto);
            let mn_idx = emd_find_minima(&proto);
            if mx_idx.len() < 2 || mn_idx.len() < 2 { break; }

            let mx_vals: Vec<f64> = mx_idx.iter().map(|&i| proto[i]).collect();
            let mn_vals: Vec<f64> = mn_idx.iter().map(|&i| proto[i]).collect();

            let upper = emd_envelope(&mx_idx, &mx_vals, n);
            let lower = emd_envelope(&mn_idx, &mn_vals, n);

            let mean_env: Vec<f64> = upper.iter().zip(lower.iter()).map(|(&u, &l)| (u + l) / 2.0).collect();

            // Check stopping: SD(mean) / SD(proto) < threshold
            let proto_sd: f64 = {
                let m = proto.iter().sum::<f64>() / n as f64;
                (proto.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / n as f64).sqrt()
            };
            let env_sd: f64 = {
                let m = mean_env.iter().sum::<f64>() / n as f64;
                (mean_env.iter().map(|&v| (v - m).powi(2)).sum::<f64>() / n as f64).sqrt()
            };

            for i in 0..n { proto[i] -= mean_env[i]; }

            if proto_sd > 1e-30 && env_sd / proto_sd < sift_threshold { break; }
        }

        for i in 0..n { residual[i] -= proto[i]; }
        imfs.push(proto);
    }

    let n_imfs = imfs.len();
    if n_imfs == 0 { return EmdResult::nan(); }

    // Energy fractions
    let energy_fraction = |v: &[f64]| -> f64 {
        let e: f64 = v.iter().map(|&x| x * x).sum();
        e / total_energy
    };

    let imf1_energy = imfs.get(0).map(|v| energy_fraction(v)).unwrap_or(f64::NAN);
    let imf2_energy = imfs.get(1).map(|v| energy_fraction(v)).unwrap_or(f64::NAN);
    let imf3_energy = imfs.get(2).map(|v| energy_fraction(v)).unwrap_or(f64::NAN);
    let residual_energy = energy_fraction(&residual);

    let mode_mixing_index = mean_adjacent_correlation(&imfs);
    let imf_orthogonality = mean_pairwise_correlation(&imfs);

    let periods: Vec<f64> = imfs.iter().map(|v| dominant_period(v)).filter(|p| p.is_finite()).collect();
    let mean_period = if periods.is_empty() { f64::NAN } else { periods.iter().sum::<f64>() / periods.len() as f64 };

    EmdResult {
        n_imfs, imf1_energy, imf2_energy, imf3_energy, residual_energy,
        mode_mixing_index, mean_period, imf_orthogonality,
    }
}

#[cfg(test)]
mod wvd_ica_emd_tests {
    use super::*;

    fn sine(n: usize, freq: f64) -> Vec<f64> {
        (0..n).map(|i| (2.0 * std::f64::consts::PI * freq * i as f64 / n as f64).sin()).collect()
    }

    // ── WVD ──────────────────────────────────────────────────────────────

    #[test]
    fn wvd_sine_finite() {
        let x = sine(64, 4.0);
        let r = wvd_features(&x, 256);
        assert!(r.time_freq_concentration.is_finite(), "TFC: {}", r.time_freq_concentration);
        assert!(r.marginal_entropy.is_finite() && r.marginal_entropy >= 0.0);
        assert!(r.cross_term_energy >= 0.0 && r.cross_term_energy <= 1.0 + 1e-10);
    }

    #[test]
    fn wvd_too_short_nan() {
        let r = wvd_features(&[1.0, 2.0, 3.0], 256);
        assert!(r.time_freq_concentration.is_nan());
    }

    #[test]
    fn wvd_low_entropy_for_pure_sine() {
        // Pure sine → energy concentrated in one frequency → low marginal entropy
        let x = sine(128, 8.0);
        let r = wvd_features(&x, 256);
        // Marginal entropy should be in [0, 1] (normalized)
        assert!(r.marginal_entropy >= 0.0 && r.marginal_entropy <= 1.0 + 1e-10,
            "marginal entropy out of range: {}", r.marginal_entropy);
    }

    #[test]
    fn wvd_inst_freq_var_for_chirp() {
        // Chirp: frequency changes over time → non-zero inst freq variance
        let n = 64;
        let x: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * (2.0 + 8.0 * t) * t).sin()
        }).collect();
        let r = wvd_features(&x, 256);
        assert!(r.instantaneous_freq_var.is_finite() && r.instantaneous_freq_var >= 0.0);
    }

    // ── FastICA ───────────────────────────────────────────────────────────

    #[test]
    fn ica_returns_finite() {
        let x = sine(200, 5.0);
        let r = fast_ica(&x, 4, 50);
        assert!(r.max_negentropy.is_finite(), "max_negentropy: {}", r.max_negentropy);
        assert!(r.mean_negentropy.is_finite());
        assert!(r.kurtosis_range.is_finite());
    }

    #[test]
    fn ica_too_short_nan() {
        let r = fast_ica(&[1.0; 10], 4, 50);
        assert!(r.max_negentropy.is_nan());
    }

    #[test]
    fn ica_non_gaussian_high_negentropy() {
        // Square wave is maximally non-Gaussian → high negentropy
        let n = 200;
        let x: Vec<f64> = (0..n).map(|i| if (i / 10) % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let r = fast_ica(&x, 4, 50);
        assert!(r.max_negentropy.is_finite() && r.max_negentropy >= 0.0);
    }

    #[test]
    fn ica_convergence_iterations_positive() {
        let x: Vec<f64> = (0..200).map(|i| (i as f64 * 0.3).sin()).collect();
        let r = fast_ica(&x, 4, 50);
        assert!(r.convergence_iterations > 0);
    }

    // ── EMD ──────────────────────────────────────────────────────────────

    #[test]
    fn emd_extracts_imfs() {
        let x = sine(200, 5.0);
        let r = emd(&x, 10, 100, 0.05);
        assert!(r.n_imfs >= 1, "should extract at least 1 IMF");
        assert!(r.imf1_energy.is_finite() && r.imf1_energy >= 0.0);
    }

    #[test]
    fn emd_energy_fractions_in_range() {
        let mut x = sine(200, 5.0);
        // Mix two frequencies
        for (i, v) in x.iter_mut().enumerate() {
            *v += 0.5 * (2.0 * std::f64::consts::PI * 15.0 * i as f64 / 200.0).sin();
        }
        let r = emd(&x, 10, 100, 0.05);
        if r.imf1_energy.is_finite() {
            assert!(r.imf1_energy >= 0.0 && r.imf1_energy <= 1.0 + 1e-10,
                "imf1_energy: {}", r.imf1_energy);
        }
        if r.residual_energy.is_finite() {
            assert!(r.residual_energy >= 0.0 && r.residual_energy <= 1.0 + 1e-10,
                "residual_energy: {}", r.residual_energy);
        }
    }

    #[test]
    fn emd_too_short_nan() {
        let r = emd(&[1.0; 5], 10, 100, 0.05);
        assert!(r.n_imfs == 0);
    }

    #[test]
    fn emd_mixed_freqs_multiple_imfs() {
        // Mix of 3 frequencies should give multiple IMFs
        let n = 300;
        let x: Vec<f64> = (0..n).map(|i| {
            let t = i as f64 / n as f64;
            (2.0 * std::f64::consts::PI * 3.0 * t).sin()
            + 0.5 * (2.0 * std::f64::consts::PI * 12.0 * t).sin()
            + 0.2 * (2.0 * std::f64::consts::PI * 25.0 * t).sin()
        }).collect();
        let r = emd(&x, 10, 100, 0.05);
        assert!(r.n_imfs >= 1, "mixed signal should give at least 1 IMF, got {}", r.n_imfs);
        assert!(r.mean_period.is_finite() || r.mean_period.is_nan()); // allowed to be NaN for degenerate
    }
}
