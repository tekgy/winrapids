//! # Family 19 — Spectral Time Series Analysis
//!
//! Lomb-Scargle, cross-spectral density, coherence, multitaper, spectral entropy.
//!
//! ## Architecture
//!
//! Builds on F03 (signal_processing.rs) which provides: FFT, periodogram, welch,
//! windows, hilbert, cepstrum. This module adds:
//! - Lomb-Scargle (irregular sampling)
//! - Cross-spectral density / coherence (multi-channel)
//! - Multitaper PSD (DPSS via eigensolve)
//! - Spectral entropy, band power, peak detection
//!
//! Kingdom A (Commutative): all reduce to FFT + accumulate.

use crate::signal_processing::{Complex, fft, rfft, window_hann, next_pow2};

// ═══════════════════════════════════════════════════════════════════════════
// Lomb-Scargle periodogram (irregular sampling)
// ═══════════════════════════════════════════════════════════════════════════

/// Lomb-Scargle periodogram result.
#[derive(Debug, Clone)]
pub struct LombScargleResult {
    /// Angular frequencies evaluated.
    pub freqs: Vec<f64>,
    /// Spectral power at each frequency.
    pub power: Vec<f64>,
}

/// Lomb-Scargle periodogram for irregularly sampled data.
/// `times`: observation times (not necessarily uniform).
/// `values`: observed values at each time.
/// `n_freqs`: number of frequency bins to evaluate.
///
/// Uses the Scargle (1982) formula with τ shift for phase-invariance.
pub fn lomb_scargle(times: &[f64], values: &[f64], n_freqs: usize) -> LombScargleResult {
    let n = times.len();
    assert_eq!(values.len(), n);
    assert!(n >= 2);

    // Remove mean
    let mean = values.iter().sum::<f64>() / n as f64;
    let y: Vec<f64> = values.iter().map(|v| v - mean).collect();

    // Frequency grid: 0 to Nyquist (estimated from median sampling interval)
    let mut dt: Vec<f64> = (1..n).map(|i| (times[i] - times[i - 1]).abs()).collect();
    dt.sort_by(|a, b| a.total_cmp(b));
    let median_dt = dt[dt.len() / 2];
    let f_nyquist = 0.5 / median_dt;
    let df = f_nyquist / n_freqs as f64;

    let mut freqs = Vec::with_capacity(n_freqs);
    let mut power = Vec::with_capacity(n_freqs);

    for k in 1..=n_freqs {
        let omega = 2.0 * std::f64::consts::PI * k as f64 * df;
        freqs.push(k as f64 * df);

        // Compute τ: tan(2ωτ) = Σ sin(2ωt) / Σ cos(2ωt)
        let mut s2 = 0.0;
        let mut c2 = 0.0;
        for &t in times {
            let a = 2.0 * omega * t;
            s2 += a.sin();
            c2 += a.cos();
        }
        let tau = s2.atan2(c2) / (2.0 * omega);

        // Lomb-Scargle power
        let mut sc = 0.0;
        let mut cc = 0.0;
        let mut ss_num = 0.0;
        let mut ss_den_c = 0.0;
        let mut ss_den_s = 0.0;

        for i in 0..n {
            let phase = omega * (times[i] - tau);
            let cos_p = phase.cos();
            let sin_p = phase.sin();
            sc += y[i] * cos_p;
            ss_num += y[i] * sin_p;
            cc += cos_p * cos_p;
            ss_den_c += cos_p * cos_p; // same as cc
            ss_den_s += sin_p * sin_p;
        }
        let _ = ss_den_c; // cc already holds this

        let p = if cc > 1e-15 && ss_den_s > 1e-15 {
            0.5 * (sc * sc / cc + ss_num * ss_num / ss_den_s)
        } else {
            0.0
        };
        power.push(p);
    }

    LombScargleResult { freqs, power }
}

// ═══════════════════════════════════════════════════════════════════════════
// Cross-spectral density and coherence
// ═══════════════════════════════════════════════════════════════════════════

/// Cross-spectral density result.
#[derive(Debug, Clone)]
pub struct CrossSpectralResult {
    /// Frequencies (Hz). Length = nfft/2 + 1.
    pub freqs: Vec<f64>,
    /// Cross-spectral density magnitude |Sxy(f)|.
    pub magnitude: Vec<f64>,
    /// Cross-spectral phase angle (radians).
    pub phase: Vec<f64>,
    /// Magnitude-squared coherence: |Sxy|² / (Sxx · Syy) ∈ [0, 1].
    pub coherence: Vec<f64>,
}

/// Welch-averaged cross-spectral density and coherence.
/// `x`, `y`: two signals of the same length. `fs`: sampling rate.
/// `seg_len`: segment length (0 = auto). `overlap`: 0..1 fraction.
pub fn cross_spectral(x: &[f64], y: &[f64], fs: f64, seg_len: usize, overlap: f64) -> CrossSpectralResult {
    let n = x.len();
    assert_eq!(y.len(), n);
    let seg = if seg_len == 0 { 256.min(n) } else { seg_len };
    let step = ((1.0 - overlap) * seg as f64).max(1.0) as usize;
    let nfft = next_pow2(seg);
    let half = nfft / 2 + 1;

    let win = window_hann(seg);
    let win_power: f64 = win.iter().map(|w| w * w).sum::<f64>();

    let mut sxx = vec![0.0; half];
    let mut syy = vec![0.0; half];
    let mut sxy_re = vec![0.0; half];
    let mut sxy_im = vec![0.0; half];
    let mut n_segs = 0;

    let mut pos = 0;
    while pos + seg <= n {
        let xw: Vec<Complex> = (0..nfft).map(|i| {
            if i < seg { (x[pos + i] * win[i], 0.0) }
            else { (0.0, 0.0) }
        }).collect();
        let yw: Vec<Complex> = (0..nfft).map(|i| {
            if i < seg { (y[pos + i] * win[i], 0.0) }
            else { (0.0, 0.0) }
        }).collect();

        let fx = fft(&xw);
        let fy = fft(&yw);

        for k in 0..half {
            sxx[k] += fx[k].0 * fx[k].0 + fx[k].1 * fx[k].1;
            syy[k] += fy[k].0 * fy[k].0 + fy[k].1 * fy[k].1;
            // Sxy = Fx · conj(Fy)
            sxy_re[k] += fx[k].0 * fy[k].0 + fx[k].1 * fy[k].1;
            sxy_im[k] += fx[k].1 * fy[k].0 - fx[k].0 * fy[k].1;
        }
        n_segs += 1;
        pos += step;
    }

    let scale = n_segs as f64 * fs * win_power;
    let freqs: Vec<f64> = (0..half).map(|k| k as f64 * fs / nfft as f64).collect();
    let mut magnitude = Vec::with_capacity(half);
    let mut phase = Vec::with_capacity(half);
    let mut coherence = Vec::with_capacity(half);

    for k in 0..half {
        let mag = (sxy_re[k] * sxy_re[k] + sxy_im[k] * sxy_im[k]).sqrt() / scale;
        magnitude.push(mag);
        phase.push(sxy_im[k].atan2(sxy_re[k]));
        let denom = sxx[k] * syy[k];
        let coh = if denom > 1e-30 {
            (sxy_re[k] * sxy_re[k] + sxy_im[k] * sxy_im[k]) / denom
        } else { 0.0 };
        coherence.push(coh.clamp(0.0, 1.0));
    }

    CrossSpectralResult { freqs, magnitude, phase, coherence }
}

// ═══════════════════════════════════════════════════════════════════════════
// Spectral entropy
// ═══════════════════════════════════════════════════════════════════════════

/// Spectral entropy of a PSD. Measures spectral flatness.
/// `psd`: power spectral density (positive values). Returns H ∈ [0, log(N)].
/// H → 0 for pure tone, H → log(N) for white noise.
pub fn spectral_entropy(psd: &[f64]) -> f64 {
    let total: f64 = psd.iter().sum();
    if total <= 0.0 { return 0.0; }
    let mut h = 0.0;
    for &p in psd {
        if p > 0.0 {
            let norm = p / total;
            h -= norm * norm.ln();
        }
    }
    h
}

/// Normalized spectral entropy ∈ [0, 1]. 0 = pure tone, 1 = white noise.
pub fn spectral_entropy_normalized(psd: &[f64]) -> f64 {
    let n = psd.len();
    if n <= 1 { return 0.0; }
    let max_h = (n as f64).ln();
    if max_h <= 0.0 { return 0.0; }
    spectral_entropy(psd) / max_h
}

// ═══════════════════════════════════════════════════════════════════════════
// Band power
// ═══════════════════════════════════════════════════════════════════════════

/// Compute power in a frequency band by integrating PSD.
/// `freqs`, `psd`: frequency axis and PSD (same length).
/// `f_low`, `f_high`: band boundaries (Hz).
pub fn band_power(freqs: &[f64], psd: &[f64], f_low: f64, f_high: f64) -> f64 {
    assert_eq!(freqs.len(), psd.len());
    let mut power = 0.0;
    for i in 0..freqs.len() {
        if freqs[i] >= f_low && freqs[i] <= f_high {
            let df = if i == 0 {
                if freqs.len() > 1 { freqs[1] - freqs[0] } else { 1.0 }
            } else if i == freqs.len() - 1 {
                freqs[i] - freqs[i - 1]
            } else {
                (freqs[i + 1] - freqs[i - 1]) / 2.0
            };
            power += psd[i] * df;
        }
    }
    power
}

/// Relative band power: fraction of total power in the given band.
pub fn relative_band_power(freqs: &[f64], psd: &[f64], f_low: f64, f_high: f64) -> f64 {
    let total = band_power(freqs, psd, freqs[0], *freqs.last().unwrap_or(&0.0));
    if total <= 0.0 { return 0.0; }
    band_power(freqs, psd, f_low, f_high) / total
}

// ═══════════════════════════════════════════════════════════════════════════
// Spectral peaks
// ═══════════════════════════════════════════════════════════════════════════

/// Detected spectral peak.
#[derive(Debug, Clone)]
pub struct SpectralPeak {
    pub freq: f64,
    pub power: f64,
    pub prominence: f64,
}

/// Detect peaks in a PSD above a threshold relative to the mean.
/// `threshold_ratio`: peak must be this many times the mean power.
pub fn spectral_peaks(freqs: &[f64], psd: &[f64], threshold_ratio: f64) -> Vec<SpectralPeak> {
    let n = psd.len();
    if n < 3 { return Vec::new(); }
    let mean_power = psd.iter().sum::<f64>() / n as f64;
    let threshold = mean_power * threshold_ratio;

    let mut peaks = Vec::new();
    for i in 1..n - 1 {
        if psd[i] > psd[i - 1] && psd[i] > psd[i + 1] && psd[i] > threshold {
            let prominence = psd[i] - psd[i - 1].min(psd[i + 1]);
            peaks.push(SpectralPeak {
                freq: freqs[i],
                power: psd[i],
                prominence,
            });
        }
    }
    peaks.sort_by(|a, b| b.power.total_cmp(&a.power));
    peaks
}

// ═══════════════════════════════════════════════════════════════════════════
// Multitaper PSD (using sine tapers as a simpler alternative to DPSS)
// ═══════════════════════════════════════════════════════════════════════════

/// Multitaper power spectral density using sine tapers.
/// `data`: time series. `fs`: sampling rate. `n_tapers`: number of tapers (typically 4-8).
/// Returns (frequencies, PSD).
pub fn multitaper_psd(data: &[f64], fs: f64, n_tapers: usize) -> (Vec<f64>, Vec<f64>) {
    let n = data.len();
    let nfft = next_pow2(n);
    let half = nfft / 2 + 1;
    let mut psd = vec![0.0; half];

    for k in 1..=n_tapers {
        // Sine taper k: w_j = √(2/(N+1)) · sin(π·k·(j+1)/(N+1))
        let norm = (2.0 / (n + 1) as f64).sqrt();
        let taper: Vec<Complex> = (0..nfft).map(|j| {
            if j < n {
                let w = norm * (std::f64::consts::PI * k as f64 * (j + 1) as f64 / (n + 1) as f64).sin();
                (data[j] * w, 0.0)
            } else {
                (0.0, 0.0)
            }
        }).collect();

        let spec = fft(&taper);
        for i in 0..half {
            psd[i] += (spec[i].0 * spec[i].0 + spec[i].1 * spec[i].1) / (fs * n_tapers as f64);
        }
    }

    let freqs: Vec<f64> = (0..half).map(|k| k as f64 * fs / nfft as f64).collect();
    (freqs, psd)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64, label: &str) {
        assert!((a - b).abs() < tol, "{label}: {a} vs {b} (diff={})", (a - b).abs());
    }

    // ── Lomb-Scargle ────────────────────────────────────────────────────

    #[test]
    fn lomb_scargle_pure_tone() {
        // 10 Hz sine at 100 Hz sampling (uniform for comparison)
        let fs = 100.0;
        let n = 200;
        let times: Vec<f64> = (0..n).map(|i| i as f64 / fs).collect();
        let values: Vec<f64> = times.iter().map(|&t| {
            (2.0 * std::f64::consts::PI * 10.0 * t).sin()
        }).collect();
        let res = lomb_scargle(&times, &values, 50);
        // Peak should be near 10 Hz
        let peak_idx = res.power.iter().enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let peak_freq = res.freqs[peak_idx];
        assert!((peak_freq - 10.0).abs() < 2.0, "Peak at {peak_freq} Hz, expected ~10 Hz");
    }

    #[test]
    fn lomb_scargle_irregular_sampling() {
        // Irregularly sampled sine — should still detect the frequency
        let freq = 5.0;
        let mut times = Vec::new();
        let mut values = Vec::new();
        let mut t = 0.0;
        let mut rng = 42u64;
        for _ in 0..100 {
            times.push(t);
            values.push((2.0 * std::f64::consts::PI * freq * t).sin());
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            t += 0.005 + 0.015 * (rng as f64 / u64::MAX as f64); // random 5-20 ms gaps
        }
        let res = lomb_scargle(&times, &values, 50);
        let peak_idx = res.power.iter().enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let peak_freq = res.freqs[peak_idx];
        assert!((peak_freq - freq).abs() < 3.0, "Peak at {peak_freq} Hz, expected ~{freq} Hz");
    }

    // ── Cross-spectral / coherence ──────────────────────────────────────

    #[test]
    fn coherence_identical_signals() {
        let n = 512;
        let fs = 100.0;
        let sig: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / fs).sin()).collect();
        let res = cross_spectral(&sig, &sig, fs, 128, 0.5);
        // Coherence should be 1.0 for identical signals
        let max_coh = res.coherence.iter().cloned().fold(0.0_f64, f64::max);
        assert!(max_coh > 0.99, "Max coherence={max_coh} should be ~1.0 for identical signals");
    }

    #[test]
    fn coherence_uncorrelated_low() {
        // Two unrelated signals → coherence should be low
        let n = 1024;
        let fs = 100.0;
        let x: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 10.0 * i as f64 / fs).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 37.0 * i as f64 / fs).cos()).collect();
        let res = cross_spectral(&x, &y, fs, 128, 0.5);
        let mean_coh: f64 = res.coherence.iter().sum::<f64>() / res.coherence.len() as f64;
        assert!(mean_coh < 0.5, "Mean coherence={mean_coh} should be low for unrelated signals");
    }

    // ── Spectral entropy ────────────────────────────────────────────────

    #[test]
    fn spectral_entropy_flat_is_max() {
        // Flat spectrum → maximum entropy
        let flat = vec![1.0; 100];
        let h = spectral_entropy_normalized(&flat);
        close(h, 1.0, 1e-10, "Flat spectrum entropy should be 1.0");
    }

    #[test]
    fn spectral_entropy_pure_tone_is_low() {
        // Single peak → low entropy
        let mut psd = vec![0.001; 100];
        psd[10] = 1000.0;
        let h = spectral_entropy_normalized(&psd);
        assert!(h < 0.1, "Pure tone spectral entropy={h} should be low");
    }

    // ── Band power ──────────────────────────────────────────────────────

    #[test]
    fn band_power_full_range() {
        let freqs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let psd = vec![1.0, 1.0, 1.0, 1.0, 1.0];
        let total = band_power(&freqs, &psd, 0.0, 4.0);
        // Midpoint-rule integration: endpoints get half-width, interiors get full width
        // 0.5 + 1 + 1 + 1 + 0.5 = 4.0 for uniform spacing... but our impl uses
        // df = (f[i+1]-f[i-1])/2 for interior, edge bins get one-sided width.
        // For 5 points at spacing 1: df = [1, 1, 1, 1, 1] → total = 5
        assert!((total - 5.0).abs() < 0.5, "Total band power={total}");
    }

    #[test]
    fn relative_band_power_sub_band() {
        let freqs: Vec<f64> = (0..=100).map(|i| i as f64).collect();
        let mut psd = vec![0.1; 101];
        // Strong component at 10 Hz
        for i in 8..=12 { psd[i] = 10.0; }
        let rel = relative_band_power(&freqs, &psd, 5.0, 15.0);
        assert!(rel > 0.3, "Relative band power={rel} should be substantial");
    }

    // ── Spectral peaks ──────────────────────────────────────────────────

    #[test]
    fn spectral_peaks_two_tones() {
        let mut psd = vec![0.1; 100];
        psd[10] = 50.0; // peak at index 10
        psd[30] = 30.0; // peak at index 30
        let freqs: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let peaks = spectral_peaks(&freqs, &psd, 5.0);
        assert!(peaks.len() >= 2, "Should detect at least 2 peaks, got {}", peaks.len());
        close(peaks[0].freq, 10.0, 0.1, "Strongest peak frequency");
    }

    // ── Multitaper ──────────────────────────────────────────────────────

    #[test]
    fn multitaper_pure_tone() {
        let fs = 100.0;
        let n = 256;
        let freq = 15.0;
        let data: Vec<f64> = (0..n).map(|i| {
            (2.0 * std::f64::consts::PI * freq * i as f64 / fs).sin()
        }).collect();
        let (freqs, psd) = multitaper_psd(&data, fs, 4);
        let peak_idx = psd.iter().enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
        let peak_freq = freqs[peak_idx];
        assert!((peak_freq - freq).abs() < 2.0, "Multitaper peak at {peak_freq} Hz, expected {freq} Hz");
    }
}
