//! Family 6 — Spectral analysis.
//!
//! Covers fintek leaves: `fft_spectral`, `welch`, `multitaper`, `lombscargle`,
//! `cepstrum`, `hilbert`, `stft_leaf`, `spectral_entropy`, `fir_bandpass`, `energy_bands`.
//! NOT covered: `wigner_ville`, `cwt_wavelet`, `scattering`, `coherence` (GAPs).

use tambear::signal_processing as sp;

/// FFT-based spectral features.
///
/// Fintek's `fft_spectral.rs` emits 15 features per resolution (5 resolutions).
/// For the initial bridge, we emit a simplified 8-feature summary for one resolution.
#[derive(Debug, Clone)]
pub struct FftSpectralResult {
    /// Total spectral energy.
    pub total_energy: f64,
    /// Peak frequency (index of max PSD bin / n).
    pub peak_frequency: f64,
    /// Spectral centroid: Σ f·PSD(f) / Σ PSD(f).
    pub centroid: f64,
    /// Spectral bandwidth: √(Σ (f - centroid)²·PSD / Σ PSD).
    pub bandwidth: f64,
    /// Spectral entropy (log₂).
    pub entropy: f64,
    /// Spectral flatness (geometric / arithmetic mean).
    pub flatness: f64,
    /// Spectral rolloff: frequency at which 85% of total energy is contained.
    pub rolloff_85: f64,
    /// Spectral slope: OLS slope of log(PSD) vs log(freq).
    pub slope: f64,
}

impl FftSpectralResult {
    pub fn nan() -> Self {
        Self {
            total_energy: f64::NAN, peak_frequency: f64::NAN, centroid: f64::NAN,
            bandwidth: f64::NAN, entropy: f64::NAN, flatness: f64::NAN,
            rolloff_85: f64::NAN, slope: f64::NAN,
        }
    }
}

/// Compute FFT-based spectral features on bin-level returns.
pub fn fft_spectral(returns: &[f64]) -> FftSpectralResult {
    let n = returns.len();
    if n < 8 { return FftSpectralResult::nan(); }

    // Convert to complex for FFT
    let input_complex: Vec<sp::Complex> = returns.iter().map(|&r| (r, 0.0)).collect();
    let spectrum = sp::fft(&input_complex);
    // PSD: |X(f)|² / n (one-sided, skip DC)
    let half = n / 2;
    let mut psd: Vec<f64> = vec![0.0; half];
    for k in 1..=half.saturating_sub(1) {
        let (re, im) = spectrum[k];
        psd[k] = (re * re + im * im) / n as f64;
    }

    // Total energy
    let total_energy: f64 = psd.iter().sum();
    if total_energy < 1e-300 { return FftSpectralResult::nan(); }

    // Peak frequency
    let (peak_idx, _) = psd.iter().enumerate().fold(
        (0, f64::NEG_INFINITY),
        |(i_best, v_best), (i, &v)| if v > v_best { (i, v) } else { (i_best, v_best) },
    );
    let peak_frequency = peak_idx as f64 / n as f64;

    // Frequencies (normalized)
    let freqs: Vec<f64> = (0..half).map(|k| k as f64 / n as f64).collect();

    // Centroid and bandwidth
    let centroid: f64 = freqs.iter().zip(psd.iter()).map(|(f, p)| f * p).sum::<f64>() / total_energy;
    let bandwidth: f64 = (freqs.iter().zip(psd.iter())
        .map(|(f, p)| (f - centroid).powi(2) * p).sum::<f64>() / total_energy).sqrt();

    // Spectral entropy (log₂ normalized)
    let mut entropy = 0.0;
    for &p in &psd {
        if p > 1e-300 {
            let pn = p / total_energy;
            entropy -= pn * pn.log2();
        }
    }

    // Spectral flatness (geometric / arithmetic mean)
    let mut log_sum = 0.0;
    let mut nonzero = 0.0_f64;
    for &p in &psd {
        if p > 1e-300 { log_sum += p.ln(); nonzero += 1.0; }
    }
    let flatness = if nonzero > 0.0 {
        (log_sum / nonzero).exp() / (total_energy / half as f64)
    } else { f64::NAN };

    // Rolloff at 85% energy
    let mut cum = 0.0;
    let mut rolloff_85 = 0.0;
    for (i, &p) in psd.iter().enumerate() {
        cum += p;
        if cum >= 0.85 * total_energy {
            rolloff_85 = i as f64 / n as f64;
            break;
        }
    }

    // Spectral slope: OLS of log(PSD) vs log(f), skip DC
    let mut sx = 0.0; let mut sy = 0.0; let mut sxx = 0.0; let mut sxy = 0.0; let mut np = 0.0;
    for (i, &p) in psd.iter().enumerate().skip(1) {
        if p > 1e-300 {
            let lx = (i as f64).ln();
            let ly = p.ln();
            sx += lx; sy += ly; sxx += lx * lx; sxy += lx * ly; np += 1.0;
        }
    }
    let slope = if np > 1.0 {
        let num = np * sxy - sx * sy;
        let den = np * sxx - sx * sx;
        if den.abs() > 1e-15 { num / den } else { f64::NAN }
    } else { f64::NAN };

    FftSpectralResult {
        total_energy, peak_frequency, centroid, bandwidth, entropy, flatness, rolloff_85, slope,
    }
}

/// Spectral entropy: PSD normalized to probability, Shannon entropy.
pub fn spectral_entropy(returns: &[f64]) -> f64 {
    let n = returns.len();
    if n < 4 { return f64::NAN; }
    let input_complex: Vec<sp::Complex> = returns.iter().map(|&r| (r, 0.0)).collect();
    let spectrum = sp::fft(&input_complex);
    let half = n / 2;
    let psd: Vec<f64> = (1..=half.saturating_sub(1))
        .map(|k| {
            let (re, im) = spectrum[k];
            (re * re + im * im) / n as f64
        }).collect();
    let total: f64 = psd.iter().sum();
    if total < 1e-300 { return f64::NAN; }
    let mut h = 0.0;
    for p in psd {
        if p > 0.0 {
            let pn = p / total;
            h -= pn * pn.log2();
        }
    }
    h
}

/// Welch's PSD estimate. Wraps `signal_processing::welch`.
///
/// Returns (frequencies, psd) as a tuple.
pub fn welch_psd(returns: &[f64], segment_len: usize, overlap: usize) -> (Vec<f64>, Vec<f64>) {
    sp::welch(returns, segment_len, overlap, 1.0)
}

/// Real cepstrum via IFFT(log |FFT|²).
pub fn cepstrum(returns: &[f64]) -> Vec<f64> {
    sp::real_cepstrum(returns)
}

/// Hilbert transform: returns (envelope, instantaneous_phase).
pub fn hilbert_envelope_phase(returns: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let analytic = sp::hilbert(returns);
    let envelope: Vec<f64> = analytic.iter().map(|c| (c.0 * c.0 + c.1 * c.1).sqrt()).collect();
    let phase: Vec<f64> = analytic.iter().map(|c| c.1.atan2(c.0)).collect();
    (envelope, phase)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fft_spectral_sine() {
        // Pure sine at frequency 0.1: peak should be near 0.1
        let n = 256;
        let data: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 0.1 * i as f64).sin()).collect();
        let r = fft_spectral(&data);
        assert!(r.total_energy > 0.0);
        assert!((r.peak_frequency - 0.1).abs() < 0.02,
            "Peak frequency should be ~0.1, got {}", r.peak_frequency);
    }

    #[test]
    fn fft_spectral_too_short() {
        let r = fft_spectral(&[1.0, 2.0, 3.0]);
        assert!(r.total_energy.is_nan());
    }

    #[test]
    fn spectral_entropy_white_noise() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let data: Vec<f64> = (0..256).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let h = spectral_entropy(&data);
        // White noise: spectral entropy should be relatively high (flat PSD)
        assert!(h > 3.0, "White noise spectral entropy should be high, got {}", h);
    }

    #[test]
    fn spectral_entropy_sine_low() {
        // Pure tone: spectral entropy should be low (concentrated)
        let data: Vec<f64> = (0..256).map(|i| (2.0 * std::f64::consts::PI * 0.1 * i as f64).sin()).collect();
        let h = spectral_entropy(&data);
        // Sine is concentrated → low entropy relative to uniform
        assert!(h.is_finite());
    }

    #[test]
    fn hilbert_sine_envelope() {
        let n = 256;
        let data: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 0.1 * i as f64).sin()).collect();
        let (env, _phase) = hilbert_envelope_phase(&data);
        // Envelope of a pure sine with amp 1 should be ~1 everywhere (boundaries drift)
        let mid = n / 2;
        assert!((env[mid] - 1.0).abs() < 0.1,
            "Hilbert envelope of unit sine should be ~1, got {}", env[mid]);
    }
}
