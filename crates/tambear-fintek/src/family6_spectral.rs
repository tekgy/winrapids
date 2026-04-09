//! Family 6 — Spectral analysis.
//!
//! Covers fintek leaves: `fft_spectral`, `welch`, `multitaper`, `lombscargle`,
//! `cepstrum`, `hilbert`, `stft_leaf`, `spectral_entropy`, `fir_bandpass`,
//! `energy_bands`, `periodicity`, `haar_wavelet`.
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

// ── Multitaper PSD (K02P02C04R02) ────────────────────────────────────────────

/// Multitaper PSD features using sine tapers.
///
/// Fintek's `multitaper.rs` outputs:
/// - spectral_centroid — power-weighted mean freq (index units)
/// - spectral_bandwidth — power-weighted spread
/// - spectral_entropy — normalized Shannon entropy of PSD
/// - f_test_significance — fraction of frequencies with F-stat > 2K
///
/// `returns`: log-return series for a single bin.
/// `n_tapers`: number of sine tapers (5 matches fintek default).
#[derive(Debug, Clone)]
pub struct MultitaperResult {
    pub centroid: f64,
    pub bandwidth: f64,
    pub entropy: f64,
    pub f_test_significance: f64,
}

impl MultitaperResult {
    pub fn nan() -> Self {
        Self { centroid: f64::NAN, bandwidth: f64::NAN, entropy: f64::NAN, f_test_significance: f64::NAN }
    }
}

/// Compute multitaper PSD features from a bin of returns.
pub fn multitaper_features(returns: &[f64], n_tapers: usize) -> MultitaperResult {
    let n = returns.len();
    if n < 20 { return MultitaperResult::nan(); }
    let k_tapers = n_tapers.min(n / 4).max(1);
    let n_freq = n / 2; // positive frequencies, skip DC

    // Build sine tapers: v_j(t) = sqrt(2/(n+1)) * sin(π*(j+1)*(t+1)/(n+1))
    let norm = (2.0 / (n + 1) as f64).sqrt();
    let mut eigenspectra = vec![vec![0.0_f64; n_freq]; k_tapers];
    let pi = std::f64::consts::PI;
    for j in 0..k_tapers {
        let order = (j + 1) as f64;
        let denom = (n + 1) as f64;
        let tapered: Vec<sp::Complex> = (0..n).map(|t| {
            let w = norm * (pi * order * (t + 1) as f64 / denom).sin();
            (returns[t] * w, 0.0)
        }).collect();
        let spectrum = sp::fft(&tapered);
        for f in 0..n_freq {
            let (re, im) = spectrum[f + 1]; // skip DC
            eigenspectra[j][f] = re * re + im * im;
        }
    }

    // MT PSD = mean of eigenspectra
    let mut mt_psd = vec![0.0_f64; n_freq];
    for f in 0..n_freq {
        for j in 0..k_tapers { mt_psd[f] += eigenspectra[j][f]; }
        mt_psd[f] /= k_tapers as f64;
    }

    let total_power: f64 = mt_psd.iter().sum();
    if total_power < 1e-30 { return MultitaperResult::nan(); }

    // Centroid and bandwidth (in bin-index units, matching fintek)
    let centroid: f64 = mt_psd.iter().enumerate()
        .map(|(k, &p)| (k + 1) as f64 * p).sum::<f64>() / total_power;
    let bandwidth: f64 = (mt_psd.iter().enumerate()
        .map(|(k, &p)| { let f = (k + 1) as f64; p * (f - centroid) * (f - centroid) })
        .sum::<f64>() / total_power).sqrt();

    // Normalized entropy
    let mut h = 0.0_f64;
    for &p in &mt_psd {
        let prob = p / total_power;
        if prob > 1e-30 { h -= prob * prob.ln(); }
    }
    let max_h = (n_freq as f64).ln();
    let entropy = if max_h > 1e-30 { h / max_h } else { f64::NAN };

    // F-test: fraction of frequencies with between-taper var / within-mean > 2K
    let f_threshold = 2.0 * k_tapers as f64;
    let mut n_sig = 0_u32;
    if k_tapers >= 2 {
        for f in 0..n_freq {
            let mean_p = mt_psd[f];
            if mean_p < 1e-30 { continue; }
            let var: f64 = eigenspectra.iter()
                .map(|es| { let d = es[f] - mean_p; d * d })
                .sum::<f64>() / (k_tapers - 1) as f64;
            if var / mean_p > f_threshold { n_sig += 1; }
        }
    }
    let f_test_significance = if n_freq > 0 { n_sig as f64 / n_freq as f64 } else { f64::NAN };

    MultitaperResult { centroid, bandwidth, entropy, f_test_significance }
}

// ── Lomb-Scargle (K02P02C03R01) ──────────────────────────────────────────────

/// Lomb-Scargle spectral features.
///
/// Fintek's `lombscargle.rs` outputs: peak_frequency, peak_power, spectral_slope,
/// false_alarm_probability.
///
/// `returns`: log-returns (values). `times`: corresponding timestamps (any units).
/// `n_freqs`: number of test frequencies (64 matches fintek default).
#[derive(Debug, Clone)]
pub struct LombScargleResult {
    pub peak_frequency: f64,
    pub peak_power: f64,
    pub spectral_slope: f64,
    pub false_alarm_probability: f64,
}

impl LombScargleResult {
    pub fn nan() -> Self {
        Self { peak_frequency: f64::NAN, peak_power: f64::NAN,
               spectral_slope: f64::NAN, false_alarm_probability: f64::NAN }
    }
}

/// Compute Lomb-Scargle periodogram features.
///
/// Times are normalized to [0,1] internally so units don't matter.
pub fn lomb_scargle_features(returns: &[f64], times: &[f64], n_freqs: usize) -> LombScargleResult {
    let n = returns.len();
    if n < 10 || times.len() != n { return LombScargleResult::nan(); }

    let t_min = times.iter().copied().fold(f64::INFINITY, f64::min);
    let t_max = times.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let t_range = t_max - t_min;
    if t_range < 1e-30 { return LombScargleResult::nan(); }
    let norm_t: Vec<f64> = times.iter().map(|&t| (t - t_min) / t_range).collect();

    let mean_y: f64 = returns.iter().sum::<f64>() / n as f64;
    let y: Vec<f64> = returns.iter().map(|&r| r - mean_y).collect();
    let var_y: f64 = y.iter().map(|&v| v * v).sum::<f64>() / n as f64;
    if var_y < 1e-30 { return LombScargleResult::nan(); }

    let pi = std::f64::consts::PI;
    let f_min = 1.0;
    let f_max = n as f64 / 2.0;
    let nf = n_freqs.max(2);
    let freqs: Vec<f64> = (0..nf).map(|i| f_min + (f_max - f_min) * i as f64 / (nf - 1) as f64).collect();

    let mut power = Vec::with_capacity(nf);
    for &freq in &freqs {
        let omega = 2.0 * pi * freq;
        let sum_sin2: f64 = norm_t.iter().map(|&t| (2.0 * omega * t).sin()).sum();
        let sum_cos2: f64 = norm_t.iter().map(|&t| (2.0 * omega * t).cos()).sum();
        let tau = sum_sin2.atan2(sum_cos2) / (2.0 * omega);
        let mut ss_yc = 0.0_f64; let mut ss_cc = 0.0_f64;
        let mut ss_ys = 0.0_f64; let mut ss_ss = 0.0_f64;
        for j in 0..n {
            let phase = omega * (norm_t[j] - tau);
            let cos_p = phase.cos(); let sin_p = phase.sin();
            ss_yc += y[j] * cos_p; ss_cc += cos_p * cos_p;
            ss_ys += y[j] * sin_p; ss_ss += sin_p * sin_p;
        }
        let pw = if ss_cc > 1e-30 && ss_ss > 1e-30 {
            0.5 * (ss_yc * ss_yc / ss_cc + ss_ys * ss_ys / ss_ss) / var_y
        } else { 0.0 };
        power.push(pw);
    }

    let (peak_idx, &peak_pow) = power.iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1)).unwrap_or((0, &0.0));
    let peak_frequency = freqs[peak_idx];

    // Spectral slope: OLS of log(power) vs log(freq), negate slope (power-law decay)
    let pairs: Vec<(f64, f64)> = freqs.iter().zip(power.iter())
        .filter(|&(_, &pw)| pw > 1e-30).map(|(&f, &pw)| (f.ln(), pw.ln())).collect();
    let spectral_slope = if pairs.len() >= 3 {
        let mx: f64 = pairs.iter().map(|(x, _)| x).sum::<f64>() / pairs.len() as f64;
        let my: f64 = pairs.iter().map(|(_, y)| y).sum::<f64>() / pairs.len() as f64;
        let (mut sxy, mut sxx) = (0.0_f64, 0.0_f64);
        for &(x, yv) in &pairs { let dx = x - mx; sxy += dx * (yv - my); sxx += dx * dx; }
        if sxx > 1e-30 { -(sxy / sxx) } else { f64::NAN }
    } else { f64::NAN };

    // False alarm probability (Baluev approximation): 1 - (1 - exp(-z))^M
    let fap = 1.0 - (1.0 - (-peak_pow).exp()).powi(nf as i32);
    let false_alarm_probability = fap.clamp(0.0, 1.0);

    LombScargleResult { peak_frequency, peak_power: peak_pow, spectral_slope, false_alarm_probability }
}

// ── STFT features (K02P03C04R01) ─────────────────────────────────────────────

/// STFT-based temporal spectral features.
///
/// Fintek's `stft_leaf.rs` outputs:
/// - spectral_centroid_var — variance of per-frame centroid
/// - spectral_flux — mean L2 frame-to-frame PSD change
/// - onset_strength — mean positive spectral flux
/// - chromagram_entropy — Shannon entropy of 12-bin folded-frequency energy
///
/// Window size: 32, hop: 16 (50% overlap), matching fintek defaults.
#[derive(Debug, Clone)]
pub struct StftFeaturesResult {
    pub centroid_var: f64,
    pub spectral_flux: f64,
    pub onset_strength: f64,
    pub chromagram_entropy: f64,
}

impl StftFeaturesResult {
    pub fn nan() -> Self {
        Self { centroid_var: f64::NAN, spectral_flux: f64::NAN,
               onset_strength: f64::NAN, chromagram_entropy: f64::NAN }
    }
}

/// Compute STFT temporal spectral features.
pub fn stft_features(returns: &[f64]) -> StftFeaturesResult {
    const WINDOW: usize = 32;
    const HOP: usize = 16;
    let n = returns.len();
    if n < WINDOW { return StftFeaturesResult::nan(); }

    // Hann window
    let hann: Vec<f64> = (0..WINDOW).map(|i| {
        0.5 * (1.0 - (2.0 * std::f64::consts::PI * i as f64 / WINDOW as f64).cos())
    }).collect();

    let n_frames = 1 + (n.saturating_sub(WINDOW)) / HOP;
    let n_bins = WINDOW / 2;
    let mut frames_psd: Vec<Vec<f64>> = Vec::with_capacity(n_frames);
    let mut centroids: Vec<f64> = Vec::with_capacity(n_frames);

    for frame_idx in 0..n_frames {
        let start = frame_idx * HOP;
        let end = (start + WINDOW).min(n);
        let len = end - start;
        let windowed: Vec<sp::Complex> = (0..WINDOW).map(|i| {
            let v = if i < len { returns[start + i] * hann[i] } else { 0.0 };
            (v, 0.0)
        }).collect();
        let spectrum = sp::fft(&windowed);
        let psd: Vec<f64> = (1..=n_bins).map(|k| {
            let (re, im) = spectrum[k];
            re * re + im * im
        }).collect();
        let total: f64 = psd.iter().sum();
        let centroid = if total > 1e-30 {
            psd.iter().enumerate().map(|(k, &p)| (k + 1) as f64 * p).sum::<f64>() / total
        } else { 0.0 };
        centroids.push(centroid);
        frames_psd.push(psd);
    }

    if n_frames < 2 { return StftFeaturesResult::nan(); }

    // centroid_var
    let mean_c: f64 = centroids.iter().sum::<f64>() / n_frames as f64;
    let centroid_var: f64 = centroids.iter().map(|&c| (c - mean_c).powi(2)).sum::<f64>() / n_frames as f64;

    // spectral_flux and onset_strength
    let mut flux_sum = 0.0_f64;
    let mut onset_sum = 0.0_f64;
    for i in 1..n_frames {
        let diff: f64 = frames_psd[i].iter().zip(frames_psd[i-1].iter())
            .map(|(&a, &b)| (a - b).powi(2)).sum::<f64>();
        flux_sum += diff.sqrt();
        let pos_flux: f64 = frames_psd[i].iter().zip(frames_psd[i-1].iter())
            .map(|(&a, &b)| (a - b).max(0.0)).sum::<f64>();
        onset_sum += pos_flux;
    }
    let n_diffs = (n_frames - 1) as f64;
    let spectral_flux = flux_sum / n_diffs;
    let onset_strength = onset_sum / n_diffs;

    // chromagram_entropy: fold n_bins into 12 chroma bins
    let mut chroma = vec![0.0_f64; 12];
    for frame_psd in &frames_psd {
        for (k, &p) in frame_psd.iter().enumerate() {
            let bin = k % 12;
            chroma[bin] += p;
        }
    }
    let chroma_total: f64 = chroma.iter().sum();
    let chromagram_entropy = if chroma_total > 1e-30 {
        let mut h = 0.0_f64;
        for &c in &chroma {
            let p = c / chroma_total;
            if p > 1e-30 { h -= p * p.ln(); }
        }
        h
    } else { f64::NAN };

    StftFeaturesResult { centroid_var, spectral_flux, onset_strength, chromagram_entropy }
}

// ── Energy bands (K02P19C3R1) ─────────────────────────────────────────────────

/// Energy band decomposition features.
///
/// Fintek's `energy_bands.rs` outputs:
/// - total_energy — Σ|returns|²
/// - low_freq_ratio — energy in lowest ¼ of PSD frequencies
/// - high_freq_ratio — energy in highest ¼ of PSD frequencies
/// - spectral_centroid — center of mass of PSD (normalized 0-1)
/// - spectral_bandwidth — bandwidth around centroid (normalized)
#[derive(Debug, Clone)]
pub struct EnergyBandsResult {
    pub total_energy: f64,
    pub low_freq_ratio: f64,
    pub high_freq_ratio: f64,
    pub spectral_centroid: f64,
    pub spectral_bandwidth: f64,
}

impl EnergyBandsResult {
    pub fn nan() -> Self {
        Self { total_energy: f64::NAN, low_freq_ratio: f64::NAN, high_freq_ratio: f64::NAN,
               spectral_centroid: f64::NAN, spectral_bandwidth: f64::NAN }
    }
}

/// Compute energy band features from a bin of returns.
pub fn energy_bands(returns: &[f64]) -> EnergyBandsResult {
    let n = returns.len();
    if n < 10 { return EnergyBandsResult::nan(); }
    let total_energy: f64 = returns.iter().map(|r| r * r).sum();
    let input_c: Vec<sp::Complex> = returns.iter().map(|&r| (r, 0.0)).collect();
    let spectrum = sp::fft(&input_c);
    let n_freq = n / 2;
    if n_freq < 4 { return EnergyBandsResult::nan(); }
    let power: Vec<f64> = (1..=n_freq).map(|k| {
        let (re, im) = spectrum[k];
        re * re + im * im
    }).collect();
    let total_psd: f64 = power.iter().sum();
    if total_psd < 1e-30 { return EnergyBandsResult::nan(); }

    let q1 = n_freq / 4;
    let q3 = 3 * n_freq / 4;
    let lo: f64 = power[..q1].iter().sum();
    let hi: f64 = power[q3..].iter().sum();
    let low_freq_ratio = lo / total_psd;
    let high_freq_ratio = hi / total_psd;

    // Normalized centroid and bandwidth (freq in [0,1])
    let centroid: f64 = power.iter().enumerate()
        .map(|(k, &p)| (k as f64 / n_freq as f64) * p).sum::<f64>() / total_psd;
    let bandwidth: f64 = (power.iter().enumerate()
        .map(|(k, &p)| { let f = k as f64 / n_freq as f64; p * (f - centroid).powi(2) })
        .sum::<f64>() / total_psd).sqrt();

    EnergyBandsResult { total_energy, low_freq_ratio, high_freq_ratio, spectral_centroid: centroid,
                        spectral_bandwidth: bandwidth }
}

// ── Periodicity (K02P19C4R1) ──────────────────────────────────────────────────

/// Periodicity strength features.
///
/// Fintek's `periodicity.rs` outputs:
/// - acf_peak — height of first ACF peak (after lag 1)
/// - acf_peak_lag — lag at which first ACF peak occurs
/// - periodicity_strength — normalized ACF peak [0,1]
/// - spectral_peak_ratio — dominant PSD peak / mean PSD
#[derive(Debug, Clone)]
pub struct PeriodicityResult {
    pub acf_peak: f64,
    pub acf_peak_lag: usize,
    pub periodicity_strength: f64,
    pub spectral_peak_ratio: f64,
}

impl PeriodicityResult {
    pub fn nan() -> Self {
        Self { acf_peak: f64::NAN, acf_peak_lag: 0, periodicity_strength: f64::NAN, spectral_peak_ratio: f64::NAN }
    }
}

/// Compute periodicity strength from a bin of returns.
pub fn periodicity(returns: &[f64]) -> PeriodicityResult {
    let n = returns.len();
    if n < 20 { return PeriodicityResult::nan(); }
    let mean: f64 = returns.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = returns.iter().map(|r| r - mean).collect();
    let var: f64 = centered.iter().map(|c| c * c).sum::<f64>() / n as f64;
    if var < 1e-30 { return PeriodicityResult::nan(); }

    // ACF peaks: scan lags 2..n/4, find first local maximum
    let max_lag = n / 4;
    let mut best_peak = 0.0_f64;
    let mut best_lag = 2_usize;
    let mut prev_acf = {
        let s: f64 = (0..n - 1).map(|t| centered[t] * centered[t + 1]).sum();
        s / (n as f64 * var)
    };
    let mut prev_prev_acf = 0.0_f64;
    for lag in 2..=max_lag {
        let s: f64 = (0..n - lag).map(|t| centered[t] * centered[t + lag]).sum();
        let acf = s / (n as f64 * var);
        // Local maximum: prev_acf > prev_prev_acf and prev_acf > acf
        if lag >= 3 && prev_acf > prev_prev_acf && prev_acf > acf && prev_acf > best_peak {
            best_peak = prev_acf;
            best_lag = lag - 1;
        }
        prev_prev_acf = prev_acf;
        prev_acf = acf;
    }

    // PSD: find peak ratio
    let input_c: Vec<sp::Complex> = returns.iter().map(|&r| (r, 0.0)).collect();
    let spectrum = sp::fft(&input_c);
    let n_freq = n / 2;
    let psd: Vec<f64> = (1..=n_freq).map(|k| {
        let (re, im) = spectrum[k]; re * re + im * im
    }).collect();
    let mean_psd: f64 = psd.iter().sum::<f64>() / psd.len() as f64;
    let peak_psd = psd.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let spectral_peak_ratio = if mean_psd > 1e-30 { peak_psd / mean_psd } else { f64::NAN };
    let periodicity_strength = (best_peak / 1.0_f64).clamp(0.0, 1.0);

    PeriodicityResult { acf_peak: best_peak, acf_peak_lag: best_lag, periodicity_strength, spectral_peak_ratio }
}

// ── Haar wavelet (K02P03C02) ──────────────────────────────────────────────────

/// Haar wavelet decomposition features (13 outputs per M-level).
///
/// Fintek's `haar_wavelet.rs` has 15 variants: 5 M-levels × 3 strategies.
/// This function takes a pre-regularized series of length `m` (a power of 2)
/// and computes the 13 scalar features. Call with your chosen M and regularization.
///
/// Use `tambear::signal_processing::haar_wavedec` under the hood.
#[derive(Debug, Clone)]
pub struct HaarWaveletResult {
    pub approx_energy: f64,
    pub total_detail_energy: f64,
    pub max_detail_level: usize,
    pub energy_concentration: f64,
    pub wavelet_entropy: f64,
    pub coarse_fine_ratio: f64,
    pub energy_decay_rate: f64,
    pub energy_decay_r2: f64,
    pub detail_kurtosis: f64,
    pub detail_skewness: f64,
    pub detail_mean_abs: f64,
    pub detail_max_abs: f64,
    pub zero_crossing_rate: f64,
}

impl HaarWaveletResult {
    pub fn nan() -> Self {
        Self { approx_energy: f64::NAN, total_detail_energy: f64::NAN, max_detail_level: 0,
               energy_concentration: f64::NAN, wavelet_entropy: f64::NAN, coarse_fine_ratio: f64::NAN,
               energy_decay_rate: f64::NAN, energy_decay_r2: f64::NAN, detail_kurtosis: f64::NAN,
               detail_skewness: f64::NAN, detail_mean_abs: f64::NAN, detail_max_abs: f64::NAN,
               zero_crossing_rate: f64::NAN }
    }
}

/// Compute Haar wavelet features from a regularized series of length m (power of 2).
///
/// `data`: regularized to `m` equispaced points before calling this function.
pub fn haar_wavelet_features(data: &[f64]) -> HaarWaveletResult {
    let m = data.len();
    if m < 2 || m & (m - 1) != 0 { return HaarWaveletResult::nan(); } // must be power of 2

    let (approx, details) = tambear::signal_processing::haar_wavedec(data, m.trailing_zeros() as usize);

    let approx_energy: f64 = approx.iter().map(|x| x * x).sum();
    let level_energies: Vec<f64> = details.iter()
        .map(|lvl| lvl.iter().map(|x| x * x).sum::<f64>()).collect();
    let total_detail_energy: f64 = level_energies.iter().sum();

    let (max_detail_level, &max_lvl_energy) = level_energies.iter().enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1)).unwrap_or((0, &0.0));
    let energy_concentration = if total_detail_energy > 1e-30 { max_lvl_energy / total_detail_energy } else { f64::NAN };

    // Wavelet entropy (normalized across levels)
    let wavelet_entropy = if total_detail_energy > 1e-30 {
        let max_h = (level_energies.len() as f64).ln();
        let h: f64 = level_energies.iter().filter(|&&e| e > 1e-30)
            .map(|&e| { let p = e / total_detail_energy; -p * p.ln() }).sum();
        if max_h > 1e-30 { h / max_h } else { f64::NAN }
    } else { f64::NAN };

    // Coarse/fine ratio: first half levels vs second half levels
    let n_lvl = level_energies.len();
    let half = n_lvl / 2;
    let coarse: f64 = level_energies[..half.max(1)].iter().sum();
    let fine: f64 = level_energies[half..].iter().sum::<f64>().max(1e-30);
    let coarse_fine_ratio = coarse / fine;

    // Energy decay rate: OLS slope of log(energy) vs level index
    let (energy_decay_rate, energy_decay_r2) = {
        let pairs: Vec<(f64, f64)> = level_energies.iter().enumerate()
            .filter(|&(_, &e)| e > 1e-30).map(|(i, &e)| (i as f64, e.ln())).collect();
        if pairs.len() >= 2 {
            let mx: f64 = pairs.iter().map(|(x, _)| x).sum::<f64>() / pairs.len() as f64;
            let my: f64 = pairs.iter().map(|(_, y)| y).sum::<f64>() / pairs.len() as f64;
            let (mut sxy, mut sxx, mut syy) = (0.0_f64, 0.0_f64, 0.0_f64);
            for &(x, y) in &pairs { let dx = x - mx; let dy = y - my; sxy += dx * dy; sxx += dx * dx; syy += dy * dy; }
            let slope = if sxx > 1e-30 { sxy / sxx } else { f64::NAN };
            let r2 = if sxx > 1e-30 && syy > 1e-30 { (sxy * sxy / (sxx * syy)).clamp(0.0, 1.0) } else { f64::NAN };
            (slope, r2)
        } else { (f64::NAN, f64::NAN) }
    };

    // All detail coefficients flattened
    let all_detail: Vec<f64> = details.iter().flat_map(|lvl| lvl.iter().copied()).collect();
    let nd = all_detail.len();
    let (detail_kurtosis, detail_skewness, detail_mean_abs, detail_max_abs, zero_crossing_rate) =
    if nd < 2 {
        (f64::NAN, f64::NAN, f64::NAN, f64::NAN, f64::NAN)
    } else {
        let mean_d: f64 = all_detail.iter().sum::<f64>() / nd as f64;
        let var_d: f64 = all_detail.iter().map(|x| (x - mean_d).powi(2)).sum::<f64>() / nd as f64;
        let std_d = var_d.sqrt();
        let skew = if std_d > 1e-30 {
            all_detail.iter().map(|x| ((x - mean_d) / std_d).powi(3)).sum::<f64>() / nd as f64
        } else { f64::NAN };
        let kurt = if std_d > 1e-30 {
            all_detail.iter().map(|x| ((x - mean_d) / std_d).powi(4)).sum::<f64>() / nd as f64 - 3.0
        } else { f64::NAN };
        let mean_abs = all_detail.iter().map(|x| x.abs()).sum::<f64>() / nd as f64;
        let max_abs = all_detail.iter().map(|x| x.abs()).fold(0.0_f64, f64::max);
        let zc = if nd >= 2 {
            let crossings = all_detail.windows(2).filter(|w| w[0] * w[1] < 0.0).count();
            crossings as f64 / (nd - 1) as f64
        } else { f64::NAN };
        (kurt, skew, mean_abs, max_abs, zc)
    };

    HaarWaveletResult { approx_energy, total_detail_energy, max_detail_level: max_detail_level + 1,
        energy_concentration, wavelet_entropy, coarse_fine_ratio, energy_decay_rate, energy_decay_r2,
        detail_kurtosis, detail_skewness, detail_mean_abs, detail_max_abs, zero_crossing_rate }
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
        let mid = n / 2;
        assert!((env[mid] - 1.0).abs() < 0.1,
            "Hilbert envelope of unit sine should be ~1, got {}", env[mid]);
    }

    #[test]
    fn multitaper_white_noise() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let data: Vec<f64> = (0..256).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let r = multitaper_features(&data, 5);
        assert!(r.centroid.is_finite());
        assert!(r.entropy > 0.0 && r.entropy <= 1.0, "normalized entropy in (0,1], got {}", r.entropy);
        assert!(r.f_test_significance >= 0.0 && r.f_test_significance <= 1.0);
    }

    #[test]
    fn multitaper_too_short() {
        let r = multitaper_features(&[1.0, 2.0, 3.0], 5);
        assert!(r.centroid.is_nan());
    }

    #[test]
    fn lomb_scargle_pure_tone_irregular() {
        let n = 100;
        let mut rng = tambear::rng::Xoshiro256::new(7);
        let times: Vec<f64> = (0..n).map(|i| i as f64 + tambear::rng::sample_normal(&mut rng, 0.0, 0.05)).collect();
        let data: Vec<f64> = times.iter().map(|&t| (2.0 * std::f64::consts::PI * 0.1 * t).sin()).collect();
        let r = lomb_scargle_features(&data, &times, 64);
        assert!(r.peak_power > 0.0);
        assert!(r.false_alarm_probability < 0.5, "significant peak: FAP={}", r.false_alarm_probability);
    }

    #[test]
    fn lomb_scargle_too_short() {
        let r = lomb_scargle_features(&[1.0, 2.0], &[0.0, 1.0], 64);
        assert!(r.peak_frequency.is_nan());
    }

    #[test]
    fn stft_features_basic() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let data: Vec<f64> = (0..200).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 1.0)).collect();
        let r = stft_features(&data);
        assert!(r.centroid_var.is_finite());
        assert!(r.spectral_flux >= 0.0);
        assert!(r.onset_strength >= 0.0);
        assert!(r.chromagram_entropy.is_finite());
    }

    #[test]
    fn stft_too_short() {
        let r = stft_features(&[1.0, 2.0]);
        assert!(r.centroid_var.is_nan());
    }

    #[test]
    fn energy_bands_low_freq_sine() {
        let n = 128;
        let data: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * 0.02 * i as f64).sin()).collect();
        let r = energy_bands(&data);
        assert!(r.total_energy > 0.0);
        assert!(r.low_freq_ratio > r.high_freq_ratio,
            "low-freq sine: lo={} hi={}", r.low_freq_ratio, r.high_freq_ratio);
        assert!(r.spectral_centroid >= 0.0 && r.spectral_centroid <= 1.0);
    }

    #[test]
    fn energy_bands_too_short() {
        let r = energy_bands(&[1.0]);
        assert!(r.total_energy.is_nan());
    }

    #[test]
    fn periodicity_sine() {
        let n = 200;
        let data: Vec<f64> = (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / 20.0).sin()).collect();
        let r = periodicity(&data);
        assert!(r.acf_peak > 0.5, "strong periodicity: acf_peak={}", r.acf_peak);
        assert!(r.spectral_peak_ratio > 1.0);
    }

    #[test]
    fn periodicity_too_short() {
        let r = periodicity(&[1.0, 2.0]);
        assert!(r.acf_peak.is_nan());
    }

    #[test]
    fn haar_wavelet_basic() {
        let m = 32_usize;
        let data: Vec<f64> = (0..m).map(|i| (i as f64 * 0.1).sin()).collect();
        let r = haar_wavelet_features(&data);
        assert!(r.approx_energy.is_finite());
        assert!(r.total_detail_energy.is_finite());
        assert!(r.wavelet_entropy.is_finite());
        assert!(r.zero_crossing_rate >= 0.0 && r.zero_crossing_rate <= 1.0);
    }

    #[test]
    fn haar_wavelet_not_power_of_two() {
        let data = vec![1.0; 30]; // not power of 2
        let r = haar_wavelet_features(&data);
        assert!(r.approx_energy.is_nan());
    }
}
