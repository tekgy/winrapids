"""
Gold Standard Oracle: Family 19 — Spectral Analysis
Generates expected values for tambear validation.

Tests:
  - Lomb-Scargle periodogram (scipy.signal.lombscargle comparison)
  - Cross-spectral coherence (identical signals → 1.0)
  - Spectral entropy (tone vs noise ordering)
  - Band power (known PSD integration)
  - Multitaper PSD (pure tone peak detection)
"""

import json
import numpy as np
from scipy.signal import lombscargle

results = {}

# ===============================================================
# 1. Lomb-Scargle: known sinusoid at 5 Hz
# ===============================================================

np.random.seed(42)
n_ls = 200
# Irregular sampling around uniform grid
times = np.sort(np.random.uniform(0, 2.0, n_ls))
freq_true = 5.0
values = np.sin(2 * np.pi * freq_true * times)

# Evaluate at 100 angular frequencies
n_freqs = 100
max_freq = 0.5 / np.median(np.diff(times))  # approx Nyquist
freqs_hz = np.linspace(0.1, max_freq, n_freqs)
angular_freqs = 2 * np.pi * freqs_hz

# scipy lombscargle expects angular frequencies, values should be zero-mean
values_zm = values - values.mean()
power_scipy = lombscargle(times, values_zm, angular_freqs, normalize=False)

# Normalize to match tambear convention (2/(n*var))
# tambear uses: P = (1/(2*var)) * (A^2 + B^2) with Scargle tau
# scipy uses raw periodogram. Let's just find the peak frequency.
peak_idx = int(np.argmax(power_scipy))
peak_freq = freqs_hz[peak_idx]

results["lomb_scargle"] = {
    "times": times.tolist(),
    "values": values.tolist(),
    "n": n_ls,
    "n_freqs": n_freqs,
    "true_freq_hz": freq_true,
    "peak_freq_hz": float(peak_freq),
    "peak_freq_close_to_true": bool(abs(peak_freq - freq_true) < 1.0),
}
print("Lomb-Scargle: true=%.1f Hz, detected=%.2f Hz" % (freq_true, peak_freq))

# ===============================================================
# 2. Cross-spectral: identical signals → coherence ≈ 1
# ===============================================================

fs = 100.0
t_cs = np.arange(0, 2.0, 1.0/fs)
signal_cs = np.sin(2 * np.pi * 10.0 * t_cs) + 0.5 * np.sin(2 * np.pi * 25.0 * t_cs)

# For identical signals, coherence = 1.0 at all frequencies
results["cross_spectral_identical"] = {
    "signal": signal_cs.tolist(),
    "fs": fs,
    "n": len(signal_cs),
    "expected_coherence_min": 0.99,  # should be ~1.0 everywhere
}
print("Cross-spectral identical: n=%d, fs=%.0f" % (len(signal_cs), fs))

# Uncorrelated signals → coherence ≈ 0
np.random.seed(77)
noise1 = np.random.randn(len(t_cs))
noise2 = np.random.randn(len(t_cs))
results["cross_spectral_uncorrelated"] = {
    "signal_x": noise1.tolist(),
    "signal_y": noise2.tolist(),
    "fs": fs,
    "n": len(noise1),
    "expected_mean_coherence_below": 0.5,  # should average well below 0.5
}
print("Cross-spectral uncorrelated: n=%d" % len(noise1))

# ===============================================================
# 3. Spectral entropy: tone vs white noise
# ===============================================================

# Pure tone PSD: one big spike
tone_psd = np.zeros(64)
tone_psd[10] = 100.0
tone_psd[11] = 1.0  # small leakage

# White noise PSD: flat
noise_psd = np.ones(64)

# Shannon entropy: H = -sum(p * log(p))
def spectral_entropy(psd):
    p = np.array(psd)
    p = p[p > 0]
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))

h_tone = spectral_entropy(tone_psd)
h_noise = spectral_entropy(noise_psd)
h_max = np.log(64)

results["spectral_entropy"] = {
    "tone_psd": tone_psd.tolist(),
    "noise_psd": noise_psd.tolist(),
    "h_tone": h_tone,
    "h_noise": h_noise,
    "h_max": float(h_max),
    "h_noise_normalized": h_noise / h_max,
    "tone_lt_noise": bool(h_tone < h_noise),
}
print("Spectral entropy: tone=%.4f, noise=%.4f, max=%.4f" % (h_tone, h_noise, h_max))

# ===============================================================
# 4. Band power: rectangular PSD
# ===============================================================

# Flat PSD of 1.0 from 0 to 50 Hz, bin width = 1 Hz
freqs_bp = np.arange(0, 50, 1.0)
psd_flat = np.ones(50) * 2.0  # PSD = 2.0 W/Hz

# Total power = 2.0 * 49 Hz = 98.0 (midpoint integration)
# Band [10, 20] Hz: 10 bins, power = 2.0 * 10 = 20.0
total_power = float(np.sum(psd_flat) * 1.0)  # approximate integral
band_10_20 = float(np.sum(psd_flat[10:20]) * 1.0)
relative = band_10_20 / total_power

results["band_power"] = {
    "freqs": freqs_bp.tolist(),
    "psd": psd_flat.tolist(),
    "f_low": 10.0,
    "f_high": 20.0,
    "band_power_approx": band_10_20,
    "total_power_approx": total_power,
    "relative_power_approx": relative,
}
print("Band power [10,20] Hz: %.1f / %.1f = %.3f" % (band_10_20, total_power, relative))

# ===============================================================
# 5. Multitaper: pure tone should show peak
# ===============================================================

fs_mt = 256.0
t_mt = np.arange(0, 1.0, 1.0/fs_mt)
freq_mt = 32.0  # exactly on a bin
signal_mt = np.sin(2 * np.pi * freq_mt * t_mt)

results["multitaper_peak"] = {
    "signal": signal_mt.tolist(),
    "fs": fs_mt,
    "n": len(signal_mt),
    "true_freq": freq_mt,
    "n_tapers": 4,
}
print("Multitaper: n=%d, fs=%.0f, true_freq=%.0f Hz" % (len(signal_mt), fs_mt, freq_mt))

# ===============================================================
# 6. Spectral peaks: known peaks in synthetic PSD
# ===============================================================

freqs_pk = np.linspace(0, 50, 101)
psd_peaks = np.ones(101) * 0.1  # baseline
psd_peaks[20] = 10.0   # peak at freq[20]
psd_peaks[60] = 5.0    # peak at freq[60]
psd_peaks[80] = 3.0    # peak at freq[80]

peak_freqs_expected = [freqs_pk[20], freqs_pk[60], freqs_pk[80]]
peak_powers_expected = [10.0, 5.0, 3.0]

results["spectral_peaks"] = {
    "freqs": freqs_pk.tolist(),
    "psd": psd_peaks.tolist(),
    "threshold_ratio": 2.0,
    "expected_peak_freqs": [float(f) for f in peak_freqs_expected],
    "expected_peak_powers": peak_powers_expected,
    "expected_n_peaks": 3,
}
print("Spectral peaks: %d peaks expected" % 3)

# ===============================================================
# Save
# ===============================================================

with open("research/gold_standard/family_19_spectral_expected.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved family_19_spectral_expected.json")
