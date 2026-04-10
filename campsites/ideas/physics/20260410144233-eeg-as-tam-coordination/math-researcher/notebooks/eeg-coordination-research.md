# EEG as TAM Coordination — Mathematical Research Notes

## The Idea

EEG (electroencephalography) brain signals coordinate distributed neural
activity via oscillatory synchronization. The analogy: TAM coordinates
distributed GPU kernels. Can EEG-inspired coordination mechanisms improve
TAM's scheduling?

## EEG Frequency Bands and Their Functions

| Band | Frequency | Function | TAM Analog? |
|---|---|---|---|
| Delta (δ) | 0.5-4 Hz | Deep sleep, homeostasis | Background maintenance, garbage collection |
| Theta (θ) | 4-8 Hz | Memory consolidation, navigation | Cache management, intermediate eviction |
| Alpha (α) | 8-13 Hz | Idle/inhibition, attention gating | Work-stealing idle detection |
| Beta (β) | 13-30 Hz | Active processing, motor | Kernel execution, active computation |
| Gamma (γ) | 30-100+ Hz | Feature binding, consciousness | Cross-kernel fusion, gather coordination |

## Relevant Mathematical Primitives

### 1. Phase-Amplitude Coupling (PAC)

Low-frequency phase modulates high-frequency amplitude:
A_γ(t) = f(φ_θ(t))

This is the canonical mechanism for multi-scale coordination in the brain.

Computation:
1. Bandpass filter at low-freq → extract phase via Hilbert transform
2. Bandpass filter at high-freq → extract amplitude envelope
3. Measure coupling: Modulation Index (MI) = KL(amplitude histogram || uniform)

Primitives needed:
- `hilbert_transform(data)` — analytic signal (FFT-based)
- `bandpass_filter(data, low, high, sample_rate)` — FIR/IIR filter
- `modulation_index(phase, amplitude, n_bins)` — KL from uniform

### 2. Coherence

Cross-spectral coherence between channels:
C_xy(f) = |S_xy(f)|² / (S_xx(f) × S_yy(f))

Already partially available via FFT + cross-spectrum.

Variants:
- Magnitude squared coherence (MSC)
- Phase coherence (imaginary part of coherency)
- Partial directed coherence (PDC) — directional
- Directed transfer function (DTF)

### 3. Granger Causality (extended)

Already have basic transfer entropy. For EEG/TAM:
- Spectral Granger causality — frequency-resolved causal influence
- Conditional Granger — controlling for other channels
- Nonlinear Granger — using kernel methods or neural networks

### 4. Phase Synchronization

Phase locking value (PLV):
PLV = |⟨exp(i(φ_x(t) - φ_y(t)))⟩_t|

Measures consistency of phase difference across time.

Related: Phase lag index (PLI), weighted PLI, debiased squared PLI.

### 5. Information-Theoretic Measures

- Transfer entropy (already have)
- Directed information
- Active information storage
- Partial information decomposition

These connect directly to the information theory catalog gaps.

## What Tambear Already Has

1. FFT / spectral analysis — foundation for coherence
2. Transfer entropy — directional information flow
3. ACF / cross-correlation — basic synchrony
4. Mutual information — statistical dependence
5. Sample entropy / complexity measures — signal complexity
6. Wavelet transform — time-frequency analysis

## What's Missing for EEG Analysis

1. **Hilbert transform** — `hilbert(data)` via FFT
   - Set negative frequencies to zero, IFFT
   - Gives instantaneous amplitude and phase
   - Foundation for PAC, PLV, all phase-based measures

2. **Bandpass filter** — `bandpass(data, low, high, order, sample_rate)`
   - Butterworth IIR or FIR (windowed sinc)
   - Needed for frequency-band decomposition

3. **Phase locking value** — `plv(phase_x, phase_y)`
   - Circular statistics primitive

4. **Cross-spectral matrix** — `cross_spectrum(channels, n_fft, window)`
   - Foundation for coherence, PDC, DTF

5. **Source localization** — beamforming / LCMV
   - Inverse problem: which sources produced these scalp measurements?
   - Requires: forward model (lead field matrix), inverse solver

## Connection to TAM Scheduling

The insight: EEG coordination is a SOLVED PROBLEM in neuroscience.
The brain coordinates ~86 billion neurons across millisecond timescales
using oscillatory phase coupling. TAM coordinates ~10⁴ GPU threads.

Phase-amplitude coupling maps to: low-frequency scheduling decisions
(which kernel to launch) modulate high-frequency execution (individual
thread behavior within the kernel).

Whether this is a useful metaphor or a genuine algorithmic insight
requires experimentation. But the mathematical primitives (Hilbert
transform, phase synchronization, coherence) are independently useful
for signal processing regardless.
