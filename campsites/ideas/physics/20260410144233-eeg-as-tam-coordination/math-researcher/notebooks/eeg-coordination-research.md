<!-- VOCABULARY_WARNING_v1 — do not remove this marker -->

# ⚠️ STOP — VOCABULARY WARNING — READ BEFORE PROCEEDING ⚠️

> **THIS DOCUMENT MAY CONTAIN OUTDATED VOCABULARY.**
>
> Tambear's vocabulary was LOCKED IN on 2026-04-17 with formal
> definitions. The terminology used in this document was current
> at the time of writing but may DIFFER from the locked vocabulary.
>
> **Do not assume any term in this document means what you think it
> means.** Words like *primitive*, *atom*, *recipe*, *method*,
> *specialist*, *operation*, *layer*, *kingdom*, *menu* may have
> meant something different at the time this document was written
> than they do in the current locked vocabulary.
>
> **Before relying on anything in this document:**
>
> 1. **Read the canonical vocabulary first** at:
>    `R:\winrapids\docs\architecture\vocabulary.md`
> 2. **Read the architecture decomposition** at:
>    `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
> 3. **Interpret this document's content through the locked lens.**
>    For every vocabulary term you encounter, ask: what does this
>    actually mean in current tambear? Use the "old term → locked
>    term" mapping table in `vocabulary.md`.
> 4. **QUESTION EVERYTHING.** Do not accept any vocabulary as
>    correct just because it sounds right or appears in this
>    document. The fact that a word is used here is NOT evidence
>    that the word's meaning here matches its current meaning.
>
> If you find inconsistencies between this document and the locked
> vocabulary, **the locked vocabulary in `vocabulary.md` is
> authoritative.** This document is a snapshot in time, not a
> current specification.
>
> Apparent agreement between this document and the locked vocabulary
> may be illusory — the same word may carry different meanings.
> CHECK THE MAPPING TABLE.

---

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


---

<!-- VOCABULARY_WARNING_v1_END — do not remove this marker -->

# ⚠️ END OF DOCUMENT — VOCABULARY WARNING REPEATED ⚠️

> **REMINDER: Vocabulary in this document may be outdated.**
>
> Canonical vocabulary lives at:
> - `R:\winrapids\docs\architecture\vocabulary.md` (terminology)
> - `R:\winrapids\docs\architecture\atoms-primitives-recipes.md`
>   (architecture decomposition)
>
> **Do not trust vocabulary appearances. Question every term.**
> Map old language to the locked vocabulary BEFORE acting on the
> content of this document. The mapping table is in
> `vocabulary.md`.
>
> Words that may carry old meanings in this document:
> *primitive*, *atom*, *recipe*, *method*, *specialist*,
> *operation*, *layer*, *kingdom*, *menu*, *scatter*,
> *Layer 0/1/2/3/4*, *3-tier*, *9 truths*.
>
> If you arrived here from inside this document and skipped the
> top banner: GO BACK AND READ IT. The locked vocabulary is not
> a suggestion; it is the only correct interpretation of any
> tambear architecture document. Documents prior to 2026-04-17
> drift; trust the locked vocabulary, not the words in front of
> you.

