# Timbre as N-Body Interaction Pattern: A Formal Connection Between Musical Perception and Parallel Computation

**Draft — 2026-03-30**
**Field**: Music Theory / Psychoacoustics / Mathematical Music Theory

---

## Abstract

We propose a formalization of musical timbre as an N-body interaction pattern among spectral partials, using the mathematical framework of decomposable accumulation from parallel computation theory. At order 1, each partial contributes independently to pitch perception. At order 2, pairs of partials create beating and consonance/dissonance. At order 3, triples create chord quality (major/minor/diminished). At order N, the full ensemble creates timbre — the emergent quality that distinguishes a violin from a flute playing the same note. We connect this hierarchy to the k-particle framework in quantum optics and the GPU scan operator family structure, identifying a common algebraic pattern: progressively higher interaction orders capture progressively richer emergent properties. The Fock boundary for timbre is the point where the number of sounding partials depends on the current acoustic state — the regime of feedback, self-oscillation, and chaotic dynamics.

---

## 1. Introduction

### 1.1 The Timbre Problem

Timbre is the attribute of auditory sensation by which a listener can judge that two sounds are dissimilar even when they have the same pitch, loudness, and duration (ANSI, 1960). Despite a century of research, timbre remains poorly formalized — typically described by negative definition (what's left after pitch, loudness, and duration are controlled) or by lists of correlated features (spectral centroid, spectral flux, attack time).

### 1.2 The Proposal

We propose that timbre is not a feature or a list of features but an *interaction pattern*: the structure of relationships among spectral partials at progressively higher orders. This is formalized using the same mathematics that describes parallelizable computation (decomposable accumulation systems) and quantum optical processes (k-particle interactions).

---

## 2. The Hierarchy of Partial Interactions

### 2.1 Order 1: Pitch

At order 1, each partial fₖ = k·f₀ is considered independently. The percept is PITCH — the fundamental frequency f₀, extracted from the pattern of harmonically-related partials.

Order-1 representation: a vector of amplitudes [a₁, a₂, a₃, ...]. Each partial contributes additively to the waveform. No interaction.

### 2.2 Order 2: Beating and Consonance

At order 2, PAIRS of partials interact. Two near-frequency partials (fₖ, fₖ + δf) create *beating* at rate δf. The perceived "roughness" or "smoothness" of a sound depends on which pairs are beating and at what rates.

Helmholtz's (1863) theory of consonance is an order-2 theory: consonance/dissonance is determined by pairwise interactions between partials. An interval is consonant when few pairs beat in the critical bandwidth (~30-40 Hz at mid-frequencies).

Order-2 representation: for each pair (i,j), the interaction strength gᵢⱼ depends on |fᵢ - fⱼ| and the critical bandwidth at that frequency. The total roughness R₂ = Σᵢ<ⱼ gᵢⱼ·aᵢ·aⱼ is a pairwise sum — a "biparticle" accumulation.

### 2.3 Order 3: Chord Quality

At order 3, TRIPLES of partials interact. Three notes sounding simultaneously create *combination tones* — phantom frequencies at f₁ + f₂ - f₃, 2f₁ - f₂, etc. — that interact with the fundamentals and partials.

Chord quality (major/minor/diminished/augmented) is not determined by any pair of notes alone. It requires three notes. The "major-ness" or "minor-ness" is an order-3 emergent property.

Order-3 representation: for each triple (i,j,k), the combination tones and their interactions with other partials. The perceptual quality Q₃ is a sum over triples — a "triphoton" accumulation.

### 2.4 Order N: Timbre

At order N (all partials interacting simultaneously), the full timbre emerges. The violin's quality comes from the specific amplitude envelope of its 15-20 significant partials, their relative phases, their attack/decay profiles, and ALL of their mutual interactions.

**Claim**: Timbre is the order-N interaction pattern of a sound's partials. It is the property that EXISTS only when all partials are considered jointly. Any subset misses some interactions. The COMPLETE timbre requires the full N-body pattern.

---

## 3. Connection to the Decomposable Accumulation Framework

### 3.1 Sound as Accumulation

The physical waveform is an additive accumulation of partials:

p(t) = Σᵢ aᵢ · sin(2πfᵢt + φᵢ)

This is a DAS (decomposable accumulation system) at order 1: each partial contributes independently. The WAVEFORM is parallelizable (liftable) at order 1.

### 3.2 Perception as Higher-Order Accumulation

But the PERCEPT is not the waveform. Perception extracts higher-order features:

- Roughness (order 2): pairwise beating
- Chord quality (order 3): triple interaction patterns
- Timbre (order N): full interaction structure

Each perceptual layer requires a higher interaction order. The perceptual system implements a HIERARCHY of partial lifts, each capturing more of the sound's structure.

### 3.3 The Fock Boundary for Timbre

The Fock boundary for musical sounds: when the NUMBER of sounding partials depends on the current acoustic state. Examples:

- **Feedback**: a guitar near its amplifier feeds back — the current sound level determines which partials self-excite, which determines the sound level...
- **Self-oscillation**: brass instruments, where the player's lip vibration couples to the tube's resonance in a nonlinear loop
- **Chaotic dynamics**: sounds with deterministic but unpredictable partial structure (turbulent wind instruments, some percussion)

Below the Fock boundary: the partial structure is FIXED (determined by the instrument's geometry). Above: the partial structure depends on the sound, which depends on the partial structure.

---

## 4. Whitacre as N-Photon Composer

Eric Whitacre's choral music provides an extreme example. In works like *Sleep* (2000) or *Lux Aurumque* (2000), 18+ voices each sustain a different pitch, creating:

- **Order 2**: ~153 beating pairs (18 choose 2)
- **Order 3**: ~816 combination-tone triples (18 choose 3)
- **Order 18**: one 18-dimensional interaction pattern

The cluster chord's ethereal quality IS the 18-body interaction. Remove any voice and the timbre changes. No subset reconstruction is possible. The composition specifically REQUIRES the full N-body pattern — Whitacre composes for the interaction, not the individual voices.

In the DAS framework: each voice is a "particle" in an 18-dimensional state space. The perceived timbre is a functional on the 18-particle distribution. This is liftable at order 18 (the full cluster) but not at lower orders (subsets lose the essential interactions).

---

## 5. Implications

### 5.1 For Music Theory

Timbre analysis should move from feature lists (spectral centroid, spectral flux) to INTERACTION ORDER analysis. The question changes from "what are the spectral features?" to "at what order do the essential timbral interactions occur?"

A violin's timbre may be well-captured at order 3-4 (the relative amplitudes and phases of the first few harmonics and their pairwise interactions). A Whitacre cluster requires order 18. A gong (with inharmonic partials creating complex beating patterns) may require order 10+.

### 5.2 For Psychoacoustics

The human perceptual system likely implements partial lifts of LIMITED ORDER. If the auditory system's effective lift order is ~7 (analogous to working memory), then sounds with essential interactions above order 7 should be perceived as "complex" or "noisy" — the perceptual system can't resolve the full pattern.

**Prediction**: there exists a perceptual "timbre ceiling" — the maximum interaction order that the auditory system can resolve as structured timbre rather than noise. Sounds with essential interactions above this ceiling are perceived as "rich but undifferentiated." This ceiling may vary across listeners and training levels.

### 5.3 For Sound Design and Synthesis

If timbre IS an N-body interaction pattern, then synthesis should target INTERACTIONS rather than individual partials. Instead of "set the amplitude of partial 5 to 0.3," the synthesis parameter should be "set the strength of the order-3 interaction among partials 3, 5, 7 to 0.3."

This is the difference between additive synthesis (order 1 — set each partial independently) and interaction synthesis (order N — set the interaction pattern directly).

---

## 6. Connection to Computational Timbre

In the GPU computation engine (codename "timbre"), a pipeline's performance fingerprint — its sharing profile across optimization levels — is called its "timbre" because:

- The partials = primitive operations (scan, sort, reduce)
- The interaction pattern = sharing structure (which operations share intermediates)
- The emergent quality = performance (the speedup that emerges from the interaction structure)
- The Fock boundary = where the computation structure depends on intermediate results

The naming is not metaphor. It is structural isomorphism. The same mathematics (N-body interaction patterns in a decomposable accumulation framework) describes both musical perception and computational optimization.

---

## References

- ANSI (1960). American National Standard Acoustical Terminology. S1.1-1960.
- Helmholtz, H. L. F. (1863). Die Lehre von den Tonempfindungen. (On the Sensations of Tone.)
- McAdams, S. & Giordano, B. L. (2009). The perception of musical timbre. Oxford Handbook of Music Psychology.
- Sethares, W. A. (2005). Tuning, Timbre, Spectrum, Scale. Springer.
- Whitacre, E. (2000). Sleep. SATB + piano. Walton Music.
