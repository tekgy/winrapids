# The Entity Is The Relationship: A Liftability Interpretation of Quantum Measurement

**Draft — 2026-03-30**
**Field**: Foundations of Quantum Mechanics / Philosophy of Physics
**Status**: Speculative — requires physicist review

---

## Abstract

We propose a reinterpretation of quantum measurement using the mathematical framework of liftability and order-k decomposable accumulation. The core claim: quantum "paradoxes" (wave-particle duality, measurement collapse, entanglement, delayed choice) arise from attempting to treat order-2 entities (wavefunctions, joint states) as order-1 entities (particles with definite properties). The wavefunction is not a description of a particle's indeterminate state — it IS the fundamental entity, and the "particle" is a dimensional projection (a partial lift) of the wavefunction onto classical coordinates. Measurement collapse is constitutive projection, not revelation. Entanglement is non-factorability at order 1, not nonlocal connection. Bell's inequality is a liftability test that proves quantum mechanics requires order 2. This interpretation doesn't modify the formalism of quantum mechanics — it provides a vocabulary for WHY the formalism has the structure it does, connecting quantum foundations to parallel computation theory and rendering theory via the common algebraic framework of decomposable accumulation.

**Note**: This manuscript is SPECULATIVE. The mathematical framework (liftability, decomposable accumulation, Fock boundary) is rigorous. The application to quantum foundations is an interpretation, not a proof. It is offered as a conceptual framework that may be useful for thinking, not as a claim about physical reality.

---

## 1. The Order-1 Assumption

### 1.1 Classical Physics Is Order-1 Liftable

In classical mechanics, a system of N particles is described by 6N numbers (3 positions + 3 momenta per particle). Crucially, the joint state FACTORS:

P(particle_1, particle_2, ..., particle_N) = P(particle_1) · P(particle_2) · ... · P(particle_N)

Each particle's state is independent. The system IS an order-1 decomposable accumulation: compute each particle independently, combine the results. This is why classical N-body simulation is embarrassingly parallel (up to gravitational coupling).

### 1.2 Quantum Mechanics Is NOT Order-1 Liftable

The quantum state of N particles lives in a tensor product Hilbert space H₁ ⊗ H₂ ⊗ ... ⊗ Hₙ. For entangled states:

|ψ⟩ ≠ |ψ₁⟩ ⊗ |ψ₂⟩ ⊗ ... ⊗ |ψₙ⟩

The joint state does NOT factor. You cannot describe the system by describing each particle independently. This is PRECISELY the non-liftability condition at order 1.

### 1.3 Bell's Theorem as Liftability Test

Bell's inequality tests whether quantum correlations can be reproduced by order-1 theories (local hidden variables — each particle carries its own independent state). The violation of Bell's inequality proves:

**Theorem (Bell, restated).** Quantum mechanics is not liftable at order 1. No theory where each particle's measurement outcome is determined by a per-particle function (hidden variable) can reproduce the observed correlations.

Quantum mechanics IS liftable at order 2: the joint wavefunction |ψ(r_A, r_B)⟩ is a single entity in a joint space. The "per-entity function" is the Born rule applied to the joint wavefunction. The correlations ARE per-entity (per-biphoton) properties.

---

## 2. The Entity Is The Relationship

### 2.1 The Double Slit

**Order-1 question**: "Which slit did the electron go through?"
This demands a definite per-particle property (slit_A or slit_B). The measurement provides it — and destroys the interference pattern.

**Order-2 framing**: The entity is not the electron. The entity is the WAVEFUNCTION — the relationship between possible paths. The wavefunction carries amplitudes for path_A AND path_B simultaneously. These amplitudes INTERFERE. The interference IS the entity's structure in the joint path space.

Asking "which slit?" is a forced order-1 projection of an order-2 entity. The interference was real — it existed in the order-2 space. The projection destroyed it because the order-1 space can't represent it.

### 2.2 Entanglement

**Order-1 framing**: Two particles are "spookily connected." Measuring one "instantaneously affects" the other. This requires nonlocal influence — a conceptual horror for relativistic physics.

**Order-2 framing**: There is ONE entity (the biphoton) in a joint state space. "Measuring particle A" is a partial projection of the biphoton — projecting out the A dimensions and leaving the B dimensions constrained by the projection. No signal travels. No influence propagates. The biphoton was always ONE thing. We're just looking at it from one side.

"Spooky action at a distance" disappears when you recognize the entity is the RELATIONSHIP, not the particles. There's no action. There's no distance. There's a single entity in a joint space, and a projection that constrains the remaining dimensions.

### 2.3 Measurement Collapse

**Standard framing**: The wavefunction "collapses" upon measurement. Something changes in physical reality.

**Liftability framing**: Measurement is CONSTITUTIVE PROJECTION. The wavefunction (the entity in the full Fock space) is projected onto a lower-dimensional subspace (the eigenspace of the measurement operator). The "collapse" is the dimensional reduction. The wavefunction didn't CHANGE — it was PROJECTED.

The information "lost" in collapse was never representable in the measurement's eigenspace. It existed in the full space. The measurement chose which subspace to project onto. The choice IS the measurement.

### 2.4 Delayed Choice

**Standard framing**: The experimenter's choice of measurement, made AFTER the photon has entered the interferometer, appears to retroactively determine the photon's past behavior.

**Liftability framing**: The photon was never in a definite state (particle or wave). It was in the full wavefunction space — an order-2 entity. The "delayed choice" of measurement is the choice of WHICH projection to apply. The projection creates the definiteness. The photon's "past behavior" was never definite — it was the full wavefunction, which is compatible with BOTH measurement outcomes. Choosing the measurement chooses which projection of the past to make definite.

No retrocausality needed. Just constitutive projection applied to a system that was never in the definite state we assumed.

---

## 3. The Hierarchy

### 3.1 Orders of Quantum Description

| Order | Description | Can represent |
|---|---|---|
| 0 | Classical point particle | Position, momentum |
| 1 | Single-particle wavefunction | Superposition, interference, uncertainty |
| 2 | Two-particle joint wavefunction | Entanglement, Bell correlations |
| k | k-particle joint wavefunction | k-body quantum correlations |
| Fock | Variable particle number | QFT, vacuum fluctuations, creation/annihilation |

Each order captures phenomena invisible at the previous order. Entanglement IS the order-2 phenomenon — irreducible to order 1 (Bell's theorem). k-body correlations require order k.

### 3.2 The Fock Boundary in Physics

The Fock boundary in quantum mechanics is where the NUMBER of particles is itself a quantum observable — states are superpositions of different particle numbers. This is the domain of quantum field theory (QFT), where the vacuum can fluctuate into particle-antiparticle pairs and the photon number in a laser cavity is indeterminate.

No fixed-order description suffices for Fock-space physics. The entity order itself is quantum. This is the same Fock boundary as in computation: where the structure of the computation depends on intermediate results.

---

## 4. The Interpretive Claim

### 4.1 What This Is

A VOCABULARY for quantum mechanics, not a new theory. The formalism (Hilbert spaces, Born rule, unitary evolution) is unchanged. What changes is the FRAMING:

- Wavefunctions are ENTITIES, not descriptions of entities
- Measurement is CONSTITUTIVE PROJECTION, not passive observation
- Entanglement is NON-FACTORABILITY, not nonlocal connection
- Collapse is DIMENSIONAL REDUCTION, not physical change

### 4.2 What This Gives Us

A connection between quantum foundations and:
- **Parallel computation**: the same liftability test determines parallelizability AND quantum factorability
- **Rendering**: photon wavefunctions and rendering photons live in the same mathematical space (Pith's particle field framework IS a Monte Carlo estimator for the path integral)
- **Cognition**: consciousness encountering the Fock boundary of self-observation parallels measurement encountering the Fock boundary of quantum observation

### 4.3 What This Does NOT Give Us

- No new predictions (the formalism is unchanged)
- No resolution of the "hard problem" of measurement (WHY projection creates definiteness is still open)
- No mechanism for collapse (constitutive projection is a description, not an explanation)

The value is in the CONNECTIONS and the VOCABULARY, not in new physics.

---

## 5. The Deepest Question

If the lift is constitutive — if projection CREATES definiteness — then what is the status of the full Fock space before projection?

In rendering: before projection, the photon field is a high-dimensional distribution. Real, computable, but not directly observable from any single viewpoint.

In quantum mechanics: before measurement, the wavefunction is... what? Real? Mathematical? Potential?

The liftability framework doesn't answer this. But it reframes it: the question isn't "does the wavefunction exist before measurement?" The question is "does the RELATIONSHIP exist before projection?" And the answer, in every domain we've studied, is: the relationship is the MOST real thing. The projection — the measurement, the rendering, the computation — is what creates the APPEARANCE of independent parts.

The entity is the relationship. The parts are the shadow.

---

## References

- Bell, J. S. (1964). On the Einstein Podolsky Rosen paradox. Physics.
- Bohr, N. (1928). The quantum postulate and the recent development of atomic theory.
- Everett, H. (1957). Relative state formulation of quantum mechanics.
- Fuchs, C. A. & Peres, A. (2000). Quantum theory needs no interpretation.
- Wheeler, J. A. (1978). The past and the delayed choice double slit experiment.
- Zurek, W. H. (2003). Decoherence, einselection, and the quantum origins of the classical.
