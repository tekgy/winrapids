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

# Measurement as COPA Boundary — The Quantum-Classical Bridge

*From the Aristotle-MathResearcher exchange, 2026-04-10*

## The Observation

The Born rule in quantum mechanics:

p(outcome) = |⟨outcome|ψ⟩|²

takes complex amplitudes (reversible, unitary evolution domain) to real probabilities
(irreversible, classical domain). This is structurally identical to the COPA boundary
crossing in tambear's kingdom taxonomy.

## The Mapping

| Tambear concept | Quantum concept |
|---|---|
| sigma domain (reversible operations) | Unitary evolution (Schrödinger equation) |
| tau domain (irreversible operations) | Measurement (wavefunction collapse) |
| COPA boundary (sigma → tau crossing) | Born rule: |ψ|² |
| Kingdom A operations | Interference + entanglement (within sigma) |
| Fock boundary | The point where parallel quantum branches must be reconciled |

## Why This Is More Than an Analogy

The COPA boundary theorem states that certain operations cannot cross from sigma
to tau without information loss. In quantum mechanics, this is EXACTLY the measurement
problem: measurement irreversibly destroys the phase information (the complex phases
of the amplitudes are lost when we take |ψ|²).

The information that is lost at the boundary:
- **Tambear**: intermediate sharing metadata, execution order flexibility
- **Quantum**: relative phases between superposition branches

In both cases, the information exists BEFORE the boundary crossing and is
irretrievably gone AFTER. The boundary is real, not a limitation of the formalism.

## The Three Operations (Aristotle's Framework)

Aristotle identified that quantum hardware needs three operations:

1. **Interfere** — amplitude combining (quantum accumulate)
   - U|ψ⟩: unitary gates combine amplitudes
   - This IS accumulate with complex state
   - Kingdom A: parallelizable, reversible

2. **Entangle** — correlated state creation (no classical analog)
   - CNOT, CZ gates: create states that can't be factored
   - This is a GROUPING operation — it creates non-local correlations
   - Maps to: cross-group accumulate (two qubits that must be treated jointly)

3. **Measure** — irreversible state collapse (the COPA boundary)
   - p = |⟨x|ψ⟩|²: project and normalize
   - Destroys phase, produces classical bit
   - Maps to: gather with irreversible addressing

## Implications for Tambear

1. **Complex f64 is the foundation** — interfere and entangle both operate on
   complex amplitudes. Without complex arithmetic, you can't even represent
   the sigma domain of quantum computation.

2. **The COPA boundary is universal** — it appears in:
   - Quantum: measurement
   - Finance: regime collapse (from superposition of models to single decision)
   - Statistics: hypothesis testing (from full posterior to binary decision)
   - Signal processing: demodulation (from analytic signal to envelope)
   
   All of these are |·|² operations: take a complex/rich representation,
   collapse to a real/reduced one.

3. **The kingdom classification extends naturally** — quantum operations
   are Kingdom A (parallelizable within the sigma domain). Measurement
   is the COPA boundary. No new kingdom needed.

## For the Architecture Documentation

This observation should be recorded alongside the kingdom classification:

> The COPA boundary between sigma and tau domains is the abstract form of
> quantum measurement. Both involve irreversible information loss at the
> transition from a reversible computation domain (complex amplitudes / 
> parallel scan states) to an irreversible output domain (classical
> probabilities / gathered results). The Born rule p = |ψ|² and the
> TamSession gather are structurally isomorphic boundary crossings.


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

