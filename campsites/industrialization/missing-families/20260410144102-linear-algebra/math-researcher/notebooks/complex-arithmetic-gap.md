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

# Complex Arithmetic Gap — What It Blocks and How to Close It

## The Gap

tambear's `Mat` is real-only (f64). There is no complex matrix type, no complex
eigendecomposition, no complex FFT output, no Hilbert transform.

## What It Blocks

### Tier 1 — Blocks existing families from being complete:
1. **Non-symmetric eigendecomposition** — eigenvalues of a real non-symmetric matrix
   can be complex (conjugate pairs). Currently `sym_eigen` only works for symmetric.
   Blocks: stability analysis, dynamical systems, matrix functions for general matrices.

2. **Full FFT** — the DFT of real data produces complex output. Currently the FFT
   infrastructure returns magnitude/phase or PSD, discarding the complex structure.
   Blocks: proper spectral manipulation, deconvolution, Wiener filtering, phase recovery.

3. **Hilbert transform** — analytic signal z(t) = x(t) + i·H[x](t).
   Computed via FFT: zero negative frequencies, IFFT.
   Blocks: instantaneous phase, instantaneous frequency, phase-amplitude coupling,
   envelope detection — entire EEG/signal processing sub-family.

### Tier 2 — Blocks new families:
4. **Quantum simulation** — wavefunctions are complex, unitary gates are complex,
   density matrices are Hermitian (complex with real diagonal).

5. **Transfer functions** — H(s) = N(s)/D(s) where s = sigma + j*omega.
   Poles/zeros live in the complex plane.
   Blocks: control theory, filter design, system identification.

6. **Characteristic functions** — phi(t) = E[exp(itX)] is the Fourier transform
   of the probability distribution. Complex-valued.
   Blocks: stable distribution fitting, deconvolution of distributions.

7. **Scattering transform** — complex wavelet coefficients with modulus operation.
   Blocks: texture classification, invariant feature extraction.

## How to Close It

### Option A: Complex Mat type (ComplexMat)
```rust
pub struct ComplexMat {
    pub real: Vec<f64>,
    pub imag: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}
```
Split storage (SoA) is better for GPU than interleaved (AoS) because
the real and imaginary parts can be processed independently in many operations.

### Option B: Paired-f64 in existing Mat
Use `Mat` with 2n columns where columns 0..n are real and n..2n are imaginary.
Less clean but no new type. Breaks existing API assumptions.

### Option C: Generic Mat<T> where T: Ring
Most general but hardest to retrofit. Would need `f64` and `Complex64` to
implement a common `Ring` trait with `zero()`, `one()`, `add()`, `mul()`.

### Recommendation: Option A

ComplexMat with split storage. Implement:
1. `complex_mat_mul(a, b)` — via 4 real mat_muls (or 3 with Gauss trick)
2. `complex_eigen(a)` — Hessenberg → QR iteration with implicit double shift
3. `complex_fft(data)` / `complex_ifft(data)` — full complex DFT
4. `hilbert_transform(data)` — via complex_fft + zero negative freqs + complex_ifft
5. `hermitian_eigen(a)` — eigendecomposition for Hermitian matrices (real eigenvalues)

### Accumulate+Gather for Complex

Complex DotProduct state is (real_sum, imag_sum) — a 2-element state vector.
This is the same scan structure as real DotProduct, just wider.

```
accumulate(Tiled, complex_mul_add, ComplexAdd)
```

where `complex_mul_add((ar,ai), (br,bi)) = (ar*br - ai*bi, ar*bi + ai*br)`
and `ComplexAdd((r1,i1), (r2,i2)) = (r1+r2, i1+i2)`.

Identity element: (0.0, 0.0). Monoid: confirmed.

This fits cleanly into the existing framework. No new kingdom. No Fock boundary.
Complex arithmetic is Kingdom A.


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

