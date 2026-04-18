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

# Math-Researcher's Next-Landscape Proposals

Written: 2026-04-10

## From mathematical families still thin

**1. Hilbert transform + analytic signal**
Blocks phase-amplitude coupling, instantaneous frequency, envelope detection.
One FFT-based primitive unlocks an entire signal processing sub-family.
Requires complex output — ties directly to ComplexMat/complex f64 foundation gap.
Build order: ComplexMat → FFT complex output → iFFT → Hilbert (~10 lines).

**2. Escort distribution as universal tilting**
`escort(probs, alpha)` primitive unifies Rényi/power-mean/generalized-dimension.
Potential paper: universal structure of parameterized information measures.
Low implementation cost, high theoretical leverage.

**3. Sinkhorn algorithm**
Entropy-regularized optimal transport. O(n²/eps²) vs O(n³) exact.
Needed for Wasserstein-p (p > 1), Wasserstein barycenters, distribution interpolation.
Kingdom C (iterative fixed-point), converges fast in practice.

**4. Copula families**
Gaussian, Clayton, Gumbel, Frank, t-copula.
Currently zero copula support. Copulas separate marginals from dependence structure.
Natural next step for the financial correlation family.

**5. Riemann zero portrait experiment**
Run the full complexity toolkit on 1000+ zeta zeros.
Compare to market eigenvalue spacings measure by measure (MFDFA, permutation entropy, RQA).
All primitives exist. Need compute time, not new math.
If measures match across multiple dimensions: universality result, not coincidence.

## From gaps noticed during paper verification

**6. Schur decomposition**
Needed for numerically robust matrix_log and matrix functions generally.
Current direct approach less stable for ill-conditioned matrices.
Standard intermediate in numerical linear algebra (LAPACK uses it for matrix_log).

**7. False nearest neighbors promotion**
Full Kennel 1992 spec already written by math-researcher.
Private implementation in family15 is correct.
Just needs extraction to tambear::complexity as a public primitive.
Scout has confirmed this as a Type A extraction — lowest-effort, high-value.

**8. SVD workup (Principle 10)**
Our SVD (Golub-Kahan bidiagonalization + QR iteration) is the most complex factorization.
Foundation for pinv, lstsq, rank, effective_rank.
Needs full Principle 10 workup against LAPACK's dgesvd at multiple scales.
Gold-standard oracle: mpmath at 50+ digits for small cases.


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

