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

# Information Theory — Paper Verification of New Primitives

Verified against original papers and canonical references.
Date: 2026-04-10, commit range c005fe0..25400e4

## Status: ALL NEW PRIMITIVES VERIFIED CORRECT

---

### 1. Hellinger Distance — CORRECT

**Reference**: Hellinger (1909), "Neue Begründung der Theorie quadratischer Formen"

**Formula**: H²(P,Q) = ½ Σ (√pᵢ - √qᵢ)²

**Implementation** (`information_theory.rs:985`):
```rust
0.5 * Σ (√pᵢ - √qᵢ)²
```

**Verification**:
- Formula matches: YES (the ½ factor is correct — some references omit it and define H² = Σ(√p-√q)²; our convention matches LeCam/Tsybakov)
- NaN for negative probs: YES (correct guard)
- H(P,P) = 0: YES (√p - √p = 0 for all i)
- H ∈ [0,1]: YES (bounded by Cauchy-Schwarz)
- Metric properties: distance version takes sqrt, so H satisfies triangle inequality

**Relationship chain**: H² = 1 - BC where BC = Bhattacharyya coefficient
- This is verified: hellinger_distance_sq(p,q) = 1 - bhattacharyya_coefficient(p,q)
  should hold. VERIFY: expand ½Σ(√p-√q)² = ½Σ(p - 2√(pq) + q) = ½(1 + 1 - 2BC) = 1 - BC ✓

### 2. Total Variation Distance — CORRECT

**Reference**: Standard measure theory. See Tsybakov (2009), Ch. 2.

**Formula**: TV(P,Q) = ½ Σ |pᵢ - qᵢ|

**Implementation** (`information_theory.rs:1009`):
```rust
0.5 * Σ |pᵢ - qᵢ|
```

**Verification**:
- The ½ factor is correct (TV ∈ [0,1], not [0,2])
- For probability distributions: TV = max_A |P(A) - Q(A)| (Scheffé's identity)
- TV(P,P) = 0: YES
- Pinsker's inequality: TV ≤ √(D_KL/2) — not tested but mathematically guaranteed

### 3. Chi-Squared Divergence — CORRECT

**Reference**: Pearson (1900), standard f-divergence.

**Formula**: χ²(P||Q) = Σ (pᵢ - qᵢ)² / qᵢ

**Implementation** (`information_theory.rs:1021`):
- Correct formula
- Returns +∞ when q=0 and p≠q: YES (correct)
- Returns 0 when p=q=0: YES (handles both-zero case)
- NOT symmetric: correctly documented

**Note**: This is the Pearson chi-squared, not the Neyman chi-squared (which divides by p).
The Neyman version: χ²(Q||P) = Σ(p-q)²/p. Both are f-divergences.

### 4. Rényi Divergence — CORRECT

**Reference**: Rényi (1961), "On measures of entropy and information"

**Formula**: D_α(P||Q) = (1/(α-1)) log Σ pᵢ^α qᵢ^{1-α}

**Implementation** (`information_theory.rs:1040`):
- General formula: YES, matches Rényi's definition
- Limit α→1: delegates to KL divergence (correct — by L'Hôpital's rule)
- α=0: D₀ = -log(Σ qᵢ [pᵢ>0]) — measures support overlap. CORRECT.
- α=∞: log max(pᵢ/qᵢ) — max-divergence. CORRECT.
- Returns +∞ when p>0, q=0: YES (correct — P not abs. cont. w.r.t. Q)
- Non-negative: CORRECT (guaranteed by Gibbs' inequality generalization)
- Monotone in α: CORRECT (proven by Rényi)
- NaN guard for negative α: YES

**Potential issue**: For α very close to 1 (but not within 1e-12), the general formula
could have precision loss. The limit delegates at |α-1| < 1e-12, which is sufficient.

### 5. Bhattacharyya Coefficient and Distance — CORRECT

**Reference**: Bhattacharyya (1943), "On a measure of divergence between two statistical populations"

**Coefficient**: BC(P,Q) = Σ √(pᵢ qᵢ)
**Distance**: D_B = -ln(BC)

**Implementation**:
- Coefficient clamps negative to 0 via `.max(0.0)`: conservative, correct
- Distance returns +∞ when BC=0: CORRECT (disjoint support)
- BC ∈ [0,1]: CORRECT (by Cauchy-Schwarz)
- D_B ∈ [0,∞): CORRECT

### 6. f-Divergence — CORRECT

**Reference**: Csiszár (1963), Ali & Silvey (1966)

**Formula**: D_f(P||Q) = Σ qᵢ f(pᵢ/qᵢ)

**Implementation** (`information_theory.rs:1108`):
- General formula with user-supplied f: CORRECT
- q=0, p=0 case: returns 0 by convention (0·f(0/0) = 0). CORRECT per Csiszár.
- q=0, p>0 case: returns +∞. CORRECT (the divergence is infinite when Q doesn't cover P).
- p=0, q>0 case: returns q·f(0). CORRECT (contribution from the tail of Q).
- Doc lists special cases — all verified:
  - f(t)=t log t → KL ✓
  - f(t)=-log t → reverse KL ✓
  - f(t)=(t-1)² → chi-squared ✓
  - f(t)=(√t-1)² → squared Hellinger (×2 factor noted) ✓
  - f(t)=|t-1|/2 → total variation ✓

### 7. Joint Entropy — CORRECT

**Reference**: Cover & Thomas (2006), Elements of Information Theory, Ch. 2.

**Formula**: H(X,Y) = -Σᵢⱼ p(i,j) log p(i,j)

**Implementation**: Delegates to `shannon_entropy` on the flattened contingency probabilities.
This is mathematically equivalent: Shannon entropy doesn't care about matrix structure.
CORRECT.

### 8. Pointwise Mutual Information — CORRECT

**Reference**: Church & Hanks (1990), Bouma (2009)

**Formula**: PMI(x,y) = log p(x,y) / (p(x)·p(y))

**Implementation** (`information_theory.rs:1147`):
- Marginals computed from contingency table: CORRECT
- p(i,j) = 0 → PMI = -∞: CORRECT (log 0)
- `positive` flag for PPMI (max(0, PMI)): CORRECT per Levy et al. (2015)
- MI = E[PMI] = Σ p(x,y) PMI(x,y): can verify this invariant

### 9. Wasserstein-1 Distance (1D) — CORRECT

**Reference**: Kantorovich (1942), Villani (2003)

For 1D: W₁ = ∫|F_P(x) - F_Q(x)| dx

**Implementation** (`information_theory.rs:1188`):
- Equal-size case: mean of |sorted_x[i] - sorted_y[i]|. CORRECT (this is the CDF integral
  when both CDFs are step functions with equal steps).
- Unequal-size case: event-merge approach computing CDF integral directly. CORRECT.
  - Iterates through sorted merge of both samples
  - Tracks CDF values for each
  - Integrates |CDF_X - CDF_Y| × (interval width)
  - This is the standard algorithm for discrete CDF Wasserstein.

**Subtlety**: The equal-size shortcut divides by n (giving mean absolute difference
of order statistics), which equals the CDF integral. For unequal sizes, the integral
is unnormalized — which matches the mathematical definition of W₁.

### 10. MMD (RBF) — CORRECT (with note)

**Reference**: Gretton et al. (2012), JMLR

**Formula**: MMD²(P,Q) = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

**Implementation** (`information_theory.rs:1248`):
- U-statistic form (excludes diagonal): CORRECT for unbiased estimation
- Median heuristic for bandwidth: CORRECT (standard practice)
- Returns MMD² not MMD: documented correctly
- Equal-n cross-term excludes i=j: CORRECT per Gretton et al.
- Unequal-n cross-term uses V-statistic: REASONABLE (no natural diagonal to exclude)

**Note**: The bandwidth median heuristic uses only first 50 points for efficiency.
This is an approximation but standard practice for large samples.

### 11. Energy Distance — CORRECT

**Reference**: Székely & Rizzo (2004, 2013)

**Formula**: E(P,Q) = 2E||X-Y|| - E||X-X'|| - E||Y-Y'||

**Implementation** (`information_theory.rs:1315`):
- Cross-term uses all n×m pairs: CORRECT (V-statistic for different samples)
- Within-term excludes diagonal (U-statistic): CORRECT
- Non-negative: clamped to 0 to guard against numerical noise. CORRECT.
- E(P,P) = 0: CORRECT (2·E_within - 2·E_within = 0)

---

## Summary

All 11 new information theory primitives are mathematically correct.
Each matches the canonical paper definition. Edge cases are handled appropriately.
The f-divergence framework correctly unifies the individual divergences.

## Still Missing (from my earlier catalog, not yet implemented)

- Differential entropy (continuous, KDE-based)
- KSG estimator (k-NN mutual information)
- Kozachenko-Leonenko entropy
- Conditional MI / Interaction information
- Directed information
- NSB / Chao-Shen / shrinkage entropy estimators
- Blahut-Arimoto (channel capacity)
- Compression-based complexity (NCD)


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

