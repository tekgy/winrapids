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

# seed-the-next-landscape — pure-math-scan proposals

Written: 2026-04-10
By: pure-math-scan (blue)

After Waves A+B (combinatorial atoms + polynomial algebra, 106 tests green).

---

## Proposals

### 1. Numerical Regime Documentation Wave (rigor)

Every primitive in Waves A+B has a numerical regime boundary — a point where the
algorithm changes behavior (returns None, loses precision, or silently degrades).
These boundaries are documented per-function but not collected anywhere as a
structured artifact that downstream callers can query.

The pattern that emerged today:
- stirling2: safe n≤25, cancellation ratio 10^10 at n=25, None beyond
- binomial_coeff: exact result fits u64, but u64 intermediates overflow at C(67,33)
  → u128 fix. The "safe" boundary is where intermediate products exceed u128.
- poly_gcd: reliable for well-separated roots, unreliable for near-common factors
- log_double_factorial: exact via double_factorial up to n=33, log-gamma beyond

**Proposal**: a structured `NumericalRegime` annotation or doc section for every
primitive, covering:
- Domain: what inputs are valid?
- Precision: what's the error bound?
- Overflow boundary: at what n does the exact path fail?
- Cancellation regime: does the formula have an alternating-sum structure?
- Preferred alternative: if this path is treacherous, which path is safe?

This is not new code — it's the rigor layer above the implementations we already
have. Would produce a pub-grade "numerical analysis" section for each primitive.

**Owner**: math-researcher or pure-math-scan
**Campsite**: `numerical-regime-documentation` under rigor

---

### 2. Op::TropicalMinPlus — Kingdom A Unlock for PELT/Viterbi (architecture)

Scout established today: PELT (changepoint detection) is Kingdom A in the
tropical semiring. min-plus: a ⊕ b = min(a,b), a ⊗ b = a + b. The identity
is +∞. The degenerate is NaN (no valid path).

Currently the Op enum has: Add, Max, Min, ArgMin, ArgMax, DotProduct, Distance.
Missing: TropicalMinPlus (and its dual TropicalMaxPlus).

Adding TropicalMinPlus to Op would:
- Unlock PELT changepoint detection as a prefix scan
- Unlock Viterbi (HMM decoding) as a prefix scan
- Unlock Floyd-Warshall (all-pairs shortest paths) as a prefix scan
- All three are currently classified Kingdom B; all three could be Kingdom A

This is a small, well-bounded addition: one Op variant, identity = f64::INFINITY,
degenerate = f64::NAN, combine = a + b (the ⊗ operation; ⊕ is handled by the
grouping). The Blelloch prefix tree padding is already correct once identity() is
defined.

**Owner**: pure-math-scan (can implement now)
**Campsite**: `op-tropical-semiring` under architecture
**Estimated scope**: <100 lines, ~10 tests

---

### 3. Additive-vs-Inclusion-Exclusion Path Audit (rigor + research)

From the garden note written during this pause: every combinatorial quantity
reachable by both an additive recurrence AND an inclusion-exclusion formula has
two computation paths — one numerically safe (recurrence, stays in naturals),
one treacherous (alternating sum, lifts to integers for cancellation).

The claim: for tambear's combinatorial atoms, we always ship the recurrence and
document the closed form. But the audit is incomplete — we haven't verified that
ALL current implementations follow this rule. Some may have slipped through as
inclusion-exclusion implementations without the recurrence alternative.

**Proposal**: audit every combinatorial function across number_theory.rs,
special_functions.rs, and information_theory.rs for:
1. Does it use an alternating sum internally?
2. Is there a recurrence alternative?
3. If the alternating sum is the only implementation, flag it and implement the
   recurrence.

Specific candidates to check:
- Stirling1: currently uses DP recurrence — correct
- Stirling2: inclusion-exclusion — documented, bounded at n≤25
- Bell: Bell triangle — correct
- Derangement: recurrence — correct
- log_multinomial: log-gamma path, no cancellation — correct
- euler_totient: direct formula via factorization — worth verifying
- möbius function: direct formula — worth verifying
- Any entropy estimators in information_theory.rs using alternating sums?

**Owner**: pure-math-scan or adversarial (adversarial has the bug-finding instinct)
**Campsite**: `combinatorial-path-audit` under rigor

---

### 4. Wave C — Extended Special Functions (missing-primitives)

The gap analysis identified these as unimplemented in special_functions.rs:
- Bessel Y_n (second kind) — needed for waveguide modes, cylindrical harmonics
- Bessel K_n (modified, second kind) — needed for Laplace kernel in spatial stats
- Airy functions Ai, Bi — needed for quantum tunneling, diffraction
- Elliptic integrals K(k), E(k), Π(n,k) — needed for pendulum exact solution,
  geodesics on ellipsoid, AGM-based algorithms
- Associated Legendre P_n^m — needed for spherical harmonics
- Spherical harmonics Y_n^m — needed for multipole expansions
- Hypergeometric ₂F₁ — universal glue for every classical orthogonal polynomial
- Lambert W function — needed for delay differential equations, entropy calculations
- Polylogarithm Li_s(z) — needed for Fermi-Dirac, Bose-Einstein distributions

The adversarial proposal #4 (special-function-poles) is the adversarial view of
this same gap — the dangerous boundaries of functions we already have. Wave C is
the forward view — implementing the functions we don't have yet.

**Owner**: pure-math-scan
**Campsite**: `wave-c-extended-special-functions` under missing-primitives
**Priority**: after Op::TropicalMinPlus (small) and the numerical regime doc

---

### 5. Polynomial Root Finding (missing-primitives)

poly_roots is not implemented yet — it's on the Wave B backlog but wasn't
included in the landing because root-finding is genuinely harder than the
other polynomial operations and deserves its own campsite.

The correct implementation:
- Companion matrix eigendecomposition for small-degree polynomials (≤12)
- Laguerre's method for higher degree (globally convergent for complex roots)
- Sturm chain for real root isolation (exact count in interval)
- Aberth-Ehrlich method for all roots simultaneously (fast parallel convergence)

These are distinct algorithms serving different use cases:
- Sturm: "how many real roots in [a,b]?" → exact integer answer
- Companion matrix: exact (up to eigenvalue precision) for all roots at once
- Laguerre: finds one root at a time, divide out, repeat
- Aberth: finds all roots simultaneously, GPU-friendly

**Owner**: pure-math-scan
**Campsite**: `polynomial-root-finding` under missing-primitives

---

## Navigator's coordination question — my read:

**Highest immediate leverage**: Op::TropicalMinPlus (proposal 2). Small scope,
unlocks three Kingdom A reclassifications (PELT, Viterbi, Floyd-Warshall), and
directly completes the op-identity-method work from earlier today.

**Rigor debt most urgent**: Numerical regime documentation (proposal 1) and the
combinatorial path audit (proposal 3). Both are cheap relative to the correctness
guarantee they provide.

**Forward investment**: Wave C special functions (proposal 4) and polynomial roots
(proposal 5). These are real work but each is well-bounded.

The rolling window primitives (navigator proposal 2) and the parallel EMA bridge
(navigator proposal 3) both sit at the intersection of my work and the GPU TA
farm. I can implement rolling_max/min/std from the same Kingdom A template once
the TropicalMinPlus Op is in place — the sliding window maximum is a deque-based
algorithm that decomposes cleanly into prefix scan with a different Op.


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

