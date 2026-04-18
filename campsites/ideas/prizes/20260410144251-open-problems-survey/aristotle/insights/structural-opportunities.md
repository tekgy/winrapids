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

# Open Problems Where Tambear Has Structural Advantage

*2026-04-10 — Aristotle*

## The Frame

Not "which problems can tambear solve?" but "which problems have structural properties that MATCH tambear's architecture?" A structural match means: the problem's difficulty lives where tambear's strengths are. A structural mismatch means: tambear can compute things relevant to the problem, but the core difficulty is elsewhere.

## Tier 1: Strong Structural Match

### 1. Riemann Hypothesis — Computational Evidence via GUE Statistics

**The match**: Montgomery's pair correlation conjecture (1973) predicts that the non-trivial zeros of ζ(s) have the same local statistics as eigenvalues of random matrices from GUE (Gaussian Unitary Ensemble). Verifying this requires computing millions of zeros and their spacing statistics.

**Why tambear**: This is exactly what tambear does — compute a quantity (zeros), then run exhaustive statistics (spacing distribution, pair correlation, n-level correlations, nearest-neighbor spacings) and compare to a theoretical distribution (GUE). The primitives already exist: FFT for zero computation (Odlyzko-Schönhage), eigendecomposition for GUE sampling, KS/AD/chi-squared tests for distribution comparison, entropy and divergence measures for distribution distance.

**The structural advantage**: Nobody has run the COMPLETE statistical battery on Riemann zeros at scale. Individual statistics (pair correlation, nearest-neighbor spacing) have been checked. But tambear's ".discover()" philosophy — run EVERY test, keep ALL results in superposition — would produce the most comprehensive statistical portrait of the zeros ever computed. If any statistic deviates from GUE, that's a potential counterexample indicator (or a new phenomenon).

**Difficulty**: The Riemann-Siegel formula for computing zeros at height T requires O(T^{1/2}) precision. For T > 10^{13} (beyond current records), this means arbitrary precision arithmetic, which tambear doesn't have yet. But the statistical analysis of EXISTING zero tables is feasible now.

**Verdict**: HIGH opportunity for novel statistical analysis. LOW probability of proving RH, but potential for discovering new empirical phenomena in the zero distribution. **The announcement**: "Most comprehensive statistical portrait of Riemann zeros reveals [new phenomenon / confirms GUE to unprecedented precision]."

### 2. Collatz Conjecture — Building on Existing Work

**The match**: The project has already done significant original work on Collatz. The three-layer decomposition (mixing, contraction, carry) with nilpotent spectral gap = 1 is a genuine contribution. The frontier is narrow: temporal equidistribution at j=3 for generic starting points.

**Why tambear**: The Markov chain analysis of Collatz IS accumulate+gather — the transition matrix operations are Kingdom A (matrix-vector multiply = tiled accumulate). The spectral analysis (eigenvalues, spectral gap) uses primitives already in the library. The carry depth analysis uses number-theoretic primitives.

**The structural advantage**: Tambear's ability to run EVERY statistical test in parallel means we can characterize Collatz orbits from more angles simultaneously than anyone else. The .discover() approach to orbit analysis — computing ALL descriptive statistics, ALL spectral properties, ALL mixing diagnostics for each orbit class — would reveal patterns invisible to targeted analysis.

**Difficulty**: The fundamental difficulty (spatial → temporal equidistribution) is a deep mathematical problem that computation alone can't solve. But computational evidence could guide the proof strategy.

**Verdict**: MODERATE opportunity. The work is already advanced. Next step: formalize the results into a paper. The four-pillar analysis + nilpotent mixing theorem + carry trichotomy is publishable NOW.

### 3. Distribution of Primes — Computational Verification of Conjectures

**The match**: Many prime number conjectures have the form "this quantity behaves like [distribution]." Twin prime constant, prime k-tuples, gaps between primes, distribution of primes in arithmetic progressions.

**Why tambear**: These are EXACTLY the kind of problems where exhaustive statistical analysis at scale produces value. Compute primes (sieve), compute the statistic of interest, compare to the conjectured distribution. Run EVERY distribution test. Flag deviations.

**The structural advantage**: Scale. GPU-accelerated sieving can enumerate primes much faster than CPU. The statistical battery is already built. The ".discover()" paradigm means we don't have to choose which test to run — we run all of them.

**Specific targets**:
- Cramér's conjecture: maximal gaps between primes grow like (log p)^2. Computational verification to larger primes than currently published.
- Maier's theorem: primes in short intervals oscillate around the expected density. Characterize the oscillation spectrum.
- Hardy-Littlewood k-tuple conjecture: verify the predicted constants for twin primes, sexy primes, cousin primes, etc. to higher precision.

**Verdict**: HIGH opportunity for producing the definitive computational dataset. LOW novelty in methodology but HIGH value in scale and completeness.

## Tier 2: Moderate Structural Match

### 4. Matrix Multiplication Exponent ω

**The problem**: What is the smallest ω such that n×n matrix multiplication can be done in O(n^ω) operations? Currently ω ≈ 2.373 (Alman-Williams 2024). The conjecture is ω = 2.

**Why partial match**: Tambear implements matrix multiplication from first principles. The tiled accumulate kernel IS the concrete instance of the abstract problem. But the problem is about ASYMPTOTIC complexity, not concrete performance. Tambear's value would be in implementing and benchmarking exotic multiplication algorithms (Strassen, Winograd, Coppersmith-Winograd variants) to find the PRACTICAL crossover points.

**Verdict**: LOW — the problem is pure combinatorics/algebra, not computational.

### 5. P vs NP — Computational Hardness Landscape

**Why partial match**: Tambear's kingdom classification (A/B/C/D) IS a practical version of complexity classification. Kingdom A = embarrassingly parallel ≈ NC. Kingdom B = sequential outer loop ≈ P. Kingdom C = data-dependent ≈ unclear. The Fock boundary = the point where self-reference makes parallelization impossible.

**The structural insight**: Tambear's experience classifying real algorithms into kingdoms provides EMPIRICAL DATA on the parallelism/sequentiality landscape. If we publish the full kingdom classification of ALL implemented algorithms, that's a dataset about computational complexity that doesn't exist elsewhere.

**Verdict**: LOW direct impact on P vs NP, but HIGH interest as a practical complexity classification.

### 6. Birch and Swinnerton-Dyer Conjecture

**The problem**: The rank of an elliptic curve E is equal to the order of vanishing of L(E, s) at s = 1.

**Why partial match**: Computing L-functions and their values at specific points IS accumulate (Euler product = product over primes, each factor is a local computation). The number theory module already has Euler product computation. The statistical analysis of the rank distribution across families of curves is exactly tambear's strength.

**Verdict**: MODERATE — the computation is feasible, the conjecture is deep.

## Tier 3: Novel Opportunities (Not Traditional Open Problems)

### 7. Universal Mathematical Constants Discovery

**The idea**: Run tambear's complete statistical battery on the outputs of thousands of mathematical functions. Look for unexpected numerical coincidences, near-integers, algebraic relations between constants. The triple identity ζ(2) = partition function = Euler product already discovered this way.

**Why tambear**: This is EXACTLY what .discover() is for. Compute everything. Compare everything. Let structural fingerprints emerge.

**The dream**: Discover a new mathematical identity or constant relationship that nobody has seen because nobody has computed this many things simultaneously and compared the results.

**Verdict**: HIGHEST novelty. This is where tambear's unique architecture (run everything, never gate) produces genuinely new mathematics. Not an open problem — a discovery methodology.

### 8. Empirical Universality Classes

**The idea**: Many mathematical phenomena exhibit universality — the same statistical behavior appears in different systems. GUE statistics in random matrices, primes, and quantum chaos. KPZ universality in growth models. Tracy-Widom in extremes.

**Why tambear**: Compute the statistics of MANY different systems and automatically detect when they share the same distribution. The ".discover()" comparison across methods is exactly the tool: if two systems produce agreeing view_agreement scores across ALL statistics, they're in the same universality class.

**Specific target**: Does the distribution of carry depths in Collatz-type maps exhibit universality? Our data shows carry depth trichotomy {0, 1, j} for m=3. What about other odd m? If the distribution changes character at the subcriticality threshold m=4, that's evidence for a universality class boundary.

**Verdict**: HIGH novelty. Connecting different parts of mathematics through computational statistics.

## Recommendation

The highest-leverage opportunities:
1. **Riemann zero statistical portrait** — achievable now, publishable, uses existing primitives
2. **Collatz formalization** — the work is done, it needs to be written up
3. **Universal constant discovery** — highest novelty, uses tambear's unique architecture
4. **Universality class detection** — connects to deep mathematics, uses .discover()

The common thread: tambear's value is not in solving open problems directly. It's in producing the most comprehensive COMPUTATIONAL EVIDENCE ever assembled for a given problem, using the run-everything philosophy. The evidence either confirms existing conjectures to new precision, or reveals new phenomena that guide theoretical work.

**The announcement that matters**: "We computed EVERY statistic for [X] at scale [Y] and found [Z]." The completeness is the differentiator. Nobody else runs every test.


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

