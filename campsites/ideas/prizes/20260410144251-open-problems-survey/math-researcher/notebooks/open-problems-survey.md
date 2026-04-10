# Open Problems Survey — What Tambear Could Contribute To

## The Question

Which open mathematical problems, if any, would benefit from tambear's
unique position: every primitive, from first principles, at scale, with
full intermediate sharing?

## Category 1: Computational Verification of Conjectures

These are conjectures where large-scale computation could provide evidence
(or counterexamples).

### 1. Collatz Conjecture (already explored — see memory)

Tekgy has done deep work here: 2-adic fixed point, Mihailescu self-correction,
cascade decay, fold at q=4. Paper 016 from session 2026-04-03.

Tambear contribution: massively parallel trajectory computation, structural
analysis of orbit statistics, spectral analysis of stopping times.

### 2. Goldbach's Conjecture

Every even integer > 2 is the sum of two primes.

Verified computationally to ~4×10¹⁸ (Oliveira e Silva 2013).

Tambear contribution: number theory sieve primitives on GPU,
parallel prime testing, Goldbach partition counting at scale.
Requires: `sieve_of_eratosthenes`, `miller_rabin`, arbitrary-precision arithmetic.

### 3. Riemann Hypothesis

All non-trivial zeros of ζ(s) have real part 1/2.

Verified computationally for first ~10¹³ zeros (Platt & Trudgian 2021).

Tambear contribution: Riemann-Siegel formula computation on GPU,
zero isolation via argument principle, Odlyzko-Schönhage algorithm.
Requires: arbitrary-precision complex arithmetic, FFT at scale.

### 4. Twin Prime Conjecture

Infinitely many primes p where p+2 is also prime.

Tambear contribution: sieve methods, counting twin primes to larger bounds,
statistical analysis of twin prime gaps.

## Category 2: Empirical Mathematics

### 5. Random Matrix Theory in Finance

Marchenko-Pastur law gives the bulk eigenvalue distribution of random
correlation matrices. Financial correlation matrices deviate from this —
the deviations encode market structure.

Already have: `marchenko_pastur_classify` in special_functions.

Open questions:
- Which eigenvalue deviations predict future returns?
- Does the Tracy-Widom distribution describe the largest eigenvalue fluctuations?
- What's the correct null model for financial correlations?

Tambear contribution: eigenvalue analysis at scale across thousands of
assets × thousands of time windows. MFDFA of eigenvalue time series.

### 6. Universal Properties of Financial Returns

Stylized facts (fat tails, volatility clustering, leverage effect) are
universal across assets and markets. WHY?

Open: Is there a single microscopic mechanism that produces all stylized facts?
Analogy to universality in critical phenomena — does the market sit at a
self-organized critical point?

Tambear contribution: family22_criticality already computes `phase_transition`
and `mfdfa`. The question is whether the multifractal spectrum is universal
across assets (same h(q) shape, different parameters).

### 7. Optimal Transport and Wasserstein Geometry

Computational optimal transport is a hot area (Cuturi 2013 — Sinkhorn
entropic regularization). Wasserstein distances define a geometry on the
space of probability distributions.

Open: efficient computation of multi-marginal optimal transport.

Tambear contribution: Sinkhorn algorithm as a primitive (matrix scaling),
Wasserstein barycenters, transport plans.
Requires: Sinkhorn iterations = accumulate(rows, exp, Add) + normalize.
This is Kingdom A — parallelizable.

## Category 3: Computational Topology

### 8. Persistent Homology at Scale

Current persistent homology is limited to ~10⁴ points (cubic complexity
of the boundary operator). GPU-accelerated implementations could push
to ~10⁶.

Already have: family14_topological in fintek.

Open: efficient persistent homology for time series (sliding window).
Cubical complexes for gridded data (images, spatial fields).

### 9. Topological Data Analysis for Financial Markets

Emerging field: use persistent homology to detect market crashes
(Gidea & Katz 2018). The Betti numbers of the Vietoris-Rips complex
built from sliding-window embeddings change before crashes.

Tambear contribution: TDA primitives + time series + financial data
= unique combination. Could validate or refute the crash prediction claims.

## Category 4: Information-Theoretic

### 10. Partial Information Decomposition (PID)

Williams & Beer (2010): decompose mutual information I(X;Y→Z) into
unique, redundant, and synergistic components. Multiple proposed measures
exist, none universally accepted.

Open: which PID axioms are "right"? Can we empirically distinguish
proposals using large-scale experiments?

Tambear contribution: implement ALL proposed PID measures, run on
identical datasets, compare. The library that has every variant can
adjudicate empirically.

## Priority Assessment

For tambear's mission (every primitive, everywhere, provably correct):

**Most aligned** (grows the catalog AND contributes to open problems):
- Random matrix theory (#5) — extends existing eigenvalue infrastructure
- Optimal transport (#7) — new primitive family, high demand
- PID (#10) — extends information theory catalog
- Persistent homology at scale (#8) — extends existing TDA

**Interesting but distant** (needs infrastructure that doesn't exist yet):
- Riemann Hypothesis (#3) — needs arbitrary-precision complex arithmetic
- Goldbach (#2) — needs number theory sieve infrastructure
- Collatz (#1) — already explored, needs 2-adic arithmetic

The convergent theme: the primitives needed for open problems are
independently useful primitives that we should build anyway.
