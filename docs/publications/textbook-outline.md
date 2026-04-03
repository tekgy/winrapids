# Textbook: Computation Through the Accumulate Lens

*Working title. Also considered: "Tam Knows: A New Foundation for Numerical Computing"*

## Audience
Graduate students and practitioners in statistics, ML, data science, scientific computing. Prerequisites: linear algebra, basic probability, some programming. The textbook TEACHES the connections that no existing textbook draws.

## The Pedagogy
Every chapter shows a domain through the accumulate lens. The student learns ONE framework and sees it applied everywhere. Instead of "learn each method separately" (50 formulas), it's "learn the pattern once, see 50 applications."

Each chapter: motivation → the traditional approach → the accumulate decomposition → what's shared with previous chapters → exercises.

## Structure

### Part I: The Framework (Chapters 1-6)
1. **What IS Computation?** — accumulate + gather, that's it. Motivating examples. Why two operations suffice.
2. **The Eight Operators** — Add, Welford, RefCentered, Affine, Särkkä, Max, ArgMax, SoftmaxWeighted. Each with examples. The associativity test.
3. **Grouping Patterns** — All→reduce, ByKey→scatter, Prefix→scan, Windowed→rolling, Tiled→GEMM, Segmented, Masked. Same operation, different parameter.
4. **The MSR Principle** — Minimum Sufficient Representation. Accumulate the minimum, extract everything. Delayed collapse. The Fock boundary.
5. **The Three Kingdoms** — Commutative (A), Sequential (B), Iterative (C). Classification of every algorithm.
6. **Transforms** — Sort, FFT, wavelet, embedding, ranking. Preprocessing that changes representation, orthogonal to kingdoms.

### Part II: The Domains (Chapters 7-16)
7. **Descriptive Statistics** — 7 fields → 41+ extractions. The MSR in its purest form. RefCentered vs naive (with the destruction gradient).
8. **Hypothesis Testing** — SAME 7 fields, different questions. "The t-test IS a descriptive statistic viewed as a hypothesis." F = t².
9. **Regression** — The cross-product matrix. GramMatrix. Normal equations. Ridge = regularized GramMatrix. Lasso via coordinate descent.
10. **Time Series** — The Affine scan family. EWM, Kalman, ARIMA, GARCH. Kingdom B in action. Adam as 4 EWM channels.
11. **Signal Processing** — FFT as the canonical transform. Convolution = multiply in frequency domain. Wavelets for multi-resolution.
12. **Information Theory** — Histograms → entropy. MI = shared information. KL = distribution distance. ALL from scatter + count.
13. **Clustering** — Distance matrices as shared infrastructure. KMeans, DBSCAN, hierarchical. Validation metrics for free.
14. **Machine Learning** — Everything composes. Random forest = histogram splits. Gradient boosting = sequential trees. SVM = kernel GramMatrix.
15. **Manifold Geometry** — 3 fields → all distances. Euclidean, hyperbolic, spherical from ONE accumulator. The ManifoldMixtureOp.
16. **Complexity and Chaos** — Cross-kingdom consumers. Embedding → distance → threshold → extract. The multiscale template.

### Part III: The Connections (Chapters 17-20)
17. **The 24 Structural Rhymes** — Kriging=GP, ANOVA=t², IRLS=EM, Adam=EWM, IRT=Fisher=IRLS=Bernoulli. The table that rewrites how we see mathematics.
18. **The IRLS Master Template** — 8 families, 1 primitive. The deepest sharing surface.
19. **MSR Across Domains** — The polynomial degree theorem. Non-polynomial MSRs. The observer's honest count.
20. **The Fock Boundary** — Where accumulate stops and self-reference begins. Partial lifts. Approximate self-knowledge. The connection to Pith's liftability.

### Part IV: The Platform (Chapters 21-25)
21. **Sharing Infrastructure** — TamSession, content-addressed intermediates, the marketplace. 5820x measured.
22. **Single-Pass Compilation** — JIT from .tbs, kernel fusion, the mega-kernel. Why 7 attention types cost less than 1 in PyTorch.
23. **Any GPU** — Multi-backend architecture. CUDA, Vulkan, Metal, CPU. The one-line seam.
24. **The .tbs Language** — Computation for everyone. 100-word vocabulary. Science linting. Auto-insert. Lock/super/discover.
25. **The Superposition Architecture** — All views, never collapse. Structural fingerprints as self-explanation. The path to AGI.

### Appendices
A. **Numerical Stability** — The centered-basis principle. Adversarial proof tables. The 4 bug classes.
B. **The Adversarial Test Suite** — How to break numerical code systematically. 120+ test vectors.
C. **Gold Standard Parity Tables** — 297+ comparisons against R/Python/scipy/sklearn.
D. **The Complete Decomposition Table** — All 500+ algorithms with their (addressing, grouping, expr, op) tuples.

## What Makes This Textbook Different
1. **ONE framework** — not a survey of methods, but a unified lens
2. **Connections first** — every chapter builds on the previous, showing what's shared
3. **Verified** — every claim backed by gold standard tests and adversarial proofs
4. **Executable** — every example is a .tbs script the student can run
5. **Any hardware** — exercises run on the student's GPU, whatever it is
6. **The rhymes** — no other textbook teaches that Kriging=GP or ANOVA=t²

## TAMBEAR Acronym
**T**ransforms · **A**ccumulate · **M**SR · **B**idirectional · **E**very GPU · **A**ssociative · **R**igorous

Seven letters. Seven principles. The name had the DNA from the start.
