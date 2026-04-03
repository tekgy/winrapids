# Manuscripts

Formal writeups of contributions from the timbre project. Each manuscript is structured for potential publication or patent filing — abstract, contribution, prior art, formal results, implications.

## Manuscript Index

### 001 — Optimal Parallel Variance via Fixed-Reference Centering
**Status**: Draft
**Contribution**: A variance formulation for GPU parallel prefix scans that is both numerically superior to Welford's algorithm (O(log n) vs O(n) error growth) AND computationally cheaper (zero-division combine). The shifted centering technique is old (1970s); applying it to parallel scans to simultaneously improve stability and speed is new.
**Audience**: Numerical analysis, GPU computing, data science systems

### 002 — The Universal Affine Combine for GPU Parallel Scans
**Status**: Draft
**Contribution**: A single associative combine function that handles all 1D linear recurrences — cumulative sums, exponential weighted means, Kalman filters (steady-state), ARIMA(1), and variable-parameter variants. The combine is 2 multiplies + 1 add regardless of which recurrence. Specialization lives entirely in the lift function.
**Audience**: GPU computing, signal processing, parallel algorithms

### 003 — The Liftability-Scannability Isomorphism
**Status**: Draft
**Contribution**: The rendering liftability principle (Pith) and the parallel scan scannability test (timbre) are the same mathematical structure — semigroup homomorphisms over decomposable accumulation. Same Fock boundary (self-reference). Same partial lifts (bounded approximation). Same "for free" corollary. Four independent discoveries (Pith, timbre, Mamba, Särkkä) of the same theorem.
**Audience**: Theoretical computer science, computer graphics, parallel computing
**Note**: Joint with Pith project. Potentially patentable as a METHOD for determining parallelizability.

### 004 — A Stateful GPU Computation Compiler
**Status**: Draft
**Contribution**: A GPU computation engine whose primary optimization is elimination (not computing what hasn't changed: 25,714x) rather than fusion (computing faster: 2.3x). The compiler is a stateful query optimizer with four injectable state inputs (provenance, dirty bitmap, residency map, specialist registry). The persistent store extends sharing across queries and sessions. Six optimization types ranked by magnitude.
**Audience**: Database systems, GPU computing, data science systems

### 005 — Recursive Lifting and the Fock Boundary
**Status**: Draft
**Contribution**: The Fock boundary (where computation becomes non-liftable due to self-reference) can be pushed outward by redefining the fundamental entity at higher orders — the "biphoton trick." Recursive application (diad → triad → N-tuple) captures progressively more inter-component interactions. Applied to compiler state: the joint (provenance × residency × fusion) state space reveals optimizations invisible to sequential optimization of each level. Musical analogy: Whitacre cluster chords as N-photon entities.
**Audience**: Theoretical computer science, quantum information (analogy), compiler design

### 006 — Self-Reference as the Universal Computational Boundary
**Status**: Draft
**Contribution**: Five apparently unrelated impossibility results — the Fock boundary, Gödel's incompleteness theorems, the halting problem, the quantum observer effect, and the AI frame problem — all arise from the same mechanism: a system that must model itself to function but cannot do so completely without circular dependency. The *partial lift* (a bounded approximation that captures enough self-knowledge to act) is the universal strategy for operating at this boundary.
**Audience**: Philosophy of computation, foundations of mathematics, theoretical computer science

### 007 — Approximate Self-Knowledge Is Sufficient: Partial Lifts as Bounded Rationality
**Status**: Draft
**Contribution**: Effective agents (biological, artificial, computational) require *partial lifts* — fixed-dimensional approximations of their own state — not complete self-knowledge. Connects Simon's bounded rationality, the Extended Kalman Filter's linearization, and partial lifts in parallel computation theory. The ORDER of approximation (dimensionality of the self-model) is the key variable for cognitive capacity; evolution optimizes this order subject to metabolic cost.
**Audience**: Epistemology, cognitive science, bounded rationality

### 008 — Exponential Weighted Means as Unconscious Bayesian Filtering
**Status**: Draft
**Contribution**: EWM smoothing with parameter α is algebraically IDENTICAL (bit-for-bit in IEEE 754) to the steady-state Kalman filter for a random walk observed in Gaussian noise, with α equaling the steady-state Kalman gain K_ss. This connection (known in control theory since Muth 1960) is virtually unknown in quantitative finance and ML, where α is treated as a heuristic tuning parameter. GPU-parallel: EWM and Kalman share the same kernel via the universal affine combine.
**Audience**: Quantitative finance, behavioral economics, signal processing

### 009 — Neural Systems as Partial-Lift Computers
**Status**: Draft
**Contribution**: Biological neural circuits implement partial lifts of different orders, with lift order (dimensionality of the maintained self-model) determining cognitive capacity. Miller's 7±2 working memory limit is the lift order of the prefrontal cortex. Attention mechanisms are adaptive lift-order selection. The cognitive hierarchy (reactive → habitual → planning → reflective) is a progression of lift orders. The Fock boundary is the limit where effective self-modeling breaks down.
**Audience**: Computational neuroscience, cognitive science, evolutionary biology

### 010 — Timbre as N-Body Interaction Pattern
**Status**: Draft
**Contribution**: Musical timbre is formalized as an N-body interaction pattern among spectral partials, using the decomposable accumulation framework. Order-1: pitch (independent partials). Order-2: beating and consonance (Helmholtz). Order-3: chord quality. Order-N: timbre (the emergent quality distinguishing instruments). Connects to the k-particle framework in quantum optics and the GPU scan operator family via a common algebraic structure. The Fock boundary for timbre is feedback/self-oscillation where partial count depends on acoustic state.
**Audience**: Music theory, psychoacoustics, mathematical music theory

### 011 — The Sharing Principle: Why Computers Should Remember
**Status**: Draft
**Contribution**: Stateless systems discard the knowledge that previous computations generated. Measured evidence from GPU data science: stateful provenance-caching systems achieve 25,714x speedup over stateless systems. The optimization hierarchy — don't compute (25,714x) > don't transfer (26x) > don't dispatch (2.3x) > compute faster (1.5-3x) — demonstrates that the industry focus on levels 3-4 leaves orders of magnitude uncaptured. The sharing principle: the most powerful optimization is recognizing when computation is unnecessary.
**Audience**: Systems design, software philosophy, computing paradigms

### 012 — The Entity Is The Relationship: A Liftability Interpretation of Quantum Measurement
**Status**: Draft (Speculative — requires physicist review)
**Contribution**: Quantum "paradoxes" arise from treating order-2 entities (wavefunctions) as order-1 entities (particles with definite properties). Bell's theorem is a liftability test proving quantum mechanics requires order 2. Measurement collapse is constitutive projection, not revelation. Entanglement is non-factorability at order 1. The formalism of quantum mechanics is unchanged; the paper provides vocabulary for WHY it has the structure it does via decomposable accumulation.
**Audience**: Foundations of quantum mechanics, philosophy of physics

### 013 — Weirdness as Dimensional Gap
**Status**: Draft
**Contribution**: Quantum "weirdness" is not a property of quantum systems but of the dimensional gap between the system's true state space and the observer's measurement capacity. Formalized as weirdness = entity_order − observation_order. The straw's "how many holes?" debate is structurally identical to wave-particle duality — a higher-dimensional object projected onto incompatible lower-dimensional descriptions. Macro objects feel non-weird only because multiple simultaneous projections close the gap trivially.
**Audience**: Philosophy of physics, foundations of quantum mechanics, epistemology

### 014 — Sort-Free DataFrames: Eliminating O(n log n) from GPU Data Science
**Status**: Draft
**Contribution**: Sort is unnecessary for the four most common sort-dependent operations (groupby, deduplication, join, top-k). Hash-based alternatives achieve 2-17x speedup because hash scatter is O(n) single-pass vs O(n log n) multi-pass sort. With a persistent group index (built once, reused via provenance cache), groupby becomes O(n_groups) metadata read + O(n) scatter — sort-free on the hot path. Proposes sort-free DataFrame design as an architectural principle: sort exists only for explicitly-requested sorted output.
**Audience**: Database systems, GPU computing, data science

### 015 — Polynomial Closure of Market Microstructure Indicators: 11 Fields Generate 90+ Bin-Level Features
**Status**: Draft
**Contribution**: An 11-field minimum sufficient representation (9 power sums + 2 order statistics) is sufficient to derive all polynomial bin-level market microstructure indicators — mean, variance, VWAP, return moments (through kurtosis), Sharpe-analog, CV, range. A 7-field core handles the majority. At 31 cadences: 341 × 8 = 2.7 KB of registers per ticker, < 1% of Blackwell SM budget, enabling zero-contention sequential accumulation per ticker. All 90+ indicators are extract expressions over the same 11 fields — the ManifoldMixtureOp pattern at signal-farm scale. Exact exceptions characterized (autocorrelation lag cross-products, scan-based leaves, true percentiles).
**Audience**: Quantitative finance, GPU computing, signal processing, market microstructure

### 016 — Why All Collatz-Type Maps Use d = 2
**Status**: Draft (revised — added Guaranteed Division Theorem)
**Contribution**: Two independent results: (A) *Algebraic* — the Guaranteed Division Theorem proves d = 2 is the unique prime modulus where +1 guarantees d-divisibility for all coprime inputs (three-line proof via φ(d) = 1). (B) *Dynamical* — at the Nyquist boundary m = 2d−1, only d ∈ {2, 3} yield contracting maps, and only d = 2 has sufficient margin (25% vs 3.8%) to prevent non-trivial cycles. The ℤ₃ Collatz (5,3) admits exactly two 2-cycles, proving contraction < 1 is necessary but not sufficient. The causal chain: φ(d) = 1 → guaranteed division → full margin → cycle-freeness. Appears novel — literature assumes d = 2 without proving it forced. Publishable independently.
**Audience**: Number theory, dynamical systems, American Mathematical Monthly
