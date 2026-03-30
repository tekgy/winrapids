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
