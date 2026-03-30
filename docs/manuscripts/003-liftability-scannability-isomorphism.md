# The Liftability-Scannability Isomorphism: A Unified Theory of Parallelizable Computation

**Draft — 2026-03-30**

---

## Abstract

We identify a precise structural isomorphism between two independently-discovered principles of parallelizable computation: the *liftability principle* from particle field rendering (Pith project) and the *scannability test* from GPU parallel prefix scans (timbre project). Both ask the same question: does a computation commute with decomposition? Both have the same formal structure: semigroup homomorphisms over decomposable accumulation. Both share the same boundary (the Fock boundary, where self-referential computation prevents decomposition), the same partial lifts (bounded approximations that extend the parallelizable domain), and the same "for free" corollary (adding capabilities without adding pipeline complexity). We argue this is not coincidence but reflects a fundamental property of parallelizable computation, independently discovered in rendering (Pith), GPU data science (timbre), sequence modeling (Mamba), and Bayesian filtering (Särkkä). We formalize the unifying structure and characterize its boundary.

---

## 1. The Two Principles

### 1.1 The Liftability Principle (Rendering)

**Context**: Particle field rendering — a GPU system that estimates rendering integrals via Monte Carlo accumulation of per-particle contributions.

**Definition (Liftability):** A rendering effect E is *liftable* if it can be expressed as a linear functional of the per-particle measure:

E(u,v) = F(Σᵢ f(pᵢ, u, v))

where f is a per-particle function (depends only on particle pᵢ and pixel (u,v), not on other particles) and F is a post-accumulation function.

**Test**: Does the integrand factor as a sum of per-particle contributions? If yes: O(N) computation via additive accumulation. If no: the effect requires inter-particle comparison.

**Classification**:
- *Fully liftable*: exact computation at O(N)
- *Partially liftable*: approximate computation with bounded error (e.g., exponential depth kernel for occlusion)
- *Fock non-liftable*: variable particle count — population-dependent effects

### 1.2 The Scannability Test (Computation)

**Context**: GPU parallel prefix scans — parallelizing sequential algorithms from O(n) to O(log n) depth.

**Definition (Scannability):** A sequential computation with update step state[t] = g(state[t-1], input[t]) is *scannable* if there exists an associative binary operator ⊕ on state-input pairs such that the prefix scan with ⊕ produces the same result as sequential application of g.

**Test**: Does the composition of two update steps form an associative operator? If yes: O(log n) parallel depth via prefix scan. If no: the computation is inherently sequential.

**Classification**:
- *Fully scannable*: exact parallel computation (linear recurrences, associative reductions)
- *Approximately scannable*: parallel computation with bounded error (linearized nonlinear systems)
- *Non-scannable (Fock)*: state-dependent computation structure — branching determined by intermediate results

---

## 2. The Isomorphism

### 2.1 Structural Correspondence

| Aspect | Liftability (Rendering) | Scannability (Computation) |
|---|---|---|
| Question | Can this effect be computed per-particle? | Can this update be composed associatively? |
| Formal structure | Linear functional of measure | Semigroup homomorphism |
| Test | Does integrand factor per-particle? | Does composition form associative operator? |
| Parallelism | O(N) via additive accumulation | O(log n) via prefix scan |
| Liftable class | Effects expressible as Σᵢ f(pᵢ) | Recurrences with associative compose |
| Partial lift | Bounded approximation (depth kernel) | Linearized approximation (EKF) |
| Fock boundary | Variable particle count | State-dependent branching |
| "For free" corollary | Adding effects doesn't add render passes | Adding operations doesn't add kernel launches |
| Construction procedure | Output: per-particle function + accumulation kernel | Output: associative operator + scan template |

### 2.2 The Common Algebraic Structure

Both principles are instances of the same mathematical structure:

**Definition (Decomposable Accumulation).** A computation C over a collection S is *decomposable* if there exists:
1. A function lift: S → M that maps each element to a monoid M
2. A monoid operation ⊕: M × M → M that is associative with identity e
3. An extraction function extract: M → Result

such that C(S) = extract(⊕ᵢ lift(sᵢ)).

For rendering: S = particles, M = accumulation space, lift = per-particle kernel, ⊕ = additive blend.
For computation: S = observations, M = state space, lift = per-element state, ⊕ = associative combine.

**Theorem (Isomorphism).** The liftability test and the scannability test are equivalent to testing whether a computation admits a decomposable accumulation. They are the same test applied to different domains.

*Proof sketch.* A liftable rendering effect E = F(Σᵢ f(pᵢ)) is a decomposable accumulation with lift = f, ⊕ = +, extract = F. A scannable recurrence with associative combine ⊕ is a decomposable accumulation with lift = per-element initialization, ⊕ = combine, extract = state projection. Both are monoid homomorphisms from the free monoid on the input set to the accumulation monoid. □

### 2.3 The Fock Boundary

In both domains, the boundary of the parallelizable class is characterized by SELF-REFERENCE:

**Rendering**: A Fock non-liftable effect requires the number of particles to depend on the current field state. The computation must observe its own output to determine its input. Example: lasing, where stimulated emission creates new photons based on the current photon density.

**Computation**: A non-scannable recurrence requires the computation structure to depend on intermediate results. The computation must observe its own state to determine its next step. Example: nonlinear Kalman where the state transition F(x) depends on the current estimate x.

**Unified**: The Fock boundary is the point where the computation needs more self-awareness than can be encoded in a fixed-dimensional per-element attribute. Below the boundary: the element carries everything it needs (the photon carries its wavelength, the scan state carries its accumulated history). Above the boundary: the element would need to carry information about the collective state, which requires observing all other elements — breaking decomposability.

### 2.4 Partial Lifts

Both domains admit bounded approximations that extend the parallelizable domain:

**Rendering**: Exact occlusion requires depth ordering (non-liftable). The exponential depth kernel exp(-λz) provides an additive approximation with error O(exp(-λΔz)). Higher λ = more accurate = closer to exact.

**Computation**: Nonlinear recurrence x[t] = g(x[t-1]) where g is not affine (non-scannable). Linearizing g around the trajectory estimate yields ĝ(x) = g(x̂) + g'(x̂)·(x - x̂), which IS affine and therefore scannable. Better trajectory estimate = more accurate = closer to exact. This is precisely the Extended Kalman Filter.

**Unified**: A partial lift replaces global self-observation (which breaks decomposability) with local per-element approximation (which preserves it). The quality of the approximation depends on how much the local approximation captures of the global state.

---

## 3. The k-Particle Hierarchy

### 3.1 Stratified Liftability

The liftability framework extends via the k-particle hierarchy:

- Order 1: per-particle functionals (most rendering effects)
- Order 2: per-pair functionals ("biphoton" — entangled pairs, correlations)
- Order k: per-k-tuple functionals (k-body interactions)

A quantity that is non-liftable at order 1 may be liftable at order 2. Example: Bell correlations ⟨σ_A · σ_B⟩ are non-liftable for single photons but liftable for biphotons (entangled pairs in ℝ⁶).

### 3.2 Application to Computation

The same hierarchy applies to computational state:

- Order 1: optimize each subsystem independently (the sequential "funnel")
- Order 2: optimize pairs jointly (the "biphoton" — joint provenance × residency state)
- Order 3: optimize triples jointly (full compiler state: provenance × residency × fusion plan)

Each order captures interactions that lower orders treat as externalities. The joint optimization at order k discovers sharing opportunities invisible at order k-1.

### 3.3 The Fock Boundary Shifts

Increasing the order pushes the Fock boundary outward: interactions that require global state at order 1 may be capturable at order 2 with a larger per-element state. The boundary is not fixed — it depends on the entity definition. The choice of entity (single photon vs biphoton, single state vs joint state) determines what is liftable.

---

## 4. Independent Discoveries

The decomposable accumulation structure has been independently discovered in at least four domains:

1. **Particle field rendering** (Pith, 2026): liftability test for rendering effects
2. **GPU data science** (timbre, 2026): scannability test for sequential algorithms
3. **Sequence modeling** (Mamba, Gu & Dao, 2023): selective state space models as parallel scans
4. **Bayesian filtering** (Särkkä, 2021): Kalman filter as associative prefix sum

Each discovery arrived at the same formal structure from a different starting point. The convergence suggests the structure is fundamental — a property of parallelizable computation itself, not an artifact of any specific domain.

---

## 5. Implications

### 5.1 A Universal Parallelizability Test

The unified framework provides a single test for parallelizability: given a computation C, determine whether C admits a decomposable accumulation. If yes: construct the monoid (lift, ⊕, extract) and execute via parallel scan/accumulation. If no: characterize which interactions prevent decomposition and determine the minimum entity order k at which decomposition becomes possible.

### 5.2 The "For Free" Corollary

Adding a new capability (rendering effect / scan operator / ML algorithm) to a decomposable accumulation system does not add pipeline complexity. The new capability requires only a new lift function. The accumulation/scan infrastructure is invariant. This is the compound value of the approach: the 135th operator is as easy to add as the 1st, and all share the same GPU execution substrate.

### 5.3 Cross-Domain Transfer

Techniques from one domain may transfer to others via the isomorphism:
- Depth kernels (rendering) → partial lifts for nonlinear filters (computation)
- Parallel Welford merge (statistics) → online softmax (attention mechanisms)
- Biphoton extensions (quantum optics) → joint state optimization (compilers)

---

## 6. Formalization

### 6.1 Definition (Decomposable Accumulation System)

A *decomposable accumulation system* (DAS) is a tuple (S, M, ⊕, e, lift, extract) where:
- S is the input set (particles, observations, tokens)
- (M, ⊕, e) is a monoid (the accumulation space with associative operation and identity)
- lift: S → M maps inputs to the monoid
- extract: M → R maps accumulated state to results

The computation over a sequence s₁, ..., sₙ is:

C(s₁,...,sₙ) = extract(lift(s₁) ⊕ lift(s₂) ⊕ ... ⊕ lift(sₙ))

### 6.2 Definition (DAS-Parallelizable)

A computation C is *DAS-parallelizable at order k* if there exists a DAS where the input set is Sᵏ (k-tuples of the original inputs). Order 1 = standard parallelism. Order k > 1 = biphoton-style entity lifting.

### 6.3 Theorem (Fock Characterization)

A computation C is *Fock non-parallelizable* if for all finite k, C is not DAS-parallelizable at order k. Equivalently: C requires the input set size itself to be a variable of the computation (Fock space).

---

## 7. Related Work

- Blelloch (1990): parallel prefix scans with associative operators
- Veach (1997): path integral formulation of light transport (liftability implicit)
- Kerbl et al. (2023): 3D Gaussian Splatting (liftability unrecognized)
- Gu & Dao (2023): Mamba selective SSM as parallel scan
- Särkkä & García-Fernández (2021): Kalman as parallel prefix sum
- Martin & Cundy (2017): parallel linear recurrent neural nets
- Bird (1987): algebraic theory of programs, list homomorphisms (the categorical precursor)

The closest theoretical precursor is Bird's work on list homomorphisms: a function h on lists is a *list homomorphism* if h(xs ++ ys) = h(xs) ⊕ h(ys) for some associative ⊕. This is precisely the decomposable accumulation condition. Our contribution is connecting this to the liftability framework (rendering), the Fock boundary (quantum mechanics), and the practical implications for GPU system design.

---

## References

- Bird, R. S. (1987). An introduction to the theory of lists.
- Blelloch, G. E. (1990). Prefix sums and their applications.
- Gu, A. & Dao, T. (2023). Mamba: Linear-time sequence modeling with selective state spaces.
- Kerbl, B. et al. (2023). 3D Gaussian Splatting for real-time radiance field rendering.
- Martin, E. & Cundy, C. (2017). Parallelizing linear recurrent neural nets over sequence length.
- Särkkä, S. & García-Fernández, Á. F. (2021). Temporal parallelization of Bayesian smoothers.
- Veach, E. (1997). Robust Monte Carlo methods for light transport simulation. PhD thesis.
