# Pure Math Gaps — Foundational Primitives Missing from Tambear

**Date**: 2026-04-10  
**Scope**: Everything below Layer 1 — the deepest atoms that applied methods call.  
**Method**: Grep audit of `crates/tambear/src/` cross-referenced against the CLAUDE.md math catalog.

---

## What We Already Have (inventory)

Before the gaps: tambear is surprisingly complete in several foundational domains.

### Number Theory — STRONG
- Primality: sieve, segmented sieve, Miller-Rabin (deterministic witness sets), next_prime, prime_count
- Modular arithmetic: gcd, lcm, extended_gcd, mod_inverse, mod_pow, mul_mod, CRT
- Quadratic residues: Legendre symbol, Jacobi symbol, Tonelli-Shanks sqrt_mod
- Number-theoretic functions: euler_totient, mobius, factorize, num_divisors, sum_divisors, divisors, sieve_totients, sieve_spf
- Discrete log: primitive_root, Baby-step Giant-step (BSGS)
- Continued fractions: continued_fraction, convergents, best_rational, cf_period
- Factorization: Pollard's rho, factorize_complete (rho + trial division)
- Diophantine: isqrt, perfect_square, sum_of_two_squares, Pell equation
- Partition function: partition_count (Euler pentagonal recurrence), euler_product_approx
- Cryptographic shells: rsa_keygen, rsa_encrypt, rsa_decrypt, dh_public_key, dh_shared_secret
- Integer linear recurrences: LinearRecurrenceStep with compose_checked, compose_saturating

### Special Functions — STRONG
- Error functions: erf, erfc (near-machine-precision, Lentz CF)
- Gamma family: log_gamma (Lanczos), gamma, log_beta, digamma, trigamma
- Incomplete functions: regularized_incomplete_beta, regularized_gamma_p/q (series + CF)
- Distribution CDFs/PDFs: normal_pdf, normal_cdf, normal_sf, normal_quantile, t_cdf, f_cdf, chi2_cdf
- Stirling: stirling_approx, stirling_approx_corrected
- Orthogonal polynomials: chebyshev_t/u, legendre_p/legendre_p_deriv, hermite_he, laguerre_l
- Gauss quadrature: gauss_legendre_nodes_weights
- Bessel functions: bessel_j0, bessel_j1, bessel_jn, bessel_i0, bessel_i1
- Softmax/log-softmax

### Linear Algebra — STRONG
- Dense matrix: LU, Cholesky, QR (Householder), SVD
- Eigendecomposition: symmetric (QR algorithm), power iteration
- Solvers: Ax=b via LU/Cholesky/QR
- Gram-Schmidt (classical + modified)
- Norms: Frobenius, L1, L∞, spectral

### Graph Algorithms — MODERATE
- Traversal: BFS, DFS, topological sort
- Shortest paths: Dijkstra, Bellman-Ford, Floyd-Warshall, A*
- MST: Kruskal, Prim
- Centrality: degree, betweenness, closeness, PageRank, eigenvector
- Flow: max flow (Ford-Fulkerson/BFS augmenting path)
- Matching: bipartite matching (Hungarian)
- Connected components (undirected)

### Numerical Methods — STRONG
- Root finding: bisection, Newton, secant, Brent
- Quadrature: Simpson, Gauss-Legendre, adaptive
- ODE: Euler, RK4, RK45 (adaptive)
- Differentiation: finite differences, complex step, Richardson extrapolation

### Computational Geometry (spatial.rs) — PARTIAL
- Convex hull 2D (Andrew's monotone chain)
- Haversine, Euclidean 2D distance
- Variograms, kriging (geostatistics)

### TDA — PARTIAL
- Persistent homology H₀ (union-find over Rips complex)
- Persistent homology H₁ (boundary matrix reduction)
- Persistence diagrams, Betti curve, betti_numbers

### Series Acceleration — STRONG
- Partial sums/cumsum, Cesàro, Aitken Δ², Shanks/Wynn epsilon, Richardson, Euler transform, Levin

### BigInt / BigFloat — PRESENT
- U256 (fixed 4-limb), BigInt (variable Vec<u64>, FFT multiply), BigFloat (arbitrary precision)

---

## Gaps — Organized by Priority (Foundation Impact)

Priority = how many other methods are blocked by this gap.  
A missing binomial coefficient blocks every combinatorial method. A missing convex hull blocks computational geometry.

---

### TIER 1 — Highest Blockers (everything depends on these)

#### 1.1 Combinatorial Atoms — MISSING ENTIRELY

These are called by: probability, statistics, information theory, ML, coding theory, everything.

| Missing Primitive | Why It Matters | What It Blocks |
|---|---|---|
| `binomial_coeff(n, k) -> u64` (exact integer) | C(n,k) for small-n exact counts | Every discrete probability calculation |
| `log_binomial(n, k) -> f64` | ln C(n,k) via log-gamma for large n | Binomial/hypergeometric/negative binomial likelihoods |
| `multinomial(n, ks: &[usize]) -> f64` | n!/(k₁!k₂!...kₘ!) | Multinomial distributions, Dirichlet-multinomial |
| `catalan(n) -> u64` | C_n = C(2n,n)/(n+1) | Parsing, tree counting, ballot problems |
| `stirling_first(n, k) -> i64` | Signed Stirling numbers of first kind | Rising factorial transforms, cycle counting |
| `stirling_second(n, k) -> u64` | Stirling numbers of second kind | Partition of set problems, Bell polynomial |
| `bell_number(n) -> u64` | B_n = Σ S(n,k) | Total partition counts |
| `fibonacci(n) -> u64` | Standard recurrence | Algorithmic analysis, matrix exponentiation demos |
| `lucas(n) -> u64` | L_n = F_{n-1} + F_{n+1} | Primality testing (Lucas-Lehmer), sequence analysis |
| `derangement(n) -> u64` | D_n = (n-1)(D_{n-2}+D_{n-1}) | Permutation statistics |
| `falling_factorial(n, k)` | n·(n-1)·...·(n-k+1) | Pochhammer symbols, hypergeometric series |
| `rising_factorial(n, k)` | n·(n+1)·...·(n+k-1) | Hypergeometric series |
| `harmonic_number(n)` | H_n = Σ 1/k | Coupon collector, expected search length |
| `double_factorial(n)` | n!! = n·(n-2)·... | Quantum mechanics, Hermite polynomial normalization |

**Note**: `binomial_pmf` and `binomial_cdf` exist in `special_functions.rs`, but these call `log_gamma` for the coefficient rather than exposing `log_binomial` as a standalone primitive. The primitive (`log_binomial`) is embedded but not surfaced.

#### 1.2 Polynomial Arithmetic — MISSING ENTIRELY

Polynomials are the algebra of algorithms. Every interpolation, every generating function, every filter design calls polynomial arithmetic.

| Missing Primitive | Why It Matters |
|---|---|
| `poly_eval(coeffs, x)` using Horner's method | Every polynomial evaluation — O(n) vs O(n²) naive |
| `poly_add(a, b)` | Building block for all polynomial operations |
| `poly_mul(a, b)` | Convolution — O(n²) schoolbook |
| `poly_mul_fft(a, b)` | O(n log n) convolution via FFT — needed for large degree |
| `poly_div(a, b) -> (quotient, remainder)` | Polynomial division via synthetic/long division |
| `poly_gcd(a, b)` | Subresultant algorithm — foundational for factorization |
| `poly_eval_multi(coeffs, xs)` | Evaluate polynomial at multiple points efficiently |
| `poly_roots_real(coeffs)` | Companion matrix eigenvalues (real roots) |
| `poly_interpolate(xs, ys)` | Lagrange/Newton interpolation → coefficient form |
| `ntt(a, modulus)` | Number Theoretic Transform for exact polynomial multiplication mod prime |

**Note**: `chebyshev_nodes` and `chebyshev_coefficients` exist in `interpolation.rs`, but these are Chebyshev-specific. No general polynomial arithmetic primitives exist.

#### 1.3 Finite Field Arithmetic — MISSING

Finite fields GF(p) and GF(p^n) underpin cryptography, error correction, and the NTT.

| Missing Primitive | Why It Matters |
|---|---|
| `gf_add(a, b, p)` | Addition in GF(p) — trivial (mod_add) but needs explicit primitive |
| `gf_mul(a, b, p)` | Multiplication in GF(p) (= mul_mod, already have this) |
| `gf2_poly_mul(a, b)` | Polynomial multiply over GF(2) (XOR convolution) — for CRC/LDPC |
| `gf2_poly_div(a, b)` | Polynomial division over GF(2) — for CRC computation |
| `gf_poly_eval(poly, x, p)` | Polynomial evaluation in GF(p) |
| `berlekamp_massey(sequence)` | Minimal LFSR — needed for stream cipher analysis |

---

### TIER 2 — Significant Gaps (block entire subfields)

#### 2.1 Graph Algorithms — Missing Key Algorithms

Despite good coverage, these are foundational graph primitives missing:

| Missing Primitive | Algorithm | Why It Matters |
|---|---|---|
| `strongly_connected_components` | Tarjan's or Kosaraju's | Dependency analysis, feedback vertex sets, causal DAG validation |
| `articulation_points` | DFS-based bridge finding | Network vulnerability analysis, graph decomposition |
| `bridge_edges` | DFS-based | Same as above |
| `biconnected_components` | DFS + stack | Planar graph testing building block |
| `euler_circuit` | Hierholzer's algorithm | Chinese postman, DNA assembly |
| `minimum_cut` | Stoer-Wagner (global min cut) | Network reliability, image segmentation |
| `maximum_matching_general` | Blossom algorithm (Edmond's) | General graph matching (bipartite only currently exists) |
| `isomorphism_check` | VF2 algorithm | Graph comparison, chemical structure matching |

#### 2.2 Computational Geometry — Missing Core Primitives

Only convex hull 2D exists. Missing:

| Missing Primitive | Why It Matters |
|---|---|
| `voronoi_2d(points)` | Nearest-facility, interpolation, mesh generation — via Fortune's algorithm |
| `delaunay_2d(points)` | Triangulation — dual of Voronoi; used in FEM, mesh interpolation |
| `point_in_polygon(point, polygon)` | Ray casting — foundational for spatial queries |
| `line_segment_intersect(a, b, c, d)` | Boolean and intersection point — geometric predicate |
| `convex_hull_3d(points)` | 3D hull — used in physics simulation, bounding volumes |
| `alpha_shape(points, alpha)` | Generalized hull for point cloud shape extraction |
| `nearest_neighbor_brute(query, points)` | O(n) reference — needed to validate spatial index |
| `minkowski_sum_2d(a, b)` | Robot motion planning, collision detection |
| `segment_distance(a, b, p)` | Point-to-segment distance — foundational for spatial queries |

**Note**: KNN exists (`knn.rs`) but it's a spatial index (k-d tree), not a geometric predicate primitive.

#### 2.3 Special Functions — Missing Families

| Missing Primitive | Why It Matters |
|---|---|
| `jacobi_poly(n, a, b, x)` | Generalizes Legendre/Chebyshev/Gegenbauer — used in spectral methods |
| `gegenbauer_poly(n, lambda, x)` | Ultraspherical polynomials — used in electrodynamics, radial functions |
| `associated_legendre(n, m, x)` | P_n^m(x) — building block for spherical harmonics |
| `spherical_harmonic(l, m, theta, phi)` | Y_l^m — essential for quantum mechanics, geophysics, graphics |
| `zernike_poly(n, m, r, theta)` | Circular wavefront analysis — optical aberration, eye tracking |
| `bessel_y0(x)`, `bessel_y1(x)`, `bessel_yn(n, x)` | Bessel functions of second kind — boundary value problems |
| `bessel_k0(x)`, `bessel_k1(x)` | Modified Bessel K — heat equation, finance (VG process) |
| `hypergeometric_2f1(a, b, c, z)` | Gauss hypergeometric — unifies many special functions |
| `hypergeometric_1f1(a, b, z)` | Kummer/confluent hypergeometric — Laguerre, Tricomi |
| `airy_ai(x)`, `airy_bi(x)` | Airy functions — WKB approximation, quantum tunneling |
| `elliptic_k(k)`, `elliptic_e(k)` | Complete elliptic integrals — pendulum, geodesics, AGM |
| `elliptic_f(phi, k)`, `elliptic_e_incomplete(phi, k)` | Incomplete elliptic integrals |
| `lambert_w(x)` | W(x) where W·e^W = x — appears in combinatorics, delay DEs |
| `clausen_function(x)` | Cl₂(x) = -∫ln|2sin(t/2)|dt — appears in polylogarithm theory |
| `polylogarithm(n, z)` | Li_n(z) — Fermi-Dirac, Bose-Einstein integrals |
| `riemann_zeta(s)` | ζ(s) for complex s — Euler products exist, proper ζ does not |
| `hurwitz_zeta(s, a)` | ζ(s,a) generalization — Lerch transcendent family |

**Note**: Bessel J₀, J₁, Jₙ, I₀, I₁ exist. The Y and K families (second kind and modified second kind) are missing. These are needed for exterior boundary problems.

#### 2.4 Integer and Combinatorial Sequences — Missing

| Missing Primitive | Why It Matters |
|---|---|
| `motzkin(n)` | Motzkin numbers — paths, secondary RNA structure |
| `narayana(n, k)` | Narayana numbers — finer Catalan decomposition |
| `euler_number(n)` | E_n (not Euler's e) — Taylor series of sec(x) |
| `bernoulli_number(n)` | B_n — appears in Euler-Maclaurin summation |
| `partition_by_k(n, k)` | Number of partitions of n into exactly k parts |
| `compositions(n, k)` | Ordered partitions of n into k positive parts |
| `necklace_count(n, k)` | Burnside/Polya — up to rotation |
| `power_set_size(n)` | 2^n — trivial but should be explicit |
| `integer_partition_list(n)` | Actual list of partitions, not just count |

---

### TIER 3 — Important But Not Immediately Blocking

#### 3.1 Abstract Algebra — Missing

| Missing Area | Specific Primitives |
|---|---|
| LLL Lattice Reduction | `lll_reduce(basis)` — post-quantum crypto, closest vector problem |
| Hermite Normal Form | `hnf(matrix)` for integer matrices |
| Smith Normal Form | `snf(matrix)` — invariant factor decomposition |
| Group operations | `dihedral_group_mul(n, a, b)`, `symmetric_group_compose(perm_a, perm_b)` |
| Permutation primitives | `perm_order(p)`, `perm_inverse(p)`, `perm_cycle_type(p)`, `perm_sign(p)` |
| Ring arithmetic (Z/nZ) | These exist as mod_*, but not as explicit Ring types with full operator algebra |
| Polynomial rings | GCD, factorization, irreducibility testing over GF(p) |

#### 3.2 Topology / TDA — Partial Coverage

TDA module exists (H₀, H₁ persistent homology), but:

| Missing Primitive | Why It Matters |
|---|---|
| `euler_characteristic(simplicial_complex)` | χ = V - E + F — topological invariant |
| `betti_numbers_simplicial(complex)` | β₀, β₁, β₂,... for general simplicial complexes |
| `persistent_homology_h2(points)` | H₂ (2-cycles) — voids in point clouds |
| `cech_complex(points, r)` | Čech complex construction — more accurate than Rips |
| `mapper_graph(data, lens, cover)` | Mapper algorithm — topological data visualization |
| `geodesic_distance_on_mesh(mesh, source)` | Fast marching method on triangulated surfaces |

#### 3.3 Logic / SAT / Constraint — Completely Missing

| Missing Primitive | Why It Matters |
|---|---|
| `dpll_sat(clauses)` | Basic SAT solver (Davis-Putnam-Logemann-Loveland) |
| `unit_propagation(formula)` | Core BCP step in CDCL SAT solvers |
| `resolution_step(clause_a, clause_b)` | Boolean resolution — building block for theorem provers |
| `unification(term_a, term_b)` | Martelli-Montanari — type inference, logic programming |
| `occurs_check(var, term)` | Unification correctness predicate |

These are genuinely foundational — every automated reasoning system calls these.

---

## Cross-Cutting Analysis

### What's surprisingly complete

1. **Number theory depth**: Miller-Rabin, Pollard rho, Pell equation, Tonelli-Shanks, BSGS discrete log — this is well past typical library coverage.

2. **Special functions rigor**: The `erfc` implementation with the Taylor/CF boundary bug caught and fixed (2026-04-10) is a good example of publication-grade quality. Lanczos log_gamma, Lentz CF for incomplete beta — solid.

3. **Series acceleration**: Aitken, Wynn epsilon, Richardson, Euler, Levin, Borwein — this is niche math most libraries don't touch. Good.

4. **Orthogonal polynomials**: Chebyshev T/U, Legendre, Hermite (probabilist's), Laguerre — solid. The Gauss-Legendre node computation via Newton's method on Legendre polynomial roots is good.

### The structural gap: no combinatorial algebra layer

Everything that counts things — binomial, Catalan, Stirling, Bell, Bernoulli, generating functions — is either absent or buried inside other computations. This is a foundational gap because:

- Every discrete probability distribution needs `log_binomial`
- Every hypergeometric test needs `log_binomial`
- Every information-theoretic bound (Hamming, Singleton, Gilbert-Varshamov) needs `log_binomial`
- Every partition statistic needs `partition_count` (exists) and `stirling_second` (missing)

### The polynomial algebra gap

No `poly_eval` (Horner), no `poly_mul`, no `poly_gcd`. These are atoms used by:
- All interpolation (we have Lagrange, Newton, Chebyshev in `interpolation.rs` but these should compose from polynomial atoms)
- All root finding for polynomials (companion matrix is in `linear_algebra.rs` but needs polynomial input)
- Error-correcting codes (Reed-Solomon needs polynomial arithmetic over GF(p))
- Signal processing FIR/IIR filter design

### The spherical harmonics gap

`associated_legendre` → `spherical_harmonic` is a dependency chain where we have the bottom (Legendre P_n), have `legendre_p_deriv`, but not `P_n^m(x)` (associated Legendre). Associated Legendre blocks:
- Spherical harmonics (quantum chemistry, graphics, geophysics)
- Multipole expansions
- Fast Multipole Method decomposition

---

## Recommended Build Order

Building in dependency order:

**Wave A — Combinatorial atoms (no deps, unlock everything)**
1. `log_binomial(n, k)` — expose what already exists in binomial_pmf internals
2. `binomial_coeff(n, k)` — exact for small n, delegate to log_binomial for large
3. `multinomial(n, ks)` — calls log_binomial
4. `falling_factorial(n, k)`, `rising_factorial(n, k)`, `double_factorial(n)` — O(k) products
5. `harmonic_number(n)` — cumsum of 1/k
6. `catalan(n)` — via log_binomial
7. `stirling_first(n, k)`, `stirling_second(n, k)` — recurrence tables
8. `bell_number(n)` — via stirling_second
9. `derangement(n)` — recurrence
10. `fibonacci(n)`, `lucas(n)` — matrix exponentiation form (O(log n))

**Wave B — Polynomial algebra (deps: log_binomial, basic arithmetic)**
1. `poly_eval(coeffs, x)` — Horner's method
2. `poly_add(a, b)`, `poly_sub(a, b)`, `poly_scale(a, c)`
3. `poly_mul(a, b)` — schoolbook O(n²)
4. `poly_mul_fft(a, b)` — via existing FFT
5. `poly_div(a, b)` — synthetic division
6. `poly_gcd(a, b)` — subresultant PRS
7. `poly_interpolate(xs, ys)` — Newton form
8. `poly_roots_real(coeffs)` — companion matrix eigenvalues

**Wave C — Extended special functions (deps: log_gamma, legendre_p)**
1. `bernoulli_number(n)` — Akiyama-Tanigawa or B_n via ζ
2. `associated_legendre(n, m, x)` — recurrence from legendre_p
3. `spherical_harmonic(l, m, theta, phi)` — calls associated_legendre
4. `bessel_y0(x)`, `bessel_y1(x)`, `bessel_yn(n, x)` — Wronskian relations from J₀/J₁
5. `bessel_k0(x)`, `bessel_k1(x)` — via I₀/I₁ and Wronskian
6. `airy_ai(x)`, `airy_bi(x)` — Stokes phenomenon, asymptotic + power series
7. `elliptic_k(k)`, `elliptic_e(k)` — AGM iteration (fastest known method)
8. `hypergeometric_2f1(a, b, c, z)` — series + analytic continuation
9. `lambert_w(x)` — Halley's method
10. `polylogarithm(n, z)` — series + functional equations

**Wave D — Graph completeness (deps: existing graph primitives)**
1. `strongly_connected_components` — Tarjan's (single DFS)
2. `articulation_points`, `bridge_edges` — DFS lowlink
3. `biconnected_components` — DFS + stack
4. `euler_circuit` — Hierholzer's
5. `minimum_cut` — Stoer-Wagner
6. `maximum_matching_general` — Edmonds' blossom (complex but foundational)

**Wave E — Computational geometry (deps: convex hull, basic geometry)**
1. `point_in_polygon(point, polygon)` — ray casting
2. `line_segment_intersect(a, b, c, d)` — determinant predicate
3. `segment_distance(a, b, p)` — point-to-segment
4. `voronoi_2d(points)` — Fortune's sweep
5. `delaunay_2d(points)` — dual of Voronoi or Bowyer-Watson
6. `convex_hull_3d(points)` — randomized incremental
7. `alpha_shape(points, alpha)` — filtered Delaunay

**Wave F — Abstract algebra (deps: linear algebra, number theory)**
1. `perm_sign(p)`, `perm_cycle_type(p)`, `perm_order(p)`, `perm_inverse(p)` — O(n) algorithms
2. `hermite_normal_form(mat)` — row reduction for integer matrices
3. `smith_normal_form(mat)` — integer matrix invariant factors
4. `lll_reduce(basis)` — Gram-Schmidt + size reduction + swap

---

## Summary Table

| Domain | Coverage | Key Gaps |
|---|---|---|
| Number theory | 90% | AKS primality (more rigorous than Miller-Rabin but not needed), quadratic sieve |
| Special functions | 70% | Bessel Y/K, Airy, elliptic integrals, 2F1 hypergeometric, spherical harmonics |
| Combinatorics | 20% | Binomial, Stirling, Bell, Catalan, generating functions — almost nothing |
| Polynomial algebra | 5% | Poly eval/mul/div/gcd completely absent |
| Graph algorithms | 65% | SCC, articulation, min-cut, general matching, isomorphism |
| Computational geometry | 25% | Voronoi, Delaunay, point-in-polygon, 3D hull |
| Abstract algebra | 30% | LLL reduction, HNF/SNF, permutation group algebra |
| TDA | 50% | H₀/H₁ done; H₂, Čech complex, Mapper missing |
| Logic/SAT | 0% | Complete gap — DPLL, resolution, unification |
| Finite fields | 10% | GF(2) polynomial arithmetic, Berlekamp-Massey |

---

*The highest-leverage gap is combinatorial atoms — they unblock the entire discrete probability and coding theory subgraph. Polynomial arithmetic is second because it underlies both algebraic algorithms and digital signal processing.*
