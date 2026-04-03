Now I have a thorough picture. Let me write the comprehensive report.

---

# Open Mathematical Prize Problems: Computational Assessment for tambear

## Part 1: Millennium Prize Problems ($1,000,000 each, Clay Mathematics Institute)

Six remain unsolved. The Poincare conjecture was solved by Perelman (2003, prize declined 2010).

---

### 1.1 Riemann Hypothesis

**Statement**: All non-trivial zeros of the Riemann zeta function have real part 1/2.

**Prize**: $1,000,000 (Clay)

**Type**: MIXED -- computation provides evidence, cannot prove.

**Current computational record**: Platt and Trudgian (2021) rigorously verified all zeros up to height t = 3.0001753328 x 10^12 lie on the critical line. Over 10^13 zeros computed total.

**What computation does**: The Odlyzko-Schonhage algorithm evaluates zeta(s) using FFT-based techniques. Each zero must be isolated and verified to lie on the critical line Re(s) = 1/2. Finding a zero OFF the critical line would disprove the hypothesis instantly (worth $1M for a disproof too).

**What a platform needs**:
- Arbitrary precision arithmetic (200-500 bits minimum for large heights, scaling with t)
- FFT-based multiplication for large-number arithmetic
- Parallel evaluation of the Riemann-Siegel formula across many t-values
- Interval arithmetic for rigorous verification (not just floating point)

**tambear fit**: MODERATE-HIGH. tambear has FFT and could build arbitrary-precision arithmetic on GPU. The Riemann-Siegel formula is embarrassingly parallel across t-values -- each zero evaluation is independent. The bottleneck is that GPU hardware natively supports f32/f64, and multi-precision emulation on GPU eliminates the parallel advantage for individual evaluations. However, the parallelism across independent t-values is massive. A platform that does N(t) evaluations in parallel across thousands of GPU cores, each at moderate precision, could push the verification frontier.

**Specific computation**: Evaluate zeta(1/2 + it) for t in some range, isolate zeros, verify each lies on the critical line. Push the record past 10^13.

---

### 1.2 P vs NP

**Statement**: Is every problem whose solution can be verified quickly also solvable quickly?

**Prize**: $1,000,000 (Clay)

**Type**: PRIMARILY THEORETICAL -- computation provides indirect evidence only.

**What computation does**: GPU-accelerated SAT solvers (ParaFROST, torchmSAT) achieve 100x speedups, but this is engineering optimization, not algorithmic. Finding a polynomial-time algorithm for an NP-complete problem would prove P=NP, but no computation can prove P != NP.

**tambear fit**: LOW for the prize itself. HIGH for related practical work (SAT solving, combinatorial optimization). tambear's graph algorithms and optimization families could contribute to practical SAT/MaxSAT solving, but this won't settle the theoretical question.

---

### 1.3 Navier-Stokes Existence and Smoothness

**Statement**: Do smooth solutions to the 3D incompressible Navier-Stokes equations always exist, and if so, do they remain smooth?

**Prize**: $1,000,000 (Clay)

**Type**: MIXED -- computation is central to the research program.

**Current state**: GPU-accelerated direct numerical simulation (DNS) of turbulence has reached 35 trillion grid points on exascale machines (Frontier, 2025). Diff-FlowFSI provides differentiable CFD on GPU.

**What computation does**:
- High-resolution DNS reveals turbulence structure at fine scales
- Detecting candidate blow-up (singularity formation) in numerical solutions
- Adaptive mesh refinement tracking vortex stretching
- Spectral methods (pseudo-spectral FFT-based solvers) are the gold standard

**What a platform needs**:
- 3D FFT (spectral solvers)
- High-order ODE integration (Runge-Kutta, spectral time-stepping)
- Massive parallelism for 3D grid computation
- Numerical stability at extreme Reynolds numbers

**tambear fit**: MODERATE. tambear has FFT, ODE solvers (euler, rk4, rk45), and the GPU pipeline. A focused effort on 3D spectral Navier-Stokes could build on the existing signal processing + numerical methods families. The real question is whether the 3D FFT can be implemented efficiently across GPU backends. This is more an applied physics problem than a number theory problem -- the computation guides theoretical work but cannot constitute a proof.

---

### 1.4 Birch and Swinnerton-Dyer Conjecture

**Statement**: The rank of an elliptic curve equals the order of vanishing of its L-function at s=1.

**Prize**: $1,000,000 (Clay)

**Type**: MIXED -- born from computation, computation continues to guide.

**History**: The conjecture itself was discovered computationally by Birch and Swinnerton-Dyer using EDSAC-2 in the 1960s. They counted points on elliptic curves mod p and noticed the connection to L-functions.

**Current state**: Full BSD verified for all non-CM curves of conductor <= 1000 and rank <= 1 (up to certain odd primes). William Stein's book "The Birch and Swinnerton-Dyer Conjecture, a Computational Approach" details the methodology.

**What computation does**:
- Count points on elliptic curves mod p for many primes p
- Compute L-functions of elliptic curves to high precision
- Verify the BSD formula numerically for specific curves
- Search for curves of high rank (current record: rank >= 29, Elkies 2006)

**What a platform needs**:
- Modular arithmetic (mod p for many primes)
- Arbitrary precision for L-function evaluation
- Parallel evaluation across many primes and curves
- Elliptic curve arithmetic

**tambear fit**: MODERATE. The core computation is "for each prime p up to some bound, count points on E mod p" -- this is embarrassingly parallel. tambear's hash scatter and parallel accumulate could parallelize the point-counting across primes. Would need an elliptic curve arithmetic layer built on top.

---

### 1.5 Yang-Mills Existence and Mass Gap

**Statement**: Prove that quantum Yang-Mills theory exists mathematically and has a positive mass gap.

**Prize**: $1,000,000 (Clay)

**Type**: MIXED -- lattice QCD provides computational evidence.

**Current state**: Lattice QCD on GPU is a major field. SIMULATeQCD runs on multi-GPU clusters. Lattice computations suggest the mass gap exists, but a rigorous mathematical proof is absent. Recent work (2025) combines lattice gauge theory with tensor network methods.

**What computation does**:
- Lattice gauge theory simulations discretize spacetime and compute gauge field configurations
- Monte Carlo sampling of the path integral
- Extraction of particle masses from correlation functions
- Requires massive parallelism and precise linear algebra

**What a platform needs**:
- SU(3) matrix arithmetic (3x3 complex matrices, group operations)
- Monte Carlo / Markov chain methods
- Eigenvalue computation for mass extraction
- Linear algebra with complex arithmetic

**tambear fit**: LOW-MODERATE. tambear has linear algebra (LU, QR, SVD, eigensolvers) and Monte Carlo integration, but lattice QCD is extremely specialized. The existing frameworks (QUDA, SIMULATeQCD) are purpose-built. Building from scratch would be a multi-year effort for marginal advantage.

---

### 1.6 Hodge Conjecture

**Statement**: Every Hodge class on a non-singular complex projective algebraic variety is a rational linear combination of algebraic cycle classes.

**Prize**: $1,000,000 (Clay)

**Type**: PRIMARILY THEORETICAL -- computation has limited role.

**Current state**: Very little computational attack surface. The objects involved (cohomology classes, algebraic varieties in high dimensions) are abstract and infinite-dimensional.

**tambear fit**: VERY LOW. This is one of the most purely theoretical of the Millennium problems. There is no known computational approach that would make progress.

---

## Part 2: Other Major Prize Problems

### 2.1 Beal Conjecture -- $1,000,000

**Statement**: If A^x + B^y = C^z where A,B,C,x,y,z are positive integers with x,y,z > 2, then A,B,C must share a common factor.

**Prize**: $1,000,000 (American Mathematical Society, funded by Andrew Beal)

**Type**: COMPUTATIONAL -- a counterexample search.

**Current record**: Peter Norvig searched all combinations with x,y,z <= 7 and A,B,C <= 250,000, plus x,y,z <= 100 and A,B,C <= 10,000. No counterexample found.

**What computation does**: Brute-force search for a counterexample. If gcd(A,B,C) = 1 and A^x + B^y = C^z with all exponents > 2, the conjecture is disproved and you win $1M.

**What a platform needs**:
- Arbitrary precision integer arithmetic (numbers get enormous: 250000^7 has ~38 digits)
- GCD computation
- Highly parallel enumeration across (A,B,C,x,y,z) space
- Smart pruning (modular arithmetic filters)

**tambear fit**: HIGH. This is a natural GPU problem. Each (A,B,x) pair produces A^x, each (B,y) produces B^y. Check if A^x + B^y is a perfect z-th power with gcd(A,B,C)=1. The search space is partitioned trivially across GPU threads. Modular pre-filtering (check mod small primes whether A^x + B^y can possibly be a z-th power) eliminates most candidates without full-precision arithmetic. tambear's hash scatter could organize results efficiently. FFT-based multiplication handles the large-number arithmetic.

**Specific computation**: Extend the search to A,B,C <= 10^6 with x,y,z <= 7. This is ~10^18 evaluations with pre-filtering, tractable on a GPU cluster over weeks.

---

### 2.2 Collatz Conjecture

**Statement**: For any positive integer, repeatedly applying n -> n/2 (even) or n -> 3n+1 (odd) eventually reaches 1.

**Prizes**: No single large official prize, but Paul Erdos said "Mathematics is not yet ripe enough for such questions" and offered $500 for a proof. Various informal prizes exist. The problem is famous enough that solving it would bring the Fields Medal equivalent of recognition.

**Type**: COMPUTATIONAL for verification, THEORETICAL for proof.

**Current record**: Verified for all n up to 2^71 (approximately 2.36 x 10^21) as of 2025, using GPU acceleration achieving 1,335x speedup over initial CPU algorithm.

**What computation does**: Verify every starting number reaches 1. Also search for cycles (numbers that loop without reaching 1). Four new path records found in the latest verification push.

**What a platform needs**:
- 64-bit or 128-bit integer arithmetic (not arbitrary precision -- numbers grow but are bounded)
- Massive parallelism (each starting number is independent)
- Bit manipulation and efficient branching
- Memory-efficient tracking of verified ranges

**tambear fit**: HIGH. This is embarrassingly parallel and fits naturally on GPU. Each thread takes a number, iterates the map, checks convergence. tambear's parallel architecture could push past 2^71 with sustained GPU effort. The GoldbachGPU paper (2026) showed 99.7% parallel efficiency at 2 GPUs for a similar verification task -- the same architecture applies to Collatz.

**Specific computation**: Push verification past 2^72 or 2^80. Each step is simple integer arithmetic -- multiply by 3, add 1, divide by 2. The bottleneck is throughput, not precision.

---

### 2.3 Goldbach's Conjecture

**Statement**: Every even integer greater than 2 is the sum of two primes.

**Prize**: Faber & Faber once offered $1M (expired 2002). No current large prize, but it remains one of the most famous open problems.

**Type**: COMPUTATIONAL for verification, THEORETICAL for proof.

**Current record**: Verified up to 4 x 10^18 (Oliveira e Silva, 2013), extended to ~9 x 10^18 by recent work. GPU frameworks (GoldbachGPU, 2026) demonstrate 16x memory reduction via bit-packed primes and full verification to 10^12 on a single RTX 3070.

**What computation does**: For each even number 2n, find primes p,q with p+q = 2n. Also track the minimum number of Goldbach representations (the "Goldbach comet").

**What a platform needs**:
- Sieve of Eratosthenes on GPU (segmented sieve)
- Bit-packed prime storage (1 bit per odd number)
- Parallel search: for each even number, scan primes to find a partition
- Memory-efficient segmented approach

**tambear fit**: HIGH. The GoldbachGPU architecture maps directly to tambear's capabilities. The segmented sieve is a parallel prefix-like operation. The verification loop is embarrassingly parallel. tambear's bit-mask infrastructure (FilterJit, mask operations) is purpose-built for exactly this kind of packed-bit computation.

**Specific computation**: Push verification past 10^19 using multi-GPU pipeline with segmented sieve.

---

### 2.4 Twin Prime Conjecture

**Statement**: There are infinitely many pairs of primes differing by 2.

**Prize**: No major formal prize, though solving it would likely earn the Fields Medal or Abel Prize.

**Type**: MIXED.

**Current record**: Largest known twin primes: 2996863034895 x 2^1290000 +/- 1 (388,342 digits, as of January 2025). Theoretical gap reduced to 246 (from infinity) by Zhang/Maynard/Tao/Polymath.

**What computation does**: Finding large twin primes doesn't prove the conjecture but provides evidence. The real contribution is primality testing of large candidates.

**What a platform needs**:
- Large-number arithmetic (hundreds of thousands of digits)
- FFT-based multiplication (essential for numbers this large)
- Parallel primality testing (Miller-Rabin, Lucas-Lehmer)

**tambear fit**: MODERATE. The FFT infrastructure exists. Primality testing at this scale requires FFT-based modular exponentiation for numbers with hundreds of thousands of digits. tambear could build this on top of existing FFT, but competing with PrimeGrid (thousands of volunteers) is hard for a single platform.

---

### 2.5 ABC Conjecture

**Statement**: For co-prime positive integers a+b=c, the radical rad(abc) is "usually" not much smaller than c.

**Prize**: $1,000,000 offered by Nobuo Kawakami specifically for disproving Mochizuki's claimed proof (by finding an inherent flaw). The conjecture itself has no formal prize, but is one of the most important in number theory.

**Type**: COMPUTATIONAL for evidence gathering, THEORETICAL for proof.

**Current state**: ABC@Home (distributed computing, Leiden University) found 23.8 million "abc triples" where rad(abc) < c as of 2014. Mochizuki's claimed proof remains disputed.

**What computation does**: Search for abc triples with high "quality" q = log(c)/log(rad(abc)). The distribution of these qualities informs the conjecture's constants.

**What a platform needs**:
- Integer factorization (to compute rad(abc))
- GCD computation
- Parallel search across (a,b) pairs
- Large number arithmetic

**tambear fit**: MODERATE-HIGH. Factoring many moderate-size numbers in parallel is GPU-friendly. Trial division up to sqrt(n) is embarrassingly parallel. The search space is a natural grid over (a,b) pairs. tambear's hash scatter and parallel accumulate could organize the factorization pipeline.

---

### 2.6 Ramsey Numbers

**Statement**: Determine R(5,5) -- the minimum n such that any 2-coloring of edges of K_n contains a monochromatic K_5.

**Prize**: Erdos offered $100 for R(5,5). Anecdotally, he said of R(6,6): "Imagine an alien force, vastly more powerful than us, landing on Earth and demanding the value of R(5,5) or they will destroy our planet. In that case, we should marshal all our computers and all our mathematicians. If they ask for R(6,6), we should launch a preemptive strike."

**Type**: COMPUTATIONAL -- SAT solving + combinatorial search.

**Current bounds**: 43 <= R(5,5) <= 46. The upper bound of 46 was established in 2024 by Angeltveit and McKay using SAT solvers (Glucose, Kissat) plus linear programming.

**What computation does**: Enumerate graph colorings, use SAT solvers to check satisfiability. The 2024 breakthrough processed 12 million cases with 5.6 million sent to SAT solvers.

**What a platform needs**:
- SAT solving infrastructure
- Graph enumeration and isomorphism testing
- Parallel case processing
- Linear programming

**tambear fit**: MODERATE. tambear has graph algorithms but not SAT solvers. Building a GPU-accelerated SAT solver is a major project. The more natural contribution would be the graph-theoretic filtering and enumeration that reduces the case space before SAT solving.

---

### 2.7 Sierpinski/Riesel Number Problems

**Statement**: Find primes of the form k * 2^n +/- 1 for specific values of k to resolve which k are Sierpinski or Riesel numbers.

**Prize**: No formal cash prize, but PrimeGrid and related projects have been running for 20+ years.

**Type**: PURELY COMPUTATIONAL.

**Current state**: PrimeGrid uses BOINC + GPU. 43 Riesel candidates remain. Recent work includes multi-GPU sieving.

**What computation does**: Primality testing (Proth's theorem, Lucas-Lehmer-Riesel) on numbers with millions of digits. Sieving to eliminate composite candidates.

**What a platform needs**:
- FFT-based modular exponentiation for huge numbers
- Sieving (trial factoring)
- Parallel primality testing

**tambear fit**: MODERATE. Similar to twin primes -- the core is FFT-based large-number arithmetic. CUDALucas already does this on GPU. tambear could build this but wouldn't have an architectural advantage over existing tools.

---

### 2.8 Odd Perfect Numbers

**Statement**: Does an odd perfect number exist?

**Prize**: No formal prize, but it's a 2000-year-old problem.

**Type**: COMPUTATIONAL for verification, THEORETICAL for proof.

**Current record**: Any odd perfect number must exceed 10^1500. If divisible by 3, it exceeds 10^360,000,000.

**What computation does**: Search for odd perfect numbers by checking divisor-sum conditions, or prove bounds that push the minimum size higher.

**What a platform needs**:
- Integer factorization
- Divisor function computation
- Modular arithmetic

**tambear fit**: LOW. The constraints are so severe that brute-force search is hopeless. Progress comes from theoretical number theory, not computation.

---

## Part 3: Problems Where Computation Already Changed History

| Problem | Year | What the Computer Did |
|---------|------|----------------------|
| Euler's Sum of Powers Conjecture | 1966 | Lander & Parkin found counterexample 27^5+84^5+110^5+133^5=144^5 on CDC 6600 |
| Four Color Theorem | 1976 | Appel & Haken proved it by exhaustive case checking (first major computer-assisted proof) |
| Mertens Conjecture | 1985 | Odlyzko & te Riele disproved using LLL lattice reduction |
| Fermat's Last Theorem (partial) | 1993 | Computational verification of cases guided Wiles's proof strategy |
| Kepler Conjecture | 1998/2014 | Hales proved by computer-assisted case analysis; Flyspeck verified formally |
| Bunkbed Conjecture | 2024 | Gladkov found monstrous counterexample (7,222 vertices) via brute-force search |
| Polya Conjecture | 1980 | Counterexample found at n=906,150,257 |
| Collatz verification | 2025 | GPU-accelerated verification to 2^71, 1,335x speedup |
| R(5,5) bound | 2024 | SAT solvers + LP reduced upper bound from 48 to 46 |
| BSD evidence | 1960s-present | Computational discovery of the conjecture itself; ongoing verification |
| Goldbach verification | 2013-2026 | Verified to 9 x 10^18, GPU frameworks achieving full device residency |
| Mersenne primes | 2024 | Luke Durant used cloud GPU cluster (NVIDIA A100s across 17 countries) to find 52nd Mersenne prime: 2^136,279,841 - 1 |

---

## Part 4: Assessment Matrix

| Problem | Prize | Theory/Computation | tambear Fit | Unique Advantage? |
|---------|-------|-------------------|-------------|-------------------|
| Riemann Hypothesis | $1M | Mixed | MODERATE-HIGH | Parallel zeta evaluation across t-values |
| P vs NP | $1M | Theoretical | LOW | None for the prize |
| Navier-Stokes | $1M | Mixed | MODERATE | FFT + ODE + GPU pipeline |
| BSD | $1M | Mixed | MODERATE | Parallel point counting |
| Yang-Mills | $1M | Mixed | LOW-MODERATE | Existing tools dominate |
| Hodge | $1M | Theoretical | VERY LOW | None |
| Beal Conjecture | $1M | Computational | **HIGH** | GPU brute force + modular filtering |
| Collatz | ~$500 + fame | Computational | **HIGH** | Embarrassingly parallel |
| Goldbach | Fame | Computational | **HIGH** | Bit-mask infrastructure is native |
| ABC | $1M (disproof) | Mixed | MODERATE-HIGH | Parallel factorization |
| Twin Primes | Fame | Mixed | MODERATE | FFT multiplication exists |
| R(5,5) | ~$100 | Computational | MODERATE | Graph algorithms exist |
| Sierpinski/Riesel | Fame | Computational | MODERATE | FFT exists |
| Odd Perfect | Fame | Theoretical | LOW | None |

---

## Part 5: What tambear Could UNIQUELY Advance

### Tier 1: Natural Fit -- Start Here

**1. Beal Conjecture Counterexample Search ($1,000,000)**

This is the single best target. Here's why:

The computation is: enumerate (A, B, C, x, y, z) with x,y,z > 2, check if A^x + B^y = C^z with gcd(A,B,C) = 1.

tambear's architecture maps to this perfectly:
- **ScatterJit** for parallel power computation: thousands of (base, exponent) pairs evaluated simultaneously
- **FilterJit** for modular pre-filtering: check mod 2, 3, 5, 7... whether A^x + B^y can be a perfect power before doing full-precision arithmetic
- **Hash scatter** for organizing results by residue class
- **FFT-based multiplication** for the large-integer arithmetic when numbers exceed 64 bits
- **Parallel prefix scan** for carry propagation in multi-precision addition

The search strategy:
1. Fix exponents (x, y, z). Start with (3,3,3), then (3,3,4), etc.
2. For each exponent triple, generate all A^x values up to some bound using parallel GPU computation
3. For each pair (A^x, B^y), compute A^x + B^y
4. Check if the result is a perfect z-th power using Newton's method on GPU (tambear has `newton` in numerical.rs)
5. If yes, check gcd(A,B,C) = 1
6. If gcd = 1: **you found a counterexample worth $1M**

Current record: A,B,C up to 250,000 for low exponents. A single RTX 6000 Pro Blackwell could push this to A,B,C up to 10^6 or beyond, which is 64x the current search volume.

**2. Goldbach Verification**

tambear's FilterJit with packed bitmasks is *literally* what GoldbachGPU built from scratch. The bit-packed prime representation, segmented sieve, and parallel verification are native to tambear's mask-not-filter architecture. The `mask_popcount`, `mask_and`, `mask_or` operations are already implemented.

The computation: for each even n, check that there exist primes p, q with p + q = n.

This maps to:
- Segmented sieve produces a bitmask of primes in each segment
- For each even n in the segment, check if `prime_mask AND rotate(prime_mask, n)` has any set bits
- If no bits set for some n: counterexample found (extremely unlikely but worth checking)

Push past the 9 x 10^18 record.

**3. Collatz Verification**

Pure throughput problem. Each number is independent. The map is:
```
if n % 2 == 0: n = n / 2
else: n = 3*n + 1
```

This fits in a single GPU thread. Launch millions of threads, each verifying a range of starting numbers. Track path records (longest sequences before reaching 1). Push past 2^71.

tambear's compute engine can dispatch this as a simple parallel map.

### Tier 2: Moderate Effort, High Impact

**4. Riemann Hypothesis Zeros**

The Odlyzko-Schonhage algorithm at its core requires:
- FFT-based evaluation of the Riemann-Siegel formula
- Multi-precision arithmetic (200+ bits at current heights)
- Parallel evaluation across independent t-values

tambear has FFT. Building multi-precision on top (using FFT for large multiply, parallel prefix for carry) is architecturally natural. Each zero evaluation is independent -- classic GPU parallelism.

The specific computation would be: implement the Riemann-Siegel Z-function as a tambear kernel, evaluate at millions of t-values simultaneously, isolate and verify zeros.

**5. ABC Triple Search**

For each coprime pair (a, b), compute c = a + b, then factorize a, b, c to compute rad(abc). The factorization of many moderate-size numbers in parallel is GPU-friendly -- trial division up to sqrt(n) for each number, launched across GPU threads.

tambear's hash scatter could organize the factorization: scatter each (a,b) pair to a hash bucket by some residue, process each bucket.

### Tier 3: Interesting But Speculative

**6. Navier-Stokes Exploration**

Build a 3D pseudo-spectral Navier-Stokes solver on tambear's FFT + RK4 infrastructure. The goal: detect candidate singularity formation at extreme Reynolds numbers. This requires 3D FFT (not yet in tambear -- currently 1D and 2D), but the architecture extends naturally.

**7. Elliptic Curve L-functions (BSD)**

Build elliptic curve point-counting on GPU. For each prime p, count points on E mod p. This is a parallel modular arithmetic problem -- one thread per prime, thousands of primes simultaneously. Accumulate the partial products of the L-function.

---

## Part 6: The Unique tambear Advantage

Most existing GPU math tools are either:
- **General-purpose** (CGBN/CUMP): provide arbitrary precision but no algorithmic structure
- **Problem-specific** (GoldbachGPU, CUDALucas): built for one problem, not reusable
- **CUDA-only**: locked to NVIDIA

tambear's unique position:
1. **Multi-backend**: Any GPU (CUDA, Vulkan, Metal via wgpu). A Beal search that runs on NVIDIA and Apple Silicon from the same binary.
2. **Algorithmic building blocks**: FFT, hash scatter, parallel prefix, bitmask operations, Newton's method, eigensolvers -- these compose into number-theoretic algorithms without starting from scratch.
3. **Accumulate-as-primitive**: The `accumulate(grouping, expr, op)` abstraction means "for each group, reduce an expression with an operator." This is exactly what number-theoretic searches do: for each exponent triple, accumulate power computations; for each even number, accumulate prime-pair checks.
4. **Numerical stability by construction**: Centered algorithms, sufficient statistics, Welford-style accumulation. When doing multi-precision arithmetic, catastrophic cancellation is the enemy. tambear's numerical hygiene applies directly.

---

## Part 7: Recommended Strategy

**Immediate (weeks)**: Implement the Beal conjecture search. It's the highest-value target ($1M), the most naturally computational, and maps directly to existing tambear primitives. Extend the search frontier by 10-100x using a single Blackwell GPU.

**Near-term (months)**: Build a multi-precision integer arithmetic layer on tambear (using FFT for multiplication, parallel prefix for carry). This unlocks Riemann zeros, large primality testing, and further Beal search expansion.

**Medium-term**: Collatz and Goldbach verification pushes using tambear's bitmask infrastructure. These are "fame" problems -- no cash prize, but the computational records are tracked by the mathematics community and would demonstrate tambear's capabilities publicly.

**Long-term**: Navier-Stokes spectral solver and BSD elliptic curve computations. These require more infrastructure but have $1M prizes and genuine scientific value.

---

Sources:
- [Millennium Prize Problems - Clay Mathematics Institute](https://www.claymath.org/millennium-problems/)
- [Millennium Prize Problems - Wikipedia](https://en.wikipedia.org/wiki/Millennium_Prize_Problems)
- [Beal Conjecture - AMS](https://www.ams.org/profession/prizes-awards/ams-supported/beal-prize-rules)
- [Beal Conjecture - Wikipedia](https://en.wikipedia.org/wiki/Beal_conjecture)
- [Collatz Conjecture Verification - Springer](https://link.springer.com/article/10.1007/s11227-025-07337-0)
- [Collatz Verification Project](https://pcbarina.fit.vutbr.cz/)
- [Riemann Hypothesis 2026 Status](https://www.mathlumen.com/articles/riemann-hypothesis-2026-status-report)
- [Riemann Hypothesis - Wikipedia](https://en.wikipedia.org/wiki/Riemann_hypothesis)
- [Goldbach Conjecture Verification](https://sweet.ua.pt/tos/goldbach.html)
- [GoldbachGPU Framework](https://arxiv.org/html/2603.02621)
- [Lock-Free GPU Goldbach Verification](https://arxiv.org/html/2603.07850)
- [Twin Prime - Wikipedia](https://en.wikipedia.org/wiki/Twin_prime)
- [R(5,5) Upper Bound](https://arxiv.org/pdf/2409.15709)
- [Bunkbed Conjecture Disproved - Quanta](https://www.quantamagazine.org/maths-bunkbed-conjecture-has-been-debunked-20241101/)
- [CGBN - NVIDIA GPU Big Number Library](https://github.com/NVlabs/CGBN)
- [ABC Conjecture - Wikipedia](https://en.wikipedia.org/wiki/Abc_conjecture)
- [GPU Turbulence Simulations - Oak Ridge](https://impact.ornl.gov/en/publications/gpu-enabled-extreme-scale-turbulence-simulations-fourier-pseudo-s/)
- [Diff-FlowFSI GPU CFD Platform](https://www.sciencedirect.com/science/article/abs/pii/S0045782525007273)
- [BSD Computational Approach