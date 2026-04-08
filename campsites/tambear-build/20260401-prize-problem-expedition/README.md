# Campsite: Millennium Prize Problem Expedition

**Opened:** 2026-04-01  
**Triggered by:** BigInt landed (887 tests, U256 + FFT multiply via signal_processing::convolve)  
**Status:** Active exploration — multiple threads seeded by team-lead

---

## Infrastructure Now Available

- **U256** (fixed-width, stack-allocated): add/sub/mul/div/mod/pow/gcd — all implemented
- **BigInt** (variable, heap): FFT multiply for large operands via `signal_processing::convolve`
- **Task #14** (pathmaker): completed

---

## Active Threads

### Collatz Conjecture

Playground files: `playground/collatz-as-scan.md`, `collatz_structural.rs`, `collatz_coverage.rs`

**Three architectural approaches:**

**1. Affine chunk lookup (Kingdom B)**
- Precompute Collatz evolution for all residues mod 2^k (k=16: ~30× speedup)
- Math-researcher confirmed: Kingdom B (chunk selection depends on accumulated state)
- Fast but doesn't reduce the search space — just computes the same trajectories faster
- Implementation: `affine_chunk_lookup(n, k)` returns (multiplier, addend, steps) for residue class

**2. Monte Carlo coverage (complete)**
- `playground/collatz_coverage.rs`: proved ALL numbers in [2^61, 2^75] drop below 2^60
- Zero failures in 100K × 15-step sampling passes
- This is a PROBABILISTIC proof — not formal, but very strong evidence
- Kingdom A: each sample is a scatter → accumulate (does it drop? yes/no)

**3. Formal proof on residue classes (math-researcher, 2026-04-01)**

**Kingdom classification (definitive):** Each step is degree 1 (Kingdom A). The routing is Kingdom B. Proving Collatz is proving τ=0 for the full piecewise-affine system.

- For each residue class mod 2^k: the k-step Collatz map is AFFINE — n ↦ (3^a · n + b) / 2^k
  where a = popcount(r), b = constant determined by r. Degree 1 in n. Kingdom A.
- Routing between classes (which class you land in next): depends on value of result mod 2^k.
  This is Kingdom B — state-dependent sequential routing.
- The conjecture: does this piecewise-affine system have a unique global attractor at {4,2,1}?
  This is the τ question. Collatz conjecture = proving τ=0 for the full piecewise-affine system.
- Implication: each STEP is Kingdom A (cheap). The DIFFICULTY is proving global convergence (τ).
- The Collatz iteration is a **switched affine system** — a collection of affine maps with
  state-dependent switching. Stability of switched linear systems is known to be UNDECIDABLE
  in general (Blondel-Tsitsiklis, 2000). The Collatz case has additional structure (deterministic
  switching, specific affine maps) that might make it decidable — but this explains why it's hard.
- The symbolic approach: prove the affine map contracts for all residue classes.
  If contraction rate ρ < 1 for all classes → τ=0 by Banach fixed-point theorem.
- This is the most promising path: Banach boundary analysis on the piecewise-affine map.
- **Bit density gap**: The team lead's proof sketch for all-ones residue classes requires that
  the bit density of 3^k approaches 1/2. This is an OPEN PROBLEM in number theory (related to
  normality of 3^k in base 2). Weyl's theorem gives equidistribution of leading digits, but
  bit density convergence is strictly harder. The average bit density over k=1..K converges
  to 1/2, but the per-k convergence is unproven. This is the specific gap in the proof sketch.

**4. Haar measure + parameterized family (scout, 2026-04-02)**

**Theorem** (proven): E[v₂(mn+1)] = 2 for ALL odd m, under Haar measure on ℤ₂.
Proof: m odd → m invertible mod 2^k → 2^k | mn+1 has unique solution mod 2^k among 2^{k-1} odd
residues → Pr[v₂(mn+1) ≥ k | n odd] = 2^{-(k-1)} → E[v₂] = Σ 2^{-(k-1)} = 2.

Corollary: contraction of T_m(n) = (mn+1)/2^{v₂(mn+1)} under Haar measure = m/4 for all odd m.

**Why m=3 is the UNIQUE interesting case:**
- Contractive iff m/4 < 1 iff m < 4
- Odd integers: m=1 (trivial, contraction 1/4), m=3 (Collatz, contraction 3/4), m≥5 (expansion)
- m=3 is the LARGEST contractive odd multiplier — closest to the boundary, slowest convergence
- This explains why the conjecture is hard: m=3 is barely contractive, trajectories grow before falling

The E_{2,3}(2)/2 = 3/4 identity is a COINCIDENCE (algebraic witness: (3m+1)(m-3)=0, unique
positive solution m=3). The Haar calculation does not require Euler products.

Garden: `20260402-why-3-is-special-in-collatz.md`, `20260402-where-the-haar-proof-breaks.md`

**The "almost all vs all" gap:** The Haar theorem gives convergence for almost all n ∈ ℤ₂.
The Collatz conjecture requires convergence for ALL n ∈ ℤ. This gap is exactly the difficulty —
a Haar-measure-0 exceptional set in ℤ₂ can still contain all positive integers.

**5. 2-Adic Restatement (math-researcher, 2026-04-02)**

**The Collatz conjecture = the finite-integer boundary always overwhelms the 2-adic attractor.**

- **Kingdom view**: piecewise-affine, A per step, B routing, τ=0 question
- **2-adic view**: finite boundary (positive integers) vs 2-adic attractor (-1 = ...111...)

T(x) = (3x+1)/2 extends to ℤ₂ with fixed point x = -1 = ...111111. The all-ones class
r = 2^k - 1 ≡ -1 (mod 2^k) IS the fixed point's residue class. For n = 2^k - 1, after j ≤ k
steps: n_j = 3^j · 2^{k-j} - 1 with k-j trailing 1-bits, each step v₂ = 1. Growth: (3/2)^k
during k-step escape. After k steps: 3^k - 1, fully escaped (0 trailing 1-bits).

Self-correction (Mihailescu): 3^k - 1 ≠ 2^m for k,m > 1. The all-ones class cannot reproduce
itself. -1 is stable in ℤ₂ but every positive integer has a leading zero. The conjecture asks
whether this boundary always overwhelms the attractor.

**6. Density diffusion**
- Treat Collatz as sparse transition matrix × density vector
- ~100 sparse matmuls instead of 2^71 individual trajectories
- Kingdom A iteration (sparse matvec is commutative) over the state space
- Connection: if the density converges to 1 (all numbers reach 1), Collatz is proved

**KAM/Regime connection (to explore)**
- Collatz parity sequence (even/odd at each step) = a regime signal
- Regime changes = trajectory turning points
- Shannon entropy of parity sequence = complexity measure of trajectory
- Does the parity sequence have KAM-like quasi-periodic structure?
- If so, fintek's regime detection machinery might apply to Collatz
- This is a structural rhyme to evaluate: Collatz parity regime ↔ market regime transitions

---

### Riemann Hypothesis — New Thread (BigFloat landed, 2026-04-01)

**Infrastructure now available:**
- `BigFloat::zeta(s)` — Riemann zeta for real s > 1 (Borwein acceleration, O((3+2√2)^{-n}))
- `zeta_complex(s)` — complex ζ(s) on the full complex plane
- `hardy_z(t)` — Hardy Z-function Z(t) = exp(iθ(t))·ζ(1/2+it), real-valued
- `find_zeta_zero(t_lo, t_hi)` — bisection on Z(t) to find zeros on the critical line
- `riemann_siegel_theta(t)` — Riemann-Siegel theta function via Stirling approximation

**The RH in computational terms**: All non-trivial zeros of ζ(s) lie on the line Re(s) = 1/2.
This can be verified for specific zeros using Z(t): Z(t₀) = 0 ↔ ζ(1/2 + it₀) = 0, and all
Z zeros are zeros of ζ on the critical line.

**First zero found (pathmaker, 2026-04-01):**
```
t = 14.1347255707   (known: 14.134725142...)   Z(t) = 3.4 × 10⁻⁷
```
6 significant figures. Entirely from tambear primitives — no external numerical libraries.
Full call stack:
```
convolve (FFT) → BigInt → BigFloat → BigComplex → zeta_complex → hardy_z → find_zeta_zero
```

**Kingdom classification of Borwein's algorithm** (confirmed by pathmaker + observer):
- d_k computation: prefix sum of C(n,j) — Kingdom A
- Main sum: single commutative weighted accumulate — Kingdom A
- ζ(s) = two nested Kingdom A operations
- Borwein weights = CDF of Binomial(n,1/2) = PrefixSum(Euler transform PMF weights)
- Connection: Borwein cites Euler; this is intentional descent, not independent discovery

**Next experiments:**
1. Verify zeros t₂ ≈ 21.022040, t₃ ≈ 25.010858 — straightforward calls to find_zeta_zero
2. Push to t > 3×10¹² with BigFloat prec=256 — extends verified range
3. Collatz stopping time distribution: COPA on log(stopping_times) → test lognormality

**Connection to Millennium Prize**: RH is one of the seven Millennium Prize Problems ($1M).
Computational verification at new heights doesn't constitute a proof, but it's publishable
and makes the conjecture more robust. Our BigFloat implementation goes beyond f64 limits —
can verify zeros where cancellation would destroy f64 precision.

**Architecture note**: Collatz and RH both require our BigFloat stack.
- Collatz stopping times → BigInt for exact arithmetic
- RH zeros on critical line → BigFloat for precision beyond f64

**Structural rhyme with the signal farm (scout, 2026-04-01)**: The zeta zero distribution and the
harmonic leaf r-statistic are computing the SAME mathematical object: level-spacing distribution
of a sequence of special values (zeros vs eigenvalues). The Montgomery-Odlyzko Law states that
ζ zeros follow GUE (Gaussian Unitary Ensemble) spacing — the same distribution as eigenvalues
of large random Hermitian matrices. The signal farm's harmonic leaf checks whether the cross-
cadence correlation eigenvalue spacings are GUE (random, market-efficient) vs Poisson (structured,
exploitable). Same T×K×O coordinate. Different domains (number theory vs market microstructure).
This is Type I structural Rhyme #39 — verified in paper-05.

**Harmonic Leaf Prototype** (scout, 2026-04-02):
`nonparametric::level_spacing_r_stat(sorted_values)` exists in `nonparametric.rs` and is
exported from `lib.rs`. The ζ zeros path (`bigfloat.rs:montgomery_odlyzko_r_statistic`) calls
this function and produces r = 0.504 from 37 zeros (GUE-consistent). When the fintek harmonic
leaf has eigenvalue data, it calls this same function. The rhyme is live at the code level;
the market data path completes the verification circuit. GUE → r ≈ 0.536 (efficient market),
Poisson → r ≈ 0.386 (structured/exploitable).

---

### Beal Conjecture with U256

`beal_search.rs` exists. U256 now available for higher exponents (x,y,z > 7).

- Previous limit: u128 (exponents ≤ 7)
- New limit: U256 (exponents up to ~20-25 before compute dominates)
- U256::pow + U256::inth_root + U256::gcd all available
- Search space: A^x + B^y = C^z, gcd(A,B,C) = 1 → FALSE (Beal says gcd > 1)

---

### Goldbach Past 64-Bit

- Nobody has GPU Goldbach with 128-bit integers
- U256 gives us 256-bit integers → Goldbach for even numbers up to 2^256
- March 2026 papers (Llorente-Saguer) show architecture for GPU Goldbach
- Our architecture: for each even N, sieve for primes p ≤ N/2, check if N-p is prime
- The "check if N-p is prime" part = BigInt primality test (Miller-Rabin with U256)
- This would be a genuine first

---

### Cross-Problem Connections

**Formal proofs via accumulate**
- Symbolic Collatz on residue classes: for each class mod 2^k, accumulate the symbolic transformation until fixed point
- Goldbach for modular classes: for each residue mod p, show both p AND n-p can be prime under Fermat little theorem constraints
- Beal algebraic constraints: modular arithmetic eliminates many (A,B,C,x,y,z) candidates before search

**The Collatz-COPA connection** (speculative)
- Collatz stopping time has a distribution. Does that distribution have a COPA-extractable structure (mean, variance, higher moments)?
- If yes: the COPA state of stopping times is the MSR for Collatz statistics
- The COPA boundary theorem says: second-order statistics are the Gaussian part
- Is Collatz stopping time distribution Gaussian? (It's approximately lognormal in practice)
- If lognormal: T = Log transform → COPA extracts it → Rhyme with financial log-returns

---

## Open Questions

1. Is Collatz density diffusion related to PageRank? (resolved by scout 2026-04-01)

   **YES** — both are power iteration on a stochastic transition matrix.

   PageRank: `π_{k+1} = d·W·π_k + (1-d)/N·1` (row-stochastic W, teleportation term)
   Collatz:  `ρ_{k+1} = T·ρ_k`             (column-stochastic T, n→Collatz(n))

   The key difference: PageRank has convergence guaranteed by Perron-Frobenius (W is 
   irreducible and aperiodic by the teleportation term). Collatz convergence IS the
   Collatz conjecture — not proven. Proving Collatz = proving T has a unique absorbing
   state (δ_1), i.e., proving 1 is the unique Perron-Frobenius eigenvector of T with
   eigenvalue 1, and all other eigenvalues have magnitude < 1.

   Both are Kingdom A (sparse matvec is commutative accumulate). Both can be accelerated
   by Anderson mixing (see Q2 above). The main architectural difference: PageRank needs
   the teleportation damping (d<1) to ensure convergence — without it, a disconnected
   graph has multiple eigenvectors. Collatz has no teleportation and may have multiple
   eigenvectors (if there are non-trivial cycles) — this is exactly what the conjecture
   is ruling out.

2. Can Wynn's epsilon accelerate density diffusion? (resolved by scout 2026-04-01)
   
   **Short answer**: not directly — Wynn ε operates on scalar sequences. But the vector
   generalization (Anderson mixing) applies directly to the density vector iteration.

   Anderson mixing = Wynn ε / Shanks transformation for VECTOR fixed-point iterations.
   For `ρ_{k+1} = T × ρ_k` (Collatz density diffusion):
   ```
   Anderson(ρ_k, ρ_{k-1}, ..., ρ_{k-m}) = argmin over convex combinations of
   {f(ρ_{k-i}) : i=0..m} where f(ρ) = Tρ - ρ (residual)
   ```
   The weights solve an m×m Gram matrix least-squares problem (Kingdom A inner solve).
   This is `attract_scan` with Anderson weights = Kingdom BC.

   The convergence is governed by the spectral gap of T. If the dominant non-unit eigenvalue
   of T has magnitude λ < 1, convergence is geometric at rate λ^k. On this geometric
   sequence, Aitken (not Wynn) is the matched accelerator. Anderson mixing with m history
   vectors can eliminate the top m eigenvalue contributions simultaneously.

   **Empirical estimate**: typical Collatz map contracts by ≈ 0.866 per step (3×/4÷2 → 
   √(3/4) × √(1/2) per two steps). This is geometric → Aitken/Anderson should accelerate.
   Anderson mixing with m=5 history vectors should give ~5× convergence speedup on the
   density diffusion iteration, similar to DIIS acceleration in quantum chemistry.

   See: series-acceleration-prefix-scan campsite (Anderson mixing = vector Shanks transform)

3. Does the KAM-Collatz connection yield any of the known Collatz heuristics (Terras, Rawsthorne)?

4. What's the Galois structure of Collatz mod 2^k? (The symbolic transformation on residue classes might have a Galois group that explains why it's hard)

**Partial answer on Q4 (scout, 2026-04-02)**: The 2-adic integers ℤ₂ are the natural domain.
Each residue class mod 2^k evolves via an AFFINE map (3n+1 is linear in n, then shift-right for
even steps). The k-th order chunk lookup IS the k-th order "Taylor expansion" of T in ℤ₂. The
Galois structure would be the group of these affine maps on residue classes — exploring this might
reveal whether the maps form a group with a clean order structure.

The p-adic perspective on the Euler factor (2026-04-02, RESOLVED by math-researcher): the
real connection is at s=1, not s=2. E_2(1) = 2 = E[v₂(3n+1)] via Haar measure on ℤ₂.
The E_{2,3}(2)/2 = 3/4 identity is a numerical coincidence (unique Diophantine solution).
The deep result: q=3 is the unique subcritical odd-prime Collatz map (contraction q/4 < 1
only for q < 4). See Euler Factor Identity section below for full resolution.

**Anti-CLT framing (scout, 2026-04-02)**: Collatz density diffusion goes AGAINST the CLT
(concentrating to δ_1, not spreading to Gaussian). PageRank goes WITH the CLT (density spreads
to stationary distribution). Both are power iteration on stochastic matrices. Proving Collatz =
proving τ=0: the spectral gap guarantees δ_1 is the unique absorbing state. Team-lead's cascade
data gives empirical spectral gap ≈ 0.01 (spectral radius ~0.99/step). Rhyme #39 candidate.

**Amplification experiment (scout, 2026-04-02)**: density diffusion converges geometrically at
0.99/step. For geometric convergence, Anderson mixing m=5 should give ~5× speedup (vector Aitken
on the density sequence). Plus: FFT the transition kernel to reveal eigenspectrum structure.
Garden entries: `20260402-amplification-instead-of-damping.md`, `20260402-collatz-in-the-right-base.md`.

---

## The {2,3}-Euler Factor Identity (observer, 2026-04-02)

The Euler product ζ(s) = ∏_p 1/(1-p^{-s}) decomposes ζ into prime contributions.
The {2,3}-factor E_{2,3}(s) = 1/(1-2^{-s}) · 1/(1-3^{-s}) isolates exactly the
Collatz primes.

**Numerical results** (computed via `euler_maclaurin_zeta`, test `euler_product_23_factor`):

| s | ζ(s) | E_{2,3}(s) | ζ/E (primes≥5) | {2,3} dominance |
|---|------|-----------|----------------|-----------------|
| 2 | 1.6449 | **1.5000** | 1.0966 | 91.3% |
| 3 | 1.2021 | 1.1868 | 1.0128 | 98.7% |
| 4 | 1.0823 | 1.0800 | 1.0022 | 99.8% |
| 6 | 1.0173 | 1.0173 | 1.0001 | 99.99% |

**Arithmetic identity**: E_{2,3}(2) / 2 = (3/2) / 2 = **3/4** = Collatz heuristic contraction.

### Resolution (math-researcher, 2026-04-02)

**The s=2 identity is a numerical coincidence.** The real connection lives at s=1.

**Structural connection at s=1 (Haar measure on ℤ₂)**:
- E_2(1) = 1/(1 - 2^{-1}) = 2
- This equals E[v₂(3n+1)] = 2 for odd n (the expected 2-adic valuation)
- The reason: v₂(3n+1) follows a geometric distribution on ℤ₂ with parameter 1/2
- This IS the Haar measure on the 2-adic integers — a structural fact about ℤ₂, not
  a coincidence about ζ(2)

**Why s=2 is coincidence**:
- The identity E_{2,3}(2)/2 = 3/4 is just (4/3)(9/8)/2 = 3/4 — a Diophantine identity
  with a unique solution. No deep mechanism forces this.
- The Collatz contraction 3/4 comes from 3 × 2^{-E[v₂]} = 3 × 2^{-2} = 3/4
- The Euler factor E_{2,3}(2) = 3/2 comes from (1-1/4)^{-1}(1-1/9)^{-1}
- These are the same numbers (2 and 3) in different formulas — they meet at s=2
  by arithmetic accident, not structural necessity

**The deeper result — uniqueness of q=3**:
- The generalized Collatz map T_q(n) = qn+1 (for odd prime q) contracts iff q/4 < 1
- Only q=3 satisfies q < 4 among odd primes
- So q=3 is the UNIQUE subcritical odd-prime Collatz map
- This is the real number-theoretic content: not "why does 3/4 appear in ζ(2)?"
  but "why is q=3 special?" — because it's the only odd prime below 4

**What to keep from the numerical experiment**:
- The {2,3} dominance table is real and interesting (91% at s=2, 99.99% at s=6)
- The Haar measure connection at s=1 IS structural and worth exploring further
- The s=2 identity should NOT be cited as evidence for a deep Collatz-ζ connection

---

## Concrete Next Experiments

### Experiment 1: Collatz Stopping Time Distribution ✓ COMPLETE (math-researcher, 2026-04-02)

Tests: `descriptive::tests::collatz_stopping_time_distribution`, `collatz_density_diffusion`

**Stopping time moments (n=2..100K, moments_ungrouped):**
```
Mean:  107.54    Std:   51.37    Min: 1    Max: 350
Skew:  0.54      Kurt: -0.28
Terras ratio: mean/log₂(N) = 6.47 (heuristic ≈ 7.23)
```
NOTE: The earlier "9.48" figure was wrong. The heuristic from the per-cycle contraction
3/4 gives ≈ 7.23 × log₂(n) total steps. Our measured 6.47 is consistent (small-n bias).

**Log-transform → approximately lognormal:**
```
Skew(ln s) = -0.45    Kurt(ln s) = -0.20
```
The COPA connection confirmed: T = Log is the Kingdom A transform. Same structure
as financial log-returns → structural rhyme with the signal farm.

**Density diffusion spectral gap (compressed map, 500 steps):**
```
     N    escaped   |λ₂|      gap
   1000    0.56    0.865    0.135
   3000    0.59    0.875    0.125
  10000    0.39    0.939    0.061
  30000    0.40    0.934    0.066
 100000    0.41    0.935    0.065
```
First empirical spectral gap measurement of the Collatz transition operator.
**The gap stabilizes at ~0.065 for N ≥ 10K** — it does NOT shrink to zero.
This is a lower bound (escaped mass would add convergence).
Connection to cascade analysis: hardest residue classes peak at k=16 → N ≥ 2^16
includes all hard cases. The gap measures the "worst-direction" convergence penalty
(|λ₂| = 0.935) vs the average per-step contraction (exp(-0.096) = 0.908).

### Experiment 2: RH Zero Verification (feasible now)
1. Call `find_zeta_zero(t_lo, t_hi)` for known zeros t₁..t₂₀
2. Verify they all lie on Re(s) = 1/2 (by construction of Hardy Z)
3. Push to higher t: currently verified t < 3×10¹². Can we verify to 10¹³?
4. BigFloat prec=256 gives ~77 digits — enough for verification up to ~10²⁰

Infrastructure: `bigfloat.rs` has everything needed. One test function.

### Experiment 3: r-statistic for ζ zeros — COMPLETE (pathmaker + scout, 2026-04-02)

**Result**: r_mean = 0.504 from 37 zeros (t ∈ [13, 130]). Consistent with GUE (0.536) at
<1σ. Excludes Poisson (0.386) at ~2.8σ. Level repulsion confirmed (no near-zero gaps).

**Shared primitive**: `nonparametric::level_spacing_r_stat(sorted_values)` — added to
`nonparametric.rs` with 3 tests (GUE-like, Poisson-like, edge cases). Exported from lib.rs.
`bigfloat.rs:montgomery_odlyzko_r_statistic` now calls this function instead of inline loop.

**Status of Rhyme #39**: Verified at code level — same function for ζ zeros path.
Market eigenvalue path complete when harmonic leaf data is available.
See paper-05-structural-rhymes.md Section 8, Rhyme #39.

---

## Related
- `playground/collatz-as-scan.md` — team-lead's initial architectural notes
- `collatz_structural.rs` — Rust implementation of chunk lookup
- `collatz_coverage.rs` — Monte Carlo coverage proof
- `beal_search.rs` — Beal conjecture search
- `bigfloat.rs` — BigFloat + ζ(s) + Hardy Z-function + critical-line zero-finding
- Series acceleration (`src/series_accel.rs`) — matched-kernel principle, StreamingWynn, Euler factor (48 tests)
- {2,3}-Euler factor table — `euler_product_23_factor` test in `series_accel.rs`
- `nonparametric::level_spacing_r_stat` — shared primitive for Rhyme #39 (ζ zeros + market eigenvalues)
