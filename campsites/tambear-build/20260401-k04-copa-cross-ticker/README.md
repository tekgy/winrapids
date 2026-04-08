# Campsite: K04 Cross-Ticker Analysis Is Entirely Super-Fock

**Opened:** 2026-04-01  
**Thread:** Math-researcher following "fractal distance via MSR" parking-lot thread  
**Status:** Active — architectural consequence for WinRapids K04 design  
**Garden:** `~/.claude/garden/2026-04-01-fractal-distance-via-msr.md` (math-researcher's notes)

---

## The Core Observation

The Bures-Wasserstein distance between two Gaussian distributions is:

```
W₂²(N(μₐ,Σₐ), N(μ_b,Σ_b)) = ‖μₐ − μ_b‖² + tr(Σₐ + Σ_b - 2(Σₐ^{1/2} Σ_b Σₐ^{1/2})^{1/2})
```

This requires ONLY the COPA states from each distribution:
- `μₐ, Σₐ = n⁻¹ Cₐ` from ticker a's COPA state `(n, μₐ, Cₐ)`
- `μ_b, Σ_b = n⁻¹ C_b` from ticker b's COPA state `(n, μ_b, C_b)`

No raw data. No re-reading tick streams. The full cross-ticker distributional geometry is
contained in the (d+1)×(d+1) COPA states accumulated during K01/K02 processing.

---

## Architectural Consequence

K01/K02 farm raw ticks → produce COPA states per ticker-cadence pair.  
K04 extracts cross-ticker distributional distances from those COPA states.  

**K04 is entirely super-Fock.** The Fock boundary is at K01. Everything in K04 operates on already-accumulated MSRs. Zero additional data passes.

For N tickers with feature dimension d:
- Each COPA state: `(d+1)×(d+1)/2 + d + 1` scalars ≈ O(d²) space
- Pairwise Bures-Wasserstein: O(d³) per pair (eigendecompose Σₐ^{1/2} Σ_b Σₐ^{1/2})
- Full N-ticker distance matrix: O(N² · d³) — no data passes, only MSR operations

For N = 1000 tickers, d = 50 features:
- Data passes needed: 0
- Compute: 10⁶ × 50³ = 1.25 × 10¹¹ FLOPS — but at O(d³) per pair, highly parallelizable
- This is a pure Galois-obstructed Kingdom A extraction: unique answer, parallel by ticker pair

---

## Progressive Hierarchy for Distance Computation

The Bures-Wasserstein distance is O(d³). But most ticker pairs are far apart and don't need it.
Math-researcher identified a progressive hierarchy that respects the MSR structure:

```
Level 0: ‖μₐ − μ_b‖                    O(d)      lower bound on W₂
Level 1: W₂(diagonal Σ only)            O(d)      diagonal Bures approx
Level 2: W₂(Gaussian, full Σ)           O(d³)     exact for Gaussian
Level ∞: Optimal transport (raw data)   O(n³)     exact for non-Gaussian
```

**Distance pushdown**: evaluate Level 0 for all N² pairs first. Only escalate to Level 1/2
for pairs within a threshold. If most tickers are well-separated (typical in practice), the
majority of pairs are resolved at O(d) cost.

This is the same principle as the MSR progressive approximation hierarchy from Algorithm 1
in paper-02: use the cheapest representation that resolves the query. The decision rule is:
"is the Level-k approximation tight enough for this pair?"

---

## What This Enables for WinRapids K04

**Current WinRapids**: K01 (tick bars), K02 (cadence bins), K03 (cross-cadence). K04 is 
the cross-ticker kingdom — the "spatial" dimension of the kingdom ladder.

**With COPA states as K04 inputs**:

1. **Distributional similarity matrix**: full N×N Bures-Wasserstein matrix. Identify ticker
   "clusters" — groups of tickers whose tick distributions are geometrically proximate.

2. **Regime detection**: COPA states change over time. Track how the distributional distance
   between ticker clusters evolves. A regime shift = sudden change in the K04 distance matrix.

3. **Arbitrage geometry**: tickers that are close in K04 distribution space but diverge in
   K01 price space are potential mean-reversion candidates. The "fractal distance" is the
   inverse of mean-reversion strength.

4. **Portfolio construction**: use the K04 distance matrix as a kernel for graph-based
   portfolio optimization. Similar to Kriging over the ticker space.

---

## The Algebraic Constancy of the Kingdom Ladder

Math-researcher's analysis (2026-04-01) corrected an initial framing:

**The rank does NOT double at each kingdom level. The rank stays at 2 throughout.**

K03 and K04 do NOT produce rank-4 tensors. They produce rank-2 cross-covariance matrices:
```
K03: Σ_{c1,c2} = cross_cov(ticker_a @ cadence1, ticker_a @ cadence2)  — d×d matrix
K04: Σ_{a,b}   = cross_cov(ticker_a @ cadence1, ticker_b @ cadence1)  — d_a×d_b matrix
```

The COPA boundary theorem (rank-2 MSR = maximum of Kingdom A) applies at EVERY kingdom level.
The Galois threshold remains d=5 at all levels — extractions flatten to matrix eigenproblems,
not rank-4 tensor decompositions (which would be NP-hard in general).

**The kingdom ladder is a semantic ladder built on an algebraic constant:**
- K0x number: which physical axes are being contracted (temporal, spatial, ticker, cadence)
- Algebra: always rank-2 COPA accumulate → rank-2 MSR → eigenextract
- COPA infrastructure: reusable at every kingdom level without modification

K03 and K04 are algebraically IDENTICAL operations. The distinction is physical:
K03 contrasts over cadence axis (temporal resolution structure).
K04 contracts over ticker axis (spatial market coupling structure).
Same computation, different meaning for the row/column indices.

**Important: cross-covariance vs marginal operations**

- Bures-Wasserstein between marginal distributions: **super-Fock** (COPA states only)
- Cross-covariance Σ_{a,b} = Cov(X_a, X_b): **requires new accumulation** (joint observations)
  `Σᵢ (x_{a,i} - μ_a)(x_{b,i} - μ_b)ᵀ` needs to observe ticker a and ticker b together

The K04 progressive hierarchy (Level 0 = mean distance, Level 1 = Bures) applies to the
marginal operations. Cross-covariance-based operations (CCA, joint PCA) require the joint
accumulation pass.

---

## Connection to COPA Boundary Theorem

The K04 distance computation (Bures-Wasserstein between COPA states) is:
- Accumulation: always Kingdom A (COPA merge is associative + commutative)
- Extraction: Kingdom A + Galois obstruction (eigendecompose Σₐ^{1/2} Σ_b Σₐ^{1/2})
- Cost: O(d³) per pair — super-Fock, MSR-touching, not data-touching

This is exactly the COPA boundary theorem operating at K04 scale. The COPA boundary
guarantees that cross-ticker distributional geometry can be computed with:
- ONE data pass (per ticker, for K02 accumulation)
- ZERO additional data passes for K04
- Full pairwise geometry from (d+1)² scalars per ticker

---

## Kingdom Classification of K04 Extractions

A clarification worth recording: K04 doesn't introduce new kingdom classifications.
The COPA boundary theorem already covers K04 — same three-layer taxonomy, larger matrices.

| K04 extraction | Problem classification | Algorithm (ρ,σ,τ) | Notes |
|---------------|----------------------|-------------------|-------|
| Cross-ticker correlation | Kingdom A (degree-2) | (0,0,0) | Direct dot product of COPA vectors |
| Mean distance ‖μₐ − μ_b‖ | Kingdom A (degree-2) | (0,0,0) | Closed-form |
| Bures-Wasserstein W₂ | Kingdom A (two degree-2 eigenproblems) | (0,1,0) | Galois-forced iteration, unique |
| Cross-ticker mutual info | Kingdom C (degree ≥ 3) | (0,1,?) | Nonlinear in covariance |

The kingdom classification depends on WHAT you extract, not WHICH level of the hierarchy.
K02 and K04 use the same rules. The K0x number classifies the physical data axis (temporal
resolution or spatial ticker); it's orthogonal to the algebraic kingdom.

## Open Questions

1. **Non-Gaussian tickers**: The Bures-Wasserstein is exact only for Gaussians. Real tick
   distributions are heavy-tailed. Does Level 2 (Bures) still provide useful correlation
   structure even when distributions aren't Gaussian? (Conjecture: yes, as a 2nd-order
   approximation whose error is bounded by the 4th cumulant.)

2. **Streaming updates**: Can COPA states be updated as new ticks arrive and K04 distances
   be updated incrementally? (Yes: COPA merge is streaming-friendly. Distance update from
   incremental COPA merge is O(d³) per new tick-batch.)

3. **Symmetry of distance matrix**: Is W₂(a,b) = W₂(b,a)? Yes — Wasserstein is symmetric.
   So we only need to compute N(N-1)/2 pairs, not N².

4. **K04 as K03 over the ticker space**: K03 is "cross-cadence accumulate on one ticker."
   K04 might be "K03 over the cross-ticker graph." Does the associativity of COPA extend
   to graphs — can we compute K04 via spanning tree of COPA merges?

---

## Related
- COPA boundary theorem: `docs/publications/paper-02-formal-proofs.md` (Theorem 7)
- Progressive approximation hierarchy: same file, Algorithm 1
- Math-researcher garden: `~/.claude/garden/2026-04-01-fractal-distance-via-msr.md`
- WinRapids K04 architecture: `R:/winrapids/CLAUDE.md` (kingdom ladder)
