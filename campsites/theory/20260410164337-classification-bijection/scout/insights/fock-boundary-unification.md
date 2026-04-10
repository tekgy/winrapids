# The Fock Boundary = Boundary of the Product Closure

*Scout, 2026-04-10*

## The unification claim

Today's theory work converged on a single statement:

**The Fock boundary IS the boundary of the product closure of the 5 atoms.**

If this holds, then Kingdom A/B/C/D classification and the sharing cluster structure
are NOT two independently derived systems that happen to correlate. They are the SAME
structure, described in two different languages:

- Kingdom classification describes it in terms of COMPUTATION (what can be parallelized)
- Sharing clusters describe it in terms of INTERMEDIATES (what can be reused)

The bijection claim (algorithm classes = sharing clusters) implies this unification.

## Why the Fock boundary connects to sharing

A computation is Kingdom A (inside the product closure) when its state-transition map
has no self-referential dependence — equivalently, when you can write `combine(a, b)`
without running the interior. This means:

**Kingdom A computations can share intermediates across arbitrary segment boundaries.**

If you can compute combine(segment_A, segment_B) algebraically, then any intermediate
computed at the boundary of segment_A is sharable with segment_B without re-running
segment_A's interior. The sharing graph is determined by the combine() structure.

**Kingdom B/C/D computations cannot.** The intermediate at the boundary of segment_A
requires running the interior to obtain. No shortcut means no sharing across that
boundary.

This is the mechanism connecting computation class to sharing cluster:
- Kingdom A → combine() exists → intermediates sharable across segment boundaries →
  dense sharing cluster
- Kingdom B → combine() requires interior → intermediates only sharable within segments →
  sparse or singleton sharing cluster
- Kingdom C → outer loop over Kingdom A inner → intermediates sharable within each
  iteration, not across → nested cluster structure
- Kingdom D → no exact combine() → approximation intermediates only → separate
  approximate cluster

## The three-criteria test as sharing predictor

Aristotle's three-criteria test (affine augmentation → associative composition →
structural independence) is also a SHARING PREDICTOR:

- Pass Test 1 (affine): sharing via matrix intermediate (gram matrix, covariance,
  companion matrix products)
- Pass Test 2 (associative): sharing via sufficient-statistic intermediate (Welford
  accumulators, LogSumExp state)
- Fail Test 3 (structural independence): NO sharing intermediate exists across
  segment boundaries

The test predicts both the Kingdom AND the sharing structure from the same analysis.

## What needs to be verified

The bijection claim requires checking both directions:

1. **Computation class → sharing cluster**: algorithms in the same Kingdom should
   cluster together in the TamSession sharing graph. Kingdom A algorithms should
   form dense sharing clusters (many shared intermediates). Kingdom B should be
   isolated or form sparse clusters.

2. **Sharing cluster → computation class**: algorithms that cluster together in the
   sharing graph should be in the same Kingdom. If two algorithms share an intermediate,
   they should both be Kingdom A (because only Kingdom A allows cross-boundary sharing).

The spring simulation that the naturalist ran verifies direction 1 topologically.
The algebraic verification (Aristotle) verifies direction 2 for the 5-atom case.
The formal writeup needs to state both directions as separate theorems.

## The test case: GARCH reclassification

GARCH was reclassified today from Kingdom B to Kingdom A (affine prefix scan via
3x3 companion matrix). If the bijection holds, GARCH's sharing cluster should also
change: it should now share intermediates with EMA, EWMA, AR(p), and Kalman filter
(all Kingdom A affine recurrences) that it previously didn't cluster with.

This is a TESTABLE prediction. Run the TamSession sharing graph on a dataset that
exercises GARCH, EMA, EWMA, AR, and Kalman. Do they share intermediates? Do they
cluster together? If yes — the bijection holds for this case and the reclassification
is structurally validated.

## The paper structure this implies

If the bijection holds:

Paper 1: "Algorithm classes are sharing clusters: a unification of computational
complexity and intermediate reuse in numerical computation."

Claims:
1. The 5-atom product closure = the set of Kingdom A algorithms (proven algebraically)
2. Kingdom A algorithms form dense sharing clusters in TamSession (proven topologically)
3. Kingdom B/C/D algorithms cluster separately with the predicted sparse structure
4. Corollary: Kingdom classification can be READ from the sharing graph without
   analyzing the algorithm — the sharing structure IS the classification

Claim 4 is the operational payoff: TAM can determine an algorithm's Kingdom by
observing its sharing behavior during execution, not by static analysis.
