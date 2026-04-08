# Challenge 29 — Kalman Filter: The General Matrix Prefix Scan

**Date**: 2026-04-06  
**Type C: Foundation Challenge — the general form subsumes all special cases**

---

## The Traditional Assumption

The Kalman filter is a specialized algorithm for linear Gaussian state-space models — a distinct addition to the signal processing section.

## Why It's More Than That

The Kalman filter is the GENERAL FORM of the matrix prefix scan (challenge 13). GARCH, AR, EWMA are all special cases.

---

## The Sarkka Connection

The garden entry `~/.claude/garden/006-the-correction-term.md` (March 30, 2026) documents that a parallel Kalman filter was implemented for the WinRapids pipeline using the Sarkka 5-tuple representation:

The Sarkka 5-tuple: `(A, b, C, η, J)` where:
- Forward part `(A, b, C)`: dynamics — how x_{t-1} maps to x_t
- Backward part `(η, J)`: observation — what measurements say about x_{t-1}

The associative combine operation on these 5-tuples IS a matrix prefix scan. The key correction term `-J_b · b_a` (the curvature of backward information through non-zero-offset segments) was found to be required — it's not in the paper's theorem statement but is required for the Blelloch tree to work correctly.

**This hard work was already done.** The parallel Kalman filter algorithm is solved. It's just not in tambear as a math primitive.

---

## Why This Matters

The Kalman filter IS the general linear Gaussian inference algorithm. Every linear time series model is a special case:

| Model | State | Transition | Observation | Notes |
|---|---|---|---|---|
| AR(p) | [x_{t}, ..., x_{t-p+1}] | companion matrix | [1, 0, ..., 0] | Observation = first component |
| ARMA(p,q) | extended state | companion + MA | standard | Slightly larger state |
| GARCH(1,1) | [σ²_t, r²_t] | 2×2 matrix | [1, 0] | Observation = variance |
| Simple EWMA | [S_t] | [α] | [1] | Direct |
| Holt's linear | [L_t, T_t] | 2×2 | [1, 0] | Level + trend |
| Kalman (general) | any x_t | A_t matrix | H_t matrix | Full generality |

**One implementation of Sarkka prefix scan covers all of these.**

---

## The Challenge Structure

The traditional assumption: Kalman filter is a "missing module" that needs to be written. 

The tekgy reality: the mathematical infrastructure (matrix prefix scan) is the SAME as challenge 13 (GARCH). The Kalman filter is just challenge 13 expressed at full generality.

Order of implementation:
1. Build the matrix prefix scan primitive (challenge 13 — GARCH motivation)
2. The Sarkka 5-tuple representation adapts this to linear Gaussian models
3. Kalman filter = special case where matrices may vary over time

The correction term from the garden entry: `-J_b · b_a` handles the case where the transition has a nonzero offset (b ≠ 0). This is the "inhomogeneous" linear recurrence case. For homogeneous recurrences (GARCH, EWMA), b=0 and the correction vanishes.

---

## Impact

Adding Kalman filter to tambear unlocks:
- Linear Gaussian filtering and smoothing
- Rauch-Tung-Striebel smoother (reverse scan)
- Extended Kalman filter (local linearization)
- Local level model, structural time series, dynamic regression
- State-space representation of ARIMA models

And for the Collatz research: the fold_irreversibility.rs temperature analysis is computing statistics of a DYNAMICAL SYSTEM. If the temperature process can be modeled as a (possibly nonlinear) state-space model, Kalman filtering gives the posterior distribution over temperature — which would give probability estimates for convergence.

---

## Most Actionable

Before implementing full Kalman filter: read the garden entry `006-the-correction-term.md` and extract the Sarkka 5-tuple combine operation. It's already correct. The work is bringing it into tambear as a general primitive.

The Blelloch scan is already implemented in the pipeline. Port the mathematical Sarkka combine to `tambear/src/kalman.rs`.

---

## Risk: RTS Smoother Sign Conventions (flagged by scout)

The garden entry covers the FORWARD Blelloch scan (filter direction). The RTS (Rauch-Tung-Striebel) smoother runs BACKWARD — same semiring, opposite traversal direction. The sign of the J (precision) term flips in the backward direction.

The garden entry's key finding was that the correction term `-J_b · b_a` only becomes visible at step 3 of the Blelloch tree (the first position computed as `combine(e01, e23)` rather than sequential). The RTS backward has the same structural issue at a different position.

**Pathmaker warning**: verify the RTS smoother against a known linear Gaussian reference BEFORE trusting it. A 2-state AR(1) model has analytical smoothed posteriors that can serve as a gold standard. Test at minimum 8 timesteps (long enough to reach the "step 3" Blelloch position where the correction first matters).

The forward filter at n=3 is where the original pathmaker found the bug. The backward smoother at n=3 (counting from the end) is the analogous high-risk position.
