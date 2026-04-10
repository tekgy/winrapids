# Holographic Sharing: First Principles Analysis

*2026-04-10 — Aristotle*

## The Question

Does TamSession's sharing infrastructure have holographic structure? In AdS/CFT, the boundary (lower-dimensional) theory encodes the bulk (higher-dimensional) theory. Is there an analogous structure in the sharing graph?

## The Proposed Mapping

| AdS/CFT | TamSession |
|---------|------------|
| Boundary | TBS surface (what the user sees: method calls, parameters) |
| Bulk | Computation graph (accumulate calls, intermediate dependencies) |
| Dictionary | IntermediateTag (content-addressed identity, compatibility predicate) |
| Entanglement entropy | Sharing degree (how many consumers share one intermediate) |
| Error correction | Redundant computation paths (if one sharing path fails, can we recover from another?) |

## Why This Might Be More Than Metaphor

The holographic principle says: the maximum information content of a volume scales with its BOUNDARY area, not its volume. In TamSession terms: the information content of the computation graph should be bounded by the SURFACE of user-visible method calls, not by the internal complexity.

This is actually true in a precise sense. The user calls N methods (boundary). Internally, these share M intermediates. The computation cost scales with M (the deduplicated intermediate count), not with N * (average intermediates per method). The sharing graph compresses the bulk computation proportionally to how much boundary information overlaps.

**The ratio bulk/boundary = the sharing factor.** If 15 methods share the same FFT, the sharing factor for FFT is 15. The total computation is proportional to the number of DISTINCT intermediates (boundary), not the total intermediate demands (bulk).

## The Error Correction Angle

In holographic codes, you can lose part of the boundary and still reconstruct the bulk. In TamSession:

- If one intermediate is corrupted (hardware error, NaN propagation), can downstream methods detect this?
- If two methods compute the same quantity independently and get different answers, the disagreement IS an error signal.

The `.discover()` superposition layer already does something like this: running multiple methods and comparing results. Agreement across methods = confidence. Disagreement = error or interesting structure.

**This is literally holographic error correction**: redundant boundary representations (multiple methods computing overlapping intermediates) enable consistency checking that detects errors in the bulk (corrupted intermediates).

## What Would Make This Rigorous

The analogy becomes a theorem if we can show:

1. **Subregion duality**: Any subset of TBS method calls determines a unique subgraph of the computation graph (the "entanglement wedge" of those methods). This is TRUE — given a set of method calls, the compiler determines exactly which intermediates are needed.

2. **Monotonicity**: Adding more method calls to the boundary can only INCREASE the accessible bulk. This is TRUE — more methods means more intermediates are computed, which means more of the computation graph is materialized.

3. **Error correction threshold**: There exists a threshold fraction of boundary methods that can be removed while still recovering all intermediates. This is TRUE IF multiple methods produce the same intermediate (the sharing degeneracy). If 3 out of 15 methods that need FFT are removed, the FFT is still computed by the remaining 12.

## The Interesting Prediction

If the analogy is deep, there should be a **Ryu-Takayanagi formula** analogue: the sharing entropy of a boundary region equals the minimal cut through the computation graph that separates that region from its complement.

In TamSession terms: the "entanglement" between two sets of methods = the number of intermediates they share. The minimal cut through the sharing graph that separates them = the bottleneck of their computational overlap.

This predicts: methods that share many intermediates are computationally "entangled" — you can't optimize one without affecting the other. Methods that share few intermediates are "separable" — they can be optimized independently.

## What I Don't Know Yet

- Does the sharing graph have the right topology to support a holographic code? (Need the spring sim to visualize.)
- Is the error correction threshold useful in practice? (Need real sharing statistics.)
- Does the Ryu-Takayanagi analogue give actionable scheduling insights? (Need to formalize.)

This needs data. The spring network simulation from the companion campsite would give the topology to test these predictions against.

## Status

Theoretical first pass. The mapping is suggestive but needs the computation graph data to go further. The error-correction prediction is the most immediately testable: run .discover() with intentionally corrupted intermediates and see if the redundancy catches it.
