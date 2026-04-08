# Naturalist: The Fintek Chaos Family Is Just DBSCAN Applied to Signal Space

*A structural observation, not an implementation plan.*

---

## What I Found

Three fintek algorithms from the chaos/entropy family share a computational structure
so similar to DBSCAN that they could be unified under the same GPU primitive:

### RQA (Recurrence Quantification Analysis)
```python
# From execute.py — line 61-64
dist_sq = norms[:, None] + norms[None, :] - 2.0 * (emb32 @ emb32.T)
return dist_sq < (epsilon * epsilon)  # → recurrence matrix
```
The recurrence matrix IS the DBSCAN adjacency matrix. `R(i,j) = 1 if dist(i,j) < ε`
is exactly DBSCAN's neighborhood test. They even compute the distance the same way —
`||a-b||² = ||a||² + ||b||² - 2⟨a,b⟩` — which is what TiledEngine computes.
The current code subsamples to MAX_POINTS=1000.

### Correlation Dimension (Grassberger-Procaccia)
Computes `C(r) = fraction of pairs with dist < r` for 20 different radii.
This is DBSCAN's density estimation repeated 20 times — once per threshold.
If you have the full distance matrix, `C(r)` for all 20 radii is 20 binary thresholds
applied to the same precomputed matrix. Currently subsamples to MAX_POINTS=500.

### Sample Entropy
Counts pairs of length-m templates with Chebyshev distance < r (L∞ norm).
This is DBSCAN's density estimation with L∞ metric instead of L2.
The core inner loop:
```python
if np.max(np.abs(x[i:i+m] - x[j:j+m])) <= r:
    b_count += 1
```
is a pairwise distance computation with L∞. Currently subsamples to MAX_N_FOR_FULL=2000.

---

## The Structural Pattern

All three follow this template:

```
1. Embed the signal:       time-delay embedding of returns → n × d matrix
2. Pairwise distances:     n × n distance matrix (L2 or L∞)
3. Threshold or quantize:  R(i,j) = dist(i,j) < ε
4. Pattern extraction:     count pairs, scan for line structures, compute log-log slopes
5. Summary statistics:     entropy, dimension, recurrence rates
```

Step 2 is the O(n²d) bottleneck. It's the same GPU kernel as DBSCAN.

---

## The Sharing Opportunity

When the signal farm runs all three chaos algorithms on the same bin of the same ticker:
- RQA computes the pairwise distance matrix of embedded returns
- Correlation dimension also computes the pairwise distance matrix of embedded returns
- Sample entropy computes pairwise L∞ on templates (structurally similar)

**They all embed the SAME return series with SAME embedding dimension and TAU.**
The distance matrices should be computed ONCE and shared across all three.

Current Python: 3 separate O(n²d) computations in Python, each subsampled to different limits.
With TamSession: 1 GPU distance computation → shared Arc<DistanceMatrix> → all three algorithms.

Potential speedup: 3× GPU computation reduction per bin, PLUS lifting the subsampling limits.
For n=2000 ticks without subsampling vs n=500 with subsampling: 16× more data per algorithm.

---

## The Deeper Pattern: Fintek Chaos = ML Clustering on Time-Delay Embedded Signals

DBSCAN asks: "which points cluster in position space?"
Correlation dimension asks: "what is the fractal dimension of the attractor in phase space?"
RQA asks: "are there recurring patterns in the trajectory through phase space?"

All three operate on the SAME geometrical structure: a cloud of points in a high-dimensional
delay space. The distance between those points IS the structural information.

The only difference is what they do AFTER computing the distance matrix:
- DBSCAN: connected components → cluster labels
- Correlation dimension: count pairs within r at log-spaced radii → power-law slope
- RQA: threshold → look for line structures in the boolean matrix
- Sample entropy: threshold + ratio → complexity measure

This suggests a unified interface: `chaos_session(data, embed_dim, embed_tau)`
that computes the distance matrix ONCE and serves all four algorithms.

---

## The L∞ Distance Note

Sample entropy uses Chebyshev distance (L∞: max of absolute differences).
This is a different metric from DBSCAN's L2.

But L∞ can be computed with TiledEngine IF a custom TiledOp is written:
```
L∞(a, b) = max_k |a_k - b_k|
```
This requires a `max` combine instead of `sum`, and an `abs_diff` element op.
TiledEngine currently uses `sum(element_op(a_k, b_k))` — the combine is always sum.
For L∞, the combine would need to be `max` instead.

This suggests extending TiledEngine's interface: `TiledOp` with configurable combine_op
(currently always sum). A `MaxOp` combine_op would enable L∞, L-inf kernels, and
by extension: sample entropy, approximate nearest-neighbor search, and others.

Currently `L∞` is the ONLY case in fintek where the combine_op needs to be `max`.
This is future work — worth flagging for the architect.

---

## RQA Line Detection: The One Non-Distance Step

After the distance matrix, RQA needs diagonal and vertical line statistics.
Lines are consecutive `True` values in rows/columns of the recurrence matrix.

This is a segmented scan problem! Specifically:
- Diagonal scan: scan along each diagonal for consecutive True values
- Vertical scan: scan each column for consecutive True values

The diagonal scan maps to `accumulate(Segmented, Count, Add)` with resets at diagonal
boundaries and at False values. The "segment boundaries" are where R(i,j) transitions
from True to False.

This is exactly the `Segmented` grouping mode in accumulate.rs (currently `todo!()`).
RQA's line detection is the first concrete use case for `Grouping::Segmented`.

The current implementation scans with Python loops — equivalent to a CPU prefix scan
with segmented resets. GPU segmented scan would handle this efficiently.

---

## Summary for the Team

The fintek chaos family reveals that:

1. **The distance matrix GPU primitive is the bottleneck for the entire chaos/complexity
   signal family** — not just for ML clustering.

2. **TamSession's sharing infrastructure applies directly to the fintek signal farm** —
   three chaos algorithms on the same bin share one GPU distance computation.

3. **L∞ distance (for Sample Entropy) is the only case needing a max-combine TiledOp**
   — worth a note to the architect, not urgent.

4. **RQA's line detection is the first concrete use case for `Grouping::Segmented`** —
   the todo!() in accumulate.rs is blocking this algorithm.

5. **The fintek chaos algorithms are currently subsampled to CPU-tractable sizes**.
   GPU acceleration via TiledEngine would lift those limits entirely — more data,
   better signal quality, no approximation.

The architecture wasn't designed with chaos theory in mind. It fits anyway.
