# .discover() IS Holographic Error Correction

*2026-04-10 — Aristotle, following team-lead directive*

## The Claim

The `.discover()` mechanism in `pipeline.rs` already implements holographic error correction. This is not a metaphor — it's a formally testable property of the sharing architecture.

## How .discover() Works (from code)

`discover()` runs multiple clustering specs (DBSCAN at different epsilons + KMeans at different k) that SHARE the same distance matrix through TamSession. It produces:
- `views`: Vec<ClusterView> — each spec's result
- `view_agreement`: f64 — pairwise Rand Index across all views (1.0 = full agreement)
- `modal_k`: usize — most common cluster count across views

The key: all views share the same DistanceMatrix intermediate. Different algorithms, same shared state.

## The Error Correction Property

**Claim**: If the shared DistanceMatrix is corrupted (NaN injection, hardware bit-flip, numerical overflow), the `view_agreement` drops.

**Why**: DBSCAN and KMeans use the distance matrix differently:
- DBSCAN: threshold comparison (d < epsilon → neighbor). A corruption changes neighborhood structure.
- KMeans: centroid assignment (argmin over distances). A corruption changes assignments.

A corruption that moves one point's distances affects DBSCAN neighborhoods and KMeans assignments DIFFERENTLY (because the algorithms have different sensitivity to distance perturbations). The Rand Index between the corrupted views drops because the algorithms no longer agree on the same structure.

**The holographic property**: The redundancy across views (multiple algorithms consuming the same intermediate) enables detecting corruption in the intermediate WITHOUT re-computing it. You don't need to check the distance matrix directly — the disagreement between views IS the check.

**The formal statement**: Let D be the true distance matrix and D' be a corrupted version. Let v_i(D) be the cluster assignment from algorithm i on D. Then:

`Rand(v_DBSCAN(D'), v_KMeans(D')) < Rand(v_DBSCAN(D), v_KMeans(D))`

whenever D' differs from D in ways that affect the two algorithms differently (which is generically true for any corruption beyond rounding).

## What Makes This Genuinely Holographic

In AdS/CFT holographic codes:
1. The bulk state (interior geometry) is encoded redundantly on the boundary (CFT observables)
2. Local corruption of boundary data can be detected by checking consistency across boundary regions
3. The code distance (number of boundary sites that must be corrupted before the error is undetectable) determines the robustness

In TamSession:
1. The intermediate (distance matrix) is the "bulk state"
2. The algorithm views (DBSCAN, KMeans) are "boundary observables"
3. Corruption of the intermediate IS detectable by checking `view_agreement`
4. The "code distance" = the number of views that must be simultaneously corrupted before the disagreement becomes undetectable

**The code distance of .discover()**: Currently, `discover()` runs ~8 specs (4 DBSCAN + 4 KMeans). The code distance is 8 — you'd need to corrupt all 8 views consistently to hide the error. Each additional spec increases the code distance by 1. Running MORE methods = MORE robust error detection.

This is EXACTLY the "run everything, never gate production" principle: running more methods isn't just producing more answers — it's INCREASING THE ERROR CORRECTION CAPABILITY of the system.

## Testable Prediction

1. Take a clean dataset with known structure (two well-separated clusters)
2. Run `discover()` → get baseline `view_agreement` (should be ~1.0)
3. Inject corruption into the distance matrix (flip sign of one entry, inject NaN, add large offset to one row)
4. Run the views against the corrupted matrix
5. Measure `view_agreement` on corrupted views

**Prediction**: view_agreement drops proportionally to the severity of the corruption. Small corruption → small drop. Large corruption → large drop. NaN injection → catastrophic drop (at least one view produces garbage).

**The null hypothesis**: view_agreement is insensitive to corruption (the algorithms agree even on corrupted data). If true, the holographic property doesn't hold.

## Connection to .discover() Beyond Clustering

The holographic error-correction property extends to ALL superposition methods, not just clustering:

- **Correlation discovery**: Pearson, Spearman, Kendall all share MomentStats. If MomentStats is corrupted, the methods disagree.
- **Regression discovery**: OLS, robust, quantile regression all share the Gram matrix. Corruption → disagreement.
- **Normality testing**: Shapiro-Wilk, Anderson-Darling, Lilliefors all share SortedOrder. Corruption → disagreement.

In general: any `.discover()` layer that runs multiple methods on shared intermediates inherits the holographic property. The MORE methods that share an intermediate, the STRONGER the error correction.

## Implications for Architecture

1. **view_agreement is a HEALTH metric**, not just a statistical result. Low agreement may mean "the algorithms genuinely disagree about structure" OR "the shared intermediate is corrupted." Layer 2 (using() override transparency) should distinguish these cases.

2. **TamSession should track view_agreement per intermediate**, not per pipeline. If 6 methods share CovMatrix and their outputs disagree, the CovMatrix is suspect.

3. **The "run everything" principle is a CORRECTNESS property**, not just a completeness property. More methods = more error detection = higher confidence in the intermediates.

## Status

Theoretical analysis complete. The prediction is immediately testable with existing code and tests. The connection to holographic codes is suggestive but not yet rigorous — would need formal proof that the Rand Index sensitivity to distance matrix perturbation matches the code distance of a stabilizer code.
