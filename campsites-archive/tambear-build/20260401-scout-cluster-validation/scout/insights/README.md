# Scout Report: Cluster Validation + HDBSCAN + KMeans++ (2026-04-01)

Two documents here. One for the pathmaker, one for the naturalist.

## cluster_validation_gold_standards.md

What the gold standard (sklearn, R) actually computes for Silhouette, Davies-Bouldin,
Calinski-Harabasz, and Gap Statistic. With exact algorithms, edge cases, and tambear
primitive decompositions.

**Headline**: Three of four validation metrics are pure consumers of the distance matrix
already cached in TamSession. Silhouette and DB can follow DBSCAN/KMeans at essentially
zero GPU cost. CH is even cheaper — directly from MSR sufficient stats, no distance matrix
needed at all.

**Build order**: CH first (free from MSR) → Silhouette (consumes TamSession) → DB (needs
centroids + K²-sized tiled) → Gap (expensive, Monte Carlo, implement last).

## hdbscan_and_kmeanspp_gold_standards.md

HDBSCAN and KMeans++ algorithm details, gold standard behavior, and tambear decompositions.

**HDBSCAN headline**: ~230 lines of CPU code on top of the existing distance matrix.
No new GPU primitives. Steps 1-2 (core distances + MRD) are pure operations on
the dist matrix already cached by DBSCAN. Sharing works perfectly here.

**KMeans++ headline**: ~40 additional lines in kmeans.rs. The seeding loop reuses
the existing ASSIGN_KERNEL for distance computation. Add the density-peak seeding
alternative as a tambear-native option (novel — not in sklearn).

**Multi-restart**: run k independent KMeans instances on separate CUDA streams,
pick best by SS_W inertia. Embarrassingly parallel.
