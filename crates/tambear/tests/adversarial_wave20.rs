//! Adversarial Wave 20 — NaN-eating in shared DistanceMatrix consumers
//!
//! Pre-flight correctness sweep for the sharing-correctness attack surface.
//! Both `knn_from_distance` and `clustering_from_distance` accept a
//! pre-built `DistanceMatrix` that may come from TamSession cache.
//! If the cached matrix contains NaN entries, each consumer fails silently
//! in a different way — with no indication that the intermediate was corrupted.
//!
//! Three confirmed bugs:
//!
//! 1. `knn_from_distance` (knn.rs:108): `if d.is_nan() { continue; }` —
//!    NaN distances are silently skipped. Point i's k-neighborhood is computed
//!    from the remaining finite distances, with no indication that the row
//!    was NaN-contaminated. Undefined neighborhood → silently truncated neighbor list.
//!
//! 2. `clustering_from_distance` density step (clustering.rs:371):
//!    `if row[j] <= epsilon_threshold` with NaN row[j] evaluates to false.
//!    NaN-distance pairs are counted as non-neighbors. Density of NaN-involved
//!    points is undercounted → core/noise classification corrupted silently.
//!
//! 3. `clustering_from_distance` border assignment step (clustering.rs:419):
//!    `if is_core[j] && row_i[j] <= epsilon_threshold` — same NaN-false pattern.
//!    Border points that have NaN distances to core points are never assigned →
//!    misclassified as noise (label -1) instead of the correct border assignment.
//!
//! Mathematical truths:
//!   - knn: if D(i,j) is NaN for any j, the k-neighborhood of point i is undefined.
//!     The result must reflect this — either empty neighbors for i, or the entire
//!     KnnResult flagged as contaminated. Silently using finite-distance neighbors
//!     produces a false claim: "these are i's k nearest neighbors" when the true
//!     nearest neighbor might have been the NaN-distance point.
//!   - DBSCAN: if D(i,j) is NaN, whether i and j are within epsilon is undefined.
//!     The density of point i is undefined → its core/border/noise classification
//!     is undefined → label[i] should not be -1 (noise) by default.
//!   - Correctness baseline: on clean data (no NaN), knn and DBSCAN should produce
//!     correct results (verified by known-output tests).
//!
//! All tests assert mathematical truths. Failures are bugs.

use tambear::intermediates::{DistanceMatrix, Metric};
use tambear::knn::knn_from_distance;
use tambear::clustering::clustering_from_distance;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Build a 4×4 symmetric distance matrix (L2Sq metric).
/// Points: [0,0], [1,0], [10,0], [11,0] — two tight clusters far apart.
///
/// Exact squared distances:
///   D(0,1) = 1,  D(0,2) = 100, D(0,3) = 121
///   D(1,2) = 81, D(1,3) = 100, D(2,3) = 1
fn clean_4x4_distance_matrix() -> DistanceMatrix {
    let data = vec![
        // row 0: distances from point 0
        0.0_f64, 1.0, 100.0, 121.0,
        // row 1: distances from point 1
        1.0,     0.0, 81.0,  100.0,
        // row 2: distances from point 2
        100.0,  81.0,  0.0,    1.0,
        // row 3: distances from point 3
        121.0, 100.0,  1.0,    0.0,
    ];
    DistanceMatrix::from_vec(Metric::L2Sq, 4, data)
}

/// Same matrix with D(0,1) and D(1,0) set to NaN.
/// Point 1's distance to point 0 is undefined.
fn nan_contaminated_4x4_distance_matrix() -> DistanceMatrix {
    let data = vec![
        // row 0: D(0,1) = NaN
        0.0_f64,    f64::NAN, 100.0, 121.0,
        // row 1: D(1,0) = NaN (symmetric)
        f64::NAN,   0.0,      81.0,  100.0,
        // row 2: clean
        100.0,     81.0,       0.0,    1.0,
        // row 3: clean
        121.0,    100.0,       1.0,    0.0,
    ];
    DistanceMatrix::from_vec(Metric::L2Sq, 4, data)
}

// ═══════════════════════════════════════════════════════════════════════════
// KNN — Test 1: Correctness on clean data
// ═══════════════════════════════════════════════════════════════════════════

/// Baseline: clean 4×4 matrix with two clusters.
/// k=1 nearest neighbor of each point:
///   point 0 → point 1 (D=1, closest)
///   point 1 → point 0 (D=1, closest)
///   point 2 → point 3 (D=1, closest)
///   point 3 → point 2 (D=1, closest)
#[test]
fn knn_clean_data_correct_neighbors() {
    let dm = clean_4x4_distance_matrix();
    let result = knn_from_distance(&dm, 1);

    assert_eq!(result.neighbors[0].len(), 1,
        "Point 0 should have 1 neighbor, got {}", result.neighbors[0].len());
    assert_eq!(result.neighbors[0][0].0, 1,
        "Point 0's nearest neighbor should be point 1, got {}", result.neighbors[0][0].0);

    assert_eq!(result.neighbors[1].len(), 1,
        "Point 1 should have 1 neighbor, got {}", result.neighbors[1].len());
    assert_eq!(result.neighbors[1][0].0, 0,
        "Point 1's nearest neighbor should be point 0, got {}", result.neighbors[1][0].0);

    assert_eq!(result.neighbors[2][0].0, 3,
        "Point 2's nearest neighbor should be point 3, got {}", result.neighbors[2][0].0);
    assert_eq!(result.neighbors[3][0].0, 2,
        "Point 3's nearest neighbor should be point 2, got {}", result.neighbors[3][0].0);
}

// ═══════════════════════════════════════════════════════════════════════════
// KNN — Test 2: NaN in distance matrix must propagate
// ═══════════════════════════════════════════════════════════════════════════

/// NaN in D(0,1) and D(1,0): the distance between points 0 and 1 is undefined.
///
/// Point 1's true nearest neighbor might be point 0 (D=1 in the clean matrix).
/// With D(0,1) = NaN, the k-neighborhood of point 1 is undefined — we don't
/// know if point 0 is closer than point 2 (D=81) because the comparison is undefined.
///
/// EXPECTED: knn_from_distance returns 0 neighbors for point 1 (undefined neighborhood),
///           OR some sentinel indicating the result is contaminated.
/// ACTUAL (BUG): point 1's NaN distance to point 0 is silently skipped (continue).
///              Point 1 gets neighbors [2, 3] as if point 0 didn't exist —
///              a false claim that points 2 and 3 are its nearest neighbors.
///
/// The test checks: when D(i,j) is NaN, knn should NOT silently use remaining distances.
/// It checks that point 1's neighbor list is empty (undefined) rather than [2] (wrong).
#[test]
fn knn_nan_distance_must_not_silently_skip() {
    let dm = nan_contaminated_4x4_distance_matrix();
    let result = knn_from_distance(&dm, 1);

    // Point 1 has NaN distance to point 0.
    // Its k=1 neighborhood is undefined — we don't know if point 0 is its true nearest.
    // BUG: the implementation returns point 2 (D=81) as the nearest neighbor,
    // silently pretending the NaN-distance point 0 doesn't exist.
    assert!(result.neighbors[1].is_empty(),
        "BUG: knn with NaN in distance row should return empty neighbors (undefined), \
         got {:?} — `if d.is_nan() {{ continue; }}` at knn.rs:108 silently skips \
         NaN distances and computes neighbors from remaining finite entries, \
         producing a false k-neighborhood that excludes the undefined pair",
        result.neighbors[1]);
}

/// Point 0 has NaN distance to point 1.
/// Same bug from the other side of the NaN pair.
#[test]
fn knn_nan_distance_point0_neighborhood_undefined() {
    let dm = nan_contaminated_4x4_distance_matrix();
    let result = knn_from_distance(&dm, 1);

    assert!(result.neighbors[0].is_empty(),
        "BUG: knn with NaN in distance row for point 0 should return empty neighbors, \
         got {:?} — same silent-skip bug as point 1",
        result.neighbors[0]);
}

/// Clean rows (points 2 and 3) should be unaffected by NaN in other rows.
/// The NaN contamination is confined to the pair (0,1) — points 2 and 3 have
/// fully finite distance rows and should still get correct neighborhoods.
#[test]
fn knn_nan_in_one_row_does_not_affect_clean_rows() {
    let dm = nan_contaminated_4x4_distance_matrix();
    let result = knn_from_distance(&dm, 1);

    // Points 2 and 3 have clean distance rows — their neighborhoods are well-defined.
    assert_eq!(result.neighbors[2][0].0, 3,
        "Point 2's nearest neighbor (clean row) should still be point 3, got {}",
        result.neighbors[2][0].0);
    assert_eq!(result.neighbors[3][0].0, 2,
        "Point 3's nearest neighbor (clean row) should still be point 2, got {}",
        result.neighbors[3][0].0);
}

// ═══════════════════════════════════════════════════════════════════════════
// DBSCAN — Test 3: Correctness on clean data
// ═══════════════════════════════════════════════════════════════════════════

/// Baseline: clean 4×4 matrix with two well-separated clusters.
/// epsilon_threshold = 2.0 (squared distance), min_samples = 2.
///
/// Within-cluster distances: D(0,1)=1, D(2,3)=1 — both ≤ 2.
/// Between-cluster distances: D(0,2)=100, D(0,3)=121, D(1,2)=81, D(1,3)=100 — all > 2.
///
/// Expected: 2 clusters, no noise. Each cluster has 2 core points.
#[test]
fn dbscan_clean_data_two_clusters() {
    let dm = clean_4x4_distance_matrix();
    let result = clustering_from_distance(&dm.data, 4, 2.0, 2);

    assert_eq!(result.n_clusters, 2,
        "Clean 4-point data with two separated clusters should give 2 clusters, got {}",
        result.n_clusters);
    assert_eq!(result.n_noise, 0,
        "No noise points expected in clean 2-cluster data, got {} noise", result.n_noise);

    // Points 0 and 1 should be in the same cluster
    assert_eq!(result.labels[0], result.labels[1],
        "Points 0 and 1 should be in the same cluster (D=1 < epsilon=2), \
         got labels {} and {}", result.labels[0], result.labels[1]);

    // Points 2 and 3 should be in the same cluster
    assert_eq!(result.labels[2], result.labels[3],
        "Points 2 and 3 should be in the same cluster (D=1 < epsilon=2), \
         got labels {} and {}", result.labels[2], result.labels[3]);

    // The two clusters should be different
    assert_ne!(result.labels[0], result.labels[2],
        "Points 0 and 2 should be in different clusters (D=100 >> epsilon=2), \
         got same label {}", result.labels[0]);
}

// ═══════════════════════════════════════════════════════════════════════════
// DBSCAN — Test 4: NaN in distance matrix corrupts density count
// ═══════════════════════════════════════════════════════════════════════════

/// NaN in D(0,1) and D(1,0): points 0 and 1's mutual reachability is undefined.
///
/// With epsilon_threshold=2.0 and min_samples=2:
///   Clean: point 0 density = 2 (itself + point 1). Core point.
///          point 1 density = 2 (itself + point 0). Core point.
///
/// With NaN D(0,1):
///   `row[1] <= 2.0` with NaN row[1] = false → point 1 not counted in point 0's density.
///   Point 0 density = 1 (only itself). Not a core point (need ≥ 2).
///   Point 1 density = 1 (only itself). Not a core point.
///
/// EXPECTED: density of NaN-involved points is undefined → result is undefined.
/// ACTUAL (BUG): density undercounted → points 0 and 1 classified as noise (label=-1),
///              even though with known D(0,1)=1 they would both be core points.
///
/// The NaN silently changes the cluster assignment from "cluster member" to "noise."
/// Undefined distance → specific wrong label. That's a false claim.
#[test]
fn dbscan_nan_distance_corrupts_density_count() {
    let dm = nan_contaminated_4x4_distance_matrix();
    let result = clustering_from_distance(&dm.data, 4, 2.0, 2);

    // BUG: points 0 and 1 get classified as noise because their mutual NaN
    // distance is silently treated as "not within epsilon."
    // Their density drops from 2 to 1, dropping them below min_samples=2.
    //
    // The correct behavior: when D(0,1) is NaN, the density of points 0 and 1
    // is undefined, and their labels should reflect that — not silently -1.
    //
    // We check: is point 0's label -1 (noise) despite being in a cluster in clean data?
    // If yes, that's the bug: NaN → false → non-core → noise (wrong).
    let clean_result = clustering_from_distance(&clean_4x4_distance_matrix().data, 4, 2.0, 2);

    // In clean data, point 0 is a cluster member (label ≥ 0)
    assert!(clean_result.labels[0] >= 0,
        "Sanity: point 0 should be in a cluster in clean data, got label {}",
        clean_result.labels[0]);

    // With NaN contamination, the behavior diverges — document the corruption
    // The bug: NaN silently makes point 0 appear as noise
    assert!(result.labels[0] >= 0,
        "BUG: DBSCAN with NaN D(0,1) misclassifies point 0 as noise (label={}) — \
         `row[j] <= epsilon_threshold` at clustering.rs:371 evaluates to false for NaN, \
         undercounting density from {} (clean) to effectively 1, below min_samples=2. \
         Undefined distance becomes specific wrong label -1 (noise).",
        result.labels[0], clean_result.labels[0]);
}

/// NaN in the entire row for point 0: all distances from point 0 are undefined.
///
/// With all D(0,j) = NaN: point 0's density = 1 (only self-distance=0).
/// Not a core point. Since its only potential core neighbors also have NaN distances
/// to it, border assignment also fails → label stays -1 (noise).
///
/// Undefined neighborhood → specific wrong label. The bug is that -1 means "noise"
/// in DBSCAN terminology, implying the point is isolated — but point 0 is NOT
/// isolated, its distances are merely undefined.
#[test]
fn dbscan_nan_full_row_misclassified_as_noise() {
    // All distances from point 0 undefined.
    let data = vec![
        // row 0: all NaN (point 0's distances undefined)
        0.0_f64, f64::NAN, f64::NAN, f64::NAN,
        // row 1: D(1,0) = NaN (symmetric), rest clean
        f64::NAN, 0.0,      1.0,     100.0,
        // row 2: D(2,0) = NaN, rest clean
        f64::NAN, 1.0,      0.0,       1.0,
        // row 3: D(3,0) = NaN, rest clean
        f64::NAN, 100.0,    1.0,       0.0,
    ];
    let dm = DistanceMatrix::from_vec(Metric::L2Sq, 4, data);
    let result = clustering_from_distance(&dm.data, 4, 2.0, 2);

    // BUG: point 0 gets label -1 (noise) because all its distances are NaN.
    // NaN comparisons evaluate to false, so no neighbors are found → density=1
    // → not core → border assignment also fails → noise.
    //
    // "Noise" is a specific claim: the point is isolated with no nearby neighbors.
    // But point 0's isolation is undefined, not confirmed.
    assert!(result.labels[0] >= 0,
        "BUG: DBSCAN with all-NaN row for point 0 classifies it as noise (label=-1) — \
         undefined distances silently produce density=1 via NaN comparison false, \
         then border assignment also fails (clustering.rs:419 same pattern), \
         giving label=-1 (noise) which is a false specific claim about an undefined point");
}
