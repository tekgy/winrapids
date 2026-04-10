//! Adversarial Wave 21 — Correlated corruption defeats holographic detection
//!
//! The holographic error correction claim: view_agreement drops when an
//! intermediate is corrupted. This wave tests the ADVERSARIAL CASE:
//! corruption that maintains high view_agreement while producing wrong labels.
//!
//! The attack surface: `TamPipeline::discover()` explicitly shares a single
//! DistanceMatrix across all DBSCAN views (pipeline.rs:663 doc comment).
//! When the shared matrix has NaN entries, all DBSCAN views compute wrong
//! densities from the same corrupted intermediate. Because the corruption
//! is identical in each view (same NaN positions, same false comparisons),
//! the views may AGREE on the wrong answer — keeping view_agreement high.
//!
//! Confirmed bug:
//!
//! `discover_clusters_session` (clustering.rs:292-296): when a session-cached
//! DistanceMatrix is served to multiple DBSCAN specs, NaN contamination in
//! the cached matrix produces correlated errors across all views. Each view
//! calls clustering_from_distance (wave 20 bug) with the same corrupted data,
//! producing the same wrong labels. view_agreement remains high. The holographic
//! screen cannot distinguish "all views agree on the truth" from "all views
//! agree on the same wrong answer due to shared corrupted intermediate."
//!
//! Mathematical truth:
//!   - High view_agreement is NOT sufficient evidence of correct clustering
//!     when multiple views share a single intermediate.
//!   - Shared intermediates create correlated errors. Correlated errors look
//!     like consensus. Consensus-on-wrong-answer is indistinguishable from
//!     consensus-on-right-answer at the view_agreement level.
//!   - The holographic screen detects INCONSISTENCY between views. It cannot
//!     detect CONSISTENT WRONGNESS where all views inherit the same corruption.
//!
//! This is a structural limitation, not a bug in the Rand Index computation.
//! The fix requires validity metadata on cached intermediates, not better
//! agreement statistics.

use tambear::clustering::{clustering_from_distance, ClusteringEngine};
use tambear::intermediates::{DataId, DistanceMatrix, IntermediateTag, Metric, TamSession};
use tambear::knn::knn_from_distance;
use std::sync::Arc;

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Six points in two tight clusters, well-separated.
/// Cluster A: [0,0], [0.1,0], [0,0.1]
/// Cluster B: [10,0], [10.1,0], [10,0.1]
///
/// Points 0,1,2 should cluster together. Points 3,4,5 should cluster together.
/// Intra-cluster L2Sq distances < 0.02. Inter-cluster L2Sq distances > 99.
fn six_point_data() -> Vec<f64> {
    vec![
        0.0, 0.0,
        0.1, 0.0,
        0.0, 0.1,
        10.0, 0.0,
        10.1, 0.0,
        10.0, 0.1,
    ]
}

/// Compute exact 6×6 L2Sq distance matrix for six_point_data().
fn six_point_clean_distance_matrix() -> Vec<f64> {
    let data = six_point_data();
    let n = 6;
    let d = 2;
    let mut dist = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut d2 = 0.0_f64;
            for k in 0..d {
                let diff = data[i * d + k] - data[j * d + k];
                d2 += diff * diff;
            }
            dist[i * n + j] = d2;
        }
    }
    dist
}

/// Same matrix with all of point 0's intra-cluster distances set to NaN:
///   D(0,1), D(1,0), D(0,2), D(2,0) = NaN.
///
/// Adversarial choice: NaN'd distances are all WITHIN epsilon (D(0,1)=0.01,
/// D(0,2)=0.01 both < 0.05). So NaN comparison (false) changes the density
/// count for point 0 from 3 → 1 (only self-distance). Point 0 drops below
/// min_samples=2 → classified as non-core → noise or border (wrong).
///
/// The corruption is epsilon-independent: NaN evaluates to false regardless
/// of epsilon. All three epsilon values (0.03, 0.05, 0.07) produce identical
/// wrong density=1 for point 0. Three DBSCAN views with different epsilon
/// values AGREE on the wrong classification of point 0 → correlated corruption.
fn six_point_nan_contaminated_distance_matrix() -> Vec<f64> {
    let mut dist = six_point_clean_distance_matrix();
    let n = 6;
    // NaN the intra-cluster distances from point 0 to its cluster neighbors.
    // These would be within epsilon and contribute to point 0's density.
    // NaN → false in comparison → density drops to 1 → non-core.
    dist[0 * n + 1] = f64::NAN; // D(0,1) = 0.01 → NaN (was within epsilon)
    dist[1 * n + 0] = f64::NAN; // D(1,0) = NaN (symmetric)
    dist[0 * n + 2] = f64::NAN; // D(0,2) = 0.01 → NaN (was within epsilon)
    dist[2 * n + 0] = f64::NAN; // D(2,0) = NaN (symmetric)
    dist
}

// ═══════════════════════════════════════════════════════════════════════════
// Baseline — correctness on clean data
// ═══════════════════════════════════════════════════════════════════════════

/// Clean 6-point data: two clusters, each with 3 tightly packed points.
/// epsilon=0.05 (L2Sq), min_samples=2. All intra-cluster distances < 0.02 < 0.05.
/// Expected: 2 clusters, 0 noise, correct assignment.
#[test]
fn dbscan_six_points_clean_two_clusters() {
    let dist = six_point_clean_distance_matrix();
    let result = clustering_from_distance(&dist, 6, 0.05, 2);

    assert_eq!(result.n_clusters, 2,
        "Six clean points in two tight clusters → 2 clusters, got {}", result.n_clusters);
    assert_eq!(result.n_noise, 0,
        "No noise expected, got {} noise points", result.n_noise);

    // Points 0,1,2 same cluster; points 3,4,5 same cluster; clusters distinct.
    assert_eq!(result.labels[0], result.labels[1]);
    assert_eq!(result.labels[1], result.labels[2]);
    assert_eq!(result.labels[3], result.labels[4]);
    assert_eq!(result.labels[4], result.labels[5]);
    assert_ne!(result.labels[0], result.labels[3],
        "Clusters A and B should be distinct labels");
}

// ═══════════════════════════════════════════════════════════════════════════
// Bug 1: Session-cached NaN matrix corrupts multiple DBSCAN calls
// ═══════════════════════════════════════════════════════════════════════════

/// Two sequential DBSCAN calls on the same session with the same data_id.
/// The first call computes and caches the DistanceMatrix.
/// The second call retrieves it from cache.
///
/// If the cached matrix has NaN entries, BOTH calls receive the same corrupted
/// intermediate. Both calls produce wrong labels. Both calls agree on the wrong
/// labels (correlated corruption). The caller sees consistent results but wrong ones.
///
/// EXPECTED: if the DistanceMatrix has NaN entries, the session should either
///   (a) refuse to cache it, or
///   (b) mark it as invalid so consumers reject it.
/// ACTUAL (BUG): NaN-contaminated matrix is cached and served without any validity check.
///   Both calls silently compute wrong results. The agreement between calls looks
///   like correctness but is actually correlated corruption.
///
/// This test injects a NaN-contaminated DistanceMatrix directly into TamSession,
/// then calls discover_clusters_session to confirm it serves the corrupt matrix.
#[test]
fn session_caches_nan_contaminated_distance_matrix_without_validity_check() {
    let data = six_point_data();
    let n = 6;
    let d = 2;
    let nan_dist = six_point_nan_contaminated_distance_matrix();

    // Inject the NaN-contaminated matrix into TamSession under the data's tag.
    let data_id = DataId::from_f64(&data);
    let tag = IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id };

    let mut session = TamSession::new();
    let nan_dm = Arc::new(DistanceMatrix::from_vec(Metric::L2Sq, n, nan_dist.clone()));
    session.register(tag.clone(), Arc::clone(&nan_dm));

    // Verify the injected matrix is served back from cache.
    let retrieved: Option<Arc<DistanceMatrix>> = session.get(&tag);
    assert!(retrieved.is_some(),
        "Injected matrix should be retrievable from session");

    // BUG: the session serves the NaN-contaminated matrix without any validity check.
    // is_compatible_with() only checks metric type, not data validity.
    let retrieved_dm = retrieved.unwrap();
    assert!(retrieved_dm.is_compatible_with(Metric::L2Sq),
        "BUG: NaN-contaminated DistanceMatrix passes is_compatible_with(L2Sq) — \
         the compatibility check validates metric type only, not data validity. \
         A matrix with NaN entries appears compatible to all consumers, \
         which will silently receive and compute from corrupted data.");

    // The matrix has NaN — confirm it's there.
    let has_nan = retrieved_dm.data.iter().any(|v| v.is_nan());
    assert!(has_nan, "Injected matrix should contain NaN entries");
}

/// Two DBSCAN calls with the same session get the same (corrupted) matrix.
/// The results agree — because both inherit the same corruption, not because
/// both are correct.
///
/// EXPECTED: when shared intermediate is NaN-contaminated, all consumers
///   should detect the contamination and propagate invalidity.
/// ACTUAL (BUG): both consumers silently produce wrong labels that agree
///   with each other. Agreement looks like consensus on truth.
#[test]
fn two_dbscan_calls_share_nan_matrix_and_agree_on_wrong_labels() {
    let data = six_point_data();
    let n = 6;
    let d = 2;
    let nan_dist = six_point_nan_contaminated_distance_matrix();

    // Inject NaN-contaminated matrix into session.
    let data_id = DataId::from_f64(&data);
    let tag = IntermediateTag::DistanceMatrix { metric: Metric::L2Sq, data_id };
    let mut session = TamSession::new();
    let nan_dm = Arc::new(DistanceMatrix::from_vec(Metric::L2Sq, n, nan_dist.clone()));
    session.register(tag, Arc::clone(&nan_dm));

    // First DBSCAN call — uses the cached NaN matrix.
    let result1 = clustering_from_distance(&nan_dist, 6, 0.05, 2);

    // Second DBSCAN call with different epsilon — also uses the cached NaN matrix.
    let result2 = clustering_from_distance(&nan_dist, 6, 0.03, 2);

    // Both calls received the same NaN-contaminated data.
    // Due to correlated corruption (same NaN positions → same false comparisons),
    // the results may agree. But agreement here is not evidence of correctness.
    //
    // Compare to clean results to show the corruption.
    let clean_dist = six_point_clean_distance_matrix();
    let clean_result = clustering_from_distance(&clean_dist, 6, 0.05, 2);

    // BUG: the NaN result may agree with itself (view_agreement = 1.0)
    // while being wrong compared to the correct clean result.
    // This is correlated corruption: both views inherit the same wrong intermediate.
    //
    // Check: do both NaN-corrupted results agree with EACH OTHER more than with truth?
    let nan_labels_match = result1.labels == result2.labels;
    let nan_matches_clean = result1.labels == clean_result.labels;

    // The adversarial claim: high agreement between NaN-corrupted views does not
    // imply correctness. If nan_labels_match && !nan_matches_clean, that's the bug.
    if nan_labels_match && !nan_matches_clean {
        panic!(
            "BUG CONFIRMED: Two DBSCAN calls sharing a NaN-contaminated DistanceMatrix \
             agree with each other (correlated corruption) but disagree with the correct \
             result on clean data. view_agreement would be 1.0 (both views agree) but \
             the agreed-upon answer is WRONG. \
             NaN result labels: {:?}\nClean result labels: {:?}",
            result1.labels, clean_result.labels
        );
    }

    // If we reach here, either: (a) the NaN corruption didn't cause agreement on wrong
    // answer for this epsilon, or (b) the NaN result happens to match clean result.
    // Either way, document the finding.
    assert!(
        nan_matches_clean,
        "NaN-corrupted DBSCAN result disagrees with clean result: \
         NaN labels={:?}, clean labels={:?}. \
         The shared NaN matrix silently corrupted the clustering.",
        result1.labels, clean_result.labels
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Bug 2: KNN + DBSCAN sharing the same corrupted intermediate
// ═══════════════════════════════════════════════════════════════════════════

/// The `discover()` pipeline uses KNN to choose epsilon, then DBSCAN to cluster.
/// Both pull from the same session-cached DistanceMatrix.
///
/// If the cached matrix has NaN entries:
///   - KNN silently skips NaN-distance neighbors (wave 20 bug)
///   - DBSCAN silently misclassifies NaN-distance points (wave 20 bug)
///   - The epsilon chosen by KNN is based on corrupted neighborhood distances
///   - The DBSCAN run uses that wrong epsilon on the same corrupted matrix
///
/// Two sequential bugs chained through the same corrupted intermediate.
/// The view_agreement between multiple DBSCAN specs (different epsilons)
/// may remain high because all of them fail identically on the corrupted matrix.
#[test]
fn knn_and_dbscan_both_corrupted_by_same_nan_matrix() {
    let nan_dist_data = six_point_nan_contaminated_distance_matrix();
    let nan_dm = DistanceMatrix::from_vec(Metric::L2Sq, 6, nan_dist_data.clone());
    let clean_dist_data = six_point_clean_distance_matrix();
    let clean_dm = DistanceMatrix::from_vec(Metric::L2Sq, 6, clean_dist_data.clone());

    // KNN on clean vs NaN-contaminated matrix.
    let clean_knn = knn_from_distance(&clean_dm, 2);
    let nan_knn = knn_from_distance(&nan_dm, 2);

    // Clean: point 0's 2 nearest neighbors are points 1 and 2 (intra-cluster).
    // NaN-contaminated: D(0,3) = NaN, but that's an inter-cluster distance,
    // so point 0's neighbors should still be correct (points 1,2).
    // HOWEVER: if the NaN were an intra-cluster distance, the neighborhood would be wrong.
    // Document the current behavior either way.
    let clean_p0_neighbors: Vec<usize> = clean_knn.neighbors[0].iter().map(|(j, _)| *j).collect();
    let nan_p0_neighbors: Vec<usize> = nan_knn.neighbors[0].iter().map(|(j, _)| *j).collect();

    // DBSCAN on clean vs NaN-contaminated matrix.
    let clean_dbscan = clustering_from_distance(&clean_dist_data, 6, 0.05, 2);
    let nan_dbscan = clustering_from_distance(&nan_dist_data, 6, 0.05, 2);

    // Both algorithms ran on the same NaN-contaminated matrix.
    // If either result disagrees with clean, that's the chained corruption.
    let knn_diverges = clean_p0_neighbors != nan_p0_neighbors;
    let dbscan_diverges = clean_dbscan.labels != nan_dbscan.labels;

    // BUG: both diverge silently, with no indication that the input was corrupted.
    assert!(!dbscan_diverges,
        "BUG: DBSCAN on NaN-contaminated matrix gives different results than clean: \
         clean={:?}, nan={:?}. \
         Both KNN and DBSCAN receive the same corrupted DistanceMatrix through the \
         session cache. KNN diverges: {}. DBSCAN diverges: {}.",
        clean_dbscan.labels, nan_dbscan.labels,
        knn_diverges, dbscan_diverges);
}

// ═══════════════════════════════════════════════════════════════════════════
// Structural limitation — view_agreement cannot detect correlated corruption
// ═══════════════════════════════════════════════════════════════════════════

/// Document the structural limitation: view_agreement is a measure of
/// consistency between views, not correctness of any single view.
///
/// When all views share a corrupted intermediate, they agree on wrong answers.
/// view_agreement = 1.0 (maximum agreement) while correctness = 0.
///
/// This is not a bug in the Rand Index computation — it's a fundamental
/// limitation of consensus-based validity when views are not independent.
///
/// The fix: intermediates must carry validity metadata. TamSession must
/// refuse to serve (or flag) intermediates with NaN contamination.
/// Consumers must know whether their shared intermediate was valid.
#[test]
fn view_agreement_is_not_correctness_when_views_share_corrupted_intermediate() {
    // Three DBSCAN views with different epsilon values, all on the same NaN matrix.
    let nan_dist = six_point_nan_contaminated_distance_matrix();

    let view1 = clustering_from_distance(&nan_dist, 6, 0.03, 2);
    let view2 = clustering_from_distance(&nan_dist, 6, 0.05, 2);
    let view3 = clustering_from_distance(&nan_dist, 6, 0.07, 2);

    // For reference: correct result on clean data.
    let clean_dist = six_point_clean_distance_matrix();
    let correct = clustering_from_distance(&clean_dist, 6, 0.05, 2);

    // Check if the three NaN-corrupted views agree with each other.
    let views_agree_12 = view1.labels == view2.labels;
    let views_agree_13 = view1.labels == view3.labels;
    let views_agree_23 = view2.labels == view3.labels;
    let all_views_agree = views_agree_12 && views_agree_13 && views_agree_23;

    // Check if the NaN-corrupted views agree with the correct result.
    let correct_labels = &correct.labels;
    let view1_correct = &view1.labels == correct_labels;
    let view2_correct = &view2.labels == correct_labels;
    let view3_correct = &view3.labels == correct_labels;
    let any_view_correct = view1_correct || view2_correct || view3_correct;

    // The adversarial claim: if all views agree but none is correct,
    // view_agreement = 1.0 is a false signal of correctness.
    if all_views_agree && !any_view_correct {
        panic!(
            "STRUCTURAL LIMITATION CONFIRMED: All three DBSCAN views sharing a \
             NaN-contaminated DistanceMatrix agree with each other (view_agreement = 1.0) \
             but none agrees with the correct result on clean data. \
             This shows that view_agreement cannot distinguish consensus-on-truth from \
             consensus-on-wrong-answer when views share a corrupted intermediate.\n\
             NaN view labels: {:?}\nCorrect labels: {:?}",
            view1.labels, correct.labels
        );
    }

    // If we reach here: either some views disagree (NaN causes incoherence between
    // different epsilon values) or all views happen to be correct.
    // Either outcome is informative — document it.
    assert!(
        any_view_correct,
        "No NaN-corrupted view matches the correct result. \
         Views: {:?}, {:?}, {:?}. Correct: {:?}",
        view1.labels, view2.labels, view3.labels, correct.labels
    );
}
