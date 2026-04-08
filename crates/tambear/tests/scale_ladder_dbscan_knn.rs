/// Scale Ladder Benchmark: DBSCAN → KNN Session Sharing
///
/// Measures the cost of DBSCAN alone, KNN alone, and DBSCAN+KNN with session
/// sharing. The shared case should show near-zero KNN GPU cost because the
/// distance matrix is already cached in TamSession.
///
/// The distance matrix is O(n²) so the ceiling is ~50K-100K points depending
/// on available memory.
///
/// Run with: cargo test --release --test scale_ladder_dbscan_knn -- --nocapture

use tambear::clustering::ClusteringEngine;
use tambear::knn::{knn_session, KnnResult};
use tambear::intermediates::TamSession;
use std::time::Instant;

/// Deterministic pseudo-random cluster data: k blobs in d dimensions.
fn generate_blob_data(n: usize, d: usize, k: usize, seed: u64) -> Vec<f64> {
    let mut state = seed;
    let mut data = Vec::with_capacity(n * d);

    for i in 0..n {
        let cluster = i % k;
        let center = (cluster as f64) * 10.0; // clusters at 0, 10, 20, ...

        for _ in 0..d {
            // xorshift64 for noise
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            let u = (state as f64) / (u64::MAX as f64);
            // Roughly [-1, 1] noise around center
            let v = center + (u * 2.0 - 1.0);
            data.push(v);
        }
    }
    data
}

#[test]
fn scale_ladder_dbscan_knn_sharing() {
    let d = 3; // 3D points
    let k_clusters = 3; // 3 blobs
    let k_neighbors = 5; // 5 nearest neighbors
    let eps = 3.0; // DBSCAN epsilon
    let min_pts = 3; // DBSCAN min_samples

    let scales: Vec<(&str, usize)> = vec![
        ("100",    100),
        ("500",    500),
        ("1K",     1_000),
        ("2K",     2_000),
        ("5K",     5_000),
        ("10K",    10_000),
        ("20K",    20_000),
        ("50K",    50_000),
    ];

    println!();
    println!("=========================================================================");
    println!("SCALE LADDER: DBSCAN + KNN Session Sharing (tambear TamSession)");
    println!("=========================================================================");
    println!("{:>6}  {:>9}  {:>9}  {:>9}  {:>9}  {:>9}  {:>8}",
             "Scale", "DBSCAN(s)", "KNN_cold", "D+K_warm", "KNN_warm", "Savings", "DistMB");
    println!("-------------------------------------------------------------------------");

    for (label, n) in &scales {
        let data = generate_blob_data(*n, d, k_clusters, 42);
        let dist_mb = (*n as f64) * (*n as f64) * 8.0 / 1e6;

        // Check if distance matrix would exceed ~4 GB
        if dist_mb > 4000.0 {
            println!("{:>6}  *** distance matrix would be {:.0} MB — skipping ***", label, dist_mb);
            continue;
        }

        // ---- Cold DBSCAN (no session reuse) ----
        let mut engine = ClusteringEngine::new().unwrap();
        let mut session_cold_dbscan = TamSession::new();

        let t0 = Instant::now();
        let cluster_result = engine.dbscan_session(
            &mut session_cold_dbscan, &data, *n, d, eps, min_pts
        ).unwrap();
        let t_dbscan_cold = t0.elapsed().as_secs_f64();

        // ---- Cold KNN (fresh session, recomputes distance matrix) ----
        let mut session_cold_knn = TamSession::new();

        let t0 = Instant::now();
        let _knn_cold: KnnResult = knn_session(
            &mut session_cold_knn, &data, *n, d, k_neighbors
        ).unwrap();
        let t_knn_cold = t0.elapsed().as_secs_f64();

        // ---- Warm: DBSCAN then KNN on SAME session ----
        let mut engine_warm = ClusteringEngine::new().unwrap();
        let mut session_warm = TamSession::new();

        let t0 = Instant::now();
        let _ = engine_warm.dbscan_session(
            &mut session_warm, &data, *n, d, eps, min_pts
        ).unwrap();
        let t_dbscan_warm = t0.elapsed().as_secs_f64();

        // KNN on the warm session — distance matrix already cached
        let t1 = Instant::now();
        let knn_warm: KnnResult = knn_session(
            &mut session_warm, &data, *n, d, k_neighbors
        ).unwrap();
        let t_knn_warm = t1.elapsed().as_secs_f64();
        let t_dbscan_knn_warm = t_dbscan_warm + t_knn_warm;

        // Savings: cold (DBSCAN+KNN separate) vs warm (shared session)
        let t_cold_total = t_dbscan_cold + t_knn_cold;
        let savings_pct = if t_cold_total > 0.0 {
            (1.0 - t_dbscan_knn_warm / t_cold_total) * 100.0
        } else {
            0.0
        };

        println!("{:>6}  {:>9.4}  {:>9.4}  {:>9.4}  {:>9.4}  {:>8.1}%  {:>8.1}",
                 label, t_dbscan_cold, t_knn_cold, t_dbscan_knn_warm,
                 t_knn_warm, savings_pct, dist_mb);
        println!("        clusters={}, knn_check: point0_neighbors={:?}",
                 cluster_result.n_clusters,
                 knn_warm.neighbor_indices(0).iter().take(3).cloned().collect::<Vec<_>>());
    }

    println!("=========================================================================");
    println!();
    println!("NOTE: DBSCAN computes the O(n^2) distance matrix on GPU.");
    println!("      KNN_cold recomputes it from scratch. KNN_warm reuses it from TamSession.");
    println!("      Savings = 1 - warm_total / (cold_dbscan + cold_knn).");
    println!("      Expected: ~50% savings (one GPU distance computation eliminated).");
}
