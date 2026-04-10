//! Adversarial Wave 18 — Final NaN-eating targets before systemic fix
//!
//! Three remaining NaN-eating instances identified in the codebase scan:
//!
//! 1. `graph::graph_laplacian` — `fold(0.0_f64, f64::max)` at graph.rs:778
//!    max_deg computed from degree sequence. NaN edge weight → NaN degree sum
//!    → fold eats NaN → max_deg = 0.0 instead of NaN.
//!    Returns (laplacian, degrees, 0.0) with NaN in the laplacian — misleading.
//!
//! 2. `hypothesis.rs:2810` — private test helper `moment_stats_from_slice`
//!    Uses NaN-eating folds for min/max. Lives in test code but test helpers
//!    should also propagate NaN — a test that silently swallows NaN is a bad test.
//!    Documented here; not a production bug but a test-correctness smell.
//!
//! 3. `numerical.rs:1172-1173` — inside `unstable_simulation_oscillates` test
//!    Brusselator amplitude range via NaN-eating fold. Not production code.
//!    Documented for completeness.
//!
//! Mathematical truths:
//!   - graph_laplacian: if any edge weight is NaN, max_deg must be NaN
//!   - graph_laplacian: degree of node i = sum of row i of adj matrix; NaN weight → NaN degree
//!   - graph_laplacian: Laplacian with NaN edges is undefined
//!
//! All tests assert mathematical truths. Failures are bugs.

use tambear::graph_laplacian;

// ═══════════════════════════════════════════════════════════════════════════
// graph_laplacian — Test 1: NaN edge weight → max_deg must be NaN
// ═══════════════════════════════════════════════════════════════════════════

/// If an edge weight is NaN, the degree of the affected node is NaN.
/// degree[i] = sum of row i = ... + NaN + ... = NaN.
/// max_deg = fold(0.0, f64::max) over degrees including NaN → eats NaN.
/// max_deg returns 0.0 or the max of NaN-free degrees instead of NaN.
///
/// EXPECTED: graph_laplacian returns max_deg = NaN when any edge weight is NaN.
/// ACTUAL (BUG): returns max_deg = 0.0 (NaN-free degrees are all 0 in this case)
///   or the max of NaN-free degrees in the general case.
#[test]
fn graph_laplacian_nan_edge_weight_max_deg_must_be_nan() {
    // 3-node graph: edge (0,1)=1.0, edge (1,2)=NaN, edge (0,2)=1.0
    // adj (symmetric):
    //   [[0, 1, 1],
    //    [1, 0, NaN],
    //    [1, NaN, 0]]
    // degree[0] = 0+1+1 = 2
    // degree[1] = 1+0+NaN = NaN
    // degree[2] = 1+NaN+0 = NaN
    // max_deg = max(0.0, 2.0, NaN, NaN) = 2.0 (NaN eaten) → BUG
    let n = 3;
    let mut adj = vec![0.0_f64; n * n];
    adj[0 * n + 1] = 1.0; adj[1 * n + 0] = 1.0;  // edge (0,1)
    adj[0 * n + 2] = 1.0; adj[2 * n + 0] = 1.0;  // edge (0,2)
    adj[1 * n + 2] = f64::NAN; adj[2 * n + 1] = f64::NAN;  // edge (1,2) = NaN

    let (_lap, degrees, max_deg) = graph_laplacian(&adj, n);

    // degree[1] and degree[2] should be NaN
    assert!(degrees[1].is_nan(),
        "degree[1] should be NaN (contains NaN edge weight), got {}", degrees[1]);
    assert!(degrees[2].is_nan(),
        "degree[2] should be NaN (contains NaN edge weight), got {}", degrees[2]);

    // max_deg should be NaN (NaN in degree sequence)
    assert!(max_deg.is_nan(),
        "BUG: graph_laplacian max_deg should be NaN when any degree is NaN, got {} \
         — fold(0.0_f64, f64::max) eats NaN from the degree sequence",
        max_deg);
}

/// When ALL edge weights are NaN (empty graph semantically), all degrees are NaN.
/// max_deg should be NaN.
/// fold(0.0, max) over all-NaN sequence: f64::max(0.0, NaN) = 0.0 for each → 0.0.
/// This is the clearest case: the fold returns 0.0 (identity) for all-NaN input.
#[test]
fn graph_laplacian_all_nan_edges_max_deg_is_nan() {
    // 2-node graph, all NaN edges
    let n = 2;
    let adj = vec![f64::NAN; n * n];

    let (_lap, degrees, max_deg) = graph_laplacian(&adj, n);

    assert!(degrees[0].is_nan(),
        "degree[0] should be NaN for all-NaN adjacency, got {}", degrees[0]);
    assert!(max_deg.is_nan(),
        "BUG: max_deg should be NaN for all-NaN adjacency, got {} \
         — fold(0.0, f64::max) returns 0.0 (identity) for all-NaN input, \
         the clearest possible demonstration of the wrong-identity bug",
        max_deg);
}

// ═══════════════════════════════════════════════════════════════════════════
// graph_laplacian — Test 2: correctness on clean graphs
// ═══════════════════════════════════════════════════════════════════════════

/// 3-node path graph: 0—1—2, uniform edge weight 1.
/// degree[0]=1, degree[1]=2, degree[2]=1. max_deg=2.
/// Laplacian: L[i][i]=degree[i], L[i][j]=-adj[i][j] for i≠j.
#[test]
fn graph_laplacian_path_graph_correct() {
    let n = 3;
    let mut adj = vec![0.0_f64; n * n];
    adj[0 * n + 1] = 1.0; adj[1 * n + 0] = 1.0;
    adj[1 * n + 2] = 1.0; adj[2 * n + 1] = 1.0;

    let (lap, degrees, max_deg) = graph_laplacian(&adj, n);

    assert!((degrees[0] - 1.0).abs() < 1e-14, "degree[0]={}", degrees[0]);
    assert!((degrees[1] - 2.0).abs() < 1e-14, "degree[1]={}", degrees[1]);
    assert!((degrees[2] - 1.0).abs() < 1e-14, "degree[2]={}", degrees[2]);
    assert!((max_deg - 2.0).abs() < 1e-14, "max_deg={}", max_deg);

    // Laplacian diagonal = degrees
    assert!((lap[0*n+0] - 1.0).abs() < 1e-14, "L[0][0]={}", lap[0*n+0]);
    assert!((lap[1*n+1] - 2.0).abs() < 1e-14, "L[1][1]={}", lap[1*n+1]);
    assert!((lap[2*n+2] - 1.0).abs() < 1e-14, "L[2][2]={}", lap[2*n+2]);

    // Off-diagonal: -adj
    assert!((lap[0*n+1] - (-1.0)).abs() < 1e-14, "L[0][1]={}", lap[0*n+1]);
    assert!((lap[1*n+0] - (-1.0)).abs() < 1e-14, "L[1][0]={}", lap[1*n+0]);
    assert!((lap[1*n+2] - (-1.0)).abs() < 1e-14, "L[1][2]={}", lap[1*n+2]);
}

/// Laplacian has zero row sums for any undirected graph (conservation).
/// L * 1 = 0 where 1 is the all-ones vector.
#[test]
fn graph_laplacian_zero_row_sums() {
    // 4-node complete graph, weight 2.0
    let n = 4;
    let mut adj = vec![2.0_f64; n * n];
    for i in 0..n { adj[i*n+i] = 0.0; }

    let (lap, _deg, _max_deg) = graph_laplacian(&adj, n);

    for i in 0..n {
        let row_sum: f64 = (0..n).map(|j| lap[i*n+j]).sum();
        assert!(row_sum.abs() < 1e-12,
            "Laplacian row {} sum should be 0, got {}", i, row_sum);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// graph_laplacian — Test 3: wrong identity class
// ═══════════════════════════════════════════════════════════════════════════

/// The fold starts at 0.0 instead of NEG_INFINITY.
/// For max_deg of a graph with a NaN degree:
///   fold(0.0, max): max(0.0, deg[0], NaN, deg[2]) → eats NaN, returns max of clean
/// If ALL clean degrees are less than 0.0 (negative weights in a weighted graph),
/// the fold would return 0.0 even though no degree equals 0.0.
///
/// Test: weighted graph with all negative edge weights.
/// Degrees are all negative. max_deg should be the least-negative degree.
/// fold(0.0, max) would return 0.0 — WRONG for negative-weight graphs.
#[test]
fn graph_laplacian_negative_weights_wrong_max_deg() {
    // 2-node graph, edge weight -1.0
    // degree[0] = 0 + (-1) = -1.0
    // degree[1] = (-1) + 0 = -1.0
    // max_deg should be -1.0
    // fold(0.0, max): max(0.0, -1.0, -1.0) = 0.0 → WRONG
    let n = 2;
    let adj = vec![0.0_f64, -1.0, -1.0, 0.0];

    let (_lap, _degrees, max_deg) = graph_laplacian(&adj, n);

    // Mathematical truth: max degree of negative-weight graph is -1.0 (least negative)
    // Actual: fold(0.0, max) returns 0.0 (wrong identity dominates)
    assert!((max_deg - (-1.0)).abs() < 1e-14,
        "BUG: graph_laplacian max_deg for negative-weight graph should be -1.0, \
         got {} — fold(0.0_f64, f64::max) uses 0.0 as identity, which dominates \
         all negative degrees, returning 0.0 instead of -1.0",
        max_deg);
}
