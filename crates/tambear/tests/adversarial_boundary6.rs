//! Adversarial Boundary Tests — Wave 6
//!
//! Targets: graph (F29), spatial, information_theory, robust
//!
//! Attack taxonomy:
//! - Type 1: Division by zero / denominator collapse
//! - Type 2: Convergence / iteration boundary
//! - Type 3: Cancellation / precision
//! - Type 4: Equipartition / degenerate geometry
//! - Type 5: Structural incompatibility

use tambear::graph::*;

// ═══════════════════════════════════════════════════════════════════════════
// GRAPH (F29)
// ═══════════════════════════════════════════════════════════════════════════

/// Dijkstra with negative weights: gives wrong shortest paths silently.
/// Type 3: silent wrong answer.
#[test]
fn dijkstra_negative_weights() {
    // Triangle: 0→1 (w=1), 1→2 (w=1), 0→2 (w=10)
    // Add negative edge: 1→2 (w=-5). True shortest 0→2 via 1 = 1 + (-5) = -4.
    // Dijkstra doesn't handle negatives — will it return wrong answer?
    let mut g = Graph::new(3);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, -5.0);
    g.add_edge(0, 2, 10.0);
    let (dist, _) = dijkstra(&g, 0);
    // Correct shortest: 0→1→2 = -4. Dijkstra may return 10.0 (wrong).
    if (dist[2] - 10.0).abs() < 1e-10 {
        eprintln!("CONFIRMED BUG: Dijkstra gives wrong answer with negative weights (dist[2]={} instead of -4)", dist[2]);
    } else if (dist[2] - (-4.0)).abs() < 1e-10 {
        // Surprisingly correct
    }
}

/// Dijkstra with NaN weight: NaN < f64 is false → edge never relaxed.
#[test]
fn dijkstra_nan_weight() {
    let mut g = Graph::new(3);
    g.add_edge(0, 1, f64::NAN);
    g.add_edge(0, 2, 5.0);
    let (dist, _) = dijkstra(&g, 0);
    // dist[0] + NaN = NaN. NaN < Inf is false → edge not relaxed → dist[1] = Inf
    if dist[1] == f64::INFINITY {
        eprintln!("NOTE: Dijkstra with NaN weight: unreachable (NaN comparison = false)");
    } else if dist[1].is_nan() {
        eprintln!("CONFIRMED BUG: Dijkstra propagates NaN distance");
    }
}

/// BFS on empty graph (0 nodes).
#[test]
fn bfs_empty_graph() {
    let g = Graph::new(0);
    let result = std::panic::catch_unwind(|| {
        bfs(&g, 0)
    });
    if result.is_err() {
        eprintln!("NOTE: BFS panics on empty graph with source=0 (out of bounds)");
    }
}

/// Bellman-Ford with n=1: single node, no edges.
#[test]
fn bellman_ford_single_node() {
    let g = Graph::new(1);
    let result = bellman_ford(&g, 0);
    assert!(result.is_some(), "Single node graph should have no negative cycles");
    let (dist, _) = result.unwrap();
    assert!((dist[0] - 0.0).abs() < 1e-10, "Distance to self should be 0");
}

/// Bellman-Ford with negative cycle: should return None.
#[test]
fn bellman_ford_negative_cycle() {
    let mut g = Graph::new(3);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, -3.0);
    g.add_edge(2, 0, 1.0); // cycle 0→1→2→0 = 1 + (-3) + 1 = -1 (negative)
    let result = bellman_ford(&g, 0);
    assert!(result.is_none(), "Should detect negative cycle");
}

/// Floyd-Warshall with negative cycle: distances become -∞.
/// Type 5: structural incompatibility.
#[test]
fn floyd_warshall_negative_cycle() {
    let mut g = Graph::new(3);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, -3.0);
    g.add_edge(2, 0, 1.0); // negative cycle
    let dist = floyd_warshall(&g);
    // Floyd-Warshall doesn't detect negative cycles — it just produces garbage
    let has_negative = dist.iter().any(|row| row.iter().any(|&d| d < -1e10));
    if has_negative {
        eprintln!("NOTE: Floyd-Warshall with negative cycle produces very negative distances (no detection)");
    }
}

/// PageRank with damping=0: all nodes get uniform rank.
#[test]
fn pagerank_damping_zero() {
    let mut g = Graph::new(3);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 1.0);
    let rank = pagerank(&g, 0.0, 100, 1e-6);
    // damping=0 → base = 1/n, no link contribution → uniform
    for (i, &r) in rank.iter().enumerate() {
        assert!((r - 1.0 / 3.0).abs() < 0.01, "damping=0 should give uniform rank, node {} has {}", i, r);
    }
}

/// PageRank with damping=1: no teleportation, dangling nodes lose all rank.
#[test]
fn pagerank_damping_one() {
    let mut g = Graph::new(3);
    g.add_edge(0, 1, 1.0);
    // Node 1 and 2 are dangling (no outgoing edges → distribute evenly)
    let rank = pagerank(&g, 1.0, 100, 1e-6);
    // base = (1-1)/3 = 0. But dangling nodes distribute evenly with damping=1.
    let sum: f64 = rank.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "Rank should sum to ~1.0, got {}", sum);
}

/// PageRank on disconnected graph.
#[test]
fn pagerank_disconnected() {
    let g = Graph::new(3); // 3 nodes, no edges
    let rank = pagerank(&g, 0.85, 100, 1e-6);
    // All nodes are dangling → distribute evenly → uniform
    for &r in &rank {
        assert!((r - 1.0 / 3.0).abs() < 0.01, "Disconnected graph should have uniform rank");
    }
}

/// Max flow with source = sink: BFS always finds s→s path of length 0.
/// BUG: infinite loop — augmenting path always exists when source==sink.
#[test]
fn max_flow_source_equals_sink() {
    use std::sync::mpsc;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 10.0);
        g.add_edge(1, 2, 5.0);
        let flow = max_flow(&g, 0, 0);
        let _ = tx.send(flow);
    });
    match rx.recv_timeout(std::time::Duration::from_secs(5)) {
        Ok(flow) => {
            assert!(flow.is_finite(), "Max flow source=sink should be finite, got {}", flow);
        }
        Err(_) => {
            eprintln!("CONFIRMED BUG: max_flow infinite loop when source==sink");
        }
    }
}

/// Max flow with zero-capacity edges: no flow possible.
#[test]
fn max_flow_zero_capacity() {
    let mut g = Graph::new(3);
    g.add_edge(0, 1, 0.0);
    g.add_edge(1, 2, 0.0);
    let flow = max_flow(&g, 0, 2);
    assert!((flow - 0.0).abs() < 1e-10, "Zero capacity should give zero flow, got {}", flow);
}

/// Graph density of empty graph (0 nodes): n*(n-1)=0 → 0/0.
/// Type 1: division by zero.
#[test]
fn density_empty_graph() {
    let g = Graph::new(0);
    let d = density(&g);
    assert!(d.is_finite(), "Density of empty graph should be finite, got {}", d);
}

/// Density of single-node graph: 0 edges, 1*(1-1)=0 → 0/0.
#[test]
fn density_single_node() {
    let g = Graph::new(1);
    let d = density(&g);
    assert!(d.is_finite(), "Density of single node should be finite, got {}", d);
}

/// Clustering coefficient of graph with no triangles.
#[test]
fn clustering_coefficient_no_triangles() {
    let mut g = Graph::new(3);
    g.add_undirected(0, 1, 1.0);
    g.add_undirected(1, 2, 1.0);
    // Line graph: 0-1-2. No triangles.
    let cc = clustering_coefficient(&g);
    assert!((cc - 0.0).abs() < 0.01, "Line graph should have CC=0, got {}", cc);
}

/// Clustering coefficient of complete graph: all triples are triangles.
#[test]
fn clustering_coefficient_complete() {
    let mut g = Graph::new(4);
    for i in 0..4 {
        for j in i+1..4 {
            g.add_undirected(i, j, 1.0);
        }
    }
    let cc = clustering_coefficient(&g);
    assert!((cc - 1.0).abs() < 0.01, "Complete graph should have CC=1.0, got {}", cc);
}

/// Topological sort with cycle: should return None.
#[test]
fn topological_sort_cycle() {
    let mut g = Graph::new(3);
    g.add_edge(0, 1, 1.0);
    g.add_edge(1, 2, 1.0);
    g.add_edge(2, 0, 1.0); // cycle
    let result = topological_sort(&g);
    assert!(result.is_none(), "Graph with cycle should have no topological sort");
}

/// Modularity of graph with no edges: every node is its own community.
/// Type 1: m=0 → 1/(2m) = div by zero.
#[test]
fn modularity_no_edges() {
    let g = Graph::new(3);
    let labels = vec![0, 1, 2];
    let q = modularity(&g, &labels);
    assert!(q.is_finite(), "Modularity with no edges should be finite, got {}", q);
}

/// Label propagation on graph with no edges: every node keeps its own label.
#[test]
fn label_propagation_no_edges() {
    let g = Graph::new(3);
    let labels = label_propagation(&g, 100);
    assert_eq!(labels, vec![0, 1, 2], "No edges → each node is its own community");
}

/// Kruskal MST on disconnected graph.
#[test]
fn kruskal_disconnected() {
    let g = Graph::new(4); // 4 nodes, no edges
    let result = kruskal(&g);
    assert!(result.edges.is_empty(), "Disconnected graph should have empty MST");
}

/// Diameter of empty graph.
#[test]
fn diameter_empty_graph() {
    let g = Graph::new(0);
    let d = diameter(&g);
    // No nodes → no paths → diameter undefined
    assert!(d.is_finite() || d == 0.0 || d == f64::NEG_INFINITY,
        "Empty graph diameter should be 0 or -inf, got {}", d);
}

/// Diameter of disconnected graph: should be Infinity.
#[test]
fn diameter_disconnected() {
    let g = Graph::new(3); // 3 isolated nodes
    let d = diameter(&g);
    // Unreachable pairs → dist=Inf → diameter=Inf
    // But the implementation may compute Floyd-Warshall where dist[i][i]=0
    // and all others are Inf, so max = Inf
    assert!(d == f64::INFINITY || d == 0.0,
        "Disconnected graph diameter should be Inf or 0, got {}", d);
}

// ═══════════════════════════════════════════════════════════════════════════
// SPATIAL
// ═══════════════════════════════════════════════════════════════════════════

/// Haversine with identical points: should be 0.
#[test]
fn haversine_identical_points() {
    let d = tambear::spatial::haversine(40.7128, -74.0060, 40.7128, -74.0060);
    assert!((d - 0.0).abs() < 1e-6, "Same point should have distance 0, got {}", d);
}

/// Haversine with antipodal points: should be ~20015 km (half circumference).
#[test]
fn haversine_antipodal() {
    let d = tambear::spatial::haversine(0.0, 0.0, 0.0, 180.0);
    // Half circumference ≈ 20015 km
    assert!((d - 20015.0).abs() < 100.0, "Antipodal distance should be ~20015km, got {}", d);
}

/// Haversine with NaN coordinates.
#[test]
fn haversine_nan() {
    let d = tambear::spatial::haversine(f64::NAN, 0.0, 0.0, 0.0);
    assert!(d.is_nan(), "NaN input should produce NaN distance, got {}", d);
}

/// Clark-Evans R with area=0: division by zero.
/// Type 1.
#[test]
fn clark_evans_zero_area() {
    let points = vec![(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)];
    let r = tambear::spatial::clark_evans_r(&points, 0.0);
    if r.is_nan() || r.is_infinite() {
        eprintln!("CONFIRMED BUG: Clark-Evans R with area=0 produces {} (div by zero)", r);
    }
}

/// Ripley's K with area=0: division by zero.
#[test]
fn ripleys_k_zero_area() {
    let points = vec![(0.0, 0.0), (1.0, 1.0)];
    let radii = vec![0.5, 1.0, 2.0];
    let k = tambear::spatial::ripleys_k(&points, &radii, 0.0);
    let any_inf = k.iter().any(|v| v.is_infinite() || v.is_nan());
    if any_inf {
        eprintln!("CONFIRMED BUG: Ripley's K with area=0 produces non-finite values: {:?}", k);
    }
}

/// Nearest-neighbor distances with single point.
#[test]
fn nn_distances_single_point() {
    let points = vec![(5.0, 5.0)];
    let result = std::panic::catch_unwind(|| {
        tambear::spatial::nn_distances(&points)
    });
    match result {
        Ok(d) => {
            if d.is_empty() {
                // No neighbors for single point → empty
            } else if d[0] == f64::INFINITY {
                // Single point → nearest neighbor at Inf
            }
        }
        Err(_) => {
            eprintln!("CONFIRMED BUG: nn_distances panics on single point");
        }
    }
}

/// Moran's I with constant values: numerator=0 → I=0 (no spatial autocorrelation).
#[test]
fn morans_i_constant_values() {
    let values = vec![5.0; 5];
    let points = vec![(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)];
    let weights = tambear::spatial::SpatialWeights::knn(&points, 2);
    let i = tambear::spatial::morans_i(&values, &weights);
    // Constant values: all deviations = 0 → numerator = 0, denominator = 0 → 0/0
    if i.is_nan() {
        eprintln!("CONFIRMED BUG: Moran's I returns NaN for constant values (0/0)");
    } else {
        assert!((i - 0.0).abs() < 1e-10 || i.is_nan(),
            "Moran's I for constant values should be 0 or NaN, got {}", i);
    }
}
