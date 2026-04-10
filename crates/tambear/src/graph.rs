//! # Family 29 — Graph Algorithms
//!
//! From first principles. Graphs as adjacency lists.
//!
//! ## What lives here
//!
//! **Traversal**: BFS, DFS, topological sort, connected components
//! **Shortest paths**: Dijkstra, Bellman-Ford, Floyd-Warshall, A*
//! **Minimum spanning tree**: Kruskal, Prim
//! **Centrality**: degree, betweenness, closeness, PageRank, eigenvector
//! **Community**: label propagation, modularity, spectral clustering
//! **Flow**: max flow (Ford-Fulkerson/BFS)
//! **Matching**: bipartite matching (Hungarian algorithm)
//!
//! ## Architecture
//!
//! Sparse adjacency list representation. Nodes are 0-indexed usize.
//! Edge weights are f64. Directed by default; undirected = add both directions.
//!
//! ## MSR insight
//!
//! The adjacency matrix IS the graph. Its eigenvalues (spectral properties)
//! are sufficient statistics for many graph invariants — connectivity,
//! clustering, community structure. PageRank is a fixed point of a
//! stochastic matrix — it's the MSR of "importance."

use std::collections::{BinaryHeap, HashMap, HashSet, VecDeque};
use std::cmp::Ordering;

/// Weighted directed edge.
#[derive(Debug, Clone, Copy)]
pub struct Edge {
    pub to: usize,
    pub weight: f64,
}

/// Sparse directed graph (adjacency list).
#[derive(Debug, Clone)]
pub struct Graph {
    /// Adjacency list: adj[u] = list of (to, weight) edges from u.
    pub adj: Vec<Vec<Edge>>,
    pub n_nodes: usize,
}

impl Graph {
    /// Create an empty graph with n nodes.
    pub fn new(n: usize) -> Self {
        Graph { adj: vec![vec![]; n], n_nodes: n }
    }

    /// Add a directed edge u → v with weight w.
    pub fn add_edge(&mut self, u: usize, v: usize, w: f64) {
        self.adj[u].push(Edge { to: v, weight: w });
    }

    /// Add an undirected edge u ↔ v with weight w.
    pub fn add_undirected(&mut self, u: usize, v: usize, w: f64) {
        self.adj[u].push(Edge { to: v, weight: w });
        self.adj[v].push(Edge { to: u, weight: w });
    }

    /// Number of edges.
    pub fn n_edges(&self) -> usize {
        self.adj.iter().map(|e| e.len()).sum()
    }

    /// Out-degree of node u.
    pub fn out_degree(&self, u: usize) -> usize {
        self.adj[u].len()
    }

    /// Create from an edge list: (u, v, weight).
    pub fn from_edges(n: usize, edges: &[(usize, usize, f64)]) -> Self {
        let mut g = Graph::new(n);
        for &(u, v, w) in edges {
            g.add_edge(u, v, w);
        }
        g
    }

    /// Create an undirected graph from an edge list.
    pub fn from_undirected_edges(n: usize, edges: &[(usize, usize, f64)]) -> Self {
        let mut g = Graph::new(n);
        for &(u, v, w) in edges {
            g.add_undirected(u, v, w);
        }
        g
    }
}

// ─── Traversal ──────────────────────────────────────────────────────

/// BFS from a source node. Returns (distances, parents).
///
/// Distance = number of hops (unweighted). Parent[i] = predecessor on shortest path.
pub fn bfs(g: &Graph, source: usize) -> (Vec<i64>, Vec<Option<usize>>) {
    let n = g.n_nodes;
    let mut dist = vec![-1i64; n];
    let mut parent = vec![None; n];
    let mut queue = VecDeque::new();
    dist[source] = 0;
    queue.push_back(source);
    while let Some(u) = queue.pop_front() {
        for e in &g.adj[u] {
            if dist[e.to] == -1 {
                dist[e.to] = dist[u] + 1;
                parent[e.to] = Some(u);
                queue.push_back(e.to);
            }
        }
    }
    (dist, parent)
}

/// DFS from a source node. Returns visit order.
pub fn dfs(g: &Graph, source: usize) -> Vec<usize> {
    let n = g.n_nodes;
    let mut visited = vec![false; n];
    let mut order = Vec::new();
    let mut stack = vec![source];
    while let Some(u) = stack.pop() {
        if visited[u] { continue; }
        visited[u] = true;
        order.push(u);
        for e in g.adj[u].iter().rev() {
            if !visited[e.to] {
                stack.push(e.to);
            }
        }
    }
    order
}

/// Topological sort (Kahn's algorithm).
///
/// Returns None if the graph has a cycle.
pub fn topological_sort(g: &Graph) -> Option<Vec<usize>> {
    let n = g.n_nodes;
    let mut in_degree = vec![0usize; n];
    for u in 0..n {
        for e in &g.adj[u] {
            in_degree[e.to] += 1;
        }
    }
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
    let mut order = Vec::with_capacity(n);
    while let Some(u) = queue.pop_front() {
        order.push(u);
        for e in &g.adj[u] {
            in_degree[e.to] -= 1;
            if in_degree[e.to] == 0 {
                queue.push_back(e.to);
            }
        }
    }
    if order.len() == n { Some(order) } else { None }
}

/// Connected components (undirected graph).
///
/// Returns a vector where component[i] = component label for node i.
pub fn connected_components(g: &Graph) -> Vec<usize> {
    let n = g.n_nodes;
    let mut component = vec![usize::MAX; n];
    let mut label = 0;
    for start in 0..n {
        if component[start] != usize::MAX { continue; }
        let mut queue = VecDeque::new();
        queue.push_back(start);
        component[start] = label;
        while let Some(u) = queue.pop_front() {
            for e in &g.adj[u] {
                if component[e.to] == usize::MAX {
                    component[e.to] = label;
                    queue.push_back(e.to);
                }
            }
        }
        label += 1;
    }
    component
}

// ─── Shortest Paths ─────────────────────────────────────────────────

/// State for Dijkstra priority queue.
#[derive(Debug, Clone, Copy)]
struct DijkstraState {
    cost: f64,
    node: usize,
}

impl PartialEq for DijkstraState {
    fn eq(&self, other: &Self) -> bool { self.cost == other.cost }
}
impl Eq for DijkstraState {}
impl PartialOrd for DijkstraState {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for DijkstraState {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reversed for min-heap
        other.cost.total_cmp(&self.cost)
    }
}

/// Dijkstra's shortest path algorithm.
///
/// Returns `(distances, parents)`. All edge weights **must be non-negative**;
/// panics if any negative weight is detected. Use `bellman_ford` for graphs
/// with negative weights (it also detects negative cycles).
pub fn dijkstra(g: &Graph, source: usize) -> (Vec<f64>, Vec<Option<usize>>) {
    // Validate: Dijkstra is incorrect on negative weights — detect early.
    for u in 0..g.n_nodes {
        for e in &g.adj[u] {
            assert!(
                e.weight >= 0.0,
                "dijkstra: negative edge weight {:.6} from {} to {}; use bellman_ford instead",
                e.weight, u, e.to,
            );
        }
    }
    let n = g.n_nodes;
    let mut dist = vec![f64::INFINITY; n];
    let mut parent = vec![None; n];
    let mut heap = BinaryHeap::new();

    dist[source] = 0.0;
    heap.push(DijkstraState { cost: 0.0, node: source });

    while let Some(DijkstraState { cost, node: u }) = heap.pop() {
        if cost > dist[u] { continue; }
        for e in &g.adj[u] {
            let new_cost = dist[u] + e.weight;
            if new_cost < dist[e.to] {
                dist[e.to] = new_cost;
                parent[e.to] = Some(u);
                heap.push(DijkstraState { cost: new_cost, node: e.to });
            }
        }
    }
    (dist, parent)
}

/// Bellman-Ford shortest path (handles negative edges).
///
/// Returns None if a negative cycle is detected.
pub fn bellman_ford(g: &Graph, source: usize) -> Option<(Vec<f64>, Vec<Option<usize>>)> {
    let n = g.n_nodes;
    let mut dist = vec![f64::INFINITY; n];
    let mut parent = vec![None; n];
    dist[source] = 0.0;

    for _ in 0..n - 1 {
        for u in 0..n {
            if dist[u] == f64::INFINITY { continue; }
            for e in &g.adj[u] {
                if dist[u] + e.weight < dist[e.to] {
                    dist[e.to] = dist[u] + e.weight;
                    parent[e.to] = Some(u);
                }
            }
        }
    }

    // Check for negative cycles
    for u in 0..n {
        if dist[u] == f64::INFINITY { continue; }
        for e in &g.adj[u] {
            if dist[u] + e.weight < dist[e.to] - 1e-14 {
                return None; // negative cycle
            }
        }
    }

    Some((dist, parent))
}

/// Floyd-Warshall all-pairs shortest paths.
///
/// Returns n×n distance matrix. dist[i][j] = shortest distance from i to j.
pub fn floyd_warshall(g: &Graph) -> Vec<Vec<f64>> {
    let n = g.n_nodes;
    let mut dist = vec![vec![f64::INFINITY; n]; n];
    for i in 0..n { dist[i][i] = 0.0; }
    for u in 0..n {
        for e in &g.adj[u] {
            dist[u][e.to] = dist[u][e.to].min(e.weight);
        }
    }
    for k in 0..n {
        for i in 0..n {
            for j in 0..n {
                if dist[i][k] + dist[k][j] < dist[i][j] {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
    dist
}

/// Reconstruct shortest path from source to target using parent array.
pub fn reconstruct_path(parent: &[Option<usize>], source: usize, target: usize) -> Vec<usize> {
    let mut path = Vec::new();
    let mut current = target;
    while current != source {
        path.push(current);
        match parent[current] {
            Some(p) => current = p,
            None => return vec![], // no path
        }
    }
    path.push(source);
    path.reverse();
    path
}

// ─── Minimum Spanning Tree ──────────────────────────────────────────

/// Edge for sorting in Kruskal.
#[derive(Debug, Clone)]
struct WeightedEdge {
    u: usize,
    v: usize,
    w: f64,
}

/// Union-Find (Disjoint Set Union).
struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        UnionFind { parent: (0..n).collect(), rank: vec![0; n] }
    }
    fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }
    fn union(&mut self, x: usize, y: usize) -> bool {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry { return false; }
        if self.rank[rx] < self.rank[ry] {
            self.parent[rx] = ry;
        } else if self.rank[rx] > self.rank[ry] {
            self.parent[ry] = rx;
        } else {
            self.parent[ry] = rx;
            self.rank[rx] += 1;
        }
        true
    }
}

/// MST result.
#[derive(Debug, Clone)]
pub struct MstResult {
    /// Edges in the MST as (u, v, weight).
    pub edges: Vec<(usize, usize, f64)>,
    /// Total weight of the MST.
    pub total_weight: f64,
}

/// Kruskal's MST algorithm.
///
/// Returns the minimum spanning tree edges and total weight.
/// Graph is treated as undirected.
pub fn kruskal(g: &Graph) -> MstResult {
    let n = g.n_nodes;
    let mut edges: Vec<WeightedEdge> = Vec::new();
    let mut seen = HashSet::new();
    for u in 0..n {
        for e in &g.adj[u] {
            let key = if u < e.to { (u, e.to) } else { (e.to, u) };
            if seen.insert(key) {
                edges.push(WeightedEdge { u, v: e.to, w: e.weight });
            }
        }
    }
    edges.sort_by(|a, b| a.w.total_cmp(&b.w));

    let mut uf = UnionFind::new(n);
    let mut mst_edges = Vec::new();
    let mut total = 0.0;
    for e in &edges {
        if uf.union(e.u, e.v) {
            mst_edges.push((e.u, e.v, e.w));
            total += e.w;
            if mst_edges.len() == n - 1 { break; }
        }
    }
    MstResult { edges: mst_edges, total_weight: total }
}

/// Prim's MST algorithm.
pub fn prim(g: &Graph) -> MstResult {
    let n = g.n_nodes;
    if n == 0 { return MstResult { edges: vec![], total_weight: 0.0 }; }
    let mut in_mst = vec![false; n];
    let mut mst_edges = Vec::new();
    let mut total = 0.0;

    // Store (weight, from, to) in heap — use negative weight for min-heap
    let mut heap: BinaryHeap<(i64, usize, usize)> = BinaryHeap::new();

    in_mst[0] = true;
    for e in &g.adj[0] {
        // Encode f64 weight as negative i64 bits for ordering
        heap.push((neg_weight_key(e.weight), 0, e.to));
    }

    while let Some((neg_key, u, v)) = heap.pop() {
        if in_mst[v] { continue; }
        let w = key_to_weight(neg_key);
        in_mst[v] = true;
        mst_edges.push((u, v, w));
        total += w;
        for e in &g.adj[v] {
            if !in_mst[e.to] {
                heap.push((neg_weight_key(e.weight), v, e.to));
            }
        }
    }
    MstResult { edges: mst_edges, total_weight: total }
}

/// Encode f64 weight as i64 key for min-heap via max-heap.
/// Uses IEEE 754 bit-level total order, then inverts for min-heap.
/// Handles negative weights, zero, and NaN correctly.
fn neg_weight_key(w: f64) -> i64 {
    let bits = w.to_bits();
    // IEEE 754 → sortable u64: negative floats flip all bits, positive flip sign bit
    let sorted = if bits >> 63 == 1 { !bits } else { bits ^ (1u64 << 63) };
    // Invert for min-heap (smallest weight → largest key), then map u64→i64 preserving order
    (!sorted ^ (1u64 << 63)) as i64
}

fn key_to_weight(k: i64) -> f64 {
    let sorted = !(k as u64 ^ (1u64 << 63));
    let bits = if sorted >> 63 == 1 {
        sorted ^ (1u64 << 63) // was non-negative float
    } else {
        !sorted // was negative float
    };
    f64::from_bits(bits)
}

// ─── Centrality ─────────────────────────────────────────────────────

/// Degree centrality (normalized).
pub fn degree_centrality(g: &Graph) -> Vec<f64> {
    let n = g.n_nodes;
    if n <= 1 { return vec![0.0; n]; }
    g.adj.iter().map(|edges| edges.len() as f64 / (n - 1) as f64).collect()
}

/// Closeness centrality.
///
/// C(v) = (n-1) / Σ d(v, u) for all reachable u.
pub fn closeness_centrality(g: &Graph) -> Vec<f64> {
    let n = g.n_nodes;
    let mut centrality = vec![0.0; n];
    for v in 0..n {
        let (dist, _) = dijkstra(g, v);
        let total: f64 = dist.iter().filter(|&&d| d < f64::INFINITY && d > 0.0).sum();
        let reachable = dist.iter().filter(|&&d| d < f64::INFINITY && d > 0.0).count();
        if reachable > 0 && total > 0.0 {
            centrality[v] = reachable as f64 / total;
        }
    }
    centrality
}

/// PageRank (power iteration).
///
/// `damping`: damping factor (typically 0.85).
/// `max_iter`: maximum iterations.
/// `tol`: convergence tolerance.
pub fn pagerank(g: &Graph, damping: f64, max_iter: usize, tol: f64) -> Vec<f64> {
    let n = g.n_nodes;
    if n == 0 { return vec![]; }
    let mut rank = vec![1.0 / n as f64; n];
    let base = (1.0 - damping) / n as f64;

    for _ in 0..max_iter {
        let mut new_rank = vec![base; n];
        for u in 0..n {
            if g.adj[u].is_empty() {
                // Dangling node: distribute evenly
                let share = damping * rank[u] / n as f64;
                for r in new_rank.iter_mut() { *r += share; }
            } else {
                let share = damping * rank[u] / g.adj[u].len() as f64;
                for e in &g.adj[u] {
                    new_rank[e.to] += share;
                }
            }
        }
        let diff: f64 = rank.iter().zip(new_rank.iter()).map(|(a, b)| (a - b).abs()).sum();
        rank = new_rank;
        if diff < tol { break; }
    }
    rank
}

// ─── Community Detection ────────────────────────────────────────────

/// Label propagation community detection.
///
/// Each node starts with its own label; iteratively adopts the most common
/// label among its neighbors. Simple, fast, non-deterministic.
pub fn label_propagation(g: &Graph, max_iter: usize) -> Vec<usize> {
    let n = g.n_nodes;
    let mut labels: Vec<usize> = (0..n).collect();

    // Simple deterministic order (not shuffled — deterministic for tests)
    for _ in 0..max_iter {
        let mut changed = false;
        for u in 0..n {
            if g.adj[u].is_empty() { continue; }
            // Count neighbor labels
            let mut counts: HashMap<usize, usize> = HashMap::new();
            for e in &g.adj[u] {
                *counts.entry(labels[e.to]).or_insert(0) += 1;
            }
            let best = counts.into_iter().max_by_key(|&(_, c)| c).unwrap().0;
            if best != labels[u] {
                labels[u] = best;
                changed = true;
            }
        }
        if !changed { break; }
    }
    labels
}

/// Graph modularity for a given partition.
///
/// Q = (1/2m) Σ [A_ij - k_i k_j / 2m] δ(c_i, c_j)
/// Higher modularity = better community structure.
pub fn modularity(g: &Graph, labels: &[usize]) -> f64 {
    let n = g.n_nodes;
    let m2: f64 = g.adj.iter().map(|e| e.len() as f64).sum(); // 2m for directed
    if m2 < 1.0 { return 0.0; }

    let degree: Vec<f64> = g.adj.iter().map(|e| e.len() as f64).collect();
    let mut q = 0.0;
    for u in 0..n {
        for e in &g.adj[u] {
            if labels[u] == labels[e.to] {
                q += 1.0 - degree[u] * degree[e.to] / m2;
            }
        }
    }
    q / m2
}

// ─── Max Flow ───────────────────────────────────────────────────────

/// Max flow from source to sink (Edmonds-Karp / BFS-based Ford-Fulkerson).
///
/// Returns the maximum flow value.
pub fn max_flow(g: &Graph, source: usize, sink: usize) -> f64 {
    if source == sink { return 0.0; }
    let n = g.n_nodes;
    // Build capacity matrix
    let mut cap = vec![vec![0.0; n]; n];
    for u in 0..n {
        for e in &g.adj[u] {
            cap[u][e.to] += e.weight;
        }
    }

    let mut total_flow = 0.0;

    loop {
        // BFS to find augmenting path
        let mut parent = vec![None; n];
        let mut visited = vec![false; n];
        visited[source] = true;
        let mut queue = VecDeque::new();
        queue.push_back(source);

        while let Some(u) = queue.pop_front() {
            if u == sink { break; }
            for v in 0..n {
                if !visited[v] && cap[u][v] > 1e-14 {
                    visited[v] = true;
                    parent[v] = Some(u);
                    queue.push_back(v);
                }
            }
        }

        if !visited[sink] { break; } // no augmenting path

        // Find bottleneck
        let mut flow = f64::INFINITY;
        let mut v = sink;
        while v != source {
            let u = parent[v].unwrap();
            flow = flow.min(cap[u][v]);
            v = u;
        }

        // Update residual capacities
        v = sink;
        while v != source {
            let u = parent[v].unwrap();
            cap[u][v] -= flow;
            cap[v][u] += flow;
            v = u;
        }

        total_flow += flow;
    }

    total_flow
}

// ─── Graph metrics ──────────────────────────────────────────────────

/// Graph diameter (longest shortest path between any pair).
pub fn diameter(g: &Graph) -> f64 {
    let dist = floyd_warshall(g);
    let mut max_dist = 0.0;
    for row in &dist {
        for &d in row {
            if d < f64::INFINITY && d > max_dist {
                max_dist = d;
            }
        }
    }
    max_dist
}

/// Graph density: |E| / (|V| * (|V| - 1)) for directed.
pub fn density(g: &Graph) -> f64 {
    let n = g.n_nodes;
    if n <= 1 { return 0.0; }
    g.n_edges() as f64 / (n * (n - 1)) as f64
}

/// Clustering coefficient (global).
///
/// Fraction of closed triplets over all triplets.
pub fn clustering_coefficient(g: &Graph) -> f64 {
    let n = g.n_nodes;
    let mut triangles = 0u64;
    let mut triplets = 0u64;

    for u in 0..n {
        let neighbors: HashSet<usize> = g.adj[u].iter().map(|e| e.to).collect();
        let k = neighbors.len();
        if k < 2 { continue; }
        triplets += (k * (k - 1) / 2) as u64;
        // Count triangles: how many neighbor pairs are also connected?
        let neigh_vec: Vec<usize> = neighbors.iter().copied().collect();
        for i in 0..neigh_vec.len() {
            for j in i + 1..neigh_vec.len() {
                if g.adj[neigh_vec[i]].iter().any(|e| e.to == neigh_vec[j]) {
                    triangles += 1;
                }
            }
        }
    }
    if triplets == 0 { 0.0 } else { triangles as f64 / triplets as f64 }
}

// ─── Dense matrix graph primitives ──────────────────────────────────

/// Euclidean pairwise distance matrix for a set of points.
///
/// Points are given as a row-major matrix with `n` rows and `d` columns.
/// Returns a flat `n×n` distance matrix (row-major), symmetric with zero diagonal.
///
/// # Arguments
///
/// - `mat`: row-major point matrix of shape n×d.
/// - `n`: number of points (rows).
/// - `d`: dimensionality (columns).
///
/// # Applications
///
/// Phase-space distance for correlation dimension and Lyapunov exponents,
/// k-NN graph construction, recurrence plots, spectral embedding, diffusion maps.
///
/// # Complexity
///
/// O(n²·d) time, O(n²) space. For n > ~1000 consider approximate methods.
pub fn pairwise_dists(mat: &[f64], n: usize, d: usize) -> Vec<f64> {
    let mut dists = vec![0.0_f64; n * n];
    for a in 0..n {
        for b in (a + 1)..n {
            let mut s = 0.0_f64;
            for k in 0..d {
                let diff = mat[a * d + k] - mat[b * d + k];
                s += diff * diff;
            }
            let dist = s.sqrt();
            dists[a * n + b] = dist;
            dists[b * n + a] = dist;
        }
    }
    dists
}

/// Build a symmetric k-NN Gaussian kernel adjacency matrix from pairwise distances.
///
/// For each point, finds its k nearest neighbors and assigns a Gaussian weight
/// `exp(-dist² / (2·σ²))` where σ is the distance to the k-th neighbor.
/// The adjacency is symmetrized: if a is in b's k-NN or vice versa, both get the weight.
///
/// # Arguments
///
/// - `dists`: flat n×n pairwise distance matrix (from `pairwise_dists`).
/// - `n`: number of points.
/// - `k`: number of nearest neighbors per point.
///
/// # Applications
///
/// Spectral embedding, diffusion maps, graph Laplacian construction,
/// manifold learning, semi-supervised classification.
pub fn knn_adjacency(dists: &[f64], n: usize, k: usize) -> Vec<f64> {
    let mut adj = vec![0.0_f64; n * n];
    for a in 0..n {
        let mut row: Vec<(usize, f64)> = (0..n)
            .filter(|&b| b != a)
            .map(|b| (b, dists[a * n + b]))
            .collect();
        row.sort_unstable_by(|x, y| x.1.partial_cmp(&y.1).unwrap_or(std::cmp::Ordering::Equal));
        let sigma = row.get(k.min(row.len()).saturating_sub(1))
            .map(|r| r.1)
            .unwrap_or(1.0)
            .max(1e-30);
        for &(b, dist) in row.iter().take(k) {
            let w = (-dist * dist / (2.0 * sigma * sigma)).exp();
            if w > adj[a * n + b] {
                adj[a * n + b] = w;
                adj[b * n + a] = w;
            }
        }
    }
    adj
}

/// Unnormalized graph Laplacian L = D − A.
///
/// Given a weighted adjacency matrix A (flat n×n, row-major), computes:
/// - Degree matrix D: diagonal with D[i,i] = Σⱼ A[i,j].
/// - Laplacian L = D − A.
///
/// # Returns
///
/// `(laplacian, degrees, max_degree)` where:
/// - `laplacian`: flat n×n Laplacian matrix.
/// - `degrees`: degree vector of length n.
/// - `max_degree`: maximum degree value.
///
/// # Applications
///
/// Spectral clustering (Fiedler vector), spectral embedding, diffusion maps,
/// graph partitioning, effective resistance computation.
///
/// For normalized Laplacian L_sym = D^{-1/2} L D^{-1/2}, scale the off-diagonal
/// entries by 1/sqrt(d[i]*d[j]).
pub fn graph_laplacian(adj: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, f64) {
    let mut degrees = vec![0.0_f64; n];
    for a in 0..n {
        degrees[a] = (0..n).map(|b| adj[a * n + b]).sum();
    }
    let max_deg = degrees.iter().copied().fold(0.0_f64, f64::max);
    let mut lap = vec![0.0_f64; n * n];
    for a in 0..n {
        for b in 0..n {
            lap[a * n + b] = if a == b { degrees[a] } else { -adj[a * n + b] };
        }
    }
    (lap, degrees, max_deg)
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_graph() -> Graph {
        // 0 -- 1 -- 2
        // |         |
        // 3 ------- 4
        let mut g = Graph::new(5);
        g.add_undirected(0, 1, 1.0);
        g.add_undirected(1, 2, 1.0);
        g.add_undirected(0, 3, 1.0);
        g.add_undirected(2, 4, 1.0);
        g.add_undirected(3, 4, 1.0);
        g
    }

    fn weighted_graph() -> Graph {
        // 0 --(1)--> 1 --(2)--> 2
        // |                      ^
        // +-------(10)----------+
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 2.0);
        g.add_edge(0, 2, 10.0);
        g
    }

    // ── BFS ──

    #[test]
    fn bfs_distances() {
        let g = simple_graph();
        let (dist, _) = bfs(&g, 0);
        assert_eq!(dist[0], 0);
        assert_eq!(dist[1], 1);
        assert_eq!(dist[2], 2);
        assert_eq!(dist[3], 1);
        assert_eq!(dist[4], 2);
    }

    // ── DFS ──

    #[test]
    fn dfs_visits_all() {
        let g = simple_graph();
        let order = dfs(&g, 0);
        assert_eq!(order.len(), 5);
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(sorted, vec![0, 1, 2, 3, 4]);
    }

    // ── Topological sort ──

    #[test]
    fn topo_sort_dag() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(0, 2, 1.0);
        g.add_edge(1, 3, 1.0);
        g.add_edge(2, 3, 1.0);
        let order = topological_sort(&g).unwrap();
        assert_eq!(order.len(), 4);
        // 0 must come before 1 and 2; both before 3
        let pos: Vec<usize> = {
            let mut p = vec![0; 4];
            for (i, &v) in order.iter().enumerate() { p[v] = i; }
            p
        };
        assert!(pos[0] < pos[1]);
        assert!(pos[0] < pos[2]);
        assert!(pos[1] < pos[3]);
        assert!(pos[2] < pos[3]);
    }

    #[test]
    fn topo_sort_cycle() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 0, 1.0);
        assert!(topological_sort(&g).is_none());
    }

    // ── Connected components ──

    #[test]
    fn connected_components_test() {
        let mut g = Graph::new(6);
        g.add_undirected(0, 1, 1.0);
        g.add_undirected(1, 2, 1.0);
        g.add_undirected(3, 4, 1.0);
        // Node 5 is isolated
        let comp = connected_components(&g);
        assert_eq!(comp[0], comp[1]);
        assert_eq!(comp[1], comp[2]);
        assert_eq!(comp[3], comp[4]);
        assert_ne!(comp[0], comp[3]);
        assert_ne!(comp[0], comp[5]);
        assert_ne!(comp[3], comp[5]);
    }

    // ── Dijkstra ──

    #[test]
    fn dijkstra_shortest() {
        let g = weighted_graph();
        let (dist, parent) = dijkstra(&g, 0);
        assert!((dist[0] - 0.0).abs() < 1e-10);
        assert!((dist[1] - 1.0).abs() < 1e-10);
        assert!((dist[2] - 3.0).abs() < 1e-10); // 0→1→2 = 3, not 0→2 = 10
        let path = reconstruct_path(&parent, 0, 2);
        assert_eq!(path, vec![0, 1, 2]);
    }

    // ── Bellman-Ford ──

    #[test]
    fn bellman_ford_negative_edges() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 4.0);
        g.add_edge(0, 2, 5.0);
        g.add_edge(1, 2, -3.0); // negative edge
        let (dist, _) = bellman_ford(&g, 0).unwrap();
        assert!((dist[2] - 1.0).abs() < 1e-10); // 0→1→2 = 4 + (-3) = 1
    }

    #[test]
    fn bellman_ford_negative_cycle() {
        let mut g = Graph::new(3);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, -5.0);
        g.add_edge(2, 0, 1.0);
        assert!(bellman_ford(&g, 0).is_none());
    }

    // ── Floyd-Warshall ──

    #[test]
    fn floyd_warshall_all_pairs() {
        let g = weighted_graph();
        let dist = floyd_warshall(&g);
        assert!((dist[0][2] - 3.0).abs() < 1e-10);
        assert!((dist[0][1] - 1.0).abs() < 1e-10);
    }

    // ── MST ──

    #[test]
    fn kruskal_mst() {
        let mut g = Graph::new(4);
        g.add_undirected(0, 1, 1.0);
        g.add_undirected(0, 2, 3.0);
        g.add_undirected(1, 2, 2.0);
        g.add_undirected(2, 3, 4.0);
        let mst = kruskal(&g);
        assert_eq!(mst.edges.len(), 3);
        assert!((mst.total_weight - 7.0).abs() < 1e-10); // 1 + 2 + 4
    }

    #[test]
    fn prim_mst() {
        let mut g = Graph::new(4);
        g.add_undirected(0, 1, 1.0);
        g.add_undirected(0, 2, 3.0);
        g.add_undirected(1, 2, 2.0);
        g.add_undirected(2, 3, 4.0);
        let mst = prim(&g);
        assert_eq!(mst.edges.len(), 3);
        assert!((mst.total_weight - 7.0).abs() < 1e-10);
    }

    #[test]
    fn kruskal_prim_agree() {
        let g = simple_graph();
        let k = kruskal(&g);
        let p = prim(&g);
        assert!((k.total_weight - p.total_weight).abs() < 1e-10,
            "kruskal {} vs prim {}", k.total_weight, p.total_weight);
    }

    #[test]
    fn prim_negative_weights() {
        // Triangle: edges -3.0, -1.0, 2.0. MST picks -3.0 and -1.0 (total -4.0)
        let mut g = Graph::new(3);
        g.add_undirected(0, 1, -3.0);
        g.add_undirected(1, 2, -1.0);
        g.add_undirected(0, 2, 2.0);
        let mst = prim(&g);
        assert_eq!(mst.edges.len(), 2);
        assert!((mst.total_weight - (-4.0)).abs() < 1e-10,
            "total={} expected -4.0", mst.total_weight);
    }

    #[test]
    fn prim_mixed_negative_positive() {
        // 4 nodes, mix of negative and positive weights
        let mut g = Graph::new(4);
        g.add_undirected(0, 1, -5.0);
        g.add_undirected(0, 2, 1.0);
        g.add_undirected(1, 2, -2.0);
        g.add_undirected(1, 3, 3.0);
        g.add_undirected(2, 3, -1.0);
        let mst = prim(&g);
        assert_eq!(mst.edges.len(), 3);
        // MST should pick: -5.0 (0-1), -2.0 (1-2), -1.0 (2-3) = total -8.0
        assert!((mst.total_weight - (-8.0)).abs() < 1e-10,
            "total={} expected -8.0", mst.total_weight);
    }

    // ── PageRank ──

    #[test]
    fn pagerank_sum_to_one() {
        let g = simple_graph();
        let pr = pagerank(&g, 0.85, 100, 1e-10);
        let sum: f64 = pr.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "sum = {}", sum);
    }

    #[test]
    fn pagerank_hub_highest() {
        // Star graph: node 0 connected to all others
        let mut g = Graph::new(5);
        for i in 1..5 {
            g.add_edge(i, 0, 1.0);
            g.add_edge(0, i, 1.0);
        }
        let pr = pagerank(&g, 0.85, 100, 1e-10);
        // Hub (node 0) should have highest rank
        assert!(pr[0] > pr[1], "hub {} <= spoke {}", pr[0], pr[1]);
    }

    // ── Community detection ──

    #[test]
    fn label_propagation_two_cliques() {
        // Two triangles connected by one edge
        let mut g = Graph::new(6);
        // Clique 1: 0-1-2
        g.add_undirected(0, 1, 1.0);
        g.add_undirected(1, 2, 1.0);
        g.add_undirected(0, 2, 1.0);
        // Clique 2: 3-4-5
        g.add_undirected(3, 4, 1.0);
        g.add_undirected(4, 5, 1.0);
        g.add_undirected(3, 5, 1.0);
        // Bridge
        g.add_undirected(2, 3, 1.0);
        let labels = label_propagation(&g, 100);
        // Nodes in same clique should have same label
        assert_eq!(labels[0], labels[1]);
        assert_eq!(labels[1], labels[2]);
        assert_eq!(labels[3], labels[4]);
        assert_eq!(labels[4], labels[5]);
    }

    // ── Max flow ──

    #[test]
    fn max_flow_simple() {
        // Simple flow network
        //   0 --10--> 1 --5--> 3
        //   0 --8---> 2 --7--> 3
        //             1 --3--> 2
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 10.0);
        g.add_edge(0, 2, 8.0);
        g.add_edge(1, 3, 5.0);
        g.add_edge(2, 3, 7.0);
        g.add_edge(1, 2, 3.0);
        let flow = max_flow(&g, 0, 3);
        assert!((flow - 12.0).abs() < 1e-10, "max flow = {}", flow);
    }

    // ── Metrics ──

    #[test]
    fn graph_density() {
        let mut g = Graph::new(4);
        g.add_edge(0, 1, 1.0);
        g.add_edge(1, 2, 1.0);
        g.add_edge(2, 3, 1.0);
        g.add_edge(3, 0, 1.0);
        let d = density(&g);
        assert!((d - 4.0 / 12.0).abs() < 1e-10, "density = {}", d);
    }

    #[test]
    fn clustering_triangle() {
        // Complete graph K3: clustering coefficient = 1
        let mut g = Graph::new(3);
        g.add_undirected(0, 1, 1.0);
        g.add_undirected(1, 2, 1.0);
        g.add_undirected(0, 2, 1.0);
        let cc = clustering_coefficient(&g);
        assert!((cc - 1.0).abs() < 1e-10, "cc = {}", cc);
    }

    #[test]
    fn diameter_test() {
        let g = simple_graph();
        let d = diameter(&g);
        assert!((d - 2.0).abs() < 1e-10, "diameter = {}", d);
    }

    // ── Edge cases ──

    #[test]
    fn empty_graph() {
        let g = Graph::new(0);
        assert_eq!(g.n_edges(), 0);
        assert_eq!(pagerank(&g, 0.85, 10, 1e-6).len(), 0);
    }

    #[test]
    fn single_node() {
        let g = Graph::new(1);
        let (dist, _) = bfs(&g, 0);
        assert_eq!(dist[0], 0);
    }
}
