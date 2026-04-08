//! # Family 27 — Topological Data Analysis
//!
//! Persistent homology (H₀ via union-find, H₁ via boundary matrix),
//! persistence diagrams, Rips complex, topological features.
//!
//! ## Architecture
//!
//! H₀ = union-find over filtered edges (Kingdom B: sequential merge).
//! H₁ = boundary matrix reduction (Kingdom B: column operations).
//! Persistence diagram = pairs of (birth, death) values.
//! Features for ML = vectorizations of persistence diagrams.

// ═══════════════════════════════════════════════════════════════════════════
// Union-Find for H₀
// ═══════════════════════════════════════════════════════════════════════════

struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
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

// ═══════════════════════════════════════════════════════════════════════════
// Persistence pairs
// ═══════════════════════════════════════════════════════════════════════════

/// A birth-death pair in a persistence diagram.
#[derive(Debug, Clone, Copy)]
pub struct PersistencePair {
    /// Homology dimension (0 = connected components, 1 = loops).
    pub dimension: usize,
    /// Filtration value at birth.
    pub birth: f64,
    /// Filtration value at death (f64::INFINITY = never dies).
    pub death: f64,
}

impl PersistencePair {
    /// Persistence = death - birth.
    pub fn persistence(&self) -> f64 {
        self.death - self.birth
    }
}

/// Persistence diagram: collection of birth-death pairs.
#[derive(Debug, Clone)]
pub struct PersistenceDiagram {
    pub pairs: Vec<PersistencePair>,
}

impl PersistenceDiagram {
    /// Filter to a specific dimension.
    pub fn dimension(&self, dim: usize) -> Vec<PersistencePair> {
        self.pairs.iter().filter(|p| p.dimension == dim).copied().collect()
    }

    /// Total persistence in given dimension.
    pub fn total_persistence(&self, dim: usize) -> f64 {
        self.dimension(dim).iter()
            .filter(|p| p.death.is_finite())
            .map(|p| p.persistence())
            .sum()
    }

    /// Maximum persistence in given dimension.
    pub fn max_persistence(&self, dim: usize) -> f64 {
        self.dimension(dim).iter()
            .filter(|p| p.death.is_finite())
            .map(|p| p.persistence())
            .fold(0.0, f64::max)
    }

    /// Number of features with persistence > threshold.
    pub fn count_above(&self, dim: usize, threshold: f64) -> usize {
        self.dimension(dim).iter()
            .filter(|p| p.death.is_finite() && p.persistence() > threshold)
            .count()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Vietoris-Rips H₀ (connected components via union-find)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute H₀ persistence (connected components) from pairwise distances.
/// `dist`: n×n symmetric distance matrix (row-major).
pub fn rips_h0(dist: &[f64], n: usize) -> PersistenceDiagram {
    assert_eq!(dist.len(), n * n);
    if n == 0 {
        return PersistenceDiagram { pairs: vec![] };
    }
    if n == 1 {
        return PersistenceDiagram {
            pairs: vec![PersistencePair { dimension: 0, birth: 0.0, death: f64::INFINITY }],
        };
    }

    // Collect all edges with distances, sort by distance
    let mut edges: Vec<(f64, usize, usize)> = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            edges.push((dist[i * n + j], i, j));
        }
    }
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut uf = UnionFind::new(n);
    let mut pairs = Vec::new();

    // All points born at filtration 0
    // When two components merge, the younger one dies
    let mut birth = vec![0.0; n];

    for (d, i, j) in &edges {
        let ri = uf.find(*i);
        let rj = uf.find(*j);
        if ri != rj {
            // Component with later birth dies (convention: higher index dies)
            let (survivor, dying) = if birth[ri] <= birth[rj] { (ri, rj) } else { (rj, ri) };
            pairs.push(PersistencePair {
                dimension: 0,
                birth: birth[dying],
                death: *d,
            });
            uf.union(*i, *j);
            // Propagate birth time to the merged root
            let new_root = uf.find(*i);
            birth[new_root] = birth[survivor];
        }
    }

    // One component survives forever
    pairs.push(PersistencePair {
        dimension: 0,
        birth: 0.0,
        death: f64::INFINITY,
    });

    PersistenceDiagram { pairs }
}

// ═══════════════════════════════════════════════════════════════════════════
// Rips H₁ (1-cycles) via boundary matrix reduction
// ═══════════════════════════════════════════════════════════════════════════

/// Compute H₁ persistence (loops) from pairwise distances.
/// Uses simplified greedy cycle-killing (NOT full boundary matrix reduction).
/// `max_edge`: maximum edge length to include (controls complex size).
///
/// **Limitation**: The greedy triangle-kills-cycle matching can produce incorrect
/// persistence pairs for H₁ in non-trivial complexes. For exact H₁, a proper
/// column reduction algorithm on the full boundary matrix is needed. H₀ output
/// from this function is exact; H₁ is approximate.
pub fn rips_h1(dist: &[f64], n: usize, max_edge: f64) -> PersistenceDiagram {
    assert_eq!(dist.len(), n * n);
    if n < 2 {
        return PersistenceDiagram { pairs: vec![] };
    }

    // Build filtered edge list
    let mut edges: Vec<(f64, usize, usize)> = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let d = dist[i * n + j];
            if d <= max_edge { edges.push((d, i, j)); }
        }
    }
    edges.sort_by(|a, b| a.0.total_cmp(&b.0));

    let n_edges = edges.len();

    // Build triangles (2-simplices) and their filtration values
    let mut triangles: Vec<(f64, usize, usize, usize)> = Vec::new();
    // Edge lookup for quick boundary computation
    let mut edge_idx = std::collections::HashMap::new();
    for (idx, &(_, i, j)) in edges.iter().enumerate() {
        edge_idx.insert((i, j), idx);
    }

    for i in 0..n {
        for j in (i + 1)..n {
            if dist[i * n + j] > max_edge { continue; }
            for k in (j + 1)..n {
                let d_ij = dist[i * n + j];
                let d_ik = dist[i * n + k];
                let d_jk = dist[j * n + k];
                if d_ik > max_edge || d_jk > max_edge { continue; }
                let filt = d_ij.max(d_ik).max(d_jk);
                triangles.push((filt, i, j, k));
            }
        }
    }
    triangles.sort_by(|a, b| a.0.total_cmp(&b.0));

    // Boundary matrix reduction (standard persistence algorithm)
    // Simplices in filtration order: vertices, then edges, then triangles
    // We only need to track which edges are "killed" by triangles for H₁

    // For H₁: an edge that creates a cycle when added (already connected components)
    // and is not the boundary of any triangle is an H₁ feature.

    // Simplified: use union-find for H₀ + track edge births for H₁
    let mut uf = UnionFind::new(n);
    let mut h0_pairs = Vec::new();
    let mut cycle_edges = Vec::new(); // edges that create cycles (potential H₁ births)

    for &(d, i, j) in &edges {
        if uf.find(i) == uf.find(j) {
            // This edge creates a cycle → H₁ birth at this filtration value
            cycle_edges.push((d, i, j));
        } else {
            uf.union(i, j);
            h0_pairs.push(PersistencePair { dimension: 0, birth: 0.0, death: d });
        }
    }
    h0_pairs.push(PersistencePair { dimension: 0, birth: 0.0, death: f64::INFINITY });

    // For each triangle, it potentially kills an H₁ cycle
    // Simple matching: each triangle kills the cycle created by its latest edge
    let mut killed: Vec<bool> = vec![false; cycle_edges.len()];

    for &(filt, i, j, k) in &triangles {
        // The three edges of the triangle
        let mut tri_edges = vec![(i, j), (i, k), (j, k)];
        // Find which cycle edge this triangle could kill (the latest-born cycle edge in the boundary)
        for (ci, &(cd, ci_a, ci_b)) in cycle_edges.iter().enumerate() {
            if killed[ci] { continue; }
            if tri_edges.contains(&(ci_a, ci_b)) {
                killed[ci] = true;
                h0_pairs.push(PersistencePair { dimension: 1, birth: cd, death: filt });
                break;
            }
        }
    }

    // Surviving H₁ cycles
    for (ci, &(cd, _, _)) in cycle_edges.iter().enumerate() {
        if !killed[ci] {
            h0_pairs.push(PersistencePair { dimension: 1, birth: cd, death: f64::INFINITY });
        }
    }

    PersistenceDiagram { pairs: h0_pairs }
}

// ═══════════════════════════════════════════════════════════════════════════
// Persistence diagram distances
// ═══════════════════════════════════════════════════════════════════════════

/// Bottleneck distance between two persistence diagrams (same dimension).
/// Approximation using greedy matching.
pub fn bottleneck_distance(a: &[PersistencePair], b: &[PersistencePair]) -> f64 {
    // Filter to finite pairs
    let a_fin: Vec<_> = a.iter().filter(|p| p.death.is_finite()).collect();
    let b_fin: Vec<_> = b.iter().filter(|p| p.death.is_finite()).collect();

    let na = a_fin.len();
    let nb = b_fin.len();

    if na == 0 && nb == 0 { return 0.0; }

    // Cost of matching point to diagonal: persistence/2
    let mut max_cost: f64 = 0.0;

    // Greedy: match each a to closest b or diagonal
    let mut used = vec![false; nb];
    for ai in &a_fin {
        let diag_cost = ai.persistence() / 2.0;
        let mut best_cost = diag_cost;
        let mut best_j = None;

        for (j, bj) in b_fin.iter().enumerate() {
            if used[j] { continue; }
            let cost = (ai.birth - bj.birth).abs().max((ai.death - bj.death).abs());
            if cost < best_cost {
                best_cost = cost;
                best_j = Some(j);
            }
        }

        if let Some(j) = best_j { used[j] = true; }
        max_cost = max_cost.max(best_cost);
    }

    // Unmatched b points go to diagonal
    for (j, bj) in b_fin.iter().enumerate() {
        if !used[j] {
            max_cost = max_cost.max(bj.persistence() / 2.0);
        }
    }

    max_cost
}

/// Wasserstein-1 distance between persistence diagrams (same dimension).
pub fn wasserstein_distance(a: &[PersistencePair], b: &[PersistencePair]) -> f64 {
    let a_fin: Vec<_> = a.iter().filter(|p| p.death.is_finite()).collect();
    let b_fin: Vec<_> = b.iter().filter(|p| p.death.is_finite()).collect();

    // Greedy matching (approximation)
    let mut total = 0.0;
    let mut used = vec![false; b_fin.len()];

    for ai in &a_fin {
        let diag_cost = ai.persistence() / 2.0;
        let mut best_cost = diag_cost;
        let mut best_j = None;

        for (j, bj) in b_fin.iter().enumerate() {
            if used[j] { continue; }
            let cost = (ai.birth - bj.birth).abs() + (ai.death - bj.death).abs();
            if cost < best_cost * 2.0 { // l1 vs l_inf comparison
                let linf = (ai.birth - bj.birth).abs().max((ai.death - bj.death).abs());
                if linf < best_cost {
                    best_cost = linf;
                    best_j = Some(j);
                }
            }
        }

        if let Some(j) = best_j {
            used[j] = true;
            total += (ai.birth - b_fin[j].birth).abs() + (ai.death - b_fin[j].death).abs();
        } else {
            total += ai.persistence() / 2.0; // distance to diagonal in L₁
        }
    }

    for (j, bj) in b_fin.iter().enumerate() {
        if !used[j] { total += bj.persistence() / 2.0; } // distance to diagonal in L₁
    }

    total
}

// ═══════════════════════════════════════════════════════════════════════════
// Persistence features for ML
// ═══════════════════════════════════════════════════════════════════════════

/// Persistence statistics: [count, total_persistence, max_persistence, mean_persistence, std_persistence].
pub fn persistence_statistics(pairs: &[PersistencePair]) -> [f64; 5] {
    let finite: Vec<f64> = pairs.iter()
        .filter(|p| p.death.is_finite())
        .map(|p| p.persistence())
        .collect();

    let n = finite.len();
    if n == 0 { return [0.0; 5]; }

    let total: f64 = finite.iter().sum();
    let max = finite.iter().copied().fold(0.0_f64, f64::max);
    let mean = total / n as f64;
    let var: f64 = finite.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n as f64;

    [n as f64, total, max, mean, var.sqrt()]
}

/// Persistence entropy: -Σ (p_i/L) · log(p_i/L) where L = total persistence.
pub fn persistence_entropy(pairs: &[PersistencePair]) -> f64 {
    let finite: Vec<f64> = pairs.iter()
        .filter(|p| p.death.is_finite())
        .map(|p| p.persistence())
        .collect();

    let total: f64 = finite.iter().sum();
    if total < 1e-15 { return 0.0; }

    -finite.iter()
        .map(|p| {
            let w = p / total;
            if w > 1e-15 { w * w.ln() } else { 0.0 }
        })
        .sum::<f64>()
}

/// Betti curve: count of alive features at each filtration threshold.
pub fn betti_curve(pairs: &[PersistencePair], thresholds: &[f64]) -> Vec<usize> {
    thresholds.iter().map(|&t| {
        pairs.iter().filter(|p| p.birth <= t && p.death > t).count()
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h0_three_points() {
        // Three points: A-B dist 1, B-C dist 2, A-C dist 3
        let dist = vec![
            0.0, 1.0, 3.0,
            1.0, 0.0, 2.0,
            3.0, 2.0, 0.0,
        ];
        let diag = rips_h0(&dist, 3);
        let h0 = diag.dimension(0);

        // Should have 3 pairs: one surviving, two dying
        assert_eq!(h0.len(), 3, "3 points → 3 H₀ pairs (2 deaths + 1 survivor)");

        let finite: Vec<_> = h0.iter().filter(|p| p.death.is_finite()).collect();
        assert_eq!(finite.len(), 2);

        // First merge at distance 1, second at distance 2
        let mut deaths: Vec<f64> = finite.iter().map(|p| p.death).collect();
        deaths.sort_by(|a, b| a.total_cmp(b));
        assert!((deaths[0] - 1.0).abs() < 1e-10);
        assert!((deaths[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn h0_two_clusters() {
        // Two clusters with gap: {0,1} close together, {2,3} close together, far apart
        let dist = vec![
            0.0, 0.1, 5.0, 5.1,
            0.1, 0.0, 5.1, 5.0,
            5.0, 5.1, 0.0, 0.1,
            5.1, 5.0, 0.1, 0.0,
        ];
        let diag = rips_h0(&dist, 4);
        let h0_finite: Vec<_> = diag.dimension(0).into_iter()
            .filter(|p| p.death.is_finite())
            .collect();

        // Should see one long-lived gap (cluster merge at ~5.0)
        let max_pers = h0_finite.iter().map(|p| p.persistence()).fold(0.0_f64, f64::max);
        assert!(max_pers > 4.0, "Should detect cluster gap, max_pers={max_pers}");
    }

    #[test]
    fn h0_single_point() {
        let dist = vec![0.0];
        let diag = rips_h0(&dist, 1);
        assert_eq!(diag.pairs.len(), 1);
        assert!(diag.pairs[0].death.is_infinite());
    }

    #[test]
    fn h1_triangle() {
        // Three points forming equilateral triangle (all edges = 1).
        // H₀: 3 components born at r=0, two merge at r=1 (2 finite-death pairs).
        // H₁: the cycle born when all 3 edges connect at r=1 is immediately filled
        //     by the 2-simplex at the same filtration value → 0 persistent H₁.
        let dist = vec![
            0.0, 1.0, 1.0,
            1.0, 0.0, 1.0,
            1.0, 1.0, 0.0,
        ];
        let diag = rips_h1(&dist, 3, 2.0);
        // H₀: two finite-death merges at r=1
        let h0_finite: Vec<_> = diag.dimension(0).into_iter()
            .filter(|p| p.death.is_finite())
            .collect();
        assert_eq!(h0_finite.len(), 2, "Equilateral triangle: 2 H₀ merges");
        let deaths_at_1 = h0_finite.iter().filter(|p| (p.death - 1.0).abs() < 1e-10).count();
        assert_eq!(deaths_at_1, 2, "Both H₀ merges should occur at r=1");
        // H₁: no persistent loop (cycle born and filled at same filtration value)
        let h1_persistent: Vec<_> = diag.dimension(1).into_iter()
            .filter(|p| p.persistence() > 1e-10)
            .collect();
        assert_eq!(h1_persistent.len(), 0, "Equilateral triangle: no persistent H₁ loop");
    }

    #[test]
    fn persistence_stats_basic() {
        let pairs = vec![
            PersistencePair { dimension: 0, birth: 0.0, death: 1.0 },
            PersistencePair { dimension: 0, birth: 0.0, death: 3.0 },
            PersistencePair { dimension: 0, birth: 0.0, death: f64::INFINITY },
        ];
        let stats = persistence_statistics(&pairs);
        assert!((stats[0] - 2.0).abs() < 1e-10, "count=2 finite pairs");
        assert!((stats[1] - 4.0).abs() < 1e-10, "total=1+3=4");
        assert!((stats[2] - 3.0).abs() < 1e-10, "max=3");
        assert!((stats[3] - 2.0).abs() < 1e-10, "mean=2");
    }

    #[test]
    fn persistence_entropy_uniform() {
        // Two pairs with equal persistence → max entropy for 2 items
        let pairs = vec![
            PersistencePair { dimension: 0, birth: 0.0, death: 1.0 },
            PersistencePair { dimension: 0, birth: 0.0, death: 1.0 },
        ];
        let ent = persistence_entropy(&pairs);
        let expected = (2.0_f64).ln();
        assert!((ent - expected).abs() < 1e-10, "Entropy={ent} vs expected={expected}");
    }

    #[test]
    fn betti_curve_decreasing() {
        // H₀ of 3 points merging at 1.0 and 2.0
        let pairs = vec![
            PersistencePair { dimension: 0, birth: 0.0, death: 1.0 },
            PersistencePair { dimension: 0, birth: 0.0, death: 2.0 },
            PersistencePair { dimension: 0, birth: 0.0, death: f64::INFINITY },
        ];
        let thresholds: Vec<f64> = (0..5).map(|i| i as f64 * 0.6).collect();
        let betti = betti_curve(&pairs, &thresholds);
        // At t=0: 3 alive, t=0.6: 3, t=1.2: 2, t=1.8: 2, t=2.4: 1
        assert_eq!(betti[0], 3);
        assert!(betti[4] <= 2);
    }

    #[test]
    fn bottleneck_identical_diagrams() {
        let pairs = vec![
            PersistencePair { dimension: 0, birth: 0.0, death: 1.0 },
            PersistencePair { dimension: 0, birth: 0.0, death: 2.0 },
        ];
        let d = bottleneck_distance(&pairs, &pairs);
        assert!((d - 0.0).abs() < 1e-10, "Identical diagrams → distance 0");
    }

    #[test]
    fn wasserstein_identical_diagrams() {
        let pairs = vec![
            PersistencePair { dimension: 0, birth: 0.0, death: 1.0 },
            PersistencePair { dimension: 0, birth: 0.0, death: 3.0 },
            PersistencePair { dimension: 0, birth: 1.0, death: 5.0 },
        ];
        let d = wasserstein_distance(&pairs, &pairs);
        assert!((d - 0.0).abs() < 1e-10,
            "Wasserstein of identical diagrams should be 0, got {d}");
    }

    #[test]
    fn wasserstein_shifted_diagram() {
        // Diagram A has one pair at (0,2), diagram B has one pair at (0,4)
        let a = vec![PersistencePair { dimension: 0, birth: 0.0, death: 2.0 }];
        let b = vec![PersistencePair { dimension: 0, birth: 0.0, death: 4.0 }];
        let d = wasserstein_distance(&a, &b);
        // L1 cost of matching: |0-0| + |2-4| = 2
        assert!(d > 0.0, "Wasserstein should be positive for different diagrams, got {d}");
    }
}
