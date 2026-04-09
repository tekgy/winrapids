//! # Spectral Gap Persistence for Sparse Transition Operators
//!
//! Investigates whether the spectral gap |1 - |λ₂|| of sparse transition
//! matrices persists as the state space N → ∞.
//!
//! ## Architecture
//!
//! Sparse deterministic maps on [0, N-1] produce transition matrices with
//! at most one nonzero per column. The eigenvalue structure is determined
//! by the cycle decomposition of the map, but computing it directly via
//! Arnoldi iteration reveals how the spectrum evolves with N.
//!
//! ## Arithmetic maps
//!
//! Generic multiply-then-divide maps parameterized by (m, d):
//!   - If d | n:  n → n / d
//!   - Otherwise: n → m·n + 1   (mod N, clamped to [0, N-1])
//!
//! The classic (m=3, d=2) gives the Collatz map. Other (m, d) pairs
//! produce different cycle structures and spectral behavior.

use crate::linear_algebra::{Mat, dot, vec_norm};

// ═══════════════════════════════════════════════════════════════════════════
// Sparse deterministic transition operator
// ═══════════════════════════════════════════════════════════════════════════

/// A deterministic map on [0, N-1]. Each state maps to exactly one target.
/// The transition matrix T has T[target[j], j] = 1.
/// States that escape [0, N-1] get self-loops (absorbing).
#[derive(Debug, Clone)]
pub struct SparseDeterministicMap {
    /// State space size.
    pub n: usize,
    /// target[j] = state that j maps to.
    pub target: Vec<usize>,
}

impl SparseDeterministicMap {
    /// Build from an arbitrary target vector. Each `target[j]` is the unique
    /// successor of state `j` in `[0, n)`. Any out-of-range targets are
    /// clamped to a self-loop (`target[j] = j`).
    ///
    /// This is the general constructor for a *functional graph* — a directed
    /// graph in which every node has exactly one outgoing edge. The struct
    /// alias [`crate::FunctionalGraph`] is provided for graph-theoretic uses.
    pub fn from_targets(target: Vec<usize>) -> Self {
        let n = target.len();
        let mut t = target;
        for j in 0..n {
            if t[j] >= n { t[j] = j; }
        }
        SparseDeterministicMap { n, target: t }
    }

    /// Build from a closure `f: usize -> usize`. Out-of-range outputs become
    /// self-loops. Convenient for defining a functional graph by a rule.
    pub fn from_fn<F: Fn(usize) -> usize>(n: usize, f: F) -> Self {
        let target: Vec<usize> = (0..n).map(|j| f(j)).collect();
        Self::from_targets(target)
    }

    /// Build from an arithmetic map parameterized by (m, d).
    ///
    /// For state n in [1, N]:
    ///   - If d | n:  n → n / d
    ///   - Otherwise: n → m·n + 1
    ///
    /// States are internally 0-indexed: state j represents value j+1.
    /// If the result exceeds N, the state gets a self-loop (absorbing).
    pub fn arithmetic(n: usize, m: usize, d: usize) -> Self {
        let mut target = vec![0usize; n];
        for j in 0..n {
            let val = j + 1; // 1-indexed value
            let next_val = if val % d == 0 {
                val / d
            } else {
                m * val + 1
            };
            // Map back to 0-indexed, with self-loop if out of range
            if next_val >= 1 && next_val <= n {
                target[j] = next_val - 1;
            } else {
                target[j] = j; // self-loop (absorbing)
            }
        }
        SparseDeterministicMap { n, target }
    }

    /// Build from an arithmetic map with modular wrapping (no absorbing states).
    ///
    /// Same as `arithmetic`, except states that would exceed N wrap:
    ///   n → ((m·n + 1) - 1) mod N + 1   (1-indexed)
    ///
    /// This makes the map a function on [0, N-1] with no self-loops from
    /// boundary effects. The resulting transition matrix is a permutation
    /// (or near-permutation), so all eigenvalues have |λ| ≤ 1.
    pub fn arithmetic_modular(n: usize, m: usize, d: usize) -> Self {
        let mut target = vec![0usize; n];
        for j in 0..n {
            let val = j + 1; // 1-indexed value
            let next_val = if val % d == 0 {
                val / d
            } else {
                // Wrap: (m*val + 1 - 1) mod N + 1 = (m*val) mod N + 1
                // But we want 1-indexed result in [1, N]
                let raw = m * val + 1;
                ((raw - 1) % n) + 1
            };
            target[j] = next_val - 1; // 0-indexed
        }
        SparseDeterministicMap { n, target }
    }

    /// Sparse matrix-vector multiply: y = T * x
    /// where T[target[j], j] = 1.
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.n);
        let mut y = vec![0.0; self.n];
        for j in 0..self.n {
            y[self.target[j]] += x[j];
        }
        y
    }

    /// Sparse matrix-transpose-vector multiply: y = T^T * x
    /// T^T has T^T[j, target[j]] = 1, so y[j] = x[target[j]].
    pub fn mul_vec_transpose(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.n);
        let mut y = vec![0.0; self.n];
        for j in 0..self.n {
            y[j] = x[self.target[j]];
        }
        y
    }

    /// Count the number of distinct cycles in the map.
    /// Returns (cycle_count, cycle_lengths, transient_count).
    pub fn cycle_structure(&self) -> CycleStructure {
        let mut visited = vec![false; self.n];
        let mut in_cycle = vec![false; self.n];
        let mut cycle_lengths = Vec::new();

        for start in 0..self.n {
            if visited[start] { continue; }

            // Trace the trajectory from `start`
            let mut path = Vec::new();
            let mut pos_map = std::collections::HashMap::new();
            let mut cur = start;

            loop {
                if visited[cur] {
                    // We hit a previously visited node from another trajectory.
                    // No new cycle from this start.
                    break;
                }
                if let Some(&cycle_start_pos) = pos_map.get(&cur) {
                    // Found a cycle: from cycle_start_pos to end of path
                    let cycle_len = path.len() - cycle_start_pos;
                    cycle_lengths.push(cycle_len);
                    for i in cycle_start_pos..path.len() {
                        in_cycle[path[i]] = true;
                    }
                    break;
                }
                pos_map.insert(cur, path.len());
                path.push(cur);
                cur = self.target[cur];
            }

            for &node in &path {
                visited[node] = true;
            }
        }

        let transient_count = (0..self.n).filter(|&i| !in_cycle[i]).count();
        cycle_lengths.sort_unstable_by(|a, b| b.cmp(a)); // descending

        CycleStructure {
            cycle_count: cycle_lengths.len(),
            cycle_lengths,
            transient_count,
        }
    }
}

/// Cycle structure of a deterministic map.
#[derive(Debug, Clone)]
pub struct CycleStructure {
    /// Number of distinct cycles.
    pub cycle_count: usize,
    /// Lengths of each cycle, sorted descending.
    pub cycle_lengths: Vec<usize>,
    /// Number of transient (non-cyclic) states.
    pub transient_count: usize,
}

// ═══════════════════════════════════════════════════════════════════════════
// Arnoldi iteration for non-symmetric sparse matrices
// ═══════════════════════════════════════════════════════════════════════════

/// Result of Arnoldi iteration.
#[derive(Debug, Clone)]
pub struct ArnoldiResult {
    /// Eigenvalues (complex: (real, imag) pairs), sorted by descending |λ|.
    pub eigenvalues: Vec<(f64, f64)>,
    /// |λ| for each eigenvalue, sorted descending.
    pub magnitudes: Vec<f64>,
}

/// Arnoldi iteration to find the top-k eigenvalues of a sparse operator.
///
/// Builds a Krylov subspace of dimension `krylov_dim`, then computes
/// eigenvalues of the resulting upper Hessenberg matrix using QR iteration.
///
/// `matvec`: closure that computes y = A*x for the sparse operator.
/// `n`: dimension of the operator.
/// `krylov_dim`: size of Krylov subspace (typically 30-100).
/// `n_eigenvalues`: how many eigenvalues to return.
pub fn arnoldi_eigenvalues<F>(
    matvec: F,
    n: usize,
    krylov_dim: usize,
    n_eigenvalues: usize,
) -> ArnoldiResult
where
    F: Fn(&[f64]) -> Vec<f64>,
{
    let k = krylov_dim.min(n);

    // Random-ish starting vector (deterministic for reproducibility)
    let mut v0 = vec![0.0; n];
    let mut rng_state = 0x12345678u64;
    for i in 0..n {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        v0[i] = (rng_state >> 33) as f64 / (1u64 << 31) as f64 - 1.0;
    }
    let norm0 = vec_norm(&v0);
    for x in v0.iter_mut() { *x /= norm0; }

    // Arnoldi factorization: A V_k = V_{k+1} H̃_k
    // V: n × (k+1) stored column-major
    // H: (k+1) × k upper Hessenberg
    let mut v_cols: Vec<Vec<f64>> = Vec::with_capacity(k + 1);
    v_cols.push(v0);

    let mut h = Mat::zeros(k + 1, k);

    for j in 0..k {
        // w = A * v_j
        let mut w = matvec(&v_cols[j]);

        // Modified Gram-Schmidt orthogonalization
        for i in 0..=j {
            let hij = dot(&v_cols[i], &w);
            h.set(i, j, hij);
            for l in 0..n {
                w[l] -= hij * v_cols[i][l];
            }
        }

        // Re-orthogonalize (one round of DGKS correction)
        for i in 0..=j {
            let correction = dot(&v_cols[i], &w);
            h.set(i, j, h.get(i, j) + correction);
            for l in 0..n {
                w[l] -= correction * v_cols[i][l];
            }
        }

        let h_next = vec_norm(&w);
        h.set(j + 1, j, h_next);

        if h_next < 1e-14 {
            // Invariant subspace found — Krylov breakdown
            // Pad with zero vector and stop
            v_cols.push(vec![0.0; n]);
            break;
        }

        let mut v_next = w;
        for x in v_next.iter_mut() { *x /= h_next; }
        v_cols.push(v_next);

        if v_cols.len() > k { break; }
    }

    // Extract the k×k upper Hessenberg matrix
    let actual_k = v_cols.len() - 1;
    let hk = extract_submatrix(&h, actual_k, actual_k);

    // Compute eigenvalues of H_k via QR iteration (implicit shifts)
    let eigs = hessenberg_qr_eigenvalues(&hk);

    // Sort by magnitude (descending)
    let mut eig_mags: Vec<((f64, f64), f64)> = eigs.iter()
        .map(|&(re, im)| ((re, im), (re * re + im * im).sqrt()))
        .collect();
    eig_mags.sort_by(|a, b| b.1.total_cmp(&a.1));

    let n_ret = n_eigenvalues.min(eig_mags.len());
    ArnoldiResult {
        eigenvalues: eig_mags[..n_ret].iter().map(|x| x.0).collect(),
        magnitudes: eig_mags[..n_ret].iter().map(|x| x.1).collect(),
    }
}

fn extract_submatrix(m: &Mat, rows: usize, cols: usize) -> Mat {
    let mut out = Mat::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            out.set(i, j, m.get(i, j));
        }
    }
    out
}

/// QR iteration for eigenvalues of an upper Hessenberg matrix.
/// Returns complex eigenvalue pairs (real, imag).
///
/// Uses implicit single-shift QR with Wilkinson shift and careful deflation.
/// Operates on the unreduced submatrix [lo..hi] at each step.
fn hessenberg_qr_eigenvalues(h: &Mat) -> Vec<(f64, f64)> {
    let n = h.rows;
    if n == 0 { return Vec::new(); }
    if n == 1 { return vec![(h.get(0, 0), 0.0)]; }

    let mut a = h.clone();
    let mut eigenvalues = Vec::with_capacity(n);
    let max_iter = 100 * n;
    let mut nn = n; // size of active matrix (indices 0..nn)

    for _iter in 0..max_iter {
        if nn == 0 { break; }
        if nn == 1 {
            eigenvalues.push((a.get(0, 0), 0.0));
            nn = 0;
            break;
        }

        // Find the bottom-most small subdiagonal element for deflation
        // Scan upward from nn-1
        let mut lo = nn - 1;
        while lo > 0 {
            let sub = a.get(lo, lo - 1).abs();
            let diag_scale = a.get(lo, lo).abs() + a.get(lo - 1, lo - 1).abs();
            if sub <= 1e-14 * diag_scale.max(1e-300) {
                a.set(lo, lo - 1, 0.0); // enforce zero
                break;
            }
            lo -= 1;
        }

        // The unreduced block is [lo..nn)
        let block_size = nn - lo;

        if block_size == 1 {
            // Deflated a 1×1 block at position nn-1
            eigenvalues.push((a.get(nn - 1, nn - 1), 0.0));
            nn -= 1;
            continue;
        }

        if block_size == 2 {
            // Deflated a 2×2 block at [nn-2, nn-1]
            let (e1, e2) = eigenvalues_2x2(
                a.get(nn - 2, nn - 2), a.get(nn - 2, nn - 1),
                a.get(nn - 1, nn - 2), a.get(nn - 1, nn - 1),
            );
            eigenvalues.push(e1);
            eigenvalues.push(e2);
            nn -= 2;
            continue;
        }

        // Apply one QR step with Wilkinson shift on the active block [lo..nn)
        let shift = wilkinson_shift(
            a.get(nn - 2, nn - 2), a.get(nn - 2, nn - 1),
            a.get(nn - 1, nn - 2), a.get(nn - 1, nn - 1),
        );

        // Implicit QR step via Givens rotations (bulge chase)
        let mut x = a.get(lo, lo) - shift;
        let mut y = a.get(lo + 1, lo);

        for k in lo..nn - 1 {
            let (c, s) = givens(x, y);

            // Apply G(k, k+1) from the left: rows k, k+1
            let col_lo = if k > lo { k - 1 } else { lo };
            for j in col_lo..nn {
                let t1 = a.get(k, j);
                let t2 = a.get(k + 1, j);
                a.set(k, j, c * t1 + s * t2);
                a.set(k + 1, j, -s * t1 + c * t2);
            }

            // Apply G(k, k+1) from the right: cols k, k+1
            let row_hi = (k + 3).min(nn);
            for i in lo..row_hi {
                let t1 = a.get(i, k);
                let t2 = a.get(i, k + 1);
                a.set(i, k, c * t1 + s * t2);
                a.set(i, k + 1, -s * t1 + c * t2);
            }

            // Set up next bulge chase step
            if k + 2 < nn {
                x = a.get(k + 1, k);
                y = a.get(k + 2, k);
            }
        }
    }

    // Extract any remaining eigenvalues that didn't converge
    while nn >= 2 {
        let (e1, e2) = eigenvalues_2x2(
            a.get(nn - 2, nn - 2), a.get(nn - 2, nn - 1),
            a.get(nn - 1, nn - 2), a.get(nn - 1, nn - 1),
        );
        eigenvalues.push(e1);
        eigenvalues.push(e2);
        nn -= 2;
    }
    if nn == 1 {
        eigenvalues.push((a.get(0, 0), 0.0));
    }

    eigenvalues
}

/// Eigenvalues of a 2×2 matrix [[a, b], [c, d]].
fn eigenvalues_2x2(a: f64, b: f64, c: f64, d: f64) -> ((f64, f64), (f64, f64)) {
    let trace = a + d;
    let det = a * d - b * c;
    let disc = trace * trace - 4.0 * det;

    if disc >= 0.0 {
        let sqrt_disc = disc.sqrt();
        ((0.5 * (trace + sqrt_disc), 0.0), (0.5 * (trace - sqrt_disc), 0.0))
    } else {
        let sqrt_disc = (-disc).sqrt();
        ((0.5 * trace, 0.5 * sqrt_disc), (0.5 * trace, -0.5 * sqrt_disc))
    }
}

/// Wilkinson shift: eigenvalue of 2×2 bottom-right block closer to a[n-1,n-1].
fn wilkinson_shift(a: f64, b: f64, c: f64, d: f64) -> f64 {
    let trace = a + d;
    let det = a * d - b * c;
    let disc = trace * trace - 4.0 * det;

    if disc >= 0.0 {
        let sqrt_disc = disc.sqrt();
        let e1 = 0.5 * (trace + sqrt_disc);
        let e2 = 0.5 * (trace - sqrt_disc);
        if (e1 - d).abs() < (e2 - d).abs() { e1 } else { e2 }
    } else {
        d
    }
}

/// Givens rotation: returns (c, s) such that [c s; -s c]^T [x; y] = [r; 0].
fn givens(x: f64, y: f64) -> (f64, f64) {
    if y.abs() < 1e-300 {
        (1.0, 0.0)
    } else if x.abs() < 1e-300 {
        (0.0, y.signum())
    } else {
        let r = (x * x + y * y).sqrt();
        (x / r, y / r)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Spectral gap analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Spectral gap result for a single N.
#[derive(Debug, Clone)]
pub struct SpectralGapResult {
    /// State space size.
    pub n: usize,
    /// Dominant eigenvalue magnitude (should be ~1 for stochastic matrices).
    pub lambda_1: f64,
    /// Second eigenvalue magnitude.
    pub lambda_2: f64,
    /// Spectral gap: |1 - |λ₂||.
    pub gap: f64,
    /// Top eigenvalues (magnitude).
    pub top_eigenvalues: Vec<f64>,
    /// Cycle structure of the map.
    pub cycles: CycleStructure,
}

/// Compute spectral gap for an arithmetic map at a given N.
pub fn spectral_gap_arithmetic(
    n: usize,
    m: usize,
    d: usize,
    krylov_dim: usize,
    n_eigenvalues: usize,
) -> SpectralGapResult {
    let map = SparseDeterministicMap::arithmetic(n, m, d);
    let cycles = map.cycle_structure();

    let result = arnoldi_eigenvalues(
        |x| map.mul_vec(x),
        n,
        krylov_dim,
        n_eigenvalues,
    );

    let lambda_1 = if result.magnitudes.is_empty() { 0.0 } else { result.magnitudes[0] };
    let lambda_2 = if result.magnitudes.len() < 2 { 0.0 } else { result.magnitudes[1] };
    let gap = (1.0 - lambda_2).abs();

    SpectralGapResult {
        n,
        lambda_1,
        lambda_2,
        gap,
        top_eigenvalues: result.magnitudes,
        cycles,
    }
}

/// Lazy random walk on the graph of an arithmetic map.
///
/// With probability `alpha`, follow the deterministic map.
/// With probability `1-alpha`, stay in place.
/// This produces a column-stochastic matrix with λ₁=1 and λ₂ < 1
/// whenever the underlying graph is connected. The spectral gap
/// measures genuine mixing rate.
pub struct LazyRandomWalk {
    pub map: SparseDeterministicMap,
    pub alpha: f64,
}

impl LazyRandomWalk {
    pub fn new(map: SparseDeterministicMap, alpha: f64) -> Self {
        LazyRandomWalk { map, alpha }
    }

    /// Stochastic mat-vec: y = (alpha * T + (1-alpha) * I) * x
    pub fn mul_vec(&self, x: &[f64]) -> Vec<f64> {
        let tx = self.map.mul_vec(x);
        let n = x.len();
        let mut y = vec![0.0; n];
        for i in 0..n {
            y[i] = self.alpha * tx[i] + (1.0 - self.alpha) * x[i];
        }
        y
    }
}

/// Compute spectral gap for a modular arithmetic map (no absorbing states).
pub fn spectral_gap_modular(
    n: usize,
    m: usize,
    d: usize,
    krylov_dim: usize,
    n_eigenvalues: usize,
) -> SpectralGapResult {
    let map = SparseDeterministicMap::arithmetic_modular(n, m, d);
    let cycles = map.cycle_structure();

    let result = arnoldi_eigenvalues(
        |x| map.mul_vec(x),
        n,
        krylov_dim,
        n_eigenvalues,
    );

    let lambda_1 = if result.magnitudes.is_empty() { 0.0 } else { result.magnitudes[0] };
    let lambda_2 = if result.magnitudes.len() < 2 { 0.0 } else { result.magnitudes[1] };
    let gap = (1.0 - lambda_2).abs();

    SpectralGapResult {
        n,
        lambda_1,
        lambda_2,
        gap,
        top_eigenvalues: result.magnitudes,
        cycles,
    }
}

/// Compute spectral gap for a lazy random walk on an arithmetic map.
pub fn spectral_gap_lazy_walk(
    n: usize,
    m: usize,
    d: usize,
    alpha: f64,
    krylov_dim: usize,
    n_eigenvalues: usize,
) -> SpectralGapResult {
    let map = SparseDeterministicMap::arithmetic(n, m, d);
    let cycles = map.cycle_structure();
    let walk = LazyRandomWalk::new(map, alpha);

    let result = arnoldi_eigenvalues(
        |x| walk.mul_vec(x),
        n,
        krylov_dim,
        n_eigenvalues,
    );

    let lambda_1 = if result.magnitudes.is_empty() { 0.0 } else { result.magnitudes[0] };
    let lambda_2 = if result.magnitudes.len() < 2 { 0.0 } else { result.magnitudes[1] };
    let gap = (1.0 - lambda_2).abs();

    SpectralGapResult {
        n,
        lambda_1,
        lambda_2,
        gap,
        top_eigenvalues: result.magnitudes,
        cycles,
    }
}

/// Run spectral gap persistence study across multiple N values.
/// Returns results for each N.
pub fn spectral_gap_persistence(
    sizes: &[usize],
    m: usize,
    d: usize,
    krylov_dim: usize,
    n_eigenvalues: usize,
) -> Vec<SpectralGapResult> {
    sizes.iter().map(|&n| {
        spectral_gap_arithmetic(n, m, d, krylov_dim, n_eigenvalues)
    }).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    // ── Sparse map construction ────────────────────────────────────────

    // ── Generic functional-graph constructors ────────────────────────────

    #[test]
    fn from_targets_general() {
        // A simple 5-node functional graph: rho-shape with tail 0 -> 1 -> 2
        // and cycle 2 -> 3 -> 4 -> 2.
        let g = SparseDeterministicMap::from_targets(vec![1, 2, 3, 4, 2]);
        assert_eq!(g.n, 5);
        let cs = g.cycle_structure();
        assert_eq!(cs.cycle_count, 1);
        assert_eq!(cs.cycle_lengths, vec![3]);
        assert_eq!(cs.transient_count, 2);
    }

    #[test]
    fn from_targets_clamps_out_of_range() {
        // Targets >= n become self-loops.
        let g = SparseDeterministicMap::from_targets(vec![99, 0, 1]);
        assert_eq!(g.target, vec![0, 0, 1]); // position 0 was 99 -> self-loop 0
    }

    #[test]
    fn from_fn_constructor() {
        // f(j) = (j + 1) mod n  — a single n-cycle
        let g = SparseDeterministicMap::from_fn(7, |j| (j + 1) % 7);
        let cs = g.cycle_structure();
        assert_eq!(cs.cycle_count, 1);
        assert_eq!(cs.cycle_lengths, vec![7]);
        assert_eq!(cs.transient_count, 0);
    }

    #[test]
    fn arithmetic_map_collatz_small() {
        // (m=3, d=2) on [1..10]
        let map = SparseDeterministicMap::arithmetic(10, 3, 2);
        // State 0 → val 1: odd → 3*1+1=4 → state 3 ✓
        assert_eq!(map.target[0], 3);
        // State 1 → val 2: even → 2/2=1 → state 0 ✓
        assert_eq!(map.target[1], 0);
        // State 3 → val 4: even → 4/2=2 → state 1 ✓
        assert_eq!(map.target[3], 1);
        // State 4 → val 5: odd → 3*5+1=16 → exceeds 10 → self-loop
        assert_eq!(map.target[4], 4);
        // State 5 → val 6: even → 6/2=3 → state 2 ✓
        assert_eq!(map.target[5], 2);
    }

    #[test]
    fn arithmetic_map_m5_d3() {
        // (m=5, d=3) on [1..15]
        let map = SparseDeterministicMap::arithmetic(15, 5, 3);
        // State 2 → val 3: 3%3==0 → 3/3=1 → state 0
        assert_eq!(map.target[2], 0);
        // State 0 → val 1: 1%3!=0 → 5*1+1=6 → state 5
        assert_eq!(map.target[0], 5);
        // State 8 → val 9: 9%3==0 → 9/3=3 → state 2
        assert_eq!(map.target[8], 2);
    }

    // ── Sparse mat-vec ─────────────────────────────────────────────────

    #[test]
    fn sparse_matvec_identity_like() {
        // Identity map: every state maps to itself
        let map = SparseDeterministicMap {
            n: 5,
            target: vec![0, 1, 2, 3, 4],
        };
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = map.mul_vec(&x);
        assert_eq!(y, x);
    }

    #[test]
    fn sparse_matvec_shift() {
        // Cyclic shift: n → (n+1) mod 4
        let map = SparseDeterministicMap {
            n: 4,
            target: vec![1, 2, 3, 0],
        };
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let y = map.mul_vec(&x);
        // T has T[1,0]=1, T[2,1]=1, T[3,2]=1, T[0,3]=1
        // y = T * [1,0,0,0] = [0,1,0,0]
        assert_eq!(y, vec![0.0, 1.0, 0.0, 0.0]);
    }

    // ── Cycle structure ────────────────────────────────────────────────

    #[test]
    fn cycle_structure_identity() {
        let map = SparseDeterministicMap {
            n: 5,
            target: vec![0, 1, 2, 3, 4],
        };
        let cs = map.cycle_structure();
        assert_eq!(cs.cycle_count, 5);
        assert!(cs.cycle_lengths.iter().all(|&l| l == 1));
        assert_eq!(cs.transient_count, 0);
    }

    #[test]
    fn cycle_structure_single_cycle() {
        let map = SparseDeterministicMap {
            n: 4,
            target: vec![1, 2, 3, 0], // 0→1→2→3→0
        };
        let cs = map.cycle_structure();
        assert_eq!(cs.cycle_count, 1);
        assert_eq!(cs.cycle_lengths, vec![4]);
        assert_eq!(cs.transient_count, 0);
    }

    #[test]
    fn cycle_structure_with_transients() {
        // 0→1→2→0 (cycle of 3), 3→4→2 (transient tail into cycle)
        let map = SparseDeterministicMap {
            n: 5,
            target: vec![1, 2, 0, 4, 2],
        };
        let cs = map.cycle_structure();
        assert_eq!(cs.cycle_count, 1);
        assert_eq!(cs.cycle_lengths, vec![3]);
        assert_eq!(cs.transient_count, 2); // states 3 and 4 are transient
    }

    // ── Arnoldi eigenvalues ────────────────────────────────────────────

    #[test]
    fn arnoldi_identity_eigenvalues() {
        // Identity matrix: all eigenvalues = 1
        let n = 10;
        let result = arnoldi_eigenvalues(
            |x| x.to_vec(),
            n,
            10,
            5,
        );
        for &mag in &result.magnitudes {
            assert!((mag - 1.0).abs() < 1e-10, "Identity eigenvalue magnitude should be 1, got {mag}");
        }
    }

    #[test]
    fn arnoldi_cyclic_shift_eigenvalues() {
        // Cyclic shift of length 8: eigenvalues are 8th roots of unity
        // |λ| = 1 for all eigenvalues
        let n = 8;
        let map = SparseDeterministicMap {
            n: 8,
            target: vec![1, 2, 3, 4, 5, 6, 7, 0],
        };
        let result = arnoldi_eigenvalues(
            |x| map.mul_vec(x),
            n,
            8,
            4,
        );
        // Top eigenvalues should all have magnitude ~1
        for &mag in &result.magnitudes {
            assert!((mag - 1.0).abs() < 0.01,
                "Cyclic shift eigenvalue magnitude should be ~1, got {mag}");
        }
    }

    #[test]
    fn arnoldi_diagonal_matrix() {
        // Diagonal matrix with well-separated eigenvalues
        let n = 8;
        let diag = vec![5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.25, 0.1];
        let result = arnoldi_eigenvalues(
            |x| {
                let mut y = vec![0.0; n];
                for i in 0..n { y[i] = diag[i] * x[i]; }
                y
            },
            n,
            8,
            4,
        );
        // Top 4 eigenvalues should be 5, 4, 3, 2
        assert!((result.magnitudes[0] - 5.0).abs() < 0.01, "λ₁={}", result.magnitudes[0]);
        assert!((result.magnitudes[1] - 4.0).abs() < 0.01, "λ₂={}", result.magnitudes[1]);
        assert!((result.magnitudes[2] - 3.0).abs() < 0.01, "λ₃={}", result.magnitudes[2]);
        assert!((result.magnitudes[3] - 2.0).abs() < 0.01, "λ₄={}", result.magnitudes[3]);
    }

    #[test]
    fn arnoldi_absorbing_state() {
        // Map with one absorbing state: 0→0, 1→0, 2→0
        // T = [[1,1,1],[0,0,0],[0,0,0]]
        // Eigenvalues: 1 (absorbing), 0, 0
        let map = SparseDeterministicMap {
            n: 3,
            target: vec![0, 0, 0],
        };
        let result = arnoldi_eigenvalues(
            |x| map.mul_vec(x),
            3,
            3,
            3,
        );
        assert!((result.magnitudes[0] - 1.0).abs() < 1e-6,
            "Absorbing state should have λ₁ ≈ 1 (col sum = 3 so actually λ₁ = 3)... got {}", result.magnitudes[0]);
    }

    // ── Spectral gap computation ───────────────────────────────────────

    #[test]
    fn spectral_gap_collatz_n100() {
        let result = spectral_gap_arithmetic(100, 3, 2, 50, 10);
        assert!(result.lambda_1 > 0.5, "λ₁ should be significant, got {}", result.lambda_1);
        assert!(result.top_eigenvalues.len() >= 2, "Should find at least 2 eigenvalues");
        eprintln!("N=100 Collatz: λ₁={:.6}, λ₂={:.6}, gap={:.6}, cycles={}",
            result.lambda_1, result.lambda_2, result.gap, result.cycles.cycle_count);
        eprintln!("  cycle lengths: {:?}", &result.cycles.cycle_lengths[..result.cycles.cycle_lengths.len().min(10)]);
        eprintln!("  transients: {}", result.cycles.transient_count);
    }

    #[test]
    fn spectral_gap_collatz_n1000() {
        let result = spectral_gap_arithmetic(1_000, 3, 2, 80, 10);
        assert!(result.lambda_1 > 0.5, "λ₁ should be significant, got {}", result.lambda_1);
        eprintln!("N=1000 Collatz: λ₁={:.6}, λ₂={:.6}, gap={:.6}, cycles={}",
            result.lambda_1, result.lambda_2, result.gap, result.cycles.cycle_count);
        eprintln!("  cycle lengths: {:?}", &result.cycles.cycle_lengths[..result.cycles.cycle_lengths.len().min(10)]);
        eprintln!("  transients: {}", result.cycles.transient_count);
    }

    #[test]
    fn spectral_gap_collatz_n10000() {
        let result = spectral_gap_arithmetic(10_000, 3, 2, 100, 10);
        assert!(result.lambda_1 > 0.5, "λ₁ should be significant, got {}", result.lambda_1);
        eprintln!("N=10000 Collatz: λ₁={:.6}, λ₂={:.6}, gap={:.6}, cycles={}",
            result.lambda_1, result.lambda_2, result.gap, result.cycles.cycle_count);
        eprintln!("  cycle lengths: {:?}", &result.cycles.cycle_lengths[..result.cycles.cycle_lengths.len().min(10)]);
        eprintln!("  transients: {}", result.cycles.transient_count);
    }

    #[test]
    fn spectral_gap_collatz_n100000() {
        let result = spectral_gap_arithmetic(100_000, 3, 2, 100, 10);
        assert!(result.lambda_1 > 0.5, "λ₁ should be significant, got {}", result.lambda_1);
        eprintln!("N=100000 Collatz: λ₁={:.6}, λ₂={:.6}, gap={:.6}, cycles={}",
            result.lambda_1, result.lambda_2, result.gap, result.cycles.cycle_count);
        eprintln!("  cycle lengths: {:?}", &result.cycles.cycle_lengths[..result.cycles.cycle_lengths.len().min(10)]);
        eprintln!("  transients: {}", result.cycles.transient_count);
    }

    // ── Persistence study ──────────────────────────────────────────────

    #[test]
    fn spectral_gap_persistence_collatz() {
        let sizes = vec![100, 500, 1_000, 5_000, 10_000];
        let results = spectral_gap_persistence(&sizes, 3, 2, 80, 10);

        eprintln!("\n{:=^60}", "");
        eprintln!("  SPECTRAL GAP PERSISTENCE: Collatz (m=3, d=2)");
        eprintln!("{:=^60}", "");
        eprintln!("{:>8} {:>10} {:>10} {:>10} {:>8} {:>10}",
            "N", "λ₁", "λ₂", "gap", "cycles", "transient");

        for r in &results {
            eprintln!("{:>8} {:>10.6} {:>10.6} {:>10.6} {:>8} {:>10}",
                r.n, r.lambda_1, r.lambda_2, r.gap, r.cycles.cycle_count, r.cycles.transient_count);
        }

        // Basic sanity: all should have found eigenvalues
        for r in &results {
            assert!(r.top_eigenvalues.len() >= 2,
                "N={}: should find at least 2 eigenvalues", r.n);
        }
    }

    #[test]
    fn spectral_gap_persistence_m5_d3() {
        let sizes = vec![100, 500, 1_000, 5_000, 10_000];
        let results = spectral_gap_persistence(&sizes, 5, 3, 80, 10);

        eprintln!("\n{:=^60}", "");
        eprintln!("  SPECTRAL GAP PERSISTENCE: (m=5, d=3)");
        eprintln!("{:=^60}", "");
        eprintln!("{:>8} {:>10} {:>10} {:>10} {:>8} {:>10}",
            "N", "λ₁", "λ₂", "gap", "cycles", "transient");

        for r in &results {
            eprintln!("{:>8} {:>10.6} {:>10.6} {:>10.6} {:>8} {:>10}",
                r.n, r.lambda_1, r.lambda_2, r.gap, r.cycles.cycle_count, r.cycles.transient_count);
        }
    }

    #[test]
    fn spectral_gap_persistence_m7_d2() {
        let sizes = vec![100, 500, 1_000, 5_000, 10_000];
        let results = spectral_gap_persistence(&sizes, 7, 2, 80, 10);

        eprintln!("\n{:=^60}", "");
        eprintln!("  SPECTRAL GAP PERSISTENCE: (m=7, d=2)");
        eprintln!("{:=^60}", "");
        eprintln!("{:>8} {:>10} {:>10} {:>10} {:>8} {:>10}",
            "N", "λ₁", "λ₂", "gap", "cycles", "transient");

        for r in &results {
            eprintln!("{:>8} {:>10.6} {:>10.6} {:>10.6} {:>8} {:>10}",
                r.n, r.lambda_1, r.lambda_2, r.gap, r.cycles.cycle_count, r.cycles.transient_count);
        }
    }

    // ── Modular variant ──────────────────────────────────────────────

    #[test]
    fn modular_map_collatz_small() {
        let map = SparseDeterministicMap::arithmetic_modular(10, 3, 2);
        // State 0 → val 1: odd → 3*1+1=4 → state 3
        assert_eq!(map.target[0], 3);
        // State 1 → val 2: even → 2/2=1 → state 0
        assert_eq!(map.target[1], 0);
        // State 4 → val 5: odd → 3*5+1=16 → (16-1)%10+1=6 → state 5
        assert_eq!(map.target[4], 5);
        // State 8 → val 9: odd → 3*9+1=28 → (28-1)%10+1=8 → state 7
        assert_eq!(map.target[8], 7);
    }

    #[test]
    fn modular_map_no_self_loops_from_boundary() {
        // With modular wrapping, no state should self-loop purely from overflow
        let map = SparseDeterministicMap::arithmetic_modular(100, 3, 2);
        let absorbing = SparseDeterministicMap::arithmetic(100, 3, 2);
        let mod_self_loops: usize = (0..100).filter(|&j| map.target[j] == j).count();
        let abs_self_loops: usize = (0..100).filter(|&j| absorbing.target[j] == j).count();
        // Modular should have far fewer self-loops than absorbing
        assert!(mod_self_loops < abs_self_loops,
            "Modular self-loops ({mod_self_loops}) should be fewer than absorbing ({abs_self_loops})");
    }

    #[test]
    fn spectral_gap_modular_collatz_persistence() {
        let sizes = vec![100, 500, 1_000, 5_000, 10_000];

        eprintln!("\n{:=^70}", "");
        eprintln!("  MODULAR SPECTRAL GAP: Collatz (m=3, d=2)");
        eprintln!("{:=^70}", "");
        eprintln!("{:>8} {:>10} {:>10} {:>10} {:>8} {:>10} {:>8}",
            "N", "λ₁", "λ₂", "gap", "cycles", "transient", "max_cyc");

        for &n in &sizes {
            let r = spectral_gap_modular(n, 3, 2, 80, 10);
            let max_cyc = r.cycles.cycle_lengths.first().copied().unwrap_or(0);
            eprintln!("{:>8} {:>10.6} {:>10.6} {:>10.6} {:>8} {:>10} {:>8}",
                r.n, r.lambda_1, r.lambda_2, r.gap, r.cycles.cycle_count,
                r.cycles.transient_count, max_cyc);
        }
    }

    #[test]
    fn spectral_gap_modular_m5_d3_persistence() {
        let sizes = vec![100, 500, 1_000, 5_000, 10_000];

        eprintln!("\n{:=^70}", "");
        eprintln!("  MODULAR SPECTRAL GAP: (m=5, d=3)");
        eprintln!("{:=^70}", "");
        eprintln!("{:>8} {:>10} {:>10} {:>10} {:>8} {:>10} {:>8}",
            "N", "λ₁", "λ₂", "gap", "cycles", "transient", "max_cyc");

        for &n in &sizes {
            let r = spectral_gap_modular(n, 5, 3, 80, 10);
            let max_cyc = r.cycles.cycle_lengths.first().copied().unwrap_or(0);
            eprintln!("{:>8} {:>10.6} {:>10.6} {:>10.6} {:>8} {:>10} {:>8}",
                r.n, r.lambda_1, r.lambda_2, r.gap, r.cycles.cycle_count,
                r.cycles.transient_count, max_cyc);
        }
    }

    // ── Lazy random walk ───────────────────────────────────────────────

    #[test]
    fn lazy_walk_preserves_stochasticity() {
        // Uniform distribution should be preserved (T is column-stochastic)
        let map = SparseDeterministicMap::arithmetic(10, 3, 2);
        let walk = LazyRandomWalk::new(map, 0.5);
        let x = vec![0.1; 10]; // uniform
        let y = walk.mul_vec(&x);
        let sum: f64 = y.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "Should preserve probability sum, got {sum}");
    }

    #[test]
    fn spectral_gap_lazy_walk_collatz() {
        let sizes = vec![100, 500, 1_000, 5_000, 10_000];

        eprintln!("\n{:=^70}", "");
        eprintln!("  LAZY WALK SPECTRAL GAP: Collatz (m=3, d=2, α=0.5)");
        eprintln!("{:=^70}", "");
        eprintln!("{:>8} {:>10} {:>10} {:>10} {:>8} {:>10}",
            "N", "λ₁", "λ₂", "gap", "cycles", "transient");

        for &n in &sizes {
            let r = spectral_gap_lazy_walk(n, 3, 2, 0.5, 80, 10);
            eprintln!("{:>8} {:>10.6} {:>10.6} {:>10.6} {:>8} {:>10}",
                r.n, r.lambda_1, r.lambda_2, r.gap, r.cycles.cycle_count,
                r.cycles.transient_count);
        }
    }

    // ── Hessenberg QR correctness ──────────────────────────────────────

    #[test]
    fn hessenberg_qr_2x2_real() {
        // [[2, 1], [1, 3]] → eigenvalues (3.618.., 1.381..)
        let h = Mat::from_vec(2, 2, vec![2.0, 1.0, 1.0, 3.0]);
        let eigs = hessenberg_qr_eigenvalues(&h);
        let mut mags: Vec<f64> = eigs.iter().map(|&(r, i)| (r*r + i*i).sqrt()).collect();
        mags.sort_by(|a, b| b.total_cmp(a));
        let expected_1 = 2.5 + (1.25_f64).sqrt();
        let expected_2 = 2.5 - (1.25_f64).sqrt();
        assert!((mags[0] - expected_1).abs() < 1e-8, "λ₁={} expected {}", mags[0], expected_1);
        assert!((mags[1] - expected_2).abs() < 1e-8, "λ₂={} expected {}", mags[1], expected_2);
    }

    #[test]
    fn hessenberg_qr_upper_triangular() {
        // Upper triangular: eigenvalues on diagonal
        let h = Mat::from_vec(3, 3, vec![
            5.0, 2.0, 1.0,
            0.0, 3.0, 4.0,
            0.0, 0.0, 1.0,
        ]);
        let eigs = hessenberg_qr_eigenvalues(&h);
        let mut mags: Vec<f64> = eigs.iter().map(|&(r, i)| (r*r + i*i).sqrt()).collect();
        mags.sort_by(|a, b| b.total_cmp(a));
        assert!((mags[0] - 5.0).abs() < 1e-8, "λ₁={}", mags[0]);
        assert!((mags[1] - 3.0).abs() < 1e-8, "λ₂={}", mags[1]);
        assert!((mags[2] - 1.0).abs() < 1e-8, "λ₃={}", mags[2]);
    }
}
