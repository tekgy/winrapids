//! Family 15 — Manifold topology / phase-space geometry.
//!
//! Pure-function bridges for fintek's manifold leaves. All take `&[f64]`
//! log-return slices (cadence-agnostic). Tambear's SVD and sym_eigen power
//! every computation here.
//!
//! Covered leaves:
//! - spectral_embedding (K02P15C2R1)  — fiedler_value, spectral_gap, cheeger, effective_resistance
//! - diff_geometry      (K02P15C3R1)  — mean_curvature, curvature_std, torsion_proxy, arc_length
//! - grassmannian       (K02P15C5R1)  — max_principal_angle, mean_principal_angle, chordal_distance, subspace_drift
//! - rmt                (K02P15C7R1)  — n_signal_eigenvalues, tracy_widom_stat, mp_ratio, spectral_rigidity
//! - embedding          (K02P14C01R01) — optimal_delay, embedding_dim, fnn_fraction, ami_first_min

// ── shared helpers ────────────────────────────────────────────────────────────

/// Build delay-embedded trajectory matrix, with optional downsampling to max_pts.
///
/// Delegates to `tambear::delay_embed` for the core embedding, then applies
/// uniform downsampling to cap the number of points at `max_pts` (for
/// production performance in large bins). Returns (flat row-major matrix, n_rows).
fn delay_embed(returns: &[f64], d: usize, tau: usize, max_pts: usize) -> (Vec<f64>, usize) {
    // Use the tambear global primitive for the embedding
    let embedded = tambear::time_series::delay_embed(returns, d, tau);
    let n_raw = embedded.len();
    if n_raw < 2 { return (Vec::new(), 0); }

    // Apply uniform downsampling if needed
    let step = if n_raw > max_pts { n_raw / max_pts } else { 1 };
    let rows: Vec<usize> = (0..n_raw).step_by(step.max(1)).collect();
    let n_rows = rows.len();
    if n_rows < 2 { return (Vec::new(), 0); }

    let mut mat = Vec::with_capacity(n_rows * d);
    for &t in &rows {
        for &val in &embedded[t] {
            mat.push(val);
        }
    }
    (mat, n_rows)
}

/// Pairwise Euclidean distance matrix for (n × d) row-major matrix.
/// Delegates to tambear::graph::pairwise_dists.
#[inline]
fn pairwise_dists(mat: &[f64], n: usize, d: usize) -> Vec<f64> {
    tambear::graph::pairwise_dists(mat, n, d)
}

/// Build symmetric k-NN Gaussian kernel adjacency from pairwise distances.
/// Delegates to tambear::graph::knn_adjacency.
#[inline]
fn knn_adjacency(dists: &[f64], n: usize, k: usize) -> Vec<f64> {
    tambear::graph::knn_adjacency(dists, n, k)
}

/// Degree-normalized graph Laplacian L = D - A. Returns (lap, degrees, max_degree).
/// Delegates to tambear::graph::graph_laplacian.
#[inline]
fn graph_laplacian(adj: &[f64], n: usize) -> (Vec<f64>, Vec<f64>, f64) {
    tambear::graph::graph_laplacian(adj, n)
}

/// Compute smallest k eigenvalues of a symmetric matrix via tambear sym_eigen.
fn smallest_eigenvalues(mat_flat: &[f64], n: usize, k: usize) -> Vec<f64> {
    if n < 2 || k == 0 { return vec![]; }
    let m = tambear::linear_algebra::Mat::from_vec(n, n, mat_flat.to_vec());
    let (mut vals, _) = tambear::linear_algebra::sym_eigen(&m);
    vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    vals.into_iter().take(k).map(|v| v.max(0.0)).collect()
}

// ── spectral_embedding (K02P15C2R1) ──────────────────────────────────────────

/// Graph Laplacian features from kNN graph of delay-embedded returns.
///
/// Corresponds to fintek's `spectral_embedding.rs` (K02P15C2R1).
#[derive(Debug, Clone)]
pub struct SpectralEmbeddingResult {
    /// 2nd smallest Laplacian eigenvalue (algebraic connectivity).
    pub fiedler_value: f64,
    /// Ratio λ₂/λ₃.
    pub spectral_gap: f64,
    /// Fiedler value / max_degree.
    pub cheeger_proxy: f64,
    /// Σ n/λᵢ for λᵢ > 0 (effective resistance).
    pub effective_resistance: f64,
}

impl SpectralEmbeddingResult {
    pub fn nan() -> Self {
        Self { fiedler_value: f64::NAN, spectral_gap: f64::NAN,
               cheeger_proxy: f64::NAN, effective_resistance: f64::NAN }
    }
}

/// Compute graph Laplacian features from per-bin log-returns.
///
/// Delay-embeds in 3D (tau=1), builds kNN=5 Gaussian kernel graph, computes
/// smallest Laplacian eigenvalues. Caps at 200 embedded points.
pub fn spectral_embedding(returns: &[f64]) -> SpectralEmbeddingResult {
    let d = 3usize;
    let k_nn = 5usize;
    let max_pts = 200usize;

    if returns.len() < 20 + d { return SpectralEmbeddingResult::nan(); }

    let (mat, n) = delay_embed(returns, d, 1, max_pts);
    if n < k_nn + 2 { return SpectralEmbeddingResult::nan(); }

    let dists = pairwise_dists(&mat, n, d);
    let adj = knn_adjacency(&dists, n, k_nn);
    let (lap, _, max_deg) = graph_laplacian(&adj, n);

    let eigs = smallest_eigenvalues(&lap, n, 4);
    if eigs.len() < 2 { return SpectralEmbeddingResult::nan(); }

    let fiedler = eigs[1];
    let spectral_gap = if eigs.len() > 2 && eigs[2] > 1e-30 { eigs[1] / eigs[2] } else { 0.0 };
    let cheeger = if max_deg > 1e-30 { fiedler / max_deg } else { 0.0 };
    let eff_res: f64 = eigs.iter().filter(|&&l| l > 1e-10).map(|l| 1.0 / l).sum::<f64>() * n as f64;

    SpectralEmbeddingResult { fiedler_value: fiedler, spectral_gap, cheeger_proxy: cheeger,
                               effective_resistance: eff_res }
}

// ── diff_geometry (K02P15C3R1) ────────────────────────────────────────────────

/// Differential geometry (curvature/torsion) of delay-embedded returns.
///
/// Corresponds to fintek's `diff_geometry.rs` (K02P15C3R1).
#[derive(Debug, Clone)]
pub struct DiffGeometryResult {
    /// Mean Menger curvature along trajectory.
    pub mean_curvature: f64,
    /// Std of point-wise curvature.
    pub curvature_std: f64,
    /// Mean magnitude of cross-product of consecutive tangents (torsion proxy).
    pub torsion_proxy: f64,
    /// Mean arc length per segment.
    pub arc_length: f64,
}

impl DiffGeometryResult {
    pub fn nan() -> Self {
        Self { mean_curvature: f64::NAN, curvature_std: f64::NAN,
               torsion_proxy: f64::NAN, arc_length: f64::NAN }
    }
}

/// Compute curvature/torsion features from per-bin log-returns.
///
/// Delay-embeds in 3D, computes Menger curvature at each triplet of consecutive
/// embedded points and the cross-product torsion proxy.
pub fn diff_geometry(returns: &[f64]) -> DiffGeometryResult {
    let d = 3usize;
    let max_pts = 2000usize;

    if returns.len() < 10 + d { return DiffGeometryResult::nan(); }

    let (mat, n_emb) = delay_embed(returns, d, 1, max_pts);
    if n_emb < 4 { return DiffGeometryResult::nan(); }

    let pt = |t: usize| -> [f64; 3] {
        [mat[t * d], mat[t * d + 1], mat[t * d + 2]]
    };

    let mut curvatures = Vec::with_capacity(n_emb);
    let mut arc_len = 0.0f64;
    let mut torsion_sum = 0.0f64;
    let mut torsion_count = 0u32;

    for t in 1..n_emb.saturating_sub(1) {
        let a = pt(t - 1);
        let b = pt(t);
        let c = pt(t + 1);

        let ba = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let bc = [c[0] - b[0], c[1] - b[1], c[2] - b[2]];
        let ba_len = (ba[0]*ba[0] + ba[1]*ba[1] + ba[2]*ba[2]).sqrt();
        let bc_len = (bc[0]*bc[0] + bc[1]*bc[1] + bc[2]*bc[2]).sqrt();
        if ba_len < 1e-30 || bc_len < 1e-30 { continue; }

        let dot = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2];
        let cos_theta = (dot / (ba_len * bc_len)).clamp(-1.0, 1.0);
        let sin_theta = (1.0 - cos_theta * cos_theta).max(0.0).sqrt();

        let ac = [c[0]-a[0], c[1]-a[1], c[2]-a[2]];
        let ac_len = (ac[0]*ac[0] + ac[1]*ac[1] + ac[2]*ac[2]).sqrt();
        let kappa = if ac_len > 1e-30 { 2.0 * sin_theta / ac_len } else { 0.0 };
        curvatures.push(kappa);
        arc_len += bc_len;
    }

    // Torsion proxy: cross-product of consecutive tangent differences
    for t in 2..n_emb.saturating_sub(1) {
        let v1 = [mat[t*d] - mat[(t-1)*d], mat[t*d+1] - mat[(t-1)*d+1], mat[t*d+2] - mat[(t-1)*d+2]];
        let v2 = [mat[(t-1)*d] - mat[(t-2)*d], mat[(t-1)*d+1] - mat[(t-2)*d+1], mat[(t-1)*d+2] - mat[(t-2)*d+2]];
        let cross = [v1[1]*v2[2] - v1[2]*v2[1], v1[2]*v2[0] - v1[0]*v2[2], v1[0]*v2[1] - v1[1]*v2[0]];
        let cross_len = (cross[0]*cross[0] + cross[1]*cross[1] + cross[2]*cross[2]).sqrt();
        if cross_len > 1e-30 { torsion_sum += cross_len; torsion_count += 1; }
    }

    if curvatures.is_empty() { return DiffGeometryResult::nan(); }
    let nc = curvatures.len() as f64;
    let mean_k = curvatures.iter().sum::<f64>() / nc;
    let std_k = (curvatures.iter().map(|k| (k - mean_k)*(k - mean_k)).sum::<f64>() / nc).sqrt();

    DiffGeometryResult {
        mean_curvature: mean_k,
        curvature_std: std_k,
        torsion_proxy: if torsion_count > 0 { torsion_sum / torsion_count as f64 } else { 0.0 },
        arc_length: arc_len / (n_emb - 1).max(1) as f64,
    }
}

// ── grassmannian (K02P15C5R1) ────────────────────────────────────────────────

/// Grassmannian (principal angle) features between first and second half subspaces.
///
/// Corresponds to fintek's `grassmannian.rs` (K02P15C5R1).
#[derive(Debug, Clone)]
pub struct GrassmannianResult {
    /// Largest principal angle between first/second half subspaces (radians).
    pub max_principal_angle: f64,
    /// Mean of all principal angles.
    pub mean_principal_angle: f64,
    /// Chordal distance: Frobenius norm of projection matrix difference.
    pub chordal_distance: f64,
    /// 1 - cos(max_principal_angle).
    pub subspace_drift: f64,
}

impl GrassmannianResult {
    pub fn nan() -> Self {
        Self { max_principal_angle: f64::NAN, mean_principal_angle: f64::NAN,
               chordal_distance: f64::NAN, subspace_drift: f64::NAN }
    }
}

/// Top-k right singular vectors of an (n × d) row-major matrix.
fn top_k_right_sv(mat: &[f64], n: usize, d: usize, k: usize) -> Vec<f64> {
    if n < 2 || d < 1 || k == 0 { return vec![0.0; d * k]; }
    let m = tambear::linear_algebra::Mat::from_vec(n, d, mat.to_vec());
    let svd = tambear::linear_algebra::svd(&m);
    // svd.vt is (d × d); rows are right singular vectors
    let k_actual = k.min(svd.sigma.len()).min(d);
    let mut out = vec![0.0f64; d * k_actual];
    for i in 0..k_actual {
        for j in 0..d {
            out[i * d + j] = svd.vt.data[i * d + j];
        }
    }
    out
}

/// Compute Grassmannian features from per-bin log-returns.
///
/// Delay-embeds in 4D, splits into first/second halves, computes SVD-based
/// principal angles between the 2-dimensional subspaces.
pub fn grassmannian(returns: &[f64]) -> GrassmannianResult {
    let d = 4usize;
    let n_comp = 2usize;
    let max_pts = 2000usize;
    let min_n = 30usize;

    if returns.len() < min_n + d { return GrassmannianResult::nan(); }

    let (mat, n_emb) = delay_embed(returns, d, 1, max_pts);
    if n_emb < 4 { return GrassmannianResult::nan(); }

    let half = n_emb / 2;
    if half < 2 { return GrassmannianResult::nan(); }

    let mat1 = &mat[..half * d];
    let mat2 = &mat[half * d..];
    let n2 = n_emb - half;
    if n2 < 2 { return GrassmannianResult::nan(); }

    // Top-2 right singular vectors from each half
    let q1 = top_k_right_sv(mat1, half, d, n_comp);
    let q2 = top_k_right_sv(mat2, n2, d, n_comp);

    // Cross-Gram matrix G = Q1^T Q2 (n_comp × n_comp)
    let mut gram = vec![0.0f64; n_comp * n_comp];
    for i in 0..n_comp {
        for j in 0..n_comp {
            gram[i * n_comp + j] = (0..d).map(|k| q1[i * d + k] * q2[j * d + k]).sum();
        }
    }

    // Singular values of gram = cos of principal angles
    let gram_m = tambear::linear_algebra::Mat::from_vec(n_comp, n_comp, gram);
    let gram_svd = tambear::linear_algebra::svd(&gram_m);
    let cos_angles: Vec<f64> = gram_svd.sigma.iter().map(|&s| s.clamp(0.0, 1.0)).collect();
    let angles: Vec<f64> = cos_angles.iter().map(|&c| c.acos()).collect();

    let max_angle = angles.iter().copied().fold(0.0f64, f64::max);
    let mean_angle = angles.iter().sum::<f64>() / angles.len().max(1) as f64;

    // Chordal distance: sqrt(sum(sin²(θᵢ)))
    let chordal = angles.iter().map(|&a| a.sin() * a.sin()).sum::<f64>().sqrt();
    let subspace_drift = 1.0 - max_angle.cos();

    GrassmannianResult { max_principal_angle: max_angle, mean_principal_angle: mean_angle,
                         chordal_distance: chordal, subspace_drift }
}

// ── rmt (K02P15C7R1) ──────────────────────────────────────────────────────────

/// Random Matrix Theory (Marchenko-Pastur) features.
///
/// Corresponds to fintek's `rmt.rs` (K02P15C7R1).
#[derive(Debug, Clone)]
pub struct RmtResult {
    /// Count of eigenvalues exceeding MP upper edge.
    pub n_signal_eigenvalues: usize,
    /// Standardized λ_max (Johnstone centering/scaling).
    pub tracy_widom_stat: f64,
    /// Fraction of total variance above MP edge.
    pub mp_ratio: f64,
    /// Spacing variance/mean ratio (GOE ≈ 0.27, Poisson ≈ 1).
    pub spectral_rigidity: f64,
}

impl RmtResult {
    pub fn nan() -> Self {
        Self { n_signal_eigenvalues: 0, tracy_widom_stat: f64::NAN,
               mp_ratio: f64::NAN, spectral_rigidity: f64::NAN }
    }
}

/// Compute RMT features from per-bin log-returns.
///
/// Delay-embeds in 8D, computes d×d covariance matrix eigenvalues, compares
/// to Marchenko-Pastur null. Caps at 2000 points.
pub fn rmt(returns: &[f64]) -> RmtResult {
    let d = 8usize;
    let max_pts = 2000usize;
    let min_n = 30usize;

    if returns.len() < min_n + d { return RmtResult::nan(); }

    let (mat, n_emb) = delay_embed(returns, d, 1, max_pts);
    if n_emb < d + 2 { return RmtResult::nan(); }

    let mat_m = tambear::linear_algebra::Mat::from_vec(n_emb, d, mat);
    let cov_m = tambear::multivariate::covariance_matrix(&mat_m, Some(1));
    let (mut eigs, _) = tambear::linear_algebra::sym_eigen(&cov_m);
    eigs.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // Marchenko-Pastur: γ = p/n, σ² = trace(Σ)/p
    let gamma = d as f64 / n_emb as f64;
    let sigma2: f64 = eigs.iter().sum::<f64>() / d as f64;
    if sigma2 < 1e-30 { return RmtResult::nan(); }
    let (_, mp_upper) = tambear::special_functions::marchenko_pastur_bounds(gamma, sigma2);

    let n_signal = eigs.iter().filter(|&&e| e > mp_upper).count();
    let total_var: f64 = eigs.iter().sum();
    let signal_var: f64 = eigs.iter().filter(|&&e| e > mp_upper).sum();
    let mp_ratio = if total_var > 1e-30 { signal_var / total_var } else { 0.0 };

    // Tracy-Widom approximation: Johnstone centering/scaling
    let lambda_max = eigs.last().copied().unwrap_or(0.0);
    let n_f = n_emb as f64;
    let p_f = d as f64;
    let mu = sigma2 * (n_f.sqrt() + p_f.sqrt()).powi(2) / n_f;
    let sigma_tw = sigma2 * (n_f.sqrt() + p_f.sqrt()) * (1.0/n_f.sqrt() + 1.0/p_f.sqrt()).sqrt().cbrt() / n_f;
    let tracy_widom = if sigma_tw > 1e-30 { (lambda_max - mu) / sigma_tw } else { f64::NAN };

    // Spectral rigidity: variance/mean of unfolded spacings
    let spacings: Vec<f64> = eigs.windows(2).map(|w| (w[1] - w[0]).abs()).filter(|&s| s > 0.0).collect();
    let spectral_rigidity = if spacings.len() >= 2 {
        let mean_s = spacings.iter().sum::<f64>() / spacings.len() as f64;
        let var_s = spacings.iter().map(|s| (s - mean_s)*(s - mean_s)).sum::<f64>() / spacings.len() as f64;
        if mean_s > 1e-30 { var_s / mean_s } else { f64::NAN }
    } else { f64::NAN };

    RmtResult { n_signal_eigenvalues: n_signal, tracy_widom_stat: tracy_widom,
                mp_ratio, spectral_rigidity }
}

// ── embedding (K02P14C01R01) ──────────────────────────────────────────────────

/// Optimal delay embedding parameters via AMI first minimum and FNN.
///
/// Corresponds to fintek's `embedding.rs` (K02P14C01R01).
#[derive(Debug, Clone)]
pub struct EmbeddingResult {
    /// Optimal delay tau (AMI first minimum in 1..MAX_TAU).
    pub optimal_delay: usize,
    /// Optimal embedding dimension (FNN drops below 10%).
    pub embedding_dim: usize,
    /// False nearest neighbor fraction at chosen dimension.
    pub fnn_fraction: f64,
    /// AMI value at the first minimum.
    pub ami_first_min: f64,
}

impl EmbeddingResult {
    pub fn nan() -> Self {
        Self { optimal_delay: 1, embedding_dim: 1, fnn_fraction: f64::NAN, ami_first_min: f64::NAN }
    }
}

const AMI_BINS: usize = 16;
const MAX_TAU: usize = 30;
const MAX_DIM: usize = 8;

fn ami_at_lag(x: &[f64], tau: usize) -> f64 {
    let n = x.len().saturating_sub(tau);
    if n < 10 { return 0.0; }
    let min_v = x.iter().copied().fold(f64::INFINITY, f64::min);
    let max_v = x.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let range = (max_v - min_v).max(1e-30);

    let mut hx = vec![0u32; AMI_BINS];
    let mut hy = vec![0u32; AMI_BINS];
    let mut hxy = vec![0u32; AMI_BINS * AMI_BINS];

    for i in 0..n {
        let bx = ((x[i] - min_v) / range * AMI_BINS as f64).floor() as usize;
        let by = ((x[i + tau] - min_v) / range * AMI_BINS as f64).floor() as usize;
        let bx = bx.min(AMI_BINS - 1);
        let by = by.min(AMI_BINS - 1);
        hx[bx] += 1;
        hy[by] += 1;
        hxy[bx * AMI_BINS + by] += 1;
    }

    let nf = n as f64;
    let mut mi = 0.0f64;
    for bx in 0..AMI_BINS {
        for by in 0..AMI_BINS {
            let c = hxy[bx * AMI_BINS + by];
            if c == 0 || hx[bx] == 0 || hy[by] == 0 { continue; }
            let pxy = c as f64 / nf;
            let px = hx[bx] as f64 / nf;
            let py = hy[by] as f64 / nf;
            mi += pxy * (pxy / (px * py)).ln();
        }
    }
    mi
}

fn fnn_frac(x: &[f64], d: usize, tau: usize) -> f64 {
    let n_emb = x.len().saturating_sub(d * tau);
    if n_emb < 10 { return 1.0; }
    let n_cap = n_emb.min(200);
    let rtol = 15.0f64;

    let mut fnn_count = 0u32;
    let mut total = 0u32;

    for i in 0..n_cap {
        let mut best_dist2 = f64::INFINITY;
        let mut best_j = 0usize;
        for j in 0..n_cap {
            if j == i { continue; }
            let mut dist2 = 0.0f64;
            for k in 0..d {
                let diff = x[i + k*tau] - x[j + k*tau];
                dist2 += diff * diff;
            }
            if dist2 < best_dist2 { best_dist2 = dist2; best_j = j; }
        }
        if best_dist2 < 1e-30 { continue; }

        // Check if adding dimension d+1 causes large jump
        let extra_i = if i + d * tau < x.len() { x[i + d * tau] } else { continue };
        let extra_j = if best_j + d * tau < x.len() { x[best_j + d * tau] } else { continue };
        let new_dist2 = best_dist2 + (extra_i - extra_j) * (extra_i - extra_j);

        if new_dist2 / best_dist2 > rtol * rtol { fnn_count += 1; }
        total += 1;
    }

    if total == 0 { 1.0 } else { fnn_count as f64 / total as f64 }
}

/// Compute optimal delay embedding parameters from per-bin log-returns.
pub fn embedding(returns: &[f64]) -> EmbeddingResult {
    if returns.len() < 50 { return EmbeddingResult::nan(); }

    // AMI: find first local minimum in 1..MAX_TAU
    let max_tau_actual = MAX_TAU.min(returns.len() / 5);
    let amis: Vec<f64> = (1..=max_tau_actual).map(|tau| ami_at_lag(returns, tau)).collect();
    let mut opt_tau = 1usize;
    let mut ami_min = amis[0];
    for (i, &val) in amis.iter().enumerate() {
        // First local minimum: ami[i] < ami[i-1] and ami[i] < ami[i+1] (or end)
        let prev_ok = i == 0 || val < amis[i - 1];
        let next_ok = i + 1 >= amis.len() || val < amis[i + 1];
        if prev_ok && next_ok && val < ami_min {
            ami_min = val;
            opt_tau = i + 1;
            break;
        }
    }

    // FNN: find embedding dimension where FNN drops below 10%
    let mut opt_dim = 1usize;
    let mut final_fnn = 1.0f64;
    for d in 1..=MAX_DIM {
        let fnn = fnn_frac(returns, d, opt_tau);
        final_fnn = fnn;
        opt_dim = d;
        if fnn < 0.1 { break; }
    }

    EmbeddingResult { optimal_delay: opt_tau, embedding_dim: opt_dim,
                      fnn_fraction: final_fnn, ami_first_min: ami_min }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── spectral_embedding ────────────────────────────────────────────────────

    #[test]
    fn spectral_embedding_too_short() {
        let r = spectral_embedding(&[0.01, -0.01, 0.005]);
        assert!(r.fiedler_value.is_nan());
    }

    #[test]
    fn spectral_embedding_random_returns_finite() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let returns: Vec<f64> = (0..100).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = spectral_embedding(&returns);
        assert!(r.fiedler_value.is_finite() && r.fiedler_value >= 0.0,
            "fiedler_value = {}", r.fiedler_value);
        assert!(r.spectral_gap.is_finite() && r.spectral_gap >= 0.0,
            "spectral_gap = {}", r.spectral_gap);
        assert!(r.cheeger_proxy.is_finite() && r.cheeger_proxy >= 0.0);
    }

    #[test]
    fn spectral_embedding_fiedler_nonnegative() {
        // Fiedler value ≥ 0 by definition (Laplacian is PSD)
        let mut rng = tambear::rng::Xoshiro256::new(7);
        let returns: Vec<f64> = (0..200).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.02)).collect();
        let r = spectral_embedding(&returns);
        if r.fiedler_value.is_finite() {
            assert!(r.fiedler_value >= -1e-10, "Fiedler must be ≥ 0, got {}", r.fiedler_value);
        }
    }

    // ── diff_geometry ─────────────────────────────────────────────────────────

    #[test]
    fn diff_geometry_too_short() {
        let r = diff_geometry(&[0.01, -0.01, 0.005]);
        assert!(r.mean_curvature.is_nan());
    }

    #[test]
    fn diff_geometry_random_returns_finite() {
        let mut rng = tambear::rng::Xoshiro256::new(13);
        let returns: Vec<f64> = (0..100).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = diff_geometry(&returns);
        assert!(r.mean_curvature.is_finite() && r.mean_curvature >= 0.0,
            "mean_curvature = {}", r.mean_curvature);
        assert!(r.curvature_std.is_finite() && r.curvature_std >= 0.0);
        assert!(r.arc_length.is_finite() && r.arc_length >= 0.0);
    }

    #[test]
    fn diff_geometry_linear_trend_low_curvature() {
        // Linear trend → nearly straight trajectory → low curvature
        let returns: Vec<f64> = (0..100).map(|i| 0.001 * i as f64).collect();
        let r = diff_geometry(&returns);
        if r.mean_curvature.is_finite() {
            assert!(r.mean_curvature < 1.0, "linear trend → low curvature, got {}", r.mean_curvature);
        }
    }

    // ── grassmannian ──────────────────────────────────────────────────────────

    #[test]
    fn grassmannian_too_short() {
        let r = grassmannian(&[0.01; 5]);
        assert!(r.max_principal_angle.is_nan());
    }

    #[test]
    fn grassmannian_random_returns_finite() {
        let mut rng = tambear::rng::Xoshiro256::new(99);
        let returns: Vec<f64> = (0..100).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = grassmannian(&returns);
        assert!(r.max_principal_angle.is_finite(), "max_principal_angle = {}", r.max_principal_angle);
        assert!(r.chordal_distance.is_finite() && r.chordal_distance >= 0.0);
        assert!(r.subspace_drift >= 0.0 && r.subspace_drift <= 2.0);
    }

    #[test]
    fn grassmannian_angles_in_range() {
        // Principal angles must be in [0, π/2]
        let mut rng = tambear::rng::Xoshiro256::new(55);
        let returns: Vec<f64> = (0..200).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = grassmannian(&returns);
        if r.max_principal_angle.is_finite() {
            assert!(r.max_principal_angle >= -1e-10 && r.max_principal_angle <= std::f64::consts::FRAC_PI_2 + 1e-10,
                "angle out of [0, π/2]: {}", r.max_principal_angle);
        }
    }

    // ── rmt ───────────────────────────────────────────────────────────────────

    #[test]
    fn rmt_too_short() {
        let r = rmt(&[0.01; 5]);
        assert!(r.tracy_widom_stat.is_nan());
    }

    #[test]
    fn rmt_random_returns_finite() {
        let mut rng = tambear::rng::Xoshiro256::new(77);
        let returns: Vec<f64> = (0..100).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = rmt(&returns);
        assert!(r.tracy_widom_stat.is_finite(), "tracy_widom = {}", r.tracy_widom_stat);
        assert!(r.mp_ratio >= 0.0 && r.mp_ratio <= 1.0 + 1e-10,
            "mp_ratio = {}", r.mp_ratio);
    }

    #[test]
    fn rmt_iid_no_signal() {
        // IID Gaussian → all eigenvalues within MP bulk → n_signal ≈ 0
        let mut rng = tambear::rng::Xoshiro256::new(123);
        let returns: Vec<f64> = (0..500).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = rmt(&returns);
        // With true IID, n_signal should be small (0-2 typically due to estimation noise)
        assert!(r.n_signal_eigenvalues <= 3,
            "IID data shouldn't have many signal eigenvalues, got {}", r.n_signal_eigenvalues);
    }

    // ── embedding ─────────────────────────────────────────────────────────────

    #[test]
    fn embedding_too_short() {
        let r = embedding(&[0.01; 10]);
        assert!(r.fnn_fraction.is_nan());
    }

    #[test]
    fn embedding_random_returns_finite() {
        let mut rng = tambear::rng::Xoshiro256::new(42);
        let returns: Vec<f64> = (0..100).map(|_| tambear::rng::sample_normal(&mut rng, 0.0, 0.01)).collect();
        let r = embedding(&returns);
        assert!(r.optimal_delay >= 1 && r.optimal_delay <= MAX_TAU,
            "optimal_delay = {}", r.optimal_delay);
        assert!(r.embedding_dim >= 1 && r.embedding_dim <= MAX_DIM,
            "embedding_dim = {}", r.embedding_dim);
        assert!(r.fnn_fraction >= 0.0 && r.fnn_fraction <= 1.0,
            "fnn_fraction = {}", r.fnn_fraction);
    }

    #[test]
    fn embedding_sine_low_dim() {
        // Pure sine wave → low embedding dimension (1-2)
        let n = 200;
        let returns: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin() * 0.01).collect();
        let r = embedding(&returns);
        assert!(r.embedding_dim <= 4,
            "sine should need low dim, got {}", r.embedding_dim);
    }
}
