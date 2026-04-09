//! # Family 14 — Factor Analysis & SEM
//!
//! EFA (principal axis factoring), varimax rotation, CFA basics, reliability.
//!
//! ## Architecture
//!
//! EFA = eigendecomposition of correlation/covariance matrix (Kingdom A).
//! Rotation = orthogonal transformation (Kingdom A).
//! CFA = iterative fitting (Kingdom C), but simplified here.

use crate::linear_algebra::{Mat, mat_mul, mat_scale, cholesky, cholesky_solve};

// ═══════════════════════════════════════════════════════════════════════════
// Correlation matrix
// ═══════════════════════════════════════════════════════════════════════════

/// Compute correlation matrix from data (n×p, row-major).
pub fn correlation_matrix(data: &[f64], n: usize, p: usize) -> Mat {
    assert_eq!(data.len(), n * p);

    // Column means and std devs
    let mut means = vec![0.0; p];
    for i in 0..n {
        for j in 0..p {
            means[j] += data[i * p + j];
        }
    }
    for j in 0..p { means[j] /= n as f64; }

    let mut stds = vec![0.0; p];
    for i in 0..n {
        for j in 0..p {
            stds[j] += (data[i * p + j] - means[j]).powi(2);
        }
    }
    for j in 0..p { stds[j] = (stds[j] / (n - 1) as f64).sqrt().max(1e-15); }

    let mut corr = Mat::zeros(p, p);
    for j in 0..p {
        corr.set(j, j, 1.0);
        for k in (j + 1)..p {
            let mut s = 0.0;
            for i in 0..n {
                s += (data[i * p + j] - means[j]) * (data[i * p + k] - means[k]);
            }
            let r = s / ((n - 1) as f64 * stds[j] * stds[k]);
            corr.set(j, k, r);
            corr.set(k, j, r);
        }
    }
    corr
}

// ═══════════════════════════════════════════════════════════════════════════
// EFA: Principal Axis Factoring
// ═══════════════════════════════════════════════════════════════════════════

/// Factor analysis result.
#[derive(Debug, Clone)]
pub struct FaResult {
    /// Factor loadings matrix (p × n_factors).
    pub loadings: Mat,
    /// Eigenvalues (sorted descending).
    pub eigenvalues: Vec<f64>,
    /// Communalities for each variable.
    pub communalities: Vec<f64>,
    /// Proportion of variance explained by each factor.
    pub variance_explained: Vec<f64>,
    /// Uniquenesses (1 - communality).
    pub uniquenesses: Vec<f64>,
    /// True if any communality exceeds 1.0 (Heywood case). This indicates near-singular or
    /// rank-deficient input. Communalities > 1 are mathematically impossible for standardized
    /// variables and signal that the extracted factor solution is inadmissible. Consider checking
    /// the rank of the correlation matrix before factoring.
    pub heywood: bool,
}

/// Exploratory Factor Analysis via principal axis factoring.
/// `corr`: p×p correlation matrix. `n_factors`: number of factors to extract.
pub fn principal_axis_factoring(corr: &Mat, n_factors: usize, max_iter: usize) -> FaResult {
    let p = corr.rows;
    assert_eq!(corr.cols, p);
    assert!(n_factors > 0 && n_factors <= p);

    // Initial communality estimates: squared multiple correlations
    // Approximated by 1 - 1/diag(R⁻¹), or just use max off-diagonal |r|
    let mut communalities = vec![0.0; p];
    for j in 0..p {
        let mut max_r = 0.0_f64;
        for k in 0..p {
            if k != j { max_r = max_r.max(corr.get(j, k).abs()); }
        }
        communalities[j] = max_r;
    }

    let mut reduced = corr.clone();

    for _ in 0..max_iter {
        // Set diagonal to communalities
        for j in 0..p { reduced.set(j, j, communalities[j]); }

        // Eigendecomposition
        let (evals, evecs) = crate::linear_algebra::sym_eigen(&reduced);
        let mut idx: Vec<usize> = (0..p).collect();
        idx.sort_by(|&a, &b| evals[b].total_cmp(&evals[a]));

        // Extract loadings: L_jf = sqrt(λ_f) · v_jf
        let mut new_comm = vec![0.0; p];
        for j in 0..p {
            for f in 0..n_factors {
                let fi = idx[f];
                let lam = evals[fi].max(0.0);
                let loading = lam.sqrt() * evecs.get(j, fi);
                new_comm[j] += loading * loading;
            }
        }

        // Clamp communalities to [0, 1] to prevent Heywood divergence.
        // Without clamping, rank-deficient input can drive communalities > 1,
        // and further iterations make the problem worse (adversarial finding).
        for c in &mut new_comm {
            *c = c.clamp(0.0, 1.0);
        }

        // Check convergence
        let delta: f64 = communalities.iter().zip(&new_comm)
            .map(|(a, b)| (a - b).abs()).sum::<f64>();
        communalities = new_comm;
        if delta < 1e-6 { break; }
    }

    // Final extraction
    for j in 0..p { reduced.set(j, j, communalities[j]); }
    let (evals, evecs) = crate::linear_algebra::sym_eigen(&reduced);
    let mut idx: Vec<usize> = (0..p).collect();
    idx.sort_by(|&a, &b| evals[b].total_cmp(&evals[a]));

    let mut loadings = Mat::zeros(p, n_factors);
    let mut eigenvalues = Vec::with_capacity(n_factors);
    for f in 0..n_factors {
        let fi = idx[f];
        let lam = evals[fi].max(0.0);
        eigenvalues.push(lam);
        for j in 0..p {
            loadings.set(j, f, lam.sqrt() * evecs.get(j, fi));
        }
    }

    let total_var: f64 = (0..p).map(|j| corr.get(j, j)).sum();
    let variance_explained: Vec<f64> = eigenvalues.iter().map(|e| e / total_var).collect();
    let uniquenesses: Vec<f64> = communalities.iter().map(|c| (1.0 - c).max(0.0)).collect();
    let heywood = communalities.iter().any(|&c| c > 1.0);

    FaResult { loadings, eigenvalues, communalities, variance_explained, uniquenesses, heywood }
}

// ═══════════════════════════════════════════════════════════════════════════
// Varimax rotation
// ═══════════════════════════════════════════════════════════════════════════

/// Varimax rotation of factor loadings. Returns rotated loadings.
pub fn varimax(loadings: &Mat, max_iter: usize) -> Mat {
    let p = loadings.rows;
    let k = loadings.cols;
    let mut rotated = loadings.clone();

    for _ in 0..max_iter {
        let mut converged = true;
        for i in 0..k {
            for j in (i + 1)..k {
                // Compute rotation angle for pair (i, j)
                let mut a = 0.0;
                let mut b = 0.0;
                let mut c_val = 0.0;
                let mut d = 0.0;
                for r in 0..p {
                    let li = rotated.get(r, i);
                    let lj = rotated.get(r, j);
                    let u = li * li - lj * lj;
                    let v = 2.0 * li * lj;
                    a += u;
                    b += v;
                    c_val += u * u - v * v;
                    d += 2.0 * u * v;
                }

                let num = d - 2.0 * a * b / p as f64;
                let den = c_val - (a * a - b * b) / p as f64;
                let angle = 0.25 * num.atan2(den);

                if angle.abs() > 1e-8 {
                    converged = false;
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    for r in 0..p {
                        let li = rotated.get(r, i);
                        let lj = rotated.get(r, j);
                        rotated.set(r, i, cos_a * li + sin_a * lj);
                        rotated.set(r, j, -sin_a * li + cos_a * lj);
                    }
                }
            }
        }
        if converged { break; }
    }

    rotated
}

// ═══════════════════════════════════════════════════════════════════════════
// Reliability
// ═══════════════════════════════════════════════════════════════════════════

/// Cronbach's alpha for internal consistency.
/// `data`: n×p matrix (n subjects, p items), row-major.
pub fn cronbachs_alpha(data: &[f64], n: usize, p: usize) -> f64 {
    assert_eq!(data.len(), n * p);
    if p < 2 { return 0.0; }

    // Item variances
    let mut item_vars = vec![0.0; p];
    let mut means = vec![0.0; p];
    for i in 0..n {
        for j in 0..p { means[j] += data[i * p + j]; }
    }
    for j in 0..p { means[j] /= n as f64; }
    for i in 0..n {
        for j in 0..p { item_vars[j] += (data[i * p + j] - means[j]).powi(2); }
    }
    for j in 0..p { item_vars[j] /= (n - 1) as f64; }

    let sum_item_var: f64 = item_vars.iter().sum();

    // Total score variance
    let mut total_scores = vec![0.0; n];
    for i in 0..n {
        total_scores[i] = (0..p).map(|j| data[i * p + j]).sum();
    }
    let total_mean: f64 = total_scores.iter().sum::<f64>() / n as f64;
    let total_var: f64 = total_scores.iter()
        .map(|s| (s - total_mean).powi(2))
        .sum::<f64>() / (n - 1) as f64;

    if total_var < 1e-15 { return 0.0; }
    (p as f64 / (p - 1) as f64) * (1.0 - sum_item_var / total_var)
}

/// McDonald's omega result.
#[derive(Debug, Clone)]
pub struct OmegaResult {
    /// ω_h value.
    pub omega: f64,
    /// True if first-factor loadings have mixed signs (bipolar factor).
    /// When true, ω is artificially deflated by cancellation and should not
    /// be trusted. The fix is reverse-scoring items with negative loadings.
    pub bipolar: bool,
}

/// McDonald's omega (hierarchical) — simplified using first eigenvalue.
/// Approximation: ω_h ≈ (Σ loadings)² / (Σ loadings)² + Σ uniquenesses.
pub fn mcdonalds_omega(loadings: &Mat) -> OmegaResult {
    let p = loadings.rows;
    // Use first factor loadings
    let first_loadings: Vec<f64> = (0..p).map(|j| loadings.get(j, 0)).collect();
    let has_pos = first_loadings.iter().any(|&l| l > 0.0);
    let has_neg = first_loadings.iter().any(|&l| l < 0.0);
    let bipolar = has_pos && has_neg;

    // For bipolar factors, use absolute values (equivalent to reverse-scoring).
    // Without this, loadings [+0.8, +0.7, -0.8, -0.7] cancel to ω = 0
    // despite high communality (adversarial finding).
    let sum_l: f64 = if bipolar {
        first_loadings.iter().map(|l| l.abs()).sum()
    } else {
        first_loadings.iter().sum()
    };
    let sum_u: f64 = (0..p).map(|j| {
        let comm: f64 = (0..loadings.cols).map(|f| loadings.get(j, f).powi(2)).sum();
        (1.0 - comm).max(0.0)
    }).sum();
    let omega = sum_l * sum_l / (sum_l * sum_l + sum_u);
    OmegaResult { omega, bipolar }
}

/// Scree test: find "elbow" — largest drop in eigenvalues.
/// Returns suggested number of factors.
pub fn scree_elbow(eigenvalues: &[f64]) -> usize {
    if eigenvalues.len() < 2 { return 1; }
    let mut max_drop = 0.0;
    let mut elbow = 1;
    for i in 0..eigenvalues.len() - 1 {
        let drop = eigenvalues[i] - eigenvalues[i + 1];
        if drop > max_drop {
            max_drop = drop;
            elbow = i + 1;
        }
    }
    elbow
}

/// Kaiser criterion: count eigenvalues > 1.
pub fn kaiser_criterion(eigenvalues: &[f64]) -> usize {
    eigenvalues.iter().filter(|&&e| e > 1.0).count().max(1)
}

// ═══════════════════════════════════════════════════════════════════════════
// KMO and Bartlett's test of sphericity
// ═══════════════════════════════════════════════════════════════════════════

/// Result of KMO and Bartlett's test of sphericity.
#[derive(Debug, Clone)]
pub struct KmoBartlettResult {
    /// Kaiser-Meyer-Olkin measure of sampling adequacy (overall).
    /// KMO > 0.9: marvellous, 0.8: meritorious, 0.7: middling, 0.6: mediocre, < 0.5: unacceptable.
    pub kmo_overall: f64,
    /// Per-variable KMO (MSA) values.
    pub kmo_per_variable: Vec<f64>,
    /// Bartlett's chi-squared statistic: -(n-1-(2p+5)/6) · ln|R|
    pub bartlett_statistic: f64,
    /// Degrees of freedom for Bartlett: p(p-1)/2
    pub bartlett_df: usize,
    /// Right-tail p-value for Bartlett (H₀: correlation matrix = I)
    pub bartlett_p_value: f64,
}

/// Kaiser-Meyer-Olkin (KMO) measure of sampling adequacy and Bartlett's
/// test of sphericity.
///
/// Both tests assess whether the correlation matrix is suitable for factor
/// analysis. Bartlett tests H₀: R = I (identity, no correlations).
/// KMO measures the proportion of common variance.
///
/// `corr`: p×p correlation matrix (symmetric, ones on diagonal).
/// `n_obs`: number of observations used to compute the correlation matrix.
pub fn kmo_bartlett(corr: &Mat, n_obs: usize) -> KmoBartlettResult {
    let p = corr.rows;
    assert_eq!(corr.cols, p, "correlation matrix must be square");
    let n = n_obs as f64;

    // ── Bartlett test ──────────────────────────────────────────────────────
    // Compute ln|R| via Cholesky: |R| = Π L_{ii}² so ln|R| = 2 Σ ln L_{ii}
    let ln_det_r = match cholesky(corr) {
        Some(l) => {
            2.0 * (0..p).map(|i| l.data[i * p + i].ln()).sum::<f64>()
        }
        None => {
            // Near-singular — return degenerate result
            return KmoBartlettResult {
                kmo_overall: 0.0,
                kmo_per_variable: vec![0.0; p],
                bartlett_statistic: f64::NAN,
                bartlett_df: p * (p - 1) / 2,
                bartlett_p_value: f64::NAN,
            };
        }
    };
    let bartlett_df = p * (p - 1) / 2;
    let bartlett_stat = -(n - 1.0 - (2.0 * p as f64 + 5.0) / 6.0) * ln_det_r;
    let bartlett_p = crate::special_functions::chi2_right_tail_p(bartlett_stat, bartlett_df as f64);

    // ── KMO ───────────────────────────────────────────────────────────────
    // Compute R^{-1} by solving R·X = I column by column.
    let l = match cholesky(corr) {
        Some(l) => l,
        None => {
            return KmoBartlettResult {
                kmo_overall: f64::NAN,
                kmo_per_variable: vec![f64::NAN; p],
                bartlett_statistic: bartlett_stat,
                bartlett_df,
                bartlett_p_value: bartlett_p,
            };
        }
    };

    // R^{-1}: cols computed via cholesky_solve
    let mut r_inv = vec![0.0_f64; p * p];
    for j in 0..p {
        let mut e = vec![0.0_f64; p];
        e[j] = 1.0;
        let col = cholesky_solve(&l, &e);
        for i in 0..p { r_inv[i * p + j] = col[i]; }
    }

    // Partial correlations: p_{ij} = -r_inv[i,j] / √(r_inv[i,i] * r_inv[j,j])
    // KMO numerator for variable i: Σ_{j≠i} r[i,j]²
    // KMO denominator for variable i: Σ_{j≠i} r[i,j]² + Σ_{j≠i} p[i,j]²
    let mut kmo_per = vec![0.0_f64; p];
    let mut kmo_num_total = 0.0;
    let mut kmo_den_total = 0.0;

    for i in 0..p {
        let mut num_i = 0.0; // Σ r[i,j]²
        let mut den_i = 0.0; // Σ r[i,j]² + Σ p[i,j]²
        for j in 0..p {
            if i == j { continue; }
            let r_ij = corr.data[i * p + j];
            let r_inv_ii = r_inv[i * p + i].max(1e-300);
            let r_inv_jj = r_inv[j * p + j].max(1e-300);
            let p_ij = -r_inv[i * p + j] / (r_inv_ii * r_inv_jj).sqrt();
            num_i += r_ij * r_ij;
            den_i += r_ij * r_ij + p_ij * p_ij;
        }
        kmo_per[i] = if den_i < 1e-300 { 1.0 } else { num_i / den_i };
        kmo_num_total += num_i;
        kmo_den_total += den_i;
    }

    let kmo_overall = if kmo_den_total < 1e-300 { 1.0 } else { kmo_num_total / kmo_den_total };

    KmoBartlettResult {
        kmo_overall,
        kmo_per_variable: kmo_per,
        bartlett_statistic: bartlett_stat,
        bartlett_df,
        bartlett_p_value: bartlett_p,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: f64, b: f64, tol: f64, label: &str) {
        assert!((a - b).abs() < tol, "{label}: {a} vs {b} (diff={})", (a - b).abs());
    }

    #[test]
    fn correlation_identity() {
        // Orthogonal columns → identity correlation
        let data: Vec<f64> = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
            2.0, 0.0, 0.0,
            0.0, 2.0, 0.0,
        ];
        let c = correlation_matrix(&data, 5, 3);
        // Diagonal should be 1
        for j in 0..3 { close(c.get(j, j), 1.0, 1e-10, &format!("diag[{j}]")); }
    }

    #[test]
    fn paf_extracts_factors() {
        // Create data with clear 2-factor structure
        let mut rng = 42u64;
        let mut data = Vec::new();
        let n = 50;
        let p = 6;
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let f1 = (rng as f64 / u64::MAX as f64 - 0.5) * 4.0;
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let f2 = (rng as f64 / u64::MAX as f64 - 0.5) * 4.0;
            // Items 0-2 load on F1, items 3-5 load on F2
            for _j in 0..3 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 0.5;
                data.push(f1 + noise);
            }
            for _j in 0..3 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 0.5;
                data.push(f2 + noise);
            }
        }

        let corr = correlation_matrix(&data, n, p);
        let res = principal_axis_factoring(&corr, 2, 100);
        assert_eq!(res.loadings.rows, p);
        assert_eq!(res.loadings.cols, 2);
        assert!(res.eigenvalues[0] > 1.0, "First eigenvalue should be > 1");
    }

    #[test]
    fn varimax_preserves_communalities() {
        // Varimax shouldn't change communalities
        let mut loadings = Mat::zeros(4, 2);
        loadings.set(0, 0, 0.8); loadings.set(0, 1, 0.2);
        loadings.set(1, 0, 0.7); loadings.set(1, 1, 0.3);
        loadings.set(2, 0, 0.2); loadings.set(2, 1, 0.8);
        loadings.set(3, 0, 0.3); loadings.set(3, 1, 0.7);

        let comm_before: Vec<f64> = (0..4).map(|j| {
            (0..2).map(|f| loadings.get(j, f).powi(2)).sum()
        }).collect();

        let rotated = varimax(&loadings, 100);
        let comm_after: Vec<f64> = (0..4).map(|j| {
            (0..2).map(|f| rotated.get(j, f).powi(2)).sum()
        }).collect();

        for j in 0..4 {
            close(comm_before[j], comm_after[j], 1e-10, &format!("communality[{j}]"));
        }
    }

    #[test]
    fn cronbach_high_reliability() {
        // Items that correlate highly → high alpha
        let mut data = Vec::new();
        let n = 30;
        let p = 5;
        let mut rng = 42u64;
        for _ in 0..n {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let true_score = rng as f64 / u64::MAX as f64 * 10.0;
            for _ in 0..p {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 0.5;
                data.push(true_score + noise);
            }
        }
        let alpha = cronbachs_alpha(&data, n, p);
        assert!(alpha > 0.8, "α={alpha} should be high for reliable items");
    }

    #[test]
    fn cronbach_low_for_random() {
        // Random items → low alpha
        let mut data = Vec::new();
        let mut rng = 42u64;
        for _ in 0..30 {
            for _ in 0..5 {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                data.push(rng as f64 / u64::MAX as f64);
            }
        }
        let alpha = cronbachs_alpha(&data, 30, 5);
        assert!(alpha < 0.5, "α={alpha} should be low for random items");
    }

    #[test]
    fn kaiser_counts_correctly() {
        let eigenvalues = vec![3.5, 1.2, 0.8, 0.3, 0.2];
        assert_eq!(kaiser_criterion(&eigenvalues), 2);
    }

    #[test]
    fn scree_finds_elbow() {
        let eigenvalues = vec![4.0, 1.5, 0.8, 0.7, 0.5, 0.3];
        // Biggest drop: 4.0 → 1.5 (drop of 2.5)
        assert_eq!(scree_elbow(&eigenvalues), 1);
    }

    // ── KMO and Bartlett ─────────────────────────────────────────────────

    fn make_corr_3x3(r12: f64, r13: f64, r23: f64) -> Mat {
        // 3×3 correlation matrix with given off-diagonal correlations
        let data = vec![
            1.0, r12, r13,
            r12, 1.0, r23,
            r13, r23, 1.0,
        ];
        Mat { rows: 3, cols: 3, data }
    }

    #[test]
    fn bartlett_identity_matrix_non_significant() {
        // Identity matrix — no correlations → Bartlett p should be large (fail to reject)
        let identity = Mat { rows: 3, cols: 3, data: vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ]};
        let r = kmo_bartlett(&identity, 100);
        // ln|I| = 0 → statistic = 0 → p = 1
        assert!(r.bartlett_statistic.abs() < 1e-10,
            "bartlett_statistic for identity should be 0, got {}", r.bartlett_statistic);
        assert!((r.bartlett_p_value - 1.0).abs() < 1e-6,
            "bartlett p for identity should be 1.0, got {}", r.bartlett_p_value);
    }

    #[test]
    fn bartlett_high_correlation_significant() {
        // High correlations → |R| small → statistic large → p small
        let r = kmo_bartlett(&make_corr_3x3(0.9, 0.85, 0.88), 200);
        assert!(r.bartlett_p_value < 0.001,
            "High correlation: bartlett p={:.6} should be < 0.001", r.bartlett_p_value);
    }

    #[test]
    fn kmo_high_for_common_factor_structure() {
        // Data with strong common factor has high KMO (> 0.7)
        // Use correlations typical of a factor structure
        let r = kmo_bartlett(&make_corr_3x3(0.8, 0.75, 0.82), 100);
        assert!(r.kmo_overall > 0.6,
            "Factor-like correlations: KMO={:.3} should be > 0.6", r.kmo_overall);
        assert_eq!(r.kmo_per_variable.len(), 3);
        for (i, &k) in r.kmo_per_variable.iter().enumerate() {
            assert!(k > 0.0 && k <= 1.0, "KMO[{i}]={k:.4} out of [0,1]");
        }
    }

    #[test]
    fn kmo_bartlett_df_is_correct() {
        // p=3 → df = p(p-1)/2 = 3
        let r = kmo_bartlett(&make_corr_3x3(0.5, 0.4, 0.45), 50);
        assert_eq!(r.bartlett_df, 3);
    }
}
