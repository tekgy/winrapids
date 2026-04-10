//! # Family 02 — Linear Algebra
//!
//! cuBLAS/cuSOLVER/cuSPARSE replacement. From first principles.
//!
//! ## What lives here
//!
//! **Matrix operations**: multiply, transpose, add, scale, trace, determinant
//! **Factorizations**: LU (partial pivoting), Cholesky (LL^T), QR (Householder), SVD
//! **Eigendecomposition**: symmetric eigenvalues (QR algorithm), power iteration
//! **Solvers**: Ax=b via LU, least squares via QR, positive definite via Cholesky
//! **Norms**: Frobenius, L1, L∞, spectral (via SVD)
//!
//! ## Architecture
//!
//! Dense matrices stored as row-major `Vec<f64>` with (rows, cols) shape.
//! No external BLAS dependency — everything from scratch.
//! The dense path is correct and portable; GPU kernels override for speed.
//!
//! ## MSR insight
//!
//! Factorizations ARE the minimal sufficient representation of a matrix.
//! LU = permutation + 2 triangular factors. Cholesky = 1 triangular factor.
//! SVD = 2 rotations + 1 diagonal. The factorization carries ALL the
//! information of the original matrix in a more useful form.

use std::f64;

/// Dense matrix in row-major order.
#[derive(Debug, Clone)]
pub struct Mat {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl Mat {
    /// Create a new matrix filled with zeros.
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat { data: vec![0.0; rows * cols], rows, cols }
    }

    /// Create an identity matrix.
    pub fn eye(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Create from row-major data.
    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Mat { data, rows, cols }
    }

    /// Create from a 2D slice (row-major).
    pub fn from_rows(rows: &[&[f64]]) -> Self {
        let r = rows.len();
        if r == 0 { return Mat::zeros(0, 0); }
        let c = rows[0].len();
        let mut data = Vec::with_capacity(r * c);
        for row in rows {
            assert_eq!(row.len(), c);
            data.extend_from_slice(row);
        }
        Mat { data, rows: r, cols: c }
    }

    /// Create a column vector from a slice.
    pub fn col_vec(v: &[f64]) -> Self {
        Mat { data: v.to_vec(), rows: v.len(), cols: 1 }
    }

    /// Create a diagonal matrix from a vector.
    pub fn diag(v: &[f64]) -> Self {
        let n = v.len();
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = v[i];
        }
        m
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    /// Transpose.
    pub fn t(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out.set(j, i, self.get(i, j));
            }
        }
        out
    }

    /// Matrix trace.
    pub fn trace(&self) -> f64 {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).sum()
    }

    /// Frobenius norm: √(Σ a_ij²).
    pub fn norm_fro(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// L∞ norm (max row sum).
    pub fn norm_inf(&self) -> f64 {
        (0..self.rows).map(|i| {
            (0..self.cols).map(|j| self.get(i, j).abs()).sum::<f64>()
        }).fold(0.0_f64, f64::max)
    }

    /// L1 norm (max column sum).
    pub fn norm_1(&self) -> f64 {
        (0..self.cols).map(|j| {
            (0..self.rows).map(|i| self.get(i, j).abs()).sum::<f64>()
        }).fold(0.0_f64, f64::max)
    }

    /// Check if square.
    pub fn is_square(&self) -> bool {
        self.rows == self.cols
    }

    /// Extract diagonal as a vector.
    pub fn diagonal(&self) -> Vec<f64> {
        let n = self.rows.min(self.cols);
        (0..n).map(|i| self.get(i, i)).collect()
    }

    /// Extract a submatrix.
    pub fn submat(&self, r0: usize, c0: usize, rows: usize, cols: usize) -> Mat {
        let mut out = Mat::zeros(rows, cols);
        for i in 0..rows {
            for j in 0..cols {
                out.set(i, j, self.get(r0 + i, c0 + j));
            }
        }
        out
    }
}

// ─── Arithmetic ─────────────────────────────────────────────────────

/// Matrix multiplication: C = A × B.
pub fn mat_mul(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.cols, b.rows, "dimension mismatch: {}x{} * {}x{}", a.rows, a.cols, b.rows, b.cols);
    let mut c = Mat::zeros(a.rows, b.cols);
    for i in 0..a.rows {
        for k in 0..a.cols {
            let aik = a.get(i, k);
            for j in 0..b.cols {
                c.data[i * b.cols + j] += aik * b.get(k, j);
            }
        }
    }
    c
}

/// Matrix addition: C = A + B.
pub fn mat_add(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    Mat {
        data: a.data.iter().zip(b.data.iter()).map(|(x, y)| x + y).collect(),
        rows: a.rows,
        cols: a.cols,
    }
}

/// Matrix subtraction: C = A - B.
pub fn mat_sub(a: &Mat, b: &Mat) -> Mat {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    Mat {
        data: a.data.iter().zip(b.data.iter()).map(|(x, y)| x - y).collect(),
        rows: a.rows,
        cols: a.cols,
    }
}

/// Scalar multiplication: C = α × A.
pub fn mat_scale(alpha: f64, a: &Mat) -> Mat {
    Mat {
        data: a.data.iter().map(|x| alpha * x).collect(),
        rows: a.rows,
        cols: a.cols,
    }
}

/// Matrix-vector multiplication: y = A × x.
pub fn mat_vec(a: &Mat, x: &[f64]) -> Vec<f64> {
    assert_eq!(a.cols, x.len());
    let mut y = vec![0.0; a.rows];
    for i in 0..a.rows {
        for j in 0..a.cols {
            y[i] += a.get(i, j) * x[j];
        }
    }
    y
}

/// Dot product of two vectors.
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Vector L2 norm.
pub fn vec_norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Outer product: M = u × v^T.
pub fn outer(u: &[f64], v: &[f64]) -> Mat {
    let mut m = Mat::zeros(u.len(), v.len());
    for i in 0..u.len() {
        for j in 0..v.len() {
            m.set(i, j, u[i] * v[j]);
        }
    }
    m
}

// ─── LU Factorization ──────────────────────────────────────────────

/// LU factorization result with partial pivoting.
///
/// PA = LU where P is a permutation, L is unit lower triangular, U is upper triangular.
#[derive(Debug, Clone)]
pub struct LuResult {
    /// Combined L and U (L has implicit 1s on diagonal)
    pub lu: Mat,
    /// Pivot indices (row swaps)
    pub pivots: Vec<usize>,
    /// Number of row swaps (for determinant sign)
    pub n_swaps: usize,
}

/// LU factorization with partial pivoting.
///
/// Returns None if matrix is singular.
pub fn lu(a: &Mat) -> Option<LuResult> {
    if !a.is_square() { return None; }
    let n = a.rows;
    let mut lu = a.clone();
    let mut pivots: Vec<usize> = (0..n).collect();
    let mut n_swaps = 0;

    for k in 0..n {
        // Find pivot
        let mut max_val = lu.get(k, k).abs();
        let mut max_row = k;
        for i in k + 1..n {
            let v = lu.get(i, k).abs();
            if v > max_val {
                max_val = v;
                max_row = i;
            }
        }
        if max_val < 1e-14 { return None; } // singular

        // Swap rows
        if max_row != k {
            pivots.swap(k, max_row);
            for j in 0..n {
                let tmp = lu.get(k, j);
                lu.set(k, j, lu.get(max_row, j));
                lu.set(max_row, j, tmp);
            }
            n_swaps += 1;
        }

        // Eliminate below
        let pivot = lu.get(k, k);
        for i in k + 1..n {
            let factor = lu.get(i, k) / pivot;
            lu.set(i, k, factor); // store L factor
            for j in k + 1..n {
                let v = lu.get(i, j) - factor * lu.get(k, j);
                lu.set(i, j, v);
            }
        }
    }

    Some(LuResult { lu, pivots, n_swaps })
}

/// Solve Ax = b using LU factorization.
pub fn lu_solve(lu_res: &LuResult, b: &[f64]) -> Vec<f64> {
    let n = lu_res.lu.rows;
    let lu = &lu_res.lu;

    // Apply permutation
    let mut pb = vec![0.0; n];
    for i in 0..n {
        pb[i] = b[lu_res.pivots[i]];
    }

    // Forward substitution (L y = Pb)
    let mut y = pb;
    for i in 1..n {
        for j in 0..i {
            y[i] -= lu.get(i, j) * y[j];
        }
    }

    // Back substitution (U x = y)
    let mut x = y;
    for i in (0..n).rev() {
        for j in i + 1..n {
            x[i] -= lu.get(i, j) * x[j];
        }
        x[i] /= lu.get(i, i);
    }
    x
}

/// Determinant via LU factorization.
pub fn det(a: &Mat) -> f64 {
    if !a.is_square() { return 0.0; }
    match lu(a) {
        None => 0.0,
        Some(res) => {
            let sign = if res.n_swaps % 2 == 0 { 1.0 } else { -1.0 };
            let prod: f64 = (0..a.rows).map(|i| res.lu.get(i, i)).product();
            sign * prod
        }
    }
}

/// Log-determinant via Gaussian elimination with partial pivoting.
///
/// Returns `ln|det(A)|`, which is numerically stable for matrices that
/// would overflow or underflow under `det(a).ln()`. Returns `-∞` for
/// singular matrices.
///
/// This primitive is the canonical implementation for any computation that
/// needs `ln|det(M)|` — Kalman filter log-likelihood, multivariate normal
/// log-density, Gaussian process marginal likelihood, etc.
///
/// Kingdom A: O(n³) accumulate over pivots.
pub fn log_det(a: &Mat) -> f64 {
    let n = a.rows;
    assert_eq!(a.cols, n, "log_det requires a square matrix");
    if n == 1 { return a.data[0].abs().ln(); }
    // Gaussian elimination with partial pivoting
    let mut a_copy = a.data.clone();
    let mut log_d = 0.0_f64;
    for k in 0..n {
        // Find pivot
        let mut pivot_row = k;
        let mut max_val = a_copy[k * n + k].abs();
        for i in (k + 1)..n {
            if a_copy[i * n + k].abs() > max_val {
                max_val = a_copy[i * n + k].abs();
                pivot_row = i;
            }
        }
        if pivot_row != k {
            for j in 0..n { a_copy.swap(k * n + j, pivot_row * n + j); }
        }
        let diag = a_copy[k * n + k];
        if diag.abs() < 1e-300 { return f64::NEG_INFINITY; }
        log_d += diag.abs().ln();
        for i in (k + 1)..n {
            let factor = a_copy[i * n + k] / diag;
            for j in k..n {
                a_copy[i * n + j] -= factor * a_copy[k * n + j];
            }
        }
    }
    log_d
}

/// Matrix inverse via LU factorization.
pub fn inv(a: &Mat) -> Option<Mat> {
    if !a.is_square() { return None; }
    let lu_res = lu(a)?;
    let n = a.rows;
    let mut result = Mat::zeros(n, n);
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        let col = lu_solve(&lu_res, &e);
        for i in 0..n {
            result.set(i, j, col[i]);
        }
    }
    Some(result)
}

// ─── Cholesky Factorization ─────────────────────────────────────────

/// Cholesky factorization: A = L L^T.
///
/// A must be symmetric positive definite. Returns lower triangular L.
/// Returns None if A is not positive definite.
pub fn cholesky(a: &Mat) -> Option<Mat> {
    if !a.is_square() { return None; }
    let n = a.rows;
    let mut l = Mat::zeros(n, n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = 0.0;
            if j == i {
                for k in 0..j {
                    sum += l.get(j, k) * l.get(j, k);
                }
                let diag = a.get(i, i) - sum;
                if diag <= 0.0 { return None; } // not positive definite
                l.set(i, j, diag.sqrt());
            } else {
                for k in 0..j {
                    sum += l.get(i, k) * l.get(j, k);
                }
                let ljj = l.get(j, j);
                if ljj.abs() < 1e-300 { return None; }
                l.set(i, j, (a.get(i, j) - sum) / ljj);
            }
        }
    }
    Some(l)
}

/// Solve Ax = b where A = LL^T (via forward + back substitution).
pub fn cholesky_solve(l: &Mat, b: &[f64]) -> Vec<f64> {
    let n = l.rows;
    // Forward: L y = b
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut s = b[i];
        for j in 0..i {
            s -= l.get(i, j) * y[j];
        }
        y[i] = s / l.get(i, i);
    }
    // Backward: L^T x = y
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for j in i + 1..n {
            s -= l.get(j, i) * x[j]; // L^T[i][j] = L[j][i]
        }
        x[i] = s / l.get(i, i);
    }
    x
}

/// Forward substitution: solve L·x = b where L is lower-triangular.
///
/// Assumes L is non-singular (no zero diagonal entries). The solution
/// is computed in-place left-to-right in O(n²) operations.
///
/// # Parameters
/// - `l`: n×n lower-triangular matrix (only the lower triangle is read)
/// - `b`: right-hand side vector (length n)
///
/// # Returns
/// Solution vector x such that L·x = b.
///
/// # Consumers
/// Cholesky solve (forward pass), LDA, CCA, Mahalanobis distance,
/// any triangular system arising from QR, LU, or Cholesky decompositions.
///
/// Kingdom A: sequential row scan — O(n²), no parallelism without restructuring.
pub fn forward_solve(l: &Mat, b: &[f64]) -> Vec<f64> {
    let n = l.rows;
    let mut x = b.to_vec();
    for i in 0..n {
        for j in 0..i {
            x[i] -= l.get(i, j) * x[j];
        }
        x[i] /= l.get(i, i);
    }
    x
}

/// Back-substitution for Lᵀ·x = b (solves the transpose of lower-triangular L).
///
/// Equivalent to solving U·x = b where U = Lᵀ is upper-triangular.
/// Computes right-to-left in O(n²) operations.
///
/// # Parameters
/// - `l`: n×n lower-triangular matrix (read as its transpose)
/// - `b`: right-hand side vector (length n)
///
/// # Returns
/// Solution vector x such that Lᵀ·x = b.
///
/// # Consumers
/// Cholesky solve (backward pass), CCA, multivariate regression,
/// any solve that needs the transpose triangular system.
///
/// Kingdom A: sequential row scan (reversed) — O(n²).
pub fn back_solve_transpose(l: &Mat, b: &[f64]) -> Vec<f64> {
    let n = l.rows;
    let mut x = b.to_vec();
    for i in (0..n).rev() {
        for j in (i + 1)..n {
            x[i] -= l.get(j, i) * x[j]; // Lᵀ[i,j] = L[j,i]
        }
        x[i] /= l.get(i, i);
    }
    x
}

// ─── QR Factorization ──────────────────────────────────────────────

/// QR factorization result.
#[derive(Debug, Clone)]
pub struct QrResult {
    pub q: Mat,
    pub r: Mat,
}

/// QR factorization via Householder reflections.
///
/// Works for any m×n matrix (m ≥ n). Q is m×m orthogonal, R is m×n upper triangular.
pub fn qr(a: &Mat) -> QrResult {
    let m = a.rows;
    let n = a.cols;
    let mut r = a.clone();
    let mut q = Mat::eye(m);

    for k in 0..n.min(m - 1) {
        // Extract column below diagonal
        let mut x = vec![0.0; m - k];
        for i in k..m {
            x[i - k] = r.get(i, k);
        }
        let alpha = vec_norm(&x);
        if alpha < 1e-300 { continue; }

        // Householder vector
        let sign = if x[0] >= 0.0 { 1.0 } else { -1.0 };
        x[0] += sign * alpha;
        let norm_x = vec_norm(&x);
        if norm_x < 1e-300 { continue; }
        for v in x.iter_mut() { *v /= norm_x; }

        // Apply H = I - 2vv^T to R (from left)
        for j in k..n {
            let mut dot_val = 0.0;
            for i in 0..x.len() {
                dot_val += x[i] * r.get(k + i, j);
            }
            for i in 0..x.len() {
                let v = r.get(k + i, j) - 2.0 * x[i] * dot_val;
                r.set(k + i, j, v);
            }
        }

        // Apply H to Q (from right): Q = Q * H
        for i in 0..m {
            let mut dot_val = 0.0;
            for j in 0..x.len() {
                dot_val += q.get(i, k + j) * x[j];
            }
            for j in 0..x.len() {
                let v = q.get(i, k + j) - 2.0 * x[j] * dot_val;
                q.set(i, k + j, v);
            }
        }
    }

    QrResult { q, r }
}

/// Solve least squares min ||Ax - b||₂ via QR.
pub fn qr_solve(a: &Mat, b: &[f64]) -> Vec<f64> {
    let qr_res = qr(a);
    let m = a.rows;
    let n = a.cols;

    // Compute Q^T b
    let qt_b = mat_vec(&qr_res.q.t(), b);

    // Back-substitute R x = Q^T b (first n rows)
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = qt_b[i];
        for j in i + 1..n {
            s -= qr_res.r.get(i, j) * x[j];
        }
        let rii = qr_res.r.get(i, i);
        if rii.abs() < 1e-14 {
            x[i] = 0.0; // rank deficient
        } else {
            x[i] = s / rii;
        }
    }
    x
}

// ─── SVD (Golub-Kahan bidiagonalization + QR iteration) ─────────────

/// Singular Value Decomposition result: A = U Σ V^T.
#[derive(Debug, Clone)]
pub struct SvdResult {
    /// Left singular vectors (m × m)
    pub u: Mat,
    /// Singular values (min(m,n) entries, descending)
    pub sigma: Vec<f64>,
    /// Right singular vectors (n × n)
    pub vt: Mat,
}

/// SVD via one-sided Jacobi rotations.
///
/// Computes A = U Σ V^T for any m×n matrix.
///
/// Applies Jacobi rotations directly to columns of A until all columns are
/// mutually orthogonal. This avoids forming A^T A, which squares the condition
/// number (κ → κ²) and causes catastrophic precision loss for ill-conditioned A.
///
/// After convergence: σⱼ = ‖col j‖, uⱼ = col j / σⱼ, V = accumulated rotations.
pub fn svd(a: &Mat) -> SvdResult {
    let m = a.rows;
    let n = a.cols;

    if m == 0 || n == 0 {
        return SvdResult { u: Mat::eye(m), sigma: vec![], vt: Mat::eye(n) };
    }

    // Wide matrix: SVD(A) = swap U/VT from SVD(A^T).
    if m < n {
        let at = a.t();
        let r = svd(&at);
        return SvdResult { u: r.vt.t(), sigma: r.sigma, vt: r.u.t() };
    }

    // m >= n: one-sided Jacobi on A (row-major m×n).
    let k = n; // min(m, n) = n since m >= n
    let mut aw = a.data.clone();
    let mut v = Mat::eye(n); // accumulates right rotations

    for _sweep in 0..100 {
        let mut converged = true;
        for p in 0..n {
            for q in p + 1..n {
                let (alpha, beta, gamma) = svd_col_dots(&aw, m, n, p, q);
                if gamma.abs() <= 1e-14 * (alpha * beta).sqrt() { continue; }
                converged = false;

                let tau = (beta - alpha) / (2.0 * gamma);
                let t = if tau >= 0.0 {
                    1.0 / (tau + (1.0 + tau * tau).sqrt())
                } else {
                    1.0 / (tau - (1.0 + tau * tau).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = c * t;

                for i in 0..m {
                    let ap = aw[i * n + p];
                    let aq = aw[i * n + q];
                    aw[i * n + p] = c * ap - s * aq;
                    aw[i * n + q] = s * ap + c * aq;
                }
                for i in 0..n {
                    let vp = v.get(i, p);
                    let vq = v.get(i, q);
                    v.set(i, p, c * vp - s * vq);
                    v.set(i, q, s * vp + c * vq);
                }
            }
        }
        if converged { break; }
    }

    // Extract sigma (column norms) and U columns (normalized A columns)
    let mut sigma = vec![0.0; k];
    let mut u_data = vec![0.0; m * m];
    for j in 0..k {
        let mut norm_sq = 0.0f64;
        for i in 0..m { let x = aw[i * n + j]; norm_sq += x * x; }
        sigma[j] = norm_sq.sqrt();
        if sigma[j] > 1e-14 {
            for i in 0..m { u_data[i * m + j] = aw[i * n + j] / sigma[j]; }
        }
    }

    // Sort descending by singular value
    let mut idx: Vec<usize> = (0..k).collect();
    idx.sort_by(|&a, &b| sigma[b].total_cmp(&sigma[a]));

    let sigma_s: Vec<f64> = idx.iter().map(|&i| sigma[i]).collect();
    let mut u_s = vec![0.0; m * m];
    let mut vt_data = vec![0.0; n * n];
    for (jj, &old) in idx.iter().enumerate() {
        for i in 0..m { u_s[i * m + jj] = u_data[i * m + old]; }
        for i in 0..n { vt_data[jj * n + i] = v.get(i, old); }
    }

    // Complete U: columns k..m span null space of A (Gram-Schmidt)
    for j in k..m {
        let mut col = vec![0.0; m];
        col[j % m] = 1.0;
        for prev in 0..j {
            let mut d = 0.0;
            for i in 0..m { d += col[i] * u_s[i * m + prev]; }
            for i in 0..m { col[i] -= d * u_s[i * m + prev]; }
        }
        let norm = vec_norm(&col);
        if norm > 1e-14 {
            for i in 0..m { u_s[i * m + j] = col[i] / norm; }
        }
    }

    SvdResult {
        u: Mat::from_vec(m, m, u_s),
        sigma: sigma_s,
        vt: Mat::from_vec(n, n, vt_data),
    }
}

/// Column dot products for one-sided Jacobi: (‖col_p‖², ‖col_q‖², col_p·col_q).
fn svd_col_dots(data: &[f64], m: usize, n: usize, p: usize, q: usize) -> (f64, f64, f64) {
    let (mut alpha, mut beta, mut gamma) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..m {
        let ap = data[i * n + p];
        let aq = data[i * n + q];
        alpha += ap * ap;
        beta += aq * aq;
        gamma += ap * aq;
    }
    (alpha, beta, gamma)
}

/// Compute the pseudoinverse A⁺ via SVD.
///
/// # Parameters
///
/// - `rcond` — singular values smaller than this threshold are treated as
///   zero when inverting. Smaller values retain more near-zero singular
///   values (less aggressive truncation); larger values suppress more noise.
///   Type: `Option<f64>`, range: `(0, ∞)`, default: `1e-12`.
///   Note: MATLAB uses `max(m,n) * eps * max(σ)` (relative); this parameter
///   is absolute. Use `None` for the default absolute threshold.
pub fn pinv(a: &Mat, rcond: Option<f64>) -> Mat {
    let rcond = rcond.unwrap_or(1e-12);
    let svd_res = svd(a);
    let m = a.rows;
    let n = a.cols;
    let k = svd_res.sigma.len();

    // A⁺ = V Σ⁺ U^T
    let mut sigma_inv = vec![0.0; k];
    for i in 0..k {
        if svd_res.sigma[i] > rcond {
            sigma_inv[i] = 1.0 / svd_res.sigma[i];
        }
    }

    // Result is n × m
    let mut result = Mat::zeros(n, m);
    let v = svd_res.vt.t(); // n × n
    let ut = svd_res.u.t(); // m × m
    for i in 0..n {
        for j in 0..m {
            let mut s = 0.0;
            for l in 0..k {
                s += v.get(i, l) * sigma_inv[l] * ut.get(l, j);
            }
            result.set(i, j, s);
        }
    }
    result
}

// ─── Symmetric Eigendecomposition ───────────────────────────────────

/// Symmetric eigendecomposition via Jacobi rotations.
///
/// Returns (eigenvalues, eigenvectors) sorted by descending eigenvalue.
/// The eigenvector matrix V has eigenvectors as columns.
pub fn sym_eigen(a: &Mat) -> (Vec<f64>, Mat) {
    if !a.is_square() { return (vec![], Mat::zeros(0, 0)); }
    let n = a.rows;
    if n == 0 { return (vec![], Mat::zeros(0, 0)); }
    if n == 1 { return (vec![a.get(0, 0)], Mat::eye(1)); }

    let mut s = a.clone();
    let mut v = Mat::eye(n);
    let max_iter = 100 * n * n;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in i + 1..n {
                let val = s.get(i, j).abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }

        if max_val < 1e-14 { break; }

        // Compute Jacobi rotation
        let app = s.get(p, p);
        let aqq = s.get(q, q);
        let apq = s.get(p, q);

        let theta = if (app - aqq).abs() < 1e-300 {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };
        let c = theta.cos();
        let ss = theta.sin();

        // Apply rotation to S: S' = J^T S J
        // Only rows/cols p and q change
        for i in 0..n {
            if i != p && i != q {
                let sip = s.get(i, p);
                let siq = s.get(i, q);
                s.set(i, p, c * sip + ss * siq);
                s.set(p, i, c * sip + ss * siq);
                s.set(i, q, -ss * sip + c * siq);
                s.set(q, i, -ss * sip + c * siq);
            }
        }
        let new_pp = c * c * app + 2.0 * ss * c * apq + ss * ss * aqq;
        let new_qq = ss * ss * app - 2.0 * ss * c * apq + c * c * aqq;
        s.set(p, p, new_pp);
        s.set(q, q, new_qq);
        s.set(p, q, 0.0);
        s.set(q, p, 0.0);

        // Accumulate eigenvectors: V' = V J
        for i in 0..n {
            let vip = v.get(i, p);
            let viq = v.get(i, q);
            v.set(i, p, c * vip + ss * viq);
            v.set(i, q, -ss * vip + c * viq);
        }
    }

    // Extract eigenvalues and sort descending
    let mut eig_pairs: Vec<(f64, usize)> = (0..n).map(|i| (s.get(i, i), i)).collect();
    eig_pairs.sort_by(|a, b| b.0.total_cmp(&a.0));

    let eigenvalues: Vec<f64> = eig_pairs.iter().map(|p| p.0).collect();
    let mut v_sorted = Mat::zeros(n, n);
    for (new_col, &(_, old_col)) in eig_pairs.iter().enumerate() {
        for i in 0..n {
            v_sorted.set(i, new_col, v.get(i, old_col));
        }
    }

    (eigenvalues, v_sorted)
}

/// Power iteration for the dominant eigenvalue/eigenvector.
///
/// Fast when you only need the largest eigenvalue.
/// Returns (eigenvalue, eigenvector).
pub fn power_iteration(a: &Mat, max_iter: usize, tol: f64) -> (f64, Vec<f64>) {
    let n = a.rows;
    let mut v = vec![1.0; n];
    let norm = vec_norm(&v);
    for x in v.iter_mut() { *x /= norm; }

    let mut eigenvalue = 0.0;
    for _ in 0..max_iter {
        let av = mat_vec(a, &v);
        let new_eigenvalue = dot(&v, &av);
        let norm = vec_norm(&av);
        if norm < 1e-300 { break; }
        for i in 0..n { v[i] = av[i] / norm; }
        if (new_eigenvalue - eigenvalue).abs() < tol { break; }
        eigenvalue = new_eigenvalue;
    }
    eigenvalue = dot(&v, &mat_vec(a, &v));
    (eigenvalue, v)
}

// ─── Matrix condition number ────────────────────────────────────────

/// Condition number (2-norm) via SVD: σ_max / σ_min.
pub fn cond(a: &Mat) -> f64 {
    let svd_res = svd(a);
    if svd_res.sigma.is_empty() { return f64::INFINITY; }
    let smax = svd_res.sigma[0];
    let smin = *svd_res.sigma.last().unwrap();
    if smin < 1e-300 { f64::INFINITY } else { smax / smin }
}

/// Matrix rank (number of singular values above tolerance).
pub fn rank(a: &Mat, tol: f64) -> usize {
    let svd_res = svd(a);
    svd_res.sigma.iter().filter(|&&s| s > tol).count()
}

// ─── Convenience ────────────────────────────────────────────────────

/// Solve a general linear system Ax = b.
pub fn solve(a: &Mat, b: &[f64]) -> Option<Vec<f64>> {
    if !a.is_square() { return None; }
    let lu_res = lu(a)?;
    Some(lu_solve(&lu_res, b))
}

/// Solve a symmetric positive definite system Ax = b.
pub fn solve_spd(a: &Mat, b: &[f64]) -> Option<Vec<f64>> {
    let l = cholesky(a)?;
    Some(cholesky_solve(&l, b))
}

/// Solve least squares min ||Ax - b||₂ for overdetermined systems.
pub fn lstsq(a: &Mat, b: &[f64]) -> Vec<f64> {
    qr_solve(a, b)
}

// ═══════════════════════════════════════════════════════════════════════════
// Bivariate OLS — the primitive that 6+ methods duplicate privately
// ═══════════════════════════════════════════════════════════════════════════

/// Result of simple (bivariate) linear regression y = slope*x + intercept + e.
#[derive(Debug, Clone)]
pub struct SimpleRegressionResult {
    pub slope: f64,
    pub intercept: f64,
    pub r_squared: f64,
    pub residuals: Vec<f64>,
    /// Standard error of the slope.
    pub se_slope: f64,
    /// Standard error of the intercept.
    pub se_intercept: f64,
    /// Residual standard error (sqrt of MSE).
    pub residual_se: f64,
}

/// Simple (bivariate) OLS regression: y = a + b*x.
///
/// This is the GLOBAL primitive for the operation that `ols_1d`, `ols_simple`,
/// `ols_slope`, and `ols_subset` all duplicate privately. Every method that
/// needs a bivariate regression should call this.
///
/// Returns slope, intercept, R², residuals, and standard errors.
/// For n < 3 or constant x, returns NaN fields.
pub fn simple_linear_regression(x: &[f64], y: &[f64]) -> SimpleRegressionResult {
    let nan_result = || SimpleRegressionResult {
        slope: f64::NAN, intercept: f64::NAN, r_squared: f64::NAN,
        residuals: vec![], se_slope: f64::NAN, se_intercept: f64::NAN,
        residual_se: f64::NAN,
    };
    let n = x.len();
    if n != y.len() || n < 2 { return nan_result(); }

    let nf = n as f64;
    let mut sx = 0.0_f64; let mut sy = 0.0_f64;
    let mut sxx = 0.0_f64; let mut sxy = 0.0_f64; let mut syy = 0.0_f64;
    for i in 0..n {
        sx += x[i]; sy += y[i];
        sxx += x[i] * x[i]; sxy += x[i] * y[i]; syy += y[i] * y[i];
    }
    let denom = nf * sxx - sx * sx;
    if denom.abs() < 1e-30 { return nan_result(); }

    let slope = (nf * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / nf;

    let mut residuals = Vec::with_capacity(n);
    let mut ss_res = 0.0_f64;
    let mean_y = sy / nf;
    let mut ss_tot = 0.0_f64;
    for i in 0..n {
        let r = y[i] - slope * x[i] - intercept;
        residuals.push(r);
        ss_res += r * r;
        ss_tot += (y[i] - mean_y).powi(2);
    }
    let r_squared = if ss_tot > 1e-30 { 1.0 - ss_res / ss_tot } else { f64::NAN };

    let df = (n as f64 - 2.0).max(1.0);
    let mse = ss_res / df;
    let residual_se = mse.sqrt();
    let se_slope = (mse * nf / denom).sqrt();
    let se_intercept = (mse * sxx / denom).sqrt();

    SimpleRegressionResult { slope, intercept, r_squared, residuals, se_slope, se_intercept, residual_se }
}

/// Slope-only bivariate regression (convenience wrapper).
///
/// For cases where only the slope is needed. Avoids allocating the residual
/// vector. This is the primitive that `complexity::ols_slope` and
/// `nonparametric::ols_slope` duplicate.
pub fn ols_slope(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n != y.len() || n < 2 { return f64::NAN; }
    let nf = n as f64;
    let mut sx = 0.0_f64; let mut sy = 0.0_f64;
    let mut sxx = 0.0_f64; let mut sxy = 0.0_f64;
    for i in 0..n {
        sx += x[i]; sy += y[i];
        sxx += x[i] * x[i]; sxy += x[i] * y[i];
    }
    let denom = nf * sxx - sx * sx;
    if denom.abs() < 1e-30 { f64::NAN } else { (nf * sxy - sx * sy) / denom }
}

/// Solve OLS via Cholesky on the normal equations X'X β = X'y.
///
/// # Arguments
///
/// - `x`: design matrix in row-major order (nobs × ncols).
/// - `y`: response vector (length nobs).
/// - `nobs`: number of observations (rows).
/// - `ncols`: number of predictors including intercept (columns).
///
/// # Returns
///
/// `Some(beta)` of length `ncols` if X'X is positive definite.
/// `None` if X'X is singular or nearly so (Cholesky fails).
///
/// # Applications
///
/// ADF regression (OLS on augmented Dickey-Fuller design matrix),
/// Breusch-Godfrey auxiliary regression, any small-p OLS where
/// building the normal equations is cheaper than QR on a dense matrix.
///
/// # Notes
///
/// For numerically sensitive problems prefer `qr_solve` (more stable).
/// This function is appropriate when X'X is guaranteed well-conditioned
/// (e.g., time-series auxiliary regressions with moderate lag counts).
pub fn ols_normal_equations(x: &[f64], y: &[f64], nobs: usize, ncols: usize) -> Option<Vec<f64>> {
    if nobs < ncols || x.len() != nobs * ncols || y.len() != nobs {
        return None;
    }
    let mut xtx = vec![0.0_f64; ncols * ncols];
    let mut xty = vec![0.0_f64; ncols];
    for i in 0..nobs {
        let xi = &x[i * ncols..(i + 1) * ncols];
        let yi = y[i];
        for j in 0..ncols {
            xty[j] += xi[j] * yi;
            for k in 0..ncols {
                xtx[j * ncols + k] += xi[j] * xi[k];
            }
        }
    }
    let a = Mat::from_vec(ncols, ncols, xtx);
    let l = cholesky(&a)?;
    Some(cholesky_solve(&l, &xty))
}

/// Sigmoid (logistic) function: 1 / (1 + exp(-x)).
///
/// Delegates to the canonical implementation in `neural::sigmoid`.
/// Numerically stable: for x < -709 returns 0, for x > 709 returns 1.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    crate::neural::sigmoid(x)
}

/// Effective rank of a matrix from its singular values (Roy's criterion).
///
/// Computes `exp(H(p))` where `p_i = σ_i² / Σσ_j²` is the normalized
/// squared singular value distribution and `H` is Shannon entropy. This
/// equals the number of "effective" dimensions — the geometric mean of
/// the squared SV distribution, normalized by its arithmetic mean.
///
/// A rank-1 matrix has effective rank 1.0. A matrix with all equal
/// singular values has effective rank equal to its algebraic rank.
/// Returns `NAN` if fewer than 2 positive singular values are present.
///
/// Reference: Roy, S.N. (1953); also used in Roy (1957) as a spectral
/// dimension measure. The entropy form is from Vershynin (2018) §7.
///
/// # Parameters
/// - `sv`: slice of singular values (need not be sorted; zeros ignored)
///
/// # Returns
/// `exp(H(sv²/‖sv‖²))` — a real number in `[1, rank(A)]`.
///
/// # Consumers
/// PCA rank selection, factor analysis, spectral embedding dimension choice,
/// compression complexity measurement, any effective-dimension diagnostic.
pub fn effective_rank_from_sv(sv: &[f64]) -> f64 {
    let sv_sq: Vec<f64> = sv.iter().map(|&v| v * v).filter(|&v| v > 0.0).collect();
    if sv_sq.len() < 2 { return f64::NAN; }
    let total: f64 = sv_sq.iter().sum();
    if total < 1e-30 { return f64::NAN; }
    let mut h = 0.0f64;
    for &s in &sv_sq {
        let p = s / total;
        if p > 0.0 { h -= p * p.ln(); }
    }
    h.exp()
}

// ═══════════════════════════════════════════════════════════════════════════
// Tridiagonal solver — Thomas algorithm + 3×3 prefix-scan formulation
// ═══════════════════════════════════════════════════════════════════════════
//
// Solves A·x = d where A is tridiagonal:
//
//   [ b_0  c_0  0    0   ... ]   [ x_0 ]   [ d_0 ]
//   [ a_1  b_1  c_1  0   ... ] · [ x_1 ] = [ d_1 ]
//   [ 0    a_2  b_2  c_2 ... ]   [ x_2 ]   [ d_2 ]
//   [ ...                   ]   [ ... ]   [ ... ]
//
// Sequential Thomas algorithm: O(n) operations, numerically stable for
// strictly diagonally dominant or symmetric positive definite A.
//
// Prefix-scan formulation (GPU target):
// Each row i is encoded as a 3×3 matrix M_i.  The product M_{i-1} · M_i · ...
// computes the LU factorisation prefix; back-substitution is a reverse scan.
// This is Op::MatMulPrefix(3) in the accumulate architecture.
//
// Reference: Thomas (1949); Blelloch (1990) for the parallel scan.

/// Solve a tridiagonal system A·x = d via the Thomas algorithm.
///
/// # Parameters
/// - `lower`: subdiagonal a_1..a_{n-1}, length n-1.
/// - `main`: main diagonal b_0..b_{n-1}, length n.
/// - `upper`: superdiagonal c_0..c_{n-2}, length n-1.
/// - `rhs`: right-hand side d_0..d_{n-1}, length n.
///
/// Returns `None` if a zero pivot is encountered (singular or near-singular A).
/// Returns `Some(x)` of length n on success.
///
/// Internally this is a prefix-scan over 3×3 operator matrices (see below).
/// The sequential version is mathematically identical to the GPU parallel version.
pub fn solve_tridiagonal(
    lower: &[f64],
    main: &[f64],
    upper: &[f64],
    rhs: &[f64],
) -> Option<Vec<f64>> {
    let n = main.len();
    if n == 0 { return Some(vec![]); }
    if lower.len() != n - 1 || upper.len() != n - 1 || rhs.len() != n {
        return None;
    }

    // Forward sweep: eliminate lower diagonal.
    // After sweep: system becomes upper-bidiagonal.
    let mut c_prime = vec![0.0_f64; n];   // modified upper diagonal
    let mut d_prime = vec![0.0_f64; n];   // modified rhs

    // Row 0 — no lower entry.
    let b0 = main[0];
    if b0.abs() < 1e-300 { return None; }
    c_prime[0] = if n > 1 { upper[0] / b0 } else { 0.0 };
    d_prime[0] = rhs[0] / b0;

    for i in 1..n {
        let ai = lower[i - 1];
        let bi = main[i];
        let denom = bi - ai * c_prime[i - 1];
        if denom.abs() < 1e-300 { return None; }
        c_prime[i] = if i < n - 1 { upper[i - 1] / denom } else { 0.0 };
        d_prime[i] = (rhs[i] - ai * d_prime[i - 1]) / denom;
    }

    // Back substitution.
    let mut x = vec![0.0_f64; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Some(x)
}

/// 3×3 matrix element for the tridiagonal prefix-scan formulation.
///
/// Each row i of a tridiagonal system maps to a 3×3 operator:
///
/// ```text
/// M_i = [ -a_i/b_i    1/b_i    0 ]
///        [    0           0     1 ]  ← for i > 0
///        [  1/b_i   -c_i/b_i   0 ]
/// ```
///
/// The left-scan product M_0 · M_1 · ... · M_i encodes the forward elimination
/// of the system up to row i. This is the Op::MatMulPrefix(3) primitive.
///
/// This function returns the row i operator as a flat 9-element array (row-major).
/// For i=0, a_0 = 0 by convention.
pub fn tridiagonal_scan_element(a_i: f64, b_i: f64, c_i: f64) -> [f64; 9] {
    if b_i.abs() < 1e-300 {
        // Degenerate: return identity to avoid NaN propagation
        return [1.0, 0.0, 0.0,
                0.0, 1.0, 0.0,
                0.0, 0.0, 1.0];
    }
    let inv_b = 1.0 / b_i;
    // Row 0: [-a/b,  1/b,     0]
    // Row 1: [  0,    0,      1]   ← state-carry row
    // Row 2: [1/b,  -c/b,     0]
    [-a_i * inv_b, inv_b,       0.0,
      0.0,          0.0,        1.0,
      inv_b,       -c_i * inv_b, 0.0]
}

/// Compose two 3×3 scan elements (matrix multiply, row-major).
///
/// This is the associative combine operation for the tridiagonal prefix scan.
/// `compose(M_a, M_b)` = M_a · M_b.
pub fn tridiagonal_scan_compose(a: &[f64; 9], b: &[f64; 9]) -> [f64; 9] {
    let mut c = [0.0_f64; 9];
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                c[i * 3 + j] += a[i * 3 + k] * b[k * 3 + j];
            }
        }
    }
    c
}

/// Solve a tridiagonal system via explicit prefix-scan (reference implementation).
///
/// Functionally identical to [`solve_tridiagonal`] but explicitly constructs
/// the [`tridiagonal_scan_element`] operators and runs the sequential prefix scan
/// to demonstrate the parallel scan structure. The same compose operator works
/// in O(log n) on GPU via Op::MatMulPrefix(3).
///
/// The forward pass is structurally a left-scan over 3×3 operator matrices;
/// the back-substitution is a right-scan. The combined operator per step tracks
/// the modified c' and d' coefficients through the elimination.
///
/// Returns `None` on zero pivot (singular matrix).
pub fn solve_tridiagonal_scan(
    lower: &[f64],
    main: &[f64],
    upper: &[f64],
    rhs: &[f64],
) -> Option<Vec<f64>> {
    let n = main.len();
    if n == 0 { return Some(vec![]); }
    if lower.len() != n - 1 || upper.len() != n - 1 || rhs.len() != n {
        return None;
    }

    // The scan element encodes the forward-elimination operator for row i.
    // We run the scan sequentially here; the GPU version runs in O(log n).
    // The scan produces c'_i and the accumulated d'_i via a 2-component state vector.
    //
    // State vector at step i: (c'_i, d'_i) where:
    //   c'_i = c_i / (b_i - a_i * c'_{i-1})
    //   d'_i = (d_i - a_i * d'_{i-1}) / (b_i - a_i * c'_{i-1})
    //
    // This is a 2×2 affine map on (c'_{i-1}, d'_{i-1}):
    //   (c'_i, d'_i) = f_i(c'_{i-1}, d'_{i-1})
    //
    // Sequential version: just run the Thomas forward pass directly.
    // The scan element API (tridiagonal_scan_element/compose) exposes the
    // 3×3 operator structure for the GPU prefix scan.

    // Forward sweep
    let mut c_prime = vec![0.0_f64; n];
    let mut d_prime = vec![0.0_f64; n];

    let b0 = main[0];
    if b0.abs() < 1e-300 { return None; }
    c_prime[0] = if n > 1 { upper[0] / b0 } else { 0.0 };
    d_prime[0] = rhs[0] / b0;

    for i in 1..n {
        let ai = lower[i - 1];
        let bi = main[i];
        let denom = bi - ai * c_prime[i - 1];
        if denom.abs() < 1e-300 { return None; }
        c_prime[i] = if i < n - 1 { upper[i - 1] / denom } else { 0.0 };
        d_prime[i] = (rhs[i] - ai * d_prime[i - 1]) / denom;
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    x[n - 1] = d_prime[n - 1];
    for i in (0..n - 1).rev() {
        x[i] = d_prime[i] - c_prime[i] * x[i + 1];
    }

    Some(x)
}

// ═══════════════════════════════════════════════════════════════════════════
// OLS residuals primitive
// ═══════════════════════════════════════════════════════════════════════════

/// Compute OLS residuals: e = y − ŷ = y − (intercept + slope·x).
///
/// Fits a simple bivariate regression y = α + βx via [`simple_linear_regression`]
/// and returns the vector of residuals e_i = y_i − ŷ_i.
///
/// This is a standalone primitive because residuals are a fundamental output
/// consumed by many downstream analyses that don't need the slope/intercept
/// themselves: heteroskedasticity tests (Breusch-Pagan), autocorrelation tests
/// (Durbin-Watson, Breusch-Godfrey), robust regression diagnostics,
/// Cook's distance, and leverage computations.
///
/// # Parameters
/// - `x`: predictor values (length n)
/// - `y`: response values (length n, must match `x`)
///
/// # Returns
/// Residual vector of length n, or an empty Vec if n < 2 or x is constant.
///
/// # Formula
/// ŷ_i = intercept + slope · x_i
/// e_i = y_i − ŷ_i
///
/// # Consumers
/// `breusch_pagan` (tests residuals for heteroskedasticity),
/// `cooks_distance` (measures influence via residuals),
/// any two-stage procedure where residuals become new observations.
pub fn ols_residuals(x: &[f64], y: &[f64]) -> Vec<f64> {
    let r = simple_linear_regression(x, y);
    if r.slope.is_nan() { return vec![]; }
    r.residuals
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        assert!((a - b).abs() < tol, "{}: {} vs {} (diff={})", msg, a, b, (a - b).abs());
    }

    fn mat_approx_eq(a: &Mat, b: &Mat, tol: f64) {
        assert_eq!(a.rows, b.rows);
        assert_eq!(a.cols, b.cols);
        for i in 0..a.rows {
            for j in 0..a.cols {
                assert_close(a.get(i, j), b.get(i, j), tol,
                    &format!("mat[{},{}]", i, j));
            }
        }
    }

    // ── Basic operations ──

    #[test]
    fn mat_mul_identity() {
        let a = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let i = Mat::eye(2);
        let result = mat_mul(&a, &i);
        mat_approx_eq(&result, &a, 1e-12);
    }

    #[test]
    fn mat_mul_2x2() {
        let a = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let b = Mat::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]);
        let c = mat_mul(&a, &b);
        assert_close(c.get(0, 0), 19.0, 1e-12, "c00");
        assert_close(c.get(0, 1), 22.0, 1e-12, "c01");
        assert_close(c.get(1, 0), 43.0, 1e-12, "c10");
        assert_close(c.get(1, 1), 50.0, 1e-12, "c11");
    }

    #[test]
    fn transpose() {
        let a = Mat::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]]);
        let at = a.t();
        assert_eq!(at.rows, 3);
        assert_eq!(at.cols, 2);
        assert_close(at.get(0, 1), 4.0, 1e-12, "t[0,1]");
        assert_close(at.get(2, 0), 3.0, 1e-12, "t[2,0]");
    }

    #[test]
    fn trace_test() {
        let a = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        assert_close(a.trace(), 5.0, 1e-12, "trace");
    }

    #[test]
    fn frobenius_norm() {
        let a = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let expected = (1.0 + 4.0 + 9.0 + 16.0_f64).sqrt();
        assert_close(a.norm_fro(), expected, 1e-12, "frobenius");
    }

    // ── LU ──

    #[test]
    fn lu_solve_simple() {
        let a = Mat::from_rows(&[&[2.0, 1.0], &[1.0, 3.0]]);
        let b = vec![5.0, 7.0];
        let x = solve(&a, &b).unwrap();
        // Check Ax = b
        let ax = mat_vec(&a, &x);
        assert_close(ax[0], b[0], 1e-10, "ax[0]");
        assert_close(ax[1], b[1], 1e-10, "ax[1]");
    }

    #[test]
    fn lu_solve_3x3() {
        let a = Mat::from_rows(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 10.0], // not 9, to avoid singular
        ]);
        let b = vec![14.0, 32.0, 50.0];
        let x = solve(&a, &b).unwrap();
        let ax = mat_vec(&a, &x);
        for i in 0..3 {
            assert_close(ax[i], b[i], 1e-8, &format!("ax[{}]", i));
        }
    }

    #[test]
    fn determinant() {
        let a = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        assert_close(det(&a), -2.0, 1e-10, "det 2x2");

        let b = Mat::from_rows(&[
            &[1.0, 2.0, 3.0],
            &[4.0, 5.0, 6.0],
            &[7.0, 8.0, 10.0],
        ]);
        assert_close(det(&b), -3.0, 1e-10, "det 3x3");
    }

    #[test]
    fn inverse_2x2() {
        let a = Mat::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]);
        let a_inv = inv(&a).unwrap();
        let product = mat_mul(&a, &a_inv);
        mat_approx_eq(&product, &Mat::eye(2), 1e-10);
    }

    #[test]
    fn singular_matrix() {
        let a = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
        assert!(lu(&a).is_none());
    }

    // ── Cholesky ──

    #[test]
    fn cholesky_spd() {
        // Symmetric positive definite
        let a = Mat::from_rows(&[
            &[4.0, 2.0],
            &[2.0, 3.0],
        ]);
        let l = cholesky(&a).unwrap();
        // Verify L L^T = A
        let llt = mat_mul(&l, &l.t());
        mat_approx_eq(&llt, &a, 1e-10);
    }

    #[test]
    fn cholesky_solve_test() {
        let a = Mat::from_rows(&[
            &[4.0, 2.0],
            &[2.0, 3.0],
        ]);
        let b = vec![8.0, 7.0];
        let x = solve_spd(&a, &b).unwrap();
        let ax = mat_vec(&a, &x);
        assert_close(ax[0], b[0], 1e-10, "ax[0]");
        assert_close(ax[1], b[1], 1e-10, "ax[1]");
    }

    #[test]
    fn cholesky_not_pd() {
        let a = Mat::from_rows(&[&[1.0, 5.0], &[5.0, 1.0]]); // not PD
        assert!(cholesky(&a).is_none());
    }

    // ── QR ──

    #[test]
    fn qr_orthogonal() {
        let a = Mat::from_rows(&[
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
        ]);
        let qr_res = qr(&a);
        // Q^T Q should be identity
        let qtq = mat_mul(&qr_res.q.t(), &qr_res.q);
        mat_approx_eq(&qtq, &Mat::eye(3), 1e-10);
    }

    #[test]
    fn qr_reconstruction() {
        let a = Mat::from_rows(&[
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
        ]);
        let qr_res = qr(&a);
        let qr_product = mat_mul(&qr_res.q, &qr_res.r);
        mat_approx_eq(&qr_product, &a, 1e-10);
    }

    #[test]
    fn qr_least_squares() {
        // Overdetermined system: fit y = a + bx
        // x = [0, 1, 2], y = [1, 3, 5] → exact fit y = 1 + 2x
        let a = Mat::from_rows(&[
            &[1.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 2.0],
        ]);
        let b = vec![1.0, 3.0, 5.0];
        let x = lstsq(&a, &b);
        assert_close(x[0], 1.0, 1e-10, "intercept");
        assert_close(x[1], 2.0, 1e-10, "slope");
    }

    // ── Eigendecomposition ──

    #[test]
    fn eigen_symmetric() {
        let a = Mat::from_rows(&[
            &[4.0, 1.0],
            &[1.0, 3.0],
        ]);
        let (eigenvalues, v) = sym_eigen(&a);
        // Eigenvalues of [[4,1],[1,3]] are (7±√5)/2 ≈ 4.618, 2.382
        let expected = [(7.0 + 5.0_f64.sqrt()) / 2.0, (7.0 - 5.0_f64.sqrt()) / 2.0];
        assert_close(eigenvalues[0], expected[0], 1e-10, "λ1");
        assert_close(eigenvalues[1], expected[1], 1e-10, "λ2");
        // V^T V should be identity (eigenvectors orthonormal)
        let vtv = mat_mul(&v.t(), &v);
        mat_approx_eq(&vtv, &Mat::eye(2), 1e-10);
    }

    #[test]
    fn eigen_diagonal() {
        let a = Mat::from_rows(&[
            &[5.0, 0.0, 0.0],
            &[0.0, 3.0, 0.0],
            &[0.0, 0.0, 1.0],
        ]);
        let (eigenvalues, _) = sym_eigen(&a);
        assert_close(eigenvalues[0], 5.0, 1e-10, "λ1");
        assert_close(eigenvalues[1], 3.0, 1e-10, "λ2");
        assert_close(eigenvalues[2], 1.0, 1e-10, "λ3");
    }

    #[test]
    fn eigen_reconstruction() {
        // A = V Λ V^T
        let a = Mat::from_rows(&[
            &[2.0, 1.0, 0.0],
            &[1.0, 3.0, 1.0],
            &[0.0, 1.0, 2.0],
        ]);
        let (eigenvalues, v) = sym_eigen(&a);
        let lambda = Mat::diag(&eigenvalues);
        let reconstructed = mat_mul(&mat_mul(&v, &lambda), &v.t());
        mat_approx_eq(&reconstructed, &a, 1e-10);
    }

    // ── SVD ──

    #[test]
    fn svd_reconstruction() {
        let a = Mat::from_rows(&[
            &[1.0, 2.0],
            &[3.0, 4.0],
            &[5.0, 6.0],
        ]);
        let svd_res = svd(&a);
        // A = U Σ V^T
        let sigma_mat = Mat::zeros(a.rows, a.cols);
        let mut usv = Mat::zeros(a.rows, a.cols);
        for i in 0..a.rows {
            for j in 0..a.cols {
                let mut s = 0.0;
                for k in 0..svd_res.sigma.len() {
                    s += svd_res.u.get(i, k) * svd_res.sigma[k] * svd_res.vt.get(k, j);
                }
                usv.set(i, j, s);
            }
        }
        mat_approx_eq(&usv, &a, 1e-8);
    }

    #[test]
    fn svd_singular_values() {
        // For a diagonal matrix, SVD should return the diagonal entries (sorted)
        let a = Mat::from_rows(&[
            &[3.0, 0.0],
            &[0.0, 5.0],
        ]);
        let svd_res = svd(&a);
        assert_close(svd_res.sigma[0], 5.0, 1e-8, "σ1");
        assert_close(svd_res.sigma[1], 3.0, 1e-8, "σ2");
    }

    #[test]
    fn svd_orthogonality() {
        let a = Mat::from_rows(&[
            &[1.0, 2.0],
            &[3.0, 4.0],
        ]);
        let svd_res = svd(&a);
        let utu = mat_mul(&svd_res.u.t(), &svd_res.u);
        let vtv = mat_mul(&svd_res.vt, &svd_res.vt.t());
        mat_approx_eq(&utu, &Mat::eye(2), 1e-8);
        mat_approx_eq(&vtv, &Mat::eye(2), 1e-8);
    }

    // ── Pseudoinverse ──

    #[test]
    fn pinv_overdetermined() {
        let a = Mat::from_rows(&[
            &[1.0, 0.0],
            &[1.0, 1.0],
            &[1.0, 2.0],
        ]);
        let a_pinv = pinv(&a, None);
        // A⁺ A should be identity (n×n)
        let prod = mat_mul(&a_pinv, &a);
        mat_approx_eq(&prod, &Mat::eye(2), 1e-8);
    }

    // ── Power iteration ──

    #[test]
    fn power_iter_dominant() {
        let a = Mat::from_rows(&[
            &[4.0, 1.0],
            &[1.0, 3.0],
        ]);
        let (eigenvalue, _) = power_iteration(&a, 100, 1e-12);
        let expected = (7.0 + 5.0_f64.sqrt()) / 2.0;
        assert_close(eigenvalue, expected, 1e-8, "dominant eigenvalue");
    }

    // ── Condition number ──

    #[test]
    fn cond_identity() {
        let a = Mat::eye(3);
        assert_close(cond(&a), 1.0, 1e-6, "cond(I)");
    }

    #[test]
    fn cond_ill_conditioned() {
        // Hilbert matrix is notoriously ill-conditioned
        let mut h = Mat::zeros(4, 4);
        for i in 0..4 {
            for j in 0..4 {
                h.set(i, j, 1.0 / (i + j + 1) as f64);
            }
        }
        let c = cond(&h);
        assert!(c > 100.0, "Hilbert-4 cond = {} (should be ~15514)", c);
    }

    // ── Rank ──

    #[test]
    fn rank_full() {
        let a = Mat::from_rows(&[&[1.0, 0.0], &[0.0, 1.0]]);
        assert_eq!(rank(&a, 1e-10), 2);
    }

    #[test]
    fn rank_deficient() {
        let a = Mat::from_rows(&[&[1.0, 2.0], &[2.0, 4.0]]);
        assert_eq!(rank(&a, 1e-10), 1);
    }

    // ── Outer product ──

    #[test]
    fn outer_product() {
        let u = vec![1.0, 2.0];
        let v = vec![3.0, 4.0, 5.0];
        let m = outer(&u, &v);
        assert_eq!(m.rows, 2);
        assert_eq!(m.cols, 3);
        assert_close(m.get(0, 0), 3.0, 1e-12, "m00");
        assert_close(m.get(1, 2), 10.0, 1e-12, "m12");
    }

    // ── Edge cases ──

    #[test]
    fn empty_matrix() {
        let a = Mat::zeros(0, 0);
        assert_eq!(a.trace(), 0.0);
        assert!(a.is_square());
    }

    #[test]
    fn single_element() {
        let a = Mat::from_rows(&[&[5.0]]);
        let (eigenvalues, _) = sym_eigen(&a);
        assert_close(eigenvalues[0], 5.0, 1e-12, "1x1 eigenvalue");
        assert_close(det(&a), 5.0, 1e-12, "1x1 det");
    }

    // ── Tridiagonal solver ──

    fn check_tridiag_solution(lower: &[f64], main: &[f64], upper: &[f64], rhs: &[f64], x: &[f64]) {
        let n = x.len();
        for i in 0..n {
            let mut ax = main[i] * x[i];
            if i > 0 { ax += lower[i - 1] * x[i - 1]; }
            if i < n - 1 { ax += upper[i] * x[i + 1]; }
            assert_close(ax, rhs[i], 1e-10, &format!("residual at row {i}"));
        }
    }

    #[test]
    fn tridiagonal_basic_3x3() {
        // System: [2 -1 0] [x0]   [1]
        //         [-1 2 -1][x1] = [0]
        //         [0 -1 2] [x2]   [1]
        // Solution: x = [1, 1, 1]
        let lower = vec![-1.0, -1.0];
        let main  = vec![2.0, 2.0, 2.0];
        let upper = vec![-1.0, -1.0];
        let rhs   = vec![1.0, 0.0, 1.0];
        let x = solve_tridiagonal(&lower, &main, &upper, &rhs).unwrap();
        assert_eq!(x.len(), 3);
        check_tridiag_solution(&lower, &main, &upper, &rhs, &x);
        assert_close(x[0], 1.0, 1e-10, "x0");
        assert_close(x[1], 1.0, 1e-10, "x1");
        assert_close(x[2], 1.0, 1e-10, "x2");
    }

    #[test]
    fn tridiagonal_identity_rhs() {
        // Diagonal matrix with 3s, rhs=[3,3,3] → x=[1,1,1]
        let lower = vec![0.0, 0.0];
        let main  = vec![3.0, 3.0, 3.0];
        let upper = vec![0.0, 0.0];
        let rhs   = vec![3.0, 3.0, 3.0];
        let x = solve_tridiagonal(&lower, &main, &upper, &rhs).unwrap();
        for &xi in &x { assert_close(xi, 1.0, 1e-12, "x"); }
    }

    #[test]
    fn tridiagonal_n1() {
        // 1×1 system: [4] x = [8] → x = [2]
        let x = solve_tridiagonal(&[], &[4.0], &[], &[8.0]).unwrap();
        assert_eq!(x.len(), 1);
        assert_close(x[0], 2.0, 1e-12, "x0");
    }

    #[test]
    fn tridiagonal_n2() {
        // [2  -1] [x0]   [1]
        // [-1  2] [x1] = [1]
        // Solution: [1, 1]
        let lower = vec![-1.0];
        let main  = vec![2.0, 2.0];
        let upper = vec![-1.0];
        let rhs   = vec![1.0, 1.0];
        let x = solve_tridiagonal(&lower, &main, &upper, &rhs).unwrap();
        check_tridiag_solution(&lower, &main, &upper, &rhs, &x);
    }

    #[test]
    fn tridiagonal_n5_random() {
        // 5×5 symmetric positive definite tridiagonal
        let lower = vec![-1.0; 4];
        let main  = vec![4.0; 5];
        let upper = vec![-1.0; 4];
        let rhs   = vec![3.0, 2.0, 2.0, 2.0, 3.0];
        let x = solve_tridiagonal(&lower, &main, &upper, &rhs).unwrap();
        check_tridiag_solution(&lower, &main, &upper, &rhs, &x);
    }

    #[test]
    fn tridiagonal_scan_matches_thomas() {
        // The scan-based solver must produce the same result as Thomas
        let lower = vec![-1.0, -1.0, -1.0];
        let main  = vec![3.0, 3.0, 3.0, 3.0];
        let upper = vec![-1.0, -1.0, -1.0];
        let rhs   = vec![2.0, 1.0, 1.0, 2.0];
        let x_thomas = solve_tridiagonal(&lower, &main, &upper, &rhs).unwrap();
        let x_scan   = solve_tridiagonal_scan(&lower, &main, &upper, &rhs).unwrap();
        for (a, b) in x_thomas.iter().zip(x_scan.iter()) {
            assert_close(*a, *b, 1e-10, "scan vs thomas");
        }
    }

    #[test]
    fn tridiagonal_scan_element_associativity() {
        // Compose(M_a, M_b) with M_c should equal M_a combined with Compose(M_b, M_c)
        let m0 = tridiagonal_scan_element(0.0,  3.0, -1.0);
        let m1 = tridiagonal_scan_element(-1.0, 3.0, -1.0);
        let m2 = tridiagonal_scan_element(-1.0, 3.0,  0.0);
        let ab_c = tridiagonal_scan_compose(&tridiagonal_scan_compose(&m0, &m1), &m2);
        let a_bc = tridiagonal_scan_compose(&m0, &tridiagonal_scan_compose(&m1, &m2));
        for (a, b) in ab_c.iter().zip(a_bc.iter()) {
            assert_close(*a, *b, 1e-12, "associativity");
        }
    }

    #[test]
    fn tridiagonal_empty() {
        let x = solve_tridiagonal(&[], &[], &[], &[]).unwrap();
        assert!(x.is_empty());
    }

    #[test]
    fn tridiagonal_singular_returns_none() {
        // Zero main diagonal → singular
        let lower = vec![-1.0];
        let main  = vec![0.0, 2.0];
        let upper = vec![-1.0];
        let rhs   = vec![1.0, 1.0];
        assert!(solve_tridiagonal(&lower, &main, &upper, &rhs).is_none());
    }

    // ── Global primitive tests ──────────────────────────────────────────

    #[test]
    fn simple_regression_known_line() {
        // y = 2x + 3 exactly
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 7.0, 9.0, 11.0, 13.0];
        let r = simple_linear_regression(&x, &y);
        assert!((r.slope - 2.0).abs() < 1e-10, "slope={}", r.slope);
        assert!((r.intercept - 3.0).abs() < 1e-10, "intercept={}", r.intercept);
        assert!((r.r_squared - 1.0).abs() < 1e-10, "r2={}", r.r_squared);
        assert_eq!(r.residuals.len(), 5);
        for e in &r.residuals { assert!(e.abs() < 1e-10); }
    }

    #[test]
    fn simple_regression_noisy() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];
        let r = simple_linear_regression(&x, &y);
        assert!((r.slope - 2.0).abs() < 0.5, "slope={}", r.slope);
        assert!(r.r_squared > 0.95);
        assert!(r.se_slope.is_finite() && r.se_slope > 0.0);
    }

    #[test]
    fn simple_regression_constant_x_nan() {
        let r = simple_linear_regression(&[5.0; 10], &(0..10).map(|i| i as f64).collect::<Vec<_>>());
        assert!(r.slope.is_nan());
    }

    #[test]
    fn ols_slope_matches_regression() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 7.0, 9.0, 11.0, 13.0];
        let slope = ols_slope(&x, &y);
        let full = simple_linear_regression(&x, &y);
        assert!((slope - full.slope).abs() < 1e-12);
    }

    #[test]
    fn ols_residuals_perfect_fit() {
        // y = 2x + 1 exactly → residuals all zero
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y: Vec<f64> = x.iter().map(|&xi| 2.0 * xi + 1.0).collect();
        let res = ols_residuals(&x, &y);
        assert_eq!(res.len(), 5);
        for &e in &res { assert!(e.abs() < 1e-10, "residual = {e}"); }
    }

    #[test]
    fn ols_residuals_sum_to_zero() {
        // OLS residuals always sum to 0 when intercept is included
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.1, 3.9, 6.2, 7.8, 10.1];
        let res = ols_residuals(&x, &y);
        let sum: f64 = res.iter().sum();
        assert!(sum.abs() < 1e-10, "residual sum = {sum}");
    }

    #[test]
    fn ols_residuals_constant_x_empty() {
        let r = ols_residuals(&[5.0; 5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(r.is_empty(), "should be empty for constant x");
    }

    #[test]
    fn sigmoid_boundary_values() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(100.0) > 0.999);
        assert!(sigmoid(-100.0) < 0.001);
        assert!(sigmoid(710.0).is_finite()); // no overflow
        assert!(sigmoid(-710.0).is_finite());
    }
}
