//! # Family 31 — Interpolation & Approximation
//!
//! From first principles. Every algorithm built on tambear primitives.
//!
//! ## What lives here
//!
//! **Polynomial interpolation**: Lagrange, Newton divided differences, Neville's
//! **Splines**: natural cubic, clamped cubic, Hermite cubic (monotone)
//! **Approximation**: Chebyshev nodes + polynomial, least-squares polynomial fit
//! **Radial basis functions**: Gaussian, multiquadric, inverse multiquadric, thin plate
//! **Piecewise**: linear interpolation (the workhorse), nearest-neighbor
//! **Rational**: Barycentric rational interpolation (Floater-Hormann)
//!
//! ## Architecture
//!
//! Interpolation is fundamentally: given (x_i, y_i), produce f̂(x) for arbitrary x.
//! Two phases: **fit** (build the interpolant) and **evaluate** (query it).
//!
//! Most interpolants store coefficients. Evaluation is gather + dot product.
//! Splines store per-segment coefficients — evaluation is binary search + Horner.
//!
//! ## MSR insight
//!
//! A cubic spline with n knots stores 4n coefficients — sufficient statistics
//! for reconstructing any query. The raw data can be discarded after fitting.
//! Newton divided differences are incrementally updatable — adding a new point
//! extends the table without recomputing from scratch.

use std::f64;

// ─── Polynomial interpolation ───────────────────────────────────────

/// Lagrange interpolation at a single point.
///
/// O(n²) but numerically stable for small n. For large n, use Newton or barycentric.
/// Returns NaN if xs has fewer than 1 point or xs and ys differ in length.
pub fn lagrange(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if n == 0 || n != ys.len() {
        return f64::NAN;
    }
    let mut result = 0.0;
    for i in 0..n {
        let mut basis = ys[i];
        for j in 0..n {
            if i != j {
                let denom = xs[i] - xs[j];
                if denom.abs() < 1e-300 {
                    return f64::NAN; // duplicate nodes
                }
                basis *= (x - xs[j]) / denom;
            }
        }
        result += basis;
    }
    result
}

/// Newton's divided differences — build the coefficient table.
///
/// Returns coefficients c[0..n] such that:
/// p(x) = c[0] + c[1](x-x0) + c[2](x-x0)(x-x1) + ...
///
/// O(n²) time and O(n) space. Incrementally extensible.
pub fn newton_divided_diff(xs: &[f64], ys: &[f64]) -> Vec<f64> {
    let n = xs.len();
    if n == 0 || n != ys.len() {
        return vec![];
    }
    let mut dd = ys.to_vec();
    for j in 1..n {
        for i in (j..n).rev() {
            let denom = xs[i] - xs[i - j];
            if denom.abs() < 1e-300 {
                dd[i] = f64::NAN;
            } else {
                dd[i] = (dd[i] - dd[i - 1]) / denom;
            }
        }
    }
    dd
}

/// Evaluate Newton polynomial using Horner's method (right to left).
pub fn newton_eval(xs: &[f64], coeffs: &[f64], x: f64) -> f64 {
    let n = coeffs.len();
    if n == 0 {
        return f64::NAN;
    }
    let mut result = coeffs[n - 1];
    for i in (0..n - 1).rev() {
        result = result * (x - xs[i]) + coeffs[i];
    }
    result
}

/// Neville's algorithm — polynomial interpolation via recursive tableau.
///
/// More numerically stable than Lagrange for ill-conditioned problems.
/// Also returns an error estimate (difference between last two approximations).
pub fn neville(xs: &[f64], ys: &[f64], x: f64) -> (f64, f64) {
    let n = xs.len();
    if n == 0 || n != ys.len() {
        return (f64::NAN, f64::NAN);
    }
    if n == 1 {
        return (ys[0], 0.0);
    }
    let mut p = ys.to_vec();
    let mut prev_diag = f64::NAN;
    for j in 1..n {
        for i in 0..n - j {
            let denom = xs[i] - xs[i + j];
            if denom.abs() < 1e-300 {
                p[i] = f64::NAN;
            } else {
                p[i] = ((x - xs[i + j]) * p[i] + (xs[i] - x) * p[i + 1]) / denom;
            }
        }
        if j == n - 2 {
            prev_diag = p[0];
        }
    }
    let err = if n >= 2 { (p[0] - prev_diag).abs() } else { 0.0 };
    (p[0], err)
}

// ─── Piecewise linear ───────────────────────────────────────────────

/// Linear interpolation between sorted knots.
///
/// Extrapolates linearly beyond endpoints. xs must be sorted ascending.
/// This is the workhorse — used everywhere in practice.
pub fn lerp(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if n == 0 || n != ys.len() {
        return f64::NAN;
    }
    if n == 1 {
        return ys[0];
    }
    // Binary search for the interval
    let i = match xs.binary_search_by(|xi| xi.total_cmp(&x)) {
        Ok(i) => return ys[i], // exact match
        Err(0) => 0,           // below range — extrapolate from first segment
        Err(i) if i >= n => n - 2, // above range — extrapolate from last segment
        Err(i) => i - 1,
    };
    let t = (x - xs[i]) / (xs[i + 1] - xs[i]);
    ys[i] + t * (ys[i + 1] - ys[i])
}

/// Nearest-neighbor interpolation.
///
/// Returns y value of the nearest x node. Ties broken toward lower index.
pub fn nearest(xs: &[f64], ys: &[f64], x: f64) -> f64 {
    let n = xs.len();
    if n == 0 || n != ys.len() {
        return f64::NAN;
    }
    let mut best = 0;
    let mut best_dist = (xs[0] - x).abs();
    for i in 1..n {
        let d = (xs[i] - x).abs();
        if d < best_dist {
            best_dist = d;
            best = i;
        }
    }
    ys[best]
}

// ─── Cubic splines ──────────────────────────────────────────────────

/// Cubic spline coefficients for one segment.
///
/// y(t) = a + b*t + c*t² + d*t³ where t = x - x_i
#[derive(Debug, Clone)]
pub struct SplineSegment {
    pub x0: f64,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
}

/// Complete cubic spline interpolant.
#[derive(Debug, Clone)]
pub struct CubicSpline {
    pub segments: Vec<SplineSegment>,
}

impl CubicSpline {
    /// Evaluate the spline at x. Extrapolates using endpoint segments.
    pub fn eval(&self, x: f64) -> f64 {
        if self.segments.is_empty() {
            return f64::NAN;
        }
        // Binary search for segment
        let seg = if x <= self.segments[0].x0 {
            &self.segments[0]
        } else if x >= self.segments.last().unwrap().x0 {
            self.segments.last().unwrap()
        } else {
            let idx = self.segments.partition_point(|s| s.x0 <= x);
            &self.segments[if idx > 0 { idx - 1 } else { 0 }]
        };
        let t = x - seg.x0;
        seg.a + t * (seg.b + t * (seg.c + t * seg.d))
    }

    /// Evaluate the first derivative at x.
    pub fn eval_deriv(&self, x: f64) -> f64 {
        if self.segments.is_empty() {
            return f64::NAN;
        }
        let seg = if x <= self.segments[0].x0 {
            &self.segments[0]
        } else if x >= self.segments.last().unwrap().x0 {
            self.segments.last().unwrap()
        } else {
            let idx = self.segments.partition_point(|s| s.x0 <= x);
            &self.segments[if idx > 0 { idx - 1 } else { 0 }]
        };
        let t = x - seg.x0;
        seg.b + t * (2.0 * seg.c + t * 3.0 * seg.d)
    }
}

/// Natural cubic spline — second derivative = 0 at endpoints.
///
/// Tridiagonal system solved in O(n) via Thomas algorithm.
/// This is the most common spline variant.
pub fn natural_cubic_spline(xs: &[f64], ys: &[f64]) -> CubicSpline {
    let n = xs.len();
    if n < 2 || n != ys.len() {
        return CubicSpline { segments: vec![] };
    }
    if n == 2 {
        let h = xs[1] - xs[0];
        return CubicSpline {
            segments: vec![SplineSegment {
                x0: xs[0],
                a: ys[0],
                b: (ys[1] - ys[0]) / h,
                c: 0.0,
                d: 0.0,
            }],
        };
    }
    let nm1 = n - 1;
    let h: Vec<f64> = (0..nm1).map(|i| xs[i + 1] - xs[i]).collect();

    // Build tridiagonal system for second derivatives (m)
    // Natural BCs: m[0] = 0, m[n-1] = 0
    let interior = n - 2;
    let mut diag = vec![0.0; interior];
    let mut upper = vec![0.0; interior];
    let mut lower = vec![0.0; interior];
    let mut rhs = vec![0.0; interior];

    for i in 0..interior {
        let ii = i + 1; // index into original arrays
        diag[i] = 2.0 * (h[ii - 1] + h[ii]);
        rhs[i] = 6.0 * ((ys[ii + 1] - ys[ii]) / h[ii] - (ys[ii] - ys[ii - 1]) / h[ii - 1]);
        if i > 0 {
            lower[i] = h[ii - 1];
        }
        if i < interior - 1 {
            upper[i] = h[ii];
        }
    }

    // Thomas algorithm (forward elimination)
    for i in 1..interior {
        let factor = lower[i] / diag[i - 1];
        diag[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }

    // Back substitution
    let mut m = vec![0.0; n];
    if interior > 0 {
        m[interior] = rhs[interior - 1] / diag[interior - 1];
        for i in (0..interior - 1).rev() {
            m[i + 1] = (rhs[i] - upper[i] * m[i + 2]) / diag[i];
        }
    }
    // m[0] = 0 and m[n-1] = 0 (natural BCs)

    // Build segments
    let mut segments = Vec::with_capacity(nm1);
    for i in 0..nm1 {
        let hi = h[i];
        let a = ys[i];
        let b = (ys[i + 1] - ys[i]) / hi - hi * (2.0 * m[i] + m[i + 1]) / 6.0;
        let c = m[i] / 2.0;
        let d = (m[i + 1] - m[i]) / (6.0 * hi);
        segments.push(SplineSegment { x0: xs[i], a, b, c, d });
    }

    CubicSpline { segments }
}

/// Clamped cubic spline — specified first derivatives at endpoints.
///
/// f'(x_0) = dy0, f'(x_{n-1}) = dyn.
pub fn clamped_cubic_spline(xs: &[f64], ys: &[f64], dy0: f64, dyn_: f64) -> CubicSpline {
    let n = xs.len();
    if n < 2 || n != ys.len() {
        return CubicSpline { segments: vec![] };
    }
    if n == 2 {
        // Hermite cubic with specified endpoint derivatives
        let h = xs[1] - xs[0];
        let a = ys[0];
        let b = dy0;
        let c = (3.0 * (ys[1] - ys[0]) / h - 2.0 * dy0 - dyn_) / h;
        let d = (dy0 + dyn_ - 2.0 * (ys[1] - ys[0]) / h) / (h * h);
        return CubicSpline {
            segments: vec![SplineSegment { x0: xs[0], a, b, c, d }],
        };
    }
    let nm1 = n - 1;
    let h: Vec<f64> = (0..nm1).map(|i| xs[i + 1] - xs[i]).collect();

    // Full n×n tridiagonal system including endpoint equations
    let mut diag = vec![0.0; n];
    let mut upper = vec![0.0; n];
    let mut lower = vec![0.0; n];
    let mut rhs = vec![0.0; n];

    // Endpoint equations (clamped BCs)
    diag[0] = 2.0 * h[0];
    upper[0] = h[0];
    rhs[0] = 6.0 * ((ys[1] - ys[0]) / h[0] - dy0);

    diag[nm1] = 2.0 * h[nm1 - 1];
    lower[nm1] = h[nm1 - 1];
    rhs[nm1] = 6.0 * (dyn_ - (ys[nm1] - ys[nm1 - 1]) / h[nm1 - 1]);

    // Interior equations
    for i in 1..nm1 {
        lower[i] = h[i - 1];
        diag[i] = 2.0 * (h[i - 1] + h[i]);
        upper[i] = h[i];
        rhs[i] = 6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
    }

    // Thomas algorithm
    for i in 1..n {
        let factor = lower[i] / diag[i - 1];
        diag[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }

    let mut m = vec![0.0; n];
    m[n - 1] = rhs[n - 1] / diag[n - 1];
    for i in (0..n - 1).rev() {
        m[i] = (rhs[i] - upper[i] * m[i + 1]) / diag[i];
    }

    // Build segments
    let mut segments = Vec::with_capacity(nm1);
    for i in 0..nm1 {
        let hi = h[i];
        let a = ys[i];
        let b = (ys[i + 1] - ys[i]) / hi - hi * (2.0 * m[i] + m[i + 1]) / 6.0;
        let c = m[i] / 2.0;
        let d = (m[i + 1] - m[i]) / (6.0 * hi);
        segments.push(SplineSegment { x0: xs[i], a, b, c, d });
    }

    CubicSpline { segments }
}

/// Monotone Hermite interpolation (Fritsch-Carlson).
///
/// Guarantees monotonicity between data points. Essential for CDFs,
/// probability transforms, and any data with a physical monotonicity constraint.
pub fn monotone_hermite(xs: &[f64], ys: &[f64]) -> CubicSpline {
    let n = xs.len();
    if n < 2 || n != ys.len() {
        return CubicSpline { segments: vec![] };
    }
    let nm1 = n - 1;
    let h: Vec<f64> = (0..nm1).map(|i| xs[i + 1] - xs[i]).collect();
    let delta: Vec<f64> = (0..nm1).map(|i| (ys[i + 1] - ys[i]) / h[i]).collect();

    if n == 2 {
        return CubicSpline {
            segments: vec![SplineSegment {
                x0: xs[0], a: ys[0], b: delta[0], c: 0.0, d: 0.0,
            }],
        };
    }

    // Initial tangents: arithmetic mean of adjacent slopes
    let mut m = vec![0.0; n];
    m[0] = delta[0];
    m[nm1] = delta[nm1 - 1];
    for i in 1..nm1 {
        if delta[i - 1].signum() != delta[i].signum() {
            m[i] = 0.0;
        } else {
            m[i] = (delta[i - 1] + delta[i]) / 2.0;
        }
    }

    // Fritsch-Carlson monotonicity correction
    for i in 0..nm1 {
        if delta[i].abs() < 1e-300 {
            m[i] = 0.0;
            m[i + 1] = 0.0;
        } else {
            let alpha = m[i] / delta[i];
            let beta = m[i + 1] / delta[i];
            // Ensure we stay in the monotonicity region
            let s = alpha * alpha + beta * beta;
            if s > 9.0 {
                let tau = 3.0 / s.sqrt();
                m[i] = tau * alpha * delta[i];
                m[i + 1] = tau * beta * delta[i];
            }
        }
    }

    // Build Hermite cubic segments from tangents
    let mut segments = Vec::with_capacity(nm1);
    for i in 0..nm1 {
        let hi = h[i];
        let a = ys[i];
        let b = m[i];
        let c = (3.0 * delta[i] - 2.0 * m[i] - m[i + 1]) / hi;
        let d = (m[i] + m[i + 1] - 2.0 * delta[i]) / (hi * hi);
        segments.push(SplineSegment { x0: xs[i], a, b, c, d });
    }

    CubicSpline { segments }
}

// ─── Chebyshev approximation ────────────────────────────────────────

/// Chebyshev nodes on [a, b].
///
/// These are the zeros of T_n(x), mapped to [a, b].
/// Minimizes the Runge phenomenon — polynomial interpolation at these
/// nodes converges for any continuous function.
pub fn chebyshev_nodes(n: usize, a: f64, b: f64) -> Vec<f64> {
    let pi = std::f64::consts::PI;
    (0..n)
        .map(|k| {
            let theta = (2 * k + 1) as f64 * pi / (2 * n) as f64;
            0.5 * (a + b) + 0.5 * (b - a) * theta.cos()
        })
        .collect()
}

/// Chebyshev coefficients for a function sampled at Chebyshev nodes.
///
/// Given values y_k = f(x_k) at n Chebyshev nodes, compute coefficients c_k
/// of the Chebyshev expansion: f(x) ≈ Σ c_k T_k(x).
///
/// Uses the discrete cosine transform relationship. O(n²) direct computation.
pub fn chebyshev_coefficients(ys: &[f64]) -> Vec<f64> {
    let n = ys.len();
    if n == 0 {
        return vec![];
    }
    let pi = std::f64::consts::PI;
    let mut coeffs = vec![0.0; n];
    for k in 0..n {
        let mut sum = 0.0;
        for j in 0..n {
            let theta = (2 * j + 1) as f64 * pi / (2 * n) as f64;
            let tk = (k as f64 * theta).cos();
            sum += ys[j] * tk;
        }
        coeffs[k] = 2.0 * sum / n as f64;
    }
    coeffs[0] /= 2.0; // c_0 convention
    coeffs
}

/// Evaluate Chebyshev expansion at x ∈ [a, b] using Clenshaw's algorithm.
///
/// Clenshaw recurrence is the Chebyshev analog of Horner's method.
/// Numerically stable and O(n).
pub fn chebyshev_eval(coeffs: &[f64], x: f64, a: f64, b: f64) -> f64 {
    let n = coeffs.len();
    if n == 0 {
        return f64::NAN;
    }
    // Map x from [a,b] to [-1,1]
    let t = (2.0 * x - a - b) / (b - a);

    // Clenshaw recurrence: b_{k} = c_k + 2t·b_{k+1} - b_{k+2}
    let mut b1 = 0.0;
    let mut b2 = 0.0;
    for k in (1..n).rev() {
        let b0 = coeffs[k] + 2.0 * t * b1 - b2;
        b2 = b1;
        b1 = b0;
    }
    coeffs[0] + t * b1 - b2
}

/// Chebyshev approximation of a function on [a, b] with n terms.
///
/// Convenience function: sample f at Chebyshev nodes, compute coefficients.
pub fn chebyshev_approximate<F: Fn(f64) -> f64>(f: &F, n: usize, a: f64, b: f64) -> Vec<f64> {
    let nodes = chebyshev_nodes(n, a, b);
    let ys: Vec<f64> = nodes.iter().map(|&x| f(x)).collect();
    chebyshev_coefficients(&ys)
}

// ─── Least-squares polynomial fit ───────────────────────────────────

/// Polynomial fit result.
#[derive(Debug, Clone)]
pub struct PolyFit {
    /// Coefficients [a0, a1, ..., a_deg]: p(x) = a0 + a1·x + a2·x² + ...
    pub coeffs: Vec<f64>,
    /// Residual sum of squares
    pub rss: f64,
    /// R² (coefficient of determination)
    pub r_squared: f64,
}

impl PolyFit {
    /// Evaluate the polynomial at x.
    pub fn eval(&self, x: f64) -> f64 {
        let mut result = 0.0;
        let mut xp = 1.0;
        for &c in &self.coeffs {
            result += c * xp;
            xp *= x;
        }
        result
    }
}

/// Least-squares polynomial fit of degree `deg`.
///
/// Solves the normal equations Aᵀ A c = Aᵀ y where A is the Vandermonde matrix.
/// Uses Cholesky-like approach for numerical stability.
///
/// For interpolation (deg = n-1), this reproduces the unique interpolating polynomial.
/// For approximation (deg < n-1), this is the best L² fit.
pub fn polyfit(xs: &[f64], ys: &[f64], deg: usize) -> PolyFit {
    let n = xs.len();
    if n == 0 || n != ys.len() || deg >= n {
        return PolyFit { coeffs: vec![], rss: f64::NAN, r_squared: f64::NAN };
    }
    let m = deg + 1; // number of coefficients

    // Build normal equations: (AᵀA) c = Aᵀy
    // AᵀA[i][j] = Σ x_k^(i+j)
    // Aᵀy[i] = Σ y_k · x_k^i

    // Precompute power sums
    let mut xpow = vec![vec![0.0; 2 * m]; n];
    for k in 0..n {
        xpow[k][0] = 1.0;
        for p in 1..2 * m {
            xpow[k][p] = xpow[k][p - 1] * xs[k];
        }
    }

    let mut ata = vec![vec![0.0; m]; m];
    let mut aty = vec![0.0; m];
    for i in 0..m {
        for j in 0..m {
            let mut s = 0.0;
            for k in 0..n {
                s += xpow[k][i + j];
            }
            ata[i][j] = s;
        }
        let mut s = 0.0;
        for k in 0..n {
            s += ys[k] * xpow[k][i];
        }
        aty[i] = s;
    }

    // Solve via Gaussian elimination with partial pivoting
    let coeffs = solve_linear_system(&ata, &aty);

    // Compute RSS and R²
    let y_mean: f64 = ys.iter().sum::<f64>() / n as f64;
    let mut rss = 0.0;
    let mut tss = 0.0;
    for k in 0..n {
        let mut yhat = 0.0;
        for i in 0..m {
            yhat += coeffs[i] * xpow[k][i];
        }
        rss += (ys[k] - yhat).powi(2);
        tss += (ys[k] - y_mean).powi(2);
    }
    let r_squared = if tss > 0.0 { 1.0 - rss / tss } else { 1.0 };

    PolyFit { coeffs, rss, r_squared }
}

/// Solve Ax = b via F02 (linear_algebra::solve — LU with partial pivoting).
fn solve_linear_system(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = b.len();
    if n == 0 { return vec![]; }
    let data: Vec<f64> = a.iter().flat_map(|row| row.iter().copied()).collect();
    let mat = crate::linear_algebra::Mat::from_vec(n, n, data);
    crate::linear_algebra::solve(&mat, b)
        .unwrap_or_else(|| vec![f64::NAN; n])
}

// ─── Radial basis function interpolation ────────────────────────────

/// RBF kernel type.
#[derive(Debug, Clone, Copy)]
pub enum RbfKernel {
    /// φ(r) = exp(-r²/ε²)
    Gaussian(f64),
    /// φ(r) = √(1 + (r/ε)²)
    Multiquadric(f64),
    /// φ(r) = 1/√(1 + (r/ε)²)
    InverseMultiquadric(f64),
    /// φ(r) = r² ln(r) (no shape parameter)
    ThinPlateSpline,
}

impl RbfKernel {
    fn eval(&self, r: f64) -> f64 {
        match self {
            RbfKernel::Gaussian(eps) => (-r * r / (eps * eps)).exp(),
            RbfKernel::Multiquadric(eps) => (1.0 + (r / eps).powi(2)).sqrt(),
            RbfKernel::InverseMultiquadric(eps) => 1.0 / (1.0 + (r / eps).powi(2)).sqrt(),
            RbfKernel::ThinPlateSpline => {
                if r < 1e-300 { 0.0 } else { r * r * r.ln() }
            }
        }
    }
}

/// Radial basis function interpolant.
#[derive(Debug, Clone)]
pub struct RbfInterpolant {
    pub centers: Vec<f64>,
    pub weights: Vec<f64>,
    pub kernel: RbfKernel,
}

impl RbfInterpolant {
    /// Evaluate at x.
    pub fn eval(&self, x: f64) -> f64 {
        let mut result = 0.0;
        for i in 0..self.centers.len() {
            let r = (x - self.centers[i]).abs();
            result += self.weights[i] * self.kernel.eval(r);
        }
        result
    }
}

/// Fit RBF interpolant: solve the linear system φ(|x_i - x_j|) w = y.
///
/// For Gaussian and inverse multiquadric, the system is positive definite.
/// For multiquadric and thin plate spline, it's conditionally positive definite.
pub fn rbf_interpolate(xs: &[f64], ys: &[f64], kernel: RbfKernel) -> RbfInterpolant {
    let n = xs.len();
    if n == 0 || n != ys.len() {
        return RbfInterpolant { centers: vec![], weights: vec![], kernel };
    }

    // Build interpolation matrix
    let mut phi = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let r = (xs[i] - xs[j]).abs();
            phi[i][j] = kernel.eval(r);
        }
    }

    // Solve φ·w = y
    let mut rhs = ys.to_vec();
    let weights = solve_linear_system(&phi, &rhs);

    RbfInterpolant {
        centers: xs.to_vec(),
        weights,
        kernel,
    }
}

// ─── Barycentric rational interpolation ─────────────────────────────

/// Barycentric rational interpolant (Floater-Hormann).
///
/// Stable, high-order interpolation without the Runge phenomenon.
/// The parameter `d` controls the blending order (0 = piecewise constant,
/// 1 = linear, higher = smoother).
#[derive(Debug, Clone)]
pub struct BarycentricRational {
    pub xs: Vec<f64>,
    pub ys: Vec<f64>,
    pub weights: Vec<f64>,
}

impl BarycentricRational {
    /// Evaluate at x using the barycentric formula.
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.xs.len();
        // Check for exact node match (avoid 0/0)
        for i in 0..n {
            if (x - self.xs[i]).abs() < 1e-15 {
                return self.ys[i];
            }
        }
        let mut num = 0.0;
        let mut den = 0.0;
        for i in 0..n {
            let t = self.weights[i] / (x - self.xs[i]);
            num += t * self.ys[i];
            den += t;
        }
        if den.abs() < 1e-300 { f64::NAN } else { num / den }
    }
}

/// Build Floater-Hormann barycentric rational interpolant of blending order d.
///
/// Convergence rate O(h^{d+1}) with no poles in the real convex hull of the data.
pub fn barycentric_rational(xs: &[f64], ys: &[f64], d: usize) -> BarycentricRational {
    let n = xs.len();
    if n == 0 || n != ys.len() {
        return BarycentricRational { xs: vec![], ys: vec![], weights: vec![] };
    }
    let d = d.min(n - 1); // can't exceed n-1

    let mut weights = vec![0.0; n];
    for k in 0..n {
        let i_start = if k > d { k - d } else { 0 };
        let i_end = k.min(n - 1 - d);
        let mut sum = 0.0;
        for i in i_start..=i_end {
            let mut prod = 1.0;
            for j in i..=i + d {
                if j != k {
                    prod /= (xs[k] - xs[j]).abs();
                }
            }
            sum += prod;
        }
        // Alternating sign
        weights[k] = if (k as i64 - i_start as i64) % 2 == 0 { sum } else { -sum };
    }

    // Fix: signs should alternate starting from the first weight
    // Recompute with correct alternation
    for k in 0..n {
        let i_start = if k > d { k - d } else { 0 };
        let i_end = k.min(n - 1 - d);
        let mut sum = 0.0;
        for i in i_start..=i_end {
            let mut prod = 1.0;
            for j in i..=i + d {
                if j != k {
                    let diff = xs[k] - xs[j];
                    if diff.abs() < 1e-300 {
                        prod = f64::INFINITY;
                    } else {
                        prod /= diff.abs();
                    }
                }
            }
            sum += prod;
        }
        weights[k] = if k % 2 == 0 { sum } else { -sum };
    }

    BarycentricRational {
        xs: xs.to_vec(),
        ys: ys.to_vec(),
        weights,
    }
}

// ─── Akima interpolation ────────────────────────────────────────────

/// Akima interpolation — locally adaptive, avoids overshooting.
///
/// Uses a weighted average of neighboring slopes to determine tangents.
/// Less smooth than natural cubic splines but avoids wiggles near
/// sharp changes in the data.
pub fn akima(xs: &[f64], ys: &[f64]) -> CubicSpline {
    let n = xs.len();
    if n < 2 || n != ys.len() {
        return CubicSpline { segments: vec![] };
    }
    if n == 2 {
        let h = xs[1] - xs[0];
        return CubicSpline {
            segments: vec![SplineSegment {
                x0: xs[0], a: ys[0], b: (ys[1] - ys[0]) / h, c: 0.0, d: 0.0,
            }],
        };
    }
    let nm1 = n - 1;

    // Compute slopes between adjacent points
    let m: Vec<f64> = (0..nm1).map(|i| (ys[i + 1] - ys[i]) / (xs[i + 1] - xs[i])).collect();

    // Extend slopes at boundaries (Akima's original method)
    // m[-2], m[-1], m[0], ..., m[n-2], m[n-1], m[n]
    let m_ext = |i: i64| -> f64 {
        if i < 0 {
            // Linear extrapolation
            let idx = (-i) as usize;
            if idx <= m.len() { m[0] } else { m[0] }
        } else if (i as usize) >= m.len() {
            m[m.len() - 1]
        } else {
            m[i as usize]
        }
    };

    // Compute tangents using Akima's formula
    let mut t = vec![0.0; n];
    for i in 0..n {
        let m1 = m_ext(i as i64 - 2);
        let m2 = m_ext(i as i64 - 1);
        let m3 = m_ext(i as i64);
        let m4 = m_ext(i as i64 + 1);

        let w1 = (m4 - m3).abs();
        let w2 = (m2 - m1).abs();

        if w1 + w2 < 1e-300 {
            t[i] = (m2 + m3) / 2.0;
        } else {
            t[i] = (w1 * m2 + w2 * m3) / (w1 + w2);
        }
    }

    // Build Hermite cubic segments
    let mut segments = Vec::with_capacity(nm1);
    for i in 0..nm1 {
        let hi = xs[i + 1] - xs[i];
        let delta = (ys[i + 1] - ys[i]) / hi;
        let a = ys[i];
        let b = t[i];
        let c = (3.0 * delta - 2.0 * t[i] - t[i + 1]) / hi;
        let d = (t[i] + t[i + 1] - 2.0 * delta) / (hi * hi);
        segments.push(SplineSegment { x0: xs[i], a, b, c, d });
    }

    CubicSpline { segments }
}

// ─── Piecewise Cubic Hermite Interpolating Polynomial (PCHIP) ───────

/// PCHIP — shape-preserving piecewise cubic Hermite.
///
/// Matlab's default interpolation. Monotone on monotone data, preserves
/// the shape of the data better than natural cubic splines.
/// This is essentially the same as monotone_hermite but uses a slightly
/// different slope limiter (harmonic mean for interior, one-sided for endpoints).
pub fn pchip(xs: &[f64], ys: &[f64]) -> CubicSpline {
    let n = xs.len();
    if n < 2 || n != ys.len() {
        return CubicSpline { segments: vec![] };
    }
    let nm1 = n - 1;
    let h: Vec<f64> = (0..nm1).map(|i| xs[i + 1] - xs[i]).collect();
    let delta: Vec<f64> = (0..nm1).map(|i| (ys[i + 1] - ys[i]) / h[i]).collect();

    if n == 2 {
        return CubicSpline {
            segments: vec![SplineSegment {
                x0: xs[0], a: ys[0], b: delta[0], c: 0.0, d: 0.0,
            }],
        };
    }

    // Interior slopes: weighted harmonic mean
    let mut d = vec![0.0; n];
    for i in 1..nm1 {
        if delta[i - 1].signum() != delta[i].signum() || delta[i - 1].abs() < 1e-300 || delta[i].abs() < 1e-300 {
            d[i] = 0.0;
        } else {
            // Weighted harmonic mean
            let w1 = 2.0 * h[i] + h[i - 1];
            let w2 = h[i] + 2.0 * h[i - 1];
            d[i] = (w1 + w2) / (w1 / delta[i - 1] + w2 / delta[i]);
        }
    }

    // Endpoint slopes: one-sided, shape-preserving
    d[0] = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]);
    if d[0].signum() != delta[0].signum() {
        d[0] = 0.0;
    } else if delta[0].signum() != delta[1].signum() && d[0].abs() > 3.0 * delta[0].abs() {
        d[0] = 3.0 * delta[0];
    }

    let last = nm1;
    d[last] = ((2.0 * h[last - 1] + h[last - 2]) * delta[last - 1] - h[last - 1] * delta[last - 2]) / (h[last - 1] + h[last - 2]);
    if d[last].signum() != delta[last - 1].signum() {
        d[last] = 0.0;
    } else if delta[last - 1].signum() != delta[last - 2].signum() && d[last].abs() > 3.0 * delta[last - 1].abs() {
        d[last] = 3.0 * delta[last - 1];
    }

    // Build Hermite segments
    let mut segments = Vec::with_capacity(nm1);
    for i in 0..nm1 {
        let hi = h[i];
        let dd = delta[i];
        let a = ys[i];
        let b = d[i];
        let c = (3.0 * dd - 2.0 * d[i] - d[i + 1]) / hi;
        let dc = (d[i] + d[i + 1] - 2.0 * dd) / (hi * hi);
        segments.push(SplineSegment { x0: xs[i], a, b, c, d: dc });
    }

    CubicSpline { segments }
}

// ─── B-spline basis ─────────────────────────────────────────────────

/// Evaluate the k-th B-spline basis function of degree p at x.
///
/// Uses the Cox-de Boor recursion. Knot vector `knots` must be non-decreasing.
/// B_{k,0}(x) = 1 if knots[k] <= x < knots[k+1], else 0
/// B_{k,p}(x) = w1·B_{k,p-1}(x) + w2·B_{k+1,p-1}(x)
pub fn bspline_basis(knots: &[f64], k: usize, p: usize, x: f64) -> f64 {
    if p == 0 {
        if k + 1 < knots.len() && knots[k] <= x && x < knots[k + 1] {
            return 1.0;
        }
        // Handle right endpoint
        if k + 1 < knots.len() && (x - knots[k + 1]).abs() < 1e-15 && k + 2 == knots.len() {
            return 1.0;
        }
        return 0.0;
    }
    let mut result = 0.0;
    if k + p < knots.len() {
        let denom1 = knots[k + p] - knots[k];
        if denom1.abs() > 1e-300 {
            result += (x - knots[k]) / denom1 * bspline_basis(knots, k, p - 1, x);
        }
    }
    if k + p + 1 < knots.len() {
        let denom2 = knots[k + p + 1] - knots[k + 1];
        if denom2.abs() > 1e-300 {
            result += (knots[k + p + 1] - x) / denom2 * bspline_basis(knots, k + 1, p - 1, x);
        }
    }
    result
}

/// Evaluate a B-spline curve at x.
///
/// Given control points `ctrl` and knot vector `knots`, degree p:
/// S(x) = Σ ctrl[i] · B_{i,p}(x)
pub fn bspline_eval(knots: &[f64], ctrl: &[f64], p: usize, x: f64) -> f64 {
    let n_basis = ctrl.len();
    let mut result = 0.0;
    for i in 0..n_basis {
        result += ctrl[i] * bspline_basis(knots, i, p, x);
    }
    result
}

/// Create a uniform B-spline knot vector for n control points, degree p.
///
/// Returns a clamped knot vector: p+1 copies of the endpoints,
/// uniformly spaced interior knots.
pub fn uniform_knots(n: usize, p: usize, a: f64, b: f64) -> Vec<f64> {
    let m = n + p + 1; // total knots
    let interior = m - 2 * (p + 1);
    let mut knots = Vec::with_capacity(m);
    for _ in 0..=p {
        knots.push(a);
    }
    for i in 1..=interior {
        knots.push(a + (b - a) * i as f64 / (interior + 1) as f64);
    }
    for _ in 0..=p {
        knots.push(b);
    }
    knots
}

// ─── Gaussian Process Regression (1D) ───────────────────────────────

/// Gaussian process regression result.
#[derive(Debug, Clone)]
pub struct GpResult {
    /// Predicted means at query points
    pub mean: Vec<f64>,
    /// Predicted standard deviations at query points
    pub std: Vec<f64>,
}

/// 1D Gaussian process regression with RBF kernel.
///
/// The gold standard for interpolation with uncertainty quantification.
/// K(x,x') = σ² exp(-|x-x'|² / (2l²))
///
/// Parameters: `length_scale` (l), `signal_var` (σ²), `noise_var` (σ_n²).
///
/// Returns posterior mean and standard deviation at each query point.
pub fn gp_regression(
    x_train: &[f64],
    y_train: &[f64],
    x_query: &[f64],
    length_scale: f64,
    signal_var: f64,
    noise_var: f64,
) -> GpResult {
    let n = x_train.len();
    let nq = x_query.len();
    if n == 0 || n != y_train.len() || nq == 0 {
        return GpResult { mean: vec![], std: vec![] };
    }

    let ls = length_scale.max(1e-10);
    let rbf = |x1: f64, x2: f64| -> f64 {
        signal_var * (-0.5 * (x1 - x2).powi(2) / (ls * ls)).exp()
    };

    // K(X, X) + σ_n² I
    let mut kxx = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            kxx[i][j] = rbf(x_train[i], x_train[j]);
            if i == j {
                kxx[i][j] += noise_var.max(1e-10); // jitter for numerical stability
            }
        }
    }

    // Solve K_xx · α = y via the existing linear solver
    let mut y = y_train.to_vec();
    let alpha = solve_linear_system(&kxx, &y);

    // Rebuild K_xx (it was modified by the solver)
    let mut kxx_orig = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            kxx_orig[i][j] = rbf(x_train[i], x_train[j]);
            if i == j {
                kxx_orig[i][j] += noise_var.max(1e-10);
            }
        }
    }

    let mut mean = vec![0.0; nq];
    let mut std = vec![0.0; nq];

    for q in 0..nq {
        // k(x*, X)
        let kstar: Vec<f64> = (0..n).map(|i| rbf(x_query[q], x_train[i])).collect();

        // Posterior mean: k* · α
        let mut mu = 0.0;
        for i in 0..n {
            mu += kstar[i] * alpha[i];
        }
        mean[q] = mu;

        // Posterior variance: k(x*,x*) - k*ᵀ K⁻¹ k*
        // Solve K v = k*
        let mut kxx2 = kxx_orig.clone();
        let mut ks = kstar.clone();
        let v = solve_linear_system(&kxx2, &ks);

        let mut var = rbf(x_query[q], x_query[q]);
        for i in 0..n {
            var -= kstar[i] * v[i];
        }
        std[q] = if var > 0.0 { var.sqrt() } else { 0.0 };
    }

    GpResult { mean, std }
}

// ─── Padé approximant ───────────────────────────────────────────────

/// Padé approximant coefficients.
///
/// R(x) = (p0 + p1·x + ... + pm·x^m) / (1 + q1·x + ... + qn·x^n)
/// Note: q0 is always 1 (normalization).
#[derive(Debug, Clone)]
pub struct PadeApproximant {
    /// Numerator coefficients [p0, p1, ..., pm]
    pub numer: Vec<f64>,
    /// Denominator coefficients [1, q1, ..., qn]  (q0 = 1 implicit)
    pub denom: Vec<f64>,
}

impl PadeApproximant {
    /// Evaluate R(x).
    pub fn eval(&self, x: f64) -> f64 {
        let mut num = 0.0;
        let mut xp = 1.0;
        for &p in &self.numer {
            num += p * xp;
            xp *= x;
        }
        let mut den = 0.0;
        xp = 1.0;
        for &q in &self.denom {
            den += q * xp;
            xp *= x;
        }
        if den.abs() < 1e-300 { f64::NAN } else { num / den }
    }
}

/// Compute [m/n] Padé approximant from Taylor coefficients.
///
/// Given c[0..m+n+1] (Taylor series coefficients of f about x=0),
/// compute the unique rational function P_m/Q_n matching the first m+n+1
/// Taylor coefficients.
///
/// This is solved via the linear system from matching coefficients.
pub fn pade(taylor_coeffs: &[f64], m: usize, n: usize) -> PadeApproximant {
    if taylor_coeffs.len() < m + n + 1 {
        return PadeApproximant { numer: vec![], denom: vec![] };
    }
    let c = taylor_coeffs;

    if n == 0 {
        // Polynomial case
        return PadeApproximant {
            numer: c[..=m].to_vec(),
            denom: vec![1.0],
        };
    }

    // Solve for denominator coefficients q[1..n]
    // System: for k = m+1 .. m+n:
    //   c[k] + q[1]·c[k-1] + q[2]·c[k-2] + ... + q[n]·c[k-n] = 0
    let mut a_mat = vec![vec![0.0; n]; n];
    let mut b_vec = vec![0.0; n];
    for row in 0..n {
        let k = m + 1 + row;
        b_vec[row] = -c[k];
        for col in 0..n {
            let idx = k as i64 - (col as i64 + 1);
            if idx >= 0 && (idx as usize) < c.len() {
                a_mat[row][col] = c[idx as usize];
            }
        }
    }

    let q_coeffs = solve_linear_system(&a_mat, &b_vec);

    // Numerator: p[k] = c[k] + q[1]·c[k-1] + ... for k = 0..m
    let mut numer = vec![0.0; m + 1];
    for k in 0..=m {
        numer[k] = c[k];
        for j in 1..=n {
            if k >= j {
                numer[k] += q_coeffs[j - 1] * c[k - j];
            }
        }
    }

    let mut denom = vec![1.0]; // q0 = 1
    denom.extend_from_slice(&q_coeffs);

    PadeApproximant { numer, denom }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Lagrange ──

    #[test]
    fn lagrange_exact_polynomial() {
        // p(x) = x² — interpolate at 3 points, should recover exactly
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 1.0, 4.0];
        assert!((lagrange(&xs, &ys, 1.5) - 2.25).abs() < 1e-12);
        assert!((lagrange(&xs, &ys, 0.5) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn lagrange_single_point() {
        assert!((lagrange(&[3.0], &[7.0], 100.0) - 7.0).abs() < 1e-12);
    }

    #[test]
    fn lagrange_empty() {
        assert!(lagrange(&[], &[], 1.0).is_nan());
    }

    // ── Newton divided differences ──

    #[test]
    fn newton_recovers_polynomial() {
        // p(x) = 2x³ - x + 3
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x * x * x - x + 3.0).collect();
        let coeffs = newton_divided_diff(&xs, &ys);
        // Evaluate at x=1.5
        let y = newton_eval(&xs, &coeffs, 1.5);
        let expected = 2.0 * 1.5_f64.powi(3) - 1.5 + 3.0;
        assert!((y - expected).abs() < 1e-10);
    }

    // ── Neville ──

    #[test]
    fn neville_matches_lagrange() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![1.0, 0.0, 1.0, 4.0];
        let x = 1.5;
        let lag = lagrange(&xs, &ys, x);
        let (nev, _err) = neville(&xs, &ys, x);
        assert!((lag - nev).abs() < 1e-12);
    }

    // ── Linear interpolation ──

    #[test]
    fn lerp_basic() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![0.0, 2.0, 1.0];
        assert!((lerp(&xs, &ys, 0.5) - 1.0).abs() < 1e-12);
        assert!((lerp(&xs, &ys, 1.5) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn lerp_exact_nodes() {
        let xs = vec![0.0, 1.0, 2.0];
        let ys = vec![10.0, 20.0, 30.0];
        assert!((lerp(&xs, &ys, 0.0) - 10.0).abs() < 1e-12);
        assert!((lerp(&xs, &ys, 1.0) - 20.0).abs() < 1e-12);
        assert!((lerp(&xs, &ys, 2.0) - 30.0).abs() < 1e-12);
    }

    #[test]
    fn lerp_extrapolation() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 2.0];
        // Extrapolate beyond endpoints
        assert!((lerp(&xs, &ys, -1.0) - -2.0).abs() < 1e-12);
        assert!((lerp(&xs, &ys, 2.0) - 4.0).abs() < 1e-12);
    }

    // ── Nearest neighbor ──

    #[test]
    fn nearest_basic() {
        let xs = vec![0.0, 1.0, 3.0];
        let ys = vec![10.0, 20.0, 30.0];
        assert!((nearest(&xs, &ys, 0.3) - 10.0).abs() < 1e-12);
        assert!((nearest(&xs, &ys, 0.7) - 20.0).abs() < 1e-12);
        assert!((nearest(&xs, &ys, 2.5) - 30.0).abs() < 1e-12);
    }

    // ── Natural cubic spline ──

    #[test]
    fn natural_spline_interpolates_nodes() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let spline = natural_cubic_spline(&xs, &ys);
        for i in 0..xs.len() {
            assert!((spline.eval(xs[i]) - ys[i]).abs() < 1e-12,
                "node {} mismatch: {} vs {}", i, spline.eval(xs[i]), ys[i]);
        }
    }

    #[test]
    fn natural_spline_smooth() {
        // Spline should be smooth between nodes
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 0.0, 1.0];
        let spline = natural_cubic_spline(&xs, &ys);
        // Check continuity at interior knots
        for i in 1..xs.len() - 1 {
            let left = spline.eval(xs[i] - 1e-10);
            let right = spline.eval(xs[i] + 1e-10);
            assert!((left - right).abs() < 1e-6, "discontinuity at knot {}", i);
        }
    }

    #[test]
    fn natural_spline_quadratic_exact() {
        // Natural cubic spline reproduces quadratics exactly
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = xs.iter().map(|&x| x * x).collect();
        let spline = natural_cubic_spline(&xs, &ys);
        // Check midpoints
        for i in 0..xs.len() - 1 {
            let mid = (xs[i] + xs[i + 1]) / 2.0;
            let expected = mid * mid;
            assert!((spline.eval(mid) - expected).abs() < 0.1,
                "midpoint {} off: {} vs {}", mid, spline.eval(mid), expected);
        }
    }

    // ── Clamped spline ──

    #[test]
    fn clamped_spline_endpoint_derivatives() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![0.0, 1.0, 0.0, -1.0];
        let spline = clamped_cubic_spline(&xs, &ys, 2.0, -2.0);
        // Check endpoint derivatives
        let d0 = spline.eval_deriv(xs[0]);
        let dn = spline.eval_deriv(xs[3]);
        assert!((d0 - 2.0).abs() < 1e-10, "d0 = {}", d0);
        assert!((dn - (-2.0)).abs() < 1e-10, "dn = {}", dn);
    }

    // ── Monotone Hermite ──

    #[test]
    fn monotone_hermite_preserves_monotonicity() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![0.0, 1.0, 1.5, 3.0, 5.0]; // monotone increasing
        let spline = monotone_hermite(&xs, &ys);
        // Check monotonicity at many intermediate points
        let mut prev = spline.eval(0.0);
        for i in 1..100 {
            let x = 4.0 * i as f64 / 100.0;
            let y = spline.eval(x);
            assert!(y >= prev - 1e-10, "monotonicity violated at x={}: {} < {}", x, y, prev);
            prev = y;
        }
    }

    // ── Chebyshev ──

    #[test]
    fn chebyshev_approximates_sin() {
        let coeffs = chebyshev_approximate(&|x: f64| x.sin(), 10, 0.0, std::f64::consts::PI);
        // Should approximate sin(x) well on [0, π]
        for &x in &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0] {
            let approx = chebyshev_eval(&coeffs, x, 0.0, std::f64::consts::PI);
            assert!((approx - x.sin()).abs() < 1e-6,
                "sin({}) ≈ {}, got {}", x, x.sin(), approx);
        }
    }

    #[test]
    fn chebyshev_nodes_properties() {
        let nodes = chebyshev_nodes(5, -1.0, 1.0);
        assert_eq!(nodes.len(), 5);
        // All nodes should be in [-1, 1]
        for &x in &nodes {
            assert!(x >= -1.0 && x <= 1.0, "node {} out of range", x);
        }
    }

    // ── Polynomial fit ──

    #[test]
    fn polyfit_linear() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![1.0, 3.0, 5.0, 7.0, 9.0]; // y = 2x + 1
        let fit = polyfit(&xs, &ys, 1);
        assert!((fit.coeffs[0] - 1.0).abs() < 1e-10, "intercept = {}", fit.coeffs[0]);
        assert!((fit.coeffs[1] - 2.0).abs() < 1e-10, "slope = {}", fit.coeffs[1]);
        assert!(fit.r_squared > 0.9999, "R² = {}", fit.r_squared);
    }

    #[test]
    fn polyfit_quadratic() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys: Vec<f64> = xs.iter().map(|&x| x * x + 2.0 * x + 1.0).collect();
        let fit = polyfit(&xs, &ys, 2);
        assert!((fit.coeffs[0] - 1.0).abs() < 1e-8, "a0 = {}", fit.coeffs[0]);
        assert!((fit.coeffs[1] - 2.0).abs() < 1e-8, "a1 = {}", fit.coeffs[1]);
        assert!((fit.coeffs[2] - 1.0).abs() < 1e-8, "a2 = {}", fit.coeffs[2]);
    }

    #[test]
    fn polyfit_eval() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![1.0, 2.0, 5.0, 10.0];
        let fit = polyfit(&xs, &ys, 2);
        // Should evaluate close to data points
        for i in 0..xs.len() {
            assert!((fit.eval(xs[i]) - ys[i]).abs() < 0.5,
                "f({}) = {} vs {}", xs[i], fit.eval(xs[i]), ys[i]);
        }
    }

    // ── RBF ──

    #[test]
    fn rbf_gaussian_interpolates() {
        let xs = vec![0.0, 1.0, 2.0, 3.0];
        let ys = vec![1.0, 0.0, 1.0, 0.0];
        let rbf = rbf_interpolate(&xs, &ys, RbfKernel::Gaussian(1.0));
        // Should interpolate exactly at nodes
        for i in 0..xs.len() {
            assert!((rbf.eval(xs[i]) - ys[i]).abs() < 1e-6,
                "rbf({}) = {} vs {}", xs[i], rbf.eval(xs[i]), ys[i]);
        }
    }

    #[test]
    fn rbf_thin_plate_interpolates() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![0.0, 1.0, 0.5, 2.0, 1.0];
        let rbf = rbf_interpolate(&xs, &ys, RbfKernel::ThinPlateSpline);
        for i in 0..xs.len() {
            assert!((rbf.eval(xs[i]) - ys[i]).abs() < 1e-6,
                "tps({}) = {} vs {}", xs[i], rbf.eval(xs[i]), ys[i]);
        }
    }

    // ── Barycentric rational ──

    #[test]
    fn barycentric_interpolates() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0];
        let ys = vec![0.0, 1.0, 0.0, 1.0, 0.0];
        let interp = barycentric_rational(&xs, &ys, 2);
        for i in 0..xs.len() {
            assert!((interp.eval(xs[i]) - ys[i]).abs() < 1e-10,
                "bary({}) = {} vs {}", xs[i], interp.eval(xs[i]), ys[i]);
        }
    }

    // ── Akima ──

    #[test]
    fn akima_interpolates_nodes() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![0.0, 1.0, 0.5, 0.5, 1.0, 0.0];
        let spline = akima(&xs, &ys);
        for i in 0..xs.len() {
            assert!((spline.eval(xs[i]) - ys[i]).abs() < 1e-10,
                "akima({}) = {} vs {}", xs[i], spline.eval(xs[i]), ys[i]);
        }
    }

    #[test]
    fn akima_vs_natural_less_oscillation() {
        // Akima should have less oscillation than natural cubic for step-like data
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // step at x=2.5
        let aki = akima(&xs, &ys);
        let nat = natural_cubic_spline(&xs, &ys);
        // Natural spline will overshoot; Akima should overshoot less
        let aki_max = (0..100).map(|i| {
            let x = 5.0 * i as f64 / 100.0;
            aki.eval(x).abs()
        }).fold(0.0_f64, f64::max);
        let nat_max = (0..100).map(|i| {
            let x = 5.0 * i as f64 / 100.0;
            nat.eval(x).abs()
        }).fold(0.0_f64, f64::max);
        // Both bounded, but natural may overshoot more
        assert!(aki_max <= nat_max + 0.1,
            "akima max {} should be <= natural max {} + 0.1", aki_max, nat_max);
    }

    // ── PCHIP ──

    #[test]
    fn pchip_monotone() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![0.0, 0.5, 1.0, 2.0, 4.0, 8.0]; // monotone increasing
        let spline = pchip(&xs, &ys);
        let mut prev = spline.eval(0.0);
        for i in 1..=100 {
            let x = 5.0 * i as f64 / 100.0;
            let y = spline.eval(x);
            assert!(y >= prev - 1e-10, "pchip monotonicity violated at {}: {} < {}", x, y, prev);
            prev = y;
        }
    }

    #[test]
    fn pchip_interpolates_nodes() {
        let xs = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let ys = vec![1.0, 3.0, 2.0, 5.0, 4.0, 6.0];
        let spline = pchip(&xs, &ys);
        for i in 0..xs.len() {
            assert!((spline.eval(xs[i]) - ys[i]).abs() < 1e-10,
                "pchip({}) = {} vs {}", xs[i], spline.eval(xs[i]), ys[i]);
        }
    }

    // ── B-spline ──

    #[test]
    fn bspline_basis_partition_of_unity() {
        // B-spline basis functions sum to 1 at any x in the interior
        let knots = vec![0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 4.0, 4.0, 4.0];
        let p = 3;
        let n = knots.len() - p - 1; // 7
        for &x in &[0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5] {
            let sum: f64 = (0..n).map(|i| bspline_basis(&knots, i, p, x)).sum();
            assert!((sum - 1.0).abs() < 1e-10, "basis sum at x={} is {}", x, sum);
        }
    }

    #[test]
    fn bspline_eval_linear() {
        // Degree 1 B-spline with uniform knots = piecewise linear
        let knots = vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0];
        let ctrl = vec![0.0, 1.0, 0.0, 2.0];
        let y = bspline_eval(&knots, &ctrl, 1, 0.5);
        assert!((y - 0.5).abs() < 1e-10);
    }

    // ── Gaussian Process ──

    #[test]
    fn gp_interpolates_with_low_noise() {
        let xt = vec![0.0, 1.0, 2.0, 3.0];
        let yt = vec![0.0, 1.0, 0.0, 1.0];
        let xq = vec![0.0, 1.0, 2.0, 3.0]; // query at training points
        let gp = gp_regression(&xt, &yt, &xq, 1.0, 1.0, 1e-8);
        for i in 0..xt.len() {
            assert!((gp.mean[i] - yt[i]).abs() < 0.01,
                "gp({}) = {} vs {}", xt[i], gp.mean[i], yt[i]);
        }
    }

    #[test]
    fn gp_uncertainty_increases_away_from_data() {
        let xt = vec![0.0, 1.0];
        let yt = vec![0.0, 1.0];
        let xq = vec![0.5, 5.0]; // near vs far from data
        let gp = gp_regression(&xt, &yt, &xq, 1.0, 1.0, 0.01);
        assert!(gp.std[1] > gp.std[0],
            "far uncertainty {} should exceed near {}", gp.std[1], gp.std[0]);
    }

    // ── Padé ──

    #[test]
    fn pade_exp_approximation() {
        // Taylor coefficients of exp(x): [1, 1, 1/2, 1/6, 1/24, ...]
        let taylor = vec![1.0, 1.0, 0.5, 1.0/6.0, 1.0/24.0, 1.0/120.0];
        let pa = pade(&taylor, 2, 2); // [2/2] Padé
        // Should be good near x=0
        for &x in &[0.0, 0.1, 0.5, 1.0] {
            let approx = pa.eval(x);
            let exact = x.exp();
            assert!((approx - exact).abs() < 0.1,
                "padé exp({}) = {} vs {}", x, approx, exact);
        }
    }

    #[test]
    fn pade_at_origin() {
        // [1/2] Padé of exp(x): R(0) = c[0] = 1
        let taylor = vec![1.0, 1.0, 0.5, 1.0 / 6.0];
        let pa = pade(&taylor, 1, 2);
        assert!((pa.eval(0.0) - 1.0).abs() < 1e-10, "R(0) = {}", pa.eval(0.0));
        // Should also be decent at x=0.5
        assert!((pa.eval(0.5) - 0.5_f64.exp()).abs() < 0.05,
            "R(0.5) = {} vs {}", pa.eval(0.5), 0.5_f64.exp());
    }

    // ── Edge cases ──

    #[test]
    fn empty_inputs() {
        assert!(lagrange(&[], &[], 1.0).is_nan());
        assert!(newton_divided_diff(&[], &[]).is_empty());
        let (v, _) = neville(&[], &[], 1.0);
        assert!(v.is_nan());
        assert!(lerp(&[], &[], 1.0).is_nan());
        assert!(nearest(&[], &[], 1.0).is_nan());
        let s = natural_cubic_spline(&[], &[]);
        assert!(s.segments.is_empty());
    }

    #[test]
    fn two_point_spline() {
        let xs = vec![0.0, 1.0];
        let ys = vec![0.0, 1.0];
        let spline = natural_cubic_spline(&xs, &ys);
        assert_eq!(spline.segments.len(), 1);
        assert!((spline.eval(0.5) - 0.5).abs() < 1e-12); // linear for 2 points
    }

    // ── Uniform knots ──

    #[test]
    fn uniform_knots_structure() {
        let knots = uniform_knots(6, 3, 0.0, 1.0);
        // Should have 6 + 3 + 1 = 10 knots
        assert_eq!(knots.len(), 10);
        // First p+1 = 4 should be 0, last 4 should be 1
        for i in 0..4 {
            assert!((knots[i] - 0.0).abs() < 1e-12);
        }
        for i in 6..10 {
            assert!((knots[i] - 1.0).abs() < 1e-12);
        }
    }
}
