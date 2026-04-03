//! Numerical methods — root finding, quadrature, ODE solvers, differentiation.
//!
//! ## Architecture
//!
//! These are the irreducible computational atoms. Every numerical algorithm
//! eventually calls one of these. No vendor libraries, no fallbacks.
//!
//! - **Root finding**: Newton, bisection, Brent, secant — find x where f(x)=0
//! - **Quadrature**: Simpson, Gauss-Legendre, adaptive — compute ∫f(x)dx
//! - **ODE solvers**: Euler, RK4, RK45 (adaptive) — solve dy/dt = f(t,y)
//! - **Differentiation**: finite differences, complex step, Richardson extrapolation
//! - **Monte Carlo**: MC integration for high-dimensional integrals
//!
//! ## .tbs integration
//!
//! ```text
//! root(f, a, b)                  # Brent's method
//! integrate(f, a, b)             # adaptive quadrature
//! ode_solve(f, y0, t_span)       # RK45
//! ```

// ═══════════════════════════════════════════════════════════════════════════
// Root finding
// ═══════════════════════════════════════════════════════════════════════════

/// Result of a root-finding algorithm.
#[derive(Debug, Clone)]
pub struct RootResult {
    pub root: f64,
    pub iterations: usize,
    pub converged: bool,
    pub function_value: f64,
}

/// Bisection method: guaranteed convergence for continuous f with f(a)*f(b) < 0.
///
/// Halves the interval each step. Convergence rate: linear (1 bit per iteration).
/// Always converges if the sign change exists. O(log₂((b-a)/tol)) iterations.
pub fn bisection(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> RootResult {
    let mut fa = f(a);
    let mut _fb = f(b);

    for iter in 0..max_iter {
        let c = (a + b) / 2.0;
        let fc = f(c);

        if fc.abs() < tol || (b - a) / 2.0 < tol {
            return RootResult { root: c, iterations: iter + 1, converged: true, function_value: fc };
        }

        if fa * fc < 0.0 {
            b = c;
            _fb = fc;
        } else {
            a = c;
            fa = fc;
        }
    }

    let c = (a + b) / 2.0;
    RootResult { root: c, iterations: max_iter, converged: false, function_value: f(c) }
}

/// Newton's method: quadratic convergence near simple roots.
///
/// x_{n+1} = x_n - f(x_n) / f'(x_n)
///
/// Requires derivative. Fastest near the root but can diverge if started far away.
pub fn newton(
    f: impl Fn(f64) -> f64,
    df: impl Fn(f64) -> f64,
    x0: f64,
    tol: f64,
    max_iter: usize,
) -> RootResult {
    let mut x = x0;

    for iter in 0..max_iter {
        let fx = f(x);
        if fx.abs() < tol {
            return RootResult { root: x, iterations: iter + 1, converged: true, function_value: fx };
        }
        let dfx = df(x);
        if dfx.abs() < 1e-15 {
            return RootResult { root: x, iterations: iter + 1, converged: false, function_value: fx };
        }
        x -= fx / dfx;
    }

    let fx = f(x);
    RootResult { root: x, iterations: max_iter, converged: false, function_value: fx }
}

/// Secant method: superlinear convergence without requiring derivatives.
///
/// x_{n+1} = x_n - f(x_n) × (x_n - x_{n-1}) / (f(x_n) - f(x_{n-1}))
///
/// Convergence order ≈ 1.618 (golden ratio). No derivative needed.
pub fn secant(f: impl Fn(f64) -> f64, x0: f64, x1: f64, tol: f64, max_iter: usize) -> RootResult {
    let mut xp = x0;
    let mut xc = x1;
    let mut fp = f(xp);

    for iter in 0..max_iter {
        let fc = f(xc);
        if fc.abs() < tol {
            return RootResult { root: xc, iterations: iter + 1, converged: true, function_value: fc };
        }
        let denom = fc - fp;
        if denom.abs() < 1e-15 {
            return RootResult { root: xc, iterations: iter + 1, converged: false, function_value: fc };
        }
        let xn = xc - fc * (xc - xp) / denom;
        xp = xc;
        fp = fc;
        xc = xn;
    }

    let fc = f(xc);
    RootResult { root: xc, iterations: max_iter, converged: false, function_value: fc }
}

/// Brent's method: combines bisection, secant, and inverse quadratic interpolation.
///
/// Guaranteed convergence like bisection, but typically converges much faster.
/// The default choice for root-finding — robust and fast.
pub fn brent(f: impl Fn(f64) -> f64, mut a: f64, mut b: f64, tol: f64, max_iter: usize) -> RootResult {
    let mut fa = f(a);
    let mut fb = f(b);

    if fa * fb > 0.0 {
        return RootResult { root: f64::NAN, iterations: 0, converged: false, function_value: f64::NAN };
    }

    if fa.abs() < fb.abs() {
        std::mem::swap(&mut a, &mut b);
        std::mem::swap(&mut fa, &mut fb);
    }

    let mut c = a;
    let mut fc = fa;
    let mut d = b - a;
    let mut e = d;

    for iter in 0..max_iter {
        if fb.abs() < tol {
            return RootResult { root: b, iterations: iter + 1, converged: true, function_value: fb };
        }

        if fa != fc && fb != fc {
            // Inverse quadratic interpolation
            let s = a * fb * fc / ((fa - fb) * (fa - fc))
                + b * fa * fc / ((fb - fa) * (fb - fc))
                + c * fa * fb / ((fc - fa) * (fc - fb));

            // Check if IQI step is acceptable
            let min_ab = if (3.0 * a + b) / 4.0 < b { (3.0 * a + b) / 4.0 } else { b };
            let max_ab = if (3.0 * a + b) / 4.0 > b { (3.0 * a + b) / 4.0 } else { b };

            if s >= min_ab && s <= max_ab && (s - b).abs() < e.abs() / 2.0 {
                e = d;
                d = s - b;
            } else {
                d = (a - b) / 2.0;
                e = d;
            }
        } else {
            // Secant step
            let s = b - fb * (b - a) / (fb - fa);
            let min_ab = if (3.0 * a + b) / 4.0 < b { (3.0 * a + b) / 4.0 } else { b };
            let max_ab = if (3.0 * a + b) / 4.0 > b { (3.0 * a + b) / 4.0 } else { b };

            if s >= min_ab && s <= max_ab && (s - b).abs() < e.abs() / 2.0 {
                e = d;
                d = s - b;
            } else {
                d = (a - b) / 2.0;
                e = d;
            }
        }

        c = b;
        fc = fb;

        if d.abs() > tol {
            b += d;
        } else {
            b += if a > b { -tol } else { tol };
        }
        fb = f(b);

        if (fb > 0.0 && fa > 0.0) || (fb < 0.0 && fa < 0.0) {
            a = c;
            fa = fc;
            d = b - c;
            e = d;
        }
    }

    RootResult { root: b, iterations: max_iter, converged: false, function_value: fb }
}

// ═══════════════════════════════════════════════════════════════════════════
// Numerical differentiation
// ═══════════════════════════════════════════════════════════════════════════

/// Central difference derivative: f'(x) ≈ (f(x+h) - f(x-h)) / 2h.
///
/// Error: O(h²). Default h = x × ε^(1/3) where ε = machine epsilon.
pub fn derivative_central(f: impl Fn(f64) -> f64, x: f64, h: Option<f64>) -> f64 {
    let h = h.unwrap_or_else(|| {
        let eps = f64::EPSILON;
        if x.abs() > 1.0 { x.abs() * eps.cbrt() } else { eps.cbrt() }
    });
    (f(x + h) - f(x - h)) / (2.0 * h)
}

/// Second derivative via central differences: f''(x) ≈ (f(x+h) - 2f(x) + f(x-h)) / h².
pub fn derivative2_central(f: impl Fn(f64) -> f64, x: f64, h: Option<f64>) -> f64 {
    let h = h.unwrap_or_else(|| {
        let eps = f64::EPSILON;
        if x.abs() > 1.0 { x.abs() * eps.powf(0.25) } else { eps.powf(0.25) }
    });
    (f(x + h) - 2.0 * f(x) + f(x - h)) / (h * h)
}

/// Richardson extrapolation for higher-accuracy derivatives.
///
/// Combines derivative estimates at different step sizes to cancel error terms.
/// Uses Neville's algorithm on the derivative tableau.
pub fn derivative_richardson(f: impl Fn(f64) -> f64, x: f64, h0: f64, n_steps: usize) -> f64 {
    let mut tableau = vec![vec![0.0; n_steps]; n_steps];

    // Fill first column with central differences at h, h/2, h/4, ...
    let mut h = h0;
    for i in 0..n_steps {
        tableau[i][0] = (f(x + h) - f(x - h)) / (2.0 * h);
        h /= 2.0;
    }

    // Richardson extrapolation
    for j in 1..n_steps {
        let factor = 4.0_f64.powi(j as i32);
        for i in j..n_steps {
            tableau[i][j] = (factor * tableau[i][j-1] - tableau[i-1][j-1]) / (factor - 1.0);
        }
    }

    tableau[n_steps - 1][n_steps - 1]
}

// ═══════════════════════════════════════════════════════════════════════════
// Quadrature (numerical integration)
// ═══════════════════════════════════════════════════════════════════════════

/// Simpson's rule: ∫_a^b f(x)dx ≈ (h/3)(f(a) + 4f(m) + f(b)) (composite).
///
/// Error: O(h⁴). Good for smooth functions. n must be even.
pub fn simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    let n = if n % 2 == 0 { n } else { n + 1 }; // Ensure even
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);

    for i in 1..n {
        let x = a + i as f64 * h;
        if i % 2 == 0 {
            sum += 2.0 * f(x);
        } else {
            sum += 4.0 * f(x);
        }
    }

    sum * h / 3.0
}

/// Gauss-Legendre quadrature (5-point).
///
/// ∫_{-1}^{1} f(x)dx ≈ Σ wᵢ f(xᵢ)
///
/// Exact for polynomials up to degree 2n-1 = 9.
/// For [a,b]: transform via x = (b-a)t/2 + (a+b)/2.
pub fn gauss_legendre_5(f: impl Fn(f64) -> f64, a: f64, b: f64) -> f64 {
    // 5-point Gauss-Legendre nodes and weights on [-1, 1]
    const NODES: [f64; 5] = [
        -0.9061798459386640,
        -0.5384693101056831,
        0.0,
        0.5384693101056831,
        0.9061798459386640,
    ];
    const WEIGHTS: [f64; 5] = [
        0.2369268850561891,
        0.4786286704993665,
        0.5688888888888889,
        0.4786286704993665,
        0.2369268850561891,
    ];

    let half_len = (b - a) / 2.0;
    let mid = (a + b) / 2.0;

    let mut sum = 0.0;
    for i in 0..5 {
        let x = half_len * NODES[i] + mid;
        sum += WEIGHTS[i] * f(x);
    }
    sum * half_len
}

/// Adaptive Simpson quadrature.
///
/// Recursively bisects intervals where the error estimate exceeds tolerance.
/// Error estimate: |S(a,b) - S(a,m) - S(m,b)| / 15 (Richardson extrapolation).
pub fn adaptive_simpson(f: impl Fn(f64) -> f64, a: f64, b: f64, tol: f64, max_depth: usize) -> f64 {
    let m = (a + b) / 2.0;
    let fa = f(a);
    let fm = f(m);
    let fb = f(b);
    let whole = (b - a) / 6.0 * (fa + 4.0 * fm + fb);

    adaptive_simpson_rec(&f, a, b, fa, fm, fb, whole, tol, max_depth)
}

fn adaptive_simpson_rec(
    f: &impl Fn(f64) -> f64,
    a: f64, b: f64,
    fa: f64, fm: f64, fb: f64,
    whole: f64,
    tol: f64,
    depth: usize,
) -> f64 {
    let m = (a + b) / 2.0;
    let lm = (a + m) / 2.0;
    let rm = (m + b) / 2.0;
    let flm = f(lm);
    let frm = f(rm);

    let left = (m - a) / 6.0 * (fa + 4.0 * flm + fm);
    let right = (b - m) / 6.0 * (fm + 4.0 * frm + fb);
    let combined = left + right;
    let error = (combined - whole) / 15.0;

    if depth == 0 || error.abs() < tol {
        combined + error // Richardson correction
    } else {
        adaptive_simpson_rec(f, a, m, fa, flm, fm, left, tol / 2.0, depth - 1)
        + adaptive_simpson_rec(f, m, b, fm, frm, fb, right, tol / 2.0, depth - 1)
    }
}

/// Trapezoidal rule (composite): ∫_a^b f(x)dx ≈ h(f(a)/2 + f(x₁) + ... + f(b)/2).
///
/// Error: O(h²). Simple but effective for periodic functions (exponential convergence).
pub fn trapezoid(f: impl Fn(f64) -> f64, a: f64, b: f64, n: usize) -> f64 {
    if n == 0 { return 0.0; }
    let h = (b - a) / n as f64;
    let mut sum = (f(a) + f(b)) / 2.0;
    for i in 1..n {
        sum += f(a + i as f64 * h);
    }
    sum * h
}

// ═══════════════════════════════════════════════════════════════════════════
// ODE solvers
// ═══════════════════════════════════════════════════════════════════════════

/// ODE solution: sequence of (t, y) pairs.
#[derive(Debug, Clone)]
pub struct OdeSolution {
    pub t: Vec<f64>,
    pub y: Vec<f64>,
}

/// Forward Euler method: y_{n+1} = y_n + h × f(t_n, y_n).
///
/// First-order. Simple but requires small step size for stability.
pub fn euler(
    f: impl Fn(f64, f64) -> f64,
    y0: f64,
    t_start: f64,
    t_end: f64,
    n_steps: usize,
) -> OdeSolution {
    let h = (t_end - t_start) / n_steps as f64;
    let mut t_vec = Vec::with_capacity(n_steps + 1);
    let mut y_vec = Vec::with_capacity(n_steps + 1);

    let mut t = t_start;
    let mut y = y0;
    t_vec.push(t);
    y_vec.push(y);

    for _ in 0..n_steps {
        y += h * f(t, y);
        t += h;
        t_vec.push(t);
        y_vec.push(y);
    }

    OdeSolution { t: t_vec, y: y_vec }
}

/// Classic Runge-Kutta 4th order (RK4).
///
/// y_{n+1} = y_n + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)
///
/// Fourth-order accuracy. The workhorse of ODE solving.
pub fn rk4(
    f: impl Fn(f64, f64) -> f64,
    y0: f64,
    t_start: f64,
    t_end: f64,
    n_steps: usize,
) -> OdeSolution {
    let h = (t_end - t_start) / n_steps as f64;
    let mut t_vec = Vec::with_capacity(n_steps + 1);
    let mut y_vec = Vec::with_capacity(n_steps + 1);

    let mut t = t_start;
    let mut y = y0;
    t_vec.push(t);
    y_vec.push(y);

    for _ in 0..n_steps {
        let k1 = f(t, y);
        let k2 = f(t + h / 2.0, y + h / 2.0 * k1);
        let k3 = f(t + h / 2.0, y + h / 2.0 * k2);
        let k4 = f(t + h, y + h * k3);
        y += h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        t += h;
        t_vec.push(t);
        y_vec.push(y);
    }

    OdeSolution { t: t_vec, y: y_vec }
}

/// Runge-Kutta-Fehlberg (RK45) adaptive step size.
///
/// Embedded 4th/5th order pair. Estimates error and adjusts step size.
/// The standard "ODE45" from MATLAB/scipy.
pub fn rk45(
    f: impl Fn(f64, f64) -> f64,
    y0: f64,
    t_start: f64,
    t_end: f64,
    tol: f64,
    h_init: f64,
) -> OdeSolution {
    // Dormand-Prince coefficients
    const A: [f64; 6] = [0.0, 1.0/4.0, 3.0/8.0, 12.0/13.0, 1.0, 1.0/2.0];
    const B: [[f64; 5]; 6] = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0/4.0, 0.0, 0.0, 0.0, 0.0],
        [3.0/32.0, 9.0/32.0, 0.0, 0.0, 0.0],
        [1932.0/2197.0, -7200.0/2197.0, 7296.0/2197.0, 0.0, 0.0],
        [439.0/216.0, -8.0, 3680.0/513.0, -845.0/4104.0, 0.0],
        [-8.0/27.0, 2.0, -3544.0/2565.0, 1859.0/4104.0, -11.0/40.0],
    ];
    // 4th order weights
    const C4: [f64; 6] = [25.0/216.0, 0.0, 1408.0/2565.0, 2197.0/4104.0, -1.0/5.0, 0.0];
    // 5th order weights
    const C5: [f64; 6] = [16.0/135.0, 0.0, 6656.0/12825.0, 28561.0/56430.0, -9.0/50.0, 2.0/55.0];

    let mut t_vec = vec![t_start];
    let mut y_vec = vec![y0];
    let mut t = t_start;
    let mut y = y0;
    let mut h = h_init.min(t_end - t_start);

    let max_steps = 100_000;
    for _ in 0..max_steps {
        if t >= t_end - 1e-15 { break; }
        h = h.min(t_end - t);

        // Compute k values
        let k1 = h * f(t, y);
        let k2 = h * f(t + A[1]*h, y + B[1][0]*k1);
        let k3 = h * f(t + A[2]*h, y + B[2][0]*k1 + B[2][1]*k2);
        let k4 = h * f(t + A[3]*h, y + B[3][0]*k1 + B[3][1]*k2 + B[3][2]*k3);
        let k5 = h * f(t + A[4]*h, y + B[4][0]*k1 + B[4][1]*k2 + B[4][2]*k3 + B[4][3]*k4);
        let k6 = h * f(t + A[5]*h, y + B[5][0]*k1 + B[5][1]*k2 + B[5][2]*k3 + B[5][3]*k4 + B[5][4]*k5);

        let ks = [k1, k2, k3, k4, k5, k6];

        // 4th and 5th order estimates
        let y4: f64 = y + C4.iter().zip(ks.iter()).map(|(&c, &k)| c * k).sum::<f64>();
        let y5: f64 = y + C5.iter().zip(ks.iter()).map(|(&c, &k)| c * k).sum::<f64>();

        // Error estimate
        let error = (y5 - y4).abs();
        let scale = tol.max(tol * y.abs());

        if error <= scale || h <= 1e-15 {
            // Accept step
            t += h;
            y = y5; // Use 5th order solution (local extrapolation)
            t_vec.push(t);
            y_vec.push(y);

            // Grow step size
            if error > 1e-30 {
                h *= 0.9 * (scale / error).powf(0.2);
            } else {
                h *= 2.0;
            }
        } else {
            // Reject step, shrink h
            h *= 0.9 * (scale / error).powf(0.25);
        }
    }

    OdeSolution { t: t_vec, y: y_vec }
}

/// Vector ODE solver: RK4 for systems dy/dt = f(t, y) where y ∈ ℝⁿ.
pub fn rk4_system(
    f: impl Fn(f64, &[f64]) -> Vec<f64>,
    y0: &[f64],
    t_start: f64,
    t_end: f64,
    n_steps: usize,
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let dim = y0.len();
    let h = (t_end - t_start) / n_steps as f64;
    let mut t_vec = Vec::with_capacity(n_steps + 1);
    let mut y_vec: Vec<Vec<f64>> = Vec::with_capacity(n_steps + 1);

    let mut t = t_start;
    let mut y = y0.to_vec();
    t_vec.push(t);
    y_vec.push(y.clone());

    for _ in 0..n_steps {
        let k1 = f(t, &y);
        let y2: Vec<f64> = (0..dim).map(|i| y[i] + h / 2.0 * k1[i]).collect();
        let k2 = f(t + h / 2.0, &y2);
        let y3: Vec<f64> = (0..dim).map(|i| y[i] + h / 2.0 * k2[i]).collect();
        let k3 = f(t + h / 2.0, &y3);
        let y4: Vec<f64> = (0..dim).map(|i| y[i] + h * k3[i]).collect();
        let k4 = f(t + h, &y4);

        for i in 0..dim {
            y[i] += h / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        t += h;
        t_vec.push(t);
        y_vec.push(y.clone());
    }

    (t_vec, y_vec)
}

// ═══════════════════════════════════════════════════════════════════════════
// Fixed-point iteration
// ═══════════════════════════════════════════════════════════════════════════

/// Fixed-point iteration: find x such that g(x) = x.
///
/// x_{n+1} = g(x_n). Converges when |g'(x*)| < 1 near the fixed point.
pub fn fixed_point(g: impl Fn(f64) -> f64, x0: f64, tol: f64, max_iter: usize) -> RootResult {
    let mut x = x0;
    for iter in 0..max_iter {
        let xn = g(x);
        if (xn - x).abs() < tol {
            return RootResult { root: xn, iterations: iter + 1, converged: true, function_value: xn - x };
        }
        x = xn;
    }
    RootResult { root: x, iterations: max_iter, converged: false, function_value: g(x) - x }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1e-6;

    fn approx(a: f64, b: f64, tol: f64) -> bool {
        if a.is_nan() && b.is_nan() { return true; }
        (a - b).abs() < tol
    }

    // ── Root finding ─────────────────────────────────────────────────────

    #[test]
    fn bisection_sqrt2() {
        // Find √2: root of x² - 2
        let r = bisection(|x| x * x - 2.0, 1.0, 2.0, 1e-12, 100);
        assert!(r.converged);
        assert!(approx(r.root, std::f64::consts::SQRT_2, 1e-10));
    }

    #[test]
    fn newton_sqrt2() {
        let r = newton(|x| x * x - 2.0, |x| 2.0 * x, 1.5, 1e-12, 50);
        assert!(r.converged);
        assert!(approx(r.root, std::f64::consts::SQRT_2, 1e-10));
        assert!(r.iterations < 10, "Newton should converge fast, took {}", r.iterations);
    }

    #[test]
    fn secant_sqrt2() {
        let r = secant(|x| x * x - 2.0, 1.0, 2.0, 1e-12, 50);
        assert!(r.converged);
        assert!(approx(r.root, std::f64::consts::SQRT_2, 1e-10));
    }

    #[test]
    fn brent_sqrt2() {
        let r = brent(|x| x * x - 2.0, 1.0, 2.0, 1e-12, 100);
        assert!(r.converged);
        assert!(approx(r.root, std::f64::consts::SQRT_2, 1e-10));
    }

    #[test]
    fn brent_cubic() {
        // x³ - x - 2 = 0 has root at x ≈ 1.5214
        let r = brent(|x| x * x * x - x - 2.0, 1.0, 2.0, 1e-10, 100);
        assert!(r.converged);
        assert!(r.function_value.abs() < 1e-9);
    }

    #[test]
    fn newton_fewer_iterations_than_bisection() {
        let nr = newton(|x| x * x - 2.0, |x| 2.0 * x, 1.5, 1e-12, 100);
        let br = bisection(|x| x * x - 2.0, 1.0, 2.0, 1e-12, 100);
        assert!(nr.iterations < br.iterations,
            "Newton {} should be fewer than bisection {}", nr.iterations, br.iterations);
    }

    // ── Differentiation ──────────────────────────────────────────────────

    #[test]
    fn derivative_sin() {
        // d/dx sin(x) = cos(x) at x = 1
        let d = derivative_central(|x| x.sin(), 1.0, Some(1e-5));
        assert!(approx(d, 1.0_f64.cos(), 1e-8), "d={}", d);
    }

    #[test]
    fn derivative2_sin() {
        // d²/dx² sin(x) = -sin(x) at x = 1
        let d2 = derivative2_central(|x| x.sin(), 1.0, Some(1e-4));
        assert!(approx(d2, -1.0_f64.sin(), 1e-4), "d2={}", d2);
    }

    #[test]
    fn richardson_polynomial() {
        // d/dx (x³) = 3x² at x = 2 → should be 12
        let d = derivative_richardson(|x| x.powi(3), 2.0, 0.1, 5);
        assert!(approx(d, 12.0, 1e-8), "d={}", d);
    }

    // ── Quadrature ───────────────────────────────────────────────────────

    #[test]
    fn simpson_x_squared() {
        // ∫₀¹ x² dx = 1/3
        let result = simpson(|x| x * x, 0.0, 1.0, 100);
        assert!(approx(result, 1.0 / 3.0, 1e-8), "result={}", result);
    }

    #[test]
    fn gauss_legendre_polynomial() {
        // ∫₀¹ x⁴ dx = 1/5 — GL5 is exact for degree ≤ 9
        let result = gauss_legendre_5(|x| x.powi(4), 0.0, 1.0);
        assert!(approx(result, 0.2, 1e-12), "result={}", result);
    }

    #[test]
    fn adaptive_simpson_sin() {
        // ∫₀^π sin(x) dx = 2
        let result = adaptive_simpson(|x| x.sin(), 0.0, std::f64::consts::PI, 1e-10, 50);
        assert!(approx(result, 2.0, 1e-8), "result={}", result);
    }

    #[test]
    fn trapezoid_linear() {
        // ∫₀¹ x dx = 0.5 (exact for linear functions)
        let result = trapezoid(|x| x, 0.0, 1.0, 1);
        assert!(approx(result, 0.5, 1e-12));
    }

    #[test]
    fn trapezoid_exp() {
        // ∫₀¹ eˣ dx = e - 1 ≈ 1.71828
        let result = trapezoid(|x| x.exp(), 0.0, 1.0, 1000);
        assert!(approx(result, std::f64::consts::E - 1.0, 1e-5), "result={}", result);
    }

    // ── ODE solvers ──────────────────────────────────────────────────────

    #[test]
    fn euler_exponential() {
        // dy/dt = y, y(0) = 1 → y(t) = eᵗ
        let sol = euler(|_t, y| y, 1.0, 0.0, 1.0, 10000);
        let y_final = *sol.y.last().unwrap();
        assert!(approx(y_final, std::f64::consts::E, 1e-3), "y(1)={}", y_final);
    }

    #[test]
    fn rk4_exponential() {
        // dy/dt = y, y(0) = 1 → y(t) = eᵗ
        let sol = rk4(|_t, y| y, 1.0, 0.0, 1.0, 100);
        let y_final = *sol.y.last().unwrap();
        assert!(approx(y_final, std::f64::consts::E, 1e-8), "y(1)={}", y_final);
    }

    #[test]
    fn rk4_better_than_euler() {
        // RK4 with 100 steps should be more accurate than Euler with 100 steps
        let sol_e = euler(|_t, y| y, 1.0, 0.0, 1.0, 100);
        let sol_r = rk4(|_t, y| y, 1.0, 0.0, 1.0, 100);
        let err_e = (sol_e.y.last().unwrap() - std::f64::consts::E).abs();
        let err_r = (sol_r.y.last().unwrap() - std::f64::consts::E).abs();
        assert!(err_r < err_e, "RK4 error {} should be < Euler error {}", err_r, err_e);
    }

    #[test]
    fn rk45_exponential() {
        let sol = rk45(|_t, y| y, 1.0, 0.0, 1.0, 1e-8, 0.1);
        let y_final = *sol.y.last().unwrap();
        assert!(approx(y_final, std::f64::consts::E, 1e-6), "y(1)={}", y_final);
    }

    #[test]
    fn rk45_sine() {
        // dy/dt = cos(t), y(0) = 0 → y(t) = sin(t)
        let sol = rk45(|t, _y| t.cos(), 0.0, 0.0, std::f64::consts::PI, 1e-8, 0.1);
        let y_final = *sol.y.last().unwrap();
        // y(π) = sin(π) = 0
        assert!(approx(y_final, 0.0, 1e-5), "y(π)={}", y_final);
    }

    #[test]
    fn rk4_system_harmonic_oscillator() {
        // x'' + x = 0 → system: y₁' = y₂, y₂' = -y₁
        // y₁(0) = 1, y₂(0) = 0 → y₁(t) = cos(t), y₂(t) = -sin(t)
        let (ts, ys) = rk4_system(
            |_t, y| vec![y[1], -y[0]],
            &[1.0, 0.0],
            0.0, std::f64::consts::PI, 1000,
        );
        let y_final = &ys[ys.len() - 1];
        // At t = π: cos(π) = -1, -sin(π) = 0
        assert!(approx(y_final[0], -1.0, 1e-5), "y1(π)={}", y_final[0]);
        assert!(approx(y_final[1], 0.0, 1e-4), "y2(π)={}", y_final[1]);
        let _ = ts; // suppress warning
    }

    // ── Fixed point ──────────────────────────────────────────────────────

    #[test]
    fn fixed_point_sqrt2() {
        // g(x) = (x + 2/x) / 2 has fixed point at √2 (Babylonian method)
        let r = fixed_point(|x| (x + 2.0 / x) / 2.0, 1.0, 1e-12, 50);
        assert!(r.converged);
        assert!(approx(r.root, std::f64::consts::SQRT_2, 1e-10));
    }
}
