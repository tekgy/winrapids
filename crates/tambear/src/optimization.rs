//! # Family 05 — Optimization
//!
//! All optimizers, from first principles.
//!
//! ## What lives here
//!
//! **Gradient-based**: gradient descent, Adam, AdaGrad, RMSProp, L-BFGS
//! **Derivative-free**: Nelder-Mead (simplex), golden section search, coordinate descent
//! **Constrained**: projected gradient, penalty method
//! **Line search**: backtracking (Armijo), Wolfe conditions
//!
//! ## Architecture
//!
//! The `ObjectiveFn` trait provides `value(&[f64]) -> f64` and optionally
//! `gradient(&[f64]) -> Vec<f64>`. Optimizers consume this trait.
//!
//! Every MLE (GARCH, GMM, logistic regression) reduces to: define an
//! ObjectiveFn, call an optimizer. The optimizer doesn't know statistics.
//!
//! ## MSR insight
//!
//! Adam's state (m, v per parameter) is the MSR of the gradient history.
//! L-BFGS stores the last k (s, y) pairs — the MSR of the Hessian approximation.
//! Both are accumulate patterns: running averages with decay.

/// Optimization result.
#[derive(Debug, Clone)]
pub struct OptResult {
    /// Optimal parameters found.
    pub x: Vec<f64>,
    /// Objective function value at optimum.
    pub f_val: f64,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the optimizer converged.
    pub converged: bool,
}

// ─── Line search ────────────────────────────────────────────────────

/// Backtracking line search with Armijo condition.
///
/// Find α such that f(x + α·d) ≤ f(x) + c₁·α·∇f(x)·d.
pub fn backtracking_line_search<F: Fn(&[f64]) -> f64>(
    f: &F,
    x: &[f64],
    grad: &[f64],
    direction: &[f64],
    alpha_init: f64,
    c1: f64,
    rho: f64,
) -> f64 {
    let n = x.len();
    let f0 = f(x);
    let slope = grad.iter().zip(direction.iter()).map(|(g, d)| g * d).sum::<f64>();
    let mut alpha = alpha_init;
    let mut x_new = vec![0.0; n];

    for _ in 0..50 {
        for i in 0..n { x_new[i] = x[i] + alpha * direction[i]; }
        if f(&x_new) <= f0 + c1 * alpha * slope {
            return alpha;
        }
        alpha *= rho;
    }
    alpha
}

/// Golden section search for 1D minimum on [a, b].
///
/// Finds x* ∈ [a, b] minimizing f(x). Derivative-free.
pub fn golden_section<F: Fn(f64) -> f64>(f: &F, mut a: f64, mut b: f64, tol: f64) -> OptResult {
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0; // ≈ 0.618
    let mut x1 = b - phi * (b - a);
    let mut x2 = a + phi * (b - a);
    let mut f1 = f(x1);
    let mut f2 = f(x2);
    let mut iter = 0;

    while (b - a).abs() > tol && iter < 1000 {
        if f1 < f2 {
            b = x2;
            x2 = x1;
            f2 = f1;
            x1 = b - phi * (b - a);
            f1 = f(x1);
        } else {
            a = x1;
            x1 = x2;
            f1 = f2;
            x2 = a + phi * (b - a);
            f2 = f(x2);
        }
        iter += 1;
    }
    let x_opt = (a + b) / 2.0;
    OptResult { x: vec![x_opt], f_val: f(x_opt), iterations: iter, converged: (b - a).abs() <= tol }
}

// ─── Gradient Descent ───────────────────────────────────────────────

/// Gradient descent with optional momentum.
///
/// Parameters:
/// - `f`: objective function
/// - `grad`: gradient function
/// - `x0`: initial guess
/// - `lr`: learning rate
/// - `momentum`: momentum coefficient (0 for vanilla GD)
/// - `max_iter`: maximum iterations
/// - `tol`: convergence tolerance on gradient norm
pub fn gradient_descent<F, G>(
    f: &F, grad: &G, x0: &[f64],
    lr: f64, momentum: f64, max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut velocity = vec![0.0; n];

    for iter in 0..max_iter {
        let g = grad(&x);
        let gnorm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < tol {
            return OptResult { f_val: f(&x), x, iterations: iter, converged: true };
        }
        for i in 0..n {
            velocity[i] = momentum * velocity[i] - lr * g[i];
            x[i] += velocity[i];
        }
    }
    OptResult { f_val: f(&x), x, iterations: max_iter, converged: false }
}

// ─── Adam ───────────────────────────────────────────────────────────

/// Adam optimizer (Kingma & Ba, 2014).
///
/// Adaptive learning rates with first and second moment estimates.
/// The default for deep learning AND many MLE problems.
pub fn adam<F, G>(
    f: &F, grad: &G, x0: &[f64],
    lr: f64, beta1: f64, beta2: f64, eps: f64,
    max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut m = vec![0.0; n]; // first moment
    let mut v = vec![0.0; n]; // second moment

    for iter in 0..max_iter {
        let g = grad(&x);
        let gnorm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < tol {
            return OptResult { f_val: f(&x), x, iterations: iter, converged: true };
        }

        let t = (iter + 1) as f64;
        for i in 0..n {
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i];
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] * g[i];
            let m_hat = m[i] / (1.0 - beta1.powf(t));
            let v_hat = v[i] / (1.0 - beta2.powf(t));
            x[i] -= lr * m_hat / (v_hat.sqrt() + eps);
        }
    }
    OptResult { f_val: f(&x), x, iterations: max_iter, converged: false }
}

// ─── AdaGrad ────────────────────────────────────────────────────────

/// AdaGrad — per-parameter adaptive learning rate.
///
/// Good for sparse gradients. Learning rate decays as 1/√t.
pub fn adagrad<F, G>(
    f: &F, grad: &G, x0: &[f64],
    lr: f64, eps: f64, max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g_sum = vec![0.0; n];

    for iter in 0..max_iter {
        let g = grad(&x);
        let gnorm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < tol {
            return OptResult { f_val: f(&x), x, iterations: iter, converged: true };
        }
        for i in 0..n {
            g_sum[i] += g[i] * g[i];
            x[i] -= lr * g[i] / (g_sum[i].sqrt() + eps);
        }
    }
    OptResult { f_val: f(&x), x, iterations: max_iter, converged: false }
}

// ─── RMSProp ────────────────────────────────────────────────────────

/// RMSProp — exponentially weighted AdaGrad.
///
/// Fixes AdaGrad's aggressive learning rate decay.
pub fn rmsprop<F, G>(
    f: &F, grad: &G, x0: &[f64],
    lr: f64, decay: f64, eps: f64, max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut v = vec![0.0; n];

    for iter in 0..max_iter {
        let g = grad(&x);
        let gnorm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < tol {
            return OptResult { f_val: f(&x), x, iterations: iter, converged: true };
        }
        for i in 0..n {
            v[i] = decay * v[i] + (1.0 - decay) * g[i] * g[i];
            x[i] -= lr * g[i] / (v[i].sqrt() + eps);
        }
    }
    OptResult { f_val: f(&x), x, iterations: max_iter, converged: false }
}

// ─── L-BFGS ─────────────────────────────────────────────────────────

/// L-BFGS (Limited-memory BFGS).
///
/// Approximates the inverse Hessian using the last `m` gradient differences.
/// The gold standard for smooth unconstrained optimization.
/// This is what GARCH MLE, GMM, and most statistical models need.
pub fn lbfgs<F, G>(
    f: &F, grad: &G, x0: &[f64],
    m: usize, max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    let mut g = grad(&x);

    // Storage for (s, y) pairs
    let mut s_history: Vec<Vec<f64>> = Vec::new();
    let mut y_history: Vec<Vec<f64>> = Vec::new();
    let mut rho_history: Vec<f64> = Vec::new();

    for iter in 0..max_iter {
        let gnorm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < tol {
            return OptResult { f_val: f(&x), x, iterations: iter, converged: true };
        }

        // Compute search direction via L-BFGS two-loop recursion
        let mut q = g.clone();
        let k = s_history.len();
        let mut alpha_vec = vec![0.0; k];

        // First loop (backward)
        for i in (0..k).rev() {
            alpha_vec[i] = rho_history[i] * dot_vec(&s_history[i], &q);
            for j in 0..n { q[j] -= alpha_vec[i] * y_history[i][j]; }
        }

        // Scale initial Hessian approximation
        let gamma = if k > 0 {
            let sy = dot_vec(&s_history[k - 1], &y_history[k - 1]);
            let yy = dot_vec(&y_history[k - 1], &y_history[k - 1]);
            if yy > 1e-300 { sy / yy } else { 1.0 }
        } else {
            1.0
        };
        for j in 0..n { q[j] *= gamma; }

        // Second loop (forward)
        for i in 0..k {
            let beta = rho_history[i] * dot_vec(&y_history[i], &q);
            for j in 0..n { q[j] += (alpha_vec[i] - beta) * s_history[i][j]; }
        }

        // Direction = -H·g
        let mut direction = q;
        for d in direction.iter_mut() { *d = -*d; }

        // Line search
        let alpha = backtracking_line_search(f, &x, &g, &direction, 1.0, 1e-4, 0.5);

        // Update x
        let mut s = vec![0.0; n];
        for i in 0..n {
            s[i] = alpha * direction[i];
            x[i] += s[i];
        }

        // Update gradient
        let g_new = grad(&x);
        let mut y = vec![0.0; n];
        for i in 0..n { y[i] = g_new[i] - g[i]; }

        let sy = dot_vec(&s, &y);
        if sy > 1e-14 {
            if s_history.len() >= m {
                s_history.remove(0);
                y_history.remove(0);
                rho_history.remove(0);
            }
            rho_history.push(1.0 / sy);
            s_history.push(s);
            y_history.push(y);
        }

        g = g_new;
    }
    OptResult { f_val: f(&x), x, iterations: max_iter, converged: false }
}

fn dot_vec(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─── Nelder-Mead ────────────────────────────────────────────────────

/// Nelder-Mead simplex method (derivative-free).
///
/// Works for any continuous function, even non-differentiable.
/// Uses reflection, expansion, contraction, and shrinkage.
pub fn nelder_mead<F: Fn(&[f64]) -> f64>(
    f: &F, x0: &[f64], step: f64, max_iter: usize, tol: f64,
) -> OptResult {
    let n = x0.len();
    let np1 = n + 1;

    // Initialize simplex
    let mut simplex: Vec<Vec<f64>> = Vec::with_capacity(np1);
    simplex.push(x0.to_vec());
    for i in 0..n {
        let mut vertex = x0.to_vec();
        vertex[i] += step;
        simplex.push(vertex);
    }
    let mut f_vals: Vec<f64> = simplex.iter().map(|v| f(v)).collect();

    let alpha = 1.0; // reflection
    let gamma = 2.0; // expansion
    let rho = 0.5;   // contraction
    let sigma = 0.5;  // shrinkage

    for iter in 0..max_iter {
        // Sort by function value
        let mut order: Vec<usize> = (0..np1).collect();
        order.sort_by(|&a, &b| f_vals[a].total_cmp(&f_vals[b]));
        let sorted_simplex: Vec<Vec<f64>> = order.iter().map(|&i| simplex[i].clone()).collect();
        let sorted_fvals: Vec<f64> = order.iter().map(|&i| f_vals[i]).collect();
        simplex = sorted_simplex;
        f_vals = sorted_fvals;

        // Check convergence (spread of function values)
        let f_range = f_vals[np1 - 1] - f_vals[0];
        if f_range < tol {
            return OptResult {
                x: simplex[0].clone(), f_val: f_vals[0],
                iterations: iter, converged: true,
            };
        }

        // Centroid of all vertices except worst
        let mut centroid = vec![0.0; n];
        for i in 0..n { // n = np1 - 1 best vertices
            for j in 0..n { centroid[j] += simplex[i][j]; }
        }
        for j in 0..n { centroid[j] /= n as f64; }

        // Reflection
        let mut xr = vec![0.0; n];
        for j in 0..n { xr[j] = centroid[j] + alpha * (centroid[j] - simplex[np1 - 1][j]); }
        let fr = f(&xr);

        if fr < f_vals[0] {
            // Try expansion
            let mut xe = vec![0.0; n];
            for j in 0..n { xe[j] = centroid[j] + gamma * (xr[j] - centroid[j]); }
            let fe = f(&xe);
            if fe < fr {
                simplex[np1 - 1] = xe;
                f_vals[np1 - 1] = fe;
            } else {
                simplex[np1 - 1] = xr;
                f_vals[np1 - 1] = fr;
            }
        } else if fr < f_vals[np1 - 2] {
            simplex[np1 - 1] = xr;
            f_vals[np1 - 1] = fr;
        } else {
            // Contraction
            let mut xc = vec![0.0; n];
            if fr < f_vals[np1 - 1] {
                // Outside contraction
                for j in 0..n { xc[j] = centroid[j] + rho * (xr[j] - centroid[j]); }
            } else {
                // Inside contraction
                for j in 0..n { xc[j] = centroid[j] + rho * (simplex[np1 - 1][j] - centroid[j]); }
            }
            let fc = f(&xc);
            if fc < f_vals[np1 - 1].min(fr) {
                simplex[np1 - 1] = xc;
                f_vals[np1 - 1] = fc;
            } else {
                // Shrink
                for i in 1..np1 {
                    for j in 0..n {
                        simplex[i][j] = simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]);
                    }
                    f_vals[i] = f(&simplex[i]);
                }
            }
        }
    }

    // Find best
    let best = f_vals.iter().enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .unwrap().0;
    OptResult {
        x: simplex[best].clone(), f_val: f_vals[best],
        iterations: max_iter, converged: false,
    }
}

// ─── Coordinate Descent ─────────────────────────────────────────────

/// Coordinate descent — optimize one variable at a time.
///
/// Each iteration minimizes f along each coordinate axis using golden section.
/// No derivatives needed. Good for separable or nearly-separable problems.
pub fn coordinate_descent<F: Fn(&[f64]) -> f64>(
    f: &F, x0: &[f64], step: f64, max_iter: usize, tol: f64,
) -> OptResult {
    let n = x0.len();
    let mut x = x0.to_vec();

    for iter in 0..max_iter {
        let f_old = f(&x);
        for dim in 0..n {
            let xd = x[dim];
            let f_1d = |t: f64| -> f64 {
                let mut xc = x.clone();
                xc[dim] = t;
                f(&xc)
            };
            let result = golden_section(&f_1d, xd - step, xd + step, tol * 0.1);
            x[dim] = result.x[0];
        }
        let f_new = f(&x);
        if (f_old - f_new).abs() < tol {
            return OptResult { x, f_val: f_new, iterations: iter, converged: true };
        }
    }
    OptResult { f_val: f(&x), x, iterations: max_iter, converged: false }
}

// ─── Projected Gradient ─────────────────────────────────────────────

/// Simple box-constrained optimization via projected gradient descent.
///
/// Enforces lower[i] ≤ x[i] ≤ upper[i] by projection after each step.
pub fn projected_gradient<F, G>(
    f: &F, grad: &G, x0: &[f64],
    lower: &[f64], upper: &[f64],
    lr: f64, max_iter: usize, tol: f64,
) -> OptResult
where
    F: Fn(&[f64]) -> f64,
    G: Fn(&[f64]) -> Vec<f64>,
{
    let n = x0.len();
    let mut x = x0.to_vec();
    // Project initial point
    for i in 0..n { x[i] = x[i].clamp(lower[i], upper[i]); }

    for iter in 0..max_iter {
        let g = grad(&x);
        let gnorm: f64 = g.iter().map(|v| v * v).sum::<f64>().sqrt();
        if gnorm < tol {
            return OptResult { f_val: f(&x), x, iterations: iter, converged: true };
        }
        for i in 0..n {
            x[i] = (x[i] - lr * g[i]).clamp(lower[i], upper[i]);
        }
    }
    OptResult { f_val: f(&x), x, iterations: max_iter, converged: false }
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Rosenbrock function: f(x,y) = (1-x)² + 100(y-x²)²
    // Minimum at (1, 1) with f = 0
    fn rosenbrock(x: &[f64]) -> f64 {
        (1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)
    }

    fn rosenbrock_grad(x: &[f64]) -> Vec<f64> {
        vec![
            -2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]),
            200.0 * (x[1] - x[0] * x[0]),
        ]
    }

    // Simple quadratic: f(x) = x[0]² + x[1]²
    fn quadratic(x: &[f64]) -> f64 {
        x[0] * x[0] + x[1] * x[1]
    }

    fn quadratic_grad(x: &[f64]) -> Vec<f64> {
        vec![2.0 * x[0], 2.0 * x[1]]
    }

    // ── Golden section ──

    #[test]
    fn golden_section_parabola() {
        let result = golden_section(&|x: f64| (x - 3.0).powi(2), 0.0, 10.0, 1e-8);
        assert!((result.x[0] - 3.0).abs() < 1e-6, "x = {}", result.x[0]);
        assert!(result.converged);
    }

    // ── Gradient Descent ──

    #[test]
    fn gd_quadratic() {
        let result = gradient_descent(
            &quadratic, &quadratic_grad, &[5.0, 3.0],
            0.1, 0.0, 1000, 1e-8,
        );
        assert!((result.x[0]).abs() < 1e-4, "x[0] = {}", result.x[0]);
        assert!((result.x[1]).abs() < 1e-4, "x[1] = {}", result.x[1]);
        assert!(result.converged);
    }

    #[test]
    fn gd_with_momentum() {
        let result = gradient_descent(
            &quadratic, &quadratic_grad, &[5.0, 3.0],
            0.1, 0.9, 1000, 1e-8,
        );
        assert!((result.x[0]).abs() < 1e-3, "x[0] = {}", result.x[0]);
    }

    // ── Adam ──

    #[test]
    fn adam_quadratic() {
        let result = adam(
            &quadratic, &quadratic_grad, &[5.0, 3.0],
            0.1, 0.9, 0.999, 1e-8, 5000, 1e-8,
        );
        assert!((result.x[0]).abs() < 0.01, "x[0] = {}", result.x[0]);
        assert!((result.x[1]).abs() < 0.01, "x[1] = {}", result.x[1]);
    }

    #[test]
    fn adam_rosenbrock() {
        let result = adam(
            &rosenbrock, &rosenbrock_grad, &[0.0, 0.0],
            0.01, 0.9, 0.999, 1e-8, 50000, 1e-10,
        );
        assert!((result.x[0] - 1.0).abs() < 0.1, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 0.1, "x[1] = {}", result.x[1]);
    }

    // ── AdaGrad ──

    #[test]
    fn adagrad_quadratic() {
        let result = adagrad(
            &quadratic, &quadratic_grad, &[5.0, 3.0],
            1.0, 1e-8, 5000, 1e-6,
        );
        assert!((result.x[0]).abs() < 0.1, "x[0] = {}", result.x[0]);
    }

    // ── RMSProp ──

    #[test]
    fn rmsprop_quadratic() {
        let result = rmsprop(
            &quadratic, &quadratic_grad, &[5.0, 3.0],
            0.01, 0.99, 1e-8, 5000, 1e-8,
        );
        assert!((result.x[0]).abs() < 0.01, "x[0] = {}", result.x[0]);
    }

    // ── L-BFGS ──

    #[test]
    fn lbfgs_quadratic() {
        let result = lbfgs(
            &quadratic, &quadratic_grad, &[5.0, 3.0],
            10, 100, 1e-10,
        );
        assert!((result.x[0]).abs() < 1e-6, "x[0] = {}", result.x[0]);
        assert!((result.x[1]).abs() < 1e-6, "x[1] = {}", result.x[1]);
        assert!(result.converged);
    }

    #[test]
    fn lbfgs_rosenbrock() {
        let result = lbfgs(
            &rosenbrock, &rosenbrock_grad, &[0.0, 0.0],
            10, 1000, 1e-10,
        );
        assert!((result.x[0] - 1.0).abs() < 0.01, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 0.01, "x[1] = {}", result.x[1]);
    }

    // ── Nelder-Mead ──

    #[test]
    fn nelder_mead_quadratic() {
        let result = nelder_mead(&quadratic, &[5.0, 3.0], 1.0, 1000, 1e-10);
        assert!((result.x[0]).abs() < 0.01, "x[0] = {}", result.x[0]);
        assert!((result.x[1]).abs() < 0.01, "x[1] = {}", result.x[1]);
    }

    #[test]
    fn nelder_mead_rosenbrock() {
        let result = nelder_mead(&rosenbrock, &[0.0, 0.0], 1.0, 5000, 1e-12);
        assert!((result.x[0] - 1.0).abs() < 0.05, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 1.0).abs() < 0.05, "x[1] = {}", result.x[1]);
    }

    // ── Coordinate Descent ──

    #[test]
    fn coord_descent_separable() {
        // f(x,y) = (x-1)² + (y-2)²  — perfectly separable
        let f = |x: &[f64]| (x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2);
        let result = coordinate_descent(&f, &[5.0, 5.0], 10.0, 100, 1e-8);
        assert!((result.x[0] - 1.0).abs() < 0.01, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 0.01, "x[1] = {}", result.x[1]);
    }

    // ── Projected Gradient ──

    #[test]
    fn projected_gradient_box() {
        // Minimize x² + y² subject to x ∈ [1, 5], y ∈ [2, 5]
        // Solution: (1, 2)
        let result = projected_gradient(
            &quadratic, &quadratic_grad, &[3.0, 4.0],
            &[1.0, 2.0], &[5.0, 5.0],
            0.1, 1000, 1e-8,
        );
        assert!((result.x[0] - 1.0).abs() < 0.01, "x[0] = {}", result.x[0]);
        assert!((result.x[1] - 2.0).abs() < 0.01, "x[1] = {}", result.x[1]);
    }

    // ── Line search ──

    #[test]
    fn backtracking_finds_step() {
        let alpha = backtracking_line_search(
            &quadratic, &[2.0, 2.0], &[4.0, 4.0], &[-4.0, -4.0],
            1.0, 1e-4, 0.5,
        );
        assert!(alpha > 0.0);
        assert!(alpha <= 1.0);
        // f(x + α·d) should be less than f(x)
        let x_new: Vec<f64> = vec![2.0 - alpha * 4.0, 2.0 - alpha * 4.0];
        assert!(quadratic(&x_new) < quadratic(&[2.0, 2.0]));
    }

    // ── Convergence comparison ──

    #[test]
    fn lbfgs_faster_than_gd() {
        // L-BFGS should converge in fewer iterations than GD on quadratic
        let gd = gradient_descent(
            &quadratic, &quadratic_grad, &[5.0, 3.0],
            0.1, 0.0, 1000, 1e-8,
        );
        let lb = lbfgs(
            &quadratic, &quadratic_grad, &[5.0, 3.0],
            5, 1000, 1e-8,
        );
        assert!(lb.iterations < gd.iterations,
            "L-BFGS {} iters vs GD {} iters", lb.iterations, gd.iterations);
    }
}
