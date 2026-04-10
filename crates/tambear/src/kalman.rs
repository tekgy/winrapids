//! # Kalman Filter and Smoother — Sarkka (2013) formulation
//!
//! Sequential (CPU) implementation of the scalar and matrix Kalman filter
//! and Rauch-Tung-Striebel (RTS) smoother. Verified against analytical
//! solutions for linear Gaussian state-space models.
//!
//! ## State-space model
//!
//! ```text
//! x_t = F * x_{t-1} + w_t,    w_t ~ N(0, Q)     (transition)
//! y_t = H * x_t    + v_t,    v_t ~ N(0, R)     (observation)
//! x_0 ~ N(x0, P0)
//! ```
//!
//! ## Filter recursion (Sarkka 2013, Algorithm 4.1)
//!
//! ```text
//! Predict: x_{t|t-1} = F * x_{t-1|t-1}
//!          P_{t|t-1} = F^2 * P_{t-1|t-1} + Q      (scalar)
//!
//! Update:  S_t = H^2 * P_{t|t-1} + R               (innovation variance)
//!          K_t = P_{t|t-1} * H / S_t               (Kalman gain)
//!          x_{t|t} = x_{t|t-1} + K_t * (y_t - H * x_{t|t-1})
//!          P_{t|t} = (1 - K_t * H) * P_{t|t-1}
//! ```
//!
//! ## Smoother recursion (RTS, Sarkka 2013, Algorithm 8.2)
//!
//! ```text
//! G_t = P_{t|t} * F / P_{t+1|t}
//! x_{t|T} = x_{t|t} + G_t * (x_{t+1|T} - F * x_{t|t})
//! P_{t|T} = P_{t|t} + G_t^2 * (P_{t+1|T} - P_{t+1|t})
//! ```
//!
//! ## Architecture note
//!
//! The parallel (GPU) version uses the Sarkka 5-tuple prefix scan with the
//! correction term `-J_b * b_a` in the eta combine (see garden 006). This
//! sequential version produces the SAME output and serves as the CPU backend
//! and the gold standard for testing the parallel version.

use crate::linear_algebra::{mat_mul, mat_add, mat_sub, inv, log_det, Mat, mat_vec};

// ─────────────────────────────────────────────────────────────────────────────
// Scalar Kalman filter (1D state, 1D observation)
// ─────────────────────────────────────────────────────────────────────────────

/// Output of the scalar Kalman filter.
#[derive(Debug, Clone)]
pub struct KalmanFilterResult {
    /// Filtered state estimates: x_{t|t}.
    pub states: Vec<f64>,
    /// Filtered state variances: P_{t|t}.
    pub variances: Vec<f64>,
    /// Kalman gains K_t.
    pub gains: Vec<f64>,
    /// Predicted state variances: P_{t|t-1}.
    pub predicted_variances: Vec<f64>,
    /// Innovation (residual): y_t - H * x_{t|t-1}.
    pub innovations: Vec<f64>,
    /// Log-likelihood of the observation sequence.
    pub log_likelihood: f64,
}

/// Run the scalar (1D) Kalman filter.
///
/// # Parameters
/// - `observations`: observed sequence y_1, …, y_n. NaN entries are skipped (missing data).
/// - `f`: state transition coefficient (x_{t} = f * x_{t-1} + w).
/// - `h`: observation coefficient (y_t = h * x_t + v).
/// - `q`: process noise variance.
/// - `r`: observation noise variance.
/// - `x0`: initial state mean.
/// - `p0`: initial state variance.
///
/// Returns `None` when the observation sequence is empty.
pub fn kalman_filter_scalar(
    observations: &[f64],
    f: f64,
    h: f64,
    q: f64,
    r: f64,
    x0: f64,
    p0: f64,
) -> Option<KalmanFilterResult> {
    let n = observations.len();
    if n == 0 {
        return None;
    }

    let mut states = Vec::with_capacity(n);
    let mut variances = Vec::with_capacity(n);
    let mut gains = Vec::with_capacity(n);
    let mut predicted_variances = Vec::with_capacity(n);
    let mut innovations = Vec::with_capacity(n);
    let mut log_likelihood = 0.0_f64;

    let mut x = x0;
    let mut p = p0;

    for &y in observations {
        // Predict
        let x_pred = f * x;
        let p_pred = f * f * p + q;

        if y.is_nan() {
            // Missing observation: propagate prediction, no update
            states.push(x_pred);
            variances.push(p_pred);
            gains.push(0.0);
            predicted_variances.push(p_pred);
            innovations.push(f64::NAN);
            x = x_pred;
            p = p_pred;
            continue;
        }

        // Update
        let innovation = y - h * x_pred;
        let s = h * h * p_pred + r;          // innovation variance
        let k = p_pred * h / s;              // Kalman gain
        let x_upd = x_pred + k * innovation;
        let p_upd = (1.0 - k * h) * p_pred;

        // Log-likelihood contribution: -0.5 * [ln(2π s) + innovation^2 / s]
        log_likelihood += -0.5 * ((2.0 * std::f64::consts::PI * s).ln() + innovation * innovation / s);

        states.push(x_upd);
        variances.push(p_upd);
        gains.push(k);
        predicted_variances.push(p_pred);
        innovations.push(innovation);

        x = x_upd;
        p = p_upd;
    }

    Some(KalmanFilterResult {
        states,
        variances,
        gains,
        predicted_variances,
        innovations,
        log_likelihood,
    })
}

/// Rauch-Tung-Striebel (RTS) smoother for the scalar case.
///
/// Runs the backward pass given the forward filter result. Returns the
/// smoothed state estimates and variances.
///
/// # Correctness requirement
/// `filter` must have been produced by [`kalman_filter_scalar`] with the
/// same `f` and `q` parameters.
pub fn rts_smoother_scalar(
    filter: &KalmanFilterResult,
    f: f64,
    q: f64,
) -> (Vec<f64>, Vec<f64>) {
    let n = filter.states.len();
    if n == 0 {
        return (vec![], vec![]);
    }

    let mut x_smooth = filter.states.clone();
    let mut p_smooth = filter.variances.clone();

    // Backward pass: t = n-2 down to 0
    for t in (0..n - 1).rev() {
        let p_pred_next = f * f * filter.variances[t] + q; // P_{t+1|t}
        if p_pred_next.abs() < 1e-300 {
            continue; // degenerate
        }
        let g = filter.variances[t] * f / p_pred_next;    // smoother gain G_t

        x_smooth[t] = filter.states[t] + g * (x_smooth[t + 1] - f * filter.states[t]);
        p_smooth[t] = filter.variances[t] + g * g * (p_smooth[t + 1] - p_pred_next);
    }

    (x_smooth, p_smooth)
}

// ─────────────────────────────────────────────────────────────────────────────
// Matrix Kalman filter (n-dimensional state, m-dimensional observation)
// ─────────────────────────────────────────────────────────────────────────────
//
// State: x_t ∈ ℝ^n_state
// Observation: y_t ∈ ℝ^n_obs
//
// x_t = F * x_{t-1} + w_t,   w_t ~ N(0, Q)
// y_t = H * x_t    + v_t,   v_t ~ N(0, R)

/// Output of the matrix Kalman filter.
#[derive(Debug, Clone)]
pub struct KalmanFilterMatrixResult {
    /// Filtered state means x_{t|t}: n_obs × n_state matrix (row-major, one row per timestep).
    pub states: Vec<Vec<f64>>,
    /// Filtered state covariances P_{t|t}: list of n_state × n_state matrices.
    pub covariances: Vec<Vec<f64>>,
    /// Log-likelihood of the observation sequence.
    pub log_likelihood: f64,
}

/// Run the multivariate Kalman filter.
///
/// # Parameters
/// - `observations`: T × n_obs matrix, row-major. Rows with any NaN are skipped.
/// - `f_mat`: n_state × n_state transition matrix (row-major).
/// - `h_mat`: n_obs × n_state observation matrix (row-major).
/// - `q_mat`: n_state × n_state process noise covariance (row-major).
/// - `r_mat`: n_obs × n_obs observation noise covariance (row-major).
/// - `x0`: initial state mean, length n_state.
/// - `p0_mat`: initial state covariance, n_state × n_state (row-major).
/// - `n_state`: state dimension.
/// - `n_obs`: observation dimension.
///
/// Returns `None` on empty observations or dimension mismatch.
pub fn kalman_filter_matrix(
    observations: &[f64],
    f_mat: &[f64],
    h_mat: &[f64],
    q_mat: &[f64],
    r_mat: &[f64],
    x0: &[f64],
    p0_mat: &[f64],
    n_state: usize,
    n_obs: usize,
) -> Option<KalmanFilterMatrixResult> {
    let t_steps = observations.len() / n_obs;
    if t_steps == 0 || observations.len() != t_steps * n_obs { return None; }
    if f_mat.len() != n_state * n_state { return None; }
    if h_mat.len() != n_obs * n_state { return None; }
    if q_mat.len() != n_state * n_state { return None; }
    if r_mat.len() != n_obs * n_obs { return None; }
    if x0.len() != n_state { return None; }
    if p0_mat.len() != n_state * n_state { return None; }

    let f = Mat { data: f_mat.to_vec(), rows: n_state, cols: n_state };
    let h = Mat { data: h_mat.to_vec(), rows: n_obs, cols: n_state };
    let q = Mat { data: q_mat.to_vec(), rows: n_state, cols: n_state };
    let r = Mat { data: r_mat.to_vec(), rows: n_obs, cols: n_obs };

    let mut x = x0.to_vec();
    let mut p = Mat { data: p0_mat.to_vec(), rows: n_state, cols: n_state };

    let mut states: Vec<Vec<f64>> = Vec::with_capacity(t_steps);
    let mut covariances: Vec<Vec<f64>> = Vec::with_capacity(t_steps);
    let mut log_likelihood = 0.0_f64;

    for t in 0..t_steps {
        let y_slice = &observations[t * n_obs..(t + 1) * n_obs];
        let has_obs = y_slice.iter().all(|v| v.is_finite());

        // Predict: x_pred = F * x
        let x_pred = mat_vec(&f, &x);
        // P_pred = F * P * F^T + Q
        let fp = mat_mul(&f, &p);
        let ft = f.t();
        let fp_ft = mat_mul(&fp, &ft);
        let p_pred = mat_add(&fp_ft, &q);

        if !has_obs {
            states.push(x_pred.clone());
            covariances.push(p_pred.data.clone());
            x = x_pred;
            p = p_pred;
            continue;
        }

        // Innovation: nu = y - H * x_pred
        let hx = mat_vec(&h, &x_pred);
        let nu: Vec<f64> = y_slice.iter().zip(hx.iter()).map(|(yi, hi)| yi - hi).collect();

        // S = H * P_pred * H^T + R
        let hp = mat_mul(&h, &p_pred);
        let ht = h.t();
        let hp_ht = mat_mul(&hp, &ht);
        let s = mat_add(&hp_ht, &r);

        // K = P_pred * H^T * S^{-1}
        let s_inv = match inv(&s) {
            Some(si) => si,
            None => {
                // Singular innovation matrix — degenerate, push prediction
                states.push(x_pred.clone());
                covariances.push(p_pred.data.clone());
                x = x_pred;
                p = p_pred;
                continue;
            }
        };
        // K = P_pred * H^T * S^{-1} (ht already computed above)
        let p_ht = mat_mul(&p_pred, &ht);
        let k = mat_mul(&p_ht, &s_inv);

        // x_upd = x_pred + K * nu
        let k_nu = mat_vec(&k, &nu);
        let x_upd: Vec<f64> = x_pred.iter().zip(k_nu.iter()).map(|(a, b)| a + b).collect();

        // P_upd = (I - K * H) * P_pred
        let kh = mat_mul(&k, &h);
        let i_mat = Mat::eye(n_state);
        let i_kh = mat_sub(&i_mat, &kh);
        let p_upd = mat_mul(&i_kh, &p_pred);

        // Log-likelihood: -0.5 * (n_obs * ln(2π) + ln|S| + nu^T S^{-1} nu)
        // log_det from linear_algebra is the canonical primitive — stable for small matrices
        let ln_det_s = log_det(&s);
        let nu_t = Mat { data: nu.clone(), rows: 1, cols: n_obs };
        let nu_t_sinv = mat_mul(&nu_t, &s_inv);
        let quad: f64 = nu_t_sinv.data.iter().zip(nu.iter()).map(|(a, b)| a * b).sum();
        log_likelihood += -0.5 * (n_obs as f64 * (2.0 * std::f64::consts::PI).ln() + ln_det_s + quad);

        states.push(x_upd.clone());
        covariances.push(p_upd.data.clone());
        x = x_upd;
        p = p_upd;
    }

    Some(KalmanFilterMatrixResult { states, covariances, log_likelihood })
}

// transpose_mat, identity_mat, and log_det are now the canonical public primitives
// Mat::t(), Mat::eye(n), and linear_algebra::log_det. All three private copies removed.

// ─────────────────────────────────────────────────────────────────────────────
// Hidden Markov Model (HMM)
// ─────────────────────────────────────────────────────────────────────────────
//
// Discrete-state, discrete-observation HMM (Rabiner 1989).
//
// Parameters:
//   pi[s]     = initial state probability P(q_1 = s)
//   A[s][s']  = transition probability P(q_{t+1}=s'|q_t=s), row-major
//   B[s][o]   = emission probability P(o_t=o|q_t=s), row-major
//
// n_states = number of hidden states
// n_obs    = number of distinct observation symbols

/// HMM parameter set.
#[derive(Debug, Clone)]
pub struct HmmParams {
    /// Initial state distribution, length n_states.
    pub pi: Vec<f64>,
    /// Transition matrix n_states × n_states, row-major. A[i][j] = P(j|i).
    pub transition: Vec<f64>,
    /// Emission matrix n_states × n_obs, row-major. B[i][o] = P(o|i).
    pub emission: Vec<f64>,
    /// Number of hidden states.
    pub n_states: usize,
    /// Number of distinct observation symbols.
    pub n_symbols: usize,
}

impl HmmParams {
    /// Uniform initialization.
    pub fn uniform(n_states: usize, n_symbols: usize) -> Self {
        let p_state = 1.0 / n_states as f64;
        let p_trans = 1.0 / n_states as f64;
        let p_emit  = 1.0 / n_symbols as f64;
        HmmParams {
            pi: vec![p_state; n_states],
            transition: vec![p_trans; n_states * n_states],
            emission: vec![p_emit; n_states * n_symbols],
            n_states,
            n_symbols,
        }
    }

    fn a(&self, i: usize, j: usize) -> f64 { self.transition[i * self.n_states + j] }
    fn b(&self, i: usize, o: usize) -> f64 { self.emission[i * self.n_symbols + o] }
}

/// Forward-backward algorithm result.
#[derive(Debug, Clone)]
pub struct HmmForwardBackwardResult {
    /// Posterior state probabilities: γ_t(s) = P(q_t=s | observations), T × n_states.
    pub gamma: Vec<Vec<f64>>,
    /// Joint probabilities: ξ_t(s,s') = P(q_t=s, q_{t+1}=s' | obs), T-1 × n_states^2.
    pub xi: Vec<Vec<f64>>,
    /// Log-likelihood of the observation sequence.
    pub log_likelihood: f64,
}

/// Run the HMM forward-backward algorithm.
///
/// `observations`: T integer indices in [0, n_symbols).
pub fn hmm_forward_backward(
    params: &HmmParams,
    observations: &[usize],
) -> Option<HmmForwardBackwardResult> {
    let t = observations.len();
    let s = params.n_states;
    if t == 0 { return None; }

    // Forward: alpha_t(i) = P(y_1,...,y_t, q_t=i)
    // Scaled to avoid underflow (Rabiner 1989, Section VI-A)
    let mut alpha = vec![vec![0.0_f64; s]; t];
    let mut scale = vec![0.0_f64; t];

    // t=0
    for i in 0..s {
        alpha[0][i] = params.pi[i] * params.b(i, observations[0]);
    }
    scale[0] = alpha[0].iter().sum::<f64>();
    if scale[0] < 1e-300 { return None; }
    for i in 0..s { alpha[0][i] /= scale[0]; }

    // t=1..T-1
    for tt in 1..t {
        let o = observations[tt];
        for j in 0..s {
            alpha[tt][j] = (0..s).map(|i| alpha[tt-1][i] * params.a(i, j)).sum::<f64>()
                * params.b(j, o);
        }
        scale[tt] = alpha[tt].iter().sum::<f64>();
        if scale[tt] < 1e-300 { return None; }
        for j in 0..s { alpha[tt][j] /= scale[tt]; }
    }

    // Backward: beta_t(i) = P(y_{t+1},...,y_T | q_t=i), scaled
    let mut beta = vec![vec![1.0_f64; s]; t];
    // beta_{T-1} = 1 (already), but scale by 1/scale[T-1]
    for i in 0..s { beta[t-1][i] /= scale[t-1]; }

    for tt in (0..t-1).rev() {
        let o_next = observations[tt + 1];
        for i in 0..s {
            beta[tt][i] = (0..s)
                .map(|j| params.a(i, j) * params.b(j, o_next) * beta[tt+1][j])
                .sum::<f64>()
                / scale[tt];
        }
    }

    // Gamma: γ_t(i) = α_t(i) * β_t(i) / Σ_j α_t(j) β_t(j)
    let mut gamma = vec![vec![0.0_f64; s]; t];
    for tt in 0..t {
        let denom: f64 = (0..s).map(|i| alpha[tt][i] * beta[tt][i]).sum();
        if denom < 1e-300 { continue; }
        for i in 0..s { gamma[tt][i] = alpha[tt][i] * beta[tt][i] / denom; }
    }

    // Xi: ξ_t(i,j) = α_t(i) * A[i,j] * B[j,o_{t+1}] * β_{t+1}(j) / P(obs)
    let mut xi = vec![vec![0.0_f64; s * s]; t - 1];
    for tt in 0..t-1 {
        let o_next = observations[tt + 1];
        let mut denom = 0.0_f64;
        for i in 0..s {
            for j in 0..s {
                xi[tt][i * s + j] = alpha[tt][i] * params.a(i,j) * params.b(j, o_next) * beta[tt+1][j];
                denom += xi[tt][i * s + j];
            }
        }
        if denom > 1e-300 {
            for v in xi[tt].iter_mut() { *v /= denom; }
        }
    }

    // Log-likelihood: Σ ln(scale_t)
    let log_likelihood = scale.iter().map(|&c| c.ln()).sum::<f64>();

    Some(HmmForwardBackwardResult { gamma, xi, log_likelihood })
}

/// Viterbi decoding: most likely hidden state sequence.
///
/// Returns the most probable hidden state sequence and its log-probability.
pub fn hmm_viterbi(params: &HmmParams, observations: &[usize]) -> Option<(Vec<usize>, f64)> {
    let t = observations.len();
    let s = params.n_states;
    if t == 0 { return None; }

    // delta[t][i] = max log-prob of any path ending in state i at time t
    let mut delta = vec![vec![f64::NEG_INFINITY; s]; t];
    let mut psi   = vec![vec![0usize; s]; t];

    // Initialize
    for i in 0..s {
        let pi_i = params.pi[i];
        let b_i0 = params.b(i, observations[0]);
        delta[0][i] = if pi_i > 0.0 && b_i0 > 0.0 {
            pi_i.ln() + b_i0.ln()
        } else {
            f64::NEG_INFINITY
        };
    }

    // Recursion
    for tt in 1..t {
        let o = observations[tt];
        for j in 0..s {
            let b_jo = params.b(j, o);
            if b_jo <= 0.0 {
                delta[tt][j] = f64::NEG_INFINITY;
                psi[tt][j] = 0;
                continue;
            }
            let log_b = b_jo.ln();
            let (best_i, best_val) = (0..s)
                .map(|i| {
                    let a_ij = params.a(i, j);
                    let v = if a_ij > 0.0 { delta[tt-1][i] + a_ij.ln() + log_b }
                            else { f64::NEG_INFINITY };
                    (i, v)
                })
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();
            delta[tt][j] = best_val;
            psi[tt][j] = best_i;
        }
    }

    // Termination
    let (best_final, best_log_prob) = (0..s)
        .map(|i| (i, delta[t-1][i]))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
        .unwrap();

    // Backtrack
    let mut path = vec![0usize; t];
    path[t-1] = best_final;
    for tt in (0..t-1).rev() {
        path[tt] = psi[tt+1][path[tt+1]];
    }

    Some((path, best_log_prob))
}

/// Baum-Welch EM algorithm for HMM parameter estimation.
///
/// Runs up to `max_iter` iterations or until log-likelihood improvement < `tol`.
/// Returns the fitted `HmmParams` and the final log-likelihood.
pub fn hmm_baum_welch(
    initial_params: &HmmParams,
    observations: &[usize],
    max_iter: usize,
    tol: f64,
) -> Option<(HmmParams, f64)> {
    let mut params = initial_params.clone();
    let s = params.n_states;
    let v = params.n_symbols;
    let t = observations.len();
    if t == 0 { return None; }

    let mut prev_ll = f64::NEG_INFINITY;

    for _iter in 0..max_iter {
        let fb = hmm_forward_backward(&params, observations)?;
        let ll = fb.log_likelihood;

        // M-step: re-estimate pi, A, B from gamma and xi
        let gamma = &fb.gamma;
        let xi = &fb.xi;

        // pi: P(q_1=i) = gamma_0(i)
        let new_pi = gamma[0].clone();

        // A: A[i,j] = Σ_{t=0}^{T-2} ξ_t(i,j) / Σ_{t=0}^{T-2} γ_t(i)
        let mut new_a = vec![0.0_f64; s * s];
        for i in 0..s {
            let denom: f64 = (0..t-1).map(|tt| gamma[tt][i]).sum();
            if denom < 1e-300 {
                // Keep row uniform if state never visited
                for j in 0..s { new_a[i * s + j] = 1.0 / s as f64; }
                continue;
            }
            for j in 0..s {
                let numer: f64 = (0..t-1).map(|tt| xi[tt][i * s + j]).sum();
                new_a[i * s + j] = numer / denom;
            }
            // Normalize row to sum to 1 (may differ from 1 due to floating point)
            let row_sum: f64 = (0..s).map(|j| new_a[i * s + j]).sum();
            if row_sum > 1e-300 {
                for j in 0..s { new_a[i * s + j] /= row_sum; }
            }
        }

        // B: B[i,o] = Σ_{t: obs_t=o} γ_t(i) / Σ_t γ_t(i)
        let mut new_b = vec![0.0_f64; s * v];
        for i in 0..s {
            let denom: f64 = (0..t).map(|tt| gamma[tt][i]).sum();
            if denom < 1e-300 {
                for o in 0..v { new_b[i * v + o] = 1.0 / v as f64; }
                continue;
            }
            for o in 0..v {
                let numer: f64 = (0..t)
                    .filter(|&tt| observations[tt] == o)
                    .map(|tt| gamma[tt][i])
                    .sum();
                new_b[i * v + o] = numer / denom;
            }
            let row_sum: f64 = (0..v).map(|o| new_b[i * v + o]).sum();
            if row_sum > 1e-300 {
                for o in 0..v { new_b[i * v + o] /= row_sum; }
            }
        }

        params.pi = new_pi;
        params.transition = new_a;
        params.emission = new_b;

        if (ll - prev_ll).abs() < tol { break; }
        prev_ll = ll;
    }

    let final_fb = hmm_forward_backward(&params, observations)?;
    Some((params, final_fb.log_likelihood))
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Scalar Kalman ───────────────────────────────────────────────────────

    #[test]
    fn kalman_scalar_random_walk_converges() {
        // Random walk: F=1, H=1, Q=0.01, R=0.1
        // Steady-state Kalman gain: K_ss = q/2 * (-1 + sqrt(1 + 4r/q))
        let q = 0.01_f64;
        let r = 0.1_f64;
        // Synthetic: x_t = x_{t-1} + N(0, sqrt(q)), y_t = x_t + N(0, sqrt(r))
        let mut rng = crate::rng::Xoshiro256::new(42);
        use crate::rng::TamRng;
        let n = 200;
        let mut x_true = 0.0_f64;
        let mut obs: Vec<f64> = Vec::with_capacity(n);
        for _ in 0..n {
            x_true += crate::rng::sample_normal(&mut rng, 0.0, q.sqrt());
            obs.push(x_true + crate::rng::sample_normal(&mut rng, 0.0, r.sqrt()));
        }
        let res = kalman_filter_scalar(&obs, 1.0, 1.0, q, r, 0.0, 1.0)
            .expect("should produce result");
        assert_eq!(res.states.len(), n);
        // Gains should converge to K_ss
        let p_inf = q / 2.0 * (-1.0 + (1.0 + 4.0 * r / q).sqrt());
        let k_ss = (p_inf + q) / (p_inf + q + r);
        let last_k = res.gains[n - 1];
        assert!((last_k - k_ss).abs() < 0.01,
            "Kalman gain should converge to {:.4}, got {:.4}", k_ss, last_k);
    }

    #[test]
    fn kalman_scalar_constant_signal() {
        // Constant signal x=5.0, observations with noise.
        // Filter should converge toward 5.0.
        let obs: Vec<f64> = (0..100).map(|_| 5.0).collect();
        let res = kalman_filter_scalar(&obs, 1.0, 1.0, 0.001, 0.1, 0.0, 10.0)
            .expect("result");
        let last = *res.states.last().unwrap();
        assert!((last - 5.0).abs() < 0.2, "final state={:.4} should be near 5.0", last);
    }

    #[test]
    fn kalman_scalar_log_likelihood_finite() {
        let obs: Vec<f64> = (0..50).map(|i| i as f64 * 0.1).collect();
        let res = kalman_filter_scalar(&obs, 1.0, 1.0, 0.1, 0.5, 0.0, 1.0)
            .expect("result");
        assert!(res.log_likelihood.is_finite(), "log-likelihood must be finite");
        assert!(res.log_likelihood < 0.0, "log-likelihood must be negative");
    }

    #[test]
    fn kalman_scalar_missing_obs() {
        // NaN in observations should be handled gracefully
        let obs = vec![1.0, f64::NAN, 3.0, f64::NAN, 5.0];
        let res = kalman_filter_scalar(&obs, 1.0, 1.0, 0.1, 0.5, 0.0, 1.0)
            .expect("result");
        assert_eq!(res.states.len(), 5);
        // Gains at NaN positions should be 0
        assert_eq!(res.gains[1], 0.0);
        assert_eq!(res.gains[3], 0.0);
    }

    #[test]
    fn rts_smoother_improves_variance() {
        // Smoother should produce smaller or equal variance than filter
        let mut rng = crate::rng::Xoshiro256::new(7);
        use crate::rng::TamRng;
        let n = 30;
        let obs: Vec<f64> = (0..n)
            .map(|_| crate::rng::sample_normal(&mut rng, 0.0, 1.0))
            .collect();
        let filter = kalman_filter_scalar(&obs, 0.9, 1.0, 0.1, 0.5, 0.0, 1.0).unwrap();
        let (_x_smooth, p_smooth) = rts_smoother_scalar(&filter, 0.9, 0.1);
        // Interior smoothed variances should be ≤ filtered variances
        for t in 1..n-1 {
            assert!(p_smooth[t] <= filter.variances[t] + 1e-10,
                "smoother variance[{}]={:.6} > filter[{}]={:.6}",
                t, p_smooth[t], t, filter.variances[t]);
        }
    }

    #[test]
    fn kalman_ar1_equivalent() {
        // AR(1): x_t = phi * x_{t-1} + e_t observed directly (H=1, R≈0)
        // With near-zero R, filter should track observations closely
        let phi = 0.8;
        let obs: Vec<f64> = vec![1.0, 0.9, 0.7, 0.5, 0.3, 0.2];
        let res = kalman_filter_scalar(&obs, phi, 1.0, 0.01, 1e-6, 0.0, 1.0).unwrap();
        // With tiny observation noise, state should follow observations
        for t in 0..obs.len() {
            assert!((res.states[t] - obs[t]).abs() < 0.01,
                "state[{}]={:.4} far from obs[{}]={:.4}", t, res.states[t], t, obs[t]);
        }
    }

    // ── Matrix Kalman ───────────────────────────────────────────────────────

    #[test]
    fn kalman_matrix_2d_scalar_equivalent() {
        // 1D state, 1D obs should match scalar version
        let obs_vec = vec![1.0, 2.0, 1.5, 2.5, 2.0];
        let f = 0.9_f64;
        let h = 1.0_f64;
        let q = 0.1_f64;
        let r = 0.5_f64;

        let scalar = kalman_filter_scalar(&obs_vec, f, h, q, r, 0.0, 1.0).unwrap();
        let mat_res = kalman_filter_matrix(
            &obs_vec,
            &[f],            // F: 1×1
            &[h],            // H: 1×1
            &[q],            // Q: 1×1
            &[r],            // R: 1×1
            &[0.0],          // x0
            &[1.0],          // P0: 1×1
            1, 1,
        ).unwrap();

        for t in 0..obs_vec.len() {
            assert!((scalar.states[t] - mat_res.states[t][0]).abs() < 1e-10,
                "state[{}]: scalar={:.8} mat={:.8}", t, scalar.states[t], mat_res.states[t][0]);
        }
    }

    // ── HMM ────────────────────────────────────────────────────────────────

    #[test]
    fn hmm_viterbi_deterministic() {
        // 2-state HMM where state 0 always emits 0, state 1 always emits 1
        let params = HmmParams {
            pi: vec![0.5, 0.5],
            transition: vec![0.9, 0.1, 0.1, 0.9],  // mostly self-transitions
            emission: vec![1.0, 0.0, 0.0, 1.0],     // deterministic
            n_states: 2,
            n_symbols: 2,
        };
        let obs = vec![0usize, 0, 0, 1, 1, 1];
        let (path, _) = hmm_viterbi(&params, &obs).unwrap();
        assert_eq!(&path[..3], &[0, 0, 0], "first 3 obs should be state 0");
        assert_eq!(&path[3..], &[1, 1, 1], "last 3 obs should be state 1");
    }

    #[test]
    fn hmm_forward_backward_gamma_sums_to_one() {
        let params = HmmParams::uniform(3, 4);
        let obs = vec![0usize, 1, 2, 3, 0, 1];
        let fb = hmm_forward_backward(&params, &obs).unwrap();
        for (t, row) in fb.gamma.iter().enumerate() {
            let sum: f64 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10,
                "gamma[{}] sums to {:.12}, expected 1.0", t, sum);
        }
    }

    #[test]
    fn hmm_baum_welch_improves_likelihood() {
        // Baum-Welch should increase (or maintain) log-likelihood
        let params = HmmParams::uniform(2, 3);
        let obs = vec![0usize, 1, 0, 2, 1, 0, 1, 2, 0, 1];
        let fb0 = hmm_forward_backward(&params, &obs).unwrap();
        let (_, ll_bw) = hmm_baum_welch(&params, &obs, 20, 1e-6).unwrap();
        assert!(ll_bw >= fb0.log_likelihood - 1e-6,
            "Baum-Welch LL={:.4} should be ≥ initial LL={:.4}", ll_bw, fb0.log_likelihood);
    }

    #[test]
    fn hmm_viterbi_empty() {
        let params = HmmParams::uniform(2, 2);
        assert!(hmm_viterbi(&params, &[]).is_none());
    }

    #[test]
    fn hmm_forward_backward_empty() {
        let params = HmmParams::uniform(2, 2);
        assert!(hmm_forward_backward(&params, &[]).is_none());
    }

    #[test]
    fn hmm_single_observation() {
        let params = HmmParams::uniform(2, 3);
        let fb = hmm_forward_backward(&params, &[1]).unwrap();
        assert_eq!(fb.gamma.len(), 1);
        let sum: f64 = fb.gamma[0].iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
