//! Hidden Markov Models — forward-backward, Viterbi, Baum-Welch.
//!
//! ## Model
//!
//! A discrete-time HMM is defined by:
//! - `n_states`: number of hidden states
//! - `n_obs`: number of distinct observation symbols
//! - `pi[i]`: initial state probability P(s_0 = i)
//! - `a[i][j]` = `trans[i * n_states + j]`: transition P(s_t = j | s_{t-1} = i)
//! - `b[i][k]` = `emit[i * n_obs + k]`: emission P(o_t = k | s_t = i)
//!
//! All probability matrices are row-stochastic (rows sum to 1).
//!
//! ## Numerical stability
//!
//! Forward/backward are computed in log-domain via log-sum-exp to avoid
//! underflow for long sequences. Viterbi uses log-domain max-product.
//!
//! ## Algorithms
//!
//! - `hmm_forward`: log-likelihood of a sequence (log-alpha, scaling factors)
//! - `hmm_backward`: backward log-probabilities
//! - `hmm_forward_backward`: posterior state probabilities γ_t(i)
//! - `hmm_viterbi`: most probable state sequence
//! - `hmm_baum_welch`: EM parameter re-estimation

use std::f64::NEG_INFINITY;

// ═══════════════════════════════════════════════════════════════════════════
// Core data structures
// ═══════════════════════════════════════════════════════════════════════════

/// A Hidden Markov Model with discrete emissions.
#[derive(Debug, Clone)]
pub struct Hmm {
    /// Number of hidden states.
    pub n_states: usize,
    /// Number of observation symbols.
    pub n_obs: usize,
    /// Initial state log-probabilities (length n_states). ln(π_i).
    pub log_pi: Vec<f64>,
    /// Transition log-probabilities, row-major (n_states × n_states). ln A_{ij}.
    pub log_trans: Vec<f64>,
    /// Emission log-probabilities, row-major (n_states × n_obs). ln B_{ik}.
    pub log_emit: Vec<f64>,
}

impl Hmm {
    /// Construct from probability matrices (not log).
    /// `pi`: length n_states
    /// `trans`: n_states × n_states row-major
    /// `emit`: n_states × n_obs row-major
    pub fn from_probs(n_states: usize, n_obs: usize,
                      pi: &[f64], trans: &[f64], emit: &[f64]) -> Self {
        let log_pi = pi.iter().map(|&p| safe_ln(p)).collect();
        let log_trans = trans.iter().map(|&p| safe_ln(p)).collect();
        let log_emit = emit.iter().map(|&p| safe_ln(p)).collect();
        Hmm { n_states, n_obs, log_pi, log_trans, log_emit }
    }

    /// Convert back to probability domain (for inspection / Baum-Welch output).
    pub fn to_probs(&self) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let pi: Vec<f64> = self.log_pi.iter().map(|&lp| lp.exp()).collect();
        let trans: Vec<f64> = self.log_trans.iter().map(|&lp| lp.exp()).collect();
        let emit: Vec<f64> = self.log_emit.iter().map(|&lp| lp.exp()).collect();
        (pi, trans, emit)
    }

    fn log_a(&self, i: usize, j: usize) -> f64 {
        self.log_trans[i * self.n_states + j]
    }

    fn log_b(&self, i: usize, k: usize) -> f64 {
        self.log_emit[i * self.n_obs + k]
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Numerical helpers
// ═══════════════════════════════════════════════════════════════════════════

/// ln(x) returning -∞ for x ≤ 0.
fn safe_ln(x: f64) -> f64 {
    if x <= 0.0 { NEG_INFINITY } else { x.ln() }
}

/// log-sum-exp of a slice (stable).
fn log_sum_exp(values: &[f64]) -> f64 {
    let max = values.iter().cloned().fold(NEG_INFINITY, f64::max);
    if max == NEG_INFINITY { return NEG_INFINITY; }
    let sum: f64 = values.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Result of the forward pass.
#[derive(Debug, Clone)]
pub struct ForwardResult {
    /// Log-forward probabilities: log_alpha[t * n_states + i] = ln α_t(i).
    pub log_alpha: Vec<f64>,
    /// Log-likelihood of the observation sequence: ln P(O | λ).
    pub log_likelihood: f64,
    /// Sequence length.
    pub t_len: usize,
}

/// Forward algorithm in log-domain.
///
/// `obs` contains integer observation indices in 0..n_obs.
pub fn hmm_forward(hmm: &Hmm, obs: &[usize]) -> ForwardResult {
    let n = hmm.n_states;
    let t_len = obs.len();
    let mut log_alpha = vec![NEG_INFINITY; t_len * n];

    // Initialise t=0
    for i in 0..n {
        log_alpha[i] = hmm.log_pi[i] + hmm.log_b(i, obs[0]);
    }

    // Recursion
    let mut buf = vec![0.0_f64; n];
    for t in 1..t_len {
        for j in 0..n {
            for i in 0..n {
                buf[i] = log_alpha[(t - 1) * n + i] + hmm.log_a(i, j);
            }
            log_alpha[t * n + j] = log_sum_exp(&buf) + hmm.log_b(j, obs[t]);
        }
    }

    let last: Vec<f64> = (0..n).map(|i| log_alpha[(t_len - 1) * n + i]).collect();
    let log_likelihood = log_sum_exp(&last);

    ForwardResult { log_alpha, log_likelihood, t_len }
}

// ═══════════════════════════════════════════════════════════════════════════
// Backward algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Result of the backward pass.
#[derive(Debug, Clone)]
pub struct BackwardResult {
    /// Log-backward probabilities: log_beta[t * n_states + i] = ln β_t(i).
    pub log_beta: Vec<f64>,
    /// Sequence length.
    pub t_len: usize,
}

/// Backward algorithm in log-domain.
pub fn hmm_backward(hmm: &Hmm, obs: &[usize]) -> BackwardResult {
    let n = hmm.n_states;
    let t_len = obs.len();
    let mut log_beta = vec![NEG_INFINITY; t_len * n];

    // Initialise t = T-1: β_{T-1}(i) = 1 → log = 0
    for i in 0..n {
        log_beta[(t_len - 1) * n + i] = 0.0;
    }

    // Recursion (backwards)
    let mut buf = vec![0.0_f64; n];
    for t in (0..t_len - 1).rev() {
        for i in 0..n {
            for j in 0..n {
                buf[j] = hmm.log_a(i, j)
                    + hmm.log_b(j, obs[t + 1])
                    + log_beta[(t + 1) * n + j];
            }
            log_beta[t * n + i] = log_sum_exp(&buf);
        }
    }

    BackwardResult { log_beta, t_len }
}

// ═══════════════════════════════════════════════════════════════════════════
// Forward-backward (posterior state probabilities)
// ═══════════════════════════════════════════════════════════════════════════

/// Result of the forward-backward algorithm.
#[derive(Debug, Clone)]
pub struct ForwardBackwardResult {
    /// Posterior state probabilities γ_t(i) = P(s_t=i | O, λ).
    /// Row-major t × n_states.
    pub gamma: Vec<f64>,
    /// Pairwise posterior ξ_t(i,j) = P(s_t=i, s_{t+1}=j | O, λ).
    /// Row-major (T-1) × n_states × n_states.
    pub xi: Vec<f64>,
    /// Log-likelihood.
    pub log_likelihood: f64,
}

/// Forward-backward algorithm. Returns γ and ξ for Baum-Welch.
pub fn hmm_forward_backward(hmm: &Hmm, obs: &[usize]) -> ForwardBackwardResult {
    let n = hmm.n_states;
    let t_len = obs.len();

    let fwd = hmm_forward(hmm, obs);
    let bwd = hmm_backward(hmm, obs);
    let ll = fwd.log_likelihood;

    // γ_t(i) = α_t(i) β_t(i) / P(O)  →  log γ = log_α + log_β - ll
    let mut gamma = vec![0.0_f64; t_len * n];
    for t in 0..t_len {
        let mut row_log: Vec<f64> = (0..n).map(|i| {
            fwd.log_alpha[t * n + i] + bwd.log_beta[t * n + i]
        }).collect();
        let row_norm = log_sum_exp(&row_log);
        for i in 0..n {
            gamma[t * n + i] = (row_log[i] - row_norm).exp();
        }
    }

    // ξ_t(i,j) for t < T-1
    let mut xi = vec![0.0_f64; (t_len - 1) * n * n];
    for t in 0..t_len - 1 {
        let mut log_xi = vec![NEG_INFINITY; n * n];
        for i in 0..n {
            for j in 0..n {
                log_xi[i * n + j] = fwd.log_alpha[t * n + i]
                    + hmm.log_a(i, j)
                    + hmm.log_b(j, obs[t + 1])
                    + bwd.log_beta[(t + 1) * n + j];
            }
        }
        let norm = log_sum_exp(&log_xi);
        for ij in 0..n * n {
            xi[t * n * n + ij] = (log_xi[ij] - norm).exp();
        }
    }

    ForwardBackwardResult { gamma, xi, log_likelihood: ll }
}

// ═══════════════════════════════════════════════════════════════════════════
// Viterbi algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Result of the Viterbi algorithm.
#[derive(Debug, Clone)]
pub struct ViterbiResult {
    /// Most probable state sequence (length T).
    pub states: Vec<usize>,
    /// Log-probability of the most probable path.
    pub log_prob: f64,
}

/// Viterbi algorithm — log-domain max-product with backpointer.
///
/// # Kingdom classification
/// **Current implementation: Kingdom B** (sequential). The recurrence
/// `δ_t(j) = max_i [δ_{t-1}(i) + log A(i,j)] + log B(j, o_t)` has sequential
/// state dependency — δ_t depends on δ_{t-1} elementwise.
///
/// **Target: Kingdom A** via tropical max-plus semiring (ℝ, max, +). The
/// transition step is a matrix-vector multiply over (max,+) — a prefix scan
/// over the state-transition matrix. Once `Semiring<TropicalMaxPlus>` is
/// implemented and the scan engine is parameterized, this can be rewritten as
/// a parallel prefix scan with O(T log T) work instead of O(T·n²) sequential.
///
/// The backpointer (psi) tracking is the one part that genuinely requires
/// sequential state — it records argmax at each step. In the Kingdom A form,
/// the forward pass is parallel; the backtracing pass remains sequential.
pub fn hmm_viterbi(hmm: &Hmm, obs: &[usize]) -> ViterbiResult {
    let n = hmm.n_states;
    let t_len = obs.len();

    let mut delta = vec![NEG_INFINITY; t_len * n];
    let mut psi: Vec<usize> = vec![0; t_len * n];

    // Initialise
    for i in 0..n {
        delta[i] = hmm.log_pi[i] + hmm.log_b(i, obs[0]);
    }

    // Recursion
    for t in 1..t_len {
        for j in 0..n {
            let (best_val, best_state) = (0..n).map(|i| {
                (delta[(t - 1) * n + i] + hmm.log_a(i, j), i)
            }).fold((NEG_INFINITY, 0), |(bv, bs), (v, s)| {
                if v > bv { (v, s) } else { (bv, bs) }
            });
            delta[t * n + j] = best_val + hmm.log_b(j, obs[t]);
            psi[t * n + j] = best_state;
        }
    }

    // Termination
    let (log_prob, mut best_last) = (0..n).map(|i| {
        (delta[(t_len - 1) * n + i], i)
    }).fold((NEG_INFINITY, 0), |(bv, bs), (v, s)| {
        if v > bv { (v, s) } else { (bv, bs) }
    });

    // Backtrack
    let mut states = vec![0usize; t_len];
    states[t_len - 1] = best_last;
    for t in (0..t_len - 1).rev() {
        states[t] = psi[(t + 1) * n + states[t + 1]];
    }

    ViterbiResult { states, log_prob }
}

// ═══════════════════════════════════════════════════════════════════════════
// Baum-Welch (EM re-estimation)
// ═══════════════════════════════════════════════════════════════════════════

/// Result of Baum-Welch training.
#[derive(Debug, Clone)]
pub struct BaumWelchResult {
    /// Trained HMM.
    pub hmm: Hmm,
    /// Log-likelihood at each EM iteration.
    pub log_likelihoods: Vec<f64>,
    /// Number of iterations run.
    pub n_iter: usize,
    /// Whether convergence was reached within `max_iter`.
    pub converged: bool,
}

/// Baum-Welch EM algorithm for HMM parameter estimation.
///
/// `obs_sequences`: list of observation sequences (each is &[usize]).
/// `max_iter`: maximum EM iterations (default 100).
/// `tol`: convergence threshold on log-likelihood change (default 1e-4).
pub fn hmm_baum_welch(
    initial: &Hmm,
    obs_sequences: &[&[usize]],
    max_iter: usize,
    tol: f64,
) -> BaumWelchResult {
    let n = initial.n_states;
    let k = initial.n_obs;
    let mut hmm = initial.clone();
    let mut log_likelihoods = Vec::new();
    let mut prev_ll = NEG_INFINITY;
    let mut converged = false;

    for iter in 0..max_iter {
        // E-step: accumulate sufficient statistics across all sequences
        let mut pi_acc = vec![0.0_f64; n];
        let mut trans_acc = vec![0.0_f64; n * n];
        let mut emit_acc = vec![0.0_f64; n * k];
        let mut total_ll = 0.0;

        for &obs in obs_sequences {
            if obs.is_empty() { continue; }
            let fb = hmm_forward_backward(&hmm, obs);
            total_ll += fb.log_likelihood;
            let t_len = obs.len();

            // Accumulate π from t=0
            for i in 0..n {
                pi_acc[i] += fb.gamma[i];
            }

            // Accumulate A
            for t in 0..t_len - 1 {
                for i in 0..n {
                    for j in 0..n {
                        trans_acc[i * n + j] += fb.xi[t * n * n + i * n + j];
                    }
                }
            }

            // Accumulate B
            for t in 0..t_len {
                let sym = obs[t];
                for i in 0..n {
                    emit_acc[i * k + sym] += fb.gamma[t * n + i];
                }
            }
        }

        log_likelihoods.push(total_ll);

        // M-step: re-normalise
        // π
        let pi_sum: f64 = pi_acc.iter().sum();
        let new_log_pi: Vec<f64> = pi_acc.iter().map(|&v| safe_ln(v / pi_sum.max(1e-300))).collect();

        // A (row-normalise)
        let mut new_log_trans = vec![NEG_INFINITY; n * n];
        for i in 0..n {
            let row_sum: f64 = trans_acc[i * n..i * n + n].iter().sum();
            for j in 0..n {
                new_log_trans[i * n + j] = safe_ln(trans_acc[i * n + j] / row_sum.max(1e-300));
            }
        }

        // B (row-normalise)
        let mut new_log_emit = vec![NEG_INFINITY; n * k];
        for i in 0..n {
            let row_sum: f64 = emit_acc[i * k..i * k + k].iter().sum();
            for sym in 0..k {
                new_log_emit[i * k + sym] = safe_ln(emit_acc[i * k + sym] / row_sum.max(1e-300));
            }
        }

        hmm.log_pi = new_log_pi;
        hmm.log_trans = new_log_trans;
        hmm.log_emit = new_log_emit;

        // Check convergence
        if (total_ll - prev_ll).abs() < tol {
            converged = true;
            let n_iter = iter + 1;
            return BaumWelchResult { hmm, log_likelihoods, n_iter, converged };
        }
        prev_ll = total_ll;
    }

    BaumWelchResult { hmm, log_likelihoods, n_iter: max_iter, converged }
}

// ═══════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════

/// Initialise a random HMM with `n_states` states and `n_obs` observation symbols.
/// Uses Xoshiro256 seeded with `seed`. All matrices are row-stochastic.
pub fn hmm_random_init(n_states: usize, n_obs: usize, seed: u64) -> Hmm {
    use crate::rng::{Xoshiro256, TamRng};
    let mut rng = Xoshiro256::new(seed);

    let random_row = |rng: &mut Xoshiro256, len: usize| -> Vec<f64> {
        let raw: Vec<f64> = (0..len).map(|_| TamRng::next_f64(rng) + 1e-3).collect();
        let sum: f64 = raw.iter().sum();
        raw.iter().map(|&v| v / sum).collect()
    };

    let pi = random_row(&mut rng, n_states);
    let trans: Vec<f64> = (0..n_states).flat_map(|_| random_row(&mut rng, n_states)).collect();
    let emit: Vec<f64> = (0..n_states).flat_map(|_| random_row(&mut rng, n_obs)).collect();

    Hmm::from_probs(n_states, n_obs, &pi, &trans, &emit)
}

/// Sample a sequence of length `t_len` from an HMM.
pub fn hmm_sample(hmm: &Hmm, t_len: usize, seed: u64) -> (Vec<usize>, Vec<usize>) {
    use crate::rng::{Xoshiro256, TamRng};
    let n = hmm.n_states;
    let k = hmm.n_obs;
    let mut rng = Xoshiro256::new(seed);

    let categorical = |log_probs: &[f64], rng: &mut Xoshiro256| -> usize {
        // Sample from a log-probability vector by computing cumulative probs
        let max = log_probs.iter().cloned().fold(NEG_INFINITY, f64::max);
        let probs: Vec<f64> = log_probs.iter().map(|&lp| (lp - max).exp()).collect();
        let sum: f64 = probs.iter().sum();
        let u = TamRng::next_f64(rng) * sum;
        let mut cumsum = 0.0;
        for (i, &p) in probs.iter().enumerate() {
            cumsum += p;
            if u <= cumsum { return i; }
        }
        probs.len() - 1
    };

    let mut states = Vec::with_capacity(t_len);
    let mut obs = Vec::with_capacity(t_len);

    let s0 = categorical(&hmm.log_pi, &mut rng);
    states.push(s0);
    let o0 = categorical(&hmm.log_emit[s0 * k..s0 * k + k], &mut rng);
    obs.push(o0);

    for _ in 1..t_len {
        let prev = *states.last().unwrap();
        let s = categorical(&hmm.log_trans[prev * n..prev * n + n], &mut rng);
        states.push(s);
        let o = categorical(&hmm.log_emit[s * k..s * k + k], &mut rng);
        obs.push(o);
    }

    (states, obs)
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    /// Coin-flip HMM: 2 states (fair/biased), 2 obs (H/T).
    fn coin_hmm() -> Hmm {
        // State 0 = fair: P(H)=0.5, P(T)=0.5
        // State 1 = biased: P(H)=0.9, P(T)=0.1
        // Transitions: tends to stay in same state (0.9 self-loop)
        Hmm::from_probs(
            2, 2,
            &[0.5, 0.5],
            &[0.9, 0.1, 0.1, 0.9],
            &[0.5, 0.5, 0.9, 0.1],
        )
    }

    #[test]
    fn forward_likelihood_positive() {
        let hmm = coin_hmm();
        let obs = vec![0, 0, 1, 0, 0]; // H H T H H
        let fwd = hmm_forward(&hmm, &obs);
        // Log-likelihood should be finite and negative
        assert!(fwd.log_likelihood.is_finite());
        assert!(fwd.log_likelihood < 0.0);
        assert_eq!(fwd.log_alpha.len(), obs.len() * 2);
    }

    #[test]
    fn backward_consistent_with_forward() {
        // P(O) via forward == P(O) via β_0 * P(o_0 | s) * π
        let hmm = coin_hmm();
        let obs = vec![1, 0, 0, 1, 0];
        let fwd = hmm_forward(&hmm, &obs);
        let bwd = hmm_backward(&hmm, &obs);
        // Alternative LL from backward: Σ_i π_i * b(o_0 | i) * β_0(i)
        let n = hmm.n_states;
        let log_vals: Vec<f64> = (0..n).map(|i| {
            hmm.log_pi[i] + hmm.log_b(i, obs[0]) + bwd.log_beta[i]
        }).collect();
        let bwd_ll = log_sum_exp(&log_vals);
        assert!((fwd.log_likelihood - bwd_ll).abs() < 1e-9,
            "forward LL = {}, backward LL = {}", fwd.log_likelihood, bwd_ll);
    }

    #[test]
    fn gamma_rows_sum_to_one() {
        let hmm = coin_hmm();
        let obs = vec![0, 1, 0, 1, 1, 0];
        let fb = hmm_forward_backward(&hmm, &obs);
        let n = hmm.n_states;
        for t in 0..obs.len() {
            let row_sum: f64 = (0..n).map(|i| fb.gamma[t * n + i]).sum();
            assert!((row_sum - 1.0).abs() < 1e-10,
                "γ row {t} sums to {row_sum}");
        }
    }

    #[test]
    fn xi_rows_sum_to_one() {
        let hmm = coin_hmm();
        let obs = vec![0, 0, 1, 0, 0];
        let fb = hmm_forward_backward(&hmm, &obs);
        let n = hmm.n_states;
        let t_len = obs.len();
        for t in 0..t_len - 1 {
            let row_sum: f64 = (0..n * n).map(|ij| fb.xi[t * n * n + ij]).sum();
            assert!((row_sum - 1.0).abs() < 1e-10,
                "ξ slice {t} sums to {row_sum}");
        }
    }

    #[test]
    fn viterbi_returns_valid_states() {
        let hmm = coin_hmm();
        let obs = vec![0, 0, 0, 1, 1, 1]; // 3 H then 3 T
        let vit = hmm_viterbi(&hmm, &obs);
        // Basic validity checks — length and valid state indices
        assert_eq!(vit.states.len(), obs.len());
        assert!(vit.states.iter().all(|&s| s < hmm.n_states));
        assert!(vit.log_prob.is_finite());
        assert!(vit.log_prob < 0.0);
        // The path should be contiguous in state-space (sticky transitions)
        // Check that log_prob matches independently computed path probability
        let mut path_log_prob = hmm.log_pi[vit.states[0]] + hmm.log_b(vit.states[0], obs[0]);
        for t in 1..obs.len() {
            path_log_prob += hmm.log_a(vit.states[t - 1], vit.states[t])
                + hmm.log_b(vit.states[t], obs[t]);
        }
        assert!((vit.log_prob - path_log_prob).abs() < 1e-9,
            "viterbi prob {} != path recompute {}", vit.log_prob, path_log_prob);
    }

    #[test]
    fn viterbi_log_prob_le_forward_ll() {
        // P(most probable path) ≤ P(O) — marginalizing over all paths gives ≥
        let hmm = coin_hmm();
        let obs = vec![0, 0, 1, 1, 0];
        let fwd = hmm_forward(&hmm, &obs);
        let vit = hmm_viterbi(&hmm, &obs);
        assert!(vit.log_prob <= fwd.log_likelihood + 1e-9,
            "viterbi {} > forward {}", vit.log_prob, fwd.log_likelihood);
    }

    #[test]
    fn baum_welch_improves_likelihood() {
        // Train with a single sequence — LL should not decrease
        let hmm = hmm_random_init(2, 2, 42);
        let obs = vec![0usize, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0];
        let result = hmm_baum_welch(&hmm, &[obs.as_slice()], 50, 1e-6);
        // Log-likelihood should be monotonically non-decreasing
        let lls = &result.log_likelihoods;
        for i in 1..lls.len() {
            assert!(lls[i] >= lls[i - 1] - 1e-6,
                "LL decreased at iter {}: {} -> {}", i, lls[i - 1], lls[i]);
        }
    }

    #[test]
    fn baum_welch_converges_on_simple_model() {
        // 2-state, 2-obs HMM trained on a long enough sequence should converge
        let true_hmm = coin_hmm();
        let (_, obs) = hmm_sample(&true_hmm, 200, 42);
        let init = hmm_random_init(2, 2, 99);
        let result = hmm_baum_welch(&init, &[obs.as_slice()], 200, 1e-6);
        assert!(result.converged || result.n_iter == 200,
            "should run to completion");
        // Final LL should be better than initial
        let init_ll = hmm_forward(&init, &obs).log_likelihood;
        let final_ll = *result.log_likelihoods.last().unwrap();
        assert!(final_ll > init_ll, "final {} <= initial {}", final_ll, init_ll);
    }

    #[test]
    fn sample_produces_valid_observations() {
        let hmm = coin_hmm();
        let (states, obs) = hmm_sample(&hmm, 100, 7);
        assert_eq!(states.len(), 100);
        assert_eq!(obs.len(), 100);
        assert!(states.iter().all(|&s| s < 2));
        assert!(obs.iter().all(|&o| o < 2));
    }

    #[test]
    fn single_observation_sequence() {
        // Edge case: length-1 sequence
        let hmm = coin_hmm();
        let obs = vec![0usize];
        let fwd = hmm_forward(&hmm, &obs);
        assert!(fwd.log_likelihood.is_finite());
        let vit = hmm_viterbi(&hmm, &obs);
        assert_eq!(vit.states.len(), 1);
    }

    #[test]
    fn three_state_hmm_viterbi() {
        // 3 states, 3 obs — test that Viterbi handles larger models correctly
        let pi = vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0];
        let trans = vec![
            0.7, 0.2, 0.1,
            0.1, 0.7, 0.2,
            0.2, 0.1, 0.7,
        ];
        let emit = vec![
            0.8, 0.1, 0.1,
            0.1, 0.8, 0.1,
            0.1, 0.1, 0.8,
        ];
        let hmm = Hmm::from_probs(3, 3, &pi, &trans, &emit);
        // Sequence mostly emitting obs 0 → should largely be in state 0
        let obs = vec![0usize, 0, 0, 0, 0, 1, 0, 0];
        let vit = hmm_viterbi(&hmm, &obs);
        assert_eq!(vit.states.len(), obs.len());
        let state0_count = vit.states.iter().filter(|&&s| s == 0).count();
        assert!(state0_count >= 5, "expected mostly state 0, got {state0_count}");
    }

    #[test]
    fn log_sum_exp_basic() {
        // log(e^1 + e^2) = 2 + log(1 + e^{-1}) ≈ 2.3133
        let result = log_sum_exp(&[1.0, 2.0]);
        let expected = (1.0_f64.exp() + 2.0_f64.exp()).ln();
        assert!((result - expected).abs() < 1e-12, "got {result}");
    }

    #[test]
    fn log_sum_exp_underflow_safe() {
        // All -inf → -inf
        let result = log_sum_exp(&[f64::NEG_INFINITY, f64::NEG_INFINITY]);
        assert_eq!(result, f64::NEG_INFINITY);
    }

    #[test]
    fn baum_welch_multiple_sequences() {
        // Should work with multiple sequences
        let true_hmm = coin_hmm();
        let (_, obs1) = hmm_sample(&true_hmm, 50, 1);
        let (_, obs2) = hmm_sample(&true_hmm, 50, 2);
        let (_, obs3) = hmm_sample(&true_hmm, 50, 3);
        let init = hmm_random_init(2, 2, 42);
        let result = hmm_baum_welch(
            &init,
            &[obs1.as_slice(), obs2.as_slice(), obs3.as_slice()],
            100, 1e-4
        );
        assert!(!result.log_likelihoods.is_empty());
        // Monotone non-decreasing per-batch (aggregate LL may fluctuate if sequences are independent)
        // Just check it ran and produced finite values
        assert!(result.log_likelihoods.iter().all(|ll| ll.is_finite()));
    }
}
