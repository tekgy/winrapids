//! # Family 34 — Bayesian Methods
//!
//! MCMC (Metropolis-Hastings, Gibbs), variational inference, Bayesian regression.
//!
//! ## Architecture
//!
//! MCMC = sequential sampling (Kingdom B).
//! Variational inference = optimization of ELBO (Kingdom C).
//! Bayesian regression = conjugate update (Kingdom A — closed form).

// ═══════════════════════════════════════════════════════════════════════════
// Metropolis-Hastings MCMC
// ═══════════════════════════════════════════════════════════════════════════

/// MCMC chain result.
#[derive(Debug, Clone)]
pub struct McmcChain {
    /// Samples (n_samples × d).
    pub samples: Vec<Vec<f64>>,
    /// Acceptance rate.
    pub acceptance_rate: f64,
}

/// Metropolis-Hastings sampler with Xoshiro256** PRNG.
///
/// `log_target`: log of unnormalized target density.
/// `initial`: starting point (length d).
/// `proposal_sd`: proposal standard deviation per dimension.
/// `n_samples`: number of samples to draw.
/// `burnin`: number of initial samples to discard.
/// `seed`: RNG seed for reproducibility.
pub fn metropolis_hastings(
    log_target: &dyn Fn(&[f64]) -> f64,
    initial: &[f64],
    proposal_sd: f64,
    n_samples: usize,
    burnin: usize,
    seed: u64,
) -> McmcChain {
    let d = initial.len();
    let total = n_samples + burnin;
    let mut current = initial.to_vec();
    let mut current_lp = log_target(&current);
    let mut samples = Vec::with_capacity(n_samples);
    let mut accepted = 0usize;
    let mut rng = crate::rng::Xoshiro256::new(seed);

    for iter in 0..total {
        // Propose: current + N(0, proposal_sd²)
        let mut proposal = current.clone();
        for j in 0..d {
            let z = crate::rng::sample_normal(&mut rng, 0.0, proposal_sd);
            proposal[j] += z;
        }

        let proposal_lp = log_target(&proposal);
        let log_alpha = proposal_lp - current_lp;

        let u = crate::rng::TamRng::next_f64(&mut rng).max(1e-300);
        if u.ln() < log_alpha {
            current = proposal;
            current_lp = proposal_lp;
            accepted += 1;
        }

        if iter >= burnin {
            samples.push(current.clone());
        }
    }

    McmcChain {
        samples,
        acceptance_rate: accepted as f64 / total as f64,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Bayesian linear regression (conjugate Normal-InverseGamma)
// ═══════════════════════════════════════════════════════════════════════════

/// Bayesian linear regression posterior.
#[derive(Debug, Clone)]
pub struct BayesLinearResult {
    /// Posterior mean of β.
    pub beta_mean: Vec<f64>,
    /// Posterior covariance of β (d×d, row-major).
    pub beta_cov: Vec<f64>,
    /// Posterior mean of σ².
    pub sigma2_mean: f64,
    /// Posterior shape parameter (α) of Inverse-Gamma for σ².
    pub alpha_post: f64,
    /// Posterior scale parameter (β) of Inverse-Gamma for σ².
    pub beta_post: f64,
}

/// Bayesian linear regression with conjugate Normal-InverseGamma prior.
/// Prior: β|σ² ~ N(β₀, σ²·Λ₀⁻¹), σ² ~ InvGamma(α₀, β₀).
/// `x`: n×d design matrix (row-major, includes intercept if desired).
/// `y`: response (length n).
/// `prior_precision`: Λ₀ (d×d, row-major). Use small diagonal for vague prior.
/// `prior_mean`: β₀ (length d).
/// `alpha0`, `beta0`: InverseGamma prior parameters for σ².
pub fn bayesian_linear_regression(
    x: &[f64], y: &[f64], n: usize, d: usize,
    prior_mean: &[f64], prior_precision: &[f64],
    alpha0: f64, beta0: f64,
) -> BayesLinearResult {
    assert_eq!(x.len(), n * d);
    assert_eq!(y.len(), n);
    assert_eq!(prior_mean.len(), d);
    assert_eq!(prior_precision.len(), d * d);

    // Posterior precision: Λ_n = Λ₀ + X'X
    let mut lambda_n = prior_precision.to_vec();
    for i in 0..n {
        for j in 0..d {
            for k in 0..d {
                lambda_n[j * d + k] += x[i * d + j] * x[i * d + k];
            }
        }
    }

    // Posterior mean: β_n = Λ_n⁻¹ (Λ₀ β₀ + X'y)
    let mut rhs = vec![0.0; d];
    for j in 0..d {
        for k in 0..d { rhs[j] += prior_precision[j * d + k] * prior_mean[k]; }
        for i in 0..n { rhs[j] += x[i * d + j] * y[i]; }
    }

    // If underdetermined (n < d), regularize posterior precision with ridge
    let lam_mat = crate::linear_algebra::Mat::from_vec(d, d, lambda_n.clone());
    let l = match crate::linear_algebra::cholesky(&lam_mat) {
        Some(l) => l,
        None => {
            // Add ridge regularization and retry
            let mut lambda_reg = lambda_n.clone();
            for j in 0..d { lambda_reg[j * d + j] += 1e-6; }
            let lam_reg_mat = crate::linear_algebra::Mat::from_vec(d, d, lambda_reg);
            crate::linear_algebra::cholesky(&lam_reg_mat)
                .expect("posterior precision not positive definite even with regularization")
        }
    };
    let beta_mean = crate::linear_algebra::cholesky_solve(&l, &rhs);

    // Posterior for σ²: InvGamma(α_n, β_n)
    let alpha_n = alpha0 + n as f64 / 2.0;

    // β_n = β₀ + 0.5 * (y'y + β₀'Λ₀β₀ - β_n'Λ_n β_n)
    let yty: f64 = y.iter().map(|yi| yi * yi).sum();
    let mut b0_lam_b0 = 0.0;
    for j in 0..d {
        for k in 0..d {
            b0_lam_b0 += prior_mean[j] * prior_precision[j * d + k] * prior_mean[k];
        }
    }
    let mut bn_lamn_bn = 0.0;
    for j in 0..d {
        for k in 0..d {
            bn_lamn_bn += beta_mean[j] * lambda_n[j * d + k] * beta_mean[k];
        }
    }
    let beta_n = beta0 + 0.5 * (yty + b0_lam_b0 - bn_lamn_bn);

    let sigma2_mean = beta_n / (alpha_n - 1.0).max(0.5);

    // Posterior covariance of β = σ²_mean · Λ_n⁻¹
    let mut beta_cov = vec![0.0; d * d];
    for j in 0..d {
        let mut ej = vec![0.0; d];
        ej[j] = 1.0;
        let col = crate::linear_algebra::cholesky_solve(&l, &ej);
        for k in 0..d { beta_cov[j * d + k] = sigma2_mean * col[k]; }
    }

    BayesLinearResult { beta_mean, beta_cov, sigma2_mean, alpha_post: alpha_n, beta_post: beta_n }
}

// ═══════════════════════════════════════════════════════════════════════════
// MCMC diagnostics
// ═══════════════════════════════════════════════════════════════════════════

/// Effective sample size (ESS) for a 1D chain via autocorrelation.
pub fn effective_sample_size(samples: &[f64]) -> f64 {
    let n = samples.len();
    if n < 10 { return n as f64; }

    let moments = crate::descriptive::moments_ungrouped(samples);
    let mean = moments.mean();
    let var = moments.variance(0);
    if var < 1e-15 { return n as f64; }

    // Compute all autocorrelations up to n/2
    let max_lag = n / 2;
    let mut rhos = Vec::with_capacity(max_lag);
    for lag in 1..=max_lag {
        let rho: f64 = samples[..n - lag].iter()
            .zip(samples[lag..].iter())
            .map(|(a, b)| (a - mean) * (b - mean))
            .sum::<f64>() / (n as f64 * var);
        rhos.push(rho);
    }

    // Geyer's initial monotone sequence estimator (IMSE): sum consecutive PAIRS
    // of autocorrelations (ρ_{2k-1} + ρ_{2k}), enforcing that pair sums are
    // both positive AND non-increasing. The monotone constraint prevents a
    // single noisy pair from causing premature truncation, which is the root
    // cause of ESS overestimation for highly autocorrelated chains.
    let mut pair_sums = Vec::new();
    let mut k = 0;
    while 2 * k + 1 < rhos.len() {
        let pair_sum = rhos[2 * k] + rhos[2 * k + 1];
        if pair_sum <= 0.0 { break; }
        pair_sums.push(pair_sum);
        k += 1;
    }
    // Enforce monotonicity: replace each pair sum with min(self, previous)
    for i in 1..pair_sums.len() {
        if pair_sums[i] > pair_sums[i - 1] {
            pair_sums[i] = pair_sums[i - 1];
        }
    }
    let mut sum_rho: f64 = pair_sums.iter().sum();
    // Add the last odd autocorrelation if it's positive and we haven't truncated
    if 2 * k < rhos.len() && rhos[2 * k] > 0.0 {
        sum_rho += rhos[2 * k];
    }

    let tau = 1.0 + 2.0 * sum_rho;
    n as f64 / tau.max(1.0)
}

/// R-hat (potential scale reduction factor) for multiple chains.
/// Each chain is a slice of samples for a single parameter.
pub fn r_hat(chains: &[&[f64]]) -> f64 {
    if chains.len() < 2 { return 1.0; } // R-hat undefined for single chain; return converged
    let m = chains.len() as f64;
    let n = chains[0].len() as f64;

    let chain_means: Vec<f64> = chains.iter().map(|c| c.iter().sum::<f64>() / n).collect();
    let overall_mean: f64 = chain_means.iter().sum::<f64>() / m;

    let b = n / (m - 1.0) * chain_means.iter().map(|&cm| (cm - overall_mean).powi(2)).sum::<f64>();
    let w: f64 = chains.iter().zip(&chain_means)
        .map(|(c, &cm)| c.iter().map(|s| (s - cm).powi(2)).sum::<f64>() / (n - 1.0))
        .sum::<f64>() / m;

    let var_hat = (n - 1.0) / n * w + b / n;
    // w=0 means all chains have zero within-chain variance (constant chains).
    // If var_hat is also 0, chains are identically converged: R̂ = 1.0 by definition.
    // If var_hat > 0 with w=0, chains are unmixed at different levels: R̂ = Inf.
    if w == 0.0 {
        return if var_hat == 0.0 { 1.0 } else { f64::INFINITY };
    }
    (var_hat / w).sqrt()
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

    // ── Metropolis-Hastings ─────────────────────────────────────────────

    #[test]
    fn mh_normal_target() {
        // Target: N(3.0, 1.0). Posterior mean should be ~3.0.
        let log_target = |x: &[f64]| -0.5 * (x[0] - 3.0).powi(2);
        let chain = metropolis_hastings(&log_target, &[0.0], 1.0, 5000, 1000, 42);
        let mean: f64 = chain.samples.iter().map(|s| s[0]).sum::<f64>() / chain.samples.len() as f64;
        assert!((mean - 3.0).abs() < 0.3, "MH mean={mean} should be ~3.0");
        assert!(chain.acceptance_rate > 0.15 && chain.acceptance_rate < 0.85,
            "Acceptance rate={}", chain.acceptance_rate);
    }

    #[test]
    fn mh_2d_target() {
        // Target: bivariate N([1, 2], I)
        let log_target = |x: &[f64]| -0.5 * ((x[0] - 1.0).powi(2) + (x[1] - 2.0).powi(2));
        let chain = metropolis_hastings(&log_target, &[0.0, 0.0], 1.0, 5000, 1000, 42);
        let m0: f64 = chain.samples.iter().map(|s| s[0]).sum::<f64>() / chain.samples.len() as f64;
        let m1: f64 = chain.samples.iter().map(|s| s[1]).sum::<f64>() / chain.samples.len() as f64;
        assert!((m0 - 1.0).abs() < 0.5, "Dim 0 mean={m0} should be ~1.0");
        assert!((m1 - 2.0).abs() < 0.5, "Dim 1 mean={m1} should be ~2.0");
    }

    // ── Bayesian linear regression ──────────────────────────────────────

    #[test]
    fn bayes_linear_known() {
        // y = 2 + 3x + noise. Should recover β ≈ [2, 3].
        let n = 50;
        let d = 2; // intercept + x
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = 42u64;
        for i in 0..n {
            let xi = i as f64 / n as f64;
            x.push(1.0); // intercept
            x.push(xi);
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            let noise = (rng as f64 / u64::MAX as f64 - 0.5) * 0.5;
            y.push(2.0 + 3.0 * xi + noise);
        }
        let prior_mean = vec![0.0; d];
        let mut prior_prec = vec![0.0; d * d];
        prior_prec[0] = 0.01; prior_prec[3] = 0.01; // vague prior

        let res = bayesian_linear_regression(&x, &y, n, d, &prior_mean, &prior_prec, 1.0, 1.0);
        assert!((res.beta_mean[0] - 2.0).abs() < 0.5, "Intercept={}", res.beta_mean[0]);
        assert!((res.beta_mean[1] - 3.0).abs() < 0.5, "Slope={}", res.beta_mean[1]);
        assert!(res.sigma2_mean > 0.0, "σ² should be positive");
    }

    // ── ESS ─────────────────────────────────────────────────────────────

    #[test]
    fn ess_iid_is_n() {
        // IID samples → ESS ≈ n
        let samples: Vec<f64> = (0..100).map(|i| (i as f64 * 1.23456).sin()).collect();
        let ess = effective_sample_size(&samples);
        assert!(ess > 50.0, "ESS={ess} should be large for ~IID samples");
    }

    // ── R-hat ───────────────────────────────────────────────────────────

    #[test]
    fn rhat_converged_chains() {
        // Two chains sampling from same distribution
        let c1: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin()).collect();
        let c2: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1 + 0.5).sin()).collect();
        let rh = r_hat(&[&c1, &c2]);
        assert!(rh < 1.5, "R-hat={rh} should be ~1.0 for converged chains");
    }

    // ── Regression: Xoshiro256 RNG quality in MH ───────────────────────
    // With LCG, variance estimation was biased due to lattice structure.
    // Xoshiro256 passes BigCrush → correct moments on standard targets.
    #[test]
    fn mh_variance_recovery_regression() {
        // Target: N(0, 1). Chain variance should be ~1.0.
        let log_target = |x: &[f64]| -0.5 * x[0] * x[0];
        let chain = metropolis_hastings(&log_target, &[0.0], 1.0, 10000, 2000, 12345);
        let samples: Vec<f64> = chain.samples.iter().map(|s| s[0]).collect();
        let n = samples.len() as f64;
        let mean = samples.iter().sum::<f64>() / n;
        let var = samples.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n - 1.0);

        assert!((mean).abs() < 0.15, "Mean should be ~0, got {mean}");
        assert!((var - 1.0).abs() < 0.3, "Variance should be ~1.0, got {var}");
    }

    #[test]
    fn mh_different_seeds_give_different_chains() {
        let log_target = |x: &[f64]| -0.5 * (x[0] - 5.0).powi(2);
        let c1 = metropolis_hastings(&log_target, &[0.0], 1.0, 100, 10, 42);
        let c2 = metropolis_hastings(&log_target, &[0.0], 1.0, 100, 10, 99);
        // Different seeds should produce different samples
        let different = c1.samples.iter().zip(c2.samples.iter())
            .any(|(a, b)| (a[0] - b[0]).abs() > 1e-10);
        assert!(different, "Different seeds should produce different chains");
    }
}
