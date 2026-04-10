# F34 Bayesian Methods — Gold Standard Implementations

Created: 2026-04-01
By: scout
Session: Day 1 tambear-math expedition

---

## Purpose

Pre-load gold standard implementations for Family 34 (Bayesian Methods).
Primary oracles: `PyMC` (Python), `Stan` via `cmdstanpy`/`rstan` (R), `numpyro` (JAX),
`arviz` (diagnostics), R `brms` (hierarchical), R `loo` (model comparison).

Central architectural question: **MCMC is fundamentally sequential — what does tambear provide?**

Answer (from navigator F34 sharing surface):
- **Likelihood evaluation** = `accumulate(All, LogLik(params), Add)` — GPU-parallelized over data
- **Gradient of log-likelihood** = same accumulate with gradient expr (F10/F05 GradientOracle)
- **Markov chain step** = CPU-side (sequential by definition)
- **Multiple chains** = embarrassingly parallel GPU streams (N chains × same kernel)
- **VI (ADVI)** = Adam on ELBO (F05 optimizer) + reparameterized GradientOracle

---

## Section 1: Gold Standard Libraries — Reference Map

| Method | Python (primary) | R (primary) | Validation role |
|--------|-----------------|-------------|----------------|
| Full MCMC (NUTS) | `pymc` | `rstan`, `brms` | Chain R-hat, ESS |
| Variational (ADVI) | `pymc` (`pm.fit`), `pyro` | `rstan` (vb) | ELBO, posterior mean |
| MAP / Laplace | `pymc` (`pm.find_MAP`) | `rstan` (optimize) | Point estimate |
| HMC diagnostics | `arviz` | `posterior` (R pkg) | R-hat < 1.01, ESS |
| Conjugate closed-form | manual formulas | manual formulas | Exact algebra |
| Hierarchical | `pymc` | `brms`, `lme4` (compare) | Pooling shrinkage |
| Model comparison | `arviz` (WAIC, LOO) | `loo` (R pkg) | ELPD, p_waic |
| Particle filter (SMC) | `pymc` (SMC), `pyabc` | `pomp` (R) | ESS trajectory |

---

## Section 2: PyMC — Primary Python Oracle

### Installation and version

```python
import pymc as pm
import arviz as az
import numpy as np

# CRITICAL: PyMC v4+ (PyMC5 = same API) uses PyTensor, NOT Theano/Aesara.
# pm.__version__ should be >= 4.0.0
# PyMC3 (< 4.0) is DEAD — uses Theano, different API entirely.
print(pm.__version__)   # Expect "5.x.x" as of 2025+

# If you see `import theano` in example code: that's PyMC3. Discard it.
# PyMC5 uses `import pytensor` — that's the current API.
```

### Anatomy of a PyMC model

```python
import pymc as pm
import numpy as np

# Synthetic data:
np.random.seed(42)
N = 100
X = np.random.normal(0, 1, N)
true_alpha, true_beta, true_sigma = 1.5, 2.3, 0.5
y = true_alpha + true_beta * X + np.random.normal(0, true_sigma, N)

with pm.Model() as linear_model:
    # Priors — every random variable declared inside `with pm.Model()`:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta  = pm.Normal('beta',  mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)   # positive-only

    # Deterministic transform (NOT a random variable — no prior):
    mu = pm.Deterministic('mu', alpha + beta * X)   # or just: mu = alpha + beta * X

    # Likelihood — observed= marks this as data, not latent:
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)
```

### MAP (Laplace point estimate)

```python
with linear_model:
    map_estimate = pm.find_MAP()

# map_estimate is a dict: {'alpha': float, 'beta': float, 'sigma': float}
print(map_estimate['alpha'])   # ≈ 1.5
print(map_estimate['beta'])    # ≈ 2.3
print(map_estimate['sigma'])   # ≈ 0.5

# Validation target: map_estimate values match OLS estimates when using
# flat/weak priors and N is large. For Normal-Normal conjugate model,
# MAP is identical to MLE (prior doesn't dominate).
```

### ADVI (Variational Inference)

```python
with linear_model:
    # Mean-field ADVI (default): fully factored Gaussian approximation
    approx = pm.fit(n=30000, method='advi')
    # approx.hist: ELBO history (convergence diagnostic)

    # IMPORTANT: pm.fit returns an Approximation object, NOT InferenceData
    # To get samples from the variational posterior:
    vi_trace = approx.sample(1000)
    # vi_trace IS InferenceData (same format as MCMC trace)

# ELBO should increase (become less negative) and plateau:
import matplotlib
approx.plot_convergence()    # plots ELBO vs iteration

# Mean estimates from VI:
vi_summary = az.summary(vi_trace)
# Check: vi_summary['mean']['alpha'] ≈ 1.5, but with variance from mean-field approx

# Full-rank ADVI (more accurate, slower):
approx_fr = pm.fit(n=30000, method='fullrank_advi')
```

### NUTS (No-U-Turn Sampler) — Full MCMC

```python
with linear_model:
    # Standard MCMC call:
    idata = pm.sample(
        draws=2000,      # samples per chain after warmup
        tune=1000,       # warmup (discarded) — Stan calls this warm-up
        chains=4,        # number of independent chains
        cores=4,         # parallel chains (uses multiprocessing)
        target_accept=0.8,  # default; increase to 0.9 for difficult posteriors
        return_inferencedata=True,  # ALWAYS True in PyMC5 (default)
        progressbar=True,
    )

# idata is an arviz.InferenceData object — NOT raw numpy arrays
# Structure: idata.posterior, idata.sample_stats, idata.observed_data, etc.
```

### Critical: InferenceData structure

```python
# idata.posterior: xarray.Dataset with dims (chain, draw, *param_dims)
idata.posterior['alpha']          # shape (4 chains, 2000 draws)
idata.posterior['alpha'].values   # numpy array shape (4, 2000)

# Flatten across chains:
alpha_samples = idata.posterior['alpha'].values.reshape(-1)  # shape (8000,)

# DO NOT do: idata['alpha'] — that fails. Always go through idata.posterior

# Sample stats (NUTS-specific diagnostics):
idata.sample_stats['diverging']   # bool array — any divergences = problem
idata.sample_stats['tree_depth']  # NUTS tree depth (should not be max_treedepth=10)
idata.sample_stats['step_size']   # adapted step size ε
idata.sample_stats['energy']      # Hamiltonian energy (for BFMI diagnostic)
```

### arviz diagnostics — the convergence checklist

```python
import arviz as az

# Full summary table:
summary = az.summary(idata)
# Columns: mean, sd, hdi_3%, hdi_97%, mcse_mean, mcse_sd, ess_bulk, ess_tail, r_hat

# CONVERGENCE CRITERIA (standard cutoffs from Vehtari et al. 2021):
# R-hat < 1.01: chains are mixing well (stricter than old < 1.1 rule)
# ESS_bulk > 400: enough effective samples for central estimates
# ESS_tail > 400: enough effective samples for tail quantiles (important for CIs)
# Zero divergences: no divergences in sample_stats['diverging']

# Individual statistics:
r_hat = az.rhat(idata)           # per-parameter R-hat
ess   = az.ess(idata)            # per-parameter ESS (bulk and tail)

print(summary[['mean', 'sd', 'r_hat', 'ess_bulk']])

# Trace plots (visual convergence):
az.plot_trace(idata)             # chains side by side; should look like "caterpillars"
az.plot_posterior(idata)         # posterior histograms with HDI
az.plot_pair(idata, divergences=True)  # pairwise parameter scatterplots

# BFMI (Bayesian Fraction of Missing Information) — energy diagnostic:
bfmi = az.bfmi(idata)
# BFMI < 0.3 suggests poor HMC geometry; try non-centered parameterization
```

### Validation targets for linear regression

```python
# With N=100, true_alpha=1.5, true_beta=2.3, true_sigma=0.5, seed=42:
expected = {
    'alpha':  {'mean_approx': 1.5,  'sd_approx': 0.05},
    'beta':   {'mean_approx': 2.3,  'sd_approx': 0.05},
    'sigma':  {'mean_approx': 0.5,  'sd_approx': 0.04},
}
# R-hat should be < 1.01 for all three parameters.
# ESS_bulk should be > 800 (of 8000 total draws with 4 chains × 2000).
```

---

## Section 3: Stan — The Reference Compiler Backend

Stan compiles models to C++ via `cmdstan`. Two Python interfaces: `cmdstanpy` (official)
and `pystan` (older). R interface: `rstan` and `cmdstanr`.

### Installation (Python)

```python
import cmdstanpy
# One-time install of CmdStan binary:
cmdstanpy.install_cmdstan()

from cmdstanpy import CmdStanModel
```

### Stan model: Bayesian linear regression

```stan
// File: linear_regression.stan
data {
    int<lower=0> N;         // number of observations
    vector[N] X;            // predictor
    vector[N] y;            // outcome
}
parameters {
    real alpha;
    real beta;
    real<lower=0> sigma;    // constrained to positive (HalfNormal implicitly)
}
model {
    // Priors (same as PyMC model above):
    alpha ~ normal(0, 10);
    beta  ~ normal(0, 10);
    sigma ~ half_normal(0, 1);

    // Likelihood (vectorized — this is one line in Stan):
    y ~ normal(alpha + beta * X, sigma);
}
```

```python
# Compile and sample:
model = CmdStanModel(stan_file='linear_regression.stan')

data_dict = {'N': N, 'X': X, 'y': y}
fit = model.sample(
    data=data_dict,
    chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    adapt_delta=0.8,     # target_accept equivalent
)

# Results:
fit.summary()             # pandas DataFrame with mean, sd, R-hat, ESS
fit.stan_variable('alpha')  # numpy array shape (8000,) = 4 chains × 2000 draws

# Or load into arviz for same diagnostic tools as PyMC:
import arviz as az
idata = az.from_cmdstanpy(fit)
az.summary(idata)
```

### Stan via R (`rstan`)

```r
library(rstan)
options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

stan_code <- "
data {
  int<lower=0> N;
  vector[N] X;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta  ~ normal(0, 10);
  sigma ~ half_normal(0, 1);
  y ~ normal(alpha + beta * X, sigma);
}
"

fit <- stan(
  model_code = stan_code,
  data = list(N=N, X=X, y=y),
  chains = 4,
  warmup = 1000,
  iter = 3000,     # total = warmup + sampling
  seed = 42,
)

# Extract:
print(fit, pars=c('alpha','beta','sigma'))  # summary table with R-hat
alpha_draws <- extract(fit, pars='alpha')$alpha  # numeric vector (8000,)

# Diagnostics:
check_hmc_diagnostics(fit)   # prints divergence/treedepth/BFMI warnings
```

### Stan variational inference

```python
# Stan's ADVI (mean-field VI):
fit_vb = model.variational(
    data=data_dict,
    algorithm='meanfield',   # or 'fullrank'
    iter=10000,
    output_samples=1000,
)
fit_vb.variational_params_dict   # {'alpha': mean_float, 'sigma_alpha': std_float, ...}
fit_vb.variational_sample        # samples from VI posterior

# In R:
fit_vb <- vb(stan_model, data=data_list, algorithm='meanfield', iter=10000)
print(fit_vb)
```

### Stan MAP (optimization)

```python
fit_map = model.optimize(data=data_dict)
fit_map.optimized_params_dict   # {'alpha': float, 'beta': float, 'sigma': float}
# equivalent to pm.find_MAP() in PyMC
```

---

## Section 4: MCMC Algorithms — Mechanics and Gold Standards

### 4.1 Metropolis-Hastings

Simplest MCMC. Propose θ' ~ q(·|θ), accept with probability min(1, p(θ'|x)/p(θ|x) · q(θ|θ')/q(θ'|θ)).
For symmetric proposals: Metropolis algorithm (q cancels).

```python
import numpy as np
from scipy import stats

def metropolis_hastings(log_posterior, theta_init, n_samples, proposal_std=0.1, seed=42):
    """
    Reference implementation of Metropolis-Hastings MCMC.
    log_posterior: callable theta -> float
    Returns: (n_samples, len(theta_init)) array of accepted samples
    """
    rng = np.random.default_rng(seed)
    theta = np.array(theta_init, dtype=float)
    samples = np.zeros((n_samples, len(theta)))
    n_accepted = 0

    for i in range(n_samples):
        # Symmetric Gaussian proposal:
        theta_prop = theta + rng.normal(0, proposal_std, size=len(theta))

        # Accept/reject:
        log_alpha = log_posterior(theta_prop) - log_posterior(theta)
        if np.log(rng.uniform()) < log_alpha:
            theta = theta_prop
            n_accepted += 1

        samples[i] = theta

    acceptance_rate = n_accepted / n_samples
    return samples, acceptance_rate

# Example: posterior for binomial success rate p given 6 successes in 10 trials
# Prior: p ~ Beta(1, 1) (uniform), Likelihood: Binomial(10, p)
# True posterior: Beta(7, 5)
def log_posterior_binomial(params):
    p = params[0]
    if p <= 0 or p >= 1:
        return -np.inf
    return stats.beta.logpdf(p, 7, 5)  # equivalent: beta(1+6, 1+4)

samples_mh, acc_rate = metropolis_hastings(log_posterior_binomial, [0.5], 50000)
print(f"Acceptance rate: {acc_rate:.3f}")  # Should be ~0.4-0.7 for good mixing
print(f"Posterior mean: {samples_mh[5000:, 0].mean():.4f}")  # Should ≈ 7/12 = 0.583
print(f"True mean: {7/12:.4f}")

# Validation: compare to exact Beta(7,5):
true_dist = stats.beta(7, 5)
print(f"True mean: {true_dist.mean():.4f}")  # 0.5833...
print(f"True std:  {true_dist.std():.4f}")   # 0.1153...
```

Validation targets:
- Posterior mean ≈ 0.583 (exact: 7/12)
- Posterior std ≈ 0.115 (exact: sqrt(7·5 / (12²·13)))
- Acceptance rate ≈ 0.40–0.70 (well-tuned step size)

### 4.2 Gibbs Sampling

Samples each parameter from its full conditional, holding others fixed.
Requires closed-form conditionals — works well for conjugate models.

```python
# Gibbs sampling for Normal-Normal conjugate model:
# Model: y_i ~ N(mu, sigma²=1), prior mu ~ N(mu0=0, tau²=10)
# Posterior: mu | y ~ N(mu_n, sigma_n²)
#   where sigma_n² = 1 / (n/sigma² + 1/tau²)
#         mu_n = sigma_n² * (sum(y)/sigma² + mu0/tau²)

def gibbs_normal_model(y, n_samples=10000, mu0=0, tau2=10, sigma2=1, seed=42):
    """
    Gibbs sampler for mu in Normal-Normal model.
    In this 1-parameter case, it reduces to direct sampling from posterior.
    With multiple parameters, Gibbs cycles through each conditional.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    ybar = y.mean()

    # Conjugate posterior parameters:
    sigma_n2 = 1.0 / (n / sigma2 + 1.0 / tau2)
    mu_n = sigma_n2 * (n * ybar / sigma2 + mu0 / tau2)

    # Direct sampling from N(mu_n, sigma_n2):
    samples = rng.normal(mu_n, np.sqrt(sigma_n2), n_samples)
    return samples, mu_n, sigma_n2

y = np.array([2.1, 1.8, 2.3, 2.0, 1.9])
samples_gibbs, mu_n, sigma_n2 = gibbs_normal_model(y)
print(f"Posterior mean: {mu_n:.4f}")    # ≈ 2.013 (pulled toward 0 slightly)
print(f"Posterior std:  {np.sqrt(sigma_n2):.4f}")
print(f"Sample mean:    {samples_gibbs.mean():.4f}")  # Should ≈ mu_n
```

Gold standard for Gibbs: JAGS (R `rjags`) — the canonical Gibbs sampler.

```r
library(rjags)

jags_model_code <- "
model {
  for(i in 1:N) {
    y[i] ~ dnorm(mu, tau)    # JAGS uses precision tau = 1/sigma^2
  }
  mu ~ dnorm(mu0, tau0)
}
"

data_jags <- list(
  N = 5,
  y = c(2.1, 1.8, 2.3, 2.0, 1.9),
  mu0 = 0,
  tau0 = 1/10,    # precision = 1/tau^2
  tau  = 1/1      # precision = 1/sigma^2
)

jags_fit <- jags.model(textConnection(jags_model_code), data=data_jags, n.chains=4)
update(jags_fit, 1000)   # burn-in
samples_jags <- coda.samples(jags_fit, variable.names=c('mu'), n.iter=5000)
summary(samples_jags)
# Mean should ≈ mu_n from analytic formula
```

### 4.3 HMC (Hamiltonian Monte Carlo)

Gradient-based proposals using Hamiltonian dynamics. Leapfrog integrator.
See navigator F34 for the Affine-scan decomposition of the leapfrog step.

Gold standard: Stan with explicit HMC settings (not NUTS).

```python
# Stan HMC with fixed step count (not adaptive):
# In CmdStanPy, set algorithm='hmc' and engine='static':
fit_hmc = model.sample(
    data=data_dict,
    chains=4,
    iter_warmup=1000,
    iter_sampling=2000,
    # HMC-specific (not NUTS):
    # These are advanced options, default NUTS is preferred
)

# Leapfrog reference implementation:
def leapfrog(theta, r, grad_log_posterior, step_size, n_steps):
    """
    HMC leapfrog integrator.
    theta: position (parameters), shape (d,)
    r: momentum, shape (d,)
    Returns: (theta_new, r_new) after n_steps leapfrog steps
    """
    # Half-step momentum:
    _, grad = grad_log_posterior(theta)
    r = r + (step_size / 2) * grad

    for _ in range(n_steps - 1):
        theta = theta + step_size * r        # full position step
        _, grad = grad_log_posterior(theta)
        r = r + step_size * grad             # full momentum step

    # Final full position step + half momentum step:
    theta = theta + step_size * r
    _, grad = grad_log_posterior(theta)
    r = r + (step_size / 2) * grad

    return theta, r
```

### 4.4 NUTS (No-U-Turn Sampler)

Adaptive HMC — automatically determines leapfrog path length via tree doubling.
Gold standards: Stan (reference implementation), PyMC (uses PyTensor backend), numpyro (JAX).

```python
# numpyro — JAX-based NUTS, fastest for pure numerical work:
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import jax
import jax.numpy as jnp

def model_numpyro(X, y=None):
    alpha = numpyro.sample('alpha', dist.Normal(0, 10))
    beta  = numpyro.sample('beta',  dist.Normal(0, 10))
    sigma = numpyro.sample('sigma', dist.HalfNormal(1))
    mu = alpha + beta * X
    with numpyro.plate('obs', len(X)):
        numpyro.sample('y', dist.Normal(mu, sigma), obs=y)

nuts_kernel = NUTS(model_numpyro, target_accept_prob=0.8)
mcmc = MCMC(nuts_kernel, num_warmup=1000, num_samples=2000, num_chains=4)
mcmc.run(jax.random.PRNGKey(42), X=X, y=y)

mcmc.print_summary()  # mean, std, 5%, 95%, n_eff, r_hat per parameter

# Get samples:
samples_np = mcmc.get_samples()      # dict: param_name -> jax array (n_samples,)
alpha_samples_np = np.array(samples_np['alpha'])  # shape (8000,)
```

Validation: PyMC and numpyro NUTS should agree on posterior means within 0.01
for well-specified models with N=100 and 8000 post-warmup draws.

### 4.5 Sequential Monte Carlo (SMC / Particle Filter)

SMC runs a sequence of annealed distributions from prior to posterior.
More robust than MCMC for multimodal posteriors; parallelizable (K particles).

```python
# PyMC SMC:
with pm.Model() as model_smc:
    alpha = pm.Normal('alpha', mu=0, sigma=10)
    beta  = pm.Normal('beta',  mu=0, sigma=10)
    sigma = pm.HalfNormal('sigma', sigma=1)
    mu = alpha + beta * X
    y_obs = pm.Normal('y_obs', mu=mu, sigma=sigma, observed=y)

    # SMC-NUTS (default): resamples particles using NUTS kernel
    idata_smc = pm.sample_smc(
        draws=2000,      # particles
        chains=2,
        kernel=pm.smc.kernels.MH,   # or NUTS
        progressbar=True,
    )

# Diagnostics:
# idata_smc.sample_stats contains 'log_marginal_likelihood'
# = log p(y) estimate (model evidence) — useful for Bayes factors
log_ml = idata_smc.sample_stats['log_marginal_likelihood'].mean()
print(f"Log marginal likelihood estimate: {log_ml:.3f}")
```

```python
# pyabc — Approximate Bayesian Computation (likelihood-free):
import pyabc

# For models where likelihood is intractable but simulation is easy:
def model_abc(params):
    """Simulate from model given params dict."""
    rng_local = np.random.default_rng()
    return {"y": rng_local.normal(params['mu'], params['sigma'], 50)}

def distance_abc(x, x0):
    """Summary statistic distance."""
    return abs(np.mean(x['y']) - np.mean(x0['y']))

prior_abc = pyabc.Distribution(
    mu=pyabc.RV("norm", 0, 10),
    sigma=pyabc.RV("halfnorm", scale=1),
)

abc = pyabc.ABCSMC(model_abc, prior_abc, distance_abc)
# abc.new("sqlite:///results.db", {"y": observed_data})
# history = abc.run(minimum_epsilon=0.1, max_nr_populations=5)
```

---

## Section 5: Variational Inference — VI-Only Deep Dive

VI finds the closest member of a parametric family to the true posterior,
by maximizing the ELBO = E_q[log p(x,θ)] - E_q[log q(θ)] = E_q[log p(x|θ)] - KL[q||p].

### ADVI in PyMC (primary target)

```python
with linear_model:
    # Mean-field ADVI (default):
    advi = pm.ADVI()
    approx_mf = advi.fit(n=30000, obj_optimizer=pm.adam(learning_rate=0.01))

    # Track ELBO:
    print(approx_mf.hist[:10])   # ELBO values at each step (negative, should increase)
    print(approx_mf.hist[-1])    # Final ELBO

    # Full-rank ADVI (correlated posterior):
    approx_fr = pm.FullRankADVI().fit(n=30000)

# Convert to InferenceData for arviz:
trace_vi = approx_mf.sample(2000)
summary_vi = az.summary(trace_vi)
```

### pyro / numpyro SVI (JAX backend)

```python
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO
from numpyro.infer.autoguide import AutoNormal
import optax

# AutoNormal guide = mean-field VI with Normal approximation:
guide = AutoNormal(model_numpyro)
optimizer = numpyro.optim.Adam(0.01)
svi = SVI(model_numpyro, guide, optimizer, loss=Trace_ELBO())

svi_state = svi.init(jax.random.PRNGKey(0), X=X, y=y)

# Run VI:
for i in range(10000):
    svi_state, loss = svi.update(svi_state, X=X, y=y)
    if i % 1000 == 0:
        print(f"Step {i}, ELBO: {-loss:.3f}")  # loss = -ELBO

# Get posterior samples:
params = svi.get_params(svi_state)
posterior_samples = guide.sample_posterior(
    jax.random.PRNGKey(1), params, sample_shape=(2000,)
)
```

### Validation: ELBO convergence criterion

```python
# ELBO should:
# 1. Increase monotonically (on average — stochastic, so noisy)
# 2. Plateau (converge) within n iterations
# 3. Final value: compare between mean-field and full-rank VI
#    Full-rank ELBO >= mean-field ELBO (looser lower bound for full-rank)

# Convergence check:
def check_vi_convergence(elbo_history, window=500, tol=1e-3):
    """Check if VI converged: last window ELBO changes < tol (relative)."""
    last = np.array(elbo_history[-window:])
    rel_change = np.abs(np.diff(last)).mean() / (np.abs(last.mean()) + 1e-10)
    return rel_change < tol

# Warning: VI posterior means are often good, but VI UNDERESTIMATES uncertainty.
# Mean-field VI assumes independence — correlations are lost.
# For fintek: VI means are good point estimates; VIs are NOT reliable for CIs.
```

---

## Section 6: Conjugate Models — Closed-Form Posterior

Conjugate models have exact analytical posteriors. No MCMC needed.
These ARE tambear accumulate primitives (Kingdom A, closed-form).

### 6.1 Beta-Binomial (Bernoulli success rate)

```python
# Model: X ~ Binomial(n, p),  p ~ Beta(a0, b0)
# Posterior: p | X ~ Beta(a0 + X, b0 + n - X)
# = exact Bayesian update by adding sufficient statistics

def beta_binomial_posterior(successes, trials, prior_a=1.0, prior_b=1.0):
    """
    Conjugate Bayesian update for binomial success rate.
    prior_a, prior_b: Beta prior hyperparameters (default = uniform = Beta(1,1))
    Returns posterior Beta parameters.
    """
    post_a = prior_a + successes
    post_b = prior_b + (trials - successes)
    return post_a, post_b

# Example: 14 successes in 20 trials, uniform prior
a, b = beta_binomial_posterior(14, 20)
print(f"Posterior: Beta({a}, {b})")  # Beta(15, 7)
print(f"Posterior mean: {a/(a+b):.4f}")   # 15/22 = 0.6818
print(f"Posterior std:  {np.sqrt(a*b/((a+b)**2*(a+b+1))):.4f}")  # 0.0974

from scipy.stats import beta
post_dist = beta(a, b)
ci_95 = post_dist.ppf([0.025, 0.975])
print(f"95% credible interval: [{ci_95[0]:.4f}, {ci_95[1]:.4f}]")

# Tambear accumulate form:
# post_a = prior_a + accumulate(All, Indicator(y==1), Add)
# post_b = prior_b + accumulate(All, Indicator(y==0), Add)
# = two scalar accumulators over the data
```

### 6.2 Normal-Normal (known variance)

```python
# Model: y_i ~ N(mu, sigma²=known), mu ~ N(mu0, tau²)
# Posterior: mu | y ~ N(mu_n, sigma_n²)
# where:
#   1/sigma_n² = 1/tau² + n/sigma²          (precision addition)
#   mu_n = sigma_n² * (mu0/tau² + n*ybar/sigma²)  (precision-weighted mean)

def normal_normal_posterior(y, sigma2, mu0=0.0, tau2=1.0):
    """Conjugate Normal-Normal posterior for unknown mean."""
    n = len(y)
    ybar = np.mean(y)

    precision_post = 1.0/tau2 + n/sigma2
    sigma_n2 = 1.0 / precision_post
    mu_n = sigma_n2 * (mu0/tau2 + n*ybar/sigma2)

    return mu_n, sigma_n2

y_data = np.array([2.1, 1.8, 2.3, 2.0, 1.9])
mu_n, sigma_n2 = normal_normal_posterior(y_data, sigma2=1.0)
print(f"Posterior mean: {mu_n:.6f}")   # slightly shrunk from ybar=2.02 toward 0
print(f"Posterior std:  {np.sqrt(sigma_n2):.6f}")

# Tambear:
# n = accumulate(All, 1, Add)
# ybar = accumulate(All, y, Add) / n
# Both are scalar outputs from a single pass over data
```

### 6.3 Dirichlet-Multinomial (categorical proportions)

```python
# Model: counts x ~ Multinomial(n, p), p ~ Dirichlet(alpha)
# Posterior: p | x ~ Dirichlet(alpha + x)
# = add observed counts to prior pseudocounts

def dirichlet_multinomial_posterior(counts, prior_alpha):
    """
    Conjugate Dirichlet-Multinomial posterior.
    counts: observed counts per category, shape (K,)
    prior_alpha: Dirichlet prior, shape (K,) (e.g., all-ones = uniform)
    Returns posterior Dirichlet alpha parameters.
    """
    return np.array(prior_alpha) + np.array(counts)

counts = np.array([10, 5, 15, 20])   # 4 categories
prior = np.ones(4)                    # uniform over simplex
post_alpha = dirichlet_multinomial_posterior(counts, prior)
print(f"Posterior alpha: {post_alpha}")  # [11, 6, 16, 21]

# Posterior mean of p:
post_mean = post_alpha / post_alpha.sum()
print(f"Posterior mean p: {post_mean}")

# Tambear: counts = accumulate(ByCategory, 1, Add) — histogram primitive
```

### 6.4 Inverse-Gamma for variance (Normal-InvGamma)

```python
# Model: y_i ~ N(mu, sigma²=unknown), sigma² ~ InvGamma(a0, b0)
# Posterior: sigma² | y ~ InvGamma(a0 + n/2, b0 + sum((y-mu)²)/2)

from scipy.stats import invgamma

def invgamma_posterior(y, mu, a0=0.001, b0=0.001):
    """Conjugate InvGamma-Normal posterior for unknown variance."""
    n = len(y)
    ss = np.sum((y - mu)**2)  # sum of squared deviations
    a_n = a0 + n / 2.0
    b_n = b0 + ss / 2.0
    return a_n, b_n

a_n, b_n = invgamma_posterior(y_data, mu=2.0)
post_sigma2 = invgamma(a_n, scale=b_n)
print(f"Posterior E[sigma²]: {post_sigma2.mean():.4f}")   # ≈ 0.045 for this data
print(f"Posterior mode: {b_n/(a_n+1):.4f}")               # mode of InvGamma

# Tambear: ss = accumulate(All, (y - mu)^2, Add) — one pass, scalar output
```

---

## Section 7: Hierarchical Models (Partial Pooling)

### 7.1 PyMC hierarchical model

```python
import pymc as pm
import numpy as np

# 8-schools problem (canonical hierarchical model, Rubin 1981):
# y_j | theta_j ~ N(theta_j, sigma_j²)  (known school-level SEs)
# theta_j ~ N(mu, tau²)
# mu ~ N(0, 5²), tau ~ HalfCauchy(5)

y_schools = np.array([28, 8, -3, 7, -1, 1, 18, 12], dtype=float)
sigma_schools = np.array([15, 10, 16, 11, 9, 11, 10, 18], dtype=float)
J = len(y_schools)

with pm.Model() as eight_schools_centered:
    # Hyperpriors:
    mu  = pm.Normal('mu',  mu=0,  sigma=5)
    tau = pm.HalfCauchy('tau', beta=5)

    # School effects (centered parameterization):
    theta = pm.Normal('theta', mu=mu, sigma=tau, shape=J)

    # Likelihood:
    obs = pm.Normal('obs', mu=theta, sigma=sigma_schools, observed=y_schools)

    # This model has DIVERGENCES in NUTS — it's the canonical example for
    # non-centered parameterization:
    idata_centered = pm.sample(1000, tune=1000, chains=4, target_accept=0.95)
    print(f"Divergences: {idata_centered.sample_stats['diverging'].sum().item()}")
```

### 7.2 Non-centered parameterization (CRITICAL trap avoidance)

```python
with pm.Model() as eight_schools_noncentered:
    # Hyperpriors (same):
    mu  = pm.Normal('mu',  mu=0,  sigma=5)
    tau = pm.HalfCauchy('tau', beta=5)

    # Non-centered: theta = mu + tau * theta_raw
    theta_raw = pm.Normal('theta_raw', mu=0, sigma=1, shape=J)
    theta = pm.Deterministic('theta', mu + tau * theta_raw)

    obs = pm.Normal('obs', mu=theta, sigma=sigma_schools, observed=y_schools)

    # This converges WITHOUT divergences:
    idata_nc = pm.sample(1000, tune=1000, chains=4, target_accept=0.9)
    print(f"Divergences: {idata_nc.sample_stats['diverging'].sum().item()}")  # Should be 0

# Validation targets for 8-schools (from Rubin 1981 / Stan manual):
# mu posterior: mean ≈ 8, sd ≈ 5
# tau posterior: mean ≈ 7, sd ≈ 6 (heavy-tailed, slow convergence)
# theta[0] (school A): mean ≈ 11, 95% CI ≈ [-2, 31]
```

**Critical**: centered vs non-centered parameterization can change NUTS efficiency by 100x.
The non-centered form separates the funnel geometry. Always prefer non-centered for
hierarchical models with tau near 0 (weak pooling / small J).

### 7.3 brms (R) — the R hierarchical Bayes reference

`brms` wraps Stan with an lme4-style formula interface.

```r
library(brms)

# Load sleepstudy data (same as lme4 example):
library(lme4)
data(sleepstudy)

# Bayesian mixed effects with brms:
fit_brms <- brm(
  Reaction ~ Days + (Days | Subject),  # same formula as lme4
  data  = sleepstudy,
  family = gaussian(),
  prior = c(
    prior(normal(250, 50), class = Intercept),
    prior(normal(10, 10),  class = b, coef = Days),
    prior(cauchy(0, 10),   class = sd),      # random effect SDs
    prior(cauchy(0, 10),   class = sigma)    # residual SD
  ),
  chains = 4,
  iter   = 2000,
  warmup = 1000,
  cores  = 4,
  seed   = 42,
)

summary(fit_brms)        # Stan-style summary with R-hat
plot(fit_brms)           # trace + posterior plots
pp_check(fit_brms)       # posterior predictive check (key validation)

# Posterior predictive:
preds <- posterior_predict(fit_brms, newdata = sleepstudy[1:5,])
# shape (n_draws, 5)

# Compare to lme4 fixed effects:
library(lme4)
fit_lme4 <- lmer(Reaction ~ Days + (Days | Subject), data=sleepstudy)
fixef(fit_lme4)   # fixed: Intercept ≈ 251.4, Days ≈ 10.5
```

Validation: brms fixed effects should match lme4 fixed effects within 1 se
when using weakly informative priors and N is large.

---

## Section 8: WAIC and LOO-CV (Bayesian Model Comparison)

### 8.1 arviz (Python)

```python
import arviz as az

# After fitting multiple models:
with linear_model:
    idata = pm.sample(2000, tune=1000, chains=4)
    pm.compute_log_likelihood(idata)  # needed for LOO/WAIC

# LOO-CV (leave-one-out cross-validation via PSIS):
loo_result = az.loo(idata, pointwise=True)
print(loo_result)
# loo_result.elpd_loo   = expected log predictive density (higher = better)
# loo_result.p_loo      = effective number of parameters (should be < n/5)
# loo_result.looic      = -2 * elpd_loo (lower = better, like AIC)

# WAIC (widely applicable information criterion):
waic_result = az.waic(idata, pointwise=True)
print(waic_result)
# waic_result.elpd_waic  = ELPD estimate via WAIC
# waic_result.p_waic     = effective parameters
# waic_result.waic       = -2 * elpd_waic

# Compare two models:
compare_result = az.compare({'linear': idata, 'quadratic': idata_quad})
print(compare_result)
# Table: elpd_diff, weight, se, dse — higher weight = preferred

# Pareto k values (LOO reliability):
az.plot_khat(loo_result)   # k < 0.5 = good, 0.5-0.7 = ok, > 0.7 = problematic
```

### 8.2 R `loo` package

```r
library(loo)

# After Stan fit, extract log-likelihood:
log_lik <- extract_log_lik(fit_stan, parameter_name="log_lik", merge_chains=FALSE)
# Must have log_lik[] computed in Stan generated quantities block

# LOO-CV:
loo_result <- loo(log_lik)
print(loo_result)
# elpd_loo, p_loo, looic and SE estimates
# Pareto k warnings if some k > 0.7

# WAIC:
waic_result <- waic(log_lik)
print(waic_result)

# Compare two models:
comp <- loo_compare(loo_model1, loo_model2)
print(comp)
# Row 1 = better model, elpd_diff gives comparison with SE
```

Stan model must include generated quantities block for log-likelihood:

```stan
generated quantities {
    vector[N] log_lik;
    for (i in 1:N) {
        log_lik[i] = normal_lpdf(y[i] | alpha + beta * X[i], sigma);
    }
}
```

---

## Section 9: Key Traps and Pitfalls

### Trap 1: PyMC version confusion

```python
# PyMC3 (dead, Theano backend) — DO NOT USE:
# import pymc3 as pm3   <- Theano dependency, won't install cleanly
# pm3.sample(1000)      <- Different API

# PyMC4 = PyMC5 (same package, PyTensor backend) — CURRENT:
import pymc as pm
pm.sample(...)   # returns InferenceData, NOT MultiTrace

# Symptom: code using pm.trace_to_dataframe() or accessing trace['alpha']
# with dict indexing = PyMC3 code. In PyMC5: idata.posterior['alpha']
```

### Trap 2: idata access patterns

```python
# WRONG: idata['alpha']  <- KeyError
# WRONG: idata.alpha     <- AttributeError
# CORRECT:
idata.posterior['alpha']              # xarray DataArray
idata.posterior['alpha'].values       # numpy array (chains, draws)
idata.posterior['alpha'].values.flatten()  # all draws combined

# WRONG: alpha_mean = idata.posterior['alpha'].mean()  <- scalar xarray
# CORRECT:
alpha_mean = float(idata.posterior['alpha'].mean())
# or:
alpha_mean = idata.posterior['alpha'].values.mean()
```

### Trap 3: Stan centered vs non-centered

```
# Centered hierarchical model:
theta[j] ~ normal(mu, tau)   <- OK when J is large AND tau >> 0
                                 BAD when J is small OR tau near 0

# Non-centered (reparameterized):
theta_raw[j] ~ normal(0, 1)
theta[j] = mu + tau * theta_raw[j]   <- ALWAYS numerically better

# Diagnostic: BFMI < 0.3 or divergences with centered model
# Fix: switch to non-centered
```

### Trap 4: R-hat interpretation

```python
# OLD rule (Gelman & Rubin 1992): R-hat < 1.1 = "converged"
# NEW rule (Vehtari et al. 2021): R-hat < 1.01 = converged
#                                  R-hat > 1.01 = run longer / debug
# arviz.rhat() implements the new rank-normalized R-hat (more reliable)

# ESS: need > 400 for reliable estimates
# ESS_bulk: for mean/median
# ESS_tail: for quantiles / credible intervals
# If ESS_tail < 400: CI estimates are unreliable even if means look OK
```

### Trap 5: pm.sample() default changed

```python
# PyMC v4+: return_inferencedata=True is the DEFAULT
# PyMC3: returned MultiTrace object
# If you see code checking `isinstance(trace, pm.backends.base.MultiTrace)`:
# that's PyMC3 legacy code

# PyMC5 example:
with model:
    idata = pm.sample(1000)   # returns InferenceData
    # NOT: trace = pm.sample(1000)  # <- old name convention still works
```

### Trap 6: MCMC for large N is slow — consider alternatives

```python
# For N > 10,000 and simple models: ADVI converges in <1 minute
# For N > 100,000: use mini-batch VI (pyro/numpyro) or Laplace approx
# Full MCMC on large N: each log-likelihood evaluation = O(N) GPU work
# That's fine for tambear (GPU-parallel likelihood) but chain is still serial

# For N > 1,000,000 with tambear backend:
# Each NUTS step calls GradientOracle (O(N) GPU accumulate) — fast
# But N_steps × N_chains sequential evaluations accumulate:
# 4 chains × 3000 total steps × 100 leapfrog steps = 1.2M oracle calls
# At 1ms each (GPU accumulate over 1M rows): 1200 seconds = 20 minutes
# Rule of thumb: full MCMC at scale costs 10-100x VI
```

### Trap 7: Divergences signal geometry problems

```python
# Check ALWAYS after sampling:
n_divergent = idata.sample_stats['diverging'].sum().item()
if n_divergent > 0:
    print(f"WARNING: {n_divergent} divergent transitions")
    # Fixes (in order of preference):
    # 1. Switch to non-centered parameterization
    # 2. Increase target_accept (0.8 -> 0.9 -> 0.95)
    # 3. Increase tune steps (1000 -> 2000)
    # 4. Reparameterize (log transform for scale params)
    # Any divergences = posterior samples are biased. Do not report results.
```

---

## Section 10: Tambear Decomposition

### Log-posterior accumulate

The central tambear operation for Bayesian likelihood evaluation:

```
// For N iid observations y_i:
log p(x | θ) = Σᵢ log p(yᵢ | θ)
             = accumulate(All, LogLikTerm(θ, yᵢ), Add)

// This is GPU-parallelized over data (embarrassingly parallel)
// The same GradientOracle structure as F05/F10 MLE
```

For Bayesian regression (Normal likelihood):
```
LogLikTerm(θ, yᵢ) = -log(σ) - 0.5 * ((yᵢ - α - β·xᵢ)/σ)²
                   = -log(σ) - 0.5 * rᵢ²/σ²
```

Log-prior terms are per-parameter, not per-observation:
```
log p(θ) = -‖β‖²/(2σ_prior²)   (Normal / Ridge prior)
         = -|β|/b               (Laplace / LASSO prior)
// These are O(d) scalar computations on CPU, not GPU accumulates
```

### Gradient for HMC/VI

```
∂ log p(θ|x)/∂θ = ∂ log p(x|θ)/∂θ + ∂ log p(θ)/∂θ
                     GradientOracle        CPU prior gradient

GradientOracle.value_and_gradient(θ):
  // Same as F05/F10 — one accumulate pass:
  = accumulate(All, (LogLikTerm(θ,yᵢ), GradTerm(θ,yᵢ)), (Add, Add))
  // Returns scalar log-likelihood AND gradient vector in one GPU pass
```

### Multiple chains = parallel GPU streams

```
// K independent chains = K independent GradientOracle calls
// These are embarrassingly parallel:
// Chain 1: θ₁ → oracle(θ₁) → θ₁'
// Chain 2: θ₂ → oracle(θ₂) → θ₂'
// Chain 3: θ₃ → oracle(θ₃) → θ₃'
// Chain 4: θ₄ → oracle(θ₄) → θ₄'
//
// If each oracle call uses 10% GPU utilization:
// 4 chains in parallel = ~40% utilization = no collision
// This is the parallel MCMC pattern for tambear

// But: within a chain, each leapfrog step is SEQUENTIAL (each step
// depends on previous gradient evaluation at new θ position)
```

### Conjugate posteriors as accumulate

```
// Beta-Binomial:
// post_a = prior_a + accumulate(All, IsSuccess(yᵢ), Add)
// post_b = prior_b + accumulate(All, IsFailure(yᵢ), Add)
// = 2 scalar outputs from 1 GPU pass

// Normal-Normal (known sigma²):
// n     = accumulate(All, 1,    Add)
// ybar  = accumulate(All, y_i,  Add) / n
// = 2 outputs from 1 GPU pass
// Then CPU applies conjugate update formula

// Dirichlet-Multinomial:
// post_alpha_k = prior_alpha_k + accumulate(ByCategory, 1, Add)
// = histogram primitive (K outputs from 1 GPU pass)
```

### Tambear primitive map

| Bayesian operation | Primitive | Kingdom |
|-------------------|-----------|---------|
| Log-likelihood evaluation | `accumulate(All, LogLikTerm(θ), Add)` | A |
| Log-likelihood gradient | `accumulate(All, GradTerm(θ), Add)` | A |
| Beta-Binomial update | `accumulate(All, IsSuccess/IsFailure, Add)` | A |
| Normal-Normal update | `accumulate(All, y/1, Add)` (mean + count) | A |
| Dirichlet-Multinomial | `accumulate(ByCategory, 1, Add)` | A |
| VI ELBO (reparameterized) | GradientOracle + KL closed-form | A + CPU |
| HMC leapfrog | GradientOracle × L_steps (sequential) | C |
| NUTS tree doubling | GradientOracle × tree_depth (sequential) | C |
| Multiple MCMC chains | K × GradientOracle (parallel streams) | A (parallel) |
| Laplace MAP | L-BFGS on combined oracle (F05) | C |
| Laplace Hessian (diagonal) | `accumulate(All, grad²ᵢ, Add)` | A |
| LOO-CV log scores | `accumulate(All, log p(yᵢ|θ̂), Add)` | A |

---

## Section 11: Validation Datasets

### 11.1 8-schools (canonical hierarchical)

```python
import numpy as np
y_schools     = np.array([28., 8., -3., 7., -1., 1., 18., 12.])
sigma_schools = np.array([15., 10., 16., 11., 9., 11., 10., 18.])
# Source: Rubin (1981), Bayesian Data Analysis (Gelman et al.)
# Expected (from Stan reference):
# mu posterior:  mean ≈ 7.9, sd ≈ 5.1
# tau posterior: mean ≈ 6.5, sd ≈ 5.5 (skewed right)
# theta[0] (school A): mean ≈ 11.5, sd ≈ 8.7
```

### 11.2 Radon contamination (standard hierarchical)

```python
import pymc as pm

# Available via PyMC docs datasets:
# counties, county index, radon measurements, floor indicator
# 919 measurements across 85 Minnesota counties
# Standard validation: county-level random intercepts, floor as fixed effect
# Expected partial pooling: county means shrunk toward grand mean
```

### 11.3 Simple binomial (closed-form validation)

```python
# 6 successes in 10 trials, Beta(1,1) prior:
# Exact posterior: Beta(7, 5)
# Exact mean: 7/12 = 0.5833...
# Exact std:  sqrt(7*5 / (12^2 * 13)) = sqrt(35/1872) = 0.1368...
# 95% CI: [0.2840, 0.8483] from Beta(7,5).ppf([0.025, 0.975])
from scipy.stats import beta
d = beta(7, 5)
print(d.mean())            # 0.5833...
print(d.std())             # 0.1368...
print(d.ppf([0.025, 0.975]))  # [0.2840, 0.8483]
# All MCMC / VI / Laplace results on this problem should match within tolerance
```

### 11.4 Stochastic volatility (advanced, Kingdom C)

```python
# S&P 500 returns or synthetic log-returns:
# y_t ~ N(0, exp(h_t))   (returns are Normal with time-varying volatility)
# h_t ~ N(rho * h_{t-1}, sigma²)  (log-volatility is AR(1))
# mu = mean log-volatility, rho = persistence, sigma = vol-of-vol
# This is Kingdom C: the latent h_t process requires FFBS or HMC
# Standard validation in Stan/PyMC documentation
```

---

## Section 12: Quick-Reference Checklist

### Before running MCMC:

- [ ] Model specified with explicit priors (not just defaults)
- [ ] Priors inspected with `pm.sample_prior_predictive()`
- [ ] Non-centered parameterization for hierarchical models
- [ ] target_accept ≥ 0.8 (increase to 0.9 for difficult models)
- [ ] At least 4 chains, tune ≥ 1000

### After sampling:

- [ ] Zero divergences: `idata.sample_stats['diverging'].sum().item() == 0`
- [ ] R-hat < 1.01 for ALL parameters: `az.summary(idata)['r_hat'].max() < 1.01`
- [ ] ESS_bulk > 400 for ALL parameters
- [ ] ESS_tail > 400 for ALL parameters (especially for CI reporting)
- [ ] Trace plots look like "caterpillars" (good mixing)
- [ ] Posterior predictive check: `pm.sample_posterior_predictive(idata)` + `az.plot_ppc()`

### For tambear validation:

- [ ] Log-likelihood accumulate matches PyMC log-likelihood evaluation
- [ ] MAP estimate matches `pm.find_MAP()` within 1e-4
- [ ] Laplace CI (diagonal Hessian) approximately matches MCMC CI for unimodal posteriors
- [ ] VI ELBO matches PyMC ADVI final ELBO within 1% (relative)
- [ ] Conjugate posteriors match exact formulas to machine precision
