"""
Gold Standard Oracle: Family 34 — Bayesian Methods
Generates exact expected values for tambear validation.

Tests:
  - Bayesian linear regression (conjugate Normal-InverseGamma posterior)
  - ESS (effective sample size for known AR(1) autocorrelation)
  - R-hat (potential scale reduction factor)
  - MCMC posterior moments vs known target
"""

import json
import numpy as np
from scipy import linalg

results = {}

# ===============================================================
# 1. Bayesian Linear Regression — Conjugate Posterior
# ===============================================================
# Model: y = X*beta + eps, eps ~ N(0, sigma^2)
# Prior: beta|sigma^2 ~ N(beta0, sigma^2 * Lambda0^{-1})
#        sigma^2 ~ InvGamma(alpha0, beta0_ig)
# Posterior: beta|sigma^2,y ~ N(beta_n, sigma^2 * Lambda_n^{-1})
#            sigma^2|y ~ InvGamma(alpha_n, beta_n_ig)

np.random.seed(42)
n = 20
d = 2  # intercept + slope

# Design matrix: [1, x_i]
x_vals = np.linspace(0, 1, n)
X = np.column_stack([np.ones(n), x_vals])

# True: y = 2 + 3x + noise
true_beta = np.array([2.0, 3.0])
noise = 0.3 * np.random.randn(n)
y = X @ true_beta + noise

# Prior: vague
beta0 = np.array([0.0, 0.0])
Lambda0 = np.diag([0.01, 0.01])
alpha0 = 1.0
beta0_ig = 1.0

# Posterior precision
Lambda_n = Lambda0 + X.T @ X
Lambda_n_inv = linalg.inv(Lambda_n)

# Posterior mean
rhs = Lambda0 @ beta0 + X.T @ y
beta_n = linalg.solve(Lambda_n, rhs)

# Posterior sigma^2
alpha_n = alpha0 + n / 2.0
yty = y @ y
b0_L0_b0 = beta0 @ Lambda0 @ beta0
bn_Ln_bn = beta_n @ Lambda_n @ beta_n
beta_n_ig = beta0_ig + 0.5 * (yty + b0_L0_b0 - bn_Ln_bn)
sigma2_mean = beta_n_ig / (alpha_n - 1.0)

# Posterior covariance of beta
beta_cov = sigma2_mean * Lambda_n_inv

results["bayes_linear"] = {
    "x": X.flatten().tolist(),
    "y": y.tolist(),
    "n": n, "d": d,
    "prior_mean": beta0.tolist(),
    "prior_precision": Lambda0.flatten().tolist(),
    "alpha0": alpha0,
    "beta0_ig": beta0_ig,
    "beta_mean": beta_n.tolist(),
    "beta_cov": beta_cov.flatten().tolist(),
    "sigma2_mean": float(sigma2_mean),
    "alpha_post": float(alpha_n),
    "beta_post": float(beta_n_ig),
}
print("Bayesian linear: beta_mean=%s" % beta_n)
print("  sigma2_mean=%.6f, alpha_n=%.1f, beta_n=%.6f" % (sigma2_mean, alpha_n, beta_n_ig))

# ===============================================================
# 2. ESS: known AR(1) chain
# ===============================================================
# For AR(1) with autocorrelation rho at lag 1:
# Theoretical ESS = n * (1-rho) / (1+rho)

np.random.seed(55)
n_ess = 1000
rho = 0.9
chain_ar1 = [0.0]
for i in range(1, n_ess):
    chain_ar1.append(rho * chain_ar1[-1] + np.sqrt(1 - rho**2) * np.random.randn())
chain_ar1 = np.array(chain_ar1)

# Theoretical ESS
ess_theoretical = n_ess * (1 - rho) / (1 + rho)

# Empirical ESS via autocorrelation (same formula as tambear)
mean_c = chain_ar1.mean()
var_c = np.mean((chain_ar1 - mean_c)**2)
max_lag = n_ess // 2
rhos = []
for lag in range(1, max_lag + 1):
    rho_lag = np.mean((chain_ar1[:-lag] - mean_c) * (chain_ar1[lag:] - mean_c)) / var_c
    rhos.append(rho_lag)

# Initial positive sequence
sum_rho = 0.0
k = 0
pair_sums = []
while 2*k + 1 < len(rhos):
    ps = rhos[2*k] + rhos[2*k + 1]
    if ps <= 0:
        break
    pair_sums.append(ps)
    k += 1
# Monotone constraint
for i in range(1, len(pair_sums)):
    if pair_sums[i] > pair_sums[i-1]:
        pair_sums[i] = pair_sums[i-1]
sum_rho = sum(pair_sums)
if 2*k < len(rhos) and rhos[2*k] > 0:
    sum_rho += rhos[2*k]
tau = 1.0 + 2.0 * sum_rho
ess_empirical = n_ess / max(tau, 1.0)

results["ess_ar1"] = {
    "chain": chain_ar1.tolist(),
    "n": n_ess,
    "rho": rho,
    "ess_theoretical": float(ess_theoretical),
    "ess_empirical": float(ess_empirical),
}
print("ESS AR(1) rho=%.1f: theoretical=%.1f, empirical=%.1f" % (rho, ess_theoretical, ess_empirical))

# ===============================================================
# 3. R-hat: converged vs diverged chains
# ===============================================================

np.random.seed(77)

# Converged: two chains from N(0, 1)
c1_conv = np.random.randn(200)
c2_conv = np.random.randn(200)

# R-hat calculation
def compute_rhat(chains):
    m = len(chains)
    n = len(chains[0])
    chain_means = [np.mean(c) for c in chains]
    overall_mean = np.mean(chain_means)
    B = n / (m - 1) * sum((cm - overall_mean)**2 for cm in chain_means)
    W = np.mean([np.var(c, ddof=1) for c in chains])
    var_hat = (n - 1) / n * W + B / n
    return np.sqrt(var_hat / W) if W > 0 else float('inf')

rhat_conv = compute_rhat([c1_conv, c2_conv])

# Diverged: two chains from different distributions
c1_div = np.random.randn(200)
c2_div = np.random.randn(200) + 5.0  # shifted by 5

rhat_div = compute_rhat([c1_div, c2_div])

results["rhat"] = {
    "converged_c1": c1_conv.tolist(),
    "converged_c2": c2_conv.tolist(),
    "rhat_converged": float(rhat_conv),
    "diverged_c1": c1_div.tolist(),
    "diverged_c2": c2_div.tolist(),
    "rhat_diverged": float(rhat_div),
}
print("R-hat converged=%.4f, diverged=%.4f" % (rhat_conv, rhat_div))

# ===============================================================
# 4. ESS: IID samples
# ===============================================================

np.random.seed(33)
iid_samples = np.random.randn(500)
mean_iid = iid_samples.mean()
var_iid = np.mean((iid_samples - mean_iid)**2)

rhos_iid = []
for lag in range(1, 251):
    r = np.mean((iid_samples[:-lag] - mean_iid) * (iid_samples[lag:] - mean_iid)) / var_iid
    rhos_iid.append(r)

# For IID: ESS should be close to n
k_iid = 0
pair_sums_iid = []
while 2*k_iid + 1 < len(rhos_iid):
    ps = rhos_iid[2*k_iid] + rhos_iid[2*k_iid + 1]
    if ps <= 0:
        break
    pair_sums_iid.append(ps)
    k_iid += 1
for i in range(1, len(pair_sums_iid)):
    if pair_sums_iid[i] > pair_sums_iid[i-1]:
        pair_sums_iid[i] = pair_sums_iid[i-1]
sum_rho_iid = sum(pair_sums_iid)
if 2*k_iid < len(rhos_iid) and rhos_iid[2*k_iid] > 0:
    sum_rho_iid += rhos_iid[2*k_iid]
tau_iid = 1.0 + 2.0 * sum_rho_iid
ess_iid = 500 / max(tau_iid, 1.0)

results["ess_iid"] = {
    "n": 500,
    "ess": float(ess_iid),
}
print("ESS IID (n=500): %.1f" % ess_iid)

# ===============================================================
# Save
# ===============================================================

with open("research/gold_standard/family_34_expected.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved family_34_expected.json")
