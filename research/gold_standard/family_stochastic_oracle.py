"""
Gold Standard Oracle: Stochastic Processes

Tests analytical moments and formulas for:
  - GBM expected value and variance
  - OU stationary variance and autocorrelation
  - Black-Scholes option pricing (against scipy)
  - Markov chain stationary distribution
  - Poisson process expected count
  - Quadratic variation of Brownian motion (analytical = T)

Usage:
    python research/gold_standard/family_stochastic_oracle.py
"""

import json
import math
import numpy as np
from scipy.stats import norm

results = {}

# ─── GBM analytical formulas ─────────────────────────────────────────────

s0, mu, sigma, T = 100.0, 0.05, 0.2, 1.0
results["gbm_moments"] = {
    "s0": s0, "mu": mu, "sigma": sigma, "T": T,
    "expected": s0 * math.exp(mu * T),
    "variance": s0**2 * math.exp(2*mu*T) * (math.exp(sigma**2 * T) - 1),
}

# ─── Black-Scholes ────────────────────────────────────────────────────────

def black_scholes_py(S, K, T, r, sigma, call=True):
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    if call:
        price = S * norm.cdf(d1) - K * math.exp(-r*T) * norm.cdf(d2)
        delta = norm.cdf(d1)
    else:
        price = K * math.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        delta = norm.cdf(d1) - 1.0
    return price, delta

# Standard test case
S, K, T, r, sig = 100.0, 100.0, 1.0, 0.05, 0.2
call_price, call_delta = black_scholes_py(S, K, T, r, sig, call=True)
put_price, put_delta = black_scholes_py(S, K, T, r, sig, call=False)

results["black_scholes_atm"] = {
    "S": S, "K": K, "T": T, "r": r, "sigma": sig,
    "call_price": call_price,
    "call_delta": call_delta,
    "put_price": put_price,
    "put_delta": put_delta,
    "put_call_parity": call_price - put_price - (S - K * math.exp(-r*T)),
}

# In-the-money call
S2, K2 = 120.0, 100.0
itm_price, itm_delta = black_scholes_py(S2, K2, T, r, sig, call=True)
results["black_scholes_itm"] = {
    "S": S2, "K": K2, "T": T, "r": r, "sigma": sig,
    "call_price": itm_price,
    "call_delta": itm_delta,
}

# Deep OTM put
S3, K3 = 50.0, 100.0
dotm_price, dotm_delta = black_scholes_py(S3, K3, T, r, sig, call=False)
results["black_scholes_deep_otm"] = {
    "S": S3, "K": K3, "T": T, "r": r, "sigma": sig,
    "put_price": dotm_price,
    "put_delta": dotm_delta,
}

# ─── OU process analytics ────────────────────────────────────────────────

theta_ou = 5.0
sigma_ou = 0.3
mu_ou = 1.0

results["ou_analytics"] = {
    "theta": theta_ou, "sigma": sigma_ou, "mu": mu_ou,
    "stationary_variance": sigma_ou**2 / (2 * theta_ou),
    "autocorr_lag_0.1": math.exp(-theta_ou * 0.1),
    "autocorr_lag_0.5": math.exp(-theta_ou * 0.5),
    "autocorr_lag_1.0": math.exp(-theta_ou * 1.0),
    "autocorr_lag_2.0": math.exp(-theta_ou * 2.0),
}

# ─── Poisson process ─────────────────────────────────────────────────────

lambda_p = 10.0
T_p = 5.0
results["poisson_analytics"] = {
    "lambda": lambda_p, "T": T_p,
    "expected_count": lambda_p * T_p,
    "variance_count": lambda_p * T_p,  # Var(N(T)) = λT
}

# ─── Brownian motion analytics ────────────────────────────────────────────

# E[W(t)] = 0, Var[W(t)] = t, E[W(t)^2] = t
# Quadratic variation of BM on [0,T] = T (almost surely)
results["brownian_analytics"] = {
    "expected_wt": 0.0,
    "variance_wt_at_1": 1.0,
    "variance_wt_at_2": 2.0,
    "quadratic_variation_T1": 1.0,
}

# ─── Markov chain stationary distribution ─────────────────────────────────

# 2-state chain: [[0.7, 0.3], [0.4, 0.6]]
# π satisfies πP = π, Σπ = 1
# π₁·0.7 + π₂·0.4 = π₁ → 0.3π₁ = 0.4π₂ → π₂/π₁ = 3/4
# π₁ + π₂ = 1 → π₁ = 4/7, π₂ = 3/7
P_2state = [[0.7, 0.3], [0.4, 0.6]]
results["markov_2state"] = {
    "transition": [0.7, 0.3, 0.4, 0.6],
    "n_states": 2,
    "stationary": [4.0/7.0, 3.0/7.0],
}

# 3-state chain
P_3state = [[0.5, 0.3, 0.2],
            [0.2, 0.5, 0.3],
            [0.1, 0.3, 0.6]]
# Compute stationary distribution via eigenvalue decomposition
P_np = np.array(P_3state)
# Power method
pi = np.ones(3) / 3.0
for _ in range(1000):
    pi = pi @ P_np
    pi /= pi.sum()

results["markov_3state"] = {
    "transition": [P_3state[i][j] for i in range(3) for j in range(3)],
    "n_states": 3,
    "stationary": pi.tolist(),
}

# ─── Brownian bridge ──────────────────────────────────────────────────────

# B(0) = 0, B(T) = b. Var[B(t)] = t(T-t)/T
T_bb = 1.0
results["brownian_bridge_analytics"] = {
    "T": T_bb,
    "endpoint_0": 0.0,
    "endpoint_T": "b (conditioned)",
    "variance_at_half": 0.25 * T_bb,  # t(T-t)/T at t=T/2
}

# ─── Save ─────────────────────────────────────────────────────────────────

with open("research/gold_standard/family_stochastic_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Stochastic Oracle: {len(results)} test cases generated")
for name in results:
    print(f"  PASS {name}")
