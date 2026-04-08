"""
Gold standard oracle for Family 00: Special Functions
Compares tambear special_functions.rs against scipy.special.
"""
import json
import numpy as np
from scipy import special, stats

results = {}

# ─── 1. Error function vs scipy.special.erf ──────────────────────────
erf_inputs = [0.0, 0.5, 1.0, 1.5, 2.0, -1.0, 3.0]
results["erf"] = {
    "inputs": erf_inputs,
    "values": [float(special.erf(x)) for x in erf_inputs],
}

# ─── 2. Complementary error function ─────────────────────────────────
results["erfc"] = {
    "inputs": erf_inputs,
    "values": [float(special.erfc(x)) for x in erf_inputs],
}

# ─── 3. Log-gamma vs scipy.special.gammaln ────────────────────────────
lgamma_inputs = [0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 0.1]
results["log_gamma"] = {
    "inputs": lgamma_inputs,
    "values": [float(special.gammaln(x)) for x in lgamma_inputs],
}

# ─── 4. Gamma function vs scipy.special.gamma ────────────────────────
gamma_inputs = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
results["gamma"] = {
    "inputs": gamma_inputs,
    "values": [float(special.gamma(x)) for x in gamma_inputs],
}

# ─── 5. Log-beta vs scipy.special.betaln ─────────────────────────────
lbeta_pairs = [(1.0, 1.0), (2.0, 3.0), (0.5, 0.5), (5.0, 10.0)]
results["log_beta"] = {
    "pairs": [[a, b] for a, b in lbeta_pairs],
    "values": [float(special.betaln(a, b)) for a, b in lbeta_pairs],
}

# ─── 6. Normal CDF vs scipy.stats.norm.cdf ───────────────────────────
ncdf_inputs = [-3.0, -2.0, -1.0, 0.0, 1.0, 1.96, 2.0, 3.0]
results["normal_cdf"] = {
    "inputs": ncdf_inputs,
    "values": [float(stats.norm.cdf(x)) for x in ncdf_inputs],
}

# ─── 7. t-distribution CDF vs scipy.stats.t.cdf ──────────────────────
t_inputs = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
t_dfs = [5.0, 10.0, 30.0]
results["t_cdf"] = {
    "inputs": t_inputs,
    "dfs": t_dfs,
    "values": {
        str(df): [float(stats.t.cdf(x, df)) for x in t_inputs]
        for df in t_dfs
    },
}

# ─── 8. Chi-square CDF vs scipy.stats.chi2.cdf ───────────────────────
chi2_inputs = [1.0, 3.84, 5.99, 9.21, 15.0]
chi2_dfs = [1.0, 2.0, 5.0]
results["chi2_cdf"] = {
    "inputs": chi2_inputs,
    "dfs": chi2_dfs,
    "values": {
        str(df): [float(stats.chi2.cdf(x, df)) for x in chi2_inputs]
        for df in chi2_dfs
    },
}

# ─── 9. F-distribution CDF vs scipy.stats.f.cdf ──────────────────────
f_inputs = [1.0, 2.0, 3.84, 5.0, 10.0]
results["f_cdf"] = {
    "inputs": f_inputs,
    "d1": 3.0,
    "d2": 20.0,
    "values": [float(stats.f.cdf(x, 3, 20)) for x in f_inputs],
}

# ─── 10. Digamma vs scipy.special.digamma ────────────────────────────
dg_inputs = [0.5, 1.0, 2.0, 5.0, 10.0]
results["digamma"] = {
    "inputs": dg_inputs,
    "values": [float(special.digamma(x)) for x in dg_inputs],
}

# ─── 11. Trigamma vs scipy.special.polygamma(1, x) ───────────────────
results["trigamma"] = {
    "inputs": dg_inputs,
    "values": [float(special.polygamma(1, x)) for x in dg_inputs],
}

# ─── 12. Regularized incomplete gamma (lower) ────────────────────────
rig_pairs = [(1.0, 1.0), (2.0, 3.0), (5.0, 5.0), (0.5, 0.5)]
results["regularized_gamma_p"] = {
    "pairs": [[a, x] for a, x in rig_pairs],
    "values": [float(special.gammainc(a, x)) for a, x in rig_pairs],
}

# ─── 13. Regularized incomplete beta ─────────────────────────────────
rib_triples = [(0.5, 0.5, 0.5), (2.0, 3.0, 0.4), (1.0, 1.0, 0.5), (5.0, 10.0, 0.3)]
results["regularized_incomplete_beta"] = {
    "triples": [[x, a, b] for x, a, b in rib_triples],
    "values": [float(special.betainc(a, b, x)) for x, a, b in rib_triples],
}

# ─── 14. Tail probabilities (two-tailed normal) ──────────────────────
z_vals = [1.0, 1.645, 1.96, 2.0, 2.576, 3.0]
results["normal_two_tail_p"] = {
    "z_values": z_vals,
    "p_values": [float(2 * stats.norm.sf(abs(z))) for z in z_vals],
}

# ─── 15. t two-tailed p-values ───────────────────────────────────────
results["t_two_tail_p"] = {
    "t_values": [1.0, 2.0, 3.0],
    "df": 10.0,
    "p_values": [float(2 * stats.t.sf(abs(t), 10)) for t in [1.0, 2.0, 3.0]],
}

# ─── Write output ────────────────────────────────────────────────────
with open("R:/winrapids/research/gold_standard/family_00_special_functions_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Wrote {len(results)} test cases")
for k, v in results.items():
    vals = {kk: vv for kk, vv in v.items() if kk not in ("data",)}
    print(f"  {k}")
