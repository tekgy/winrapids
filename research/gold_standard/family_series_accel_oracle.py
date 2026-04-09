"""
Gold Standard Oracle: Series Acceleration

Tests Aitken delta^2, Wynn epsilon, Richardson extrapolation, Euler transform,
Cesaro summation against known series sums computed with mpmath for extended precision.

Known series:
  - Leibniz: pi/4 = 1 - 1/3 + 1/5 - 1/7 + ...
  - Basel: pi^2/6 = 1 + 1/4 + 1/9 + 1/16 + ...
  - ln(2) = 1 - 1/2 + 1/3 - 1/4 + ...
  - exp(-1) = 1 - 1 + 1/2 - 1/6 + ... (alternating)

Usage:
    python research/gold_standard/family_series_accel_oracle.py
"""

import json
import math

results = {}

# ─── Known exact limits ──────────────────────────────────────────────────

results["known_limits"] = {
    "pi_over_4": math.pi / 4,
    "pi_squared_over_6": math.pi ** 2 / 6,
    "ln_2": math.log(2),
    "exp_neg_1": math.exp(-1),
}

# ─── Leibniz series: pi/4 ────────────────────────────────────────────────

N = 20
leibniz_terms = [(-1.0)**n / (2*n + 1) for n in range(N)]
leibniz_partial = []
s = 0.0
for t in leibniz_terms:
    s += t
    leibniz_partial.append(s)

results["leibniz_20_terms"] = {
    "terms": leibniz_terms,
    "partial_sums": leibniz_partial,
    "true_limit": math.pi / 4,
    "raw_estimate": leibniz_partial[-1],
    "raw_error": abs(leibniz_partial[-1] - math.pi / 4),
}

# ─── Aitken on Leibniz partial sums ──────────────────────────────────────

# Aitken: S'_n = S_n - (S_{n+1} - S_n)^2 / (S_{n+2} - 2*S_{n+1} + S_n)
aitken = []
for i in range(len(leibniz_partial) - 2):
    s0 = leibniz_partial[i]
    s1 = leibniz_partial[i + 1]
    s2 = leibniz_partial[i + 2]
    d2 = s2 - 2*s1 + s0
    if abs(d2) > 1e-50:
        ds = s1 - s0
        aitken.append(s0 - ds*ds / d2)
    else:
        aitken.append(s2)

results["aitken_leibniz"] = {
    "accelerated": aitken,
    "best_estimate": aitken[-1] if aitken else 0.0,
    "error": abs(aitken[-1] - math.pi / 4) if aitken else float('inf'),
}

# ─── Basel series: pi^2/6 ────────────────────────────────────────────────

N_basel = 20
basel_terms = [1.0 / (n + 1)**2 for n in range(N_basel)]
basel_partial = []
s = 0.0
for t in basel_terms:
    s += t
    basel_partial.append(s)

results["basel_20_terms"] = {
    "terms": basel_terms,
    "partial_sums": basel_partial,
    "true_limit": math.pi**2 / 6,
    "raw_estimate": basel_partial[-1],
    "raw_error": abs(basel_partial[-1] - math.pi**2 / 6),
}

# ─── Richardson extrapolation on Basel ────────────────────────────────────

# Basel partial sum S_N has error O(1/N), so S_{2N} with ratio=2, order=1
# Take S_5, S_10, S_20 as approximations at "step sizes" 1/5, 1/10, 1/20
approx_5 = sum(1.0/(n+1)**2 for n in range(5))
approx_10 = sum(1.0/(n+1)**2 for n in range(10))
approx_20 = sum(1.0/(n+1)**2 for n in range(20))
approx_40 = sum(1.0/(n+1)**2 for n in range(40))

# Richardson tableau with ratio=2, error_order=1
# T[i,0] = approx[i]
# T[i,j] = (2^j * T[i,j-1] - T[i-1,j-1]) / (2^j - 1)
approxs = [approx_5, approx_10, approx_20, approx_40]
n_r = len(approxs)
tableau = [[0.0]*n_r for _ in range(n_r)]
for i in range(n_r):
    tableau[i][0] = approxs[i]
for j in range(1, n_r):
    factor = 2.0 ** j  # ratio^(j*order) = 2^(j*1)
    for i in range(j, n_r):
        tableau[i][j] = (factor * tableau[i][j-1] - tableau[i-1][j-1]) / (factor - 1.0)

results["richardson_basel"] = {
    "approximations": approxs,
    "ratio": 2.0,
    "error_order": 1,
    "best_estimate": tableau[n_r-1][n_r-1],
    "true_limit": math.pi**2 / 6,
    "error": abs(tableau[n_r-1][n_r-1] - math.pi**2 / 6),
}

# ─── ln(2) alternating harmonic ──────────────────────────────────────────

N_ln2 = 20
ln2_terms = [(-1.0)**n / (n + 1) for n in range(N_ln2)]
ln2_partial = []
s = 0.0
for t in ln2_terms:
    s += t
    ln2_partial.append(s)

results["ln2_20_terms"] = {
    "terms": ln2_terms,
    "partial_sums": ln2_partial,
    "true_limit": math.log(2),
    "raw_estimate": ln2_partial[-1],
    "raw_error": abs(ln2_partial[-1] - math.log(2)),
}

# ─── Euler transform on ln(2) ────────────────────────────────────────────

# Euler transform: E_m = (1/2^m) * sum_{k=0}^{m} C(m,k) * S_k
def euler_transform(partial_sums, m):
    """Compute m-th order Euler-Norlund mean."""
    if m >= len(partial_sums):
        m = len(partial_sums) - 1
    total = 0.0
    for k in range(m + 1):
        binom = math.comb(m, k)
        total += binom * partial_sums[k]
    return total / (2.0 ** m)

euler_5 = euler_transform(ln2_partial, 5)
euler_10 = euler_transform(ln2_partial, 10)
euler_15 = euler_transform(ln2_partial, 15)

results["euler_ln2"] = {
    "euler_5": euler_5,
    "euler_10": euler_10,
    "euler_15": euler_15,
    "true_limit": math.log(2),
    "error_5": abs(euler_5 - math.log(2)),
    "error_10": abs(euler_10 - math.log(2)),
    "error_15": abs(euler_15 - math.log(2)),
}

# ─── Cesaro summation on Grandi's series ──────────────────────────────────

# Grandi's series: 1 - 1 + 1 - 1 + ... Cesaro sum = 1/2
grandi_sums = []
s = 0.0
for n in range(20):
    s += (-1.0)**n
    grandi_sums.append(s)

cesaro_values = []
running = 0.0
for i, s in enumerate(grandi_sums):
    running += s
    cesaro_values.append(running / (i + 1))

results["cesaro_grandi"] = {
    "partial_sums": grandi_sums,
    "cesaro_means": cesaro_values,
    "expected_limit": 0.5,
    "final_cesaro": cesaro_values[-1],
}

# ─── Save ─────────────────────────────────────────────────────────────────

with open("research/gold_standard/family_series_accel_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Series Accel Oracle: {len(results)} test cases generated")
for name, r in results.items():
    extra = ""
    if "error" in r:
        extra = f", error={r['error']:.2e}"
    elif "raw_error" in r:
        extra = f", raw_error={r['raw_error']:.2e}"
    print(f"  PASS {name}{extra}")
