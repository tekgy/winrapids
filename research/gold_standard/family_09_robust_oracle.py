"""
Gold standard oracle for Family 09: Robust Statistics
Compares tambear robust.rs against statsmodels/scipy.
"""
import json
import numpy as np
from scipy import stats

rng = np.random.default_rng(42)
results = {}

# ─── Test data: normal with outliers ──────────────────────────────────
clean = rng.normal(5.0, 1.0, size=50).tolist()
outliers = [50.0, -30.0, 100.0]
contaminated = clean + outliers
arr = np.array(contaminated)

# ─── 1. Huber M-estimate vs statsmodels ──────────────────────────────
from statsmodels.robust import scale as robust_scale
from statsmodels.robust.robust_linear_model import RLM
import statsmodels.api as sm

# Huber M-estimator of location
hub = sm.robust.scale.Huber()
loc, scale = hub(arr)
results["huber_m"] = {
    "data": contaminated,
    "k": 1.345,
    "location": float(loc),
    "scale": float(scale),
}

# ─── 2. MAD (scaled) vs scipy ────────────────────────────────────────
mad_val = stats.median_abs_deviation(arr, scale=1.0)
results["mad_unscaled"] = {
    "data": contaminated,
    "mad": float(mad_val),
}

# ─── 3. Qn scale estimator ───────────────────────────────────────────
# Qn = 2.2219 * {|xi - xj|; i < j}_(k)  where k = binom(h,2), h = floor(n/2)+1
# No direct scipy/statsmodels, compute manually
n = len(contaminated)
arr_sorted = np.sort(arr)
diffs = []
for i in range(n):
    for j in range(i+1, n):
        diffs.append(abs(arr[i] - arr[j]))
diffs.sort()
h = n // 2 + 1
k = h * (h - 1) // 2
# Qn = c_n * d_n * first_quartile_of_diffs
# Approximate: Qn ≈ 2.2219 * diffs[k-1] for large n
# The constant 2.2219 is for consistency with normal
qn_raw = diffs[k - 1]
# Finite-sample correction factors are complex; just record raw value
results["qn_scale"] = {
    "data": contaminated,
    "qn_raw_kth_diff": float(qn_raw),
    "k": k,
    "h": h,
    "n": n,
}

# ─── 4. Weight functions (mathematical definitions) ──────────────────
# Huber weight: w(u) = 1 if |u| <= k, else k/|u|
# Bisquare weight: w(u) = (1 - (u/k)^2)^2 if |u| <= k, else 0
def huber_w(u, k=1.345):
    return 1.0 if abs(u) <= k else k / abs(u)

def bisquare_w(u, k=4.685):
    return (1 - (u/k)**2)**2 if abs(u) <= k else 0.0

results["huber_weight"] = {
    "inputs": [0.0, 0.5, 1.0, 1.345, 2.0, 5.0],
    "k": 1.345,
    "weights": [huber_w(u) for u in [0.0, 0.5, 1.0, 1.345, 2.0, 5.0]],
}

results["bisquare_weight"] = {
    "inputs": [0.0, 1.0, 2.0, 4.0, 4.685, 5.0],
    "k": 4.685,
    "weights": [bisquare_w(u) for u in [0.0, 1.0, 2.0, 4.0, 4.685, 5.0]],
}

# ─── 5. Medcouple ────────────────────────────────────────────────────
# Symmetric data: MC ≈ 0
symmetric = rng.normal(0, 1, size=100).tolist()
# Right-skewed: MC > 0
right_skewed = rng.exponential(2.0, size=100).tolist()

results["medcouple_symmetric"] = {
    "data": symmetric,
    "expected_range": [-0.2, 0.2],  # should be near 0
}

results["medcouple_right_skewed"] = {
    "data": right_skewed,
    "expected_sign": "positive",
}

# ─── Write output ────────────────────────────────────────────────────
with open("R:/winrapids/research/gold_standard/family_09_robust_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Wrote {len(results)} test cases")
for k, v in results.items():
    vals = {kk: vv for kk, vv in v.items()
            if kk not in ("data", "inputs", "symmetric", "right_skewed")}
    print(f"  {k}: {vals}")
