"""
Gold standard oracle for Family 10: Information Theory
Compares tambear information_theory.rs against scipy.stats and sklearn.
"""
import json
import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score

rng = np.random.default_rng(42)
results = {}

# ─── 1. Shannon entropy vs scipy.stats.entropy ───────────────────────
# Uniform distribution
probs_uniform = [0.25, 0.25, 0.25, 0.25]
# Peaked distribution
probs_peaked = [0.7, 0.1, 0.1, 0.1]
# Near-deterministic
probs_determ = [0.97, 0.01, 0.01, 0.01]

results["shannon_entropy"] = {
    "uniform_probs": probs_uniform,
    "uniform_entropy": float(stats.entropy(probs_uniform, base=np.e)),
    "peaked_probs": probs_peaked,
    "peaked_entropy": float(stats.entropy(probs_peaked, base=np.e)),
    "deterministic_probs": probs_determ,
    "deterministic_entropy": float(stats.entropy(probs_determ, base=np.e)),
}

# ─── 2. Shannon entropy from counts ──────────────────────────────────
counts = [10.0, 20.0, 30.0, 40.0]
total = sum(counts)
probs_from_counts = [c / total for c in counts]
results["entropy_from_counts"] = {
    "counts": counts,
    "entropy": float(stats.entropy(probs_from_counts, base=np.e)),
}

# ─── 3. KL divergence vs scipy.stats.entropy(p, q) ───────────────────
p = [0.4, 0.3, 0.2, 0.1]
q = [0.25, 0.25, 0.25, 0.25]
results["kl_divergence"] = {
    "p": p,
    "q": q,
    "kl_pq": float(stats.entropy(p, q, base=np.e)),  # KL(p||q)
}

# ─── 4. Cross entropy ────────────────────────────────────────────────
# H(p,q) = H(p) + KL(p||q)
hp = float(stats.entropy(p, base=np.e))
kl = float(stats.entropy(p, q, base=np.e))
results["cross_entropy"] = {
    "p": p,
    "q": q,
    "cross_entropy": hp + kl,
}

# ─── 5. JS divergence ────────────────────────────────────────────────
m = [(pi + qi) / 2 for pi, qi in zip(p, q)]
js = 0.5 * stats.entropy(p, m, base=np.e) + 0.5 * stats.entropy(q, m, base=np.e)
results["js_divergence"] = {
    "p": p,
    "q": q,
    "js": float(js),
}

# ─── 6. Mutual information from contingency table ────────────────────
# 3x3 contingency table
contingency = [
    10.0, 2.0, 1.0,
    3.0, 15.0, 2.0,
    1.0, 3.0, 12.0,
]
nx, ny = 3, 3
total = sum(contingency)
ct = np.array(contingency).reshape(nx, ny)

# Compute MI manually using scipy
# MI = Σ p(x,y) * log(p(x,y) / (p(x)*p(y)))
pxy = ct / total
px = pxy.sum(axis=1)
py = pxy.sum(axis=0)
mi = 0.0
for i in range(nx):
    for j in range(ny):
        if pxy[i, j] > 0:
            mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))

results["mutual_information"] = {
    "contingency": contingency,
    "nx": nx,
    "ny": ny,
    "mi": float(mi),
}

# ─── 7. Conditional entropy H(Y|X) ───────────────────────────────────
# H(Y|X) = H(X,Y) - H(X)
# H(X,Y) = -Σ p(x,y) log p(x,y)
h_xy = -np.sum(pxy[pxy > 0] * np.log(pxy[pxy > 0]))
h_x = -np.sum(px[px > 0] * np.log(px[px > 0]))
h_y_given_x = h_xy - h_x
results["conditional_entropy"] = {
    "contingency": contingency,
    "nx": nx,
    "ny": ny,
    "h_y_given_x": float(h_y_given_x),
}

# ─── 8. Variation of information ──────────────────────────────────────
h_y = -np.sum(py[py > 0] * np.log(py[py > 0]))
vi = h_xy + h_xy - h_x - h_y  # = H(X|Y) + H(Y|X)
# Actually: VI = H(X,Y) - MI, but also = H(X|Y) + H(Y|X)
# VI = H(X) + H(Y) - 2*MI
vi2 = h_x + h_y - 2*mi
results["variation_of_information"] = {
    "contingency": contingency,
    "nx": nx,
    "ny": ny,
    "vi": float(vi2),
}

# ─── 9. Clustering MI scores vs sklearn ──────────────────────────────
labels_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]
labels_pred = [0, 0, 1, 1, 1, 2, 2, 2, 0, 2]

mi_score = mutual_info_score(labels_true, labels_pred)
nmi_score = normalized_mutual_info_score(labels_true, labels_pred, average_method='arithmetic')
ami_score = adjusted_mutual_info_score(labels_true, labels_pred)

results["clustering_mi"] = {
    "labels_true": labels_true,
    "labels_pred": labels_pred,
    "mi": float(mi_score),
    "nmi_arithmetic": float(nmi_score),
    "ami": float(ami_score),
}

# ─── 10. Rényi entropy ───────────────────────────────────────────────
# H_alpha(p) = 1/(1-alpha) * log(Σ p^alpha)
alpha = 2.0
renyi = (1.0/(1.0 - alpha)) * np.log(np.sum(np.array(probs_peaked)**alpha))
results["renyi_entropy"] = {
    "probs": probs_peaked,
    "alpha": alpha,
    "renyi": float(renyi),
}

# ─── 11. Tsallis entropy ─────────────────────────────────────────────
# S_q(p) = 1/(q-1) * (1 - Σ p^q)
q_tsallis = 2.0
tsallis = (1.0 / (q_tsallis - 1.0)) * (1.0 - np.sum(np.array(probs_peaked)**q_tsallis))
results["tsallis_entropy"] = {
    "probs": probs_peaked,
    "q": q_tsallis,
    "tsallis": float(tsallis),
}

# ─── Write output ────────────────────────────────────────────────────
with open("R:/winrapids/research/gold_standard/family_10_information_theory_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Wrote {len(results)} test cases")
for k, v in results.items():
    vals = {kk: vv for kk, vv in v.items()
            if kk not in ("x", "y", "labels_true", "labels_pred", "contingency",
                          "uniform_probs", "peaked_probs", "deterministic_probs",
                          "counts", "p", "q", "probs")}
    print(f"  {k}: {vals}")
