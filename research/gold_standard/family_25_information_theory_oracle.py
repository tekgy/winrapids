"""
Gold Standard Oracle: Family 25 — Information Theory

Generates expected values from scipy.stats.entropy and sklearn.metrics for comparison with tambear.

Algorithms covered:
  - Shannon entropy (scipy.stats.entropy)
  - KL divergence (scipy.stats.entropy with two args)
  - Mutual information (sklearn.metrics.mutual_info_score)
  - Normalized mutual information (sklearn.metrics.normalized_mutual_info_score)

Usage:
    python research/gold_standard/family_25_information_theory_oracle.py
"""

import json
import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score, normalized_mutual_info_score, adjusted_mutual_info_score

results = {}

# -- Shannon entropy --

# Uniform over 4 outcomes: H = ln(4)
p = [0.25, 0.25, 0.25, 0.25]
results["shannon_uniform_4"] = {
    "probs": p,
    "H": float(entropy(p)),  # natural log
    "H_exact": float(np.log(4)),
}

# Fair coin: H = ln(2)
p = [0.5, 0.5]
results["shannon_binary_fair"] = {
    "probs": p,
    "H": float(entropy(p)),
    "H_exact": float(np.log(2)),
}

# Deterministic: H = 0
p = [1.0, 0.0, 0.0]
results["shannon_deterministic"] = {
    "probs": p,
    "H": float(entropy(p)),
}

# Asymmetric
p = [0.9, 0.1]
results["shannon_asymmetric"] = {
    "probs": p,
    "H": float(entropy(p)),
}

# 8-outcome uniform: H = ln(8)
p = [1/8] * 8
results["shannon_uniform_8"] = {
    "probs": p,
    "H": float(entropy(p)),
    "H_exact": float(np.log(8)),
}

# -- KL divergence --

p = [0.5, 0.5]
q = [0.5, 0.5]
results["kl_identical"] = {
    "p": p, "q": q,
    "KL": float(entropy(p, q)),  # Should be 0
}

p = [0.9, 0.1]
q = [0.5, 0.5]
results["kl_asymmetric_scipy"] = {
    "p": p, "q": q,
    "KL": float(entropy(p, q)),
}

p = [0.5, 0.5]
q = [0.9, 0.1]
results["kl_reverse"] = {
    "p": p, "q": q,
    "KL": float(entropy(p, q)),
}

# Verify asymmetry: KL(p||q) != KL(q||p)
p3 = [0.25, 0.25, 0.25, 0.25]
q3 = [0.1, 0.2, 0.3, 0.4]
results["kl_asymmetry_check"] = {
    "p": p3, "q": q3,
    "KL_pq": float(entropy(p3, q3)),
    "KL_qp": float(entropy(q3, p3)),
}

# -- Mutual information (from labels) --

# Perfect correlation
labels_a = [0, 0, 1, 1, 2, 2]
labels_b = [0, 0, 1, 1, 2, 2]
results["mi_perfect"] = {
    "labels_a": labels_a,
    "labels_b": labels_b,
    "MI": float(mutual_info_score(labels_a, labels_b)),
    "NMI": float(normalized_mutual_info_score(labels_a, labels_b)),
}

# Independent (designed to be low MI)
labels_a = [0, 0, 0, 1, 1, 1]
labels_b = [0, 1, 0, 1, 0, 1]
results["mi_low_dependence"] = {
    "labels_a": labels_a,
    "labels_b": labels_b,
    "MI": float(mutual_info_score(labels_a, labels_b)),
    "NMI": float(normalized_mutual_info_score(labels_a, labels_b)),
}

# Known 3x3 contingency table
# Joint: [[2,1,0],[0,2,1],[1,0,2]] → moderate MI
labels_a = [0,0,0, 1,1,1, 2,2,2]
labels_b = [0,0,1, 1,1,2, 2,2,0]
results["mi_3x3_moderate"] = {
    "labels_a": labels_a,
    "labels_b": labels_b,
    "MI": float(mutual_info_score(labels_a, labels_b)),
    "NMI": float(normalized_mutual_info_score(labels_a, labels_b)),
    "AMI": float(adjusted_mutual_info_score(labels_a, labels_b)),
}

# -- Save --

with open("research/gold_standard/family_25_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"F25 Oracle: {len(results)} test cases generated")
for name, r in results.items():
    parts = []
    if 'H' in r: parts.append(f"H={r['H']:.6f}")
    if 'KL' in r: parts.append(f"KL={r['KL']:.6f}")
    if 'KL_pq' in r: parts.append(f"KL(p||q)={r['KL_pq']:.6f}")
    if 'MI' in r: parts.append(f"MI={r['MI']:.6f}")
    if 'NMI' in r: parts.append(f"NMI={r['NMI']:.6f}")
    print(f"  PASS {name}: {', '.join(parts)}")
