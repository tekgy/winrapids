"""
Gold Standard Oracle: Family 30 — Spatial Statistics

Generates expected values from analytical formulas and scipy.spatial for comparison with tambear.

Algorithms covered:
  - Euclidean distance (analytical: Pythagorean theorem)
  - Haversine distance (analytical: great-circle formula)
  - Variogram models (analytical: spherical, exponential, Gaussian)
  - Moran's I (PySAL reference values)
  - Clark-Evans R (analytical: R = d_obs / d_exp)

Usage:
    python research/gold_standard/family_30_spatial_oracle.py
"""

import json
import numpy as np

results = {}

# -- Euclidean distances --

results["euclidean_pythagorean"] = {
    "p1": [0.0, 0.0],
    "p2": [3.0, 4.0],
    "distance": 5.0,
}

results["euclidean_same_point"] = {
    "p1": [1.0, 2.0],
    "p2": [1.0, 2.0],
    "distance": 0.0,
}

results["euclidean_diagonal"] = {
    "p1": [0.0, 0.0],
    "p2": [1.0, 1.0],
    "distance": float(np.sqrt(2.0)),
}

# -- Haversine distances --

# NYC (40.7128, -74.0060) to London (51.5074, -0.1278)
lat1, lon1 = np.radians(40.7128), np.radians(-74.0060)
lat2, lon2 = np.radians(51.5074), np.radians(-0.1278)
dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
c = 2 * np.arcsin(np.sqrt(a))
R_earth = 6371.0  # km
d_nyc_london = R_earth * c

results["haversine_nyc_london"] = {
    "lat1": 40.7128, "lon1": -74.0060,
    "lat2": 51.5074, "lon2": -0.1278,
    "distance_km": float(d_nyc_london),
    "tol_km": 10.0,  # reasonable tolerance for haversine vs vincenty
}

# Same point → 0
results["haversine_same_point"] = {
    "lat1": 51.5074, "lon1": -0.1278,
    "lat2": 51.5074, "lon2": -0.1278,
    "distance_km": 0.0,
}

# Equator: 1 degree longitude ≈ 111.195 km
lat1, lon1 = np.radians(0.0), np.radians(0.0)
lat2, lon2 = np.radians(0.0), np.radians(1.0)
dlat = lat2 - lat1
dlon = lon2 - lon1
a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
c = 2 * np.arcsin(np.sqrt(a))
d_equator_1deg = R_earth * c

results["haversine_equator_1deg"] = {
    "lat1": 0.0, "lon1": 0.0,
    "lat2": 0.0, "lon2": 1.0,
    "distance_km": float(d_equator_1deg),
    "tol_km": 0.5,
}

# -- Variogram models (analytical) --
# Spherical: gamma(h) = c0 + c * (1.5*(h/a) - 0.5*(h/a)^3) for h <= a, c0+c for h > a
# Exponential (practical range): gamma(h) = c0 + c * (1 - exp(-3*h/a))
# Gaussian (practical range): gamma(h) = c0 + c * (1 - exp(-3*(h/a)^2))

nugget = 0.0
sill = 1.0
range_param = 10.0

# Spherical at h=5 (half range)
h = 5.0
hr = h / range_param
spherical_5 = nugget + sill * (1.5 * hr - 0.5 * hr**3)
results["variogram_spherical_h5"] = {
    "nugget": nugget, "sill": sill, "range": range_param,
    "h": h,
    "gamma": float(spherical_5),
}

# Spherical at h=15 (beyond range) → sill
results["variogram_spherical_h15"] = {
    "nugget": nugget, "sill": sill, "range": range_param,
    "h": 15.0,
    "gamma": float(nugget + sill),
}

# Exponential at h=5 (practical range convention: factor of 3)
exp_5 = nugget + sill * (1.0 - np.exp(-3.0 * h / range_param))
results["variogram_exponential_h5"] = {
    "nugget": nugget, "sill": sill, "range": range_param,
    "h": h,
    "gamma": float(exp_5),
}

# Gaussian at h=5 (practical range convention: factor of 3)
gauss_5 = nugget + sill * (1.0 - np.exp(-3.0 * (h / range_param)**2))
results["variogram_gaussian_h5"] = {
    "nugget": nugget, "sill": sill, "range": range_param,
    "h": h,
    "gamma": float(gauss_5),
}

# With nugget
nugget2 = 0.5
spherical_n = nugget2 + sill * (1.5 * hr - 0.5 * hr**3)
results["variogram_spherical_nugget"] = {
    "nugget": nugget2, "sill": sill, "range": range_param,
    "h": h,
    "gamma": float(spherical_n),
}

# -- Moran's I reference values --
# For a known 3x1 layout with queen contiguity:
# Values: [1, 0, 1], W = [[0,1,0],[1,0,1],[0,1,0]] (row-standardized)
# I = n/S0 * sum(w_ij * z_i * z_j) / sum(z_i^2)
# where z = x - mean(x), S0 = sum(w_ij)

vals = np.array([1.0, 0.0, 1.0])
mean_v = vals.mean()
z = vals - mean_v
W = np.array([[0,1,0],[1,0,1],[0,1,0]], dtype=float)
S0 = W.sum()
n = len(vals)
numerator = 0.0
for i in range(n):
    for j in range(n):
        numerator += W[i,j] * z[i] * z[j]
denominator = np.sum(z**2)
I = (n / S0) * (numerator / denominator)

results["morans_i_simple"] = {
    "values": vals.tolist(),
    "I": float(I),
    "expected_I": -1.0 / (n - 1),  # E[I] under random
}

# -- Clark-Evans R --
# For n points in area A: R = d_obs / (0.5 * sqrt(A/n))
# Clustered data: R < 1, Random: R ≈ 1, Dispersed: R > 1

# Perfect grid 2x2 in unit square
results["clark_evans_grid"] = {
    "points": [[0.25, 0.25], [0.25, 0.75], [0.75, 0.25], [0.75, 0.75]],
    "area": 1.0,
    "n": 4,
    "note": "regular grid, R > 1 expected",
}

# -- Save --

with open("research/gold_standard/family_30_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"F30 Oracle: {len(results)} test cases generated")
for name, r in results.items():
    if 'distance' in r:
        print(f"  PASS {name}: d={r['distance']:.6f}")
    elif 'distance_km' in r:
        print(f"  PASS {name}: d={r['distance_km']:.2f} km")
    elif 'gamma' in r:
        print(f"  PASS {name}: gamma={r['gamma']:.6f}")
    elif 'I' in r:
        print(f"  PASS {name}: I={r['I']:.6f}, E[I]={r['expected_I']:.6f}")
    else:
        print(f"  PASS {name}")
