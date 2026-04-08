"""
Gold Standard Oracle: Family 05 — Optimization

Generates expected values from scipy.optimize for comparison with tambear.

Algorithms covered:
  - Golden section / brent → scipy.optimize.minimize_scalar
  - Gradient descent / Adam / L-BFGS → scipy.optimize.minimize
  - Nelder-Mead → scipy.optimize.minimize(method='Nelder-Mead')
  - Coordinate descent (separable) → analytical
  - Projected gradient (box) → scipy.optimize.minimize(method='L-BFGS-B', bounds=...)

Usage:
    python research/gold_standard/family_05_optimization_oracle.py
"""

import json
import numpy as np
from scipy.optimize import minimize, minimize_scalar


def rosenbrock(x):
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    return np.array([
        -2 * (1 - x[0]) - 400 * x[0] * (x[1] - x[0]**2),
        200 * (x[1] - x[0]**2)
    ])

def quadratic(x):
    return x[0]**2 + x[1]**2

def quadratic_grad(x):
    return np.array([2*x[0], 2*x[1]])


results = {}

# ── Golden section / scalar minimization ──

# min (x-3)^2 on [0,10]
r = minimize_scalar(lambda x: (x-3)**2, bounds=(0, 10), method='bounded')
results["golden_section_parabola"] = {
    "x_star": float(r.x),
    "f_star": float(r.fun),
    "true_x": 3.0,
    "true_f": 0.0,
    "tol": 1e-6,
}

# min cos(x) on [2,5]
r = minimize_scalar(lambda x: np.cos(x), bounds=(2, 5), method='bounded')
results["golden_section_cos"] = {
    "x_star": float(r.x),
    "f_star": float(r.fun),
    "true_x": float(np.pi),
    "true_f": -1.0,
    "tol": 1e-6,
}

# ── Quadratic: all methods should find (0,0) ──

x0 = np.array([5.0, 3.0])

for method in ['CG', 'BFGS', 'L-BFGS-B', 'Nelder-Mead', 'Powell']:
    jac = quadratic_grad if method in ['CG', 'BFGS', 'L-BFGS-B'] else None
    r = minimize(quadratic, x0, method=method, jac=jac,
                 options={'maxiter': 10000, 'gtol': 1e-10})
    results[f"quadratic_{method.lower().replace('-', '')}"] = {
        "x_star": r.x.tolist(),
        "f_star": float(r.fun),
        "true_x": [0.0, 0.0],
        "true_f": 0.0,
        "converged": bool(r.success),
        "niter": int(r.nit),
        "tol": 1e-4,
    }

# ── Rosenbrock: harder test ──

x0_rosen = np.array([-1.0, 1.0])

for method in ['L-BFGS-B', 'Nelder-Mead', 'CG', 'BFGS']:
    jac = rosenbrock_grad if method in ['CG', 'BFGS', 'L-BFGS-B'] else None
    r = minimize(rosenbrock, x0_rosen, method=method, jac=jac,
                 options={'maxiter': 50000, 'gtol': 1e-10})
    results[f"rosenbrock_{method.lower().replace('-', '')}"] = {
        "x_star": r.x.tolist(),
        "f_star": float(r.fun),
        "true_x": [1.0, 1.0],
        "true_f": 0.0,
        "converged": bool(r.success),
        "niter": int(r.nit),
        "tol": 0.01,
    }

# ── Box-constrained: min x^2+y^2 subject to x∈[1,5], y∈[2,5] ──

r = minimize(quadratic, [3.0, 4.0], method='L-BFGS-B', jac=quadratic_grad,
             bounds=[(1, 5), (2, 5)])
results["box_constrained"] = {
    "x_star": r.x.tolist(),
    "f_star": float(r.fun),
    "true_x": [1.0, 2.0],
    "true_f": 5.0,
    "converged": bool(r.success),
    "tol": 1e-4,
}

# ── Beale function (harder 2D test) ──
# f(x,y) = (1.5-x+xy)^2 + (2.25-x+xy^2)^2 + (2.625-x+xy^3)^2
# Minimum at (3, 0.5), f=0

def beale(x):
    return ((1.5 - x[0] + x[0]*x[1])**2 +
            (2.25 - x[0] + x[0]*x[1]**2)**2 +
            (2.625 - x[0] + x[0]*x[1]**3)**2)

r = minimize(beale, [0.0, 0.0], method='Nelder-Mead',
             options={'maxiter': 50000, 'xatol': 1e-10})
results["beale_nelder_mead"] = {
    "x_star": r.x.tolist(),
    "f_star": float(r.fun),
    "true_x": [3.0, 0.5],
    "true_f": 0.0,
    "converged": bool(r.success),
    "tol": 0.01,
}

# ── Booth function: f(x,y) = (x+2y-7)^2 + (2x+y-5)^2, min at (1,3) ──

def booth(x):
    return (x[0] + 2*x[1] - 7)**2 + (2*x[0] + x[1] - 5)**2

r = minimize(booth, [0.0, 0.0], method='L-BFGS-B',
             jac=lambda x: np.array([
                 2*(x[0]+2*x[1]-7) + 4*(2*x[0]+x[1]-5),
                 4*(x[0]+2*x[1]-7) + 2*(2*x[0]+x[1]-5)]))
results["booth_lbfgsb"] = {
    "x_star": r.x.tolist(),
    "f_star": float(r.fun),
    "true_x": [1.0, 3.0],
    "true_f": 0.0,
    "converged": bool(r.success),
    "tol": 1e-6,
}

# ── Save results ──

with open("research/gold_standard/family_05_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"F05 Oracle: {len(results)} test cases generated")
for name, r in results.items():
    x = r['x_star'] if isinstance(r['x_star'], list) else [r['x_star']]
    true = r['true_x'] if isinstance(r['true_x'], list) else [r['true_x']]
    err = max(abs(a-b) for a, b in zip(x, true))
    status = "PASS" if err < r['tol'] else "FAIL"
    print(f"  {status} {name}: x*={[round(v,6) for v in x]}, "
          f"f*={r['f_star']:.6e}, err={err:.2e}")
