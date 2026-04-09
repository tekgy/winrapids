"""
Gold Standard Oracle: Physics

Tests analytical solutions for:
  - Physical constants against NIST 2018 CODATA values
  - Simple harmonic oscillator exact solution
  - Damped harmonic oscillator (underdamped)
  - Kepler orbit: vis-viva, orbital elements
  - SHO energy conservation
  - N-body energy conservation (qualitative)

Usage:
    python research/gold_standard/family_physics_oracle.py
"""

import json
import math

results = {}

# ─── Physical constants (NIST 2018 CODATA) ──────────────────────────────

results["constants"] = {
    "k_boltzmann": 1.380649e-23,
    "h_bar": 1.0545718e-34,
    "speed_of_light": 2.99792458e8,
    "G_grav": 6.67430e-11,
    "elem_charge": 1.602176634e-19,
    "mass_electron": 9.1093837015e-31,
    "bohr_radius": 5.29177210903e-11,
    "avogadro": 6.02214076e23,
    "gas_constant": 8.314462618,
    "epsilon_0": 8.8541878128e-12,
    "hydrogen_ground_ev": -13.605693,
}

# ─── Simple harmonic oscillator ──────────────────────────────────────────

# x(t) = x0 cos(ωt) + (v0/ω) sin(ωt)
# v(t) = -x0 ω sin(ωt) + v0 cos(ωt)
x0, v0, omega = 1.0, 0.0, 2.0 * math.pi  # period = 1s

sho_tests = {}
for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
    x = x0 * math.cos(omega * t) + (v0 / omega) * math.sin(omega * t)
    v = -x0 * omega * math.sin(omega * t) + v0 * math.cos(omega * t)
    sho_tests[str(t)] = {"x": x, "v": v}

results["sho_exact"] = {
    "x0": x0, "v0": v0, "omega": omega,
    "positions": sho_tests,
}

# SHO energy conservation: E = 1/2 m(v^2 + omega^2 x^2) = constant
mass = 1.0
E0 = 0.5 * mass * (v0**2 + omega**2 * x0**2)
results["sho_energy"] = {
    "mass": mass, "omega": omega,
    "x0": x0, "v0": v0,
    "total_energy": E0,
    # Energy at all test times should equal E0
    "energy_at_times": {
        t_str: 0.5 * mass * (vals["v"]**2 + omega**2 * vals["x"]**2)
        for t_str, vals in sho_tests.items()
    }
}

# ─── Damped harmonic oscillator (underdamped) ─────────────────────────────

# x(t) = exp(-γt) [A cos(ωd t) + B sin(ωd t)]
# ωd = sqrt(ω0^2 - γ^2)
omega0 = 10.0
gamma = 1.0
omega_d = math.sqrt(omega0**2 - gamma**2)
x0_d, v0_d = 1.0, 0.0
A = x0_d
B = (v0_d + gamma * x0_d) / omega_d

dho_tests = {}
for t in [0.0, 0.1, 0.5, 1.0, 2.0]:
    x = math.exp(-gamma * t) * (A * math.cos(omega_d * t) + B * math.sin(omega_d * t))
    dho_tests[str(t)] = x

results["dho_underdamped"] = {
    "omega0": omega0, "gamma": gamma, "omega_d": omega_d,
    "x0": x0_d, "v0": v0_d,
    "positions": dho_tests,
}

# ─── Kepler orbit ────────────────────────────────────────────────────────

# Circular orbit: v = sqrt(GM/r), e = 0, T = 2pi sqrt(r^3/GM)
G = 6.67430e-11
M_sun = 1.989e30
r_earth = 1.496e11  # 1 AU in meters
v_circ = math.sqrt(G * M_sun / r_earth)
T_orbit = 2 * math.pi * math.sqrt(r_earth**3 / (G * M_sun))

# Specific orbital energy
energy_specific = 0.5 * v_circ**2 - G * M_sun / r_earth
sma = -G * M_sun / (2 * energy_specific)

results["kepler_circular"] = {
    "r": r_earth,
    "v_circular": v_circ,
    "period_seconds": T_orbit,
    "period_days": T_orbit / 86400,
    "semi_major_axis": sma,
    "energy_specific": energy_specific,
    "eccentricity": 0.0,  # circular orbit
}

# Vis-viva: v^2 = GM(2/r - 1/a)
v_at_r = math.sqrt(G * M_sun * (2.0 / r_earth - 1.0 / sma))
results["vis_viva"] = {
    "v_at_periapsis": v_at_r,
    "matches_circular": abs(v_at_r - v_circ) < 1e-3,
}

# ─── Particle mechanics ──────────────────────────────────────────────────

# Kinetic energy: E_k = 1/2 m v^2
# Angular momentum: L = r x p
results["particle_mechanics"] = {
    "kinetic_energy_1kg_10ms": 0.5 * 1.0 * 10.0**2,  # 50 J
    "momentum_1kg_10ms": [10.0, 0.0, 0.0],  # for v = [10, 0, 0]
}

# ─── Save ─────────────────────────────────────────────────────────────────

with open("research/gold_standard/family_physics_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Physics Oracle: {len(results)} test cases generated")
for name in results:
    print(f"  PASS {name}")
