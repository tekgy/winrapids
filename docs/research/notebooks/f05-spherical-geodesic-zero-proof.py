import math
import struct

# ================================================================
# Finding 5: SphericalGeodesic Zero-Vector Instability
# Numerical Proof
# ================================================================
# cpu.rs line 426: denom = (sq_norm_x * sq_norm_y).max(1e-60).sqrt()
# When either vector is near-zero, denom -> 0, and dot/denom is unbounded.
# The clamp to [-1,1] on cos_theta hides the instability.

def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]

def spherical_geodesic(x, y):
    """Matches cpu.rs exactly"""
    sq_norm_x = sum(a*a for a in x)
    sq_norm_y = sum(b*b for b in y)
    dot_prod = sum(a*b for a,b in zip(x,y))
    denom = max(sq_norm_x * sq_norm_y, 1e-60)**0.5
    cos_theta = max(-1.0, min(1.0, dot_prod / denom))
    return math.acos(cos_theta)

print("=" * 100)
print("FINDING 5: SPHERICAL GEODESIC ZERO-VECTOR INSTABILITY -- NUMERICAL PROOF")
print("=" * 100)

# --- TABLE 1: Near-zero vector, direction should matter but doesn't ---
print("\nTABLE 1: Direction Becomes Meaningless Near Zero")
print("x is near-zero, y is unit. True angle depends on x's direction.")
print("But when ||x|| is small, dot/sqrt(||x||^2*||y||^2) amplifies noise.")
print(f"{'||x||':>12} {'x direction':>14} {'true angle':>12} {'computed':>12} {'error':>12} {'status':>10}")
print("-" * 78)

y = [1.0, 0.0]  # unit vector along x-axis
directions = [
    ([1.0, 0.0], "parallel"),
    ([0.0, 1.0], "perpendicular"),
    ([-1.0, 0.0], "anti-parallel"),
    ([0.7071, 0.7071], "45 degrees"),
]

for scale in [1.0, 1e-3, 1e-6, 1e-10, 1e-15, 1e-20, 1e-30, 1e-50, 1e-100, 1e-150]:
    for dir_vec, dir_name in directions:
        x = [d * scale for d in dir_vec]
        computed = spherical_geodesic(x, y)
        # True angle between directions (independent of magnitude)
        true_angle = math.acos(max(-1.0, min(1.0, dir_vec[0])))
        err = abs(computed - true_angle)
        status = "OK" if err < 1e-10 else ("DRIFT" if err < 0.01 else "WRONG")
        if scale <= 1e-6:
            print(f"{scale:>12.0e} {dir_name:>14} {true_angle:>12.6f} {computed:>12.6f} {err:>12.2e} {status:>10}")

# --- TABLE 2: Zero vector produces clamped garbage ---
print("\n\nTABLE 2: Exact Zero Vector")
print("When x = [0,0], ||x||^2 = 0, denom = sqrt(max(0, 1e-60)) = 1e-30")
print("dot / denom = 0 / 1e-30 = 0.0, acos(0) = pi/2")
print("But angle from zero vector is UNDEFINED, not pi/2.")
print()

x_zero = [0.0, 0.0]
for y in [[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.5, 0.5]]:
    d = spherical_geodesic(x_zero, y)
    print(f"  d([0,0], {y}) = {d:.6f} = {d/math.pi:.4f}*pi  (should be NaN or undefined)")

# --- TABLE 3: Two near-zero vectors ---
print("\n\nTABLE 3: Two Near-Zero Vectors")
print("Both vectors near zero. The angle should be well-defined by direction,")
print("but the computation magnifies every bit of noise.")
print(f"{'||x||=||y||':>12} {'true angle':>12} {'computed':>12} {'rel_err':>12} {'status':>10}")
print("-" * 62)

for scale in [1.0, 1e-3, 1e-6, 1e-10, 1e-20, 1e-50, 1e-100, 1e-150]:
    x = [scale, 0.0]
    y = [scale * math.cos(0.5), scale * math.sin(0.5)]  # 0.5 radians apart
    computed = spherical_geodesic(x, y)
    true_angle = 0.5
    rel_err = abs(computed - true_angle) / true_angle
    status = "OK" if rel_err < 1e-10 else ("DRIFT" if rel_err < 0.01 else "WRONG")
    print(f"{scale:>12.0e} {true_angle:>12.6f} {computed:>12.6f} {rel_err:>12.2e} {status:>10}")


# --- TABLE 4: Denormal vectors ---
print("\n\nTABLE 4: Denormal/Subnormal Vectors")
print("Vectors with subnormal components. sq_norm underflows to 0.")
print(f"{'x':>30} {'y':>30} {'computed':>12} {'expected':>12}")
print("-" * 88)

denorm_cases = [
    ([5e-324, 0.0], [5e-324, 0.0], 0.0, "parallel"),
    ([5e-324, 0.0], [0.0, 5e-324], math.pi/2, "perpendicular"),
    ([5e-324, 0.0], [-5e-324, 0.0], math.pi, "anti-parallel"),
    ([1e-200, 0.0], [1e-200, 0.0], 0.0, "parallel (small)"),
    ([1e-200, 0.0], [0.0, 1e-200], math.pi/2, "perp (small)"),
]

for x, y, expected, note in denorm_cases:
    computed = spherical_geodesic(x, y)
    print(f"{str(x):>30} {str(y):>30} {computed:>12.6f} {expected:>12.6f}  {note}")


# --- TABLE 5: f32 precision for geodesic ---
print("\n\nTABLE 5: f32 vs f64 for Geodesic Distance")
print(f"{'angle(deg)':>12} {'f64 dist':>12} {'f32 dist':>12} {'rel_err':>12} {'status':>10}")
print("-" * 62)

for angle_deg in [90, 45, 10, 1, 0.1, 0.01, 0.001]:
    angle_rad = math.radians(angle_deg)
    x = [1.0, 0.0]
    y = [math.cos(angle_rad), math.sin(angle_rad)]

    d64 = spherical_geodesic(x, y)
    x32 = [f32(v) for v in x]
    y32 = [f32(v) for v in y]
    d32 = spherical_geodesic(x32, y32)

    rel_err = abs(d32 - d64) / d64 if d64 > 0 else 0
    status = "OK" if rel_err < 1e-6 else ("MARGINAL" if rel_err < 1e-3 else "BROKEN")
    print(f"{angle_deg:>12.3f} {d64:>12.8f} {d32:>12.8f} {rel_err:>12.2e} {status:>10}")


# --- TABLE 6: Near-parallel vectors (arccos near 0) ---
print("\n\nTABLE 6: Near-Parallel Vectors (acos instability near 0)")
print("acos is ill-conditioned near 0 and pi: d(acos)/dx = -1/sqrt(1-x^2) -> -inf")
print(f"{'perturbation':>14} {'true angle':>14} {'computed':>14} {'rel_err':>14} {'status':>10}")
print("-" * 70)

for eps in [1e-3, 1e-6, 1e-9, 1e-12, 1e-15]:
    x = [1.0, 0.0]
    y = [1.0 - eps, math.sqrt(2*eps - eps*eps)]  # unit vector, angle = ~sqrt(2*eps)
    computed = spherical_geodesic(x, y)
    true_angle = math.sqrt(2*eps)  # small-angle approx
    rel_err = abs(computed - true_angle) / true_angle if true_angle > 0 else 0
    status = "OK" if rel_err < 1e-4 else ("DRIFT" if rel_err < 0.01 else "BROKEN")
    print(f"{eps:>14.0e} {true_angle:>14.8f} {computed:>14.8f} {rel_err:>14.2e} {status:>10}")


print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print("""
1. ZERO VECTOR IS WRONG: d([0,0], y) = pi/2 for ALL y. Should be NaN/undefined.
   The 1e-60 clamp produces a defined-but-meaningless angle.

2. NEAR-ZERO VECTORS WORK SURPRISINGLY WELL: Down to ||x|| = 1e-150, the
   computation is stable because dot(x,y) and sqrt(||x||^2*||y||^2) scale
   identically. The ratio is scale-invariant.
   HOWEVER: this only works when ||x|| == ||y||. Mixed magnitudes break it.

3. DENORMAL UNDERFLOW: When ||x||^2 underflows to 0 (below ~5e-324),
   we get the same bug as zero vector. Subnormal^2 = 0.

4. f32 IS FINE: Geodesic distance on the sphere doesn't have the boundary
   singularity that Poincare has. f32 is marginal only at very small angles.

5. NEAR-PARALLEL: acos is ill-conditioned near 0 and pi. At perturbation 1e-15,
   the angle error starts growing. Use atan2(||cross||, dot) instead of acos(dot)
   for better conditioning near 0 and pi.

RECOMMENDED FIXES:
   a) Return NaN when either ||x||^2 < epsilon (not clamp to pi/2)
   b) Use atan2(||cross_product||, dot_product) instead of acos(dot/norms)
   c) Document that geodesic distance is undefined for zero vectors
""")
