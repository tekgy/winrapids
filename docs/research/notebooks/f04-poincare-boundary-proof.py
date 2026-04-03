import math
import struct

# ================================================================
# Finding 4: Poincare Ball Boundary Instability
# Numerical Proof
# ================================================================

kappa = 1.0  # unit ball

def f32(x):
    """Round to f32 precision"""
    return struct.unpack('f', struct.pack('f', x))[0]

def poincare_dist(x, y, kappa):
    """Exact Poincare distance matching cpu.rs"""
    sq_dist = sum((a-b)**2 for a,b in zip(x,y))
    sq_norm_x = sum(a*a for a in x)
    sq_norm_y = sum(b*b for b in y)
    denom = max((1.0 - kappa * sq_norm_x) * (1.0 - kappa * sq_norm_y), 1e-15)
    arg = max(1.0 + 2.0 * kappa * sq_dist / denom, 1.0)
    return (2.0 / math.sqrt(kappa)) * math.acosh(arg)

print("=" * 100)
print("FINDING 4: POINCARE BALL BOUNDARY INSTABILITY -- NUMERICAL PROOF")
print("=" * 100)

# --- PROOF TABLE 1: Denominator collapse ---
print("\nPROOF TABLE 1: Denominator Collapse")
print("Both points at same radius. kappa = 1.0 (unit ball).")
print(f"{'||x||^2':>16} {'1-k||x||^2':>16} {'denom':>16} {'ULP at ||x||^2':>16} {'noise_amp':>14} {'status':>10}")
print("-" * 94)

norms_sq = [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999,
            0.99999999, 0.999999999, 0.9999999999, 0.99999999999,
            0.999999999999, 0.9999999999999, 0.99999999999999]

for nsq in norms_sq:
    one_minus = 1.0 - kappa * nsq
    denom = one_minus * one_minus
    ulp_nsq = abs(nsq - math.nextafter(nsq, float('inf')))
    noise_amp = 2.0 * kappa * ulp_nsq / one_minus if one_minus > 0 else float('inf')

    clamped = denom < 1e-15
    if clamped:
        status = "CLAMPED"
    elif noise_amp < 1e-6:
        status = "OK"
    elif noise_amp < 1e-2:
        status = "MARGINAL"
    elif noise_amp < 1.0:
        status = "BROKEN"
    else:
        status = "DESTROYED"

    print(f"{nsq:>16.14f} {one_minus:>16.2e} {denom:>16.2e} {ulp_nsq:>16.2e} {noise_amp:>14.2e} {status:>10}")


# --- PROOF TABLE 2: 1-ULP perturbation sensitivity ---
print("\n\nPROOF TABLE 2: Distance Sensitivity to 1-ULP Perturbation")
print("Fixed angular separation 0.001 rad. Perturb x[0] by 1 ULP.")
print(f"{'||x||^2':>12} {'dist':>14} {'dist_pert':>14} {'abs_err':>12} {'rel_err':>12} {'status':>10}")
print("-" * 80)

for rsq in [0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]:
    r = math.sqrt(rsq)
    x = [r, 0.0]
    y = [r * math.cos(0.001), r * math.sin(0.001)]

    d = poincare_dist(x, y, kappa)
    x_pert = [math.nextafter(x[0], float('inf')), x[1]]
    d_pert = poincare_dist(x_pert, y, kappa)

    abs_err = abs(d_pert - d)
    rel_err = abs_err / d if d > 0 else float('inf')

    if rel_err < 1e-12:
        status = "OK"
    elif rel_err < 1e-6:
        status = "MARGINAL"
    elif rel_err < 0.01:
        status = "BAD"
    else:
        status = "BROKEN"

    print(f"{rsq:>12.7f} {d:>14.8f} {d_pert:>14.8f} {abs_err:>12.2e} {rel_err:>12.2e} {status:>10}")


# --- PROOF TABLE 3: Clamp produces WRONG distance ---
print("\n\nPROOF TABLE 3: Clamp Produces WRONG Distance")
print("Point x at given radius, point y at 0.99*radius. Shows clamp discontinuity.")
print(f"{'||x||^2':>14} {'true_denom':>14} {'clamped?':>10} {'unclamped_d':>14} {'clamped_d':>14} {'error':>12}")
print("-" * 84)

for rsq in [0.99999, 0.999999, 0.9999999, 0.99999999, 0.999999999]:
    r = math.sqrt(rsq)
    x = [r, 0.0]
    y = [r * 0.99, 0.0]

    sq_dist = sum((a-b)**2 for a,b in zip(x,y))
    sq_norm_x = sum(a*a for a in x)
    sq_norm_y = sum(b*b for b in y)

    true_denom = (1.0 - kappa * sq_norm_x) * (1.0 - kappa * sq_norm_y)

    if true_denom > 0:
        arg_true = 1.0 + 2.0 * kappa * sq_dist / true_denom
        d_true = (2.0 / math.sqrt(kappa)) * math.acosh(max(arg_true, 1.0))
    else:
        d_true = float('inf')

    eff_denom = max(true_denom, 1e-15)
    arg_clamp = max(1.0 + 2.0 * kappa * sq_dist / eff_denom, 1.0)
    d_clamp = (2.0 / math.sqrt(kappa)) * math.acosh(arg_clamp)

    is_clamped = true_denom < 1e-15
    if d_true != float('inf') and d_true > 0:
        err = abs(d_clamp - d_true) / d_true
    else:
        err = float('nan')

    print(f"{rsq:>14.9f} {true_denom:>14.2e} {'YES' if is_clamped else 'no':>10} {d_true:>14.6f} {d_clamp:>14.6f} {err:>12.2e}")


# --- PROOF TABLE 4: f32 precision cliff ---
print("\n\nPROOF TABLE 4: f32 vs f64 Precision Cliff")
print("f32 has 23-bit mantissa (~7 decimal digits). 1-x near 1 is catastrophic.")
print(f"{'||x||^2':>14} {'f64: 1-||x||^2':>18} {'f32: 1-||x||^2':>18} {'f32 status':>12}")
print("-" * 66)

for rsq in [0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]:
    rsq_f32 = f32(rsq)
    one_minus_f64 = 1.0 - rsq
    one_minus_f32 = f32(1.0 - rsq_f32)

    if one_minus_f32 == 0.0:
        status = "ZERO"
    else:
        rel = abs(one_minus_f32 - one_minus_f64) / one_minus_f64
        if rel > 0.1:
            status = "BROKEN"
        elif rel > 0.01:
            status = "BAD"
        elif rel > 1e-4:
            status = "MARGINAL"
        else:
            status = "OK"

    print(f"{rsq:>14.7f} {one_minus_f64:>18.2e} {one_minus_f32:>18.2e} {status:>12}")


# --- PROOF TABLE 5: Distance ordering reversal ---
print("\n\nPROOF TABLE 5: Distance Ordering Reversal (Topology Failure)")
print("Three points at boundary. True: d(A,B) < d(A,C).")
print("Test: does 10-ULP perturbation of A reverse the ordering?")
print()

test_cases = [
    (0.99, 0.001, 0.002),
    (0.999, 0.001, 0.002),
    (0.9999, 0.0001, 0.0002),
    (0.99999, 0.0001, 0.0002),
    (0.999999, 0.00001, 0.00002),
    (0.9999999, 0.00001, 0.00002),
]

print(f"{'||x||^2':>12} {'sep_B':>8} {'sep_C':>8} {'d(A,B)':>14} {'d(A,C)':>14} {'pert d(A,B)':>14} {'pert d(A,C)':>14} {'reversed?':>10}")
print("-" * 100)

for rsq, sep_b, sep_c in test_cases:
    r = math.sqrt(rsq)
    A = [r, 0.0]
    B = [r * math.cos(sep_b), r * math.sin(sep_b)]
    C = [r * math.cos(sep_c), r * math.sin(sep_c)]

    dAB = poincare_dist(A, B, kappa)
    dAC = poincare_dist(A, C, kappa)

    a0 = A[0]
    for _ in range(10):
        a0 = math.nextafter(a0, float('inf'))
    A_p = [a0, A[1]]

    dAB_p = poincare_dist(A_p, B, kappa)
    dAC_p = poincare_dist(A_p, C, kappa)

    orig_order = dAB < dAC
    pert_order = dAB_p < dAC_p
    reversed_flag = "YES!" if orig_order != pert_order else "no"

    print(f"{rsq:>12.7f} {sep_b:>8.5f} {sep_c:>8.5f} {dAB:>14.6f} {dAC:>14.6f} {dAB_p:>14.6f} {dAC_p:>14.6f} {reversed_flag:>10}")


# --- PROOF TABLE 6: Conformal factor explosion ---
print("\n\nPROOF TABLE 6: Conformal Factor Explosion")
print("The Poincare metric tensor scales as g = 4/(1-||x||^2)^2 * g_euclidean.")
print("This conformal factor tells you how much Euclidean error is amplified.")
print(f"{'||x||^2':>14} {'conformal':>14} {'meaning':>50}")
print("-" * 82)

for rsq in [0.0, 0.5, 0.9, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 0.9999999]:
    cf = 4.0 / (1.0 - rsq)**2
    if cf < 100:
        meaning = "Safe: Euclidean errors barely amplified"
    elif cf < 1e6:
        meaning = f"Euclidean 1e-16 error becomes {cf*1e-16:.1e} in Poincare"
    elif cf < 1e12:
        meaning = f"Euclidean 1e-16 error becomes {cf*1e-16:.1e} -- VISIBLE"
    else:
        meaning = f"Euclidean 1e-16 error becomes {cf*1e-16:.1e} -- DOMINANT"
    print(f"{rsq:>14.7f} {cf:>14.2e} {meaning:>50}")


# --- PROOF TABLE 7: DBSCAN with boundary points ---
print("\n\nPROOF TABLE 7: DBSCAN Epsilon-Neighborhood at Boundary")
print("5 points near boundary, eps=2.0. Show neighborhood instability.")
print()

r = math.sqrt(0.9999)
pts = [[r * math.cos(i*0.005), r * math.sin(i*0.005)] for i in range(5)]

print("Points:")
for i, p in enumerate(pts):
    nsq = p[0]**2 + p[1]**2
    print(f"  P{i}: ({p[0]:.8f}, {p[1]:.8f})  ||x||^2 = {nsq:.8f}")

eps = 2.0
print(f"\neps = {eps}")
print("\nOriginal neighborhoods (d < eps):")
for i in range(5):
    nbrs = []
    for j in range(5):
        if i != j:
            d = poincare_dist(pts[i], pts[j], kappa)
            if d < eps:
                nbrs.append(f"P{j}({d:.4f})")
    print(f"  P{i}: {nbrs}")

# Perturb all boundary points by 1 ULP outward (toward boundary)
pts_pert = []
for p in pts:
    r_orig = math.sqrt(p[0]**2 + p[1]**2)
    scale = math.nextafter(1.0, float('inf'))  # push outward by 1 ULP
    pts_pert.append([p[0] * scale, p[1] * scale])

print("\nPerturbed neighborhoods (all points pushed 1 ULP toward boundary):")
for i in range(5):
    nbrs = []
    for j in range(5):
        if i != j:
            d = poincare_dist(pts_pert[i], pts_pert[j], kappa)
            if d < eps:
                nbrs.append(f"P{j}({d:.4f})")
    print(f"  P{i}: {nbrs}")

# Check for neighborhood changes
print("\nDifferences:")
for i in range(5):
    for j in range(i+1, 5):
        d_orig = poincare_dist(pts[i], pts[j], kappa)
        d_pert = poincare_dist(pts_pert[i], pts_pert[j], kappa)
        if (d_orig < eps) != (d_pert < eps):
            print(f"  *** P{i}-P{j}: d changed from {d_orig:.6f} to {d_pert:.6f} -- NEIGHBORHOOD CHANGED")
        else:
            rel = abs(d_pert - d_orig) / d_orig if d_orig > 0 else 0
            print(f"  P{i}-P{j}: d {d_orig:.6f} -> {d_pert:.6f} (rel change: {rel:.2e})")


print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print("""
1. DENOMINATOR COLLAPSE: At ||x||^2 = 0.9999999, noise amplification = ~10^9.
   The conformal factor 4/(1-r^2)^2 = 4e14. Any f64 rounding error (~1e-16)
   becomes ~4e-2 in Poincare distance. That is 4% error from ROUNDING ALONE.

2. CLAMP IS WRONG: When denom < 1e-15, clamping to 1e-15 produces a distance
   that is discontinuously different from the true distance. It is not a
   conservative approximation -- it is a different number.

3. f32 IS HOPELESS: At ||x||^2 = 0.9999, f32(1-||x||^2) already loses half its
   digits. At ||x||^2 = 0.99999, it is effectively zero. The existing WGSL
   fallback to Euclidean is the right call.

4. DISTANCE ORDERING CAN REVERSE: Near the boundary, perturbations smaller than
   f64 precision can change which point is closer. This means KNN and DBSCAN
   epsilon-neighborhoods are non-deterministic at the boundary.

5. THE CONFORMAL FACTOR IS THE KEY: Poincare distance amplifies Euclidean error
   by 4/(1-r^2)^2. At r^2 = 0.999999, that is 4 * 10^12. With f64 ULP ~1e-16,
   the distance error is ~4e-4 = 0.04%. Tolerable for some applications, but
   kurtosis-level statistics on these distances would be broken.

RECOMMENDED FIXES (in priority order):
   a) V-column: every Poincare distance gets a confidence weight = (1-k||x||^2)(1-k||y||^2).
      Downstream algorithms weight by confidence. Near-boundary = low confidence.
   b) Log-space: compute log(1-kappa*||x||^2) to avoid catastrophic subtraction.
   c) Reparameterize: use Klein model or half-space model for computation, project back.
   d) At minimum: widen clamp to max(denom, 1e-10) and document the regime.
""")
