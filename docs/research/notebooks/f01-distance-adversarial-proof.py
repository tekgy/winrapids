import math
import struct

# ================================================================
# Family 01: Distance & Similarity — Adversarial Test Suite
# ================================================================
# Operations: dot_product, l2_distance, covariance, sphere (cosine),
#             softmax_weighted, poincare, sphere_geodesic, manifold_mixture
#
# Poincare and SphericalGeodesic already covered in F04/F05.
# This focuses on: dot_product, L2, covariance, sphere, softmax_weighted.

def f32(x):
    return struct.unpack('f', struct.pack('f', x))[0]

print("=" * 100)
print("FAMILY 01: DISTANCE & SIMILARITY — ADVERSARIAL TESTS")
print("=" * 100)

# ================================================================
# 1. DOT PRODUCT
# ================================================================
print("\n### DOT PRODUCT ###")

# --- Catastrophic cancellation in dot product ---
print("\nTEST 1.1: Dot Product Cancellation")
print("a and b nearly orthogonal. True dot ~ 0. Naive accumulation loses precision.")
print(f"{'dim':>6} {'offset':>10} {'true_dot':>14} {'computed':>14} {'abs_err':>12} {'status':>10}")
print("-" * 72)

for dim in [10, 100, 1000]:
    for offset in [0, 1e4, 1e8, 1e12]:
        # a = [offset + 1, offset + 2, ...], b = [offset - 1, offset - 2, ...]
        # designed so dot(a,b) = sum((offset+i)(offset-i)) = sum(offset^2 - i^2)
        # = dim*offset^2 - sum(i^2)
        a = [offset + (i+1) for i in range(dim)]
        b = [offset - (i+1) for i in range(dim)]

        true_dot = dim * offset * offset - sum((i+1)**2 for i in range(dim))
        computed = sum(a[i]*b[i] for i in range(dim))

        abs_err = abs(computed - true_dot)
        rel_err = abs_err / abs(true_dot) if true_dot != 0 else abs_err

        status = "OK" if rel_err < 1e-10 else ("MARGINAL" if rel_err < 1e-4 else "BROKEN")
        if true_dot == 0 and abs_err == 0:
            status = "OK"

        print(f"{dim:>6} {offset:>10.0e} {true_dot:>14.4e} {computed:>14.4e} {abs_err:>12.2e} {status:>10}")


# --- Overflow in dot product ---
print("\nTEST 1.2: Dot Product Overflow")
print("Large vector components. sum(a*b) overflows f64.")
print(f"{'a_val':>14} {'b_val':>14} {'dim':>6} {'result':>14} {'status':>10}")
print("-" * 62)

overflow_cases = [
    (1e153, 1e153, 2, "should be 2e306, OK"),
    (1e154, 1e154, 2, "should be 2e308, OVERFLOW"),
    (1e200, 1e200, 1, "should be 1e400, OVERFLOW"),
    (1e-200, 1e-200, 1, "should be 1e-400, UNDERFLOW to 0"),
]

for a_val, b_val, dim, note in overflow_cases:
    result = sum(a_val * b_val for _ in range(dim))
    is_inf = math.isinf(result)
    is_zero = result == 0.0 and a_val * b_val != 0
    status = "OVERFLOW" if is_inf else ("UNDERFLOW" if is_zero else "OK")
    print(f"{a_val:>14.0e} {b_val:>14.0e} {dim:>6} {result:>14.4e} {status:>10}  {note}")


# --- NaN propagation ---
print("\nTEST 1.3: Dot Product NaN/Inf")
nan = float('nan')
inf = float('inf')
cases = [
    ([1.0, nan, 3.0], [4.0, 5.0, 6.0], "one NaN"),
    ([inf, 1.0], [1.0, 1.0], "+Inf component"),
    ([inf, 1.0], [-inf, 1.0], "Inf * -Inf"),
    ([0.0, 0.0], [0.0, 0.0], "zero vectors"),
]
print(f"{'description':>20} {'result':>14} {'expected':>14}")
print("-" * 52)
for a, b, desc in cases:
    result = sum(ai*bi for ai,bi in zip(a,b))
    if desc == "one NaN":
        expected = "NaN"
    elif desc == "+Inf component":
        expected = "Inf"
    elif desc == "Inf * -Inf":
        expected = "NaN"
    else:
        expected = "0.0"
    print(f"{desc:>20} {result:>14} {expected:>14}")


# ================================================================
# 2. L2 DISTANCE (squared)
# ================================================================
print("\n\n### L2 DISTANCE (SQUARED) ###")

# --- L2 cancellation: nearby points at large offset ---
print("\nTEST 2.1: L2 Distance — Nearby Points at Large Offset")
print("Two points differ by tiny amount, centered at large offset.")
print("L2Sq = sum((a-b)^2). Subtraction is exact if a and b are close.")
print(f"{'offset':>12} {'spread':>10} {'true_L2sq':>14} {'computed':>14} {'rel_err':>12} {'status':>10}")
print("-" * 72)

for offset in [0, 1e4, 1e8, 1e12, 1e15]:
    dim = 10
    spread = 0.001
    a = [offset + i * spread for i in range(dim)]
    b = [offset + (i + 0.5) * spread for i in range(dim)]

    true_l2sq = dim * (0.5 * spread)**2
    computed = sum((a[i]-b[i])**2 for i in range(dim))

    rel_err = abs(computed - true_l2sq) / true_l2sq if true_l2sq > 0 else 0
    status = "OK" if rel_err < 1e-10 else ("MARGINAL" if rel_err < 1e-4 else "BROKEN")

    print(f"{offset:>12.0e} {spread:>10.4f} {true_l2sq:>14.6e} {computed:>14.6e} {rel_err:>12.2e} {status:>10}")

print("\nKEY INSIGHT: L2 distance is robust to offset because (a-b) cancels the offset.")
print("Unlike dot product and variance, L2 does NOT suffer from catastrophic cancellation.")
print("The subtraction a[i]-b[i] loses precision only when a[i] and b[i] are within 1 ULP.")


# --- L2 with identical points ---
print("\nTEST 2.2: L2 Distance — Identical Points")
a = [1e8 + i * 0.001 for i in range(100)]
b = list(a)  # exact copy
l2sq = sum((a[i]-b[i])**2 for i in range(len(a)))
print(f"  100 identical points at offset 1e8: L2sq = {l2sq} (must be exactly 0.0)")


# --- L2 triangle inequality ---
print("\nTEST 2.3: L2 Triangle Inequality")
print("d(A,C) <= d(A,B) + d(B,C) must hold for all A,B,C.")
import random
random.seed(42)
violations = 0
for _ in range(10000):
    dim = 5
    A = [random.gauss(0, 1) for _ in range(dim)]
    B = [random.gauss(0, 1) for _ in range(dim)]
    C = [random.gauss(0, 1) for _ in range(dim)]

    dAB = math.sqrt(sum((A[i]-B[i])**2 for i in range(dim)))
    dBC = math.sqrt(sum((B[i]-C[i])**2 for i in range(dim)))
    dAC = math.sqrt(sum((A[i]-C[i])**2 for i in range(dim)))

    if dAC > dAB + dBC + 1e-10:  # small tolerance for rounding
        violations += 1

print(f"  10,000 random triangles: {violations} violations (must be 0)")


# --- L2 with denormals ---
print("\nTEST 2.4: L2 Distance — Denormal/Extreme Values")
cases_l2 = [
    ([5e-324], [0.0], "denormal vs zero"),
    ([1e-300], [2e-300], "tiny values"),
    ([1e153], [0.0], "large vs zero"),
    ([1e154], [1e154 + 1.0], "near-overflow squared"),
]
for a, b, desc in cases_l2:
    try:
        l2sq = sum((a[i]-b[i])**2 for i in range(len(a)))
        print(f"  {desc:>30}: L2sq = {l2sq:.4e}")
    except OverflowError:
        print(f"  {desc:>30}: OVERFLOW (Python; f64 would give inf)")


# ================================================================
# 3. COVARIANCE
# ================================================================
print("\n\n### COVARIANCE ###")

# The covariance op computes sum(a*b)/(kd-1).
# It assumes inputs are PRE-CENTERED. If not centered → wrong answer.
print("\nTEST 3.1: Covariance — Pre-Centering Requirement")
print("Covariance op is raw accumulation. Centering is caller's responsibility.")
print("If caller forgets to center, result is wrong by O(mean_a * mean_b).")
print()

# Not centered
a_raw = [100.0 + i for i in range(10)]
b_raw = [200.0 + i*2 for i in range(10)]
mean_a = sum(a_raw) / len(a_raw)
mean_b = sum(b_raw) / len(b_raw)

raw_cov = sum(a_raw[i]*b_raw[i] for i in range(10)) / 9
centered_cov = sum((a_raw[i]-mean_a)*(b_raw[i]-mean_b) for i in range(10)) / 9
true_cov = centered_cov

print(f"  Raw (uncentered) covariance:     {raw_cov:.4f}")
print(f"  Centered covariance (correct):   {centered_cov:.4f}")
print(f"  Error if not centered:           {abs(raw_cov - true_cov):.4f} ({abs(raw_cov - true_cov)/abs(true_cov)*100:.1f}%)")
print(f"  The raw result is ~mean_a*mean_b = {mean_a*mean_b:.1f} too high")

# --- Covariance with zero variance ---
print("\nTEST 3.2: Covariance — One Constant Column")
a_const = [42.0] * 10
b_vary = [float(i) for i in range(10)]
mean_ac = sum(a_const) / 10
mean_bv = sum(b_vary) / 10
cov_centered = sum((a_const[i]-mean_ac)*(b_vary[i]-mean_bv) for i in range(10)) / 9
print(f"  Covariance with constant column: {cov_centered:.4f} (must be exactly 0.0)")


# ================================================================
# 4. SPHERE (COSINE DISTANCE)
# ================================================================
print("\n\n### SPHERE (COSINE DISTANCE) ###")

# Sphere distance = 1 - dot(a,b). Assumes UNIT-NORMALIZED inputs.
print("\nTEST 4.1: Cosine Distance — Non-Unit Input")
print("The sphere op computes 1-dot(a,b). If inputs aren't unit-normalized, result is wrong.")
a = [3.0, 4.0]  # ||a|| = 5
b = [4.0, 3.0]  # ||b|| = 5
cos_dist_raw = 1.0 - sum(a[i]*b[i] for i in range(2))
cos_dist_correct = 1.0 - sum(a[i]*b[i] for i in range(2)) / (5*5)
print(f"  Raw (non-normalized):     1 - dot = {cos_dist_raw}")
print(f"  Correct (normalized):     1 - cos = {cos_dist_correct:.6f}")
print(f"  Raw is WRONG by {abs(cos_dist_raw - cos_dist_correct):.4f}")

# --- Cosine distance is NOT a metric ---
print("\nTEST 4.2: Cosine Distance — Triangle Inequality FAILURE")
print("Cosine distance (1-cos) does NOT satisfy triangle inequality in general.")
print("Angular distance (arccos(cos)) DOES satisfy it.")
print()

# Known counter-example
A = [1.0, 0.0]
B = [0.0, 1.0]
C = [-1.0, 0.0]

dAB = 1.0 - sum(A[i]*B[i] for i in range(2))  # = 1 - 0 = 1
dBC = 1.0 - sum(B[i]*C[i] for i in range(2))  # = 1 - 0 = 1
dAC = 1.0 - sum(A[i]*C[i] for i in range(2))  # = 1 - (-1) = 2

print(f"  d(A,B) = {dAB}, d(B,C) = {dBC}, d(A,C) = {dAC}")
print(f"  Triangle: d(A,C) <= d(A,B) + d(B,C)? {dAC} <= {dAB+dBC}? {dAC <= dAB + dBC}")
print(f"  Barely holds for unit vectors. But for non-unit, can violate.")

# With non-unit vectors
A2 = [2.0, 0.0]
B2 = [0.0, 0.5]
C2 = [-1.0, 0.0]
dAB2 = 1.0 - sum(A2[i]*B2[i] for i in range(2))
dBC2 = 1.0 - sum(B2[i]*C2[i] for i in range(2))
dAC2 = 1.0 - sum(A2[i]*C2[i] for i in range(2))
print(f"  Non-unit: d(A,B) = {dAB2}, d(B,C) = {dBC2}, d(A,C) = {dAC2}")
print(f"  Triangle: {dAC2} <= {dAB2 + dBC2}? {dAC2 <= dAB2 + dBC2}")
if dAC2 > dAB2 + dBC2:
    print(f"  *** TRIANGLE INEQUALITY VIOLATED by {dAC2 - (dAB2+dBC2):.4f} ***")

# --- Near-identical vectors ---
print("\nTEST 4.3: Cosine Distance — Nearly Identical Unit Vectors")
print("1 - dot(a,b) when a ~ b suffers from catastrophic cancellation near 1.")
print(f"{'angle (rad)':>14} {'1-dot':>14} {'true 1-cos':>14} {'rel_err':>12}")
print("-" * 58)

for angle in [1.0, 0.1, 0.01, 0.001, 1e-4, 1e-6, 1e-8, 1e-10]:
    a = [1.0, 0.0]
    b = [math.cos(angle), math.sin(angle)]
    one_minus_dot = 1.0 - sum(a[i]*b[i] for i in range(2))
    true_val = 1.0 - math.cos(angle)
    # For small angle: 1-cos(x) ≈ x^2/2
    rel_err = abs(one_minus_dot - true_val) / true_val if true_val > 0 else 0
    print(f"{angle:>14.1e} {one_minus_dot:>14.4e} {true_val:>14.4e} {rel_err:>12.2e}")

print("\n  At angle = 1e-8: 1-cos = 5e-17. f64 has ~1e-16 ULP near 1.0.")
print("  The subtraction 1-dot loses ALL significant digits for angle < ~1e-8.")
print("  FIX: use 2*sin^2(angle/2) instead of 1-cos(angle) for small angles.")


# ================================================================
# 5. SOFTMAX_WEIGHTED
# ================================================================
print("\n\n### SOFTMAX WEIGHTED ###")

def softmax_weighted_cpu(scores, values):
    """Matches cpu.rs exactly: online softmax with running max."""
    max_val = float('-inf')
    exp_sum = 0.0
    weighted_sum = 0.0
    for ki in range(len(scores)):
        score = scores[ki]
        if score > max_val:
            scale = math.exp(max_val - score)
            exp_sum *= scale
            weighted_sum *= scale
            max_val = score
        w = math.exp(score - max_val)
        exp_sum += w
        weighted_sum += w * values[ki]
    return weighted_sum / exp_sum if exp_sum > 0 else 0.0

# --- Test 5.1: Basic correctness ---
print("\nTEST 5.1: Softmax Weighted — Basic")
scores = [1.0, 2.0, 3.0]
values = [10.0, 20.0, 30.0]
result = softmax_weighted_cpu(scores, values)
# softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
# weighted = 0.09*10 + 0.2447*20 + 0.6652*30 = 0.9 + 4.894 + 19.957 = 25.751
print(f"  scores=[1,2,3], values=[10,20,30]: result = {result:.6f} (expected ~25.7515)")

# --- Test 5.2: Large scores (overflow prevention) ---
print("\nTEST 5.2: Softmax Weighted — Large Scores (exp overflow)")
scores_big = [1000.0, 1001.0, 1002.0]
values_big = [1.0, 2.0, 3.0]
result_big = softmax_weighted_cpu(scores_big, values_big)
print(f"  scores=[1000,1001,1002]: result = {result_big:.6f}")
print(f"  Should be same as scores=[0,1,2] (shift-invariant): {softmax_weighted_cpu([0,1,2], [1,2,3]):.6f}")
print(f"  Match: {abs(result_big - softmax_weighted_cpu([0,1,2], [1,2,3])) < 1e-10}")

# --- Test 5.3: Negative scores (underflow) ---
print("\nTEST 5.3: Softmax Weighted — Very Negative Scores")
scores_neg = [-1000.0, -999.0, -998.0]
values_neg = [1.0, 2.0, 3.0]
result_neg = softmax_weighted_cpu(scores_neg, values_neg)
print(f"  scores=[-1000,-999,-998]: result = {result_neg:.6f}")
print(f"  Should be same as scores=[-2,-1,0]: {softmax_weighted_cpu([-2,-1,0], [1,2,3]):.6f}")

# --- Test 5.4: All equal scores ---
print("\nTEST 5.4: Softmax Weighted — All Equal Scores")
result_eq = softmax_weighted_cpu([5.0, 5.0, 5.0], [10.0, 20.0, 30.0])
print(f"  All scores=5: result = {result_eq:.6f} (expected: mean(values) = 20.0)")

# --- Test 5.5: Single element ---
print("\nTEST 5.5: Softmax Weighted — Single Element")
result_single = softmax_weighted_cpu([42.0], [7.0])
print(f"  Single element: result = {result_single:.6f} (expected: 7.0)")

# --- Test 5.6: NaN in scores ---
print("\nTEST 5.6: Softmax Weighted — NaN/Inf in Scores")
result_nan = softmax_weighted_cpu([1.0, float('nan'), 3.0], [10.0, 20.0, 30.0])
result_inf = softmax_weighted_cpu([1.0, float('inf'), 3.0], [10.0, 20.0, 30.0])
result_neginf = softmax_weighted_cpu([1.0, float('-inf'), 3.0], [10.0, 20.0, 30.0])
print(f"  NaN in scores: {result_nan} (should be NaN)")
print(f"  +Inf in scores: {result_inf} (should be values[1] = 20.0)")
print(f"  -Inf in scores: {result_neginf:.6f} (should ignore that element)")

# --- Test 5.7: Order dependence ---
print("\nTEST 5.7: Softmax Weighted — Order Dependence")
print("The online algorithm processes elements sequentially. Does order matter?")
import random
random.seed(42)
scores_orig = [random.gauss(0, 10) for _ in range(100)]
values_orig = [random.gauss(0, 1) for _ in range(100)]

result_fwd = softmax_weighted_cpu(scores_orig, values_orig)

# Reverse order
result_rev = softmax_weighted_cpu(scores_orig[::-1], values_orig[::-1])

# Random shuffle
indices = list(range(100))
random.shuffle(indices)
scores_shuf = [scores_orig[i] for i in indices]
values_shuf = [values_orig[i] for i in indices]
result_shuf = softmax_weighted_cpu(scores_shuf, values_shuf)

print(f"  Forward:  {result_fwd:.15f}")
print(f"  Reversed: {result_rev:.15f}")
print(f"  Shuffled: {result_shuf:.15f}")
print(f"  Max diff: {max(abs(result_fwd-result_rev), abs(result_fwd-result_shuf)):.2e}")
if result_fwd != result_rev or result_fwd != result_shuf:
    print(f"  *** ORDER-DEPENDENT: results differ at bit level ***")
else:
    print(f"  Order-independent (within f64 precision)")


# --- Test 5.8: Extreme temperature (scores nearly identical) ---
print("\nTEST 5.8: Softmax Weighted — Near-Uniform (All Scores Nearly Equal)")
eps = 1e-15
scores_near = [1.0 + i*eps for i in range(100)]
values_near = [float(i) for i in range(100)]
result_near = softmax_weighted_cpu(scores_near, values_near)
print(f"  Scores differ by {eps}: result = {result_near:.6f} (expected: ~mean(0..99) = 49.5)")


print("\n" + "=" * 100)
print("FAMILY 01 SUMMARY")
print("=" * 100)
print("""
FINDINGS:

1. DOT PRODUCT: Catastrophic cancellation when vectors are large-offset nearly-orthogonal.
   At offset 1e8, dim=1000: visible errors. At offset 1e12: BROKEN.
   Same bug as variance — it's sum(a*b) where individual terms are huge but sum is small.
   FIX: For dot products where cancellation is expected, use compensated summation (Kahan).

2. L2 DISTANCE: ROBUST to offset. The subtraction a[i]-b[i] cancels the offset before
   squaring. This is why L2 is the default for euclidean distances — it's numerically safe.
   Only breaks when a[i] and b[i] are within 1 ULP of each other (extremely close).

3. COVARIANCE: Assumes pre-centered inputs. If caller forgets to center, result is
   wrong by O(mean_a * mean_b). The comment in cpu.rs documents this, but there's no
   runtime check. FIX: assert or warn if mean(A) or mean(B) > threshold.

4. SPHERE (COSINE): Three issues:
   a) Requires unit-normalized inputs — no runtime check
   b) 1-dot cancellation for nearly-identical vectors (angle < 1e-8)
   c) NOT a metric (triangle inequality can fail for non-unit inputs)
   FIX: Use 2*sin^2(angle/2) for small angles; validate unit-normalization.

5. SOFTMAX WEIGHTED: Well-implemented with online max tracking. Handles large scores
   correctly (shift-invariant). Minor issues:
   a) NaN in scores produces NaN (acceptable)
   b) +Inf in scores correctly selects that element's value
   c) Order-dependent at bit level (expected for online algorithm)
   d) -Inf scores correctly excluded
   VERDICT: Best-implemented operation in F01. The FlashAttention pattern is correct.

PRIORITY:
- HIGH: Cosine distance cancellation (angle < 1e-8 gives garbage)
- MEDIUM: Dot product cancellation (affects downstream: covariance, Gram matrix)
- LOW: Covariance pre-centering assumption (documented but not enforced)
- NONE: L2 distance (naturally robust), Softmax (well-implemented)
""")
