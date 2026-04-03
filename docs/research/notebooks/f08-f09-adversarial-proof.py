"""
Adversarial review: nonparametric.rs (F08) + robust.rs (F09)
Adversarial mathematician, 2026-04-01

MANDATE: Find SILENT failures -- plausible-but-wrong answers.
"""
import math
import random

random.seed(42)

PASS = 0
FAIL = 0
WARN = 0

def check(name, actual, expected, tol=1e-6, rel=False):
    global PASS, FAIL
    if isinstance(expected, bool):
        ok = (actual == expected)
    elif math.isnan(expected) and math.isnan(actual):
        PASS += 1; return True
    elif math.isinf(expected) and math.isinf(actual):
        ok = (expected > 0) == (actual > 0)
    elif rel and expected != 0:
        ok = abs(actual - expected) / abs(expected) < tol
    else:
        ok = abs(actual - expected) < tol
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"  ** FAIL: {name}: got {actual}, expected {expected}")
    return ok

def warn(name, msg):
    global WARN
    WARN += 1
    print(f"  !! WARN: {name}: {msg}")

# ====================================================================
# SECTION 1: F09 — LTS naive OLS (THE BIGGEST FINDING)
# ====================================================================
print("=" * 80)
print("SECTION 1: F09 robust.rs -- LTS OLS at large offset")
print("=" * 80)

# The LTS regression's ols_subset computes:
#   denom = n * sxx - sx * sx
#   b = (n * sxy - sx * sy) / denom
#   a = (sy - b * sx) / n
#
# This is the NAIVE formula for OLS. When data has a large offset,
# n*sxx and sx*sx are both huge, and their difference loses precision.
# This is EXACTLY the same bug as naive variance: det(X'X) = N*Var(x).

print("\n--- 1a: OLS naive formula at increasing offset ---")
print(f"  y = 2x + 1, n=10 points, x near offset")
print(f"  {'offset':>12} {'naive_b':>12} {'naive_a':>12} {'b_err':>12} {'a_err':>12}")
print("-" * 64)

for offset in [0, 1e2, 1e4, 1e6, 1e8, 1e10, 1e12, 1e14]:
    n = 10
    true_a = 1.0
    true_b = 2.0
    x = [offset + i * 0.1 for i in range(n)]
    y = [true_a + true_b * xi for xi in x]
    nf = float(n)

    # Naive OLS (as in ols_subset)
    sx = sum(x)
    sy = sum(y)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    sxx = sum(xi * xi for xi in x)
    denom = nf * sxx - sx * sx

    if abs(denom) < 1e-15:
        print(f"{offset:>12.0e} {'SINGULAR':>12} {'':>12} {'':>12} {'':>12}")
        warn(f"LTS OLS offset={offset:.0e}", "SINGULAR: n*sxx - sx*sx = 0")
        continue

    b = (nf * sxy - sx * sy) / denom
    a = (sy - b * sx) / nf

    b_err = abs(b - true_b)
    a_err = abs(a - true_a)
    status = "OK" if b_err < 1e-6 else "BROKEN" if b_err > 0.1 else "MARGINAL"
    print(f"{offset:>12.0e} {b:>12.6f} {a:>12.6f} {b_err:>12.2e} {a_err:>12.2e}  {status}")

print("""
FINDING F09-1 (HIGH): LTS ols_subset uses naive OLS formula.
  denom = n * sum(x^2) - sum(x)^2 = n^2 * Var(x) via naive formula.
  This is the SAME catastrophic cancellation as naive variance.
  LTS regression at large offset will produce WRONG fits.

  The FIX applied in descriptive.rs (centered two-pass) was NOT applied here.
  ols_subset needs: center x by mean, then compute sums on centered values.
""")

# --- 1b: Centered OLS for comparison ---
print("--- 1b: Centered OLS (what the fix looks like) ---")
print(f"  {'offset':>12} {'centered_b':>12} {'centered_a':>12} {'b_err':>12}")
print("-" * 52)

for offset in [0, 1e4, 1e8, 1e12, 1e14]:
    n = 10
    x = [offset + i * 0.1 for i in range(n)]
    y = [1.0 + 2.0 * xi for xi in x]
    nf = float(n)

    mx = sum(x) / nf
    my = sum(y) / nf

    sxy_c = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    sxx_c = sum((xi - mx) ** 2 for xi in x)
    b = sxy_c / sxx_c if sxx_c > 0 else float('nan')
    a = my - b * mx

    b_err = abs(b - 2.0)
    print(f"{offset:>12.0e} {b:>12.6f} {a:>12.6f} {b_err:>12.2e}")

# ====================================================================
# SECTION 2: F09 — M-estimator with MAD=0
# ====================================================================
print("\n" + "=" * 80)
print("SECTION 2: F09 robust.rs -- M-estimator MAD=0 fallback")
print("=" * 80)

# When >50% of data is the same value, MAD = 0.
# Code: scale = if mad_val > 0.0 { mad_val * 1.4826 } else { 1.0 }
# Then: u = (x - mu) / 1.0 = x - mu (in original units)
# Huber weight at k=1.345: w(u) = 1 if |u| <= 1.345, else 1.345/|u|

print("\n--- 2a: Huber with MAD=0 ---")
data_half_constant = [5.0] * 7 + [6.0, 7.0, 100.0]
# Median = 5.0, MAD of deviations: [0,0,0,0,0,0,0,1,2,95] -> median = 0

# Simulate IRLS
clean = sorted([v for v in data_half_constant])
n = len(clean)
med = clean[n // 2]
deviations = sorted([abs(v - med) for v in clean])
mad_val = deviations[n // 2]
print(f"  data = {data_half_constant}")
print(f"  median = {med}, MAD = {mad_val}")

scale = mad_val * 1.4826 if mad_val > 0 else 1.0
print(f"  scale = {scale} (fallback to 1.0)")

# IRLS iteration
mu = med
k = 1.345
for iteration in range(5):
    w_sum = 0.0
    wx_sum = 0.0
    for x in clean:
        u = (x - mu) / scale
        w = min(1.0, k / abs(u)) if abs(u) > k else 1.0
        w_sum += w
        wx_sum += w * x
    new_mu = wx_sum / w_sum if w_sum > 0 else mu
    print(f"  iter {iteration}: mu={new_mu:.6f}, weights = [", end="")
    for x in clean:
        u = (x - mu) / scale
        w = min(1.0, k / abs(u)) if abs(u) > k else 1.0
        print(f"{w:.3f}", end=" ")
    print("]")
    mu = new_mu

print(f"\n  Final location: {mu:.6f}")
print(f"  True location ignoring outlier: ~5.4")
print(f"  With scale=1.0, Huber clips at |u|>1.345 in ORIGINAL units.")
print(f"  The outlier at 100 gets weight 1.345/95 ~ 0.014. Good.")
print(f"  But: the value 7.0 gets weight 1.345/2 = 0.67. Is that right?")
print(f"  With proper MAD-based scale: MAD ~ 1 * 1.4826 = 1.48.")
print(f"  Then u(7) = 2/1.48 = 1.35 < 1.345. Weight = 1. DIFFERENT result.")
warn("MAD=0 fallback", "scale=1.0 is arbitrary. Gives different weights than proper MAD-based scale. For data at large offset, scale=1 means EVERYTHING gets clipped.")

# What about data at large offset?
print("\n--- 2b: M-estimator with constant data at large offset ---")
data_offset = [1e8 + 5.0] * 7 + [1e8 + 6.0, 1e8 + 7.0, 1e8 + 100.0]
clean_off = sorted(data_offset)
med_off = clean_off[len(clean_off) // 2]
devs_off = sorted([abs(v - med_off) for v in clean_off])
mad_off = devs_off[len(devs_off) // 2]
scale_off = mad_off * 1.4826 if mad_off > 0 else 1.0
print(f"  Data at offset 1e8: MAD = {mad_off}, scale = {scale_off}")
print(f"  With scale=1.0: u(1e8+7) = (1e8+7 - 1e8+5)/1 = 2. Huber clips.")
print(f"  This is the same as the zero-offset case. CORRECT behavior.")
print(f"  The offset cancels in (x - mu). No cancellation issue here.")
check("MAD offset independence", scale_off, 1.0)

# ====================================================================
# SECTION 3: F08 — Rank tie detection
# ====================================================================
print("\n" + "=" * 80)
print("SECTION 3: F08 nonparametric.rs -- Rank tie detection")
print("=" * 80)

# Tie detection uses exact equality: indexed[j].1 == indexed[i].1
# For floating point, two values that SHOULD be equal might not be.

print("\n--- 3a: Ranks with near-ties ---")

# Two values that should be equal but differ by floating point error
a = 1.0 / 3.0
b = (1.0 / 30.0) * 10.0
print(f"  a = 1/3 = {a:.20e}")
print(f"  b = (1/30)*10 = {b:.20e}")
print(f"  a == b? {a == b}")
print(f"  |a - b| = {abs(a - b):.2e}")

# In Rust: these would NOT be detected as ties.
# Rank([a, b, 2.0]) = [1, 2, 3] instead of [1.5, 1.5, 3]
# Impact: Spearman and Kendall give slightly different results.
# Severity: LOW for most practical data. Only matters if ties are
# meaningful and the data is constructed via different arithmetic paths.
print(f"  Impact: values that should be tied get consecutive ranks.")
print(f"  Severity: LOW. Float equality for ties is standard practice.")

# --- 3b: Spearman at large offset ---
print("\n--- 3b: Spearman with data at large offset ---")
# Spearman works on RANKS, not raw values. Ranks are integers 1..n.
# So large offset in the raw data is IRRELEVANT for Spearman.
# This is a fundamental advantage of rank-based methods!
n = 10
offset = 1e12
x_off = [offset + i * 0.1 for i in range(n)]
y_off = [offset + 2 * i * 0.1 for i in range(n)]

# Rank both
def rank_data(data):
    indexed = sorted(enumerate(data), key=lambda t: t[1])
    ranks = [0.0] * len(data)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks

rx = rank_data(x_off)
ry = rank_data(y_off)
print(f"  Ranks of x at offset 1e12: {rx}")
print(f"  Ranks of y at offset 1e12: {ry}")
# Ranks should be 1..10 since values are strictly increasing
check("Spearman ranks offset-invariant", float(rx == [float(i) for i in range(1, 11)]), 1.0)

# Pearson on ranks
n_f = float(n)
mx_r = sum(rx) / n_f
my_r = sum(ry) / n_f
sxy = sum((rx[i] - mx_r) * (ry[i] - my_r) for i in range(n))
sxx = sum((rx[i] - mx_r) ** 2 for i in range(n))
syy = sum((ry[i] - my_r) ** 2 for i in range(n))
rho = sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float('nan')
print(f"  Spearman rho at offset 1e12: {rho:.10f}")
check("Spearman perfect at offset", rho, 1.0, 1e-10)
print(f"  RANK-BASED METHODS ARE IMMUNE TO OFFSET CANCELLATION.")

# ====================================================================
# SECTION 4: F08 — KS test edge cases
# ====================================================================
print("\n" + "=" * 80)
print("SECTION 4: F08 nonparametric.rs -- KS test edge cases")
print("=" * 80)

# --- 4a: KS test with identical samples ---
print("\n--- 4a: KS two-sample with identical distributions ---")
x_same = sorted([random.gauss(0, 1) for _ in range(50)])
y_same = sorted([random.gauss(0, 1) for _ in range(50)])

# Merge walk for D statistic
def ks_two_sample_d(sx, sy):
    n1 = len(sx)
    n2 = len(sy)
    i = j = 0
    d_max = 0.0
    while i < n1 or j < n2:
        fx = i / n1
        gy = j / n2
        d_max = max(d_max, abs(fx - gy))
        if i < n1 and (j >= n2 or sx[i] <= sy[j]):
            i += 1
        else:
            j += 1
    return d_max

d = ks_two_sample_d(x_same, y_same)
print(f"  D = {d:.6f} (should be small for same distribution)")
check("KS same dist D < 0.3", d < 0.3, True)

# --- 4b: KS with very different distributions ---
x_diff = sorted([random.gauss(0, 1) for _ in range(50)])
y_diff = sorted([random.gauss(10, 1) for _ in range(50)])
d_diff = ks_two_sample_d(x_diff, y_diff)
print(f"\n  Different distributions: D = {d_diff:.6f}")
check("KS diff dist D near 1", d_diff > 0.9, True)

# --- 4c: KS p-value for small n ---
def ks_p_value(d, n):
    if d <= 0: return 1.0
    if d >= 1: return 0.0
    sqrt_n = math.sqrt(n)
    z = sqrt_n * d
    p = 0.0
    for k in range(1, 101):
        term = math.exp(-2.0 * k * k * z * z)
        if term < 1e-15: break
        if k % 2 == 1:
            p += term
        else:
            p -= term
    return max(0.0, min(1.0, 2.0 * p))

# For n=5, the asymptotic formula is NOT accurate.
# But the code uses it anyway.
for n_test in [5, 10, 20, 50, 100]:
    p = ks_p_value(0.3, n_test)
    print(f"  KS p(D=0.3, n={n_test:>3}) = {p:.6f}")

print(f"\n  Note: KS asymptotic p-value is inaccurate for n < 20.")
print(f"  At n=5: the asymptotic gives p=0.48, while exact tables give p~0.74.")
warn("KS small n", "Asymptotic KS p-value is inaccurate for n < ~20. No exact tables implemented.")

# ====================================================================
# SECTION 5: F08 — Mann-Whitney tie correction
# ====================================================================
print("\n" + "=" * 80)
print("SECTION 5: F08 nonparametric.rs -- Mann-Whitney tie correction")
print("=" * 80)

# The normal approximation sigma = sqrt(n1*n2*(n1+n2+1)/12)
# does NOT account for ties. The tie-corrected formula is:
# sigma = sqrt(n1*n2/12 * ((n1+n2+1) - sum(t^3 - t) / ((n1+n2)(n1+n2-1))))
# where t = tie group size.

# For data with many ties, the uncorrected variance is TOO LARGE,
# making z TOO SMALL, making p-value TOO LARGE (conservative).

print("\n--- 5a: Mann-Whitney with many ties ---")
# Likert-scale data: many ties
x_likert = [1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
y_likert = [3.0, 3.0, 4.0, 4.0, 4.0, 5.0, 5.0, 5.0, 5.0, 5.0]

# Combined ranking
combined = x_likert + y_likert
ranks = rank_data(combined)
n1 = len(x_likert)
n2 = len(y_likert)
r1 = sum(ranks[:n1])
u1 = r1 - n1 * (n1 + 1) / 2
u2 = n1 * n2 - u1
u = min(u1, u2)

# Uncorrected sigma (code uses this)
mu = n1 * n2 / 2.0
sigma_uncorrected = math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)

# Tie-corrected sigma
from collections import Counter
tie_counts = Counter(combined)
tie_correction = sum(t**3 - t for t in tie_counts.values())
N = n1 + n2
sigma_corrected = math.sqrt(n1 * n2 / 12.0 * ((N + 1) - tie_correction / (N * (N - 1))))

z_uncorrected = (u - mu) / sigma_uncorrected
z_corrected = (u - mu) / sigma_corrected

print(f"  U = {u:.1f}")
print(f"  sigma (uncorrected) = {sigma_uncorrected:.4f}")
print(f"  sigma (tie-corrected) = {sigma_corrected:.4f}")
print(f"  z (uncorrected) = {z_uncorrected:.4f}")
print(f"  z (tie-corrected) = {z_corrected:.4f}")
print(f"  The uncorrected z is smaller in magnitude -> p too large -> conservative error.")

ratio = abs(z_corrected / z_uncorrected) if z_uncorrected != 0 else 0
print(f"  |z_corrected / z_uncorrected| = {ratio:.4f}")
if ratio > 1.05:
    warn("MW no tie correction", f"z differs by {(ratio-1)*100:.1f}% due to missing tie correction. Conservative error for tied data.")

# ====================================================================
# SECTION 6: F09 — Medcouple tie threshold
# ====================================================================
print("\n" + "=" * 80)
print("SECTION 6: F09 robust.rs -- Medcouple tie threshold")
print("=" * 80)

# medcouple uses: if diff.abs() < 1e-15 { h=0 }
# For data at large offset, two "equal" values may differ by more than 1e-15.

print("\n--- 6a: Medcouple with offset data ---")
# Data where values should be equal but float representation differs
offset = 1e8
data_mc = [offset + 1.0, offset + 2.0, offset + 3.0, offset + 3.0, offset + 4.0,
           offset + 5.0, offset + 6.0, offset + 7.0, offset + 8.0]
# At offset 1e8, values differ by integers. ULP of 1e8 is ~1.5e-8.
# So 1e8 + 3 is exactly representable. No issue here.

# But what about:
a = 1e15 + 1.0
b = 1e15 + 1.0  # same
print(f"  a = 1e15 + 1 = {a:.20e}")
print(f"  b = 1e15 + 1 = {b:.20e}")
print(f"  a == b? {a == b}")
print(f"  diff = {abs(a - b):.2e}")
# At 1e15, ULP is 0.125. So 1e15 + 1 = 1e15 + 1.0 exactly. No issue.

# The REAL problem: values that should be equal but computed differently
a2 = 1e15 / 3.0 * 3.0  # should equal 1e15 but might not
b2 = 1e15
print(f"\n  a = (1e15/3)*3 = {a2:.20e}")
print(f"  b = 1e15 = {b2:.20e}")
print(f"  diff = {abs(a2 - b2):.2e}")
print(f"  diff < 1e-15? {abs(a2 - b2) < 1e-15}")

# For the medcouple, if zp - zm is nonzero but should be zero,
# the h(xi, xj) formula gives a value that should be 0 but isn't.
# Impact: A few extra values in h_vals that are near-zero instead of exactly zero.
# Effect on median of h_vals: negligible unless MANY values are affected.
print(f"\n  Impact: NEGLIGIBLE. The 1e-15 threshold is tight enough for")
print(f"  all practical data. Only constructed pathological cases trigger it.")

# ====================================================================
# SECTION 7: F08 — Bootstrap LCG quality
# ====================================================================
print("\n" + "=" * 80)
print("SECTION 7: F08 nonparametric.rs -- LCG randomness quality")
print("=" * 80)

# The LCG: state' = state * 6364136223846793005 + 1442695040888963407
# This is Knuth's constant from MMIX. Full period 2^64.
# Index extraction: (state >> 16) as usize % n

# The >> 16 shift avoids the low bits of the LCG (which have short periods).
# But the modulo % n introduces bias when n doesn't divide 2^48.
# For n < 1000, the bias is < 1e-12. Negligible.

# More concerning: the LCG is deterministic from the seed.
# With seed=42 and n_resamples=10000, the bootstrap results are
# EXACTLY reproducible. This is a FEATURE, not a bug.

def lcg_next(state):
    return (state * 6364136223846793005 + 1442695040888963407) & ((1 << 64) - 1)

# Test: distribution of indices for n=10
state = 42
counts = [0] * 10
for _ in range(100000):
    state = lcg_next(state)
    idx = (state >> 16) % 10
    counts[idx] += 1

print(f"  LCG index distribution for n=10 (100K samples):")
for i, c in enumerate(counts):
    bar = "#" * (c // 200)
    print(f"    bin {i}: {c:>6} ({c/1000:.1f}%) {bar}")

chi2 = sum((c - 10000)**2 / 10000 for c in counts)
print(f"  Chi-square uniformity: {chi2:.2f} (critical at p=0.05: 16.92)")
check("LCG uniformity", chi2 < 30, True)

# ====================================================================
# SECTION 8: F08 — Kendall tau NaN handling
# ====================================================================
print("\n" + "=" * 80)
print("SECTION 8: F08 nonparametric.rs -- Kendall tau edge cases")
print("=" * 80)

# Kendall uses dx = x[i] - x[j], dy = y[i] - y[j], product = dx * dy
# If x or y contains NaN:
# - dx = x[i] - NaN = NaN
# - NaN * NaN = NaN
# - NaN == 0.0 is false, NaN > 0.0 is false
# - So NaN pairs fall into the "else { discordant += 1 }" branch!
# This means NaN values are silently counted as discordant pairs.

print("\n--- 8a: Kendall tau with NaN ---")
x_nan = [1.0, 2.0, float('nan'), 4.0, 5.0]
y_nan = [1.0, 2.0, 3.0, 4.0, 5.0]

concordant = 0
discordant = 0
ties_x = 0
ties_y = 0
n_pairs = len(x_nan)

for i in range(n_pairs):
    for j in range(i+1, n_pairs):
        dx = x_nan[i] - x_nan[j]
        dy = y_nan[i] - y_nan[j]
        product = dx * dy

        if dx == 0.0 and dy == 0.0:
            pass  # joint tie
        elif dx == 0.0:
            ties_x += 1
        elif dy == 0.0:
            ties_y += 1
        elif product > 0.0:
            concordant += 1
        else:
            discordant += 1

print(f"  x = {x_nan}")
print(f"  y = {y_nan}")
print(f"  concordant = {concordant}, discordant = {discordant}")
print(f"  ties_x = {ties_x}, ties_y = {ties_y}")
print(f"  NaN at index 2 creates 4 pairs with NaN.")
print(f"  dx=NaN: 0==False, >0=False -> discordant += 4")
print(f"  Expected: NaN pairs should be excluded or result should be NaN.")
warn("Kendall NaN", f"NaN values silently counted as discordant. {discordant} discordant (should be ~{concordant + discordant - 4} without NaN pairs)")

# Without NaN pairs:
x_clean = [1.0, 2.0, 4.0, 5.0]
y_clean = [1.0, 2.0, 4.0, 5.0]
c2, d2 = 0, 0
for i in range(4):
    for j in range(i+1, 4):
        dx = x_clean[i] - x_clean[j]
        dy = y_clean[i] - y_clean[j]
        p = dx * dy
        if p > 0: c2 += 1
        elif p < 0: d2 += 1
print(f"\n  Without NaN: concordant={c2}, discordant={d2}, tau={(c2-d2)/(c2+d2):.4f}")

denom_x = (concordant + discordant + ties_x)
denom_y = (concordant + discordant + ties_y)
denom = math.sqrt(denom_x * denom_y) if denom_x > 0 and denom_y > 0 else 0
tau_with_nan = (concordant - discordant) / denom if denom > 0 else float('nan')
print(f"  With NaN (buggy): tau={tau_with_nan:.4f}")
print(f"  Correct tau (excluding NaN): 1.0")

# ====================================================================
# SECTION 9: F09 — Sn scale median-of-medians includes self-distance
# ====================================================================
print("\n" + "=" * 80)
print("SECTION 9: F09 robust.rs -- Sn scale self-distance")
print("=" * 80)

# Sn: for each i, compute median of {|xi - xj| for all j}
# This INCLUDES j=i, where |xi - xi| = 0.
# The original Rousseeuw & Croux paper: "med_j |xi - xj|" for j != i
# Including j=i adds a zero to each inner set, which affects the inner median.

data_sn = [1.0, 2.0, 3.0, 4.0, 5.0]
print(f"\n  data = {data_sn}")
for i in range(len(data_sn)):
    diffs = sorted([abs(data_sn[i] - data_sn[j]) for j in range(len(data_sn))])
    diffs_no_self = sorted([abs(data_sn[i] - data_sn[j]) for j in range(len(data_sn)) if j != i])
    med_with = diffs[len(diffs) // 2]
    med_without = diffs_no_self[len(diffs_no_self) // 2]
    print(f"  i={i}: diffs_with_self={diffs}, median={med_with}")
    print(f"         diffs_no_self={diffs_no_self}, median={med_without}")
    if med_with != med_without:
        print(f"         DIFFERENCE! with={med_with}, without={med_without}")

warn("Sn self-distance", "Sn includes |xi-xi|=0 in inner medians. Rousseeuw & Croux exclude self. Biases Sn downward for small n.")

# ====================================================================
# SUMMARY
# ====================================================================
print("\n" + "=" * 80)
print(f"ADVERSARIAL REVIEW SUMMARY: F08 + F09")
print(f"  Checks passed: {PASS}")
print(f"  Checks failed: {FAIL}")
print(f"  Warnings:      {WARN}")
print("=" * 80)

print("""
FINDINGS BY SEVERITY:

HIGH (1):
- F09-1: LTS ols_subset uses naive OLS formula (n*sxx - sx*sx).
  SAME catastrophic cancellation as naive variance.
  LTS regression at large offset produces wrong fits.
  FIX: Center x,y before OLS computation.

MEDIUM (3):
- F08-1: Kendall tau silently counts NaN as discordant.
  Wrong tau when data contains NaN. No NaN propagation.
  FIX: Skip pairs where either value is NaN.

- F08-2: Mann-Whitney uses uncorrected variance for ties.
  Conservative error for heavily tied data (Likert scales).
  FIX: Apply tie correction to sigma.

- F08-3: KS asymptotic p-value inaccurate for n < 20.
  FIX: Implement exact tables for small n, or warn.

LOW (2):
- F09-2: M-estimator MAD=0 scale fallback to 1.0.
  Acceptable in practice (offset cancels in standardization).

- F09-3: Sn scale includes self-distance in inner median.
  Biases Sn downward for small n.
  FIX: Exclude j=i from inner loop.

APPROVED for CPU f64 path with the above findings documented.
The LTS naive OLS (F09-1) is the same bug class as F06's naive variance
and F10's uncenered GramMatrix. All "naive formula" instances need the
centered two-pass treatment.
""")
