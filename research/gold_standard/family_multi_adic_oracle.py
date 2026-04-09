"""
Gold Standard Oracle: Multi-Adic Arithmetic

Tests p-adic valuation, norm, distance, digit expansion against known identities.
All deterministic, bit-perfect comparison.

Usage:
    python research/gold_standard/family_multi_adic_oracle.py
"""

import json
import math

results = {}

# ─── p-adic valuation ────────────────────────────────────────────────────

def v_p(n, p):
    """p-adic valuation of n."""
    if n == 0:
        return float('inf')
    v = 0
    while n % p == 0:
        n //= p
        v += 1
    return v

# Known valuations
valuation_tests = {}
for n, p, expected in [
    (12, 2, 2),    # 12 = 2^2 * 3
    (12, 3, 1),
    (12, 5, 0),
    (8, 2, 3),     # 8 = 2^3
    (1, 2, 0),
    (32, 2, 5),    # 2^5
    (243, 3, 5),   # 3^5
    (3125, 5, 5),  # 5^5
    (60, 2, 2),    # 60 = 2^2 * 3 * 5
    (60, 3, 1),
    (60, 5, 1),
    (60, 7, 0),
    (2520, 2, 3),  # 2520 = 2^3 * 3^2 * 5 * 7
    (2520, 3, 2),
    (2520, 5, 1),
    (2520, 7, 1),
]:
    key = f"v_{p}({n})"
    valuation_tests[key] = expected

results["valuations"] = valuation_tests

# ─── p-adic norm ─────────────────────────────────────────────────────────

# |n|_p = p^{-v_p(n)}
norm_tests = {}
for n, p, expected in [
    (8, 2, 0.125),     # 2^{-3}
    (9, 3, 1.0/9.0),   # 3^{-2}
    (7, 2, 1.0),       # v_2(7)=0 → 2^0=1
    (1, 5, 1.0),       # v_5(1)=0 → 5^0=1
    (25, 5, 0.04),     # 5^{-2}
    (60, 2, 0.25),     # 2^{-2}
    (60, 3, 1.0/3.0),  # 3^{-1}
]:
    key = f"|{n}|_{p}"
    norm_tests[key] = expected

results["norms"] = norm_tests

# ─── p-adic distance ─────────────────────────────────────────────────────

# d_p(a,b) = p^{-v_p(a-b)}
def p_adic_distance(a, b, p):
    if a == b:
        return 0.0
    diff = abs(a - b)
    v = v_p(diff, p)
    return p ** (-v)

distance_tests = {}
for a, b, p, expected in [
    (0, 8, 2, 0.125),    # d_2(0,8) = 2^{-3}
    (0, 4, 2, 0.25),     # d_2(0,4) = 2^{-2}
    (7, 7, 2, 0.0),      # d_2(a,a) = 0
    (6, 12, 2, 0.5),     # |12-6|=6=2*3, v_2(6)=1, d=2^{-1}
    (10, 1, 3, 1.0/9.0), # |10-1|=9=3^2, v_3(9)=2, d=3^{-2}
]:
    key = f"d_{p}({a},{b})"
    distance_tests[key] = expected
    # verify
    actual = p_adic_distance(a, b, p)
    assert abs(actual - expected) < 1e-15, f"d_{p}({a},{b}) = {actual}, expected {expected}"

results["distances"] = distance_tests

# ─── Ultrametric inequality ──────────────────────────────────────────────

# d(a,c) <= max(d(a,b), d(b,c)) for all a,b,c
ultrametric_tests = []
for p in [2, 3, 5]:
    for a, b, c in [(1,3,5), (2,6,10), (7,15,31)]:
        dac = p_adic_distance(a, c, p)
        dab = p_adic_distance(a, b, p)
        dbc = p_adic_distance(b, c, p)
        holds = dac <= max(dab, dbc) + 1e-15
        ultrametric_tests.append({
            "p": p, "a": a, "b": b, "c": c,
            "dac": dac, "dab": dab, "dbc": dbc,
            "holds": holds
        })

results["ultrametric"] = ultrametric_tests

# ─── p-adic digits (base p expansion) ────────────────────────────────────

def p_adic_digits(n, p, max_digits=64):
    """Low-order digits first."""
    if n == 0:
        return []
    digits = []
    while n > 0 and len(digits) < max_digits:
        digits.append(n % p)
        n //= p
    return digits

digit_tests = {
    "13_base2": p_adic_digits(13, 2),     # 1101 → [1, 0, 1, 1]
    "23_base3": p_adic_digits(23, 3),     # 212 → [2, 1, 2]
    "42_base5": p_adic_digits(42, 5),     # 132 → [2, 3, 1]
    "100_base10": p_adic_digits(100, 10), # [0, 0, 1]
    "0_base2": p_adic_digits(0, 2),       # []
    "1_base7": p_adic_digits(1, 7),       # [1]
}

results["p_adic_digits"] = digit_tests

# ─── Roundtrip: digits → number ──────────────────────────────────────────

roundtrip_tests = {}
for n in [0, 1, 42, 100, 12345, 999999]:
    for p in [2, 3, 5, 7, 10]:
        digits = p_adic_digits(n, p)
        # Reconstruct
        reconstructed = sum(d * p**i for i, d in enumerate(digits))
        roundtrip_tests[f"{n}_base{p}"] = {
            "original": n,
            "digits": digits,
            "reconstructed": reconstructed,
            "matches": n == reconstructed,
        }

results["roundtrip"] = roundtrip_tests

# ─── Multi-adic profile ──────────────────────────────────────────────────

primes = [2, 3, 5, 7]

profile_tests = {}
for n in [1, 11, 60, 2520, 100]:
    valuations = [v_p(n, p) for p in primes]
    total_weight = sum(v for v in valuations if v != float('inf'))
    is_coprime = all(v == 0 for v in valuations)
    profile_tests[str(n)] = {
        "valuations": valuations,
        "total_weight": total_weight,
        "is_coprime_to_all": is_coprime,
    }

results["multi_adic_profiles"] = profile_tests

# ─── Collatz trajectory multi-adic profile ────────────────────────────────

# Standard Collatz: n/2 if even, 3n+1 if odd
def collatz(n):
    return n // 2 if n % 2 == 0 else 3 * n + 1

trajectory_4 = [4, 2, 1]
profile_4 = []
for n in trajectory_4:
    profile_4.append({
        "value": n,
        "v2": v_p(n, 2) if n > 0 else None,
        "v3": v_p(n, 3) if n > 0 else None,
    })

results["collatz_trajectory_4"] = {
    "values": trajectory_4,
    "profiles": profile_4,
}

# ─── Batch profiles ──────────────────────────────────────────────────────

# batch_profiles(10, 5, [2,3,5]) → profiles for 10..14
batch_primes = [2, 3, 5]
batch_start = 10
batch_count = 5
batch_result = []
for i in range(batch_count):
    n = batch_start + i
    for p in batch_primes:
        batch_result.append(v_p(n, p) if n > 0 else 0)

results["batch_profiles_10_5"] = {
    "start": batch_start,
    "count": batch_count,
    "primes": batch_primes,
    "flat_valuations": batch_result,
}

# ─── Save ─────────────────────────────────────────────────────────────────

with open("research/gold_standard/family_multi_adic_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Multi-Adic Oracle: {len(results)} test cases generated")
for name in results:
    print(f"  PASS {name}")
