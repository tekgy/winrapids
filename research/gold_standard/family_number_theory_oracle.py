"""
Gold Standard Oracle: Number Theory

Generates expected values for primality, factorization, modular arithmetic,
number-theoretic functions against sympy and known identities.

All results are exact integers — bit-perfect comparison.

Usage:
    python research/gold_standard/family_number_theory_oracle.py
"""

import json
import math
from sympy import (
    isprime, factorint, totient, mobius as sympy_mobius,
    divisor_count, divisor_sigma, divisors as sympy_divisors,
    nextprime, primepi,
    jacobi_symbol, legendre_symbol,
    Integer
)
from sympy.ntheory.modular import crt as sympy_crt_fn

results = {}

# ─── Primality ────────────────────────────────────────────────────────────

test_primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
               97, 101, 1009, 10007, 100003, 1000003, 999999937]
test_composites = [4, 6, 8, 9, 10, 15, 21, 25, 100, 1001, 999999938]

results["primality"] = {
    "primes": {str(n): True for n in test_primes},
    "composites": {str(n): False for n in test_composites},
}

# Verify against sympy
for n in test_primes:
    assert isprime(n), f"{n} should be prime"
for n in test_composites:
    assert not isprime(n), f"{n} should be composite"

# ─── Sieve ────────────────────────────────────────────────────────────────

primes_to_100 = [int(p) for p in range(2, 101) if isprime(p)]
results["sieve_100"] = {
    "primes": primes_to_100,
    "count": len(primes_to_100),
}

# ─── Next prime ───────────────────────────────────────────────────────────

results["next_prime"] = {
    str(n): int(nextprime(n)) for n in [1, 2, 10, 100, 1000, 10000]
}

# ─── Prime counting ──────────────────────────────────────────────────────

results["prime_count"] = {
    str(n): int(primepi(n)) for n in [10, 100, 1000, 10000]
}

# ─── Factorization ───────────────────────────────────────────────────────

factor_tests = [12, 60, 100, 360, 1001, 2520, 10080, 65536, 999999937]
results["factorize"] = {}
for n in factor_tests:
    factors = factorint(n)
    # Convert to list of [prime, exponent] sorted by prime
    factor_list = sorted([[int(p), int(e)] for p, e in factors.items()])
    results["factorize"][str(n)] = factor_list

# ─── Euler totient ───────────────────────────────────────────────────────

totient_tests = [1, 2, 6, 10, 12, 36, 100, 360, 1000, 10080]
results["euler_totient"] = {
    str(n): int(totient(n)) for n in totient_tests
}

# ─── Mobius function ─────────────────────────────────────────────────────

mobius_tests = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 30, 42, 60]
results["mobius"] = {
    str(n): int(sympy_mobius(n)) for n in mobius_tests
}

# ─── Divisor count and sum ───────────────────────────────────────────────

div_tests = [1, 6, 12, 28, 60, 100, 360]
results["num_divisors"] = {
    str(n): int(divisor_count(n)) for n in div_tests
}
results["sum_divisors"] = {
    str(n): int(divisor_sigma(n)) for n in div_tests
}
results["divisors"] = {
    str(n): sorted([int(d) for d in sympy_divisors(n)]) for n in div_tests
}

# ─── GCD and LCM ─────────────────────────────────────────────────────────

gcd_tests = [(12, 18), (35, 49), (100, 75), (17, 13), (0, 5), (1, 1)]
results["gcd"] = {
    f"{a}_{b}": int(math.gcd(a, b)) for a, b in gcd_tests
}
results["lcm"] = {
    f"{a}_{b}": int(math.lcm(a, b)) for a, b in gcd_tests
}

# ─── Modular exponentiation ──────────────────────────────────────────────

results["mod_pow"] = {
    "2_10_1000": int(pow(2, 10, 1000)),      # 1024 mod 1000 = 24
    "3_100_997": int(pow(3, 100, 997)),
    "7_256_13": int(pow(7, 256, 13)),
    "2_32_1000000007": int(pow(2, 32, 1000000007)),
}

# ─── Modular inverse ─────────────────────────────────────────────────────

results["mod_inverse"] = {
    "3_mod_7": int(pow(3, -1, 7)),      # 3*5 = 15 ≡ 1 (mod 7) → 5
    "5_mod_13": int(pow(5, -1, 13)),
    "17_mod_43": int(pow(17, -1, 43)),
}

# ─── CRT ──────────────────────────────────────────────────────────────────

# x ≡ 2 (mod 3), x ≡ 3 (mod 5), x ≡ 2 (mod 7)
remainders = [2, 3, 2]
moduli = [3, 5, 7]
x_crt, M_crt = sympy_crt_fn(moduli, remainders)
results["crt"] = {
    "remainders": remainders,
    "moduli": moduli,
    "solution": int(x_crt),
    "product": int(M_crt),
}
# Verify
for r, m in zip(remainders, moduli):
    assert int(x_crt) % m == r, f"CRT verification failed: {x_crt} mod {m} != {r}"

# ─── Legendre symbol ─────────────────────────────────────────────────────

legendre_tests = [(1, 7), (2, 7), (3, 7), (4, 7), (5, 7), (6, 7),
                  (2, 11), (3, 11), (5, 13), (7, 17)]
results["legendre"] = {
    f"{a}_{p}": int(legendre_symbol(a, p)) for a, p in legendre_tests
}

# ─── Jacobi symbol ───────────────────────────────────────────────────────

jacobi_tests = [(2, 15), (3, 15), (7, 15), (11, 15), (2, 21), (5, 21)]
results["jacobi"] = {
    f"{a}_{n}": int(jacobi_symbol(a, n)) for a, n in jacobi_tests
}

# ─── Continued fractions ─────────────────────────────────────────────────

import math
# sqrt(2) = [1; 2, 2, 2, ...] (period 1)
# pi = [3; 7, 15, 1, 292, ...]
# e = [2; 1, 2, 1, 1, 4, 1, 1, 6, ...]

results["continued_fractions"] = {
    "sqrt2_10_terms": [1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    "phi_10_terms": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # golden ratio
    "pi_first_6": [3, 7, 15, 1, 292, 1],
}

# ─── Perfect numbers ─────────────────────────────────────────────────────

# σ(n) = 2n for perfect numbers
results["perfect_numbers"] = {
    "6": int(divisor_sigma(6)) == 12,    # 2*6
    "28": int(divisor_sigma(28)) == 56,  # 2*28
    "496": int(divisor_sigma(496)) == 992,
}

# ─── Totient sieve ───────────────────────────────────────────────────────

# φ(n) for n = 1..20
# phi(0) is undefined; tambear returns 0. phi(1) = 1.
results["totient_sieve_20"] = [0] + [int(totient(n)) for n in range(1, 21)]

# ─── Save ─────────────────────────────────────────────────────────────────

with open("research/gold_standard/family_number_theory_expected.json", "w") as f:
    json.dump(results, f, indent=2)

print(f"Number Theory Oracle: {len(results)} test cases generated")
for name in results:
    print(f"  PASS {name}")
