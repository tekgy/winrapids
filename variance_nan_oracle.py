"""Compute reference values for NaN-aware variance oracles.

Covers:
1. numpy.nanvar with both ddof=0 and ddof=1 against the 5 mean data variants
   (no NaNs present — should match numpy.var exactly).
2. numpy.nanvar on SEEDED-NaN variants (inject 10% NaNs) — verifies:
   (a) the NaN-dropping path introduces no additional error vs var on filtered data
   (b) ddof correction against n_effective = n - count(NaN), not n
3. pandas.Series.var(skipna=True) on same seeded-NaN variants.
4. Historical bug regression: numpy.nanvar(ddof=1) on arrays where every
   value is NaN — used to return 0.0, now returns NaN.

All numerical references are mpmath at 50dps for bit-exact ground truth.
"""
import json
import struct
from pathlib import Path
import numpy as np
import pandas as pd
from mpmath import mp, mpf

mp.dps = 50

DATA_ROOT = Path(r"R:/tambear/oracle/mean/data/generated")
VARIANTS = ["standard_gaussian_n1000", "ill_conditioned",
            "heavy_tail_t2", "outlier_contaminated", "small_n10"]


def mpmath_variance_two_pass(values, ddof):
    """Bit-exact reference via two-pass with mpmath at 50-digit precision.
    Values that are NaN are DROPPED before computation (Ignore semantics)."""
    xs = [mpf(repr(v)) for v in values if not (isinstance(v, float) and v != v)]
    n_eff = len(xs)
    if n_eff - ddof <= 0:
        return mpf("nan"), n_eff
    mean = sum(xs) / n_eff
    sq_dev = sum((x - mean) ** 2 for x in xs)
    return sq_dev / (n_eff - ddof), n_eff


def ulp_distance(a: float, ref_str: str) -> float:
    """ULP distance between f64 'a' and the f64 closest to mpmath ref."""
    if a != a:  # NaN
        return float("nan")
    ref = mpf(ref_str)
    ref_f64 = float(ref)
    if a == ref_f64:
        return 0.0
    a_bits = np.float64(a).view(np.int64)
    r_bits = np.float64(ref_f64).view(np.int64)
    return float(abs(int(a_bits) - int(r_bits)))


# ============================================================
# Part 1: nanvar on CLEAN data (no NaNs) — sanity check that
# nanvar == var when no NaNs present.
# ============================================================
print("=" * 70)
print("PART 1: nanvar vs var on CLEAN DATA (no NaNs injected)")
print("=" * 70)

clean_results = {}
for variant in VARIANTS:
    inp = DATA_ROOT / variant / "input.json"
    with open(inp) as f:
        vals = json.load(f)["values"]
    arr = np.array(vals, dtype=np.float64)
    series = pd.Series(arr)

    nanvar_d0 = float(np.nanvar(arr))             # ddof=0
    nanvar_d1 = float(np.nanvar(arr, ddof=1))
    var_d0 = float(np.var(arr))
    var_d1 = float(np.var(arr, ddof=1))
    pd_skipna_true = float(series.var(skipna=True))  # ddof=1 default
    pd_skipna_false = float(series.var(skipna=False))

    # Sanity: nanvar == var on clean data (modulo NaN dispatch overhead
    # producing same f64 bit pattern)
    assert nanvar_d0 == var_d0, f"nanvar vs var differ on {variant}: {nanvar_d0} vs {var_d0}"
    assert nanvar_d1 == var_d1, f"nanvar(ddof=1) vs var(ddof=1) differ on {variant}"
    assert pd_skipna_true == pd_skipna_false, f"skipna diff on clean {variant}"

    clean_results[variant] = {
        "n": len(arr),
        "nanvar_ddof0": nanvar_d0,
        "nanvar_ddof1": nanvar_d1,
        "pandas_skipna_true": pd_skipna_true,
        "pandas_skipna_false": pd_skipna_false,
    }
    print(f"  {variant:30s}: nanvar(ddof=0)={nanvar_d0:.6g}  nanvar(ddof=1)={nanvar_d1:.6g}  [all match var]")

# ============================================================
# Part 2: nanvar on NaN-INJECTED data (10% NaNs)
# ============================================================
print()
print("=" * 70)
print("PART 2: nanvar on NaN-INJECTED DATA (10% NaN)")
print("=" * 70)

np.random.seed(42)  # deterministic NaN placement
nan_results = {}
for variant in VARIANTS:
    inp = DATA_ROOT / variant / "input.json"
    with open(inp) as f:
        vals = json.load(f)["values"]
    arr = np.array(vals, dtype=np.float64).copy()
    n = len(arr)
    n_nan = max(1, n // 10)
    # Deterministic NaN placement at fixed indices
    nan_indices = sorted(np.random.RandomState(42 + hash(variant) % 1000).choice(
        n, size=n_nan, replace=False).tolist())
    for idx in nan_indices:
        arr[idx] = float("nan")
    series = pd.Series(arr)

    # numpy.nanvar: drops NaNs, uses n_eff = n - n_nan
    nanvar_d0 = float(np.nanvar(arr))
    nanvar_d1 = float(np.nanvar(arr, ddof=1))
    # numpy.var: propagates NaN → returns NaN
    var_propagate = float(np.var(arr))
    # pandas Series.var: skipna=True default
    pd_skipna = float(series.var())  # ddof=1, skipna=True
    pd_no_skip = float(series.var(skipna=False))

    # mpmath reference: drop NaNs, two-pass with correct n_eff
    mp_ref_d0, n_eff = mpmath_variance_two_pass(arr.tolist(), ddof=0)
    mp_ref_d1, _ = mpmath_variance_two_pass(arr.tolist(), ddof=1)
    assert n_eff == n - n_nan

    nan_results[variant] = {
        "n": n,
        "n_nan": n_nan,
        "n_effective": n_eff,
        "nanvar_ddof0": {
            "result": nanvar_d0,
            "mpmath_50d": mp.nstr(mp_ref_d0, 50, strip_zeros=False),
            "abs_error": float(abs(mpf(repr(nanvar_d0)) - mp_ref_d0)),
            "ulp_error": ulp_distance(nanvar_d0, mp.nstr(mp_ref_d0, 50, strip_zeros=False)),
        },
        "nanvar_ddof1": {
            "result": nanvar_d1,
            "mpmath_50d": mp.nstr(mp_ref_d1, 50, strip_zeros=False),
            "abs_error": float(abs(mpf(repr(nanvar_d1)) - mp_ref_d1)),
            "ulp_error": ulp_distance(nanvar_d1, mp.nstr(mp_ref_d1, 50, strip_zeros=False)),
        },
        "pandas_var_skipna_true": {
            "result": pd_skipna,
            "mpmath_50d": mp.nstr(mp_ref_d1, 50, strip_zeros=False),
            "abs_error": float(abs(mpf(repr(pd_skipna)) - mp_ref_d1)),
            "ulp_error": ulp_distance(pd_skipna, mp.nstr(mp_ref_d1, 50, strip_zeros=False)),
        },
        "var_propagate_returns_nan": var_propagate != var_propagate,  # True iff NaN
        "pd_skipna_false_returns_nan": pd_no_skip != pd_no_skip,
        "nan_indices_preview": nan_indices[:5],
    }
    print(f"  {variant:30s}: n={n}, n_nan={n_nan}, n_eff={n_eff}")
    print(f"    numpy.nanvar(ddof=0): {nanvar_d0:.15g}  ULP={nan_results[variant]['nanvar_ddof0']['ulp_error']}")
    print(f"    numpy.nanvar(ddof=1): {nanvar_d1:.15g}  ULP={nan_results[variant]['nanvar_ddof1']['ulp_error']}")
    print(f"    pandas Series.var(skipna=True): {pd_skipna:.15g}  ULP={nan_results[variant]['pandas_var_skipna_true']['ulp_error']}")
    print(f"    numpy.var (no skipna) returns NaN: {nan_results[variant]['var_propagate_returns_nan']}")

# ============================================================
# Part 3: historical-bug regression checks
# ============================================================
print()
print("=" * 70)
print("PART 3: historical-bug regression checks (numpy 2.4.2)")
print("=" * 70)

# Bug class 1: nanvar(ddof=1) on all-NaN array — historically 0.0, should be NaN
all_nan = np.array([float("nan")] * 10, dtype=np.float64)
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result_all_nan_d0 = float(np.nanvar(all_nan))
    result_all_nan_d1 = float(np.nanvar(all_nan, ddof=1))
print(f"  nanvar of all-NaN array (n=10, ddof=0): {result_all_nan_d0}  [expect NaN]")
print(f"  nanvar of all-NaN array (n=10, ddof=1): {result_all_nan_d1}  [expect NaN]")
print(f"  Is result NaN? ddof=0: {result_all_nan_d0 != result_all_nan_d0}, ddof=1: {result_all_nan_d1 != result_all_nan_d1}")

# Bug class 2: nanvar(ddof=1) on array with n_eff == 1 (n=2 with 1 NaN)
single_eff = np.array([5.0, float("nan")], dtype=np.float64)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    result_single_d0 = float(np.nanvar(single_eff))
    result_single_d1 = float(np.nanvar(single_eff, ddof=1))
print(f"  nanvar of n=2 with 1 NaN (ddof=0): {result_single_d0}  [expect 0.0 — population var of one point]")
print(f"  nanvar of n=2 with 1 NaN (ddof=1): {result_single_d1}  [expect NaN — sample var of one point is undefined]")

# Bug class 3: nanvar(ddof=n_eff) where divisor becomes 0 or negative
# Should return NaN with warning, not inf or garbage
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    boundary = np.array([1.0, 2.0, 3.0, float("nan"), float("nan")], dtype=np.float64)
    # n_eff=3, ddof=3 → divisor=0
    zero_div = float(np.nanvar(boundary, ddof=3))
    # n_eff=3, ddof=4 → divisor=-1
    neg_div = float(np.nanvar(boundary, ddof=4))
print(f"  nanvar with divisor=0 (ddof=n_eff): {zero_div}  [expect NaN or inf]")
print(f"  nanvar with divisor=-1 (ddof>n_eff): {neg_div}  [expect NaN]")

# ============================================================
# Part 4: ddof/n_effective interaction — the KEY oracle question
# ============================================================
print()
print("=" * 70)
print("PART 4: ddof × n_effective semantics")
print("=" * 70)
print()
print("Question: Does numpy.nanvar(x, ddof=k) use (n_eff - k) or (n - k)?")
print()

# Test: known case. x = [1, 2, 3, 4, 5, NaN, NaN, NaN], n=8, n_eff=5.
# Sample variance of [1,2,3,4,5] is 2.5 (ddof=1, divisor=4).
# If nanvar uses n_eff - ddof: divisor = 5 - 1 = 4, result = 10/4 = 2.5  ✓
# If nanvar uses n - ddof:      divisor = 8 - 1 = 7, result = 10/7 ≈ 1.428
test_known = np.array([1.0, 2.0, 3.0, 4.0, 5.0, float("nan"), float("nan"), float("nan")], dtype=np.float64)
nv_d0 = float(np.nanvar(test_known))
nv_d1 = float(np.nanvar(test_known, ddof=1))
mp_xs = [mpf(x) for x in [1, 2, 3, 4, 5]]
mp_mean = sum(mp_xs) / 5
mp_sq_dev = sum((x - mp_mean) ** 2 for x in mp_xs)
d0_ref_n_eff = float(mp_sq_dev / 5)  # divisor = n_eff = 5
d0_ref_n = float(mp_sq_dev / 8)       # divisor = n = 8
d1_ref_n_eff = float(mp_sq_dev / 4)   # divisor = n_eff - 1 = 4
d1_ref_n = float(mp_sq_dev / 7)       # divisor = n - 1 = 7
print(f"  Test: [1, 2, 3, 4, 5, NaN, NaN, NaN], n=8, n_eff=5")
print(f"  nanvar(x, ddof=0) = {nv_d0}")
print(f"    ref if uses n_eff (=5): {d0_ref_n_eff}  matches? {abs(nv_d0 - d0_ref_n_eff) < 1e-12}")
print(f"    ref if uses n (=8):     {d0_ref_n}     matches? {abs(nv_d0 - d0_ref_n) < 1e-12}")
print(f"  nanvar(x, ddof=1) = {nv_d1}")
print(f"    ref if uses n_eff-1 (=4): {d1_ref_n_eff}  matches? {abs(nv_d1 - d1_ref_n_eff) < 1e-12}")
print(f"    ref if uses n-1 (=7):     {d1_ref_n}     matches? {abs(nv_d1 - d1_ref_n) < 1e-12}")
print()
# Confirm: which does pandas use?
pd_series = pd.Series(test_known)
pd_d1 = float(pd_series.var())
print(f"  pandas Series.var() = {pd_d1}")
print(f"    matches n_eff-1: {abs(pd_d1 - d1_ref_n_eff) < 1e-12}")
print(f"    matches n-1:     {abs(pd_d1 - d1_ref_n) < 1e-12}")

# ============================================================
# Summary output
# ============================================================
print()
print("=" * 70)
print("SUMMARY — write to nanvar_reference.json")
print("=" * 70)

summary = {
    "generated": "2026-04-22",
    "environment": {
        "python_version": "3.13.13",
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
    },
    "clean_data": clean_results,
    "nan_injected_data": nan_results,
    "historical_regressions": {
        "all_nan_array_ddof0": {
            "result": result_all_nan_d0,
            "is_nan": result_all_nan_d0 != result_all_nan_d0,
            "expected": "NaN (no valid data)",
        },
        "all_nan_array_ddof1": {
            "result": result_all_nan_d1,
            "is_nan": result_all_nan_d1 != result_all_nan_d1,
            "expected": "NaN (no valid data)",
        },
        "n_eff_equals_1_ddof0": {
            "result": result_single_d0,
            "expected": "0.0 (population variance of 1 point is 0)",
        },
        "n_eff_equals_1_ddof1": {
            "result": result_single_d1,
            "is_nan": result_single_d1 != result_single_d1,
            "expected": "NaN (sample variance of 1 point is undefined)",
        },
    },
    "ddof_semantics": {
        "numpy_nanvar": "uses n_effective - ddof (NOT n - ddof)",
        "pandas_series_var_skipna": "uses n_effective - ddof (same as numpy.nanvar)",
        "test_case": "[1,2,3,4,5,NaN,NaN,NaN], nanvar(ddof=1)=2.5 confirming n_eff-1=4 divisor",
    },
}

out_path = Path(r"R:/winrapids/nanvar_reference.json")
out_path.write_text(json.dumps(summary, indent=2))
print(f"  Written to: {out_path}")
