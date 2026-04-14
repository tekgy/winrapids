"""
Gold-standard parity runner for tambear trig functions.

Compares tambear (via f64 native math, representing the Remez+Payne-Hanek
implementation) against:
  - numpy (platform libm / SVML)
  - mpmath at 100-digit precision (gold standard)

Usage:
    python parity_runner.py          # runs sin + cos
    python parity_runner.py --fn sin # runs just sin

Note: Until tambear exposes a Python binding for its Rust trig recipes, we
compare numpy vs mpmath directly. This establishes the gold-standard baseline
and documents ulp budgets. When Rust bindings ship, this script gains a
tambear= column.

Output format per function:
  - Worst-case ulp deviation (numpy vs mpmath)
  - Inputs that produce the worst deviation
  - Summary suitable for parity-table.md
"""

import argparse
import struct
import math
import mpmath
import numpy as np

# ── Gold standard: 100 digits ────────────────────────────────────────────────
mpmath.mp.dps = 100


def f64_to_bits(x: float) -> int:
    return struct.unpack('Q', struct.pack('d', x))[0]


def ulps_between(a: float, b: float) -> int:
    """ULP distance between two f64 values. Returns 0 for NaN==NaN (by bits)."""
    if math.isnan(a) and math.isnan(b):
        return 0
    if math.isnan(a) or math.isnan(b):
        return 2**63  # sentinel: NaN vs non-NaN
    if math.isinf(a) and math.isinf(b):
        return 0 if a == b else 2**63
    # Treat -0 == +0
    ba, bb = f64_to_bits(a), f64_to_bits(b)
    if ba == 0x8000000000000000:
        ba = 0
    if bb == 0x8000000000000000:
        bb = 0
    # Convert sign-magnitude to two's complement ordering
    if ba >> 63:
        ba = 0x8000000000000000 - ba
    if bb >> 63:
        bb = 0x8000000000000000 - bb
    return abs(ba - bb)


# ── Synthetic ground truth test points ──────────────────────────────────────
# sin(n·π/6) for n in 0..12: exact values known.
SIN_EXACT = [
    (0,     0.0),
    (1,     0.5),                      # sin(π/6)
    (2,     math.sqrt(3)/2),           # sin(π/3)
    (3,     1.0),                      # sin(π/2)
    (4,     math.sqrt(3)/2),           # sin(2π/3)
    (5,     0.5),                      # sin(5π/6)
    (6,     0.0),                      # sin(π) — catastrophic cancellation
    (7,    -0.5),                      # sin(7π/6)
    (8,    -math.sqrt(3)/2),           # sin(4π/3)
    (9,    -1.0),                      # sin(3π/2)
    (10,   -math.sqrt(3)/2),           # sin(5π/3)
    (11,   -0.5),                      # sin(11π/6)
    (12,    0.0),                      # sin(2π)
]

COS_EXACT = [
    (0,    1.0),
    (1,    math.sqrt(3)/2),            # cos(π/6)
    (2,    0.5),                       # cos(π/3)
    (3,    0.0),                       # cos(π/2) — near cancellation
    (4,   -0.5),                       # cos(2π/3)
    (5,   -math.sqrt(3)/2),            # cos(5π/6)
    (6,   -1.0),                       # cos(π)
    (7,   -math.sqrt(3)/2),            # cos(7π/6)
    (8,   -0.5),                       # cos(4π/3)
    (9,    0.0),                       # cos(3π/2) — near cancellation
    (10,   0.5),                       # cos(5π/3)
    (11,   math.sqrt(3)/2),            # cos(11π/6)
    (12,   1.0),                       # cos(2π)
]


def adversarial_sin_cos_inputs():
    """Adversarial inputs matching adversarial.rs sin_cos_adversarial."""
    pts = []
    pi = math.pi

    # Multiples of π/4 up to ±50π
    for k in range(-200, 201):
        pts.append(k * pi / 4)

    # Near-zero inputs
    for e in range(-52, 1):
        pts.append(2.0 ** e)
        pts.append(-(2.0 ** e))

    # Large inputs (Payne-Hanek territory for tambear)
    for mag in [1e4, 1e5, 1e6, 1e7, 1e10, 1e15, 1e17]:
        pts.append(mag)
        pts.append(-mag)

    # Kahan hard inputs
    for v in [3.141592653589793, 6.283185307179586, 1.5707963267948966, 355.0, 1e5]:
        pts.append(v)
        pts.append(-v)

    # Dense sweep through [0, 2π]
    pts.extend(np.linspace(0, 2*pi, 2000).tolist())

    # Remove duplicates, keep finite
    seen = set()
    out = []
    for x in pts:
        if math.isfinite(x):
            key = f64_to_bits(x)
            if key not in seen:
                seen.add(key)
                out.append(x)
    return sorted(out)


def run_parity(fn_name: str, numpy_fn, mpmath_fn, exact_pairs):
    """
    Run parity analysis for a trig function.

    Returns dict with:
      - worst_ulp_numpy_vs_mpmath: int
      - worst_input: float
      - exact_point_results: list of dicts
      - large_arg_worst: (float, int)
      - summary lines for parity-table.md
    """
    inputs = adversarial_sin_cos_inputs()
    print(f"\n{'='*60}")
    print(f"  {fn_name.upper()} parity analysis")
    print(f"  {len(inputs)} adversarial inputs")
    print(f"{'='*60}")

    # ── Exact known points ───────────────────────────────────────────────────
    print(f"\n-- Exact known points (n*pi/6 grid) --")
    exact_results = []
    for (n, expected_exact) in exact_pairs:
        x = n * math.pi / 6
        np_val = float(numpy_fn(x))
        mp_val = float(mpmath_fn(mpmath.mpf(x)))
        ulp_np_mp = ulps_between(np_val, mp_val)
        ulp_np_ex = ulps_between(np_val, expected_exact)
        ulp_mp_ex = ulps_between(mp_val, expected_exact)
        exact_results.append({
            'n': n,
            'x': x,
            'expected': expected_exact,
            'numpy': np_val,
            'mpmath': mp_val,
            'ulp_np_mp': ulp_np_mp,
            'ulp_np_exact': ulp_np_ex,
            'ulp_mp_exact': ulp_mp_ex,
        })
        marker = "  <-- ZERO" if expected_exact == 0.0 else ""
        print(f"  n={n:2d} x={x:+.4f}  np_vs_mp={ulp_np_mp} ulp  np_vs_exact={ulp_np_ex} ulp{marker}")

    # ── Full adversarial sweep ───────────────────────────────────────────────
    print(f"\n-- Full adversarial sweep --")
    worst_ulp = 0
    worst_input = 0.0
    total = len(inputs)
    threshold_counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

    # Process in batches for speed
    np_vals = numpy_fn(np.array(inputs))
    mp_vals_f64 = []
    for x in inputs:
        try:
            mp_val = float(mpmath_fn(mpmath.mpf(x)))
        except Exception:
            mp_val = float('nan')
        mp_vals_f64.append(mp_val)

    ulp_dists = []
    for i, (x, np_v, mp_v) in enumerate(zip(inputs, np_vals, mp_vals_f64)):
        d = ulps_between(float(np_v), mp_v)
        ulp_dists.append(d)
        if d > worst_ulp:
            worst_ulp = d
            worst_input = x
        for k in sorted(threshold_counts.keys()):
            if d <= k:
                threshold_counts[k] += 1

    print(f"  worst ulp (numpy vs mpmath): {worst_ulp} at x={worst_input:.18e}")
    for ulp_limit, count in sorted(threshold_counts.items()):
        pct = 100.0 * count / total
        print(f"  <= {ulp_limit} ulp: {count}/{total} ({pct:.1f}%)")

    # ── Large argument analysis (Payne-Hanek domain) ─────────────────────────
    print(f"\n-- Large argument subset (|x| > 2^20·π/2 ≈ 1.6e6) --")
    large_threshold = 1_647_100.0
    large_inputs = [x for x in inputs if abs(x) > large_threshold]
    if large_inputs:
        large_worst_ulp = 0
        large_worst_input = 0.0
        for x in large_inputs:
            np_v = float(numpy_fn(np.array([x]))[0])
            mp_v = float(mpmath_fn(mpmath.mpf(x)))
            d = ulps_between(np_v, mp_v)
            if d > large_worst_ulp:
                large_worst_ulp = d
                large_worst_input = x
        print(f"  {len(large_inputs)} large-arg inputs, worst: {large_worst_ulp} ulp at x={large_worst_input:.18e}")
    else:
        large_worst_ulp = None
        large_worst_input = None
        print("  (none in this sweep)")

    return {
        'fn': fn_name,
        'n_inputs': total,
        'worst_ulp_numpy_vs_mpmath': worst_ulp,
        'worst_input': worst_input,
        'exact_results': exact_results,
        'large_arg_worst_ulp': large_worst_ulp,
        'large_arg_worst_input': large_worst_input,
        'ulp_dist_counts': threshold_counts,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fn', default='all', choices=['all', 'sin', 'cos'])
    args = parser.parse_args()

    results = {}
    if args.fn in ('all', 'sin'):
        results['sin'] = run_parity(
            'sin', np.sin, mpmath.sin, SIN_EXACT
        )
    if args.fn in ('all', 'cos'):
        results['cos'] = run_parity(
            'cos', np.cos, mpmath.cos, COS_EXACT
        )

    # ── Policy gap analysis ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  POLICY GAP ANALYSIS")
    print("="*60)

    print("""
IEEE 754-2019 special case policy gaps (places where implementations
may differ by design, not by error):

sin(0.0)  = +0.0 (IEEE), sin(-0.0) = -0.0 (IEEE) — sign preservation
cos(0.0)  = 1.0, cos(-0.0) = 1.0 — cos is even, -0 treated as 0
sin(π)    = numpy returns ~1.2e-16 (not 0.0) — expected, π is irrational in f64
cos(π/2)  = numpy returns ~6.1e-17 (not 0.0) — same reason

numpy vs mpmath: these are measurement differences, NOT policy gaps.
The gap numpy shows vs mpmath for sin(kπ) is the range-reduction error
in x.sin() (libm) vs mpmath's exact arithmetic. tambear uses Payne-Hanek
for |x| >= 2^20·π/2, matching mpmath's precision for large arguments.

tambear defaults (scientist judgment):
- precision = compensated (not strict): the additional cost is ~10% for
  a factor-of-2+ reduction in worst-case ulp. Worth it.
- angle_unit = radians (conventional, expected by users).
- range_reduction = auto: Cody-Waite where valid, Payne-Hanek beyond.
""")

    print("\nSUMMARY TABLE (numpy vs mpmath gold standard):")
    print(f"{'Function':<10} {'Inputs':<8} {'Worst ulp':<12} {'<= 1 ulp %':<14} {'<= 2 ulp %':<14}")
    print("-" * 60)
    for fn, r in results.items():
        n = r['n_inputs']
        w = r['worst_ulp_numpy_vs_mpmath']
        c1 = r['ulp_dist_counts'].get(1, 0)
        c2 = r['ulp_dist_counts'].get(2, 0)
        pct1 = 100.0 * c1 / n if n > 0 else 0
        pct2 = 100.0 * c2 / n if n > 0 else 0
        print(f"  {fn:<8} {n:<8} {w:<12} {pct1:<14.1f} {pct2:<14.1f}")


if __name__ == '__main__':
    main()
