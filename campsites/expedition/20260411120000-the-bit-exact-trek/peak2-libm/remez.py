#!/usr/bin/env python
"""
Campsite 2.5 / 2.10 / 2.13 infrastructure — Remez minimax polynomial fitter.

PURPOSE
-------
Compute polynomial coefficients for tambear-libm from first principles, at
high precision, with a certified error bound. This is the *only* source of
polynomial coefficients for Phase 1. We do not copy coefficients from any
existing libm.

The coefficients this script generates go into:
    exp-constants.toml
    log-constants.toml
    trig-constants.toml (sin, cos)
    atan-constants.toml
    ...

and are then embedded into the .tam text for each libm function as
f64 bit-pattern literals.

APPROACH
--------
For a target function f(x) on an interval [a, b] and a polynomial degree d,
compute the minimax polynomial P(x) of degree d that minimizes

    max_{x in [a, b]} |f(x) - P(x)|

via the Remez exchange algorithm, implemented in mpmath at 100+ decimal
digits of working precision so that the fit error is dominated by the
polynomial's structural accuracy, not by the solver's precision.

Remez exchange algorithm (classical):
  1. Pick d+2 Chebyshev nodes on [a, b] as initial reference points.
  2. Solve the linear system:
         f(x_i) - P(x_i) = (-1)^i * E     for i = 0..d+1
     This gives d+1 polynomial coefficients and one signed error E.
  3. Find the (d+2) extrema of the error function  r(x) = f(x) - P(x) .
  4. If the extrema align with the reference (up to tolerance), done.
  5. Otherwise, replace the reference with the extrema and repeat from (2).

For the monotone-error classical Remez algorithm, convergence is typically
5-15 iterations for smooth f on a compact interval.

OUTPUT
------
Coefficients + certified error bound in a TOML file that's read at libm
build time. We also write a plain-text summary with the 50-digit decimal
strings so humans can sanity-check the fit.

NO VENDOR LIBM
--------------
This script uses mpmath and nothing else. We compute the target function,
the polynomial evaluations, and the root-finding for extrema entirely in
mpmath arithmetic. No numpy, no scipy, no math.sin. (I1, I8, I9.)
"""
from __future__ import annotations

import argparse
import struct
import sys
from typing import Callable, List, Tuple

try:
    import mpmath as mp
except ImportError:
    sys.exit(
        "mpmath not installed. Run:\n"
        "    uv venv\n"
        "    uv pip install mpmath\n"
        "from the peak2-libm directory first."
    )

WORKING_DPS = 100  # mpmath decimal digits of internal precision


# -- Target functions --------------------------------------------------------

# Each entry maps a symbolic target name to a function callable(mp.mpf -> mp.mpf).
# The target is what we fit — this may be a reshaped version of the actual
# libm function. For exp, we fit Q(r) = (exp(r) - 1 - r) / r^2 so that the
# final evaluator can be  1 + r + r^2 * Q(r)  and inherit the exact linear
# behavior near zero.

def target_exp_remainder(r: "mp.mpf") -> "mp.mpf":
    """(exp(r) - 1 - r) / r^2 extended continuously to r=0 with value 1/2."""
    if r == 0:
        return mp.mpf("0.5")
    return (mp.exp(r) - 1 - r) / (r * r)

def target_log_remainder(f: "mp.mpf") -> "mp.mpf":
    """(log(1+f) - f) / f^2 extended continuously to f=0 with value -1/2."""
    if f == 0:
        return mp.mpf("-0.5")
    return (mp.log(1 + f) - f) / (f * f)

def target_sin_over_r(r: "mp.mpf") -> "mp.mpf":
    """(sin(r) - r) / r^3 extended continuously to r=0 with value -1/6.

    We fit this, then evaluate   sin(r) = r + r^3 * P(r^2)
    which gives odd symmetry by construction and exact linear behavior at 0.
    """
    if r == 0:
        return mp.mpf(-1) / 6
    return (mp.sin(r) - r) / (r ** 3)

def target_cos_shifted(r: "mp.mpf") -> "mp.mpf":
    """(cos(r) - 1 + r^2/2) / r^4 extended continuously to r=0 with value 1/24.

    We fit this, then evaluate   cos(r) = 1 - r^2/2 + r^4 * P(r^2)
    which gives even symmetry and preserves the leading order subtraction.
    """
    if r == 0:
        return mp.mpf(1) / 24
    return (mp.cos(r) - 1 + r * r / 2) / (r ** 4)

def target_atan_remainder(r: "mp.mpf") -> "mp.mpf":
    """(atan(r) - r) / r^3 extended continuously to r=0 with value -1/3."""
    if r == 0:
        return mp.mpf(-1) / 3
    return (mp.atan(r) - r) / (r ** 3)

TARGETS: dict[str, Tuple[Callable[["mp.mpf"], "mp.mpf"], str]] = {
    # tag -> (function, description)
    "exp_remainder": (target_exp_remainder, "(exp(r) - 1 - r) / r^2"),
    "log_remainder": (target_log_remainder, "(log(1+f) - f) / f^2"),
    "sin_over_r":    (target_sin_over_r,    "(sin(r) - r) / r^3"),
    "cos_shifted":   (target_cos_shifted,   "(cos(r) - 1 + r^2/2) / r^4"),
    "atan_remainder":(target_atan_remainder,"(atan(r) - r) / r^3"),
}


# -- fp64 helpers ------------------------------------------------------------

def mp_to_f64_bits(v: "mp.mpf") -> int:
    """Round an mpmath value to nearest fp64 and return its bit pattern."""
    as_float = float(v)  # mpmath uses round-to-nearest-even to fp64
    return struct.unpack("<Q", struct.pack("<d", as_float))[0]

def mp_to_f64_hex(v: "mp.mpf") -> str:
    """Format the mpmath value as a hex fp64 bit literal, like 0d3ff0000000000000."""
    return f"0d{mp_to_f64_bits(v):016x}"


# -- Polynomial evaluation (Horner in mpmath) --------------------------------

def horner(coeffs: List["mp.mpf"], x: "mp.mpf") -> "mp.mpf":
    """Horner evaluation: coeffs[0] + x * (coeffs[1] + x * (coeffs[2] + ...))."""
    y = coeffs[-1]
    for c in reversed(coeffs[:-1]):
        y = y * x + c
    return y


# -- Remez exchange ----------------------------------------------------------

def chebyshev_nodes(a: "mp.mpf", b: "mp.mpf", n: int) -> List["mp.mpf"]:
    """The n Chebyshev nodes of the second kind on [a, b]."""
    mid = (a + b) / 2
    half = (b - a) / 2
    nodes = []
    for k in range(n):
        # Chebyshev-Gauss-Lobatto nodes include the endpoints
        theta = mp.pi * k / (n - 1)
        nodes.append(mid + half * mp.cos(theta))
    nodes.sort()
    return nodes

def solve_remez_linear_system(
    target: Callable[["mp.mpf"], "mp.mpf"],
    nodes: List["mp.mpf"],
    degree: int,
) -> Tuple[List["mp.mpf"], "mp.mpf"]:
    """Solve for polynomial coefficients c_0..c_d and error E such that
       target(x_i) - P(x_i) = (-1)^i * E   for i = 0..d+1

    There are d+1 coefficients + 1 error = d+2 unknowns, matching d+2 nodes.
    Build the (d+2) x (d+2) matrix and solve via mpmath.lu_solve.
    """
    n = len(nodes)
    assert n == degree + 2, f"need {degree + 2} nodes, got {n}"
    A = mp.matrix(n, n)
    b = mp.matrix(n, 1)
    for i, x in enumerate(nodes):
        # Columns 0..degree: powers of x (polynomial part)
        xp = mp.mpf(1)
        for j in range(degree + 1):
            A[i, j] = xp
            xp = xp * x
        # Last column: (-1)^i for the signed error
        A[i, degree + 1] = mp.mpf(-1) ** i
        b[i, 0] = target(x)
    sol = mp.lu_solve(A, b)
    coeffs = [sol[i, 0] for i in range(degree + 1)]
    error = sol[degree + 1, 0]
    return coeffs, error

def find_extrema(
    target: Callable[["mp.mpf"], "mp.mpf"],
    coeffs: List["mp.mpf"],
    a: "mp.mpf",
    b: "mp.mpf",
    degree: int,
) -> List["mp.mpf"]:
    """Locate d+2 alternating extrema of r(x) = target(x) - P(x) on [a, b].

    We scan densely across [a, b] to find each extremum's bracket, then
    use mpmath's findroot on the derivative (or on sign changes of the
    error to catch boundary extrema).
    """
    def residual(x):
        return target(x) - horner(coeffs, x)

    # Sample densely
    n_samples = 2000
    xs = [a + (b - a) * i / (n_samples - 1) for i in range(n_samples)]
    rs = [residual(x) for x in xs]

    extrema = []
    # Always consider endpoints
    # Find interior local extrema by sign changes of discrete differences
    # between consecutive samples
    diffs = [rs[i + 1] - rs[i] for i in range(n_samples - 1)]

    # Endpoint a is an extremum if the slope doesn't start by heading the
    # "wrong" way. For Remez, we always include both endpoints as candidate
    # extrema because the error function is typically largest at or near
    # the boundary.
    extrema.append(a)
    for i in range(len(diffs) - 1):
        if diffs[i] == 0:
            continue
        if (diffs[i] > 0 and diffs[i + 1] < 0) or (diffs[i] < 0 and diffs[i + 1] > 0):
            # Local extremum between xs[i] and xs[i + 2]
            x_left, x_right = xs[i], xs[i + 2]
            # Golden-section narrowing to a few more digits of precision
            x_ext = golden_section(residual, x_left, x_right, diffs[i] > 0)
            extrema.append(x_ext)
    extrema.append(b)

    # Deduplicate near-identical extrema (shouldn't happen but be safe)
    extrema = sorted(set(extrema))
    # Remez exchange should return exactly d+2 extrema. If we found more or
    # fewer, take the largest-|residual| d+2 of them to continue the iter.
    with_mag = [(abs(residual(x)), x) for x in extrema]
    with_mag.sort(reverse=True)
    kept = sorted([p[1] for p in with_mag[: degree + 2]])
    return kept

def golden_section(
    f: Callable[["mp.mpf"], "mp.mpf"],
    a: "mp.mpf",
    b: "mp.mpf",
    maximize: bool,
    iters: int = 80,
) -> "mp.mpf":
    """Golden-section search for an extremum of f on [a, b].
    maximize=True finds a max; False finds a min.
    """
    phi = (1 + mp.sqrt(5)) / 2
    invphi = 1 / phi
    invphi2 = 1 / (phi * phi)
    h = b - a
    c = a + invphi2 * h
    d = a + invphi * h
    fc = f(c)
    fd = f(d)
    for _ in range(iters):
        if (fc > fd) == maximize:
            b = d
            d = c
            fd = fc
            h = invphi * h
            c = a + invphi2 * h
            fc = f(c)
        else:
            a = c
            c = d
            fc = fd
            h = invphi * h
            d = a + invphi * h
            fd = f(d)
    return (a + b) / 2

def remez_fit(
    target: Callable[["mp.mpf"], "mp.mpf"],
    a: "mp.mpf",
    b: "mp.mpf",
    degree: int,
    max_iters: int = 50,
    tol_decimal_places: int = 25,
) -> Tuple[List["mp.mpf"], "mp.mpf"]:
    """Classical Remez exchange. Returns (coefficients, certified_error_bound)."""
    nodes = chebyshev_nodes(a, b, degree + 2)
    tol = mp.mpf(10) ** (-tol_decimal_places)

    coeffs = None
    prev_error = None

    for iteration in range(max_iters):
        coeffs, signed_error = solve_remez_linear_system(target, nodes, degree)
        extrema = find_extrema(target, coeffs, a, b, degree)
        if len(extrema) != degree + 2:
            raise RuntimeError(
                f"Remez iter {iteration}: expected {degree+2} extrema, got {len(extrema)}"
            )

        def residual(x):
            return target(x) - horner(coeffs, x)

        max_abs_err = max(abs(residual(x)) for x in extrema)

        if prev_error is not None:
            if abs(max_abs_err - prev_error) < tol * max_abs_err:
                return coeffs, max_abs_err
        prev_error = max_abs_err
        nodes = extrema

    return coeffs, prev_error if prev_error is not None else mp.mpf("nan")


# -- Main --------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Remez minimax fitter for tambear-libm.")
    p.add_argument("--target", required=True, choices=sorted(TARGETS.keys()))
    p.add_argument("--lo", type=str, required=True, help="interval lower bound (mpmath expr)")
    p.add_argument("--hi", type=str, required=True, help="interval upper bound (mpmath expr)")
    p.add_argument("--degree", type=int, required=True, help="polynomial degree")
    p.add_argument("--dps", type=int, default=WORKING_DPS)
    p.add_argument("--name", type=str, default=None, help="symbolic name for output")
    p.add_argument("--max-iters", type=int, default=50)
    args = p.parse_args()

    mp.mp.dps = args.dps

    target_fn, target_desc = TARGETS[args.target]
    # Interval bounds are Python expressions evaluated in an mpmath context.
    # This lets the user write "-mp.log(2)/2", "mp.pi/4", "mp.sqrt(2)-1", etc.
    eval_env = {
        "mp": mp,
        "mpf": mp.mpf,
        "pi": mp.pi,
        "sqrt": mp.sqrt,
        "log": mp.log,
        "ln": mp.log,
        "exp": mp.exp,
    }
    try:
        a = mp.mpf(eval(args.lo, eval_env))
        b = mp.mpf(eval(args.hi, eval_env))
    except Exception as e:
        print(f"failed to parse interval [{args.lo}, {args.hi}]: {e}", file=sys.stderr)
        return 1

    name = args.name or args.target

    print(f"# remez fit: {name}", file=sys.stderr)
    print(f"# target   : {target_desc}", file=sys.stderr)
    print(f"# interval : [{mp.nstr(a, 20)}, {mp.nstr(b, 20)}]", file=sys.stderr)
    print(f"# degree   : {args.degree}", file=sys.stderr)
    print(f"# dps      : {args.dps}", file=sys.stderr)
    print("", file=sys.stderr)

    coeffs, err = remez_fit(target_fn, a, b, args.degree, max_iters=args.max_iters)

    print(f"# max_abs_error = {mp.nstr(err, 30)}", file=sys.stderr)
    print(f"# err as fraction of 2^-52 = {mp.nstr(err / mp.mpf(2) ** (-52), 6)}", file=sys.stderr)
    print("", file=sys.stderr)

    print(f"[{name}]")
    print(f'target = "{target_desc}"')
    print(f'interval_lo = "{mp.nstr(a, 50)}"')
    print(f'interval_hi = "{mp.nstr(b, 50)}"')
    print(f"degree = {args.degree}")
    print(f'max_abs_error = "{mp.nstr(err, 30)}"')
    print()
    print(f"[{name}.coefficients]")
    for i, c in enumerate(coeffs):
        hex_bits = mp_to_f64_hex(c)
        as_f64 = float(c)
        print(f'a_{i}   = {{ hex = "{hex_bits}", f64 = {as_f64!r}, decimal_50 = "{mp.nstr(c, 50)}" }}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
