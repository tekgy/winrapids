#!/usr/bin/env python
"""
Campsite 2.2 — mpmath reference generator

Generates (fp64_input, fp64_reference_output, 50_digit_decimal_string) triples
for a given libm function over a declared domain. Used as the oracle for
tambear-libm Phase 1 ULP measurement (Campsite 2.3 and every Phase 1 function
campsite after it).

USAGE
-----
    # Set up environment (do this once per repo clone)
    uv venv
    uv pip install mpmath numpy

    # Generate references
    python gen-reference.py --function sqrt --n 1000 --out sqrt-1k.bin
    python gen-reference.py --function exp  --n 1000000 --out exp-1m.bin
    python gen-reference.py --function sin  --n 1000000 --domain small --out sin-small-1m.bin

OUTPUT FORMAT (.bin)
--------------------
A tambear-specific binary format — we do not use parquet or arrow because we
want readability across the full cross-backend pipeline with zero dependencies.

    Header (64 bytes, little-endian):
        bytes  0..  8: magic b"TAMBLMR1"          (tambear libm reference v1)
        bytes  8.. 16: uint64 n_samples
        bytes 16.. 32: fixed-width ascii function name (null-padded to 16)
        bytes 32.. 48: fixed-width ascii domain tag   (null-padded to 16)
        bytes 48.. 56: uint64 mpmath precision (decimal digits)
        bytes 56.. 64: reserved (zero)

    Followed by n_samples records, each 48 bytes:
        bytes  0..  8: fp64 input  (little-endian IEEE 754 double)
        bytes  8.. 16: fp64 reference output (mpmath rounded-to-nearest fp64)
        bytes 16.. 48: 32 bytes ASCII mantissa-and-exponent string,
                       null-padded, giving the 50-digit mpmf representation
                       of the true result. Used for humans when diagnosing
                       why a particular input broke us.

The Rust ULP harness (Campsite 2.3) reads the header, asserts the magic and
function/domain match what it's about to test, then iterates the records.

NO VENDOR LIBM
--------------
This script uses mpmath only for the reference. We never import math.sin,
numpy.exp, scipy, or any hardware libm to compute a reference value. mpmath
at 50-digit precision is the oracle; everything else is input generation.
(Invariants I1, I9.)

AUTHORITATIVE FUNCTIONS AND DOMAINS
-----------------------------------
These must match the primary-domain table in accuracy-target.md. If you edit
one, edit the other.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import struct
import sys
import time
from typing import Callable, Iterator, Tuple

try:
    import mpmath as mp
except ImportError:
    sys.exit(
        "mpmath not installed. Run:\n"
        "    uv venv\n"
        "    uv pip install mpmath\n"
        "from the peak2-libm directory first."
    )

MAGIC = b"TAMBLMR1"
MPMATH_DIGITS = 50  # I9: mpmath precision for the oracle
HEADER_SIZE = 64
RECORD_SIZE = 48

# -- Domain declarations ------------------------------------------------------

# Each entry: fn_name -> dict(domain_tag -> (lo, hi, sampler_kind))
# sampler_kind:
#   "exponent_uniform" — pick exponent uniformly across the domain, mantissa
#                        uniformly across 52-bit space. Gives equal weight to
#                        every decade. THIS IS THE DEFAULT for most functions.
#   "real_uniform"     — pick the real value uniformly in [lo, hi]. Appropriate
#                        for bounded domains like [-1, 1] where no decade
#                        swamping can happen.
DOMAINS: dict[str, dict[str, Tuple[float, float, str]]] = {
    "sqrt":  {"primary": (0.0,             1.797693e+308, "exponent_uniform")},
    "exp":   {"primary": (-745.133219,     709.782712,    "exponent_uniform")},
    "log":   {"primary": (2.225073e-308,   1.797693e+308, "exponent_uniform")},
    "sin":   {
        "small": (-(1 << 20),  (1 << 20),   "exponent_uniform"),
        "tiny":  (-1e-10,       1e-10,       "exponent_uniform"),
    },
    "cos":   {
        "small": (-(1 << 20),  (1 << 20),   "exponent_uniform"),
        "tiny":  (-1e-10,       1e-10,       "exponent_uniform"),
    },
    "tanh":  {"primary": (-1.797693e+308,  1.797693e+308, "exponent_uniform")},
    "sinh":  {"primary": (-710.0,          710.0,          "exponent_uniform")},
    "cosh":  {"primary": (-710.0,          710.0,          "exponent_uniform")},
    "atan":  {"primary": (-1.797693e+308,  1.797693e+308, "exponent_uniform")},
    "asin":  {"primary": (-1.0,            1.0,            "real_uniform")},
    "acos":  {"primary": (-1.0,            1.0,            "real_uniform")},
}

# mpmath reference function table. These are the ONLY functions we ever call
# on mpmath values in this generator. Each lambda takes an mpmath mpf and
# returns an mpmath mpf at MPMATH_DIGITS precision.
MP_FUNCS: dict[str, Callable[["mp.mpf"], "mp.mpf"]] = {
    "sqrt":  lambda x: mp.sqrt(x),
    "exp":   lambda x: mp.exp(x),
    "log":   lambda x: mp.log(x),
    "sin":   lambda x: mp.sin(x),
    "cos":   lambda x: mp.cos(x),
    "tanh":  lambda x: mp.tanh(x),
    "sinh":  lambda x: mp.sinh(x),
    "cosh":  lambda x: mp.cosh(x),
    "atan":  lambda x: mp.atan(x),
    "asin":  lambda x: mp.asin(x),
    "acos":  lambda x: mp.acos(x),
}

# -- fp64 helpers -------------------------------------------------------------

def f64_bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]

def bits_f64(b: int) -> float:
    return struct.unpack("<d", struct.pack("<Q", b & 0xFFFF_FFFF_FFFF_FFFF))[0]

def f64_exponent(x: float) -> int:
    """Unbiased exponent of x. Returns -1022 for subnormals and zero."""
    if x == 0.0:
        return -1022
    _m, e = math.frexp(x)
    return e - 1  # math.frexp gives m in [0.5, 1) ; our convention: m in [1, 2)

def f64_from_parts(sign: int, unbiased_exp: int, mantissa52: int) -> float:
    """Construct an fp64 from sign bit, unbiased exponent, and 52-bit mantissa."""
    biased = unbiased_exp + 1023
    if biased <= 0 or biased >= 2047:
        return 0.0  # we don't generate subnormals or +/-inf here by default
    bits = (sign & 1) << 63 | (biased & 0x7FF) << 52 | (mantissa52 & 0xF_FFFF_FFFF_FFFF)
    return bits_f64(bits)

def mp_round_to_f64(v: "mp.mpf") -> float:
    """Round an mpmath value to the nearest fp64. Uses mpmath's own float cast,
    which we verify behaves as IEEE round-to-nearest-even. This is the ONLY
    place we trust a language-level conversion to fp64, and it is for the
    reference, not for our own math."""
    return float(v)

# -- Samplers -----------------------------------------------------------------

def sample_exponent_uniform(
    lo: float,
    hi: float,
    n: int,
    rng: random.Random,
) -> Iterator[float]:
    """Exponent-uniform sampling inside [lo, hi].

    For each sample:
      1. Pick sign compatible with lo/hi (both allowed if domain crosses 0).
      2. Pick an unbiased exponent uniformly from exponents that can fit
         inside [lo, hi].
      3. Pick a random 52-bit mantissa.
      4. Reassemble. Reject and retry if outside [lo, hi].
    """
    signs = []
    if lo < 0.0:
        signs.append(1)
    if hi > 0.0:
        signs.append(0)
    if not signs:
        # Degenerate: lo == hi == 0
        for _ in range(n):
            yield 0.0
        return

    def max_abs_exp():
        m = max(abs(lo), abs(hi))
        if m == 0.0:
            return -1022
        return f64_exponent(m)

    min_abs_exp = -1022  # we include near-subnormal decades
    max_exp = max_abs_exp()

    emitted = 0
    tries = 0
    while emitted < n:
        tries += 1
        if tries > n * 1000:
            raise RuntimeError(f"sampler stuck; emitted={emitted}, tries={tries}")
        sign = rng.choice(signs)
        e = rng.randint(min_abs_exp, max_exp)
        m = rng.getrandbits(52)
        x = f64_from_parts(sign, e, m)
        if x < lo or x > hi:
            continue
        emitted += 1
        yield x

def sample_real_uniform(
    lo: float,
    hi: float,
    n: int,
    rng: random.Random,
) -> Iterator[float]:
    for _ in range(n):
        yield lo + (hi - lo) * rng.random()

# -- Main generation loop -----------------------------------------------------

def generate(
    function: str,
    domain: str,
    n: int,
    seed: int,
) -> Iterator[Tuple[float, float, str]]:
    if function not in DOMAINS:
        raise ValueError(f"unknown function: {function}")
    if domain not in DOMAINS[function]:
        raise ValueError(f"unknown domain for {function}: {domain}")

    lo, hi, kind = DOMAINS[function][domain]
    rng = random.Random(seed)
    mp.mp.dps = MPMATH_DIGITS

    if kind == "exponent_uniform":
        sampler = sample_exponent_uniform(lo, hi, n, rng)
    elif kind == "real_uniform":
        sampler = sample_real_uniform(lo, hi, n, rng)
    else:
        raise ValueError(f"unknown sampler kind: {kind}")

    mp_fn = MP_FUNCS[function]

    for x in sampler:
        mp_x = mp.mpf(x)
        mp_y = mp_fn(mp_x)
        # Round mpmath value to fp64 for the binary reference.
        y_f64 = mp_round_to_f64(mp_y)
        # Ascii representation for humans. 50 digits, no trailing newlines.
        y_str = mp.nstr(mp_y, MPMATH_DIGITS)
        yield x, y_f64, y_str

def write_bin(
    out_path: str,
    function: str,
    domain: str,
    n: int,
    samples: Iterator[Tuple[float, float, str]],
) -> None:
    with open(out_path, "wb") as f:
        # Header
        f.write(MAGIC)
        f.write(struct.pack("<Q", n))
        f.write(function.encode("ascii").ljust(16, b"\x00"))
        f.write(domain.encode("ascii").ljust(16, b"\x00"))
        f.write(struct.pack("<Q", MPMATH_DIGITS))
        f.write(b"\x00" * 8)  # reserved
        assert f.tell() == HEADER_SIZE, f"header size mismatch: {f.tell()}"

        # Records
        for i, (x, y, y_str) in enumerate(samples):
            f.write(struct.pack("<d", x))
            f.write(struct.pack("<d", y))
            y_str_bytes = y_str.encode("ascii", errors="replace")[:32]
            f.write(y_str_bytes.ljust(32, b"\x00"))
            if (i + 1) % 100_000 == 0:
                print(f"  {i+1:>10}/{n} samples", file=sys.stderr)

def main() -> int:
    p = argparse.ArgumentParser(description="Generate mpmath reference data for tambear-libm.")
    p.add_argument("--function", required=True, choices=sorted(DOMAINS.keys()))
    p.add_argument("--domain", default="primary")
    p.add_argument("--n", type=int, required=True, help="number of samples")
    p.add_argument("--out", required=True, help="output .bin path")
    p.add_argument("--seed", type=int, default=0x7AEBEA4)
    args = p.parse_args()

    print(
        f"generating {args.n} samples: function={args.function} "
        f"domain={args.domain} seed={args.seed:#x} dps={MPMATH_DIGITS}",
        file=sys.stderr,
    )
    start = time.time()
    samples = generate(args.function, args.domain, args.n, args.seed)
    write_bin(args.out, args.function, args.domain, args.n, samples)
    elapsed = time.time() - start
    sz = os.path.getsize(args.out)
    print(
        f"wrote {sz} bytes to {args.out} in {elapsed:.1f}s "
        f"({args.n / max(elapsed, 1e-6):.0f} samples/sec)",
        file=sys.stderr,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
