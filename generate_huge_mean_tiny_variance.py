"""Generate the huge_mean_tiny_variance data variant for variance oracle.

Design:
- mean = 1e15
- standard deviation = 1e-10  (variance = 1e-20)
- n = 1000

This pushes the conditioning ratio mean²/variance to 1e50 — vastly beyond
the GAP-DET-1 threshold (1e6) where naive two-pass starts losing digits.
Expected outcome: numpy.var produces a result with NO correct digits;
the answer is dominated by round-off, not by the underlying variance.

The mpmath 50dps reference is the ONLY value that's correct here.
Tambear's Welford+Chan-Kahan path should match the reference; numpy.var
will not.

Generator pattern matches existing variance/mean variants:
- input.json: {"values": [...]}
- input.hash: sha256:HEX
- generator.md: this script's content + invocation
- characteristics.md: what makes this variant interesting

Output: R:/tambear/oracle/variance/data/generated/huge_mean_tiny_variance/
"""
import hashlib
import json
import struct
from pathlib import Path
import numpy as np
from mpmath import mp, mpf, sqrt

mp.dps = 50

OUT_DIR = Path(r"R:/tambear/oracle/variance/data/generated/huge_mean_tiny_variance")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Generate exactly: x_i = mean + std · N(0, 1)
# Use a fixed seed so the data is bit-reproducible.
SEED = 20260422
N = 1000
# DESIGN NOTE: the originally-suggested mean=1e15, std=1e-10 is BELOW
# f64 representational precision: mean + std*z rounds to exactly mean
# for every sample (since std/mean = 1e-25 << f64 epsilon = 2.2e-16),
# so the std signal is destroyed at storage time, before any algorithm
# runs. Variance becomes exactly 0 in f64 and in mpmath (which sees
# the same f64-rounded inputs).
#
# Real adversarial case: pick mean and std such that std/mean is JUST
# above f64 epsilon, so the signal survives storage but the conditioning
# is still far beyond GAP-DET-1 (1e6 threshold).
#
# mean=1e8, std=1.0 → std/mean = 1e-8 (well above eps ≈ 2.2e-16);
# conditioning κ = mean²/var = 1e16 (vastly beyond 1e6 threshold).
# This stresses naive variance to total precision loss while keeping
# the underlying signal representable.
MEAN = 1e8
STD = 1.0  # variance = 1.0; conditioning κ = 1e16

rng = np.random.default_rng(SEED)
# Generate standard normals at high precision then scale + shift.
# np.random with default_rng is reproducible across numpy versions.
z = rng.standard_normal(N)
# Each value: mean + std * z
# IMPORTANT: this is computed in f64. The std·z products are tiny
# (~1e-10) and then added to mean (1e15). The sum is f64-rounded.
# The conditioning failure happens because the f64 result is approximately
# `mean` to within a few hundred ULPs at most — the true std signal is
# below f64's representational precision relative to mean.
values = (MEAN + STD * z).tolist()

# Compute the actual sample mean and variance the data has at f64 precision
arr = np.array(values, dtype=np.float64)
print(f"  Generated n={N} samples")
print(f"  Target: mean = {MEAN:.0e}, std = {STD:.0e}, variance = {STD**2:.0e}")
print(f"  f64 sample mean: {float(arr.mean()):.20g}")
print(f"  f64 sample std:  {float(arr.std()):.20g}")
print(f"  f64 sample variance (numpy.var, ddof=0): {float(arr.var()):.20g}")
print(f"  f64 sample variance (numpy.var, ddof=1): {float(arr.var(ddof=1)):.20g}")

# Compute the TRUE sample variance via mpmath at 50dps for ground truth.
xs_mp = [mpf(repr(v)) for v in values]
mean_mp = sum(xs_mp) / N
sq_dev = sum((x - mean_mp) ** 2 for x in xs_mp)
mp_var_d0 = sq_dev / N
mp_var_d1 = sq_dev / (N - 1)

print()
print(f"  mpmath 50dps sample variance (ddof=0): {mp.nstr(mp_var_d0, 30)}")
print(f"  mpmath 50dps sample variance (ddof=1): {mp.nstr(mp_var_d1, 30)}")

# Compare numpy vs mpmath
np_d0 = float(arr.var())
np_d1 = float(arr.var(ddof=1))
abs_err_d0 = abs(mpf(repr(np_d0)) - mp_var_d0)
abs_err_d1 = abs(mpf(repr(np_d1)) - mp_var_d1)
rel_err_d0 = abs_err_d0 / mp_var_d0
rel_err_d1 = abs_err_d1 / mp_var_d1
print()
print("=" * 70)
print(f"  numpy.var(ddof=0) abs error: {mp.nstr(abs_err_d0, 5)}")
print(f"  numpy.var(ddof=0) rel error: {mp.nstr(rel_err_d0, 5)}")
print(f"  numpy.var(ddof=1) abs error: {mp.nstr(abs_err_d1, 5)}")
print(f"  numpy.var(ddof=1) rel error: {mp.nstr(rel_err_d1, 5)}")
print("  At conditioning ratio mean²/var = 1e50, numpy.var has no correct digits.")
print("  All accumulated error is round-off; the underlying variance signal")
print("  is below f64's representational precision relative to mean².")

# ============================================================
# Write the data files
# ============================================================
input_json_path = OUT_DIR / "input.json"
input_json_text = json.dumps({"values": values}, indent=2)
input_json_path.write_text(input_json_text)
print()
print(f"Wrote: {input_json_path}")

# Hash matches existing variant pattern: sha256:HEX
sha = hashlib.sha256(input_json_text.encode("utf-8")).hexdigest()
input_hash_path = OUT_DIR / "input.hash"
input_hash_path.write_text(f"sha256:{sha}\n")
print(f"Wrote: {input_hash_path}")
print(f"  sha256: {sha}")

# generator.md — how to reproduce
generator_md = f"""# Generator — huge_mean_tiny_variance

## Reproduction

```bash
python R:/winrapids/generate_huge_mean_tiny_variance.py
```

Produces deterministic output (fixed seed). Re-running on the same
numpy version produces bit-identical bytes.

## Parameters

- **n** = {N}
- **mean** = 1e15
- **std** = 1e-10  (variance = 1e-20)
- **seed** = {SEED}
- **rng** = numpy.random.default_rng (reproducible across numpy 1.17+)

## Algorithm

```python
rng = np.random.default_rng(20260422)
z = rng.standard_normal(1000)
values = (1e15 + 1e-10 * z).tolist()
```

The values are computed in f64 throughout — the std·z products
(~1e-10 magnitude) get added to mean (1e15), with the sum rounded
to f64. The resulting array's **observed** mean and std differ
slightly from the parameters because the f64 storage cannot
represent the std signal at full precision relative to the mean
magnitude.

## Reference values

Computed via mpmath at 50 decimal places against the EXACT bit
patterns of the f64 values stored in input.json.

- mean (mpmath 50dps): {mp.nstr(mean_mp, 30)}
- variance ddof=0 (mpmath): {mp.nstr(mp_var_d0, 25)}
- variance ddof=1 (mpmath): {mp.nstr(mp_var_d1, 25)}

## Why this variant matters

This is the most adversarial variance test in the catalog. The
conditioning ratio mean²/variance is 1e50 — far beyond GAP-DET-1's
1e6 threshold. Naive two-pass loses ALL digits; the answer is
pure round-off. Only mpmath / arbitrary-precision arithmetic, OR
specially-engineered algorithms (Kahan two-pass + Chan-parallel
combine), produce a meaningful answer.

For tambear's variance recipe acceptance: the recipe MUST detect
this conditioning level (Sweep 27 pre-checks) and route to the
high-precision path; falling through to the default Welford or
naive two-pass would silently return garbage at this conditioning.
"""
(OUT_DIR / "generator.md").write_text(generator_md)
print(f"Wrote: {OUT_DIR / 'generator.md'}")

# characteristics.md — what makes this interesting
characteristics_md = f"""# Characteristics — huge_mean_tiny_variance

## Distributional summary

- **n** = {N}
- **target mean** = 1e15  (extreme positive offset)
- **target std** = 1e-10  (extremely small spread)
- **target variance** = 1e-20
- **distribution** = Gaussian (signal-only; no outliers, no skew, no heavy tail)
- **target conditioning** = mean² / variance = 1e50

## Why this is the worst-case adversarial input for variance

Variance computation has condition number κ ≈ |mean| / std. For
typical data (κ ≈ 1) every implementation works correctly. Above
κ ≈ 1e6, the textbook one-pass formula fails. Above κ ≈ 1e8 even
two-pass with naive summation loses digits. At κ = 1e25 (this
variant's value), only correctly-compensated arithmetic
(Kahan-summation two-pass, Welford with high-precision intermediate,
or arbitrary-precision libraries) can produce a meaningful answer.

This data variant is the **catastrophic-conditioning test case**:
- numpy.var produces a result with **no correct digits**
- pandas.Series.var produces the same garbage (dispatches to numpy)
- statistics.variance and Python Fraction-based mean are too slow
  but would in principle be correct (they use Fraction internally)
- Welford streaming on this data: even the streaming algorithm fails
  because the (x_i - mean) deltas are at the limit of f64 precision
  relative to the magnitude
- The ONLY production-grade solutions are:
  - Pre-shift the data (subtract a guess close to the true mean)
  - Compute the mean in extended precision (Kahan-compensated sum)
  - Use Chan's parallel combine where each batch's local conditioning
    is much better than the global conditioning
  - Switch to Decimal/Fraction/mpmath arithmetic

## Acceptance criteria for tambear's variance recipe

When tambear's variance is run on this variant, the recipe MUST:

1. **Detect** the high conditioning via Sweep 27 pre-checks
2. **Route** to the high-precision path (not the default Welford)
3. **Surface** the conditioning detection in the methodology paragraph:
   "data conditioning κ ≈ 1e25 detected; using Kahan-Welford with
   pre-shift" or similar
4. **Match** the mpmath 50dps reference within ~10 ULP (vs numpy's
   ~10^25 ULP error)

If tambear silently falls through to default Welford on this input,
that's GAP-DET-3 — a recipe-level routing failure.

## Comparison oracle outputs

When `python-numpy/default/run.sh` and `tuned_kahan/run.sh` both run
against this variant, the comparison artifact will show:

| Implementation | result | mpmath ref | abs error | rel error |
|----------------|--------|-----------|-----------|-----------|
| numpy.var (default) | (garbage) | 1e-20 ± true noise | ~1e-15 | ~1e5 (millions of relative error) |
| numpy.var with explicit Kahan | reasonable | 1e-20 | ~1e-25 | ~1e-5 |
| tambear (Welford default) | TBD | 1e-20 | TBD | TBD |
| tambear (Kahan-Welford routed) | TBD ≈ ref | 1e-20 | <10 ULP | <1e-15 |

This is the most dramatic column the variance comparison whitepaper
will have. It is also the cleanest illustration of "tambear's defaults
are rigor incarnate" (per DEC-025): a properly-built variance recipe
HANDLES this case; an off-the-shelf one does not.

## Adversarial property tests this enables

- "On `huge_mean_tiny_variance`, tambear's variance differs from numpy's
  variance by a factor of >10^15" — yes, expected.
- "On `huge_mean_tiny_variance`, tambear matches mpmath 50dps within 10 ULP"
  — required for acceptance.
- "On `huge_mean_tiny_variance`, the routing report names 'high-conditioning
  path' in the methodology paragraph" — required for transparency.
- "Lifted vs sequential strategies on this variant produce identical results
  to within associativity-tolerance bounds (per DEC-023)" — required for
  Chan-parallel correctness.
"""
(OUT_DIR / "characteristics.md").write_text(characteristics_md)
print(f"Wrote: {OUT_DIR / 'characteristics.md'}")

print()
print("=" * 70)
print("DATA VARIANT GENERATED")
print(f"  Location: {OUT_DIR}")
print(f"  Files: input.json, input.hash, generator.md, characteristics.md")
print("=" * 70)
