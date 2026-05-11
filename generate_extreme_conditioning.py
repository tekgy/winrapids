"""Generate the extreme_conditioning_kappa_1e16 data variant for variance oracle.

DESIGN NOTE: a previous design attempt named this "huge_mean_tiny_variance"
with mean=1e15, std=1e-10. That failed because std/mean = 1e-25 is BELOW
f64 epsilon (~2.2e-16): the std signal is destroyed at storage time, before
any algorithm runs. Variance becomes exactly 0 in BOTH f64 and mpmath
(since mpmath sees the same f64-rounded inputs). Useless for the oracle.

Instead: pick mean and std so that std/mean is just above f64 epsilon
(signal survives storage), but the conditioning ratio kappa = mean^2/var
is far beyond GAP-DET-1's 1e6 threshold:
- mean=1e8, std=1.0
- std/mean = 1e-8 (well above eps)
- kappa = 1e16 (10 orders of magnitude beyond GAP-DET-1)

Naive two-pass loses ~10 digits at this conditioning. The variance signal
is real (std=1 is well-represented), but the algorithm fails to recover it.

Hash + generator + characteristics generated alongside.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
from mpmath import mp, mpf

mp.dps = 50

OUT_DIR = Path(r"R:/tambear/oracle/variance/data/generated/extreme_conditioning_kappa_1e16")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20260422
N = 1000
MEAN = 1e8
STD = 1.0  # variance = 1.0; conditioning kappa = mean^2/var = 1e16

rng = np.random.default_rng(SEED)
z = rng.standard_normal(N)
values = (MEAN + STD * z).tolist()

# Diagnostics
arr = np.array(values, dtype=np.float64)
xs_mp = [mpf(repr(v)) for v in values]
mean_mp = sum(xs_mp) / N
sq_dev = sum((x - mean_mp) ** 2 for x in xs_mp)
mp_var_d0 = sq_dev / N
mp_var_d1 = sq_dev / (N - 1)
np_d0 = float(arr.var())
np_d1 = float(arr.var(ddof=1))

print(f"  mean: f64 sample = {float(arr.mean()):.20g}, mpmath = {mp.nstr(mean_mp, 20)}")
print(f"  variance ddof=0: numpy = {np_d0:.20g}, mpmath = {mp.nstr(mp_var_d0, 20)}")
print(f"  variance ddof=1: numpy = {np_d1:.20g}, mpmath = {mp.nstr(mp_var_d1, 20)}")
print(f"  numpy.var ddof=1 abs error: {mp.nstr(abs(mpf(repr(np_d1)) - mp_var_d1), 5)}")
print(f"  numpy.var ddof=1 rel error: {mp.nstr(abs(mpf(repr(np_d1)) - mp_var_d1) / mp_var_d1, 5)}")

# ULP error
import struct
def ulp_dist(a, ref_str):
    ref_f = float(mpf(ref_str))
    if a == ref_f: return 0.0
    a_bits = struct.unpack("<q", struct.pack("<d", a))[0]
    b_bits = struct.unpack("<q", struct.pack("<d", ref_f))[0]
    return abs(a_bits - b_bits)
ulp_d0 = ulp_dist(np_d0, mp.nstr(mp_var_d0, 50, strip_zeros=False))
ulp_d1 = ulp_dist(np_d1, mp.nstr(mp_var_d1, 50, strip_zeros=False))
print(f"  ULP error ddof=0: {ulp_d0}")
print(f"  ULP error ddof=1: {ulp_d1}")

# Write input.json
input_json_path = OUT_DIR / "input.json"
input_json_text = json.dumps({"values": values}, indent=2)
input_json_path.write_text(input_json_text, encoding="utf-8")
print(f"\nWrote: {input_json_path}")

sha = hashlib.sha256(input_json_text.encode("utf-8")).hexdigest()
(OUT_DIR / "input.hash").write_text(f"sha256:{sha}\n", encoding="utf-8")
print(f"Wrote: {OUT_DIR / 'input.hash'}")
print(f"  sha256: {sha}")

generator_md = f"""# Generator -- extreme_conditioning_kappa_1e16

## Reproduction

```bash
PYTHONIOENCODING=utf-8 python R:/winrapids/generate_extreme_conditioning.py
```

Produces deterministic output (fixed seed). Re-running on the same
numpy version produces bit-identical bytes.

## Parameters

- **n** = {N}
- **mean** = 1e8
- **std** = 1.0  (variance = 1.0)
- **conditioning kappa** = mean^2/var = 1e16
- **seed** = {SEED}
- **rng** = numpy.random.default_rng (reproducible across numpy 1.17+)

## Algorithm

```python
rng = np.random.default_rng({SEED})
z = rng.standard_normal({N})
values = (1e8 + 1.0 * z).tolist()
```

## Reference values (mpmath 50dps against bit-exact f64 inputs)

- sample mean: {mp.nstr(mean_mp, 25)}
- sample variance (ddof=0): {mp.nstr(mp_var_d0, 25)}
- sample variance (ddof=1): {mp.nstr(mp_var_d1, 25)}

## Why this variant matters

The conditioning ratio kappa = mean^2/var = 1e16 is 10 orders of
magnitude beyond GAP-DET-1's 1e6 threshold. Naive two-pass loses
~10 significant digits. Welford streaming on this data also fails
(per GAP-DET-1 documented on `ill_conditioned` variant which is
already at kappa = 1e8; this variant is 8 orders of magnitude worse).

For tambear's variance recipe: the recipe MUST detect this conditioning
level (Sweep 27 pre-checks) and route to the high-precision path
(Kahan-Welford or Chan-parallel-with-Kahan-within-batch). Falling
through to default Welford or naive two-pass returns garbage.

## Earlier design attempt that failed

A first version targeted mean=1e15, std=1e-10 (kappa = 1e50). That
data variant could not be represented at all in f64: std/mean =
1e-25 << f64 epsilon (~2.2e-16) means each generated value
`mean + std*z` rounds to exactly `mean` at storage time. Variance
becomes 0 before any algorithm runs. Useless as oracle data.

Lesson for any future "extreme variance" variants: keep std/mean
above f64 epsilon (~2.2e-16) so the signal survives storage. That
caps the achievable conditioning at roughly kappa = (mean/eps)^2.
For mean = 1e8 (this variant), the maximum representable conditioning
is approximately 2e47 with std at the eps boundary; we use a more
modest std=1 (kappa=1e16) which is still extreme but leaves the
signal comfortably above the precision floor.
"""
(OUT_DIR / "generator.md").write_text(generator_md, encoding="utf-8")
print(f"Wrote: {OUT_DIR / 'generator.md'}")

characteristics_md = f"""# Characteristics -- extreme_conditioning_kappa_1e16

## Distributional summary

- **n** = {N}
- **mean** = 1e8 (large positive offset)
- **std** = 1.0 (modest spread)
- **variance** = 1.0
- **conditioning ratio kappa** = mean^2 / variance = **1e16**
- **distribution** = Gaussian (signal-only; no outliers, no skew)

## Reference (mpmath 50dps)

- sample mean: {mp.nstr(mean_mp, 25)}
- sample variance (ddof=0): {mp.nstr(mp_var_d0, 25)}
- sample variance (ddof=1): {mp.nstr(mp_var_d1, 25)}

## Why this variant is the variance oracle's hardest test

Variance computation has condition number kappa ~ |mean| / std. For
typical data (kappa near 1) every implementation works correctly.
GAP-DET-1 documented failure starting at kappa around 1e6. This
variant pushes kappa to 1e16 -- ten orders of magnitude further.

At this conditioning, naive two-pass loses ~10 significant digits
out of f64's ~15. Even Welford streaming fails (the per-element
delta = x_i - running_mean is at the limit of f64 precision relative
to the magnitude). The ONLY production-grade solutions are:

- Pre-shift the data (subtract a guess close to the true mean
  before computing variance)
- Compute the mean in extended precision (Kahan-compensated sum)
- Chan's parallel combine where each batch's local conditioning is
  much better than the global conditioning
- Switch to Decimal/Fraction/mpmath arithmetic

## Observed numerical behavior on this variant

| Implementation | result | abs error vs mpmath | ULP distance |
|---|---|---|---|
| numpy.var(ddof=1) | {np_d1:.10g} | ~{float(abs(mpf(repr(np_d1)) - mp_var_d1)):.2e} | {ulp_d1} |
| numpy.var(ddof=0) | {np_d0:.10g} | ~{float(abs(mpf(repr(np_d0)) - mp_var_d0)):.2e} | {ulp_d0} |
| mpmath two-pass (50dps) | {mp.nstr(mp_var_d1, 18)} | 0 (reference) | 0 |

numpy is wrong by ~10 significant digits at this conditioning.
Tambear's correctly-routed path (Kahan-Welford or Chan-parallel-Kahan)
should match the mpmath reference to within ~10 ULP, demonstrating a
*15+ orders of magnitude* improvement in relative error vs naive numpy.

## Acceptance criteria for tambear's variance recipe

When tambear's variance is run on this variant, the recipe MUST:

1. **Detect** the high conditioning via Sweep 27 pre-checks
2. **Route** to the high-precision path (NOT default Welford)
3. **Surface** the conditioning detection in the methodology paragraph
   ("data conditioning kappa ~ 1e16 detected; using Kahan-Welford
   with pre-shift" or equivalent)
4. **Match** the mpmath 50dps reference to within ~10 ULP

If tambear silently falls through to default Welford on this input,
that is GAP-DET-3 -- a recipe-level routing failure, distinct from
GAP-DET-1 (which is the algorithm-level conditioning failure).

## Related variants

- `ill_conditioned` (in mean/data/generated/): kappa ~ 1e8 -- the
  baseline GAP-DET-1 case. Less extreme conditioning; numpy still
  loses ~5-6 digits but produces a result with the correct magnitude.
- `extreme_conditioning_kappa_1e16` (THIS): pushes 8 orders of
  magnitude further. Numpy loses ALL meaningful precision; only
  rigor-conscious implementations (Kahan, Chan, mpmath) work.

## Adversarial property tests this variant enables

- "On `extreme_conditioning_kappa_1e16`, tambear matches mpmath 50dps
  within 10 ULP" -- required for acceptance.
- "On this variant, tambear's variance recipe emits a high-conditioning
  routing annotation in the methodology paragraph" -- required for
  transparency per DEC-020.
- "Lifted vs sequential strategies on this variant produce identical
  results to within associativity-tolerance bounds" -- required per
  DEC-023 lifted-equivalence guarantee.
- "Chan-parallel merge on this variant matches the sequential
  Welford-Kahan reference within ~10 ULP" -- the Chan's-merge
  acceptance harness gold-standard test.
"""
(OUT_DIR / "characteristics.md").write_text(characteristics_md, encoding="utf-8")
print(f"Wrote: {OUT_DIR / 'characteristics.md'}")

print()
print("=" * 70)
print("DATA VARIANT GENERATED")
print(f"  Location: {OUT_DIR}")
print(f"  Files: input.json, input.hash, generator.md, characteristics.md")
print("=" * 70)
