# Pitfall: One-Pass Variance Catastrophic Cancellation

*Discovered and verified by scout, 2026-04-11.*

---

## The bug

The current `variance()` recipe in `crates/tambear-primitives/src/recipes/mod.rs`
uses the one-pass formula:

```
variance = (Σx² - (Σx)²/n) / (n-1)
```

This formula computes `Σx²` and `(Σx)²/n` separately, then subtracts them.
When the mean is large relative to the variance, both quantities are nearly equal
and the subtraction destroys all significant bits.

## Verified failure cases (Python)

```python
# Case 1: mild (mean = 1e9, variance = 9.167)
data = [1e9 + x for x in range(10)]
# True variance: 9.1667
# One-pass formula: 0.0  (EXACT ZERO — 100% relative error)
# sum_sq and (sum_x)^2/n are bit-identical in fp64

# Case 2: extreme (tiny variance around huge mean)
data = [1e9 + x/1e6 for x in range(100)]
# True variance: 8.4e-10
# One-pass formula: 165.5  (relative error: 2e+11)
```

In Case 1, `sum_sq = 1.0000000090000001e+19` and `(sum_x)^2/n = 1.0000000090000001e+19`
are *identical fp64 values*. The subtraction is exactly zero. No bits of the variance
survive.

## Why the existing tests don't catch this

All tests in `recipes/mod.rs` use small integer inputs (1..5, 1..10). These avoid
the problem because the mean is small (3.0) relative to the variance.

The GPU tests in `gpu_end_to_end.rs` use `sin(i)*10+5` for i in 0..10000. Mean ~5.0,
variance ~50. Still avoids the problem.

Financial data (prices ~100-500, daily changes ~0.01-1.0) WILL trigger this.

## When to expect this in production

Any dataset where:
```
mean^2 / variance >> 2^52 (about 4.5e15)
```
equivalently: `mean / std_dev >> 2^26 (about 6.7e7)`

For price data: mean = 200, std_dev = 0.01 → ratio = 20000. Safe.
For accumulated price data with small daily changes: mean = 200, std_dev = 0.001 → ratio = 200000. Safe.
For ratio data with large offset: mean = 1e9, std_dev = 3.0 → ratio = 3.3e8. Danger zone.

## The fix: Welford's algorithm

```rust
// Welford's one-pass stable variance
// Accumulates (n, mean, M2) with:
//   delta  = x - mean
//   mean  += delta / n
//   M2    += delta * (x - mean_new)
// Final: variance = M2 / (n - 1)
```

Key insight: Welford centers each element on the running mean, so it never computes
`x²` directly. The subtraction `x - mean` involves numbers of comparable magnitude
(both on the order of the variance's scale), so there is no catastrophic cancellation.

Welford is numerically stable and produces correct results even when mean >> std_dev.

Welford IS order-dependent — the result changes slightly if elements are processed
in different orders. This is acceptable for a single-thread sequential accumulation.
For GPU parallel variance, the per-block Welford triples must be merged using the
parallel Welford merge formula (see `peak6-determinism/scout-rfa-terrain.md`).

## What needs to change

1. The `variance()` recipe in `recipes/mod.rs` should be replaced with a Welford
   formulation. The current accumulate+gather structure doesn't directly support
   sequential updates (Welford's update depends on the running mean), so this
   may require a new recipe structure or a new Grouping type.

2. As a minimum acceptable fix: add a test that fails for the one-pass formula
   with large-mean data. This makes the bug visible and prevents regression.

3. The long-term fix is Welford as the primary variance recipe, with the one-pass
   formula kept as a documented "fast but potentially wrong" variant.

## Action items

- [ ] Add a failing test for the one-pass formula with `data = [1e9 + x for x in 0..10]`
      (this test should fail currently — it documents the known bug)
- [ ] Design Welford accumulation in the .tam IR (needs IR Architect input)
- [ ] Replace the `variance()` recipe with Welford
- [ ] Document in the recipe: "one-pass formula is numerically unstable when mean >> std_dev;
      use Welford for production data"

Filed: scout, 2026-04-11. Related pitfall: P08 in `pitfalls/README.md`.
