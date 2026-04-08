# Expedition Metrics

**Maintained by**: Observer
**Last updated**: 2026-04-01

## The Five Metrics That Matter

Fewer metrics, more signal. These five tell us whether we're building something **right**, not just fast.

### 1. Verified / Implemented

```
Algorithms with gold standard parity PASS / Algorithms implemented in Rust
```

**Current: 148 / ~153** (F00: 19 special functions, F02: 34 linear algebra, F03: 16 signal processing, F06: 22, F07: 16, F08: 10, F09: 12, F10: 7 regression, F25: 11, F26: 10, F31: 12 interpolation, F32: 16 numerical ✓, plus softmax, dot product, L2, DBSCAN, accumulate primitives)

This is the quality metric. An algorithm isn't "done" until it matches an external trusted implementation on identical data. Implementation without verification is hypothesis, not proof.

**Why this matters most**: Every unverified algorithm is a liability. One wrong statistical function poisons every downstream consumer. The whole trust model of "V columns carry confidence" breaks if the DO columns are wrong.

### 2. Adversarial Coverage

```
Algorithms with adversarial test suite PASS / Algorithms implemented
```

**Current: 0 / ~15**

An algorithm can match scipy on normal data and still break on offset data, edge cases, or extreme values. The adversarial mathematician's TC-CANCEL-GRAD test is the canary: if kurtosis breaks at offset 1e4, we catch it before a user does.

**Why it matters**: Silent wrong answers are worse than crashes. Financial data is NEVER centered near zero (prices are ~$100-$5000). Every moment computation on real data hits the cancellation zone unless properly centered.

### 3. Sharing Surface Realized

```
Sharing opportunities actually working in session / Sharing opportunities identified
```

**Current: ~5 / many** (DBSCAN→KNN, stats→regression, manifold distance caching verified)

This measures whether the architectural promise is real. If two algorithms SHOULD share an intermediate but DON'T (because the tags don't match, or the session lookup misses), the 5,820× speedup claim erodes.

**Why it matters**: Sharing is the entire competitive advantage. Without it, tambear is just another math library. With it, it's a compilation system.

### 4. MSR Compression Ratio

```
Leaves computable from MSR / Total leaves in family
```

**Current: 41 / ~90 claimed for F06** (see msr_11field_leaf_enumeration.py)

Measures how much of each family's output is derivable from the accumulated sufficient statistics alone. Higher = more sharing downstream.

**Why it matters**: If the MSR doesn't cover a leaf, that leaf needs raw data access — which means another GPU pass. The whole "accumulate once, extract many" promise depends on MSR coverage being high.

### 5. Cross-Platform Parity

```
(Algorithm, backend) pairs with quantified precision / Total (algorithm, backend) pairs
```

**Current: 0 / ~45** (~15 algorithms × 3 backends)

For every algorithm on every backend (CPU f64, CUDA f64, WGSL f32), we need the max relative error documented. Not "it works" — the actual number.

**Why it matters**: The multi-backend promise is only real if we know the precision cost. "Variance works on WGSL" is meaningless without "max relative error = 3e-4 for data in range [0, 1000]."

---

## Dashboard (updated per session)

| Metric | Value | Target | Trend |
|--------|:-----:|:------:|:-----:|
| Verified / Implemented | 148/153 | 100% | +34(F02) +16(F03) +7(F10) +19(F00) +12(F31) +16(F32) +10(F26) +12(F09) +11(F25) |
| Adversarial Coverage | 0/15 | 100% | — |
| Sharing Realized | 5/? | high | — |
| MSR Compression (F06) | 41/90 | >80% | — |
| Cross-Platform Parity | 0/45 | 100% | — |

## What These DON'T Track (and why)

- **Lines of code**: Not a proxy for progress. 10 lines of correct math > 1000 lines of untested code.
- **Number of algorithms "implemented"**: Counts without verification are vanity metrics.
- **Performance benchmarks**: Important but secondary. A fast wrong answer is worse than a slow right one. Benchmark AFTER verification.
- **Test count**: 300 tests that don't compare against external tools tell us less than 10 tests that do.
