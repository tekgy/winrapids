# Family 25: Information Theory — Adversarial Test Suite

**Author**: Adversarial Mathematician
**Date**: 2026-04-01
**Status**: PROVEN numerically
**Code**: `crates/tambear/src/information_theory.rs`
**Proof script**: `docs/research/notebooks/f07-hypothesis-adversarial-proof.py` (Section 3)

---

## Operations Tested

| Operation | Code Location | Verdict |
|-----------|--------------|---------|
| shannon_entropy | information_theory.rs:66-68 | OK |
| renyi_entropy | information_theory.rs:84-109 | OK (alpha limits correct) |
| tsallis_entropy | information_theory.rs:115-125 | OK |
| kl_divergence | information_theory.rs:136-139 | OK (returns Inf when q=0, p>0) |
| js_divergence | information_theory.rs:146-150 | OK (symmetric, bounded) |
| cross_entropy | information_theory.rs:156-163 | OK |
| mutual_information | information_theory.rs:175-203 | OK (clamps to >= 0) |
| normalized_mutual_info | information_theory.rs:214-243 | OK |
| variation_of_information | information_theory.rs:248-268 | OK |
| conditional_entropy | information_theory.rs:273-287 | OK |
| joint_histogram | information_theory.rs:306-323 | **HIGH** (i32 overflow) |
| adjusted_mutual_info | information_theory.rs:370-397 | **MEDIUM** (O(n) memory) |
| probabilities | information_theory.rs:37-41 | **LOW** (no input validation) |

---

## Finding F25-1: Joint Histogram i32 Overflow (HIGH)

**Bug**: `joint_histogram` computes composite keys as `x * ny as i32 + y` where x and y are i32. When `nx * ny > i32::MAX` (2,147,483,647), the composite key overflows, wrapping to negative values that index wrong bins in the scatter.

**Threshold**: `sqrt(i32::MAX) = 46,340`. Any categorical variables with more than 46K categories will silently produce wrong contingency tables.

**Example**:
```
nx = 50000, ny = 50000
max composite key = 49999 * 50000 + 49999 = 2,499,999,999
i32::MAX = 2,147,483,647
OVERFLOW -> wraps to negative -> wrong scatter bin
```

**Impact**: Silent wrong MI, NMI, AMI for high-cardinality categorical data. No error or warning.

**Fix**: Either:
- Use i64 keys (requires `scatter_phi` to accept i64)
- Validate `nx * ny < i32::MAX` and return error if exceeded

---

## Finding F25-2: AMI Expected MI Memory Scaling (MEDIUM)

**Bug**: `expected_mutual_info` allocates `vec![0.0f64; n_int + 1]` for log-factorials. Memory usage:

| n | log_fact memory |
|---|----------------|
| 1K | 8 KB |
| 100K | 800 KB |
| 1M | 8 MB |
| 10M | 80 MB |
| 100M | 800 MB |
| 1B | 8 GB (OOM) |

**Impact**: AMI crashes with OOM for datasets > ~100M points. No graceful degradation.

**Fix**: Use Stirling's approximation for `log(n!)` when n > threshold (e.g., 1000). Stirling: `log(n!) ~ n*log(n) - n + 0.5*log(2*pi*n)`. Error < 1/(12n). This gives O(1) memory for arbitrarily large n.

---

## Finding F25-3: No Input Validation (LOW)

- `shannon_entropy(&[f64])` accepts any slice. Negative values silently treated as 0 (via `p_log_p`'s `if p <= 0.0` check).
- `kl_divergence(p, q)` does not verify `sum(p) == 1` or `sum(q) == 1`. Invalid distributions produce wrong results without warning.
- `probabilities(counts)` correctly handles all-zero by returning zeros.

**Impact**: Low -- callers are responsible for valid inputs. But a debug_assert would catch misuse.

---

## Test Vectors

### TV-F25-ENT-01: Shannon uniform
```
probs = [0.25, 0.25, 0.25, 0.25]
expected: ln(4) ~ 1.38629
```

### TV-F25-ENT-02: Shannon deterministic
```
probs = [1.0, 0.0, 0.0]
expected: 0.0
```

### TV-F25-ENT-03: Shannon concentrated
```
probs = [1e-15/9]*9 + [1 - 1e-15]
expected: < 1e-12
```

### TV-F25-REN-01: Renyi converges to Shannon at alpha=1
```
probs = [0.3, 0.5, 0.2]
renyi(probs, 1.0) must equal shannon(probs) within 1e-10
renyi(probs, 0.9999) must equal shannon(probs) within 1e-3
```

### TV-F25-REN-02: Renyi uniform invariance
```
probs = [0.25, 0.25, 0.25, 0.25]
For all alpha in {0.5, 1.0, 2.0, 10.0}: renyi ~ ln(4)
```

### TV-F25-KL-01: Identical distributions
```
p = q = [0.3, 0.5, 0.2]
expected: 0.0
```

### TV-F25-KL-02: Disjoint support
```
p = [0.5, 0.5, 0.0], q = [0.0, 0.0, 1.0]
expected: +Infinity
```

### TV-F25-KL-03: Asymmetry
```
p = [0.9, 0.1], q = [0.1, 0.9]
kl(p,q) == kl(q,p) (this specific case is symmetric by swap)
kl(p,q) > 0
```

### TV-F25-JS-01: Bounded by ln(2)
```
p = [1.0, 0.0], q = [0.0, 1.0]
expected: ln(2) ~ 0.69315 (maximum JS divergence)
```

### TV-F25-JS-02: Symmetry
```
For any p, q: js(p,q) == js(q,p)
```

### TV-F25-CE-01: Cross-entropy identity
```
H(P, P) = H(P)
```

### TV-F25-CE-02: Cross-entropy decomposition
```
H(P, Q) = H(P) + D_KL(P || Q)
```

### TV-F25-MI-01: Independent variables
```
table = [0.15, 0.10, 0.45, 0.30] (2x2, p(x,y) = p(x)*p(y))
expected MI: 0.0
```

### TV-F25-MI-02: Perfect correlation
```
table = [5.0, 0.0, 0.0, 5.0] (2x2, Y=X)
expected MI: ln(2)
```

### TV-F25-NMI-01: Perfect NMI
```
Same as MI-02
expected NMI(arithmetic): 1.0
```

### TV-F25-VI-01: Perfect agreement
```
Same as MI-02
expected VI: 0.0
```

### TV-F25-AMI-01: Perfect agreement
```
labels_true = [0,0,1,1,2,2], labels_pred = [0,0,1,1,2,2]
expected AMI: 1.0
```

### TV-F25-AMI-02: Independent labeling
```
labels_true = [0,0,0,1,1,1,2,2,2]
labels_pred = [0,1,2,0,1,2,0,1,2]
expected AMI: <= 0.01 (near zero, possibly negative)
```

### TV-F25-JH-01: Overflow detection (BUG CHECK)
```
nx = 50000, ny = 50000
Expected: error or correct result
Currently: silent overflow, wrong bins
```

---

## Priority Summary

| Finding | Severity | Impact | Fix |
|---------|----------|--------|-----|
| F25-1: i32 overflow | **HIGH** | Silent wrong contingency tables | i64 keys or validation |
| F25-2: AMI OOM | **MEDIUM** | Crash at n > 100M | Stirling approximation |
| F25-3: No input validation | **LOW** | Wrong results from invalid input | debug_assert |
