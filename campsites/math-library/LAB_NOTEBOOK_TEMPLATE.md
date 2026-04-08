# Lab Notebook: [Algorithm Name]

**Family**: F## — [Family Name]
**Author**: [pathmaker/naturalist/etc.]
**Date**: YYYY-MM-DD
**Status**: DRAFT | VALIDATED | SIGNED-OFF

---

## 1. Hypothesis

What does this algorithm compute? What are we claiming about our implementation?

- **Mathematical definition**: [Formal definition, cite original paper/textbook]
- **Our claim**: [What we believe tambear achieves — correctness, performance, composability]
- **Prediction**: [What we expect to see in results before running experiments]

## 2. Accumulate Decomposition

How does this reduce to `accumulate(grouping, expr, op) + gather(addressing)`?

| Step | Primitive | Grouping | Expr | Op | Notes |
|------|-----------|----------|------|----|-------|
| 1 | accumulate | ? | ? | ? | |
| 2 | gather | ? addressing | | | |
| 3 | fused_expr | | | | extraction from MSR |

**MSR fields consumed**: [list fields from session]
**MSR fields produced**: [list fields registered to session]
**Polynomial degree**: [degree of extraction — determines sharing family]
**Sharing surface**: [which other algorithms share this accumulation?]

**Is this the minimum decomposition?** [Yes/No — if no, what's simpler?]

## 3. Design Decisions

| Decision | Choice | Alternatives Considered | Rationale |
|----------|--------|------------------------|-----------|
| Formula variant | | | |
| Bias correction | population / sample / both | | |
| NaN handling | propagate / skip / error | | |
| Numerical method | naive / Welford / Kahan / RefCentered | | |
| Centering | yes / no | | Required if polynomial degree > 1 and data not near zero |

## 4. Implementation

**File**: `crates/tambear/src/[module].rs`
**Lines**: [start-end]
**Public API**: `fn name(args) -> Result`

```rust
// Key code snippet (the mathematical core, not boilerplate)
```

**Full version**: Computes from raw data
**Sufficient version**: Extracts from already-accumulated MSR fields

## 5. Results

### 5a. Synthetic Ground Truth

| Test | Input | Expected | Got | Match? | Tolerance |
|------|-------|----------|-----|:------:|-----------|
| Known answer | [describe] | [value] | [value] | ✓/✗ | 1e-N |

### 5b. Edge Cases

| Case | Input | Expected Behavior | Actual | Pass? |
|------|-------|-------------------|--------|:-----:|
| n=1 | | | | |
| n=2 | | | | |
| All identical | | | | |
| All NaN | | | | |
| Contains Inf | | | | |
| Near overflow | | | | |
| Denormals | | | | |

### 5c. Adversarial Cases

| Test ID | Category | Input | Expected | Got | Pass? |
|---------|----------|-------|----------|-----|:-----:|
| TC-CANCEL-GRAD | Catastrophic cancellation | offset + noise | stable result | | |
| [from adversarial suite] | | | | | |

### 5d. Gold Standard Parity (observer sign-off required)

| Oracle | Dataset | Our Value | Oracle Value | Abs Diff | Rel Diff | Tolerance | Pass? |
|--------|---------|-----------|-------------|----------|----------|-----------|:-----:|
| scipy | standard_normal_1000 | | | | | 1e-10 | |
| scipy | uniform_01_1000 | | | | | 1e-10 | |
| scipy | tick_prices_10k | | | | | 1e-10 | |
| R | [dataset] | | | | | 1e-10 | |

## 6. Benchmark

**Hardware**: [GPU model, CPU model, RAM]
**Data**: [n rows, d cols, dtype]

| Implementation | Time | Notes |
|---------------|------|-------|
| tambear standalone (this algo only) | | |
| tambear composable (with session sharing) | | |
| scipy/numpy | | |
| R | | |
| cu* (if applicable) | | |

**Sharing speedup**: [composable / standalone ratio]

## 7. Surprise

What was unexpected? What did we learn that wasn't in the hypothesis?

-

## 8. Composability Contract

```toml
[algorithm]
name = ""
family = ""

[inputs]
required = []
optional = []

[outputs]
primary = ""
secondary = []

[sufficient_stats]
consumes = []
produces = []

[sharing]
provides_to_session = []
consumes_from_session = []

[assumptions]
requires_sorted = false
requires_positive = false
requires_no_nan = false
minimum_n = 1
```

## 9. Cross-Platform Verification

| Backend | Precision | Result | Matches CPU f64? | Max Relative Error |
|---------|-----------|--------|:-:|-----|
| CPU f64 (reference) | f64 | | baseline | 0 |
| CUDA f64 | f64 | | | |
| WGSL f32 | f32 | | | |

## 10. Open Questions

- [ ] [Question 1]
- [ ] [Question 2]

---

**Observer sign-off**: [ ] NOT REVIEWED | [ ] REVIEWED — ISSUES | [✓] SIGNED OFF
**Sign-off date**:
**Sign-off notes**:
