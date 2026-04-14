# Trig Family Expedition — Lab Notebook

Observer role. Documents what IS, not what we hope.
Scientific record of every experiment, benchmark, and accuracy measurement.

---

## Baseline Accuracy Survey — 2026-04-13

### Context

First measurement at expedition start. The accuracy report harness (commit 325d9a7)
was run to establish ground-truth numbers for all currently-shipped libm recipes.
This run captures the state post-commit 0bbae82 (Remez refit + Payne-Hanek for sin/cos).

### Method

- Harness: `cargo test --test libm_accuracy_report -- --ignored --nocapture`
- Directory: `crates/tambear/`
- Reference oracle: Rust's native `f64::sin`, `f64::cos`, `f64::exp`, `f64::ln`
  (correctly-rounded on most x86-64 platforms via hardware FPU or glibc; not
  formally guaranteed by the Rust spec but empirically reliable).
- Exception: erf, erfc, tgamma, lgamma have no `f64::*` counterparts in stable
  Rust — cross-strategy comparison used instead (strict/compensated vs
  correctly_rounded). This is an internal consistency check, not an external oracle.
- Input sets: adversarial generator (commit 325d9a7) split into three classes:
  - `sweep`: dense linspace interior (low-bit-entropy mantissas)
  - `region_boundaries`: ±1 ulp from domain transitions
  - `landmarks`: mathematical constants and powers-of-two
- Platform: Windows 11 Pro, AMD Ryzen (build host)
- Build: `test` profile (unoptimized + debuginfo)

### Raw Results

#### exp (natural exponential)

| Strategy          | Input set        |    n | worst | p99 | p95 | p50 |
|-------------------|-----------------|-----:|------:|----:|----:|----:|
| strict            | sweep            | 1553 |     1 |   1 |   1 |   0 |
| strict            | region_boundaries|  170 |     1 |   0 |   0 |   0 |
| strict            | landmarks        |  168 |     1 |   1 |   0 |   0 |
| compensated       | sweep            | 1553 |     1 |   1 |   0 |   0 |
| compensated       | region_boundaries|  170 |     0 |   0 |   0 |   0 |
| compensated       | landmarks        |  168 |     1 |   1 |   0 |   0 |
| correctly_rounded | sweep            | 1553 |     1 |   1 |   0 |   0 |
| correctly_rounded | region_boundaries|  170 |     0 |   0 |   0 |   0 |
| correctly_rounded | landmarks        |  168 |     1 |   1 |   0 |   0 |

Note: 1 reference-infinite value skipped in all sweep rows.

#### log (natural logarithm)

| Strategy          | Input set        |    n | worst | p99 | p95 | p50 |
|-------------------|-----------------|-----:|------:|----:|----:|----:|
| strict            | sweep            | 1286 |     1 |   1 |   1 |   0 |
| strict            | region_boundaries|  295 |     1 |   1 |   1 |   0 |
| strict            | landmarks        |  312 |     1 |   1 |   1 |   0 |
| compensated       | sweep            | 1286 |     1 |   1 |   1 |   0 |
| compensated       | region_boundaries|  295 |     1 |   1 |   1 |   0 |
| compensated       | landmarks        |  312 |     1 |   1 |   1 |   0 |
| correctly_rounded | sweep            | 1286 |     1 |   1 |   1 |   0 |
| correctly_rounded | region_boundaries|  295 |     1 |   0 |   0 |   0 |
| correctly_rounded | landmarks        |  312 |     0 |   0 |   0 |   0 |

#### sin / cos (post Payne-Hanek — commit 0bbae82)

| Recipe | Strategy          | Input set        |    n | worst | p99 | p95 | p50 |
|--------|-------------------|-----------------|-----:|------:|----:|----:|----:|
| sin    | strict            | sweep            | 9821 |     1 |   0 |   0 |   0 |
| sin    | strict            | region_boundaries|  184 |     0 |   0 |   0 |   0 |
| sin    | strict            | landmarks        |  172 |     0 |   0 |   0 |   0 |
| sin    | compensated       | (all same as strict) |
| sin    | correctly_rounded | (all same as strict) |
| cos    | strict            | sweep            | 9821 |     1 |   0 |   0 |   0 |
| cos    | strict            | region_boundaries|  184 |     0 |   0 |   0 |   0 |
| cos    | strict            | landmarks        |  172 |     0 |   0 |   0 |   0 |
| cos    | compensated       | (all same as strict) |
| cos    | correctly_rounded | (all same as strict) |

**Key observation:** All three strategies (strict / compensated / correctly_rounded) return
identical results for sin and cos. The implementation currently aliases all three to
`sin_strict` / `cos_strict`. The "three strategies" structure exists in code but not in math.

#### erf / erfc (oracle: self-comparison against correctly_rounded)

All strategies: worst = 0 ulps across all input sets. Internal consistency is perfect.
Caveat: this is a cross-strategy check, not comparison to an independent oracle.

#### tgamma / lgamma (oracle: self-comparison against correctly_rounded)

All strategies: worst = 0 ulps across all input sets. Same caveat.

### Conclusions

1. **sin/cos baseline is ≤ 1 ulp** on the adversarial test suite post-Payne-Hanek. The
   commit message claimed "1-2 ulps worst case"; the measurement shows worst = 1 ulp
   on the sweep, 0 ulps on landmarks and region boundaries. This is tighter than claimed.

2. **exp and log are ≤ 1 ulp** across all strategies and input sets — better than the
   spec.toml stated target of ≤ 4 ulps (strict) and ≤ 2 ulps (compensated).

3. **The three-strategy differentiation is inoperative for sin/cos.** All three functions
   call the same kernel. This is a known TODO (the code contains explicit `// Worst-case ≤ 2 ulps`
   comments but compensated/correctly_rounded are one-liners delegating to strict).
   This is the first thing to watch: will pathmaker implement genuine differentiation,
   or will the three-strategy skeleton remain a shell?

4. **erf/erfc/tgamma/lgamma oracle concern:** zero ulps vs. self is a consistency check,
   not a correctness verification. The correctly_rounded strategy could be consistently
   wrong and this test would still pass. Need an external oracle (mpmath, R's pgamma/
   lgamma, or Wolfram Alpha spot-checks) to verify these are actually correct.

### Next Steps / Open Questions

- When new functions are implemented (tan, cot, atan, sinh, etc.), the *first* thing
  to measure is the adversarial report output for those functions. Add to the harness.
- The sin/cos `|x| ≤ 1e6` domain cap in the adversarial generator should be removed
  after Payne-Hanek proves out. The current generator deliberately avoids testing
  large |x| — this means the ≤ 1 ulp claim is only validated in-domain.
- Oracle gap: for functions without stable Rust counterparts, we need external
  validation. mpmath at 50-digit precision is the right reference.

---

## Structural Observation — Strategy Shell vs. Strategy Implementation

### 2026-04-13

The compensated/correctly_rounded variants of sin and cos are empty wrappers:

```rust
pub fn sin_compensated(x: f64) -> f64 { sin_strict(x) }
pub fn sin_correctly_rounded(x: f64) -> f64 { sin_strict(x) }
```

This is the same pattern observed in the commit log's pre-Payne-Hanek era.
The `exp` recipe has genuine differentiation (three distinct polynomial degrees,
DD range reduction in compensated, etc.). The `sin` recipe does not yet.

**Hypothesis to test:** When tan, atan2, sinh, etc. are implemented, will they arrive
with genuine three-strategy differentiation, or will the pattern repeat?

**Why this matters:** The accuracy report can't distinguish "three strategies, identical
output" from "one strategy, three wrappers." A reviewer attacking publishability would
ask: "what is the actual accuracy difference between your strict and correctly_rounded
sin implementations?" Right now the answer is: none.

**What to watch for:** The first new function that arrives with non-trivial compensated/
correctly_rounded variants. That's the moment to measure the actual delta.

---

## Confound Notice — Oracle Trustworthiness

### 2026-04-13

The accuracy report uses Rust's `f64::sin` as the oracle for sin/cos. On x86-64
Windows (this platform), `f64::sin` delegates to the CRT's sin, which on MSVC is
known to NOT be correctly-rounded — it can be off by up to 1 ulp vs. the true value.

**Implication:** Measuring tambear's sin against MSVC's sin means our "0 ulps" or
"1 ulp" measurements are relative to a reference that may itself be 1 ulp off. Two
errors could cancel (making us look better) or compound (making us look worse, or
correct errors hiding).

**Severity assessment:** For a library claiming ≤ 2 ulps, this is a significant
confound. The measurement "worst = 1 ulp" could conceal:
- tambear is 0 ulps off true, oracle is 1 ulp off (we look 1 ulp off)
- tambear is 1 ulp off true, oracle is 1 ulp off (we look 0 ulps off)

**Resolution path:** The correctness claim needs validation against a known
correctly-rounded oracle (mpmath, MPFR, or Sollya's gappa). The accuracy_report
harness as written is a fast development-cycle check, not a publishability claim.

---

## Domain Cap Observation — sin/cos adversarial generator

### 2026-04-13

The `sin_cos_adversarial()` generator caps at |x| ≤ 1e6. The commit message for
325d9a7 explicitly notes this: "the current recipe uses a DD-precision range
reduction that degrades for very large arguments."

Post-commit 0bbae82, Payne-Hanek is implemented. The generator cap was NOT updated.

**This means:** The accuracy report is not testing Payne-Hanek paths at all.
The 9821-input sweep covers |x| ≤ 1e6. Payne-Hanek activates at |x| ≥ 2^20·π/2
≈ 1,647,099.3. The current test sweep doesn't reach there.

**The adversarial generator for sin/cos needs extension to cover the Payne-Hanek
regime.** This is a gap in the test coverage, not in the implementation.

**Test to run:** Extend `sin_cos_adversarial()` to include:
- |x| in [1e6, 1e9] — exercises Payne-Hanek
- Known hard Payne-Hanek cases (large multiples of π where the reduced argument
  is tiny — these are the points that triggered the extension loop in the implementation)
- Multiples of 355 (355/113 ≈ π, so 355 mod π is small — medium-sized hard case)

This extension should happen before any accuracy claims about sin/cos on large arguments.

---
