# Invariant-claim divergence pattern — sweep-35 convergence note

**Date**: 2026-05-10 (adversarial, end of Task #8 completion)
**Status**: OBSERVATION — not a bug report, a structural finding across four separate findings this session

---

## The pattern

Four independent findings this session share the same structural shape:
a **label** (docstring, task status, test name, code comment) claims a property about a **substrate**
(code, data structure, implementation), and the substrate doesn't satisfy the property.

| Finding | Label | Substrate | Divergence |
|---|---|---|---|
| TI-CMP-1 | `ieee_eq` semantics documented as value-equality | implementation compared `precision_bits` | representation equality ≠ value equality |
| TI-CMP-2 | `total_cmp` docstring says "matches f64::total_cmp NaN ordering" | negative-NaN payload comparison direction inverted | wrong sort direction for negative-NaN payloads |
| Tasks #1, #2 | status = `completed` | `R:\tambear\crates\tambear\src\recipes\elementary\` does not exist | nothing was implemented in the target codebase |
| expm1.rs ULP budget | module docstring: "≤ 2 ULP strict" | test asserts `max_ulps=4` vs platform libm (not mpmath) | the documented budget and the test budget are different numbers AND the oracle is wrong |

## X-over-Y framing (Pattern 2)

This is **implementation-over-label**: when a label (docstring, task status, test name) and the
implementation disagree, the implementation is what runs — the label is fiction. The failure mode
is trusting the label and acting as if the property holds when it doesn't.

- Y (convenient): the label that claims the property
- X (durable): the implementation / substrate / oracle output
- Failure mode: someone reads the docstring, concludes the invariant holds, doesn't check the code

## Why this is systemic, not local

All four instances appeared in one sweep. They're not correlated by bug class (memory layout,
control flow, build system, oracle choice — four different substrates). What they share is a
**review gap**: each label was written when the implementation was believed to have the property,
and no coupling exists between the label and the implementation that would fail if the property
were later violated.

The antibody class is: **property claims need a failing test that fails if the property is violated.**
- `ieee_eq` semantics → the new `ieee_eq_same_value_different_precision_*` tests in `cmp_tests.rs` are this antibody (now passing after TI-CMP-1 fix).
- `total_cmp` NaN ordering → `total_cmp_neg_nan_payload_ordering_matches_f64_convention` (now passing after TI-CMP-2 fix).
- Task status → no programmatic antibody exists for "task status matches directory contents"; the antibody is human review + `substrate-over-routing` discipline.
- expm1 ULP budget → the new `expm1_ulp_budget_vs_mpmath_oracle_le2` test in the gauntlet (currently `#[ignore]`, fires when Phase A lands).

## What to watch for

When writing documentation or updating task status: the claim is only as good as the test
that would fail if the claim were violated. If no such test exists, the claim is unfalsified
and should be treated as a hypothesis, not a fact.

Specific surfaces to re-check at Phase A landing:
1. Does the new expm1 actually achieve ≤ 2 ULP vs mpmath (not vs platform libm)?
2. Are there other docstring ULP claims in `R:\tambear\crates\tambear\src\` that assert a budget
   but test against platform libm instead of mpmath? (`observer` Watch #2 originally flagged this.)
3. Are there other task completions marked in the task list where the named file/module
   doesn't exist in `R:\tambear\`?

---

*Logged by adversarial at close of Task #8. Route to navigator for sweep-35 retrospective.*
