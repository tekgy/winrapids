# Logbook: Peak 4 Campsites 4.1, 4.3, 4.4, 4.5, 4.7

**Date:** 2026-04-11  
**Role:** Test Oracle (Gold Standard Scientist)  
**Campsites completed:** 4.1 (TamBackend trait + NullBackend), 4.3 (cross-backend agreement), 4.4/4.5 (ToleranceSpec), 4.7 (hard-cases suite)

---

## What was done

Created `crates/tambear-tam-test-harness/` — a standalone crate with zero
production dependencies.  Contains:

- `backend.rs` — `TamBackend` trait, `BackendRegistry`, `NullBackend` (smoke test only)
- `tolerance.rs` — `ToleranceSpec::bit_exact()` and `ToleranceSpec::within_ulp(bound)` with `ulp_distance()`
- `harness.rs` — `run_all_backends`, `compare_backends`, `assert_cross_backend_agreement`
- `hard_cases.rs` — 13 adversarial input generators covering every known fp failure mode

34 tests pass.  4 tests are `#[ignore]` pending backends or Peak 6.

---

## What almost went wrong

**The `ulp_distance` sign handling.** The standard bit-manipulation trick for ULP
distance requires handling negative floats carefully: IEEE 754 stores negatives in
sign-magnitude order, but `ulp_distance` needs a monotone mapping from float value
to integer for subtraction to work.  The correct transform is: if the sign bit is
set (value is negative), flip all bits except the sign bit.  Equivalently:
`if bits < 0 { bits ^ i64::MAX }` — this converts sign-magnitude to two's-complement
monotone ordering.

The first version I wrote just did `(a_bits - b_bits).unsigned_abs()` on signed
integers, which produces the wrong answer for negative floats.  The test
`negative_values_ordered_correctly` caught this before it shipped.  The fix is
in `tolerance.rs:ordered_bits()`.  Every future reader of this function should
understand why the `^ i64::MAX` is there.

**The `denormals` generator.** Initial implementation used `step * i as f64` where
`step = MIN_POSITIVE / n`.  When `i == n`, this gives exactly `MIN_POSITIVE` —
a normal value, not subnormal.  The test `denormals_are_subnormal` caught it.
Fix: use `n + 1` as the denominator so all `n` values stay strictly below
`MIN_POSITIVE`.

---

## What tempted me off-path

**Using `f64::EPSILON` for the ULP baseline.** Every tutorial on floating-point
comparison reaches for `EPSILON` as the tolerance.  `EPSILON` is approximately
2.2e-16 for f64, which sounds tiny — but as a tolerance policy it's completely
wrong for this use case.  A 1-ULP error near 1.0 is about `1.1e-16` (half of
EPSILON), which sounds fine — but a 1-ULP error near 1e-300 is about `1e-316`,
vastly smaller than EPSILON.  ULP-based comparison is correct-by-construction;
epsilon-based comparison silently widens the tolerance as values get smaller.

Refused this entirely.  The trek's tolerance policy is ULP-based or bit-exact.
No epsilons anywhere in this crate.

**Building "convenience" wrappers for the tolerance API.** There was an impulse
to add `ToleranceSpec::from_function_name("tam_exp")` that auto-looks up the
documented ULP bound.  Refused: the ULP bounds don't exist yet (tambear-libm is
still being built in Peak 2), and adding lookup logic here would create a
dependency on libm documentation that doesn't exist.  The caller provides the
bound explicitly.  When libm lands and documents its ULP bounds, callers
reference them directly.

---

## What the next traveler should know

**Plugging in a real backend:** implement `TamBackend`, call `registry.push(Box::new(your_backend))`,
then `run_all_backends(&registry, &program, &inputs)`.  The trait shape will not
change when real IR types arrive from Peak 1 — the only migration is replacing
the `TamProgram`/`Inputs`/`Outputs` placeholder types in `lib.rs` with imports
from `tambear-tam-ir`.

**The `#[ignore = "xfail_nondeterministic"]` tests:** these are not broken tests.
They are the Test Oracle's guarantee that reduction-based comparisons are
*explicitly blocked* until Peak 6 enforces determinism.  Removing the `ignore`
annotation before Peak 6 lands is an invariant violation (I5).  The Test Oracle
removes them — not the backend implementer.

**The `all_cases_returns_expected_count` test:** there are currently 13 hard cases.
If you add a new case, increment this count.  If you remove one, document why in
the logbook.  The count is a forcing function — it ensures new cases are noticed.

**mpmath coordination:** the math-researcher team is building `peak2-libm/gen-reference.py`
which produces the reference files this harness will use for gold-standard comparison
(campsites 2.2 and 2.3).  The ULP harness in tambear (2.3) and the tolerance
infrastructure here (4.4/4.5) are complementary.  When Peak 2 lands a new libm
function, the Test Oracle adds a row to the parity table and fills in the measured
ULP error.
