# Logbook: Peak 4 Campsites 4.6 and 4.8

**Date:** 2026-04-12
**Role:** Test Oracle (Gold Standard Scientist)
**Campsites completed:** 4.6 (e2e recipe tests through TamBackend harness), 4.8 (oracle runner infrastructure)

---

## What was done

**Campsite 4.6** — ported the `gpu_end_to_end.rs` tests to the TamBackend harness path.
Created `tests/e2e_recipes.rs` with 8 tests (5 passing, 3 ignored):
- `cpu_sum_exact_value_1_to_10000` — Σ1..=10000 = 50,005,000 exactly (bit-exact)
- `cpu_variance_pass_sums_{nice,trap}_data` — 3-slot accumulation; trap data documents the gather-side catastrophic cancellation hazard (kernel is correct; gather step loses precision)
- `cpu_pearson_perfect_linear_is_one` — r=1.0 within 2 ULP on y=2.5x+7
- `e2e_harness_detects_real_vs_null_disagreement` — harness violation-detection path confirmed working

**Campsite 4.8** — oracle runner infrastructure (I9′ scientist track).
Created `src/oracle_runner.rs`:
- `load_oracle_entry(toml_path, base_dir)` — reads adversarial's TOML oracle entries
- `run_oracle(entry, candidate_name, candidate)` — produces structured `OracleReport`
- String expression evaluator for corpus values: handles `"ln(2)"`, `"-ln(2)"`, `"ln(2) + 1e-15"`, `"10*ln(2)"`, `"inf"`, `"-inf"`, `"nan"`, and unary minus
- `BitExactConstraint` enum: `Exact(u64)` for hex literals, `NonzeroSubnormal` for class constraints
- Static identity registry: `exp_log_roundtrip`, `exp_negation`, `exp_additivity`, `exp_one_returns_e` — pre-compiled Rust closures keyed by name

Created `tests/oracle_runner_tests.rs` with 5 passing tests including calibration baseline (`f64::exp` ≤ 1 ULP from mpmath), broken-candidate detection, and NaN propagation verification.

---

## What almost went wrong

**`rust,no_run` vs `rust,ignore` in doc-tests.** The module-level example in `ulp_harness.rs` uses a runtime file path that doesn't exist in the doc-test environment. I initially marked it `rust,no_run`, thinking this would prevent execution. It doesn't — `no_run` still compiles, and the compile fails because the path doesn't exist as a compile-time string constant. The fix is `rust,ignore`, which skips compilation entirely. This is a subtle distinction: `no_run` = "compile but don't run," `ignore` = "don't even compile."

**`include_str!` path resolution in integration tests.** Integration test files in `tests/` resolve `include_str!` paths starting from the `tests/` directory. First attempt used `../../campsites/...` (which would work from `src/`), but needed `../../../campsites/...` to reach the workspace root from `tests/`. This isn't obvious from the Rust docs. The compiler's error message was helpful: `help: there is a file with the same name in a different directory` with the corrected path.

**Signed-zero ULP gap — the B1 gap.** The adversarial's review of `exp-design.md` flagged that `exp(-inf)` must return `+0.0`, not `-0.0`. A ULP comparison would pass either, because `+0.0` and `-0.0` are 1 ULP apart in the IEEE ordered representation (adjacent values) — not 0 ULP. A 1-ULP tolerance accepts the wrong sign. This is why `BitExactConstraint::Exact` exists: the bit pattern `0x0000000000000000` is `+0.0` and `0x8000000000000000` is `-0.0`. Both pass a "within 1 ULP" check. Only a bit-pattern check catches the distinction. The oracle runner enforces this via `bit_exact_checks`, not via the ULP path.

**Scientific notation in the expression evaluator.** The corpus entry `"ln(2) + 1e-15"` caused the expression evaluator to fail. My arithmetic parser split on `-` (lowest precedence, right-to-left) and found the `-` in `1e-15`, splitting into `"1e"` and `"15"`. The fix: when scanning right-to-left for binary `-`, skip any `-` immediately preceded by `e` or `E` (it's a scientific notation exponent, not a binary minus). This also affects `+` in `1e+15` forms. The IEEE `1e-15` is not a subtraction operation.

**Identity closures and the multi-function problem.** The design doc had `Box<dyn Fn(f64, &dyn Fn(f64) -> f64) -> f64>` — a single candidate closure. The adversarial (correctly) flagged that `exp_log_roundtrip` requires two functions. Rather than a single candidate fn, each identity closure receives `&HashMap<String, Box<dyn Fn(f64) -> f64>>`. When `tam_ln` isn't registered, `exp_log_roundtrip` returns `f64::NAN` as residual, and the runner treats NaN residual as "function not available — skip, not fail." The adversarial suggested a named-dispatch closure (`dispatch("tam_ln", x)`) as a cleaner interface — agreed, earmarked for the next refactor.

**`TamAtan` match arms missing.** The pathmaker added `TamAtan` to `ast.rs` (part of I11 NaN propagation work) but didn't propagate the variant to `print.rs`, `verify.rs`, and `interp.rs`. This blocked the build when I tried to run the oracle runner tests. Added the three missing match arms as stubs (all panic with the campsite 5.2 stub message, matching the existing pattern for unimplemented transcendentals).

**The injection set ULP "0.0" design decision.** Injection corpus inputs that aren't in the reference binary get `candidate(x)` as their own reference — so ULP = 0 always for those inputs. This is intentional: injection sets exist to test NaN/inf propagation via `special_value_failures`, not precision. But it's visually misleading — a report showing `injection[special_values]: max_ulp=0, passes=true` looks like "perfect accuracy" when it means "NaN propagation verified, precision unverified." This is documented in the code and is a Phase 2 refinement target. The bit_exact_checks section handles sign-sensitive precision for special inputs.

---

## What tempted me off the path

**Refactoring the identity closure interface immediately.** The adversarial's named-dispatch suggestion (`dispatch("tam_ln", x)`) is cleaner than the HashMap at the call site. There was a pull to refactor right away. Resisted — the current design works, the tests pass, and the refactor is contained. Changing a working interface before it's exercised by real code is speculative. The right moment is when the second function's identity checks get added.

**Making injection ULP reports "real" by querying mpmath at runtime.** The synthetic-reference approach feels like a shortcut. The alternative — calling Python/mpmath at runtime for each injection input — would give true ULP numbers for every corpus input. But it would make the runner depend on Python, violate I8 in spirit (the runner would be calling an external evaluation system), and be wrong for the `cody_waite_exact` inputs (F3 from adversarial — these MUST NOT be re-evaluated via mpmath). The current design is the right one: injection sets test propagation, bit_exact_checks test correctness, random sample tests precision.
