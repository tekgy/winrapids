# Handoff

## State
Peak 4 fully closed. Oracle runner (d4141ec) + tam_exp.toml (1259620) complete. Six constraint_checks: finite on 709.7827128933839, infinite_positive on 709.7827128933840/710.0/800.0, nonzero_subnormal_positive on -745.0/-745.133. 67 tests green, 6 ignored. VB-005 MITIGATED — fsqrt NaN guard mandatory, no runner changes needed for fsqrt oracle.

## Next
1. **Peak 2 tam_exp lands**: remove `#[ignore]` from `tam_exp_passes_oracle` in `tests/oracle_runner_tests.rs`, swap `|x| x.exp()` for `tambear_libm::tam_exp`.
2. **fsqrt oracle entry** (when adversarial is ready): standard shape, claimed_max_ulp = 0.0, special_values NaN injection, bit_exact_checks for sqrt(+0)/sqrt(+inf). No runner changes needed.

## Context
Injection sets use synthetic-reference path (ULP=0, sign not enforced) for inputs not in reference binary. Every sign-sensitive injection output needs a constraint_checks or bit_exact_checks backstop — documented in logbook `peak4-campsites-4.6-4.8.md`. BitExactConstraint is hex-only; NamedConstraint::parse errors on unknown names at load time (no silent skip).
