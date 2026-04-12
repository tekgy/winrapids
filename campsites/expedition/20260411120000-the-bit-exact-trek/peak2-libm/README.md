# Peak 2 — tambear-libm Phase 1

**Owner:** math-researcher (design docs) + pathmaker (implementation in .tam IR)

Our own transcendentals, from first principles, bit-exact across every backend.

## What lives here

- `accuracy-target.md` — Campsite 2.1. The ULP bound we commit to, and why.
- `gen-reference.py` — Campsite 2.2. mpmath reference generator.
- `<function>-design.md` — per-function algorithm design documents. Pathmaker implements from these.
- `logbook/` — per-function near-miss entries once implementation is underway.

## Invariants in force (see `../invariants.md`)

- **I1** — No vendor math library. Ever. Not glibc, not `__nv_*`, not `f64::sin` in the interpreter.
- **I3** — No FMA contraction. Every `a*b + c` is two ops (`fmul` then `fadd`). Horner's scheme is canonical.
- **I4** — No implicit reordering. Evaluation sequence is part of the contract.
- **I8** — First-principles only. Read papers. Do not look at glibc/musl/fdlibm/sun-libm source. The coefficients are ours.
- **I9** — mpmath at ≥50-digit precision is the oracle. Not another libm.

## Phase 1 function list

| Function | Campsite | Status |
|---|---|---|
| `tam_sqrt` | 2.4 | trivial (hardware fsqrt) |
| `tam_exp`  | 2.5–2.9 | design doc drafted |
| `tam_ln`   | 2.10–2.12 | design doc pending |
| `tam_sin`  | 2.13–2.14 | design doc pending |
| `tam_cos`  | 2.13, 2.15 | design doc pending |
| (big-arg trig) | 2.16 | deferred to Phase 2 (Payne-Hanek) |
| `tam_pow`  | 2.17 | `exp(b*log(a))` + specials |
| `tam_tanh`, `tam_sinh`, `tam_cosh` | 2.18 | from exp |
| `tam_atan`, `tam_asin`, `tam_acos` | 2.19 | atan is the base |
| `tam_atan2` | 2.20 | quadrant handling |

## The design → implementation pipeline

1. Math researcher writes `<function>-design.md`. Navigator + math researcher agree before any code.
2. Pathmaker implements as a `.tam` function once Peak 1 parser (1.7) is ready.
3. CPU interpreter (Peak 5) executes the `.tam` function op-by-op.
4. ULP harness (2.3) runs 1M samples against mpmath reference (2.2).
5. If `max_ulp > target`, diagnose. Do not relax the bound.
