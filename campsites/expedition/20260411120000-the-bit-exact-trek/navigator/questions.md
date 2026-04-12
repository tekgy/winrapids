# Navigator Questions

If the trek-plan.md is silent on your question and you can't find unblocked work,
drop the question here. Navigator answers within one session.

Format: `[role] [date] — <the question>`

---

<!-- questions below, newest first -->

## Q1 — 2026-04-12 — [math-researcher] Cody-Waite range reduction needs new IR ops; four paths, pathmaker picks one

**Blocker for:** Campsite 2.6 (`tam_exp.tam`) and everything downstream that composes on top of exp/log (which is, in practice, all of Peak 2 except atan and hyperbolics).

### The gap

After reading the Phase 1 `.tam` spec (`peak1-tam-ir/spec.md`) end-to-end, every design doc I wrote depends on IR ops that don't exist. The spec has:

- fp ops: `fadd/fsub/fmul/fdiv/fsqrt/fneg/fabs` (§5.3)
- int ops: `iadd.i32/isub.i32/imul.i32/icmp_lt` (§5.4) — no shifts, no masks, no `i64`
- comparisons: `fcmp_gt/fcmp_lt/fcmp_eq` (§5.5)
- `select.f64` and `select.i32` (§5.6)
- transcendental stubs: `tam_exp/tam_ln/tam_sin/tam_cos/tam_pow` (§5.7)
- §11 explicitly: "Type casts between `f64` and `i32`" are **out of scope**

The spec has no fp→int conversion, no bit reinterpret, no `i64`, no integer bitwise ops, and no `ldexp`. That is not sufficient to express Cody-Waite range reduction or exponent-based range reduction for log.

**What I need and why:**

| Need | For which function | Reason |
|---|---|---|
| fp→integer conversion (or workaround) | exp, sin, cos, pow | Compute `n = round(x / ln2)` to drive range reduction |
| Building `2^n` as fp64 | exp, tanh/sinh/cosh via exp | Polynomial reassembly: `result = poly(r) * 2^n` |
| Read fp64 exponent field | log | `log(x) = log(m) + e · ln2` requires extracting `e` |
| `and.i32` + integer constants | sin, cos | `k mod 4` quadrant dispatch (workable without, but uglier) |

**The "magic number" trick** (`round_ties_even(y) = (y + 1.5·2^52) - 1.5·2^52` in fp64) lets me produce `n_f64` (an f64 holding an integer value) without any cast. That works for the Cody-Waite multiplication `n_f64 * ln2_hi`. **But it does not solve the `2^n` reassembly.** There is no way to turn `n_f64` into `2^n_f64` without either (i) an `ldexp`-like op, (ii) integer manipulation of the exponent field, or (iii) a runtime table lookup (which functions cannot do — only kernels have buffer access).

### Four paths — pathmaker picks one

**Path A — Minimal (2 ops):** `f64_to_i32_rne` + `ldexp.f64`. Plus repeal spec §11's "no fp↔int casts" line. Small amendment, unblocks exp/sin/cos/pow. Log still needs a separate `frexp`-like op or `log_exponent_of.f64 → f64` to read the input's exponent.

**Path B — Bitcast + i64 ops (~7 ops):** `bitcast.f64.i64`, `bitcast.i64.f64`, `shr.i64`, `and.i64`, `or.i64`, `iadd.i64`, `isub.i64`. Unblocks everything (Peak 2 AND Peak 6 RFA which needs the same ops). More ops but more general: they're reusable for Kahan/Neumaier/TwoSum, any future bit-level fp work. **Team-lead signaled preference for this path** in the 2026-04-12 check-in. I also recommend this.

**Path C — Full libm-friendly (~10 ops):** Path B plus `f64_to_i32_rne` (so sin/cos quadrant dispatch is clean) plus `and.i32` plus `fp_is_integer.f64` (for pow's integer-b detection). Largest amendment, no workarounds needed anywhere.

**Path D — Dedicated semantic ops (4 ops):** `ldexp.f64`, `frexp_exponent.f64` (returns unbiased exponent as f64 integer), `round_ties_even.f64` (f64→f64 integer via magic-number semantics, but as an op so backends use native `cvt.rne`), `and.i32`. No new integer types, spec stays small. Con: these ops are libm-specific, don't compose into RFA primitives or anything else. Peak 6 still needs Path B for RFA regardless.

### My recommendation: Path B

Because:
1. Team-lead already weighed in for it.
2. Peak 6 RFA needs the same bitcast + i64 ops, so adding them once serves both peaks.
3. The i64 ops are reusable primitives for any future bit-level work (Kahan, TwoSum, TwoProd, frexp/ldexp, fpclassify).
4. Clean universal backend lowerings: PTX `mov.b64` + integer ops, SPIR-V `OpBitcast` + `OpShiftRightLogical`/`OpBitwiseAnd`, CPU `f64::to_bits`/`f64::from_bits`.
5. The spec stays honest — no "exp-flavored" ops that only exist to serve one function.

**Caveat:** Even with Path B, I still want either `f64_to_i32_rne` OR an acceptance that sin/cos quadrant dispatch uses fcmp-cascades instead of `k mod 4 as integer`. The magic-number trick works for `n_f64`, but translating that to an i32 for `and`-with-3 is the one place where Path B by itself is awkward. Either pick Path C (B + `f64_to_i32_rne`) or I absorb the fcmp-cascade cost.

### What I need from pathmaker

1. **Decide the path** (A/B/C/D or variant).
2. **Confirm timeline.** Is this a Phase 1 spec amendment (fast, unblocks 2.6 today) or a Phase 2 deferral (blocks all Peak 2 implementations until Phase 2 starts)?
3. **Confirm `tam_atan` stub addition.** Per adversarial's matrix and my `atan-design.md`, I need exactly one new transcendental stub for `tam_atan`. The other inverse-trig functions (`asin`, `acos`, `atan2`) and all three hyperbolics compose as regular `.tam` functions — no additional stubs required.
4. **Update spec §11.** Whatever path, §11's "Type casts between f64 and i32 are out of scope" line must move.

### What's not blocked by this question

- **`tam_sqrt`** — trivial, lowers to existing `fsqrt.f64`. Implementable now.
- **`tam_atan`, `tam_asin`, `tam_acos`, `tam_atan2`** — only need fp ops, fcmp, and select. The only ask is the one `tam_atan` stub. Could be implemented in Phase 1 spec as-is once pathmaker adds that stub.
- **`tam_tanh`, `tam_sinh`, `tam_cosh`** — compose over `tam_exp`. Implementable *structurally* in the current spec, but blocked in practice because `tam_exp` itself is blocked. Unblocks with exp.

### Impact on my design docs

None algorithmically. The math is the same. If pathmaker picks a path other than "exactly what exp-design.md spelled out," I patch the four function docs (exp, log, sin-cos, pow) in ~30 minutes to match the actual ops. No architectural rework.

### Impact on accuracy-target.md

None. The 1-ULP target is invariant.
