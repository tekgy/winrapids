# Navigator Questions

If the trek-plan.md is silent on your question and you can't find unblocked work,
drop the question here. Navigator answers within one session.

Format: `[role] [date] — <the question>`

---

<!-- questions below, newest first -->

## Q2 — 2026-04-12 — [math-researcher → pathmaker] Pre-review opinion on OrderStrategy registry entry format

Navigator routed pathmaker's OrderStrategy registry format to me for math-side review (check-ins.md, 2026-04-12 "FOUR — OrderStrategy registry format review incoming"). The specific question navigator asked: does a registry entry's `formal_spec` field need to carry anything beyond prose — specifically, do error bounds and algebraic properties need to be machine-checkable in Phase 1?

My pre-review opinion, so pathmaker has something to react to when they arrive with the draft:

### Recommendation: prose is sufficient for Phase 1, BUT the format must reserve structured slots for future machine-checkability.

**Why prose is enough today:**
1. Every OrderStrategy entry in Phase 1 has a human owner who reviews it manually (me for math, pathmaker for the Rust type, adversarial for testing). The review is the checker, not the type system.
2. Machine-checkable error bounds at the registry level would require a small DSL for expressing `|S - T| ≤ n · 2^-80 · M + 7 · ε · |T|` as a structured tree. That's real work for a Phase 1 feature that nobody is yet mis-using.
3. Prose lets us capture the *why* alongside the *what*. An error bound formula tells you the constant; prose tells you the composition reasoning that produced the constant. The why is where mistakes hide and review catches them.

**Why the format MUST reserve slots for structure:**
1. Phase 2 will add more OrderStrategy variants (RFA with K=4 for higher accuracy, Kahan with compensated summation, tree-fixed-fanout-4, etc.). Each needs the same metadata. If we start with free-form prose, Phase 2 has to retrofit structure or live with unstructured fields.
2. The summit test (Peak 7.11) needs per-kernel tolerances vs mpmath. That tolerance derives from the OrderStrategy entry. Machine-readable → Peak 4's ToleranceSpec auto-populates; prose-only → scientist writes it by hand per kernel, drift-prone.
3. The registry is the contract between math-researcher's design doc and pathmaker's backend emitter. Future reviewers need to diff entries and detect regressions; structure enables that.

### Proposed fields for an OrderStrategy registry entry

Pathmaker, treat this as a math-side wishlist — adjust the Rust type shape to fit whatever the verifier + printer can actually enforce. Illustrative TOML:

```toml
[[order_strategy]]
name = "rfa_bin_exponent_aligned"
description = "Reproducible floating-point accumulation via bin-indexed accumulator (Demmel-Nguyen 2013)"

# Which dataflow patterns this strategy is valid for
applies_to = ["accumulate(All, *, +)", "gather(FixedBlockOrder, +)"]

# The algebraic invariant the strategy upholds (prose in Phase 1)
invariant = """
For any permutation π of the input multiset x[0..n], the result is bit-identical:
  rfa_sum(x) == rfa_sum(π(x))
This holds because:
  (1) AddFloatToIndexed uses the (r | 1) tie-breaking trick, making each
      deposit's rounding direction independent of accumulator state.
  (2) AddIndexedToIndexed is commutative and associative over window-aligned
      fixed-point addition.
  (3) ConvertIndexedToFloat is a deterministic function of the indexed state.
"""

[order_strategy.accuracy]
# Prose description — always required
description = """
|S - T| ≤ n · 2^-80 · M + 7 · ε · |T|
where T is the exact real-valued sum, S is the RFA result, M = max|x_i|,
and ε = 2^-53 (fp64 machine epsilon). See Demmel-Nguyen ARITH 2013 for
the proof and Ahrens-Demmel-Nguyen EECS-2016-121 for the constants.
"""

# Machine-readable formula string — Phase 1 stores it as opaque bytes,
# Phase 2 parses it into an AST for auto-tolerance computation.
formula = "n * pow(2, -80) * M + 7 * eps * abs(T)"
variables = { n = "summand_count", M = "max_abs_input", T = "true_sum", eps = "f64_epsilon" }

# Generation parameters for backend code emission
[order_strategy.params]
bin_width_bits = 40
fold_k = 3
max_deposits_before_renorm = 2048
max_summands = 18446744073709551616  # 2^64

# Cross-backend determinism class
[order_strategy.determinism]
run_to_run = true      # same GPU, same dispatch, same bits
gpu_to_gpu = true      # different tree shapes produce same bits (RFA property)
cpu_to_gpu = true      # bit-exact between CPU interpreter and any GPU backend

# Backend capability requirements
[order_strategy.backends]
cpu = { supported = true, notes = "Pure Rust, no libm" }
ptx = { supported = true, notes = "Requires bitcast + i64 ops (spec §5.4a/b)" }
spirv = { supported = true, notes = "Requires OpBitcast + Int64 capability; ESC-001 subnormal caveat" }

# I8 audit trail — citations only, no source code
references = [
  "Demmel & Nguyen, Fast Reproducible Floating-Point Summation, ARITH 2013",
  "Demmel & Nguyen, Parallel Reproducible Summation, IEEE TC 64(7):2060-2070, 2015",
  "Ahrens, Demmel, Nguyen, Efficient Reproducible Floating Point Summation and BLAS, UCB/EECS-2016-121, 2016",
]
```

### The critical answer to navigator's specific question

> "Does the formal spec of `rfa_bin_exponent_aligned` need to carry the error bound formula `|S - T| ≤ n·2^-80·M + 7·ε·|T|` in a machine-checkable form, or is prose sufficient for Phase 1?"

**Prose is sufficient for Phase 1, BUT the registry entry must carry the formula as a separate `formula` string field alongside the prose description.** The formula doesn't need to be parsed or evaluated by the Rust type system — it just needs to exist as a machine-readable string that future-Peak-4's tolerance computer can consume. Think of it like a `#[test]` attribute's expected-value string: the type system treats it as opaque bytes, but the downstream test runner parses it. Phase 2 adds the parser; Phase 1 stores the string so Phase 2 has something to parse.

**Why this matters:** if pathmaker ships registry entries with only prose, Peak 4's ToleranceSpec ends up with a hand-maintained mapping of `kernel name → tolerance`, which is exactly the kind of drift-prone shadow state the trek is designed to eliminate. If the registry entry carries the formula as a string, Peak 4 can route `tolerance_for(kernel)` through the registry, and in Phase 2 the string becomes a parsed AST with the same call site. Zero migration pain.

**The tiny cost:** one extra string field per registry entry. Pathmaker's TOML or Rust struct gains a `formula: String` or `Option<String>`. That's it.

### What I'd reject

- **Requiring full machine-checkable error bounds in Phase 1.** Too much surface area, too easy to get wrong, not enough consumers to justify the work yet.
- **Prose-only with no structured slots.** Guarantees a Phase 2 rewrite of the registry.
- **A dedicated DSL for error bounds parsed at commit time.** Maybe Phase 3. Not now.

### What I'd specifically want in every entry, even if prose

At minimum, every OrderStrategy entry should carry:
1. `name` — the canonical key referenced by `ReduceBlockAdd.order_strategy` field.
2. `description` — one-sentence prose summary.
3. `invariant` — prose description of the algebraic property the strategy upholds (commutativity, associativity, deterministic rounding, etc.). This is what I review from the math side.
4. `accuracy.description` — prose error bound.
5. `accuracy.formula` — machine-readable string with the bound as a formula (Phase 1 opaque, Phase 2 parsed).
6. `determinism` — a three-boolean record (run_to_run, gpu_to_gpu, cpu_to_gpu) so Peak 4 knows what tolerance policy to use (bit_exact vs within_ulp_bound).
7. `backends` — supported + notes per backend, so the verifier can reject kernels that use an order strategy on an incapable backend.
8. `references` — citations, not source code. The I8 audit trail.

### Action for pathmaker when you arrive

Send me the draft Rust struct or TOML. I'll green-light or propose tweaks. Expected turnaround: <30 minutes. The above is my opinion; your authority on the Rust type shape.

**For `rfa_bin_exponent_aligned` specifically**, the full content is already in `peak6-determinism/rfa-design.md` — the registry entry is a condensed view of the same information. Cross-reference the doc rather than duplicating the whole thing.

---

## Q1 — 2026-04-12 — [math-researcher] Cody-Waite range reduction needs new IR ops; four paths, pathmaker picks one

**STATUS: RESOLVED by pathmaker commit `d36f64c` (2026-04-12).** Path C chosen (bitcast + i64 ops + f64_to_i32_rn + ldexp.f64). All four IR op families delivered. Campsite 2.6 is unblocked.

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
