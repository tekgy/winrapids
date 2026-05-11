# BZ Algorithm Preconditions — F13 Antibody Map

**Created:** 2026-05-08 (continuation session, tambear-sweep31-finish)
**Author:** aristotle
**Brief:** F13 says every rule with a scope precondition needs an antibody at construction time. The multi-limb arith.rs unstub adds four new BZ algorithms, each a rule-with-scope-precondition. This doc enumerates the preconditions per algorithm and proposes the antibody for each, so pathmaker can file them as part of the impl.

**Inputs:**
- DEC-031 §3.5 (BZ reference); DESIGN.md §3 algorithm-decision table
- Brent-Zimmermann *Modern Computer Arithmetic* (2nd ed.) Algorithms 3.1, 3.3, 3.5, 3.10
- F13 (ratified 2026-05-08) — `R:\winrapids\campsites\tambear-formalize\survey\20260508123003-aristotle\f13-antibodies-for-scope-precondition-rules.md`
- ty.rs canonical-form definition (top-bit-set at position `(p-1) % 64` of top limb; `limbs.len() == ceil(p/64)`)
- arith.rs current state (f64-fast-path FIRST; multi-limb FALLBACK panic'd)

---

## Master rule

**For every BZ algorithm preconditions list below, the antibody is enforced at function entry. Out-of-scope inputs produce a debug_assert panic in test builds. Release builds may rely on call-site invariants from the type system (e.g., `BigFloatKind::Normal` already excludes ±0/±Inf/NaN), but the documented preconditions stay in source as inline comments.**

The pattern: each precondition is one of (a) a regime check that implies algorithm correctness, OR (b) a guard-bit/iteration-count derivation that depends on `p`. Both kinds need explicit code, not assumed-by-convention behavior.

---

## BZ Algorithm 3.1 — Add / Sub (multi-limb path)

### Surface

Two `BigFloatKind::Normal` operands. Compute `c = a + b` (or `a - b`) at result precision `p_res = max(p_a, p_b)`, rounded per `rounding`.

### Preconditions

**R1.1 — Operand alignment shift is finite.**
After exponent alignment, the smaller-magnitude operand is shifted right by `|a.exponent - b.exponent|` bits relative to the larger. If shift ≥ p_res + 2, the smaller operand contributes nothing except possibly to the rounding direction (via guard + sticky bits).
- **Antibody:** explicit branch `if shift >= p_res + 2` that returns `larger` with possible rounding adjustment; otherwise full alignment + add.
- **Without antibody:** silent loss of the rounding tie-break case, OR a UB-shift if shift ≥ 64 (limb-shift overflow).
- **Test witness:** `bf(1.0).add(&bf(1e-300), RNE)` at p=200 — shift is 996 bits; result must be `1.0` exactly with rounding direction correctly determined.

**R1.2 — True subtraction handles cancellation.**
When sign(a) ≠ sign(b) and `|a| ≈ |b|`, magnitude after subtract may have far fewer significant bits than either operand (catastrophic cancellation). Renormalization must scan from MSB, find the new top bit, shift mantissa left, decrement exponent.
- **Antibody:** dedicated `renormalize_after_subtract` that handles arbitrary-leading-zero-count case. Test: `bf(1.0 + 1e-100).sub(&bf(1.0), RNE)` at p=200 — result has only ~330 significant bits below the original 200; renormalize must shift up.
- **Without antibody:** result has leading-zero limbs while claiming `kind: Normal`; canonical-form invariant (top-bit-set at position (p-1)%64) silently violated.
- **Test witness:** `bf(1.5).sub(&bf(1.5).sub(&bf(1e-100), RNE), RNE)` — exercises the cancellation path explicitly.

**R1.3 — Exponent stays in i64 range.**
If both operands have top bit set and full add produces carry-out, exponent increments. For p ≤ 1024 and reasonable inputs, exponent fits easily in i64. But the antibody is still needed as a structural invariant.
- **Antibody:** `debug_assert!(c.exponent.checked_add_or_sub(1).is_some())` or equivalent at the carry-handling site.
- **Without antibody:** silent exponent UB at extreme inputs.

**R1.4 — Result canonicalization.**
After add/sub + renormalize, `c.limbs[top].leading_zeros() == (64 - ((p_res - 1) % 64) - 1)` and `c.limbs.len() == ceil(p_res / 64)`. Round per RoundingMode using bits below position p_res.
- **Antibody:** explicit `canonicalize` step at end of normal_add. Plus integration test that `c.limbs.len() == ceil(c.precision_bits / 64)` for many random inputs.

---

## BZ Algorithm 3.3 — Schoolbook Multiplication

### Surface

Two `BigFloatKind::Normal` operands. Compute `c = a × b` at `p_res = max(p_a, p_b)`, rounded per `rounding`.

### Preconditions

**R3.1 — Full unrounded product fits in 2n limbs.**
Schoolbook of `n × n` limbs produces `2n` limbs. For DEC-031 in-scope precision (≤ 1024 bits = 16 limbs), the full product is up to 32 limbs (2048 bits). All in scope; tier cap saturates at 1024 (per §3.8).
- **Antibody:** allocate `Vec::with_capacity(2 * n)` for the intermediate product. Explicit check that `n_a + n_b ≤ 32` in debug builds (Karatsuba threshold; FFT excluded).
- **Without antibody:** silent OOM at extreme p. Realistically, the §3.8 saturation kicks in first, so this is defense-in-depth.

**R3.2 — Top-bit position detection is two-cases.**
Both operands have their top bits set (canonical form). Their product has top bit at either position `2p - 1` (when both operands round-up to ≥ 1.0) OR position `2p - 2` (when both ≤ 1.0 mantissa). Renormalization must check.
- **Antibody:** explicit if/else on the top limb's leading_zeros after multiply, picking the shift amount accordingly.
- **Without antibody:** **THIS IS THE SILENT-FAILURE FOOTPRINT WORTH FLAGGING.** If implementation always shifts by p, the case where actual top is at 2p-2 produces a value off by factor of 2. The BigFloat is canonical-form-correct (top bit set) but its exponent is wrong. No assertion fires. No round-trip test catches it because the exponent error is an exact factor of 2.
- **Test witness:** `bf(0.6).mul(&bf(0.6), RNE)` at p=200 — actual product 0.36, top bit at position 2p-2. If wrong shift, result is 0.72.

**R3.3 — Guard bits below round position.**
After the 2n-limb product, the round position is at bit `2p - p_res`. To round correctly per RoundingMode, need at least 2 guard bits + sticky bit below this position. With `p_res ≤ p_a + p_b`, this is automatic; with `p_res = max(p_a, p_b)` and balanced operands, equally automatic.
- **Antibody:** the schoolbook FULLY computes 2n limbs before rounding, so guard bits are always available. Antibody is the "do not truncate before rounding" property of the schoolbook implementation. Comment + integration test on rounding direction at hard cases (ties).

---

## BZ Algorithm 3.5 — Newton-Raphson Reciprocal (for Division)

### Surface

`BigFloatKind::Normal` operands. Compute `c = a / b` at `p_res = max(p_a, p_b)`, rounded per `rounding`.

Strategy: compute `1/b` via Newton iteration at p_res + 50 guard bits, then multiply by `a`, then round.

### Preconditions

**R5.1 — f64 initial guess is well-defined.**
Initial guess `x_0 = f64::from(f64-approximation-of-b).recip()`. Requires `b` to fit in f64 without overflow/underflow.
- **Antibody:** explicit handling of the case where b's exponent is outside f64's [-1022, 1023]. For extreme b, scale b by power-of-2 to bring it into f64 range, do Newton, scale back.
- **Without antibody:** division of very large or very small numbers silently produces wrong result (Newton converges to wrong limit when initial guess is f64::INFINITY or f64::MIN_POSITIVE).
- **Test witness:** `bf(1.0).div(&bf(2.0_f64.powi(1100)), RNE)` — b is finite as BigFloat but overflows f64; Newton initial guess via raw f64 reciprocal is broken.

**R5.2 — Iteration count `⌈log₂(p_work / 53)⌉ + 2` is sufficient.**
Newton doubles correct bits per iteration. Starting from 53 bits (f64 initial guess), reach `p_work = p_res + 50` bits in `⌈log₂(p_work / 53)⌉` iterations. Add 2 for safety per BZ.
- **Antibody:** iteration count COMPUTED from p_work, NOT hardcoded. Code: `let n_iter = ((p_work as f64) / 53.0).log2().ceil() as u32 + 2;`. For p_work=200+50=250, n_iter = ⌈log₂(250/53)⌉ + 2 = 3 + 2 = 5. For p_work=1024+50=1074, n_iter = 5 + 2 = 7.
- **Without antibody (hardcoded count):** for p > some threshold, Newton is non-converged at end of loop; result has fewer correct bits than claimed. Silent precision loss.
- **Test witness:** at p=1024, divide two operands and verify result has at least 1024 correct bits via mpmath comparison (scientist's task).

**R5.3 — 50 guard bits is sufficient for correct final round.**
After Newton at p_work = p_res + 50, the reciprocal is correct to within ~0.5 ULP of p_work. Multiplying by a (correct to within 0.5 ULP at its own precision) and rounding to p_res is correct iff the cumulative error is < 0.5 ULP at p_res. With 50 guard bits, headroom is `2^50` — plenty for any sane operand class.
- **Antibody:** documented in DESIGN.md. The 50 is locked in §3 dispatch table. Comment in code: `// Per BZ §3.5 + DESIGN.md §3: 50 guard bits suffice for in-scope p ≤ 1024.`
- **Without antibody:** for some hard rounding cases (ties exactly at p_res), 50 guard bits may be insufficient if Newton's iterate has worse-than-claimed accuracy. Defense: verify against mpmath at all hard cases.

**R5.4 — Sign of result.**
`sign(a/b) = sign(a) XOR sign(b)`, regardless of magnitude. For special-value paths, already handled in arith.rs's `div`. For Normal/Normal, the multi-limb path sees abs values; the sign is composed at end via `mul_sign`.
- **Antibody:** explicit `result.sign = mul_sign(a.sign, b.sign)` after the magnitude division. Already correct in current shape; preserve.

---

## BZ Algorithm 3.10 — Newton Iteration Sqrt

### Surface

`BigFloatKind::Normal` operand with sign=false (negative-and-not-zero already returns NaN). Compute `c = sqrt(a)` at `p_res = a.precision_bits`, rounded per `rounding`.

Strategy: Newton iteration `x_{n+1} = (x_n + a/x_n) / 2` at p_work = p_res + 50, initial guess from f64 sqrt.

### Preconditions

**R10.1 — Even-exponent form for the working argument.**
Newton converges quadratically when input is in `[0.25, 1)` or similar. Implementation: extract `a = m × 2^e` with mantissa `m ∈ [1, 2)`, e even (so sqrt(2^e) = 2^(e/2) exactly). If e is odd, halve via `a = (m/2) × 2^(e+1)` so the new e' = e+1 is even.
- **Antibody:** explicit `if exponent % 2 != 0 { adjust }` step at function entry. Working_a has even exponent.
- **Without antibody:** Newton's first iteration produces an iterate of wrong magnitude (off by factor sqrt(2)); subsequent iterations don't recover.
- **Test witness:** `bf(2.0).sqrt(RNE)` — a's exponent is 1 (odd); result must equal sqrt(2) ≈ 1.41421...

**R10.2 — Newton iteration count.** Same as R5.2. Doubles correct bits per iteration; `⌈log₂(p_work / 53)⌉ + 1` iterations sufficient (sqrt is slightly faster than reciprocal because doubling is cheaper than multiply-and-subtract).
- **Antibody:** count COMPUTED from p_work.

**R10.3 — Final exponent reconstruction.**
After Newton converges on `sqrt(working_a)`, multiply by `2^(e_working / 2)` to recover the true sqrt. e_working is even by R10.1, so e_working / 2 is integer.
- **Antibody:** explicit `final_exponent = working_exponent / 2 + adjustment` step.
- **Without antibody:** result magnitude wrong by factor 2^k for some k.

**R10.4 — Sign of result is positive (when not NaN).**
sqrt of positive Normal returns positive Normal. sqrt of -0 returns -0 (already handled in arith.rs's special-value dispatch). sqrt of any other negative returns NaN (already handled).
- **Antibody:** the special-value dispatch at arith.rs:351-376 already enforces. The multi-limb path entered only with sign=false-and-not-zero; preserve.

---

## Cross-cutting antibody pattern

Every algorithm above shares the same antibody schema:

1. **Function entry:** debug_assert preconditions on operand class (already in arith.rs's match dispatch — preserve).
2. **Initial setup:** scale to working representation (R5.1's f64-range adjustment, R10.1's even-exponent form). Document why.
3. **Iteration / accumulation:** count COMPUTED from p_work, not hardcoded. Document the formula in source.
4. **Result construction:** canonicalize (R1.4, R3.2). Verify `limbs.len() == ceil(p / 64)` and top-bit-set invariant.
5. **Final round:** apply RoundingMode. Document the round position formula.

The F13 mechanical-artifact for each algorithm is the integration test that exercises the precondition boundary AND the mpmath comparison that catches silent drift. Both are in scope for scientist's Task #8 (cross-precision proptest harness) — the scope-precondition-witness shape lives there, not in arith.rs's source comments.

The shape: source code documents the rule + scope; tests witness the boundary; mpmath comparison catches silent drift inside scope.

---

## Recommendations to pathmaker (priority order)

1. **Compute iteration counts from p_work**, never hardcode. The single most common silent-failure mode in Newton-method implementations. Code review for this on PR.

2. **Top-bit position detection in BZ Alg 3.3 must be two-cases** (R3.2). The factor-of-2 silent error is the worst kind — passes most tests, fails one where rounding direction depends on the lost bit.

3. **f64-range adjustment for Newton initial guesses** (R5.1, R10.1). Without scaling, large-magnitude inputs to div/sqrt silently produce wrong results because Newton's initial guess is f64::INFINITY.

4. **Renormalization after subtract** (R1.2). Cancellation case is the easiest to forget because most subtraction tests don't trigger it. Specifically test `a - (a - epsilon)` patterns.

5. **Document each algorithm's preconditions inline** in arith.rs at function entry, in the form `// R5.1: f64 initial guess requires...`. Future readers (including future-aristotle and future-pathmaker) need this context to extend or audit.

---

## Status

Phase 1-3 walk on the BZ algorithm preconditions, framed under F13's antibody pattern. Map of each algorithm's preconditions + the antibody shape for each. Recommendations to pathmaker prioritized by silent-failure cost.

This is preventive substrate. If pathmaker's implementation already handles all of these (which is plausible — math-researcher's DESIGN.md §3 dispatch table is detailed), this doc is a redundant safety net. If any are missed, the doc surfaces them at PR review time.

Output landed at this campsite. Ready for pathmaker's diff. Will pressure-test post-diff per Task #10.
