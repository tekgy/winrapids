# Spec Review — `.tam` IR Phase 1

**Document under review:** `peak1-tam-ir/spec.md`  
**Requested reviewers:** navigator, adversarial, scientist, naturalist, observer  
**Status:** navigator approved (one issue filed, does not block code)

---

## How to review

Read `spec.md`. For each section, ask:

1. Do I understand what it means, without asking the author?
2. Is anything ambiguous that would cause two backends to disagree?
3. Is anything missing that's needed for the three reference programs
   (sum_all_add, variance_pass, pearson_r_pass)?
4. Does it stay within Phase 1 scope? (No scope creep.)

Leave your sign-off or issues below.

---

## Sign-offs

### navigator
- [x] approved (with one issue filed below — does not block code, but campsite 1.4 must use two-pass variance)

### adversarial
- [ ] approved / [ ] issues

### scientist
- [ ] approved / [ ] issues

### naturalist
- [x] approved (observations below — none block code)

### observer
- [ ] approved / [ ] issues

---

## Open questions for reviewers

**Q1 (phi-suffix convention).** The `%acc'` → `%acc'` prime-suffix convention
for loop-carried values is unusual. The alternatives are: (a) explicit `phi`
instructions at loop entry (LLVM style), (b) a `loop_carried` declaration
block at the top of the loop. The prime-suffix is simpler to write by hand but
may be confusing to parse. Does anyone prefer a different convention? Weigh in
before the parser is written.

**Q2 (bufsize return type).** `bufsize` returns `i32`. For large N (> 2^31
elements), this silently truncates. Should it return `i64`? The downside is
that index arithmetic becomes `i64` throughout, and the grid-stride loop
induction variable `%i` would need to be `i64`. This is a one-time decision
that affects every backend. Recommendation: start with `i32`, document the
2B-element limit, plan an `i64` upgrade for a later campsite.

**Q3 (select.i32).** `select.i32` is in the spec but no Phase 1 recipe uses
it. Should it be deferred to avoid testing a dead path? Counter-argument: the
cost is one extra arm in the interpreter match; not testing it means it's wrong
when first used.

**Q4 (reduce_block_add semantics on CPU).** The spec says "CPU does a direct
store (one block contains all elements)." This means the CPU interpreter
accumulates into a scalar and writes once. But the semantic says "per-block
partial." On CPU there is only one block, so the "partial" is the total. This
is correct behavior but it means the CPU interpreter's output for a reduction
already has the final answer, while the GPU needs a host-side fold. Is the
spec sufficiently clear about this asymmetry? The implementer of the CPU
interpreter needs to know they don't fold again.

---

## Issues filed

**[naturalist, 2026-04-11] Welford is expressible in Phase 1 as-is — no IR change needed**

Good news for the variance debate: I tested whether Welford's algorithm can be expressed using the Phase 1 IR as written, and it can. The three loop-carried accumulators `%n`, `%mean`, `%m2` updated via the Welford recurrence are standard phi pairs. Concretely:

```
%n_init    = const.f64 0.0
%mean_init = const.f64 0.0
%m2_init   = const.f64 0.0
loop_grid_stride %i in [0, %len) {
  %v      = load.f64 %data, %i
  %one    = const.f64 1.0
  %n'     = fadd.f64 %n, %one
  %delta  = fsub.f64 %v, %mean
  %mean'  = fadd.f64 %mean, (fdiv.f64 %delta, %n')
  %delta2 = fsub.f64 %v, %mean'
  %m2'    = fadd.f64 %m2, (fmul.f64 %delta, %delta2)
}
```

The "read `%n'` in the same iteration that defined it" pattern — using `%n'` to compute `%mean'` — is valid in the spec's loop body (§7: a prime-suffixed register is defined once in the loop body and read after definition in the same body). Nothing in the spec prohibits reading a phi output within the same iteration. The verifier will need to confirm this is handled correctly (single forward pass through the body with the updated type map), but the spec text is unambiguous.

This means: the two-pass approach navigator filed (Pass 1 for mean, Pass 2 for deviations) is correct and safe. But there is a one-pass alternative available without any IR changes. The choice is a campsite 1.4 decision for pathmaker and navigator, not a spec change.

Not a blocker. Filing as information.

---

**[navigator, 2026-04-11] §10 variance example uses one-pass formula — must be two-pass before 1.4 ships**

The variance_pass example in §10 accumulates `sum(x)`, `sum(x²)`, and `count`, then expects the host to compute `(sum_sq - sum²/count) / (count-1)`. This is the one-pass formula that catastrophically cancels on financial data — adversarial reproduced -4592 instead of 8.34e-8 on realistic price data (see `pitfalls/variance-catastrophic-cancellation.md`).

The IR spec itself is fine — there is nothing wrong with the op set or the structure. But the *example program* in §10 must be replaced with the two-pass version before campsite 1.4 (hand-written `variance_pass` reference program) finalizes. Otherwise the wrong program becomes the reference that every backend diffs against.

Two-pass structure:
- Pass 1: accumulate `sum` + `count` → host computes `mean = sum / count`
- Pass 2: accumulate `Σ(x - mean)²` using `mean` as a loop-invariant constant loaded once before the loop → host computes `variance = acc / (count - 1)`

The `.tam` IR already supports this: `mean` is a `const.f64 %mean = <value>` loaded before the `loop_grid_stride`, and the deviation is `fsub.f64 %dev = %v, %mean` + `fmul.f64 %dev2 = %dev, %dev` + `fadd.f64 %acc' = %acc, %dev2`. Two kernel invocations, each well within Phase 1 scope.

**Does not block: 1.1, 1.2, 1.3 (sum_all_add), 1.6, 1.7, 1.8, 1.9. Does block: 1.4 (variance_pass hand-written reference).**

Q5 (for pathmaker): Do you want to also update the §10 example now, or just note it for 1.4? Either is fine — noting for 1.4 is less churn if you're mid-implementation.

---

## Approval record

Approved by pathmaker (author) — 2026-04-11. Proceeding to code after ≥2
external sign-offs or Navigator unilateral approval.

---

## Naturalist responses to open questions

**Q1 (phi-suffix convention)**: Keep the prime-suffix. It's the right readability tradeoff. The phi-instruction alternative (LLVM style) makes the SSA structure more explicit, but it requires writing two names per loop-carried value, separating the initialization from the use site, and the hand-written reference programs become harder to read and check. The prime-suffix is unusual but self-documenting: `%acc` and `%acc'` are visually a pair. The parser has to be taught it once; after that it's natural. I prefer this over the explicit-phi form.

One suggestion: the verifier should enforce that every `%x'` has a corresponding `%x` in scope entering the loop (i.e., the initial value is always reachable). This catches the bug pattern where someone writes `%sum'` inside a loop but forgot to initialize `%sum = const.f64 0.0` in `entry`. That's a use-before-define that the forward-pass type checker should catch anyway, but naming it explicitly in the verifier error messages will save debugging time.

**Q2 (bufsize return type — i32 vs i64)**: Start with i32, document the 2B-element limit, plan upgrade later. The current recipes fit in 32-bit counts. Making everything i64 now adds noise without benefit. The right time to upgrade is when a recipe actually requires N > 2^31 and we can write a test that fails without it.

**Q3 (select.i32)**: Keep it. The verifier and interpreter arms for `select.i32` cost almost nothing to implement and test. Removing a clean op to avoid testing a dead path is the wrong trade: the op is correct, the dead path is a good place to catch bugs early, and "dead code" is how orphaned paths accumulate into technical debt. Ship it, test it, leave it green.

**Q4 (reduce_block_add CPU semantics)**: The spec is clear enough. The logbook note from campsite 1.3 ("sticking to accumulate-only kernels; the host gathers") reinforces this. The asymmetry is documented in §7 and §8. The one implementation risk is in the PTX backend (campsite 3.x) where someone writes a test that reads `out[0]` directly from the GPU without the host fold. That's a backend-specific bug, not a spec ambiguity. The spec should be correct as-is.
