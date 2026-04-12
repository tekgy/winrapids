# Logbook — Campsite 1.1: `.tam` IR spec

**Date:** 2026-04-11  
**Author:** pathmaker  
**Status:** complete

---

## What was done

Wrote `peak1-tam-ir/spec.md`: a 12-section, ~2-page specification covering
types (i32, i64, f64, pred, buf<f64>), the complete Phase 1 op set (31 ops),
SSA register conventions, the phi-suffix convention for loop-carried values,
text encoding grammar, a full `variance_pass` example, and explicit out-of-scope
list. Also wrote `spec-review.md` with four open questions for reviewers.

---

## What almost went wrong

**The scope creep reflex.** Writing a spec with full creative freedom, the
pull toward completeness is strong. Three times I caught myself starting to
add things: (1) integer casts between i32 and f64 — useful, but no recipe
needs them; (2) multiple-exit functions with early `ret` — natural, but the
verifier is simpler without it; (3) a binary encoding alongside the text
encoding — "might as well specify both." Stopped each time by asking the
single question: "does a recipe we actually have require this?" The answer was
no every time.

The spec deliberately contains an explicit "does NOT exist in Phase 1" section
to document these refusals. Future implementers should not have to reverse-
engineer which omissions were deliberate.

---

## What tempted me off-path

**The `%acc'` phi convention.** The cleanest alternative would be explicit
`phi` instructions at loop entry, LLVM-style:

```
loop_grid_stride %i in [0, %n) {
  phi %acc = [%acc_init, entry], [%acc_next, loop_body]
  ...
  %acc_next = fadd.f64 %acc, %v
}
```

This is more principled — it makes the phi node visible as a first-class
instruction. But it requires two names per loop-carried value, complicates
hand-writing the reference programs, and makes the variance example harder
to read. The prime-suffix convention lets you write `%acc` and `%acc'`
as a natural "old value → new value" pair. I chose readability over formalism
here, filed it as Q1 in the review doc, and noted that if reviewers prefer
the explicit phi style it can be changed before the parser is written — that's
the right time to decide.

---

## What the next traveler should know

**The reduce_block_add asymmetry.** The CPU interpreter's "one block = all
elements" shortcut means the interpreter's output is already final, while the
PTX path writes partial sums that need host folding. This is correct and
intentional, but it's the kind of thing that causes "CPU interpreter and PTX
disagree on variance" bugs if the person writing the PTX translator forgets
the host fold. Campsite 3.11 will hit this. Make sure the PTX translator's
test explicitly checks the folded value, not just `out[0]` of the raw output.

**The invariant table in §12.** Every invariant (I3, I4, I5, I7, I8) is
cross-referenced to the spec section that upholds it. If you add an op, check
which invariants it might touch and add a row. This table is a maintenance
contract, not a one-time check.
