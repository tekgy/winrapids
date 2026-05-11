# Formal structure of `is_kernel_share_compatible_with` — math-researcher memo

Date: 2026-04-23
From: navigator (task #19)
For: aristotle, pathmaker

---

## Conclusion (one sentence)

`is_kernel_share_compatible_with` is a **preorder** (reflexive + transitive,
NOT symmetric) — renaming it to `is_kernel_tolerance_with` would introduce
a mathematical misnomer, because tolerance relations require symmetry.

**Recommendation: do NOT rename.**

---

## The relation, formally

Define `P(a, b)` = `a.is_kernel_share_compatible_with(b)` ("producer `a`
can serve consumer `b`").

Reading shape.rs:417–458 directly:

| Producer → Consumer | Result | Reason |
|---------------------|--------|--------|
| Static(n) → Static(m) | n == m | exact match required |
| UpTo(a) → UpTo(b) | a >= b | larger bound serves smaller |
| Dynamic → Dynamic | true | trivially compatible |
| Static(_) → Dynamic | false | GAP-SHARE-1 constant-folding risk |
| Dynamic → Static(_) | false | no size guarantee |
| UpTo(_) → Dynamic | false | consumer might exceed bound |
| Dynamic → UpTo(_) | false | producer has no cap |
| Adaptive → _ | false | unresolved, blocked |
| _ → Adaptive | true | runtime guard on dispatcher path |

---

## Is it reflexive?

`P(a, a)` for each variant:

- `Static(n) → Static(n)`: `n == n` = **true**
- `UpTo(a) → UpTo(a)`: `a >= a` = **true**
- `Dynamic → Dynamic`: **true**
- `Adaptive → Adaptive`: falls on `(Adaptive, _)` arm = **false**

**Adaptive breaks reflexivity.** The relation is reflexive everywhere except
`Adaptive`. This is intentional — Adaptive represents an unresolved shape,
and an unresolved producer cannot safely serve even itself. The comment
at line 448-452 ("two unresolved shapes may resolve to incompatible sizes")
makes this explicit.

For the purposes of the resolved-shape subset (Static, Dynamic, UpTo),
the relation IS reflexive.

---

## Is it symmetric?

**No.** Concrete counterexample:

```
P(UpTo(100), UpTo(50)):  a=100, b=50,  a >= b  = TRUE
P(UpTo(50),  UpTo(100)): a=50,  b=100, a >= b  = FALSE
```

A 100-element kernel can serve a 50-element consumer (it over-covers).
A 50-element kernel cannot serve a 100-element consumer (it under-covers).

This asymmetry is the DEFINING FEATURE of the relation — it encodes
"does producer cover consumer?" which is inherently directional.

**A tolerance relation requires symmetry. This relation lacks it.
`is_kernel_tolerance_with` would be mathematically wrong.**

---

## Is it transitive?

**Yes** (on the resolved-shape subset). Proof sketch:

- `UpTo(a) → UpTo(b) → UpTo(c)`: `a >= b` and `b >= c` implies `a >= c`. ✓
- `Static(n) → Static(n) → Static(n)`: trivially. ✓
- Any chain involving `false` collapses immediately. ✓
- The cross-variant arms (Static→UpTo, UpTo→Static) are consistent
  with transitivity: `UpTo(100) → Static(50)` (50 <= 100 = true),
  `Static(50) → UpTo(100)` (50 <= 100 = true), and
  `UpTo(100) → UpTo(100)` (100 >= 100 = true). ✓

The relation is transitive on the resolved-shape subset.

---

## What kind of relation is it?

On the resolved-shape subset {Static, Dynamic, UpTo}:

- **Reflexive**: yes
- **Transitive**: yes
- **Symmetric**: NO

This is a **preorder** (reflexive + transitive), specifically a partial
preorder since some pairs are incomparable (e.g., Static(5) and Static(6)
are both `false` in both directions — neither serves the other).

It is NOT:
- A tolerance relation (requires symmetry)
- A partial order in the strict sense (would require antisymmetry, but
  the relation is not antisymmetric either — Static(n)→Static(n) is
  symmetric, but that's the only case)

The best name for the concept is already the name it has:
**"kernel share compatible with"** — `a` is compatible with `b` as a
kernel reuse source. Compatibility is directional. The name is accurate.

---

## What `is_kernel_tolerance_with` would imply

A "tolerance relation" in mathematics is:
1. Reflexive
2. Symmetric (if a tolerates b, then b tolerates a)
3. NOT required to be transitive

Using that name would imply that if producer A can serve consumer B,
then consumer B can serve producer A. That is false by the counterexample
above and would confuse any reader familiar with the terminology.

---

## Recommendation

**Do not rename.** The current name `is_kernel_share_compatible_with` is
accurate and descriptive. The rename to `is_kernel_tolerance_with` would:

1. Introduce a mathematical misnomer (tolerance relations are symmetric;
   this relation is not)
2. Confuse readers who know the formal definition of tolerance relations
3. Provide no benefit — "compatible with" already conveys directionality

If the goal was to name the relation's formal class, the correct term
would be "preorder" or "compatibility preorder" — but the current
English name already communicates the semantics better than a formal
term would.

The alias `is_share_compatible_with` (back-compat) should remain
pointing to `is_kernel_share_compatible_with` as documented in the
existing alias docstring.
