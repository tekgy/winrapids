# Internal tameness contracts — the shape behind the BZ bug class

**Status**: Research note, drafted 2026-05-09. Anchored on naturalist's open question from `2026-05-08-the-fourth-instance.md` ("are these four separate adversarial-input categories, or four faces of one boundary-condition class?") and audit of the ~12 BZ-arithmetic bugs that surfaced and were fixed during the same session.

**Companion to**:
- `holonomic-architecture.md` — names the general class (non-holonomic local defenses)
- F13.C graduation condition (aristotle, 2026-05-09) — antibody graduates from local-pattern to structural-invariant only when required at every signature

This doc names the **specific** structural shape underneath the general class.

---

## The naturalist's question

From `~/.claude/garden/2026-05/2026-05-08-the-fourth-instance.md` closing:

> *"Cancellation/borrow, sign of exp_shift, NaN payload, seed overflow at subnormal boundary — are these four separate adversarial-input categories, or are they four faces of one boundary-condition class?"*

The naturalist's 2026-05-09 follow-up surfaced the general answer: non-holonomic local defenses on a tameness invariant. Fixing the bugs requires making the antibody *structural at every signature*, not just at the public API entry.

This doc adds the **specific structural pattern** behind the bug class.

---

## The audit

Twelve+ bugs surfaced and fixed in 2026-05-08/09 across BigFloat arithmetic:

| Site | Original framing | Boundary | Fix |
|---|---|---|---|
| `normal_add_multilimb` cancellation/borrow (#8) | wrong arithmetic on near-equal values | "normal-with-zero-limbs" never happens in practice | flip kind to Zero on all-zero limbs |
| `newton_reciprocal` exp_shift sign (#9) | unscaling sign error | seed-as-scaled-f64 boundary | corrected sign in the unscale step |
| `div` NaN payload drop (#10) | inconsistent special-value handling | NaN-payload preservation across ops | match add/mul/sqrt's preserve-payload pattern |
| `newton_reciprocal` seed overflow (#11) | Newton diverges on subnormal `b` | f64 subnormal where `1/b = ±Inf` | `is_finite()` guard on the reciprocal seed |
| `canonicalize_and_round` sticky-bit loss on left-shift n=1 (#13) | sticky bit dropped during renormalization | exactly-one-bit left-shift case | preserve sticky into the shifted position |
| `canonicalize_and_round` mag=0 with nonzero round/sticky (attack22) | total cancellation left sub-ULP info unhandled | mag=0 + nonzero (round, sticky) | apply IEEE rounding to sub-ULP — directed modes return ±1 ULP at sub-ULP exponent |
| `add` exp_diff i64 overflow (attack18) | `large.exp - small.exp` cast to u64 overflows i64 | `i64::MAX/MIN` exponent boundary | `saturating_sub` instead of `-` |
| `mul` exp_at_lsb chain overflow (attack23) | `a.exp + b.exp - p_a - p_b + 2` chain overflows i64 | `i64::MAX/MIN` exponent boundary | `saturating_add` / `saturating_sub` chain |
| `to_f64` mantissa-rounding carry-out (attack25) | `unbiased + 1` overflows when `unbiased = i64::MAX` | `i64::MAX` exponent boundary | `saturating_add(1)` |
| `to_f64` biased_exp computation (attack24) | `unbiased + F64_EXP_BIAS` overflows | `i64` exponent boundary | `saturating_add` |
| `div` NaN preservation (attack15/17) | inputs were already-fixed cases that proved the original NaN-payload fix held | NaN-payload preservation | tests confirm the b9dfeb9 fix |
| (others surfaced + fixed in same session) | various | various edges of i64 / f64 / canonical-form | various |

All twelve plus follow-ons share **one structural shape**: each is at a boundary where the algorithm's intermediate state could exit a "safe subspace" of its representation type, and the original implementation *implicitly* assumed it wouldn't.

---

## The shape

**Every operation in BigFloat arithmetic carries an implicit tameness contract on its intermediate state.**

The contract is *narrower* than the type allows. The type `BigFloat` admits:
- Exponents anywhere in `i64::MIN..=i64::MAX`
- Limbs of arbitrary bit patterns including all-zero
- NaN payloads of arbitrary content
- Subnormal-class values whose `to_f64` reciprocal produces `±Inf`

But each algorithm step operates safely only on a *subset* of those values. The subset is what I'm calling the **internal tameness predicate** for that step:

- `normal_add_multilimb`'s subspace excludes "Normal-with-all-zero-limbs" (canonical-form precondition)
- `newton_reciprocal`'s subspace excludes f64 inputs where `1.0/b` overflows to `±Inf` (Newton-seed precondition)
- `to_f64`'s carry-out path's subspace excludes `unbiased = i64::MAX` (i64-arithmetic precondition)
- `canonicalize_and_round`'s subspace assumed `mag != 0 || (round, sticky) == (0, 0)` (sub-ULP-implies-mag precondition)
- `div`'s special-value dispatch's subspace assumed payloads weren't carried (consistency-with-add-mul-sqrt precondition — but its precondition was *wrong*, not violated)

The bugs aren't separate categories. They're **all** instances of one shape:

> An operation has an implicit tameness predicate on its intermediate state. The predicate is narrower than the type. For "normal" inputs, the algorithm stays inside the predicate's subspace. Adversarial inputs push intermediate state to the predicate's *boundary* — where the implicit contract is violated and the implementation has no defined behavior.

When the boundary is crossed, the failure mode is:
- Silent wrong answer (sticky-bit lost, NaN payload dropped, sign error)
- Debug panic (i64 overflow)
- Numerical divergence (Newton iteration with `±Inf` seed)
- Loss of canonical form (Normal kind with zero limbs)

The fix in each case follows the same pattern:

1. **Make the predicate explicit** — name what state the algorithm requires
2. **Detect violations at the boundary** — saturating arithmetic; explicit checks for edge cases; payload preservation in dispatch
3. **Define the behavior** — saturate to ±Inf; flip kind to Zero; preserve payload; fall through to safe path

---

## Connection to the holonomic lens

Naturalist's 2026-05-09 essay generalized this class as "non-holonomic local defenses on a tameness invariant." Concretely:

- Each bug's *original* defense was at the public API entry — "we accepted tame inputs, so internal computation is safe."
- The defense was **non-holonomic** because the path from "tame input" to "internal state at boundary" depended on the algorithm's internal control flow. Different paths (different branches, different iteration counts, different operand sizes) led to different intermediate states. The "tame input" predicate didn't propagate through the path.
- The fix in each case made the antibody **signature-level** — saturating arithmetic at every i64-arithmetic site, every check at every special-value dispatch, every iteration's seed verified before use. Per F13.C: graduation from local pattern to structural invariant requires the predicate at every call site, not just public-API.

The specific shape adds: **the predicate isn't on the input type. It's on the intermediate representation's safe subspace.** The signature-level antibody isn't "is the input tame?" — it's "did this step's intermediate state stay inside the operation's tameness predicate?"

---

## Methodology consequence

For every BigFloat operation (and by extension, every operation on a precision-rich type):

**Audit pass — "implicit tameness contracts":**

1. List every intermediate state the algorithm produces (exponent computations, limb subtractions, special-value dispatches, type conversions).
2. For each intermediate, name the **implicit tameness predicate** — what subspace of the type does the algorithm assume the value lives in?
3. For each predicate, identify the **boundary** — the inputs that push the intermediate to the predicate's edge.
4. For each boundary, decide:
   - **Saturate** (saturating arithmetic returns the limit value, which is a defined output for the operation)
   - **Detect and branch** (explicit check; fall through to alternate code path)
   - **Reject** (the predicate is part of the operation's contract; violations are user error and panic with a clear message)
5. Make the chosen behavior **structural** — at every signature, not just the public API.

This is the audit the team did organically over 2026-05-08/09 by responding to adversarial-generator failures. The doc names it as a *deliberate audit pattern* so future operations get the audit *before* shipping rather than *after* adversarial generators surface failures.

---

## Tooling opportunity

The audit pattern is mechanical enough that it could be partially automated. Candidate lints:

- **i64-arithmetic-without-saturation**: any `+`, `-`, `*` on `i64` values inside a numerical operation. Suggest `saturating_*` or explicit overflow handling. (Most of attack18/23/24/25 would have been caught at lint time.)
- **Mantissa-rounding-without-carry-bump-check**: any `unbiased + 1` pattern in a rounding-carry context. Same fix pattern.
- **Limb-zero-without-kind-flip**: any path that subtracts limbs without checking the all-zero case. Cancellation-to-Zero antibody.
- **Special-value dispatch consistency**: NaN/Inf/Zero handling across operations. If `add`/`mul`/`sqrt` preserve payload but `div` doesn't, the inconsistency is a lint.
- **f64-fast-path-without-result-finiteness**: any f64 operation used as a seed for higher-precision computation. Verify the result is finite before relying on it.

These aren't requirements; they're opportunities. The lint pattern would catch the *known shape* of this bug class before adversarial generators do.

---

## Connection to the broader pattern

This doc is the fourth *lens application* artifact from 2026-05-08/09:

1. `holonomic-architecture.md` — the foundational lens (recipes content-addressed; IR provenance-addressed; F13 antibodies graduate to structural at signature level)
2. `confident-wrong-narratives.md` — apparatus-first investigation discipline (math-researcher's sin-2^1000 false narrative)
3. `tambear-libm-factoring.md` — synthesis of past-Claude April 13 design + oracle empirics (the periodic table; the complementary-argument transform; tambear factoring libm)
4. This doc — internal tameness contracts as the specific shape behind the BZ bug class

Each is the holonomic lens applied to a different question. The lens is the *connector*; the substrate is what the lens reveals. Per the discipline added to global CLAUDE.md today: read past-me before writing. The BZ bugs were the substrate; this doc is the connection.

---

## Open questions

1. **Beyond BigFloat?** The same shape may apply to other precision-rich types — DoubleDouble, the complex-number stack when it lands, eventually the symbolic stack. Worth running the audit pass on existing operations as they're written, not just retroactively when adversarial generators fire.
2. **Does the lint pattern want to be a `cargo-tambear-audit` tool?** Or just a documented checklist applied during code review? The cost-benefit depends on how often the bug class recurs.
3. **Connection to F13.C's graduation condition.** F13.C says "antibody graduates to structural when required at every signature." The audit pass produces the list of every-signature requirements. Are they the same list, or does F13.C cover something the audit misses?
4. **The "implicit tameness predicate" is a *type refinement*.** In refinement-types languages (Liquid Haskell, Refinement Types in Rust via creusot etc.), this would be expressible as a type-system constraint. Tambear doesn't use refinement types currently. Is there a lightweight version (e.g., debug assertions on intermediate state, or runtime-checked newtype wrappers) that's worth introducing for the operations where this pattern is densest?
5. **Intermediate vs final state.** The audit framing focuses on *intermediate* state. Some bugs (NaN payload drop in #10) were about the *output* state — a final-state inconsistency across operations. Is that a separate shape, or a special case of the same shape (where the "intermediate" is the dispatch decision)?

---

## What this doc is and isn't

- **Is**: a research note naming the specific structural shape behind the BZ bug class. Anchored on twelve+ fixes from 2026-05-08/09. Companion to the holonomic-architecture doc; not a replacement.
- **Isn't**: a complete audit. The audit pass for existing BigFloat operations hasn't been done; this doc names the pattern, doesn't enumerate every site.
- **Isn't**: a tooling commitment. The lint candidates are opportunities, not commitments.
- **Isn't**: a refactor mandate. The bugs are fixed; the shape is for future operations and methodology.

The naturalist asked the question. The audit-by-doing happened over 2026-05-08/09. This doc names what was found.
