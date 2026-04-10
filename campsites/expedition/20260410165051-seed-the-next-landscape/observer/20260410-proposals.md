# Observer's Next-Landscape Proposals

Written: 2026-04-10

## oracle-coverage-map (verification family)

A systematic map of which primitives have been verified against high-precision
oracles vs. which haven't. Not "how many tests pass" but "what territory is
still unclaimed."

**Why this matters:** The Padé [6/6] failure wasn't found by reading the
implementation. It was found by stating a mathematical truth and seeing the
code violate it. `exp(t·I) = e^t·I` is a theorem — the test asserted the
theorem, the code falsified it. You can't find that class of bug by inspection.
You can only find it by systematic oracle coverage.

**Current state:** Two workup files exist (erfc.md, pearson_r.md) doing this
for two primitives. The coverage map extends that to all 120+ primitives.

**Format (navigator's suggestion):**
Three-tier status per primitive:
- Not verified (no oracle comparison exists)
- Spot-checked (verified on a handful of known values)
- Fully oracled (verified against mpmath at 50+ digits, adversarial edge cases,
  multiple scales)

**Open design question from navigator:** Track status only, or also track
which tests would be needed to reach full coverage? Status-only is auditable
right now. Test-specification is more useful but harder to maintain.

**Suggested owner:** Observer + scientist jointly — observer tracks coverage
state, scientist executes workups for uncovered primitives.

**Connection to KingdomProof idea (navigator garden):** The coverage map is
the same concept applied to mathematical correctness that KingdomProof applies
to kingdom classification. Both are systematic verification layers that
currently exist as informal convention.
