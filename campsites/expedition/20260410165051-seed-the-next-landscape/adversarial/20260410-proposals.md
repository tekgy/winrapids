# Adversarial's Next-Landscape Proposals

Written: 2026-04-10
Context: Wave 18 complete. 144 tests green. 39 bugs documented across 18 waves.
NaN-eating fold pattern eliminated from all confirmed production paths.

---

**1. `adversarial-validity-semantics`**
Make the three validity policies (Propagate/Ignore/Error) explicit as a design axis.
Every function should declare which policy it uses. Currently implicit and inconsistent.
This is the structural fix that makes the NEXT class of NaN bugs impossible —
instead of finding them wave by wave, the policy declaration forces the question
at implementation time.
Role: adversarial + aristotle (policy design) + pathmaker (trait/attribute implementation)

**2. `adversarial-singularity-class`**
The OTHER bug class: tridiagonal zero-pivot returns identity matrix,
matrix_exp singular-Q returns eye. These are singularity-as-identity violations —
same pattern as wrong-identity in NaN-eating, but for degenerate matrix inputs.
Needs adversarial workups like the NaN-eating waves.
Connection: op-identity-method campsite (Aristotle) — `degenerate()` is the
right return for singular inputs, not `identity()`.
Role: adversarial

**3. `adversarial-catastrophic-cancellation`**
Sum-based methods at scale 1e15+. Welford merge is covered; raw sum/mean
paths in non-Welford code are not. This is a different bug class from NaN-eating:
the answer is finite but WRONG, not NaN. Harder to detect because tests pass.
Gold standard: Kahan summation or pairwise summation as oracle.
Role: adversarial + scientist (oracle workup)

**4. `adversarial-special-function-poles`**
gamma at negative integers, digamma at 0, beta at 0/1.
Classical dangerous boundaries, none tested adversarially yet.
These are published in the mathematical literature — exact behavior is known
(poles, branch cuts, asymptotic expansions). Oracle exists.
Role: adversarial + math-researcher (pole behavior verification)

---

## Navigator's note on sequencing

adversarial-validity-semantics (#1) is the highest-leverage structural fix —
it prevents future bug classes rather than fixing one at a time.
But it requires a design decision (policy declaration mechanism) before
implementation. Route #1 to aristotle first for the design question.

adversarial-singularity-class (#2) can start immediately — same wave structure
as NaN-eating, different input class.

adversarial-catastrophic-cancellation (#3) and adversarial-special-function-poles (#4)
are independent and can run in parallel with #2.
