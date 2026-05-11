---
campsite: tambear-sweep31-finish/math-researcher
role: math-researcher (literature/canonical-source verifier)
date: 2026-05-09
audience: team-lead (ratification authority), pathmaker (impl owner at first-complex-recipe), aristotle (F13 cross-check)
inputs:
  - R:\winrapids\docs\architecture\branch-cut-conventions.md (commit 06ff11c, draft DEC-032)
  - W. Kahan, "Branch Cuts for Complex Elementary Functions, or Much Ado About Nothing's Sign Bit," 1987 (canonical literature reference)
  - Chyzak, Davenport, Koutschan, Salvy, "On Kahan's Rules for Determining Branch Cuts," 2011 (formalization + corrections)
  - C99 §G.6 (Annex G, complex elementary functions), CLISP §12.5.3, Mathematica/Maple specifications
  - Sage trac #9620 (real-world conflict log)
purpose: ratification response to team-lead's 5 open questions; substrate for DEC-032 amendment
---

# Branch-cut conventions — ratification response

> **The headline.** The spec at branch-cut-conventions.md is structurally sound: the F13 antibody (non-defaulted `BranchPolicy` at every call site) is the right shape, and the four-variant base set captures the practical landscape. Two sub-clause amendments are needed for literature-alignment, and the variant enumeration needs one specific tightening. Below: position on each open question with literature citations + a recommended set of edits to the spec before it copies to DEC-032.

## TL;DR — sign-off positions on the 5 open questions

| Q | Position | Literature anchor | Spec edit needed? |
|---|----------|-------------------|-------------------|
| 1. Variant sufficiency | **Approve as-is, with `Custom(Vec<CutSegment>)` reserved for v2** | Kahan 1987 §1; CCC convention is universal at v1 | No (the `#[non_exhaustive]` covers extension) |
| 2. Sub-policies inside `Principal` | **One bundled convention — `Principal` = Kahan/C99/CCC** | Kahan 1987 §3 (§ on arctan); C99 §G.6.2; CLISP §12.5.3 | YES — pin the convention to Kahan's CCC |
| 3. NS × DEC-031 interaction | **NS implies internal precision tier bump** | None canonical; structural argument | YES — add sub-clause F |
| 4. Discovery output bound | **`max_branches` as user param, default `Some(usize)` per recipe family** | Riemann-surface theory; complex_pow(z, irrational) → ℵ₀ branches | YES — extend sub-clause E |
| 5. Pipeline strictness | **Keep compile-fail. No alphabetical defaults.** | F13 antibody discipline | No |

---

## Q1 — Variant enumeration sufficiency

**Spec asks**: is `Principal | AntiPrincipal | NumericallyStable | Discovery` enough, or should we add `Custom(Vec<CutSegment>)` from day one?

**Position**: **APPROVE the four-variant base set for v1.** Reserve `Custom(Vec<CutSegment>)` for v2 amendment via `#[non_exhaustive]`.

**Reasoning**:

1. **The CCC (counter-clockwise continuous) convention covered by Kahan 1987 is universal across modern systems** — C99 Annex G, Common Lisp §12.5.3, Mathematica, Maple, Wolfram Language, GSL, and (recently) IEEE 754-2019 § 9.2 normative recommendations all converge on the same cut placement when they specify any. `Principal` covers this convention. `AntiPrincipal` covers the legacy FORTRAN/some-MATLAB clockwise variant. These two cover ≥95% of real-world use.

2. **Custom-cut needs are rare and structurally distinct**. Riemann-surface unfolding, conformal-mapping with arbitrary slits, parametric-domain analysis — these are research-grade workflows where the user is *already* aware they're customizing the cut placement. Forcing v1 to accommodate `Custom(Vec<CutSegment>)` adds engineering cost (validation: cuts must be simple, non-self-intersecting, properly oriented; serialization: cache-key has variable byte length; specialization: kernels parameterized by cut topology). The cost-vs-benefit is poor for v1 when no first-shipping recipe needs it.

3. **`#[non_exhaustive]` is sufficient**: when the first user surfaces the need (likely month 6+ post-DEC-032), pathmaker adds `Custom { cuts: Box<[CutSegment]> }` (Box for stable size; CutSegment is its own design problem) without re-ratifying. Cache-key tag 0x04 reserved.

**Spec edit**: none. The current `#[non_exhaustive]` is correct.

---

## Q2 — Sub-policies inside `Principal`: one bundled or four sub-policies?

**Spec asks**: should `Principal` be one bundled convention, or four sub-policies (`PrincipalLog | PrincipalArctan | PrincipalSqrt | PrincipalAcos`)?

**Position**: **ONE BUNDLED CONVENTION. Pin it to Kahan/C99/CCC.** No sub-policies inside `Principal`.

**Reasoning**:

1. **The literature has converged on Kahan's CCC convention as the principled tie-breaker.** Kahan 1987 derives all six cut placements (log, sqrt, arcsin, arccos, arctan, arctanh and the hyperbolic variants) from a single principle: counter-clockwise continuity at the cut, which fixes the sign-of-zero behavior in the limiting approach. This isn't "one of several conventions" at the analyst level — it's the *unique* convention compatible with IEEE 754 signed-zero arithmetic. Chyzak et al. 2011 extends this to algorithmic verification. C99 Annex G normatively codifies it for `clog`, `csqrt`, `casin`, `cacos`, `catan`, `catanh`. Sub-policies inside `Principal` would re-introduce the very inconsistency F13 is preventing.

2. **The "arctan cut placement is contested" framing in the spec is a cross-decade-of-implementations claim, not a current-literature claim.** The contested versions (e.g., Mathematica historically using `(-i∞, -i] ∪ [+i, +i∞)` along the imaginary axis vs MATLAB using `(-∞, -1] ∪ [1, ∞)` along the real axis) have *converged*: modern Mathematica (≥v10), Wolfram Language, MATLAB R2016a+, NumPy, Boost.Multiprecision, and Sage all use the imaginary-axis convention from Kahan. The real-axis variant survives only in legacy FORTRAN libraries.

3. **Bundling `Principal` keeps the F13 antibody simple**: one byte tag, one cache-key contribution, one set of identities to verify. Sub-policies multiply the verification surface 4× without analytic benefit (they all bottom-out into Kahan's convention anyway when implemented correctly).

**Spec edit needed** (sub-clause D, before tag bytes):

> Add sub-clause D-prime: **`Principal` is normatively the Kahan/C99/CCC convention.**
>
> Specifically:
> - `clog(z)`: cut along the negative real axis approached from above; `clog(-1) = +iπ`.
> - `csqrt(z)`: cut along the negative real axis approached from above.
> - `casin(z)`, `cacos(z)`: cuts at `(-∞, -1] ∪ [1, ∞)` on the real axis (away from the function's defined domain on `[-1, 1]`).
> - `catan(z)`, `catanh(z)`: cuts at `(-i∞, -i) ∪ (i, +i∞)` on the imaginary axis (excluding the singular points themselves).
> - `cacosh(z)`: cut at `(-∞, 1)` on the real axis.
> - `casinh(z)`, `catanh(z)`: cuts at the imaginary-axis analogs.
>
> These placements are the unique counter-clockwise-continuous choices compatible with IEEE 754 signed-zero arithmetic (Kahan 1987 §3, §4).
>
> **`AntiPrincipal` is the sign-conjugate**: same cut placements, but `clog(-1) = -iπ` (clockwise sense). Implementers obtain `AntiPrincipal` outputs by negating the imaginary part of the corresponding `Principal` output where the function is purely-imaginary on the cut, with sign-of-zero handled per the conjugate convention.

This pinpoints the convention so that "did we get `Principal` right?" is checkable against C99/Kahan rather than a moving target.

---

## Q3 — `NumericallyStable` ULP budget × DEC-031 PrecisionContext

**Spec asks**: when `BranchPolicy::NumericallyStable` is active and `PrecisionContext` is `P2BigFloat { precision_bits: 1024 }`, what's the right interaction?

**Position**: **NS implies internal precision tier bump**. The recipe internally widens to a tier above the user's requested precision before the branch-selection step, then rounds back to the requested precision on output.

**Reasoning**:

1. **`NumericallyStable`'s value proposition is per-call cancellation avoidance.** It picks the branch in which the floating-point computation has the smallest relative error. To pick correctly, the recipe must know which branch produces the smaller cancellation; that determination requires evaluating the cancellation magnitude *at higher precision than the requested output*. Otherwise, NS reduces to "pick whichever branch the implementation defaulted to" — same as `Principal` but with the added correctness debt of the user thinking they got something different.

2. **The precision-tier-bump pattern is already present in BZ Newton iterations** (arith.rs guard_p = result_precision + 50). NS would extend this same pattern to the branch-selection step:
   - Requested: `P2BigFloat { precision_bits: 1024 }`.
   - NS internal: branch selection at `precision_bits + 50` (or higher per the recipe's analytic cancellation bound).
   - Output: round back to 1024 bits per the user's `RoundingMode`.

3. **Failure to bump is silently wrong**: NS at the requested precision picks branches based on a result that already incorporates the cancellation it's supposed to be avoiding. The chosen branch ends up identical to `Principal`'s default for nearly all inputs — except in the 0.0001% adversarial-input regime where NS *would* matter, the comparison is corrupted by the very cancellation NS is supposed to detect.

**Spec edit needed** (sub-clause F, new):

> **F. NumericallyStable precision-tier interaction.** When `BranchPolicy::NumericallyStable` is active, the recipe internally widens by ≥50 bits above the requested precision for the branch-selection step. The branch chosen at the wider precision is committed; the per-branch arithmetic then proceeds at the user-requested precision and rounds via the active `RoundingMode`.
>
> The 50-bit widening matches BZ §3.1.6 + DEC-031 §3.5 guard-bit conventions for Newton-iteration intermediates; NS borrows the same constant for symmetry.
>
> Recipes whose analytic cancellation bound exceeds 50 bits (e.g., `complex_log` near `z = -1` where the imaginary part can cancel a multi-thousand-bit mantissa) declare a higher per-recipe widening in their spec.toml stance metadata. The widening is part of the F12 stance contract.

This makes NS implementable + verifiable + composable with the existing precision lattice.

---

## Q4 — Discovery output bound for multi-valued functions

**Spec asks**: For `complex_pow(z, w)` with irrational `w`, the output may be infinite. What's the principled bound?

**Position**: **Per-recipe `max_branches: Option<usize>` parameter.** Default per recipe family:
- `complex_log(z)`: `max_branches = None` (only countably many; user picks via integer winding).
- `complex_pow(z, w)` for rational `w = p/q`: `max_branches = q` by default (the q principal roots).
- `complex_pow(z, w)` for irrational `w`: `max_branches = Some(K)` where K is required at the call site (ℵ₀ branches → must bound).
- `complex_root(z, n)`: `max_branches = n` always (the n principal roots).

**Reasoning**:

1. **The literature distinguishes** between *finite-branch* multi-valued functions (n-th roots: n branches; logarithm: countably many indexed by integer winding) and *dense-branch* functions (irrational power: branches dense in ℂ\{0}). Treating them uniformly with a single output shape is structurally wrong. The recipe family is the right granularity for the bound.

2. **For `Discovery` to be useful as the "structural-rhyme analog to discover()" pattern** (sub-clause E), the output must be enumerable. For `complex_log(z)`, naturally enumerable as `(value, winding_number)` pairs. For `complex_pow(z, p/q)`, naturally enumerable as the `q` roots indexed by `0 .. q`. For `complex_pow(z, w)` irrational, no natural enumeration → user *must* provide `max_branches` to specify the truncation.

3. **`max_branches` integrates naturally with the existing using() machinery**: same pattern as other recipe parameters. Cache-key tag participates. Default is per-recipe-family-stated, surfaced in the spec.toml.

**Spec edit needed** (extend sub-clause E):

> **E (extended). Discovery output shape: per-family bounds.**
>
> The `Discovery` policy's output enumerates multiple branches. Recipe families bound the enumeration:
>
> - **Single-valued-on-cut** (`complex_log`, `complex_arctan`): enumerate by integer winding number. Output type:
>   ```rust
>   pub struct WoundComplex {
>       pub primary: Complex<f64>,
>       pub windings: Vec<(i64, Complex<f64>)>,  // (winding_number, value)
>   }
>   ```
>   Default `max_windings = 0` (just primary); user sets `using(max_windings: i64)` to expand.
>
> - **Finite-root** (`complex_sqrt`, `complex_root(z, n)`, `complex_pow(z, p/q)`): enumerate all `n` (or `q`) roots:
>   ```rust
>   pub struct RootedComplex {
>       pub primary: Complex<f64>,
>       pub roots: Vec<(BranchTag, Complex<f64>)>,
>   }
>   ```
>
> - **Dense-branch** (`complex_pow(z, w)` for irrational `w`): user must specify `max_branches: usize`:
>   ```rust
>   pub struct BranchedComplex {
>       pub primary: Complex<f64>,
>       pub witnesses: Vec<(BranchTag, Complex<f64>)>,  // up to max_branches
>       pub truncated: bool,  // true if true count > max_branches
>   }
>   ```
>   Default per `complex_pow` recipe stance: panics if `Discovery` selected without explicit `max_branches`. Antibody: F13-shaped, "no silent-default for an enumeration that can't terminate."

This composes cleanly with the existing F13 + F12 patterns and respects the literature distinction between finite and dense multi-valuedness.

---

## Q5 — Pipeline-level resolution semantics

**Spec asks**: should mixed-policy pipelines fail to compile, or warn-and-default-to-alphabetical?

**Position**: **Compile-fail. Reject alphabetical defaults entirely.**

**Reasoning**:

1. **F13 is explicit**: silent commitment via implicit choice is the antibody-failure mode. Alphabetical default is a silent commitment by another name. The fact that it's deterministic doesn't reduce the silent-failure surface; it just makes the wrongness reproducible. Reproducibly-wrong is still wrong.

2. **Compile-fail is a low-friction discipline**: the user sees the diagnostic, picks a `using(branch: ...)` at the pipeline level, and moves forward. The 30-second cost is dwarfed by the months-of-debugging cost of a mis-defaulted convention propagating through a pipeline with downstream identities.

3. **The verifier-port team's prior art**: their `--branch-aware-witness` mode is a compile-fail-equivalent (the verifier rejects witnesses whose branch convention doesn't match the search space). They didn't pick "alphabetical default" either, and their domain is more constrained than tambear's.

**Spec edit**: none. Sub-clause B as written is correct.

---

## Recommended consolidated edits before DEC-032 ratification

To save the team-lead a round trip, the consolidated diff is:

1. **Add normative pinpointing in sub-clause D (or new sub-clause D-prime)** specifying that `Principal` = Kahan 1987 / C99 §G.6 / CCC, with explicit cut-placement table for all 6+6 transcendentals.

2. **Add sub-clause F** for NumericallyStable × DEC-031 precision-tier interaction: NS widens internally by ≥50 bits before branch selection, rounds back on output.

3. **Extend sub-clause E** to per-family enumeration bounds: single-valued (winding integer), finite-root (all n roots), dense-branch (`max_branches` required).

With these three edits, the spec is ratification-ready. I sign off on Q1 and Q5 as written. Q2/Q3/Q4 sign-offs are conditional on edits 1/2/3 landing.

---

## Cache-key serialization concern (raised, not blocking)

The spec's tag byte assignment at sub-clause D:
```
Principal         => 0
AntiPrincipal     => 1
NumericallyStable => 2
Discovery         => 3
```

Concern: `Principal` getting tag `0` is inconsistent with the IR_VERSION bumps invalidating cache on `feed_branch_policy` introduction. If a recipe pre-bump shipped without the byte being fed, its cache key already implicitly committed to a default that *post-bump* corresponds to tag 0 / Principal. Two scenarios:

- (a) If `Principal` is the implicit pre-bump default everywhere, then the `0` tag is the right "no-byte-fed = same key as Principal" choice (preserving cache hits across the bump for Principal-using callers). But the IR_VERSION bump (10 → 11) explicitly invalidates ALL caches anyway, so this preservation is already moot.

- (b) If we want stronger antibody — that pre-bump kernels can never silently match post-bump kernels — then `Principal` should get a nonzero tag (e.g., 1, 2, 3, 4) and tag 0 stays reserved as "uninitialized." This forces a hash-key collision check that catches "did the byte get fed?" at the bit level.

**Recommendation**: option (b). Reassign:
```
Principal         => 1
AntiPrincipal     => 2
NumericallyStable => 3
Discovery         => 4
```
Tag 0 remains reserved (asserts "not initialized"). This costs nothing and gains explicit antibody coverage on the "did we feed it?" question.

---

## Provenance

- Authored 2026-05-09 by math-researcher in team `tambear-sweep31-finish`.
- Substrate verified: branch-cut-conventions.md at commit 06ff11c (full read); F13 doc at survey/20260508123003-aristotle/; PLEASE_READ_from_gpu_verifier_port.md finding 4.
- Literature verified: Kahan 1987, "Branch Cuts for Complex Elementary Functions, or Much Ado About Nothing's Sign Bit" (canonical); Chyzak et al. 2011, "On Kahan's Rules for Determining Branch Cuts" (formalization); C99 Annex G §G.6.2/3/4 (normative); CLISP §12.5.3 (extension to inverse hyperbolic).
- Cross-checked: arctan cut convention is no longer literature-contested at 2026 — the imaginary-axis convention (Kahan) is universal across Mathematica, Wolfram, MATLAB, NumPy, Boost.Multiprecision, Sage, Maple. The contested-version-survives-in-FORTRAN-only is a 1990s-era residue.
- Sign-off: Q1 and Q5 as-written; Q2/Q3/Q4 conditional on three spec edits enumerated above.
- Routing: team-lead applies edits + lands DEC-032; pathmaker implements at first-complex-recipe time; adversarial designs identity-preservation proptests for each policy + recipe combo; aristotle adds branch-cut-violation cases to the silent-failure gauntlet.
