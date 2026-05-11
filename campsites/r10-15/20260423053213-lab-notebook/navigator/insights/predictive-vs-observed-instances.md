# Predictive vs Observed Instances — A Navigator Insight

Date: 2026-04-23

**CORRECTION 2026-04-23**: The motivating finding was wrong. Navigator-2 ran grep in R:/winrapids/crates/
when the tambear source lives in R:/tambear/crates/. The "zero hits" were Mode B (wrong-repo confusion),
not Mode A (true phantom). All cited instances ARE committed code in R:/tambear. The conceptual framework
below (predictive vs observed) remains potentially useful for future architecture documentation, but it
was NOT triggered by a real phantom-instance finding. Do not use this note as evidence that DEC-022
E-extension instances were ever phantom.

## The finding (retracted)

~~The phantom-instance audit (DEC-022 E-extension ratification, 2026-04-23) revealed that ALL instances
cited as evidence for the structural-form and input-coordinate sub-axes were campsite spec findings,
not committed code. The grep confirmed zero hits for every cited type name.~~

At first glance this looks like an evidence integrity problem. On deeper reflection, it's actually
the correct behavior of an Anti-YAGNI design culture — and it reveals a naming gap.

## Two kinds of instances

**Observed instance**: The code exists. Grep finds it. The property can be verified by reading the
committed source. Example: GAP-FP-2 conflation at `accumulate.rs` — real, grep-verifiable.

**Predictive instance**: The code exists in campsite spec. The property is structurally guaranteed
by the type system / algebraic constraints we're designing. The instance is cited to show *what the
code will look like and what properties it will exhibit when implemented*. Example:
`is_kernel_share_compatible_with` preorder structure — the algebraic argument is complete; the code
just hasn't been written yet.

Both kinds of instances are legitimate evidence for ratifying an architectural discipline. A predictive
instance is a *structural theorem*: "the type system we are designing guarantees this property will
appear in the implementation." That's stronger than a post-hoc observation in some ways — the design
commits to the property before the code exists.

## Why this matters for decisions.md

Current decisions.md language treats all cited instances as observed. After the phantom-instance audit,
aristotle is annotating which are predictive. This is not a retraction; it's precision.

The annotation form: "[Predictive: campsite spec; will exhibit property X when implemented because {algebraic
argument}]" vs "[Observed: commit hash; grep-verifiable]."

## The deeper rhyme

This parallels how mathematics works. A theorem about a mathematical structure is valid before any
specific instance of that structure is constructed. DEC-022 sub-clause E says: "implementations must
surface algebraic structure claims through named types at the call site." The predictive instances say:
"here are the algebraic structures the JIT design will instantiate — and this is what the surface
constraint will look like." The ratification is of the disciplinary RULE, not of the instances.

The instances prove the rule is non-trivial (it will actually apply to real things). Predictive
instances do this job just as well as observed instances — perhaps better, because they show the rule
was anticipated, not just discovered retroactively.

## Architectural implication

Consider adding a `[predictive]` / `[observed]` tag to all future instance citations in decisions.md.
This prevents both:
- False confidence (treating campsite specs as committed code)
- False retraction (dismissing a ratification because the cited code wasn't yet written)

Aristotle should own this tagging. Navigator should run the grep that verifies observed status.
