# Oracle Runner Design — Campsite 4.8 (I9′ scientist track)

**Owner:** scientist
**Status:** design doc — code follows after campsite 4.6 (waiting for pathmaker's IR ops commit)
**Date:** 2026-04-12

---

## What this is

Navigator accepted Aristotle's I9′ Move (v4): the oracle becomes a continuous, registered
artifact — not a final audit. The adversarial owns **corpus curation** (what inputs to test,
which special cases, which identities). The scientist owns the **runner**: the infrastructure
that takes a named entry from the shared `oracles/` registry and executes it against a
tambear-libm function, comparing TESTED results to CLAIMED ULP bounds.

This design doc is step 1. Code is step 2.

---

## Ownership boundary (precise)

| Who | Owns |
|-----|------|
| Adversarial | `oracles/<fn>.toml` — the corpus entries: input sets, expected outputs, identity checks, claimed ULP bounds |
| Scientist | `oracle_runner::run_oracle(entry, candidate_fn)` — reads the entry, runs the candidate, reports TESTED vs CLAIMED |
| Math-researcher | supplies the TAMBLMR1 `.bin` reference files (via `gen-reference.py`) that the runner uses as mpmath truth |

The runner never curates inputs. The runner never decides what "close enough" means. Both
of those are in the oracle entry written by adversarial. The runner only executes the entry's
instructions and produces a structured report.

---

## Oracle entry format (`oracles/<fn>.toml`)

Each function has one entry file. Adversarial writes it; scientist reads it. Format:

```toml
[function]
name = "tam_exp"
primary_domain = [-708.0, 709.0]      # domain over which ULP bound is claimed
claimed_max_ulp = 1.0                 # the published Phase 1 bound
reference_bin = "peak2-libm/exp-1m.bin"  # TAMBLMR1 file produced by gen-reference.py

[corpus]
# Named injection sets. Each is a list of f64 values.
# Runner runs every value in every set; report groups results by set name.
special_values = [0.0, -0.0, "inf", "-inf", "nan", -708.4, 709.8, -745.0]
argument_reduction_boundaries = ["ln(2)", "2*ln(2)", "ln(2)/2"]
near_zero = [1e-300, 1e-200, 2.2e-308]  # subnormal boundary
near_overflow = [709.0, 709.78, 710.0]
cody_waite_constants = ["ln(2)", "2*ln(2)", "3*ln(2)", "10*ln(2)"]

[identities]
# Each identity is: eval the expression, compare to 0.0 ± tolerance_ulp ULPs.
# The runner evaluates the expression symbolically using the candidate function.
# Variable `x` is drawn from the primary_domain sample set.
[[identities.checks]]
name = "exp_log_roundtrip"
expr = "exp(log(x)) - x"   # should be ~0 for x > 0
tolerance_ulp = 2.0
domain = [1e-300, 1e300]   # positive reals only

[[identities.checks]]
name = "exp_negation"
expr = "exp(x) * exp(-x) - 1.0"
tolerance_ulp = 2.0
domain = [-700.0, 700.0]
```

The `special_values` field can use string expressions for constants that aren't representable
as literal `f64` in TOML (e.g. `"ln(2)"` → evaluated via mpmath at 50 digits, then
faithfully rounded to the nearest fp64). The runner evaluates these strings once at load time
using the same mpmath infrastructure as the reference generator.

**Bit-pattern exact checks** (`[bit_exact_checks]` section, separate from injection sets):

```toml
[bit_exact_checks]
# Each entry: input → required bit pattern.
# Used for signed-zero, quiet-NaN-exact, and overflow-boundary tests.
# These CANNOT use ULP distance — signed zero (+0 vs -0) differs by 1 ULP but
# must be bit-exact per IEEE 754. NaN payloads are also bit-pattern-specific.
#
# Adversarial B1 requirement: exp(-inf) must be +0.0, not -0.0
[[bit_exact_checks.cases]]
name = "exp_neg_inf_positive_zero"
input = "-inf"
expected_bits = "0x0000000000000000"  # +0.0
rationale = "IEEE 754: exp(-inf) = +0, not -0. Some libms return -0 due to sign-propagation bugs."

[[bit_exact_checks.cases]]
name = "exp_underflow_positive_zero"
input = -746.0
expected_bits = "0x0000000000000000"  # +0.0
rationale = "Underflow path must not set the sign bit. exp(-746) < MIN_SUBNORMAL rounds to +0."
```

These are checked with `candidate(input).to_bits() == expected_bits`, not with ULP distance.
A ULP comparison would accept `-0.0` as 0 ULP from `+0.0` (they are adjacent in the IEEE
ordered set), but the requirement is stricter: the sign bit must be correct.

---

## Runner API (Rust, in `tambear-tam-test-harness`)

```rust
/// A named oracle entry loaded from `oracles/<fn>.toml`.
pub struct OracleEntry {
    pub function_name: String,
    pub claimed_max_ulp: f64,
    /// Records from the TAMBLMR1 reference binary.
    pub reference_records: Vec<RefRecord>,
    /// Named injection sets: (set_name, vec_of_f64_inputs).
    pub injection_sets: Vec<(String, Vec<f64>)>,
    /// Bit-pattern exact checks (signed-zero, NaN payload, overflow boundary).
    pub bit_exact_checks: Vec<BitExactCheck>,
    /// Identity checks.
    pub identity_checks: Vec<IdentityCheck>,
}

/// A check that the bit pattern of candidate(input) equals expected_bits exactly.
///
/// Used for signed-zero (+0 vs -0), NaN payloads, and cases where ULP distance
/// would pass a wrong answer (ULP(+0, -0) = 1 but the sign must be correct).
pub struct BitExactCheck {
    pub name: String,
    pub input: f64,
    pub expected_bits: u64,
    pub rationale: String,  // shown in failure output
}

pub struct BitExactResult {
    pub name: String,
    pub input: f64,
    pub expected_bits: u64,
    pub actual_bits: u64,
    pub passes: bool,
}

pub struct IdentityCheck {
    pub name: String,
    pub tolerance_ulp: f64,
    // The expression is pre-compiled to a Rust closure by the loader.
    // This avoids eval-at-runtime complexity in Phase 1.
    pub check: Box<dyn Fn(f64, &dyn Fn(f64) -> f64) -> f64>,
    pub domain: (f64, f64),
}

/// Load an oracle entry from the TOML file + its TAMBLMR1 reference binary.
pub fn load_oracle_entry(toml_path: &str) -> Result<OracleEntry, String>;

/// Run a candidate function against all components of an oracle entry.
pub fn run_oracle(entry: &OracleEntry, candidate: impl Fn(f64) -> f64) -> OracleReport;

pub struct OracleReport {
    pub function_name: String,
    pub claimed_max_ulp: f64,
    /// ULP report over the full reference set.
    pub random_sample_report: UlpReport,
    /// Per injection-set ULP reports.
    pub injection_reports: Vec<(String, UlpReport)>,
    /// Bit-pattern exact check results (signed-zero, NaN payload, etc).
    pub bit_exact_results: Vec<BitExactResult>,
    /// Identity check results.
    pub identity_results: Vec<IdentityResult>,
    /// true if all components pass their claimed bounds.
    pub passes: bool,
}

pub struct IdentityResult {
    pub name: String,
    pub max_residual_ulp: f64,
    pub claimed_tolerance_ulp: f64,
    pub passes: bool,
    /// Worst-case input (for the failure message).
    pub worst_case_input: Option<f64>,
}
```

`OracleReport::passes` is `true` iff:
1. `random_sample_report.max_ulp ≤ claimed_max_ulp`
2. Every injection set: `max_ulp ≤ claimed_max_ulp` (same bound — no looser spec for special cases)
3. Every bit-exact check: `actual_bits == expected_bits`
4. Every identity check: `max_residual_ulp ≤ tolerance_ulp`

Note: bit-exact checks are strictly STRONGER than ULP checks. `+0.0` and `-0.0` are 1 ULP
apart in the IEEE ordered set (adjacent values) — a ULP check would pass either one. The
bit-exact check enforces the sign. This distinction matters for underflow paths and
signed-infinity results. The adversarial review of exp-design.md (B1) identified this gap.

---

## What the runner does NOT do

- Does not decide what inputs to use (that's the corpus entry).
- Does not decide what "acceptable" means (that's `claimed_max_ulp` in the entry).
- Does not evaluate identity expressions at runtime. Phase 1: the runner uses a fixed set
  of hard-coded identity check functions (one per function, registered by name). The TOML
  identity section names them; the runner looks them up in a static registry. If a name
  is unknown, the runner fails with an error, not silently skips.
- Does not run across multiple backends in Phase 1. The runner takes a single `candidate`
  closure. Cross-backend comparison is `assert_cross_backend_agreement`'s job.

---

## Integration with the existing harness

The runner builds on `ulp_harness.rs` (campsite 2.3). It reuses:
- `read_reference_bin` — loads the TAMBLMR1 file
- `UlpReport::measure` — measures the random sample set
- `ulp_distance_with_special` — handles NaN/inf edge cases

The runner adds:
- TOML parsing of the oracle entry (using `toml` crate, already common in this workspace)
- Injection-set handling (just calling `UlpReport::measure` on a smaller input set)
- Identity check dispatch (static registry, Phase 1)

New crate dependency: `toml` (for loading `.toml` oracle entries). Or use a simpler
custom parser for the subset of TOML we use — the entry format above is simple enough
that a hand-written parser could work. Decision: use the `toml` crate for correctness,
since oracle entries must be machine-writable by the adversarial role and not subtly
mis-parsed.

---

## Phase 1 scope (what lands at campsite 4.8)

1. `OracleEntry`, `OracleReport`, `IdentityResult` structs.
2. `load_oracle_entry(path)` — loads TOML + TAMBLMR1 reference.
3. `run_oracle(entry, candidate)` — produces full `OracleReport`.
4. Static identity registry for `tam_exp` (the first function to land from Peak 2):
   - `exp_log_roundtrip`: `|exp(log(x)) - x|` for x ∈ [1e-300, 1e300]
   - `exp_negation`: `|exp(x) * exp(-x) - 1.0|` for x ∈ [-700, 700]
5. Integration test: `oracle_runner_exp_passes_claimed_bound` — reads `exp-1k.bin` (already
   generated), runs `f64::exp` as candidate (calibration baseline), verifies TESTED ≤ CLAIMED.
6. Integration test: `oracle_runner_identity_exp_log` — verifies the identity check
   infrastructure works with a known-good candidate.

Phase 2 (when more Peak 2 functions land): add identity registrations per function, add
the adversarial's corpus entries for each function.

---

## Campsite 4.8 sequencing

- **Blocked on:** campsite 4.6 (which is blocked on pathmaker's IR ops commit).
- **Blocked on:** adversarial providing at least one oracle entry (`oracles/tam_exp.toml`).
  The runner can be written and tested with the calibration baseline (`f64::exp`) before
  any tambear-libm function exists, so code development can start once 4.6 is done.
- **Not blocked by:** actual tambear-libm implementations. The runner accepts any `Fn(f64) -> f64`.

---

## What campsite 4.8 delivers

A test that, when run, produces output like:

```
oracle: tam_exp
  random sample: max_ulp=0.00, mean_ulp=0.00, passes=true [calibration: f64::exp]
  injection[special_values]: max_ulp=0.00, passes=true
  injection[argument_reduction_boundaries]: max_ulp=0.00, passes=true
  identity[exp_log_roundtrip]: max_residual=1.23 ULP, tolerance=2.0 ULP, passes=true
  identity[exp_negation]: max_residual=0.51 ULP, tolerance=2.0 ULP, passes=true
  OVERALL: PASS (claimed 1.0 ULP, tested 0.00 ULP on random sample)
```

When the real `tam_exp.tam` implementation lands (Peak 2 campsite 2.6+), the same
test runs against it. If anything fails, the report names the failing category and the
worst-case input. Navigator sees the failure immediately.

---

## Why design doc before code

The runner is the interface between adversarial (corpus) and scientist (execution). Both
sides need to agree on the oracle entry format before either writes a line. This doc is
that agreement proposal. If adversarial has feedback on the TOML format or the identity
check interface, route it back before code starts.

The `IdentityCheck.check` as a pre-compiled closure (not runtime eval) is the one design
decision most likely to draw feedback. The alternative (a tiny expression evaluator) would
let the adversarial write arbitrary expressions in TOML. Rejected for Phase 1: an expression
evaluator is another thing that can be wrong, and the oracle must not contain bugs. Static
pre-compiled closures are auditable; string expressions aren't. Phase 2 can revisit if the
function catalog grows large enough to make per-function registration impractical.
