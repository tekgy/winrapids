---
campsite: tambear-formalize/sweep-31/math-researcher
role: math-researcher (covering scientist's gap per team-lead's 2026-05-08 routing)
date: 2026-05-08
sweep: 31 — BigFloat type-level home
gap-covered: Gap 1 — Oracle validation infrastructure (was scientist's)
status: draft for navigator review + team-lead routing
audience: pathmaker (implementation lead at Sweep 34); team-lead/Tekgy (ratification); aristotle (Gap 2 cross-reference)
inputs:
  - DEC-031 §3.9 (mpmath as PEER, not authority); §5 (Sweep 34 bridge sequence)
  - Sweep 31 DESIGN.md (R:\winrapids\campsites\tambear-formalize\sweep-31\math-researcher\DESIGN.md) — BigFloat v2 surface
  - Existing oracle catalog at R:\tambear\oracle\ — variance, mean, centered_moment, raw_moment, ewma, dd_subnormal_regimes, eft_primitives, moment_stats_merge (8 dirs + TEMPLATE + LIFT_EQUIVALENCE_TEMPLATE.md + README.md)
  - Sweep 10 oracle-bridge at R:\tambear\sweeps\10-oracle-bridge\ (README + STATE + STEPS)
  - PLEASE_READ_from_gpu_verifier_port.md §3 (verification-tier vs discovery-tier — strict ≠ sieve)
  - libm assumption-doc set, especially asin-rational-kernel.md §5 (corpus-design-as-claim)
---

# Sweep 31 Gap 1 — Oracle Validation Infrastructure

> **The headline.** DEC-031 §3.9 inverts the oracle authority relationship: tambear's BigFloat at p=200 (default) or p=500 (chains-E/F/G recommended floor) becomes the gold; mpmath/MPFR/SymPy become peers we cross-validate against. **This isn't a downgrade of mpmath** — mpmath stays in our test infrastructure forever. It's an upgrade of tambear: when our BigFloat and mpmath disagree, the workflow doesn't default to "mpmath is right, tambear is wrong." It triages the disagreement and resolves it, sometimes in our favor (file the bug upstream), sometimes in mpmath's (regression-test ourselves). The "bit-perfect or bug-found" Tambear Contract §10 promise is what this design operationalizes.
>
> **What's surprising**: the existing `oracle/variance/` catalog already has the right shape (data/ + per-package implementations + comparison/ + README.md + SPEC.md + generate_outputs.py). Sweep 34 is migration, not new construction. Most of the oracle infrastructure that DEC-031 §3.9 calls for **already exists** — sweeps 4 (winrapids substrate port) + 10 (oracle bridge plumbing) + the 8 oracle dirs all converge here.
>
> **What's load-bearing**: the discovery-tier-vs-verification-tier distinction from PLEASE_READ §3. Most "oracle" infrastructure in the wild is discovery-tier (numerical sieve, accept ULP-close). DEC-031 demands verification-tier (strict, bit-exact at requested precision OR the disagreement is investigated). We need to be loud about which tier we're operating in at every test site.

---

## 1. The six pieces — team-lead's checklist, answered

### 1.1 Validation harness shape

**Per-op proptest structure**: for each of the v2 surface (add, sub, mul, div, sqrt, cmp, from_f64, from_dd, to_f64, to_dd):

```
∀ inputs (sampled per per-op input distribution),
∀ p ∈ {sampled precisions},
∀ rounding ∈ {RoundToNearestTiesEven, RoundToNearestTiesAwayFromZero,
              RoundTowardZero, RoundTowardPositiveInfinity,
              RoundTowardNegativeInfinity}:
   tambear_BigFloat_op(inputs, p, rounding) == mpmath_op(inputs, p, rounding)
```

**Bit-exact equality is the assertion**, not ULP-close, at every test site. The verification-tier discipline (per PLEASE_READ §3): if mpmath and tambear differ by a single bit, that's a finding to investigate, not a tolerance to absorb. Discovery-tier sieves use ULP tolerances; we don't.

**Per-op input distributions** (verification-tier corpora):

| Op | Input distribution | Why |
|---|---|---|
| `from_f64` | full f64 bit pattern coverage: ±0, ±Inf, NaN, all subnormals from 2^-1074, MIN_POSITIVE neighborhood, mathematical constants ±1 ULP, powers of 2 from 2^-1074 to 2^1023, dense linspace in [-1, 1], log-spaced |x| in (0, f64::MAX] | exhaustive at boundaries; landmark coverage matches asin-corpus-design pattern |
| `to_f64` | full BigFloat range — sample BigFloat values from `from_f64` distribution + mpmath-generated BigFloats at p=200/500/1024 | inverse symmetry with from_f64 |
| `from_dd` | DDs constructed via from_f64 (lo=0, exercises diamond short-circuit) + DDs from EFT outputs (lo≠0, exercises hi+lo path) | both paths in §3.1 boundary table |
| `to_dd` | BigFloat values at p ≥ 106 (Strict) + at p ∈ [53, 106) (RoundingEquivalent) | three-regime per §3.1 |
| `cmp` | pairs from from_f64 distribution × itself; cross-product with NaN, ±Inf, ±0 to exercise tag dispatch | total order coverage |
| `add / sub` | pairs from from_f64 distribution × itself, including same-sign / opposite-sign / cancellation-prone (a, -a + ε) | catastrophic cancellation regime |
| `mul` | pairs from from_f64 distribution × itself; near-overflow products; near-underflow products; ±0 × finite; ±Inf × ±Inf | overflow/underflow boundaries |
| `div` | pairs from from_f64 distribution × itself; div-by-zero; div-by-Inf; near-overflow quotients | denominator-near-zero regime |
| `sqrt` | from_f64 distribution restricted to non-negative; ±0; ±Inf; subnormals; NaN; sqrt of perfect squares (exact) | exact-result detection |

**Precision sweep**: bit-exact at `p ∈ {53, 64, 106, 107, 200, 500, 1023, 1024}` per call. Includes the tier boundaries (53 = round-trip-identity exception; 106 = DD-source Strict boundary; 107 = first BigFloat-only precision; 1024 = §3.8 saturation cap).

**Rounding-mode sweep**: all 5 modes per call (per DEC-031 §1's `RoundingMode` enum). Verification-tier means each rounding mode produces a distinct expected output; the test asserts the right one.

**Test-corpus scale**: ~1000 inputs × 8 precisions × 5 rounding modes = 40,000 assertions per op. With 10 ops in v2 surface = 400,000 assertions. CI runs this nightly (proptest with shrinking enabled); per-PR runs a smaller sample (proptest cases=64 default).

### 1.2 Tolerances and scales — three-tier strictness

The "tolerance" for verification-tier is: **bit-exact equality** (`tambear_result.bits == mpmath_result.bits` at the requested precision). Not ULP-close. Not relative-error. Bit-exact.

**Three regression severities** when disagreement fires:

| Severity | Trigger | Action |
|---|---|---|
| **CRITICAL** | tambear vs mpmath differ on a "core arithmetic" op (add/sub/mul/div/sqrt) at any p ≥ 53 | block CI; investigate immediately; one of us has a bug |
| **HIGH** | tambear vs mpmath differ on conversion ops (from_f64/to_f64/from_dd/to_dd) at tier boundaries | block CI; usually a tier-dispatch issue or a rounding-mode disagreement |
| **MEDIUM** | tambear vs mpmath differ in NaN payload preservation | report but don't block; IEEE 754 allows variation |

Per the verification-tier discipline, **even MEDIUM disagreements get logged and investigated** — they're not noise, they're data about IEEE 754 conformance differences between the two implementations.

**Cross-precision consistency check** (a verification-tier idiom not in mpmath's standard test suites): for any input x and op f, compute f(x) at p=200 and at p=500. The p=500 result, rounded to p=200, MUST equal the p=200 result. **If it doesn't, our BigFloat is precision-inconsistent — a structural bug in the rounding logic.** This catches off-by-one errors in the guard-bit analysis (BZ §3.1.6) that pure tambear-vs-mpmath comparison would miss.

### 1.3 Bidirectional bug-finding workflow

When tambear-BigFloat and mpmath disagree, the triage chain:

1. **Cross-check at higher precision**: compute the result in both at p=2000 (well above any production use). The "true" answer should be representable at p=200/500 unambiguously. The implementation that converges to it is correct.
2. **Cross-check via Brent-Zimmermann reference**: the BZ algorithm specifications give exact procedures. Walk the procedure by hand (or with a third implementation); identify which library's intermediate step produces the wrong bits.
3. **Cross-check with MPFR (mpmath's underlying C lib)**: mpmath wraps MPFR. If our BigFloat agrees with MPFR-direct but disagrees with mpmath, the bug is in mpmath (rare, but has happened — mpmath's `mp.dps` rounding has historical bugs around tier boundaries).
4. **Cross-check with SymPy at exact rational precision**: for inputs expressible as exact rationals (e.g., `0.5`, `1.0/3.0` interpreted as the rational 1/3), SymPy gives the algebraic answer. Our BigFloat at finite precision is correct iff it's the correctly-rounded approximation of SymPy's exact value.
5. **Decide who's right**: the consensus of (1)-(4) determines who has the bug.
6. **File upstream when tambear is right**: bugs in mpmath/MPFR/SymPy get reported to those projects with a minimal reproducer derived from our test corpus. **This is the "bit-perfect or bug-found" promise made operational** — even users who never adopt tambear benefit from our rigor.
7. **Regression-test when tambear is wrong**: add the failing case to `tests/big_float_regressions.rs` with a comment linking to the disagreement origin. Future-tambear can never silently re-introduce the bug.

**Triage doc convention**: when a CRITICAL/HIGH disagreement fires, the investigator writes a brief at `R:\tambear\oracle\big_float\disagreements\YYYYMMDD-<short-slug>.md` documenting (a) inputs, (b) tambear output, (c) mpmath output, (d) higher-precision oracle result, (e) BZ-walk verdict, (f) resolution + commit hash. This becomes the disagreement archive — every line is a publication-grade artifact of the rigor process.

### 1.4 Sweep 34 bridge — migration from existing oracle catalog

Substrate verified: `R:\tambear\oracle\` already has 8 implemented dirs (variance, mean, centered_moment, raw_moment, ewma, dd_subnormal_regimes, eft_primitives, moment_stats_merge) plus TEMPLATE/ and LIFT_EQUIVALENCE_TEMPLATE.md. The existing structure per `oracle/variance/`:

```
oracle/variance/
  README.md           — claim + scope
  SPEC.md             — full 15-section template
  data/generated/     — input corpora (multiple distributions)
  generate_outputs.py — runs each implementation against data, emits output.json
  python-numpy/, python-pandas/, python-polars/, python-scipy/,
  python-statistics/, python-statsmodels/, R-base/, rust-accurate/
                      — per-package outputs + parameters.toml
  comparison/         — cross-implementation comparison artifacts
  validation/         — validation outputs against mpmath gold
```

**Migration to BigFloat as gold (Sweep 34)**: three concrete edits per oracle dir.

1. **Add `tambear-bigfloat-200/` and `tambear-bigfloat-500/` directories** with their own `output.json` from running tambear's BigFloat at p=200 and p=500. These slot into the existing per-package convention.
2. **Update `validation/` to compute against `tambear-bigfloat-500/` rather than `mpmath/`**. The mpmath outputs stay in the catalog as peer reference; the GOLD column in `comparison/` switches to the BigFloat-500 column.
3. **Add `disagreements/` subdirectory** for the triage docs from §1.3.

The 8 existing oracle dirs each get this 3-edit migration. **No new directory structure**; the existing template stretches to absorb BigFloat as the gold without architectural change. **This is why DEC-031 §5's Sweep 34 estimate is 3-5 days, not weeks** — most of the work is already done; we're slotting BigFloat in.

**Per-recipe migration table for Sweep 34**:

| Existing oracle dir | Migration action | Notes |
|---|---|---|
| `oracle/mean/` | add bigfloat-200, bigfloat-500 outputs | mean is BZ Algorithm 3.1 territory; tambear BigFloat will compute it at p=500 from raw f64 inputs |
| `oracle/variance/` | same | variance has the GAP-DET-1 conditioning issue at f64; BigFloat-500 should reproduce mpmath-50dps exactly |
| `oracle/centered_moment/` | same | parameterized variance; same migration |
| `oracle/raw_moment/` | same | same |
| `oracle/ewma/` | same | recurrence; AffineCompose; precision-sensitive at low decay rates |
| `oracle/dd_subnormal_regimes/` | **specifically informs Sweep 31's DD↔BigFloat boundary tests** — already targets the §3.1 boundary regimes | this dir is MOST relevant for our gap; cross-link bidirectionally |
| `oracle/eft_primitives/` | add bigfloat outputs as the verification-tier reference for two_sum/two_product_fma exact outputs | EFT primitives are bit-exact by definition; BigFloat oracle confirms via independent path |
| `oracle/moment_stats_merge/` | same as variance | Welford+Pebay merge across batches |

### 1.5 Libm verification-tier integration

Per PLEASE_READ §3 + #4 from the GPU verifier port team:

> **Strict verification is much stricter than numerical-sieve discovery**. The Rust port's verifier rejects ~95% of candidates the SI's numerical sieve accepts. **Implication for tambear**: when a recipe's "validation" is built, decide explicitly whether it's the discovery-tier sieve or the verification-tier strict check. They should NOT be the same.

For the libm formalization sweep (downstream of Sweep 31), the integration shape:

**Discovery-tier test** (already exists): `assert!(ulps_between(sin_strict(x), x.sin()) <= 2)`. Compares against vendor libm; tolerates ULP slack. This catches gross algorithmic errors but accepts ULP-noise. **Lives at the recipe's `tests` module.**

**Verification-tier test** (Sweep 31 enables): `assert_eq!(sin_strict(x).to_bits(), bigfloat_sin(BigFloat::from_f64(x, 500)).to_dd().to_f64().to_bits())`. Bit-exact, BigFloat at p=500 as gold, strict-tier polynomial as the unit-under-test, rounded back to f64. **This catches polynomial-coefficient bugs (the asin P_S2/P_S5 class) at the bit level.**

The two tests live side-by-side; both run in CI. Discovery-tier is fast (vendor libm is fast); verification-tier is slower (BigFloat at p=500 adds ~100x to per-input cost) but runs on the adversarial corpus from `recipes/libm/adversarial.rs::float_landmarks`.

**API shape** for the verification-tier test harness:

```rust
/// Verification-tier oracle test: compute the recipe's mathematical
/// reference at BigFloat(p=500), round to f64, assert bit-exact match
/// against the recipe's strict-tier output.
///
/// # Panics
///
/// Panics on ANY bit-mismatch (not ULP-close). The panic message
/// includes:
/// - input x
/// - recipe output bits
/// - BigFloat-rounded bits
/// - region classification (kernel boundary, half-angle path, etc.)
/// - corpus tag (which adversarial generator produced this input)
pub fn verify_libm_strict<F1, F2>(
    inputs: &[f64],
    strict_recipe: F1,
    bigfloat_oracle: F2,
    region_classifier: fn(f64) -> &'static str,
)
where
    F1: Fn(f64) -> f64,
    F2: Fn(BigFloat) -> BigFloat,
{
    for &x in inputs {
        let recipe_out = strict_recipe(x);
        let bf_in = BigFloat::from_f64(x, 500);
        let bf_out = bigfloat_oracle(bf_in);
        let oracle_out = bf_out.to_f64();
        if recipe_out.to_bits() != oracle_out.to_bits() {
            let region = region_classifier(x);
            panic!(
                "verification-tier failure for x={x:e} (region: {region}):\n\
                 recipe   = {recipe_out:e} (bits {:#018x})\n\
                 oracle   = {oracle_out:e} (bits {:#018x})\n\
                 ulp diff = {}",
                recipe_out.to_bits(),
                oracle_out.to_bits(),
                ulps_between(recipe_out, oracle_out),
            );
        }
    }
}
```

**The region_classifier closure ties into asin-rational-kernel.md §5's corpus-design-as-claim work**: each adversarial input carries a region tag (e.g., `"kernel_boundary"`, `"half_angle_path"`, `"trivial_tiny_x"`); failures get logged with their region. Over time, the disagreement archive shows which regions are bug-prone — the corpus-design metric Aristotle's analysis flagged.

### 1.6 Sweep-10 oracle-bridge cross-reference

Sweep 10 (Oracle Bridge, PLANNED per SWEEP_HISTORY.md) was rescoped during the 2026-04-21 crate-architecture consolidation into a dual-mission cross-language oracle effort:

- Implement `tambear::primitives::oracle_compare` (currently scaffolded only — `abs_within` / `rel_within` / `ulp_within` / `NanPolicy` + slice helpers + `FirstMismatch` with ~200 tests).
- Land the first proof-of-methodology oracle at `oracle/<recipe>/` — `log` is the leading candidate.
- Wire one fintek `*_expected.json` through the new plumbing.

**What survives**: all of it. Sweep 10's `oracle_compare` tooling is the substrate Sweep 34 uses. The `abs_within`/`rel_within`/`ulp_within` helpers are exactly what the verification-tier `verify_libm_strict` harness needs — except we use `ulp_within(0)` (zero ULP tolerance, i.e., bit-exact) for verification-tier and `ulp_within(N)` for discovery-tier. Same plumbing, different tolerance.

**What gets reshaped**: the "log is the leading candidate" line. With BigFloat as gold, log goes from "first oracle to land" to "first verification-tier libm test." The discovery-tier log-vs-vendor-libm test was the original Sweep 10 deliverable; the verification-tier log-vs-BigFloat-at-500-bits test is the Sweep 34 follow-on. **Both ship; they're orthogonal.**

**Sweep 10 STATE doc** (`R:\tambear\sweeps\10-oracle-bridge\STATE.md`) and STEPS doc would be updated to reflect that Sweep 31 lands BigFloat first, then Sweep 34 promotes it to gold. The Sweep 10 plumbing is precisely what makes that promotion mechanical.

**The `oracle_compare` API** (per Sweep 10 scaffold): `assert_within_ulps(actual, expected, max_ulps, label)`. We extend with `assert_bit_exact(actual, expected, label)` for verification-tier. **Same module; same naming convention; one more function.**

---

## 2. The discovery-tier vs verification-tier table

Per PLEASE_READ §3, this distinction is load-bearing. Making it explicit per Sweep 31's deliverables:

| Tier | Tolerance | Reference | When to use | Cost |
|---|---|---|---|---|
| **Discovery-tier** | `ULP-close` (typically ≤ 2 ULPs) | vendor libm (`f64::sin`, etc.) | hand-picked tests; sanity checks; CI smoke tests | fast (~ns per input) |
| **Verification-tier** | bit-exact | tambear BigFloat at p=500, rounded to dest precision | adversarial corpora; pre-release validation; bug audits | slower (~µs per input due to BigFloat arithmetic) |

The two tiers coexist at every recipe's test module. **The recipe's spec.toml stance metadata declares which tier is the contract** (per F12 / aristotle's schema). For `_strict` strategies, the contract is verification-tier with a per-strategy ULP budget (`≤ 2 ulps`). The spec.toml's claim is honored iff the verification-tier test passes within that budget.

**Critically**: a `_strict` strategy that passes discovery-tier `≤ 2 ulps` against vendor libm but FAILS verification-tier bit-exact against BigFloat-at-500 means the vendor libm and tambear's BigFloat disagree. This is the third disagreement class beyond the two in §1.3 — **vendor-libm-vs-bigfloat disagreement**, where tambear's strict polynomial is correct, vendor libm is correct (within its own ULP budget), and our BigFloat is correct. Three different "correct"s. The triage chain in §1.3 still applies; we just have three reference points instead of two.

This is **why** the libm assumption-doc set (Cody-Waite reduction, asin rational P/Q, IEEE 754-2019 §9.2 exactness) matters — every assumption is a constraint that BigFloat's oracle must satisfy. If BigFloat at p=500 disagrees with the libm assumption-doc's stated exact behavior (e.g., `cospi(integer) = ±1` exactly), one of them is wrong.

---

## 3. Test-suite scale and CI shape

**Nightly run** (full proptest cases, full precision sweep, full rounding-mode sweep):
- ~400,000 BigFloat assertions (per §1.1)
- ~50,000 libm verification-tier assertions (32 recipes × 1500 adversarial inputs avg)
- BigFloat at p=500 cost ≈ 5 µs/op; libm verification cost ≈ 50 µs/input (BigFloat-call dominates)
- Total: ~3 seconds for BigFloat suite; ~40 minutes for libm verification suite
- Plus disagreement triage if anything fires

**Per-PR run** (proptest cases=64, precision={53, 200, 500}, rounding=RoundToNearestTiesEven):
- ~5,000 BigFloat assertions
- ~2,000 libm verification-tier assertions (sample of the adversarial corpus)
- ~3 seconds total — fits in standard PR CI budget

**Disagreement-triage SLA**: CRITICAL within 24 hours; HIGH within a week; MEDIUM logged in `disagreements/` for batched periodic review.

---

## 4. Open questions for ratification

1. **mpmath integration mechanism**: subprocess (Python via pyo3 or `Command::new("python")`) or static gold corpus (pre-computed JSON)? My recommendation: **both**, per aristotle's §4 sharpening from the schema-doc dialogue. Static corpus for CI (reproducible, fast); live mpmath subprocess for dev loop (refit-and-rerun-coefficients). Sweep 34 ships static; Sweep 35 wires live.

2. **Precision floor for "default" oracle gold**: DEC-031 §3.9 says p=200 default, p=500 recommended floor. Should the verification-tier test default to 500 (per chains-E/F/G) and accept p=200 only with explicit opt-in? **My recommendation: yes** — default to 500. The 200 case is for fast-path discovery-tier-adjacent uses; the verification-tier we ship to consumers should be 500.

3. **Cross-precision consistency check** (§1.2): every op test runs at TWO precisions (p₁, p₂) and asserts the round-down agrees. Cost: doubles the test count. Catches precision-rounding bugs that single-precision can't. **My recommendation: yes for v1**; the cost is bearable and the catch is real.

4. **NaN payload preservation**: the gauntlet (Aristotle's Gap 2) covers this surface. Is "NaN payload preserved" a verification-tier requirement or a MEDIUM-only check? **My recommendation: MEDIUM** — IEEE 754 explicitly allows variation. We assert sign-of-NaN preservation but not full payload.

5. **Float subnormal regime in libm tests**: per DEC-031 chains-E/F/G, subnormal inputs hit different error bounds. Should the libm verification-tier test SKIP subnormal-input cases (they're outside the polynomial-fit domain), or run them at higher BigFloat precision (p=2000)? **My recommendation: run at p=2000**, document the result. The asin polynomial doesn't have subnormal inputs in its valid domain, so subnormal-x simply tests the "tiny-x trivial path" — `asin(2^-1074) = 2^-1074` exactly, BigFloat will confirm.

6. **Libm `_compensated` and `_correctly_rounded` paths**: per F4 audit, ~30 of 32 are aliases to `_strict`. The verification-tier test should run against the actual implementation called (not the named strategy), so all three strategies of an aliased recipe pass identically. **Once aristotle's F12 schema lands, the lint catches the aliasing AND the verification-tier test confirms the alias is honored at the bit level.** Two complementary checks.

---

## 5. Cross-references

**Inward** (this doc cites):
- Sweep 31 DESIGN.md (the v2 surface specification)
- DEC-031 §3.9, §5, §6 #13
- libm assumption-doc set: trig_reduce-sharing, cody-waite-payne-hanek-crossover, ieee-754-2019-pi-scaled-exactness, asin-rational-kernel (the §5 corpus-design-as-claim ties to §1.5)
- aristotle's spec-toml-stances-schema.md (F12 lint shapes the verification-tier expectation)
- aristotle's default-is-a-claim.md (the prose claim is what verification validates)

**Outward** (other docs should cite this):
- Sweep 31 implementation tasks (via DESIGN.md)
- Sweep 34 oracle-migration tasks (this doc IS the migration plan)
- libm formalization sweep (this doc IS the verification-tier integration plan)
- Aristotle's silent-failure gauntlet (Gap 2) — cross-link bidirectionally; the gauntlet's regression-witnesses populate `disagreements/`

---

## 6. Dependencies and sequencing

```
Sweep 31 (BigFloat impl)
  └─→ Sweep 32 (cache-key plumbing for the precision-coordinate axis)
      └─→ Sweep 33 (TAM routing for force-cpu-for-bigfloat)
          └─→ Sweep 34 (THIS DOC — oracle-migration to BigFloat-as-gold)
              └─→ Sweep 35 (BigFloat transcendentals — exp/log/sin/cos at BigFloat precision)
                  └─→ libm formalization sweep (verification-tier integration)
```

Sweep 31 + Sweep 34 together complete the oracle infrastructure. Sweep 35 is BigFloat transcendentals (Brent-Zimmermann §4, AGM-based or series-based at high precision); Sweep 35 unblocks the verification-tier test for libm recipes that the discovery-tier doesn't catch (the asin P_S2/P_S5 class).

**The full chain DEC-031 § 5 references** (Sweep 31 → 32 → 33 → 34 → 35+): this doc fills in the Sweep 34 piece. Aristotle's gauntlet (Gap 2) fills in the structural-failure-mode coverage. Together they ratify the "bit-perfect or bug-found" promise at design time, before code lands.

---

## 7. Provenance

- Authored 2026-05-08 by math-researcher in team `tambear-formalize`, covering scientist's gap per team-lead's 2026-05-08 routing (scientist + adversarial in zombie state from rate-limit cascade).
- Substrate verified: existing oracle catalog at `R:\tambear\oracle\` (8 dirs + TEMPLATE + variance/SPEC.md as the deepest exemplar); Sweep 10 oracle-bridge at `R:\tambear\sweeps\10-oracle-bridge\`; PLEASE_READ_from_gpu_verifier_port.md §3 (verification-tier vs discovery-tier framing); DEC-031 §3.9 + §5.
- Cross-checks: existing `oracle/variance/` already has the right shape (data + per-package + comparison + validation); BigFloat-as-gold is a 3-edit-per-dir migration, not a rebuild. The Sweep 34 estimate of 3-5 days holds.
- Aristotle has Gap 2 (silent-failure proptest gauntlet) at task #29; bidirectional cross-link expected once that doc lands.
- This is a draft. Open questions in §4 require navigator/aristotle/team-lead review. Ratification gates Sweep 34 execution (which can run in parallel with Sweep 31 implementation; the gauntlet, Sweep 32, and Sweep 33 don't sequence-block Sweep 34's test-infrastructure design).
