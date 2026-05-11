---
campsite: tambear-formalize/survey/20260508123003-math-researcher
role: math-researcher
date: 2026-05-08
subject: tambear-trig campsite — survey of math content
status: complete
---

# tambear-trig — Math Survey

> Vocabulary lock applies. Five tiers: Pipelines (5) → Recipes (4) → Atoms (3) → Op/Expr (2) → Primitives (1). Older notes inside this campsite use *primitive* / *atom* / *layer* with pre-lock meanings; map per `R:\winrapids\docs\architecture\vocabulary.md` before quoting.

## TL;DR

The 13 sub-campsites at `R:\winrapids\campsites\tambear-trig\` are **scope-marker scaffolds**, not knowledge stores. The actual trig math lives at `R:\winrapids\crates\tambear\src\recipes\libm\` — 18 `.rs` files (~6270 lines) + 44 `.spec.toml` files. **None of it has shipped to `R:\tambear\`.** The destination has zero recipes/libm/.

This material is **publication-grade and ripe to formalize.** The math is correct and well-referenced (Muller 2018 / Cody-Waite 1980 / Payne-Hanek 1983 / fdlibm structurally / Hart 1968). Recent commits show real adversarial-found bugs being fixed in the source. The work to do is mechanical: lower onto the locked-vocabulary atoms+primitives substrate at the destination, refresh spec.tomls to use the lock-correct `primitives_used` taxonomy, oracle against mpmath at 50-digit, run through the adversarial harness.

---

## What's in the campsite directory

13 sub-campsites under `R:\winrapids\campsites\tambear-trig\`:

| Sub-campsite | Owning role | Content |
|---|---|---|
| `20260414142355-catalog` | math-researcher | One-line creation note: "Enumerate the complete trig catalog + history + uses" |
| `20260414142356-angle-units` | math-researcher | One-line: "Angle unit parameterization design (rad/deg/grad/turns/pi)" |
| `20260414142356-atoms-gaps` | aristotle | One-line: "New atoms/exprs/ops required for full trig family" |
| `20260414142356-autodiscover` | adversarial | One-line: "Probe catalog for trig auto-dispatch" |
| `20260414142356-compilation` | pathmaker | (creation note only) |
| `20260414142356-defaults` | scientist | (creation note only) |
| `20260414142356-implementations` | pathmaker | **Has one substantive file**: `scout/insights/asin-polynomial-audit.md` — fdlibm-lineage audit of two pre-fix bugs in asin's P_S2 (digit transposition `2.012255…` vs correct `2.012125…`) and P_S5 (invented constant `-3.25e-6` vs correct `+3.479e-5`). Both fixed in source as of `bbda152`. |
| `20260414142356-notation` | aristotle | (creation note only) |
| `20260414142356-shared-pass` | naturalist | One-line: "Shared-pass optimizations: sincos, hyperbolic_bundle, TrigSharedIntermediate" |
| `20260414142356-spec-tomls` | pathmaker | (creation note only) |
| `20260414142356-tbs-syntax` | pathmaker | (creation note only) |
| `20260414142356-variants` | pathmaker | (creation note only) |
| `20260414144052-shared-pass` | naturalist | duplicate / re-creation |

None of these appear in `campsite list --status active`. They're scaffolded but unlogged. The total signal in the campsite tree is the asin polynomial audit + the topic-list recoverable from the 13 directory names.

## What's actually on disk (the real material)

`R:\winrapids\crates\tambear\src\recipes\libm\`:

### Implemented .rs files (~6270 lines, 18 files)

| File | Lines | What it owns |
|---|---:|---|
| `sin.rs` | 932 | sin/cos kernels with Cody-Waite 3-part π/2 + Payne-Hanek 1200-bit 2/π reduction; Remez-refit minimax polynomials in mpmath at 80-digit; quadrant fixup |
| `tan.rs` | 506 | tan with separate kernel structure; pole handling near (n+½)·π |
| `erf.rs` | 487 | erf, erfc — fdlibm rational approximation, 1 ulp tightening |
| `exp.rs` | 479 | exp via range reduction k·ln(2) + r, three-strategy lowering |
| `pi_scaled_inv.rs` | 400 | asinpi, acospi, atanpi, atan2pi — IEEE 754-2019 pi-scaled inverses |
| `gamma.rs` | 376 | tgamma, lgamma — Lanczos approximation |
| `log.rs` | 339 | log via mantissa+exponent split, polynomial on reduced argument |
| `hyperbolic.rs` | 316 | sinh, cosh, tanh — sinh uses expm1 path for small x; tanh saturation never exactly ±1 for finite x (per `d54b749`) |
| `atan.rs` | 309 | atan + atan2; 4-sub-interval Cody-Waite reduction (or 5-interval AT constants per `4d2b5e9`) |
| `pi_scaled.rs` | 289 | sinpi, cospi, tanpi — exact integer/half-integer/quarter-integer; (-1)^n sign flips per `1f9d347`, `5b89ab7` |
| `inv_hyperbolic.rs` | 268 | asinh, acosh, atanh — log-form identities with cancellation guards |
| `asin.rs` | 261 | asin via fdlibm rational P/Q on |x| ≤ 0.5; half-angle reduction beyond; coefficients refit and bug-audited (P_S2/P_S5) |
| `rare_trig.rs` | 199 | versin (via 2·sin²(x/2) — no cancellation), haversin, gudermannian, all special cases |
| `inv_recip.rs` | 170 | sec/csc/cot + asec/acsc/acot — composed from sin/cos/tan/asin/acos/atan |
| `sincos.rs` | 139 | fused sincos — single reduction, both kernels |
| `sincos_pi.rs` | 61 | fused sincospi |
| `mod.rs` | 49 | module declarations |
| `adversarial.rs` | 690 | generator-backed harness: float-landmark inputs, ±1 ulp neighborhoods of region transitions, dense linspace sweeps |

### Spec.toml files (44 total)

Each `.spec.toml` is a per-recipe SPEC under the template established by `oracle/variance/SPEC.md`. The template covers: name, layer, family tags, description, long_description, references with kind+note, decomposition.primitives_used, decomposition.kingdom, sharing.{reads,writes}, parameters (with default + domain), outputs (with shape + has_v_column), writeup.methods_template.

Spec-only (no matching `.rs`) — these declare scope; the math lives inside one of the larger `.rs` files: `acos`, `acosh`, `acospi`, `acot`, `acsc`, `asec`, `asinh`, `asinpi`, `atanh`, `atanpi`, `atan2`, `cos`, `cosh`, `cospi`, `cot`, `csc`, `gudermannian`, `haversin`, `inv_gudermannian`, `sec`, `sinh`, `sinpi`, `tanh`, `tanpi`, `versin`.

Spec + dedicated `.rs`: `asin`, `atan`, `exp`, `log`, `sin`, `sincos`, `sincospi`, `tan`.

## Quality assessment — passes the Filter Test

Sampling `sin.spec.toml` against the 10-point Tambear Contract:

1. **Custom-implemented, our way** — yes. fdlibm referenced structurally (P/Q rational form, 4-interval atan reduction, kernel-evaluation skeleton); coefficients independently refit in mpmath at 80-digit.
2. **Atom decomposition** — declared `kingdom = "A"`. Per-element transcendental lowers to `accumulate(All, Expr::Sin, Op::Identity)` once Expr is extended.
3. **Shareable intermediates** — `sharing.reads = ["trig_reduce"]`, `sharing.writes = ["trig_reduce"]`. The (q, r_hi, r_lo) tuple from range reduction is the cache subject. sincos / cos / atan2pi all consume it. atan writes `atan_result` for atan2 to consume.
4. **Every parameter tunable** — three parameters per recipe: `precision: {strict, compensated, correctly_rounded}` with documented ulp budgets, `angle_unit: {radians, degrees, gradians, turns, pi_scaled}` with documented multipliers, `range_reduction: {auto, cody_waite, payne_hanek}` with documented domain validity.
5. **Every measure in every family** — comprehensive. Full circular, hyperbolic, reciprocal, pi-scaled, inverse, fused, rare. IEEE 754-2019 §9.2 sinPi/cosPi/tanPi. Gudermannian + inverse.
6. **Optimized for 2026 hardware** — n/a yet (recipes are scalar f64, no SIMD/wavefront work). The atom decomposition will deliver this when JIT lands.
7. **No vendor lock-in** — yes. Pure Rust + tambear primitives. No vendor libm, no SLEEF, no Intel SVML.
8. **No OS lock-in** — yes. Pure f64 arithmetic.
9. **Lifting to TAM** — Kingdom A; TAM schedules per-element with no boundaries.
10. **Publication-grade rigor** — references documented (Muller 2018 ch. 11 / Payne-Hanek 1983 / Cody-Waite 1980 / fdlibm / Hart 1968). Polynomial error bounds stated (sin <2.2e-17, cos <1.4e-18, both below ½ ulp). Special cases enumerated (NaN, ±∞, ±0, integer/half-integer/quarter-integer). Adversarial harness exists.

The asin polynomial audit by scout is exactly the kind of artifact Filter Test §10 asks for: every published reference (fdlibm hex constants `0x3FC9C1550E884455` etc.), every assumption (single rational P/Q on |x| ≤ 0.5), every bug found and fixed.

## Math content — what's covered

### Circular family
- **sin, cos, tan** — full radian-domain implementation. Cody-Waite reduction for `|x| < 2^20·π/2`, Payne-Hanek beyond. Remez-refit kernels.
- **asin, acos, atan, atan2** — inverse trig with proper sub-interval reduction. acos = π/2 − asin (with cancellation handling near asin(1)). atan2 quadrant logic per IEEE 754-2019.

### Hyperbolic family
- **sinh, cosh, tanh** — sinh uses expm1 path for small |x|. tanh saturation behavior: never returns exactly ±1 for finite x (`d54b749` regression).
- **asinh, acosh, atanh** — log-form identities (asinh(x) = log(x + √(x²+1)), acosh(x) = log(x + √(x²−1)), atanh(x) = ½·log((1+x)/(1−x))) with cancellation guards near 0 and near ±1.

### Reciprocal family
- **sec, csc, cot, asec, acsc, acot** — composed from circular. `inv_recip.rs` (170 lines, thin).

### Pi-scaled family (IEEE 754-2019 §9.2)
- **sinpi, cospi, tanpi** — sinpi(integer) = ±0, sinpi(n+½) = ±1, cospi(integer) = ±1, cospi(n+½) = 0, tanpi(integer) = 0, tanpi(n+½) = ±∞, tanpi(n+¼) = ±1 — all EXACT. (−1)^n sign-flip handled correctly post-`1f9d347` and `5b89ab7`.
- **asinpi, acospi, atanpi, atan2pi** — inverse pi-scaled. atan2pi exact at quarter-integer diagonals (`bffe087`).
- **sincospi** — fused.

### Rare/exotic
- **versin = 1 − cos(x)** — implemented as 2·sin²(x/2) to avoid catastrophic cancellation for small x.
- **haversin = (1 − cos(x))/2 = sin²(x/2)** — used in great-circle distance formula.
- **gudermannian gd(x) = atan(sinh(x))** — connects circular and hyperbolic. Returns nextDown(π/2) for sinh-overflowed finite x.
- **inv_gudermannian gd⁻¹(x) = atanh(sin(x)) = log(sec(x) + tan(x))** — spec'd, .rs body lives in rare_trig or hyperbolic.

### Special functions (non-trig but in same module)
- **erf, erfc** — fdlibm rational approximation, 1 ulp tightening (`69d1ea9`)
- **gamma (tgamma), lgamma** — Lanczos
- **exp, log** — range reduction + polynomial; the founding pair of the libm tier per `MATH_ROADMAP.md`'s Sweep cluster A

## Bug-fix history (genuine adversarial value)

The recent winrapids commit log shows the harness is finding real bugs. Each line is a bug that (a) escaped hand-picked tests, (b) was found by float-landmark / region-transition generators, (c) was fixed in source:

| Commit | Bug |
|---|---|
| `5b89ab7` | tanpi quarter-integer sign — `integer_flips` was wrong; `tanpi(n+0.25) = +1`, `tanpi(n+0.75) = −1` for all n (tan has period π, not 2π — n-parity does NOT flip) |
| `1f9d347` | cospi general-path missing (−1)^n sign flip |
| `bffe087` | atan2pi exact quarter-integers for `|y|=|x|` diagonal |
| `4d2b5e9` | atan 5-interval AT constants; sinh expm1 path; sinpi sign; gudermannian boundary near sinh-overflow |
| `bbda152` | asin/acos/atan/atan2 generator-backed adversarial sweeps (the asin P_S2/P_S5 fix) |
| `d54b749` | tanh saturation: never exactly ±1 for finite x |
| `fadc620` | cospi external-oracle accuracy test |
| `7fcbd1d` | softmax/log_softmax NaN propagation (not pure trig but uses trig stack) |

These are non-trivial bugs. The tanpi quarter-integer one is a published-libm error-class — most libms get this wrong because they reason about tan as having period 2π (it has period π). Catching it is exactly what publication-grade rigor looks like.

## Drift from the locked vocabulary

Items to fix during formalization, not blockers:

1. **`primitives_used` lists recipes**, not primitives. Specs cite `payne_hanek_rem_pio2` and `cody_waite_rem_pio2` under `[decomposition].primitives_used`. Under the lock these are recipes (named compositions of `frint` + `fmadd` + table lookups). They should appear in a separate `recipes_used` list, with `primitives_used` reserved for `frint`, `fmadd`, `compensated_horner`, `fsqrt`, `dd_add`, etc.
2. **Constants** are `f64::consts::PI` / `f64::consts::FRAC_PI_2` (Rust stdlib) instead of a tambear `primitives::constants` table with f64 + DoubleDouble pairs. Cosmetic but the architecture doc calls for it.
3. **`_compensated` and `_correctly_rounded` paths sometimes alias `_strict`** (rare_trig.rs is the clearest case — versin / haversin / gudermannian's `_compensated` and `_correctly_rounded` just call `_strict`). Honest stubs (no silent precision claim), but should be flagged with `#[precision(strict)]` aliases or actual DD-precision implementations.
4. **No `#[precision(...)]` attributes** in source. The architecture doc specifies these as the per-recipe lowering tag; the libm recipes pre-date the attribute design.
5. **No `#[tags(...)]` attributes** for multi-family membership. Every spec.toml has a `family = ["libm", "trigonometric", ...]` array; this needs to be lifted to the source.

## Risks / questions to surface before formalizing

- **Expr-enum extension** (substrate-correction post-aristotle, 2026-05-08): the `Expr` enum at `R:\tambear\crates\tambear\src\accumulate.rs:165-226` is **already partially populated** for trig — `Sin / Cos / Tan / Asin / Acos / Atan / Sinh / Cosh / Tanh` (lines 195-203) plus `Atan2` (line 218) plus `Sqrt / Ln / Exp` (lines 186-188). DEC-007 freezes Op; Expr is open by design. The actual completeness gap for libm-trig is narrower than I initially framed: **~22 new Expr variants** to add for full libm-trig coverage:
  - Inverse hyperbolic: `Asinh, Acosh, Atanh`
  - Special functions: `Erf, Erfc, Gamma, LGamma`
  - Reciprocal trig: `Sec, Csc, Cot, Asec, Acsc, Acot`
  - Pi-scaled: `SinPi, CosPi, TanPi, AsinPi, AcosPi, AtanPi, Atan2Pi`
  - Rare: `Versin, Haversin, Gudermannian, InvGudermannian`
  - Pathmaker decision, not a vocabulary question.
- **Sharing fingerprint shape**: spec.tomls write `trig_reduce` and `atan_result` as cache keys. Sweep 12B renamed `IntermediateTag` → `ComputedTag` (commit `7c80609`). The cache-key fingerprint per Sweep 4H is BLAKE3 of (computation_name, args, assumptions). Need to align: what's the canonical fingerprint for "(q, r_hi, r_lo) under range_reduction=cody_waite"?
- **25 of 44 spec.tomls have no dedicated .rs.** They reference the math inside one of the larger files. When formalizing into `R:\tambear\`, decide whether to (a) split into one .rs per spec.toml (uniform but more files), (b) preserve grouped .rs files (sin.rs hosting sin+cos+sincos kernels — pragmatic). I lean (b): the kernel sharing structure is the natural unit, and the spec.toml provides the per-recipe surface.
- **Reimplement-not-port philosophy** per `MATH_ROADMAP.md`: the winrapids files are reference. The right move is: copy `.spec.toml` (refresh under lock), reimplement `.rs` against locked atoms+primitives substrate, run winrapids `.rs` AS reference oracle (parity test), run mpmath at 50-digit AS gold-standard oracle (correctness test), run adversarial harness (regression test). Three oracles per recipe.

## Stance classification — per aristotle's stance-awareness finding

Per `vocabulary.md` Part II's five-stance taxonomy (pure-math / diagnostic / override-transparency / workflow / discovery), the winrapids libm recipes are a **pure-math + override-transparency hybrid**: each function ships `_strict / _compensated / _correctly_rounded` triplets. The user picks at the call site (which IS the override-transparency contract realized at function-name level).

This has structural implications for the "ready to formalize" predicate. Per stance:

| Stance | Per-recipe floor for "ready to ship" |
|---|---|
| pure-math (`_strict` only) | mpmath-50-digit oracle + adversarial harness coverage. ewma-tier rigor. |
| override-transparency (full triplet) | Triplet actually implemented (no aliasing) + override-vs-default ULP-divergence comparison test. Higher floor. |
| diagnostic (auto-strategy-selection) | Not yet designed for libm. Future: `sin(x)` could pick `_strict` vs `_correctly_rounded` based on input magnitude or tolerance hint. |
| discovery (`.discover_trig(x)`) | Not present. Future v2 follow-on. |

Per-recipe stance audit of winrapids/libm — **substrate-corrected 2026-05-08 after F4 audit pass** (mechanical grep of every `_compensated` / `_correctly_rounded` body; previous table in this section was wrong, claimed "real" where the body is `name_strict(x)` aliasing):

| Recipe | _strict | _compensated | _correctly_rounded | F12 status |
|---|---|---|---|---|
| `exp` | real | **real** (only one) | **real** (only one) | F12-compliant if spec.toml declares (currently undeclared) |
| `log` | real | **real** (only one) | **real** (only one) | F12-compliant if spec.toml declares (currently undeclared) |
| `sin / cos / sincos / sincospi` | real | undeclared alias | undeclared alias | F12 violation today |
| `tan / cot / sec / csc` | real | undeclared alias | undeclared alias | F12 violation today |
| `asin / acos` | real | undeclared alias | undeclared alias | F12 violation today |
| `atan / atan2` | real | undeclared alias | undeclared alias | F12 violation today |
| `sinh / cosh / tanh` | real | undeclared alias | undeclared alias | F12 violation today |
| `asinh / acosh / atanh` | real | undeclared alias | undeclared alias | F12 violation today |
| `asec / acsc / acot` | real | undeclared alias | undeclared alias | F12 violation today |
| `sinpi / cospi / tanpi / asinpi / acospi / atanpi / atan2pi` | real | undeclared alias | undeclared alias | F12 violation today |
| `versin / haversin / gudermannian` | real | undeclared alias | undeclared alias | F12 violation today |
| `erf / erfc / tgamma / lgamma` | real | undeclared alias | undeclared alias | F12 violation today |

**The iceberg is the whole tree, not just rare-trig.** Only `exp` and `log` have real triplets. All other libm recipes — ~50 functions across 14 files — alias compensated and correctly_rounded to strict, undeclared. Per F12 (per aristotle's deconstruction at `survey/20260508123003-aristotle/default-is-a-claim.md`), every one is a contract violation today: the spec.toml declares `precision: {strict, compensated, correctly_rounded}` as a parameter domain, the methods_template makes ULP commitments per strategy, but the implementation only realizes the strict claim.

**Fix path** (pathmaker decision per recipe):
- **(a)** Real implementations of compensated and correctly_rounded paths via DD arithmetic. ~50 functions × 2 paths = ~100 implementation pieces. Highest effort.
- **(b)** Declared aliasing in spec.toml under F12. ~50 spec.toml `[stances.override_transparency.strategy.X] state = "aliased_to"` blocks with rationale. Mechanical; ~hours rather than weeks. Cheapest.
- **(c)** Reduce parameter domain to `{strict}` for recipes where compensated/correctly_rounded never need to ship. Closes the false-claim surface; loses the extension path.

Most likely outcome: (b) for the bulk of libm with spec.toml declarations; (a) reserved for the recipes where consumers actually need the higher-precision strategy (sin/cos with their existing 80-digit-mpmath polynomial coefficients are the natural candidates — the strict polynomial is *already* near 1 ulp; the DD path would be straightforward to implement on top of the existing coefficient table).

**`exp` and `log` are the model.** Only these two recipes have real triplets. They land BEFORE the trig recipes (Phase C1-C2 per the commit log). The pattern they demonstrate (real DD path on top of refit polynomial coefficients) IS the pattern that the rest of libm-trig should adopt, in order, for any recipe where (a) is chosen over (b).

**The recipe stance metadata MUST be added at formalization time.** Pulling the trig family into `R:\tambear\` without per-strategy `state` declarations would re-import the same contract violation. The first commit of the libm-trig sweep should include the spec.toml stance blocks, even if they're all `aliased_to = "strict"` to start. Then the v2 sweeps replace `aliased_to` with `real` as the DD paths actually land.

The **rare-trig stance gap** under F12 (per aristotle's deconstruction at `survey/20260508123003-aristotle/default-is-a-claim.md`): aliasing is permitted IF declared in spec.toml with rationale. Today versin/haversin/gudermannian's compensated/correctly_rounded paths alias _strict undeclared — that's the contract violation. Fix is one of:
- **(a)** Implement real DD paths for the compensated/correctly_rounded strategies. (Higher effort; gains are negligible because the rare-trig recipes already compose sin/cos which carry their own precision contracts.)
- **(b)** Add an explicit declaration in the spec.toml with rationale, e.g.:

  ```toml
  [stances.override_transparency.strategy.compensated]
  state = "aliased_to"
  target = "strict"
  rationale = "rare-trig delegates to sin/cos which carry their own precision; \
               local DD path adds nothing at landmark inputs (versin landmarks \
               are at integer multiples of π where sin returns exact 0)."
  ```

Both (a) and (b) are F12-compliant. (b) is the cheaper fix. Pathmaker decision.

**Proposed spec.toml extension** to make stance explicit (F12-compliant per aristotle's deconstruction):

```toml
[stances.pure_math]
canonical_strategy = "strict"
oracle = "mpmath_50dps"
adversarial_harness = "winrapids/libm/adversarial.rs::sin_*"

[stances.override_transparency]
strategies = ["strict", "compensated", "correctly_rounded"]
divergence_test = "tests/sin_strategy_comparison.rs"

[stances.override_transparency.strategy.strict]
state = "real"

[stances.override_transparency.strategy.compensated]
state = "real"  # or "aliased_to" with target + rationale, or "stubbed_pending" with sweep ref

[stances.override_transparency.strategy.correctly_rounded]
state = "real"

# Per F12: every declared strategy is in {real, aliased_to, stubbed_pending}.
# Undeclared = state defaults to "real"; lint fails if impl missing.
# Declared aliasing/stubbing requires a target (for aliased_to) or sweep ref (for stubbed_pending) plus rationale.
```

**Lint contract** (mechanical F12 enforcement):
- Every strategy in `strategies` must have a `[stances.<stance>.strategy.<name>]` block.
- `state = "real"` ⇒ corresponding pub fn must exist.
- `state = "aliased_to"` ⇒ `target` must reference a strategy whose state is `real`; `rationale` must be present.
- `state = "stubbed_pending"` ⇒ `sweep_ref` must reference an open sweep/issue; `rationale` must be present.
- Missing `[stances.<stance>.strategy.<name>]` block ⇒ defaulted to `real`, fails if impl missing.

**Fidelity metrics — two-component split** (per aristotle's refinement after the F4 audit; supersedes single `claim_fidelity`):

A single fidelity number conflated two distinct things. Two recipes can share `claim_fidelity = 0.33` for very different reasons. The split:

- **`declared_fidelity`** — fraction of strategies with explicit `state` declaration. Under F12 this MUST be `1.0` for any spec.toml-compliant recipe. The transparency floor: every strategy is *labeled* (real / aliased_to / stubbed_pending). Dropping below 1.0 is the F12 violation.
- **`real_fidelity`** — fraction of strategies in `state = "real"`. The completeness story. 1.0 means full triplet shipped; 0.33 means one of three is real (the other two declared aliases or stubs). Below 1.0 is *not* a violation under F12 — it's just incomplete-but-honest.

For current pi_scaled (after F12-compliant declaration is added):
- `declared_fidelity = 1.0` (all three strategies labeled — strict as real, compensated/correctly_rounded as aliased_to)
- `real_fidelity = 0.33` (one of three is real)

For current pi_scaled BEFORE F12-compliant declaration:
- `declared_fidelity = 0.0` (no strategy explicitly labeled)
- `real_fidelity` undefined or 0.33 by inference

Same surface number, opposite contract status. The split makes the difference legible in CI.

The lint enforces `declared_fidelity = 1.0` as a hard CI failure. `real_fidelity` is a tracked metric, never a gate — it's just the completeness number. This is the engineering shape of F12: transparency is required; completeness is voluntary.

**Tolerance-derived evidence floor** (per aristotle, accepted): the spec.toml `precision` default is implicitly a tolerance commitment. `sin to ≤ 2 ulps` ⇒ `_strict` is sufficient. `sin to ≤ 1 ulp` ⇒ `_correctly_rounded` is required. The current `default = "compensated"` is a middle-budget commitment, neither the cheapest nor the strictest. Worth flagging in the writeup metadata that the default is a budget choice, not neutral.

## Pull-priority assessment

**This is the highest-leverage in-flight work.** Reasons:

1. Per `MATH_ROADMAP.md`, the first math sweep cluster is libm-core / libm-trig / libm-hyperbolic / libm-special. The work is essentially done; just unported.
2. Sweep 10 (Oracle Bridge, PLANNED) lists `log` as the leading proof-of-methodology candidate. Pulling trig brings exp + log + the trig family in as a coherent libm bundle once Sweep 10 lands.
3. Sweep 8 substrate (atoms + Op-trait redesign + accumulate executors) is already at the destination. These recipes need exactly one new Expr variant family to bottom out.
4. The adversarial harness alone is worth pulling — generator-backed float-landmark + region-transition attacks are a reusable pattern for every subsequent libm/special recipe.
5. No cross-campsite blockers. Trig recipes don't depend on JIT (they're sequential per-element math), don't depend on explicit Validity dispatcher (NaN propagation hand-coded and correct), don't depend on DEC-029 knowledge layer.

## Proposed sweep scaffold

If team-lead picks "pull trig next":

- **Sweep slug**: `libm-trig` (or `libm-elementary` if exp + log come along — recommended).
- **Scope**: 18-25 recipes. Order: exp → log → sqrt+cbrt (if not already) → sin/cos via shared `reduce_trig` + `eval_sincos` → tan → asin → acos → atan/atan2 → hyperbolic family → inv_hyperbolic → reciprocal family → pi-scaled family (sinpi/cospi/tanpi/atan2pi/asinpi/acospi/atanpi) → rare (versin/haversin/gudermannian).
- **Per-recipe deliverable**: spec.toml refreshed under lock + .rs first-principles atom-decomposed + 3 oracles (winrapids parity, mpmath 50-digit, adversarial harness) + README at `oracle/<recipe>/`.
- **First-pull payload**: probably `sin/cos/sincos` since they are (a) most-referenced by everything else, (b) hardest to get right, (c) bring `reduce_trig` shared cache as a load-bearing artifact for downstream recipes. After that, atan/atan2 (because atan2pi depends on it). Then exp/log for completeness.

## Files for pathmaker / scientist to pre-read

If pathmaker takes implementation:
- `R:\winrapids\crates\tambear\src\recipes\libm\sin.rs` — the canonical pattern
- `R:\winrapids\crates\tambear\src\recipes\libm\sin.spec.toml` — the canonical spec template
- `R:\winrapids\crates\tambear\src\recipes\libm\adversarial.rs` — harness to port
- `R:\winrapids\crates\tambear\src\recipes\libm\pi_scaled.rs` — example of exactness-contract handling

If scientist takes oracle seeding:
- `R:\winrapids\crates\tambear\src\recipes\libm\adversarial.rs::float_landmarks(lo, hi)` — input set generator
- `R:\winrapids\campsites\tambear-trig\20260414142356-implementations\scout\insights\asin-polynomial-audit.md` — example of bug-finding-grade audit
- `R:\tambear\oracle\variance\SPEC.md` — the 15-section template to clone per recipe

If aristotle takes definitional questions:
- `R:\winrapids\docs\architecture\vocabulary.md` Tier 2 — does `Expr` need a freeze decision parallel to DEC-007 for `Op`?
- The atan 4-interval-vs-5-interval reduction (commit `4d2b5e9`) is the kind of "what's the canonical algorithm" question — different libm sources give 4-interval (fdlibm), 5-interval (some textbook variants), or Padé (research-grade, lower legibility).

## Status

- Survey: complete
- Findings reported to navigator
- Drift items catalogued for formalization
- No blockers found
- Recommend: pull trig next, ahead of sweep-8 closeout if math-team can run in parallel with infrastructure-team (the recipes don't depend on JIT closing).
