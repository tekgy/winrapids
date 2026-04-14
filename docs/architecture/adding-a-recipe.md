# Adding a New Recipe to Tambear

This is the playbook for adding a new mathematical recipe to tambear,
distilled from the first three `.spec.toml` migrations (exp, correlation_matrix,
factor_analysis) and the five Phase-C libm implementations (exp, log, sin/cos,
erf/erfc, gamma).

## The five-artifact pattern

Every production recipe ships with five artifacts:

1. **The Rust implementation** — pure math, three precision strategies, in
   `crates/tambear/src/recipes/<family>/<name>.rs`.
2. **The `.spec.toml` schema** — single source of truth for parameters,
   outputs, defaults, metadata. Co-located next to the `.rs`.
3. **Oracle-backed tests** — in the `.rs` under `#[cfg(test)]`, asserting
   mathematical correctness against a reference (mpmath, std::f64::*, R,
   scipy — whichever applies).
4. **Adversarial tests** — region boundaries, near-poles, NaN, ±0, ±∞,
   extreme values. In `recipes/libm/adversarial.rs` for libm; colocated
   otherwise.
5. **Schema registration** — one `match` arm in
   `recipes/pipelines/schema.rs` that routes the recipe name to the
   toml-loading `OnceLock` accessor.

## Step-by-step

### 1. Place the recipe in the family tree

```
crates/tambear/src/recipes/
├── libm/                    # elementary transcendentals
├── statistics/              # correlation, factor analysis, descriptive
├── multivariate/            # (future) PCA, ICA, MDS
├── time_series/             # (future) ARIMA, spectral, state-space
└── pipelines/               # orchestration layer — don't put recipes here
```

If your family doesn't exist, create it as a subdirectory under `recipes/`
with its own `mod.rs` that documents the family.

### 2. Write the `.spec.toml`

Minimal skeleton — flesh out the sections as applicable:

```toml
[recipe]
name = "my_recipe"
layer = "recipe"  # or "expr", "primitive", "atom"
family = ["category_1", "category_2"]  # flat tags, not hierarchical
description = "One-line human-readable summary."

long_description = """
Multi-paragraph prose explaining what the recipe does, when to use it,
what its assumptions are, and what's hard about computing it. This shows
up in the IDE side-panel and the generated writeup.
"""

[[recipe.references]]
citation = "Author Year"
kind = "paper"        # paper | book | implementation | textbook | standard
note = "What this reference contributes."

[decomposition]
primitives_used = ["fmadd", "compensated_horner"]  # which primitives it reaches
kingdom = "A"  # A | B | C | D — see CLAUDE.md

[sharing]
reads = []     # IntermediateTag names this recipe consumes from TamSession
writes = []    # IntermediateTag names it registers for downstream consumers

# ── Parameters (0 or more) ──
[[parameters]]
key = "method"           # TBS name
display_name = "Method"  # IDE label
description = "Detailed prose help text."
kind = "method"          # bool | int | float | string | method
advanced = false         # false = main UI, true = advanced section

[parameters.default]
# Any subset of the four dimensions:
using = "pearson"              # bare value, coerced by `kind`
autodiscover = "probe_name"    # name of an algorithm_properties probe
# sweep = [...]                # array of bare values
# [parameters.default.superposition]
# values = [...]
# combiner = { kind = "keep_all" }

[parameters.domain]
kind = "enum"
values = ["pearson", "spearman", "kendall"]
# or: kind = "range" with min/max
# or: kind = "int_range" with min/max
# or: kind = "free_string"

# ── Outputs (1 or more) ──
[[outputs]]
semantic_name = "result"
description = "What this column contains."
dtype = "f64"            # f32 | f64 | i32 | i64 | bool | string
has_v_column = true      # true if recipe emits a V-column confidence signal

[outputs.shape]
kind = "vector"          # scalar | vector | matrix | table
length_from = "input_rows"  # or: length = 42, or: length_from = "choice:n_factors"

# ── Writeup template (optional) ──
[writeup]
methods_template = """
Prose template with {placeholder} fields filled in from parameter values
and outputs at manuscript-generation time.
"""
```

### 3. Write the Rust implementation

Follow the three-strategy pattern (`exp`, `log`, `sin` in `recipes/libm/`
are canonical examples):

```rust
pub fn my_recipe_strict(...) -> ... { /* fast path, ≤ 4 ulps target */ }
pub fn my_recipe_compensated(...) -> ... { /* mid path, ≤ 2 ulps target */ }
pub fn my_recipe_correctly_rounded(...) -> ... { /* gold path, ≤ 1 ulp target */ }
```

Handle all special cases at the top: NaN, ±∞, ±0, domain violations.
Use `primitives::hardware::*` for arithmetic, `primitives::compensated::*`
for EFTs and Horner, `primitives::double_double::*` for ~106-bit working
precision, `primitives::specialist::*` for Kulisch-level gold standards.

### 4. Register the schema

In `crates/tambear/src/recipes/pipelines/schema.rs`:

```rust
// Add the match arm:
pub fn schema_for(recipe: &RecipeRef) -> Option<&'static RecipeSchema> {
    match recipe {
        RecipeRef::Recipe(name) => match name.as_str() {
            // ... existing arms ...
            "my_recipe" => Some(my_recipe_schema_from_toml()),
            _ => None,
        },
        // ...
    }
}

// Add the OnceLock cell + loader:
static MY_RECIPE_SCHEMA_CELL: OnceLock<&'static RecipeSchema> = OnceLock::new();

fn my_recipe_schema_from_toml() -> &'static RecipeSchema {
    MY_RECIPE_SCHEMA_CELL.get_or_init(|| {
        let toml_str = include_str!("../<family>/my_recipe.spec.toml");
        let owned = toml_schema::parse_spec_toml(toml_str)
            .expect("my_recipe.spec.toml must parse");
        Box::leak(Box::new(leak_into_static(&owned)))
    })
}
```

That's the entire bridge from `.spec.toml` to the pipeline layer.

### 5. Write tests

In your recipe's `.rs` file under `#[cfg(test)]`:

```rust
#[test]
fn recipe_boundary_semantics() {
    // NaN, ±∞, ±0, domain violations — each case explicit
}

#[test]
fn recipe_known_values() {
    // my_recipe(known_input) matches mathematical-closed-form exact value
}

#[test]
fn recipe_strict_within_budget() {
    check_recipe(my_recipe_strict, "strict", 4);  // or whatever ulp budget
}

// ... same for compensated and correctly_rounded
```

If the recipe is a libm function, also add it to
`recipes/libm/adversarial.rs` — extend the sample generators and the
`libm_accuracy_report` test to cover the new function.

### 6. Verify

```bash
cd crates/tambear
cargo test --lib recipes::<family>::<name>    # your recipe's tests
cargo test --lib recipes::pipelines           # schema + invoke integration
cargo test --lib                              # full regression
```

All green, no regressions — commit.

## The compose-default pattern

When writing the `[parameters.default]` block, remember that the four
dimensions are **composable**, not mutually exclusive:

```toml
[parameters.default]
using = "pearson"
autodiscover = "normality_probe"
```

This means: by default, use Pearson, AND also run the normality probe.
If the user overrides `using = "spearman"`, the probe still runs — the
effective binding keeps both dimensions active. The user sees side-by-
side: their choice (Spearman) and what tambear would have recommended
(Spearman, because the probe fired).

If you want "user must pick, tambear won't decide," leave `using`
unspecified:

```toml
[parameters.default]
# no `using` — user must provide or the probe must decide
autodiscover = "my_probe"
```

If you want a hard default with no advice, leave `autodiscover` unspecified:

```toml
[parameters.default]
using = "default_method"
# no `autodiscover` — tambear doesn't run an advisory probe
```

## The advice flow

When a recipe runs with a user override that the probe would have
disagreed with, the bridge layer emits an `Advice` entry in the
`InvokeResponse.advice` array. The IDE renders these as
"Tambear recommends… / You picked…" panels per step.

The advice source is `primitives::oracle::algorithm_properties::format_advice()`
— one catalog, consumed by tests, by lints, by the IDE bridge.

## Worked examples

- **exp** (`recipes/libm/exp.spec.toml` + `exp.rs`) — one parameter
  (precision), one output, simple. The pilot.
- **correlation_matrix** (`recipes/statistics/correlation_matrix.spec.toml`)
  — three parameters, three outputs. Demonstrates compose-default:
  method has BOTH `using = "pearson"` AND `autodiscover = "normality_probe"`.
- **factor_analysis** (`recipes/statistics/factor_analysis.spec.toml`)
  — five parameters, six outputs. Demonstrates autodiscover-only
  default: `n_factors` has NO `using` value, so tambear decides at
  runtime via `parallel_analysis_or_kaiser`.

## What NOT to do

- **Don't hand-write a `RecipeSchema` const.** The three pilot recipes
  have migrated; all new recipes go straight to `.spec.toml`.
- **Don't wrap a vendor library.** Every recipe is from-first-principles
  Rust. See the Tambear Contract in `CLAUDE.md` for the full rule.
- **Don't skip the three strategies.** Every recipe has strict,
  compensated, and correctly_rounded. If one is obviously redundant
  (e.g., a pure integer recipe with no rounding), document why.
- **Don't put implementation logic in the `.spec.toml`.** The toml is
  metadata — what the IDE renders, what the runtime reads for defaults.
  The math stays in the `.rs`.
- **Don't forget to add references.** Papers, textbooks, implementations
  — the `[[recipe.references]]` block feeds the generated writeup's
  citation list.

## Checklist before shipping

- [ ] `.spec.toml` at `recipes/<family>/<name>.spec.toml`
- [ ] `.rs` with three-strategy entry points at same path
- [ ] `schema.rs` match arm + `OnceLock` cell + loader fn
- [ ] Tests: boundary semantics, known values, per-strategy ulp budget
- [ ] Adversarial coverage (libm: extend adversarial.rs; other: co-locate)
- [ ] `[[recipe.references]]` entries for any papers/books cited
- [ ] `long_description` fleshes out the "when to use" and "when not to"
- [ ] All parameter defaults thought through (using? autodiscover? both?)
- [ ] Domains specified for every parameter (enum / range / int_range / free_string)
- [ ] `cargo test --lib` all green
- [ ] Commit message describes the recipe's structural contribution, not
      just its name
