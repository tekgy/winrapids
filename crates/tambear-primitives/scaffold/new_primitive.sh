#!/bin/bash
# Generate a new primitive folder with all files pre-populated.
# Usage: ./scaffold/new_primitive.sh <name> "<description>" "<formula>"
#
# Example:
#   ./scaffold/new_primitive.sh mean_cubic "Cubic mean: (Σxᵢ³/n)^(1/3)" "(sum_cubed / count)^(1/3)"

set -e

NAME="$1"
DESC="$2"
FORMULA="$3"

if [ -z "$NAME" ] || [ -z "$DESC" ]; then
    echo "Usage: $0 <name> <description> [formula]"
    echo "Example: $0 mean_cubic 'Cubic mean' '(sum_cubed/count)^(1/3)'"
    exit 1
fi

DIR="src/primitives/$NAME"
if [ -d "$DIR" ]; then
    echo "ERROR: $DIR already exists"
    exit 1
fi

mkdir -p "$DIR/tests"

# --- mod.rs ---
cat > "$DIR/mod.rs" << RUST
//! $DESC
//!
//! # Formula
//! \`\`\`text
//! ${FORMULA:-TODO: write formula}
//! \`\`\`
//!
//! # Recipe
//! TODO: decompose into accumulate+gather steps.
//! See ARCHITECTURE.md for the pattern.
//!
//! # Kingdom
//! TODO: classify (A/B/C/D) and justify.

/// $DESC
///
/// Returns NaN for empty input or if any value is NaN.
pub fn $NAME(data: &[f64]) -> f64 {
    if data.is_empty() { return f64::NAN; }
    if crate::nan_guard::has_nan(data) { return f64::NAN; }
    todo!("implement $NAME")
}

#[cfg(test)]
mod tests;
RUST

# --- tests/mod.rs ---
cat > "$DIR/tests/mod.rs" << RUST
use super::*;

#[test]
fn basic() {
    todo!("write basic test for $NAME")
}

#[test]
fn empty_is_nan() {
    assert!($NAME(&[]).is_nan());
}

#[test]
fn nan_propagates() {
    assert!($NAME(&[1.0, f64::NAN, 3.0]).is_nan());
}
RUST

# --- params.toml ---
cat > "$DIR/params.toml" << TOML
[primitive]
name = "$NAME"
family = ["TODO"]
kingdom = "TODO"
description = "$DESC"

[[params]]
name = "data"
type = "&[f64]"
required = true
description = "Input values"

[returns]
type = "f64"
description = "TODO"

[identity]
value = "TODO"
rationale = "TODO"

[degenerate]
value = "NaN"
condition = "empty input or NaN in data"

[references]
# scipy = ""
# r = ""
# matlab = ""
TOML

# --- README.md ---
cat > "$DIR/README.md" << MD
# $NAME

$DESC

## Formula

\`\`\`
${FORMULA:-TODO}
\`\`\`

## When to use

TODO

## Composes with

TODO

## Recipe (accumulate+gather decomposition)

\`\`\`
TODO: list the accumulate and gather steps
\`\`\`
MD

echo "Created $DIR/"
echo "  mod.rs        — implementation (TODO)"
echo "  tests/mod.rs  — tests (TODO)"
echo "  params.toml   — metadata (TODO)"
echo "  README.md     — docs (TODO)"
echo ""
echo "Next steps:"
echo "  1. Implement the function in mod.rs"
echo "  2. Write the recipe (accumulate+gather decomposition)"
echo "  3. Add tests"
echo "  4. Add 'pub mod $NAME;' to src/primitives/mod.rs"
echo "  5. Add pub use re-export"
echo "  6. Add catalog entry"
echo "  7. cargo test -p tambear-primitives && cargo test -p tambear-tam"
