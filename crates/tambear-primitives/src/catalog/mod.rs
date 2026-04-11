//! Catalog: searchable index of all primitives.
//!
//! Search by name, alias, family tag, parameter, formula fragment,
//! ecosystem name (scipy/R/MATLAB/Julia), or any combination.
//! The catalog is the API that IDE, TBS, and agents use to discover
//! what exists and how to use it.

/// Metadata for a single primitive. Search-oriented — everything
/// a consumer needs to FIND and USE this primitive.
#[derive(Debug, Clone)]
pub struct PrimitiveEntry {
    /// Canonical name (= folder name). The primary key.
    pub name: &'static str,
    /// Human description — what it computes, in one sentence.
    pub description: &'static str,
    /// Formula in plain text. Searchable.
    pub formula: &'static str,
    /// All names this primitive is known by — aliases, abbreviations,
    /// ecosystem names, colloquial names. ALL searchable.
    pub aliases: &'static [&'static str],
    /// Family tags — multi-membership. Not hierarchical.
    pub families: &'static [&'static str],
    /// Kingdom classification.
    pub kingdom: &'static str,
    /// Semiring (if this primitive is or uses one).
    pub semiring: Option<&'static str>,
    /// What other primitives does this CALL internally.
    pub composes_from: &'static [&'static str],
    /// Related primitives — alternatives, generalizations, specializations.
    pub related: &'static [&'static str],
    /// Names in other ecosystems for cross-reference search.
    pub ecosystem_names: &'static [&'static str],
    /// Parameter names (searchable — "what takes a bandwidth parameter?").
    pub param_names: &'static [&'static str],
    /// Completeness flags.
    pub has_workup: bool,
    pub has_adversarial: bool,
    pub has_oracle: bool,
}

// ═══════════════════════════════════════════════════════════════════
// The catalog
// ═══════════════════════════════════════════════════════════════════

pub static CATALOG: &[PrimitiveEntry] = &[
    PrimitiveEntry {
        name: "nan_guard",
        description: "NaN-propagating min/max and safe sorting",
        formula: "nan_min(a,b) = NaN if either is NaN, else min(a,b)",
        aliases: &["nan_min", "nan_max", "nan_propagating", "sorted_total", "sorted_finite", "has_nan", "finite_only"],
        families: &["numerical", "guard", "foundation"],
        kingdom: "A",
        semiring: None,
        composes_from: &[],
        related: &["total_cmp"],
        ecosystem_names: &["numpy.nanmin", "numpy.nanmax", "R: na.rm=TRUE"],
        param_names: &[],
        has_workup: false,
        has_adversarial: false,
        has_oracle: false,
    },
    PrimitiveEntry {
        name: "semiring",
        description: "Algebraic semiring trait + 6 standard instances",
        formula: "(S, add, mul, zero, one) where add is associative+commutative, mul distributes",
        aliases: &["Additive", "TropicalMinPlus", "TropicalMaxPlus", "LogSumExp", "Boolean", "MaxTimes",
                   "tropical", "min-plus", "max-plus", "log-semiring", "boolean semiring"],
        families: &["algebra", "foundation", "semiring", "scan"],
        kingdom: "A",
        semiring: Some("all"),
        composes_from: &["nan_guard"],
        related: &["monoid", "group", "ring", "field"],
        ecosystem_names: &[],
        param_names: &[],
        has_workup: false,
        has_adversarial: true,
        has_oracle: false,
    },
    PrimitiveEntry {
        name: "log_sum_exp",
        description: "Numerically stable log(sum(exp(x))) via max-subtraction",
        formula: "lse(x) = max(x) + ln(sum(exp(x - max(x))))",
        aliases: &["logsumexp", "lse", "log-add-exp", "log_add_exp", "softmax_denominator",
                   "log_sum_exp_pair", "pairwise_lse"],
        families: &["information_theory", "numerical", "probabilistic", "semiring"],
        kingdom: "A",
        semiring: Some("LogSumExp"),
        composes_from: &[],
        related: &["softmax", "log_softmax", "log_likelihood", "partition_function"],
        ecosystem_names: &["scipy.special.logsumexp", "torch.logsumexp", "tf.reduce_logsumexp",
                           "numpy.logaddexp", "R: matrixStats::logSumExp"],
        param_names: &["values"],
        has_workup: false,
        has_adversarial: true,
        has_oracle: false,
    },
    PrimitiveEntry {
        name: "prefix_scan",
        description: "Generic prefix scan over any semiring — THE Kingdom A operation",
        formula: "output[i] = add(input[0], ..., input[i])",
        aliases: &["scan", "cumulative", "prefix_sum", "cumsum", "running_sum", "running_min",
                   "running_max", "inclusive_scan", "exclusive_scan", "segmented_scan",
                   "parallel_scan", "blelloch", "reduce", "fold"],
        families: &["scan", "parallel", "foundation", "kingdom_a"],
        kingdom: "A",
        semiring: Some("any"),
        composes_from: &["semiring"],
        related: &["cumsum", "cumprod", "running_min", "running_max", "segmented_reduce"],
        ecosystem_names: &["numpy.cumsum", "itertools.accumulate", "std::iter::scan",
                           "thrust::inclusive_scan", "R: cumsum"],
        param_names: &["data", "semiring"],
        has_workup: false,
        has_adversarial: false,
        has_oracle: false,
    },
    PrimitiveEntry {
        name: "softmax",
        description: "Normalized exponential. Composes from log_sum_exp.",
        formula: "softmax(x)_i = exp(x_i - lse(x))",
        aliases: &["normalized_exponential", "soft_max", "log_softmax", "softargmax",
                   "boltzmann", "gibbs_distribution"],
        families: &["probabilistic", "neural", "attention", "classification"],
        kingdom: "A",
        semiring: None,
        composes_from: &["log_sum_exp"],
        related: &["log_sum_exp", "sigmoid", "hardmax", "sparsemax", "gumbel_softmax"],
        ecosystem_names: &["scipy.special.softmax", "torch.softmax", "torch.log_softmax",
                           "tf.nn.softmax", "R: exp(x)/sum(exp(x))"],
        param_names: &["x"],
        has_workup: false,
        has_adversarial: true,
        has_oracle: false,
    },
];

// ═══════════════════════════════════════════════════════════════════
// Search API
// ═══════════════════════════════════════════════════════════════════

/// Universal search — matches against name, aliases, description,
/// formula, families, ecosystem names, and parameter names.
/// Case-insensitive. Returns all matching entries ranked by relevance.
pub fn search(query: &str) -> Vec<&'static PrimitiveEntry> {
    let q = query.to_lowercase();
    let mut scored: Vec<(usize, &PrimitiveEntry)> = CATALOG.iter()
        .filter_map(|e| {
            let mut score = 0usize;
            // Exact name match — highest
            if e.name.to_lowercase() == q { score += 100; }
            // Name contains
            if e.name.to_lowercase().contains(&q) { score += 50; }
            // Alias exact match
            if e.aliases.iter().any(|a| a.to_lowercase() == q) { score += 80; }
            // Alias contains
            if e.aliases.iter().any(|a| a.to_lowercase().contains(&q)) { score += 30; }
            // Ecosystem name contains
            if e.ecosystem_names.iter().any(|n| n.to_lowercase().contains(&q)) { score += 40; }
            // Family match
            if e.families.iter().any(|f| f.to_lowercase() == q) { score += 20; }
            // Description contains
            if e.description.to_lowercase().contains(&q) { score += 10; }
            // Formula contains
            if e.formula.to_lowercase().contains(&q) { score += 10; }
            // Param name match
            if e.param_names.iter().any(|p| p.to_lowercase().contains(&q)) { score += 15; }

            if score > 0 { Some((score, e)) } else { None }
        })
        .collect();

    scored.sort_by(|a, b| b.0.cmp(&a.0)); // highest score first
    scored.into_iter().map(|(_, e)| e).collect()
}

/// Search by family tag.
pub fn by_family(family: &str) -> Vec<&'static PrimitiveEntry> {
    let f = family.to_lowercase();
    CATALOG.iter().filter(|e| e.families.iter().any(|fam| fam.to_lowercase() == f)).collect()
}

/// Search by kingdom.
pub fn by_kingdom(kingdom: &str) -> Vec<&'static PrimitiveEntry> {
    CATALOG.iter().filter(|e| e.kingdom == kingdom).collect()
}

/// Find what a primitive composes from (its dependencies).
pub fn dependencies_of(name: &str) -> Vec<&'static PrimitiveEntry> {
    let entry = CATALOG.iter().find(|e| e.name == name);
    match entry {
        Some(e) => e.composes_from.iter()
            .filter_map(|dep| CATALOG.iter().find(|c| c.name == *dep))
            .collect(),
        None => vec![],
    }
}

/// Find primitives that depend on (compose from) a given primitive.
pub fn dependents_of(name: &str) -> Vec<&'static PrimitiveEntry> {
    CATALOG.iter()
        .filter(|e| e.composes_from.contains(&name))
        .collect()
}

/// Primitives missing workups (Principle 10 gap).
pub fn missing_workups() -> Vec<&'static PrimitiveEntry> {
    CATALOG.iter().filter(|e| !e.has_workup).collect()
}

/// Primitives missing adversarial tests.
pub fn missing_adversarial() -> Vec<&'static PrimitiveEntry> {
    CATALOG.iter().filter(|e| !e.has_adversarial).collect()
}

/// Total count.
pub fn count() -> usize {
    CATALOG.len()
}

/// All family tags across all primitives.
pub fn all_families() -> Vec<&'static str> {
    let mut families: Vec<&str> = CATALOG.iter()
        .flat_map(|e| e.families.iter().copied())
        .collect();
    families.sort();
    families.dedup();
    families
}

/// All aliases across all primitives.
pub fn all_aliases() -> Vec<(&'static str, &'static str)> {
    CATALOG.iter()
        .flat_map(|e| e.aliases.iter().map(move |a| (*a, e.name)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_by_name() {
        let r = search("log_sum_exp");
        assert_eq!(r[0].name, "log_sum_exp");
    }

    #[test]
    fn search_by_alias() {
        let r = search("logsumexp");
        assert_eq!(r[0].name, "log_sum_exp");
    }

    #[test]
    fn search_by_scipy_name() {
        let r = search("scipy.special.logsumexp");
        assert_eq!(r[0].name, "log_sum_exp");
    }

    #[test]
    fn search_by_torch_name() {
        let r = search("torch.logsumexp");
        assert_eq!(r[0].name, "log_sum_exp");
    }

    #[test]
    fn search_cumsum_finds_prefix_scan() {
        let r = search("cumsum");
        assert!(r.iter().any(|e| e.name == "prefix_scan"), "cumsum should find prefix_scan");
    }

    #[test]
    fn search_tropical_finds_semiring() {
        let r = search("tropical");
        assert!(r.iter().any(|e| e.name == "semiring"));
    }

    #[test]
    fn search_boltzmann_finds_softmax() {
        let r = search("boltzmann");
        assert!(r.iter().any(|e| e.name == "softmax"));
    }

    #[test]
    fn dependencies_of_softmax() {
        let deps = dependencies_of("softmax");
        assert!(deps.iter().any(|e| e.name == "log_sum_exp"));
    }

    #[test]
    fn dependents_of_log_sum_exp() {
        let deps = dependents_of("log_sum_exp");
        assert!(deps.iter().any(|e| e.name == "softmax"));
    }

    #[test]
    fn by_family_foundation() {
        let r = by_family("foundation");
        assert!(r.len() >= 3);
    }

    #[test]
    fn all_families_sorted() {
        let fams = all_families();
        for i in 1..fams.len() {
            assert!(fams[i] > fams[i-1]);
        }
    }

    #[test]
    fn count_is_five() {
        assert_eq!(count(), 5);
    }
}
