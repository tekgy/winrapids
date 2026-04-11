//! Catalog: the index of all primitives in the flat catalog.
//!
//! Built at compile time from the primitives module tree.
//! At runtime, provides search, listing, and metadata queries.
//!
//! This is what the IDE, TBS compiler, and agents call to discover
//! what primitives exist and how they compose.

/// Metadata for a single primitive, derived from params.toml.
#[derive(Debug, Clone)]
pub struct PrimitiveEntry {
    /// Canonical name (folder name). e.g. "log_sum_exp"
    pub name: &'static str,
    /// Human description.
    pub description: &'static str,
    /// Family tags for multi-membership search.
    pub families: &'static [&'static str],
    /// Kingdom classification (A, B, C, D).
    pub kingdom: &'static str,
    /// What semiring this primitive belongs to (if applicable).
    pub semiring: Option<&'static str>,
    /// What this primitive produces for TamSession sharing.
    pub sharing_tag: Option<&'static str>,
    /// Names of primitives that consume this one's output.
    pub consumers: &'static [&'static str],
    /// Whether this primitive has a complete workup (Principle 10).
    pub has_workup: bool,
    /// Whether adversarial tests exist.
    pub has_adversarial: bool,
    /// Whether oracle validation exists.
    pub has_oracle: bool,
}

/// The compile-time catalog. Every primitive registers here.
/// Adding a primitive = adding one entry to this array.
pub static CATALOG: &[PrimitiveEntry] = &[
    PrimitiveEntry {
        name: "nan_guard",
        description: "NaN-propagating comparisons and guards (nan_min, nan_max, sorted_total, sorted_finite)",
        families: &["numerical", "guard", "foundation"],
        kingdom: "A",
        semiring: None,
        sharing_tag: None,
        consumers: &["semiring", "log_sum_exp", "prefix_scan", "every_primitive_that_handles_raw_input"],
        has_workup: false,
        has_adversarial: false,
        has_oracle: false,
    },
    PrimitiveEntry {
        name: "semiring",
        description: "Semiring trait + 6 instances (Additive, TropicalMinPlus, TropicalMaxPlus, LogSumExp, Boolean, MaxTimes)",
        families: &["algebra", "foundation", "semiring"],
        kingdom: "A",
        semiring: Some("all"),
        sharing_tag: None,
        consumers: &["prefix_scan", "hmm_forward", "viterbi", "shortest_path", "attention"],
        has_workup: false,
        has_adversarial: true,
        has_oracle: false,
    },
    PrimitiveEntry {
        name: "log_sum_exp",
        description: "Numerically stable log(sum(exp(x))) via max-subtraction trick",
        families: &["information_theory", "numerical", "probabilistic", "semiring"],
        kingdom: "A",
        semiring: Some("LogSumExp"),
        sharing_tag: Some("LogSumExpResult"),
        consumers: &["hmm_forward", "softmax", "log_softmax", "mixture_log_likelihood", "bayes_model_evidence", "attention_weights"],
        has_workup: false,
        has_adversarial: true,
        has_oracle: false,
    },
    PrimitiveEntry {
        name: "prefix_scan",
        description: "Generic prefix scan over any semiring (inclusive, exclusive, segmented, reduce)",
        families: &["scan", "parallel", "foundation", "kingdom_a"],
        kingdom: "A",
        semiring: Some("any"),
        sharing_tag: None,
        consumers: &["cumsum", "running_min", "running_max", "hmm_forward", "viterbi", "ema", "garch_filter"],
        has_workup: false,
        has_adversarial: false,
        has_oracle: false,
    },
    PrimitiveEntry {
        name: "softmax",
        description: "Normalized exponential distribution. Composes from log_sum_exp.",
        families: &["probabilistic", "neural", "attention"],
        kingdom: "A",
        semiring: None,
        sharing_tag: None,
        consumers: &["attention", "classification", "mixture_weights", "boltzmann"],
        has_workup: false,
        has_adversarial: true,
        has_oracle: false,
    },
];

/// Search the catalog by name substring (case-insensitive).
pub fn search(query: &str) -> Vec<&'static PrimitiveEntry> {
    let q = query.to_lowercase();
    CATALOG.iter().filter(|e| e.name.to_lowercase().contains(&q)).collect()
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

/// All primitives that consume a given primitive's output.
pub fn consumers_of(name: &str) -> Vec<&'static PrimitiveEntry> {
    // Find primitives whose consumers list includes `name`,
    // OR whose name appears in `name`'s consumers list.
    let n = name.to_lowercase();
    CATALOG.iter().filter(|e| {
        e.consumers.iter().any(|c| c.to_lowercase() == n)
    }).collect()
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

/// List all family tags across all primitives.
pub fn all_families() -> Vec<&'static str> {
    let mut families: Vec<&str> = CATALOG.iter()
        .flat_map(|e| e.families.iter().copied())
        .collect();
    families.sort();
    families.dedup();
    families
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_finds_log_sum_exp() {
        let results = search("log_sum");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].name, "log_sum_exp");
    }

    #[test]
    fn by_family_foundation() {
        let results = by_family("foundation");
        assert!(results.len() >= 2); // nan_guard, semiring, prefix_scan
    }

    #[test]
    fn by_kingdom_a() {
        let results = by_kingdom("A");
        assert_eq!(results.len(), CATALOG.len()); // all are Kingdom A currently
    }

    #[test]
    fn consumers_of_log_sum_exp() {
        // log_sum_exp's entry lists "softmax" as a consumer
        let entry = search("log_sum_exp");
        assert!(entry[0].consumers.contains(&"softmax"), "softmax should be in log_sum_exp consumers");

        // consumers_of finds entries that HAVE "softmax" in their consumers list
        let results = consumers_of("softmax");
        let names: Vec<&str> = results.iter().map(|e| e.name).collect();
        assert!(names.contains(&"log_sum_exp"), "log_sum_exp should list softmax as consumer");
    }

    #[test]
    fn missing_workups_is_all() {
        // None have workups yet
        assert_eq!(missing_workups().len(), CATALOG.len());
    }

    #[test]
    fn all_families_deduped() {
        let fams = all_families();
        for i in 1..fams.len() {
            assert!(fams[i] > fams[i-1], "not sorted/deduped");
        }
    }

    #[test]
    fn count_matches() {
        assert_eq!(count(), 5);
    }
}
