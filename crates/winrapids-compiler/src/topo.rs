//! Topological sort (Kahn's algorithm).
//!
//! Deterministic: tie-breaks by identity hash for reproducible ordering.

use std::collections::{BinaryHeap, HashMap, HashSet};
use std::cmp::Reverse;

/// Kahn's algorithm topological sort.
///
/// Input: nodes (identity strings) and their dependencies (identity → set of dependency identities).
/// Output: identities in topological order (dependencies before dependents).
///
/// Tie-breaking: lexicographic on identity hash (deterministic).
/// Panics if the graph has cycles (should never happen in a well-formed primitive DAG).
pub fn topo_sort(
    nodes: &[String],
    dep_graph: &HashMap<String, HashSet<String>>,
) -> Vec<String> {
    // Compute in-degree
    let mut in_degree: HashMap<&str, usize> = HashMap::new();
    for node in nodes {
        in_degree.entry(node).or_insert(0);
    }
    for (node, deps) in dep_graph {
        let _ = in_degree.entry(node).or_insert(0);
        for dep in deps {
            let _ = in_degree.entry(dep).or_insert(0); // ensure dep exists
        }
        // in_degree[node] = number of deps that are in our node set
        let count = deps.iter().filter(|d| in_degree.contains_key(d.as_str())).count();
        *in_degree.get_mut(node.as_str()).unwrap() = count;
    }

    // Min-heap for deterministic tie-breaking (Reverse for min-heap)
    let mut ready: BinaryHeap<Reverse<&str>> = BinaryHeap::new();
    for (&node, &deg) in &in_degree {
        if deg == 0 {
            ready.push(Reverse(node));
        }
    }

    let mut result = Vec::with_capacity(nodes.len());

    while let Some(Reverse(node)) = ready.pop() {
        result.push(node.to_string());

        // Find all nodes that depend on `node` and decrement their in-degree
        for (other, deps) in dep_graph {
            if deps.contains(node) {
                let deg = in_degree.get_mut(other.as_str()).unwrap();
                *deg -= 1;
                if *deg == 0 {
                    ready.push(Reverse(other.as_str()));
                }
            }
        }
    }

    assert_eq!(
        result.len(), nodes.len(),
        "Cycle detected in primitive DAG: sorted {} of {} nodes",
        result.len(), nodes.len()
    );

    result
}
