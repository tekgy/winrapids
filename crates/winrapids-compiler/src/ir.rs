//! Arena-based IR with built-in CSE.
//!
//! The Polars pattern: NodeId = u32 index into a flat Vec.
//! CSE is free — HashMap<identity, NodeId> deduplicates on insert.
//!
//! Two nodes with the same identity are the same computation.
//! Identity = BLAKE3(op, input_identities, canonical_params).

use std::collections::HashMap;

/// Index into the Arena's node vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// The 9 primitives + Data (input leaf).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum PrimitiveOp {
    /// Raw input data (leaf node, no computation).
    Data,
    /// Parallel prefix scan (cumsum, Welford, EWM, etc.)
    Scan,
    /// Sort
    Sort,
    /// Full reduction (sum, mean, etc.)
    Reduce,
    /// Tiled reduction (per-tile aggregates)
    TiledReduce,
    /// Scatter (write to arbitrary indices)
    Scatter,
    /// Gather (read from arbitrary indices)
    Gather,
    /// Binary search (sorted array lookup)
    Search,
    /// Stream compaction (filter)
    Compact,
    /// Fused element-wise expression (one kernel launch)
    FusedExpr,
}

/// One node in the primitive IR DAG.
#[derive(Clone, Debug)]
pub struct Node {
    /// Which primitive operation.
    pub op: PrimitiveOp,
    /// Input node identities (not NodeIds — identities survive CSE).
    pub input_identities: Vec<String>,
    /// Canonical parameters sorted by key. e.g. [("agg","add"), ("window","20")]
    pub params: Vec<(String, String)>,
    /// Human-readable output name (e.g. "cs", "out").
    pub output_name: String,
    /// BLAKE3-based identity hash (hex, 24 chars). Two nodes with same identity
    /// are the same computation — this IS the CSE key.
    pub identity: String,
}

/// Compute the identity hash for a node.
/// Identity = BLAKE3(op_debug, input_identities, sorted_params) truncated to 12 bytes hex.
fn compute_identity(op: &PrimitiveOp, inputs: &[String], params: &[(String, String)]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(format!("{:?}", op).as_bytes());
    for inp in inputs {
        hasher.update(inp.as_bytes());
        hasher.update(b"|");
    }
    for (k, v) in params {
        hasher.update(k.as_bytes());
        hasher.update(b"=");
        hasher.update(v.as_bytes());
        hasher.update(b",");
    }
    let hash = hasher.finalize();
    // 12 hex chars = 6 bytes, matches E04's MD5[:12]
    hex::encode(&hash.as_bytes()[..6])
}

/// The IR arena. Nodes live here. CSE is built in.
#[derive(Debug)]
pub struct Arena {
    /// All nodes, indexed by NodeId.
    pub nodes: Vec<Node>,
    /// Identity → NodeId for CSE deduplication.
    seen: HashMap<String, NodeId>,
}

impl Arena {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            seen: HashMap::new(),
        }
    }

    /// Add a node, or return the existing NodeId if a node with the same identity exists.
    /// This IS CSE — deduplication by identity hash.
    pub fn add_or_dedup(
        &mut self,
        op: PrimitiveOp,
        input_identities: Vec<String>,
        params: Vec<(String, String)>,
        output_name: String,
    ) -> NodeId {
        let identity = compute_identity(&op, &input_identities, &params);

        if let Some(&existing) = self.seen.get(&identity) {
            return existing;
        }

        let id = NodeId(self.nodes.len() as u32);
        self.nodes.push(Node {
            op,
            input_identities,
            params,
            output_name,
            identity: identity.clone(),
        });
        self.seen.insert(identity, id);
        id
    }

    /// Look up a node by its identity hash.
    pub fn get_by_identity(&self, identity: &str) -> Option<NodeId> {
        self.seen.get(identity).copied()
    }

    /// Get a node by its NodeId.
    pub fn get(&self, id: NodeId) -> &Node {
        &self.nodes[id.0 as usize]
    }

    /// Number of unique nodes (after CSE).
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Is the arena empty?
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

// We need hex encoding. Inline a minimal hex encoder to avoid adding a dep.
mod hex {
    pub fn encode(bytes: &[u8]) -> String {
        let mut s = String::with_capacity(bytes.len() * 2);
        for &b in bytes {
            s.push_str(&format!("{:02x}", b));
        }
        s
    }
}
