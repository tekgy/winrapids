//! Provenance hash computation.
//!
//! A buffer's provenance is its computational identity:
//! hash(input_provenances + computation_identity).
//!
//! Same inputs + same computation = same provenance = reuse.
//! This is the CSE identity from the compiler, extended to persist
//! across plans and sessions via the store.
//!
//! Uses BLAKE3 truncated to 128 bits (16 bytes). Fast (>1 GB/s),
//! collision-resistant, deterministic.

/// Compute provenance from input provenances and computation identity.
///
/// Input order matters: `f(a, b)` and `f(b, a)` have different provenances
/// unless the computation itself is commutative (which is encoded in the
/// computation_id string).
///
/// # Examples
/// ```
/// use winrapids_store::provenance_hash;
///
/// let input_a = winrapids_store::data_provenance("price:AAPL:2026-03-30");
/// let prov = provenance_hash(&[input_a], "scan:add:w=20");
/// assert_eq!(prov.len(), 16);
/// ```
pub fn provenance_hash(
    input_provenances: &[[u8; 16]],
    computation_id: &str,
) -> [u8; 16] {
    let mut hasher = blake3::Hasher::new();
    for p in input_provenances {
        hasher.update(p);
    }
    hasher.update(computation_id.as_bytes());
    let hash = hasher.finalize();
    let mut result = [0u8; 16];
    result.copy_from_slice(&hash.as_bytes()[..16]);
    result
}

/// Compute provenance for raw input data (leaf provenance).
///
/// The data identity string should uniquely identify the data:
/// e.g., "price:AAPL:2026-03-30:1s" for ticker AAPL, date, 1-second cadence.
///
/// When the raw data changes (new ticks arrive), the identity string changes,
/// which cascades through all dependent provenances — automatically
/// invalidating stale computations.
pub fn data_provenance(data_identity: &str) -> [u8; 16] {
    let hash = blake3::hash(data_identity.as_bytes());
    let mut result = [0u8; 16];
    result.copy_from_slice(&hash.as_bytes()[..16]);
    result
}

/// Format a provenance hash as a short hex string (first 8 chars).
/// For human-readable logging and debugging.
pub fn prov_hex(provenance: &[u8; 16]) -> String {
    provenance.iter()
        .take(4)
        .map(|b| format!("{:02x}", b))
        .collect()
}
