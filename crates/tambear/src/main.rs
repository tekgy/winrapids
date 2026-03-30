//! Tambear integration test / sanity check.
//!
//! Run with: cargo run --bin tambear-test
//!
//! This file is the integration point — as scout, naturalist, and observer
//! complete their pieces, this test will be extended to run end-to-end.

fn main() {
    println!("tambear: sort-free DataFrame engine");
    println!("Tam doesn't sort. Tam knows.");
    println!();
    println!("Status:");
    println!("  [ ] HashScatterEngine::groupby  (scout)");
    println!("  [ ] GroupIndex::build           (scout)");
    println!("  [ ] RefCenteredStatsEngine      (naturalist)");
    println!("  [ ] AffineOp in winrapids-scan  (naturalist)");
    println!("  [ ] .tb format spec             (observer)");
    println!("  [ ] Python API                  (observer)");
}
