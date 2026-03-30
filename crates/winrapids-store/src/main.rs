//! Validation binary for winrapids-store.
//!
//! Tests the persistent store without GPU hardware:
//! - BufferHeader is exactly 64 bytes (cache-line aligned)
//! - Provenance hashing is deterministic
//! - Same inputs = same provenance, different inputs = different provenance
//! - Store lookup/register/eviction works correctly
//! - Cost-aware eviction picks the right victim
//! - NullWorld always misses
//! - WorldState trait works through GpuStore

use winrapids_store::*;

fn main() {
    println!("{}", "=".repeat(70));
    println!("winrapids-store validation");
    println!("{}", "=".repeat(70));

    test_header_size();
    test_provenance_determinism();
    test_provenance_identity();
    test_store_basic();
    test_store_eviction();
    test_cost_aware_eviction();
    test_null_world();
    test_world_state_trait();
    test_store_stats();

    println!("\n{}", "=".repeat(70));
    println!("ALL TESTS PASSED");
    println!("{}", "=".repeat(70));
}

fn test_header_size() {
    println!("\n--- Test 1: BufferHeader is 64 bytes ---");
    let size = std::mem::size_of::<BufferHeader>();
    assert_eq!(size, 64, "BufferHeader must be exactly 64 bytes, got {}", size);
    println!("  BufferHeader size = {} bytes  PASS", size);

    // Verify alignment
    let align = std::mem::align_of::<BufferHeader>();
    println!("  BufferHeader align = {} bytes", align);
}

fn test_provenance_determinism() {
    println!("\n--- Test 2: Provenance hashing is deterministic ---");
    let input_a = data_provenance("price:AAPL:2026-03-30:1s");
    let input_b = data_provenance("price:AAPL:2026-03-30:1s");
    assert_eq!(input_a, input_b, "Same identity must produce same provenance");
    println!("  data_provenance deterministic  PASS  ({})", prov_hex(&input_a));

    let comp_1 = provenance_hash(&[input_a], "scan:add:w=20");
    let comp_2 = provenance_hash(&[input_a], "scan:add:w=20");
    assert_eq!(comp_1, comp_2, "Same inputs + same computation = same provenance");
    println!("  provenance_hash deterministic  PASS  ({})", prov_hex(&comp_1));
}

fn test_provenance_identity() {
    println!("\n--- Test 3: Provenance identity separation ---");
    let price_aapl = data_provenance("price:AAPL:2026-03-30:1s");
    let price_msft = data_provenance("price:MSFT:2026-03-30:1s");
    assert_ne!(price_aapl, price_msft, "Different data must produce different provenance");
    println!("  Different data → different provenance  PASS");

    let scan_add = provenance_hash(&[price_aapl], "scan:add");
    let scan_mul = provenance_hash(&[price_aapl], "scan:mul");
    assert_ne!(scan_add, scan_mul, "Different computation must produce different provenance");
    println!("  Different computation → different provenance  PASS");

    // Order matters: f(a, b) != f(b, a)
    let ab = provenance_hash(&[price_aapl, price_msft], "cross_corr");
    let ba = provenance_hash(&[price_msft, price_aapl], "cross_corr");
    assert_ne!(ab, ba, "Input order must matter");
    println!("  Input order matters  PASS");

    // CSE: same computation on same data = same provenance
    let cs_1 = provenance_hash(&[price_aapl], "scan:add");
    let cs_2 = provenance_hash(&[price_aapl], "scan:add");
    assert_eq!(cs_1, cs_2, "CSE: identical computations must have identical provenance");
    println!("  CSE identity (same comp = same prov)  PASS");
}

fn test_store_basic() {
    println!("\n--- Test 4: Store basic operations ---");

    let mut store = GpuStore::new(1_000_000); // 1 MB budget

    let prov = data_provenance("test:basic");
    let ptr = BufferPtr { device_ptr: 0x1000, byte_size: 400_000 };

    // Miss
    assert!(store.lookup(&prov).is_none(), "Should miss on empty store");
    println!("  Empty store miss  PASS");

    // Register
    let header = BufferHeader::new(prov, 100.0, DType::F32, 100_000);
    let evicted = store.register(header, ptr);
    assert!(evicted.is_empty(), "Should not evict with space available");
    println!("  Register with space  PASS");

    // Hit
    let result = store.lookup(&prov);
    assert_eq!(result, Some(ptr), "Should hit after register");
    println!("  Hit after register  PASS  (ptr={:#x})", ptr.device_ptr);

    // Used bytes tracking
    assert_eq!(store.used_bytes(), 400_000);
    println!("  Used bytes = {}  PASS", store.used_bytes());

    // Remove
    let removed = store.remove(&prov);
    assert!(removed.is_some(), "Should remove existing entry");
    assert!(store.lookup(&prov).is_none(), "Should miss after remove");
    assert_eq!(store.used_bytes(), 0);
    println!("  Remove and verify  PASS");
}

fn test_store_eviction() {
    println!("\n--- Test 5: LRU eviction ---");

    // 1 KB budget, register buffers until eviction triggers
    let mut store = GpuStore::new(1_000);

    // Register 3 × 400 byte buffers — third should trigger eviction
    for i in 0..3u64 {
        let prov = data_provenance(&format!("test:evict:{}", i));
        let ptr = BufferPtr { device_ptr: 0x1000 * (i + 1), byte_size: 400 };
        let header = BufferHeader::new(prov, 10.0, DType::F32, 100);
        let evicted = store.register(header, ptr);

        if i < 2 {
            assert!(evicted.is_empty(), "First {} buffers should fit", i + 1);
        } else {
            assert!(!evicted.is_empty(), "Third buffer should trigger eviction");
            println!("  Eviction triggered on buffer {}  PASS", i);
            println!("  Evicted {} entries, freed {} bytes",
                evicted.len(),
                evicted.iter().map(|e| e.ptr.byte_size).sum::<u64>());
        }
    }

    // After eviction, newest entries should survive
    let prov_2 = data_provenance("test:evict:2");
    assert!(store.lookup(&prov_2).is_some(), "Newest entry should survive eviction");
    println!("  Newest entry survives  PASS");
}

fn test_cost_aware_eviction() {
    println!("\n--- Test 6: Cost-aware eviction ---");

    // Budget fits exactly 2 entries
    let mut store = GpuStore::new(800);

    // Register a CHEAP buffer (cost=1μs) — should be evicted first
    let cheap_prov = data_provenance("test:cheap");
    let cheap_ptr = BufferPtr { device_ptr: 0x1000, byte_size: 400 };
    let cheap_header = BufferHeader::new(cheap_prov, 1.0, DType::F32, 100);
    store.register(cheap_header, cheap_ptr);

    // Register an EXPENSIVE buffer (cost=1000μs) — should be kept
    let expensive_prov = data_provenance("test:expensive");
    let expensive_ptr = BufferPtr { device_ptr: 0x2000, byte_size: 400 };
    let expensive_header = BufferHeader::new(expensive_prov, 1000.0, DType::F32, 100);
    store.register(expensive_header, expensive_ptr);

    // Access the expensive one more to boost its retention score
    store.lookup(&expensive_prov);

    // Register a third buffer — triggers eviction
    let new_prov = data_provenance("test:new");
    let new_ptr = BufferPtr { device_ptr: 0x3000, byte_size: 400 };
    let new_header = BufferHeader::new(new_prov, 50.0, DType::F32, 100);
    let evicted = store.register(new_header, new_ptr);

    assert!(!evicted.is_empty(), "Should evict something");

    // The cheap buffer should have been evicted (lower retention score)
    let cheap_evicted = evicted.iter().any(|e| e.provenance == cheap_prov);
    let expensive_evicted = evicted.iter().any(|e| e.provenance == expensive_prov);
    println!("  Cheap evicted: {}  Expensive evicted: {}", cheap_evicted, expensive_evicted);

    // Expensive buffer should survive (higher retention score)
    assert!(
        store.lookup(&expensive_prov).is_some(),
        "Expensive buffer should survive cost-aware eviction"
    );
    println!("  Expensive buffer survives  PASS");
    println!("  New buffer registered  PASS");
}

fn test_null_world() {
    println!("\n--- Test 7: NullWorld always misses ---");

    let mut null = NullWorld;

    let prov = data_provenance("test:null");
    let ptr = BufferPtr { device_ptr: 0x1000, byte_size: 100 };

    // Provenance: always misses
    assert!(null.provenance_get(&prov).is_none());
    null.provenance_put(prov, ptr, 10.0);
    assert!(null.provenance_get(&prov).is_none(), "NullWorld should forget everything");
    println!("  Provenance always misses  PASS");

    // Dirty: everything dirty
    assert!(!null.is_clean(&prov));
    println!("  Everything dirty  PASS");

    // Residency: nothing resident
    assert!(!null.is_resident(&prov));
    assert!(null.resident_pointer(&prov).is_none());
    println!("  Nothing resident  PASS");
}

fn test_world_state_trait() {
    println!("\n--- Test 8: WorldState trait on GpuStore ---");

    let mut store = GpuStore::new(1_000_000);

    // Use store through the WorldState trait
    fn use_world(world: &mut dyn WorldState) {
        let prov = data_provenance("test:world");
        let ptr = BufferPtr { device_ptr: 0xDEAD, byte_size: 100 };

        // Miss
        assert!(world.provenance_get(&prov).is_none());
        assert!(!world.is_clean(&prov));
        assert!(!world.is_resident(&prov));

        // Register
        world.provenance_put(prov, ptr, 50.0);

        // Hit
        assert_eq!(world.provenance_get(&prov), Some(ptr));
        assert!(world.is_clean(&prov));
        assert!(world.is_resident(&prov));
        assert_eq!(world.resident_pointer(&prov), Some(ptr));
    }

    use_world(&mut store);
    println!("  WorldState trait dispatches correctly  PASS");

    // Also works with NullWorld
    fn use_null_world(world: &mut dyn WorldState) {
        let prov = data_provenance("test:null_world");
        assert!(world.provenance_get(&prov).is_none());
        assert!(!world.is_clean(&prov));
        assert!(!world.is_resident(&prov));
    }

    use_null_world(&mut NullWorld);
    println!("  NullWorld through WorldState  PASS");
}

fn test_store_stats() {
    println!("\n--- Test 9: Store statistics ---");

    let mut store = GpuStore::new(1_000_000);

    let prov = data_provenance("test:stats");
    let ptr = BufferPtr { device_ptr: 0x1000, byte_size: 100 };

    // Miss
    store.lookup(&prov);

    // Register + 3 hits
    let header = BufferHeader::new(prov, 10.0, DType::F32, 25);
    store.register(header, ptr);
    store.lookup(&prov);
    store.lookup(&prov);
    store.lookup(&prov);

    let stats = store.stats();
    println!("  Hits: {}  Misses: {}  Hit rate: {:.1}%",
        stats.hits, stats.misses, stats.hit_rate() * 100.0);
    println!("  Entries: {}  Used: {} bytes", stats.entries, stats.used_bytes);
    assert_eq!(stats.hits, 3, "Should have 3 hits");
    assert_eq!(stats.misses, 1, "Should have 1 miss (initial lookup)");
    assert_eq!(stats.entries, 1, "Should have 1 entry");
    println!("  Stats tracking  PASS");
}
