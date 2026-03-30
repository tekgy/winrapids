//! Local context validation — kernel generation tests.

use winrapids_local::ops::*;
use winrapids_local::engine::generate_local_context_kernel;
use winrapids_local::cache::cache_key;

fn main() {
    println!("{}", "=".repeat(70));
    println!("winrapids-local validation");
    println!("{}", "=".repeat(70));

    test_basic_gather();
    test_full_feature_set();
    test_minimal_spec();
    test_identity_key();
    test_boundary_handling();

    println!("\n{}", "=".repeat(70));
    println!("ALL LOCAL CONTEXT TESTS PASSED");
    println!("{}", "=".repeat(70));
}

fn test_basic_gather() {
    println!("\n--- Test 1: Basic gather (raw values only) ---");

    let spec = LocalContextSpec {
        offsets: vec![-1, 0, 1],
        features: vec![
            LocalFeature::RawValue { offset_idx: 0 },  // data[i-1]
            LocalFeature::RawValue { offset_idx: 1 },  // data[i]
            LocalFeature::RawValue { offset_idx: 2 },  // data[i+1]
        ],
    };

    let kernel = generate_local_context_kernel(&spec);
    assert!(kernel.contains("local_context"), "Should contain kernel name");
    assert!(kernel.contains("vals[0]"), "Should output vals[0]");
    assert!(kernel.contains("vals[2]"), "Should output vals[2]");
    assert_eq!(spec.output_width(), 3);

    // No intermediates needed for raw values
    assert!(!kernel.contains("local_mean"));
    assert!(!kernel.contains("local_std"));

    println!("  3 offsets, 3 raw values: {} bytes  PASS", kernel.len());
}

fn test_full_feature_set() {
    println!("\n--- Test 2: Full feature set (FinTek offsets) ---");

    let offsets = vec![-10, -5, -3, -1, 0, 1, 3, 5, 10];

    let spec = LocalContextSpec {
        offsets: offsets.clone(),
        features: vec![
            LocalFeature::Delta { offset_idx: 3 },       // delta vs lag -1
            LocalFeature::Delta { offset_idx: 0 },       // delta vs lag -10
            LocalFeature::LogRatio { offset_idx: 3 },    // log return vs -1
            LocalFeature::Direction { offset_idx: 3 },   // direction vs -1
            LocalFeature::LocalMean,
            LocalFeature::LocalStd,
            LocalFeature::Slope,
            LocalFeature::PeakDetect,
        ],
    };

    let kernel = generate_local_context_kernel(&spec);

    // Should contain all feature computations
    assert!(kernel.contains("center - vals[3]"), "Delta feature");
    assert!(kernel.contains("log(center / vals[3])"), "LogRatio feature");
    assert!(kernel.contains("local_mean"), "Local mean intermediate");
    assert!(kernel.contains("local_std"), "Local std intermediate");
    assert!(kernel.contains("slope"), "Slope intermediate");
    assert!(kernel.contains("is_peak"), "Peak detection");

    assert_eq!(spec.output_width(), 8);

    println!("  9 offsets, 8 features: {} bytes  PASS", kernel.len());

    // Feature names
    let names: Vec<String> = spec.features.iter()
        .map(|f| f.name(&offsets))
        .collect();
    println!("  Features: {:?}", names);
}

fn test_minimal_spec() {
    println!("\n--- Test 3: Minimal spec (single delta) ---");

    let spec = LocalContextSpec {
        offsets: vec![-1, 0],
        features: vec![
            LocalFeature::Delta { offset_idx: 0 },  // data[i] - data[i-1] = diff(1)
        ],
    };

    let kernel = generate_local_context_kernel(&spec);
    assert_eq!(spec.output_width(), 1);

    // Should NOT compute unnecessary intermediates
    assert!(!kernel.contains("local_mean"));
    assert!(!kernel.contains("local_std"));
    assert!(!kernel.contains("slope"));
    assert!(!kernel.contains("peak"));

    println!("  1 output (diff): {} bytes  PASS", kernel.len());
}

fn test_identity_key() {
    println!("\n--- Test 4: Identity and cache keys ---");

    let spec1 = LocalContextSpec {
        offsets: vec![-1, 0, 1],
        features: vec![LocalFeature::Delta { offset_idx: 0 }],
    };
    let spec2 = LocalContextSpec {
        offsets: vec![-5, 0, 5],
        features: vec![LocalFeature::Delta { offset_idx: 0 }],
    };
    let spec3 = LocalContextSpec {
        offsets: vec![-1, 0, 1],
        features: vec![LocalFeature::LogRatio { offset_idx: 0 }],
    };

    // Different offsets -> different identity
    assert_ne!(spec1.identity_key(), spec2.identity_key());
    // Different features -> different identity
    assert_ne!(spec1.identity_key(), spec3.identity_key());
    // Same spec -> same identity
    let spec1b = spec1.clone();
    assert_eq!(spec1.identity_key(), spec1b.identity_key());

    println!("  Identity key uniqueness  PASS");

    // BLAKE3 cache keys
    assert_ne!(cache_key(&spec1), cache_key(&spec2),
        "Different offsets must have different cache keys");
    assert_ne!(cache_key(&spec1), cache_key(&spec3),
        "Different features must have different cache keys");
    assert_eq!(cache_key(&spec1), cache_key(&spec1b),
        "Same spec must have same cache key");
    println!("  BLAKE3 cache key uniqueness  PASS");
}

fn test_boundary_handling() {
    println!("\n--- Test 5: Boundary handling ---");

    let spec = LocalContextSpec {
        offsets: vec![-100, 0, 100],
        features: vec![
            LocalFeature::RawValue { offset_idx: 0 },
            LocalFeature::RawValue { offset_idx: 2 },
        ],
    };

    let kernel = generate_local_context_kernel(&spec);

    // Should have boundary checks
    assert!(kernel.contains("src >= 0 && src < n"), "Should bounds-check offset reads");
    // Out-of-bounds defaults to center value
    assert!(kernel.contains("center"), "Should default to center for OOB");

    println!("  Boundary handling present  PASS");
}
