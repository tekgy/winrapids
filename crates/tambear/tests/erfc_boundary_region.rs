use tambear::special_functions::erfc;

#[test]
fn erfc_taylor_region_near_boundary() {
    fn ulps(a: f64, b: f64) -> i64 {
        if a == b { return 0; }
        let ai = a.to_bits() as i64;
        let bi = b.to_bits() as i64;
        (ai - bi).abs()
    }
    // Oracle from mpmath at 50dp
    let cases: &[(f64, f64)] = &[
        (0.5,   0.47950012218695348),
        (0.75,  0.28883914694213895),
        (1.0,   0.15729920705028513),
        (1.1,   0.11979519672936965),
        (1.2,   0.08968999994940674),
        (1.3,   0.06599467609380771),
        (1.3859292911256333, 0.049995790296440877),  // 1.96/sqrt(2)
        (1.4,   0.04771488869416672),
        (1.499, 0.03978437591283688),  // just inside Taylor boundary
        // boundary and just outside:
        (1.501, 0.039663380261695014),  // just outside Taylor, in CF
        (1.5,   0.039797964586491284),  // exactly at boundary
    ];
    let mut max_ulps = 0i64;
    for &(arg, oracle) in cases {
        let got = erfc(arg);
        let u = ulps(got, oracle);
        println!("erfc({:.4}) got={:.17e} oracle={:.17e} ULPs={}", arg, got, oracle, u);
        max_ulps = max_ulps.max(u);
    }
    println!("max ULPs = {}", max_ulps);
}
