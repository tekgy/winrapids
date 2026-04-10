use tambear::special_functions::erfc;

#[test]
fn erfc_ulp_scan_cf_region() {
    fn ulps(a: f64, b: f64) -> i64 {
        let ai = a.to_bits() as i64;
        let bi = b.to_bits() as i64;
        (ai - bi).abs()
    }
    // Oracle: mpmath at 50 dp, converted to f64
    let cases: &[(f64, f64)] = &[
        (1.0,   0.15729920705028513_f64),
        (2.0,   0.004677734981047266_f64),
        (3.0,   2.209049699858544e-05_f64),
        (3.5355339059327378, 5.733031437583871e-07_f64),
        (4.0,   1.541725790028002e-08_f64),
        (4.242640687119285,  1.9731752900753987e-09_f64),
        (5.0,   1.537459794428035e-12_f64),
        (6.0,   2.1519736712498913e-17_f64),
        (7.0,   4.183825607779414e-23_f64),
        (8.0,   1.1224297172982926e-29_f64),
    ];
    let mut max_ulps = 0i64;
    for &(arg, oracle) in cases {
        let got = erfc(arg);
        let u = ulps(got, oracle);
        println!("erfc({:.4}) got={:.17e} oracle={:.17e} ULPs={}", arg, got, oracle, u);
        max_ulps = max_ulps.max(u);
    }
    println!("max ULPs = {}", max_ulps);
    // Claim: ≤ 10 ULP everywhere in CF region (conservative, tighten after measurement)
    assert!(max_ulps <= 10, "max ULPs {} exceeds 10", max_ulps);
}
