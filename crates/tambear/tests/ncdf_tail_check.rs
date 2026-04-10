use tambear::special_functions::{normal_cdf, erfc};

#[test]
fn ncdf_tail_ulp_scan() {
    // ULP count helper: counts integer distance between f64 bit patterns
    fn ulps(a: f64, b: f64) -> i64 {
        let ai = a.to_bits() as i64;
        let bi = b.to_bits() as i64;
        (ai - bi).abs()
    }
    // Oracle values from mpmath at 50 dp for erfc at various arguments
    // erfc(arg) where arg = -x/sqrt(2), x negative → large positive args
    let erfc_cases: &[(f64, f64)] = &[
        // (arg, mpmath oracle for erfc(arg))
        (1.0,  0.15729920705028513_f64),
        (2.0,  0.004677734981047265_f64),
        (3.0,  2.2090496998585441e-05_f64),
        (3.5355339059327378,  5.73303143758389082e-07_f64),  // x=-5
        (4.0,  1.5418042952029988e-08_f64),
        (4.242640687119285,   1.97317529007539618e-09_f64),   // x=-6
        (5.0,  1.5374597944280348e-12_f64),
    ];
    println!("erfc ULP scan:");
    for &(arg, oracle) in erfc_cases {
        let got = erfc(arg);
        let ulp = ulps(got, oracle);
        println!("  erfc({:.4}) got={:.17e} oracle={:.17e} ULPs={}", arg, got, oracle, ulp);
    }
}
