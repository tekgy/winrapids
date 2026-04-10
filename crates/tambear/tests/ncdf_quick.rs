use tambear::special_functions::normal_cdf;

#[test]
fn ncdf_quick() {
    let cases: &[(f64, f64)] = &[
        (0.0,   5.00000000000000000e-01),
        (0.5,   6.91462461274013118e-01),
        (1.0,   8.41344746068542926e-01),
        (1.5,   9.33192798731141915e-01),
        (2.0,   9.77249868051820791e-01),
        (3.0,   9.98650101968369897e-01),
        (4.0,   9.99968328758166880e-01),
        (5.0,   9.99999713348428076e-01),
        (-1.0,  1.58655253931457046e-01),
        (-3.0,  1.34989803163009458e-03),
        (-5.0,  2.86651571879193912e-07),
        (6.0,   9.99999999013412300e-01),
        (-6.0,  9.86587645037698091e-10),
        (0.1,   5.39827837277028988e-01),
        (-0.5,  3.08537538725986882e-01),
    ];
    let mut max_rel = 0.0_f64;
    for &(x, oracle) in cases {
        let got = normal_cdf(x);
        let rel = if oracle == 0.0 { got.abs() } else { (got - oracle).abs() / oracle.abs() };
        println!("x={:8.4} got={:.17e} oracle={:.17e} rel_err={:.3e}", x, got, oracle, rel);
        max_rel = max_rel.max(rel);
    }
    println!("max_rel_err = {:.3e}", max_rel);
    assert!(max_rel < 2e-15, "max relative error {max_rel:.3e} exceeds 2e-15");
}
