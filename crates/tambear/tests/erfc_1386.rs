use tambear::special_functions::{erfc, normal_cdf};

#[test]
fn erfc_1386_precise() {
    fn ulps(a: f64, b: f64) -> i64 {
        if a == b { return 0; }
        let ai = a.to_bits() as i64;
        let bi = b.to_bits() as i64;
        (ai - bi).abs()
    }
    let arg = 1.96_f64 / std::f64::consts::SQRT_2;
    println!("arg = {:.17e}", arg);
    
    let erfc_got = erfc(arg);
    let erfc_oracle = 0.04999579029644084_f64; // mpmath at 50dp
    println!("erfc(arg) got    = {:.17e}", erfc_got);
    println!("erfc(arg) oracle = {:.17e}", erfc_oracle);
    println!("erfc ULPs = {}", ulps(erfc_got, erfc_oracle));
    
    // 0.5 * erfc
    let half_erfc = 0.5 * erfc_got;
    println!("0.5 * erfc_got   = {:.17e}", half_erfc);
    
    // normal_cdf(-1.96)
    let ncdf = normal_cdf(-1.96);
    let ncdf_oracle = 0.024997895148220435_f64;
    println!("ncdf(-1.96) got  = {:.17e}", ncdf);
    println!("ncdf(-1.96) oracle={:.17e}", ncdf_oracle);
    println!("ncdf ULPs = {}", ulps(ncdf, ncdf_oracle));
}
