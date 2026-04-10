use tambear::time_series::bocpd;

fn main() {
    let mut data = vec![0.0f64; 100];
    for i in 50..100 {
        data[i] = 1.0;
    }
    let cps = bocpd(&data, 500, 1.0 / 200.0, None);
    println!("Detected CPs: {:?}", cps);
    if cps.iter().any(|&cp| (cp as isize - 50).abs() <= 10) {
        println!("SUCCESS: Detected CP near 50");
    } else {
        println!("FAILURE: Did not detect CP near 50");
        std::process::exit(1);
    }
}
