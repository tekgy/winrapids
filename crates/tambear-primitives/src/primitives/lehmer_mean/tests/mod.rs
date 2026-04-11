use super::*;

#[test]
fn p1_is_arithmetic() {
    let data = vec![1.0, 2.0, 3.0];
    assert!((lehmer_mean(&data, 1.0) - 2.0).abs() < 1e-14);
}

#[test]
fn p0_is_harmonic() {
    let data = vec![1.0, 2.0, 4.0];
    let lehmer = lehmer_mean(&data, 0.0);
    let harmonic = crate::mean_harmonic::mean_harmonic(&data);
    assert!((lehmer - harmonic).abs() < 1e-12);
}

#[test]
fn p2_is_contraharmonic() {
    let data = vec![2.0, 4.0, 6.0];
    let lehmer = lehmer_mean(&data, 2.0);
    let contra = mean_contraharmonic(&data);
    assert!((lehmer - contra).abs() < 1e-12);
}

#[test]
fn contraharmonic_ge_arithmetic() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let contra = mean_contraharmonic(&data);
    let arith = crate::mean_arithmetic::mean_arithmetic(&data);
    assert!(contra >= arith - 1e-14);
}

#[test]
fn empty_is_nan() {
    assert!(lehmer_mean(&[], 1.0).is_nan());
    assert!(mean_contraharmonic(&[]).is_nan());
}

#[test]
fn negative_data_is_nan() {
    assert!(lehmer_mean(&[1.0, -1.0], 0.5).is_nan());
}
