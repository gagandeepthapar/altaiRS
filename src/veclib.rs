use super::meta::types::*;
use super::meta::utils::*;

use ndarray::Axis;
use ndarray::{array, s, stack, Array1, ArrayBase, Data, Ix1};
use rand::{self, prelude::Distribution};

pub fn skew(u1: Vector3) -> Generic3D {
    let ncol = u1.shape()[1];
    let nu1 = &u1 * -1.;

    let mut skewblock = Generic3D::zeros((3, 3, ncol));

    // column 1
    u1.slice(s![2, ..])
        .assign_to(skewblock.slice_mut(s![1, 0, ..]));
    nu1.slice(s![1, ..])
        .assign_to(skewblock.slice_mut(s![2, 0, ..]));

    // column 2
    nu1.slice(s![2, ..])
        .assign_to(skewblock.slice_mut(s![0, 1, ..]));
    u1.slice(s![0, ..])
        .assign_to(skewblock.slice_mut(s![2, 1, ..]));

    // column 3
    u1.slice(s![1, ..])
        .assign_to(skewblock.slice_mut(s![0, 2, ..]));
    nu1.slice(s![0, ..])
        .assign_to(skewblock.slice_mut(s![1, 2, ..]));

    skewblock
}

pub fn unit(u1: Generic2D) -> (Generic2D, Generic1D) {
    let norm = u1.pow2().sum_axis(Axis(0)).sqrt(); // col-wise norm
    (u1 / norm.clone(), norm)
}

pub fn sat(val: f64, lims: (f64, f64)) -> f64 {
    let val = if val < lims.0 { lims.0 } else { val };
    let val = if lims.1 < val { lims.1 } else { val };
    val
}

pub fn clamp<T>(val: T, lims: (T, T)) -> T
where
    T: Ord,
{
    std::cmp::min(std::cmp::max(val, lims.0), lims.1)
}

pub fn runit(num: usize) -> Vector3 {
    let mut randr = rand::rngs::OsRng;
    let unif = rand::distributions::Uniform::new(-1., 1.);
    let uz = unif
        .sample_iter(&mut randr)
        .take(num * 3)
        .collect::<Vec<f64>>();

    let (u1, _) = unit(Vector3::from_shape_vec((3, num), uz).unwrap());
    u1
}

pub fn fdot(u1: Vector3, u2: Vector3) -> Generic1D {
    let (u1, u2) = sanitize_dims(u1, u2);
    (u1 * u2).sum_axis(Axis(0)) // sum(a_i * b_i)
}

pub fn mfcross<A>(u1: &ArrayBase<A, Ix1>, u2: &ArrayBase<A, Ix1>) -> Array1<f64>
where
    A: Data<Elem = f64>,
{
    array![
        u1[1] * u2[2] - u1[2] * u1[1],
        u1[2] * u2[0] - u1[0] * u2[2],
        u1[0] * u2[1] - u1[1] * u2[0]
    ]
}

pub fn fcross(u1: &Vector3, u2: &Vector3) -> Vector3 {
    // let (u1, u2) = sanitize_dims(u1, u2);
    let x = &u1.row(1) * &u2.row(2) - &u1.row(2) * &u2.row(1);
    let y = &u1.row(2) * &u2.row(0) - &u1.row(0) * &u2.row(2);
    let z = &u1.row(0) * &u2.row(1) - &u1.row(1) * &u2.row(0);
    ndarray::stack![Axis(0), x, y, z]
}

pub fn vangle(u1: Vector3, u2: Vector3) -> Generic1D {
    let (u1u, _) = unit(u1); // unit vectors
    let (u2u, _) = unit(u2);
    Generic1D::from_vec(
        fdot(u1u, u2u)
            .iter()
            .map(|&v| sat(v, (-1., 1.)).acos())
            .collect::<Vec<f64>>(),
    )
}

pub fn radec2uvec(ra: Generic1D, dec: Generic1D) -> Vector3 {
    // Sanitize Dims
    let ra = Generic2D::from_shape_vec((1, ra.len()), ra.to_vec()).unwrap();
    let dec = Generic2D::from_shape_vec((1, dec.len()), dec.to_vec()).unwrap();
    let (ra, dec) = sanitize_dims(ra, dec);

    let u1 = ra.cos() * dec.cos();
    let u2 = ra.sin() * dec.cos();
    let u3 = dec.sin();

    stack!(Axis(0), u1.row(0), u2.row(0), u3.row(0))
}
