use crate::meta::utils::*;

use crate::types::*;
use ndarray::Axis;
use ndarray::{s, stack, Data};

pub fn skew<A>(u1: &Vector3Slice<A>) -> Generic3D
where
    A: Data<Elem = f64>,
{
    let ncol = u1.shape()[1];
    let nu1 = u1 * -1.;

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

pub fn unit<A>(u1: &GenericSlice2D<A>) -> (Generic2D, Scalar)
where
    A: Data<Elem = f64>,
{
    let norm = u1.pow2().sum_axis(Axis(0)).sqrt(); // col-wise norm
    (u1 / &norm, norm)
}

pub fn sat<A>(vals: &GenericSlice1D<A>, lims: (f64, f64)) -> Generic1D
where
    A: Data<Elem = f64>,
{
    Generic1D::from_iter(vals.iter().map(|&v| {
        if v < lims.0 {
            lims.0
        } else if v > lims.1 {
            lims.1
        } else {
            v
        }
    }))
}

pub fn clamp<T>(val: T, lims: (T, T)) -> T
where
    T: Ord,
{
    std::cmp::min(std::cmp::max(val, lims.0), lims.1)
}

pub fn fdot<A>(u1: &Vector3Slice<A>, u2: &Vector3Slice<A>) -> Scalar
where
    A: Data<Elem = f64>,
{
    let (u1, u2) = sanitize_dims(u1, u2);
    (u1 * u2).sum_axis(Axis(0)) // sum(a_i * b_i)
}

pub fn fcross<A>(u1: &Vector3Slice<A>, u2: &Vector3Slice<A>) -> Vector3
where
    A: Data<Elem = f64>,
{
    let (u1, u2) = sanitize_dims(u1, u2);
    stack![
        Axis(0),
        &u1.row(1) * &u2.row(2) - &u1.row(2) * &u2.row(1),
        &u1.row(2) * &u2.row(0) - &u1.row(0) * &u2.row(2),
        &u1.row(0) * &u2.row(1) - &u1.row(1) * &u2.row(0)
    ]
}

pub fn vangle<A>(u1: &Vector3Slice<A>, u2: &Vector3Slice<A>) -> Scalar
where
    A: Data<Elem = f64>,
{
    let (u1, u2) = sanitize_dims(u1, u2);
    let (u1u, _) = unit(&u1); // unit vectors
    let (u2u, _) = unit(&u2);

    Generic1D::from_iter(
        sat(&fdot(&u1u, &u2u), (-1., 1.))
            .iter()
            .map(|dprod| dprod.acos()),
    )
}

pub fn radec2uvec<A>(ra: GenericSlice1D<A>, dec: GenericSlice1D<A>) -> Vector3
where
    A: Data<Elem = f64>,
{
    // Sanitize Dims
    let ra = Generic2D::from_shape_vec((1, ra.len()), ra.to_vec()).unwrap();
    let dec = Generic2D::from_shape_vec((1, dec.len()), dec.to_vec()).unwrap();
    let (ra, dec) = sanitize_dims(&ra, &dec);

    let u1 = ra.cos() * dec.cos();
    let u2 = ra.sin() * dec.cos();
    let u3 = dec.sin();

    stack!(Axis(0), u1.row(0), u2.row(0), u3.row(0))
}

#[cfg(test)]
mod tests {
    use ndarray::concatenate;

    use super::*;

    fn x(n: usize) -> Generic2D {
        concatenate![Axis(0), Generic2D::ones((1, n)), Generic2D::zeros((2, n))]
    }
    fn y(n: usize) -> Generic2D {
        concatenate![
            Axis(0),
            Generic2D::zeros((1, n)),
            Generic2D::ones((1, n)),
            Generic2D::zeros((1, n))
        ]
    }
    fn z(n: usize) -> Generic2D {
        concatenate![Axis(0), Generic2D::zeros((2, n)), Generic2D::ones((1, n))]
    }

    #[test]
    fn test_cross_xy() {
        let ux = x(1);
        let uy = y(1);
        assert_eq!(fcross(&ux, &uy), z(1))
    }

    #[test]
    fn test_cross_xz() {
        let ux = x(1);
        let uz = z(1);
        assert_eq!(fcross(&ux, &uz), -1. * y(1))
    }

    #[test]
    fn test_cross_yz() {
        let uy = y(1);
        let uz = z(1);
        assert_eq!(fcross(&uy, &uz), &x(1))
    }

    #[test]
    fn test_dot_xx() {
        let u1 = x(1);
        let u2 = x(1);
        assert_eq!(fdot(&u1, &u2), Generic1D::from_vec(vec![1.]))
    }

    #[test]
    fn test_dot_xy() {
        let u1 = x(1);
        let u2 = y(1);
        assert_eq!(fdot(&u1, &u2), Generic1D::from_vec(vec![0.]))
    }
}
