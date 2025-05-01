use crate::meta::utils::sanitize_dims;
use crate::types::*;
use crate::veclib::*;

use ndarray::{concatenate, s, stack, Axis, Data};

pub fn qinv<A>(q21: &Quaternion4Slice<A>) -> Quaternion4
where
    A: Data<Elem = f64>,
{
    stack![
        Axis(0),
        q21.row(0).to_owned() * -1.,
        q21.row(1).to_owned() * -1.,
        q21.row(2).to_owned() * -1.,
        q21.row(3).to_owned() * 1.,
    ]
}

pub fn qmult<A>(q32: &Quaternion4Slice<A>, q21: &Quaternion4Slice<A>) -> Quaternion4
where
    A: Data<Elem = f64>,
{
    // Markley 2.84a
    // Sanitize Dimensions
    let (q32, q21) = sanitize_dims(q32, q21);

    // Extract Vectors and Scalars
    let (qv, pv) = (q32.slice(s![0..3, ..]), q21.slice(s![0..3, ..]));
    let (q4, p4) = (
        q32.row(3).insert_axis(Axis(0)),
        q21.row(3).insert_axis(Axis(0)),
    );

    // Compute vector portion
    let qvec = concatenate![Axis(0), q4, q4, q4] * &pv - fcross(&qv, &pv)
        + concatenate![Axis(0), p4, p4, p4] * &qv;

    // Compute scalar portion
    let qscl = (qinv(&q32) * &q21).sum_axis(Axis(0)).insert_axis(Axis(0));

    concatenate![Axis(0), qvec, qscl]
}

pub fn qcomp<A>(q32: &Quaternion4Slice<A>, q21: &Quaternion4Slice<A>) -> Quaternion4
where
    A: Data<Elem = f64>,
{
    let (q31, _) = unit(&qmult(q32, q21));
    q31
}

pub fn qangle<A>(q32: &Quaternion4Slice<A>, q21: &Quaternion4Slice<A>) -> Scalar
where
    A: Data<Elem = f64>,
{
    let (q32, q21) = sanitize_dims(q32, q21);
    let dq = qcomp(&q32, &qinv(&q21));
    let (_, sintheta) = unit(&dq.slice(s![0..3, ..]));
    let costheta = dq.slice(s![3, ..]);

    Generic1D::from_iter(
        sintheta
            .iter()
            .zip(costheta.iter())
            .map(|(&s, &c)| 2.0 * s.atan2(c)),
    )
}

pub fn qxform<A>(q21: &Quaternion4Slice<A>, u1: &Vector3Slice<A>) -> Vector3
where
    A: Data<Elem = f64>,
{
    // Sanitize dimensions
    let (q21, u1) = sanitize_dims(q21, u1);

    // Markley 2.130
    let qu1 = concatenate![
        Axis(0),
        u1,
        Generic1D::zeros(u1.shape()[1]).insert_axis(Axis(0))
    ]; // Create quaternion form of vector [x; 0];

    qcomp(&q21, &qcomp(&qu1, &qinv(&q21))) // u2 = q \otimes u1 \otimes qinv(q)
        .slice(s![0..3, ..]) // only grab first 3 rows
        .to_owned()
}

pub fn qdot<A>(w2: &Vector3Slice<A>, q21: &Quaternion4Slice<A>) -> Quaternion4
where
    A: Data<Elem = f64>,
{
    let (w2, q21) = sanitize_dims(w2, q21);

    // Markley 3.79d
    0.5 * qmult(
        &concatenate![
            Axis(0),
            w2,
            Generic1D::zeros(w2.shape()[1]).insert_axis(Axis(0))
        ],
        &q21,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn q_ident(n: usize) -> Generic2D {
        concatenate![
            Axis(0),
            Generic2D::zeros((3, n)),
            Generic1D::ones(n).insert_axis(Axis(0))
        ]
    }

    #[test]
    fn test_qinv_identity() {
        let qi = q_ident(1);
        assert_eq!(qinv(&qi), qi);
    }

    #[test]
    fn test_qinv() {
        let qi = Generic2D::from_shape_vec((4, 1), vec![1., 2., 3., 4.]).unwrap();
        let true_inv = Generic2D::from_shape_vec((4, 1), vec![-1., -2., -3., 4.]).unwrap();
        assert_eq!(qinv(&qi), true_inv);
    }

    #[test]
    fn test_qmult_zero() {
        let q_a = q_ident(1);
        let q_b = q_ident(1);
        let q_c = q_ident(1);

        assert_eq!(qmult(&q_a, &q_b), q_c);
    }

    #[test]
    fn test_qangle_zero() {
        let q_a = q_ident(1);
        let q_b = q_ident(1);
        let q_c = array![0.];

        assert_eq!(qangle(&q_a, &q_b), q_c);
    }

    #[test]
    fn test_qdot_zero_rate() {
        let w = Generic2D::zeros((3, 1));
        let q21 = q_ident(1);

        assert_eq!(qdot(&w, &q21), Generic2D::zeros((4, 1)))
    }
}
