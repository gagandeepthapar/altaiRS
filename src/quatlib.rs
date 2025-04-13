use super::meta::types::*;
use super::meta::utils::*;

use ndarray::{concatenate, s, stack, Axis};

use super::veclib::*;

// pub fn randq(num: usize) -> Quaternion4 {
//     let mut randr = rand::rngs::OsRng;
//     let unif = rand::distributions::Uniform::new(-PI, PI);
//     let ax = runit(num);
//     let theta = Generic1D::from_shape_vec(
//         num,
//         unif.sample_iter(&mut randr).take(num).collect::<Vec<f64>>(),
//     )
//     .unwrap();
//     let st = (theta.clone() / 2.).sin();
//     let ct = (theta / 2.).cos();
//     let q1 = &ax.row(0) * st.clone();
//     let q2 = &ax.row(1) * st.clone();
//     let q3 = &ax.row(2) * st;
//     let q4 = ct;

//     ndarray::stack![Axis(0), q1, q2, q3, q4]
// }

pub fn qangle(q32: &Quaternion4, q21: &Quaternion4) -> Generic1D {
    // let (q32, q21) = sanitize_dims(q32, q21);
    let dq = qcomp(q32, &qinv(q21));
    let (_, st) = unit(ndarray::stack![Axis(0), dq.row(0), dq.row(1), dq.row(2)]);
    let st = st.to_vec();
    let ct = dq.row(3).to_vec();

    Generic1D::from_iter(st.iter().zip(ct.iter()).map(|(&s, &c)| 2.0 * s.atan2(c)))
}

pub fn qinv(q21: &Quaternion4) -> Quaternion4 {
    ndarray::stack![
        Axis(0),
        -1. * &q21.row(0),
        -1. * &q21.row(1),
        -1. * &q21.row(2),
        q21.row(3)
    ]
}

pub fn qcomp(q32: &Quaternion4, q21: &Quaternion4) -> Quaternion4 {
    let (q31, _) = unit(qmult(q32, q21));
    q31
}

pub fn qmult(q32: &Quaternion4, q21: &Quaternion4) -> Quaternion4 {
    // Markley 2.84a
    // let (q32, q21) = sanitize_dims(q32, q21);
    let n_col = q32.shape()[1];

    let psi_q32 = psi_q(q32.to_owned());
    let mut q31 = Quaternion4::zeros((4, n_col));
    q31.axis_iter_mut(Axis(1))
        .zip(
            psi_q32
                .axis_iter(Axis(2))
                .zip(q32.axis_iter(Axis(1)).zip(q21.axis_iter(Axis(1)))),
        )
        .for_each(|(q_31, (psiq, (q_32, q_21)))| {
            let qmat = concatenate![Axis(1), psiq, q_32.into_shape_with_order((4, 1)).unwrap()];
            (qmat.dot(&q_21)).assign_to(q_31);
        });

    q31
}

fn mat_q_form(q21: Quaternion4, scl: f64) -> Generic3D {
    let n_col = q21.shape()[1];

    let qvec = q21.slice(s![0..3, ..]).to_owned();
    let nqvec = qvec.to_owned() * -1.;
    let qscl = q21.slice(s![3, ..]);

    let mut psiq = Generic3D::zeros((4, 3, n_col));
    psiq.slice_mut(s![0, 0, ..]).assign(&qscl);
    psiq.slice_mut(s![1, 1, ..]).assign(&qscl);
    psiq.slice_mut(s![2, 2, ..]).assign(&qscl);

    let qvec_skew = scl * skew(qvec).into_shape_with_order((3, 3, n_col)).unwrap();

    qvec_skew
        .slice(s![1.., 0, ..])
        .assign_to(psiq.slice_mut(s![1..3, 0, ..]));

    qvec_skew
        .slice(s![0, 1, ..])
        .assign_to(psiq.slice_mut(s![0, 1, ..]));

    qvec_skew
        .slice(s![2, 1, ..])
        .assign_to(psiq.slice_mut(s![2, 1, ..]));

    qvec_skew
        .slice(s![0..2, 2, ..])
        .assign_to(psiq.slice_mut(s![0..2, 2, ..]));

    psiq.slice_mut(s![3, 0, ..]).assign(&nqvec.slice(s![0, ..]));
    psiq.slice_mut(s![3, 1, ..]).assign(&nqvec.slice(s![1, ..]));
    psiq.slice_mut(s![3, 2, ..]).assign(&nqvec.slice(s![2, ..]));

    psiq
}

pub fn psi_q(q21: Quaternion4) -> Generic3D {
    mat_q_form(q21, -1.)
}

pub fn xi_q(q21: Quaternion4) -> Generic3D {
    mat_q_form(q21, 1.)
}

pub fn qxform(q21: Quaternion4, u1: Vector3) -> Vector3 {
    let (q21, u1) = sanitize_dims(q21, u1);
    let (u1u, u1m) = unit(u1);
    let n_col = u1u.shape()[0];

    let xi_q = xi_q(q21.to_owned());
    let psi_q = psi_q(q21);

    let mut u2 = Vector3::zeros((3, n_col));
    u2.axis_iter_mut(Axis(1)).enumerate().for_each(|(ii, u)| {
        (u1m[ii]
            * xi_q
                .slice(s![.., .., ii])
                .t()
                .dot(&psi_q.slice(s![.., .., ii]))
                .dot(&u1u.slice(s![.., ii])))
        .assign_to(u);
    });

    u2
}

pub fn dcm2quat(T21: TransformMatrix33) -> Quaternion4 {
    let n_col = T21.shape()[2];
    let t21_tr = Generic1D::from_iter(
        T21.axis_iter(Axis(2))
            .map(|T| T[[0, 0]] + T[[1, 1]] + T[[2, 2]]),
    );

    let qA = stack![
        Axis(1),
        1. + 2. * T21.slice(s![0, 0, ..]).to_owned() - t21_tr.to_owned(),
        T21.slice(s![0, 1, ..]).to_owned() + T21.slice(s![1, 0, ..]).to_owned(),
        T21.slice(s![0, 2, ..]).to_owned() + T21.slice(s![2, 0, ..]).to_owned(),
        T21.slice(s![1, 2, ..]).to_owned() - T21.slice(s![2, 1, ..]).to_owned()
    ];

    let qB = stack![
        Axis(1),
        T21.slice(s![1, 0, ..]).to_owned() + T21.slice(s![0, 1, ..]).to_owned(),
        1. + 2. * T21.slice(s![1, 1, ..]).to_owned() - t21_tr.to_owned(),
        T21.slice(s![1, 2, ..]).to_owned() + T21.slice(s![2, 1, ..]).to_owned(),
        T21.slice(s![2, 0, ..]).to_owned() - T21.slice(s![0, 2, ..]).to_owned(),
    ];

    let qC = stack![
        Axis(1),
        T21.slice(s![2, 0, ..]).to_owned() + T21.slice(s![0, 2, ..]).to_owned(),
        T21.slice(s![1, 2, ..]).to_owned() + T21.slice(s![2, 1, ..]).to_owned(),
        1. + 2. * T21.slice(s![2, 2, ..]).to_owned() - t21_tr.to_owned(),
        T21.slice(s![0, 1, ..]).to_owned() - T21.slice(s![1, 0, ..]).to_owned(),
    ];

    let qD = stack![
        Axis(1),
        T21.slice(s![1, 2, ..]).to_owned() - T21.slice(s![2, 1, ..]).to_owned(),
        T21.slice(s![2, 0, ..]).to_owned() - T21.slice(s![0, 2, ..]).to_owned(),
        T21.slice(s![0, 1, ..]).to_owned() + T21.slice(s![1, 0, ..]).to_owned(),
        1. + t21_tr
    ];

    qA
}
