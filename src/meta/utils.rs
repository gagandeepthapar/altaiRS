use crate::types::{Generic2D, GenericSlice2D};
use ndarray::Data;

enum DIMCHECK {
    NOOP,
    REP1,
    REP2,
}

fn check_dims(d1: usize, d2: usize) -> DIMCHECK {
    if d1 == d2 {
        return DIMCHECK::NOOP;
    }
    if d1 == 1 {
        return DIMCHECK::REP1;
    }
    if d2 == 1 {
        return DIMCHECK::REP2;
    }
    panic!("Cannot broadcast dimensions: {} VS {}", d1, d2);
}

pub fn sanitize_dims<A>(u1: &GenericSlice2D<A>, u2: &GenericSlice2D<A>) -> (Generic2D, Generic2D)
where
    A: Data<Elem = f64>,
{
    let (u1r, u2r) = match check_dims(u1.shape()[1], u2.shape()[1]) {
        DIMCHECK::NOOP => (u1.to_owned(), u2.to_owned()),
        DIMCHECK::REP1 => {
            let u1r = Generic2D::ones((u1.shape()[0], u2.shape()[1])) * u1;
            (u1r, u2.to_owned())
        }
        DIMCHECK::REP2 => {
            let u2r = Generic2D::ones((u2.shape()[0], u1.shape()[1])) * u2;
            (u1.to_owned(), u2r)
        }
    };
    (u1r, u2r)
}
