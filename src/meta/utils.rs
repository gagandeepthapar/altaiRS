use super::types::Generic2D;

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

pub fn sanitize_dims(u1: Generic2D, u2: Generic2D) -> (Generic2D, Generic2D) {
    let (u1r, u2r) = match check_dims(u1.shape()[1], u2.shape()[1]) {
        DIMCHECK::NOOP => (u1, u2),
        DIMCHECK::REP1 => {
            let u1r = Generic2D::ones((u1.shape()[0], u2.shape()[1])) * u1;
            (u1r, u2)
        }
        DIMCHECK::REP2 => {
            let u2r = Generic2D::ones((u2.shape()[0], u1.shape()[1])) * u2;
            (u1, u2r)
        }
    };
    (u1r, u2r)
}
