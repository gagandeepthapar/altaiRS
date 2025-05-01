use ndarray::{Ix1, Ix2, Ix3};

pub type Generic1D = ndarray::Array1<f64>;
pub type Generic2D = ndarray::Array2<f64>;
pub type Generic3D = ndarray::Array3<f64>;

pub type Scalar = ndarray::Array1<f64>;
pub type Vector3 = ndarray::Array2<f64>;
pub type Quaternion4 = ndarray::Array2<f64>;
pub type Matrix33 = ndarray::Array3<f64>;

pub type GenericSlice1D<A> = ndarray::ArrayBase<A, Ix1>;
pub type GenericSlice2D<A> = ndarray::ArrayBase<A, Ix2>;
pub type GenericSlice3D<A> = ndarray::ArrayBase<A, Ix3>;

pub type Vector3Slice<A> = ndarray::ArrayBase<A, Ix2>;
pub type Quaternion4Slice<A> = ndarray::ArrayBase<A, Ix2>;
pub type TransformMatrix33Slice<A> = ndarray::ArrayBase<A, Ix3>;
