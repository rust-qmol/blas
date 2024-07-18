use libopenblas::__BindgenComplex;
use core::num::{complex::Complex, float::Float};

mod l1;
mod l2;
mod l3;

pub(crate) mod libopenblas;

impl<T: Float> __BindgenComplex<T> {
    pub(crate) fn to_complex(&self) -> Complex<T> {
        Complex {
            re: self.re,
            im: self.im,
        }
    }

    pub(crate) fn from_complex(complex: Complex<T>) -> Self {
        __BindgenComplex {
            re: complex.re,
            im: complex.im,
        }
    }
}
