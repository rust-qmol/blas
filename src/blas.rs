use libopenblas::__BindgenComplex;
use core::num::{complex::Complex, float::Float};
use std::{ffi::c_void, ptr::{addr_of, addr_of_mut}};

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

    pub(crate) fn as_ptr(&self) -> *const c_void{
        addr_of!(self) as *const c_void
    }

    pub(crate) fn as_mut_ptr(mut self: &mut Self) -> *mut c_void{
        addr_of_mut!(self) as *mut c_void
    }
}
