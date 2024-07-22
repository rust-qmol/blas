#[allow(non_snake_case, non_upper_case_globals, non_camel_case_types)]
pub(crate) mod libopenblas;

pub mod l1;
mod l2;
mod l3;

mod raw;

use core::num::{complex::Complex, float::Float};
use libopenblas::__BindgenComplex;
use std::{
    ffi::c_void,
    ptr::{addr_of, addr_of_mut},
};

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

    pub(crate) unsafe fn as_ptr(self) -> *const c_void {
        addr_of!(self) as *const c_void
    }

    pub(crate) unsafe fn as_mut_ptr(mut self) -> *mut c_void {
        addr_of_mut!(self) as *mut c_void
    }
}
