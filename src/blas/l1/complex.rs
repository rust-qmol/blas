use core::num::complex::Complex;
use std::ffi::c_void;

use crate::{
    blas::libopenblas::{cblas_dzasum, cblas_scasum},
    vector::Vector,
};

impl Vector<Complex<f32>> {
    pub fn asum(&self) -> f32 {
        unsafe {
            cblas_scasum(
                self.data.len() as i32,
                self.data.as_ptr() as *const c_void,
                0,
            )
        }
    }
}

impl Vector<Complex<f64>> {
    pub fn asum(&self) -> f64 {
        unsafe {
            cblas_dzasum(
                self.data.len() as i32,
                self.data.as_ptr() as *const c_void,
                0,
            )
        }
    }
}
