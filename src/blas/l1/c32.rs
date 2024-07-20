use core::num::complex::Complex;

use crate::{
    blas::libopenblas::{
        __BindgenComplex, cblas_cdotc_sub, cblas_cscal, cblas_cdotu_sub, cblas_crotg, cblas_csrot, cblas_icamax,
        cblas_icamin, cblas_scasum, cblas_scnrm2, cblas_caxpy, cblas_ccopy, cblas_cswap,
    },
    vector::Vector,
};

impl Vector<Complex<f32>> {
    pub fn swap(mut x: Self, mut y: Self) -> (Self, Self) {
        unsafe { cblas_cswap(x.blas_len(), x.as_mut_void_ptr(), 0, y.as_mut_void_ptr(), 0) }
        (x, y)
    }

    pub fn drot(mut x: Self, mut y: Self, c: f32, s: f32) -> (Self, Self) {
        unsafe {
            cblas_csrot(
                x.blas_len(),
                x.as_mut_void_ptr(),
                0,
                y.as_mut_void_ptr(),
                0,
                c,
                s,
            )
        }
        (x, y)
    }
    pub fn rotg(
        a: Complex<f32>,
        b: Complex<f32>,
        mut c: f32,
        s: Complex<f32>,
    ) -> (Complex<f32>, Complex<f32>, f32, Complex<f32>) {
        let mut a_bind = __BindgenComplex::from_complex(a);
        let mut b_bind = __BindgenComplex::from_complex(b);
        let mut s_bind = __BindgenComplex::from_complex(s);
        unsafe {
            cblas_crotg(
                a_bind.as_mut_ptr(),
                b_bind.as_mut_ptr(),
                &mut c,
                s_bind.as_mut_ptr(),
            )
        };
        (
            a_bind.to_complex(),
            b_bind.to_complex(),
            c,
            s_bind.to_complex(),
        )
    }

    pub fn asum(&self) -> f32 {
        unsafe { cblas_scasum(self.blas_len(), self.as_void_ptr(), 0) }
    }
    pub fn axpy(&mut self, alpha: Complex<f32>, rhs: Self) -> &mut Self {
        unsafe {
            cblas_caxpy(
                self.blas_len(),
                __BindgenComplex::from_complex(alpha).as_ptr(),
                rhs.as_void_ptr(),
                0,
                self.as_mut_void_ptr(),
                0,
            )
        }
        self
    }
    pub fn copy(&self, rhs: &mut Self) {
        unsafe {
            cblas_ccopy(
                self.blas_len(),
                self.as_void_ptr(),
                0,
                rhs.as_mut_void_ptr(),
                0,
            )
        }
    }
    pub fn dotc(&self, rhs: Self) -> Self {
        let mut res = self.clone();
        unsafe {
            cblas_cdotc_sub(
                self.blas_len(),
                self.as_void_ptr(),
                0,
                rhs.as_void_ptr(),
                0,
                res.as_mut_void_ptr(),
            )
        }
        res
    }
    pub fn dotu(&self, rhs: Self) -> Self {
        let mut res = self.clone();
        unsafe {
            cblas_cdotu_sub(
                self.blas_len(),
                self.as_void_ptr(),
                0,
                rhs.as_void_ptr(),
                0,
                res.as_mut_void_ptr(),
            )
        }
        res
    }
    pub fn nrm2(&self) -> f32 {
        unsafe { cblas_scnrm2(self.blas_len(), self.as_void_ptr(), 0) }
    }
    pub fn scal(&mut self, alpha: Complex<f32>) -> &mut Self {
        let alpha_bind = __BindgenComplex::from_complex(alpha);
        unsafe { cblas_cscal(self.blas_len(), alpha_bind.as_ptr(), self.as_mut_void_ptr(), 0) };
        self
    }
    pub fn iamax(&self) -> usize {
        unsafe { cblas_icamax(self.blas_len(), self.as_void_ptr(), 0) }
    }
    pub fn iamin(&self) -> usize {
        unsafe { cblas_icamin(self.blas_len(), self.as_void_ptr(), 0) }
    }
}
