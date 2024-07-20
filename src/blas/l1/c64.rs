use core::num::complex::Complex;

use crate::{
    blas::libopenblas::{
        __BindgenComplex, cblas_dzasum, cblas_dznrm2, cblas_izamax, cblas_izamin, cblas_zaxpy,
        cblas_zcopy, cblas_zdotc_sub, cblas_zdotu_sub, cblas_zdrot, cblas_zrotg, cblas_zscal,
        cblas_zswap,
    },
    vector::Vector,
};

impl Vector<Complex<f64>> {
    pub fn swap(mut x: Self, mut y: Self) -> (Self, Self) {
        unsafe { cblas_zswap(x.blas_len(), x.as_mut_void_ptr(), 0, y.as_mut_void_ptr(), 0) }
        (x, y)
    }

    pub fn drot(mut x: Self, mut y: Self, c: f64, s: f64) -> (Self, Self) {
        unsafe {
            cblas_zdrot(
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
        a: Complex<f64>,
        b: Complex<f64>,
        mut c: f64,
        s: Complex<f64>,
    ) -> (Complex<f64>, Complex<f64>, f64, Complex<f64>) {
        let mut a_bind = __BindgenComplex::from_complex(a);
        let mut b_bind = __BindgenComplex::from_complex(b);
        let mut s_bind = __BindgenComplex::from_complex(s);
        unsafe {
            cblas_zrotg(
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

    pub fn asum(&self) -> f64 {
        unsafe { cblas_dzasum(self.blas_len(), self.as_void_ptr(), 0) }
    }
    pub fn axpy(&mut self, alpha: Complex<f64>, rhs: Self) -> &mut Self {
        unsafe {
            cblas_zaxpy(
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
            cblas_zcopy(
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
            cblas_zdotc_sub(
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
            cblas_zdotu_sub(
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
    pub fn nrm2(&self) -> f64 {
        unsafe { cblas_dznrm2(self.blas_len(), self.as_void_ptr(), 0) }
    }
    pub fn scal(&mut self, alpha: Complex<f64>) -> &mut Self {
        let alpha_bind = __BindgenComplex::from_complex(alpha);
        unsafe {
            cblas_zscal(
                self.blas_len(),
                alpha_bind.as_ptr(),
                self.as_mut_void_ptr(),
                0,
            )
        };
        self
    }
    pub fn iamax(&self) -> usize {
        unsafe { cblas_izamax(self.blas_len(), self.as_void_ptr(), 0) }
    }
    pub fn iamin(&self) -> usize {
        unsafe { cblas_izamin(self.blas_len(), self.as_void_ptr(), 0) }
    }
}
