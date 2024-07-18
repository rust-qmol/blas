use crate::{
    blas::libopenblas::{cblas_idamax, cblas_idamin, cblas_dasum, cblas_daxpy, cblas_dcopy, cblas_ddot, cblas_dnrm2, cblas_drot, cblas_drotg, cblas_drotm, cblas_drotmg, cblas_dscal, cblas_dswap},
    vector::Vector,
};

impl Vector<f64> {
    pub fn swap(mut x: Self, mut y: Self) -> (Self, Self){
        unsafe { cblas_dswap(x.blas_len(), x.as_mut_ptr(), 0, y.as_mut_ptr(), 0) }
        (x, y)
    }
    
    pub fn rot(mut x: Self, mut y: Self, c: f64, s: f64) -> (Self, Self){
        unsafe { cblas_drot(x.blas_len(), x.as_mut_ptr(), 0, y.as_mut_ptr(), 0, c, s) }
        (x, y)
    }
    pub fn rotg(mut a: f64, mut b: f64, mut c: f64, mut s: f64) -> (f64, f64, f64, f64){
        unsafe { cblas_drotg(&mut a, &mut b, &mut c, &mut s) };
        (a, b, c, s)
    }
    pub fn rotm(mut x: Self, mut y: Self, param: [f64;5]) -> (Self, Self){
        unsafe { cblas_drotm(x.blas_len(), x.as_mut_ptr(), 0, y.as_mut_ptr(), 0, param.as_ptr()) }
        (x, y)
    }
    pub fn rotmg(mut d1: f64, mut d2: f64, mut x1: f64, y1: f64, mut param: [f64;5]) -> (f64, f64, f64, [f64;5]){
        unsafe { cblas_drotmg(&mut d1, &mut d2, &mut x1, y1, param.as_mut_ptr()) }
        (d1, d2, x1, param)
    }

    pub fn asum(&self) -> f64 {
        unsafe { cblas_dasum(self.blas_len(), self.as_ptr(), 0) }
    }
    pub fn axpy(&mut self, alpha: f64, rhs: Self) -> &mut Self {
        unsafe { cblas_daxpy(self.blas_len(), alpha, rhs.as_ptr(), 0, self.as_mut_ptr(), 0) }
        self
    }
    pub fn copy(&self, rhs: &mut Self){
        unsafe { cblas_dcopy(self.blas_len(), self.as_ptr(), 0, rhs.as_mut_ptr(), 0) }
    }        
    pub fn dot(&self, rhs: Self) -> f64{
        unsafe { cblas_ddot(self.blas_len(), self.as_ptr(), 0, rhs.as_ptr(), 0) }
    }
    pub fn nrm2(&self) -> f64{
        unsafe { cblas_dnrm2(self.blas_len(), self.as_ptr(), 0) }
    }
    pub fn scal(&mut self, alpha: f64) -> &mut Self{
        unsafe { cblas_dscal(self.blas_len(), alpha, self.as_mut_ptr(), 0) };
        self
    }
    pub fn iamax(&self) -> usize {
        unsafe { cblas_idamax(self.blas_len(), self.as_ptr(), 0) }
    }
    pub fn iamin(&self) -> usize {
        unsafe { cblas_idamin(self.blas_len(), self.as_ptr(), 0) }
    }
}

