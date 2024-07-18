use crate::{
    blas::libopenblas::{cblas_dsdot, cblas_isamax, cblas_isamin, cblas_sasum, cblas_saxpy, cblas_scopy, cblas_sdot, cblas_sdsdot, cblas_snrm2, cblas_srot, cblas_srotg, cblas_srotm, cblas_srotmg, cblas_sscal, cblas_sswap},
    vector::Vector,
};

impl Vector<f32> {
    pub fn swap(mut x: Self, mut y: Self) -> (Self, Self){
        unsafe { cblas_sswap(x.blas_len(), x.as_mut_ptr(), 0, y.as_mut_ptr(), 0) }
        (x, y)
    }
    
    pub fn rot(mut x: Self, mut y: Self, c: f32, s: f32) -> (Self, Self){
        unsafe { cblas_srot(x.blas_len(), x.as_mut_ptr(), 0, y.as_mut_ptr(), 0, c, s) }
        (x, y)
    }
    pub fn rotg(mut a: f32, mut b: f32, mut c: f32, mut s: f32) -> (f32, f32, f32, f32){
        unsafe { cblas_srotg(&mut a, &mut b, &mut c, &mut s) };
        (a, b, c, s)
    }
    pub fn rotm(mut x: Self, mut y: Self, param: [f32;5]) -> (Self, Self){
        unsafe { cblas_srotm(x.blas_len(), x.as_mut_ptr(), 0, y.as_mut_ptr(), 0, param.as_ptr()) }
        (x, y)
    }
    pub fn rotmg(mut d1: f32, mut d2: f32, mut x1: f32, y1: f32, mut param: [f32;5]) -> (f32, f32, f32, [f32;5]){
        unsafe { cblas_srotmg(&mut d1, &mut d2, &mut x1, y1, param.as_mut_ptr()) }
        (d1, d2, x1, param)
    }

    pub fn asum(&self) -> f32 {
        unsafe { cblas_sasum(self.blas_len(), self.as_ptr(), 0) }
    }
    pub fn axpy(&mut self, alpha: f32, rhs: Self) -> &mut Self {
        unsafe { cblas_saxpy(self.blas_len(), alpha, rhs.as_ptr(), 0, self.as_mut_ptr(), 0) }
        self
    }
    pub fn copy(&self, rhs: &mut Self){
        unsafe { cblas_scopy(self.blas_len(), self.as_ptr(), 0, rhs.as_mut_ptr(), 0) }
    }        
    pub fn dot(&self, rhs: Self) -> f32{
        unsafe { cblas_sdot(self.blas_len(), self.as_ptr(), 0, rhs.as_ptr(), 0) }
    }
    pub fn sdsdot(&self, scalar: f32, rhs: Self) -> f32{
        unsafe { cblas_sdsdot(self.blas_len(), scalar, self.as_ptr(), 0, rhs.as_ptr(), 0) }
    }
    pub fn dsdot(&self, rhs: Self) -> f64{
        unsafe { cblas_dsdot(self.blas_len(), self.as_ptr(), 0, rhs.as_ptr(), 0) }
    }
    pub fn nrm2(&self) -> f32{
        unsafe { cblas_snrm2(self.blas_len(), self.as_ptr(), 0) }
    }
    pub fn scal(&mut self, alpha: f32) -> &mut Self{
        unsafe { cblas_sscal(self.blas_len(), alpha, self.as_mut_ptr(), 0) };
        self
    }
    pub fn iamax(&self) -> usize {
        unsafe { cblas_isamax(self.blas_len(), self.as_ptr(), 0) }
    }
    pub fn iamin(&self) -> usize {
        unsafe { cblas_isamin(self.blas_len(), self.as_ptr(), 0) }
    }
}

