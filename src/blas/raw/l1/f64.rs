use crate::blas::libopenblas::{
    blasint, cblas_dasum, cblas_daxpy, cblas_dcopy, cblas_ddot, cblas_dnrm2, cblas_drot,
    cblas_drotg, cblas_drotm, cblas_drotmg, cblas_dscal, cblas_dswap, cblas_idamax, cblas_idamin,
};

use super::{BlasLeve1FloatRaw, BlasLeve1Raw};

impl BlasLeve1Raw for f64 {
    type Point = f64;
    type Float = f64;
    type EleInput = Self::Float;

    unsafe fn as_ptr(&self) -> *const Self::Point {
        self
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut Self::Point {
        self
    }

    unsafe fn asum(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        incx: crate::blas::libopenblas::blasint,
    ) -> Self::Float {
        cblas_dasum(n, x, incx)
    }

    unsafe fn axpy(
        n: blasint,
        alpha: Self::EleInput,
        x: *const Self::Point,
        incx: blasint,
        y: *mut Self::Point,
        incy: blasint,
    ) {
        cblas_daxpy(n, alpha, x, incx, y, incy)
    }

    unsafe fn copy(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        incx: crate::blas::libopenblas::blasint,
        y: *mut Self::Point,
        incy: crate::blas::libopenblas::blasint,
    ) {
        cblas_dcopy(n, x, incx, y, incy)
    }

    unsafe fn nrm2(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        inc_x: crate::blas::libopenblas::blasint,
    ) -> Self::Float {
        cblas_dnrm2(n, x, inc_x)
    }

    unsafe fn rot(
        n: crate::blas::libopenblas::blasint,
        x: *mut Self::Point,
        inc_x: crate::blas::libopenblas::blasint,
        y: *mut Self::Point,
        inc_y: crate::blas::libopenblas::blasint,
        c: Self::Float,
        s: Self::Float,
    ) {
        cblas_drot(n, x, inc_x, y, inc_y, c, s)
    }

    unsafe fn rotg(
        a: *mut Self::Point,
        b: *mut Self::Point,
        c: *mut Self::Float,
        s: *mut Self::Point,
    ) {
        cblas_drotg(a, b, c, s)
    }

    unsafe fn scal(
        n: crate::blas::libopenblas::blasint,
        alpha: Self::EleInput,
        x: *mut Self::Point,
        inc_x: crate::blas::libopenblas::blasint,
    ) {
        cblas_dscal(n, alpha, x, inc_x)
    }

    unsafe fn swap(
        n: crate::blas::libopenblas::blasint,
        x: *mut Self::Point,
        incx: crate::blas::libopenblas::blasint,
        y: *mut Self::Point,
        incy: crate::blas::libopenblas::blasint,
    ) {
        cblas_dswap(n, x, incx, y, incy)
    }

    unsafe fn amax_index(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        incx: crate::blas::libopenblas::blasint,
    ) -> usize {
        cblas_idamax(n, x, incx)
    }

    unsafe fn amin_index(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        incx: crate::blas::libopenblas::blasint,
    ) -> usize {
        cblas_idamin(n, x, incx)
    }
}

impl BlasLeve1FloatRaw for f64 {
    unsafe fn dot(
        n: blasint,
        x: *const Self::Float,
        incx: blasint,
        y: *const Self::Float,
        incy: blasint,
    ) -> Self::Float {
        cblas_ddot(n, x, incx, y, incy)
    }

    unsafe fn rotm(
        n: blasint,
        x: *mut Self::Float,
        inc_x: blasint,
        y: *mut Self::Float,
        inc_y: blasint,
        p: *const Self::Float,
    ) {
        cblas_drotm(n, x, inc_x, y, inc_y, p)
    }

    unsafe fn rotmg(
        d1: *mut Self::Float,
        d2: *mut Self::Float,
        b1: *mut Self::Float,
        b2: Self::Float,
        p: *mut Self::Float,
    ) {
        cblas_drotmg(d1, d2, b1, b2, p)
    }
}
