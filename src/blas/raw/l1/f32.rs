use crate::blas::libopenblas::{
    blasint, cblas_isamax, cblas_isamin, cblas_sasum, cblas_saxpy, cblas_scopy, cblas_sdot,
    cblas_snrm2, cblas_srot, cblas_srotg, cblas_srotm, cblas_srotmg, cblas_sscal, cblas_sswap,
};

use super::{BlasLeve1FloatRaw, BlasLeve1Raw};

impl BlasLeve1Raw for f32 {
    type Point = f32;
    type Float = f32;
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
        cblas_sasum(n, x, incx)
    }

    unsafe fn axpy(
        n: blasint,
        alpha: Self::EleInput,
        x: *const Self::Point,
        incx: blasint,
        y: *mut Self::Point,
        incy: blasint,
    ) {
        cblas_saxpy(n, alpha, x, incx, y, incy)
    }

    unsafe fn copy(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        incx: crate::blas::libopenblas::blasint,
        y: *mut Self::Point,
        incy: crate::blas::libopenblas::blasint,
    ) {
        cblas_scopy(n, x, incx, y, incy)
    }

    unsafe fn nrm2(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        inc_x: crate::blas::libopenblas::blasint,
    ) -> Self::Float {
        cblas_snrm2(n, x, inc_x)
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
        cblas_srot(n, x, inc_x, y, inc_y, c, s)
    }

    unsafe fn rotg(
        a: *mut Self::Point,
        b: *mut Self::Point,
        c: *mut Self::Float,
        s: *mut Self::Point,
    ) {
        cblas_srotg(a, b, c, s)
    }

    unsafe fn scal(
        n: crate::blas::libopenblas::blasint,
        alpha: Self::EleInput,
        x: *mut Self::Point,
        inc_x: crate::blas::libopenblas::blasint,
    ) {
        cblas_sscal(n, alpha, x, inc_x)
    }

    unsafe fn swap(
        n: crate::blas::libopenblas::blasint,
        x: *mut Self::Point,
        incx: crate::blas::libopenblas::blasint,
        y: *mut Self::Point,
        incy: crate::blas::libopenblas::blasint,
    ) {
        cblas_sswap(n, x, incx, y, incy)
    }

    unsafe fn amax_index(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        incx: crate::blas::libopenblas::blasint,
    ) -> usize {
        cblas_isamax(n, x, incx)
    }

    unsafe fn amin_index(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        incx: crate::blas::libopenblas::blasint,
    ) -> usize {
        cblas_isamin(n, x, incx)
    }
}

impl BlasLeve1FloatRaw for f32 {
    unsafe fn dot(
        n: blasint,
        x: *const Self::Float,
        incx: blasint,
        y: *const Self::Float,
        incy: blasint,
    ) -> Self::Float {
        cblas_sdot(n, x, incx, y, incy)
    }

    unsafe fn rotm(
        n: blasint,
        x: *mut Self::Point,
        inc_x: blasint,
        y: *mut Self::Point,
        inc_y: blasint,
        p: *const Self::Point,
    ) {
        cblas_srotm(n, x, inc_x, y, inc_y, p)
    }

    unsafe fn rotmg(
        d1: *mut Self::Point,
        d2: *mut Self::Point,
        b1: *mut Self::Point,
        b2: Self::Point,
        p: *mut Self::Point,
    ) {
        cblas_srotmg(d1, d2, b1, b2, p)
    }
}
