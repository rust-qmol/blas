use crate::blas::libopenblas::{
    blasint, cblas_dasum, cblas_daxpy, cblas_dcopy, cblas_ddot, cblas_dnrm2, cblas_drot,
    cblas_drotg, cblas_drotm, cblas_drotmg, cblas_dscal, cblas_dswap, cblas_idamax, cblas_idamin,
};

use super::{BlasLevel1FloatRaw, BlasLevel1Raw};

impl BlasLevel1Raw for f64 {
    type Pointer = f64;
    type Coeff = f64;
    type EleInput = Self::Coeff;

    unsafe fn as_ptr(&self) -> *const Self::Pointer {
        self
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut Self::Pointer {
        self
    }

    unsafe fn asum(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Pointer,
        incx: crate::blas::libopenblas::blasint,
    ) -> Self::Coeff {
        cblas_dasum(n, x, incx)
    }

    unsafe fn axpy(
        n: blasint,
        alpha: Self::EleInput,
        x: *const Self::Pointer,
        incx: blasint,
        y: *mut Self::Pointer,
        incy: blasint,
    ) {
        cblas_daxpy(n, alpha, x, incx, y, incy)
    }

    unsafe fn copy(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Pointer,
        incx: crate::blas::libopenblas::blasint,
        y: *mut Self::Pointer,
        incy: crate::blas::libopenblas::blasint,
    ) {
        cblas_dcopy(n, x, incx, y, incy)
    }

    unsafe fn nrm2(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Pointer,
        inc_x: crate::blas::libopenblas::blasint,
    ) -> Self::Coeff {
        cblas_dnrm2(n, x, inc_x)
    }

    unsafe fn rot(
        n: crate::blas::libopenblas::blasint,
        x: *mut Self::Pointer,
        inc_x: crate::blas::libopenblas::blasint,
        y: *mut Self::Pointer,
        inc_y: crate::blas::libopenblas::blasint,
        c: Self::Coeff,
        s: Self::Coeff,
    ) {
        cblas_drot(n, x, inc_x, y, inc_y, c, s)
    }

    unsafe fn rotg(
        a: *mut Self::Pointer,
        b: *mut Self::Pointer,
        c: *mut Self::Pointer,
        s: *mut Self::Pointer,
    ) {
        cblas_drotg(a, b, c, s)
    }

    unsafe fn scal(
        n: crate::blas::libopenblas::blasint,
        alpha: Self::EleInput,
        x: *mut Self::Pointer,
        inc_x: crate::blas::libopenblas::blasint,
    ) {
        cblas_dscal(n, alpha, x, inc_x)
    }

    unsafe fn swap(
        n: crate::blas::libopenblas::blasint,
        x: *mut Self::Pointer,
        incx: crate::blas::libopenblas::blasint,
        y: *mut Self::Pointer,
        incy: crate::blas::libopenblas::blasint,
    ) {
        cblas_dswap(n, x, incx, y, incy)
    }

    unsafe fn amax_index(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Pointer,
        incx: crate::blas::libopenblas::blasint,
    ) -> usize {
        cblas_idamax(n, x, incx)
    }

    unsafe fn amin_index(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Pointer,
        incx: crate::blas::libopenblas::blasint,
    ) -> usize {
        cblas_idamin(n, x, incx)
    }
}

impl BlasLevel1FloatRaw for f64 {
    unsafe fn dot(
        n: blasint,
        x: *const Self::Coeff,
        incx: blasint,
        y: *const Self::Coeff,
        incy: blasint,
    ) -> Self::Coeff {
        cblas_ddot(n, x, incx, y, incy)
    }

    unsafe fn rotm(
        n: blasint,
        x: *mut Self::Coeff,
        inc_x: blasint,
        y: *mut Self::Coeff,
        inc_y: blasint,
        p: *const Self::Coeff,
    ) {
        cblas_drotm(n, x, inc_x, y, inc_y, p)
    }

    unsafe fn rotmg(
        d1: *mut Self::Coeff,
        d2: *mut Self::Coeff,
        b1: *mut Self::Coeff,
        b2: Self::Coeff,
        p: *mut Self::Coeff,
    ) {
        cblas_drotmg(d1, d2, b1, b2, p)
    }
}
