use core::num::complex::Complex;
use std::{
    ffi::c_void,
    ptr::{addr_of, addr_of_mut},
};

use crate::blas::libopenblas::{
    blasint, cblas_caxpy, cblas_ccopy, cblas_cdotc_sub, cblas_cdotu_sub, cblas_crotg, cblas_cscal,
    cblas_csrot, cblas_cswap, cblas_icamax, cblas_icamin, cblas_scasum, cblas_scnrm2,
};

use super::{BlasLevel1ComplexRaw, BlasLevel1Raw};

impl BlasLevel1Raw for Complex<f32> {
    type Pointer = c_void;
    type Coeff = f32;
    type EleInput = *const Self::Pointer;

    unsafe fn as_ptr(&self) -> *const Self::Pointer {
        addr_of!(self) as *const Self::Pointer
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut Self::Pointer {
        addr_of_mut!(*self) as *mut Self::Pointer
    }

    unsafe fn asum(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Pointer,
        incx: crate::blas::libopenblas::blasint,
    ) -> Self::Coeff {
        cblas_scasum(n, x, incx)
    }

    unsafe fn axpy(
        n: blasint,
        alpha: Self::EleInput,
        x: *const Self::Pointer,
        incx: blasint,
        y: *mut Self::Pointer,
        incy: blasint,
    ) {
        cblas_caxpy(n, alpha, x, incx, y, incy)
    }

    unsafe fn copy(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Pointer,
        incx: crate::blas::libopenblas::blasint,
        y: *mut Self::Pointer,
        incy: crate::blas::libopenblas::blasint,
    ) {
        cblas_ccopy(n, x, incx, y, incy)
    }

    unsafe fn nrm2(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Pointer,
        inc_x: crate::blas::libopenblas::blasint,
    ) -> Self::Coeff {
        cblas_scnrm2(n, x, inc_x)
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
        cblas_csrot(n, x, inc_x, y, inc_y, c, s)
    }

    unsafe fn rotg(
        a: *mut Self::Pointer,
        b: *mut Self::Pointer,
        c: *mut Self::Coeff,
        s: *mut Self::Pointer,
    ) {
        cblas_crotg(a, b, c, s)
    }

    unsafe fn scal(
        n: crate::blas::libopenblas::blasint,
        alpha: Self::EleInput,
        x: *mut Self::Pointer,
        inc_x: crate::blas::libopenblas::blasint,
    ) {
        cblas_cscal(n, alpha, x, inc_x)
    }

    unsafe fn swap(
        n: crate::blas::libopenblas::blasint,
        x: *mut Self::Pointer,
        incx: crate::blas::libopenblas::blasint,
        y: *mut Self::Pointer,
        incy: crate::blas::libopenblas::blasint,
    ) {
        cblas_cswap(n, x, incx, y, incy)
    }

    unsafe fn amax_index(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Pointer,
        incx: crate::blas::libopenblas::blasint,
    ) -> usize {
        cblas_icamax(n, x, incx)
    }

    unsafe fn amin_index(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Pointer,
        incx: crate::blas::libopenblas::blasint,
    ) -> usize {
        cblas_icamin(n, x, incx)
    }
}

impl BlasLevel1ComplexRaw for Complex<f32> {
    unsafe fn dotc_sub(
        n: blasint,
        x: *const Self::Pointer,
        incx: blasint,
        y: *const Self::Pointer,
        incy: blasint,
        ret: *mut Self::Pointer,
    ) {
        unsafe { cblas_cdotc_sub(n, x, incx, y, incy, ret) }
    }

    unsafe fn dotu_sub(
        n: blasint,
        x: *const Self::Pointer,
        incx: blasint,
        y: *const Self::Pointer,
        incy: blasint,
        ret: *mut Self::Pointer,
    ) {
        unsafe { cblas_cdotu_sub(n, x, incx, y, incy, ret) }
    }
}
