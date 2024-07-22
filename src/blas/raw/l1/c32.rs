use core::num::complex::Complex;
use std::{
    ffi::c_void,
    ptr::{addr_of, addr_of_mut},
};

use crate::blas::libopenblas::{
    blasint, cblas_caxpy, cblas_ccopy, cblas_cdotc_sub, cblas_cdotu_sub,
    cblas_crotg, cblas_cscal, cblas_csrot, cblas_cswap, cblas_icamax, cblas_icamin, cblas_scasum,
    cblas_scnrm2,
};

use super::{BlasLeve1ComplexRaw, BlasLeve1Raw};

impl BlasLeve1Raw for Complex<f32> {
    type Point = c_void;
    type Float = f32;
    type EleInput = *const Self::Point;

    unsafe fn as_ptr(&self) -> *const Self::Point {
        addr_of!(self) as *const Self::Point
    }

    unsafe fn as_mut_ptr(&mut self) -> *mut Self::Point {
        addr_of_mut!(*self) as *mut Self::Point
    }

    unsafe fn asum(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        incx: crate::blas::libopenblas::blasint,
    ) -> Self::Float {
        cblas_scasum(n, x, incx)
    }

    unsafe fn axpy(
        n: blasint,
        alpha: Self::EleInput,
        x: *const Self::Point,
        incx: blasint,
        y: *mut Self::Point,
        incy: blasint,
    ) {
        cblas_caxpy(n, alpha, x, incx, y, incy)
    }

    unsafe fn copy(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        incx: crate::blas::libopenblas::blasint,
        y: *mut Self::Point,
        incy: crate::blas::libopenblas::blasint,
    ) {
        cblas_ccopy(n, x, incx, y, incy)
    }

    unsafe fn nrm2(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        inc_x: crate::blas::libopenblas::blasint,
    ) -> Self::Float {
        cblas_scnrm2(n, x, inc_x)
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
        cblas_csrot(n, x, inc_x, y, inc_y, c, s)
    }

    unsafe fn rotg(
        a: *mut Self::Point,
        b: *mut Self::Point,
        c: *mut Self::Float,
        s: *mut Self::Point,
    ) {
        cblas_crotg(a, b, c, s)
    }

    unsafe fn scal(
        n: crate::blas::libopenblas::blasint,
        alpha: Self::EleInput,
        x: *mut Self::Point,
        inc_x: crate::blas::libopenblas::blasint,
    ) {
        cblas_cscal(n, alpha, x, inc_x)
    }

    unsafe fn swap(
        n: crate::blas::libopenblas::blasint,
        x: *mut Self::Point,
        incx: crate::blas::libopenblas::blasint,
        y: *mut Self::Point,
        incy: crate::blas::libopenblas::blasint,
    ) {
        cblas_cswap(n, x, incx, y, incy)
    }

    unsafe fn amax_index(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        incx: crate::blas::libopenblas::blasint,
    ) -> usize {
        cblas_icamax(n, x, incx)
    }

    unsafe fn amin_index(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        incx: crate::blas::libopenblas::blasint,
    ) -> usize {
        cblas_icamin(n, x, incx)
    }
}

impl BlasLeve1ComplexRaw for Complex<f32> {
    unsafe fn dotc_sub(
        n: blasint,
        x: *const Self::Point,
        incx: blasint,
        y: *const Self::Point,
        incy: blasint,
        ret: *mut Self::Point,
    ) {
        unsafe { cblas_cdotc_sub(n, x, incx, y, incy, ret) }
    }

    unsafe fn dotu_sub(
        n: blasint,
        x: *const Self::Point,
        incx: blasint,
        y: *const Self::Point,
        incy: blasint,
        ret: *mut Self::Point,
    ) {
        unsafe { cblas_cdotu_sub(n, x, incx, y, incy, ret) }
    }
}
