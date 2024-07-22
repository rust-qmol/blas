use core::num::complex::Complex;
use std::{
    ffi::c_void,
    ptr::{addr_of, addr_of_mut},
};

use crate::blas::libopenblas::{
    blasint, cblas_dzasum, cblas_dznrm2, cblas_icamax, cblas_icamin, cblas_zaxpy, cblas_zcopy, cblas_zdotc_sub, cblas_zdotu_sub, cblas_zdrot, cblas_zrotg,
    cblas_zscal, cblas_zswap,
};

use super::{BlasLeve1ComplexRaw, BlasLeve1Raw};

impl BlasLeve1Raw for Complex<f64> {
    type Point = c_void;
    type Float = f64;
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
        cblas_dzasum(n, x, incx)
    }

    unsafe fn axpy(
        n: blasint,
        alpha: Self::EleInput,
        x: *const Self::Point,
        incx: blasint,
        y: *mut Self::Point,
        incy: blasint,
    ) {
        cblas_zaxpy(n, alpha, x, incx, y, incy)
    }

    unsafe fn copy(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        incx: crate::blas::libopenblas::blasint,
        y: *mut Self::Point,
        incy: crate::blas::libopenblas::blasint,
    ) {
        cblas_zcopy(n, x, incx, y, incy)
    }

    unsafe fn nrm2(
        n: crate::blas::libopenblas::blasint,
        x: *const Self::Point,
        inc_x: crate::blas::libopenblas::blasint,
    ) -> Self::Float {
        cblas_dznrm2(n, x, inc_x)
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
        cblas_zdrot(n, x, inc_x, y, inc_y, c, s)
    }

    unsafe fn rotg(
        a: *mut Self::Point,
        b: *mut Self::Point,
        c: *mut Self::Float,
        s: *mut Self::Point,
    ) {
        cblas_zrotg(a, b, c, s)
    }

    unsafe fn scal(
        n: crate::blas::libopenblas::blasint,
        alpha: Self::EleInput,
        x: *mut Self::Point,
        inc_x: crate::blas::libopenblas::blasint,
    ) {
        cblas_zscal(n, alpha, x, inc_x)
    }

    unsafe fn swap(
        n: crate::blas::libopenblas::blasint,
        x: *mut Self::Point,
        incx: crate::blas::libopenblas::blasint,
        y: *mut Self::Point,
        incy: crate::blas::libopenblas::blasint,
    ) {
        cblas_zswap(n, x, incx, y, incy)
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

impl BlasLeve1ComplexRaw for Complex<f64> {
    unsafe fn dotc_sub(
        n: blasint,
        x: *const Self::Point,
        incx: blasint,
        y: *const Self::Point,
        incy: blasint,
        ret: *mut Self::Point,
    ) {
        unsafe { cblas_zdotc_sub(n, x, incx, y, incy, ret) }
    }

    unsafe fn dotu_sub(
        n: blasint,
        x: *const Self::Point,
        incx: blasint,
        y: *const Self::Point,
        incy: blasint,
        ret: *mut Self::Point,
    ) {
        unsafe { cblas_zdotu_sub(n, x, incx, y, incy, ret) }
    }
}
