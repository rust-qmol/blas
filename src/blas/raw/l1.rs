mod c32;
mod c64;
mod f32;
mod f64;

use std::ffi::c_void;

use crate::blas::libopenblas::blasint;

pub trait BlasLeve1Raw {
    type Point;
    type Float;
    type EleInput;

    unsafe fn as_ptr(&self) -> *const Self::Point;
    unsafe fn as_mut_ptr(&mut self) -> *mut Self::Point;

    unsafe fn asum(n: blasint, x: *const Self::Point, incx: blasint) -> Self::Float;
    unsafe fn axpy(
        n: blasint,
        alpha: Self::EleInput,
        x: *const Self::Point,
        incx: blasint,
        y: *mut Self::Point,
        incy: blasint,
    );
    unsafe fn copy(
        n: blasint,
        x: *const Self::Point,
        incx: blasint,
        y: *mut Self::Point,
        incy: blasint,
    );
    unsafe fn nrm2(n: blasint, x: *const Self::Point, inc_x: blasint) -> Self::Float;
    unsafe fn rot(
        n: blasint,
        x: *mut Self::Point,
        inc_x: blasint,
        y: *mut Self::Point,
        inc_y: blasint,
        c: Self::Float,
        s: Self::Float,
    );
    unsafe fn rotg(
        a: *mut Self::Point,
        b: *mut Self::Point,
        c: *mut Self::Float,
        s: *mut Self::Point,
    );
    unsafe fn scal(n: blasint, alpha: Self::EleInput, x: *mut Self::Point, inc_x: blasint);
    unsafe fn swap(
        n: blasint,
        x: *mut Self::Point,
        incx: blasint,
        y: *mut Self::Point,
        incy: blasint,
    );
    unsafe fn amax_index(n: blasint, x: *const Self::Point, incx: blasint) -> usize;
    unsafe fn amin_index(n: blasint, x: *const Self::Point, incx: blasint) -> usize;
}

pub trait BlasLeve1FloatRaw
where
    Self: BlasLeve1Raw,
{
    unsafe fn dot(
        n: blasint,
        x: *const <Self as BlasLeve1Raw>::Point,
        incx: blasint,
        y: *const <Self as BlasLeve1Raw>::Point,
        incy: blasint,
    ) -> <Self as BlasLeve1Raw>::Float;
    unsafe fn rotm(
        n: blasint,
        x: *mut Self::Point,
        inc_x: blasint,
        y: *mut Self::Point,
        inc_y: blasint,
        p: *const Self::Point,
    );
    unsafe fn rotmg(
        d1: *mut Self::Point,
        d2: *mut Self::Point,
        b1: *mut Self::Point,
        b2: Self,
        p: *mut Self::Point,
    );
}

pub trait BlasLeve1ComplexRaw
where
    Self: Sized,
    Self: BlasLeve1Raw<Point = c_void>,
{
    unsafe fn dotc_sub(
        n: blasint,
        x: *const Self::Point,
        incx: blasint,
        y: *const Self::Point,
        incy: blasint,
        ret: *mut Self::Point,
    );
    unsafe fn dotu_sub(
        n: blasint,
        x: *const Self::Point,
        incx: blasint,
        y: *const Self::Point,
        incy: blasint,
        ret: *mut Self::Point,
    );
}
