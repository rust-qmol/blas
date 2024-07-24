mod c32;
mod c64;
mod f32;
mod f64;

use std::ffi::c_void;

use crate::blas::libopenblas::blasint;

pub trait BlasLevel1Raw {
    type Pointer;
    type Coeff;
    type EleInput;

    unsafe fn as_ptr(&self) -> *const Self::Pointer;
    unsafe fn as_mut_ptr(&mut self) -> *mut Self::Pointer;

    unsafe fn asum(n: blasint, x: *const Self::Pointer, incx: blasint) -> Self::Coeff;
    unsafe fn axpy(
        n: blasint,
        alpha: Self::EleInput,
        x: *const Self::Pointer,
        incx: blasint,
        y: *mut Self::Pointer,
        incy: blasint,
    );
    unsafe fn copy(
        n: blasint,
        x: *const Self::Pointer,
        incx: blasint,
        y: *mut Self::Pointer,
        incy: blasint,
    );
    unsafe fn nrm2(n: blasint, x: *const Self::Pointer, inc_x: blasint) -> Self::Coeff;
    unsafe fn rot(
        n: blasint,
        x: *mut Self::Pointer,
        inc_x: blasint,
        y: *mut Self::Pointer,
        inc_y: blasint,
        c: Self::Coeff,
        s: Self::Coeff,
    );
    unsafe fn rotg(
        a: *mut Self::Pointer,
        b: *mut Self::Pointer,
        c: *mut Self::Coeff,
        s: *mut Self::Pointer,
    );
    unsafe fn scal(n: blasint, alpha: Self::EleInput, x: *mut Self::Pointer, inc_x: blasint);
    unsafe fn swap(
        n: blasint,
        x: *mut Self::Pointer,
        incx: blasint,
        y: *mut Self::Pointer,
        incy: blasint,
    );
    unsafe fn amax_index(n: blasint, x: *const Self::Pointer, incx: blasint) -> usize;
    unsafe fn amin_index(n: blasint, x: *const Self::Pointer, incx: blasint) -> usize;
}

pub trait BlasLevel1FloatRaw
where
    Self: BlasLevel1Raw,
{
    unsafe fn dot(
        n: blasint,
        x: *const <Self as BlasLevel1Raw>::Pointer,
        incx: blasint,
        y: *const <Self as BlasLevel1Raw>::Pointer,
        incy: blasint,
    ) -> <Self as BlasLevel1Raw>::Coeff;
    unsafe fn rotm(
        n: blasint,
        x: *mut Self::Pointer,
        inc_x: blasint,
        y: *mut Self::Pointer,
        inc_y: blasint,
        p: *const Self::Pointer,
    );
    unsafe fn rotmg(
        d1: *mut Self::Pointer,
        d2: *mut Self::Pointer,
        b1: *mut Self::Pointer,
        b2: Self,
        p: *mut Self::Pointer,
    );
}

pub trait BlasLevel1ComplexRaw
where
    Self: Sized,
    Self: BlasLevel1Raw<Pointer = c_void>,
{
    unsafe fn dotc_sub(
        n: blasint,
        x: *const Self::Pointer,
        incx: blasint,
        y: *const Self::Pointer,
        incy: blasint,
        ret: *mut Self::Pointer,
    );
    unsafe fn dotu_sub(
        n: blasint,
        x: *const Self::Pointer,
        incx: blasint,
        y: *const Self::Pointer,
        incy: blasint,
        ret: *mut Self::Pointer,
    );
}
