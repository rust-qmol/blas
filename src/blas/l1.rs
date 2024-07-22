use core::num::complex::Complex;
use std::ffi::c_void;

use crate::vector::Vector;

use super::{
    libopenblas::blasint,
    raw::l1::{BlasLeve1ComplexRaw, BlasLeve1FloatRaw, BlasLeve1Raw},
};

pub trait BlasLeve1
where
    Self: Clone,
{
    type Ele: BlasLeve1Raw;
    type Point = <Self::Ele as BlasLeve1Raw>::Point;
    type Float = <Self::Ele as BlasLeve1Raw>::Float;

    fn blas_len(&self) -> blasint;
    fn as_ptr(&self) -> *const <Self::Ele as BlasLeve1Raw>::Point;
    fn as_mut_ptr(&mut self) -> *mut <Self::Ele as BlasLeve1Raw>::Point;

    fn swap(mut x: Self, mut y: Self) -> (Self, Self) {
        unsafe { Self::Ele::swap(x.blas_len(), x.as_mut_ptr(), 1, y.as_mut_ptr(), 1) }
        (x, y)
    }

    fn rot(
        mut x: Self,
        mut y: Self,
        c: <Self::Ele as BlasLeve1Raw>::Float,
        s: <Self::Ele as BlasLeve1Raw>::Float,
    ) -> (Self, Self) {
        unsafe { Self::Ele::rot(x.blas_len(), x.as_mut_ptr(), 1, y.as_mut_ptr(), 1, c, s) }
        (x, y)
    }
    fn rotg(
        mut a: Self::Ele,
        mut b: Self::Ele,
        mut c: <Self::Ele as BlasLeve1Raw>::Float,
        mut s: Self::Ele,
    ) -> (
        Self::Ele,
        Self::Ele,
        <Self::Ele as BlasLeve1Raw>::Float,
        Self::Ele,
    ) {
        unsafe { Self::Ele::rotg(a.as_mut_ptr(), b.as_mut_ptr(), &mut c, s.as_mut_ptr()) };
        (a, b, c, s)
    }

    fn asum(&self) -> <Self::Ele as BlasLeve1Raw>::Float {
        unsafe { Self::Ele::asum(self.blas_len(), self.as_ptr(), 1) }
    }
    fn axpy(&mut self, alpha: <Self::Ele as BlasLeve1Raw>::EleInput, rhs: Self) -> &mut Self {
        unsafe {
            Self::Ele::axpy(
                self.blas_len(),
                alpha,
                rhs.as_ptr(),
                1,
                self.as_mut_ptr(),
                1,
            )
        }
        self
    }
    fn copy(&self, rhs: &mut Self) {
        unsafe { Self::Ele::copy(self.blas_len(), self.as_ptr(), 1, rhs.as_mut_ptr(), 1) }
    }
    fn nrm2(&self) -> <Self::Ele as BlasLeve1Raw>::Float {
        unsafe { Self::Ele::nrm2(self.blas_len(), self.as_ptr(), 1) }
    }
    fn scal(&mut self, alpha: <Self::Ele as BlasLeve1Raw>::EleInput) -> &mut Self {
        unsafe { Self::Ele::scal(self.blas_len(), alpha, self.as_mut_ptr(), 1) };
        self
    }
    fn iamax(&self) -> usize {
        unsafe { Self::Ele::amax_index(self.blas_len(), self.as_ptr(), 1) }
    }
    fn iamin(&self) -> usize {
        unsafe { Self::Ele::amin_index(self.blas_len(), self.as_ptr(), 1) }
    }
}

pub trait BlasLeve1Float<Ele: BlasLeve1FloatRaw<Point = Ele>>
where
    Self: BlasLeve1<Ele = Ele>,
{
    fn dot(&self, rhs: Self) -> <Ele as BlasLeve1Raw>::Float {
        unsafe { Ele::dot(self.blas_len(), self.as_ptr(), 1, rhs.as_ptr(), 1) }
    }
    fn rotm(mut x: Self, mut y: Self, param: [Ele; 5]) -> (Self, Self) {
        unsafe {
            Ele::rotm(
                x.blas_len(),
                x.as_mut_ptr(),
                1,
                y.as_mut_ptr(),
                1,
                param.as_ptr(),
            )
        }
        (x, y)
    }
    fn rotmg(
        mut d1: Ele,
        mut d2: Ele,
        mut x1: Ele,
        y1: Ele,
        mut param: [Ele; 5],
    ) -> (Ele, Ele, Ele, [Ele; 5]) {
        unsafe {
            Ele::rotmg(
                d1.as_mut_ptr(),
                d2.as_mut_ptr(),
                x1.as_mut_ptr(),
                y1,
                param.as_mut_ptr(),
            )
        }
        (d1, d2, x1, param)
    }
}

pub trait BlasLeve1Complex<Ele: BlasLeve1ComplexRaw>
where
    Self: BlasLeve1<Ele = Ele>,
{
    fn dotc_sub(x: Self, y: Self, mut ret: Ele) -> Ele {
        unsafe {
            Ele::dotc_sub(x.blas_len(), y.as_ptr(), 1, y.as_ptr(), 1, ret.as_mut_ptr());
        }
        ret
    }
    fn dotu_sub(x: Self, y: Self, mut ret: Ele) -> Ele {
        unsafe {
            Ele::dotu_sub(x.blas_len(), y.as_ptr(), 1, y.as_ptr(), 1, ret.as_mut_ptr());
        }
        ret
    }
}

macro_rules! vector_float_leve1_impl {
    ($t: ident) => {
        impl BlasLeve1 for Vector<$t> {
            type Ele = $t;

            type Point = <Self::Ele as BlasLeve1Raw>::Point;

            type Float = <Self::Ele as BlasLeve1Raw>::Float;

            fn blas_len(&self) -> blasint {
                self.data.len() as blasint
            }

            fn as_ptr(&self) -> *const <Self::Ele as BlasLeve1Raw>::Point {
                self.data.as_ptr()
            }

            fn as_mut_ptr(&mut self) -> *mut <Self::Ele as BlasLeve1Raw>::Point {
                self.data.as_mut_ptr()
            }
        }

        impl BlasLeve1Float<$t> for Vector<$t> {}
    };
}

vector_float_leve1_impl!(f32);
vector_float_leve1_impl!(f64);

macro_rules! vector_complex_leve1_impl {
    ($t: ident) => {
        impl BlasLeve1 for Vector<Complex<$t>> {
            type Ele = Complex<$t>;

            type Point = <Self::Ele as BlasLeve1Raw>::Point;

            type Float = <Self::Ele as BlasLeve1Raw>::Float;

            fn blas_len(&self) -> blasint {
                self.data.len() as blasint
            }

            fn as_ptr(&self) -> *const <Self::Ele as BlasLeve1Raw>::Point {
                self.data.as_ptr() as *const c_void
            }

            fn as_mut_ptr(&mut self) -> *mut <Self::Ele as BlasLeve1Raw>::Point {
                self.data.as_mut_ptr() as *mut c_void
            }
        }

        impl BlasLeve1Complex<Complex<$t>> for Vector<Complex<$t>> {}
    };
}

vector_complex_leve1_impl!(f32);
vector_complex_leve1_impl!(f64);
