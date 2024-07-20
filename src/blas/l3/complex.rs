use crate::blas::libopenblas::{cblas_cgemm, cblas_zgemm};
use crate::matrix::mtype::{Lower, Upper};
use crate::{
    blas::libopenblas::CBLAS_LAYOUT,
    matrix::{
        mtype::{ColMajor, Full, RowMajor},
        Matrix,
    },
};

use crate::blas::__BindgenComplex;
use core::num::complex::Complex;

macro_rules! mm_raw_impl {
    ($layout_enum: ident, $layout_expr: expr, $uplo_enum: ident, $method: ident, $t: ident) => {
        impl Matrix<Complex<$t>, Full, $layout_enum, $uplo_enum> {
            pub fn mm(alpha: Complex<$t>, a: Self, b: Self, beta: Complex<$t>, mut c: Self) {
                let alpha_bind = __BindgenComplex::from_complex(alpha);
                let beta_bind = __BindgenComplex::from_complex(beta);
                unsafe {
                    $method(
                        $layout_expr,
                        a.trans,
                        b.trans,
                        a.blas_row(),
                        b.blas_col(),
                        a.blas_col(),
                        alpha_bind.as_ptr(),
                        a.as_void_ptr(),
                        0,
                        b.as_void_ptr(),
                        0,
                        beta_bind.as_ptr(),
                        c.as_mut_void_ptr(),
                        0,
                    )
                }
            }
        }
    };
}

macro_rules! mm_impl {
    ($method: ident, $t: ident) => {
        mm_raw_impl!(ColMajor, CBLAS_LAYOUT::CblasColMajor, Upper, $method, $t);
        mm_raw_impl!(ColMajor, CBLAS_LAYOUT::CblasColMajor, Lower, $method, $t);
        mm_raw_impl!(RowMajor, CBLAS_LAYOUT::CblasRowMajor, Upper, $method, $t);
        mm_raw_impl!(RowMajor, CBLAS_LAYOUT::CblasRowMajor, Lower, $method, $t);
    };
}

mm_impl!(cblas_cgemm, f32);
mm_impl!(cblas_zgemm, f64);
