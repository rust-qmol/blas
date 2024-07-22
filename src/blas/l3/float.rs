use crate::matrix::mtype::{Lower, Upper};
use crate::{
    blas::libopenblas::{cblas_dgemm, cblas_sgemm, CBLAS_LAYOUT},
    matrix::{
        mtype::{ColMajor, Full, RowMajor},
        Matrix,
    },
};

macro_rules! mm_raw_impl {
    ($layout_enum: ident, $layout_expr: expr, $uplo_enum: ident, $method: ident, $t: ident) => {
        impl Matrix<$t, Full, $layout_enum, $uplo_enum> {
            pub fn mm(alpha: $t, a: Self, b: Self, beta: $t, c: Self) {
                unsafe {
                    $method(
                        $layout_expr,
                        a.trans,
                        b.trans,
                        a.blas_row(),
                        b.blas_col(),
                        a.blas_col(),
                        alpha,
                        a.as_ptr(),
                        0,
                        b.as_ptr(),
                        0,
                        beta,
                        c.as_mut_ptr(),
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

mm_impl!(cblas_sgemm, f32);
mm_impl!(cblas_dgemm, f64);
