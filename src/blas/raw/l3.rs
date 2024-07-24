mod c32;
mod c64;
mod f32;
mod f64;

use crate::blas::libopenblas::{
    blasint, CBLAS_DIAG, CBLAS_ORDER, CBLAS_SIDE, CBLAS_TRANSPOSE, CBLAS_UPLO,
};

pub trait BlasLeve3Raw {
    type Pointer;
    type Coeff;

    unsafe fn gemm(
        order: CBLAS_ORDER,
        trans_a: CBLAS_TRANSPOSE,
        trans_b: CBLAS_TRANSPOSE,
        m: blasint,
        n: blasint,
        k: blasint,
        alpha: Self::Coeff,
        a: *const Self::Pointer,
        lda: blasint,
        b: *const Self::Pointer,
        ldb: blasint,
        beta: Self::Coeff,
        c: *mut Self::Pointer,
        ldc: blasint,
    );

    unsafe fn symm(
        order: CBLAS_ORDER,
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        m: blasint,
        n: blasint,
        alpha: Self::Coeff,
        a: *const Self::Pointer,
        lda: blasint,
        b: *const Self::Pointer,
        ldb: blasint,
        beta: Self::Coeff,
        c: *mut Self::Pointer,
        ldc: blasint,
    );

    unsafe fn syrk(
        order: CBLAS_ORDER,
        uplo: CBLAS_UPLO,
        trans: CBLAS_TRANSPOSE,
        n: blasint,
        k: blasint,
        alpha: Self::Coeff,
        a: *const Self::Pointer,
        lda: blasint,
        beta: Self::Coeff,
        c: *mut Self::Pointer,
        ldc: blasint,
    );

    unsafe fn syr2k(
        order: CBLAS_ORDER,
        uplo: CBLAS_UPLO,
        trans: CBLAS_TRANSPOSE,
        n: blasint,
        k: blasint,
        alpha: Self::Coeff,
        a: *const Self::Pointer,
        lda: blasint,
        b: *const Self::Pointer,
        ldb: blasint,
        beta: Self::Coeff,
        c: *mut Self::Pointer,
        ldc: blasint,
    );

    unsafe fn trmm(
        order: CBLAS_ORDER,
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        trans_a: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        m: blasint,
        n: blasint,
        alpha: Self::Coeff,
        a: *const Self::Pointer,
        lda: blasint,
        b: *mut Self::Pointer,
        ldb: blasint,
    );

    unsafe fn trsm(
        order: CBLAS_ORDER,
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        trans_a: CBLAS_TRANSPOSE,
        diag: CBLAS_DIAG,
        m: blasint,
        n: blasint,
        alpha: Self::Coeff,
        a: *const Self::Pointer,
        lda: blasint,
        b: *mut Self::Pointer,
        ldb: blasint,
    );
}

trait BlasLeve3ComplexRaw
where
    Self: BlasLeve3Raw,
{
    type CoeffFloat;

    unsafe fn hemm(
        order: CBLAS_ORDER,
        side: CBLAS_SIDE,
        uplo: CBLAS_UPLO,
        m: blasint,
        n: blasint,
        alpha: Self::Coeff,
        a: *const Self::Pointer,
        lda: blasint,
        b: *const Self::Pointer,
        ldb: blasint,
        beta: Self::Coeff,
        c: *mut Self::Pointer,
        ldc: blasint,
    );

    unsafe fn herk(
        order: CBLAS_ORDER,
        uplo: CBLAS_UPLO,
        trans: CBLAS_TRANSPOSE,
        n: blasint,
        k: blasint,
        alpha: Self::CoeffFloat,
        a: *const Self::Pointer,
        lda: blasint,
        beta: Self::CoeffFloat,
        c: *mut Self::Pointer,
        ldc: blasint,
    );

    unsafe fn her2k(
        order: CBLAS_ORDER,
        uplo: CBLAS_UPLO,
        trans: CBLAS_TRANSPOSE,
        n: blasint,
        k: blasint,
        alpha: Self::Coeff,
        a: *const Self::Pointer,
        lda: blasint,
        b: *const Self::Pointer,
        ldb: blasint,
        beta: Self::CoeffFloat,
        c: *mut Self::Pointer,
        ldc: blasint,
    );
}
