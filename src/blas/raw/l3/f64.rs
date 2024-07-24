use crate::blas::libopenblas::{
    cblas_dgemm, cblas_dsymm, cblas_dsyr2k, cblas_dsyrk, cblas_dtrmm, cblas_dtrsm,
};

use super::BlasLeve3Raw;

impl BlasLeve3Raw for f64 {
    type Pointer = f64;
    type Coeff = f64;

    unsafe fn gemm(
        order: crate::blas::libopenblas::CBLAS_LAYOUT,
        trans_a: crate::blas::libopenblas::CBLAS_TRANSPOSE,
        trans_b: crate::blas::libopenblas::CBLAS_TRANSPOSE,
        m: crate::blas::libopenblas::blasint,
        n: crate::blas::libopenblas::blasint,
        k: crate::blas::libopenblas::blasint,
        alpha: Self::Coeff,
        a: *const Self::Pointer,
        lda: crate::blas::libopenblas::blasint,
        b: *const Self::Pointer,
        ldb: crate::blas::libopenblas::blasint,
        beta: Self::Coeff,
        c: *mut Self::Pointer,
        ldc: crate::blas::libopenblas::blasint,
    ) {
        cblas_dgemm(
            order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        )
    }

    unsafe fn symm(
        order: crate::blas::libopenblas::CBLAS_LAYOUT,
        side: crate::blas::libopenblas::CBLAS_SIDE,
        uplo: crate::blas::libopenblas::CBLAS_UPLO,
        m: crate::blas::libopenblas::blasint,
        n: crate::blas::libopenblas::blasint,
        alpha: Self::Coeff,
        a: *const Self::Pointer,
        lda: crate::blas::libopenblas::blasint,
        b: *const Self::Pointer,
        ldb: crate::blas::libopenblas::blasint,
        beta: Self::Coeff,
        c: *mut Self::Pointer,
        ldc: crate::blas::libopenblas::blasint,
    ) {
        cblas_dsymm(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
    }

    unsafe fn syrk(
        order: crate::blas::libopenblas::CBLAS_LAYOUT,
        uplo: crate::blas::libopenblas::CBLAS_UPLO,
        trans: crate::blas::libopenblas::CBLAS_TRANSPOSE,
        n: crate::blas::libopenblas::blasint,
        k: crate::blas::libopenblas::blasint,
        alpha: Self::Coeff,
        a: *const Self::Pointer,
        lda: crate::blas::libopenblas::blasint,
        beta: Self::Coeff,
        c: *mut Self::Pointer,
        ldc: crate::blas::libopenblas::blasint,
    ) {
        cblas_dsyrk(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
    }

    unsafe fn syr2k(
        order: crate::blas::libopenblas::CBLAS_LAYOUT,
        uplo: crate::blas::libopenblas::CBLAS_UPLO,
        trans: crate::blas::libopenblas::CBLAS_TRANSPOSE,
        n: crate::blas::libopenblas::blasint,
        k: crate::blas::libopenblas::blasint,
        alpha: Self::Coeff,
        a: *const Self::Pointer,
        lda: crate::blas::libopenblas::blasint,
        b: *const Self::Pointer,
        ldb: crate::blas::libopenblas::blasint,
        beta: Self::Coeff,
        c: *mut Self::Pointer,
        ldc: crate::blas::libopenblas::blasint,
    ) {
        cblas_dsyr2k(
            order, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        )
    }

    unsafe fn trmm(
        order: crate::blas::libopenblas::CBLAS_LAYOUT,
        side: crate::blas::libopenblas::CBLAS_SIDE,
        uplo: crate::blas::libopenblas::CBLAS_UPLO,
        trans_a: crate::blas::libopenblas::CBLAS_TRANSPOSE,
        diag: crate::blas::libopenblas::CBLAS_DIAG,
        m: crate::blas::libopenblas::blasint,
        n: crate::blas::libopenblas::blasint,
        alpha: Self::Coeff,
        a: *const Self::Pointer,
        lda: crate::blas::libopenblas::blasint,
        b: *mut Self::Pointer,
        ldb: crate::blas::libopenblas::blasint,
    ) {
        cblas_dtrmm(
            order, side, uplo, trans_a, diag, m, n, alpha, a, lda, b, ldb,
        )
    }

    unsafe fn trsm(
        order: crate::blas::libopenblas::CBLAS_LAYOUT,
        side: crate::blas::libopenblas::CBLAS_SIDE,
        uplo: crate::blas::libopenblas::CBLAS_UPLO,
        trans_a: crate::blas::libopenblas::CBLAS_TRANSPOSE,
        diag: crate::blas::libopenblas::CBLAS_DIAG,
        m: crate::blas::libopenblas::blasint,
        n: crate::blas::libopenblas::blasint,
        alpha: Self::Coeff,
        a: *const Self::Pointer,
        lda: crate::blas::libopenblas::blasint,
        b: *mut Self::Pointer,
        ldb: crate::blas::libopenblas::blasint,
    ) {
        cblas_dtrsm(
            order, side, uplo, trans_a, diag, m, n, alpha, a, lda, b, ldb,
        )
    }
}
