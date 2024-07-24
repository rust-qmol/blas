use core::num::complex::Complex;
use std::os::raw::c_void;

use crate::blas::libopenblas::{
    cblas_zgemm, cblas_zhemm, cblas_zher2k, cblas_zherk, cblas_zsymm, cblas_zsyr2k, cblas_zsyrk,
    cblas_ztrmm, cblas_ztrsm,
};

use super::{BlasLeve3ComplexRaw, BlasLeve3Raw};

impl BlasLeve3Raw for Complex<f64> {
    type Pointer = c_void;
    type Coeff = *const c_void;

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
        cblas_zgemm(
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
        cblas_zsymm(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
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
        cblas_zsyrk(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
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
        cblas_zsyr2k(
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
        cblas_ztrmm(
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
        cblas_ztrsm(
            order, side, uplo, trans_a, diag, m, n, alpha, a, lda, b, ldb,
        )
    }
}

impl BlasLeve3ComplexRaw for Complex<f64> {
    type CoeffFloat = f64;

    unsafe fn hemm(
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
        cblas_zhemm(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
    }

    unsafe fn herk(
        order: crate::blas::libopenblas::CBLAS_LAYOUT,
        uplo: crate::blas::libopenblas::CBLAS_UPLO,
        trans: crate::blas::libopenblas::CBLAS_TRANSPOSE,
        n: crate::blas::libopenblas::blasint,
        k: crate::blas::libopenblas::blasint,
        alpha: Self::CoeffFloat,
        a: *const Self::Pointer,
        lda: crate::blas::libopenblas::blasint,
        beta: Self::CoeffFloat,
        c: *mut Self::Pointer,
        ldc: crate::blas::libopenblas::blasint,
    ) {
        cblas_zherk(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
    }

    unsafe fn her2k(
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
        beta: Self::CoeffFloat,
        c: *mut Self::Pointer,
        ldc: crate::blas::libopenblas::blasint,
    ) {
        cblas_zher2k(
            order, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        )
    }
}
