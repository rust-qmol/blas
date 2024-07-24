use core::num::complex::Complex;
use std::os::raw::c_void;

use crate::blas::libopenblas::{
    cblas_cgemm, cblas_chemm, cblas_cher2k, cblas_cherk, cblas_csymm, cblas_csyr2k, cblas_csyrk,
    cblas_ctrmm, cblas_ctrsm,
};

use super::{BlasLeve3ComplexRaw, BlasLeve3Raw};

impl BlasLeve3Raw for Complex<f32> {
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
        cblas_cgemm(
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
        cblas_csymm(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
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
        cblas_csyrk(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
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
        cblas_csyr2k(
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
        cblas_ctrmm(
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
        cblas_ctrsm(
            order, side, uplo, trans_a, diag, m, n, alpha, a, lda, b, ldb,
        )
    }
}

impl BlasLeve3ComplexRaw for Complex<f32> {
    type CoeffFloat = f32;

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
        cblas_chemm(order, side, uplo, m, n, alpha, a, lda, b, ldb, beta, c, ldc)
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
        cblas_cherk(order, uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
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
        cblas_cher2k(
            order, uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        )
    }
}
