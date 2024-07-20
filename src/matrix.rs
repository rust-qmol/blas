pub mod mtype;

mod index_ele;
mod ops;
mod ptr;

use core::num::{complex::Complex, Num};
use std::marker::PhantomData;

use crate::blas::libopenblas::CBLAS_TRANSPOSE;

use mtype::{Band, Full, Packed, Upper};

pub type GeMatrix<T, LAYOUT> = Matrix<T, Full, LAYOUT, Upper>;
pub type GbMatrix<T, LAYOUT> = Matrix<T, Band, LAYOUT, Upper>;

pub type SyMatrix<T, LAYOUT, UPLO> = Matrix<T, Full, LAYOUT, UPLO>;
pub type SpMatrix<T, LAYOUT, UPLO> = Matrix<T, Packed, LAYOUT, UPLO>;
pub type SbMatrix<T, LAYOUT, UPLO> = Matrix<T, Band, LAYOUT, UPLO>;

pub type HeMatrix<T, LAYOUT, UPLO> = Matrix<Complex<T>, Full, LAYOUT, UPLO>;
pub type HpMatrix<T, LAYOUT, UPLO> = Matrix<Complex<T>, Packed, LAYOUT, UPLO>;
pub type HbMatrix<T, LAYOUT, UPLO> = Matrix<Complex<T>, Band, LAYOUT, UPLO>;

pub type TrMatrix<T, LAYOUT, UPLO> = Matrix<T, Full, LAYOUT, UPLO>;
pub type TpMatrix<T, LAYOUT, UPLO> = Matrix<T, Packed, LAYOUT, UPLO>;
pub type TbMatrix<T, LAYOUT, UPLO> = Matrix<T, Band, LAYOUT, UPLO>;

pub struct Matrix<T: Num, S, LAYOUT, UPLO> {
    pub(crate) row: usize,
    pub(crate) col: usize,
    pub(crate) data: Vec<T>,
    pub(crate) kl: usize,
    pub(crate) ku: usize,
    pub(crate) trans: CBLAS_TRANSPOSE,
    // enum
    storage: PhantomData<S>,
    layout: PhantomData<LAYOUT>,
    uplo: PhantomData<UPLO>,
}
