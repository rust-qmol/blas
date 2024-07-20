use std::{
    ffi::c_void,
    ptr::{addr_of, addr_of_mut},
};

use core::num::Num;

use crate::blas::libopenblas::blasint;

use super::{
    mtype::{Band, ColMajor, Full, Lower, Packed, RowMajor, Upper},
    Matrix,
};

macro_rules! ptr_raw_impl {
    ($storage_enum: ident, $layout_enum: ident, $uplo_enum: ident) => {
        impl<T: Num> Matrix<T, $storage_enum, $layout_enum, $uplo_enum> {
            pub(crate) fn blas_row(&self) -> blasint {
                self.row as blasint
            }
            pub(crate) fn blas_col(&self) -> blasint {
                self.col as blasint
            }

            pub(crate) fn as_ptr(&self) -> *const T {
                self.data.as_ptr()
            }
            pub(crate) fn as_mut_ptr(&mut self) -> *mut T {
                self.data.as_mut_ptr() as *mut T
            }
            pub(crate) fn as_void_ptr(&self) -> *const c_void {
                addr_of!(self.data) as *const c_void
            }
            pub(crate) fn as_mut_void_ptr(&mut self) -> *mut c_void {
                addr_of_mut!(self.data) as *mut c_void
            }
        }
    };
}

macro_rules! ptr_impl {
    ($storage_enum: ident) => {
        ptr_raw_impl!($storage_enum, ColMajor, Upper);
        ptr_raw_impl!($storage_enum, ColMajor, Lower);
        ptr_raw_impl!($storage_enum, RowMajor, Upper);
        ptr_raw_impl!($storage_enum, RowMajor, Lower);
    };
}

ptr_impl!(Full);
ptr_impl!(Packed);
ptr_impl!(Band);
