mod index;
mod ops;

use core::num::Num;
use std::ffi::c_void;

use crate::blas::libopenblas::blasint;

#[repr(C)]
#[derive(Debug, Clone)]
pub struct Vector<T: Num> {
    pub(crate) data: Vec<T>,
}

impl<T: Num> Vector<T> {
    pub fn from_vec(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub(crate) fn blas_len(&self) -> blasint {
        self.data.len() as blasint
    }

    pub(crate) fn as_ptr(&self) -> *const T{
        self.data.as_ptr()
    }

    pub(crate) fn as_mut_ptr(&mut self) -> *mut T{
        self.data.as_mut_ptr()
    }

    pub(crate) fn as_void_ptr(&self) -> *const c_void{
        self.data.as_ptr() as *const c_void
    }

    pub(crate) fn as_mut_void_ptr(&mut self) -> *mut c_void{
        self.data.as_mut_ptr() as *mut c_void
    }
}
