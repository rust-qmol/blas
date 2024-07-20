use std::ops::Index;

use core::num::Num;

use crate::matrix::{
    mtype::{ColMajor, Full, Lower, RowMajor, Upper},
    Matrix,
};

macro_rules! index_cacl {
    ($n: expr, $i: expr, $j: expr) => {
        $n * $i + $j
    };
}

impl<T: Num> Index<[usize; 2]> for Matrix<T, Full, ColMajor, Upper> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.data[index_cacl!(self.col, index[0], index[1])]
    }
}

impl<T: Num> Index<[usize; 2]> for Matrix<T, Full, RowMajor, Upper> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.data[index_cacl!(self.row, index[1], index[0])]
    }
}

impl<T: Num> Index<[usize; 2]> for Matrix<T, Full, ColMajor, Lower> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.data[index_cacl!(self.col, index[0], index[1])]
    }
}

impl<T: Num> Index<[usize; 2]> for Matrix<T, Full, RowMajor, Lower> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        &self.data[index_cacl!(self.row, index[1], index[0])]
    }
}
