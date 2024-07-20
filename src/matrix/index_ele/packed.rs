use std::ops::Index;

use core::num::Num;

use crate::matrix::{
    mtype::{ColMajor, Lower, Packed, RowMajor, Upper},
    Matrix,
};

macro_rules! index_cacl_1 {
    ($i: expr, $j: expr) => {
        $i + ($j + 1) * $j / 2
    };
}

macro_rules! index_cacl_2 {
    ($n: expr, $i: expr, $j: expr) => {
        $i + (2 * $n - $j - 1) * $j / 2
    };
}

impl<T: Num> Index<[usize; 2]> for Matrix<T, Packed, ColMajor, Upper> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let i = index.iter().min().unwrap();
        let j = index.iter().max().unwrap();
        &self.data[index_cacl_2!(self.row, i, j)]
    }
}

impl<T: Num> Index<[usize; 2]> for Matrix<T, Packed, ColMajor, Lower> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let i = index.iter().max().unwrap();
        let j = index.iter().min().unwrap();
        &self.data[index_cacl_1!(i, j)]
    }
}

impl<T: Num> Index<[usize; 2]> for Matrix<T, Packed, RowMajor, Upper> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let i = index.iter().min().unwrap();
        let j = index.iter().max().unwrap();
        &self.data[index_cacl_1!(i, j)]
    }
}

impl<T: Num> Index<[usize; 2]> for Matrix<T, Packed, RowMajor, Lower> {
    type Output = T;

    fn index(&self, index: [usize; 2]) -> &Self::Output {
        let i = index.iter().max().unwrap();
        let j = index.iter().min().unwrap();
        &self.data[index_cacl_2!(self.row, i, j)]
    }
}
