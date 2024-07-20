use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use core::num::Num;

use super::{
    mtype::{Band, ColMajor, Full, Lower, Packed, RowMajor, Upper},
    Matrix,
};

macro_rules! vector_ops_raw_impl {
    ($bound_assign:ident, $method_assign:ident, $bound:ident, $method:ident, $storage_enum: ident, $layout_enum: ident, $uplo_enum: ident) => {
        impl<T: Num> $bound_assign for Matrix<T, $storage_enum, $layout_enum, $uplo_enum> {
            fn $method_assign(&mut self, rhs: Self) {
                self.data
                    .iter_mut()
                    .zip(rhs.data.iter())
                    .for_each(|(a, b)| $bound_assign::$method_assign(a, *b));
            }
        }

        impl<T: Num> $bound for Matrix<T, $storage_enum, $layout_enum, $uplo_enum> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                let mut r = self;
                $bound_assign::$method_assign(&mut r, rhs);
                r
            }
        }
    };
}

macro_rules! vector_ops_impl {
    ($storage_enum: ident, $layout_enum: ident, $uplo_enum: ident) => {
        vector_ops_raw_impl!(
            AddAssign,
            add_assign,
            Add,
            add,
            $storage_enum,
            $layout_enum,
            $uplo_enum
        );
        vector_ops_raw_impl!(
            SubAssign,
            sub_assign,
            Sub,
            sub,
            $storage_enum,
            $layout_enum,
            $uplo_enum
        );
        vector_ops_raw_impl!(
            MulAssign,
            mul_assign,
            Mul,
            mul,
            $storage_enum,
            $layout_enum,
            $uplo_enum
        );
        vector_ops_raw_impl!(
            DivAssign,
            div_assign,
            Div,
            div,
            $storage_enum,
            $layout_enum,
            $uplo_enum
        );

        impl<T: Num> Neg for Matrix<T, $storage_enum, $layout_enum, $uplo_enum> {
            type Output = Matrix<T, $storage_enum, $layout_enum, $uplo_enum>;

            fn neg(self) -> Self::Output {
                Self {
                    row: self.row,
                    col: self.col,
                    kl: self.kl,
                    ku: self.ku,
                    data: self.data.iter().map(|x| -(*x)).collect(),
                    trans: self.trans,
                    storage: std::marker::PhantomData,
                    layout: std::marker::PhantomData,
                    uplo: std::marker::PhantomData,
                }
            }
        }
    };
}

macro_rules! vector_ops_impl_for_other_storage {
    ($storage_enum: ident) => {
        vector_ops_impl!($storage_enum, ColMajor, Upper);
        vector_ops_impl!($storage_enum, ColMajor, Lower);
        vector_ops_impl!($storage_enum, RowMajor, Upper);
        vector_ops_impl!($storage_enum, RowMajor, Lower);
    };
}

vector_ops_impl_for_other_storage!(Full);
vector_ops_impl_for_other_storage!(Packed);
vector_ops_impl_for_other_storage!(Band);
