use core::num::Num;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use super::Vector;

macro_rules! vector_ops_raw_impl {
    ($vec_name:ident, $bound_assign:ident, $method_assign:ident, $bound:ident, $method:ident) => {
        impl<T: Num> $bound_assign for $vec_name<T> {
            fn $method_assign(&mut self, rhs: Self) {
                self.data
                    .iter_mut()
                    .zip(rhs.data.iter())
                    .for_each(|(mut a, b)| $bound_assign::$method_assign(a, *b));
            }
        }

        impl<T: Num> $bound for $vec_name<T> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                let mut r = self;
                $bound_assign::$method_assign(&mut r, rhs);
                r
            }
        }
    };
}

vector_ops_raw_impl!(Vector, AddAssign, add_assign, Add, add);
vector_ops_raw_impl!(Vector, SubAssign, sub_assign, Sub, sub);
vector_ops_raw_impl!(Vector, MulAssign, mul_assign, Mul, mul);
vector_ops_raw_impl!(Vector, DivAssign, div_assign, Div, div);

impl<T: Num> Neg for Vector<T> {
    type Output = Vector<T>;

    fn neg(self) -> Self::Output {
        Self {
            data: self.data.iter().map(|x| -(*x)).collect(),
        }
    }
}
