use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use core::num::Num;

use super::Vector;

macro_rules! vector_ops_raw_impl {
    ($bound_assign:ident, $method_assign:ident, $bound:ident, $method:ident) => {
        impl<T: Num> $bound_assign for Vector<T> {
            fn $method_assign(&mut self, rhs: Self) {
                self.data
                    .iter_mut()
                    .zip(rhs.data.iter())
                    .for_each(|(a, b)| $bound_assign::$method_assign(a, *b));
            }
        }

        impl<T: Num> $bound for Vector<T> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                let mut r = self;
                $bound_assign::$method_assign(&mut r, rhs);
                r
            }
        }
    };
}

vector_ops_raw_impl!(AddAssign, add_assign, Add, add);
vector_ops_raw_impl!(SubAssign, sub_assign, Sub, sub);
vector_ops_raw_impl!(MulAssign, mul_assign, Mul, mul);
vector_ops_raw_impl!(DivAssign, div_assign, Div, div);

impl<T: Num> Neg for Vector<T> {
    type Output = Vector<T>;

    fn neg(self) -> Self::Output {
        Self {
            data: self.data.iter().map(|x| -(*x)).collect(),
        }
    }
}
