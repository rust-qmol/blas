#[test]
fn vector_f32_test() {
    use crate::blas::l1::{BlasLevel1, BlasLevel1Float};
    use crate::vector::Vector;
    let v1 = Vector::<f32>::from_vec(vec![0.1, 0.2, 0.3, 0.4, 0.5]);
    println!("v1: {:?}", v1);
    let v2 = Vector::<f32>::from_vec(vec![0.9, 0.8, 0.7, 0.6, 0.5]);
    println!("v2: {:?}", v2);
    println!("");

    println!("ops test:");
    println!("+: {:?}", v1.clone() + v2.clone());
    println!("-: {:?}", v1.clone() - v2.clone());
    println!("*: {:?}", v1.clone() * v2.clone());
    println!("/: {:?}", v1.clone() / v2.clone());
    println!("");

    println!("blas leve1 test:");
    println!("blas len: {:?}", v1.blas_len());
    println!("asum: {:?}", v1.asum());
    println!("iamax: {:?}", v1.iamax());
    println!("iamin: {:?}", v1.iamin());
    println!("nrm2: {:?}", v1.nrm2());
    println!("scal: {:?}", v1.clone().scal(2.0));
    println!("dot: {:?}", v1.dot(v2.clone()));
    println!("swap: {:?}", Vector::<f32>::swap(v1, v2));
}

#[test]
fn vector_c32_test() {
    use crate::blas::l1::{BlasLevel1, BlasLevel1Complex};
    use crate::vector::Vector;
    use core::num::{complex::Complex, zero::Zero};
    let v1 = Vector::<Complex<f32>>::from_vec(vec![
        Complex { re: 0.1, im: 0.2 },
        Complex { re: 0.3, im: 0.4 },
        Complex { re: 0.5, im: 0.6 },
        Complex { re: 0.7, im: 0.8 },
        Complex { re: 0.9, im: 1.0 },
    ]);
    println!("v1: {:?}", v1);
    let v2 = Vector::<Complex<f32>>::from_vec(vec![
        Complex { re: 0.2, im: 0.1 },
        Complex { re: 0.4, im: 0.3 },
        Complex { re: 0.6, im: 0.5 },
        Complex { re: 0.8, im: 0.7 },
        Complex { re: 1.0, im: 0.9 },
    ]);
    println!("v2: {:?}", v2);
    println!("");

    println!("ops test:");
    println!("+: {:?}", v1.clone() + v2.clone());
    println!("-: {:?}", v1.clone() - v2.clone());
    println!("*: {:?}", v1.clone() * v2.clone());
    println!("/: {:?}", v1.clone() / v2.clone());
    println!("");

    println!("blas leve1 test:");
    println!("blas len: {:?}", v1.blas_len());
    println!("asum: {:?}", v1.asum());
    println!("iamax: {:?}", v1.iamax());
    println!("iamin: {:?}", v1.iamin());
    println!("nrm2: {:?}", v1.nrm2());
    println!(
        "dotc_sub: {:?}",
        Vector::<Complex<f32>>::dotc_sub(v1.clone(), v2.clone(), Complex::<f32>::zero())
    );
    println!(
        "dotu_sub: {:?}",
        Vector::<Complex<f32>>::dotu_sub(v1.clone(), v2.clone(), Complex::<f32>::zero())
    );
    println!("swap: {:?}", Vector::<Complex<f32>>::swap(v1, v2));
}
