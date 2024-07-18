use cmake::Config;
use std::process::Command;

fn main() {
    Command::new("git")
        .args(&["submodule", "init"])
        .status()
        .expect("git submodule init failed");
    Command::new("git")
        .args(&["submodule", "update"])
        .status()
        .expect("git submodule update faild");

    let dst = Config::new("OpenBLAS").build();

    println!("cargo:rustc-link-search=native={}/lib", dst.display());
    println!("cargo:rustc-link-lib=static=openblas");

    let bindings = bindgen::Builder::default()
        .clang_args(&[format!("-I{}/include/openblas", dst.display())])
        .header(format!("{}/include/openblas/cblas.h", dst.display()))
        .header(format!("{}/include/openblas/lapack.h", dst.display()))
        .allowlist_file(format!("{}/include/openblas/lapack.h", dst.display()))
        .allowlist_file(format!("{}/include/openblas/cblas.h", dst.display()))
        .wrap_unsafe_ops(true)
        .generate_comments(true)
        .layout_tests(false)
        .generate()
        .expect("Unable to generate bindings");

    let out_path = std::env::current_dir().unwrap();
    bindings
        .write_to_file(out_path.join("src/blas/libopenblas.rs"))
        .expect("Couldn't write bindings!");
}
