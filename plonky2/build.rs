// Links the zeknox CUDA library when the `cuda` feature is enabled.
fn main() {
    println!("cargo:rerun-if-env-changed=ZEKNOX_LIB_DIR");
    println!("cargo:rerun-if-env-changed=CUDA_HOME");
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let zeknox = std::env::var("ZEKNOX_LIB_DIR")
        .expect("feature `cuda` needs ZEKNOX_LIB_DIR (directory containing libzeknox.a built for this GPU)");
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/opt/cuda".into());
    println!("cargo:rerun-if-changed={zeknox}/libzeknox.a");
    println!("cargo:rustc-link-search=native={zeknox}");
    println!("cargo:rustc-link-search=native={cuda}/lib64");
    println!("cargo:rustc-link-lib=static=zeknox");
    println!("cargo:rustc-link-lib=dylib=cudart");
    println!("cargo:rustc-link-lib=dylib=stdc++");
    println!("cargo:rustc-link-lib=dylib=gomp");
}
