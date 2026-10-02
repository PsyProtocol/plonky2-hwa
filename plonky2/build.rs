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

    // NVTX v3 is header-only; a two-function C shim exposes push/pop to Rust.
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let shim = out.join("nvtx_shim.c");
    std::fs::write(
        &shim,
        "#include <nvtx3/nvToolsExt.h>\n\
         void plonky2_nvtx_push(const char *name) { nvtxRangePushA(name); }\n\
         void plonky2_nvtx_pop(void) { nvtxRangePop(); }\n",
    )
    .unwrap();
    let obj = out.join("nvtx_shim.o");
    let ok = std::process::Command::new(std::env::var("CC").unwrap_or_else(|_| "cc".into()))
        .args(["-O2", "-fPIC", "-c"])
        .arg(format!("-I{cuda}/include"))
        .arg(&shim)
        .arg("-o")
        .arg(&obj)
        .status()
        .expect("run C compiler for the NVTX shim")
        .success();
    assert!(ok, "compiling the NVTX shim failed");
    let lib = out.join("libplonky2_nvtx.a");
    let _ = std::fs::remove_file(&lib);
    assert!(std::process::Command::new("ar").arg("rcs").arg(&lib).arg(&obj).status().unwrap().success());
    println!("cargo:rustc-link-search=native={}", out.display());
    println!("cargo:rustc-link-lib=static=plonky2_nvtx");

    // Quotient polynomial kernel. PLONKY2_CUDA_ARCH selects the target (default sm_120).
    println!("cargo:rerun-if-changed=cuda/quotient.cu");
    println!("cargo:rerun-if-env-changed=PLONKY2_CUDA_ARCH");
    let arch = std::env::var("PLONKY2_CUDA_ARCH").unwrap_or_else(|_| "120".into());
    let qobj = out.join("quotient.o");
    let ok = std::process::Command::new(format!("{cuda}/bin/nvcc"))
        .args(["-O3", "-std=c++17", "-Xcompiler", "-fPIC", "-c"])
        .arg(format!("-arch=sm_{arch}"))
        .arg("cuda/quotient.cu")
        .arg("-o")
        .arg(&qobj)
        .status()
        .expect("run nvcc for cuda/quotient.cu")
        .success();
    assert!(ok, "compiling cuda/quotient.cu failed");
    let qlib = out.join("libplonky2_quotient.a");
    let _ = std::fs::remove_file(&qlib);
    assert!(std::process::Command::new("ar").arg("rcs").arg(&qlib).arg(&qobj).status().unwrap().success());
    println!("cargo:rustc-link-lib=static=plonky2_quotient");
    println!("cargo:rustc-link-lib=dylib=dl");
}
