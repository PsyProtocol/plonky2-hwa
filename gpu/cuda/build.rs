// Copyright (C) 2019-2022 Aleo Systems Inc.
// This file is part of the snarkVM library.

// The snarkVM library is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// The snarkVM library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with the snarkVM library. If not, see <https://www.gnu.org/licenses/>.

use std::env;

fn main() {
    println!("cargo:rerun-if-env-changed=NVCC");
    println!("cargo:rerun-if-env-changed=CUDA_ARCH");

    // The native kernels are only needed (and only linked) when the host runtime is enabled.
    // Without the `runtime` feature no device calls are made, so skip nvcc entirely and let the
    // crate build on machines without a CUDA toolchain.
    if env::var_os("CARGO_FEATURE_RUNTIME").is_none() {
        return;
    }

    let nvcc = match env::var("NVCC") {
        Ok(var) => which::which(var),
        Err(_) => which::which("nvcc"),
    };
    let nvcc = nvcc.unwrap_or_else(|_| {
        panic!(
            "plonky2_cuda `runtime` feature is enabled but `nvcc` was not found. \
             Install the CUDA toolkit or set the NVCC env var."
        )
    });

    // GPU architecture is configurable; default to sm_75 (Turing) but let callers target their card.
    let arch = env::var("CUDA_ARCH").unwrap_or_else(|_| "sm_75".to_string());

    let mut build = cc::Build::new();
    build.cuda(true);
    build.compiler(nvcc);
    build.no_default_flags(true);
    build.warnings(false);
    build.extra_warnings(false);
    build.flag("-O5");
    build.flag(&format!("-arch={arch}"));
    build.flag("-maxrregcount=255");
    build.file("plonky2_gpu.cu").compile("plonky2_cuda");

    println!("cargo:rerun-if-changed=plonky2_gpu.cu");
    println!("cargo:rerun-if-changed=plonky2_gpu_impl.cuh");
    println!("cargo:rerun-if-changed=def.cuh");
    println!("cargo:rerun-if-changed=constants.cuh");
    println!("cargo:rerun-if-env-changed=CXXFLAGS");
}
