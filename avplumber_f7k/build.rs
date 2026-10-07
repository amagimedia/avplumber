include!("../build_support/ffmpeg_prefix.rs");

fn main() {
    configure_ffmpeg_prefix();
    if std::env::var_os("CARGO_FEATURE_CPP_NODES").is_some() {
        compile_cpp_nodes();
    }
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let src = manifest_dir.join("src/graph/capability.rs");
    let dest = manifest_dir.join("include/avplumber_ids.h");
    println!("cargo:rerun-if-changed={}", src.display());
    println!(
        "cargo:rerun-if-changed={}",
        manifest_dir.join("cbindgen.toml").display()
    );

    let config =
        cbindgen::Config::from_file(manifest_dir.join("cbindgen.toml")).expect("cbindgen.toml");
    let bindings = cbindgen::Builder::new()
        .with_config(config)
        .with_src(&src)
        .generate()
        .expect("cbindgen");
    let mut generated = Vec::new();
    bindings.write(&mut generated);
    let generated = String::from_utf8(generated).expect("cbindgen utf-8");
    match std::fs::read_to_string(&dest) {
        Ok(existing) if existing == generated => {}
        _ => std::fs::write(&dest, generated).expect("write avplumber_ids.h"),
    }
}

fn compile_cpp_nodes() {
    let manifest_dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let workspace = manifest_dir.parent().unwrap().to_path_buf();
    let archive = workspace.join("deps/avcpp/build/src/libavcpp.a");
    println!("cargo:rerun-if-changed={}", archive.display());
    if !archive.is_file() {
        panic!(
            "cpp_nodes requires {} built against the same FFmpeg as this crate \
             (FFMPEG_PREFIX / PKG_CONFIG_PATH; do not mix the distro libavutil).\n\
             Position-independent code is required to link the archive into Rust.\n\
             Makefile recipe (deps/avcpp/build, target avcpp-static), plus the PIC flag:\n\
             \n  \
             rm -rf deps/avcpp/build && mkdir -p deps/avcpp/build\n  \
             cd deps/avcpp/build && PKG_CONFIG_PATH=\"$PKG_CONFIG_PATH\" cmake \
             -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
             -DCMAKE_POSITION_INDEPENDENT_CODE=ON ..\n  \
             make -C deps/avcpp/build avcpp-static",
            archive.display()
        );
    }

    let out_dir = PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let tu = out_dir.join("cpp_tu");
    // `#include "node_common.hpp"` searches the source file's directory before
    // -I, and src/nodes/node_common.hpp is the C++ core header. Compiling a
    // symlink that lives elsewhere makes that search miss, so -I hits the shim.
    // The .cpp's `../util.hpp` and `../video_parameters.hpp` then resolve next
    // to that symlink.
    force_symlink(
        &tu.join("nodes/rescale_video.cpp"),
        &workspace.join("src/nodes/rescale_video.cpp"),
    );
    force_symlink(&tu.join("util.hpp"), &workspace.join("src/util.hpp"));
    force_symlink(
        &tu.join("video_parameters.hpp"),
        &workspace.join("src/video_parameters.hpp"),
    );
    let avcpp_inc = out_dir.join("avcpp_inc");
    force_symlink(&avcpp_inc.join("avcpp"), &workspace.join("deps/avcpp/src"));

    let ffmpeg_include = ffmpeg_include_dir().unwrap_or_else(|| {
        panic!("cpp_nodes could not find FFmpeg headers; set FFMPEG_PREFIX to the build this crate links")
    });

    for path in [
        manifest_dir.join("cpp/include/node_common.hpp"),
        manifest_dir.join("cpp/stubs.cpp"),
        workspace.join("src/nodes/rescale_video.cpp"),
        workspace.join("src/util.hpp"),
        workspace.join("src/video_parameters.hpp"),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .pic(true)
        .file(tu.join("nodes/rescale_video.cpp"))
        .file(manifest_dir.join("cpp/stubs.cpp"))
        .file(manifest_dir.join("cpp/instance_bridge.cpp"))
        .include(manifest_dir.join("cpp/include"))
        .include(manifest_dir.join("include"))
        .include(&avcpp_inc)
        .include(workspace.join("deps/include"))
        .include(workspace.join("src"))
        .include(&ffmpeg_include)
        .flag("-Wno-unused-function");
    if std::env::var_os("CARGO_FEATURE_CPP_CUDA").is_some() {
        compile_cuda_transform(&mut build, &manifest_dir, &workspace, &tu, &ffmpeg_include);
    }
    build.compile("avplumber_cpp_nodes");

    println!(
        "cargo:rustc-link-search=native={}",
        archive.parent().unwrap().display()
    );
    println!("cargo:rustc-link-lib=static=avcpp");
    if let Some(libdir) = ffmpeg_libdir() {
        println!("cargo:rustc-link-search=native={}", libdir.display());
    }
    // After libavcpp.a, so its undefined swscale/avutil refs resolve to the
    // same shared libraries the ffmpeg feature linked.
    for lib in [
        "swscale",
        "swresample",
        "avfilter",
        "avformat",
        "avcodec",
        "avdevice",
        "avutil",
    ] {
        println!("cargo:rustc-link-lib=dylib={lib}");
    }
    println!("cargo:rustc-link-lib=pthread");
    if std::env::var_os("CARGO_FEATURE_CPP_CUDA").is_some() {
        println!("cargo:rustc-link-lib=dl");
    }
}

fn compile_cuda_transform(
    build: &mut cc::Build,
    manifest_dir: &Path,
    workspace: &Path,
    tu: &Path,
    ffmpeg_include: &Path,
) {
    let nvcc = std::env::var_os("NVCC")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("/usr/local/cuda/bin/nvcc"));
    if !nvcc.is_file() {
        panic!(
            "cpp_cuda needs nvcc to build the compositor kernels (set NVCC). looked at {}",
            nvcc.display()
        );
    }
    let hw = workspace.join("src/nodes/hwaccel");
    let header = workspace.join("objs/src/nodes/hwaccel/cuda_rect_scale.ptx.h");
    embed_rect_kernel(&nvcc, &hw.join("cuda_rect_scale.cu"), &header);
    println!("cargo:rerun-if-changed={}", hw.join("cuda_rect_scale.cu").display());
    println!("cargo:rerun-if-changed={}", hw.join("cuda_rect_table.h").display());
    println!("cargo:rerun-if-changed={}", hw.join("graphic_color.h").display());

    // cuda_transform.cpp's quoted includes are relative to the file as named.
    // The symlink lives outside src/nodes, so ../node_common.hpp is the shim.
    let mirrored = tu.join("nodes/hwaccel");
    force_symlink(
        &mirrored.join("cuda_transform.cpp"),
        &hw.join("cuda_transform.cpp"),
    );
    for name in [
        "cuda_rect_frame.hpp",
        "cuda_rect_draw.hpp",
        "cuda_rect_texture.h",
        "cuda_rect_table.h",
        "graphic_color.h",
    ] {
        force_symlink(&mirrored.join(name), &hw.join(name));
    }
    force_symlink(
        &tu.join("nodes/node_common.hpp"),
        &manifest_dir.join("cpp/include/node_common.hpp"),
    );
    force_symlink(&tu.join("hwaccel.hpp"), &workspace.join("src/hwaccel.hpp"));
    force_symlink(&tu.join("avbuffer.hpp"), &workspace.join("src/avbuffer.hpp"));
    let prim = workspace.join("src/mixer/primitives");
    for name in [
        "compositor_color.hpp",
        "compositor_layers.hpp",
        "compositor_geometry.hpp",
        "pixel_layout.hpp",
    ] {
        force_symlink(&tu.join("mixer/primitives").join(name), &prim.join(name));
    }

    for path in [
        hw.join("cuda_transform.cpp"),
        hw.join("cuda_rect_draw.cpp"),
        workspace.join("deps/cuda_loader/cuda_drvapi_dynlink.c"),
        manifest_dir.join("cpp/cuda_loader_init.cpp"),
    ] {
        println!("cargo:rerun-if-changed={}", path.display());
    }

    build
        .file(mirrored.join("cuda_transform.cpp"))
        .file(hw.join("cuda_rect_draw.cpp"))
        .file(manifest_dir.join("cpp/cuda_loader_init.cpp"))
        .include(ffmpeg_include)
        .define("HAVE_CUDA_RECT_SCALE", "1");

    cc::Build::new()
        .pic(true)
        .file(workspace.join("deps/cuda_loader/cuda_drvapi_dynlink.c"))
        .include(workspace.join("deps/cuda_loader"))
        .include(workspace.join("deps/include"))
        .compile("avplumber_cuda_loader");
}

fn embed_rect_kernel(nvcc: &Path, cu: &Path, header: &Path) {
    let fatbin = header.with_extension("fatbin");
    if let Some(parent) = fatbin.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    // sm_75 is the T4. The PTX fallback is the newest of these, matching the
    // Makefile's CUDA_FATBIN_ARCHS / CUDA_FATBIN_PTX_ARCH.
    let status = Command::new(nvcc)
        .args([
            "-fatbin",
            "-gencode",
            "arch=compute_75,code=sm_75",
            "-gencode",
            "arch=compute_80,code=sm_80",
            "-gencode",
            "arch=compute_86,code=sm_86",
            "-gencode",
            "arch=compute_89,code=sm_89",
            "-gencode",
            "arch=compute_89,code=compute_89",
            "-o",
        ])
        .arg(&fatbin)
        .arg(cu)
        .status()
        .unwrap_or_else(|err| panic!("failed to run {}: {err}", nvcc.display()));
    if !status.success() {
        panic!("nvcc failed to build {}", cu.display());
    }
    let bytes = std::fs::read(&fatbin).unwrap_or_else(|err| panic!("read {}: {err}", fatbin.display()));
    if bytes.is_empty() {
        panic!("{} is empty", fatbin.display());
    }
    let mut text = String::from("const unsigned char avpl_rect_scale_ptx[] = {\n");
    for (i, byte) in bytes.iter().enumerate() {
        text.push_str(&format!("0x{byte:02x},"));
        if i % 16 == 15 {
            text.push('\n');
        }
    }
    text.push_str(&format!(
        "\n}};\nconst unsigned int avpl_rect_scale_ptx_len = {};\n",
        bytes.len()
    ));
    std::fs::write(header, text).unwrap_or_else(|err| panic!("write {}: {err}", header.display()));
}

fn force_symlink(link: &Path, target: &Path) {
    if let Some(parent) = link.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let _ = std::fs::remove_file(link);
    std::os::unix::fs::symlink(target, link)
        .unwrap_or_else(|err| panic!("symlink {} -> {}: {err}", link.display(), target.display()));
}
