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

    cc::Build::new()
        .cpp(true)
        .std("c++17")
        .pic(true)
        .file(tu.join("nodes/rescale_video.cpp"))
        .file(manifest_dir.join("cpp/stubs.cpp"))
        .include(manifest_dir.join("cpp/include"))
        .include(manifest_dir.join("include"))
        .include(&avcpp_inc)
        .include(workspace.join("deps/include"))
        .include(workspace.join("src"))
        .include(&ffmpeg_include)
        .flag("-Wno-unused-function")
        .compile("avplumber_cpp_nodes");

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
}

fn force_symlink(link: &Path, target: &Path) {
    if let Some(parent) = link.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    let _ = std::fs::remove_file(link);
    std::os::unix::fs::symlink(target, link)
        .unwrap_or_else(|err| panic!("symlink {} -> {}: {err}", link.display(), target.display()));
}
