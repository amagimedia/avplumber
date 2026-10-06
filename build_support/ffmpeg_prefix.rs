// Shared by the package build scripts.
//
// A custom FFmpeg install is not on the default loader path, so the binaries,
// cdylibs and tests that link it need an rpath. `ffmpeg9` is also checked
// against the headers actually being compiled, because the feature flag and
// the distro pkg-config result can otherwise silently disagree.

use std::path::{Path, PathBuf};
use std::process::Command;

fn configure_ffmpeg_prefix() {
    println!("cargo:rerun-if-env-changed=FFMPEG_PREFIX");
    println!("cargo:rerun-if-env-changed=FFMPEG_INCLUDE_DIR");
    println!("cargo:rerun-if-env-changed=FFMPEG_LIBS_DIR");
    println!("cargo:rerun-if-env-changed=FFMPEG_PKG_CONFIG_PATH");
    println!("cargo:rerun-if-env-changed=PKG_CONFIG_PATH");

    if cfg!(unix) {
        if let Some(libdir) = ffmpeg_libdir() {
            let rpath = format!("-Wl,-rpath,{}", libdir.display());
            // Cargo rejects a link-arg instruction for a target kind the
            // package does not have, so match the manifest.
            let manifest = std::fs::read_to_string(
                PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap()).join("Cargo.toml"),
            )
            .unwrap_or_default();
            if manifest.contains("cdylib") {
                println!("cargo:rustc-cdylib-link-arg={rpath}");
            }
            if manifest.contains("[[bin]]") {
                println!("cargo:rustc-link-arg-bins={rpath}");
            }
            println!("cargo:rustc-link-arg-tests={rpath}");
        }
    }

    if std::env::var_os("CARGO_FEATURE_FFMPEG9").is_none() {
        return;
    }
    let include = ffmpeg_include_dir().unwrap_or_else(|| {
        panic!(
            "feature ffmpeg9 needs FFmpeg 9 headers (libavcodec 63). \
             Build them with scripts/build-ffmpeg.sh and source target/ffmpeg.env, \
             or point FFMPEG_PREFIX / FFMPEG_INCLUDE_DIR / FFMPEG_PKG_CONFIG_PATH \
             at an existing FFmpeg 9 install."
        )
    });
    let header = include.join("libavcodec/version_major.h");
    println!("cargo:rerun-if-changed={}", header.display());
    let major = libavcodec_major(&header).unwrap_or_else(|| {
        panic!(
            "feature ffmpeg9: could not read LIBAVCODEC_VERSION_MAJOR from {}",
            header.display()
        )
    });
    if major != 63 {
        panic!(
            "feature ffmpeg9 requires libavcodec major version 63 (FFmpeg 9); {} defines {major}. \
             The system FFmpeg is a different major version; source target/ffmpeg.env after \
             scripts/build-ffmpeg.sh, or pass the matching ffmpeg* feature instead.",
            header.display()
        );
    }
}

fn resolve_prefix() -> Option<PathBuf> {
    let raw = std::env::var("FFMPEG_PREFIX").ok()?;
    let path = PathBuf::from(&raw);
    if path.is_absolute() {
        return Some(path);
    }
    let mut dir = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    loop {
        let candidate = dir.join(&path);
        if candidate.exists() {
            return Some(candidate);
        }
        if !dir.pop() {
            break;
        }
    }
    None
}

fn ffmpeg_libdir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("FFMPEG_LIBS_DIR") {
        let dir = PathBuf::from(dir);
        if dir.is_dir() {
            return Some(dir);
        }
    }
    if let Some(prefix) = resolve_prefix() {
        let lib = prefix.join("lib");
        if lib.is_dir() {
            return Some(lib);
        }
    }
    if let Ok(pc) = std::env::var("FFMPEG_PKG_CONFIG_PATH") {
        let pc = PathBuf::from(pc);
        if pc.file_name().and_then(|name| name.to_str()) == Some("pkgconfig") {
            return pc.parent().map(Path::to_path_buf);
        }
    }
    None
}

fn ffmpeg_include_dir() -> Option<PathBuf> {
    if let Ok(dir) = std::env::var("FFMPEG_INCLUDE_DIR") {
        let dir = PathBuf::from(dir);
        if dir.join("libavcodec/version_major.h").is_file() {
            return Some(dir);
        }
    }
    if let Some(prefix) = resolve_prefix() {
        let include = prefix.join("include");
        if include.join("libavcodec/version_major.h").is_file() {
            return Some(include);
        }
    }
    let mut cmd = Command::new("pkg-config");
    cmd.args(["--variable=includedir", "libavcodec"]);
    if let Ok(pc) = std::env::var("FFMPEG_PKG_CONFIG_PATH") {
        cmd.env("PKG_CONFIG_PATH", pc);
    }
    let output = cmd.output().ok()?;
    if !output.status.success() {
        return None;
    }
    let dir = String::from_utf8(output.stdout).ok()?;
    let dir = PathBuf::from(dir.trim());
    dir.join("libavcodec/version_major.h")
        .is_file()
        .then_some(dir)
}

fn libavcodec_major(header: &Path) -> Option<u32> {
    let text = std::fs::read_to_string(header).ok()?;
    for line in text.lines() {
        let Some(rest) = line.trim().strip_prefix("#define") else {
            continue;
        };
        let mut parts = rest.split_whitespace();
        if parts.next() == Some("LIBAVCODEC_VERSION_MAJOR") {
            return parts.next()?.parse().ok();
        }
    }
    None
}
