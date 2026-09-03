fn main() {
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
