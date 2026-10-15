include!("../build_support/ffmpeg_prefix.rs");
include!("../build_support/cpp_nodes.rs");

fn main() {
    configure_ffmpeg_prefix();
    if std::env::var_os("CARGO_FEATURE_CPP_NODES").is_some() {
        compile_cpp_nodes();
    }
}
