//! avplumber_nodes — the media nodes, one node per file, node logic only.
//!
//! Everything these nodes are built out of lives in the framework crate:
//! the libav-facing helpers in [`avplumber_f7k::libav`] (codec lookup,
//! dictionaries, the send/receive pump), the node-authoring helpers in
//! [`avplumber_f7k::scaffold`], and the contracts themselves in
//! [`avplumber_f7k::graph`]. The dependency runs one way only — the framework
//! knows nothing about these nodes — so an embedder can write its own node set
//! against the same helpers without going near this crate.
//!
//! [`register_media_nodes`] installs the whole set on an
//! [`Instance`](avplumber_f7k::Instance); nodes that need libav are compiled
//! only with the `ffmpeg` feature.

#[cfg(all(
    feature = "ffmpeg",
    not(any(
        feature = "ffmpeg6",
        feature = "ffmpeg7",
        feature = "ffmpeg7_1",
        feature = "ffmpeg8"
    ))
))]
compile_error!(
    "feature `ffmpeg` selects no ABI on its own: build with `ffmpeg6`, `ffmpeg7`, \
     `ffmpeg7_1` or `ffmpeg8` (each implies `ffmpeg`), matching the FFmpeg this \
     crate links against"
);

use avplumber_f7k::Instance;

#[cfg(feature = "ffmpeg")]
pub mod decode;
#[cfg(feature = "ffmpeg")]
pub mod demux;
#[cfg(feature = "ffmpeg")]
pub mod encode;
#[cfg(feature = "ffmpeg")]
pub mod input;
pub mod mux;
pub mod null_sink;
#[cfg(feature = "ffmpeg")]
pub mod output;

/// Registers every media node type. Separate from `Instance::new` so an
/// embedder that only wants the substrate does not pay for libav.
pub fn register_media_nodes(inst: &Instance) {
    avplumber_f7k::register_spec::<null_sink::NullSinkSpec>(inst);
    // `mux` only orders timestamps and describes the container, so it needs no
    // libav and stays available in the default build.
    avplumber_f7k::register_spec::<mux::MuxSpec>(inst);
    #[cfg(feature = "ffmpeg")]
    {
        avplumber_f7k::register_spec::<input::InputSpec>(inst);
        avplumber_f7k::register_spec::<demux::DemuxSpec>(inst);
        avplumber_f7k::register_spec::<decode::VideoDecoderSpec>(inst);
        avplumber_f7k::register_spec::<decode::AudioDecoderSpec>(inst);
        avplumber_f7k::register_spec::<encode::VideoEncoderSpec>(inst);
        avplumber_f7k::register_spec::<encode::AudioEncoderSpec>(inst);
        avplumber_f7k::register_spec::<output::OutputSpec>(inst);
    }
}
