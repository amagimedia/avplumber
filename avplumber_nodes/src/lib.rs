//! avplumber_nodes — the media nodes, one node per file, node logic only.
//!
//! Everything these nodes are built out of lives in the framework crate:
//! the libav-facing helpers in [`avplumber_f7k::libav`] (codec lookup,
//! dictionaries, the send/receive pump), the node-authoring helpers in
//! [`avplumber_f7k::node_api`], and the contracts themselves in
//! [`avplumber_f7k::graph`]. The dependency runs one way only — the framework
//! knows nothing about these nodes — so an embedder can write its own node set
//! against the same helpers without going near this crate.
//!
//! [`register_media_nodes`] installs the whole set on an
//! [`Instance`](avplumber_f7k::Instance). Rust nodes that need libav are
//! compiled with the `ffmpeg` feature. C++ nodes (`cpp_nodes`, `cpp_cuda`,
//! `cpp_mixer`) are compiled by this crate and linked from that same function.

#[cfg(all(
    feature = "ffmpeg",
    not(any(
        feature = "ffmpeg6",
        feature = "ffmpeg7",
        feature = "ffmpeg7_1",
        feature = "ffmpeg8",
        feature = "ffmpeg9"
    ))
))]
compile_error!(
    "feature `ffmpeg` selects no ABI on its own: build with `ffmpeg6`, `ffmpeg7`, \
     `ffmpeg7_1`, `ffmpeg8` or `ffmpeg9` (each implies `ffmpeg`), matching the \
     FFmpeg this crate links against"
);

use avplumber_f7k::Instance;

#[cfg(feature = "ffmpeg")]
pub mod bsf;
#[cfg(feature = "ffmpeg")]
pub mod decode;
#[cfg(feature = "ffmpeg")]
pub mod demux;
#[cfg(feature = "ffmpeg")]
pub mod encode;
#[cfg(feature = "ffmpeg")]
pub mod filter_video;
pub mod force_fps;
#[cfg(feature = "ffmpeg")]
pub mod force_keyframe;
#[cfg(feature = "ffmpeg")]
pub mod input;
pub mod mux;
pub mod null_sink;
#[cfg(feature = "ffmpeg")]
pub mod output;
pub mod realtime;
pub mod sentinel;

/// Registers every media node type. Separate from `Instance::new` so an
/// embedder that only wants the substrate does not pay for these nodes.
mod correction_cfg;

pub fn register_media_nodes(inst: &Instance) {
    #[cfg(feature = "cpp_nodes")]
    link_cpp_nodes(inst);
    avplumber_f7k::register_spec::<null_sink::NullSinkSpec>(inst);
    // `mux` only orders timestamps and describes the container, so it needs no
    // libav and stays available in the default build.
    avplumber_f7k::register_spec::<mux::MuxSpec>(inst);
    // `force_fps` only restamps and clones media, so it needs no libav either.
    avplumber_f7k::register_spec::<force_fps::ForceFpsSpec>(inst);
    // Likewise `realtime`: it paces by a clock and restamps, nothing more.
    avplumber_f7k::register_spec::<realtime::RealtimeSpec>(inst);
    avplumber_f7k::register_spec::<sentinel::SentinelSpec>(inst);
    #[cfg(feature = "ffmpeg")]
    {
        avplumber_f7k::register_spec::<force_keyframe::ForceKeyframeSpec>(inst);
        avplumber_f7k::register_spec::<input::InputSpec>(inst);
        avplumber_f7k::register_spec::<demux::DemuxSpec>(inst);
        avplumber_f7k::register_spec::<decode::VideoDecoderSpec>(inst);
        avplumber_f7k::register_spec::<decode::AudioDecoderSpec>(inst);
        avplumber_f7k::register_spec::<encode::VideoEncoderSpec>(inst);
        avplumber_f7k::register_spec::<encode::AudioEncoderSpec>(inst);
        avplumber_f7k::register_spec::<filter_video::FilterVideoSpec>(inst);
        avplumber_f7k::register_spec::<output::OutputSpec>(inst);
        avplumber_f7k::register_spec::<bsf::BsfSpec>(inst);
        avplumber_f7k::register_spec::<resample::ResampleSpec>(inst);
    }
}

#[cfg(feature = "ffmpeg")]
pub mod resample;

#[cfg(feature = "cpp_nodes")]
fn link_cpp_nodes(inst: &Instance) {
    // SAFETY: each symbol is defined by the translation unit this crate
    // compiled and returns 1 once that registrar is queued.
    let linked = unsafe { avp_cpp_link_rescale_video() };
    assert_eq!(linked, 1, "rescale_video C++ node was not linked");
    let linked = unsafe { avp_cpp_link_one_to_many() };
    assert_eq!(linked, 1, "one_to_many C++ node was not linked");
    #[cfg(feature = "cpp_cuda")]
    {
        let linked = unsafe { avp_cpp_link_cuda_transform() };
        assert_eq!(linked, 1, "cuda_transform C++ node was not linked");
    }
    #[cfg(feature = "cpp_mixer")]
    {
        let linked = unsafe { avp_cpp_link_mixer() };
        assert_eq!(linked, 1, "mixer commands were not linked");
    }

    avplumber_f7k::abi::cpp_bridge::install(avplumber_f7k::abi::cpp_bridge::CppBridge {
        bind: avp_cpp_bind_instance,
        on_drop: avp_cpp_instance_drop,
        cuda_init: cuda_init(),
        hwaccel_put: hwaccel_put(),
    });
    // `Instance::new` already replayed whatever static constructors queued.
    // The link calls above queue the same registrars without depending on
    // `.init_array`, so replay again for this instance.
    inst.replay_module_inits();
}

#[cfg(feature = "cpp_nodes")]
unsafe extern "C" {
    fn avp_cpp_link_rescale_video() -> i32;
    fn avp_cpp_link_one_to_many() -> i32;
    fn avp_cpp_bind_instance(wrapper: *mut std::ffi::c_void, inner: *mut std::ffi::c_void);
    fn avp_cpp_instance_drop(inner: *mut std::ffi::c_void);
}

#[cfg(feature = "cpp_cuda")]
unsafe extern "C" {
    fn avp_cpp_link_cuda_transform() -> i32;
    fn avp_cpp_cuda_init();
    fn avp_cpp_hwaccel_put(
        wrapper: *mut std::ffi::c_void,
        name: *const std::ffi::c_char,
        device: *mut std::ffi::c_void,
    ) -> i32;
}

#[cfg(feature = "cpp_mixer")]
unsafe extern "C" {
    fn avp_cpp_link_mixer() -> i32;
}

#[cfg(feature = "cpp_cuda")]
fn cuda_init() -> Option<unsafe extern "C" fn()> {
    Some(avp_cpp_cuda_init)
}

#[cfg(all(feature = "cpp_nodes", not(feature = "cpp_cuda")))]
fn cuda_init() -> Option<unsafe extern "C" fn()> {
    None
}

#[cfg(feature = "cpp_cuda")]
fn hwaccel_put() -> Option<
    unsafe extern "C" fn(
        *mut std::ffi::c_void,
        *const std::ffi::c_char,
        *mut std::ffi::c_void,
    ) -> i32,
> {
    Some(avp_cpp_hwaccel_put)
}

#[cfg(all(feature = "cpp_nodes", not(feature = "cpp_cuda")))]
fn hwaccel_put() -> Option<
    unsafe extern "C" fn(
        *mut std::ffi::c_void,
        *const std::ffi::c_char,
        *mut std::ffi::c_void,
    ) -> i32,
> {
    None
}
