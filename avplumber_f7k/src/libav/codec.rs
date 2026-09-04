//! Codec lookup, format names, and the `Spec` ⇄ `AVCodecContext` conversions the
//! decode and encode nodes share.
//!
//! Everything a node needs to open a codec lives here, so `src/nodes/decode.rs`
//! and `src/nodes/encode.rs` stay about node behaviour.

use std::ffi::{CStr, CString};

use rsmpeg::avcodec::{AVCodec, AVCodecContext, AVCodecRef};
use rsmpeg::avutil::AVFrame;
use rusty_ffmpeg::ffi;

use crate::graph::buffer::AvpRational;
use crate::graph::spec::{ChannelLayout, PacketSpec, Spec};
use crate::libav::dict::Options;
use crate::libav::error::av_error;

/// `avcodec_get_name`: what libav calls this codec id, which is also the key a
/// `codec_map` is looked up by.
pub fn codec_name(codec_id: ffi::AVCodecID) -> String {
    let name = unsafe { ffi::avcodec_get_name(codec_id) };
    if name.is_null() {
        return "unknown".into();
    }
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

/// A decoder by name, or libav's default for `codec_id`.
///
/// C++ resolves this at `node.add` (it can read the input codecpar by walking up
/// the graph); under Spec flow the codec id only arrives at runtime, so an
/// unknown name fails at `group.start`.
pub fn find_decoder(
    name: Option<&str>,
    codec_id: ffi::AVCodecID,
) -> Result<AVCodecRef<'static>, String> {
    match name {
        Some(name) => {
            let c_name = CString::new(name).map_err(|_| "codec name contains a NUL".to_string())?;
            AVCodec::find_decoder_by_name(&c_name)
                .ok_or_else(|| format!("unknown decoder `{name}`"))
        }
        None => AVCodec::find_decoder(codec_id).ok_or_else(|| {
            format!(
                "no decoder for {} in this libavcodec build",
                codec_name(codec_id)
            )
        }),
    }
}

pub fn find_encoder(name: &str) -> Result<AVCodecRef<'static>, String> {
    let c_name = CString::new(name).map_err(|_| "codec name contains a NUL".to_string())?;
    AVCodec::find_encoder_by_name(&c_name).ok_or_else(|| format!("unknown encoder `{name}`"))
}

/// A format name that may carry the `?` prefix C++ uses for "prefer this, but a
/// codec that cannot do it is not an error".
pub struct FormatChoice {
    pub value: i32,
    pub optional: bool,
}

impl FormatChoice {
    fn split(spec: &str) -> (&str, bool) {
        match spec.strip_prefix('?') {
            Some(rest) => (rest, true),
            None => (spec, false),
        }
    }
}

/// `av_get_pix_fmt` with the `?`-optional prefix.
pub fn parse_pix_fmt(spec: &str) -> Result<FormatChoice, String> {
    let (name, optional) = FormatChoice::split(spec);
    let c_name = CString::new(name).map_err(|_| "pixel_format contains a NUL".to_string())?;
    let value = unsafe { ffi::av_get_pix_fmt(c_name.as_ptr()) };
    if value == ffi::AV_PIX_FMT_NONE {
        return Err(format!("unknown pixel format `{name}`"));
    }
    Ok(FormatChoice { value, optional })
}

/// `av_get_sample_fmt` with the `?`-optional prefix.
pub fn parse_sample_fmt(spec: &str) -> Result<FormatChoice, String> {
    let (name, optional) = FormatChoice::split(spec);
    let c_name = CString::new(name).map_err(|_| "sample_format contains a NUL".to_string())?;
    let value = unsafe { ffi::av_get_sample_fmt(c_name.as_ptr()) };
    if value == ffi::AV_SAMPLE_FMT_NONE {
        return Err(format!("unknown sample format `{name}`"));
    }
    Ok(FormatChoice { value, optional })
}

pub fn pix_fmt_name(value: i32) -> String {
    let name = unsafe { ffi::av_get_pix_fmt_name(value) };
    if name.is_null() {
        return format!("pix_fmt {value}");
    }
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

pub fn sample_fmt_name(value: i32) -> String {
    let name = unsafe { ffi::av_get_sample_fmt_name(value) };
    if name.is_null() {
        return format!("sample_fmt {value}");
    }
    unsafe { CStr::from_ptr(name) }
        .to_string_lossy()
        .into_owned()
}

/// `avcodec_open2` with the leftover-option reporting C++ does.
///
/// Raw ffi rather than [`AVCodecContext::open`]: that takes an owned
/// [`rsmpeg::avutil::AVDictionary`] and hands the remainder back as a new one,
/// while [`Options`] is the same `*mut *mut AVDictionary` protocol the rest of
/// the crate uses for `avformat_open_input`.
pub fn open_codec(
    ctx: &mut AVCodecContext,
    mut options: Options,
    node: &str,
) -> Result<(), String> {
    let ret =
        unsafe { ffi::avcodec_open2(ctx.as_mut_ptr(), std::ptr::null(), options.as_mut_ptr()) };
    if ret < 0 {
        return Err(format!("avcodec_open2 failed: {}", av_error(ret)));
    }
    options.warn_leftovers(node, "the codec");
    Ok(())
}

/// The decoder's input description: codec parameters plus the time base its
/// packets are stamped in.
pub fn apply_packet_spec(ctx: &mut AVCodecContext, spec: &PacketSpec) -> Result<(), String> {
    let codecpar = spec
        .codecpar
        .as_ref()
        .ok_or("the packet spec carries no codec parameters")?;
    ctx.apply_codecpar(codecpar)
        .map_err(|e| format!("avcodec_parameters_to_context failed: {e}"))?;
    // What `frame.time_base` and every rescale downstream is relative to.
    ctx.set_pkt_timebase(spec.time_base.into());
    if spec.time_base.num > 0 && spec.time_base.den > 0 {
        ctx.set_time_base(spec.time_base.into());
    }
    if spec.frame_rate.num > 0 && spec.frame_rate.den > 0 {
        ctx.set_framerate(spec.frame_rate.into());
    }
    Ok(())
}

/// What a decoded video frame is, for the encoder downstream.
///
/// Read from the frame rather than the context because that is what the sink
/// will actually receive: a decoder may negotiate a different pixel format than
/// its context advertises, and `get_format` can pick one later still.
pub fn video_spec_of(frame: &AVFrame, time_base: AvpRational, frame_rate: AvpRational) -> Spec {
    Spec::Video {
        width: frame.width,
        height: frame.height,
        pix_fmt: frame.format,
        frame_rate,
        sar: frame.sample_aspect_ratio.into(),
        time_base,
    }
}

pub fn audio_spec_of(frame: &AVFrame, time_base: AvpRational) -> Spec {
    Spec::Audio {
        sample_rate: frame.sample_rate,
        sample_fmt: frame.format,
        layout: ChannelLayout::from_av(&frame.ch_layout),
        time_base,
    }
}

/// Whether two specs describe the same format, so a re-delivered `Spec` is a
/// no-op instead of a codec reopen. Time base is part of it: an encoder's
/// `time_base` comes straight from here.
pub fn same_spec(a: &Spec, b: &Spec) -> bool {
    match (a, b) {
        (
            Spec::Video {
                width: aw,
                height: ah,
                pix_fmt: af,
                frame_rate: ar,
                sar: asar,
                time_base: atb,
            },
            Spec::Video {
                width: bw,
                height: bh,
                pix_fmt: bf,
                frame_rate: br,
                sar: bsar,
                time_base: btb,
            },
        ) => (aw, ah, af, ar, asar, atb) == (bw, bh, bf, br, bsar, btb),
        (
            Spec::Audio {
                sample_rate: ar,
                sample_fmt: af,
                layout: al,
                time_base: atb,
            },
            Spec::Audio {
                sample_rate: br,
                sample_fmt: bf,
                layout: bl,
                time_base: btb,
            },
        ) => (ar, af, al, atb) == (br, bf, bl, btb),
        (Spec::Packet(a), Spec::Packet(b)) => same_packet_spec(a, b),
        (Spec::Mux { streams: a }, Spec::Mux { streams: b }) => {
            a.len() == b.len()
                && a.iter().zip(b).all(|(a, b)| {
                    same_packet_spec(&a.spec, &b.spec) && a.id == b.id && a.metadata == b.metadata
                })
        }
        _ => false,
    }
}

/// One stream's codec parameters, shared by the two variants that carry them.
fn same_packet_spec(a: &PacketSpec, b: &PacketSpec) -> bool {
    a.codec_id == b.codec_id
        && a.time_base == b.time_base
        && a.frame_rate == b.frame_rate
        && a.extra_data == b.extra_data
}

/// The encoder's side of the same contract: a `Spec::Video`/`Spec::Audio` fills
/// in what C++ read from `IVideoFormatSource`/`IAudioMetadataSource` and
/// `ITimeBaseSource`.
pub fn apply_media_spec(ctx: &mut AVCodecContext, spec: &Spec) -> Result<(), String> {
    match spec {
        Spec::Video {
            width,
            height,
            pix_fmt,
            frame_rate,
            sar,
            time_base,
        } => {
            if *width <= 0 || *height <= 0 {
                return Err(format!("the input spec has size {width}x{height}"));
            }
            ctx.set_width(*width);
            ctx.set_height(*height);
            ctx.set_pix_fmt(*pix_fmt);
            ctx.set_sample_aspect_ratio((*sar).into());
            if frame_rate.num > 0 && frame_rate.den > 0 {
                ctx.set_framerate((*frame_rate).into());
            }
            set_time_base(ctx, *time_base)
        }
        Spec::Audio {
            sample_rate,
            sample_fmt,
            layout,
            time_base,
        } => {
            if *sample_rate <= 0 {
                return Err(format!("the input spec has sample rate {sample_rate}"));
            }
            ctx.set_sample_rate(*sample_rate);
            ctx.set_sample_fmt(*sample_fmt);
            // Safety: a freshly allocated context's layout is zeroed, and this
            // runs once, before open.
            let mut ch_layout = unsafe { std::mem::zeroed::<ffi::AVChannelLayout>() };
            unsafe { layout.apply_to(&mut ch_layout) }?;
            ctx.set_ch_layout(ch_layout);
            set_time_base(ctx, *time_base)
        }
        other => Err(format!(
            "expected a video or audio spec, got {:?}",
            other.media()
        )),
    }
}

fn set_time_base(ctx: &mut AVCodecContext, time_base: AvpRational) -> Result<(), String> {
    if time_base.num <= 0 || time_base.den <= 0 {
        return Err(format!(
            "the input spec has an unusable time base {}/{}",
            time_base.num, time_base.den
        ));
    }
    ctx.set_time_base(time_base.into());
    Ok(())
}

/// What an opened encoder produces, as the [`Spec::Packet`] `mux` and `output`
/// consume — the replacement for C++'s three-phase `initFromFormatContext`
/// handshake with the muxer.
pub fn packet_spec_of(ctx: &AVCodecContext) -> PacketSpec {
    PacketSpec::from_codecpar(
        &ctx.extract_codecpar(),
        ctx.time_base.into(),
        ctx.framerate.into(),
    )
}
