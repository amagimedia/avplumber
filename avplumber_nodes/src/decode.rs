//! `dec_video` / `dec_audio` — libavcodec decoding. Port of C++
//! `src/nodes/decoders.cpp`.
//!
//! The decoder opens on the [`Spec::Packet`] its producer publishes, which is
//! what replaces the C++ `findNodeUp<IStreamsInput>()` walk and the
//! `InputStreamMetadata` the C++ decoder read its `AVStream` from. It then
//! publishes the [`Spec::Video`]/[`Spec::Audio`] of the **first decoded frame**,
//! so an encoder downstream never has to ask what it is receiving.
//!
//! Seek precision is in band: the `resume_at` a [`EdgeEvent::FlushStop`]
//! carries is the position the source aimed for, and frames the codec produces
//! below it are dropped (see [`InputHandler::on_flush_stop`]). That replaces
//! C++ `discardUntil`, which a seek coordinator called from outside the graph,
//! and with it `flush_magic`/`waiting_for_frame`, which only existed to make
//! that call land on the right frame.
//!
//! Hardware decoding is one device reference plus the pixel format the
//! `get_format` callback picks: `hw_device_ctx` names the device `hwaccel.init`
//! opened, `pixel_format: "cuda"` makes the callback choose the surface format,
//! and libavcodec allocates the frame pool itself. Frames then leave this node
//! on the device, and the [`Spec::Video`] published with them says which
//! software format the surface holds, so a hardware encoder downstream can
//! describe its own pool before the first frame reaches it.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use rsmpeg::avcodec::AVCodecContext;
use rusty_ffmpeg::ffi;
use serde_json::Value;

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::buffer::{AvpMediaType, AvpRational};
use avplumber_f7k::graph::edge::{Edge, EdgeEvent};
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::media::{FrameExt, Media, PacketExt, Ts};
use avplumber_f7k::graph::node::Blocked;
use avplumber_f7k::graph::pad::NodePads;
use avplumber_f7k::graph::spec::{PacketSpec, Spec};
use avplumber_f7k::graph::timebase::ts_cmp;
use avplumber_f7k::libav::codec;
use avplumber_f7k::libav::dict::Options;
use avplumber_f7k::libav::pump::{Progress, Pump, PumpKind};
use avplumber_f7k::scaffold::{Blocking, BlockingIo, InputHandler, PARK_TIMEOUT_MS, SingleInput};
use avplumber_f7k::services::hwaccel::HwDevice;

/// The parameters both decoder types share; C++ has one template for both.
///
/// The node keeps this whole struct rather than copying the fields it needs out
/// of it, so each parameter is declared exactly once. The two fields marked
/// build-time below are the exception a reader has to know about: after
/// [`DecoderParams::build`] they say what the script asked for, not what the node
/// runs on.
#[derive(Debug, serde::Deserialize)]
pub struct DecoderParams {
    /// A specific implementation, e.g. `h264_qsv`; libavcodec's default when absent.
    #[serde(default)]
    codec: Option<String>,
    /// `{"<input codec name>": "<implementation>"}`, consulted only when `codec`
    /// is absent.
    #[serde(default)]
    codec_map: BTreeMap<String, String>,
    /// Requested output format, `?`-prefixed to accept the decoder's own when it
    /// cannot oblige. Video only.
    ///
    /// Build-time only: parsed into [`Decoder::pixel_format`], which is what the
    /// `get_format` callback reads. Runtime code must not consult this string.
    #[serde(default)]
    pixel_format: Option<String>,
    /// Passed to `avcodec_open2`; unconsumed entries are logged.
    #[serde(default)]
    options: Option<Value>,
    /// The name of a device `hwaccel.init` opened. Resolved while the node is
    /// built, into [`Decoder::hwaccel`].
    #[serde(default)]
    hwaccel: Option<String>,
    /// Use the device only when the *input* stream is one of these codecs, by
    /// libavcodec's name for it (`"h264"`). A string or a list of them; absent
    /// means every codec. C++ has the same gate, because setting
    /// `hw_device_ctx` on a decoder that has no hardware path for the stream
    /// has been seen to corrupt frames.
    #[serde(default)]
    hwaccel_only_for_codecs: Option<Value>,
}

impl DecoderParams {
    fn build(self, name: &str, media: AvpMediaType, ctx: &BuildCtx<'_>) -> Result<Decoder, String> {
        let hwaccel = match &self.hwaccel {
            Some(device) => Some(resolve_hwaccel(ctx, device)?),
            None => None,
        };
        let hwaccel_codecs = match &self.hwaccel_only_for_codecs {
            Some(Value::String(one)) => Some(vec![one.clone()]),
            Some(Value::Array(many)) => Some(
                many.iter()
                    .map(|v| {
                        v.as_str()
                            .map(str::to_string)
                            .ok_or_else(|| "hwaccel_only_for_codecs holds codec names".to_string())
                    })
                    .collect::<Result<Vec<_>, _>>()?,
            ),
            Some(Value::Null) | None => None,
            Some(other) => {
                return Err(format!(
                    "hwaccel_only_for_codecs expects a codec name or a list of them, got {other}"
                ));
            }
        };
        if hwaccel.is_none() && hwaccel_codecs.is_some() {
            log::warn!("{name}: hwaccel_only_for_codecs without hwaccel does nothing");
        }
        let pixel_format = match (self.pixel_format.as_deref(), media) {
            (Some(spec), AvpMediaType::VIDEO) => {
                let choice = codec::parse_pix_fmt(spec)?;
                Some(Arc::new(PixelFormatRequest {
                    node: name.into(),
                    value: choice.value,
                    optional: choice.optional,
                }))
            }
            (Some(spec), _) => {
                log::warn!("{name}: ignoring pixel_format `{spec}` on a non-video decoder");
                None
            }
            (None, _) => None,
        };

        Ok(Decoder {
            io: BlockingIo::new(name),
            media,
            params: self,
            pixel_format,
            hwaccel,
            hwaccel_codecs,
            state: Mutex::new(State::new(media, name)),
        })
    }
}

/// `dec_video`. A newtype so both decoders share one parameter set while each
/// keeps its own `TYPE_NAME`.
#[derive(Debug, serde::Deserialize)]
#[serde(transparent)]
pub struct VideoDecoderSpec(DecoderParams);

impl NodeSpec for VideoDecoderSpec {
    const TYPE_NAME: &'static str = "dec_video";
    type Node = Blocking<Decoder>;

    fn build(self, name: &str, ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        self.0.build(name, AvpMediaType::VIDEO, ctx).map(Blocking)
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(transparent)]
pub struct AudioDecoderSpec(DecoderParams);

impl NodeSpec for AudioDecoderSpec {
    const TYPE_NAME: &'static str = "dec_audio";
    type Node = Blocking<Decoder>;

    fn build(self, name: &str, ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        self.0.build(name, AvpMediaType::AUDIO, ctx).map(Blocking)
    }
}

/// What `pixel_format` asks for, in a shape the libav callback can read.
struct PixelFormatRequest {
    node: String,
    value: ffi::AVPixelFormat,
    optional: bool,
}

/// The C++ `get_format` lambda. `opaque` is the [`PixelFormatRequest`] the node
/// keeps alive for exactly as long as it owns the context that points at it.
unsafe extern "C" fn choose_pix_fmt(
    ctx: *mut ffi::AVCodecContext,
    formats: *const ffi::AVPixelFormat,
) -> ffi::AVPixelFormat {
    let request = unsafe { &*((*ctx).opaque as *const PixelFormatRequest) };
    if formats.is_null() {
        return ffi::AV_PIX_FMT_NONE;
    }
    let mut offered = formats;
    while unsafe { *offered } != ffi::AV_PIX_FMT_NONE {
        if unsafe { *offered } == request.value {
            return request.value;
        }
        offered = unsafe { offered.add(1) };
    }
    log::warn!(
        "{}: the decoder does not support pixel_format {}",
        request.node,
        codec::pix_fmt_name(request.value)
    );
    if request.optional {
        let best = unsafe { *formats };
        log::info!(
            "{}: using the decoder's own {} instead",
            request.node,
            codec::pix_fmt_name(best)
        );
        return best;
    }
    ffi::AV_PIX_FMT_NONE
}

struct State {
    /// `None` until the input spec arrives; the codec cannot be opened before.
    ctx: Option<AVCodecContext>,
    /// The [`Spec::Packet`] `ctx` was opened for, so a re-delivered identical
    /// spec is a no-op.
    input_spec: Option<Spec>,
    /// The last spec published downstream, for the same reason.
    output_spec: Option<Spec>,
    /// Frames are stamped in the input's time base, which is also what
    /// `pkt_timebase` was set to.
    time_base: AvpRational,
    frame_rate: AvpRational,
    pump: Pump,
    /// C++ `last_pts_`, for the out-of-order warning.
    last_pts: Ts,
    /// Whether the packet last handed to the codec was a keyframe: C++ does not
    /// warn about out-of-order output right after one.
    last_key: bool,
    /// `Eof` seen: drain the codec, forward it, finish.
    eof: bool,
    /// A [`EdgeEvent::Drain`] was taken: the codec has been asked for
    /// everything it holds, and once that is out it is flushed so it can take
    /// packets again. Unlike `eof` this does not end the node.
    draining: bool,
    dropped_early: u64,
    /// The position the last `FlushStop` aimed for: frames below it are
    /// dropped, and the first frame at or past it clears it. C++
    /// `discard_until_`, but delivered on the edge instead of called from
    /// outside the graph.
    resume_at: Option<Ts>,
    /// Frames dropped under the current `resume_at`, for one log line when it
    /// clears.
    dropped_before_resume: u64,
}

impl State {
    fn new(media: AvpMediaType, node: &str) -> Self {
        Self {
            ctx: None,
            input_spec: None,
            output_spec: None,
            time_base: AvpRational::default(),
            frame_rate: AvpRational::default(),
            pump: Pump::new(PumpKind::Decode, media, node),
            last_pts: Ts::invalid(),
            last_key: false,
            eof: false,
            draining: false,
            dropped_early: 0,
            resume_at: None,
            dropped_before_resume: 0,
        }
    }
}

pub struct Decoder {
    io: BlockingIo,
    /// `VIDEO` or `AUDIO`: which node type this is.
    media: AvpMediaType,
    /// What the script asked for, verbatim: [`DecoderParams`] documents each
    /// field, and holding it whole is what keeps them from being declared twice.
    params: DecoderParams,
    /// `params.pixel_format`, parsed. Held by the node rather than by [`State`] so
    /// the pointer the context carries in `opaque` stays valid for the context's
    /// whole life.
    pixel_format: Option<Arc<PixelFormatRequest>>,
    /// The device `params.hwaccel` named, resolved at build.
    hwaccel: Option<Arc<HwDevice>>,
    /// `params.hwaccel_only_for_codecs`, parsed. `None` means every codec.
    hwaccel_codecs: Option<Vec<String>>,
    state: Mutex<State>,
}

/// The device a node named, or an error that says so at `node.add`.
fn resolve_hwaccel(ctx: &BuildCtx<'_>, name: &str) -> Result<Arc<HwDevice>, String> {
    ctx.hwaccel(name)
}

impl InputHandler for Decoder {
    /// C++ built its decoder from the input `AVStream`; here the same
    /// description arrives as a spec, and a *changed* one reopens the codec.
    fn on_spec(&self, spec: Spec) -> Result<Option<Spec>, NodeError> {
        let state = &mut *self.state.lock().unwrap();
        let Spec::Packet(packet_spec) = &spec else {
            log::warn!(
                "{}: ignoring a {:?} spec on the input; a decoder needs a packet spec",
                self.io.name,
                spec.media()
            );
            return Ok(None);
        };
        if let Some(published) = &state.input_spec {
            if codec::same_spec(published, &spec) {
                log::debug!(
                    "{}: input spec re-delivered unchanged, keeping the decoder",
                    self.io.name
                );
                return Ok(None);
            }
            log::info!(
                "{}: input format changed, reopening the decoder",
                self.io.name
            );
            if state.pump.has_output() {
                log::warn!(
                    "{}: dropping frames the previous decoder had already produced",
                    self.io.name
                );
            }
        }

        let ctx = self
            .open(packet_spec)
            .map_err(|message| self.io.error(NodePhase::Spec, message))?;
        state.time_base = packet_spec.time_base;
        state.frame_rate = ctx.framerate.into();
        state.ctx = Some(ctx);
        state.pump.reset();
        state.last_pts = Ts::invalid();
        state.last_key = false;
        state.input_spec = Some(spec);
        // Nothing to publish yet: the output format is read off the first
        // decoded frame, in `emit`.
        Ok(None)
    }

    fn on_buffer(&self, buffer: Media) -> Result<Option<Media>, NodeError> {
        let state = &mut *self.state.lock().unwrap();
        if state.ctx.is_none() {
            // Only reachable when a producer pushed packets before its spec;
            // there is nothing to decode them with yet.
            state.dropped_early += 1;
            return Ok(None);
        }
        state.last_key = matches!(&buffer, Media::Packet(packet) if packet.is_key());
        state.pump.load(buffer);
        Ok(None)
    }

    fn on_flush(&self) {
        let state = &mut *self.state.lock().unwrap();
        if let Some(ctx) = state.ctx.as_mut() {
            ctx.flush_buffers();
        }
        state.pump.reset();
        state.last_pts = Ts::invalid();
        // A new discontinuity supersedes whatever the previous one aimed for.
        state.resume_at = None;
        state.draining = false;
        state.dropped_before_resume = 0;
    }

    fn on_flush_stop(&self, resume_at: Option<Ts>) {
        let state = &mut *self.state.lock().unwrap();
        if let Some(ts) = resume_at {
            log::debug!(
                "{}: dropping decoded frames before {}/{}",
                self.io.name,
                ts.val,
                fmt_tb(ts.tb)
            );
        }
        state.resume_at = resume_at;
        state.dropped_before_resume = 0;
    }

    /// The codec keeps producing after its last input, so this only starts the
    /// drain; [`SingleInput::before_take`] finishes when it is over.
    fn on_eof(&self) -> Result<Blocked, NodeError> {
        let state = &mut *self.state.lock().unwrap();
        if let Some(ctx) = state.ctx.as_mut() {
            state.pump.flush(ctx);
        }
        state.eof = true;
        Ok(Blocked::Again)
    }

    /// The source has run out of packets for now, so whatever the codec is
    /// holding has to come out: NVDEC keeps the last frame until the next
    /// packet arrives, and a paused seek to the end of a recording is exactly
    /// the case where no next packet exists. The codec is flushed once it is
    /// empty, in `before_take`, so it can decode again.
    fn on_drain(&self) {
        let state = &mut *self.state.lock().unwrap();
        if state.eof || state.draining {
            return;
        }
        let Some(ctx) = state.ctx.as_mut() else {
            return;
        };
        state.pump.flush(ctx);
        state.draining = true;
    }

    fn on_closed(&self) {
        let state = &mut *self.state.lock().unwrap();
        self.log_drops(state);
    }
}

impl SingleInput for Decoder {
    fn io(&self) -> &BlockingIo {
        &self.io
    }

    fn pads(&self) -> NodePads {
        NodePads::siso(AvpMediaType::PACKET, self.media)
    }

    fn start(&self) {
        let state = &mut *self.state.lock().unwrap();
        state.pump.reset();
        if let Some(ctx) = state.ctx.as_mut() {
            ctx.flush_buffers();
        }
        state.last_pts = Ts::invalid();
        state.last_key = false;
        state.eof = false;
        state.dropped_early = 0;
        state.resume_at = None;
        state.dropped_before_resume = 0;
        // `input_spec`/`output_spec` deliberately survive: the codec is still
        // open for them, and the edge re-arms its latched spec on a restart, so
        // the re-delivery is recognised as "unchanged" instead of reopening.
    }

    /// Decoded frames go downstream before anything new is taken in, and a
    /// codec that refused the packet we are holding is offered it again.
    fn before_take(&self) -> Result<Option<Blocked>, NodeError> {
        let state = &mut *self.state.lock().unwrap();
        let out = self.io.output()?;
        if let Some(buffer) = state.pump.take_output() {
            let buffer = self.stamp(state, buffer);
            if self.below_resume_at(state, &buffer) {
                return Ok(Some(Blocked::Again));
            }
            return self.emit(state, &out, buffer).map(Some);
        }
        // Drained after `Eof`: pass the marker on and finish.
        if state.eof {
            self.log_drops(state);
            out.push_event(EdgeEvent::Eof);
            return Ok(Some(Blocked::Done));
        }
        // Drained after `Drain`: everything the codec held is out, so put it
        // back in a state that takes packets. libavcodec requires the flush
        // before anything can be sent again.
        if state.draining {
            if let Some(ctx) = state.ctx.as_mut() {
                ctx.flush_buffers();
            }
            state.pump.rearm();
            state.draining = false;
            log::debug!("{}: drained, ready for packets again", self.io.name);
        }
        if state.pump.is_loaded() {
            return Ok(Some(match self.drive(state)? {
                Progress::Moved => Blocked::Again,
                // Neither direction moved, which for a decoder means the codec
                // wants time rather than data. Park instead of spinning.
                Progress::Stalled => {
                    self.io.wait(PARK_TIMEOUT_MS);
                    Blocked::Again
                }
            }));
        }
        Ok(None)
    }
}

impl Decoder {
    fn drive(&self, state: &mut State) -> Result<Progress, NodeError> {
        let ctx = state
            .ctx
            .as_mut()
            .ok_or_else(|| self.io.error(NodePhase::Process, "the decoder is not open"))?;
        state
            .pump
            .drive(ctx)
            .map_err(|message| self.io.error(NodePhase::Process, message))
    }

    /// Publishes the stamped frame's spec if it is new, then pushes it.
    fn emit(
        &self,
        state: &mut State,
        out: &Arc<dyn Edge>,
        buffer: Media,
    ) -> Result<Blocked, NodeError> {
        self.publish_spec(state, out, &buffer);
        // Parks for room, unless a flush has reached the input meanwhile: then
        // this frame is from before the discontinuity and is dropped.
        let input = self.io.input()?;
        self.io.push_from(&input, out, buffer)
    }

    /// The in-band `discardUntil`: after a seek that could only land on a
    /// keyframe, what the codec produces before the position the source aimed
    /// for is discarded. The first frame at or past it clears the cutoff; a
    /// frame without a timestamp is let through, since it cannot be placed on
    /// either side.
    fn below_resume_at(&self, state: &mut State, buffer: &Media) -> bool {
        let Some(cutoff) = state.resume_at else {
            return false;
        };
        let ts = buffer.ts();
        if !ts.is_valid() {
            return false;
        }
        if ts_cmp(ts.val, ts.tb, cutoff.val, cutoff.tb).is_lt() {
            state.dropped_before_resume += 1;
            return true;
        }
        if state.dropped_before_resume > 0 {
            log::debug!(
                "{}: dropped {} frame(s) before {}/{}, resuming at {}",
                self.io.name,
                state.dropped_before_resume,
                cutoff.val,
                fmt_tb(cutoff.tb),
                ts.val
            );
        }
        state.resume_at = None;
        state.dropped_before_resume = 0;
        false
    }

    /// libavcodec 6 and 7/8 disagree about whether `frame.time_base` comes back
    /// filled, and everything downstream reads timestamps through it, so it is
    /// set here from `pkt_timebase` unconditionally.
    fn stamp(&self, state: &mut State, buffer: Media) -> Media {
        let mut buffer = buffer;
        if let Media::Video(frame) | Media::Audio(frame) = &mut buffer {
            let ts = Ts {
                val: frame.pts,
                tb: state.time_base,
            };
            frame.set_ts(ts);
            if !state.last_key
                && state.last_pts.is_valid()
                && ts.is_valid()
                && ts_cmp(ts.val, ts.tb, state.last_pts.val, state.last_pts.tb).is_lt()
            {
                log::warn!(
                    "{}: got an out of order frame from the decoder: {} -> {}",
                    self.io.name,
                    state.last_pts.val,
                    ts.val
                );
            }
            if ts.is_valid() {
                state.last_pts = ts;
            }
        }
        buffer
    }

    /// The decoded format, read from the frame rather than the context: that is
    /// what the consumer will actually receive.
    fn publish_spec(&self, state: &mut State, out: &Arc<dyn Edge>, buffer: &Media) {
        let spec = match buffer {
            Media::Video(frame) => codec::video_spec_of(frame, state.time_base, state.frame_rate),
            Media::Audio(frame) => codec::audio_spec_of(frame, state.time_base),
            _ => return,
        };
        if let Some(published) = &state.output_spec {
            if codec::same_spec(published, &spec) {
                return;
            }
            log::info!(
                "{}: decoded format changed, re-publishing the spec: {spec:?}",
                self.io.name
            );
        } else {
            log::info!("{}: decoding to {spec:?}", self.io.name);
        }
        out.push_event(EdgeEvent::Spec(spec.clone()));
        state.output_spec = Some(spec);
    }

    fn open(&self, spec: &PacketSpec) -> Result<AVCodecContext, String> {
        let codecpar = spec
            .codecpar
            .as_ref()
            .ok_or("the input spec carries no codec parameters")?;
        let wanted = match self.media {
            AvpMediaType::AUDIO => ffi::AVMEDIA_TYPE_AUDIO,
            _ => ffi::AVMEDIA_TYPE_VIDEO,
        };
        if codecpar.codec_type != wanted {
            return Err(format!(
                "the input stream is not {}",
                if wanted == ffi::AVMEDIA_TYPE_AUDIO {
                    "audio"
                } else {
                    "video"
                }
            ));
        }

        let codec_id = spec.codec_id as ffi::AVCodecID;
        let input_name = codec::codec_name(codec_id);
        let implementation = match &self.params.codec {
            Some(name) => Some(name.clone()),
            None => match self.params.codec_map.get(&input_name) {
                Some(name) => {
                    log::info!(
                        "{}: detected codec {input_name}, using implementation {name}",
                        self.io.name
                    );
                    Some(name.clone())
                }
                None => {
                    if !self.params.codec_map.is_empty() {
                        log::info!(
                            "{}: detected codec {input_name}, not in codec_map, using the \
                             libavcodec default",
                            self.io.name
                        );
                    }
                    None
                }
            },
        };

        let codec = codec::find_decoder(implementation.as_deref(), codec_id)?;
        let mut ctx = AVCodecContext::new(&codec);
        codec::apply_packet_spec(&mut ctx, spec)?;
        if let Some(device) = &self.hwaccel {
            match &self.hwaccel_codecs {
                Some(allowed) if !allowed.iter().any(|c| *c == input_name) => {
                    log::info!(
                        "{}: {input_name} is not in hwaccel_only_for_codecs, decoding in software",
                        self.io.name
                    );
                }
                _ => {
                    // The device only; the frame pool is libavcodec's business,
                    // and `get_format` (from `pixel_format`) is what keeps the
                    // frames on it.
                    ctx.set_hw_device_ctx(device.device_ref());
                    log::info!(
                        "{}: decoding {input_name} on hardware device `{}`",
                        self.io.name,
                        device.name
                    );
                }
            }
        }
        if let Some(request) = &self.pixel_format {
            // Safety: `request` is owned by the node, which outlives every
            // context it builds, and rsmpeg does not use `opaque` itself.
            unsafe {
                rsmpeg::UnsafeDerefMut::deref_mut(&mut ctx).opaque =
                    Arc::as_ptr(request) as *mut c_void;
            }
            ctx.set_get_format(Some(choose_pix_fmt));
        }
        codec::open_codec(
            &mut ctx,
            Options::from_json(self.params.options.as_ref())?,
            &self.io.name,
        )?;
        log::info!(
            "{}: opened decoder {} for {input_name}",
            self.io.name,
            codec.name().to_string_lossy()
        );
        Ok(ctx)
    }

    fn log_drops(&self, state: &State) {
        if state.dropped_early > 0 {
            log::info!(
                "{}: dropped {} packet(s) that arrived before the input spec",
                self.io.name,
                state.dropped_early
            );
        }
    }
}

fn fmt_tb(tb: AvpRational) -> String {
    format!("{}/{}", tb.num, tb.den)
}

#[cfg(test)]
mod tests {
    use super::*;
    use avplumber_f7k::graph::AVP_NOPTS;
    use avplumber_f7k::graph::media::test_media;

    fn build_decoder(params: serde_json::Value) -> Result<Decoder, String> {
        let instance = avplumber_f7k::Instance::new();
        let ctx = BuildCtx {
            instance: &instance,
            name: "dec",
            params: &params,
            sync_group: None,
        };
        serde_json::from_value::<DecoderParams>(params.clone())
            .map_err(|e| e.to_string())?
            .build("dec", AvpMediaType::VIDEO, &ctx)
    }

    fn decoder() -> Decoder {
        build_decoder(serde_json::json!({})).expect("a decoder with no parameters builds")
    }

    /// A device that was never initialized is a `node.add` error, not a
    /// silently software decoder.
    #[test]
    fn naming_an_unknown_hardware_device_fails_to_build() {
        let Err(message) = build_decoder(serde_json::json!({"hwaccel": "gpu"})) else {
            panic!("building must fail: no hwaccel.init has run");
        };
        assert!(message.contains("no hardware device `gpu`"), "{message}");
    }

    /// The codec gate is parsed from either shape C++ accepts.
    #[test]
    fn hwaccel_only_for_codecs_takes_a_name_or_a_list() {
        let one = build_decoder(serde_json::json!({"hwaccel_only_for_codecs": "h264"}))
            .expect("without hwaccel it only warns");
        assert_eq!(
            one.hwaccel_codecs.as_deref(),
            Some(&["h264".to_string()][..])
        );
        let many = build_decoder(serde_json::json!({"hwaccel_only_for_codecs": ["h264", "hevc"]}))
            .expect("a list");
        assert_eq!(
            many.hwaccel_codecs.as_deref(),
            Some(&["h264".to_string(), "hevc".to_string()][..])
        );
        let none = build_decoder(serde_json::json!({})).expect("absent");
        assert_eq!(none.hwaccel_codecs, None);
        let Err(bad) = build_decoder(serde_json::json!({"hwaccel_only_for_codecs": 7})) else {
            panic!("a number is not a codec name");
        };
        assert!(bad.contains("codec name"), "{bad}");
    }

    fn ms(val: i64) -> Ts {
        Ts {
            val,
            tb: AvpRational { num: 1, den: 1000 },
        }
    }

    /// The cutoff drops what lies below it, in any time base, and the first
    /// frame at or past it clears it.
    #[test]
    fn frames_below_resume_at_are_dropped_until_the_first_one_past_it() {
        let dec = decoder();
        let state = &mut *dec.state.lock().unwrap();
        // 1/100 s: the same instant as 10 ms, in a coarser base.
        state.resume_at = Some(Ts {
            val: 1,
            tb: AvpRational { num: 1, den: 100 },
        });

        assert!(dec.below_resume_at(state, &test_media(AvpMediaType::VIDEO, 5)));
        assert!(dec.below_resume_at(state, &test_media(AvpMediaType::VIDEO, 9)));
        assert_eq!(state.dropped_before_resume, 2);
        assert!(state.resume_at.is_some(), "still armed before the cutoff");

        assert!(!dec.below_resume_at(state, &test_media(AvpMediaType::VIDEO, 10)));
        assert!(
            state.resume_at.is_none(),
            "the frame at the cutoff clears it"
        );
        assert_eq!(state.dropped_before_resume, 0);

        // Cleared: an earlier timestamp is no longer dropped.
        assert!(!dec.below_resume_at(state, &test_media(AvpMediaType::VIDEO, 3)));
    }

    /// A frame with no timestamp cannot be placed on either side, so it passes
    /// and leaves the cutoff armed.
    #[test]
    fn a_frame_without_pts_passes_and_keeps_the_cutoff() {
        let dec = decoder();
        let state = &mut *dec.state.lock().unwrap();
        state.resume_at = Some(ms(10));
        assert!(!dec.below_resume_at(state, &test_media(AvpMediaType::VIDEO, AVP_NOPTS)));
        assert!(state.resume_at.is_some());
        assert!(dec.below_resume_at(state, &test_media(AvpMediaType::VIDEO, 4)));
    }

    /// `FlushStop` arms the cutoff, and a later `FlushStart` disarms it: a new
    /// discontinuity supersedes what the previous one aimed for.
    #[test]
    fn flush_stop_arms_and_flush_start_disarms() {
        let dec = decoder();
        dec.on_flush_stop(Some(ms(200)));
        assert_eq!(
            dec.state.lock().unwrap().resume_at.map(|ts| ts.val),
            Some(200)
        );
        dec.on_flush();
        assert!(dec.state.lock().unwrap().resume_at.is_none());
        dec.on_flush_stop(None);
        assert!(dec.state.lock().unwrap().resume_at.is_none());
    }
}
