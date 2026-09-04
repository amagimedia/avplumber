//! `dec_video` / `dec_audio` — libavcodec decoding. Port of C++
//! `src/nodes/decoders.cpp`.
//!
//! The decoder opens on the [`Spec::Packet`] its producer publishes, which is
//! what replaces the C++ `findNodeUp<IStreamsInput>()` walk and the
//! `InputStreamMetadata` the C++ decoder read its `AVStream` from. It then
//! publishes the [`Spec::Video`]/[`Spec::Audio`] of the **first decoded frame**,
//! so an encoder downstream never has to ask what it is receiving.
//!
//! Deferred: `hwaccel`/`hwaccel_only_for_codecs` (rejected rather than silently
//! ignored), and `flush_magic`/`waiting_for_frame`, which only mean anything
//! together with the `discardUntil` seek machinery.

use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use rsmpeg::avcodec::AVCodecContext;
use rusty_ffmpeg::ffi;
use serde_json::Value;

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::buffer::{AvpMediaType, AvpRational};
use avplumber_f7k::graph::edge::{Edge, EdgeEvent, EdgeItem};
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::media::{FrameExt, Media, PacketExt, Ts};
use avplumber_f7k::graph::node::{Blocked, Node, NodeBody, NodeKind};
use avplumber_f7k::graph::pad::{NodePads, PadDecl};
use avplumber_f7k::graph::spec::{PacketSpec, Spec};
use avplumber_f7k::graph::timebase::ts_cmp;
use avplumber_f7k::libav::codec;
use avplumber_f7k::libav::dict::Options;
use avplumber_f7k::libav::pump::{Progress, Pump, PumpKind};
use avplumber_f7k::scaffold::{BlockingStep, EdgeSlot, Park, Pushed, blocking_body, push_blocking};

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
    /// Build-time only: rejected outright, see the module docs. Always `None` on a
    /// node that was built successfully.
    #[serde(default)]
    hwaccel: Option<Value>,
    /// Build-time only, like `hwaccel`.
    #[serde(default)]
    hwaccel_only_for_codecs: Option<Value>,
}

impl DecoderParams {
    fn build(self, name: &str, media: AvpMediaType) -> Result<Decoder, String> {
        if self.hwaccel.is_some() || self.hwaccel_only_for_codecs.is_some() {
            return Err("hwaccel is not implemented in the Rust core yet".into());
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
            name: name.into(),
            media,
            params: self,
            pixel_format,
            park: Arc::new(Park::default()),
            input: EdgeSlot::default(),
            out: EdgeSlot::default(),
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
    type Node = Decoder;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        self.0.build(name, AvpMediaType::VIDEO)
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(transparent)]
pub struct AudioDecoderSpec(DecoderParams);

impl NodeSpec for AudioDecoderSpec {
    const TYPE_NAME: &'static str = "dec_audio";
    type Node = Decoder;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        self.0.build(name, AvpMediaType::AUDIO)
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
    dropped_early: u64,
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
            dropped_early: 0,
        }
    }
}

pub struct Decoder {
    name: String,
    /// `VIDEO` or `AUDIO`: which node type this is.
    media: AvpMediaType,
    /// What the script asked for, verbatim: [`DecoderParams`] documents each
    /// field, and holding it whole is what keeps them from being declared twice.
    params: DecoderParams,
    /// `params.pixel_format`, parsed. Held by the node rather than by [`State`] so
    /// the pointer the context carries in `opaque` stays valid for the context's
    /// whole life.
    pixel_format: Option<Arc<PixelFormatRequest>>,
    park: Arc<Park>,
    input: EdgeSlot,
    out: EdgeSlot,
    state: Mutex<State>,
}

impl Node for Decoder {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> NodeKind {
        NodeKind::Blocking
    }

    fn pads(&self) -> NodePads {
        NodePads {
            sources: vec![PadDecl {
                name: "in".into(),
                media: AvpMediaType::PACKET,
            }],
            sinks: vec![PadDecl {
                name: "out".into(),
                media: self.media,
            }],
        }
    }

    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.input.bind(edge);
    }

    fn bind_sink(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.out.bind(edge);
    }

    fn start(&self) {
        self.park.reset();
        let state = &mut *self.state.lock().unwrap();
        state.pump.reset();
        if let Some(ctx) = state.ctx.as_mut() {
            ctx.flush_buffers();
        }
        state.last_pts = Ts::invalid();
        state.last_key = false;
        state.eof = false;
        state.dropped_early = 0;
        // `input_spec`/`output_spec` deliberately survive: the codec is still
        // open for them, and the edge re-arms its latched spec on a restart, so
        // the re-delivery is recognised as "unchanged" instead of reopening.
    }

    fn interrupt(&self) {
        self.park.interrupt();
    }

    fn take_body(self: Arc<Self>) -> NodeBody {
        blocking_body(self)
    }
}

impl BlockingStep for Decoder {
    fn step(&self) -> Result<Blocked, NodeError> {
        let input = self
            .input
            .require(&self.name, NodePhase::Process, "input")?;
        let out = self.out.require(&self.name, NodePhase::Process, "output")?;
        let mut guard = self.state.lock().unwrap();
        let state = &mut *guard;

        if self.park.is_interrupted() {
            return Ok(Blocked::Done);
        }

        // Decoded output goes downstream before anything new is taken in.
        if let Some(buffer) = state.pump.take_output() {
            return self.emit(state, &out, buffer);
        }
        // Drained after `Eof`: pass the marker on and finish.
        if state.eof {
            self.log_drops(state);
            out.push_event(EdgeEvent::Eof);
            return Ok(Blocked::Done);
        }
        // The codec refused the packet we are holding: offer it again.
        if state.pump.is_loaded() {
            return match self.drive(state)? {
                Progress::Moved => Ok(Blocked::Again),
                // Neither direction moved, which for a decoder means the codec
                // wants time rather than data. Park instead of spinning.
                Progress::Stalled => {
                    self.park.wait(avplumber_f7k::scaffold::PARK_TIMEOUT_MS);
                    Ok(Blocked::Again)
                }
            };
        }

        match input.take(-1) {
            Some(EdgeItem::Event(EdgeEvent::Spec(spec))) => {
                self.on_spec(state, spec)?;
                Ok(Blocked::Again)
            }
            Some(EdgeItem::Event(EdgeEvent::Eof)) => {
                if let Some(ctx) = state.ctx.as_mut() {
                    state.pump.flush(ctx);
                }
                state.eof = true;
                Ok(Blocked::Again)
            }
            Some(EdgeItem::Event(EdgeEvent::FlushStart)) => {
                if let Some(ctx) = state.ctx.as_mut() {
                    ctx.flush_buffers();
                }
                state.pump.reset();
                state.last_pts = Ts::invalid();
                out.push_event(EdgeEvent::FlushStart);
                Ok(Blocked::Again)
            }
            Some(EdgeItem::Event(EdgeEvent::FlushStop)) => {
                out.push_event(EdgeEvent::FlushStop);
                Ok(Blocked::Again)
            }
            Some(EdgeItem::Buffer(buffer)) => {
                if state.ctx.is_none() {
                    // Only reachable when a producer pushed packets before its
                    // spec; there is nothing to decode them with yet.
                    state.dropped_early += 1;
                    return Ok(Blocked::Again);
                }
                state.last_key = matches!(&buffer, Media::Packet(packet) if packet.is_key());
                state.pump.load(buffer);
                Ok(Blocked::Again)
            }
            // A blocking take came back empty: the edge is closed, or the
            // executor interrupted it to stop this node.
            None => {
                self.log_drops(state);
                Ok(Blocked::Done)
            }
        }
    }
}

impl Decoder {
    fn error(&self, phase: NodePhase, message: impl Into<String>) -> NodeError {
        NodeError::new(&self.name, phase, message)
    }

    fn drive(&self, state: &mut State) -> Result<Progress, NodeError> {
        let ctx = state
            .ctx
            .as_mut()
            .ok_or_else(|| self.error(NodePhase::Process, "the decoder is not open"))?;
        state
            .pump
            .drive(ctx)
            .map_err(|message| self.error(NodePhase::Process, message))
    }

    /// Stamps the frame, publishes its spec if it is new, then pushes it.
    fn emit(
        &self,
        state: &mut State,
        out: &Arc<dyn Edge>,
        buffer: Media,
    ) -> Result<Blocked, NodeError> {
        let buffer = self.stamp(state, buffer);
        self.publish_spec(state, out, &buffer);
        match push_blocking(&self.park, out, buffer, || Ok(()))? {
            Pushed::Ok => Ok(Blocked::Again),
            Pushed::Interrupted => Ok(Blocked::Done),
            Pushed::Closed => {
                log::info!("{}: output edge closed, finishing", self.name);
                Ok(Blocked::Done)
            }
        }
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
                    self.name,
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
                self.name
            );
        } else {
            log::info!("{}: decoding to {spec:?}", self.name);
        }
        out.push_event(EdgeEvent::Spec(spec.clone()));
        state.output_spec = Some(spec);
    }

    /// C++ built its decoder from the input `AVStream`; here the same
    /// description arrives as a spec, and a *changed* one reopens the codec.
    fn on_spec(&self, state: &mut State, spec: Spec) -> Result<(), NodeError> {
        let Spec::Packet(packet_spec) = &spec else {
            log::warn!(
                "{}: ignoring a {:?} spec on the input; a decoder needs a packet spec",
                self.name,
                spec.media()
            );
            return Ok(());
        };
        if let Some(published) = &state.input_spec {
            if codec::same_spec(published, &spec) {
                log::debug!(
                    "{}: input spec re-delivered unchanged, keeping the decoder",
                    self.name
                );
                return Ok(());
            }
            log::info!("{}: input format changed, reopening the decoder", self.name);
            if state.pump.has_output() {
                log::warn!(
                    "{}: dropping frames the previous decoder had already produced",
                    self.name
                );
            }
        }

        let ctx = self
            .open(packet_spec)
            .map_err(|message| self.error(NodePhase::Spec, message))?;
        state.time_base = packet_spec.time_base;
        state.frame_rate = ctx.framerate.into();
        state.ctx = Some(ctx);
        state.pump.reset();
        state.last_pts = Ts::invalid();
        state.last_key = false;
        state.input_spec = Some(spec);
        Ok(())
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
                        self.name
                    );
                    Some(name.clone())
                }
                None => {
                    if !self.params.codec_map.is_empty() {
                        log::info!(
                            "{}: detected codec {input_name}, not in codec_map, using the \
                             libavcodec default",
                            self.name
                        );
                    }
                    None
                }
            },
        };

        let codec = codec::find_decoder(implementation.as_deref(), codec_id)?;
        let mut ctx = AVCodecContext::new(&codec);
        codec::apply_packet_spec(&mut ctx, spec)?;
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
            &self.name,
        )?;
        log::info!(
            "{}: opened decoder {} for {input_name}",
            self.name,
            codec.name().to_string_lossy()
        );
        Ok(ctx)
    }

    fn log_drops(&self, state: &State) {
        if state.dropped_early > 0 {
            log::info!(
                "{}: dropped {} packet(s) that arrived before the input spec",
                self.name,
                state.dropped_early
            );
        }
    }
}
