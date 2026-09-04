//! `enc_video` / `enc_audio` — libavcodec encoding. Port of C++
//! `src/nodes/encoders.cpp`.
//!
//! The encoder opens on the [`Spec::Video`]/[`Spec::Audio`] its producer
//! publishes: that one message carries what C++ collected from three separate
//! `findNodeUp` interfaces (`IVideoFormatSource`/`IAudioMetadataSource`,
//! `IFrameRateSource` and `ITimeBaseSource`). It then publishes the
//! [`Spec::Packet`] of the opened context, which is what replaces C++'s
//! three-phase `setOutput`/`openEncoder`/`codecParameters` handshake with the
//! muxer: `mux` and `output` read the codec parameters off the edge instead of
//! reaching back into this node.
//!
//! Because that handshake is gone the encoder never sees the container, so it
//! **always** encodes with `AV_CODEC_FLAG_GLOBAL_HEADER` (design doc §4) instead
//! of copying `AVFMT_GLOBALHEADER` off the output format. Extradata therefore
//! always reaches `output` through the spec; the muxers that need it in-band
//! (mpegts and friends) re-insert it themselves.
//!
//! Deferred: `hwaccel` (rejected rather than silently ignored), and
//! `INeedsOutputFrameSize`, the hint an audio encoder gives a resampler about
//! its `frame_size`. Without a resampler in the graph the input has to be
//! chunked correctly already, which it is when the sample rate is unchanged.

use std::collections::VecDeque;
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
use avplumber_f7k::graph::spec::Spec;
use avplumber_f7k::graph::timebase::ts_cmp;
use avplumber_f7k::libav::codec;
use avplumber_f7k::libav::dict::Options;
use avplumber_f7k::libav::pump::{Progress, Pump, PumpKind};
use avplumber_f7k::scaffold::{BlockingStep, EdgeSlot, Park, Pushed, blocking_body, push_blocking};

/// The parameters both encoder types share; C++ has one template for both.
///
/// The node keeps this whole struct rather than copying fields out of it, so each
/// parameter is declared exactly once. `hwaccel` is the one field that means
/// nothing after a successful build.
#[derive(Debug, serde::Deserialize)]
pub struct EncoderParams {
    /// Required, unlike a decoder's: nothing else says what to produce.
    codec: String,
    /// Passed to `avcodec_open2` — bitrate, preset, gop size and the rest all
    /// live here. Unconsumed entries are logged.
    #[serde(default)]
    options: Option<Value>,
    /// Stamp output packets with the input frame's timestamp instead of the
    /// encoder's own. A hack for the 1:1 `pcm_*` "encoders", see [`Encoder`].
    #[serde(default)]
    timestamps_passthrough: bool,
    /// Build-time only: rejected outright, see the module docs. Always `None` on a
    /// node that was built successfully.
    #[serde(default)]
    hwaccel: Option<Value>,
}

impl EncoderParams {
    fn build(self, name: &str, media: AvpMediaType) -> Result<Encoder, String> {
        if self.hwaccel.is_some() {
            return Err("hwaccel is not implemented in the Rust core yet".into());
        }
        Ok(Encoder {
            name: name.into(),
            media,
            params: self,
            park: Arc::new(Park::default()),
            input: EdgeSlot::default(),
            out: EdgeSlot::default(),
            state: Mutex::new(State::new(name)),
        })
    }
}

/// `enc_video`. A newtype so both encoders share one parameter set while each
/// keeps its own `TYPE_NAME`.
#[derive(Debug, serde::Deserialize)]
#[serde(transparent)]
pub struct VideoEncoderSpec(EncoderParams);

impl NodeSpec for VideoEncoderSpec {
    const TYPE_NAME: &'static str = "enc_video";
    type Node = Encoder;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        self.0.build(name, AvpMediaType::VIDEO)
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(transparent)]
pub struct AudioEncoderSpec(EncoderParams);

impl NodeSpec for AudioEncoderSpec {
    const TYPE_NAME: &'static str = "enc_audio";
    type Node = Encoder;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        self.0.build(name, AvpMediaType::AUDIO)
    }
}

struct State {
    /// `None` until the input spec arrives; the codec cannot be opened before.
    ctx: Option<AVCodecContext>,
    /// The [`Spec::Video`]/[`Spec::Audio`] `ctx` was opened for, so a re-delivered
    /// identical spec is a no-op.
    input_spec: Option<Spec>,
    /// The opened encoder's time base: what its packets are stamped in, and what
    /// the published [`Spec::Packet`] told the muxer.
    time_base: AvpRational,
    pump: Pump,
    /// C++ `prev_ts_`: the last accepted input timestamp, for the
    /// backwards-PTS check.
    prev_pts: Ts,
    /// Input timestamps waiting to be stamped onto output packets, oldest first;
    /// only filled when `timestamps_passthrough` is set.
    input_pts: VecDeque<Ts>,
    /// Whether the "passthrough is not 1:1" warning has been logged. Both of its
    /// wordings mean the same thing, and repeating either per frame would flood
    /// the log.
    passthrough_warned: bool,
    /// `Eof` seen: drain the codec, forward it, finish.
    eof: bool,
    dropped_early: u64,
}

impl State {
    fn new(node: &str) -> Self {
        Self {
            ctx: None,
            input_spec: None,
            time_base: AvpRational::default(),
            // `out_media` is unused when encoding: the output is always a packet.
            pump: Pump::new(PumpKind::Encode, AvpMediaType::PACKET, node),
            prev_pts: Ts::invalid(),
            input_pts: VecDeque::new(),
            passthrough_warned: false,
            eof: false,
            dropped_early: 0,
        }
    }
}

pub struct Encoder {
    name: String,
    /// `VIDEO` or `AUDIO`: which node type this is.
    media: AvpMediaType,
    /// What the script asked for, verbatim: [`EncoderParams`] documents each field,
    /// and holding it whole is what keeps them from being declared twice.
    params: EncoderParams,
    park: Arc<Park>,
    input: EdgeSlot,
    out: EdgeSlot,
    state: Mutex<State>,
}

impl Node for Encoder {
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
                media: self.media,
            }],
            sinks: vec![PadDecl {
                name: "out".into(),
                media: AvpMediaType::PACKET,
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
        state.prev_pts = Ts::invalid();
        state.input_pts.clear();
        state.passthrough_warned = false;
        state.eof = false;
        state.dropped_early = 0;
        // `input_spec` deliberately survives: the codec is still open for it, and
        // the edge re-arms its latched spec on a restart, so the re-delivery is
        // recognised as "unchanged" instead of reopening. The output spec is
        // latched on the output edge for the same reason, so the muxer keeps the
        // codec parameters of the encoder that is still running.
    }

    fn interrupt(&self) {
        self.park.interrupt();
    }

    fn take_body(self: Arc<Self>) -> NodeBody {
        blocking_body(self)
    }
}

impl BlockingStep for Encoder {
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

        // Encoded packets go downstream before anything new is taken in.
        if let Some(buffer) = state.pump.take_output() {
            return self.emit(state, &out, buffer);
        }
        // Drained after `Eof`: pass the marker on and finish.
        if state.eof {
            self.log_drops(state);
            out.push_event(EdgeEvent::Eof);
            return Ok(Blocked::Done);
        }
        // The codec refused the frame we are holding: offer it again.
        if state.pump.is_loaded() {
            return match self.drive(state)? {
                Progress::Moved => Ok(Blocked::Again),
                // Neither direction moved, which for an encoder means it wants
                // time rather than data. Park instead of spinning.
                Progress::Stalled => {
                    self.park.wait(avplumber_f7k::scaffold::PARK_TIMEOUT_MS);
                    Ok(Blocked::Again)
                }
            };
        }

        match input.take(-1) {
            Some(EdgeItem::Event(EdgeEvent::Spec(spec))) => {
                self.on_spec(state, &out, spec)?;
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
                state.prev_pts = Ts::invalid();
                state.input_pts.clear();
                out.push_event(EdgeEvent::FlushStart);
                Ok(Blocked::Again)
            }
            Some(EdgeItem::Event(EdgeEvent::FlushStop)) => {
                out.push_event(EdgeEvent::FlushStop);
                Ok(Blocked::Again)
            }
            Some(EdgeItem::Buffer(buffer)) => {
                self.load(state, buffer);
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

impl Encoder {
    fn error(&self, phase: NodePhase, message: impl Into<String>) -> NodeError {
        NodeError::new(&self.name, phase, message)
    }

    /// Hands one frame to the pump, or drops it like C++ does.
    fn load(&self, state: &mut State, buffer: Media) {
        if state.ctx.is_none() {
            // Only reachable when a producer pushed frames before its spec;
            // there is nothing to encode them with yet.
            state.dropped_early += 1;
            return;
        }
        let mut buffer = buffer;
        let time_base = state.time_base;
        let ts = buffer.ts();
        if state.prev_pts.is_valid()
            && ts.is_valid()
            && ts_cmp(ts.val, ts.tb, state.prev_pts.val, state.prev_pts.tb).is_lt()
        {
            log::warn!(
                "{}: input PTS went backwards {} -> {}, discarding frame",
                self.name,
                state.prev_pts.val,
                ts.val
            );
            return;
        }
        // libavcodec reads `frame.pts` in the *context's* time base and ignores
        // `frame.time_base`, so a frame stamped in another base is rescaled here
        // instead of being silently misread.
        if let Media::Video(frame) | Media::Audio(frame) = &mut buffer
            && ts.is_valid()
            && ts.tb != time_base
        {
            frame.set_ts(ts.rescale(time_base));
        }
        if self.params.timestamps_passthrough {
            state.input_pts.push_back(ts);
        }
        if ts.is_valid() {
            state.prev_pts = ts;
        }
        state.pump.load(buffer);
    }

    fn drive(&self, state: &mut State) -> Result<Progress, NodeError> {
        let ctx = state
            .ctx
            .as_mut()
            .ok_or_else(|| self.error(NodePhase::Process, "the encoder is not open"))?;
        let progress = state
            .pump
            .drive(ctx)
            .map_err(|message| self.error(NodePhase::Process, message))?;
        // The frame was taken but nothing came out, i.e. the encoder is holding
        // it back — which is exactly what `timestamps_passthrough` cannot model.
        if self.params.timestamps_passthrough
            && !state.passthrough_warned
            && !state.pump.is_loaded()
            && !state.pump.has_output()
        {
            state.passthrough_warned = true;
            log::warn!(
                "{}: encoder does buffer but we overwrite timestamps, this may cause desync!",
                self.name
            );
        }
        Ok(progress)
    }

    /// Stamps the packet and pushes it. Unlike the decoder there is no spec to
    /// publish here: an encoder's output format is known at open time.
    fn emit(
        &self,
        state: &mut State,
        out: &Arc<dyn Edge>,
        buffer: Media,
    ) -> Result<Blocked, NodeError> {
        let buffer = self.stamp(state, buffer);
        match push_blocking(&self.park, out, buffer, || Ok(()))? {
            Pushed::Ok => Ok(Blocked::Again),
            Pushed::Interrupted => Ok(Blocked::Done),
            Pushed::Closed => {
                log::info!("{}: output edge closed, finishing", self.name);
                Ok(Blocked::Done)
            }
        }
    }

    /// Every packet leaves with the encoder's time base written into it:
    /// libavcodec 6 and 7/8 disagree about whether `packet.time_base` comes back
    /// filled, and `mux`/`output` rescale through it.
    fn stamp(&self, state: &mut State, buffer: Media) -> Media {
        let mut buffer = buffer;
        let tb = state.time_base;
        if let Media::Packet(packet) = &mut buffer {
            let mut pts = Ts {
                val: packet.pts,
                tb,
            };
            let mut dts = Ts {
                val: packet.dts,
                tb,
            };
            if self.params.timestamps_passthrough {
                match state.input_pts.pop_front() {
                    // The HACK C++ has for the `pcm_*` "encoders", which emit one
                    // packet per frame: PTS *and* DTS come from the frame.
                    Some(source) if source.is_valid() => {
                        pts = source.rescale(tb);
                        dts = pts;
                    }
                    _ => {
                        if !state.passthrough_warned {
                            state.passthrough_warned = true;
                            log::warn!(
                                "{}: more packets than input timestamps, keeping the encoder's \
                                 own on this one",
                                self.name
                            );
                        }
                    }
                }
            }
            packet.set_ts_dts(pts, dts);
        }
        buffer
    }

    /// C++ built its encoder from three `findNodeUp` interfaces plus the output
    /// stream; here the same description arrives as one spec, and a *changed* one
    /// reopens the codec.
    fn on_spec(&self, state: &mut State, out: &Arc<dyn Edge>, spec: Spec) -> Result<(), NodeError> {
        if spec.media() != self.media {
            log::warn!(
                "{}: ignoring a {:?} spec on the input; this encoder needs {:?}",
                self.name,
                spec.media(),
                self.media
            );
            return Ok(());
        }
        if let Some(open_for) = &state.input_spec {
            if codec::same_spec(open_for, &spec) {
                log::debug!(
                    "{}: input spec re-delivered unchanged, keeping the encoder",
                    self.name
                );
                return Ok(());
            }
            log::info!("{}: input format changed, reopening the encoder", self.name);
            if state.pump.has_output() {
                log::warn!(
                    "{}: dropping packets the previous encoder had already produced",
                    self.name
                );
            }
        }

        let ctx = self
            .open(&spec)
            .map_err(|message| self.error(NodePhase::Spec, message))?;
        // What the muxer needs, extradata included, straight from the opened
        // context — published before the first packet reaches the edge.
        let packet_spec = codec::packet_spec_of(&ctx);
        state.time_base = ctx.time_base.into();
        state.ctx = Some(ctx);
        state.pump.reset();
        state.prev_pts = Ts::invalid();
        state.input_pts.clear();
        state.passthrough_warned = false;
        state.input_spec = Some(spec);
        out.push_event(EdgeEvent::Spec(Spec::Packet(packet_spec)));
        Ok(())
    }

    fn open(&self, spec: &Spec) -> Result<AVCodecContext, String> {
        let codec = codec::find_encoder(&self.params.codec)?;
        let wanted = match self.media {
            AvpMediaType::AUDIO => ffi::AVMEDIA_TYPE_AUDIO,
            _ => ffi::AVMEDIA_TYPE_VIDEO,
        };
        if codec.type_ != wanted {
            return Err(format!(
                "`{}` is not {} encoder",
                self.params.codec,
                if wanted == ffi::AVMEDIA_TYPE_AUDIO {
                    "an audio"
                } else {
                    "a video"
                }
            ));
        }

        let mut ctx = AVCodecContext::new(&codec);
        codec::apply_media_spec(&mut ctx, spec)?;
        // Always, since this node cannot see the container — see the module docs.
        ctx.set_flags(ctx.flags | ffi::AV_CODEC_FLAG_GLOBAL_HEADER as i32);
        codec::open_codec(
            &mut ctx,
            Options::from_json(self.params.options.as_ref())?,
            &self.name,
        )?;
        log::info!(
            "{}: opened encoder {} at {} bit/s, time base {}/{}",
            self.name,
            codec.name().to_string_lossy(),
            ctx.bit_rate,
            ctx.time_base.num,
            ctx.time_base.den
        );
        if self.media == AvpMediaType::AUDIO && ctx.frame_size > 0 {
            log::info!(
                "{}: the encoder wants {} samples per frame",
                self.name,
                ctx.frame_size
            );
        }
        Ok(ctx)
    }

    fn log_drops(&self, state: &State) {
        if state.dropped_early > 0 {
            log::info!(
                "{}: dropped {} frame(s) that arrived before the input spec",
                self.name,
                state.dropped_early
            );
        }
    }
}
