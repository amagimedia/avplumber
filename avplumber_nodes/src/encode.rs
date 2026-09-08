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
use avplumber_f7k::graph::edge::{Edge, EdgeEvent};
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::media::{FrameExt, Media, PacketExt, Ts};
use avplumber_f7k::graph::node::Blocked;
use avplumber_f7k::graph::pad::NodePads;
use avplumber_f7k::graph::spec::Spec;
use avplumber_f7k::graph::timebase::ts_cmp;
use avplumber_f7k::libav::codec;
use avplumber_f7k::libav::dict::Options;
use avplumber_f7k::libav::pump::{Progress, Pump, PumpKind};
use avplumber_f7k::scaffold::{Blocking, BlockingIo, InputHandler, PARK_TIMEOUT_MS, SingleInput};

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
    /// What a `FlushStart` (a seek or another discontinuity upstream) does to
    /// the codec, see [`FlushMode`].
    #[serde(default)]
    flush: FlushMode,
    /// Build-time only: rejected outright, see the module docs. Always `None` on a
    /// node that was built successfully.
    #[serde(default)]
    hwaccel: Option<Value>,
}

/// How the codec takes an in-band discontinuity. An `Eof` is different: it
/// always drains the codec, that is how a recording gets its last packets.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FlushMode {
    /// The codec is left alone; only what this node holds itself is dropped
    /// and the backwards-PTS reference forgotten. What the codec buffers
    /// internally is a few frames of delay at most, harmless in a live
    /// output whose timestamps stay monotonic across the seek (the `realtime`
    /// node's), and `avcodec_flush_buffers` is not a safe way to get rid of
    /// them: libx264 implements it by draining, which stops its lookahead
    /// thread for good, so every frame after the first seek failed with
    /// "Generic error in an external library".
    #[default]
    Keep,
    /// The codec is closed and reopened: nothing from before the discontinuity
    /// comes out after it, and the next packet is a fresh keyframe. For a
    /// recorder that wants a clean cut; costs a codec open per seek.
    Reopen,
}

impl EncoderParams {
    fn build(self, name: &str, media: AvpMediaType) -> Result<Encoder, String> {
        if self.hwaccel.is_some() {
            return Err("hwaccel is not implemented in the Rust core yet".into());
        }
        Ok(Encoder {
            io: BlockingIo::new(name),
            media,
            params: self,
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
    type Node = Blocking<Encoder>;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        self.0.build(name, AvpMediaType::VIDEO).map(Blocking)
    }
}

#[derive(Debug, serde::Deserialize)]
#[serde(transparent)]
pub struct AudioEncoderSpec(EncoderParams);

impl NodeSpec for AudioEncoderSpec {
    const TYPE_NAME: &'static str = "enc_audio";
    type Node = Blocking<Encoder>;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        self.0.build(name, AvpMediaType::AUDIO).map(Blocking)
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
    /// A reopen at a `FlushStart` produced different codec parameters: publish
    /// them before the next packet.
    pending_spec: Option<Spec>,
    /// A reopen at a `FlushStart` failed. `on_flush` cannot fail itself, so the
    /// next step reports it.
    pending_error: Option<String>,
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
            pending_spec: None,
            pending_error: None,
        }
    }
}

pub struct Encoder {
    io: BlockingIo,
    /// `VIDEO` or `AUDIO`: which node type this is.
    media: AvpMediaType,
    /// What the script asked for, verbatim: [`EncoderParams`] documents each field,
    /// and holding it whole is what keeps them from being declared twice.
    params: EncoderParams,
    state: Mutex<State>,
}

impl InputHandler for Encoder {
    /// C++ built its encoder from three `findNodeUp` interfaces plus the output
    /// stream; here the same description arrives as one spec, and a *changed* one
    /// reopens the codec. What comes back is the muxer's half of the old
    /// handshake: the opened context's codec parameters, extradata included,
    /// published before the first packet reaches the edge.
    fn on_spec(&self, spec: Spec) -> Result<Option<Spec>, NodeError> {
        let state = &mut *self.state.lock().unwrap();
        if spec.media() != self.media {
            log::warn!(
                "{}: ignoring a {:?} spec on the input; this encoder needs {:?}",
                self.io.name,
                spec.media(),
                self.media
            );
            return Ok(None);
        }
        if let Some(open_for) = &state.input_spec {
            if codec::same_spec(open_for, &spec) {
                log::debug!(
                    "{}: input spec re-delivered unchanged, keeping the encoder",
                    self.io.name
                );
                return Ok(None);
            }
            log::info!(
                "{}: input format changed, reopening the encoder",
                self.io.name
            );
            if state.pump.has_output() {
                log::warn!(
                    "{}: dropping packets the previous encoder had already produced",
                    self.io.name
                );
            }
        }

        let ctx = self
            .open(&spec)
            .map_err(|message| self.io.error(NodePhase::Spec, message))?;
        let packet_spec = codec::packet_spec_of(&ctx);
        state.time_base = ctx.time_base.into();
        state.ctx = Some(ctx);
        state.pump.reset();
        state.prev_pts = Ts::invalid();
        state.input_pts.clear();
        state.passthrough_warned = false;
        state.input_spec = Some(spec);
        Ok(Some(Spec::Packet(packet_spec)))
    }

    fn on_buffer(&self, buffer: Media) -> Result<Option<Media>, NodeError> {
        let state = &mut *self.state.lock().unwrap();
        self.load(state, buffer);
        Ok(None)
    }

    /// A discontinuity drops what this node holds — the frame loaded into the
    /// pump and packets not yet pushed — and forgets the backwards-PTS
    /// reference. What happens to the codec is the `flush` parameter's call,
    /// see [`FlushMode`].
    fn on_flush(&self) {
        let state = &mut *self.state.lock().unwrap();
        state.pump.reset();
        state.prev_pts = Ts::invalid();
        state.input_pts.clear();
        if self.params.flush == FlushMode::Reopen
            && let Some(spec) = state.input_spec.clone()
            && state.ctx.is_some()
        {
            match self.open(&spec) {
                Ok(ctx) => {
                    let packet_spec = Spec::Packet(codec::packet_spec_of(&ctx));
                    let unchanged = state
                        .ctx
                        .as_ref()
                        .map(|old| {
                            codec::same_spec(
                                &Spec::Packet(codec::packet_spec_of(old)),
                                &packet_spec,
                            )
                        })
                        .unwrap_or(false);
                    if !unchanged {
                        state.pending_spec = Some(packet_spec);
                    }
                    state.time_base = ctx.time_base.into();
                    state.ctx = Some(ctx);
                    log::debug!("{}: reopened the encoder at a discontinuity", self.io.name);
                }
                Err(message) => {
                    state.ctx = None;
                    state.pending_error = Some(format!(
                        "reopening the encoder at a discontinuity failed: {message}"
                    ));
                }
            }
        }
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

    fn on_closed(&self) {
        let state = &mut *self.state.lock().unwrap();
        self.log_drops(state);
    }
}

impl SingleInput for Encoder {
    fn io(&self) -> &BlockingIo {
        &self.io
    }

    fn pads(&self) -> NodePads {
        NodePads::siso(self.media, AvpMediaType::PACKET)
    }

    fn start(&self) {
        let state = &mut *self.state.lock().unwrap();
        state.pump.reset();
        if state.eof {
            // The codec was drained: libavcodec only takes new input after
            // `avcodec_flush_buffers`, which encoders do not reliably support
            // (see `on_flush`). Forgetting both makes the spec the edge re-arms
            // on restart open a fresh one.
            state.ctx = None;
            state.input_spec = None;
        }
        state.prev_pts = Ts::invalid();
        state.input_pts.clear();
        state.passthrough_warned = false;
        state.eof = false;
        state.dropped_early = 0;
        state.pending_spec = None;
        state.pending_error = None;
        // Otherwise `input_spec` deliberately survives: the codec is still open
        // for it, and the edge re-arms its latched spec on a restart, so the
        // re-delivery is recognised as "unchanged" instead of reopening. The
        // output spec is latched on the output edge for the same reason, so the
        // muxer keeps the codec parameters of the encoder that is still running.
    }

    /// Encoded packets go downstream before anything new is taken in, and a
    /// codec that refused the frame we are holding is offered it again.
    fn before_take(&self) -> Result<Option<Blocked>, NodeError> {
        let state = &mut *self.state.lock().unwrap();
        let out = self.io.output()?;
        if let Some(message) = state.pending_error.take() {
            return Err(self.io.error(NodePhase::Process, message));
        }
        if let Some(spec) = state.pending_spec.take() {
            out.push_event(EdgeEvent::Spec(spec));
        }
        if let Some(buffer) = state.pump.take_output() {
            return self.emit(state, &out, buffer).map(Some);
        }
        // Drained after `Eof`: pass the marker on and finish.
        if state.eof {
            self.log_drops(state);
            out.push_event(EdgeEvent::Eof);
            return Ok(Some(Blocked::Done));
        }
        if state.pump.is_loaded() {
            return Ok(Some(match self.drive(state)? {
                Progress::Moved => Blocked::Again,
                // Neither direction moved, which for an encoder means it wants
                // time rather than data. Park instead of spinning.
                Progress::Stalled => {
                    self.io.wait(PARK_TIMEOUT_MS);
                    Blocked::Again
                }
            }));
        }
        Ok(None)
    }
}

impl Encoder {
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
                self.io.name,
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
            .ok_or_else(|| self.io.error(NodePhase::Process, "the encoder is not open"))?;
        let progress = state
            .pump
            .drive(ctx)
            .map_err(|message| self.io.error(NodePhase::Process, message))?;
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
                self.io.name
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
        self.io.push(out, buffer)
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
                                self.io.name
                            );
                        }
                    }
                }
            }
            packet.set_ts_dts(pts, dts);
        }
        buffer
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
            &self.io.name,
        )?;
        log::info!(
            "{}: opened encoder {} at {} bit/s, time base {}/{}",
            self.io.name,
            codec.name().to_string_lossy(),
            ctx.bit_rate,
            ctx.time_base.num,
            ctx.time_base.den
        );
        if self.media == AvpMediaType::AUDIO && ctx.frame_size > 0 {
            log::info!(
                "{}: the encoder wants {} samples per frame",
                self.io.name,
                ctx.frame_size
            );
        }
        Ok(ctx)
    }

    fn log_drops(&self, state: &State) {
        if state.dropped_early > 0 {
            log::info!(
                "{}: dropped {} frame(s) that arrived before the input spec",
                self.io.name,
                state.dropped_early
            );
        }
    }
}
