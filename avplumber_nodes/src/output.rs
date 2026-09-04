//! `output` — writes a muxed packet stream into a container. Port of C++
//! `src/nodes/output.cpp`.
//!
//! The container is created from the latched [`Spec::Mux`] that `mux` publishes:
//! one message carrying every stream's codec parameters, id and metadata. It
//! replaces C++'s three-phase `initFromFormatContext` /
//! `initFromFormatContextPostOpenPreWriteHeader` / `initFromFormatContextPostOpen`
//! handshake, in which this node reached up the chain for the muxer, which
//! reached further up for each encoder. Nothing reaches anywhere here — `mux`
//! and `output` share an edge, not an `AVFormatContext` (design doc §7).
//!
//! Two behaviours worth knowing about:
//! - Packets are rescaled into the time base libavformat *settled* on per stream,
//!   which is not always the one we asked for (mp4 rounds to milliseconds,
//!   mpegts uses 1/90000). That rescale can collapse two distinct DTS values into
//!   one, and `av_interleaved_write_frame` rejects a non-increasing DTS outright,
//!   so there is a second monotonic guard here: `mux`'s ran in the source time
//!   base and cannot see the collapse.
//! - The trailer is written whenever the node stops, not only on
//!   [`EdgeEvent::Eof`]. C++ writes it from `flush()`, i.e. on the EOF marker
//!   only, and leaves an unfinalized — unplayable — MP4 behind when a group is
//!   stopped mid-stream.
//!
//! Deferred: `seek_table` / `seek_table_text`, which belong with seek.

use std::ffi::CString;
use std::sync::{Arc, Mutex};

use rsmpeg::avformat::AVFormatContextOutput;
use rusty_ffmpeg::ffi;
use serde_json::Value;

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::buffer::{AvpMediaType, AvpRational};
use avplumber_f7k::graph::edge::{Edge, EdgeEvent, EdgeItem};
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::media::{Media, PacketExt, Ts};
use avplumber_f7k::graph::node::{Blocked, Node, NodeBody, NodeKind};
use avplumber_f7k::graph::pad::{NodePads, PadDecl};
use avplumber_f7k::graph::spec::{MuxStream, Spec};
use avplumber_f7k::graph::timebase::rescale;
use avplumber_f7k::libav::codec;
use avplumber_f7k::libav::dict::Options;
use avplumber_f7k::scaffold::{BlockingStep, EdgeSlot, blocking_body};

/// C++ `errors_ > 20`: a muxer that rejects one packet is usually still usable,
/// one that rejects twenty in a row is not.
const MAX_CONSECUTIVE_ERRORS: u32 = 20;

/// The node keeps this whole struct rather than copying fields out of it, so each
/// parameter is declared exactly once. The `seek_table*` fields are the only ones
/// that mean nothing after a successful build.
#[derive(Debug, serde::Deserialize)]
pub struct OutputSpec {
    url: String,
    /// Muxer name, e.g. `mpegts`; inferred from the url when absent.
    #[serde(default)]
    format: Option<String>,
    /// Passed to the protocol and then to `avformat_write_header`; unconsumed
    /// entries are logged.
    #[serde(default)]
    options: Option<Value>,
    /// Build-time only: rejected outright, see the module docs. Always `None` on a
    /// node that was built successfully.
    #[serde(default)]
    seek_table: Option<String>,
    /// Build-time only, like `seek_table`.
    #[serde(default)]
    seek_table_text: Option<String>,
}

impl NodeSpec for OutputSpec {
    const TYPE_NAME: &'static str = "output";
    type Node = StreamOutput;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        if self.seek_table.is_some() || self.seek_table_text.is_some() {
            return Err("seek_table is not implemented in the Rust core yet".into());
        }
        // Unlike `input`, nothing is opened here: the container cannot be created
        // before its stream list arrives, and that is a runtime message.
        Ok(StreamOutput {
            name: name.into(),
            params: self,
            input: EdgeSlot::default(),
            state: Mutex::new(State::default()),
        })
    }
}

/// One output stream, as libavformat left it.
struct Stream {
    /// The time base the muxer settled on, which everything written is rescaled
    /// into — not necessarily the one the spec asked for.
    time_base: AvpRational,
    /// The spec's time base, for a packet that carries none of its own.
    source_time_base: AvpRational,
    /// Last DTS handed to the muxer, in `time_base`.
    prev_dts: Option<i64>,
    /// How often the monotonic guard had to push the DTS forward.
    forced_dts: u64,
}

#[derive(Default)]
struct State {
    /// `None` until the container description arrives; also the "header written"
    /// flag, since [`StreamOutput::open`] writes it before handing the context
    /// over, and [`StreamOutput::finalize`] takes the context back out.
    ctx: Option<AVFormatContextOutput>,
    /// The [`Spec::Mux`] `ctx` was built for, so a re-delivered identical
    /// description is a no-op instead of a second file.
    mux_spec: Option<Spec>,
    streams: Vec<Stream>,
    /// Consecutive write failures, C++ `errors_`.
    errors: u32,
    dropped_early: u64,
    dropped_unknown: u64,
}

pub struct StreamOutput {
    name: String,
    /// What the script asked for, verbatim: [`OutputSpec`] documents each field,
    /// and holding it whole is what keeps them from being declared twice.
    params: OutputSpec,
    input: EdgeSlot,
    state: Mutex<State>,
}

impl Node for StreamOutput {
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
            sinks: Vec::new(),
        }
    }

    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.input.bind(edge);
    }

    fn start(&self) {
        let state = &mut *self.state.lock().unwrap();
        state.errors = 0;
        state.dropped_early = 0;
        state.dropped_unknown = 0;
        // `ctx` and `mux_spec` are always `None` here: this node finalizes and
        // closes in `stop()`, so a restart re-opens the file from the description
        // the edge re-delivers.
    }

    fn stop(&self) {
        let state = &mut *self.state.lock().unwrap();
        self.finalize(state);
    }

    fn take_body(self: Arc<Self>) -> NodeBody {
        blocking_body(self)
    }
}

impl BlockingStep for StreamOutput {
    fn step(&self) -> Result<Blocked, NodeError> {
        let input = self
            .input
            .require(&self.name, NodePhase::Process, "input")?;
        let state = &mut *self.state.lock().unwrap();

        match input.take(-1) {
            Some(EdgeItem::Event(EdgeEvent::Spec(spec))) => {
                self.on_spec(state, spec)?;
                Ok(Blocked::Again)
            }
            Some(EdgeItem::Buffer(buffer)) => self.write(state, buffer),
            Some(EdgeItem::Event(EdgeEvent::Eof)) => {
                log::info!(
                    "{}: end of stream, finishing {}",
                    self.name,
                    self.params.url
                );
                // The trailer is `stop()`'s job, which the executor calls as soon
                // as this returns.
                Ok(Blocked::Done)
            }
            // Nothing of ours to discard, and libavformat's interleaving buffer
            // must not be: it holds packets already accepted for the file.
            Some(EdgeItem::Event(EdgeEvent::FlushStart | EdgeEvent::FlushStop)) => {
                Ok(Blocked::Again)
            }
            // A blocking take came back empty: the edge is closed, or the executor
            // interrupted it to stop this node.
            None => Ok(Blocked::Done),
        }
    }
}

impl StreamOutput {
    fn error(&self, phase: NodePhase, message: impl Into<String>) -> NodeError {
        NodeError::new(&self.name, phase, message)
    }

    /// The container description, which is the one thing this node cannot start
    /// without.
    fn on_spec(&self, state: &mut State, spec: Spec) -> Result<(), NodeError> {
        let Spec::Mux { streams } = &spec else {
            log::warn!(
                "{}: ignoring a {} spec on the input; output needs the container description a \
                 `mux` publishes",
                self.name,
                spec.variant_name()
            );
            return Ok(());
        };
        if let Some(open_for) = &state.mux_spec {
            if codec::same_spec(open_for, &spec) {
                log::debug!(
                    "{}: container description re-delivered unchanged, keeping {} open",
                    self.name,
                    self.params.url
                );
                return Ok(());
            }
            // The header is on disk and names the streams; a different set of them
            // needs a different file. Failing lets the group restart, which is
            // what reopens it.
            return Err(self.error(
                NodePhase::Spec,
                format!(
                    "the stream layout of {} changed after its header was written",
                    self.params.url
                ),
            ));
        }

        let (ctx, streams) = self
            .open(streams)
            .map_err(|message| self.error(NodePhase::Spec, message))?;
        state.ctx = Some(ctx);
        state.streams = streams;
        state.errors = 0;
        state.mux_spec = Some(spec);
        Ok(())
    }

    /// Creates the container, its streams and its header. Everything libav needs
    /// comes from `streams`; the encoders that produced those parameters are not
    /// consulted, and need not still exist.
    fn open(&self, streams: &[MuxStream]) -> Result<(AVFormatContextOutput, Vec<Stream>), String> {
        let c_url = CString::new(self.params.url.as_str())
            .map_err(|_| format!("url `{}` contains a NUL", self.params.url))?;
        let c_format = match self.params.format.as_deref() {
            Some(name) => {
                Some(CString::new(name).map_err(|_| format!("format `{name}` contains a NUL"))?)
            }
            None => None,
        };
        // The same dictionary the protocol takes from is then offered to the
        // muxer, exactly as C++ passes one `opts` to `openOutput` and
        // `writeHeader` in turn.
        let mut options = Options::from_json(self.params.options.as_ref())?.into_av_dictionary();
        let mut ctx = AVFormatContextOutput::builder()
            .maybe_format_name(c_format.as_deref())
            .filename(&c_url)
            .options(&mut options)
            .build()
            .map_err(|error| format!("opening {} failed: {error}", self.params.url))?;

        let format_name = ctx.oformat().name().to_string_lossy().into_owned();
        let oformat = ctx.oformat().as_ptr();
        for (index, stream) in streams.iter().enumerate() {
            let codecpar = stream.spec.codecpar.as_ref().ok_or_else(|| {
                format!("stream {index} of the container description has no codec parameters")
            })?;
            // C++ made this check in the encoder, the only place that knew both
            // the codec and the container. Here the container knows both.
            let supported =
                unsafe { ffi::avformat_query_codec(oformat, codecpar.codec_id, COMPLIANCE_NORMAL) };
            if supported == 0 {
                return Err(format!(
                    "codec {} of stream {index} is not supported by container {format_name}",
                    codec::codec_name(codecpar.codec_id)
                ));
            }

            let mut out = ctx.new_stream();
            out.set_codecpar(codecpar.clone());
            out.set_time_base(stream.spec.time_base.into());
            if stream.spec.frame_rate.den != 0 {
                out.set_avg_frame_rate(stream.spec.frame_rate.into());
                // No rsmpeg setter, and unlike `avg_frame_rate` this one is
                // informational — but `ffprobe` and some players read it.
                unsafe { rsmpeg::UnsafeDerefMut::deref_mut(&mut *out) }.r_frame_rate =
                    stream.spec.frame_rate.into();
            }
            if let Some(id) = stream.id {
                unsafe { rsmpeg::UnsafeDerefMut::deref_mut(&mut *out) }.id = id;
            }
            if !stream.metadata.is_empty() {
                out.set_metadata(Options::from_pairs(&stream.metadata)?.into_av_dictionary());
            }
        }

        ctx.write_header(&mut options).map_err(|error| {
            format!("writing the header of {} failed: {error}", self.params.url)
        })?;
        Options::from_av_dictionary(options).warn_leftovers(&self.name, "the muxer");

        let states = ctx
            .streams()
            .iter()
            .zip(streams)
            .map(|(out, wanted)| {
                let time_base: AvpRational = out.time_base.into();
                if time_base != wanted.spec.time_base {
                    log::info!(
                        "{}: {format_name} put stream {} in time base {}/{}, not {}/{}",
                        self.name,
                        out.index,
                        time_base.num,
                        time_base.den,
                        wanted.spec.time_base.num,
                        wanted.spec.time_base.den
                    );
                }
                Stream {
                    time_base,
                    source_time_base: wanted.spec.time_base,
                    prev_dts: None,
                    forced_dts: 0,
                }
            })
            .collect::<Vec<_>>();
        log::info!(
            "{}: writing {} stream(s) as {format_name} to {}",
            self.name,
            states.len(),
            self.params.url
        );
        Ok((ctx, states))
    }

    /// Rescales one packet into its stream's time base and hands it to the muxer.
    fn write(&self, state: &mut State, buffer: Media) -> Result<Blocked, NodeError> {
        let Media::Packet(mut packet) = buffer else {
            log::warn!("{}: dropping a buffer that is not a packet", self.name);
            return Ok(Blocked::Again);
        };
        if state.ctx.is_none() {
            state.dropped_early += 1;
            return Ok(Blocked::Again);
        }
        let index = packet.stream_index;
        let Some(stream) = usize::try_from(index)
            .ok()
            .and_then(|index| state.streams.get_mut(index))
        else {
            state.dropped_unknown += 1;
            return Ok(Blocked::Again);
        };

        // libavformat leaves `pkt.time_base` unset on the way in, so producers
        // fill it (`PacketExt::set_ts_dts`); the spec's is the fallback for one
        // that did not.
        let from = if packet.time_base.den != 0 {
            packet.time_base.into()
        } else {
            stream.source_time_base
        };
        let to = stream.time_base;
        let mut pts = Ts {
            val: packet.pts,
            tb: from,
        }
        .rescale(to);
        let mut dts = Ts {
            val: packet.dts,
            tb: from,
        }
        .rescale(to);

        if dts.is_valid() {
            if let Some(prev) = stream.prev_dts
                && dts.val <= prev
            {
                stream.forced_dts += 1;
                if stream.forced_dts == 1 {
                    log::warn!(
                        "{}: DTS {} of stream {index} is not past the previous {prev} in time base \
                         {}/{}, forcing it forward (further ones are only counted)",
                        self.name,
                        dts.val,
                        to.num,
                        to.den
                    );
                }
                dts.val = prev + 1;
            }
            stream.prev_dts = Some(dts.val);
            // Whatever moved DTS must not leave it ahead of PTS.
            if pts.is_valid() && pts.val < dts.val {
                pts.val = dts.val;
            }
        }
        let duration = if packet.duration > 0 {
            rescale(packet.duration, from, to)
        } else {
            0
        };
        packet.set_ts_dts(pts, dts);
        packet.set_duration(duration);

        let ctx = state
            .ctx
            .as_mut()
            .ok_or_else(|| self.error(NodePhase::Process, "the container is not open"))?;
        // Interleaved, like C++ `octx_.writePacket`: the muxer reorders across
        // streams, which is what mp4 and mpegts need. It also takes the packet.
        match ctx.interleaved_write_frame(&mut packet) {
            Ok(()) => {
                state.errors = 0;
                Ok(Blocked::Again)
            }
            Err(error) => {
                state.errors += 1;
                log::error!("{}: writing a packet failed: {error}", self.name);
                if state.errors > MAX_CONSECUTIVE_ERRORS {
                    return Err(self.error(
                        NodePhase::Process,
                        format!(
                            "{} consecutive write failures on {}, giving up",
                            state.errors, self.params.url
                        ),
                    ));
                }
                Ok(Blocked::Again)
            }
        }
    }

    /// Writes the trailer and closes the file. Called from `stop()`, which the
    /// executor runs on this node's own thread once the body is done — for any
    /// reason, including an error or a group stop.
    fn finalize(&self, state: &mut State) {
        let Some(mut ctx) = state.ctx.take() else {
            return;
        };
        match ctx.write_trailer() {
            Ok(()) => log::info!("{}: finalized {}", self.name, self.params.url),
            Err(error) => log::error!(
                "{}: writing the trailer of {} failed: {error}",
                self.name,
                self.params.url
            ),
        }
        // Drop closes the protocol; `mux_spec` goes with it so a restart opens the
        // file again instead of taking the re-delivered description for a no-op.
        drop(ctx);
        state.mux_spec = None;
        self.log_drops(state);
        for (index, stream) in state.streams.iter().enumerate() {
            if stream.forced_dts > 0 {
                log::info!(
                    "{}: forced the DTS of {} packet(s) of stream {index} forward",
                    self.name,
                    stream.forced_dts
                );
            }
        }
        state.streams.clear();
    }

    fn log_drops(&self, state: &State) {
        if state.dropped_early > 0 {
            log::info!(
                "{}: dropped {} packet(s) that arrived before the container description",
                self.name,
                state.dropped_early
            );
        }
        if state.dropped_unknown > 0 {
            log::warn!(
                "{}: dropped {} packet(s) addressed to a stream this container has not got",
                self.name,
                state.dropped_unknown
            );
        }
    }
}

/// `FF_COMPLIANCE_NORMAL`, which the bindings generate as unsigned.
const COMPLIANCE_NORMAL: i32 = ffi::FF_COMPLIANCE_NORMAL as i32;
