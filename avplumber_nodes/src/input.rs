//! `input` — opens a container and pushes its packets. Port of C++
//! `src/nodes/input.cpp`.
//!
//! Publishes [`Spec::Catalog`] on its output before the first packet: that is
//! what replaces C++ `findNodeUp<IStreamsInput>()`, and it is also the answer to
//! the [`EdgeHint`]s a `demux` posts back up this edge — a `streams_filter` this
//! node evaluates (only the `AVFormatContext`'s owner can) and the stream
//! selection it turns into `AVDISCARD_ALL` on everything unwanted.
//!
//! Deferred: `getObject("streams"/"programs")` (the Rust control layer has no
//! `object.get` yet), and seek.

use std::ffi::{CString, c_int, c_void};
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};
use std::sync::{Arc, Mutex};

use rsmpeg::avformat::AVFormatContextInput;
use rusty_ffmpeg::ffi;
use serde_json::Value;

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::buffer::{AvpMediaType, AvpRational};
use avplumber_f7k::graph::edge::{Edge, EdgeEvent, EdgeHint};
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::media::{Media, PacketExt, Ts};
use avplumber_f7k::graph::node::{Blocked, Node, NodeBody, NodeKind};
use avplumber_f7k::graph::pad::{NodePads, PadDecl};
use avplumber_f7k::graph::spec::{CatalogStream, PacketSpec, Spec, StreamSelection};
use avplumber_f7k::libav::codec;
use avplumber_f7k::libav::dict::Options;
use avplumber_f7k::libav::error::{av_error, code_of, is_eagain};
use avplumber_f7k::scaffold::{BlockingStep, EdgeSlot, Park, Pushed, blocking_body, push_blocking};

/// C++ `int timeout = 5`.
const DEFAULT_TIMEOUT_S: f64 = 5.0;
/// `deadline_us` value that means "wait as long as it takes".
const NO_DEADLINE: i64 = i64::MAX;

/// The node keeps this whole struct rather than copying fields out of it, so each
/// parameter is declared exactly once. `format`, `options` and `initial_timeout`
/// are consumed by the open in [`InputSpec::build`] and mean nothing afterwards;
/// `eof_mode` and `timeout` are validated there into the two derived fields on
/// [`StreamInput`].
#[derive(Debug, serde::Deserialize)]
pub struct InputSpec {
    url: String,
    /// Demuxer name, e.g. `mpegts`; inferred from the url when absent.
    #[serde(default)]
    format: Option<String>,
    /// Passed to `avformat_open_input`; unconsumed entries are logged.
    #[serde(default)]
    options: Option<Value>,
    /// Seconds a single blocking libav call may take; negative disables it.
    #[serde(default)]
    timeout: Option<f64>,
    /// Same, for opening the container. Defaults to `timeout`.
    #[serde(default)]
    initial_timeout: Option<f64>,
    /// `drain` (default) sends [`EdgeEvent::Eof`] downstream, `none` stays quiet.
    #[serde(default)]
    eof_mode: Option<String>,
    /// Milliseconds to keep the node alive after EOF.
    #[serde(default)]
    stop_delay: Option<i64>,
}

impl NodeSpec for InputSpec {
    const TYPE_NAME: &'static str = "input";
    type Node = StreamInput;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        let eof_drain = match self.eof_mode.as_deref() {
            None | Some("drain") => true,
            Some("none") => false,
            Some(other) => {
                return Err(format!(
                    "unknown eof_mode `{other}` (expected drain or none)"
                ));
            }
        };
        let timeout_s = self.timeout.unwrap_or(DEFAULT_TIMEOUT_S);
        let initial_timeout_s = self.initial_timeout.unwrap_or(timeout_s);

        // Opened here, like C++ `init()`: a bad url fails `node.add`, and a group
        // restart rebuilds the node and reopens.
        let interrupt = Arc::new(InterruptState::default());
        interrupt.arm(initial_timeout_s);
        let ctx = open_input(
            name,
            &self.url,
            self.format.as_deref(),
            self.options.as_ref(),
            &interrupt,
        );
        interrupt.disarm();
        let ctx = ctx?;
        log_streams(name, &self.url, &ctx);

        Ok(StreamInput {
            name: name.into(),
            params: self,
            timeout_s,
            eof_drain,
            interrupt,
            park: Arc::new(Park::default()),
            out: EdgeSlot::default(),
            state: Mutex::new(Some(State {
                ctx,
                filter: None,
                catalog_published: false,
                eof_sent: false,
                finish_at_us: None,
            })),
        })
    }
}

/// Read by the libav interrupt callback, written by the node's own thread and by
/// [`Node::interrupt`].
#[derive(Default)]
struct InterruptState {
    /// Monotonic microseconds past which a blocking libav call gives up.
    deadline_us: AtomicI64,
    should_end: AtomicBool,
    /// Set by the callback so the node can tell a timeout from a plain stop.
    timed_out: AtomicBool,
}

impl InterruptState {
    fn arm(&self, seconds: f64) {
        let deadline = if seconds.is_finite() && seconds >= 0.0 {
            now_us().saturating_add((seconds * 1e6) as i64)
        } else {
            NO_DEADLINE
        };
        self.deadline_us.store(deadline, Ordering::Release);
    }

    fn disarm(&self) {
        self.deadline_us.store(NO_DEADLINE, Ordering::Release);
    }

    fn should_end(&self) -> bool {
        self.should_end.load(Ordering::Acquire)
    }

    fn request_end(&self) {
        self.should_end.store(true, Ordering::Release);
    }

    fn take_timed_out(&self) -> bool {
        self.timed_out.swap(false, Ordering::AcqRel)
    }

    fn reset(&self) {
        self.should_end.store(false, Ordering::Release);
        self.timed_out.store(false, Ordering::Release);
        self.disarm();
    }
}

/// `av_gettime_relative`: monotonic, so a wallclock jump cannot fake a timeout.
fn now_us() -> i64 {
    unsafe { ffi::av_gettime_relative() }
}

unsafe extern "C" fn interrupt_cb(opaque: *mut c_void) -> c_int {
    // `opaque` is the `Arc<InterruptState>` the node keeps alive for exactly as
    // long as it owns the context that holds this callback.
    let state = unsafe { &*(opaque as *const InterruptState) };
    if state.should_end.load(Ordering::Acquire) {
        return 1;
    }
    let deadline = state.deadline_us.load(Ordering::Acquire);
    if deadline != NO_DEADLINE && now_us() > deadline {
        state.timed_out.store(true, Ordering::Release);
        return 1;
    }
    0
}

struct State {
    ctx: AVFormatContextInput,
    /// The `streams_filter` the last [`EdgeHint::StreamsFilter`] asked about; it
    /// is echoed in the catalog so the consumer can tell which question was
    /// answered.
    filter: Option<String>,
    catalog_published: bool,
    eof_sent: bool,
    /// `stop_delay` is running: monotonic microseconds to finish at.
    finish_at_us: Option<i64>,
}

pub struct StreamInput {
    name: String,
    /// What the script asked for, verbatim: [`InputSpec`] documents each field, and
    /// holding it whole is what keeps them from being declared twice.
    params: InputSpec,
    /// `params.timeout`, defaulted.
    timeout_s: f64,
    /// `params.eof_mode`, validated.
    eof_drain: bool,
    /// Aborts a blocking call *inside* libav, through the context's
    /// `AVIOInterruptCB`.
    interrupt: Arc<InterruptState>,
    /// Where the node waits on the Rust side: for room on a full output edge, and
    /// for the `stop_delay` countdown. [`Park`] documents why it is a field of its
    /// own rather than part of `state`.
    park: Arc<Park>,
    out: EdgeSlot,
    state: Mutex<Option<State>>,
}

impl Node for StreamInput {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> NodeKind {
        NodeKind::Blocking
    }

    fn pads(&self) -> NodePads {
        NodePads {
            sources: Vec::new(),
            sinks: vec![PadDecl {
                name: "out".into(),
                media: AvpMediaType::PACKET,
            }],
        }
    }

    fn bind_sink(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.out.bind(edge);
    }

    fn start(&self) {
        self.interrupt.reset();
        self.park.reset();
        if let Some(state) = self.state.lock().unwrap().as_mut() {
            // Re-publish the catalog for this run: the edge may have been reset,
            // and a consumer rebuilt with us needs the spec again.
            state.catalog_published = false;
            state.eof_sent = false;
            state.finish_at_us = None;
        }
    }

    /// Two blocking sites in two different worlds, so two wake mechanisms. This
    /// node declares no source pads, so the executor's "interrupt the node's
    /// source edges" pass finds nothing — this call is the only handle anyone has
    /// on a running `input`.
    fn interrupt(&self) {
        self.interrupt.request_end();
        self.park.interrupt();
    }

    fn stop(&self) {
        // C++ closes in the destructor. Closing here releases the file or socket
        // as soon as the body is done; packets already pushed are refcounted and
        // outlive the context.
        if self.state.lock().unwrap().take().is_some() {
            log::debug!("{}: closed {}", self.name, self.params.url);
        }
    }

    fn take_body(self: Arc<Self>) -> NodeBody {
        blocking_body(self)
    }
}

impl BlockingStep for StreamInput {
    fn step(&self) -> Result<Blocked, NodeError> {
        let out = self.out.require(&self.name, NodePhase::Process, "output")?;
        let mut guard = self.state.lock().unwrap();
        let state = guard
            .as_mut()
            .ok_or_else(|| self.error(NodePhase::Process, "the input is already closed"))?;

        if self.interrupt.should_end() {
            return Ok(Blocked::Done);
        }

        // EOF already announced, `stop_delay` running down.
        if let Some(finish_at) = state.finish_at_us {
            if now_us() >= finish_at {
                log::info!("{}: stop_delay elapsed, finishing input", self.name);
                return Ok(Blocked::Done);
            }
            self.park.wait(10);
            return Ok(Blocked::Again);
        }

        self.serve_hints(state, &out)?;

        self.interrupt.arm(self.timeout_s);
        let read = state.ctx.read_packet();
        self.interrupt.disarm();

        let mut packet = match read {
            Ok(Some(packet)) => packet,
            Ok(None) => return self.on_eof(state, &out),
            Err(error) => {
                if self.interrupt.should_end() {
                    log::info!("{}: read interrupted by stop", self.name);
                    return Ok(Blocked::Done);
                }
                if self.interrupt.take_timed_out() {
                    return Err(self.error(
                        NodePhase::Process,
                        format!(
                            "timeout of {}s exceeded reading {}",
                            self.timeout_s, self.params.url
                        ),
                    ));
                }
                // Unlike C++, which throws on every non-EOF code: a device that
                // has nothing right now is not a failure.
                if code_of(&error).is_some_and(is_eagain) {
                    return Ok(Blocked::Again);
                }
                return Err(self.error(
                    NodePhase::Process,
                    format!("reading {} failed: {error}", self.params.url),
                ));
            }
        };

        if packet.is_corrupt() {
            log::warn!("{}: got incomplete packet, dropping", self.name);
            return Ok(Blocked::Again);
        }
        if !packet.ts().is_valid() && !packet.dts().is_valid() {
            log::warn!("{}: got packet without PTS & DTS, dropping", self.name);
            return Ok(Blocked::Again);
        }

        // libavformat leaves `pkt.time_base` unset; everything downstream reads
        // timestamps through it, so it is stamped from the stream here, once.
        let index = packet.stream_index;
        let time_base = stream_time_base(&state.ctx, index).ok_or_else(|| {
            self.error(
                NodePhase::Process,
                format!("packet from unknown stream {index}"),
            )
        })?;
        let (pts, dts) = (packet.pts, packet.dts);
        packet.set_ts_dts(
            Ts {
                val: pts,
                tb: time_base,
            },
            Ts {
                val: dts,
                tb: time_base,
            },
        );

        match push_blocking(&self.park, &out, Media::Packet(packet), || {
            // The anti-deadlock rule: a producer with no room still answers the
            // question its consumer is parked on.
            self.serve_hints(state, &out)
        })? {
            Pushed::Ok => Ok(Blocked::Again),
            Pushed::Interrupted => Ok(Blocked::Done),
            Pushed::Closed => {
                log::info!("{}: output edge closed, finishing", self.name);
                Ok(Blocked::Done)
            }
        }
    }
}

impl StreamInput {
    fn error(&self, phase: NodePhase, message: impl Into<String>) -> NodeError {
        NodeError::new(&self.name, phase, message)
    }

    /// Drains the consumer's hints and (re-)publishes the catalog when it no
    /// longer answers the current question. Idempotent, and cheap when nothing
    /// changed: one uncontended lock.
    fn serve_hints(&self, state: &mut State, out: &Arc<dyn Edge>) -> Result<(), NodeError> {
        for hint in out.take_hints() {
            match hint {
                EdgeHint::StreamsFilter(filter) => {
                    if state.filter.as_deref() != Some(filter.as_str()) {
                        log::info!("{}: streams_filter is now `{filter}`", self.name);
                        state.filter = Some(filter);
                        state.catalog_published = false;
                    }
                }
                EdgeHint::Streams(selection) => self.apply_discard(state, &selection),
            }
        }
        if !state.catalog_published {
            self.publish_catalog(state, out)?;
        }
        Ok(())
    }

    fn publish_catalog(&self, state: &mut State, out: &Arc<dyn Edge>) -> Result<(), NodeError> {
        let filter = state.filter.clone();
        let matches = match filter.as_deref() {
            Some(filter) => self.match_streams(state, filter)?,
            None => vec![true; state.ctx.nb_streams as usize],
        };
        let streams = state
            .ctx
            .streams()
            .iter()
            .zip(matches)
            .map(|(stream, matches_filter)| {
                let codecpar = stream.codecpar();
                CatalogStream {
                    index: stream.index,
                    codec_type: codecpar.codec_type,
                    spec: PacketSpec::from_codecpar(
                        &codecpar,
                        stream.time_base.into(),
                        stream.avg_frame_rate.into(),
                    ),
                    matches_filter,
                }
            })
            .collect::<Vec<_>>();

        log::debug!(
            "{}: publishing a catalog of {} streams (filter {:?})",
            self.name,
            streams.len(),
            filter
        );
        // A control event, so it goes out even with the buffer side full.
        out.push_event(EdgeEvent::Spec(Spec::Catalog { filter, streams }));
        state.catalog_published = true;
        Ok(())
    }

    /// `avformat_match_stream_specifier` per stream. Only this node can evaluate
    /// a specifier, since only it owns the `AVFormatContext`.
    fn match_streams(&self, state: &mut State, filter: &str) -> Result<Vec<bool>, NodeError> {
        let specifier = CString::new(filter).map_err(|_| {
            self.error(
                NodePhase::Spec,
                format!("streams_filter `{filter}` contains a NUL"),
            )
        })?;
        let ctx = state.ctx.as_mut_ptr();
        let count = unsafe { (*ctx).nb_streams } as usize;
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let stream = unsafe { *(*ctx).streams.add(i) };
            let matched =
                unsafe { ffi::avformat_match_stream_specifier(ctx, stream, specifier.as_ptr()) };
            if matched < 0 {
                return Err(self.error(
                    NodePhase::Spec,
                    format!("invalid streams_filter `{filter}`: {}", av_error(matched)),
                ));
            }
            out.push(matched > 0);
        }
        Ok(out)
    }

    /// C++ `discardAllStreams()` + `enableStream(i)`: libavformat then skips the
    /// unwanted streams inside `av_read_frame`, so they cost no packet at all.
    ///
    /// One exception, and C++ has it too: whatever
    /// `avformat_find_stream_info` already pulled into libavformat's read-ahead
    /// buffer is delivered regardless — a packet or two per stream, once.
    fn apply_discard(&self, state: &mut State, selection: &StreamSelection) {
        for stream in state.ctx.streams_mut() {
            let discard = if selection.contains(stream.index) {
                ffi::AVDISCARD_DEFAULT
            } else {
                ffi::AVDISCARD_ALL
            };
            stream.set_discard(discard);
        }
        log::info!(
            "{}: demuxing streams {:?} only",
            self.name,
            selection.enabled
        );
    }

    fn on_eof(&self, state: &mut State, out: &Arc<dyn Edge>) -> Result<Blocked, NodeError> {
        log::info!("{}: end of input", self.name);
        if let Some(delay_ms) = self.params.stop_delay {
            // C++ sends the EOF marker here regardless of eof_mode.
            self.send_eof_once(state, out);
            state.finish_at_us = Some(now_us().saturating_add(delay_ms.saturating_mul(1000)));
            log::info!("{}: delaying finish by {delay_ms} ms", self.name);
            return Ok(Blocked::Again);
        }
        if self.eof_drain {
            self.send_eof_once(state, out);
        }
        Ok(Blocked::Done)
    }

    fn send_eof_once(&self, state: &mut State, out: &Arc<dyn Edge>) {
        if !state.eof_sent {
            out.push_event(EdgeEvent::Eof);
            state.eof_sent = true;
        }
    }
}

fn stream_time_base(ctx: &AVFormatContextInput, index: i32) -> Option<AvpRational> {
    ctx.streams()
        .iter()
        .find(|stream| stream.index == index)
        .map(|stream| stream.time_base.into())
}

fn log_streams(node: &str, url: &str, ctx: &AVFormatContextInput) {
    log::info!("{node}: opened {url}, {} streams:", ctx.nb_streams);
    for stream in ctx.streams() {
        let codecpar = stream.codecpar();
        let kind = match codecpar.codec_type {
            ffi::AVMEDIA_TYPE_VIDEO => "video",
            ffi::AVMEDIA_TYPE_AUDIO => "audio",
            ffi::AVMEDIA_TYPE_SUBTITLE => "subtitle",
            ffi::AVMEDIA_TYPE_DATA => "data",
            _ => "other",
        };
        log::info!(
            "{node}:   {}: {kind} {} tb {}/{}",
            stream.index,
            codec::codec_name(codecpar.codec_id),
            stream.time_base.num,
            stream.time_base.den
        );
    }
}

/// Raw ffi rather than [`AVFormatContextInput::builder`]: the builder allocates
/// its own context, leaving nowhere to install the interrupt callback before the
/// open — and the open is exactly the call a timeout must be able to abort.
fn open_input(
    node: &str,
    url: &str,
    format: Option<&str>,
    options: Option<&Value>,
    interrupt: &Arc<InterruptState>,
) -> Result<AVFormatContextInput, String> {
    let c_url = CString::new(url).map_err(|_| format!("url `{url}` contains a NUL"))?;
    let input_format = match format {
        Some(name) => {
            let c_name = CString::new(name).map_err(|_| "format contains a NUL".to_string())?;
            let found = unsafe { ffi::av_find_input_format(c_name.as_ptr()) };
            if found.is_null() {
                return Err(format!("unknown input format `{name}`"));
            }
            found
        }
        None => std::ptr::null(),
    };
    let mut options = Options::from_json(options)?;

    let mut ctx = unsafe { ffi::avformat_alloc_context() };
    if ctx.is_null() {
        return Err("avformat_alloc_context failed".into());
    }
    unsafe {
        (*ctx).interrupt_callback = ffi::AVIOInterruptCB {
            callback: Some(interrupt_cb),
            opaque: Arc::as_ptr(interrupt) as *mut c_void,
        };
    }

    let opened = unsafe {
        ffi::avformat_open_input(&mut ctx, c_url.as_ptr(), input_format, options.as_mut_ptr())
    };
    if opened < 0 {
        // `avformat_open_input` frees and nulls the context on failure.
        return Err(format!("opening {url} failed: {}", av_error(opened)));
    }
    let found = unsafe { ffi::avformat_find_stream_info(ctx, std::ptr::null_mut()) };
    if found < 0 {
        unsafe { ffi::avformat_close_input(&mut ctx) };
        return Err(format!(
            "finding stream info of {url} failed: {}",
            av_error(found)
        ));
    }
    options.warn_leftovers(node, "the demuxer");

    Ok(unsafe { AVFormatContextInput::from_raw(NonNull::new(ctx).expect("open_input succeeded")) })
}
