//! `mux` — merges several packet streams into one in DTS order and describes the
//! output container. Port of C++ `src/nodes/mux.cpp`.
//!
//! Two things C++ did through the `IMuxer`/`IEncoder` handshake happen here over
//! the edge instead. C++ let `output` drive a three-phase
//! `initFromFormatContext` → `…PostOpenPreWriteHeader` → `…PostOpen` walk, during
//! which this node reached *up* each input with `findNodeUp<IEncoder>()` to fetch
//! a codec, created the `AVStream` for it and handed the stream back to the
//! encoder. Under Spec flow each input publishes its own [`Spec::Packet`], and
//! this node merges them into one [`Spec::Mux`] on its output — the whole
//! container description, in stream-index order, before a single packet follows
//! it. Emission is gated on that message, so `output` can never see a packet for
//! a stream it has not created yet.
//!
//! Consequences worth knowing:
//!
//! * **Stream order is the order of `src`**, and each input pad is named after
//!   the edge it takes: `src[i]` is output stream `i`. C++ derived the index from
//!   `octx.addStream()` instead, which amounted to the same order.
//! * **`fix_timestamps` works in each packet's own time base.** C++ rescaled into
//!   the output `AVStream`'s time base first, which it knew from the third phase
//!   of the handshake; this node owns no `AVFormatContext` (design doc §7: "mux
//!   and output share an edge, not an `AVFormatContext`"), so `output` does that
//!   rescale — and repeats the monotonic-DTS guard afterwards, because rescaling
//!   can collapse two distinct DTS values into one.
//! * **`allow_no_encoder` is accepted and does nothing**: a `Spec::Packet` on an
//!   input is all this node needs, encoder or not. An input that reaches EOF
//!   without ever publishing one is an error, since its stream cannot be created.
//!
//! Deliberate deviations from C++, both documented at their site: a packet with a
//! PTS but no DTS is muxed rather than dropped, and an input that is already
//! `known_to_be_broken` no longer consumes another input's `ts_sort_wait` grace.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::buffer::{AvpMediaType, AvpRational};
use avplumber_f7k::graph::edge::{Edge, EdgeEvent, EdgeItem, Push};
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::media::{Media, Ts};
use avplumber_f7k::graph::node::{Node, NodeBody, NodeKind, Tick};
use avplumber_f7k::graph::pad::{NodePads, PadDecl};
use avplumber_f7k::graph::poll_ctx::NodePollContext;
use avplumber_f7k::graph::spec::{MuxStream, PacketSpec, Spec};
use avplumber_f7k::graph::timebase::{MILLISECONDS, tb_cmp, ts_cmp};
use avplumber_f7k::scaffold::{EdgeSlot, PollStep, poll_body};

#[cfg(feature = "ffmpeg")]
use avplumber_f7k::graph::media::PacketExt;

/// C++ `sync_wait_max_ms_`.
const DEFAULT_TS_SORT_WAIT_MS: i64 = 2500;

/// How many consecutive shifted packets on one stream make the fixing severe
/// enough to move every stream instead. C++ `calculateGlobalShift`.
const SEVERE_SHIFT_PACKETS: u64 = 10;

#[derive(Debug, serde::Deserialize)]
pub struct MuxSpec {
    /// The envelope's own `src` list, which the control layer leaves in the node
    /// parameters: this is what fixes the output stream order, so the node reads
    /// it instead of relying on the order its pads happen to be bound in (a
    /// reconstruction rebinds from a map, in which order is arbitrary).
    ///
    /// Build-time only: parsed into [`StreamMuxer::pad_names`].
    #[serde(default)]
    src: Option<Value>,
    /// Force strictly increasing DTS per stream. Off by default, like C++.
    #[serde(default)]
    fix_timestamps: bool,
    /// Seconds in a script, milliseconds inside; `0` disables the grace.
    #[serde(default)]
    ts_sort_wait: Option<f64>,
    /// Accepted for script compatibility only, see the module docs.
    #[serde(default)]
    allow_no_encoder: Option<bool>,
    /// Container-level stream ids, positional over `src`. Build-time only: folded
    /// into the per-input state.
    #[serde(default)]
    stream_ids: Vec<i32>,
    /// Per-stream metadata dictionaries, positional over `src`. Build-time only,
    /// like `stream_ids`.
    #[serde(default)]
    metadata: Vec<Value>,
}

impl NodeSpec for MuxSpec {
    const TYPE_NAME: &'static str = "mux";
    type Node = StreamMuxer;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        let names = src_names(self.src.as_ref())?;
        if self.stream_ids.len() > names.len() {
            return Err("too many stream_ids".into());
        }
        if self.metadata.len() > names.len() {
            return Err("too many metadata dictionaries in list".into());
        }
        if self.allow_no_encoder == Some(false) {
            // The one case where the parameter would have changed C++ behaviour is
            // the default, so saying nothing here would be misleading.
            log::info!(
                "{name}: allow_no_encoder has no effect in the Rust core; an input's own spec \
                 describes its stream"
            );
        }

        let mut inputs = Vec::with_capacity(names.len());
        for (i, edge_name) in names.iter().enumerate() {
            if names[..i].contains(edge_name) {
                return Err(format!("`{edge_name}` is listed twice in src"));
            }
            inputs.push(Input {
                name: edge_name.clone(),
                index: i as i32,
                id: self.stream_ids.get(i).copied(),
                metadata: match self.metadata.get(i) {
                    Some(value) => metadata_pairs(value)?,
                    None => Vec::new(),
                },
                edge: None,
                spec: None,
                head: None,
                at_eof: false,
                idle: false,
                idle_since: None,
                warned: false,
                known_to_be_broken: false,
                prev_dts: Ts::invalid(),
                shift: 0,
                shifted_for: 0,
            });
        }

        let sync_wait_max_ms = match self.ts_sort_wait {
            Some(seconds) => (seconds * 1000.0) as i64,
            None => DEFAULT_TS_SORT_WAIT_MS,
        };
        Ok(StreamMuxer {
            name: name.into(),
            params: self,
            sync_wait_max_ms,
            pad_names: names,
            out: EdgeSlot::default(),
            state: Mutex::new(State::new(inputs)),
        })
    }
}

/// The `src` list as ordered edge names. `control::bind_named` binds them
/// positionally against the pads declared from this same list, so pad `i` and
/// `src[i]` are the same name either way.
fn src_names(src: Option<&Value>) -> Result<Vec<String>, String> {
    match src {
        Some(Value::String(one)) => Ok(vec![one.clone()]),
        Some(Value::Array(items)) => items
            .iter()
            .map(|item| {
                item.as_str()
                    .map(str::to_string)
                    .ok_or_else(|| "every entry of src must be an edge name".to_string())
            })
            .collect(),
        Some(_) => Err("src must be an edge name or a list of them".into()),
        None => Err("mux needs its input edges listed in src".into()),
    }
}

/// One `metadata` entry, stringified exactly as C++ `parametersToDict` does.
/// Kept out of [`avplumber_f7k::libav::dict`] because the description travels as plain
/// pairs — only `output` turns them into an `AVDictionary`.
fn metadata_pairs(value: &Value) -> Result<Vec<(String, String)>, String> {
    match value {
        Value::Null => Ok(Vec::new()),
        Value::Object(map) => Ok(map
            .iter()
            .map(|(key, value)| {
                let value = match value {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                (key.clone(), value)
            })
            .collect()),
        _ => Err("every metadata entry must be an object of \"key\": value pairs".into()),
    }
}

struct Input {
    /// The edge this input takes from, which is also its pad name.
    name: String,
    /// Output stream index: this input's position in `src`.
    index: i32,
    id: Option<i32>,
    metadata: Vec<(String, String)>,
    edge: Option<Arc<dyn Edge>>,
    /// What this input carries, for the [`Spec::Mux`] description.
    spec: Option<PacketSpec>,
    /// The packet taken but not yet muxed. C++ `peek`s the queue instead; owning
    /// the head means nothing is cloned.
    head: Option<Media>,
    at_eof: bool,
    /// Set every round: whether this input had nothing to offer.
    idle: bool,
    /// Media time (ms) of the least packet when this input first went idle, so
    /// the grace is a stream-time budget. C++ `idle_since`.
    idle_since: Option<i64>,
    /// Whether the sync-wait warning has been logged for the current idle spell.
    warned: bool,
    /// Nothing more is expected here, so it must not hold the others up.
    known_to_be_broken: bool,
    /// Last DTS emitted for this stream, for `fix_timestamps`.
    prev_dts: Ts,
    /// How far the last packet had to be moved, in `prev_dts`'s time base.
    shift: i64,
    /// Consecutive shifted packets.
    shifted_for: u64,
}

impl Input {
    /// The bound edge. [`StreamMuxer::step`] fails the node before anything below
    /// it runs if a pad is unbound, so this cannot fire.
    fn edge(&self) -> &Arc<dyn Edge> {
        self.edge.as_ref().expect("mux checks its bindings first")
    }

    /// Everything a restart must forget; the configuration above it survives.
    fn reset(&mut self) {
        self.spec = None;
        self.head = None;
        self.at_eof = false;
        self.idle = false;
        self.idle_since = None;
        self.warned = false;
        self.known_to_be_broken = false;
        self.prev_dts = Ts::invalid();
        self.shift = 0;
        self.shifted_for = 0;
    }
}

struct State {
    inputs: Vec<Input>,
    /// A packet already taken from an input whose output had no room.
    pending: Option<Media>,
    /// Whether the [`Spec::Mux`] description has gone out.
    published: bool,
    /// The amount every stream is moved by, once the fixing turned severe.
    global_shift: Ts,
    /// The deadline armed on the previous poll, i.e. the C++ `wait(max_wait)`
    /// this node is in the middle of.
    wait_until: Option<Instant>,
    dropped_nopts: u64,
}

impl State {
    fn new(inputs: Vec<Input>) -> Self {
        Self {
            inputs,
            pending: None,
            published: false,
            global_shift: Ts {
                val: 0,
                tb: AvpRational { num: 1, den: 1 },
            },
            wait_until: None,
            dropped_nopts: 0,
        }
    }
}

pub struct StreamMuxer {
    name: String,
    /// What the script asked for, verbatim: [`MuxSpec`] documents each field, and
    /// holding it whole is what keeps them from being declared twice.
    params: MuxSpec,
    /// `params.ts_sort_wait` in milliseconds, C++ `sync_wait_max_ms_`; `<= 0`
    /// disables the grace.
    sync_wait_max_ms: i64,
    /// The input pad names, so [`Node::pads`] does not take the state lock.
    pad_names: Vec<String>,
    out: EdgeSlot,
    state: Mutex<State>,
}

impl Node for StreamMuxer {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }

    /// One source pad per `src` entry, named after it. Declaring them (rather
    /// than leaving the pads implicit) is what makes the stream indices survive a
    /// reconstruction: pads are rebound from a map, but this list comes from the
    /// stored parameters, in order.
    fn pads(&self) -> NodePads {
        NodePads {
            sources: self
                .pad_names
                .iter()
                .map(|name| PadDecl {
                    name: name.clone(),
                    media: AvpMediaType::PACKET,
                })
                .collect(),
            sinks: vec![PadDecl {
                name: "out".into(),
                media: AvpMediaType::PACKET,
            }],
        }
    }

    fn bind_source(&self, pad: &str, edge: Arc<dyn Edge>) {
        let mut state = self.state.lock().unwrap();
        match state.inputs.iter_mut().find(|input| input.name == pad) {
            Some(input) => input.edge = Some(edge),
            None => log::warn!("{}: no input pad `{pad}` to bind", self.name),
        }
    }

    fn bind_sink(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.out.bind(edge);
    }

    fn start(&self) {
        let state = &mut *self.state.lock().unwrap();
        for input in &mut state.inputs {
            input.reset();
        }
        state.pending = None;
        state.published = false;
        state.global_shift = Ts {
            val: 0,
            tb: AvpRational { num: 1, den: 1 },
        };
        state.wait_until = None;
        state.dropped_nopts = 0;
    }

    fn take_body(self: Arc<Self>) -> NodeBody {
        poll_body(self)
    }
}

impl PollStep for StreamMuxer {
    fn step(&self, ctx: &mut NodePollContext) -> Result<Tick, NodeError> {
        let out = self.out.require(&self.name, NodePhase::Poll, "output")?;
        let state = &mut *self.state.lock().unwrap();

        if let Some(unbound) = state.inputs.iter().find(|input| input.edge.is_none()) {
            // A muxer cannot do without one of its streams, so this fails the
            // group instead of stalling on an input that will never speak.
            return Err(self.error(
                NodePhase::Poll,
                format!("input `{}` is not bound", unbound.name),
            ));
        }

        // A packet already taken goes out before anything new is considered, so a
        // full output cannot lose it.
        if let Some(buffer) = state.pending.take() {
            match self.emit(state, ctx, &out, buffer) {
                Emitted::Ok => {}
                Emitted::Parked => return Ok(Tick::Idle),
                Emitted::Closed => return Ok(Tick::Done),
            }
        }

        // Did the grace armed on the previous poll run out? Taken here so no path
        // below can act on a stale deadline. An input that is still idle keeps its
        // media-time `idle_since`, so re-arming never extends its budget.
        let grace_expired = state
            .wait_until
            .take()
            .is_some_and(|deadline| Instant::now() >= deadline);

        let published = state.published;
        for input in &mut state.inputs {
            self.fill(input, published);
        }

        if state
            .inputs
            .iter()
            .all(|input| input.at_eof && input.head.is_none())
        {
            // Describe the container on the way out when every input managed to
            // describe itself: `output` turns a description plus an EOF into a
            // valid empty file, while an EOF on its own leaves no file at all.
            // An input that finished without a spec is left alone — its stream
            // cannot be created, and shutting down cleanly is friendlier than
            // failing a graph whose source simply held nothing.
            if !state.published && state.inputs.iter().all(|input| input.spec.is_some()) {
                self.publish(state, &out)?;
            }
            self.log_drops(state);
            out.push_event(EdgeEvent::Eof);
            return Ok(Tick::Done);
        }

        // Nothing may reach `output` before it knows which streams to create.
        if !state.published {
            match self.publish(state, &out)? {
                Published::Ok => {}
                Published::Waiting => {
                    for input in &state.inputs {
                        if input.spec.is_none() {
                            ctx.wait_readable(input.edge().clone());
                        }
                    }
                    return Ok(Tick::Idle);
                }
            }
        }

        // The least DTS across the inputs. C++ `mux.cpp:105-147`.
        let mut least: Option<(usize, Ts)> = None;
        let mut candidates = 0usize;
        let mut nopts = 0u64;
        for (i, input) in state.inputs.iter_mut().enumerate() {
            input.idle = true;
            let usable = match &input.head {
                Some(buffer) => {
                    let ts = order_ts(buffer);
                    if ts.is_valid() {
                        Some(ts)
                    } else {
                        // Nothing can order it. Dropped, and retried at once
                        // rather than waited on: the next item may already be in
                        // the queue behind it, and its wake is long consumed.
                        log::warn!(
                            "{}: dropping a packet with no timestamp on `{}`",
                            self.name,
                            input.name
                        );
                        input.head = None;
                        nopts += 1;
                        None
                    }
                }
                None => None,
            };
            match usable {
                Some(ts) => {
                    input.idle = false;
                    input.known_to_be_broken = false;
                    input.idle_since = None;
                    input.warned = false;
                    candidates += 1;
                    if least
                        .is_none_or(|(_, best)| ts_cmp(ts.val, ts.tb, best.val, best.tb).is_lt())
                    {
                        least = Some((i, ts));
                    }
                }
                None => {
                    // An input at EOF has nothing more to give; C++ marks it on
                    // its EOF marker, which is what lets the others proceed.
                    if input.at_eof {
                        input.known_to_be_broken = true;
                    }
                    if input.known_to_be_broken {
                        candidates += 1;
                    }
                }
            }
        }
        state.dropped_nopts += nopts;

        let Some((least_i, least_ts)) = least else {
            if nopts > 0 {
                return Ok(Tick::Again);
            }
            // Nothing to order: wait for whichever inputs can still deliver.
            for input in &state.inputs {
                if !input.at_eof {
                    ctx.wait_readable(input.edge().clone());
                }
            }
            return Ok(Tick::Idle);
        };

        let have_all = candidates == state.inputs.len();
        if grace_expired {
            // The grace ran out with nothing new, so stop holding the muxing up:
            // C++ `should_emit = !got_packet`.
            for input in state.inputs.iter_mut().filter(|input| input.idle) {
                input.known_to_be_broken = true;
            }
        } else if !have_all
            && self.sync_wait_max_ms > 0
            && self.grace(state, ctx, least_i, least_ts, candidates) == Grace::Park
        {
            return Ok(Tick::Idle);
        }

        let index = state.inputs[least_i].index;
        let Some(mut buffer) = state.inputs[least_i].head.take() else {
            return Ok(Tick::Again);
        };
        if self.params.fix_timestamps {
            self.fix(state, least_i, &mut buffer);
            self.calculate_global_shift(state);
        }
        set_index(&mut buffer, index);
        match self.emit(state, ctx, &out, buffer) {
            Emitted::Ok => Ok(Tick::Again),
            Emitted::Parked => Ok(Tick::Idle),
            Emitted::Closed => Ok(Tick::Done),
        }
    }
}

/// Where one `offer` left the buffer.
enum Emitted {
    Ok,
    /// The output was full; the buffer is stashed and the node is parked on it.
    Parked,
    Closed,
}

/// Whether the container description could be assembled yet.
enum Published {
    Ok,
    Waiting,
}

/// What the `ts_sort_wait` grace decided.
#[derive(PartialEq, Eq)]
enum Grace {
    /// Every input has either delivered or run out of grace.
    Emit,
    /// Parked on the idle inputs and on the remaining grace.
    Park,
}

impl StreamMuxer {
    fn error(&self, phase: NodePhase, message: impl Into<String>) -> NodeError {
        NodeError::new(&self.name, phase, message)
    }

    /// Drains one input until a buffer sits at its head, recording the events on
    /// the way. The C++ `peek`, minus the copy.
    fn fill(&self, input: &mut Input, published: bool) {
        if input.head.is_some() || input.at_eof {
            return;
        }
        let edge = input.edge().clone();
        loop {
            match edge.try_take() {
                Some(EdgeItem::Buffer(buffer)) => {
                    input.head = Some(buffer);
                    return;
                }
                Some(EdgeItem::Event(EdgeEvent::Spec(Spec::Packet(spec)))) => {
                    if published {
                        // The container has been described and `output` has its
                        // header; a stream cannot be re-declared now.
                        log::warn!(
                            "{}: `{}` re-published its spec after the container was described, \
                             keeping the original stream",
                            self.name,
                            input.name
                        );
                    } else {
                        input.spec = Some(spec);
                    }
                }
                Some(EdgeItem::Event(EdgeEvent::Spec(other))) => {
                    log::warn!(
                        "{}: ignoring a {} spec on `{}`; a muxer input carries encoded packets",
                        self.name,
                        other.variant_name(),
                        input.name
                    );
                }
                Some(EdgeItem::Event(EdgeEvent::Eof)) => {
                    input.at_eof = true;
                    return;
                }
                Some(EdgeItem::Event(EdgeEvent::FlushStart)) => {
                    // Only this input's timestamp tracking is reset, and the
                    // marker is not forwarded: `output` has already written a
                    // header for these streams, and what a muxer should do with a
                    // flush belongs with seek support (out of scope).
                    input.prev_dts = Ts::invalid();
                    input.shift = 0;
                    input.shifted_for = 0;
                }
                Some(EdgeItem::Event(EdgeEvent::FlushStop)) => {}
                None => {
                    if edge.is_closed() {
                        log::debug!(
                            "{}: `{}` closed without an EOF marker, treating it as finished",
                            self.name,
                            input.name
                        );
                        input.at_eof = true;
                    }
                    return;
                }
            }
        }
    }

    /// The whole output container, in stream-index order — what replaces C++'s
    /// three-phase handshake with `output`.
    fn publish(&self, state: &mut State, out: &Arc<dyn Edge>) -> Result<Published, NodeError> {
        let mut streams = Vec::with_capacity(state.inputs.len());
        for input in &state.inputs {
            let Some(spec) = &input.spec else {
                if input.at_eof {
                    return Err(self.error(
                        NodePhase::Spec,
                        format!(
                            "`{}` reached EOF without ever describing its stream, so the output \
                             container cannot be built",
                            input.name
                        ),
                    ));
                }
                return Ok(Published::Waiting);
            };
            streams.push(MuxStream {
                spec: spec.clone(),
                id: input.id,
                metadata: input.metadata.clone(),
            });
        }
        log::info!(
            "{}: output container of {} stream(s): {:?}",
            self.name,
            streams.len(),
            self.pad_names
        );
        out.push_event(EdgeEvent::Spec(Spec::Mux { streams }));
        state.published = true;
        Ok(Published::Ok)
    }

    /// How long an input with nothing to offer may hold the others up. C++
    /// `mux.cpp:161-215`, with its blocking `wait(max_wait)` turned into a park on
    /// the idle inputs plus a deadline: the next poll sees `grace_expired` and
    /// emits, exactly like C++'s `should_emit = !got_packet`.
    fn grace(
        &self,
        state: &mut State,
        ctx: &mut NodePollContext,
        least_i: usize,
        least_ts: Ts,
        candidates: usize,
    ) -> Grace {
        // Anchored in the *media* time of the least packet, like C++: the budget
        // is stream time, and only the waiting itself is wall clock.
        let now_ms = least_ts.rescale(MILLISECONDS).val;
        let least_name = state.inputs[least_i].name.clone();
        let total = state.inputs.len();
        let sync_wait = self.sync_wait_max_ms;
        let mut candidates = candidates;
        let mut max_wait = 0i64;
        // Deliberately skipping the broken inputs, which C++ does not: they were
        // counted as candidates already, and letting them run their grace down a
        // second time here spends a *live* input's budget for it.
        for input in state
            .inputs
            .iter_mut()
            .filter(|input| input.idle && !input.known_to_be_broken)
        {
            let Some(since) = input.idle_since else {
                input.idle_since = Some(now_ms);
                max_wait = max_wait.max(sync_wait);
                continue;
            };
            let mut diff = now_ms - since;
            if diff < 0 {
                log::warn!(
                    "{}: time went backwards in the muxer: `{}` idle since {since} ms, `{}` is at \
                     {now_ms} ms",
                    self.name,
                    input.name,
                    least_name
                );
                diff = 0;
            }
            let to_wait = sync_wait - diff;
            if to_wait <= 0 {
                // Waited long enough: treat it as if it had nothing left to give.
                candidates += 1;
                if !input.warned {
                    input.warned = true;
                    log::warn!(
                        "{}: sync wait timeout exceeded: {diff} ms on `{}`",
                        self.name,
                        input.name
                    );
                }
            } else {
                max_wait = max_wait.max(to_wait);
            }
        }
        if candidates == total {
            return Grace::Emit;
        }

        // Woken by either an idle input delivering or the grace running out; the
        // clamp only guards against a rounding surprise, since an input that still
        // has grace left leaves `max_wait` positive.
        for input in &state.inputs {
            if input.idle {
                ctx.wait_readable(input.edge().clone());
            }
        }
        let deadline = Instant::now() + Duration::from_millis(max_wait.max(1) as u64);
        state.wait_until = Some(deadline);
        ctx.wait_deadline(deadline);
        Grace::Park
    }

    /// C++ `fix_timestamps`: force strictly increasing DTS per stream, keep PTS at
    /// or after DTS, and record how far the packet had to move so every stream can
    /// be shifted by the same amount later.
    ///
    /// In the packet's own time base (see the module docs), and skipped entirely
    /// for a packet without a DTS — there is nothing to force monotonic, and C++
    /// never got here because it dropped such packets outright.
    fn fix(&self, state: &mut State, i: usize, buffer: &mut Media) {
        let global_shift = state.global_shift;
        let input = &mut state.inputs[i];
        let mut pts = buffer.ts();
        let mut dts = dts_of(buffer);
        if !dts.is_valid() {
            return;
        }
        if global_shift.val != 0 {
            dts = dts + global_shift.rescale(dts.tb);
            if pts.is_valid() {
                pts = pts + global_shift.rescale(pts.tb);
            }
        }
        if input.prev_dts.is_valid() {
            if ts_cmp(dts.val, dts.tb, input.prev_dts.val, input.prev_dts.tb).is_le() {
                // C++ compares raw values, assuming one time base per stream; the
                // rescale keeps this right if a producer ever changes its own.
                let forced = input.prev_dts.rescale(dts.tb).val + 1;
                log::info!(
                    "{}: non-increasing DTS on `{}`: {} -> {}, fixing to {forced}",
                    self.name,
                    input.name,
                    input.prev_dts.val,
                    dts.val
                );
                input.shift = forced - dts.val;
                input.shifted_for += 1;
                dts = Ts {
                    val: forced,
                    tb: dts.tb,
                };
            } else {
                input.shift = 0;
                input.shifted_for = 0;
            }
        }
        if pts.is_valid() && ts_cmp(pts.val, pts.tb, dts.val, dts.tb).is_lt() {
            log::info!(
                "{}: PTS < DTS on `{}`: {} < {}, fixing",
                self.name,
                input.name,
                pts.val,
                dts.val
            );
            pts = dts;
        }
        input.prev_dts = dts;
        set_stamps(buffer, pts, dts);
    }

    /// C++ `calculateGlobalShift`: once one stream has needed shifting for more
    /// than [`SEVERE_SHIFT_PACKETS`] packets in a row, move *every* stream by the
    /// same amount instead, so the fixing cannot desync audio from video.
    fn calculate_global_shift(&self, state: &mut State) {
        if !state
            .inputs
            .iter()
            .any(|input| input.shifted_for > SEVERE_SHIFT_PACKETS)
        {
            return;
        }
        let mut max_shift = Ts {
            val: 0,
            tb: AvpRational { num: 1, den: 1 },
        };
        let mut coarsest: Option<AvpRational> = None;
        for input in state
            .inputs
            .iter_mut()
            .filter(|input| input.prev_dts.is_valid())
        {
            let tb = input.prev_dts.tb;
            log::info!(
                "{}: `{}` was shifted by {} in {}/{}",
                self.name,
                input.name,
                input.shift,
                tb.num,
                tb.den
            );
            if ts_cmp(input.shift, tb, max_shift.val, max_shift.tb).is_gt() {
                max_shift = Ts {
                    val: input.shift,
                    tb,
                };
            }
            // The *coarsest* time base wins, so the shared shift cannot be rounded
            // to a different amount per stream — which is the desync this whole
            // detour exists to avoid.
            if coarsest.is_none_or(|best| tb_cmp(tb, best).is_gt()) {
                coarsest = Some(tb);
            }
            input.shift = 0;
            input.shifted_for = 0;
        }
        let Some(coarsest) = coarsest else { return };
        let target = max_shift + state.global_shift;
        let mut shifted = target.rescale(coarsest);
        if ts_cmp(shifted.val, shifted.tb, target.val, target.tb).is_lt() {
            // Never round down: a shift that shrinks re-introduces the collision
            // it was computed to fix.
            shifted = Ts {
                val: shifted.val + 1,
                tb: coarsest,
            };
        }
        log::info!(
            "{}: shifting everything by {} in {}/{}",
            self.name,
            shifted.val,
            coarsest.num,
            coarsest.den
        );
        state.global_shift = shifted;
    }

    fn emit(
        &self,
        state: &mut State,
        ctx: &mut NodePollContext,
        out: &Arc<dyn Edge>,
        buffer: Media,
    ) -> Emitted {
        match out.offer(buffer) {
            Ok(()) => Emitted::Ok,
            Err((Push::Full, buffer)) => {
                state.pending = Some(buffer);
                ctx.wait_writable(out.clone());
                Emitted::Parked
            }
            Err((Push::Closed, _)) => {
                log::info!("{}: output edge closed, finishing", self.name);
                Emitted::Closed
            }
            // The edge took it and discarded it: nothing left to retry with.
            Err((Push::Dropped | Push::Accepted, _)) => Emitted::Ok,
        }
    }

    fn log_drops(&self, state: &State) {
        if state.dropped_nopts > 0 {
            log::info!(
                "{}: dropped {} packet(s) that carried no timestamp",
                self.name,
                state.dropped_nopts
            );
        }
    }
}

/// The timestamp a muxer orders by: DTS, which is decode order, and PTS when the
/// producer left no DTS.
///
/// C++ ordered by the same pair but then *dropped* any packet whose DTS was
/// invalid. Deliberately not reproduced: a container that carries only PTS (a
/// remux without a decoder in between) would lose every packet, and `output`
/// rescales the two timestamps independently anyway.
fn order_ts(buffer: &Media) -> Ts {
    let dts = dts_of(buffer);
    if dts.is_valid() { dts } else { buffer.ts() }
}

#[cfg(feature = "ffmpeg")]
fn dts_of(buffer: &Media) -> Ts {
    match buffer {
        Media::Packet(packet) => packet.dts(),
        other => other.ts(),
    }
}

/// Without libav there are no packets, so the ordering half of this node is what
/// [`Media::Stub`] can exercise: it carries one timestamp, which stands in for
/// both.
#[cfg(not(feature = "ffmpeg"))]
fn dts_of(buffer: &Media) -> Ts {
    buffer.ts()
}

#[cfg(feature = "ffmpeg")]
fn set_stamps(buffer: &mut Media, pts: Ts, dts: Ts) {
    if let Media::Packet(packet) = buffer {
        packet.set_ts_dts(pts, dts);
    }
}

#[cfg(not(feature = "ffmpeg"))]
fn set_stamps(_buffer: &mut Media, _pts: Ts, _dts: Ts) {}

#[cfg(feature = "ffmpeg")]
fn set_index(buffer: &mut Media, index: i32) {
    if let Media::Packet(packet) = buffer {
        packet.set_stream_index(index);
    }
}

#[cfg(not(feature = "ffmpeg"))]
fn set_index(_buffer: &mut Media, _index: i32) {}

#[cfg(test)]
mod tests {
    use super::*;
    use avplumber_f7k::Instance;
    use avplumber_f7k::graph::buffer::AVP_NOPTS;
    use avplumber_f7k::graph::buffered_edge::BufferedEdge;
    use avplumber_f7k::graph::edge::Wakeup;
    use avplumber_f7k::graph::media::test_media;
    use std::sync::atomic::AtomicBool;

    /// A muxer plus its edges, stepped by hand.
    ///
    /// No executor: `step` is called directly, so every decision the node makes
    /// is observable one poll at a time, and the `ts_sort_wait` grace can be
    /// waited out with a `sleep` instead of a whole runtime.
    struct Harness {
        node: StreamMuxer,
        inputs: Vec<Arc<dyn Edge>>,
        out: Arc<dyn Edge>,
        ctx: NodePollContext,
    }

    /// What came out of the muxer, in order.
    #[derive(PartialEq, Eq, Debug)]
    enum Out {
        /// A buffer and the timestamp it carries.
        Buffer(i64),
        /// The container description, as its stream count.
        Mux(usize),
        Eof,
        Other,
    }

    impl Harness {
        /// One input pad per name, all edges empty. `params` is the script's, with
        /// `src` filled in from `names` the way the control layer leaves it.
        fn new(names: &[&str], mut params: Value) -> Self {
            params["src"] = names.to_vec().into();
            let spec: MuxSpec = serde_json::from_value(params.clone()).expect("mux parameters");
            // `MuxSpec::build` never touches the instance; it is here because the
            // signature asks for one.
            let instance = Instance::new();
            let node = spec
                .build(
                    "m",
                    &BuildCtx {
                        instance: &instance,
                        name: "m",
                        params: &params,
                    },
                )
                .expect("mux node");

            let inputs: Vec<Arc<dyn Edge>> = names
                .iter()
                .map(|_| Arc::new(BufferedEdge::new(16)) as Arc<dyn Edge>)
                .collect();
            for (name, edge) in names.iter().zip(&inputs) {
                node.bind_source(name, edge.clone());
            }
            let out: Arc<dyn Edge> = Arc::new(BufferedEdge::new(64));
            node.bind_sink("out", out.clone());
            node.start();

            Self {
                node,
                inputs,
                out,
                ctx: NodePollContext::new(
                    Arc::new(AtomicBool::new(false)),
                    Arc::new(Wakeup::new()),
                ),
            }
        }

        /// One input's `Spec::Packet`, in the same `1/1000` [`test_media`] stamps
        /// its buffers in.
        fn describe(&self, input: usize) {
            self.inputs[input]
                .push_event(EdgeEvent::Spec(Spec::Packet(PacketSpec::new(MILLISECONDS))));
        }

        fn feed(&self, input: usize, stamps: &[i64]) {
            for stamp in stamps {
                assert!(
                    self.inputs[input]
                        .offer(test_media(AvpMediaType::PACKET, *stamp))
                        .is_ok(),
                    "the test input edges are large enough for {} stamps",
                    stamps.len()
                );
            }
        }

        fn finish(&self, input: usize) {
            self.inputs[input].push_event(EdgeEvent::Eof);
        }

        fn step(&mut self) -> Tick {
            let tick = self.node.step(&mut self.ctx).expect("mux step");
            // What the executors do between polls; without it the park from one
            // step would leak into the next.
            self.ctx.clear_park();
            tick
        }

        /// Steps until the node finishes. `budget` only exists so a regression
        /// fails the test instead of hanging it.
        fn run_until_done(&mut self, budget: usize) {
            for _ in 0..budget {
                if self.step() == Tick::Done {
                    return;
                }
            }
            panic!("mux did not finish within {budget} steps");
        }

        fn drain(&self) -> Vec<Out> {
            let mut items = Vec::new();
            while let Some(item) = self.out.try_take() {
                items.push(match item {
                    EdgeItem::Buffer(buffer) => Out::Buffer(order_ts(&buffer).val),
                    EdgeItem::Event(EdgeEvent::Spec(Spec::Mux { streams })) => {
                        Out::Mux(streams.len())
                    }
                    EdgeItem::Event(EdgeEvent::Eof) => Out::Eof,
                    _ => Out::Other,
                });
            }
            items
        }
    }

    #[test]
    fn packets_leave_in_least_timestamp_order() {
        let mut mux = Harness::new(&["a", "b"], serde_json::json!({}));
        for input in [0, 1] {
            mux.describe(input);
        }
        // Interleaved so neither input is simply drained before the other, and
        // `a` outlives `b` so the tail of one stream after the other's EOF is
        // covered too.
        mux.feed(0, &[0, 40, 80]);
        mux.feed(1, &[10, 20, 30]);
        for input in [0, 1] {
            mux.finish(input);
        }

        mux.run_until_done(32);
        assert_eq!(
            mux.drain(),
            vec![
                Out::Mux(2),
                Out::Buffer(0),
                Out::Buffer(10),
                Out::Buffer(20),
                Out::Buffer(30),
                Out::Buffer(40),
                Out::Buffer(80),
                Out::Eof,
            ],
            "the container description comes first, then a merge, then one EOF"
        );
    }

    #[test]
    fn nothing_leaves_before_every_input_described_its_stream() {
        // The grace is off, so the only thing holding the muxer back is the
        // missing description.
        let mut mux = Harness::new(&["a", "b"], serde_json::json!({ "ts_sort_wait": 0.0 }));
        mux.describe(0);
        mux.feed(0, &[0]);

        assert_eq!(mux.step(), Tick::Idle);
        assert_eq!(
            mux.drain(),
            Vec::new(),
            "`output` must not see a packet for a stream it has not created"
        );

        mux.describe(1);
        assert_eq!(mux.step(), Tick::Again);
        assert_eq!(mux.drain(), vec![Out::Mux(2), Out::Buffer(0)]);
    }

    #[test]
    fn an_idle_input_holds_the_others_up_only_for_the_grace() {
        const GRACE_MS: u64 = 50;
        let mut mux = Harness::new(
            &["a", "b"],
            serde_json::json!({ "ts_sort_wait": GRACE_MS as f64 / 1000.0 }),
        );
        for input in [0, 1] {
            mux.describe(input);
        }
        // `b` says nothing at all: it is neither at EOF nor delivering, which is
        // exactly the input `ts_sort_wait` exists for.
        mux.feed(0, &[0]);

        assert_eq!(mux.step(), Tick::Idle);
        assert_eq!(
            mux.drain(),
            vec![Out::Mux(2)],
            "the description goes out at once, but the packet waits for `b`"
        );

        std::thread::sleep(Duration::from_millis(GRACE_MS * 2));
        assert_eq!(mux.step(), Tick::Again);
        assert_eq!(
            mux.drain(),
            vec![Out::Buffer(0)],
            "once the grace is spent, `b` stops holding the muxing up"
        );
    }

    #[test]
    fn a_packet_without_a_timestamp_is_dropped_and_the_next_one_tried_at_once() {
        let mut mux = Harness::new(&["a"], serde_json::json!({}));
        mux.describe(0);
        mux.feed(0, &[AVP_NOPTS, 10]);
        mux.finish(0);

        mux.run_until_done(16);
        assert_eq!(mux.drain(), vec![Out::Mux(1), Out::Buffer(10), Out::Eof]);
    }

    /// `fix_timestamps` rewrites the packet, which only a real `AVPacket` can
    /// carry — `Media::Stub` is immutable, so the default build sees the ordering
    /// half of this node and this case belongs to the libav one.
    #[cfg(feature = "ffmpeg")]
    #[test]
    fn fix_timestamps_forces_a_repeated_dts_forward() {
        let mut mux = Harness::new(&["a"], serde_json::json!({ "fix_timestamps": true }));
        mux.describe(0);
        // Repeated, then backwards: both are what a muxer rejects.
        mux.feed(0, &[100, 100, 90]);
        mux.finish(0);

        mux.run_until_done(16);
        assert_eq!(
            mux.drain(),
            vec![
                Out::Mux(1),
                Out::Buffer(100),
                Out::Buffer(101),
                Out::Buffer(102),
                Out::Eof,
            ]
        );
    }

    #[test]
    fn without_fix_timestamps_the_stamps_are_left_alone() {
        let mut mux = Harness::new(&["a"], serde_json::json!({}));
        mux.describe(0);
        mux.feed(0, &[100, 100, 90]);
        mux.finish(0);

        mux.run_until_done(16);
        assert_eq!(
            mux.drain(),
            vec![
                Out::Mux(1),
                Out::Buffer(100),
                Out::Buffer(100),
                Out::Buffer(90),
                Out::Eof,
            ],
            "ordering is this node's job; fixing is opt-in"
        );
    }

    #[test]
    fn stream_ids_and_metadata_are_positional_over_src() {
        let mut mux = Harness::new(
            &["a", "b"],
            serde_json::json!({
                "stream_ids": [7, 9],
                "metadata": [{ "language": "deu" }, { "language": "eng", "rotate": 90 }],
            }),
        );
        for input in [0, 1] {
            mux.describe(input);
            mux.finish(input);
        }

        // Every input is at EOF without a packet, so one step describes the
        // container and finishes.
        assert_eq!(mux.step(), Tick::Done);
        let Some(EdgeItem::Event(EdgeEvent::Spec(Spec::Mux { streams }))) = mux.out.try_take()
        else {
            panic!("mux must describe its container");
        };
        assert_eq!(streams.len(), 2);
        assert_eq!((streams[0].id, streams[1].id), (Some(7), Some(9)));
        assert_eq!(streams[0].metadata, vec![("language".into(), "deu".into())]);
        assert_eq!(
            streams[1].metadata,
            vec![
                ("language".to_string(), "eng".to_string()),
                // Non-string values are stringified, like C++ `parametersToDict`.
                ("rotate".to_string(), "90".to_string()),
            ]
        );
    }

    #[test]
    fn a_duplicated_src_entry_is_rejected() {
        let params = serde_json::json!({ "src": ["a", "a"] });
        let spec: MuxSpec = serde_json::from_value(params.clone()).expect("mux parameters");
        let instance = Instance::new();
        let built = spec.build(
            "m",
            &BuildCtx {
                instance: &instance,
                name: "m",
                params: &params,
            },
        );
        // `StreamMuxer` is not `Debug`, so the `Ok` side cannot be unwrapped into
        // a message.
        let Err(error) = built else {
            panic!("two streams cannot share one input edge");
        };
        assert!(error.contains("twice"), "unexpected error: {error}");
    }
}
