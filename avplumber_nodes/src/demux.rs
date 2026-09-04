//! `demux` — splits one packet stream into per-stream outputs. Port of C++
//! `src/nodes/demux.cpp`.
//!
//! Where C++ resolved its `routing` at `node.add` by walking up to the
//! `IStreamsInput` node, this node waits for the [`Spec::Catalog`] its producer
//! publishes, and answers with two [hints](EdgeHint) on the same edge: the
//! `streams_filter` only the container's owner can evaluate, and the stream
//! selection it should discard everything outside of. See
//! [`routing`](avplumber_f7k::graph::routing) for the key grammar.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::buffer::AvpMediaType;
use avplumber_f7k::graph::edge::{Edge, EdgeEvent, EdgeHint, EdgeItem, Push};
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::media::{Media, PacketExt};
use avplumber_f7k::graph::node::{Node, NodeBody, NodeKind, Tick};
use avplumber_f7k::graph::pad::{NodePads, PadDecl};
use avplumber_f7k::graph::poll_ctx::NodePollContext;
use avplumber_f7k::graph::routing::{self, MEDIA_TYPE_VIDEO, RouteKey};
use avplumber_f7k::graph::spec::{CatalogStream, Spec, StreamSelection};
use avplumber_f7k::scaffold::{EdgeSlot, PollStep, poll_body};

#[derive(Debug, serde::Deserialize)]
pub struct DemuxSpec {
    /// `{"<source spec>": "<destination edge>"}`, e.g. `{"v": "e_v", "?a:1": "e_a"}`.
    ///
    /// Build-time only: parsed into [`StreamDemuxer::routes`], which is what
    /// routing decisions read.
    routing: BTreeMap<String, String>,
    /// An ffmpeg stream specifier restricting which streams the `v`/`a`/`d`
    /// ordinals count. Evaluated upstream, by the node that owns the container.
    #[serde(default)]
    streams_filter: Option<String>,
    /// Drop everything until the first keyframe on a routed video stream.
    #[serde(default)]
    wait_for_keyframe: Option<bool>,
}

impl NodeSpec for DemuxSpec {
    const TYPE_NAME: &'static str = "demux";
    type Node = StreamDemuxer;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        if self.routing.is_empty() {
            return Err("routing must name at least one stream".into());
        }
        let routes = self
            .routing
            .iter()
            .map(|(key, edge)| {
                Ok(Route {
                    pad: key.clone(),
                    key: routing::parse_route_key(key)?,
                    edge_name: edge.clone(),
                })
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(StreamDemuxer {
            name: name.into(),
            routes,
            params: self,
            input: EdgeSlot::default(),
            outs: Mutex::new(BTreeMap::new()),
            state: Mutex::new(State::default()),
        })
    }

    /// The routing values *are* the output edge names, so a script does not
    /// repeat them in `dst`.
    fn bindings(&self) -> Vec<(avplumber_f7k::core::PadDirection, String, String)> {
        self.routing
            .iter()
            .map(|(key, edge)| (avplumber_f7k::core::PadDirection::Output, key.clone(), edge.clone()))
            .collect()
    }
}

struct Route {
    /// The routing key verbatim, which is also this route's output pad name.
    pad: String,
    key: RouteKey,
    edge_name: String,
}

#[derive(Default)]
struct State {
    /// Resolved `stream_index → output`, in catalog order. A handful of entries,
    /// so a linear scan beats a map.
    map: Vec<(i32, Arc<dyn Edge>)>,
    /// Just the indices, to recognise a re-delivered catalog that changes nothing.
    indices: Vec<i32>,
    /// Routed video streams, for `wait_for_keyframe`.
    video: Vec<i32>,
    resolved: bool,
    waiting_for_keyframe: bool,
    /// A packet taken from the input whose output had no room. Held here so a
    /// full output cannot lose it.
    pending: Option<(Arc<dyn Edge>, Media)>,
    dropped_early: u64,
    dropped_unrouted: u64,
}

pub struct StreamDemuxer {
    name: String,
    /// `params.routing`, parsed.
    routes: Vec<Route>,
    /// What the script asked for, verbatim: [`DemuxSpec`] documents each field, and
    /// holding it whole is what keeps them from being declared twice.
    params: DemuxSpec,
    input: EdgeSlot,
    outs: Mutex<BTreeMap<String, Arc<dyn Edge>>>,
    state: Mutex<State>,
}

impl Node for StreamDemuxer {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }

    fn pads(&self) -> NodePads {
        NodePads {
            sources: vec![PadDecl {
                name: "in".into(),
                media: AvpMediaType::PACKET,
            }],
            sinks: self
                .routes
                .iter()
                .map(|route| PadDecl {
                    name: route.pad.clone(),
                    media: AvpMediaType::PACKET,
                })
                .collect(),
        }
    }

    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        // At `node.add` time, well before any executor thread: the producer
        // normally drains this before it ever publishes a catalog, so the very
        // first one already answers our filter.
        self.post_filter_hint(&edge);
        self.input.bind(edge);
    }

    fn bind_sink(&self, pad: &str, edge: Arc<dyn Edge>) {
        self.outs.lock().unwrap().insert(pad.into(), edge);
    }

    fn start(&self) {
        let mut state = self.state.lock().unwrap();
        *state = State {
            waiting_for_keyframe: self.params.wait_for_keyframe.unwrap_or(false),
            ..State::default()
        };
        drop(state);
        // Re-posted for safety: a restart may have handed this edge a producer
        // that has never seen our filter.
        if let Some(input) = self.input.get() {
            self.post_filter_hint(&input);
        }
    }

    fn take_body(self: Arc<Self>) -> NodeBody {
        poll_body(self)
    }
}

impl PollStep for StreamDemuxer {
    fn step(&self, ctx: &mut NodePollContext) -> Result<Tick, NodeError> {
        let input = self.input.require(&self.name, NodePhase::Poll, "input")?;
        let mut state = self.state.lock().unwrap();

        // A held-back packet goes out before anything new is taken, so ordering
        // within a stream survives a full output.
        if let Some((edge, buffer)) = state.pending.take() {
            match self.emit(&mut state, ctx, edge, buffer) {
                Emitted::Ok => {}
                Emitted::Parked => return Ok(Tick::Idle),
                Emitted::Closed => return Ok(Tick::Done),
            }
        }

        let Some(item) = input.try_take() else {
            if input.is_closed() {
                return Ok(Tick::Done);
            }
            ctx.wait_readable(input);
            return Ok(Tick::Idle);
        };

        match item {
            EdgeItem::Event(EdgeEvent::Spec(spec)) => {
                self.on_catalog(&mut state, &input, spec)?;
                Ok(Tick::Again)
            }
            EdgeItem::Event(EdgeEvent::Eof) => {
                for (_, edge) in &state.map {
                    edge.push_event(EdgeEvent::Eof);
                }
                self.log_drops(&state);
                Ok(Tick::Done)
            }
            EdgeItem::Event(event @ (EdgeEvent::FlushStart | EdgeEvent::FlushStop)) => {
                for (_, edge) in &state.map {
                    edge.push_event(event.clone());
                }
                Ok(Tick::Again)
            }
            EdgeItem::Buffer(buffer) => self.route(&mut state, ctx, buffer),
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

impl StreamDemuxer {
    fn post_filter_hint(&self, input: &Arc<dyn Edge>) {
        if let Some(filter) = &self.params.streams_filter {
            input.post_hint(EdgeHint::StreamsFilter(filter.clone()));
        }
    }

    fn route(
        &self,
        state: &mut State,
        ctx: &mut NodePollContext,
        buffer: Media,
    ) -> Result<Tick, NodeError> {
        let index = match &buffer {
            Media::Packet(packet) => packet.stream_index,
            other => {
                return Err(NodeError::new(
                    &self.name,
                    NodePhase::Poll,
                    format!("expected a packet, got {:?}", other.media_type()),
                ));
            }
        };

        if !state.resolved {
            // The catalog has not arrived (or answered a different filter) yet.
            // Only reachable when a producer published before draining our hint.
            state.dropped_early += 1;
            return Ok(Tick::Again);
        }

        let Some((_, edge)) = state.map.iter().find(|(routed, _)| *routed == index) else {
            // Upstream discarding is the mechanism; this catches the few packets
            // already in flight when the selection hint landed.
            state.dropped_unrouted += 1;
            return Ok(Tick::Again);
        };
        let edge = edge.clone();

        if state.waiting_for_keyframe {
            // C++ quirk kept deliberately: one shared flag, cleared by the first
            // keyframe on *any* routed video stream, and until then every routed
            // packet is dropped — audio included.
            let is_key = matches!(&buffer, Media::Packet(packet) if packet.is_key());
            if is_key && state.video.contains(&index) {
                log::debug!("{}: keyframe on stream {index}, forwarding now", self.name);
                state.waiting_for_keyframe = false;
            } else {
                return Ok(Tick::Again);
            }
        }

        match self.emit(state, ctx, edge, buffer) {
            Emitted::Ok => Ok(Tick::Again),
            Emitted::Parked => Ok(Tick::Idle),
            Emitted::Closed => Ok(Tick::Done),
        }
    }

    fn emit(
        &self,
        state: &mut State,
        ctx: &mut NodePollContext,
        edge: Arc<dyn Edge>,
        buffer: Media,
    ) -> Emitted {
        match edge.offer(buffer) {
            Ok(()) => Emitted::Ok,
            Err((Push::Full, buffer)) => {
                state.pending = Some((edge.clone(), buffer));
                ctx.wait_writable(edge);
                Emitted::Parked
            }
            Err((Push::Closed, _)) => {
                log::info!("{}: an output edge closed, finishing", self.name);
                Emitted::Closed
            }
            // The edge took it and discarded it: nothing left to retry with.
            Err((Push::Dropped | Push::Accepted, _)) => Emitted::Ok,
        }
    }

    /// Resolves `routing` against a catalog whose `filter` answers ours, then
    /// tells the producer what to keep and the consumers what they will get.
    fn on_catalog(
        &self,
        state: &mut State,
        input: &Arc<dyn Edge>,
        spec: Spec,
    ) -> Result<(), NodeError> {
        let (filter, streams) = match spec {
            Spec::Catalog { filter, streams } => (filter, streams),
            other => {
                log::warn!(
                    "{}: ignoring a {:?} spec on the input; expected a stream catalog",
                    self.name,
                    other.media()
                );
                return Ok(());
            }
        };
        if filter.as_deref() != self.params.streams_filter.as_deref() {
            // Answers a different question — the producer had not drained our
            // hint yet. Ignore it and wait; the next catalog will match.
            log::debug!(
                "{}: catalog is for streams_filter {filter:?}, waiting for {:?}",
                self.name,
                self.params.streams_filter
            );
            return Ok(());
        }

        let resolved = self.resolve_routes(&streams)?;
        let indices = resolved.iter().map(|(index, _)| *index).collect::<Vec<_>>();
        if state.resolved && indices == state.indices {
            log::debug!(
                "{}: catalog re-delivered unchanged, keeping routes",
                self.name
            );
            return Ok(());
        }

        let mut map = Vec::with_capacity(resolved.len());
        let mut video = Vec::new();
        for (index, stream) in &resolved {
            let edge = self.output(&stream.pad)?;
            if stream.codec_type == MEDIA_TYPE_VIDEO {
                video.push(*index);
            }
            // What this output will carry, so a decoder downstream can open
            // without ever seeing the container.
            edge.push_event(EdgeEvent::Spec(Spec::Packet(stream.spec.clone())));
            map.push((*index, edge));
        }

        log::info!(
            "{}: routing streams {:?} to {:?}",
            self.name,
            indices,
            resolved.iter().map(|(_, s)| &s.pad).collect::<Vec<_>>()
        );
        // Everything else is demuxed for nothing, so ask the producer to discard
        // it inside libavformat instead.
        input.post_hint(EdgeHint::Streams(StreamSelection::new(indices.clone())));

        state.map = map;
        state.indices = indices;
        state.video = video;
        state.resolved = true;
        Ok(())
    }

    /// One entry per route that resolved, as `(stream index, the stream)`.
    fn resolve_routes(&self, streams: &[CatalogStream]) -> Result<Vec<(i32, Resolved)>, NodeError> {
        let mut out: Vec<(i32, Resolved)> = Vec::with_capacity(self.routes.len());
        for route in &self.routes {
            let Some(index) = routing::resolve(&route.key, streams) else {
                if route.key.optional {
                    log::info!(
                        "{}: no optional stream `{}`, leaving `{}` unbound",
                        self.name,
                        route.pad,
                        route.edge_name
                    );
                    continue;
                }
                return Err(NodeError::new(
                    &self.name,
                    NodePhase::Spec,
                    format!("no stream `{}` in the input", route.pad),
                ));
            };
            if let Some((_, first)) = out.iter().find(|(routed, _)| *routed == index) {
                // C++ `addStream` throws on the same thing.
                return Err(NodeError::new(
                    &self.name,
                    NodePhase::Spec,
                    format!(
                        "routing keys `{}` and `{}` both resolve to stream {index}",
                        first.pad, route.pad
                    ),
                ));
            }
            let stream = streams
                .iter()
                .find(|stream| stream.index == index)
                .expect("resolve returned a catalog stream");
            out.push((
                index,
                Resolved {
                    pad: route.pad.clone(),
                    codec_type: stream.codec_type,
                    spec: stream.spec.clone(),
                },
            ));
        }
        Ok(out)
    }

    fn output(&self, pad: &str) -> Result<Arc<dyn Edge>, NodeError> {
        self.outs.lock().unwrap().get(pad).cloned().ok_or_else(|| {
            NodeError::new(
                &self.name,
                NodePhase::Spec,
                format!("output pad `{pad}` is not bound"),
            )
        })
    }

    fn log_drops(&self, state: &State) {
        if state.dropped_early > 0 {
            log::info!(
                "{}: dropped {} packet(s) that arrived before the catalog",
                self.name,
                state.dropped_early
            );
        }
        if state.dropped_unrouted > 0 {
            log::info!(
                "{}: dropped {} packet(s) of unrouted streams",
                self.name,
                state.dropped_unrouted
            );
        }
    }
}

/// One route that found its stream.
struct Resolved {
    pad: String,
    codec_type: i32,
    spec: avplumber_f7k::graph::spec::PacketSpec,
}
