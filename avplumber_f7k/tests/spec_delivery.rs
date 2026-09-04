//! `Spec` is delivered exactly once, and `peek` + `pop` act on the same
//! storage. Both edge kinds keep two of them — latched `Spec` vs queue, and on
//! `DirectEdge` queue vs `inflight` — so every accessor has to agree on which
//! one wins. Default (non-ffmpeg) build: `Media::Stub` only.

#![cfg(not(feature = "ffmpeg"))]

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex, OnceLock};

use avplumber_f7k::{
    AvpMediaType, AvpRational, BufferedEdge, DirectEdge, Edge, EdgeEvent, EdgeItem, Media, Node,
    NodeKind, NodePollContext, Push, SisoNode, SisoPollAdapter, Spec, Tick, Wakeup,
};

fn stub(pts: i64) -> Media {
    Media::Stub {
        kind: AvpMediaType::VIDEO,
        pts,
    }
}

fn video_spec(width: i32) -> Spec {
    Spec::Video {
        width,
        height: 8,
        pix_fmt: 0,
        frame_rate: AvpRational { num: 25, den: 1 },
        sar: AvpRational { num: 1, den: 1 },
        time_base: AvpRational { num: 1, den: 1000 },
    }
}

#[derive(Default, Debug)]
struct Drained {
    specs: usize,
    buffers: Vec<i64>,
    other_events: usize,
}

fn drain(edge: &dyn Edge) -> Drained {
    let mut out = Drained::default();
    while let Some(item) = edge.try_take() {
        match item {
            EdgeItem::Buffer(buf) => out.buffers.push(buf.ts().val),
            EdgeItem::Event(EdgeEvent::Spec(_)) => out.specs += 1,
            EdgeItem::Event(_) => out.other_events += 1,
        }
    }
    out
}

#[test]
fn a_pushed_spec_is_delivered_exactly_once() {
    let edge = BufferedEdge::new(4);
    edge.push_event(EdgeEvent::Spec(video_spec(16)));
    assert_eq!(edge.push(stub(1)), Push::Accepted);

    let first = drain(&edge);
    assert_eq!(first.specs, 1, "one push_event(Spec) is one delivery");
    assert_eq!(first.buffers, vec![1]);

    edge.rearm_spec();
    let second = drain(&edge);
    assert_eq!(second.specs, 1, "rearm re-delivers the latch, once");
    assert!(second.buffers.is_empty());

    let third = drain(&edge);
    assert_eq!(third.specs, 0, "a delivered latch stays delivered");
}

struct Forward;

impl SisoNode for Forward {
    type Inner = ();

    fn name(&self) -> &str {
        "forward"
    }
    fn on_spec(&self, spec: &Spec) -> Result<((), Spec), String> {
        Ok(((), spec.clone()))
    }
    fn process(&self, _inner: &mut (), buf: Media) -> Result<Option<Media>, String> {
        Ok(Some(buf))
    }
}

/// Poll the node the way the executors do, until it parks or finishes.
fn pump(node: &dyn Node) {
    let cancel = Arc::new(AtomicBool::new(false));
    let tick = Arc::new(Wakeup::new());
    for _ in 0..64 {
        let mut ctx = NodePollContext::new(cancel.clone(), tick.clone());
        match node.poll(&mut ctx) {
            Tick::Again => continue,
            Tick::Idle | Tick::Done => break,
        }
    }
}

#[test]
fn a_spec_reaches_the_tail_of_a_poll_chain_exactly_once() {
    let edges: Vec<Arc<dyn Edge>> = (0..4)
        .map(|_| Arc::new(BufferedEdge::new(8)) as Arc<dyn Edge>)
        .collect();
    let nodes: Vec<Arc<dyn Node>> = (0..3)
        .map(|i| {
            let node = Arc::new(SisoPollAdapter::new(Forward));
            node.bind_source("in", edges[i].clone());
            node.bind_sink("out", edges[i + 1].clone());
            node as Arc<dyn Node>
        })
        .collect();

    edges[0].push_event(EdgeEvent::Spec(video_spec(16)));
    assert_eq!(edges[0].push(stub(1)), Push::Accepted);
    assert_eq!(edges[0].push(stub(2)), Push::Accepted);
    for _ in 0..4 {
        for node in &nodes {
            pump(node.as_ref());
        }
    }

    let tail = drain(edges[3].as_ref());
    assert_eq!(
        tail.specs, 1,
        "three forwarding nodes must not multiply the Spec"
    );
    assert_eq!(tail.buffers, vec![1, 2]);
}

#[test]
fn buffered_peek_then_pop_consumes_the_latch_not_a_queued_buffer() {
    let edge = BufferedEdge::new(4);
    edge.push_event(EdgeEvent::Spec(video_spec(16)));
    assert!(matches!(
        edge.take(0),
        Some(EdgeItem::Event(EdgeEvent::Spec(_)))
    ));
    assert_eq!(edge.push(stub(1)), Push::Accepted);
    assert_eq!(edge.push(stub(2)), Push::Accepted);
    // Reconstruction and bind both rearm the latch with buffers still queued.
    edge.rearm_spec();

    assert!(matches!(
        edge.peek_clone(0),
        Some(EdgeItem::Event(EdgeEvent::Spec(_)))
    ));
    edge.pop();

    let rest = drain(&edge);
    assert_eq!(
        rest.buffers,
        vec![1, 2],
        "pop must remove the peeked latch delivery, not a buffer behind it"
    );
    assert_eq!(rest.specs, 0, "the popped delivery must not come back");
}

/// The C ABI idiom (`avp_edge_peek` + `avp_edge_peek_consume`): look at the
/// head, then remove what was looked at.
struct PeekPopConsumer {
    input: OnceLock<Arc<dyn Edge>>,
    seen: Mutex<Vec<&'static str>>,
}

impl Node for PeekPopConsumer {
    fn name(&self) -> &str {
        "peek-pop"
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }
    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        let _ = self.input.set(edge);
    }
    fn poll(&self, _ctx: &mut NodePollContext) -> Tick {
        let Some(input) = self.input.get() else {
            return Tick::Idle;
        };
        let Some(item) = input.peek_clone(0) else {
            return Tick::Idle;
        };
        self.seen.lock().unwrap().push(match item {
            EdgeItem::Buffer(_) => "buffer",
            EdgeItem::Event(EdgeEvent::FlushStop) => "flush-stop",
            EdgeItem::Event(_) => "event",
        });
        input.pop();
        Tick::Again
    }
}

#[test]
fn direct_peek_then_pop_keeps_the_inflight_buffer_behind_a_queued_event() {
    let direct = Arc::new(DirectEdge::new());
    let logical: Arc<dyn Edge> = direct.clone();
    let consumer = Arc::new(PeekPopConsumer {
        input: OnceLock::new(),
        seen: Mutex::new(Vec::new()),
    });
    consumer.bind_source("in", logical.clone());
    direct.set_consumer(consumer.clone());

    // The ordinary first offer after an event: event queued, buffer inflight.
    logical.push_event(EdgeEvent::FlushStop);
    assert_eq!(logical.push(stub(7)), Push::Accepted);

    assert_eq!(
        consumer.seen.lock().unwrap().as_slice(),
        ["flush-stop", "buffer"],
        "pop must not consume the inflight buffer while an event is ahead of it"
    );
    assert_eq!(
        logical.occupied(),
        0,
        "an offer reported Accepted must leave nothing behind"
    );
}
