//! `null_sink` — drains one input edge and counts what came through.
//!
//! Port of C++ `src/nodes/null_sink.cpp`, plus the counters the Rust tests need
//! as a terminal assertion point (C++ tests read the sink's edge instead).

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::edge::{Edge, EdgeEvent, EdgeItem};
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::node::{Blocked, Node, NodeBody, NodeKind};
use avplumber_f7k::graph::spec::Spec;
use avplumber_f7k::scaffold::{BlockingStep, EdgeSlot, blocking_body};

/// What one `null_sink` instance has seen, cumulative across restarts.
///
/// Kept in a process-wide registry keyed by node name (below) so a test can
/// read it without holding the node: a group restart replaces the `Node`, and
/// the whole point of this sink is to survive that and keep counting.
#[derive(Default)]
pub struct Counters {
    pub buffers: AtomicU64,
    pub specs: AtomicU64,
    pub eof: AtomicBool,
    pub last_spec: Mutex<Option<Spec>>,
}

impl Counters {
    pub fn buffers(&self) -> u64 {
        self.buffers.load(Ordering::Relaxed)
    }

    pub fn specs(&self) -> u64 {
        self.specs.load(Ordering::Relaxed)
    }

    pub fn saw_eof(&self) -> bool {
        self.eof.load(Ordering::Acquire)
    }
}

fn registry() -> &'static Mutex<HashMap<String, Arc<Counters>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<Counters>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The counters of the `null_sink` named `name`, creating them if that node has
/// not been built yet — so a test can grab the handle before `group.start`.
pub fn counters(name: &str) -> Arc<Counters> {
    registry()
        .lock()
        .unwrap()
        .entry(name.to_string())
        .or_default()
        .clone()
}

/// Zeroes the counters of `name`, for a test that reuses a node name.
pub fn reset(name: &str) {
    let counters = counters(name);
    counters.buffers.store(0, Ordering::Relaxed);
    counters.specs.store(0, Ordering::Relaxed);
    counters.eof.store(false, Ordering::Release);
    *counters.last_spec.lock().unwrap() = None;
}

/// No parameters, like C++ (`src` is the envelope's, not the node's).
#[derive(Debug, serde::Deserialize)]
pub struct NullSinkSpec {}

impl NodeSpec for NullSinkSpec {
    const TYPE_NAME: &'static str = "null_sink";
    type Node = NullSink;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        Ok(NullSink {
            name: name.into(),
            input: EdgeSlot::default(),
            counters: counters(name),
        })
    }
}

pub struct NullSink {
    name: String,
    input: EdgeSlot,
    counters: Arc<Counters>,
}

impl Node for NullSink {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> NodeKind {
        NodeKind::Blocking
    }

    // Deliberately no `pads()`: C++ `DECLNODE_ATD(null_sink, NullSink)` accepts
    // every media type, and a vertex that declares no pads makes
    // `core::pad_media` skip the media-type check entirely.

    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.input.bind(edge);
    }

    fn take_body(self: Arc<Self>) -> NodeBody {
        blocking_body(self)
    }
}

impl BlockingStep for NullSink {
    fn step(&self) -> Result<Blocked, NodeError> {
        let edge = self
            .input
            .require(&self.name, NodePhase::Process, "input")?;
        match edge.take(-1) {
            Some(EdgeItem::Buffer(_)) => {
                self.counters.buffers.fetch_add(1, Ordering::Relaxed);
                Ok(Blocked::Again)
            }
            Some(EdgeItem::Event(EdgeEvent::Spec(spec))) => {
                self.counters.specs.fetch_add(1, Ordering::Relaxed);
                *self.counters.last_spec.lock().unwrap() = Some(spec);
                Ok(Blocked::Again)
            }
            Some(EdgeItem::Event(EdgeEvent::Eof)) => {
                self.counters.eof.store(true, Ordering::Release);
                Ok(Blocked::Done)
            }
            // Nothing is buffered here, so a flush is a no-op.
            Some(EdgeItem::Event(EdgeEvent::FlushStart | EdgeEvent::FlushStop)) => {
                Ok(Blocked::Again)
            }
            // A blocking `take` came back empty: the edge is closed, or the
            // executor interrupted it to stop this node.
            None => Ok(Blocked::Done),
        }
    }
}
