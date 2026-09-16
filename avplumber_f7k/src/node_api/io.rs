//! The name and the two rebindable edge slots a node keeps.

use std::sync::{Arc, Mutex};

use crate::graph::edge::{Edge, EdgeEvent, EdgeItem};
use crate::graph::error::{NodeError, NodePhase};

/// One bound edge, for the nodes with a single input or a single output.
///
/// A `Mutex<Option<_>>` rather than a `OnceLock`: `bind_source`/`bind_sink` run
/// again whenever a script rebinds a pad or a reconstruction re-establishes the
/// links, and the last binding is the live one.
#[derive(Default)]
pub struct EdgeSlot {
    edge: Mutex<Option<Arc<dyn Edge>>>,
}

impl EdgeSlot {
    pub fn bind(&self, edge: Arc<dyn Edge>) {
        *self.edge.lock().unwrap() = Some(edge);
    }

    pub fn get(&self) -> Option<Arc<dyn Edge>> {
        self.edge.lock().unwrap().clone()
    }

    /// The bound edge, or the error every media node would otherwise spell out
    /// itself. An unbound pad is a script mistake, so it fails the group.
    pub fn require(
        &self,
        node: &str,
        phase: NodePhase,
        what: &str,
    ) -> Result<Arc<dyn Edge>, NodeError> {
        self.get()
            .ok_or_else(|| NodeError::new(node, phase, format!("{what} edge is not bound")))
    }
}

/// What a node keeps besides its own state: its name, and one rebindable slot
/// per direction.
///
/// [`Blocking`](crate::node_api::Blocking) and [`Polling`](crate::node_api::Polling)
/// bind the slots and answer [`Node::name`](crate::graph::node::Node::name) from
/// here, so the node itself implements neither. A node with several pads on one
/// side keeps those itself, overrides the `bind_*` hook of its node trait, and
/// leaves that slot empty.
pub struct Io {
    pub name: String,
    pub input_slot: EdgeSlot,
    pub output_slot: EdgeSlot,
    /// Which phase an unbound edge is reported in: the one the body runs in.
    phase: NodePhase,
}

impl Io {
    /// `phase` is the body's: [`NodePhase::Poll`] for a
    /// [`PollNode`](crate::node_api::PollNode). [`BlockingIo`](crate::node_api::BlockingIo)
    /// fills it in for a blocking one.
    pub fn new(name: &str, phase: NodePhase) -> Self {
        Self {
            name: name.into(),
            input_slot: EdgeSlot::default(),
            output_slot: EdgeSlot::default(),
            phase,
        }
    }

    /// The bound input edge, or the error that fails the group: an unbound pad
    /// is a script mistake, not something to wait out.
    pub fn input(&self) -> Result<Arc<dyn Edge>, NodeError> {
        self.input_slot.require(&self.name, self.phase, "input")
    }

    /// The bound output edge, like [`Self::input`].
    pub fn output(&self) -> Result<Arc<dyn Edge>, NodeError> {
        self.output_slot.require(&self.name, self.phase, "output")
    }

    pub fn error(&self, phase: NodePhase, message: impl Into<String>) -> NodeError {
        NodeError::new(&self.name, phase, message)
    }
}

/// Whether the next item on `input` is a `FlushStart`: what a node holding a
/// buffer it produced from that input checks before waiting any longer.
pub fn flush_at_head(input: &Arc<dyn Edge>) -> bool {
    matches!(
        input.peek_clone(0),
        Some(EdgeItem::Event(EdgeEvent::FlushStart))
    )
}
