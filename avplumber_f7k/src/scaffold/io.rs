//! The name and the edges a node built on the [`scaffold`](crate::scaffold)
//! wrappers keeps, and, for a blocking body, the park it waits on.

use std::ops::Deref;
use std::sync::Arc;

use crate::graph::edge::Edge;
use crate::graph::error::{NodeError, NodePhase};
use crate::graph::media::Media;
use crate::graph::node::Blocked;
use crate::scaffold::edge_slot::EdgeSlot;
use crate::scaffold::park::{Park, Pushed, push_blocking};

/// What a node keeps besides its own state: its name, and one rebindable slot
/// per direction.
///
/// [`Blocking`](crate::scaffold::Blocking) and [`Polling`](crate::scaffold::Polling)
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
    /// [`PollNode`](crate::scaffold::PollNode). [`BlockingIo::new`] fills it in
    /// for a blocking one.
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

/// [`Io`] plus the [`Park`] a blocking body waits on: for room on its output,
/// and wherever else it has to sleep interruptibly.
pub struct BlockingIo {
    io: Io,
    pub(crate) park: Arc<Park>,
}

impl BlockingIo {
    pub fn new(name: &str) -> Self {
        Self {
            io: Io::new(name, NodePhase::Process),
            park: Arc::new(Park::default()),
        }
    }

    /// Pushes `buffer`, parking for room instead of dropping it. `Blocked::Done`
    /// means the node was interrupted while parked, or the edge is closed;
    /// either way it is finished.
    pub fn push(&self, edge: &Arc<dyn Edge>, buffer: Media) -> Result<Blocked, NodeError> {
        self.push_with(edge, buffer, || Ok(()))
    }

    /// [`Self::push`] with the anti-deadlock hook of [`push_blocking`]:
    /// `while_parked` runs on every wake, which is where a producer answers the
    /// hints its parked consumer posted.
    pub fn push_with(
        &self,
        edge: &Arc<dyn Edge>,
        buffer: Media,
        while_parked: impl FnMut() -> Result<(), NodeError>,
    ) -> Result<Blocked, NodeError> {
        Ok(
            match push_blocking(&self.park, edge, buffer, while_parked)? {
                Pushed::Ok => Blocked::Again,
                Pushed::Interrupted => Blocked::Done,
                Pushed::Closed => {
                    log::info!("{}: output edge closed, finishing", self.name);
                    Blocked::Done
                }
            },
        )
    }

    /// Set by [`Node::interrupt`](crate::graph::node::Node::interrupt); a body
    /// checks it before blocking on something the interrupt cannot reach.
    pub fn is_interrupted(&self) -> bool {
        self.park.is_interrupted()
    }

    /// Sleeps at most `timeout_ms`, less when interrupted or when the output
    /// edge drains: what a body does when libav wants time rather than data.
    pub fn wait(&self, timeout_ms: u64) {
        self.park.wait(timeout_ms)
    }
}

impl Deref for BlockingIo {
    type Target = Io;

    fn deref(&self) -> &Io {
        &self.io
    }
}
