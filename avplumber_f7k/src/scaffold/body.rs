//! The fallible step a node writes, and the infallible
//! [`NodeBody`] it becomes.

use std::sync::Arc;

use crate::graph::error::NodeError;
use crate::graph::node::{Blocked, Node, NodeBody, Tick};
use crate::graph::poll_ctx::NodePollContext;

/// The real body of a blocking node.
///
/// [`Node::process`] returns `Blocked`, so it cannot report a failure — the
/// executor would see a clean finish and the group would never restart. These
/// nodes implement `step` and hand the executor a fallible body through
/// [`blocking_body`].
pub trait BlockingStep: Node {
    fn step(&self) -> Result<Blocked, NodeError>;
}

pub fn blocking_body<N: BlockingStep + 'static>(node: Arc<N>) -> NodeBody {
    NodeBody::Blocking(Box::new(move || node.step()))
}

/// Same for the cooperative nodes and [`Node::poll`].
pub trait PollStep: Node {
    fn step(&self, ctx: &mut NodePollContext) -> Result<Tick, NodeError>;
}

pub fn poll_body<N: PollStep + 'static>(node: Arc<N>) -> NodeBody {
    NodeBody::Poll(Box::new(move |ctx| node.step(ctx)))
}
