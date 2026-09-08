//! [`PollNode`], what a cooperative node writes, and [`Polling`], the [`Node`]
//! it runs as.

use std::ops::Deref;
use std::sync::Arc;

use crate::graph::edge::Edge;
use crate::graph::error::NodeError;
use crate::graph::node::{Node, NodeKind, Tick};
use crate::graph::pad::NodePads;
use crate::graph::poll_ctx::NodePollContext;
use crate::scaffold::io::Io;

/// A node that never blocks: it shares an event loop
/// ([`AsyncExecutor`](crate::exec::AsyncExecutor)) with every other node of its
/// clock domain, registers on the [`NodePollContext`] what it is waiting for,
/// and returns [`Tick::Idle`]. The C++ non-blocking node shape.
///
/// Keep an [`Io`] built with [`NodePhase::Poll`](crate::graph::error::NodePhase::Poll),
/// hand it out through [`Self::io`], implement [`Self::step`]. Wrap the node in
/// [`Polling`] to get the `Node`.
pub trait PollNode: Send + Sync + 'static {
    fn io(&self) -> &Io;

    /// For the media-type check at connect. Declaring none skips the check.
    fn pads(&self) -> NodePads {
        NodePads::default()
    }

    /// One cooperative step. `Idle` after registering waiters on `ctx`, `Again`
    /// to be called right back, `Done` to finish. `Err` fails the node like a
    /// blocking one — except on the fused Direct path, which is why
    /// [`Self::direct_consumer_is_infallible`] exists.
    fn step(&self, ctx: &mut NodePollContext) -> Result<Tick, NodeError>;

    /// The opt-in to consuming from a [`DirectEdge`](crate::graph::DirectEdge),
    /// whose producer runs this node's `step` from inside its own `offer`. That
    /// path has no executor to report an error to, so `true` is a promise that
    /// `step` never returns `Err`. Connecting a Direct edge to a node that does
    /// not opt in is rejected.
    fn direct_consumer_is_infallible(&self) -> bool {
        false
    }

    /// Where an input edge lands. The default is the one slot; a node with
    /// several input pads overrides this and keeps them itself.
    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.io().input_slot.bind(edge);
    }

    /// Same, for an output edge.
    fn bind_sink(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.io().output_slot.bind(edge);
    }

    /// Before the first step of a run, on the event loop.
    fn start(&self) {}

    /// After the last step, on the event loop, however the body ended.
    fn stop(&self) {}

    /// [`Node::interrupt`]: any thread, any time, must not block. A poll body
    /// is never parked anywhere the executor cannot reach, so most nodes need
    /// nothing here.
    fn interrupt(&self) {}
}

/// The [`Node`] a [`PollNode`] runs as. See [`Blocking`](crate::scaffold::Blocking)
/// for why this is a newtype.
pub struct Polling<N>(pub N);

impl<N: PollNode> Node for Polling<N> {
    fn name(&self) -> &str {
        &self.0.io().name
    }

    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }

    fn direct_consumer_is_infallible(&self) -> bool {
        self.0.direct_consumer_is_infallible()
    }

    fn pads(&self) -> NodePads {
        self.0.pads()
    }

    fn bind_source(&self, pad: &str, edge: Arc<dyn Edge>) {
        self.0.bind_source(pad, edge);
    }

    fn bind_sink(&self, pad: &str, edge: Arc<dyn Edge>) {
        self.0.bind_sink(pad, edge);
    }

    fn start(&self) {
        self.0.start();
    }

    fn stop(&self) {
        self.0.stop();
    }

    fn interrupt(&self) {
        self.0.interrupt();
    }

    fn poll(&self, ctx: &mut NodePollContext) -> Result<Tick, NodeError> {
        self.0.step(ctx)
    }
}

impl<N> Deref for Polling<N> {
    type Target = N;

    fn deref(&self) -> &N {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;

    use super::*;
    use crate::graph::BufferedEdge;
    use crate::graph::buffer::AvpMediaType;
    use crate::graph::edge::{EdgeItem, Wakeup};
    use crate::graph::error::NodePhase;

    /// Forwards buffers; fails on an event, unless it promised not to.
    struct Fwd {
        io: Io,
        infallible: bool,
    }

    impl PollNode for Fwd {
        fn io(&self) -> &Io {
            &self.io
        }

        fn pads(&self) -> NodePads {
            NodePads::siso(AvpMediaType::VIDEO, AvpMediaType::VIDEO)
        }

        fn direct_consumer_is_infallible(&self) -> bool {
            self.infallible
        }

        fn step(&self, ctx: &mut NodePollContext) -> Result<Tick, NodeError> {
            let input = self.io.input()?;
            match input.try_take() {
                None => {
                    ctx.wait_readable(input);
                    Ok(Tick::Idle)
                }
                Some(EdgeItem::Buffer(buffer)) => {
                    let _ = self.io.output()?.offer(buffer);
                    Ok(Tick::Again)
                }
                Some(EdgeItem::Event(_)) => Err(self.io.error(NodePhase::Poll, "an event")),
            }
        }
    }

    fn fwd(infallible: bool) -> Polling<Fwd> {
        Polling(Fwd {
            io: Io::new("f", NodePhase::Poll),
            infallible,
        })
    }

    fn ctx() -> NodePollContext {
        NodePollContext::new(Arc::new(AtomicBool::new(false)), Arc::new(Wakeup::new()))
    }

    #[test]
    fn wrapper_supplies_the_node_contract_from_io() {
        let node = fwd(true);
        assert_eq!(node.name(), "f");
        assert_eq!(node.kind(), NodeKind::Poll);
        assert!(node.direct_consumer_is_infallible());
        assert!(!fwd(false).direct_consumer_is_infallible());
        assert_eq!(
            node.pads(),
            NodePads::siso(AvpMediaType::VIDEO, AvpMediaType::VIDEO)
        );

        let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(4));
        let output: Arc<dyn Edge> = Arc::new(BufferedEdge::new(4));
        node.bind_source("in", input.clone());
        node.bind_sink("out", output.clone());
        let mut ctx = ctx();
        assert_eq!(node.poll(&mut ctx).unwrap(), Tick::Idle);
        assert!(
            input
                .offer(crate::graph::media::test_media(AvpMediaType::VIDEO, 1))
                .is_ok()
        );
        assert_eq!(node.poll(&mut ctx).unwrap(), Tick::Again);
        assert_eq!(output.occupied(), 1);
    }

    #[test]
    fn step_errors_come_out_of_poll() {
        let node = fwd(false);
        let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(4));
        node.bind_source("in", input.clone());
        input.push_event(crate::graph::edge::EdgeEvent::Eof);
        let err = node.poll(&mut ctx()).unwrap_err();
        assert_eq!((err.node.as_str(), err.phase), ("f", NodePhase::Poll));
        assert_eq!(err.message, "an event");
    }

    #[test]
    fn unbound_input_fails_the_step_in_the_poll_phase() {
        let err = fwd(false).poll(&mut ctx()).unwrap_err();
        assert_eq!(err.phase, NodePhase::Poll);
        assert!(err.message.contains("input"), "{}", err.message);
    }
}
