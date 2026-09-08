//! [`BlockingNode`], what a node with its own thread writes, and [`Blocking`],
//! the [`Node`] it runs as.

use std::ops::Deref;
use std::sync::Arc;

use crate::graph::edge::Edge;
use crate::graph::error::NodeError;
use crate::graph::node::{Blocked, Node, NodeKind};
use crate::graph::pad::NodePads;
use crate::scaffold::io::BlockingIo;

/// A node whose body blocks: on its input's `take(-1)`, inside libav, or on its
/// [`BlockingIo`] park. It gets its own OS thread
/// ([`BlockingExecutor`](crate::exec::BlockingExecutor)), which is the right
/// place for every libav call — none of them has an asynchronous form.
///
/// The C++ `process()` shape: keep a [`BlockingIo`], hand it out through
/// [`Self::io`], implement [`Self::step`]. Everything else has a default. Wrap
/// the node in [`Blocking`] to get the `Node`.
pub trait BlockingNode: Send + Sync + 'static {
    fn io(&self) -> &BlockingIo;

    /// For the media-type check at connect. Declaring none skips the check,
    /// which a sink that accepts every media type wants.
    fn pads(&self) -> NodePads {
        NodePads::default()
    }

    /// One step: block for input, push what came of it. `Done` finishes the
    /// node; `Err` fails it, and the supervisor restarts the group per its
    /// policy.
    fn step(&self) -> Result<Blocked, NodeError>;

    /// Where an input edge lands. The default is the one slot; a node with
    /// several input pads overrides this and keeps them itself.
    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.io().input_slot.bind(edge);
    }

    /// Same, for an output edge.
    fn bind_sink(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.io().output_slot.bind(edge);
    }

    /// Before the first step of a run, on the body's thread. The park is
    /// already reset.
    fn start(&self) {}

    /// After the last step, on the body's thread, however the body ended.
    fn stop(&self) {}

    /// [`Node::interrupt`] beyond waking the park, which [`Blocking`] does
    /// itself: a node blocked *inside* libav has to be told through its own
    /// callback. Any thread, any time, must not block.
    fn interrupt(&self) {}
}

/// The [`Node`] a [`BlockingNode`] runs as.
///
/// A newtype rather than a blanket impl so that [`Polling`](crate::scaffold::Polling)
/// can exist beside it: two blanket impls of `Node` would overlap.
pub struct Blocking<N>(pub N);

impl<N: BlockingNode> Node for Blocking<N> {
    fn name(&self) -> &str {
        &self.0.io().name
    }

    fn kind(&self) -> NodeKind {
        NodeKind::Blocking
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
        self.0.io().park.reset();
        self.0.start();
    }

    fn stop(&self) {
        self.0.stop();
    }

    fn interrupt(&self) {
        self.0.io().park.interrupt();
        self.0.interrupt();
    }

    fn process(&self) -> Result<Blocked, NodeError> {
        self.0.step()
    }
}

impl<N> Deref for Blocking<N> {
    type Target = N;

    fn deref(&self) -> &N {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;
    use crate::graph::BufferedEdge;
    use crate::graph::buffer::AvpMediaType;
    use crate::graph::edge::Push;
    use crate::graph::error::NodePhase;
    use crate::graph::media::test_media;

    /// Fails on the third step, and counts the hooks.
    struct Counting {
        io: BlockingIo,
        steps: AtomicUsize,
        hooks: Mutex<Vec<&'static str>>,
    }

    impl BlockingNode for Counting {
        fn io(&self) -> &BlockingIo {
            &self.io
        }

        fn pads(&self) -> NodePads {
            NodePads::output(AvpMediaType::VIDEO)
        }

        fn step(&self) -> Result<Blocked, NodeError> {
            let step = self.steps.fetch_add(1, Ordering::SeqCst);
            if step == 2 {
                return Err(self.io.error(NodePhase::Process, "third step"));
            }
            self.io.push(
                &self.io.output()?,
                test_media(AvpMediaType::VIDEO, step as i64),
            )
        }

        fn start(&self) {
            self.hooks.lock().unwrap().push("start");
        }

        fn stop(&self) {
            self.hooks.lock().unwrap().push("stop");
        }

        fn interrupt(&self) {
            self.hooks.lock().unwrap().push("interrupt");
        }
    }

    fn node() -> Blocking<Counting> {
        Blocking(Counting {
            io: BlockingIo::new("c"),
            steps: AtomicUsize::new(0),
            hooks: Mutex::new(Vec::new()),
        })
    }

    #[test]
    fn wrapper_supplies_the_node_contract_from_io() {
        let node = node();
        assert_eq!(node.name(), "c");
        assert_eq!(node.kind(), NodeKind::Blocking);
        assert_eq!(node.pads(), NodePads::output(AvpMediaType::VIDEO));

        let out: Arc<dyn Edge> = Arc::new(BufferedEdge::new(4));
        node.bind_sink("whatever", out.clone());
        node.start();
        assert_eq!(node.process().unwrap(), Blocked::Again);
        assert_eq!(node.process().unwrap(), Blocked::Again);
        let err = node.process().unwrap_err();
        assert_eq!((err.node.as_str(), err.phase), ("c", NodePhase::Process));
        assert_eq!(err.message, "third step");
        node.stop();
        assert_eq!(out.occupied(), 2);
        assert_eq!(*node.hooks.lock().unwrap(), vec!["start", "stop"]);
    }

    #[test]
    fn unbound_output_fails_the_step() {
        let node = node();
        let err = node.process().unwrap_err();
        assert_eq!(err.phase, NodePhase::Process);
        assert!(err.message.contains("output"), "{}", err.message);
    }

    #[test]
    fn interrupt_reaches_the_park_and_the_node() {
        let node = node();
        let out: Arc<dyn Edge> = Arc::new(BufferedEdge::new(1));
        node.bind_sink("out", out.clone());
        node.start();
        // Fill the output so the next push would park; an interrupt that landed
        // first turns that push into `Done` instead of a wait.
        assert_eq!(
            out.push(test_media(AvpMediaType::VIDEO, 100)),
            Push::Accepted
        );
        node.interrupt();
        assert_eq!(node.process().unwrap(), Blocked::Done);
        assert_eq!(*node.hooks.lock().unwrap(), vec!["start", "interrupt"]);
        // `start` clears the interrupt for the next run.
        node.start();
        assert!(!node.io.is_interrupted());
    }
}
