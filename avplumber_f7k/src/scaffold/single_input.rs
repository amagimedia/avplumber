//! A blocking node with one input, written as reactions to what arrives on it
//! rather than as a `take` loop.
//!
//! Every consumer of an edge sees the same five things — a `Spec`, a buffer,
//! `FlushStart`, `FlushStop`, `Eof` — and most of what happens to them is the
//! same in every node: a flush is answered and forwarded, an `Eof` finishes
//! the node once it says it is done, a control event goes on to the output.
//! [`InputHandler`] is the part that differs, [`react`] is the part that does
//! not, and [`SingleInput`] is the blocking loop that feeds the two, so a node
//! writes neither the loop nor the classification.
//!
//! How a node keeps its state is its own business: every hook takes `&self`,
//! and the node locks a mutex, bumps an atomic or reads a field as it sees
//! fit. The framework asks only for `Send + Sync`, because the executor shares
//! the node with the control thread. Nothing is held across the wait for
//! input, so a node's state is free to read from elsewhere while the body
//! blocks.

use std::sync::Arc;

use crate::graph::edge::{Edge, EdgeEvent, EdgeItem};
use crate::graph::error::NodeError;
use crate::graph::media::{Media, Ts};
use crate::graph::node::Blocked;
use crate::graph::pad::NodePads;
use crate::graph::spec::Spec;
use crate::scaffold::blocking::BlockingNode;
use crate::scaffold::io::BlockingIo;

/// What a node does with each kind of item on its input.
pub trait InputHandler: Send + Sync + 'static {
    /// A (re-)delivered stream description. Reconfigure, and return the
    /// description of what this node will produce if it is known here — a
    /// transform's output format, an encoder's codec parameters — to have it
    /// published on the output before anything else. `None` when there is
    /// nothing to publish yet: a decoder learns its output format from the first
    /// decoded frame, a sink never publishes.
    fn on_spec(&self, spec: Spec) -> Result<Option<Spec>, NodeError>;

    /// One buffer. Whatever comes back is pushed to the output, parking for
    /// room. `None` for a node that consumes the buffer, or that produces later:
    /// a codec hands its output over in [`SingleInput::before_take`].
    fn on_buffer(&self, buffer: Media) -> Result<Option<Media>, NodeError>;

    /// The input is being flushed: discard whatever is buffered here. The
    /// `FlushStart` is forwarded once this returns. A node that buffers nothing
    /// needs nothing here.
    fn on_flush(&self) {}

    /// The flush is over; what follows is at the new position. `resume_at` is
    /// the position the source aimed for when its reposition could only land on
    /// a keyframe before it, so a node that turns buffered input into
    /// timestamped output — a decoder — drops what lies below it. The
    /// `FlushStop` is forwarded, payload intact, once this returns.
    fn on_flush_stop(&self, _resume_at: Option<Ts>) {}

    /// [`EdgeEvent::Drain`]: give up what the node is holding *inside a codec*,
    /// without ending anything. The event is forwarded once this returns, so a
    /// node that holds nothing of the kind can ignore it. Whatever the node
    /// produces goes out the usual way, from `before_take`.
    fn on_drain(&self) {}

    /// The input ended. `Done` finishes the node, after `Eof` is forwarded. A
    /// node that has to drain first — a codec — returns `Again`, keeps stepping
    /// from [`SingleInput::before_take`], and forwards `Eof` itself when the
    /// drain is over.
    fn on_eof(&self) -> Result<Blocked, NodeError> {
        Ok(Blocked::Done)
    }

    /// The input came back empty: the edge is closed, or a stop interrupted the
    /// wait. The node is finished; this is the place for a final log line.
    fn on_closed(&self) {}
}

/// What [`react`] made of one item.
pub enum Reaction {
    Again,
    Done,
    /// The handler produced a buffer; the caller pushes it the way its body
    /// pushes.
    Produced(Media),
}

/// Classifies one item and calls the matching hook, forwarding the control
/// events to `output` when there is one. The blocking loop of [`SingleInput`]
/// and the cooperative Siso adapters both end in this.
pub fn react<H: InputHandler + ?Sized>(
    handler: &H,
    output: Option<&Arc<dyn Edge>>,
    item: EdgeItem,
) -> Result<Reaction, NodeError> {
    let forward = |event: EdgeEvent| {
        if let Some(out) = output {
            out.push_event(event);
        }
    };
    match item {
        EdgeItem::Event(EdgeEvent::Spec(spec)) => {
            if let Some(published) = handler.on_spec(spec)? {
                forward(EdgeEvent::Spec(published));
            }
            Ok(Reaction::Again)
        }
        EdgeItem::Buffer(buffer) => Ok(match handler.on_buffer(buffer)? {
            Some(produced) => Reaction::Produced(produced),
            None => Reaction::Again,
        }),
        EdgeItem::Event(EdgeEvent::FlushStart) => {
            handler.on_flush();
            forward(EdgeEvent::FlushStart);
            Ok(Reaction::Again)
        }
        EdgeItem::Event(EdgeEvent::Drain) => {
            handler.on_drain();
            forward(EdgeEvent::Drain);
            Ok(Reaction::Again)
        }
        EdgeItem::Event(EdgeEvent::FlushStop { resume_at }) => {
            handler.on_flush_stop(resume_at);
            forward(EdgeEvent::FlushStop { resume_at });
            Ok(Reaction::Again)
        }
        EdgeItem::Event(EdgeEvent::Eof) => match handler.on_eof()? {
            Blocked::Again => Ok(Reaction::Again),
            Blocked::Done => {
                forward(EdgeEvent::Eof);
                Ok(Reaction::Done)
            }
        },
    }
}

/// A blocking node with one input: the reactions of [`InputHandler`], plus
/// what the loop that feeds them needs. It is a [`BlockingNode`] through the
/// blanket impl below, so it is registered as `Blocking<Self>` like any other.
///
/// The loop, once per step: return if interrupted; run [`Self::before_take`];
/// block on the input; [`react`]; push what was produced.
pub trait SingleInput: InputHandler {
    fn io(&self) -> &BlockingIo;

    /// For the media-type check at connect. Declaring none skips the check.
    fn pads(&self) -> NodePads {
        NodePads::default()
    }

    /// Runs before the input is read. `Some` is this step's result and skips
    /// the read: emit what an earlier step produced, drive a codec that is
    /// still holding an input, finish once a drain is over. The default reads
    /// straight away.
    fn before_take(&self) -> Result<Option<Blocked>, NodeError> {
        Ok(None)
    }

    /// Before the first step of a run, on the body's thread. The park is
    /// already reset.
    fn start(&self) {}

    /// After the last step, on the body's thread, however the body ended.
    fn stop(&self) {}

    /// [`BlockingNode::interrupt`]: any thread, any time, must not block — so
    /// it must not take a lock the body may be holding.
    fn interrupt(&self) {}
}

impl<N: SingleInput> BlockingNode for N {
    fn io(&self) -> &BlockingIo {
        SingleInput::io(self)
    }

    fn pads(&self) -> NodePads {
        SingleInput::pads(self)
    }

    fn step(&self) -> Result<Blocked, NodeError> {
        let io = SingleInput::io(self);
        if io.is_interrupted() {
            return Ok(Blocked::Done);
        }
        let input = io.input()?;
        if let Some(blocked) = self.before_take()? {
            return Ok(blocked);
        }
        let Some(item) = input.take(-1) else {
            self.on_closed();
            return Ok(Blocked::Done);
        };
        match react(self, io.output_slot.get().as_ref(), item)? {
            Reaction::Again => Ok(Blocked::Again),
            Reaction::Done => Ok(Blocked::Done),
            Reaction::Produced(buffer) => io.push_from(&input, &io.output()?, buffer),
        }
    }

    fn start(&self) {
        SingleInput::start(self);
    }

    fn stop(&self) {
        SingleInput::stop(self);
    }

    fn interrupt(&self) {
        SingleInput::interrupt(self);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicI64, AtomicUsize, Ordering};

    use super::*;
    use crate::graph::BufferedEdge;
    use crate::graph::buffer::{AvpMediaType, AvpRational};
    use crate::graph::error::NodePhase;
    use crate::graph::media::test_media;
    use crate::graph::node::Node;
    use crate::scaffold::blocking::Blocking;

    /// Records what arrives, echoes buffers, publishes a spec of its own, and
    /// drains for one step after `Eof`. Its state is a mutex and an atomic,
    /// which is the node's choice, not the scaffold's.
    struct Echo {
        io: BlockingIo,
        seen: Mutex<Vec<&'static str>>,
        drain_steps: AtomicUsize,
        /// The `resume_at` the last `FlushStop` carried, `i64::MIN` for none.
        resume_seen: AtomicI64,
    }

    impl Echo {
        fn saw(&self, what: &'static str) {
            self.seen.lock().unwrap().push(what);
        }
    }

    fn spec() -> Spec {
        Spec::Video {
            width: 2,
            height: 2,
            pix_fmt: 0,
            sw_pix_fmt: -1,
            frame_rate: AvpRational { num: 1, den: 1 },
            sar: AvpRational { num: 1, den: 1 },
            time_base: AvpRational { num: 1, den: 1000 },
        }
    }

    impl InputHandler for Echo {
        fn on_spec(&self, _spec: Spec) -> Result<Option<Spec>, NodeError> {
            self.saw("spec");
            Ok(Some(spec()))
        }

        fn on_buffer(&self, buffer: Media) -> Result<Option<Media>, NodeError> {
            self.saw("buffer");
            if buffer.ts().val < 0 {
                return Err(self.io.error(NodePhase::Process, "negative pts"));
            }
            Ok(Some(buffer))
        }

        fn on_flush(&self) {
            self.saw("flush");
        }

        fn on_flush_stop(&self, resume_at: Option<Ts>) {
            self.saw("flush-stop");
            self.resume_seen
                .store(resume_at.map_or(i64::MIN, |ts| ts.val), Ordering::SeqCst);
        }

        fn on_eof(&self) -> Result<Blocked, NodeError> {
            self.saw("eof");
            self.drain_steps.store(1, Ordering::SeqCst);
            Ok(Blocked::Again)
        }

        fn on_closed(&self) {
            self.saw("closed");
        }
    }

    impl SingleInput for Echo {
        fn io(&self) -> &BlockingIo {
            &self.io
        }

        fn before_take(&self) -> Result<Option<Blocked>, NodeError> {
            if self.drain_steps.swap(0, Ordering::SeqCst) == 1 {
                self.saw("drained");
                self.io.output()?.push_event(EdgeEvent::Eof);
                return Ok(Some(Blocked::Done));
            }
            Ok(None)
        }

        fn start(&self) {
            self.saw("start");
        }

        fn stop(&self) {
            self.saw("stop");
        }
    }

    fn echo() -> (Blocking<Echo>, Arc<dyn Edge>, Arc<dyn Edge>) {
        let node = Blocking(Echo {
            io: BlockingIo::new("echo"),
            seen: Mutex::new(Vec::new()),
            drain_steps: AtomicUsize::new(0),
            resume_seen: AtomicI64::new(i64::MIN),
        });
        let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(8));
        let output: Arc<dyn Edge> = Arc::new(BufferedEdge::new(8));
        node.bind_source("in", input.clone());
        node.bind_sink("out", output.clone());
        (node, input, output)
    }

    fn events(edge: &dyn Edge) -> Vec<&'static str> {
        let mut out = Vec::new();
        while let Some(item) = edge.try_take() {
            out.push(match item {
                EdgeItem::Buffer(_) => "buffer",
                EdgeItem::Event(EdgeEvent::Spec(_)) => "spec",
                EdgeItem::Event(EdgeEvent::FlushStart) => "flush-start",
                EdgeItem::Event(EdgeEvent::FlushStop { resume_at: None }) => "flush-stop",
                EdgeItem::Event(EdgeEvent::FlushStop { resume_at: Some(_) }) => "flush-stop+resume",
                EdgeItem::Event(EdgeEvent::Drain) => "drain",
                EdgeItem::Event(EdgeEvent::Eof) => "eof",
            });
        }
        out
    }

    /// A `FlushStart` pushed on an edge drops the buffers queued ahead of it,
    /// so the buffer is stepped through before the flush is queued.
    #[test]
    fn every_item_reaches_its_hook_and_control_events_are_forwarded() {
        let (node, input, output) = echo();
        node.start();

        input.push_event(EdgeEvent::Spec(spec()));
        input.push(test_media(AvpMediaType::VIDEO, 1));
        assert_eq!(node.process().unwrap(), Blocked::Again, "the spec");
        assert_eq!(node.process().unwrap(), Blocked::Again, "the buffer");
        assert_eq!(
            events(&*output),
            vec!["spec", "buffer"],
            "the published spec, then the echoed buffer"
        );

        input.push_event(EdgeEvent::FlushStart);
        input.push_event(EdgeEvent::FlushStop {
            resume_at: Some(Ts {
                val: 42,
                tb: AvpRational { num: 1, den: 1000 },
            }),
        });
        input.push_event(EdgeEvent::Eof);
        assert_eq!(node.process().unwrap(), Blocked::Again, "flush start");
        assert_eq!(node.process().unwrap(), Blocked::Again, "flush stop");
        assert_eq!(
            node.process().unwrap(),
            Blocked::Again,
            "eof starts the drain"
        );
        assert_eq!(node.process().unwrap(), Blocked::Done, "the drain finishes");
        node.stop();

        assert_eq!(
            *node.seen.lock().unwrap(),
            vec![
                "start",
                "spec",
                "buffer",
                "flush",
                "flush-stop",
                "eof",
                "drained",
                "stop"
            ]
        );
        assert_eq!(
            node.resume_seen.load(Ordering::SeqCst),
            42,
            "the hook sees the resume position"
        );
        assert_eq!(
            events(&*output),
            vec!["flush-start", "flush-stop+resume", "eof"],
            "both flush markers forwarded, the payload intact, then the drain's own Eof"
        );
    }

    #[test]
    fn a_hook_error_is_the_step_error() {
        let (node, input, _output) = echo();
        input.push(test_media(AvpMediaType::VIDEO, -1));
        let err = node.process().unwrap_err();
        assert_eq!((err.node.as_str(), err.phase), ("echo", NodePhase::Process));
        assert_eq!(err.message, "negative pts");
    }

    #[test]
    fn a_closed_input_finishes_through_on_closed() {
        let (node, input, _output) = echo();
        input.interrupt();
        assert_eq!(node.process().unwrap(), Blocked::Done);
        assert_eq!(*node.seen.lock().unwrap(), vec!["closed"]);
    }

    #[test]
    fn eof_finishes_at_once_when_the_handler_does_not_drain() {
        struct Sink {
            io: BlockingIo,
        }
        impl InputHandler for Sink {
            fn on_spec(&self, _: Spec) -> Result<Option<Spec>, NodeError> {
                Ok(None)
            }
            fn on_buffer(&self, _: Media) -> Result<Option<Media>, NodeError> {
                Ok(None)
            }
        }
        impl SingleInput for Sink {
            fn io(&self) -> &BlockingIo {
                &self.io
            }
        }
        let node = Blocking(Sink {
            io: BlockingIo::new("sink"),
        });
        let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(2));
        node.bind_source("in", input.clone());
        input.push(test_media(AvpMediaType::AUDIO, 0));
        input.push_event(EdgeEvent::Eof);
        assert_eq!(node.process().unwrap(), Blocked::Again);
        assert_eq!(node.process().unwrap(), Blocked::Done);
    }
}
