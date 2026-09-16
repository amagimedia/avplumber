//! A cooperative node with one input, written as [`InputHandler`] reactions
//! rather than a `try_take` match on every event kind.
//!
//! [`PollInput`] is to [`PollNode`] what [`SingleInput`](crate::node_api::SingleInput)
//! is to [`BlockingNode`](crate::node_api::BlockingNode): the node writes
//! `on_spec` / `on_buffer` / `on_flush`, and this module owns the classification,
//! the output stash, and backpressure. Hooks stay `&self` so the node can keep
//! counters in atomics, a grid in a mutex, or mix the two — the adapter never
//! wraps the node's state.

use std::collections::VecDeque;
use std::ops::Deref;
use std::sync::{Arc, Mutex};

use crate::graph::edge::{Edge, Push};
use crate::graph::error::{NodeError, NodePhase};
use crate::graph::grain::Grain;
use crate::graph::node::Polled;
use crate::graph::pad::NodePads;
use crate::graph::poll_ctx::NodePollContext;
use crate::node_api::io::{Io, flush_at_head};
use crate::node_api::poll::PollNode;
use crate::node_api::single_input::{InputHandler, Reaction, react};

/// Where one `offer` of a stashed buffer left the node.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DrainPending {
    /// The stash is empty.
    Drained,
    /// The output was full; the remainder stays queued and waiters are armed.
    Parked,
    Closed,
}

/// [`Io`] plus the output stash a cooperative body needs: `try_take` already
/// consumed the input, so a 1:N step that cannot `offer` yet keeps the extra
/// buffers here, not in the node's algorithm state.
///
/// The mutex is the bus, not a requirement on how the node keeps counters or
/// grid state.
pub struct PollIo {
    io: Io,
    pending: Mutex<VecDeque<Grain>>,
}

impl PollIo {
    pub fn new(name: &str) -> Self {
        Self {
            io: Io::new(name, NodePhase::Poll),
            pending: Mutex::new(VecDeque::new()),
        }
    }

    pub fn clear_pending(&self) {
        self.pending.lock().unwrap().clear();
    }

    pub fn extend_pending(&self, buffers: impl IntoIterator<Item = Grain>) {
        self.pending.lock().unwrap().extend(buffers);
    }

    /// Drop stashed output that a `FlushStart` now at the head of `input` has
    /// made stale. `true` if anything was discarded.
    pub fn drop_stale_pending(&self, input: &Arc<dyn Edge>) -> bool {
        if self.pending.lock().unwrap().is_empty() || !flush_at_head(input) {
            return false;
        }
        self.clear_pending();
        true
    }

    /// Offers the stash oldest-first. On [`DrainPending::Parked`] the rest stays
    /// queued, the output is waited writable, and `input` is waited for a flush
    /// so a seek can abandon what was produced.
    pub fn drain_pending(
        &self,
        ctx: &mut NodePollContext,
        input: &Arc<dyn Edge>,
        output: &Arc<dyn Edge>,
    ) -> DrainPending {
        let mut pending = self.pending.lock().unwrap();
        while let Some(buffer) = pending.pop_front() {
            match output.offer(buffer) {
                Ok(()) | Err((Push::Dropped | Push::Accepted, _)) => {}
                Err((Push::Full, buffer)) => {
                    pending.push_front(buffer);
                    ctx.wait_writable(output.clone());
                    ctx.wait_flush(input.clone());
                    return DrainPending::Parked;
                }
                Err((Push::Closed, _)) => {
                    log::info!("{}: output closed, discarding", self.name);
                    pending.clear();
                    return DrainPending::Closed;
                }
            }
        }
        DrainPending::Drained
    }
}

impl Deref for PollIo {
    type Target = Io;

    fn deref(&self) -> &Io {
        &self.io
    }
}

/// A Poll node with one input and (usually) one output, driven by
/// [`InputHandler`]. Register as `Polling<Self>`.
///
/// [`Self::on_buffer`] may return several buffers; they sit on [`PollIo`] until
/// the output has room. That stash is not the node's algorithm.
pub trait PollInput: InputHandler {
    fn io(&self) -> &PollIo;

    fn pads(&self) -> NodePads {
        NodePads::default()
    }

    fn direct_consumer_is_infallible(&self) -> bool {
        false
    }

    fn start(&self) {}

    fn stop(&self) {}

    fn interrupt(&self) {}

    fn set_object(&self, key: &str, _value: &serde_json::Value) -> Result<(), String> {
        Err(format!("{} has no object `{key}` to set", self.io().name))
    }

    fn get_object(&self, key: &str) -> Result<serde_json::Value, String> {
        Err(format!("{} has no object `{key}` to get", self.io().name))
    }
}

impl<N: PollInput> PollNode for N {
    fn io(&self) -> &crate::node_api::Io {
        PollInput::io(self)
    }

    fn pads(&self) -> NodePads {
        PollInput::pads(self)
    }

    fn direct_consumer_is_infallible(&self) -> bool {
        PollInput::direct_consumer_is_infallible(self)
    }

    fn start(&self) {
        PollInput::io(self).clear_pending();
        PollInput::start(self);
    }

    fn stop(&self) {
        PollInput::stop(self);
    }

    fn interrupt(&self) {
        PollInput::interrupt(self);
    }

    fn set_object(&self, key: &str, value: &serde_json::Value) -> Result<(), String> {
        PollInput::set_object(self, key, value)
    }

    fn get_object(&self, key: &str) -> Result<serde_json::Value, String> {
        PollInput::get_object(self, key)
    }

    fn step(&self, ctx: &mut NodePollContext) -> Result<Polled, NodeError> {
        let io = PollInput::io(self);
        let input = io.input()?;
        let output = io.output()?;

        io.drop_stale_pending(&input);
        match io.drain_pending(ctx, &input, &output) {
            DrainPending::Parked => return Ok(Polled::Idle),
            DrainPending::Closed => return Ok(Polled::Done),
            DrainPending::Drained => {}
        }
        if output.is_full() && !output.is_closed() {
            ctx.wait_writable(output);
            return Ok(Polled::Idle);
        }

        let Some(item) = input.try_take() else {
            if input.is_closed() {
                self.on_closed();
                return Ok(Polled::Done);
            }
            ctx.wait_readable(input);
            return Ok(Polled::Idle);
        };

        match react(self, Some(&output), item)? {
            Reaction::Again => Ok(Polled::Again),
            Reaction::Done => Ok(Polled::Done),
            Reaction::Produced(buffers) => {
                io.extend_pending(buffers);
                Ok(match io.drain_pending(ctx, &input, &output) {
                    DrainPending::Parked => Polled::Idle,
                    DrainPending::Closed => Polled::Done,
                    DrainPending::Drained => Polled::Again,
                })
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    use super::*;
    use crate::graph::BufferedEdge;
    use crate::graph::edge::{Edge, EdgeEvent, EdgeItem, Wakeup};
    use crate::graph::error::NodeError;
    use crate::graph::grain::{Grain, test_media};
    use crate::graph::media::AvpMediaType;
    use crate::graph::node::{Node, Polled};
    use crate::graph::pad::NodePads;
    use crate::graph::spec::Spec;
    use crate::node_api::poll::Polling;

    /// One input frame becomes two outputs. The count is an atomic: the node
    /// has no mutex of its own.
    struct Dup {
        io: PollIo,
        emitted: AtomicU64,
    }

    impl InputHandler for Dup {
        fn on_spec(&self, spec: Spec) -> Result<Option<Spec>, NodeError> {
            Ok(Some(spec))
        }

        fn on_buffer(&self, buffer: Grain) -> Result<Vec<Grain>, NodeError> {
            self.emitted.fetch_add(2, Ordering::Relaxed);
            let copy = buffer.clone();
            Ok(vec![buffer, copy])
        }
    }

    impl PollInput for Dup {
        fn io(&self) -> &PollIo {
            &self.io
        }

        fn pads(&self) -> NodePads {
            NodePads::siso(AvpMediaType::VIDEO, AvpMediaType::VIDEO)
        }

        fn get_object(&self, key: &str) -> Result<serde_json::Value, String> {
            match key {
                "emitted" => Ok(serde_json::json!(self.emitted.load(Ordering::Relaxed))),
                other => Err(format!("unknown {other}")),
            }
        }
    }

    fn ctx() -> NodePollContext {
        NodePollContext::new(Arc::new(AtomicBool::new(false)), Arc::new(Wakeup::new()))
    }

    #[test]
    fn one_buffer_can_produce_several_without_a_node_mutex() {
        let node = Polling(Dup {
            io: PollIo::new("dup"),
            emitted: AtomicU64::new(0),
        });
        let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(8));
        let output: Arc<dyn Edge> = Arc::new(BufferedEdge::new(8));
        node.bind_source("in", input.clone());
        node.bind_sink("out", output.clone());
        node.start();

        input.push_event(EdgeEvent::Spec(Spec::Video {
            width: 2,
            height: 2,
            pix_fmt: 0,
            sw_pix_fmt: -1,
            frame_rate: Default::default(),
            sar: Default::default(),
            time_base: Default::default(),
        }));
        assert!(input.offer(test_media(AvpMediaType::VIDEO, 1)).is_ok());

        let mut ctx = ctx();
        assert_eq!(node.poll(&mut ctx).unwrap(), Polled::Again, "the spec");
        assert_eq!(node.poll(&mut ctx).unwrap(), Polled::Again, "the buffer");

        let mut pts = Vec::new();
        while let Some(item) = output.try_take() {
            if let EdgeItem::Buffer(buf) = item {
                pts.push(buf.ts().val);
            }
        }
        assert_eq!(pts, vec![1, 1]);
        assert_eq!(node.get_object("emitted").unwrap(), serde_json::json!(2));
    }
}
