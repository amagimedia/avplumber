//! Optional helpers for the linear single-input / single-output case
//! (C++ `NodeSISO`). Not libavformat: no mux/demux here.
//!
//! `Node` is still the only runtime contract. Implement it directly for
//! sources, sinks, fan-in/fan-out, or anything that is not a transform of a
//! single stream. `SisoNode` exists so those transforms do not each
//! reimplement "take from input, classify Spec vs buffer vs flush, push to
//! output."
//!
//! Three wrappers, same `SisoNode` body:
//! - [`SisoAdapter`] — blocking (`take(-1)`), own OS thread
//! - [`SisoPollAdapter`] — cooperative poll, shared event loop
//! - [`SisoAsyncAdapter`] — `run_async`, same event loop, locals
//!   survive `.await` (output backpressure, later clock-gated nodes)
//!
//! The first two are [`SingleInput`]/[`PollNode`] implementations behind the
//! [`Blocking`]/[`Polling`] wrappers, so they carry an [`Io`] like every other
//! scaffolded node, and all three drive the same [`InputHandler`] through
//! [`react`]: what is Siso-specific is only the `Inner` and the three hooks.

use std::sync::{Arc, Mutex};

use crate::graph::edge::{Edge, Push};
use crate::graph::error::{NodeError, NodePhase};
use crate::graph::media::{Media, Ts};
use crate::graph::node::{Node, NodeFuture, NodeKind, Tick};
use crate::graph::poll_ctx::NodePollContext;
use crate::graph::spec::Spec;
use crate::scaffold::blocking::Blocking;
use crate::scaffold::io::{BlockingIo, Io};
use crate::scaffold::poll::{PollNode, Polling};
use crate::scaffold::single_input::{InputHandler, Reaction, SingleInput, react};

/// Per-buffer transform whose codec/size/layout state is rebuilt from `Spec`.
///
/// Single input, single output — the C++ `NodeSISO` case. Not a container
/// node (`AVFormatContext` / mux / demux).
///
/// Most filters, rescalers, resamplers, and bitstream filters cannot process
/// a buffer until the stream `Spec` is known, and must rebuild that state
/// when it changes mid-stream. `Inner` is that state: `on_spec` constructs
/// it, `process` uses it. There is no `Inner` until the first `Spec` arrives
/// (the edge always delivers a latched `Spec` before buffers).
///
/// Three `on_spec` shapes, same hook:
/// - identity: return the incoming `Spec` and a dummy `Inner` (firewall);
/// - query: read the format, configure `Inner`, forward the same `Spec`
///   (encoder);
/// - transform: return a *new* `Spec` describing the output
///   (rescale, resample, filters). Downstream readers see the transformed
///   format on their adjacent edge, not the original.
///
/// `process` returning `Ok(None)` drops the buffer without producing output.
/// Wrap in a schedule-specific adapter to get a `Node`.
pub trait SisoNode: Send + Sync + 'static {
    type Inner: Send;

    fn name(&self) -> &str;
    /// Opts `SisoPollAdapter` into Direct input. Return `true` only when
    /// `on_spec` and `process` are guaranteed not to return `Err`.
    fn direct_consumer_is_infallible(&self) -> bool {
        false
    }
    fn on_spec(&self, spec: &Spec) -> Result<(Self::Inner, Spec), String>;
    fn process(&self, inner: &mut Self::Inner, buf: Media) -> Result<Option<Media>, String>;
    fn on_flush(&self, _inner: &mut Self::Inner) {}
    /// See [`InputHandler::on_flush_stop`].
    fn on_flush_stop(&self, _inner: &mut Self::Inner, _resume_at: Option<Ts>) {}
}

/// The [`InputHandler`] every adapter drives: the `Inner` and the
/// [`SisoNode`] hooks behind the generic ones.
struct SisoCore<F: SisoNode> {
    f: F,
    inner: Mutex<Option<F::Inner>>,
}

impl<F: SisoNode> SisoCore<F> {
    fn new(f: F) -> Self {
        Self {
            f,
            inner: Mutex::new(None),
        }
    }

    fn error(&self, phase: NodePhase, message: impl Into<String>) -> NodeError {
        NodeError::new(self.f.name(), phase, message)
    }
}

impl<F: SisoNode> InputHandler for SisoCore<F> {
    fn on_spec(&self, spec: Spec) -> Result<Option<Spec>, NodeError> {
        let (inner, out_spec) = self
            .f
            .on_spec(&spec)
            .map_err(|message| self.error(NodePhase::Spec, message))?;
        *self.inner.lock().unwrap() = Some(inner);
        Ok(Some(out_spec))
    }

    fn on_buffer(&self, buf: Media) -> Result<Option<Media>, NodeError> {
        let mut guard = self.inner.lock().unwrap();
        let inner = guard
            .as_mut()
            .ok_or_else(|| self.error(NodePhase::Process, "buffer received before initial Spec"))?;
        self.f
            .process(inner, buf)
            .map_err(|message| self.error(NodePhase::Process, message))
    }

    fn on_flush(&self) {
        if let Some(inner) = self.inner.lock().unwrap().as_mut() {
            self.f.on_flush(inner);
        }
    }

    fn on_flush_stop(&self, resume_at: Option<Ts>) {
        if let Some(inner) = self.inner.lock().unwrap().as_mut() {
            self.f.on_flush_stop(inner, resume_at);
        }
    }
}

/// The blocking body: a [`SingleInput`] over [`SisoCore`], so the loop is the
/// scaffold's. Like every scaffolded node it fails on an unbound pad.
pub struct SisoBlocking<F: SisoNode> {
    core: SisoCore<F>,
    io: BlockingIo,
}

/// Blocking wrapper: [`SisoBlocking`] as a `Node`.
pub type SisoAdapter<F> = Blocking<SisoBlocking<F>>;

impl<F: SisoNode> Blocking<SisoBlocking<F>> {
    pub fn new(f: F) -> Self {
        Blocking(SisoBlocking {
            io: BlockingIo::new(f.name()),
            core: SisoCore::new(f),
        })
    }
}

impl<F: SisoNode> InputHandler for SisoBlocking<F> {
    fn on_spec(&self, spec: Spec) -> Result<Option<Spec>, NodeError> {
        self.core.on_spec(spec)
    }

    fn on_buffer(&self, buf: Media) -> Result<Option<Media>, NodeError> {
        self.core.on_buffer(buf)
    }

    fn on_flush(&self) {
        self.core.on_flush()
    }

    fn on_flush_stop(&self, resume_at: Option<Ts>) {
        self.core.on_flush_stop(resume_at)
    }
}

impl<F: SisoNode> SingleInput for SisoBlocking<F> {
    fn io(&self) -> &BlockingIo {
        &self.io
    }
}

/// The cooperative body: `try_take` + `Tick::Idle`, no private thread.
///
/// Output backpressure stashes the produced buffer and waits writable.
/// That stash is the Poll cost of not keeping locals across a park.
/// Use this as a Direct consumer only when `on_spec` and `process` are
/// infallible, and opt in with `SisoNode::direct_consumer_is_infallible`; scheduled
/// Poll bodies propagate their errors to supervision, while the fused Direct
/// `Node::poll` contract cannot carry `NodeError`.
pub struct SisoPolling<F: SisoNode> {
    core: SisoCore<F>,
    io: Io,
    pending: Mutex<Option<Media>>,
}

/// Cooperative wrapper: [`SisoPolling`] as a `Node`.
pub type SisoPollAdapter<F> = Polling<SisoPolling<F>>;

impl<F: SisoNode> Polling<SisoPolling<F>> {
    pub fn new(f: F) -> Self {
        Polling(SisoPolling {
            io: Io::new(f.name(), NodePhase::Poll),
            core: SisoCore::new(f),
            pending: Mutex::new(None),
        })
    }
}

impl<F: SisoNode> SisoPolling<F> {
    fn offer_or_park(&self, out: &Arc<dyn Edge>, buf: Media, ctx: &mut NodePollContext) -> Tick {
        match out.offer(buf) {
            Ok(()) => Tick::Again,
            Err((Push::Dropped, _)) => Tick::Again,
            Err((Push::Closed, _)) => Tick::Done,
            Err((Push::Full, buf)) => {
                *self.pending.lock().unwrap() = Some(buf);
                ctx.wait_writable(out.clone());
                Tick::Idle
            }
            Err((Push::Accepted, _)) => Tick::Again,
        }
    }
}

impl<F: SisoNode> PollNode for SisoPolling<F> {
    fn io(&self) -> &Io {
        &self.io
    }

    fn direct_consumer_is_infallible(&self) -> bool {
        self.core.f.direct_consumer_is_infallible()
    }

    fn step(&self, ctx: &mut NodePollContext) -> Result<Tick, NodeError> {
        let out = self.io.output()?;
        let pending = self.pending.lock().unwrap().take();
        if let Some(buf) = pending {
            return Ok(self.offer_or_park(&out, buf, ctx));
        }
        if out.is_full() {
            ctx.wait_writable(out);
            return Ok(Tick::Idle);
        }
        let input = self.io.input()?;
        let Some(item) = input.try_take() else {
            if input.is_closed() {
                return Ok(Tick::Done);
            }
            ctx.wait_readable(input);
            return Ok(Tick::Idle);
        };
        Ok(match react(&self.core, Some(&out), item)? {
            Reaction::Again => Tick::Again,
            Reaction::Done => Tick::Done,
            Reaction::Produced(buf) => self.offer_or_park(&out, buf, ctx),
        })
    }
}

/// Async wrapper: one future, produced buffers stay on the stack across
/// `wait_writable`. Same event loop as Poll. Implements `Node` directly, since
/// the scaffold has no async node trait yet.
pub struct SisoAsyncAdapter<F: SisoNode> {
    core: SisoCore<F>,
    io: Io,
}

impl<F: SisoNode> SisoAsyncAdapter<F> {
    pub fn new(f: F) -> Self {
        Self {
            io: Io::new(f.name(), NodePhase::Async),
            core: SisoCore::new(f),
        }
    }
}

impl<F: SisoNode> Node for SisoAsyncAdapter<F> {
    fn name(&self) -> &str {
        &self.io.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Async
    }

    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.io.input_slot.bind(edge);
    }
    fn bind_sink(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.io.output_slot.bind(edge);
    }

    fn run_async(self: Arc<Self>) -> NodeFuture {
        Box::pin(async move {
            let input = self.io.input()?;
            let out = self.io.output()?;
            loop {
                if out.is_full() && !out.is_closed() {
                    out.wait_writable().await;
                    continue;
                }
                if input.occupied() == 0 && !input.is_closed() {
                    input.wait_readable().await;
                }
                let Some(item) = input.try_take() else {
                    if input.is_closed() {
                        return Ok(());
                    }
                    continue;
                };
                match react(&self.core, Some(&out), item)? {
                    Reaction::Again => {}
                    Reaction::Done => return Ok(()),
                    Reaction::Produced(mut buf) => loop {
                        match out.offer(buf) {
                            Ok(()) => break,
                            Err((Push::Dropped, _)) => break,
                            Err((Push::Closed, _)) => return Ok(()),
                            Err((Push::Full, back)) => {
                                buf = back;
                                out.wait_writable().await;
                            }
                            Err((Push::Accepted, _)) => break,
                        }
                    },
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::task::{Context, Poll as TaskPoll, Waker};
    use std::time::Duration;

    use super::*;
    use crate::graph::BufferedEdge;
    use crate::graph::buffer::{AvpMediaType, AvpRational};
    use crate::graph::edge::{EdgeEvent, EdgeItem, Push, Wakeup};
    use crate::graph::error::NodePhase;
    use crate::graph::media::Media;
    use crate::graph::node::Blocked;
    use crate::graph::node::{NodeBody, Tick};
    use crate::graph::poll_ctx::NodePollContext;
    use crate::graph::spec::Spec;

    struct Identity {
        name: &'static str,
    }

    impl SisoNode for Identity {
        type Inner = ();
        fn name(&self) -> &str {
            self.name
        }
        fn on_spec(&self, spec: &Spec) -> Result<((), Spec), String> {
            Ok(((), spec.clone()))
        }
        fn process(&self, _inner: &mut (), buf: Media) -> Result<Option<Media>, String> {
            Ok(Some(buf))
        }
    }

    struct FailsOnSpec {
        name: &'static str,
    }

    impl SisoNode for FailsOnSpec {
        type Inner = ();

        fn name(&self) -> &str {
            self.name
        }

        fn on_spec(&self, _spec: &Spec) -> Result<((), Spec), String> {
            Err("unsupported input format".into())
        }

        fn process(&self, _inner: &mut (), _buf: Media) -> Result<Option<Media>, String> {
            unreachable!("a failed Spec must not install processing state")
        }
    }

    struct FailsOnBuffer;

    impl SisoNode for FailsOnBuffer {
        type Inner = ();

        fn name(&self) -> &str {
            "buffer_error"
        }

        fn on_spec(&self, spec: &Spec) -> Result<((), Spec), String> {
            Ok(((), spec.clone()))
        }

        fn process(&self, _inner: &mut (), _buf: Media) -> Result<Option<Media>, String> {
            Err("decoder rejected buffer".into())
        }
    }

    fn video_spec() -> Spec {
        Spec::Video {
            width: 8,
            height: 8,
            pix_fmt: 0,
            frame_rate: AvpRational { num: 1, den: 1 },
            sar: AvpRational { num: 1, den: 1 },
            time_base: AvpRational { num: 1, den: 1000 },
        }
    }

    fn stub(pts: i64) -> Media {
        crate::graph::media::test_media(AvpMediaType::VIDEO, pts)
    }

    fn bind_pair<N: Node>(node: &N) -> (Arc<dyn Edge>, Arc<dyn Edge>) {
        let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(8));
        let output: Arc<dyn Edge> = Arc::new(BufferedEdge::new(8));
        node.bind_source("src", input.clone());
        node.bind_sink("dst", output.clone());
        (input, output)
    }

    fn poll_ctx() -> NodePollContext {
        NodePollContext::new(Arc::new(AtomicBool::new(false)), Arc::new(Wakeup::new()))
    }

    fn take_bufs(edge: &dyn Edge) -> Vec<i64> {
        let mut pts = Vec::new();
        while let Some(item) = edge.try_take() {
            if let EdgeItem::Buffer(buf) = item {
                pts.push(buf.ts().val);
            }
        }
        pts
    }

    fn pump_until_idle(node: &impl Node, ctx: &mut NodePollContext) {
        for _ in 0..32 {
            match node.poll(ctx).unwrap() {
                Tick::Idle | Tick::Done => return,
                Tick::Again => {}
            }
        }
        panic!("poll helper did not become idle");
    }

    #[test]
    fn poll_idles_on_empty_input() {
        let node = SisoPollAdapter::new(Identity { name: "p" });
        let (_in, _out) = bind_pair(&node);
        let mut ctx = poll_ctx();
        assert_eq!(node.poll(&mut ctx).unwrap(), Tick::Idle);
        assert!(ctx.needs_park());
    }

    #[test]
    fn poll_forwards_spec_then_buffer() {
        let node = SisoPollAdapter::new(Identity { name: "p" });
        let (input, output) = bind_pair(&node);
        input.push_event(EdgeEvent::Spec(video_spec()));
        assert_eq!(input.push(stub(7)), Push::Accepted);

        let mut ctx = poll_ctx();
        pump_until_idle(&node, &mut ctx);
        assert_eq!(take_bufs(&*output), vec![7]);
    }

    #[test]
    fn poll_idles_when_output_is_full() {
        let node = SisoPollAdapter::new(Identity { name: "p" });
        let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(1));
        let output: Arc<dyn Edge> = Arc::new(BufferedEdge::new(1));
        node.bind_source("src", input.clone());
        node.bind_sink("dst", output.clone());

        input.push_event(EdgeEvent::Spec(video_spec()));
        let mut ctx = poll_ctx();
        pump_until_idle(&node, &mut ctx);
        while output.try_take().is_some() {}

        assert_eq!(output.push(stub(1)), Push::Accepted);
        assert_eq!(input.push(stub(9)), Push::Accepted);
        assert_eq!(node.poll(&mut ctx).unwrap(), Tick::Idle);
        assert!(ctx.needs_park());

        let _ = output.try_take();
        assert_eq!(node.poll(&mut ctx).unwrap(), Tick::Again);
        assert_eq!(take_bufs(&*output), vec![9]);
    }

    fn poll_future_ready<T>(fut: impl std::future::Future<Output = T>) -> T {
        let mut fut = std::pin::pin!(fut);
        let waker = Waker::noop();
        let mut cx = Context::from_waker(&waker);
        match fut.as_mut().poll(&mut cx) {
            TaskPoll::Ready(v) => v,
            TaskPoll::Pending => panic!("format async helper parked with input already queued"),
        }
    }

    #[test]
    fn async_forwards_spec_then_buffer_then_eof() {
        let node = Arc::new(SisoAsyncAdapter::new(Identity { name: "a" }));
        let (input, output) = bind_pair(node.as_ref());
        input.push_event(EdgeEvent::Spec(video_spec()));
        assert_eq!(input.push(stub(3)), Push::Accepted);
        input.push_event(EdgeEvent::Eof);

        let NodeBody::Async(fut) = node.clone().take_body() else {
            panic!("expected async body");
        };
        poll_future_ready(fut).unwrap();
        assert_eq!(take_bufs(&*output), vec![3]);
    }

    #[test]
    fn scheduled_adapters_propagate_siso_spec_errors() {
        let blocking = Arc::new(SisoAdapter::new(FailsOnSpec { name: "blocking" }));
        let (input, _output) = bind_pair(blocking.as_ref());
        input.push_event(EdgeEvent::Spec(video_spec()));
        let NodeBody::Blocking(mut step) = blocking.take_body() else {
            panic!("expected blocking body");
        };
        let error = step().unwrap_err();
        assert_eq!(error.node, "blocking");
        assert_eq!(error.phase, NodePhase::Spec);
        assert_eq!(error.message, "unsupported input format");

        let poll = Arc::new(SisoPollAdapter::new(FailsOnSpec { name: "poll" }));
        let (input, _output) = bind_pair(poll.as_ref());
        input.push_event(EdgeEvent::Spec(video_spec()));
        let NodeBody::Poll(mut step) = poll.take_body() else {
            panic!("expected poll body");
        };
        let error = step(&mut poll_ctx()).unwrap_err();
        assert_eq!(error.node, "poll");
        assert_eq!(error.phase, NodePhase::Spec);
        assert_eq!(error.message, "unsupported input format");

        let asynchronous = Arc::new(SisoAsyncAdapter::new(FailsOnSpec { name: "async" }));
        let (input, _output) = bind_pair(asynchronous.as_ref());
        input.push_event(EdgeEvent::Spec(video_spec()));
        let NodeBody::Async(future) = asynchronous.take_body() else {
            panic!("expected async body");
        };
        let error = poll_future_ready(future).unwrap_err();
        assert_eq!(error.node, "async");
        assert_eq!(error.phase, NodePhase::Spec);
        assert_eq!(error.message, "unsupported input format");
    }

    #[test]
    fn scheduled_poll_adapter_propagates_siso_process_errors() {
        let node = Arc::new(SisoPollAdapter::new(FailsOnBuffer));
        let (input, _output) = bind_pair(node.as_ref());
        input.push_event(EdgeEvent::Spec(video_spec()));
        assert_eq!(input.push(stub(7)), Push::Accepted);
        let NodeBody::Poll(mut step) = node.take_body() else {
            panic!("expected poll body");
        };
        let mut ctx = poll_ctx();
        let error = (0..4)
            .find_map(|_| step(&mut ctx).err())
            .expect("the queued buffer must reach SisoNode::process");
        assert_eq!(error.node, "buffer_error");
        assert_eq!(error.phase, NodePhase::Process);
        assert_eq!(error.message, "decoder rejected buffer");
    }

    /// A blocking adapter whose output already holds one buffer of one, with
    /// the Spec consumed and one buffer queued on its input.
    fn blocked_on_full_output() -> (Arc<SisoAdapter<Identity>>, Arc<dyn Edge>) {
        let node = Arc::new(SisoAdapter::new(Identity { name: "blocking" }));
        let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(2));
        let output: Arc<dyn Edge> = Arc::new(BufferedEdge::new(1));
        node.bind_source("src", input.clone());
        node.bind_sink("dst", output.clone());

        assert_eq!(output.push(stub(1)), Push::Accepted);
        input.push_event(EdgeEvent::Spec(video_spec()));
        assert_eq!(input.push(stub(9)), Push::Accepted);
        assert_eq!(node.process().unwrap(), Blocked::Again, "the Spec");
        (node, output)
    }

    fn take_one(edge: &dyn Edge) -> Option<i64> {
        match edge.try_take() {
            Some(EdgeItem::Buffer(buf)) => Some(buf.ts().val),
            _ => None,
        }
    }

    #[test]
    fn blocking_adapter_parks_on_a_full_output_without_losing_the_buffer() {
        let (node, output) = blocked_on_full_output();
        let drained = {
            let output = output.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                take_one(&*output)
            })
        };

        assert_eq!(
            node.process().unwrap(),
            Blocked::Again,
            "parks, then pushes 9"
        );
        assert_eq!(drained.join().unwrap(), Some(1));
        assert_eq!(take_bufs(&*output), vec![9]);
    }

    #[test]
    fn blocking_adapter_interrupt_releases_a_full_output_park() {
        let (node, output) = blocked_on_full_output();
        let interrupter = {
            let node = node.clone();
            std::thread::spawn(move || {
                std::thread::sleep(Duration::from_millis(20));
                node.interrupt();
            })
        };

        assert_eq!(node.process().unwrap(), Blocked::Done);
        interrupter.join().unwrap();
        assert_eq!(
            take_bufs(&*output),
            vec![1],
            "nothing was pushed past the park"
        );
    }
}
