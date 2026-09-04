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

use std::sync::{Arc, Mutex, OnceLock};

use crate::graph::edge::{Edge, EdgeEvent, EdgeItem, EdgeWaker, Push, Wakeup};
use crate::graph::error::{NodeError, NodePhase};
use crate::graph::media::Media;
use crate::graph::node::{Blocked, Node, NodeBody, NodeFuture, NodeKind, Tick};
use crate::graph::poll_ctx::NodePollContext;
use crate::graph::spec::Spec;

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
    fn on_spec(&self, spec: &Spec) -> Result<(Self::Inner, Spec), String>;
    fn process(&self, inner: &mut Self::Inner, buf: Media) -> Result<Option<Media>, String>;
    fn on_flush(&self, _inner: &mut Self::Inner) {}
}

enum Drive {
    Again,
    Done,
    Push(Media),
}

struct SisoIo<F: SisoNode> {
    f: F,
    input: OnceLock<Arc<dyn Edge>>,
    output: OnceLock<Arc<dyn Edge>>,
    inner: Mutex<Option<F::Inner>>,
}

impl<F: SisoNode> SisoIo<F> {
    fn new(f: F) -> Self {
        Self {
            f,
            input: OnceLock::new(),
            output: OnceLock::new(),
            inner: Mutex::new(None),
        }
    }

    fn bind_source(&self, edge: Arc<dyn Edge>) {
        let _ = self.input.set(edge);
    }
    fn bind_sink(&self, edge: Arc<dyn Edge>) {
        let _ = self.output.set(edge);
    }

    fn drive(&self, item: EdgeItem) -> Result<Drive, NodeError> {
        match item {
            EdgeItem::Event(EdgeEvent::Spec(spec)) => match self.f.on_spec(&spec) {
                Ok((new_inner, out_spec)) => {
                    *self.inner.lock().unwrap() = Some(new_inner);
                    if let Some(out) = self.output.get() {
                        out.push_event(EdgeEvent::Spec(out_spec));
                    }
                    Ok(Drive::Again)
                }
                Err(message) => Err(NodeError::new(self.f.name(), NodePhase::Spec, message)),
            },
            EdgeItem::Buffer(buf) => {
                let mut guard = self.inner.lock().unwrap();
                match guard.as_mut() {
                    Some(inner) => match self.f.process(inner, buf) {
                        Ok(Some(out_buf)) => Ok(Drive::Push(out_buf)),
                        Ok(None) => Ok(Drive::Again),
                        Err(message) => {
                            Err(NodeError::new(self.f.name(), NodePhase::Process, message))
                        }
                    },
                    None => Err(NodeError::new(
                        self.f.name(),
                        NodePhase::Process,
                        "buffer received before initial Spec",
                    )),
                }
            }
            EdgeItem::Event(EdgeEvent::Eof) => Ok(Drive::Done),
            EdgeItem::Event(EdgeEvent::FlushStart) => {
                let mut guard = self.inner.lock().unwrap();
                if let Some(inner) = guard.as_mut() {
                    self.f.on_flush(inner);
                }
                drop(guard);
                if let Some(out) = self.output.get() {
                    out.push_event(EdgeEvent::FlushStart);
                }
                Ok(Drive::Again)
            }
            EdgeItem::Event(EdgeEvent::FlushStop) => {
                if let Some(out) = self.output.get() {
                    out.push_event(EdgeEvent::FlushStop);
                }
                Ok(Drive::Again)
            }
        }
    }
}

/// Blocking wrapper: `take(-1)` on a dedicated thread.
pub struct SisoAdapter<F: SisoNode> {
    io: SisoIo<F>,
    pending: Mutex<Option<Media>>,
    writable: Arc<Wakeup>,
}

struct BlockingWritable(Arc<Wakeup>);

impl EdgeWaker for BlockingWritable {
    fn wake(&self) {
        self.0.notify();
    }
}

impl<F: SisoNode> SisoAdapter<F> {
    pub fn new(f: F) -> Self {
        Self {
            io: SisoIo::new(f),
            pending: Mutex::new(None),
            writable: Arc::new(Wakeup::new()),
        }
    }

    fn offer_or_retry(&self, buf: Media) -> Blocked {
        let Some(out) = self.io.output.get() else {
            return Blocked::Again;
        };
        match out.offer(buf) {
            Ok(()) | Err((Push::Dropped | Push::Accepted, _)) => Blocked::Again,
            Err((Push::Closed, _)) => Blocked::Done,
            Err((Push::Full, buf)) => {
                *self.pending.lock().unwrap() = Some(buf);
                out.notify_writable(Box::new(BlockingWritable(self.writable.clone())));
                if out.is_full() {
                    // Bounded waiting keeps executor cancellation responsive
                    // even when the downstream group has already stopped.
                    self.writable.wait(10);
                }
                Blocked::Again
            }
        }
    }

    fn process_result(&self) -> Result<Blocked, NodeError> {
        let pending = self.pending.lock().unwrap().take();
        if let Some(buf) = pending {
            return Ok(self.offer_or_retry(buf));
        }
        let input = match self.io.input.get() {
            Some(edge) => edge,
            None => return Ok(Blocked::Done),
        };
        match input.take(-1) {
            Some(item) => match self.io.drive(item)? {
                Drive::Again => Ok(Blocked::Again),
                Drive::Done => Ok(Blocked::Done),
                Drive::Push(buf) => Ok(self.offer_or_retry(buf)),
            },
            None => Ok(Blocked::Done),
        }
    }
}

impl<F: SisoNode> Node for SisoAdapter<F> {
    fn name(&self) -> &str {
        self.io.f.name()
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Blocking
    }

    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.io.bind_source(edge);
    }
    fn bind_sink(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.io.bind_sink(edge);
    }

    fn process(&self) -> Blocked {
        self.process_result().unwrap_or(Blocked::Done)
    }

    fn take_body(self: Arc<Self>) -> NodeBody {
        NodeBody::Blocking(Box::new(move || self.process_result()))
    }
}

/// Cooperative wrapper: `try_take` + `Tick::Idle`, no private thread.
///
/// Output backpressure stashes the produced buffer and waits writable.
/// That stash is the Poll cost of not keeping locals across a park.
/// Use this as a Direct consumer only when `on_spec` and `process` are
/// infallible; scheduled Poll bodies propagate their errors to supervision,
/// while the fused Direct `Node::poll` contract cannot carry `NodeError`.
pub struct SisoPollAdapter<F: SisoNode> {
    io: SisoIo<F>,
    pending: Mutex<Option<Media>>,
}

impl<F: SisoNode> SisoPollAdapter<F> {
    pub fn new(f: F) -> Self {
        Self {
            io: SisoIo::new(f),
            pending: Mutex::new(None),
        }
    }

    fn offer_or_park(&self, buf: Media, ctx: &mut NodePollContext) -> Tick {
        let Some(out) = self.io.output.get() else {
            return Tick::Again;
        };
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

    fn poll_result(&self, ctx: &mut NodePollContext) -> Result<Tick, NodeError> {
        let pending = self.pending.lock().unwrap().take();
        if let Some(buf) = pending {
            return Ok(self.offer_or_park(buf, ctx));
        }
        if let Some(out) = self.io.output.get()
            && out.is_full()
        {
            ctx.wait_writable(out.clone());
            return Ok(Tick::Idle);
        }
        let input = match self.io.input.get() {
            Some(edge) => edge,
            None => return Ok(Tick::Done),
        };
        match input.try_take() {
            None => {
                if input.is_closed() {
                    Ok(Tick::Done)
                } else {
                    ctx.wait_readable(input.clone());
                    Ok(Tick::Idle)
                }
            }
            Some(item) => match self.io.drive(item)? {
                Drive::Again => Ok(Tick::Again),
                Drive::Done => Ok(Tick::Done),
                Drive::Push(buf) => Ok(self.offer_or_park(buf, ctx)),
            },
        }
    }
}

impl<F: SisoNode> Node for SisoPollAdapter<F> {
    fn name(&self) -> &str {
        self.io.f.name()
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Poll
    }

    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.io.bind_source(edge);
    }
    fn bind_sink(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.io.bind_sink(edge);
    }

    fn poll(&self, ctx: &mut NodePollContext) -> Tick {
        self.poll_result(ctx).unwrap_or(Tick::Done)
    }

    fn take_body(self: Arc<Self>) -> NodeBody {
        NodeBody::Poll(Box::new(move |ctx| self.poll_result(ctx)))
    }
}

/// Async wrapper: one future, produced buffers stay on the stack across
/// `wait_writable`. Same event loop as Poll.
pub struct SisoAsyncAdapter<F: SisoNode> {
    io: SisoIo<F>,
}

impl<F: SisoNode> SisoAsyncAdapter<F> {
    pub fn new(f: F) -> Self {
        Self { io: SisoIo::new(f) }
    }
}

impl<F: SisoNode> Node for SisoAsyncAdapter<F> {
    fn name(&self) -> &str {
        self.io.f.name()
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Async
    }

    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.io.bind_source(edge);
    }
    fn bind_sink(&self, _pad: &str, edge: Arc<dyn Edge>) {
        self.io.bind_sink(edge);
    }

    fn run_async(self: Arc<Self>) -> NodeFuture {
        Box::pin(async move {
            let Some(input) = self.io.input.get().cloned() else {
                return Ok(());
            };
            let output = self.io.output.get().cloned();
            loop {
                if let Some(out) = output.as_ref() {
                    if out.is_full() && !out.is_closed() {
                        out.wait_writable().await;
                        continue;
                    }
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
                match self.io.drive(item)? {
                    Drive::Again => {}
                    Drive::Done => return Ok(()),
                    Drive::Push(mut buf) => {
                        let Some(out) = output.as_ref() else {
                            continue;
                        };
                        loop {
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
                        }
                    }
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicBool;
    use std::task::{Context, Poll as TaskPoll, Waker};

    use super::*;
    use crate::graph::BufferedEdge;
    use crate::graph::buffer::{AvpMediaType, AvpRational};
    use crate::graph::edge::{EdgeEvent, Push, Wakeup};
    use crate::graph::error::NodePhase;
    use crate::graph::media::Media;
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
        #[cfg(not(feature = "ffmpeg"))]
        {
            Media::Stub {
                kind: AvpMediaType::VIDEO,
                pts,
            }
        }
        #[cfg(feature = "ffmpeg")]
        {
            let _ = pts;
            panic!("scaffold tests expect the default (non-ffmpeg) Media::Stub");
        }
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
            match node.poll(ctx) {
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
        assert_eq!(node.poll(&mut ctx), Tick::Idle);
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
        assert_eq!(node.poll(&mut ctx), Tick::Idle);
        assert!(ctx.needs_park());

        let _ = output.try_take();
        assert_eq!(node.poll(&mut ctx), Tick::Again);
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

    #[test]
    fn blocking_adapter_retries_a_full_output_without_losing_the_buffer() {
        let node = SisoAdapter::new(Identity { name: "blocking" });
        let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(2));
        let output: Arc<dyn Edge> = Arc::new(BufferedEdge::new(1));
        node.bind_source("src", input.clone());
        node.bind_sink("dst", output.clone());

        assert_eq!(output.push(stub(1)), Push::Accepted);
        input.push_event(EdgeEvent::Spec(video_spec()));
        assert_eq!(input.push(stub(9)), Push::Accepted);

        while input.occupied() > 0 {
            assert_eq!(node.process(), Blocked::Again);
        }
        assert_eq!(node.process(), Blocked::Again);
        assert_eq!(take_bufs(&*output), vec![1]);
        assert_eq!(node.process(), Blocked::Again);
        assert_eq!(take_bufs(&*output), vec![9]);
    }
}
