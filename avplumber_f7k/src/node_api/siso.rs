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
//! The first two are [`SingleInput`]/[`PollInput`] implementations behind the
//! [`Blocking`](crate::node_api::Blocking)/[`Polling`] wrappers, so they carry
//! [`BlockingIo`]/[`PollIo`] like every other node written this way, and all three
//! drive the same [`InputHandler`] through [`react`]: what is Siso-specific is
//! only the `InputState` and the three hooks.

use std::sync::{Arc, Mutex};

use crate::graph::edge::{Edge, EdgeEvent, Push};
use crate::graph::error::{NodeError, NodePhase};
use crate::graph::grain::Grain;
use crate::graph::node::{Node, NodeFuture, NodeKind};
use crate::graph::pad::NodePads;
use crate::graph::spec::Spec;
use crate::graph::timestamp::Ts;
use crate::node_api::blocking::{Blocking, BlockingIo};
use crate::node_api::io::Io;
use crate::node_api::poll::Polling;
use crate::node_api::poll_input::{PollInput, PollIo};
use crate::node_api::single_input::{EofAction, InputHandler, Reaction, SingleInput, react};

/// Per-buffer transform whose codec/size/layout state is rebuilt from `Spec`.
///
/// Single input, single output — the C++ `NodeSISO` case. Not a container
/// node (`AVFormatContext` / mux / demux), and not a codec that parks on
/// `EAGAIN` (those stay on [`SingleInput`](crate::node_api::SingleInput)).
///
/// `InputState` is processing state opened from the **input** `Spec`: `on_spec`
/// constructs it, `process` / `on_eof` use it. It is not pad/edge state (that
/// is the `Spec` on the edge, plus the adapter's last-input copy for
/// `same_as`). There is no `InputState` until the first `Spec` arrives; buffers
/// before that are dropped and counted. An identical re-delivery is a no-op in
/// the adapter (`Spec::same_as`); a changed input spec replaces `InputState`.
/// JSON and anything that must survive a format change live on `Self`.
///
/// Three `on_spec` shapes, same hook:
/// - identity: return the incoming `Spec` and a dummy `InputState`;
/// - query: read the format, configure `InputState`, forward the same `Spec`;
/// - transform: return a *new* `Spec` describing the output (rescale,
///   resample, bitstream filter). Downstream readers see the transformed
///   format on their adjacent edge, not the original.
///
/// `process` returns every grain this input produced: empty to drop, one to
/// forward, several when one input becomes many. `on_eof` is the same for
/// whatever was still inside `InputState`. Wrap in a schedule-specific adapter
/// to get a `Node`.
pub trait SisoNode: Send + Sync + 'static {
    type InputState: Send;

    fn name(&self) -> &str;

    /// For the media-type check at connect. Declaring none skips the check.
    fn pads(&self) -> NodePads {
        NodePads::default()
    }

    /// Opts `SisoPollAdapter` into Direct input. Return `true` only when
    /// `on_spec`, `process` and `on_eof` are guaranteed not to return `Err`.
    fn direct_consumer_is_infallible(&self) -> bool {
        false
    }

    fn on_spec(&self, spec: &Spec) -> Result<(Self::InputState, Spec), String>;

    fn process(&self, input_state: &mut Self::InputState, buf: Grain)
    -> Result<Vec<Grain>, String>;

    fn flush(&self, _input_state: &mut Self::InputState) {}

    /// See [`InputHandler::on_flush_stop`].
    fn flush_stop(&self, _input_state: &mut Self::InputState, _resume_at: Option<Ts>) {}

    /// Drain anything still inside `InputState`. Default: nothing. The adapter
    /// pushes the result, then forwards `Eof` and finishes.
    fn on_eof(&self, _input_state: &mut Self::InputState) -> Result<Vec<Grain>, String> {
        Ok(Vec::new())
    }

    /// Start of a run, with `InputState` already installed when the spec survived
    /// from the previous run. Flush codec buffers here; do not drop `InputState`.
    fn start(&self, _input_state: &mut Self::InputState) {}

    /// [`crate::node_api::NodeObjects`] for this transform, if any. Default: none.
    fn objects(&self) -> Option<&dyn crate::node_api::NodeObjects> {
        None
    }
}

struct SisoState<I> {
    input_state: Option<I>,
    input_spec: Option<Spec>,
    dropped_early: u64,
}

impl<I> Default for SisoState<I> {
    fn default() -> Self {
        Self {
            input_state: None,
            input_spec: None,
            dropped_early: 0,
        }
    }
}

/// The [`InputHandler`] every adapter drives: the `InputState` and the
/// [`SisoNode`] hooks behind the generic ones.
struct SisoCore<F: SisoNode> {
    f: F,
    state: Mutex<SisoState<F::InputState>>,
}

impl<F: SisoNode> SisoCore<F> {
    fn new(f: F) -> Self {
        Self {
            f,
            state: Mutex::new(SisoState::default()),
        }
    }

    fn error(&self, phase: NodePhase, message: impl Into<String>) -> NodeError {
        NodeError::new(self.f.name(), phase, message)
    }

    fn start_run(&self) {
        let mut state = self.state.lock().unwrap();
        state.dropped_early = 0;
        if let Some(input_state) = state.input_state.as_mut() {
            self.f.start(input_state);
        }
    }
}

impl<F: SisoNode> InputHandler for SisoCore<F> {
    fn on_spec(&self, spec: Spec) -> Result<Option<Spec>, NodeError> {
        let mut state = self.state.lock().unwrap();
        if let Some(known) = &state.input_spec {
            if known.same_as(&spec) {
                log::debug!("{}: input spec re-delivered unchanged", self.f.name());
                return Ok(None);
            }
            log::info!("{}: input format changed, rebuilding", self.f.name());
        }
        let (input_state, out_spec) = self
            .f
            .on_spec(&spec)
            .map_err(|message| self.error(NodePhase::Spec, message))?;
        state.input_state = Some(input_state);
        state.input_spec = Some(spec);
        Ok(Some(out_spec))
    }

    fn on_buffer(&self, buf: Grain) -> Result<Vec<Grain>, NodeError> {
        let mut state = self.state.lock().unwrap();
        let Some(input_state) = state.input_state.as_mut() else {
            state.dropped_early += 1;
            return Ok(Vec::new());
        };
        self.f
            .process(input_state, buf)
            .map_err(|message| self.error(NodePhase::Process, message))
    }

    fn on_flush(&self) {
        if let Some(input_state) = self.state.lock().unwrap().input_state.as_mut() {
            self.f.flush(input_state);
        }
    }

    fn on_flush_stop(&self, resume_at: Option<Ts>) {
        if let Some(input_state) = self.state.lock().unwrap().input_state.as_mut() {
            self.f.flush_stop(input_state, resume_at);
        }
    }

    fn on_eof(&self) -> Result<EofAction, NodeError> {
        let mut state = self.state.lock().unwrap();
        let Some(input_state) = state.input_state.as_mut() else {
            return Ok(EofAction::Done);
        };
        let produced = self
            .f
            .on_eof(input_state)
            .map_err(|message| self.error(NodePhase::Process, message))?;
        if produced.is_empty() {
            Ok(EofAction::Done)
        } else {
            Ok(EofAction::Emit(produced))
        }
    }

    fn on_closed(&self) {
        let n = self.state.lock().unwrap().dropped_early;
        if n > 0 {
            log::info!(
                "{}: dropped {n} buffer(s) that arrived before the input spec",
                self.f.name()
            );
        }
    }
}

macro_rules! impl_siso_input_handler {
    ($ty:ident) => {
        impl<F: SisoNode> InputHandler for $ty<F> {
            fn on_spec(&self, spec: Spec) -> Result<Option<Spec>, NodeError> {
                self.core.on_spec(spec)
            }
            fn on_buffer(&self, buf: Grain) -> Result<Vec<Grain>, NodeError> {
                self.core.on_buffer(buf)
            }
            fn on_flush(&self) {
                self.core.on_flush()
            }
            fn on_flush_stop(&self, resume_at: Option<Ts>) {
                self.core.on_flush_stop(resume_at)
            }
            fn on_eof(&self) -> Result<EofAction, NodeError> {
                self.core.on_eof()
            }
            fn on_closed(&self) {
                self.core.on_closed()
            }
        }
    };
}

/// The blocking body: a [`SingleInput`] over [`SisoCore`], so the loop is the
/// wrapper's. Like every other node written this way, it fails on an unbound pad.
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

    pub fn inner(&self) -> &F {
        &self.0.core.f
    }
}

impl_siso_input_handler!(SisoBlocking);

impl<F: SisoNode> SingleInput for SisoBlocking<F> {
    fn io(&self) -> &BlockingIo {
        &self.io
    }

    fn pads(&self) -> NodePads {
        self.core.f.pads()
    }

    fn start(&self) {
        self.core.start_run();
    }

    fn objects(&self) -> Option<&dyn crate::node_api::NodeObjects> {
        self.core.f.objects()
    }
}

/// The cooperative body: `try_take` + `Polled::Idle`, no private thread.
///
/// Output backpressure stashes produced buffers on [`PollIo`] and waits
/// writable. That stash is the Poll cost of not keeping locals across a park.
/// Use this as a Direct consumer only when `on_spec` and `process` are
/// infallible, and opt in with `SisoNode::direct_consumer_is_infallible`; scheduled
/// Poll bodies propagate their errors to supervision, while the fused Direct
/// `Node::poll` contract cannot carry `NodeError`.
pub struct SisoPolling<F: SisoNode> {
    core: SisoCore<F>,
    io: PollIo,
}

/// Cooperative wrapper: [`SisoPolling`] as a `Node`.
pub type SisoPollAdapter<F> = Polling<SisoPolling<F>>;

impl<F: SisoNode> Polling<SisoPolling<F>> {
    pub fn new(f: F) -> Self {
        Polling(SisoPolling {
            io: PollIo::new(f.name()),
            core: SisoCore::new(f),
        })
    }

    pub fn inner(&self) -> &F {
        &self.0.core.f
    }
}

impl_siso_input_handler!(SisoPolling);

impl<F: SisoNode> PollInput for SisoPolling<F> {
    fn io(&self) -> &PollIo {
        &self.io
    }

    fn pads(&self) -> NodePads {
        self.core.f.pads()
    }

    fn direct_consumer_is_infallible(&self) -> bool {
        self.core.f.direct_consumer_is_infallible()
    }

    fn start(&self) {
        self.core.start_run();
    }

    fn objects(&self) -> Option<&dyn crate::node_api::NodeObjects> {
        self.core.f.objects()
    }
}

/// Async wrapper: one future, produced buffers stay on the stack across
/// `wait_writable`. Same event loop as Poll. Implements `Node` directly, since
/// the authoring API has no async node trait yet.
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

    fn pads(&self) -> NodePads {
        self.core.f.pads()
    }

    fn start(&self) {
        self.core.start_run();
    }

    fn set_object(&self, key: &str, value: &serde_json::Value) -> Result<(), String> {
        crate::node_api::objects::set_on(self.core.f.objects(), self.name(), key, value)
    }

    fn get_object(&self, key: &str) -> Result<serde_json::Value, String> {
        crate::node_api::objects::get_on(self.core.f.objects(), self.name(), key)
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
                let (buffers, then_eof) = match react(&self.core, Some(&out), item)? {
                    Reaction::Again => continue,
                    Reaction::Done => return Ok(()),
                    Reaction::Produced(buffers) => (buffers, false),
                    Reaction::ProducedThenEof(buffers) => (buffers, true),
                };
                for mut buf in buffers {
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
                if then_eof {
                    out.push_event(EdgeEvent::Eof);
                    return Ok(());
                }
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::task::{Context, Poll as TaskPoll, Waker};
    use std::time::Duration;

    use super::*;
    use crate::graph::BufferedEdge;
    use crate::graph::edge::{EdgeEvent, EdgeItem, Push, Wakeup};
    use crate::graph::error::NodePhase;
    use crate::graph::grain::Grain;
    use crate::graph::media::{AvpMediaType, AvpRational};
    use crate::graph::node::Processed;
    use crate::graph::node::{Node, NodeBody, Polled};
    use crate::graph::poll_ctx::NodePollContext;
    use crate::graph::spec::Spec;

    struct Identity {
        name: &'static str,
    }

    impl SisoNode for Identity {
        type InputState = ();
        fn name(&self) -> &str {
            self.name
        }
        fn on_spec(&self, spec: &Spec) -> Result<((), Spec), String> {
            Ok(((), spec.clone()))
        }
        fn process(&self, _input_state: &mut (), buf: Grain) -> Result<Vec<Grain>, String> {
            Ok(vec![buf])
        }
    }

    struct Objects {
        name: &'static str,
        hits: AtomicUsize,
    }

    impl SisoNode for Objects {
        type InputState = ();
        fn name(&self) -> &str {
            self.name
        }
        fn on_spec(&self, spec: &Spec) -> Result<((), Spec), String> {
            Ok(((), spec.clone()))
        }
        fn process(&self, _input_state: &mut (), buf: Grain) -> Result<Vec<Grain>, String> {
            Ok(vec![buf])
        }
        fn objects(&self) -> Option<&dyn crate::node_api::NodeObjects> {
            Some(self)
        }
    }

    impl crate::node_api::NodeObjects for Objects {
        fn set_object(&self, key: &str, value: &serde_json::Value) -> Result<(), String> {
            match key {
                "n" => {
                    self.hits
                        .store(value.as_u64().unwrap_or(0) as usize, Ordering::SeqCst);
                    Ok(())
                }
                other => Err(format!("{}: no `{other}`", self.name)),
            }
        }
        fn get_object(&self, key: &str) -> Result<serde_json::Value, String> {
            match key {
                "n" => Ok(serde_json::json!(self.hits.load(Ordering::SeqCst))),
                other => Err(format!("{}: no `{other}`", self.name)),
            }
        }
    }

    struct CountingSpec {
        name: &'static str,
        specs: Arc<AtomicUsize>,
    }

    impl SisoNode for CountingSpec {
        type InputState = ();
        fn name(&self) -> &str {
            self.name
        }
        fn on_spec(&self, spec: &Spec) -> Result<((), Spec), String> {
            self.specs.fetch_add(1, Ordering::SeqCst);
            Ok(((), spec.clone()))
        }
        fn process(&self, _input_state: &mut (), buf: Grain) -> Result<Vec<Grain>, String> {
            Ok(vec![buf])
        }
    }

    struct FailsOnSpec {
        name: &'static str,
    }

    impl SisoNode for FailsOnSpec {
        type InputState = ();

        fn name(&self) -> &str {
            self.name
        }

        fn on_spec(&self, _spec: &Spec) -> Result<((), Spec), String> {
            Err("unsupported input format".into())
        }

        fn process(&self, _input_state: &mut (), _buf: Grain) -> Result<Vec<Grain>, String> {
            unreachable!("a failed Spec must not install processing state")
        }
    }

    struct FailsOnBuffer;

    impl SisoNode for FailsOnBuffer {
        type InputState = ();

        fn name(&self) -> &str {
            "buffer_error"
        }

        fn on_spec(&self, spec: &Spec) -> Result<((), Spec), String> {
            Ok(((), spec.clone()))
        }

        fn process(&self, _input_state: &mut (), _buf: Grain) -> Result<Vec<Grain>, String> {
            Err("decoder rejected buffer".into())
        }
    }

    struct Dup;

    impl SisoNode for Dup {
        type InputState = ();
        fn name(&self) -> &str {
            "dup"
        }
        fn pads(&self) -> NodePads {
            NodePads::siso(AvpMediaType::VIDEO, AvpMediaType::VIDEO)
        }
        fn on_spec(&self, spec: &Spec) -> Result<((), Spec), String> {
            Ok(((), spec.clone()))
        }
        fn process(&self, _input_state: &mut (), buf: Grain) -> Result<Vec<Grain>, String> {
            Ok(vec![buf.clone(), buf])
        }
    }

    /// Holds each buffer until the next one (or EOF) so a 1:1 delay still
    /// drains the last grain from `on_eof`.
    struct Delay;

    impl SisoNode for Delay {
        type InputState = Option<Grain>;
        fn name(&self) -> &str {
            "delay"
        }
        fn on_spec(&self, spec: &Spec) -> Result<(Option<Grain>, Spec), String> {
            Ok((None, spec.clone()))
        }
        fn process(
            &self,
            input_state: &mut Option<Grain>,
            buf: Grain,
        ) -> Result<Vec<Grain>, String> {
            Ok(input_state.replace(buf).into_iter().collect())
        }
        fn on_eof(&self, input_state: &mut Option<Grain>) -> Result<Vec<Grain>, String> {
            Ok(input_state.take().into_iter().collect())
        }
    }

    fn video_spec() -> Spec {
        video_spec_width(8)
    }

    fn video_spec_width(width: i32) -> Spec {
        Spec::Video {
            width,
            height: 8,
            pix_fmt: 0,
            sw_pix_fmt: -1,
            frame_rate: AvpRational { num: 1, den: 1 },
            sar: AvpRational { num: 1, den: 1 },
            time_base: AvpRational { num: 1, den: 1000 },
        }
    }

    fn stub(pts: i64) -> Grain {
        crate::graph::grain::test_media(AvpMediaType::VIDEO, pts)
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
                pts.push(buf.ts().ticks());
            }
        }
        pts
    }

    fn drain_kinds(edge: &dyn Edge) -> Vec<&'static str> {
        let mut out = Vec::new();
        while let Some(item) = edge.try_take() {
            out.push(match item {
                EdgeItem::Buffer(_) => "buffer",
                EdgeItem::Event(EdgeEvent::Spec(_)) => "spec",
                EdgeItem::Event(EdgeEvent::FlushStart) => "flush-start",
                EdgeItem::Event(EdgeEvent::FlushStop { .. }) => "flush-stop",
                EdgeItem::Event(EdgeEvent::Drain) => "drain",
                EdgeItem::Event(EdgeEvent::Eof) => "eof",
            });
        }
        out
    }

    fn pump_until_idle(node: &impl Node, ctx: &mut NodePollContext) {
        for _ in 0..32 {
            match node.poll(ctx).unwrap() {
                Polled::Idle | Polled::Done => return,
                Polled::Again => {}
            }
        }
        panic!("poll helper did not become idle");
    }

    #[test]
    fn poll_idles_on_empty_input() {
        let node = SisoPollAdapter::new(Identity { name: "p" });
        let (_in, _out) = bind_pair(&node);
        let mut ctx = poll_ctx();
        assert_eq!(node.poll(&mut ctx).unwrap(), Polled::Idle);
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
    fn poll_adapter_forwards_object_keys() {
        let node = SisoPollAdapter::new(Objects {
            name: "o",
            hits: AtomicUsize::new(0),
        });
        node.set_object("n", &serde_json::json!(3)).unwrap();
        assert_eq!(node.get_object("n").unwrap(), serde_json::json!(3));
        assert_eq!(node.inner().hits.load(Ordering::SeqCst), 3);
        assert!(node.set_object("x", &serde_json::Value::Null).is_err());
    }

    #[test]
    fn blocking_adapter_forwards_object_keys() {
        let node = SisoAdapter::new(Objects {
            name: "o",
            hits: AtomicUsize::new(0),
        });
        node.set_object("n", &serde_json::json!(5)).unwrap();
        assert_eq!(node.get_object("n").unwrap(), serde_json::json!(5));
    }

    #[test]
    fn poll_adapter_without_objects_errors() {
        let node = SisoPollAdapter::new(Identity { name: "p" });
        let err = node
            .set_object("n", &serde_json::json!(1))
            .expect_err("Identity has no NodeObjects");
        assert!(err.contains("p") && err.contains("n"), "{err}");
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
        assert_eq!(node.poll(&mut ctx).unwrap(), Polled::Idle);
        assert!(ctx.needs_park());

        let _ = output.try_take();
        assert_eq!(node.poll(&mut ctx).unwrap(), Polled::Again);
        assert_eq!(take_bufs(&*output), vec![9]);
    }

    /// Production change that would make this fail: `SisoCore::on_spec` always
    /// constructing a new `InputState` and republishing, even when the input format
    /// is the one already installed.
    #[test]
    fn an_unchanged_spec_is_not_rebuilt_or_republished() {
        let specs = Arc::new(AtomicUsize::new(0));
        let node = SisoPollAdapter::new(CountingSpec {
            name: "c",
            specs: specs.clone(),
        });
        let (input, output) = bind_pair(&node);
        input.push_event(EdgeEvent::Spec(video_spec()));
        let mut ctx = poll_ctx();
        pump_until_idle(&node, &mut ctx);
        assert_eq!(specs.load(Ordering::SeqCst), 1);
        assert_eq!(drain_kinds(&*output), vec!["spec"]);

        input.push_event(EdgeEvent::Spec(video_spec()));
        pump_until_idle(&node, &mut ctx);
        assert_eq!(
            specs.load(Ordering::SeqCst),
            1,
            "on_spec must not run for an identical re-delivery"
        );
        assert!(
            drain_kinds(&*output).is_empty(),
            "an unchanged spec must not be republished"
        );
    }

    #[test]
    fn one_buffer_can_become_several() {
        let node = SisoPollAdapter::new(Dup);
        let (input, output) = bind_pair(&node);
        assert_eq!(
            node.pads(),
            NodePads::siso(AvpMediaType::VIDEO, AvpMediaType::VIDEO)
        );
        input.push_event(EdgeEvent::Spec(video_spec()));
        assert_eq!(input.push(stub(4)), Push::Accepted);
        let mut ctx = poll_ctx();
        pump_until_idle(&node, &mut ctx);
        assert_eq!(take_bufs(&*output), vec![4, 4]);
    }

    #[test]
    fn on_eof_emits_held_buffers_then_forwards_eof() {
        let node = SisoAdapter::new(Delay);
        let (input, output) = bind_pair(&node);
        node.start();
        input.push_event(EdgeEvent::Spec(video_spec()));
        assert_eq!(input.push(stub(1)), Push::Accepted);
        assert_eq!(input.push(stub(2)), Push::Accepted);
        input.push_event(EdgeEvent::Eof);

        assert_eq!(node.process().unwrap(), Processed::Again, "spec");
        assert_eq!(node.process().unwrap(), Processed::Again, "holds 1");
        assert_eq!(
            node.process().unwrap(),
            Processed::Again,
            "emits 1, holds 2"
        );
        assert_eq!(node.process().unwrap(), Processed::Done, "emits 2 then eof");
        let mut pts = Vec::new();
        let mut kinds = Vec::new();
        while let Some(item) = output.try_take() {
            match item {
                EdgeItem::Buffer(buf) => {
                    kinds.push("buffer");
                    pts.push(buf.ts().ticks());
                }
                EdgeItem::Event(EdgeEvent::Spec(_)) => kinds.push("spec"),
                EdgeItem::Event(EdgeEvent::Eof) => kinds.push("eof"),
                EdgeItem::Event(_) => kinds.push("other"),
            }
        }
        assert_eq!(kinds, vec!["spec", "buffer", "buffer", "eof"]);
        assert_eq!(pts, vec![1, 2]);
    }

    #[test]
    fn a_buffer_before_the_spec_is_dropped_not_an_error() {
        let node = SisoPollAdapter::new(Identity { name: "early" });
        let (input, output) = bind_pair(&node);
        assert_eq!(input.push(stub(1)), Push::Accepted);
        let mut ctx = poll_ctx();
        pump_until_idle(&node, &mut ctx);
        assert!(take_bufs(&*output).is_empty());
        input.push_event(EdgeEvent::Spec(video_spec()));
        assert_eq!(input.push(stub(2)), Push::Accepted);
        pump_until_idle(&node, &mut ctx);
        assert_eq!(take_bufs(&*output), vec![2]);
    }

    #[test]
    fn a_changed_spec_rebuilds() {
        let specs = Arc::new(AtomicUsize::new(0));
        let node = SisoPollAdapter::new(CountingSpec {
            name: "c",
            specs: specs.clone(),
        });
        let (input, output) = bind_pair(&node);
        input.push_event(EdgeEvent::Spec(video_spec()));
        let mut ctx = poll_ctx();
        pump_until_idle(&node, &mut ctx);
        assert_eq!(drain_kinds(&*output), vec!["spec"]);
        input.push_event(EdgeEvent::Spec(video_spec_width(16)));
        pump_until_idle(&node, &mut ctx);
        assert_eq!(specs.load(Ordering::SeqCst), 2);
        assert_eq!(drain_kinds(&*output), vec!["spec"]);
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
        assert_eq!(node.process().unwrap(), Processed::Again, "the Spec");
        (node, output)
    }

    fn take_one(edge: &dyn Edge) -> Option<i64> {
        match edge.try_take() {
            Some(EdgeItem::Buffer(buf)) => Some(buf.ts().ticks()),
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
            Processed::Again,
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

        assert_eq!(node.process().unwrap(), Processed::Done);
        interrupter.join().unwrap();
        assert_eq!(
            take_bufs(&*output),
            vec![1],
            "nothing was pushed past the park"
        );
    }
}
