//! Native `Node` contract. Execution strategy is `NodeBody`, taken once at
//! start.

use std::ffi::c_void;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use crate::graph::capability::AvpInterfaceId;
use crate::graph::edge::Edge;
use crate::graph::error::NodeError;
use crate::graph::pad::NodePads;
use crate::graph::poll_ctx::NodePollContext;
use crate::graph::spec::Spec;

/// Result of one [`Node::process`] (blocking body). There is no Idle: the
/// body waits inside `take(-1)` instead of yielding. Failure is the `Err` side
/// of the `Result` the method returns, not a variant here.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Blocked {
    Again,
    Done,
}

/// Result of one [`Node::poll`] (cooperative body). Idle is C++
/// `processWhenSignalled` / `sleepAndProcess` then return; Again is
/// `yieldAndProcess`. Failure is the `Err` side of the `Result`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tick {
    Again,
    Idle,
    Done,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum NodeKind {
    Blocking,
    Poll,
    Async,
}

impl NodeKind {
    pub fn is_blocking(self) -> bool {
        matches!(self, NodeKind::Blocking)
    }
}

pub type NodeFuture = Pin<Box<dyn Future<Output = Result<(), NodeError>> + Send>>;

pub enum NodeBody {
    Blocking(Box<dyn FnMut() -> Result<Blocked, NodeError> + Send>),
    Poll(Box<dyn FnMut(&mut NodePollContext) -> Result<Tick, NodeError> + Send>),
    Async(NodeFuture),
}

pub trait Node: Send + Sync + 'static {
    fn name(&self) -> &str;
    fn kind(&self) -> NodeKind {
        NodeKind::Blocking
    }
    /// Opts a Poll node into use as a Direct-edge consumer. Returning `true`
    /// promises that its fused `poll` path cannot fail, because Direct
    /// execution has no `NodeError` channel back to the supervisor.
    fn direct_consumer_is_infallible(&self) -> bool {
        false
    }
    fn pads(&self) -> NodePads {
        NodePads::default()
    }

    fn start(&self) {}
    /// Teardown, on the body's own thread once it has returned.
    fn stop(&self) {}
    /// Asynchronous "come back now" request, C++ `IInterruptible::interrupt`.
    ///
    /// Called by an executor from [`Executor::stop`](crate::exec::Executor::stop)
    /// — another thread, while the body may be parked inside libav or on an edge
    /// — so it must not block, and may arrive more than once or before the body
    /// starts. Interrupting the node's *source* edges is not enough for a node
    /// that has none, like `input`: only the node itself can be told to abandon
    /// a blocking read. [`Self::stop`] keeps its meaning; this is a request, that
    /// is the teardown.
    fn interrupt(&self) {}
    fn set_generation(&self, _generation: u64) {}
    fn on_spec(&self, spec: &Spec) -> Result<Spec, String> {
        Ok(spec.clone())
    }

    /// One step of a blocking body. `Err` fails the node: the executor reports
    /// it and the supervisor restarts the group, per its policy.
    fn process(&self) -> Result<Blocked, NodeError> {
        Ok(Blocked::Done)
    }
    /// One step of a cooperative body; `Err` fails the node the same way.
    ///
    /// One path cannot carry the error: a [`DirectEdge`](crate::graph::DirectEdge)
    /// runs its consumer's `poll` from inside the producer's `offer`, where there
    /// is no executor to report to. That is what
    /// [`Self::direct_consumer_is_infallible`] promises never happens; an `Err`
    /// there is logged and ends the fused run, nothing more.
    fn poll(&self, _ctx: &mut NodePollContext) -> Result<Tick, NodeError> {
        Ok(Tick::Done)
    }

    fn run_async(self: Arc<Self>) -> NodeFuture {
        Box::pin(async { Ok(()) })
    }

    fn query_interface(&self, _iface: AvpInterfaceId) -> Option<*const c_void> {
        None
    }

    /// A named knob the control layer can turn (`node.object.set`), C++
    /// `IInputsObjects::setObject`. Called from the control thread while the
    /// body runs, so an implementation must not take a lock the body holds
    /// across a blocking call. The default node has none.
    fn set_object(&self, key: &str, _value: &serde_json::Value) -> Result<(), String> {
        Err(format!("{} has no object `{key}` to set", self.name()))
    }

    /// A named value the control layer can read (`node.object.get`), C++
    /// `IReturnsObjects::getObject`. Same threading rule as [`Self::set_object`].
    fn get_object(&self, key: &str) -> Result<serde_json::Value, String> {
        Err(format!("{} has no object `{key}` to get", self.name()))
    }
    fn bind_source(&self, _name: &str, _edge: Arc<dyn Edge>) {}
    fn bind_sink(&self, _name: &str, _edge: Arc<dyn Edge>) {}

    /// The body the executor drives, taken once at start. The default calls
    /// [`Self::process`] / [`Self::poll`] / [`Self::run_async`] by
    /// [`Self::kind`]; override it only for a body that needs locals of its own
    /// across steps.
    fn take_body(self: Arc<Self>) -> NodeBody {
        match self.kind() {
            NodeKind::Blocking => NodeBody::Blocking(Box::new(move || self.process())),
            NodeKind::Poll => NodeBody::Poll(Box::new(move |ctx| self.poll(ctx))),
            NodeKind::Async => NodeBody::Async(self.run_async()),
        }
    }
}
