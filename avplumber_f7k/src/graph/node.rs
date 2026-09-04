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

/// Result of one `Node::process` (blocking body). There is no Idle: the
/// body waits inside `take(-1)` instead of yielding.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Blocked {
    Again,
    Done,
}

/// Result of one `Node::poll` (cooperative body). Idle is C++
/// `processWhenSignalled` / `sleepAndProcess` then return; Again is
/// `yieldAndProcess`.
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
    fn direct_poll_is_infallible(&self) -> bool {
        false
    }
    fn pads(&self) -> NodePads {
        NodePads::default()
    }

    fn start(&self) {}
    fn stop(&self) {}
    fn set_generation(&self, _generation: u64) {}
    fn on_spec(&self, spec: &Spec) -> Result<Spec, String> {
        Ok(spec.clone())
    }

    fn process(&self) -> Blocked {
        Blocked::Done
    }
    fn poll(&self, _ctx: &mut NodePollContext) -> Tick {
        Tick::Done
    }

    fn run_async(self: Arc<Self>) -> NodeFuture {
        Box::pin(async { Ok(()) })
    }

    fn query_interface(&self, _iface: AvpInterfaceId) -> Option<*const c_void> {
        None
    }
    fn bind_source(&self, _name: &str, _edge: Arc<dyn Edge>) {}
    fn bind_sink(&self, _name: &str, _edge: Arc<dyn Edge>) {}

    fn take_body(self: Arc<Self>) -> NodeBody {
        match self.kind() {
            NodeKind::Blocking => NodeBody::Blocking(Box::new(move || Ok(self.process()))),
            NodeKind::Poll => NodeBody::Poll(Box::new(move |ctx| Ok(self.poll(ctx)))),
            NodeKind::Async => NodeBody::Async(self.run_async()),
        }
    }
}
