//! Executor substrate. Set-level: runs a set of nodes.

pub mod async_rt;
pub mod blocking;

pub use async_rt::AsyncExecutor;
pub use blocking::BlockingExecutor;

use std::sync::Arc;

use crate::graph::{Edge, Node, NodeError};

/// Id of one start of a node set. Every node in the set carries the same value.
///
/// The supervisor increments it on each group start or restart and passes it
/// through [`Executor::configure_run`] before [`Executor::start`]. An executor
/// does not interpret it: it stamps the value onto every [`NodeOutcome`] it
/// reports. That is what lets the supervisor ignore a worker still unwinding
/// from a previous run. The same id fences the edge leases
/// ([`crate::graph::generation_writer`]), which every node — native or C —
/// holds instead of the bare edge, so the previous generation cannot take or
/// push on the next one's edges.
pub type Generation = u64;

/// Invoked once when a node's body ends, from the worker that ran it.
///
/// Concurrent calls (different nodes finishing together) are expected. The
/// executor does not interpret [`NodeOutcome`]; it only delivers it. The
/// supervisor's reporter posts to the group manager, which is how a fault
/// becomes a restart. Without [`Executor::configure_run`], outcomes only hit
/// a debug log.
pub type OutcomeReporter = Arc<dyn Fn(NodeOutcome) + Send + Sync>;

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub enum ExecCtxId {
    Blocking,
    EventLoop { name: String },
    TickSource { name: String },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExecutorState {
    Created,
    Running,
    Stopping,
    Stopped,
}

#[derive(Debug, Clone)]
pub enum NodeOutcome {
    Completed {
        name: String,
        generation: Generation,
    },
    Failed {
        name: String,
        generation: Generation,
        err: NodeError,
    },
    Panicked {
        name: String,
        generation: Generation,
        message: String,
    },
    Cancelled {
        name: String,
        generation: Generation,
    },
}

impl NodeOutcome {
    pub fn name(&self) -> &str {
        match self {
            Self::Completed { name, .. }
            | Self::Failed { name, .. }
            | Self::Panicked { name, .. }
            | Self::Cancelled { name, .. } => name,
        }
    }

    pub fn generation(&self) -> Generation {
        match self {
            Self::Completed { generation, .. }
            | Self::Failed { generation, .. }
            | Self::Panicked { generation, .. }
            | Self::Cancelled { generation, .. } => *generation,
        }
    }

    pub fn is_fault(&self) -> bool {
        matches!(self, Self::Failed { .. } | Self::Panicked { .. })
    }
}

pub trait Executor: Send + Sync {
    fn add_node(&self, node: Arc<dyn Node>, sources: Vec<Arc<dyn Edge>>, sinks: Vec<Arc<dyn Edge>>);
    fn remove_node(&self, name: &str);
    fn configure_run(&self, _generation: Generation, _reporter: OutcomeReporter) {}
    fn start(&self) -> Result<(), String>;
    fn stop(&self);
    fn join(&self);
    fn tick(&self) {}
    fn state(&self) -> ExecutorState;
}
