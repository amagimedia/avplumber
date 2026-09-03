//! Executor substrate. Set-level: runs a set of nodes.

pub mod async_rt;
pub mod blocking;

pub use async_rt::AsyncExecutor;
pub use blocking::BlockingExecutor;

use std::sync::Arc;

use crate::graph::{Edge, Node, NodeError};

pub type Generation = u64;
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
