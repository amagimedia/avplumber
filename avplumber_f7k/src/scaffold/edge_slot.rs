//! One bound edge, for the nodes with a single input or a single output.

use std::sync::{Arc, Mutex};

use crate::graph::edge::Edge;
use crate::graph::error::{NodeError, NodePhase};

/// One bound edge, for the nodes with a single input or a single output.
///
/// A `Mutex<Option<_>>` rather than a `OnceLock`: `bind_source`/`bind_sink` run
/// again whenever a script rebinds a pad or a reconstruction re-establishes the
/// links, and the last binding is the live one.
#[derive(Default)]
pub struct EdgeSlot {
    edge: Mutex<Option<Arc<dyn Edge>>>,
}

impl EdgeSlot {
    pub fn bind(&self, edge: Arc<dyn Edge>) {
        *self.edge.lock().unwrap() = Some(edge);
    }

    pub fn get(&self) -> Option<Arc<dyn Edge>> {
        self.edge.lock().unwrap().clone()
    }

    /// The bound edge, or the error every media node would otherwise spell out
    /// itself. An unbound pad is a script mistake, so it fails the group.
    pub fn require(
        &self,
        node: &str,
        phase: NodePhase,
        what: &str,
    ) -> Result<Arc<dyn Edge>, NodeError> {
        self.get()
            .ok_or_else(|| NodeError::new(node, phase, format!("{what} edge is not bound")))
    }
}
