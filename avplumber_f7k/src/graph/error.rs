use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodePhase {
    Start,
    Process,
    Poll,
    Async,
    Spec,
    Stop,
}

#[derive(Clone, Debug)]
pub struct NodeError {
    pub node: String,
    pub phase: NodePhase,
    pub message: String,
}

impl NodeError {
    pub fn new(node: impl Into<String>, phase: NodePhase, message: impl Into<String>) -> Self {
        Self {
            node: node.into(),
            phase,
            message: message.into(),
        }
    }
}

impl fmt::Display for NodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} during {:?}: {}", self.node, self.phase, self.message)
    }
}

impl std::error::Error for NodeError {}
