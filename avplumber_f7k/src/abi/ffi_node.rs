//! C node vtable wrapped as a native `Node`.

use std::ffi::c_void;

use crate::graph::capability::AvpInterfaceId;
use crate::graph::error::{NodeError, NodePhase};
use crate::graph::node::{Node, NodeKind, Polled, Processed};
use crate::graph::poll_ctx::NodePollContext;

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AvpNodeVtable {
    /// Callbacks receive the `self` pointer from `avp_node_set_impl`, not the
    /// stable `AvpNode` handle. Store that handle on the impl in the factory
    /// if start/process/poll need to bind edges or query the node by name.
    pub start: Option<extern "C" fn(*mut c_void)>,
    pub stop: Option<extern "C" fn(*mut c_void)>,
    pub destroy: Option<extern "C" fn(*mut c_void)>,
    pub process: Option<extern "C" fn(*mut c_void) -> i32>,
    pub poll: Option<extern "C" fn(*mut c_void) -> i32>,
    pub query_interface: Option<extern "C" fn(*mut c_void, u32) -> *const c_void>,
    pub direct_consumer_is_infallible: i32,
}

pub struct FfiNode {
    name: String,
    self_ptr: *mut c_void,
    vtable: &'static AvpNodeVtable,
}

unsafe impl Send for FfiNode {}
unsafe impl Sync for FfiNode {}

impl FfiNode {
    pub fn new(name: String, self_ptr: *mut c_void, vtable: &'static AvpNodeVtable) -> Self {
        Self {
            name,
            self_ptr,
            vtable,
        }
    }

    fn flow_blocked(&self, code: i32) -> Result<Processed, NodeError> {
        match code {
            0..=2 => Ok(Processed::Again),
            3 => Ok(Processed::Done),
            4 => Err(NodeError::new(
                &self.name,
                NodePhase::Process,
                "C node returned AVP_FLOW_ERROR",
            )),
            _ => Err(NodeError::new(
                &self.name,
                NodePhase::Process,
                format!("C node returned invalid AvpFlow code {code}"),
            )),
        }
    }
    fn flow_tick(&self, code: i32) -> Result<Polled, NodeError> {
        match code {
            0 | 1 => Ok(Polled::Again),
            2 => Ok(Polled::Idle),
            3 => Ok(Polled::Done),
            4 => Err(NodeError::new(
                &self.name,
                NodePhase::Poll,
                "C node returned AVP_FLOW_ERROR",
            )),
            _ => Err(NodeError::new(
                &self.name,
                NodePhase::Poll,
                format!("C node returned invalid AvpFlow code {code}"),
            )),
        }
    }
}

impl Drop for FfiNode {
    fn drop(&mut self) {
        // Deliberately touches nothing but the C side: the handle may itself be
        // mid-teardown here. Whoever abandons an unpublished generation clears
        // the pending slot that pointed at this state.
        if let Some(destroy) = self.vtable.destroy {
            destroy(self.self_ptr);
        }
    }
}

impl Node for FfiNode {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        if self.vtable.process.is_some() {
            NodeKind::Blocking
        } else {
            NodeKind::Poll
        }
    }
    fn direct_consumer_is_infallible(&self) -> bool {
        self.vtable.direct_consumer_is_infallible != 0
    }
    fn start(&self) {
        if let Some(f) = self.vtable.start {
            f(self.self_ptr);
        }
    }
    fn stop(&self) {
        if let Some(f) = self.vtable.stop {
            f(self.self_ptr);
        }
    }
    /// A C vtable is one of the two kinds at a time (`kind` picks by which
    /// entry point it defines). `AVP_FLOW_ERROR` becomes a `NodeError` here, so
    /// a C node fails its group the way a native `Err` does.
    fn process(&self) -> Result<Processed, NodeError> {
        match self.vtable.process {
            Some(f) => self.flow_blocked(f(self.self_ptr)),
            None => Ok(Processed::Done),
        }
    }
    fn poll(&self, _ctx: &mut NodePollContext) -> Result<Polled, NodeError> {
        match self.vtable.poll {
            Some(f) => self.flow_tick(f(self.self_ptr)),
            None => Ok(Polled::Done),
        }
    }
    fn query_interface(&self, iface: AvpInterfaceId) -> Option<*const c_void> {
        let f = self.vtable.query_interface?;
        let p = f(self.self_ptr, iface as u32);
        if p.is_null() { None } else { Some(p) }
    }
}

pub type VtableNode = FfiNode;
