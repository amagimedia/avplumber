//! C ABI adapters.

pub mod control;
pub mod convert;
pub mod edge_ops;
pub mod ffi_node;
pub mod graph_mgmt;
pub mod node;
pub mod registry;
pub mod types;

pub use control::{avp_core_exec_command, avp_core_serve_tcp};
pub use edge_ops::{
    AvpPeek, avp_edge_current_spec, avp_edge_notify_readable, avp_edge_notify_writable,
    avp_edge_occupied, avp_edge_peek, avp_edge_peek_consume, avp_edge_peek_release, avp_edge_pop,
    avp_edge_push, avp_edge_push_event, avp_edge_take,
};
pub use ffi_node::{AvpNodeVtable, FfiNode, VtableNode};
pub use graph_mgmt::{
    avp_create_edge, avp_create_group, avp_create_node, avp_destroy_edge, avp_destroy_group,
    avp_destroy_node, avp_group_add, avp_group_add_checked, avp_group_remove, avp_group_status,
    avp_lookup_group, avp_lookup_node, avp_restart_group, avp_start_group, avp_stop_group,
};
pub use node::{
    avp_node_bind_sink, avp_node_bind_source, avp_node_impl, avp_node_name,
    avp_node_query_interface, avp_node_set_impl,
};
pub use registry::{
    avp_core_query_service, avp_register_media_type, avp_register_module_init,
    avp_register_node_factory, avp_shared_get, avp_shared_put,
};
pub use types::{AvpBuffer, AvpSpec, EdgeCoupling};

use std::sync::Arc;

use crate::Instance;
use crate::exec::ExecCtxId;
use crate::factory::RestartPolicy;
use crate::graph::{Edge, Node};

pub type AvpCore = Instance;

pub struct AvpGroup {
    pub(crate) core: *mut AvpCore,
    pub(crate) name: String,
}

pub struct AvpNode {
    pub(crate) core: *mut AvpCore,
    pub(crate) name: String,
    pub(crate) c_name: std::ffi::CString,
    pub(crate) node: Arc<dyn Node>,
    pub(crate) pads: crate::NodePads,
    pub(crate) self_ptr: *mut std::ffi::c_void,
    pub(crate) vtable: Option<&'static crate::abi::AvpNodeVtable>,
    pub(crate) pending_self_ptr: *mut std::ffi::c_void,
    pub(crate) pending_vtable: Option<&'static crate::abi::AvpNodeVtable>,
    /// Generation writer handles returned to C code remain valid until the
    /// stable node handle is destroyed, so late helper threads get fencing
    /// instead of a dangling pointer.
    #[allow(clippy::vec_box)] // Box addresses are the C ABI handle identity.
    pub(crate) producer_leases: Vec<Box<AvpEdge>>,
    #[allow(clippy::vec_box)] // Staged handles must also keep stable addresses.
    pub(crate) pending_producer_leases: Vec<Box<AvpEdge>>,
    /// Direct readers need generation identity too: an old C consumer may
    /// finish after the replacement generation has started.
    #[allow(clippy::vec_box)] // Box addresses are the C ABI handle identity.
    pub(crate) direct_reader_leases: Vec<Box<AvpEdge>>,
    #[allow(clippy::vec_box)] // Staged handles must also keep stable addresses.
    pub(crate) pending_direct_reader_leases: Vec<Box<AvpEdge>>,
    pub(crate) building: bool,
    pub(crate) exec_ctx: ExecCtxId,
    pub(crate) restart: RestartPolicy,
    pub(crate) service_hint: Option<String>,
}

pub struct AvpEdge {
    pub(crate) name: String,
    pub(crate) edge: Arc<dyn Edge>,
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_core_create() -> *mut AvpCore {
    Box::into_raw(Box::new(Instance::new()))
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_core_destroy(core: *mut AvpCore) {
    if core.is_null() {
        return;
    }
    unsafe {
        let instance = Box::from_raw(core);
        instance.shutdown();
        drop(instance);
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_string_free(s: *mut std::os::raw::c_char) {
    if s.is_null() {
        return;
    }
    unsafe {
        drop(std::ffi::CString::from_raw(s));
    }
}

pub(crate) struct PlaceholderNode;

impl crate::graph::Node for PlaceholderNode {
    fn name(&self) -> &str {
        "<placeholder>"
    }
    fn process(&self) -> crate::graph::Blocked {
        crate::graph::Blocked::Done
    }
}
