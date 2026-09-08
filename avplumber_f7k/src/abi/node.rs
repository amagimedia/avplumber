//! C node handle: attach impl/vtable, bind named edges, query capabilities.

use std::ffi::c_void;
use std::os::raw::c_char;

use crate::AvpInterfaceId;
use crate::AvpMediaType;
use crate::PadDirection;
use crate::abi::{AvpEdge, AvpNode, AvpNodeVtable, FfiNode};
use crate::graph::{generation_reader, generation_writer};

/// Attach the C++/Rust object + vtable to the AvpNode. For a C-vtable node this
/// wraps `(self_ptr, vtable)` into a `VtableNode` stored in `node.node`.
#[unsafe(no_mangle)]
pub extern "C" fn avp_node_set_impl(
    node: *mut AvpNode,
    self_ptr: *mut c_void,
    vtable: *const AvpNodeVtable,
) {
    let node = unsafe { &mut *node };
    if node.building {
        node.pending_self_ptr = self_ptr;
        node.pending_vtable = if vtable.is_null() {
            None
        } else {
            // SAFETY: node vtables have static storage by ABI contract.
            Some(unsafe { &*vtable })
        };
        return;
    }
    if vtable.is_null() {
        node.vtable = None;
        return;
    }
    // SAFETY: the vtable is a C-side static (or &'static lifetime by contract).
    let vtable_ref: &'static AvpNodeVtable = unsafe { &*vtable };
    node.vtable = Some(vtable_ref);
    let vt = FfiNode::new(
        node.name.clone(),
        node as *mut _ as *mut c_void,
        self_ptr,
        vtable_ref,
    );
    let native: std::sync::Arc<dyn crate::Node> = std::sync::Arc::new(vt);
    let core = node.core;
    let name = node.name.clone();
    if core.is_null() {
        node.self_ptr = self_ptr;
        node.vtable = Some(vtable_ref);
        node.node = native;
    } else {
        // End the direct handle borrow before the core reacquires the handle
        // map in graph -> nodes -> handles order.
        let _ = node;
        unsafe { &*core }.replace_node_impl(&name, native, self_ptr, vtable_ref);
    }
}

/// Returns the C++ object pointer set via `avp_node_set_impl`, or null for a
/// pure-Rust node.
#[unsafe(no_mangle)]
pub extern "C" fn avp_node_impl(node: *mut AvpNode) -> *mut c_void {
    if let Some(self_ptr) = crate::abi::ffi_node::callback_impl(node.cast()) {
        return self_ptr;
    }
    let node = unsafe { &*node };
    if crate::abi::ffi_node::is_factory_handle(node as *const _ as *mut c_void) {
        node.pending_self_ptr
    } else {
        node.self_ptr
    }
}

/// Borrowed name (valid while the AvpNode handle lives; caller must NOT free).
#[unsafe(no_mangle)]
pub extern "C" fn avp_node_name(node: *mut AvpNode) -> *const c_char {
    unsafe { (*node).c_name.as_ptr() }
}

/// Capability query on this node only — no graph walk. Returns the
/// per-interface vtable pointer, or null.
#[unsafe(no_mangle)]
pub extern "C" fn avp_node_query_interface(node: *mut AvpNode, iface_id: u32) -> *const c_void {
    let node = unsafe { &*node };
    let iface = match AvpInterfaceId::from_raw(iface_id) {
        Some(id) => id,
        None => return std::ptr::null(),
    };
    node.node.query_interface(iface).unwrap_or(std::ptr::null())
}

/// Bind a named edge to this node as a source (consumer side). Returns the
/// endpoint handle. The matching `avp_node_bind_sink` on the producer (same
/// edge name) completes the link; `avp_create_edge` does both in one call.
#[unsafe(no_mangle)]
pub extern "C" fn avp_node_bind_source(
    node: *mut AvpNode,
    edge_name: *const c_char,
    _media: AvpMediaType,
    capacity: usize,
) -> *mut AvpEdge {
    bind_endpoint(node, edge_name, capacity, PadDirection::Input)
}

/// Bind a named edge to this node as a sink (producer side).
#[unsafe(no_mangle)]
pub extern "C" fn avp_node_bind_sink(
    node: *mut AvpNode,
    edge_name: *const c_char,
    _media: AvpMediaType,
    capacity: usize,
) -> *mut AvpEdge {
    bind_endpoint(node, edge_name, capacity, PadDirection::Output)
}

fn bind_endpoint(
    node: *mut AvpNode,
    edge_name: *const c_char,
    capacity: usize,
    direction: PadDirection,
) -> *mut AvpEdge {
    if node.is_null() {
        return std::ptr::null_mut();
    }
    let node = unsafe { &mut *node };
    if node.core.is_null() {
        return std::ptr::null_mut();
    }
    let core = unsafe { &*node.core };
    let name = cstr_to_string(edge_name);
    if capacity != 0 {
        core.plan_capacity(&name, capacity);
    }
    let declared = match direction {
        PadDirection::Input => &node.pads.sources,
        PadDirection::Output => &node.pads.sinks,
    };
    let pad = if declared.len() == 1 {
        declared[0].name.as_str()
    } else if declared.iter().any(|pad| pad.name == name) || declared.is_empty() {
        name.as_str()
    } else {
        return std::ptr::null_mut();
    };
    let edge = match core.bind_edge(&node.name, pad, direction, &name) {
        Ok(edge) => edge.edge,
        Err(_) => return std::ptr::null_mut(),
    };
    let c_node = node.vtable.is_some() || node.pending_vtable.is_some();
    if c_node && (direction == PadDirection::Output || edge.is_direct()) {
        let generation = crate::factory::build_generation()
            .or_else(|| {
                core.node(&node.name)
                    .map(|instance| instance.active_generation)
            })
            .unwrap_or_else(|| edge.writer_generation())
            .max(edge.writer_generation());
        let lease_edge = match direction {
            PadDirection::Input => generation_reader(edge, generation),
            PadDirection::Output => generation_writer(edge, generation),
        };
        let mut lease = Box::new(AvpEdge {
            name,
            edge: lease_edge,
            media_vtables: core.media_vtables.clone(),
        });
        let pointer = lease.as_mut() as *mut AvpEdge;
        match (direction, node.building) {
            (PadDirection::Input, true) => node.pending_direct_reader_leases.push(lease),
            (PadDirection::Input, false) => node.direct_reader_leases.push(lease),
            (PadDirection::Output, true) => node.pending_producer_leases.push(lease),
            (PadDirection::Output, false) => node.producer_leases.push(lease),
        }
        return pointer;
    }
    let mut handles = core.edge_handles.lock().unwrap();
    let handle = handles.entry(name.clone()).or_insert_with(|| {
        Box::new(AvpEdge {
            name,
            edge: edge.clone(),
            media_vtables: core.media_vtables.clone(),
        })
    });
    handle.as_ref() as *const AvpEdge as *mut AvpEdge
}

pub(crate) fn cstr_to_string(s: *const c_char) -> String {
    if s.is_null() {
        return String::new();
    }
    unsafe { std::ffi::CStr::from_ptr(s).to_string_lossy().into_owned() }
}
