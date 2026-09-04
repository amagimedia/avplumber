//! C graph/group construction: create nodes/edges/groups, start/stop groups.
//! Thin adapter over `Instance` methods in `core.rs`.

use std::ffi::CString;
use std::os::raw::c_char;

use crate::NodeRequest;
use crate::abi::EdgeCoupling;
use crate::abi::{AvpCore, AvpEdge, AvpGroup, AvpNode};
use crate::graph::generation_writer;

fn set_err(err_out: *mut *const c_char, msg: &str) {
    if err_out.is_null() {
        return;
    }
    let c = CString::new(msg).unwrap_or_else(|_| CString::new("<bad utf8>").unwrap());
    unsafe {
        *err_out = c.into_raw();
    }
}

/// Create a node by type name. Params are UTF-8 JSON. Returns NULL on error;
/// *err holds a caller-freed message.
#[unsafe(no_mangle)]
pub extern "C" fn avp_create_node(
    core: *mut AvpCore,
    type_name: *const c_char,
    instance_name: *const c_char,
    json_params: *const c_char,
    err: *mut *const c_char,
) -> *mut AvpNode {
    if core.is_null() {
        set_err(err, "null core");
        return std::ptr::null_mut();
    }
    let inst = unsafe { &mut *core };
    let type_name = crate::abi::node::cstr_to_string(type_name);
    let instance_name = crate::abi::node::cstr_to_string(instance_name);
    let params = crate::abi::node::cstr_to_string(json_params);
    match NodeRequest::from_json(&type_name, &instance_name, &params)
        .and_then(|request| inst.create_node(request))
    {
        Ok(native) => {
            let c_name = CString::new(instance_name.clone()).unwrap();
            let mut handle = inst
                .pending_node_handles
                .lock()
                .unwrap()
                .remove(&instance_name)
                .unwrap_or_else(|| {
                    Box::new(AvpNode {
                        core,
                        c_name,
                        node: native.node.clone(),
                        pads: native.pads.clone(),
                        self_ptr: std::ptr::null_mut(),
                        vtable: None,
                        pending_self_ptr: std::ptr::null_mut(),
                        pending_vtable: None,
                        producer_leases: Vec::new(),
                        pending_producer_leases: Vec::new(),
                        direct_reader_leases: Vec::new(),
                        pending_direct_reader_leases: Vec::new(),
                        building: false,
                        name: instance_name.clone(),
                        exec_ctx: native.exec_ctx.clone(),
                        restart: native.restart,
                        service_hint: native.service_hint.clone(),
                    })
                });
            handle.core = core;
            handle.node = native.node;
            handle.pads = native.pads;
            handle.exec_ctx = native.exec_ctx;
            handle.restart = native.restart;
            handle.service_hint = native.service_hint;
            let ptr = handle.as_ref() as *const AvpNode as *mut AvpNode;
            inst.node_handles
                .lock()
                .unwrap()
                .insert(instance_name.clone(), handle);
            ptr
        }
        Err(e) => {
            set_err(err, &e.to_string());
            std::ptr::null_mut()
        }
    }
}

/// Create + connect an edge. `coupling` NULL = core default (buffered).
#[unsafe(no_mangle)]
pub extern "C" fn avp_create_edge(
    core: *mut AvpCore,
    name: *const c_char,
    producer: *mut AvpNode,
    out_pad: *const c_char,
    consumer: *mut AvpNode,
    in_pad: *const c_char,
    coupling: *const EdgeCoupling,
) -> *mut AvpEdge {
    if core.is_null() || producer.is_null() || consumer.is_null() {
        return std::ptr::null_mut();
    }
    let inst = unsafe { &mut *core };
    let edge_name = crate::abi::node::cstr_to_string(name);
    let prod_name = unsafe { (*producer).name.clone() };
    let cons_name = unsafe { (*consumer).name.clone() };
    let out_pad = crate::abi::node::cstr_to_string(out_pad);
    let in_pad = crate::abi::node::cstr_to_string(in_pad);
    let coupling = unsafe { coupling.as_ref() }
        .copied()
        .unwrap_or_default()
        .to_native();
    let native = match inst.connect_edge(
        &edge_name, &prod_name, &out_pad, &cons_name, &in_pad, coupling,
    ) {
        Ok(edge) => edge,
        Err(_) => return std::ptr::null_mut(),
    };
    let logical = native.edge;
    inst.edge_handles.lock().unwrap().insert(
        edge_name.clone(),
        Box::new(AvpEdge {
            name: edge_name.clone(),
            edge: logical.clone(),
            media_vtables: inst.media_vtables.clone(),
        }),
    );
    let generation = inst
        .node(&prod_name)
        .map(|node| node.active_generation)
        .unwrap_or_else(|| logical.writer_generation());
    let mut lease = Box::new(AvpEdge {
        name: edge_name.clone(),
        edge: generation_writer(logical, generation),
        media_vtables: inst.media_vtables.clone(),
    });
    let ptr = lease.as_mut() as *mut AvpEdge;
    unsafe { &mut *producer }.producer_leases.push(lease);
    ptr
}

/// Create a supervisor group (ordered start/stop of a named node set).
#[unsafe(no_mangle)]
pub extern "C" fn avp_create_group(core: *mut AvpCore, name: *const c_char) -> *mut AvpGroup {
    if core.is_null() {
        return std::ptr::null_mut();
    }
    let inst = unsafe { &mut *core };
    let name = crate::abi::node::cstr_to_string(name);
    if inst.create_group(&name).is_err() {
        return std::ptr::null_mut();
    }
    let handle = Box::new(AvpGroup {
        core,
        name: name.clone(),
    });
    let ptr = handle.as_ref() as *const AvpGroup as *mut AvpGroup;
    inst.group_handles.lock().unwrap().insert(name, handle);
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_group_add(group: *mut AvpGroup, node: *mut AvpNode) {
    if group.is_null() || node.is_null() {
        log::error!("avp_group_add rejected null group or node");
        return;
    }
    let group = unsafe { &*group };
    let node = unsafe { &*node };
    if let Err(error) = unsafe { &*group.core }.add_group_member(&group.name, &node.name) {
        log::error!("avp_group_add rejected membership: {error}");
    }
}

/// Fallible group membership API. Returns 0 on success, -1 with a
/// caller-freed error on invalid handles, lifecycle, or policy membership.
#[unsafe(no_mangle)]
pub extern "C" fn avp_group_add_checked(
    group: *mut AvpGroup,
    node: *mut AvpNode,
    err: *mut *const c_char,
) -> i32 {
    if group.is_null() || node.is_null() {
        set_err(err, "null group or node");
        return -1;
    }
    let group = unsafe { &*group };
    let node = unsafe { &*node };
    match unsafe { &*group.core }.add_group_member(&group.name, &node.name) {
        Ok(()) => 0,
        Err(error) => {
            set_err(err, &error.to_string());
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_group_remove(group: *mut AvpGroup, node: *mut AvpNode) {
    if group.is_null() || node.is_null() {
        return;
    }
    let group = unsafe { &*group };
    let node = unsafe { &*node };
    let _ = unsafe { &*group.core }.remove_group_member(&group.name, &node.name);
}

/// 0 ok, -1 err. Idempotent on a started group.
#[unsafe(no_mangle)]
pub extern "C" fn avp_start_group(group: *mut AvpGroup, err: *mut *const c_char) -> i32 {
    if group.is_null() {
        set_err(err, "null group");
        return -1;
    }
    let group = unsafe { &*group };
    match unsafe { &*group.core }.start_group(&group.name) {
        Ok(()) => 0,
        Err(e) => {
            set_err(err, &e.to_string());
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_stop_group(group: *mut AvpGroup, err: *mut *const c_char) -> i32 {
    if group.is_null() {
        set_err(err, "null group");
        return -1;
    }
    let group = unsafe { &*group };
    match unsafe { &*group.core }.stop_group(&group.name) {
        Ok(()) => 0,
        Err(error) => {
            set_err(err, &error.to_string());
            -1
        }
    }
}

/// Requests a generation-fenced reconstruction of a running or failed group.
/// Returns 0 when accepted, -1 with a caller-freed error otherwise.
#[unsafe(no_mangle)]
pub extern "C" fn avp_restart_group(group: *mut AvpGroup, err: *mut *const c_char) -> i32 {
    if group.is_null() {
        set_err(err, "null group");
        return -1;
    }
    let group = unsafe { &*group };
    match unsafe { &*group.core }.restart_group(&group.name) {
        Ok(()) => 0,
        Err(error) => {
            set_err(err, &error.to_string());
            -1
        }
    }
}

/// Writes a caller-owned JSON status string to `out_status`.
#[unsafe(no_mangle)]
pub extern "C" fn avp_group_status(
    group: *mut AvpGroup,
    out_status: *mut *mut c_char,
    err: *mut *const c_char,
) -> i32 {
    if group.is_null() || out_status.is_null() {
        set_err(err, "null group or status output");
        return -1;
    }
    unsafe {
        *out_status = std::ptr::null_mut();
    }
    let group = unsafe { &*group };
    let result = unsafe { &*group.core }
        .group_status(&group.name)
        .and_then(|status| {
            serde_json::to_string(&status)
                .map_err(|error| crate::CoreError::Operation(error.to_string()))
        })
        .and_then(|json| {
            CString::new(json).map_err(|error| crate::CoreError::Operation(error.to_string()))
        });
    match result {
        Ok(status) => {
            unsafe {
                *out_status = status.into_raw();
            }
            0
        }
        Err(error) => {
            set_err(err, &error.to_string());
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_destroy_node(core: *mut AvpCore, node: *mut AvpNode) {
    if core.is_null() || node.is_null() {
        return;
    }
    let inst = unsafe { &mut *core };
    let name = unsafe { (*node).name.clone() };
    if inst.destroy_node(&name).is_err() {
        return;
    }
    inst.node_handles.lock().unwrap().remove(&name);
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_destroy_edge(core: *mut AvpCore, edge: *mut AvpEdge) {
    if core.is_null() || edge.is_null() {
        return;
    }
    let inst = unsafe { &mut *core };
    let name = unsafe { (*edge).name.clone() };
    let _ = inst.destroy_edge(&name);
    inst.edge_handles.lock().unwrap().remove(&name);
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_destroy_group(core: *mut AvpCore, group: *mut AvpGroup) {
    if core.is_null() || group.is_null() {
        return;
    }
    let inst = unsafe { &mut *core };
    let name = unsafe { (*group).name.clone() };
    let _ = inst.destroy_group(&name);
    inst.group_handles.lock().unwrap().remove(&name);
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_lookup_node(core: *mut AvpCore, name: *const c_char) -> *mut AvpNode {
    if core.is_null() {
        return std::ptr::null_mut();
    }
    let inst = unsafe { &*core };
    let name = crate::abi::node::cstr_to_string(name);
    inst.node_handles
        .lock()
        .unwrap()
        .get(&name)
        .map(|b| b.as_ref() as *const AvpNode as *mut AvpNode)
        .unwrap_or(std::ptr::null_mut())
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_lookup_group(core: *mut AvpCore, name: *const c_char) -> *mut AvpGroup {
    if core.is_null() {
        return std::ptr::null_mut();
    }
    let inst = unsafe { &*core };
    let name = crate::abi::node::cstr_to_string(name);
    inst.group_handles
        .lock()
        .unwrap()
        .get(&name)
        .map(|b| b.as_ref() as *const AvpGroup as *mut AvpGroup)
        .unwrap_or(std::ptr::null_mut())
}
