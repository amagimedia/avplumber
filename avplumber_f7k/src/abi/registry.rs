//! C factories, media vtables, service lookup, and the instance-shared object
//! map. The smoke test registers Rust factories via `register_factory` instead.

use std::ffi::c_void;
use std::os::raw::c_char;
use std::sync::{Arc, Mutex};

use crate::abi::{AvpCore, AvpNode};
use crate::factory::BuiltNode;
use crate::graph::AvpServiceId;
use crate::graph::media::{AvpMediaType, AvpMediaVtable};
/// C factory: `(AvpCore*, AvpNode* node, const char* json_params) -> AvpNode*`.
/// Creates the impl, calls `avp_node_set_impl(node, ...)`, returns `node`.
pub type AvpNodeFactoryFn =
    extern "C" fn(*mut AvpCore, *mut AvpNode, *const c_char) -> *mut AvpNode;

/// Register a C factory under a type name. Rust nodes use `register_factory`
/// or `register_spec`; this wraps a C function pointer for C++ nodes.
#[unsafe(no_mangle)]
pub extern "C" fn avp_register_node_factory(
    core: *mut AvpCore,
    type_name: *const c_char,
    c_fn: AvpNodeFactoryFn,
) {
    // SAFETY: `core` is the Instance the caller is registering on. Every field
    // this touches sits behind a mutex, so a shared borrow is enough.
    let inst = unsafe { &*core };
    let type_name = crate::abi::node::cstr_to_string(type_name);
    inst.factories
        .lock()
        .unwrap()
        // Do not capture `core`. Module init runs inside `Instance::new` before
        // that wrapper is returned or boxed, so a pointer taken then dangles.
        // `BuildCtx::instance` is the live Instance at `node.add`.
        .register_built(&type_name, move |inst_name, params, ctx| {
            let params_c = std::ffi::CString::new(params).map_err(|e| e.to_string())?;
            let core = ctx.instance as *const crate::Instance as *mut AvpCore;
            let existing = unsafe { &*core }
                .node_handles
                .lock()
                .unwrap()
                .get(inst_name)
                .map(|handle| handle.as_ref() as *const AvpNode as *mut AvpNode);
            let mut new_handle = if existing.is_none() {
                Some(Box::new(AvpNode {
                    core,
                    name: inst_name.to_string(),
                    c_name: std::ffi::CString::new(inst_name).map_err(|e| e.to_string())?,
                    node: Arc::new(crate::abi::PlaceholderNode),
                    pads: crate::NodePads::default(),
                    self_ptr: std::ptr::null_mut(),
                    vtable: None,
                    pending_self_ptr: std::ptr::null_mut(),
                    pending_vtable: None,
                    producer_leases: Vec::new(),
                    pending_producer_leases: Vec::new(),
                    direct_reader_leases: Vec::new(),
                    pending_direct_reader_leases: Vec::new(),
                    building: false,
                    exec_ctx: crate::exec::ExecCtxId::Blocking,
                    restart: crate::factory::RestartPolicy::Off,
                    service_hint: None,
                }))
            } else {
                None
            };
            let node_ptr = existing.unwrap_or_else(|| {
                new_handle.as_mut().expect("new handle allocated").as_mut() as *mut AvpNode
            });
            unsafe {
                (*node_ptr).pads = crate::NodePads::default();
                (*node_ptr).pending_self_ptr = std::ptr::null_mut();
                (*node_ptr).pending_vtable = None;
                (*node_ptr).pending_producer_leases.clear();
                (*node_ptr).pending_direct_reader_leases.clear();
                (*node_ptr).building = true;
            }
            let ret = c_fn(core, node_ptr, params_c.as_ptr());
            unsafe {
                (*node_ptr).building = false;
            }
            let pending_vtable = unsafe { (*node_ptr).pending_vtable };
            let pending_self_ptr = unsafe { (*node_ptr).pending_self_ptr };
            if ret.is_null() || ret != node_ptr || pending_vtable.is_none() {
                if let Some(vtable) = pending_vtable {
                    drop(crate::abi::FfiNode::new(
                        inst_name.to_string(),
                        pending_self_ptr,
                        vtable,
                        crate::NodePads::default(),
                    ));
                }
                unsafe {
                    (*node_ptr).pending_self_ptr = std::ptr::null_mut();
                    (*node_ptr).pending_vtable = None;
                    (*node_ptr).pending_producer_leases.clear();
                    (*node_ptr).pending_direct_reader_leases.clear();
                }
                return Err(if ret.is_null() {
                    "node factory returned null".to_string()
                } else if ret != node_ptr {
                    "node factory returned a different AvpNode handle".to_string()
                } else {
                    "node factory did not install a vtable".to_string()
                });
            }
            let pads = unsafe { (*node_ptr).pads.clone() };
            let native: Arc<dyn crate::Node> = Arc::new(crate::abi::FfiNode::new(
                inst_name.to_string(),
                pending_self_ptr,
                pending_vtable.expect("checked above"),
                pads,
            ));
            if let Some(mut handle) = new_handle {
                handle.self_ptr = pending_self_ptr;
                handle.vtable = pending_vtable;
                handle.pending_self_ptr = std::ptr::null_mut();
                handle.pending_vtable = None;
                // Leases stay staged: `create_node` publishes them once it has
                // decided this construction is the live generation.
                handle.node = native.clone();
                unsafe { &*core }
                    .pending_node_handles
                    .lock()
                    .unwrap()
                    .insert(inst_name.to_string(), handle);
            }
            Ok(BuiltNode::from_node(native))
        });
}

/// Process-global, const-initialized, so a C++ static constructor can append
/// before `main`. Replayed into every `Instance` rather than consumed: each
/// core needs the factories, and a later `Instance::new` must not find the
/// queue empty.
static MODULE_INITS: Mutex<Vec<extern "C" fn(*mut AvpCore)>> = Mutex::new(Vec::new());

/// Queue a registrar. `Instance::new` replays the queue. The callback must
/// register factories only; the `AvpCore` it receives does not outlive the call.
#[unsafe(no_mangle)]
pub extern "C" fn avp_register_module_init(init: extern "C" fn(*mut AvpCore)) {
    let mut inits = MODULE_INITS.lock().unwrap();
    if !inits.contains(&init) {
        inits.push(init);
    }
}

/// # Safety
/// `core` must point at a live `Instance` for the duration of the call and
/// must not be stored by the registrars.
pub(crate) unsafe fn replay_module_inits(core: *mut AvpCore) {
    let inits = MODULE_INITS.lock().unwrap().clone();
    for init in inits {
        init(core);
    }
}

/// Register retain/release for an opaque C++-owned media type (EGL, metadata).
#[unsafe(no_mangle)]
pub extern "C" fn avp_register_media_type(
    core: *mut AvpCore,
    media: AvpMediaType,
    vtable: *const AvpMediaVtable,
) {
    if vtable.is_null() {
        return;
    }
    let inst = unsafe { &mut *core };
    let v = unsafe { *vtable };
    inst.media_vtables.lock().unwrap().insert(media, v);
}

/// Look up a core service by ID. Returns the const service vtable, or null
/// if this build did not register it.
#[unsafe(no_mangle)]
pub extern "C" fn avp_core_query_service(core: *mut AvpCore, service_id: u32) -> *const c_void {
    let inst = unsafe { &*core };
    let Some(id) = AvpServiceId::from_raw(service_id) else {
        return std::ptr::null();
    };
    inst.services.query(id)
}

/// Instance-wide string-keyed objects (legacy C++ sharing, not typed services).
#[unsafe(no_mangle)]
pub extern "C" fn avp_shared_get(
    core: *mut AvpCore,
    type_key: *const c_char,
    name: *const c_char,
) -> *mut c_void {
    let inst = unsafe { &*core };
    let tk = crate::abi::node::cstr_to_string(type_key);
    let n = crate::abi::node::cstr_to_string(name);
    inst.shared
        .lock()
        .unwrap()
        .get(&(tk, n))
        .map(|e| e.obj)
        .unwrap_or(std::ptr::null_mut())
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_shared_put(
    core: *mut AvpCore,
    type_key: *const c_char,
    name: *const c_char,
    obj: *mut c_void,
    vtable: *const AvpMediaVtable,
) {
    let inst = unsafe { &*core };
    let tk = crate::abi::node::cstr_to_string(type_key);
    let n = crate::abi::node::cstr_to_string(name);
    let entry = crate::SharedEntry {
        obj,
        vtable: if vtable.is_null() {
            None
        } else {
            Some(unsafe { *vtable })
        },
    };
    inst.shared.lock().unwrap().insert((tk, n), entry);
}
