//! Slot the nodes crate fills when it links C++ nodes.
//!
//! The framework never names those nodes and never links their objects. The
//! nodes crate installs the function pointers after the objects are in the
//! binary; factory creation, `hwaccel.init`, and instance drop call through
//! the slot. An empty slot means this process has no C++ nodes.

use std::ffi::{c_char, c_void};
use std::sync::{Arc, Mutex};

use crate::Instance;

pub struct CppBridge {
    pub bind: unsafe extern "C" fn(*mut c_void, *mut c_void),
    pub on_drop: unsafe extern "C" fn(*mut c_void),
    pub cuda_init: Option<unsafe extern "C" fn()>,
    pub hwaccel_put: Option<
        unsafe extern "C" fn(*mut c_void, *const c_char, *mut c_void) -> i32,
    >,
}

static BRIDGE: Mutex<Option<CppBridge>> = Mutex::new(None);

pub fn install(bridge: CppBridge) {
    *BRIDGE.lock().unwrap() = Some(bridge);
}

/// The C++ shared-object map keys an `InstanceData` by the inner allocation.
/// Mixer commands call this before `avp_cpp_instance`, so the two lookups
/// agree.
#[unsafe(no_mangle)]
pub extern "C" fn avp_cpp_ensure_instance(core: *mut Instance) {
    if !core.is_null() {
        bind_instance(unsafe { &*core });
    }
}

pub(crate) fn bind_instance(inst: &Instance) {
    let Some(bind) = BRIDGE.lock().unwrap().as_ref().map(|bridge| bridge.bind) else {
        return;
    };
    unsafe {
        bind(
            inst as *const Instance as *mut c_void,
            Arc::as_ptr(&inst.inner) as *mut c_void,
        );
    }
}

pub(crate) fn drop_instance(inner: *mut c_void) {
    let Some(on_drop) = BRIDGE.lock().unwrap().as_ref().map(|bridge| bridge.on_drop) else {
        return;
    };
    unsafe { on_drop(inner) };
}

pub(crate) fn publishes_hwaccel() -> bool {
    BRIDGE.lock().unwrap().as_ref().is_some_and(|bridge| {
        bridge.cuda_init.is_some() && bridge.hwaccel_put.is_some()
    })
}

pub(crate) fn publish_hwaccel(
    inst: &Instance,
    name: &str,
    device: *mut c_void,
) -> Result<(), String> {
    let (init, put) = {
        let guard = BRIDGE.lock().unwrap();
        let Some(bridge) = guard.as_ref() else {
            return Ok(());
        };
        match (bridge.cuda_init, bridge.hwaccel_put) {
            (Some(init), Some(put)) => (init, put),
            _ => return Ok(()),
        }
    };
    bind_instance(inst);
    let name_c = std::ffi::CString::new(name).map_err(|_| "the device name contains a NUL")?;
    unsafe {
        init();
        if put(inst as *const Instance as *mut c_void, name_c.as_ptr(), device) != 0 {
            return Err(format!(
                "failed to publish hwaccel `{name}` to the C++ nodes"
            ));
        }
    }
    Ok(())
}
