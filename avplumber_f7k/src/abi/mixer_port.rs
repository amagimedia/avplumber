//! Named node/group operations used by the C++ mixer graph on a Rust instance.

use std::ffi::{CStr, CString};
use std::os::raw::c_char;

use crate::abi::AvpCore;

fn set_err(err_out: *mut *mut c_char, msg: &str) {
    if err_out.is_null() {
        return;
    }
    let c = CString::new(msg).unwrap_or_else(|_| CString::new("<bad utf8>").unwrap());
    unsafe {
        *err_out = c.into_raw();
    }
}

fn cstr<'a>(ptr: *const c_char) -> &'a str {
    if ptr.is_null() {
        return "";
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().unwrap_or("")
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_node_exists(core: *mut AvpCore, name: *const c_char) -> i32 {
    if core.is_null() {
        return 0;
    }
    let inst = unsafe { &*core };
    i32::from(inst.node(cstr(name)).is_some())
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_node_object_set(
    core: *mut AvpCore,
    node: *const c_char,
    key: *const c_char,
    json: *const c_char,
    err: *mut *mut c_char,
) -> i32 {
    if core.is_null() {
        set_err(err, "null core");
        return -1;
    }
    let inst = unsafe { &*core };
    let node_name = cstr(node);
    let Some(found) = inst.node(node_name) else {
        set_err(err, &format!("Mixer node no longer exists: {node_name}"));
        return -1;
    };
    let value: serde_json::Value = match serde_json::from_str(cstr(json)) {
        Ok(value) => value,
        Err(error) => {
            set_err(err, &error.to_string());
            return -1;
        }
    };
    match found.node.set_object(cstr(key), &value) {
        Ok(()) => 0,
        Err(error) => {
            set_err(err, &error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_node_object_get(
    core: *mut AvpCore,
    node: *const c_char,
    key: *const c_char,
    json_out: *mut *mut c_char,
    err: *mut *mut c_char,
) -> i32 {
    if core.is_null() {
        set_err(err, "null core");
        return -1;
    }
    let inst = unsafe { &*core };
    let node_name = cstr(node);
    let Some(found) = inst.node(node_name) else {
        set_err(err, &format!("Mixer node no longer exists: {node_name}"));
        return -1;
    };
    match found.node.get_object(cstr(key)) {
        Ok(value) => {
            let text = match serde_json::to_string(&value) {
                Ok(text) => text,
                Err(error) => {
                    set_err(err, &error.to_string());
                    return -1;
                }
            };
            if !json_out.is_null() {
                match CString::new(text) {
                    Ok(c) => unsafe { *json_out = c.into_raw() },
                    Err(error) => {
                        set_err(err, &error.to_string());
                        return -1;
                    }
                }
            }
            0
        }
        Err(error) => {
            set_err(err, &error);
            -1
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_group_start_named(
    core: *mut AvpCore,
    name: *const c_char,
    err: *mut *mut c_char,
) -> i32 {
    group_op(core, name, err, true)
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_group_stop_named(
    core: *mut AvpCore,
    name: *const c_char,
    err: *mut *mut c_char,
) -> i32 {
    group_op(core, name, err, false)
}

fn group_op(core: *mut AvpCore, name: *const c_char, err: *mut *mut c_char, start: bool) -> i32 {
    if core.is_null() {
        set_err(err, "null core");
        return -1;
    }
    let inst = unsafe { &*core };
    let name = cstr(name);
    if inst.group(name).is_none() {
        set_err(err, &format!("Mixer group no longer exists: {name}"));
        return -1;
    }
    let result = if start {
        inst.start_group(name)
    } else {
        inst.stop_group(name)
    };
    match result {
        Ok(()) => 0,
        Err(error) => {
            set_err(err, &error.to_string());
            -1
        }
    }
}
