//! C wrapper over the Rust control parser.

use crate::abi::AvpCore;
use std::ffi::CString;
use std::os::raw::c_char;

#[unsafe(no_mangle)]
pub extern "C" fn avp_core_exec_command(
    core: *mut AvpCore,
    line: *const c_char,
    out_reply: *mut *mut c_char,
) -> i32 {
    if core.is_null() || line.is_null() {
        return -1;
    }
    let line = unsafe { std::ffi::CStr::from_ptr(line) }.to_string_lossy();
    match exec_line(unsafe { &*core }, line.trim()) {
        Ok(s) => {
            if !out_reply.is_null() {
                let c = CString::new(s).unwrap_or_else(|_| CString::new("ok").unwrap());
                unsafe {
                    *out_reply = c.into_raw();
                }
            }
            0
        }
        Err(e) => {
            if !out_reply.is_null() {
                let c = CString::new(e).unwrap_or_else(|_| CString::new("error").unwrap());
                unsafe {
                    *out_reply = c.into_raw();
                }
            }
            -1
        }
    }
}

/// Serves the control protocol on `0.0.0.0:port` for the rest of the process.
/// Returns the bound port, or -1. The `AvpCore` must outlive the process, which
/// is the only way a C embedder holds one.
#[unsafe(no_mangle)]
pub extern "C" fn avp_core_serve_tcp(core: *mut AvpCore, port: u16) -> i32 {
    if core.is_null() {
        return -1;
    }
    // The handle is an `Arc<Instance>` sharing the embedder's inner state:
    // `Instance` is `Arc<InstanceInner>` underneath, so this is a clone of the
    // pointer, not of the graph.
    let instance = std::sync::Arc::new(unsafe { &*core }.share());
    match crate::control::tcp::serve(instance, ("0.0.0.0", port)) {
        Ok(server) => {
            let bound = server.local_addr().port();
            std::mem::forget(server);
            i32::from(bound)
        }
        Err(error) => {
            log::error!("cannot serve the control protocol on port {port}: {error}");
            -1
        }
    }
}

pub fn exec_line(core: &AvpCore, line: &str) -> Result<String, String> {
    crate::control::exec_line(core, line)
}
