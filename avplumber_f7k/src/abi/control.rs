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

#[unsafe(no_mangle)]
pub extern "C" fn avp_core_serve_tcp(_core: *mut AvpCore, _port: u16) -> i32 {
    // TCP listener is a thin loop over exec_command; embedders can call that.
    -1
}

pub fn exec_line(core: &AvpCore, line: &str) -> Result<String, String> {
    crate::control::exec_line(core, line)
}
