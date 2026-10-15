//! Process-global command table for handlers the core match does not know.
//! Builtins stay in [`super::exec_line`] and cannot be shadowed.

use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};
use std::sync::Mutex;

use crate::abi::AvpCore;

pub type AvpCommandFn = unsafe extern "C" fn(*mut AvpCore, *const c_char, *mut *mut c_char) -> i32;

static COMMANDS: Mutex<Option<HashMap<String, AvpCommandFn>>> = Mutex::new(None);

fn table() -> std::sync::MutexGuard<'static, Option<HashMap<String, AvpCommandFn>>> {
    COMMANDS.lock().unwrap()
}

/// Names handled by the match in `exec_line`, not by this table.
pub fn is_builtin(name: &str) -> bool {
    matches!(
        name,
        "node.add"
            | "edge.add"
            | "queue.plan_capacity"
            | "group.start"
            | "group.stop"
            | "group.restart"
            | "group.status"
            | "node.object.set"
            | "node.object.get"
            | "seek"
            | "pause"
            | "resume"
            | "speed.set"
            | "speed.get"
            | "playback.status"
            | "hwaccel.init"
            | "hello"
            | "version"
            | "bye"
    )
}

#[unsafe(no_mangle)]
pub extern "C" fn avp_register_command(name: *const c_char, handler: AvpCommandFn) -> i32 {
    if name.is_null() || handler as usize == 0 {
        return -1;
    }
    let name = unsafe { CStr::from_ptr(name) }.to_string_lossy();
    let name = name.trim().to_ascii_lowercase();
    if name.is_empty() || is_builtin(&name) {
        return -1;
    }
    let mut guard = table();
    let map = guard.get_or_insert_with(HashMap::new);
    if map.contains_key(&name) {
        return -1;
    }
    map.insert(name, handler);
    0
}

/// `Some` when a handler is registered. The handler is called without the table lock.
pub fn dispatch(core: &AvpCore, name: &str, args: &str) -> Option<Result<String, String>> {
    let key = name.trim().to_ascii_lowercase();
    let handler = {
        let guard = table();
        let map = guard.as_ref()?;
        *map.get(&key)?
    };
    let args = CString::new(args).unwrap_or_else(|_| CString::new("").unwrap());
    let mut reply: *mut c_char = std::ptr::null_mut();
    let rc = unsafe { handler(core as *const AvpCore as *mut AvpCore, args.as_ptr(), &mut reply) };
    let text = if reply.is_null() {
        String::new()
    } else {
        let owned = unsafe { CStr::from_ptr(reply) }.to_string_lossy().into_owned();
        unsafe { free(reply.cast()) };
        owned
    };
    Some(if rc == 0 {
        Ok(text)
    } else {
        Err(if text.is_empty() {
            format!("command {key} failed")
        } else {
            text
        })
    })
}

unsafe extern "C" {
    fn free(ptr: *mut c_void);
}
