//! C callers register commands beside the builtin match. A builtin name and a
//! second registration of the same name are rejected.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_void};

use avplumber_f7k::control::exec_line;
use avplumber_f7k::Instance;

unsafe extern "C" {
    fn malloc(size: usize) -> *mut c_void;
    fn avp_register_command(
        name: *const c_char,
        handler: unsafe extern "C" fn(*mut Instance, *const c_char, *mut *mut c_char) -> i32,
    ) -> i32;
}

unsafe extern "C" fn echo(
    _core: *mut Instance,
    args: *const c_char,
    reply: *mut *mut c_char,
) -> i32 {
    if args.is_null() || reply.is_null() {
        return -1;
    }
    let bytes = unsafe { CStr::from_ptr(args) }.to_bytes_with_nul();
    let buf = unsafe { malloc(bytes.len()) } as *mut c_char;
    if buf.is_null() {
        return -1;
    }
    unsafe {
        std::ptr::copy_nonoverlapping(args, buf, bytes.len());
        *reply = buf;
    }
    0
}

#[test]
fn registered_command_echoes_and_builtins_stay_owned() {
    let inst = Instance::new();
    let echo_name = CString::new("test.echo").unwrap();
    let builtin = CString::new("node.add").unwrap();
    unsafe {
        assert_eq!(avp_register_command(builtin.as_ptr(), echo), -1);
        assert_eq!(avp_register_command(echo_name.as_ptr(), echo), 0);
        assert_eq!(avp_register_command(echo_name.as_ptr(), echo), -1);
    }
    assert_eq!(
        exec_line(&inst, "test.echo hello").unwrap(),
        "hello"
    );
    assert_eq!(exec_line(&inst, "TEST.ECHO again").unwrap(), "again");
    assert!(exec_line(&inst, "test.missing").is_err());
}
