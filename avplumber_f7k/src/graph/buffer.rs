//! Native scalar types shared with the C headers by layout, not by
//! conversion. `AvpBuffer` / `AvpSpec` live in `abi/`.

use std::ffi::c_void;

#[repr(C)]
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub struct AvpRational {
    pub num: i32,
    pub den: i32,
}

pub const AVP_NOPTS: i64 = i64::MIN;

#[repr(C)]
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AvpMediaType {
    PACKET = 1,
    VIDEO = 2,
    AUDIO = 3,
    EGL = 4,
    METADATA = 5,
}

impl AvpMediaType {
    pub fn is_ffmpeg(self) -> bool {
        matches!(self, Self::PACKET | Self::VIDEO | Self::AUDIO)
    }
}

/// Function table for C++-owned media (`Media::Opaque`). Same layout as the
/// C header; native code holds it because OpaqueFrame calls it on Drop/Clone.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct AvpMediaVtable {
    pub retain: extern "C" fn(*mut c_void),
    pub release: extern "C" fn(*mut c_void),
    pub get_pts: extern "C" fn(*mut c_void) -> i64,
    pub get_time_base: extern "C" fn(*mut c_void, *mut AvpRational),
}
