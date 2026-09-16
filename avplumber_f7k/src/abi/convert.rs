use std::ffi::c_void;

use crate::abi::AvpGrain;
use crate::graph::grain::{Grain, OpaqueGrain};
use crate::graph::media::{AvpMediaType, AvpMediaVtable};

pub fn grain_to_avp(m: Grain) -> AvpGrain {
    match m {
        #[cfg(feature = "ffmpeg")]
        Grain::Packet(p) => AvpGrain {
            media: AvpMediaType::PACKET,
            ptr: p.into_raw().as_ptr() as *mut c_void,
        },
        #[cfg(feature = "ffmpeg")]
        Grain::Video(f) => AvpGrain {
            media: AvpMediaType::VIDEO,
            ptr: f.into_raw().as_ptr() as *mut c_void,
        },
        #[cfg(feature = "ffmpeg")]
        Grain::Audio(f) => AvpGrain {
            media: AvpMediaType::AUDIO,
            ptr: f.into_raw().as_ptr() as *mut c_void,
        },
        Grain::Opaque(o) => {
            let media = o.media();
            AvpGrain {
                media,
                ptr: o.into_raw(),
            }
        }
        #[cfg(not(feature = "ffmpeg"))]
        Grain::Stub { kind, pts } => AvpGrain {
            media: kind,
            ptr: pts as usize as *mut c_void,
        },
    }
}

/// Project an owned Rust grain into a C handle without transferring it.
/// The returned pointer remains valid only while `grain` remains alive.
pub fn grain_as_avp(grain: &Grain) -> AvpGrain {
    match grain {
        #[cfg(feature = "ffmpeg")]
        Grain::Packet(packet) => AvpGrain {
            media: AvpMediaType::PACKET,
            ptr: packet.as_ptr() as *mut c_void,
        },
        #[cfg(feature = "ffmpeg")]
        Grain::Video(frame) => AvpGrain {
            media: AvpMediaType::VIDEO,
            ptr: frame.as_ptr() as *mut c_void,
        },
        #[cfg(feature = "ffmpeg")]
        Grain::Audio(frame) => AvpGrain {
            media: AvpMediaType::AUDIO,
            ptr: frame.as_ptr() as *mut c_void,
        },
        Grain::Opaque(frame) => AvpGrain {
            media: frame.media(),
            ptr: frame.as_ptr(),
        },
        #[cfg(not(feature = "ffmpeg"))]
        Grain::Stub { kind, pts } => AvpGrain {
            media: *kind,
            ptr: *pts as usize as *mut c_void,
        },
    }
}

/// Release one C-owned grain reference after an edge accepts its retained
/// candidate.
pub fn release_avp_grain(grain: AvpGrain, opaque_vtable: Option<AvpMediaVtable>) -> bool {
    if grain.is_null() {
        return false;
    }
    match grain.media {
        AvpMediaType::EGL | AvpMediaType::METADATA => {
            let Some(vtable) = opaque_vtable else {
                return false;
            };
            (vtable.release)(grain.ptr);
        }
        #[cfg(feature = "ffmpeg")]
        AvpMediaType::PACKET => {
            let mut packet = grain.ptr as *mut rusty_ffmpeg::ffi::AVPacket;
            unsafe { rusty_ffmpeg::ffi::av_packet_free(&mut packet) };
        }
        #[cfg(feature = "ffmpeg")]
        AvpMediaType::VIDEO | AvpMediaType::AUDIO => {
            let mut frame = grain.ptr as *mut rusty_ffmpeg::ffi::AVFrame;
            unsafe { rusty_ffmpeg::ffi::av_frame_free(&mut frame) };
        }
        #[cfg(not(feature = "ffmpeg"))]
        _ => {}
    }
    true
}

/// Retain/clone a candidate for an edge push while leaving the caller's
/// original reference untouched until the edge accepts the candidate.
pub fn clone_avp_grain(grain: AvpGrain, opaque_vtable: Option<AvpMediaVtable>) -> Option<Grain> {
    if grain.is_null() {
        return None;
    }
    match grain.media {
        AvpMediaType::EGL | AvpMediaType::METADATA => {
            let vtable = opaque_vtable?;
            (vtable.retain)(grain.ptr);
            OpaqueGrain::new(grain.ptr, vtable, grain.media).map(Grain::Opaque)
        }
        #[cfg(feature = "ffmpeg")]
        AvpMediaType::PACKET => {
            let raw = unsafe {
                rusty_ffmpeg::ffi::av_packet_clone(grain.ptr as *const rusty_ffmpeg::ffi::AVPacket)
            };
            let packet = std::ptr::NonNull::new(raw)?;
            Some(Grain::Packet(unsafe {
                rsmpeg::avcodec::AVPacket::from_raw(packet)
            }))
        }
        #[cfg(feature = "ffmpeg")]
        AvpMediaType::VIDEO | AvpMediaType::AUDIO => {
            let raw = unsafe {
                rusty_ffmpeg::ffi::av_frame_clone(grain.ptr as *const rusty_ffmpeg::ffi::AVFrame)
            };
            let frame = std::ptr::NonNull::new(raw)?;
            let frame = unsafe { rsmpeg::avutil::AVFrame::from_raw(frame) };
            if grain.media == AvpMediaType::VIDEO {
                Some(Grain::Video(frame))
            } else {
                Some(Grain::Audio(frame))
            }
        }
        #[cfg(not(feature = "ffmpeg"))]
        _ => Some(Grain::Stub {
            kind: grain.media,
            pts: grain.ptr as usize as i64,
        }),
    }
}
