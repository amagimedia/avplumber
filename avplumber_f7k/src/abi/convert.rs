use std::ffi::c_void;

use crate::abi::AvpBuffer;
use crate::graph::buffer::{AvpMediaType, AvpMediaVtable};
use crate::graph::media::{Media, OpaqueFrame};

pub fn media_to_avp(m: Media) -> AvpBuffer {
    match m {
        #[cfg(feature = "ffmpeg")]
        Media::Packet(p) => AvpBuffer {
            media: AvpMediaType::PACKET,
            ptr: p.into_raw().as_ptr() as *mut c_void,
        },
        #[cfg(feature = "ffmpeg")]
        Media::Video(f) => AvpBuffer {
            media: AvpMediaType::VIDEO,
            ptr: f.into_raw().as_ptr() as *mut c_void,
        },
        #[cfg(feature = "ffmpeg")]
        Media::Audio(f) => AvpBuffer {
            media: AvpMediaType::AUDIO,
            ptr: f.into_raw().as_ptr() as *mut c_void,
        },
        Media::Opaque(o) => {
            let media = o.media();
            AvpBuffer {
                media,
                ptr: o.into_raw(),
            }
        }
        #[cfg(not(feature = "ffmpeg"))]
        Media::Stub { kind, pts } => AvpBuffer {
            media: kind,
            ptr: pts as usize as *mut c_void,
        },
    }
}

/// Project an owned Rust media value into a C buffer without transferring it.
/// The returned pointer remains valid only while `media` remains alive.
pub fn media_as_avp(media: &Media) -> AvpBuffer {
    match media {
        #[cfg(feature = "ffmpeg")]
        Media::Packet(packet) => AvpBuffer {
            media: AvpMediaType::PACKET,
            ptr: packet.as_ptr() as *mut c_void,
        },
        #[cfg(feature = "ffmpeg")]
        Media::Video(frame) => AvpBuffer {
            media: AvpMediaType::VIDEO,
            ptr: frame.as_ptr() as *mut c_void,
        },
        #[cfg(feature = "ffmpeg")]
        Media::Audio(frame) => AvpBuffer {
            media: AvpMediaType::AUDIO,
            ptr: frame.as_ptr() as *mut c_void,
        },
        Media::Opaque(frame) => AvpBuffer {
            media: frame.media(),
            ptr: frame.as_ptr(),
        },
        #[cfg(not(feature = "ffmpeg"))]
        Media::Stub { kind, pts } => AvpBuffer {
            media: *kind,
            ptr: *pts as usize as *mut c_void,
        },
    }
}

/// Release one C-owned buffer reference after an edge accepts its retained
/// candidate.
pub fn release_avp_buffer(buf: AvpBuffer, opaque_vtable: Option<AvpMediaVtable>) -> bool {
    if buf.is_null() {
        return false;
    }
    match buf.media {
        AvpMediaType::EGL | AvpMediaType::METADATA => {
            let Some(vtable) = opaque_vtable else {
                return false;
            };
            (vtable.release)(buf.ptr);
        }
        #[cfg(feature = "ffmpeg")]
        AvpMediaType::PACKET => {
            let mut packet = buf.ptr as *mut rusty_ffmpeg::ffi::AVPacket;
            unsafe { rusty_ffmpeg::ffi::av_packet_free(&mut packet) };
        }
        #[cfg(feature = "ffmpeg")]
        AvpMediaType::VIDEO | AvpMediaType::AUDIO => {
            let mut frame = buf.ptr as *mut rusty_ffmpeg::ffi::AVFrame;
            unsafe { rusty_ffmpeg::ffi::av_frame_free(&mut frame) };
        }
        #[cfg(not(feature = "ffmpeg"))]
        _ => {}
    }
    true
}

/// Retain/clone a candidate for an edge push while leaving the caller's
/// original reference untouched until the edge accepts the candidate.
pub fn clone_avp_buffer(buf: AvpBuffer, opaque_vtable: Option<AvpMediaVtable>) -> Option<Media> {
    if buf.is_null() {
        return None;
    }
    match buf.media {
        AvpMediaType::EGL | AvpMediaType::METADATA => {
            let vtable = opaque_vtable?;
            (vtable.retain)(buf.ptr);
            OpaqueFrame::new(buf.ptr, vtable, buf.media).map(Media::Opaque)
        }
        #[cfg(feature = "ffmpeg")]
        AvpMediaType::PACKET => {
            let raw = unsafe {
                rusty_ffmpeg::ffi::av_packet_clone(buf.ptr as *const rusty_ffmpeg::ffi::AVPacket)
            };
            let packet = std::ptr::NonNull::new(raw)?;
            Some(Media::Packet(unsafe {
                rsmpeg::avcodec::AVPacket::from_raw(packet)
            }))
        }
        #[cfg(feature = "ffmpeg")]
        AvpMediaType::VIDEO | AvpMediaType::AUDIO => {
            let raw = unsafe {
                rusty_ffmpeg::ffi::av_frame_clone(buf.ptr as *const rusty_ffmpeg::ffi::AVFrame)
            };
            let frame = std::ptr::NonNull::new(raw)?;
            let frame = unsafe { rsmpeg::avutil::AVFrame::from_raw(frame) };
            if buf.media == AvpMediaType::VIDEO {
                Some(Media::Video(frame))
            } else {
                Some(Media::Audio(frame))
            }
        }
        #[cfg(not(feature = "ffmpeg"))]
        _ => Some(Media::Stub {
            kind: buf.media,
            pts: buf.ptr as usize as i64,
        }),
    }
}
