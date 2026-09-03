use std::ffi::c_void;

use crate::Instance;
use crate::abi::AvpBuffer;
use crate::graph::buffer::{AvpMediaType, AvpMediaVtable};
use crate::graph::media::{Media, OpaqueFrame};

pub fn media_to_avp(
    m: Media,
    _vtables: &std::collections::HashMap<AvpMediaType, AvpMediaVtable>,
) -> AvpBuffer {
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

pub fn avp_to_media(buf: AvpBuffer, inst: &Instance) -> Option<Media> {
    match buf.media {
        AvpMediaType::EGL | AvpMediaType::METADATA => {
            if buf.is_null() {
                return None;
            }
            let vt = *inst.media_vtables.lock().unwrap().get(&buf.media)?;
            OpaqueFrame::new(buf.ptr, vt, buf.media).map(Media::Opaque)
        }
        #[cfg(feature = "ffmpeg")]
        AvpMediaType::PACKET => {
            let p = std::ptr::NonNull::new(buf.ptr as *mut rusty_ffmpeg::ffi::AVPacket)?;
            Some(Media::Packet(unsafe {
                rsmpeg::avcodec::AVPacket::from_raw(p)
            }))
        }
        #[cfg(feature = "ffmpeg")]
        AvpMediaType::VIDEO => {
            let p = std::ptr::NonNull::new(buf.ptr as *mut rusty_ffmpeg::ffi::AVFrame)?;
            Some(Media::Video(unsafe {
                rsmpeg::avutil::AVFrame::from_raw(p)
            }))
        }
        #[cfg(feature = "ffmpeg")]
        AvpMediaType::AUDIO => {
            let p = std::ptr::NonNull::new(buf.ptr as *mut rusty_ffmpeg::ffi::AVFrame)?;
            Some(Media::Audio(unsafe {
                rsmpeg::avutil::AVFrame::from_raw(p)
            }))
        }
        #[cfg(not(feature = "ffmpeg"))]
        _ => Some(Media::Stub {
            kind: buf.media,
            pts: buf.ptr as usize as i64,
        }),
    }
}
