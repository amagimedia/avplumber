//! Native payload. Owned rsmpeg values internally; `AvpBuffer` lives in `abi/`.

use std::ffi::c_void;
use std::ptr::NonNull;

use crate::graph::buffer::{AVP_NOPTS, AvpMediaType, AvpMediaVtable, AvpRational};
use crate::graph::timebase::{rescale, ts_cmp};

#[derive(Clone, Copy, Debug)]
pub struct Ts {
    pub val: i64,
    pub tb: AvpRational,
}

impl Ts {
    pub fn invalid() -> Self {
        Self {
            val: AVP_NOPTS,
            tb: AvpRational { num: 0, den: 0 },
        }
    }

    pub fn is_valid(self) -> bool {
        self.val != AVP_NOPTS
    }

    pub fn rescale(self, to: AvpRational) -> Ts {
        if !self.is_valid() {
            return Self {
                val: AVP_NOPTS,
                tb: to,
            };
        }
        Ts {
            val: rescale(self.val, self.tb, to),
            tb: to,
        }
    }
}

impl PartialEq for Ts {
    fn eq(&self, other: &Self) -> bool {
        match (self.is_valid(), other.is_valid()) {
            (false, false) => true,
            (true, true) => ts_cmp(self.val, self.tb, other.val, other.tb).is_eq(),
            _ => false,
        }
    }
}

impl Eq for Ts {}

/// C++-owned media (EGL / Metadata). Drop/Clone go through the vtable.
pub struct OpaqueFrame {
    ptr: NonNull<c_void>,
    vtable: AvpMediaVtable,
    media: AvpMediaType,
    time_base: AvpRational,
}

unsafe impl Send for OpaqueFrame {}
unsafe impl Sync for OpaqueFrame {}

impl OpaqueFrame {
    pub fn new(ptr: *mut c_void, vtable: AvpMediaVtable, media: AvpMediaType) -> Option<Self> {
        let ptr = NonNull::new(ptr)?;
        let mut tb = AvpRational::default();
        (vtable.get_time_base)(ptr.as_ptr(), &mut tb);
        Some(Self {
            ptr,
            vtable,
            media,
            time_base: tb,
        })
    }
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr.as_ptr()
    }
    pub fn media(&self) -> AvpMediaType {
        self.media
    }
    pub fn time_base(&self) -> AvpRational {
        self.time_base
    }
    pub fn pts(&self) -> i64 {
        (self.vtable.get_pts)(self.ptr.as_ptr())
    }
    pub fn into_raw(self) -> *mut c_void {
        let p = self.ptr.as_ptr();
        std::mem::forget(self);
        p
    }
}

impl Clone for OpaqueFrame {
    fn clone(&self) -> Self {
        (self.vtable.retain)(self.ptr.as_ptr());
        Self {
            ptr: self.ptr,
            vtable: self.vtable,
            media: self.media,
            time_base: self.time_base,
        }
    }
}

impl Drop for OpaqueFrame {
    fn drop(&mut self) {
        (self.vtable.release)(self.ptr.as_ptr());
    }
}

pub enum Media {
    #[cfg(feature = "ffmpeg")]
    Packet(rsmpeg::avcodec::AVPacket),
    #[cfg(feature = "ffmpeg")]
    Video(rsmpeg::avutil::AVFrame),
    #[cfg(feature = "ffmpeg")]
    Audio(rsmpeg::avutil::AVFrame),
    Opaque(OpaqueFrame),
    #[cfg(not(feature = "ffmpeg"))]
    Stub {
        kind: AvpMediaType,
        pts: i64,
    },
}

unsafe impl Send for Media {}

impl Media {
    pub fn media_type(&self) -> AvpMediaType {
        match self {
            #[cfg(feature = "ffmpeg")]
            Media::Packet(_) => AvpMediaType::PACKET,
            #[cfg(feature = "ffmpeg")]
            Media::Video(_) => AvpMediaType::VIDEO,
            #[cfg(feature = "ffmpeg")]
            Media::Audio(_) => AvpMediaType::AUDIO,
            Media::Opaque(o) => o.media(),
            #[cfg(not(feature = "ffmpeg"))]
            Media::Stub { kind, .. } => *kind,
        }
    }

    pub fn ts(&self) -> Ts {
        match self {
            #[cfg(feature = "ffmpeg")]
            Media::Packet(p) => Ts {
                val: p.pts,
                tb: AvpRational {
                    num: p.time_base.num,
                    den: p.time_base.den,
                },
            },
            #[cfg(feature = "ffmpeg")]
            Media::Video(f) | Media::Audio(f) => Ts {
                val: f.pts,
                tb: AvpRational {
                    num: f.time_base.num,
                    den: f.time_base.den,
                },
            },
            Media::Opaque(o) => Ts {
                val: o.pts(),
                tb: o.time_base(),
            },
            #[cfg(not(feature = "ffmpeg"))]
            Media::Stub { pts, .. } => Ts {
                val: *pts,
                tb: AvpRational { num: 1, den: 1000 },
            },
        }
    }
}

impl Clone for Media {
    fn clone(&self) -> Self {
        match self {
            #[cfg(feature = "ffmpeg")]
            Media::Packet(p) => Media::Packet(clone_packet(p)),
            #[cfg(feature = "ffmpeg")]
            Media::Video(f) => Media::Video(f.clone()),
            #[cfg(feature = "ffmpeg")]
            Media::Audio(f) => Media::Audio(f.clone()),
            Media::Opaque(o) => Media::Opaque(o.clone()),
            #[cfg(not(feature = "ffmpeg"))]
            Media::Stub { kind, pts } => Media::Stub {
                kind: *kind,
                pts: *pts,
            },
        }
    }
}

#[cfg(feature = "ffmpeg")]
fn clone_packet(p: &rsmpeg::avcodec::AVPacket) -> rsmpeg::avcodec::AVPacket {
    unsafe {
        let raw = rusty_ffmpeg::ffi::av_packet_clone(p.as_ptr());
        rsmpeg::avcodec::AVPacket::from_raw(std::ptr::NonNull::new(raw).expect("av_packet_clone"))
    }
}

#[cfg(feature = "ffmpeg")]
pub trait FrameExt {
    fn ts(&self) -> Ts;
    fn set_ts(&mut self, ts: Ts);
}

#[cfg(feature = "ffmpeg")]
impl FrameExt for rsmpeg::avutil::AVFrame {
    fn ts(&self) -> Ts {
        Ts {
            val: self.pts,
            tb: AvpRational {
                num: self.time_base.num,
                den: self.time_base.den,
            },
        }
    }
    fn set_ts(&mut self, ts: Ts) {
        self.set_pts(ts.val);
        self.set_time_base(rusty_ffmpeg::ffi::AVRational {
            num: ts.tb.num,
            den: ts.tb.den,
        });
    }
}

#[cfg(feature = "ffmpeg")]
pub trait PacketExt {
    fn clone_ref(&self) -> rsmpeg::avcodec::AVPacket;
}

#[cfg(feature = "ffmpeg")]
impl PacketExt for rsmpeg::avcodec::AVPacket {
    fn clone_ref(&self) -> rsmpeg::avcodec::AVPacket {
        clone_packet(self)
    }
}
