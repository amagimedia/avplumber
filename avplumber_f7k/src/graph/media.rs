//! Native payload. Owned rsmpeg values internally; `AvpBuffer` lives in `abi/`.

use std::ffi::c_void;
use std::ptr::NonNull;

use crate::graph::buffer::{AVP_NOPTS, AvpMediaType, AvpMediaVtable, AvpRational};
use crate::graph::timebase::{finer, rescale, ts_cmp};

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

/// Sum in the finer of the two time bases, like C++ `addTS`. An invalid operand
/// makes the sum invalid: `NOPTS + shift` is not a timestamp.
///
/// To stay in one specific time base — a packet's own, say — rescale the other
/// operand into it first, which makes this the `addTSSameTB` of C++.
impl std::ops::Add for Ts {
    type Output = Ts;

    fn add(self, other: Ts) -> Ts {
        if !self.is_valid() || !other.is_valid() {
            return Ts::invalid();
        }
        let tb = finer(self.tb, other.tb);
        Ts {
            val: rescale(self.val, self.tb, tb) + rescale(other.val, other.tb, tb),
            tb,
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

/// A buffer carrying nothing but a timestamp, for unit tests.
///
/// Cfg-paired so one test body works in either build — an empty libav frame or
/// packet with the feature, [`Media::Stub`] without — and stamped in the same
/// `1/1000` `Stub` reports, so ordering and counting come out identical.
///
/// Compiled for this crate's own tests, and for anyone who asks with the
/// `testing` feature — which is how the node crates' unit tests reach it,
/// through a dev-dependency, so a release build still carries none of it.
#[cfg(all(any(test, feature = "testing"), not(feature = "ffmpeg")))]
pub fn test_media(kind: AvpMediaType, pts: i64) -> Media {
    Media::Stub { kind, pts }
}

#[cfg(all(any(test, feature = "testing"), feature = "ffmpeg"))]
pub fn test_media(kind: AvpMediaType, pts: i64) -> Media {
    use crate::graph::spec::ChannelLayout;
    use rusty_ffmpeg::ffi;

    let ts = Ts {
        val: pts,
        tb: AvpRational { num: 1, den: 1_000 },
    };
    // Smallest real buffer of each kind, not an empty one: `av_frame_clone` and
    // `av_packet_ref` need something to reference, and an edge may clone what it
    // carries (`peek_clone`).
    match kind {
        AvpMediaType::PACKET => {
            let mut packet = rsmpeg::avcodec::AVPacket::new();
            // rsmpeg has no payload allocator, and `av_new_packet` is what makes
            // a packet reference-counted, hence clonable.
            let ret =
                unsafe { ffi::av_new_packet(rsmpeg::UnsafeDerefMut::deref_mut(&mut packet), 1) };
            assert!(ret >= 0, "one-byte test packet: av_new_packet failed");
            packet.set_ts_dts(ts, ts);
            Media::Packet(packet)
        }
        AvpMediaType::VIDEO => {
            let mut frame = rsmpeg::avutil::AVFrame::new();
            frame.set_width(2);
            frame.set_height(2);
            frame.set_format(ffi::AV_PIX_FMT_GRAY8);
            frame.alloc_buffer().expect("2x2 gray8 test frame");
            frame.set_ts(ts);
            Media::Video(frame)
        }
        AvpMediaType::AUDIO => {
            let mut frame = rsmpeg::avutil::AVFrame::new();
            frame.set_nb_samples(1);
            frame.set_sample_rate(48_000);
            frame.set_format(ffi::AV_SAMPLE_FMT_S16);
            let mono = ChannelLayout::from_mask(
                ffi::AV_CHANNEL_ORDER_NATIVE as i32,
                1,
                ffi::AV_CH_LAYOUT_MONO,
            );
            unsafe { mono.apply_to(&mut rsmpeg::UnsafeDerefMut::deref_mut(&mut frame).ch_layout) }
                .expect("mono layout");
            frame.alloc_buffer().expect("one-sample mono test frame");
            frame.set_ts(ts);
            Media::Audio(frame)
        }
        other => panic!("{other:?} has no libav buffer to stand in for it"),
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

/// Timestamps as [`Ts`] plus the flag tests the container nodes need.
///
/// `stream_index` is deliberately absent: the field is readable through `Deref`
/// and rsmpeg already has an inherent `set_stream_index`. Both timestamps are
/// set together because one packet carries a single time base, and adding a
/// second `set_dts(Ts)` here would be shadowed by that inherent setter.
#[cfg(feature = "ffmpeg")]
pub trait PacketExt {
    fn clone_ref(&self) -> rsmpeg::avcodec::AVPacket;
    fn ts(&self) -> Ts;
    fn dts(&self) -> Ts;
    /// Sets PTS, DTS *and* the packet's own time base — libavformat leaves the
    /// latter unset, and everything downstream reads timestamps through it.
    /// `dts` is rescaled into `pts`'s time base.
    fn set_ts_dts(&mut self, pts: Ts, dts: Ts);
    fn is_key(&self) -> bool;
    fn is_corrupt(&self) -> bool;
}

#[cfg(feature = "ffmpeg")]
impl PacketExt for rsmpeg::avcodec::AVPacket {
    fn clone_ref(&self) -> rsmpeg::avcodec::AVPacket {
        clone_packet(self)
    }
    fn ts(&self) -> Ts {
        Ts {
            val: self.pts,
            tb: AvpRational {
                num: self.time_base.num,
                den: self.time_base.den,
            },
        }
    }
    fn dts(&self) -> Ts {
        Ts {
            val: self.dts,
            tb: AvpRational {
                num: self.time_base.num,
                den: self.time_base.den,
            },
        }
    }
    fn set_ts_dts(&mut self, pts: Ts, dts: Ts) {
        self.set_pts(pts.val);
        self.set_dts(dts.rescale(pts.tb).val);
        // rsmpeg has no `set_time_base` for packets, only for frames.
        unsafe {
            rsmpeg::UnsafeDerefMut::deref_mut(self).time_base = rusty_ffmpeg::ffi::AVRational {
                num: pts.tb.num,
                den: pts.tb.den,
            }
        };
    }
    fn is_key(&self) -> bool {
        self.flags & rusty_ffmpeg::ffi::AV_PKT_FLAG_KEY as i32 != 0
    }
    fn is_corrupt(&self) -> bool {
        self.flags & rusty_ffmpeg::ffi::AV_PKT_FLAG_CORRUPT as i32 != 0
    }
}
