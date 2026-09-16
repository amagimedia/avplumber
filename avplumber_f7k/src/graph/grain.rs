//! Native payload. Owned rsmpeg values internally; `AvpGrain` lives in `abi/`.

use std::ffi::c_void;
use std::ptr::NonNull;

#[cfg(feature = "ffmpeg")]
use crate::graph::media::AVP_NOPTS;
use crate::graph::media::{AvpMediaType, AvpMediaVtable, AvpRational};
use crate::graph::timestamp::Ts;

/// C++-owned media (EGL / Metadata). Drop/Clone go through the vtable.
pub struct OpaqueGrain {
    ptr: NonNull<c_void>,
    vtable: AvpMediaVtable,
    media: AvpMediaType,
    time_base: AvpRational,
}

unsafe impl Send for OpaqueGrain {}
unsafe impl Sync for OpaqueGrain {}

impl OpaqueGrain {
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

impl Clone for OpaqueGrain {
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

impl Drop for OpaqueGrain {
    fn drop(&mut self) {
        (self.vtable.release)(self.ptr.as_ptr());
    }
}

/// A single timestamped packet or a frame.
///
/// For consistency with FFmpeg, audio fragment which usually contains multiple samples is also called a "frame".
pub enum Grain {
    #[cfg(feature = "ffmpeg")]
    Packet(rsmpeg::avcodec::AVPacket),
    #[cfg(feature = "ffmpeg")]
    Video(rsmpeg::avutil::AVFrame),
    #[cfg(feature = "ffmpeg")]
    Audio(rsmpeg::avutil::AVFrame),
    Opaque(OpaqueGrain),
    #[cfg(not(feature = "ffmpeg"))]
    Stub {
        kind: AvpMediaType,
        pts: i64,
    },
}

unsafe impl Send for Grain {}

impl Grain {
    pub fn media_type(&self) -> AvpMediaType {
        match self {
            #[cfg(feature = "ffmpeg")]
            Grain::Packet(_) => AvpMediaType::PACKET,
            #[cfg(feature = "ffmpeg")]
            Grain::Video(_) => AvpMediaType::VIDEO,
            #[cfg(feature = "ffmpeg")]
            Grain::Audio(_) => AvpMediaType::AUDIO,
            Grain::Opaque(o) => o.media(),
            #[cfg(not(feature = "ffmpeg"))]
            Grain::Stub { kind, .. } => *kind,
        }
    }

    pub fn ts(&self) -> Ts {
        match self {
            #[cfg(feature = "ffmpeg")]
            Grain::Packet(p) => Ts {
                val: p.pts,
                tb: AvpRational {
                    num: p.time_base.num,
                    den: p.time_base.den,
                },
            },
            #[cfg(feature = "ffmpeg")]
            Grain::Video(f) | Grain::Audio(f) => Ts {
                val: f.pts,
                tb: AvpRational {
                    num: f.time_base.num,
                    den: f.time_base.den,
                },
            },
            Grain::Opaque(o) => Ts {
                val: o.pts(),
                tb: o.time_base(),
            },
            #[cfg(not(feature = "ffmpeg"))]
            Grain::Stub { pts, .. } => Ts {
                val: *pts,
                tb: AvpRational { num: 1, den: 1000 },
            },
        }
    }

    /// Restamps the buffer, time base included, for the nodes that own the
    /// timeline (a frame-rate conformer, the pacing stage). A packet keeps its
    /// DTS at the same distance from its PTS; an opaque frame cannot be
    /// restamped and is left alone.
    pub fn set_ts(&mut self, ts: Ts) {
        match self {
            #[cfg(feature = "ffmpeg")]
            Grain::Packet(p) => {
                let dts = if p.dts != AVP_NOPTS && p.pts != AVP_NOPTS {
                    Ts {
                        val: ts.val - (p.pts - p.dts),
                        tb: ts.tb,
                    }
                } else {
                    Ts::invalid()
                };
                p.set_ts_dts(ts, dts);
            }
            #[cfg(feature = "ffmpeg")]
            Grain::Video(f) | Grain::Audio(f) => f.set_ts(ts),
            Grain::Opaque(_) => {}
            #[cfg(not(feature = "ffmpeg"))]
            Grain::Stub { pts, .. } => *pts = ts.rescale(AvpRational { num: 1, den: 1000 }).val,
        }
    }
}

/// A buffer carrying nothing but a timestamp, for unit tests.
///
/// Cfg-paired so one test body works in either build — an empty libav frame or
/// packet with the feature, [`Grain::Stub`] without — and stamped in the same
/// `1/1000` `Stub` reports, so ordering and counting come out identical.
///
/// Compiled for this crate's own tests, and for anyone who asks with the
/// `testing` feature — which is how the node crates' unit tests reach it,
/// through a dev-dependency, so a release build still carries none of it.
#[cfg(all(any(test, feature = "testing"), not(feature = "ffmpeg")))]
pub fn test_media(kind: AvpMediaType, pts: i64) -> Grain {
    Grain::Stub { kind, pts }
}

#[cfg(all(any(test, feature = "testing"), feature = "ffmpeg"))]
pub fn test_media(kind: AvpMediaType, pts: i64) -> Grain {
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
            Grain::Packet(packet)
        }
        AvpMediaType::VIDEO => {
            let mut frame = rsmpeg::avutil::AVFrame::new();
            frame.set_width(2);
            frame.set_height(2);
            frame.set_format(ffi::AV_PIX_FMT_GRAY8);
            frame.alloc_buffer().expect("2x2 gray8 test frame");
            frame.set_ts(ts);
            Grain::Video(frame)
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
            Grain::Audio(frame)
        }
        other => panic!("{other:?} has no libav buffer to stand in for it"),
    }
}

impl Clone for Grain {
    fn clone(&self) -> Self {
        match self {
            #[cfg(feature = "ffmpeg")]
            Grain::Packet(p) => Grain::Packet(clone_packet(p)),
            #[cfg(feature = "ffmpeg")]
            Grain::Video(f) => Grain::Video(f.clone()),
            #[cfg(feature = "ffmpeg")]
            Grain::Audio(f) => Grain::Audio(f.clone()),
            Grain::Opaque(o) => Grain::Opaque(o.clone()),
            #[cfg(not(feature = "ffmpeg"))]
            Grain::Stub { kind, pts } => Grain::Stub {
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
