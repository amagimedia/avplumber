//! Flat C structs. Native code uses `Spec` / `Media` / `EdgeKind`; these
//! exist only to match `include/avplumber_core.h`.

use std::ffi::c_void;

use crate::graph::buffer::{AvpMediaType, AvpRational};
use crate::graph::edge::EdgeKind;
use crate::graph::spec::{ChannelLayout, Spec};

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct AvpSpec {
    pub media: AvpMediaType,
    pub width: i32,
    pub height: i32,
    pub pixel_format: i32,
    pub frame_rate: AvpRational,
    pub sample_aspect_ratio: AvpRational,
    pub sample_rate: i32,
    pub sample_format: i32,
    pub channel_order: i32,
    pub nb_channels: i32,
    pub channel_mask: u64,
    pub time_base: AvpRational,
}

impl From<&Spec> for AvpSpec {
    fn from(s: &Spec) -> Self {
        match s {
            Spec::Video {
                width,
                height,
                pix_fmt,
                frame_rate,
                sar,
                time_base,
            } => AvpSpec {
                media: AvpMediaType::VIDEO,
                width: *width,
                height: *height,
                pixel_format: *pix_fmt,
                frame_rate: *frame_rate,
                sample_aspect_ratio: *sar,
                time_base: *time_base,
                ..AvpSpec::zeroed()
            },
            Spec::Audio {
                sample_rate,
                sample_fmt,
                layout,
                time_base,
            } => AvpSpec {
                media: AvpMediaType::AUDIO,
                sample_rate: *sample_rate,
                sample_format: *sample_fmt,
                channel_order: layout.order,
                nb_channels: layout.nb_channels,
                channel_mask: layout.mask,
                time_base: *time_base,
                ..AvpSpec::zeroed()
            },
            Spec::Packet(p) => AvpSpec {
                media: AvpMediaType::PACKET,
                time_base: p.time_base,
                ..AvpSpec::zeroed()
            },
            // Both container-level variants project as lossily as `Spec::Packet`:
            // the C ABI has one flat spec, so only the first stream's time base
            // survives. `to_native()` therefore never rebuilds them.
            Spec::Catalog { streams, .. } => AvpSpec {
                media: AvpMediaType::PACKET,
                time_base: streams
                    .first()
                    .map(|s| s.spec.time_base)
                    .unwrap_or_default(),
                ..AvpSpec::zeroed()
            },
            Spec::Mux { streams } => AvpSpec {
                media: AvpMediaType::PACKET,
                time_base: streams
                    .first()
                    .map(|s| s.spec.time_base)
                    .unwrap_or_default(),
                ..AvpSpec::zeroed()
            },
        }
    }
}

impl AvpSpec {
    pub fn zeroed() -> Self {
        unsafe { std::mem::zeroed() }
    }

    /// Lossy ABI → native. CUSTOM channel maps are not recovered.
    pub fn to_native(&self) -> Spec {
        match self.media {
            AvpMediaType::AUDIO => Spec::Audio {
                sample_rate: self.sample_rate,
                sample_fmt: self.sample_format,
                layout: ChannelLayout::from_mask(
                    self.channel_order,
                    self.nb_channels,
                    self.channel_mask,
                ),
                time_base: self.time_base,
            },
            AvpMediaType::PACKET => {
                Spec::Packet(crate::graph::spec::PacketSpec::new(self.time_base))
            }
            _ => Spec::Video {
                width: self.width,
                height: self.height,
                pix_fmt: self.pixel_format,
                frame_rate: self.frame_rate,
                sar: self.sample_aspect_ratio,
                time_base: self.time_base,
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct AvpBuffer {
    pub media: AvpMediaType,
    pub ptr: *mut c_void,
}

impl AvpBuffer {
    pub fn null(media: AvpMediaType) -> Self {
        Self {
            media,
            ptr: std::ptr::null_mut(),
        }
    }
    pub fn is_null(&self) -> bool {
        self.ptr.is_null()
    }
}

unsafe impl Send for AvpBuffer {}
unsafe impl Sync for AvpBuffer {}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct EdgeCoupling {
    pub is_direct: i32,
    pub capacity: usize,
}

impl Default for EdgeCoupling {
    fn default() -> Self {
        Self {
            is_direct: 0,
            capacity: 0,
        }
    }
}

impl EdgeCoupling {
    pub fn to_native(self) -> EdgeKind {
        if self.is_direct != 0 {
            EdgeKind::Direct
        } else {
            EdgeKind::Buffered {
                capacity: self.capacity,
            }
        }
    }
}
