//! Native stream/container description. The flat `AvpSpec` is an ABI
//! projection in `abi/types.rs`.

use crate::graph::buffer::{AvpMediaType, AvpRational};

/// Channel layout carried whole. `map` holds CUSTOM/AMBISONIC indices when
/// `nb_channels` does not fit a bitmask.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct ChannelLayout {
    pub order: i32,
    pub nb_channels: i32,
    pub mask: u64,
    pub map: Vec<i32>,
}

impl ChannelLayout {
    pub fn from_mask(order: i32, nb_channels: i32, mask: u64) -> Self {
        Self {
            order,
            nb_channels,
            mask,
            map: Vec::new(),
        }
    }
}

/// Codec parameters for a packet stream. Native path keeps codecpar; the
/// non-ffmpeg build keeps a compact stand-in so the core still type-checks.
#[derive(Clone, Debug)]
pub struct PacketSpec {
    pub time_base: AvpRational,
    pub codec_id: i32,
    pub extra_data: Vec<u8>,
}

impl PacketSpec {
    pub fn new(time_base: AvpRational) -> Self {
        Self {
            time_base,
            codec_id: 0,
            extra_data: Vec::new(),
        }
    }
}

#[derive(Clone, Debug)]
pub enum Spec {
    Video {
        width: i32,
        height: i32,
        pix_fmt: i32,
        frame_rate: AvpRational,
        sar: AvpRational,
        time_base: AvpRational,
    },
    Audio {
        sample_rate: i32,
        sample_fmt: i32,
        layout: ChannelLayout,
        time_base: AvpRational,
    },
    Packet(PacketSpec),
    Mux {
        streams: Vec<PacketSpec>,
    },
}

impl Spec {
    pub fn time_base(&self) -> AvpRational {
        match self {
            Spec::Video { time_base, .. } | Spec::Audio { time_base, .. } => *time_base,
            Spec::Packet(p) => p.time_base,
            Spec::Mux { streams } => streams.first().map(|s| s.time_base).unwrap_or_default(),
        }
    }

    pub fn media(&self) -> AvpMediaType {
        match self {
            Spec::Video { .. } => AvpMediaType::VIDEO,
            Spec::Audio { .. } => AvpMediaType::AUDIO,
            Spec::Packet(_) | Spec::Mux { .. } => AvpMediaType::PACKET,
        }
    }
}
