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

#[cfg(feature = "ffmpeg")]
impl ChannelLayout {
    /// Snapshot a libav layout: its bitmask, or the channel ids of a CUSTOM one.
    pub fn from_av(layout: &rusty_ffmpeg::ffi::AVChannelLayout) -> Self {
        use rusty_ffmpeg::ffi;

        let order = layout.order as i32;
        let nb_channels = layout.nb_channels;
        let custom = layout.order == ffi::AV_CHANNEL_ORDER_CUSTOM;
        let map = unsafe { layout.u.map };
        if custom && !map.is_null() && nb_channels > 0 {
            let ids = unsafe { std::slice::from_raw_parts(map, nb_channels as usize) };
            Self {
                order,
                nb_channels,
                mask: 0,
                map: ids.iter().map(|c| c.id).collect(),
            }
        } else {
            Self {
                order,
                nb_channels,
                mask: unsafe { layout.u.mask },
                map: Vec::new(),
            }
        }
    }

    /// Write this layout into `dst`. Deliberately not `av_channel_layout_retype`
    /// (FFmpeg 7.0+): the fields are set directly, and only a CUSTOM order
    /// allocates, with `av_mallocz` so libav can free it in
    /// `av_channel_layout_uninit`.
    ///
    /// # Safety
    /// `dst` must point at a writable `AVChannelLayout` that is *uninitialised*
    /// (freshly allocated or already uninited) — a live CUSTOM layout's `map`
    /// would leak, since nothing here frees the previous contents.
    pub unsafe fn apply_to(
        &self,
        dst: *mut rusty_ffmpeg::ffi::AVChannelLayout,
    ) -> Result<(), String> {
        use rusty_ffmpeg::ffi;

        let nb_channels = self.nb_channels.max(0);
        let dst = unsafe { &mut *dst };
        dst.opaque = std::ptr::null_mut();
        dst.nb_channels = nb_channels;

        if self.order != ffi::AV_CHANNEL_ORDER_CUSTOM as i32 {
            dst.order = self.order as ffi::AVChannelOrder;
            dst.u.mask = self.mask;
            return Ok(());
        }

        if self.map.len() != nb_channels as usize || self.map.is_empty() {
            return Err(format!(
                "custom channel layout carries {} ids for {nb_channels} channels",
                self.map.len()
            ));
        }
        let bytes = std::mem::size_of::<ffi::AVChannelCustom>() * self.map.len();
        let map = unsafe { ffi::av_mallocz(bytes) } as *mut ffi::AVChannelCustom;
        if map.is_null() {
            return Err("out of memory allocating a custom channel map".into());
        }
        for (i, id) in self.map.iter().enumerate() {
            unsafe { (*map.add(i)).id = *id as ffi::AVChannel };
        }
        dst.order = ffi::AV_CHANNEL_ORDER_CUSTOM;
        dst.u.map = map;
        Ok(())
    }
}

/// Which container streams a consumer actually wants. Travels *upstream* as
/// [`EdgeHint::Streams`](crate::graph::EdgeHint::Streams); the container owner
/// discards the rest.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct StreamSelection {
    pub enabled: Vec<i32>,
}

impl StreamSelection {
    pub fn new(enabled: Vec<i32>) -> Self {
        Self { enabled }
    }

    pub fn contains(&self, index: i32) -> bool {
        self.enabled.contains(&index)
    }
}

/// Codec parameters for a packet stream. The native path carries the whole
/// codecpar; `codec_id`/`extra_data` stay for the non-ffmpeg build and the ABI
/// projection.
#[derive(Clone, Debug, Default)]
pub struct PacketSpec {
    pub time_base: AvpRational,
    /// The stream's nominal rate, for `output`'s `avg_frame_rate` and the
    /// encoder's `framerate` — 0/0 when unknown.
    pub frame_rate: AvpRational,
    pub codec_id: i32,
    pub extra_data: Vec<u8>,
    /// Owned deep copy, so it outlives the context it was read from. `Send` but
    /// not `Sync`, which is fine: a `Spec` only ever lives behind an edge mutex.
    #[cfg(feature = "ffmpeg")]
    pub codecpar: Option<rsmpeg::avcodec::AVCodecParameters>,
}

impl PacketSpec {
    pub fn new(time_base: AvpRational) -> Self {
        Self {
            time_base,
            ..Self::default()
        }
    }
}

#[cfg(feature = "ffmpeg")]
impl PacketSpec {
    /// Deep-copy codec parameters, keeping `codec_id`/`extra_data` in sync so
    /// the two representations never disagree.
    pub fn from_codecpar(
        par: &rsmpeg::avcodec::AVCodecParameters,
        time_base: AvpRational,
        frame_rate: AvpRational,
    ) -> Self {
        let extra_data = if par.extradata.is_null() || par.extradata_size <= 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(par.extradata, par.extradata_size as usize) }
                .to_vec()
        };
        Self {
            time_base,
            frame_rate,
            codec_id: par.codec_id as i32,
            extra_data,
            codecpar: Some(par.clone()),
        }
    }
}

/// One container stream as the demuxing side sees it.
#[derive(Clone, Debug)]
pub struct CatalogStream {
    /// Index within the container, i.e. what a packet's `stream_index` holds.
    pub index: i32,
    /// Raw `AVMEDIA_TYPE_*`, so routing keys like `d:0` (data) or `?D` resolve
    /// without a lossy detour through [`AvpMediaType`].
    pub codec_type: i32,
    pub spec: PacketSpec,
    /// This stream's `avformat_match_stream_specifier` result for the catalog's
    /// `filter`; `true` when no filter was requested.
    pub matches_filter: bool,
}

/// One stream `output` should create, as `mux` describes it.
#[derive(Clone, Debug, Default)]
pub struct MuxStream {
    pub spec: PacketSpec,
    /// Container-level stream id (`stream_ids` param); `None` leaves libav's.
    pub id: Option<i32>,
    pub metadata: Vec<(String, String)>,
}

impl MuxStream {
    pub fn new(spec: PacketSpec) -> Self {
        Self {
            spec,
            id: None,
            metadata: Vec::new(),
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
    /// Every stream of a container, annotated with the `streams_filter` the
    /// publisher evaluated (`None` = none was requested). Echoing the filter is
    /// what makes the [hint](crate::graph::EdgeHint) handshake
    /// order-independent: a consumer ignores a catalog answering a different
    /// question and waits for the next one.
    Catalog {
        filter: Option<String>,
        streams: Vec<CatalogStream>,
    },
    /// A whole output container, in stream-index order.
    Mux {
        streams: Vec<MuxStream>,
    },
}

impl Spec {
    pub fn time_base(&self) -> AvpRational {
        match self {
            Spec::Video { time_base, .. } | Spec::Audio { time_base, .. } => *time_base,
            Spec::Packet(p) => p.time_base,
            Spec::Catalog { streams, .. } => streams
                .first()
                .map(|s| s.spec.time_base)
                .unwrap_or_default(),
            Spec::Mux { streams } => streams
                .first()
                .map(|s| s.spec.time_base)
                .unwrap_or_default(),
        }
    }

    pub fn media(&self) -> AvpMediaType {
        match self {
            Spec::Video { .. } => AvpMediaType::VIDEO,
            Spec::Audio { .. } => AvpMediaType::AUDIO,
            Spec::Packet(_) | Spec::Catalog { .. } | Spec::Mux { .. } => AvpMediaType::PACKET,
        }
    }

    /// The variant's own name, for logs. [`Spec::media`] cannot tell `Packet`,
    /// `Catalog` and `Mux` apart, yet all three are legitimate things to find on
    /// a packet edge, so "expected a packet spec, got a packet spec" is exactly
    /// the message a node would otherwise print.
    pub fn variant_name(&self) -> &'static str {
        match self {
            Spec::Video { .. } => "video",
            Spec::Audio { .. } => "audio",
            Spec::Packet(_) => "packet",
            Spec::Catalog { .. } => "catalog",
            Spec::Mux { .. } => "mux",
        }
    }
}
