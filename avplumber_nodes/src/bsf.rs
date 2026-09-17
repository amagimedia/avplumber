//! `bsf` — a libavcodec bitstream filter chain over a packet stream, e.g.
//! `dump_extra=freq=keyframe`, which repeats SPS/PPS before every keyframe so
//! an RTP receiver can join mid-stream. Port of C++ `src/nodes/bsf.cpp`.
//!
//! The C++ node reached up the chain for the encoder's codec parameters and
//! relayed the muxer's questions past itself. Here the parameters arrive as the
//! [`Spec::Packet`] its producer publishes, and what the filter makes of them
//! (`par_out`, `time_base_out`) goes out as a new packet spec, so `mux` and
//! `output` describe the container from what actually flows.

use rsmpeg::avcodec::AVPacket;

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::grain::{Grain, PacketExt};
use avplumber_f7k::graph::media::{AvpMediaType, AvpRational};
use avplumber_f7k::graph::pad::NodePads;
use avplumber_f7k::graph::spec::{PacketSpec, Spec};
use avplumber_f7k::graph::timestamp::Ts;
use avplumber_f7k::libav::bsf;
use avplumber_f7k::libav::codec;
use avplumber_f7k::node_api::{SisoAdapter, SisoNode};

#[derive(Debug, serde::Deserialize)]
pub struct BsfSpec {
    /// The filter list, in `av_bsf_list_parse_str` syntax:
    /// `h264_mp4toannexb`, `dump_extra=freq=keyframe`, `a,b=opt=1`.
    bsf: String,
}

impl NodeSpec for BsfSpec {
    const TYPE_NAME: &'static str = "bsf";
    type Node = SisoAdapter<BitstreamFilter>;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        if self.bsf.trim().is_empty() {
            return Err("bsf: the filter list is empty".into());
        }
        // Parse once here so a typo fails `node.add`; the real context is built
        // on the spec, whose codec parameters it needs.
        bsf::Context::parse(&self.bsf)?;
        Ok(SisoAdapter::new(BitstreamFilter {
            name: name.to_string(),
            list: self.bsf,
        }))
    }
}

pub struct BitstreamFilter {
    name: String,
    list: String,
}

/// Associated type of [`SisoNode`]: must be `pub` because `BitstreamFilter` is.
pub struct FilterState {
    filter: bsf::Context,
    time_base_in: AvpRational,
    time_base_out: AvpRational,
}

impl SisoNode for BitstreamFilter {
    type InputState = FilterState;

    fn name(&self) -> &str {
        &self.name
    }

    fn pads(&self) -> NodePads {
        NodePads::siso(AvpMediaType::PACKET, AvpMediaType::PACKET)
    }

    fn on_spec(&self, spec: &Spec) -> Result<(FilterState, Spec), String> {
        let Spec::Packet(packet_spec) = spec else {
            return Err(format!(
                "bsf needs a packet spec, got {}",
                spec.variant_name()
            ));
        };
        let par_in = packet_spec
            .codecpar
            .as_ref()
            .ok_or("the input spec carries no codec parameters")?;
        let mut filter = bsf::Context::parse(&self.list)?;
        filter.init(par_in, packet_spec.time_base)?;
        let time_base_out = filter.time_base_out();
        let out_spec = Spec::Packet(PacketSpec::from_codecpar(
            &filter.par_out(),
            time_base_out,
            packet_spec.frame_rate,
        ));
        log::info!(
            "{}: `{}` on {}, time base {}/{} -> {}/{}",
            self.name,
            self.list,
            codec::codec_name(par_in.codec_id),
            packet_spec.time_base.num,
            packet_spec.time_base.den,
            time_base_out.num,
            time_base_out.den
        );
        Ok((
            FilterState {
                filter,
                time_base_in: packet_spec.time_base,
                time_base_out,
            },
            out_spec,
        ))
    }

    fn process(&self, inner: &mut FilterState, buffer: Grain) -> Result<Vec<Grain>, String> {
        let Grain::Packet(mut packet) = buffer else {
            log::warn!("{}: dropping a buffer that is not a packet", self.name);
            return Ok(Vec::new());
        };
        let (pts, dts) = (packet.ts(), packet.dts());
        if pts.tb.den != 0 && pts.tb != inner.time_base_in {
            packet.set_ts_dts(
                pts.rescale(inner.time_base_in),
                dts.rescale(inner.time_base_in),
            );
        }
        let packets = inner.filter.run(Some(&mut packet))?;
        Ok(stamp(inner.time_base_out, packets))
    }

    fn on_eof(&self, inner: &mut FilterState) -> Result<Vec<Grain>, String> {
        let packets = inner.filter.run(None)?;
        Ok(stamp(inner.time_base_out, packets))
    }

    fn flush(&self, inner: &mut FilterState) {
        inner.filter.flush();
    }

    fn start(&self, inner: &mut FilterState) {
        inner.filter.flush();
    }
}

fn stamp(tb: AvpRational, packets: Vec<AVPacket>) -> Vec<Grain> {
    packets
        .into_iter()
        .map(|mut packet| {
            let (pts, dts) = (packet.pts, packet.dts);
            packet.set_ts_dts(Ts { val: pts, tb }, Ts { val: dts, tb });
            Grain::Packet(packet)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use avplumber_f7k::Instance;
    use avplumber_f7k::graph::BufferedEdge;
    use avplumber_f7k::graph::edge::{Edge, EdgeEvent, EdgeItem};
    use avplumber_f7k::graph::grain::test_media;
    use avplumber_f7k::graph::node::{Node, Processed};
    use avplumber_f7k::graph::spec::PacketSpec;
    use rsmpeg::avcodec::AVCodecParameters;
    use rusty_ffmpeg::ffi;

    fn packet_spec() -> Spec {
        let mut par = AVCodecParameters::new();
        unsafe {
            let raw = rsmpeg::UnsafeDerefMut::deref_mut(&mut par);
            raw.codec_type = ffi::AVMEDIA_TYPE_VIDEO;
            raw.codec_id = ffi::AV_CODEC_ID_H264;
        }
        Spec::Packet(PacketSpec::from_codecpar(
            &par,
            AvpRational {
                num: 1,
                den: 90_000,
            },
            AvpRational { num: 25, den: 1 },
        ))
    }

    fn node() -> (SisoAdapter<BitstreamFilter>, Arc<dyn Edge>, Arc<dyn Edge>) {
        let instance = Instance::new();
        let params = serde_json::json!({});
        let ctx = BuildCtx {
            instance: &instance,
            name: "bsf",
            params: &params,
            sync_group: None,
        };
        let node = BsfSpec { bsf: "null".into() }
            .build("bsf", &ctx)
            .expect("null bsf builds");
        let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(8));
        let output: Arc<dyn Edge> = Arc::new(BufferedEdge::new(8));
        node.bind_source("in", input.clone());
        node.bind_sink("out", output.clone());
        node.start();
        (node, input, output)
    }

    #[test]
    fn null_filter_forwards_packets_and_eof() {
        let (node, input, output) = node();
        input.push_event(EdgeEvent::Spec(packet_spec()));
        assert!(input.offer(test_media(AvpMediaType::PACKET, 1)).is_ok());
        input.push_event(EdgeEvent::Eof);

        assert_eq!(node.process().unwrap(), Processed::Again, "spec");
        assert_eq!(node.process().unwrap(), Processed::Again, "packet");
        assert_eq!(node.process().unwrap(), Processed::Done, "eof");

        let mut pts = Vec::new();
        let mut saw_eof = false;
        while let Some(item) = output.try_take() {
            match item {
                EdgeItem::Buffer(Grain::Packet(p)) => pts.push(p.pts),
                EdgeItem::Event(EdgeEvent::Eof) => saw_eof = true,
                EdgeItem::Event(EdgeEvent::Spec(_)) => {}
                _ => panic!("unexpected item"),
            }
        }
        assert_eq!(
            pts,
            vec![90],
            "1ms in 1/1000 restamped into the 90 kHz input time base"
        );
        assert!(saw_eof);
    }
}
