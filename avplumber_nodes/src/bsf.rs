//! `bsf` — a libavcodec bitstream filter chain over a packet stream, e.g.
//! `dump_extra=freq=keyframe`, which repeats SPS/PPS before every keyframe so
//! an RTP receiver can join mid-stream. Port of C++ `src/nodes/bsf.cpp`.
//!
//! The C++ node reached up the chain for the encoder's codec parameters and
//! relayed the muxer's questions past itself. Here the parameters arrive as the
//! [`Spec::Packet`] its producer publishes, and what the filter makes of them
//! (`par_out`, `time_base_out`) goes out as a new packet spec, so `mux` and
//! `output` describe the container from what actually flows.

use std::collections::VecDeque;
use std::ffi::CString;
use std::ptr::NonNull;
use std::sync::Mutex;

use rsmpeg::avcodec::{AVCodecParameters, AVPacket};
use rusty_ffmpeg::ffi;

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::buffer::{AvpMediaType, AvpRational};
use avplumber_f7k::graph::edge::EdgeEvent;
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::media::{Media, PacketExt, Ts};
use avplumber_f7k::graph::node::Blocked;
use avplumber_f7k::graph::pad::NodePads;
use avplumber_f7k::graph::spec::{PacketSpec, Spec};
use avplumber_f7k::libav::codec;
use avplumber_f7k::libav::error::{av_error, is_eagain, is_eof};
use avplumber_f7k::scaffold::{Blocking, BlockingIo, InputHandler, SingleInput};

#[derive(Debug, serde::Deserialize)]
pub struct BsfSpec {
    /// The filter list, in `av_bsf_list_parse_str` syntax:
    /// `h264_mp4toannexb`, `dump_extra=freq=keyframe`, `a,b=opt=1`.
    bsf: String,
}

impl NodeSpec for BsfSpec {
    const TYPE_NAME: &'static str = "bsf";
    type Node = Blocking<BitstreamFilter>;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        if self.bsf.trim().is_empty() {
            return Err("bsf: the filter list is empty".into());
        }
        // Parse once here so a typo fails `node.add`; the real context is built
        // on the spec, whose codec parameters it needs.
        Filter::parse(&self.bsf)?;
        Ok(Blocking(BitstreamFilter {
            io: BlockingIo::new(name),
            params: self,
            state: Mutex::new(State::default()),
        }))
    }
}

/// An `AVBSFContext`, owned. rsmpeg wraps single filters by name only; the
/// list syntax needs the C API.
struct Filter(NonNull<ffi::AVBSFContext>);

// The context is touched only from the node's thread, under the state lock.
unsafe impl Send for Filter {}

impl Filter {
    fn parse(list: &str) -> Result<Self, String> {
        let text = CString::new(list).map_err(|_| "bsf: the filter list contains a NUL")?;
        let mut ctx: *mut ffi::AVBSFContext = std::ptr::null_mut();
        let ret = unsafe { ffi::av_bsf_list_parse_str(text.as_ptr(), &mut ctx) };
        if ret < 0 {
            return Err(format!("bsf: cannot create `{list}`: {}", av_error(ret)));
        }
        NonNull::new(ctx)
            .map(Self)
            .ok_or_else(|| format!("bsf: `{list}` produced no context"))
    }

    /// Copies the input parameters in, sets the input time base, initializes.
    fn init(
        &mut self,
        par_in: &AVCodecParameters,
        time_base_in: AvpRational,
    ) -> Result<(), String> {
        unsafe {
            let ctx = self.0.as_ptr();
            let ret = ffi::avcodec_parameters_copy((*ctx).par_in, par_in.as_ptr());
            if ret < 0 {
                return Err(format!(
                    "bsf: cannot copy codec parameters: {}",
                    av_error(ret)
                ));
            }
            (*ctx).time_base_in = time_base_in.into();
            let ret = ffi::av_bsf_init(ctx);
            if ret < 0 {
                return Err(format!("bsf: init failed: {}", av_error(ret)));
            }
        }
        Ok(())
    }

    fn par_out(&self) -> AVCodecParameters {
        let mut out = AVCodecParameters::new();
        unsafe {
            ffi::avcodec_parameters_copy(out.as_mut_ptr(), (*self.0.as_ptr()).par_out);
        }
        out
    }

    fn time_base_out(&self) -> AvpRational {
        unsafe { (*self.0.as_ptr()).time_base_out }.into()
    }

    /// `None` signals the end of the stream. The filter takes the packet.
    fn send(&mut self, packet: Option<&mut AVPacket>) -> Result<(), String> {
        let ptr = packet.map_or(std::ptr::null_mut(), |p| p.as_mut_ptr());
        let ret = unsafe { ffi::av_bsf_send_packet(self.0.as_ptr(), ptr) };
        if ret < 0 && !is_eof(ret) {
            return Err(format!("bsf: send failed: {}", av_error(ret)));
        }
        Ok(())
    }

    /// The next filtered packet, or `None` when the filter wants input or is
    /// drained.
    fn receive(&mut self) -> Result<Option<AVPacket>, String> {
        let mut out = AVPacket::new();
        let ret = unsafe { ffi::av_bsf_receive_packet(self.0.as_ptr(), out.as_mut_ptr()) };
        if ret >= 0 {
            return Ok(Some(out));
        }
        if is_eagain(ret) || is_eof(ret) {
            return Ok(None);
        }
        Err(format!("bsf: receive failed: {}", av_error(ret)))
    }

    fn flush(&mut self) {
        unsafe { ffi::av_bsf_flush(self.0.as_ptr()) };
    }
}

impl Drop for Filter {
    fn drop(&mut self) {
        let mut ctx = self.0.as_ptr();
        unsafe { ffi::av_bsf_free(&mut ctx) };
    }
}

#[derive(Default)]
struct State {
    /// `None` until the input spec arrives.
    filter: Option<Filter>,
    /// The spec the filter was built for, so an identical re-delivery is a no-op.
    input_spec: Option<Spec>,
    /// The input time base the filter expects packets in.
    time_base_in: AvpRational,
    time_base_out: AvpRational,
    /// Filtered packets waiting to go out, oldest first.
    ready: VecDeque<AVPacket>,
    /// `Eof` seen: drain, forward it, finish.
    eof: bool,
    dropped_early: u64,
}

pub struct BitstreamFilter {
    io: BlockingIo,
    params: BsfSpec,
    state: Mutex<State>,
}

impl InputHandler for BitstreamFilter {
    fn on_spec(&self, spec: Spec) -> Result<Option<Spec>, NodeError> {
        let state = &mut *self.state.lock().unwrap();
        let Spec::Packet(packet_spec) = &spec else {
            log::warn!(
                "{}: ignoring a {} spec on the input; bsf needs a packet spec",
                self.io.name,
                spec.variant_name()
            );
            return Ok(None);
        };
        if let Some(known) = &state.input_spec {
            if codec::same_spec(known, &spec) {
                log::debug!("{}: input spec re-delivered unchanged", self.io.name);
                return Ok(None);
            }
            log::info!(
                "{}: input format changed, rebuilding the filter",
                self.io.name
            );
        }
        let par_in = packet_spec.codecpar.as_ref().ok_or_else(|| {
            self.io.error(
                NodePhase::Spec,
                "the input spec carries no codec parameters",
            )
        })?;
        let mut filter =
            Filter::parse(&self.params.bsf).map_err(|m| self.io.error(NodePhase::Spec, m))?;
        filter
            .init(par_in, packet_spec.time_base)
            .map_err(|m| self.io.error(NodePhase::Spec, m))?;
        let out_spec = Spec::Packet(PacketSpec::from_codecpar(
            &filter.par_out(),
            filter.time_base_out(),
            packet_spec.frame_rate,
        ));
        log::info!(
            "{}: `{}` on {}, time base {}/{} -> {}/{}",
            self.io.name,
            self.params.bsf,
            codec::codec_name(par_in.codec_id),
            packet_spec.time_base.num,
            packet_spec.time_base.den,
            filter.time_base_out().num,
            filter.time_base_out().den
        );
        state.time_base_in = packet_spec.time_base;
        state.time_base_out = filter.time_base_out();
        state.filter = Some(filter);
        state.ready.clear();
        state.input_spec = Some(spec);
        Ok(Some(out_spec))
    }

    fn on_buffer(&self, buffer: Media) -> Result<Option<Media>, NodeError> {
        let state = &mut *self.state.lock().unwrap();
        let Media::Packet(mut packet) = buffer else {
            log::warn!("{}: dropping a buffer that is not a packet", self.io.name);
            return Ok(None);
        };
        if state.filter.is_none() {
            state.dropped_early += 1;
            return Ok(None);
        }
        // The filter reads timestamps in its input base; packets say their own.
        let (pts, dts) = (packet.ts(), packet.dts());
        if pts.tb.den != 0 && pts.tb != state.time_base_in {
            packet.set_ts_dts(
                pts.rescale(state.time_base_in),
                dts.rescale(state.time_base_in),
            );
        }
        self.run(state, Some(&mut packet))
            .map_err(|m| self.io.error(NodePhase::Process, m))?;
        Ok(None)
    }

    fn on_flush(&self) {
        let state = &mut *self.state.lock().unwrap();
        if let Some(filter) = state.filter.as_mut() {
            filter.flush();
        }
        state.ready.clear();
    }

    fn on_eof(&self) -> Result<Blocked, NodeError> {
        let state = &mut *self.state.lock().unwrap();
        if state.filter.is_some() {
            self.run(state, None)
                .map_err(|m| self.io.error(NodePhase::Process, m))?;
        }
        state.eof = true;
        Ok(Blocked::Again)
    }

    fn on_closed(&self) {
        let state = self.state.lock().unwrap();
        if state.dropped_early > 0 {
            log::info!(
                "{}: dropped {} packet(s) that arrived before the input spec",
                self.io.name,
                state.dropped_early
            );
        }
    }
}

impl SingleInput for BitstreamFilter {
    fn io(&self) -> &BlockingIo {
        &self.io
    }

    fn pads(&self) -> NodePads {
        NodePads::siso(AvpMediaType::PACKET, AvpMediaType::PACKET)
    }

    fn start(&self) {
        let state = &mut *self.state.lock().unwrap();
        state.ready.clear();
        state.eof = false;
        state.dropped_early = 0;
        if let Some(filter) = state.filter.as_mut() {
            filter.flush();
        }
        // `input_spec` and the filter survive: the re-armed latched spec is
        // recognised as unchanged instead of rebuilding.
    }

    /// Filtered packets go out before anything new is taken in.
    fn before_take(&self) -> Result<Option<Blocked>, NodeError> {
        let state = &mut *self.state.lock().unwrap();
        let out = self.io.output()?;
        if let Some(packet) = state.ready.pop_front() {
            let input = self.io.input()?;
            return self
                .io
                .push_from(&input, &out, Media::Packet(packet))
                .map(Some);
        }
        if state.eof {
            out.push_event(EdgeEvent::Eof);
            return Ok(Some(Blocked::Done));
        }
        Ok(None)
    }
}

impl BitstreamFilter {
    /// Sends one packet (or the end) and queues everything the filter gives
    /// back, stamped in its output time base.
    fn run(&self, state: &mut State, packet: Option<&mut AVPacket>) -> Result<(), String> {
        let tb = state.time_base_out;
        let filter = state.filter.as_mut().ok_or("bsf: no filter")?;
        filter.send(packet)?;
        while let Some(mut out) = filter.receive()? {
            let (pts, dts) = (out.pts, out.dts);
            out.set_ts_dts(Ts { val: pts, tb }, Ts { val: dts, tb });
            state.ready.push_back(out);
        }
        Ok(())
    }
}
