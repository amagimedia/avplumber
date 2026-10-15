//! `filter_video` — one libavfilter graph between a video spec and video frames.
//!
//! The mixer passes the graph as a string (`hwupload_cuda`, `scale_cuda`,
//! `tonemap_cuda`, `setparams`, …). Those filters stay FFmpeg's, including the
//! pinned and v210 patches in the mixer image. This node only builds the graph,
//! pushes frames, and pulls frames.
//!
//! Frame rate, size and pixel format come from the input [`Spec::Video`]. The
//! C++ node asked an upstream `IFrameRateSource` and that walk is a stub, so
//! linking `filters.cpp` fails on the first frame of every mixer graph.
//!
//! A software graph is built when the spec arrives. `defer_preliminary_init`,
//! or a hardware pixel format, waits for the first frame so the buffer source
//! can take that frame's `hw_frames_ctx` (FFmpeg 8 checks it at init). Filters
//! flagged `AVFILTER_FLAG_HWDEVICE` get the device before they are initialized;
//! `hwupload_cuda` reads it then, not from the graph afterwards.
//!
//! Pool rotation keeps the graph. A new `AVBufferRef` with the same format,
//! software format and device is the same pool; width and height on the
//! hardware context are the aligned allocation, not the visible size, so they
//! are not compared.
//!
//! `Drain` does not close the buffer source. A null frame is end of stream, and
//! the mixer sends `Drain` at the end of every loop pass and then more frames.
//! The graphs it uses emit one frame per input, and each input is pulled before
//! the next arrives, so nothing is held behind the marker. `Eof` does close the
//! source: the node is finished, and a restart builds a new graph.

use std::ffi::{CString, c_int};
use std::ptr;
use std::sync::{Arc, Mutex};

use rsmpeg::avfilter::{AVFilter, AVFilterContext, AVFilterGraph};
use rsmpeg::avutil::AVFrame;
use rsmpeg::error::RsmpegError;
use rusty_ffmpeg::ffi as av;

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::edge::EdgeEvent;
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::grain::Grain;
use avplumber_f7k::graph::media::{AvpMediaType, AvpRational};
use avplumber_f7k::graph::pad::NodePads;
use avplumber_f7k::graph::spec::Spec;
use avplumber_f7k::libav::codec;
use avplumber_f7k::libav::error::av_error;
use avplumber_f7k::node_api::{Blocking, BlockingIo, EofAction, InputHandler, SingleInput};
use avplumber_f7k::services::hwaccel::HwDevice;

/// `filter_video`. `graph` is a libavfilter description. `threads` is
/// `AVFilterGraph.nb_threads`: 0 lets FFmpeg decide, and the mixer sends 1
/// because the default starts an idle slice thread per CPU per graph.
#[derive(Debug, serde::Deserialize)]
pub struct FilterVideoSpec {
    graph: String,
    #[serde(default)]
    threads: i32,
    #[serde(default)]
    hwaccel: Option<String>,
    #[serde(default)]
    defer_preliminary_init: bool,
}

impl NodeSpec for FilterVideoSpec {
    const TYPE_NAME: &'static str = "filter_video";
    type Node = Blocking<FilterVideo>;

    fn build(self, name: &str, ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        if self.graph.trim().is_empty() {
            return Err("filter_video: graph is empty".into());
        }
        if self.threads < 0 {
            return Err("filter_video: threads must be 0 (FFmpeg's default) or positive".into());
        }
        let hwaccel = match self.hwaccel {
            Some(name) => Some(ctx.hwaccel(&name)?),
            None => None,
        };
        Ok(Blocking(FilterVideo {
            io: BlockingIo::new(name),
            graph: self.graph,
            threads: self.threads,
            defer: self.defer_preliminary_init,
            hwaccel,
            state: Mutex::new(State::default()),
        }))
    }
}

pub struct FilterVideo {
    io: BlockingIo,
    graph: String,
    threads: i32,
    defer: bool,
    hwaccel: Option<Arc<HwDevice>>,
    state: Mutex<State>,
}

#[derive(Default)]
struct State {
    source: Option<Source>,
    built: Option<Built>,
    published: Option<Spec>,
}

/// What the input spec says, which is also what the buffer source is opened with
/// until a frame disagrees.
#[derive(Clone, Copy, PartialEq, Eq)]
struct Source {
    width: i32,
    height: i32,
    pix_fmt: i32,
    sw_pix_fmt: i32,
    frame_rate: AvpRational,
    sar: AvpRational,
    time_base: AvpRational,
}

struct Built {
    session: Session,
    width: i32,
    height: i32,
    pix_fmt: i32,
    time_base: AvpRational,
    /// `None` for a software source. `Some` is the pool the buffer source was
    /// given, compared by format and device rather than by pointer.
    hw: Option<HwKey>,
    output: Spec,
}

struct Session {
    graph: AVFilterGraph,
}

/// Format, software format and device of an `AVHWFramesContext`. Two keys match
/// when all three do. A null device matches nothing: C++ treats a missing
/// `device_ref` as a different pool.
#[derive(Clone, Copy)]
struct HwKey {
    format: i32,
    sw_format: i32,
    device: usize,
}

impl PartialEq for HwKey {
    fn eq(&self, other: &Self) -> bool {
        self.device != 0
            && self.device == other.device
            && self.format == other.format
            && self.sw_format == other.sw_format
    }
}

impl SingleInput for FilterVideo {
    fn io(&self) -> &BlockingIo {
        &self.io
    }

    fn pads(&self) -> NodePads {
        NodePads::siso(AvpMediaType::VIDEO, AvpMediaType::VIDEO)
    }
}

impl InputHandler for FilterVideo {
    fn on_spec(&self, spec: Spec) -> Result<Option<Spec>, NodeError> {
        let incoming =
            source_of(spec).map_err(|message| self.io.error(NodePhase::Spec, message))?;
        // Hardware frames carry their pool on the frame, and a deferred graph
        // wants that pool rather than a guess from the spec.
        let open_now = {
            let state = &mut *self.state.lock().unwrap();
            let same = state.source == Some(incoming);
            state.source = Some(incoming);
            if same && state.built.is_some() {
                return Ok(None);
            }
            state.built = None;
            !(self.defer || codec::is_hw_pix_fmt(incoming.pix_fmt))
        };
        if !open_now {
            return Ok(None);
        }
        let built = self
            .open(&incoming, ptr::null_mut())
            .map_err(|message| self.io.error(NodePhase::Spec, message))?;
        let state = &mut *self.state.lock().unwrap();
        Ok(self.publish(state, built))
    }

    fn on_buffer(&self, buffer: Grain) -> Result<Vec<Grain>, NodeError> {
        let Grain::Video(frame) = buffer else {
            return Err(self.io.error(
                NodePhase::Process,
                format!("filter_video needs video, got {:?}", buffer.media_type()),
            ));
        };
        self.ensure(&frame)?;
        let state = &mut *self.state.lock().unwrap();
        let built = state.built.as_mut().ok_or_else(|| {
            self.io
                .error(NodePhase::Process, "filter_video: graph is not open")
        })?;
        built
            .session
            .push_pull(Some(frame))
            .map_err(|message| self.io.error(NodePhase::Process, message))
    }

    fn on_eof(&self) -> Result<EofAction, NodeError> {
        let state = &mut *self.state.lock().unwrap();
        let frames = match state.built.as_mut() {
            Some(built) => built
                .session
                .push_pull(None)
                .map_err(|message| self.io.error(NodePhase::Process, message))?,
            None => Vec::new(),
        };
        // The null frame closed the source. Drop it so a restart builds another.
        state.built = None;
        Ok(EofAction::Emit(frames))
    }
}

impl FilterVideo {
    /// Opens or keeps the graph for `frame`. A new output spec is pushed here
    /// when the graph could not be built at [`Self::on_spec`].
    fn ensure(&self, frame: &AVFrame) -> Result<(), NodeError> {
        let key = hw_key(frame.hw_frames_ctx);
        let from_frame = {
            let state = &mut *self.state.lock().unwrap();
            let source = state.source.ok_or_else(|| {
                self.io.error(
                    NodePhase::Process,
                    "filter_video: frame arrived before a video spec",
                )
            })?;
            let fits = state.built.as_ref().is_some_and(|built| {
                built.width == frame.width
                    && built.height == frame.height
                    && built.pix_fmt == frame.format
                    && built.time_base == source.time_base
                    && same_pool(built.hw, key)
            });
            if fits {
                None
            } else {
                let mut from_frame = source;
                from_frame.width = frame.width;
                from_frame.height = frame.height;
                from_frame.pix_fmt = frame.format;
                if !frame.hw_frames_ctx.is_null() {
                    from_frame.sw_pix_fmt = codec::frame_sw_pix_fmt(frame);
                }
                Some(from_frame)
            }
        };
        let Some(from_frame) = from_frame else {
            return Ok(());
        };
        let built = self
            .open(&from_frame, frame.hw_frames_ctx)
            .map_err(|message| self.io.error(NodePhase::Process, message))?;
        let publish = {
            let state = &mut *self.state.lock().unwrap();
            state.built = Some(built);
            self.take_unpublished(state)
        };
        if let Some(spec) = publish {
            if let Some(out) = self.io.output_slot.get() {
                out.push_event(EdgeEvent::Spec(spec));
            }
        }
        Ok(())
    }

    fn open(&self, source: &Source, hw_frames: *mut av::AVBufferRef) -> Result<Built, String> {
        open_graph(Open {
            desc: &self.graph,
            source,
            hw_frames,
            device: self.hwaccel.as_deref(),
            threads: self.threads,
            name: &self.io.name,
        })
    }

    /// Stores `built` and returns its spec when downstream has not seen it.
    fn publish(&self, state: &mut State, built: Built) -> Option<Spec> {
        state.built = Some(built);
        self.take_unpublished(state)
    }

    fn take_unpublished(&self, state: &mut State) -> Option<Spec> {
        let output = state.built.as_ref()?.output.clone();
        if state
            .published
            .as_ref()
            .is_some_and(|have| have.same_as(&output))
        {
            return None;
        }
        state.published = Some(output.clone());
        Some(output)
    }
}

struct Open<'a> {
    desc: &'a str,
    source: &'a Source,
    hw_frames: *mut av::AVBufferRef,
    device: Option<&'a HwDevice>,
    threads: i32,
    name: &'a str,
}

fn open_graph(open: Open<'_>) -> Result<Built, String> {
    let mut graph = AVFilterGraph::new();
    // Before any filter is allocated: FFmpeg sizes its slice threads then.
    unsafe {
        rsmpeg::UnsafeDerefMut::deref_mut(&mut graph).nb_threads = open.threads;
    }

    let c_desc = CString::new(open.desc)
        .map_err(|_| format!("filter_video: graph `{}` contains a null byte", open.desc))?;
    let mut raw_segment = ptr::null_mut();
    let ret = unsafe {
        av::avfilter_graph_segment_parse(graph.as_mut_ptr(), c_desc.as_ptr(), 0, &mut raw_segment)
    };
    if ret < 0 {
        if !raw_segment.is_null() {
            unsafe { av::avfilter_graph_segment_free(&mut raw_segment) };
        }
        return Err(format!(
            "filter_video: couldn't parse `{}`: {}",
            open.desc,
            av_error(ret)
        ));
    }
    let segment = Segment(raw_segment);

    let ret = unsafe { av::avfilter_graph_segment_create_filters(segment.0, 0) };
    if ret < 0 {
        return Err(format!(
            "filter_video: couldn't create `{}`: {}",
            open.desc,
            av_error(ret)
        ));
    }
    if let Some(device) = open.device {
        attach_device(&mut graph, device)?;
    }

    let mut inputs = ptr::null_mut();
    let mut outputs = ptr::null_mut();
    let ret = unsafe { av::avfilter_graph_segment_apply(segment.0, 0, &mut inputs, &mut outputs) };
    if ret < 0 {
        free_inout(inputs);
        free_inout(outputs);
        return Err(format!(
            "filter_video: couldn't init `{}`: {}",
            open.desc,
            av_error(ret)
        ));
    }
    drop(segment);

    let input_pad = take_pad(inputs, "input", open.desc).map_err(|message| {
        free_inout(outputs);
        message
    })?;
    let output_pad = take_pad(outputs, "output", open.desc)?;

    let buffer = AVFilter::get_by_name(c"buffer").ok_or("filter_video: no `buffer` filter")?;
    let mut src = graph
        .alloc_filter_context(&buffer, c"in0")
        .ok_or("filter_video: couldn't allocate the buffer source")?;

    let mut hw_frames = open.hw_frames;
    let owned_pool = if hw_frames.is_null() && codec::is_hw_pix_fmt(open.source.pix_fmt) {
        let device = open
            .device
            .ok_or("filter_video: hardware input has no frames context and no hwaccel device")?;
        if open.source.sw_pix_fmt == av::AV_PIX_FMT_NONE {
            return Err("filter_video: hardware input has no software format".into());
        }
        let mut pool = device.frames_ctx(
            open.source.width,
            open.source.height,
            open.source.sw_pix_fmt,
        )?;
        hw_frames = pool.as_mut_ptr();
        Some(pool)
    } else {
        None
    };
    if !hw_frames.is_null() {
        set_hw_frames(&mut src, hw_frames)?;
    }
    let hw = hw_key(hw_frames);
    drop(owned_pool);

    let args = source_args(open.source)?;
    src.init_str(Some(&args))
        .map_err(|e| format!("filter_video: buffer source: {e}"))?;
    let src_ptr = src.as_mut_ptr();
    drop(src);
    let ret = unsafe { av::avfilter_link(src_ptr, 0, input_pad.filter, input_pad.index) };
    if ret < 0 {
        return Err(format!(
            "filter_video: couldn't link the buffer source: {}",
            av_error(ret)
        ));
    }

    let sink_filter =
        AVFilter::get_by_name(c"buffersink").ok_or("filter_video: no `buffersink` filter")?;
    let mut sink = graph
        .create_filter_context(&sink_filter, c"out0", None)
        .map_err(|e| format!("filter_video: buffersink: {e}"))?;
    let sink_ptr = sink.as_mut_ptr();
    drop(sink);
    let ret = unsafe { av::avfilter_link(output_pad.filter, output_pad.index, sink_ptr, 0) };
    if ret < 0 {
        return Err(format!(
            "filter_video: couldn't link the buffer sink: {}",
            av_error(ret)
        ));
    }
    graph
        .config()
        .map_err(|e| format!("filter_video: couldn't configure `{}`: {e}", open.desc))?;

    let sink = graph
        .get_filter(c"out0")
        .ok_or("filter_video: buffer sink is gone after configure")?;
    let output = output_spec(&sink, open.source)?;
    if let Spec::Video { width, height, .. } = &output {
        log::info!(
            "{}: filter `{}` -> {}x{}",
            open.name,
            open.desc,
            width,
            height
        );
    }
    drop(sink);

    Ok(Built {
        session: Session { graph },
        width: open.source.width,
        height: open.source.height,
        pix_fmt: open.source.pix_fmt,
        time_base: open.source.time_base,
        hw,
        output,
    })
}

impl Session {
    fn push_pull(&mut self, frame: Option<AVFrame>) -> Result<Vec<Grain>, String> {
        {
            let mut src = self
                .graph
                .get_filter(c"in0")
                .ok_or("filter_video: buffer source is gone")?;
            src.buffersrc_add_frame(frame, None)
                .map_err(|e| format!("filter_video: buffersrc: {e}"))?;
        }
        let mut sink = self
            .graph
            .get_filter(c"out0")
            .ok_or("filter_video: buffer sink is gone")?;
        let time_base = sink.get_time_base();
        let mut out = Vec::new();
        loop {
            match sink.buffersink_get_frame(None) {
                Ok(mut frame) => {
                    if time_base.num > 0 && time_base.den > 0 {
                        frame.set_time_base(time_base);
                    }
                    out.push(Grain::Video(frame));
                }
                Err(RsmpegError::BufferSinkDrainError) | Err(RsmpegError::BufferSinkEofError) => {
                    break;
                }
                Err(e) => return Err(format!("filter_video: buffersink: {e}")),
            }
        }
        Ok(out)
    }
}

fn source_of(spec: Spec) -> Result<Source, String> {
    let Spec::Video {
        width,
        height,
        pix_fmt,
        sw_pix_fmt,
        frame_rate,
        sar,
        time_base,
    } = spec
    else {
        return Err(format!(
            "filter_video needs a video spec, got {}",
            spec.variant_name()
        ));
    };
    if frame_rate.num <= 0 || frame_rate.den <= 0 {
        return Err("unknown input video frame rate".into());
    }
    if time_base.num <= 0 || time_base.den <= 0 {
        return Err("unknown input time base".into());
    }
    if width <= 0 || height <= 0 {
        return Err(format!("invalid input size {width}x{height}"));
    }
    Ok(Source {
        width,
        height,
        pix_fmt,
        sw_pix_fmt,
        frame_rate,
        sar,
        time_base,
    })
}

fn source_args(source: &Source) -> Result<CString, String> {
    let sar = if source.sar.num > 0 && source.sar.den > 0 {
        source.sar
    } else {
        AvpRational { num: 1, den: 1 }
    };
    let text = format!(
        "time_base={}/{}:video_size={}x{}:pix_fmt={}:pixel_aspect={}/{}:frame_rate={}/{}",
        source.time_base.num,
        source.time_base.den,
        source.width,
        source.height,
        source.pix_fmt,
        sar.num,
        sar.den,
        source.frame_rate.num,
        source.frame_rate.den,
    );
    CString::new(text).map_err(|_| "filter_video: buffer source arguments contain a null".into())
}

fn output_spec(sink: &AVFilterContext, source: &Source) -> Result<Spec, String> {
    let width = sink.get_w();
    let height = sink.get_h();
    if width <= 0 || height <= 0 {
        return Err(format!(
            "filter_video: buffersink negotiated {width}x{height}"
        ));
    }
    let pix_fmt = sink.get_format();
    if pix_fmt == av::AV_PIX_FMT_NONE {
        return Err("filter_video: buffersink has no pixel format".into());
    }
    let frame_rate = rational_or(sink.get_frame_rate(), source.frame_rate);
    let time_base = rational_or(sink.get_time_base(), source.time_base);
    let sar_av = sink.get_sample_aspect_ratio();
    let sar = if sar_av.num > 0 && sar_av.den > 0 {
        AvpRational::from(sar_av)
    } else {
        AvpRational { num: 1, den: 1 }
    };
    Ok(Spec::Video {
        width,
        height,
        pix_fmt,
        sw_pix_fmt: sink_sw_format(sink),
        frame_rate,
        sar,
        time_base,
    })
}

fn rational_or(got: av::AVRational, fallback: AvpRational) -> AvpRational {
    if got.num > 0 && got.den > 0 {
        got.into()
    } else {
        fallback
    }
}

/// The software format of a hardware sink, or `AV_PIX_FMT_NONE` for a software
/// sink. The reference belongs to the sink.
fn sink_sw_format(sink: &AVFilterContext) -> i32 {
    let buffer = unsafe { av::av_buffersink_get_hw_frames_ctx(sink.as_ptr()) };
    if buffer.is_null() {
        return av::AV_PIX_FMT_NONE;
    }
    let data = unsafe { (*buffer).data } as *const av::AVHWFramesContext;
    if data.is_null() {
        return av::AV_PIX_FMT_NONE;
    }
    unsafe { (*data).sw_format }
}

fn hw_key(hw_frames: *mut av::AVBufferRef) -> Option<HwKey> {
    if hw_frames.is_null() {
        return None;
    }
    // Safety: a non-null frames context is an `AVHWFramesContext`, by libavutil.
    unsafe {
        let data = (*hw_frames).data as *const av::AVHWFramesContext;
        if data.is_null() {
            return None;
        }
        let device_ref = (*data).device_ref;
        let device = if device_ref.is_null() {
            0
        } else {
            (*device_ref).data as usize
        };
        Some(HwKey {
            format: (*data).format,
            sw_format: (*data).sw_format,
            device,
        })
    }
}

/// `None` then `Some` is a pool appearing, which has to rebuild the source.
/// `Some` then `None` is a frame that simply doesn't repeat the pointer; the
/// pool already on the source still describes it.
fn same_pool(opened: Option<HwKey>, frame: Option<HwKey>) -> bool {
    match (opened, frame) {
        (None, None) | (Some(_), None) => true,
        (None, Some(_)) => false,
        (Some(a), Some(b)) => a == b,
    }
}

/// `hwupload_cuda` is initialized by `avfilter_graph_segment_apply`, and FFmpeg
/// 8 requires the device on the filter before that. Setting it on the graph
/// afterwards is too late.
fn attach_device(graph: &mut AVFilterGraph, device: &HwDevice) -> Result<(), String> {
    let count = graph.nb_filters as usize;
    let filters = graph.filters;
    if count > 0 && filters.is_null() {
        return Err("filter_video: graph filter list is null".into());
    }
    for index in 0..count {
        // Safety: `nb_filters` is the length of this array, owned by the graph.
        let fctx = unsafe { *filters.add(index) };
        if fctx.is_null() {
            continue;
        }
        let filter = unsafe { (*fctx).filter };
        if filter.is_null() {
            continue;
        }
        let flags = unsafe { (*filter).flags };
        if flags & av::AVFILTER_FLAG_HWDEVICE as c_int == 0 {
            continue;
        }
        let referenced = unsafe { av::av_buffer_ref(device.as_ptr()) };
        if referenced.is_null() {
            return Err("filter_video: av_buffer_ref of the hardware device failed".into());
        }
        // The filter context frees this reference with the graph.
        unsafe { (*fctx).hw_device_ctx = referenced };
    }
    Ok(())
}

fn set_hw_frames(src: &mut AVFilterContext, hw_frames: *mut av::AVBufferRef) -> Result<(), String> {
    let params = unsafe { av::av_buffersrc_parameters_alloc() };
    if params.is_null() {
        return Err("filter_video: av_buffersrc_parameters_alloc failed".into());
    }
    let params = BufferSrcParams(params);
    let referenced = unsafe { av::av_buffer_ref(hw_frames) };
    if referenced.is_null() {
        return Err("filter_video: av_buffer_ref of hw_frames_ctx failed".into());
    }
    unsafe { (*params.0).hw_frames_ctx = referenced };
    let ret = unsafe { av::av_buffersrc_parameters_set(src.as_mut_ptr(), params.0) };
    if ret < 0 {
        return Err(format!(
            "filter_video: av_buffersrc_parameters_set: {}",
            av_error(ret)
        ));
    }
    Ok(())
}

struct Pad {
    filter: *mut av::AVFilterContext,
    index: u32,
}

/// Copies the single pad and frees the list. More or fewer than one pad is an
/// error: the mixer uses one input and one output, and so does this node.
fn take_pad(mut list: *mut av::AVFilterInOut, which: &str, graph: &str) -> Result<Pad, String> {
    if list.is_null() {
        return Err(format!("filter_video: graph `{graph}` has no {which} pad"));
    }
    let mut count = 0usize;
    let mut cursor = list;
    let mut pad = Pad {
        filter: ptr::null_mut(),
        index: 0,
    };
    while !cursor.is_null() {
        if count == 0 {
            pad = unsafe {
                Pad {
                    filter: (*cursor).filter_ctx,
                    index: (*cursor).pad_idx as u32,
                }
            };
        }
        count += 1;
        cursor = unsafe { (*cursor).next };
    }
    unsafe { av::avfilter_inout_free(&mut list) };
    if count != 1 {
        return Err(format!(
            "filter_video: graph `{graph}` has {count} {which} pad(s); this node has one"
        ));
    }
    if pad.filter.is_null() {
        return Err(format!(
            "filter_video: graph `{graph}` {which} pad has no filter"
        ));
    }
    Ok(pad)
}

fn free_inout(mut list: *mut av::AVFilterInOut) {
    if !list.is_null() {
        unsafe { av::avfilter_inout_free(&mut list) };
    }
}

/// Frees the segment before the graph. Filter contexts belong to the graph.
struct Segment(*mut av::AVFilterGraphSegment);

impl Drop for Segment {
    fn drop(&mut self) {
        if !self.0.is_null() {
            unsafe { av::avfilter_graph_segment_free(&mut self.0) };
        }
    }
}

/// Frees the parameter block and the frames reference it holds. `parameters_set`
/// has its own reference by then.
struct BufferSrcParams(*mut av::AVBufferSrcParameters);

impl Drop for BufferSrcParams {
    fn drop(&mut self) {
        if self.0.is_null() {
            return;
        }
        unsafe {
            if !(*self.0).hw_frames_ctx.is_null() {
                av::av_buffer_unref(&mut (*self.0).hw_frames_ctx);
            }
            av::av_freep(ptr::addr_of_mut!(self.0).cast());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::{Value, json};

    use avplumber_f7k::Instance;
    use avplumber_f7k::factory::{BuildCtx, NodeSpec};
    use avplumber_f7k::graph::buffered_edge::BufferedEdge;
    use avplumber_f7k::graph::edge::{Edge, EdgeEvent, EdgeItem};
    use avplumber_f7k::graph::grain::Grain;
    use avplumber_f7k::graph::media::AvpRational;
    use avplumber_f7k::graph::node::{Node, Processed};
    use avplumber_f7k::graph::spec::Spec;
    use rsmpeg::avutil::AVFrame;
    use rusty_ffmpeg::ffi;

    use super::FilterVideoSpec;

    const GRAPH: &str = "scale=16:16,setparams=range=tv";

    fn video_spec(rate_num: i32) -> Spec {
        Spec::Video {
            width: 8,
            height: 8,
            pix_fmt: ffi::AV_PIX_FMT_GRAY8,
            sw_pix_fmt: ffi::AV_PIX_FMT_NONE,
            frame_rate: AvpRational {
                num: rate_num,
                den: 1,
            },
            sar: AvpRational { num: 1, den: 1 },
            time_base: AvpRational { num: 1, den: 30 },
        }
    }

    fn gray(pts: i64) -> Grain {
        let mut frame = AVFrame::new();
        frame.set_width(8);
        frame.set_height(8);
        frame.set_format(ffi::AV_PIX_FMT_GRAY8);
        frame.alloc_buffer().expect("8x8 gray");
        frame.set_pts(pts);
        frame.set_time_base(ffi::AVRational { num: 1, den: 30 });
        Grain::Video(frame)
    }

    fn build(params: Value) -> Result<avplumber_f7k::Blocking<super::FilterVideo>, String> {
        let spec: FilterVideoSpec =
            serde_json::from_value(params.clone()).map_err(|e| e.to_string())?;
        let instance = Instance::new();
        spec.build(
            "fv",
            &BuildCtx {
                instance: &instance,
                name: "fv",
                params: &params,
                sync_group: None,
            },
        )
    }

    struct Harness {
        node: avplumber_f7k::Blocking<super::FilterVideo>,
        input: Arc<dyn Edge>,
        out: Arc<dyn Edge>,
    }

    impl Harness {
        fn new(params: Value) -> Self {
            let node = build(params).unwrap();
            let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(8));
            let out: Arc<dyn Edge> = Arc::new(BufferedEdge::new(8));
            node.bind_source("in", input.clone());
            node.bind_sink("out", out.clone());
            node.start();
            Self { node, input, out }
        }

        fn push_spec(&self, rate: i32) {
            self.input.push_event(EdgeEvent::Spec(video_spec(rate)));
        }

        fn push_frame(&self, pts: i64) {
            assert!(self.input.offer(gray(pts)).is_ok());
        }

        fn push_drain(&self) {
            self.input.push_event(EdgeEvent::Drain);
        }

        fn step(&self) -> Processed {
            self.node.process().expect("filter step")
        }

        fn take(&self) -> Vec<EdgeItem> {
            let mut items = Vec::new();
            while let Some(item) = self.out.try_take() {
                items.push(item);
            }
            items
        }
    }

    fn video(item: EdgeItem) -> AVFrame {
        match item {
            EdgeItem::Buffer(Grain::Video(frame)) => frame,
            EdgeItem::Buffer(_) => panic!("expected a video frame"),
            EdgeItem::Event(event) => panic!("expected a video frame, got {event:?}"),
        }
    }

    #[test]
    fn scale_and_setparams_publish_a_spec_and_a_frame() {
        let harness = Harness::new(json!({"graph": GRAPH, "threads": 1}));
        harness.push_spec(30);
        assert!(matches!(harness.step(), Processed::Again));
        let spec_items = harness.take();
        let [
            EdgeItem::Event(EdgeEvent::Spec(Spec::Video {
                width,
                height,
                pix_fmt,
                ..
            })),
        ] = &spec_items[..]
        else {
            panic!("expected one video spec");
        };
        assert_eq!((*width, *height, *pix_fmt), (16, 16, ffi::AV_PIX_FMT_GRAY8));

        harness.push_frame(42);
        assert!(matches!(harness.step(), Processed::Again));
        let frame = video(harness.take().pop().expect("a frame"));
        assert_eq!((frame.width, frame.height), (16, 16));
        assert_eq!(frame.format, ffi::AV_PIX_FMT_GRAY8);
        assert_eq!(frame.pts, 42);
        assert_eq!(frame.color_range, ffi::AVCOL_RANGE_MPEG);
        assert_eq!(frame.time_base.den, 30);

        harness.push_frame(43);
        assert!(matches!(harness.step(), Processed::Again));
        let again = harness.take();
        assert!(
            matches!(&again[..], [EdgeItem::Buffer(_)]),
            "a second frame does not republish the spec"
        );
        assert_eq!(video(again.into_iter().next().unwrap()).pts, 43);
    }

    #[test]
    fn drain_keeps_the_graph_open_for_the_next_frame() {
        let harness = Harness::new(json!({"graph": GRAPH}));
        harness.push_spec(30);
        harness.step();
        harness.take();

        harness.push_frame(1);
        harness.step();
        harness.push_drain();
        harness.step();
        harness.push_frame(2);
        harness.step();

        let items = harness.take();
        assert!(
            matches!(
                &items[..],
                [
                    EdgeItem::Buffer(_),
                    EdgeItem::Event(EdgeEvent::Drain),
                    EdgeItem::Buffer(_),
                ]
            ),
            "frame, drain, frame"
        );
        let mut frames = items.into_iter();
        assert_eq!(video(frames.next().unwrap()).pts, 1);
        frames.next();
        assert_eq!(video(frames.next().unwrap()).pts, 2);
    }

    #[test]
    fn eof_flushes_and_finishes() {
        let harness = Harness::new(json!({"graph": GRAPH}));
        harness.push_spec(30);
        harness.step();
        harness.take();
        harness.push_frame(1);
        harness.step();
        harness.input.push_event(EdgeEvent::Eof);
        assert!(matches!(harness.step(), Processed::Done));
        let items = harness.take();
        assert!(matches!(
            &items[..],
            [EdgeItem::Buffer(_), EdgeItem::Event(EdgeEvent::Eof)]
        ));
        assert_eq!(video(items.into_iter().next().unwrap()).pts, 1);
    }

    #[test]
    fn a_flush_does_not_drop_the_graph() {
        let harness = Harness::new(json!({"graph": GRAPH}));
        harness.push_spec(30);
        harness.step();
        harness.take();
        harness.push_frame(1);
        harness.step();
        harness.input.push_event(EdgeEvent::FlushStart);
        harness.step();
        harness.push_frame(2);
        harness.step();
        let mut items = harness.take();
        assert!(matches!(items.pop(), Some(EdgeItem::Buffer(_))));
        assert!(matches!(
            items.pop(),
            Some(EdgeItem::Event(EdgeEvent::FlushStart))
        ));
    }

    #[test]
    fn defer_waits_for_the_frame_to_publish_the_spec() {
        let harness = Harness::new(json!({"graph": GRAPH, "defer_preliminary_init": true}));
        harness.push_spec(30);
        assert!(matches!(harness.step(), Processed::Again));
        assert!(harness.take().is_empty());

        harness.push_frame(7);
        assert!(matches!(harness.step(), Processed::Again));
        let items = harness.take();
        assert!(matches!(
            &items[..],
            [
                EdgeItem::Event(EdgeEvent::Spec(Spec::Video {
                    width: 16,
                    height: 16,
                    ..
                })),
                EdgeItem::Buffer(_),
            ]
        ));
        assert_eq!(video(items.into_iter().nth(1).unwrap()).pts, 7);
    }

    #[test]
    fn missing_frame_rate_is_an_error() {
        let harness = Harness::new(json!({"graph": "null"}));
        harness.push_spec(0);
        let err = harness.node.process().unwrap_err();
        assert!(err.to_string().contains("frame rate"), "{err}");
    }

    #[test]
    fn an_unknown_filter_fails_when_the_spec_arrives() {
        let harness = Harness::new(json!({"graph": "not_a_real_filter"}));
        harness.push_spec(30);
        let err = harness.node.process().unwrap_err();
        assert!(err.to_string().contains("not_a_real_filter"), "{err}");
    }

    #[test]
    fn a_source_only_graph_is_rejected() {
        let harness = Harness::new(json!({"graph": "color=c=black:s=16x16:r=30"}));
        harness.push_spec(30);
        let err = harness.node.process().unwrap_err();
        assert!(err.to_string().contains("input"), "{err}");
    }

    #[test]
    fn empty_graph_and_bad_threads_and_unknown_device_fail_at_build() {
        let err = match build(json!({"graph": "  "})) {
            Err(err) => err,
            Ok(_) => panic!("empty graph was accepted"),
        };
        assert!(err.contains("empty"), "{err}");
        let err = match build(json!({"graph": "null", "threads": -1})) {
            Err(err) => err,
            Ok(_) => panic!("negative threads were accepted"),
        };
        assert!(err.contains("threads"), "{err}");
        let err = match build(json!({"graph": "null", "hwaccel": "nope"})) {
            Err(err) => err,
            Ok(_) => panic!("unknown device was accepted"),
        };
        assert!(err.contains("nope"), "{err}");
    }
}
