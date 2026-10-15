//! One CUDA frame through the C++ `cuda_transform` node on the Rust graph.
//!
//! The node is the mixer's canvas draw (crop, scale, place), not a libavfilter
//! wrapper. The input is a 64×64 NV12 surface on the device `hwaccel.init`
//! opened; the canvas is 32×32, so the compositor kernel has to run. A 1:1
//! canvas would pass the input frame through undrawn.
//!
//! `cargo test -p avplumber_nodes --features cpp_cuda,ffmpeg9 --test cpp_cuda_transform`
//! with `FFMPEG_PREFIX` pointing at a CUDA build of the patched FFmpeg, and
//! `libavcpp.a` built against that same tree. `NVCC` builds the kernel image.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use avplumber_f7k::control;
use avplumber_f7k::services::hwaccel::HwDevice;
use avplumber_f7k::{
    AvpMediaType, Edge, EdgeEvent, EdgeItem, Grain, Instance, Node, NodeError, NodeKind, NodePads,
    NodePhase, Processed, Push, register_factory,
};
use rsmpeg::UnsafeDerefMut;
use rsmpeg::avutil::{AVFrame, AVHWFramesContext};
use rusty_ffmpeg::ffi;

const PTS: i64 = 12_345;
const SRC: i32 = 64;
const CANVAS: i32 = 32;

#[derive(Clone, Debug)]
struct Drawn {
    width: i32,
    height: i32,
    format: i32,
    sw_format: i32,
    pts: i64,
    time_base_num: i32,
    time_base_den: i32,
    center_y: u8,
}

struct FrameSource {
    name: String,
    sink: OnceLock<Arc<dyn Edge>>,
    sent: AtomicBool,
    device: Arc<HwDevice>,
}

impl FrameSource {
    fn new(name: String, device: Arc<HwDevice>) -> Self {
        Self {
            name,
            sink: OnceLock::new(),
            sent: AtomicBool::new(false),
            device,
        }
    }
}

impl Node for FrameSource {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Blocking
    }
    fn pads(&self) -> NodePads {
        NodePads::output(AvpMediaType::VIDEO)
    }
    fn bind_sink(&self, _pad: &str, edge: Arc<dyn Edge>) {
        let _ = self.sink.set(edge);
    }
    fn process(&self) -> Result<Processed, NodeError> {
        if self.sent.swap(true, Ordering::AcqRel) {
            return Ok(Processed::Done);
        }
        let sink = self.sink.get().expect("sink bound before process");
        let frame = cuda_nv12(&self.device).map_err(|err| {
            NodeError::new(&self.name, NodePhase::Process, format!("cuda frame: {err}"))
        })?;
        match sink.push(Grain::Video(frame)) {
            Push::Accepted => {}
            other => {
                return Err(NodeError::new(
                    &self.name,
                    NodePhase::Process,
                    format!("source push: {other:?}"),
                ));
            }
        }
        sink.push_event(EdgeEvent::Eof);
        Ok(Processed::Done)
    }
}

struct FrameSink {
    name: String,
    source: OnceLock<Arc<dyn Edge>>,
    seen: Arc<Mutex<Vec<Drawn>>>,
}

impl FrameSink {
    fn new(name: String, seen: Arc<Mutex<Vec<Drawn>>>) -> Self {
        Self {
            name,
            source: OnceLock::new(),
            seen,
        }
    }
}

impl Node for FrameSink {
    fn name(&self) -> &str {
        &self.name
    }
    fn kind(&self) -> NodeKind {
        NodeKind::Blocking
    }
    fn pads(&self) -> NodePads {
        NodePads::input(AvpMediaType::VIDEO)
    }
    fn bind_source(&self, _pad: &str, edge: Arc<dyn Edge>) {
        let _ = self.source.set(edge);
    }
    fn process(&self) -> Result<Processed, NodeError> {
        let source = self.source.get().expect("source bound before process");
        match source.take(-1) {
            Some(EdgeItem::Buffer(Grain::Video(frame))) => {
                let drawn = inspect(&frame).map_err(|err| {
                    NodeError::new(&self.name, NodePhase::Process, format!("inspect: {err}"))
                })?;
                self.seen.lock().unwrap().push(drawn);
                Ok(Processed::Again)
            }
            Some(EdgeItem::Event(EdgeEvent::Eof)) => Ok(Processed::Done),
            Some(EdgeItem::Buffer(_)) => Err(NodeError::new(
                &self.name,
                NodePhase::Process,
                "unexpected non-video buffer",
            )),
            Some(EdgeItem::Event(_)) => Err(NodeError::new(
                &self.name,
                NodePhase::Process,
                "unexpected edge event",
            )),
            None => Ok(Processed::Again),
        }
    }
}

fn cuda_nv12(device: &HwDevice) -> Result<AVFrame, String> {
    let mut pool: AVHWFramesContext = device
        .frames_ctx(SRC, SRC, ffi::AV_PIX_FMT_NV12)
        .map_err(|err| format!("frames ctx: {err}"))?;
    let mut software = software_nv12()?;
    let mut hardware = AVFrame::new();
    pool.get_buffer(&mut hardware)
        .map_err(|err| format!("av_hwframe_get_buffer: {err}"))?;
    let rc = unsafe {
        ffi::av_hwframe_transfer_data(hardware.as_mut_ptr(), software.as_mut_ptr(), 0)
    };
    if rc < 0 {
        return Err(format!("av_hwframe_transfer_data: {rc}"));
    }
    hardware.set_pts(PTS);
    hardware.set_time_base(ffi::AVRational { num: 1, den: 1000 });
    Ok(hardware)
}

fn software_nv12() -> Result<AVFrame, String> {
    let mut frame = AVFrame::new();
    frame.set_width(SRC);
    frame.set_height(SRC);
    frame.set_format(ffi::AV_PIX_FMT_NV12);
    frame
        .alloc_buffer()
        .map_err(|err| format!("nv12 alloc: {err}"))?;
    unsafe {
        let raw = UnsafeDerefMut::deref_mut(&mut frame);
        fill_plane(raw.data[0], raw.linesize[0], SRC as usize, SRC as usize, 180);
        fill_plane(
            raw.data[1],
            raw.linesize[1],
            SRC as usize,
            (SRC / 2) as usize,
            128,
        );
    }
    Ok(frame)
}

fn fill_plane(data: *mut u8, stride: i32, width: usize, height: usize, value: u8) {
    assert!(!data.is_null());
    let stride = stride as usize;
    for row in 0..height {
        for col in 0..width {
            unsafe {
                *data.add(row * stride + col) = value;
            }
        }
    }
}

fn inspect(frame: &AVFrame) -> Result<Drawn, String> {
    let raw = unsafe { &*frame.as_ptr() };
    if raw.hw_frames_ctx.is_null() {
        return Err("output frame has no hw_frames_ctx".into());
    }
    let sw_format = unsafe {
        let ctx = (*raw.hw_frames_ctx).data as *const ffi::AVHWFramesContext;
        (*ctx).sw_format
    };
    let mut software = AVFrame::new();
    software.set_format(ffi::AV_PIX_FMT_NV12);
    software.set_width(raw.width);
    software.set_height(raw.height);
    software
        .alloc_buffer()
        .map_err(|err| format!("download alloc: {err}"))?;
    let rc =
        unsafe { ffi::av_hwframe_transfer_data(software.as_mut_ptr(), frame.as_ptr() as *mut _, 0) };
    if rc < 0 {
        return Err(format!("av_hwframe_transfer_data (download): {rc}"));
    }
    let downloaded = unsafe { &*software.as_ptr() };
    let center_y = unsafe {
        *downloaded.data[0].add(
            (CANVAS as usize / 2) * downloaded.linesize[0] as usize + (CANVAS as usize / 2),
        )
    };
    Ok(Drawn {
        width: raw.width,
        height: raw.height,
        format: raw.format,
        sw_format,
        pts: raw.pts,
        time_base_num: raw.time_base.num,
        time_base_den: raw.time_base.den,
        center_y,
    })
}

struct StopGroup<'a>(&'a Instance);

impl Drop for StopGroup<'_> {
    fn drop(&mut self) {
        let _ = control::exec_line(self.0, "group.stop g");
    }
}

#[test]
fn cuda_transform_scales_a_cuda_frame() {
    let inst = Instance::new();
    avplumber_nodes::register_media_nodes(&inst);
    control::exec_line(
        &inst,
        r#"hwaccel.init {"name":"gpu","type":"cuda"}"#,
    )
    .unwrap_or_else(|err| panic!("hwaccel.init: {err}"));
    let device = inst
        .services()
        .hwaccels
        .get("gpu")
        .unwrap_or_else(|err| panic!("hwaccel gpu: {err}"));

    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_factory = seen.clone();
    register_factory(&inst, "frame_source", move |name, _params| {
        Ok(Arc::new(FrameSource::new(name.to_string(), device.clone())))
    });
    register_factory(&inst, "frame_sink", move |name, _params| {
        Ok(Arc::new(FrameSink::new(
            name.to_string(),
            seen_factory.clone(),
        )))
    });

    let _stop = StopGroup(&inst);
    let draw = r#"node.add {"type":"cuda_transform","name":"draw","group":"g","hwaccel":"gpu","src":"raw","dst":"canvas","outputs":[{"dst":"canvas","width":32,"height":32,"sw_format":"nv12","layers":[{"dst_x":0,"dst_y":0,"dst_w":32,"dst_h":32}]}]}"#;
    for line in [
        r#"node.add {"type":"frame_source","name":"src","group":"g","dst":"raw"}"#,
        draw,
        r#"node.add {"type":"frame_sink","name":"sink","group":"g","src":"canvas"}"#,
        "group.start g",
    ] {
        control::exec_line(&inst, line).unwrap_or_else(|err| panic!("{line}: {err}"));
    }

    let deadline = Instant::now() + Duration::from_secs(20);
    let mut outcomes = Vec::new();
    while Instant::now() < deadline {
        let status = control::exec_line(&inst, "group.status g").expect("group.status");
        let status: serde_json::Value = serde_json::from_str(&status).expect("group.status json");
        outcomes = status["outcomes"]
            .as_array()
            .expect("outcomes")
            .iter()
            .map(|outcome| outcome.as_str().expect("outcome string").to_owned())
            .collect();
        for outcome in &outcomes {
            assert!(
                !outcome.starts_with("failed:") && !outcome.starts_with("panicked:"),
                "node failed: {outcome}; all outcomes: {outcomes:?}"
            );
        }
        if outcomes.iter().any(|o| o.starts_with("completed:sink:"))
            && outcomes.iter().any(|o| o.starts_with("completed:draw:"))
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        outcomes.iter().any(|o| o.starts_with("completed:draw:"))
            && outcomes.iter().any(|o| o.starts_with("completed:sink:")),
        "group did not finish cleanly: {outcomes:?}"
    );

    let frames = seen.lock().unwrap().clone();
    assert_eq!(frames.len(), 1, "expected one canvas frame, got {frames:?}");
    let frame = &frames[0];
    assert_eq!(frame.width, CANVAS);
    assert_eq!(frame.height, CANVAS);
    assert_eq!(frame.format, ffi::AV_PIX_FMT_CUDA);
    assert_eq!(frame.sw_format, ffi::AV_PIX_FMT_NV12);
    assert_eq!(frame.pts, PTS);
    assert_eq!(frame.time_base_num, 1);
    assert_eq!(frame.time_base_den, 1000);
    assert!(
        frame.center_y > 100,
        "center luma is {}, the canvas looks undrawn or black: {frame:?}",
        frame.center_y
    );
}
