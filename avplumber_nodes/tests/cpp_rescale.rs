//! Software `AVFrame` through the C++ `rescale_video` node on the Rust graph.
//!
//! `cpp_nodes` implies `ffmpeg` and `async`. The ABI selector is separate:
//! `cargo test -p avplumber_nodes --features cpp_nodes,ffmpeg9 --test cpp_rescale`
//! with `FFMPEG_PREFIX` pointing at the same FFmpeg `libavcpp.a` was built against.

use std::ffi::CString;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use avplumber_f7k::control;
use avplumber_f7k::{
    AvpMediaType, Edge, EdgeEvent, EdgeItem, Grain, Instance, Node, NodeError, NodeKind, NodePads,
    NodePhase, Processed, Push, register_factory,
};
use rsmpeg::UnsafeDerefMut;
use rsmpeg::avutil::AVFrame;
use rusty_ffmpeg::ffi;

const PTS: i64 = 12_345;
const TAG_KEY: &str = "avp_tag";
const TAG_VAL: &str = "kept";
const SIDE_MAGIC: [u8; 4] = [0xde, 0xad, 0xbe, 0xef];

#[derive(Clone, Debug)]
struct Scaled {
    width: i32,
    height: i32,
    format: i32,
    pts: i64,
    time_base_num: i32,
    time_base_den: i32,
    tag: Option<String>,
    side_data: bool,
    center: [u8; 3],
}

struct FrameSource {
    name: String,
    sink: OnceLock<Arc<dyn Edge>>,
    sent: AtomicBool,
}

impl FrameSource {
    fn new(name: String) -> Self {
        Self {
            name,
            sink: OnceLock::new(),
            sent: AtomicBool::new(false),
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
        match sink.push(Grain::Video(software_frame())) {
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
    seen: Arc<Mutex<Vec<Scaled>>>,
}

impl FrameSink {
    fn new(name: String, seen: Arc<Mutex<Vec<Scaled>>>) -> Self {
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
                self.seen.lock().unwrap().push(inspect(&frame));
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

fn software_frame() -> AVFrame {
    let mut frame = AVFrame::new();
    frame.set_width(64);
    frame.set_height(64);
    frame.set_format(ffi::AV_PIX_FMT_YUV420P);
    frame.alloc_buffer().expect("64x64 yuv420p");
    frame.set_pts(PTS);
    frame.set_time_base(ffi::AVRational { num: 1, den: 1000 });
    unsafe {
        let raw = UnsafeDerefMut::deref_mut(&mut frame);
        fill_plane(raw.data[0], raw.linesize[0], 64, 64, 180);
        fill_plane(raw.data[1], raw.linesize[1], 32, 32, 128);
        fill_plane(raw.data[2], raw.linesize[2], 32, 32, 128);
        let key = CString::new(TAG_KEY).unwrap();
        let val = CString::new(TAG_VAL).unwrap();
        let rc = ffi::av_dict_set(&mut raw.metadata, key.as_ptr(), val.as_ptr(), 0);
        assert_eq!(rc, 0, "av_dict_set");
        let side = ffi::av_frame_new_side_data(raw, ffi::AV_FRAME_DATA_A53_CC, SIDE_MAGIC.len());
        assert!(!side.is_null(), "av_frame_new_side_data");
        std::ptr::copy_nonoverlapping(SIDE_MAGIC.as_ptr(), (*side).data, SIDE_MAGIC.len());
    }
    frame
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

fn inspect(frame: &AVFrame) -> Scaled {
    let raw = unsafe { &*frame.as_ptr() };
    let mut center = [0u8; 3];
    if raw.width == 32 && raw.height == 32 && raw.format == ffi::AV_PIX_FMT_RGB24 {
        let stride = raw.linesize[0] as usize;
        unsafe {
            let p = raw.data[0].add(16 * stride + 16 * 3);
            center = [*p, *p.add(1), *p.add(2)];
        }
    }
    Scaled {
        width: raw.width,
        height: raw.height,
        format: raw.format,
        pts: raw.pts,
        time_base_num: raw.time_base.num,
        time_base_den: raw.time_base.den,
        tag: dict_get(raw.metadata, TAG_KEY),
        side_data: has_side_magic(raw),
        center,
    }
}

fn dict_get(dict: *mut ffi::AVDictionary, key: &str) -> Option<String> {
    let key = CString::new(key).unwrap();
    let entry = unsafe { ffi::av_dict_get(dict, key.as_ptr(), std::ptr::null(), 0) };
    if entry.is_null() {
        return None;
    }
    let value = unsafe { std::ffi::CStr::from_ptr((*entry).value) };
    Some(value.to_string_lossy().into_owned())
}

fn has_side_magic(raw: &ffi::AVFrame) -> bool {
    for i in 0..raw.nb_side_data {
        let side = unsafe { &**raw.side_data.add(i as usize) };
        if side.type_ == ffi::AV_FRAME_DATA_A53_CC
            && side.size == SIDE_MAGIC.len()
            && unsafe { std::slice::from_raw_parts(side.data, side.size) } == SIDE_MAGIC
        {
            return true;
        }
    }
    false
}

struct StopGroup<'a>(&'a Instance);

impl Drop for StopGroup<'_> {
    fn drop(&mut self) {
        let _ = control::exec_line(self.0, "group.stop g");
    }
}

#[test]
fn rescale_video_scales_a_software_frame() {
    let inst = Instance::new();
    avplumber_nodes::register_media_nodes(&inst);
    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_factory = seen.clone();
    register_factory(&inst, "frame_source", |name, _params| {
        Ok(Arc::new(FrameSource::new(name.to_string())))
    });
    register_factory(&inst, "frame_sink", move |name, _params| {
        Ok(Arc::new(FrameSink::new(
            name.to_string(),
            seen_factory.clone(),
        )))
    });

    let _stop = StopGroup(&inst);
    for line in [
        r#"node.add {"type":"frame_source","name":"src","group":"g","dst":"raw"}"#,
        r#"node.add {"type":"rescale_video","name":"scale","group":"g","src":"raw","dst":"scaled","dst_width":32,"dst_height":32,"dst_pixel_format":"rgb24"}"#,
        r#"node.add {"type":"frame_sink","name":"sink","group":"g","src":"scaled"}"#,
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
            && outcomes.iter().any(|o| o.starts_with("completed:scale:"))
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    assert!(
        outcomes.iter().any(|o| o.starts_with("completed:scale:"))
            && outcomes.iter().any(|o| o.starts_with("completed:sink:")),
        "group did not finish cleanly: {outcomes:?}"
    );

    let frames = seen.lock().unwrap().clone();
    assert_eq!(frames.len(), 1, "expected one scaled frame, got {frames:?}");
    let frame = &frames[0];
    assert_eq!(frame.width, 32);
    assert_eq!(frame.height, 32);
    assert_eq!(frame.format, ffi::AV_PIX_FMT_RGB24);
    assert_eq!(frame.pts, PTS);
    assert_eq!(frame.time_base_num, 1);
    assert_eq!(frame.time_base_den, 1000);
    assert_eq!(frame.tag.as_deref(), Some(TAG_VAL));
    assert!(frame.side_data, "A53 side data was not preserved");
    for (channel, value) in frame.center.iter().enumerate() {
        assert!(
            *value > 100,
            "center channel {channel} is {value}, frame looks unscaled or black: {frame:?}"
        );
    }
}
