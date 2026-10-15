//! Hard cut between two CUDA sources on the Rust graph.
//!
//! Two 64×64 NV12 sources fan out through `one_to_many` into two
//! `mixer_compositor` slots. Each slot's post fan-out is also `one_to_many`:
//! `mixer.cut` sets that mask, then the `mixer_selector` commits when the
//! preview slot's frame carries the new `avp.mixer.scene` revision.
//! `mixer_snapshot` sits on the selector output and forwards it. A cut does
//! not use `transition_cuda`.
//!
//! `cargo test -p avplumber_nodes --features cpp_mixer,ffmpeg9 --test cpp_mixer_cut`
//! on a machine with nvcc and the patched CUDA FFmpeg.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use avplumber_f7k::control;
use avplumber_f7k::services::hwaccel::HwDevice;
use avplumber_f7k::{
    AvpMediaType, Edge, EdgeItem, Grain, Instance, Node, NodeError, NodeKind, NodePads, NodePhase,
    Processed, Push, register_factory,
};
use rsmpeg::UnsafeDerefMut;
use rsmpeg::avutil::{AVFrame, AVHWFramesContext};
use rusty_ffmpeg::ffi;

const SIZE: i32 = 64;
const DARK: u8 = 40;
const BRIGHT: u8 = 220;

struct FrameSource {
    name: String,
    sink: OnceLock<Arc<dyn Edge>>,
    device: Arc<HwDevice>,
    luma: u8,
}

impl FrameSource {
    fn new(name: String, device: Arc<HwDevice>, luma: u8) -> Self {
        Self {
            name,
            sink: OnceLock::new(),
            device,
            luma,
        }
    }
}

/// Same clock as the mixer's `monotonicNs()` (`std::chrono::steady_clock`).
/// A cut discards input stamped before that instant, so elapsed-since-start
/// timestamps never become eligible.
fn monotonic_ns() -> i64 {
    #[repr(C)]
    struct Timespec {
        tv_sec: i64,
        tv_nsec: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(clk_id: i32, tp: *mut Timespec) -> i32;
    }
    let mut ts = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let rc = unsafe { clock_gettime(1, &mut ts) };
    assert_eq!(rc, 0, "clock_gettime");
    ts.tv_sec
        .saturating_mul(1_000_000_000)
        .saturating_add(ts.tv_nsec)
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
        let sink = self.sink.get().expect("sink bound before process");
        let frame = cuda_nv12(&self.device, self.luma, monotonic_ns()).map_err(|err| {
            NodeError::new(&self.name, NodePhase::Process, format!("cuda frame: {err}"))
        })?;
        match sink.push(Grain::Video(frame)) {
            Push::Accepted => {
                std::thread::sleep(Duration::from_millis(30));
                Ok(Processed::Again)
            }
            Push::Full | Push::Dropped => Ok(Processed::Again),
            Push::Closed => Ok(Processed::Done),
        }
    }
}

struct FrameSink {
    name: String,
    source: OnceLock<Arc<dyn Edge>>,
    seen: Arc<Mutex<Vec<u8>>>,
}

impl FrameSink {
    fn new(name: String, seen: Arc<Mutex<Vec<u8>>>) -> Self {
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
                let y = center_y(&frame).map_err(|err| {
                    NodeError::new(&self.name, NodePhase::Process, format!("inspect: {err}"))
                })?;
                self.seen.lock().unwrap().push(y);
                Ok(Processed::Again)
            }
            Some(EdgeItem::Buffer(_)) => Err(NodeError::new(
                &self.name,
                NodePhase::Process,
                "unexpected non-video buffer",
            )),
            Some(EdgeItem::Event(_)) => Ok(Processed::Again),
            None => Ok(Processed::Again),
        }
    }
}

fn cuda_nv12(device: &HwDevice, luma: u8, pts: i64) -> Result<AVFrame, String> {
    let mut pool: AVHWFramesContext = device
        .frames_ctx(SIZE, SIZE, ffi::AV_PIX_FMT_NV12)
        .map_err(|err| format!("frames ctx: {err}"))?;
    let mut software = software_nv12(luma)?;
    let mut hardware = AVFrame::new();
    pool.get_buffer(&mut hardware)
        .map_err(|err| format!("av_hwframe_get_buffer: {err}"))?;
    let rc = unsafe {
        ffi::av_hwframe_transfer_data(hardware.as_mut_ptr(), software.as_mut_ptr(), 0)
    };
    if rc < 0 {
        return Err(format!("av_hwframe_transfer_data: {rc}"));
    }
    hardware.set_pts(pts);
    hardware.set_time_base(ffi::AVRational {
        num: 1,
        den: 1_000_000_000,
    });
    // The SDR canvas refuses a frame whose color tags do not match it.
    unsafe {
        let raw = UnsafeDerefMut::deref_mut(&mut hardware);
        raw.color_trc = ffi::AVCOL_TRC_BT709;
        raw.color_primaries = ffi::AVCOL_PRI_BT709;
        raw.colorspace = ffi::AVCOL_SPC_BT709;
        raw.color_range = ffi::AVCOL_RANGE_MPEG;
    }
    Ok(hardware)
}

fn software_nv12(luma: u8) -> Result<AVFrame, String> {
    let mut frame = AVFrame::new();
    frame.set_width(SIZE);
    frame.set_height(SIZE);
    frame.set_format(ffi::AV_PIX_FMT_NV12);
    frame
        .alloc_buffer()
        .map_err(|err| format!("nv12 alloc: {err}"))?;
    unsafe {
        let raw = UnsafeDerefMut::deref_mut(&mut frame);
        fill_plane(raw.data[0], raw.linesize[0], SIZE as usize, SIZE as usize, luma);
        fill_plane(
            raw.data[1],
            raw.linesize[1],
            SIZE as usize,
            (SIZE / 2) as usize,
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

fn center_y(frame: &AVFrame) -> Result<u8, String> {
    let raw = unsafe { &*frame.as_ptr() };
    if raw.hw_frames_ctx.is_null() {
        return Err("output frame has no hw_frames_ctx".into());
    }
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
    Ok(unsafe {
        *downloaded.data[0].add(
            (SIZE as usize / 2) * downloaded.linesize[0] as usize + (SIZE as usize / 2),
        )
    })
}

struct StopGroup<'a>(&'a Instance);

impl Drop for StopGroup<'_> {
    fn drop(&mut self) {
        let _ = control::exec_line(self.0, "group.stop g");
    }
}

fn exec(inst: &Instance, line: &str) {
    control::exec_line(inst, line).unwrap_or_else(|err| panic!("{line}: {err}"));
}

fn outcomes(inst: &Instance) -> Vec<String> {
    let status = control::exec_line(inst, "group.status g").expect("group.status");
    let status: serde_json::Value = serde_json::from_str(&status).expect("group.status json");
    status["outcomes"]
        .as_array()
        .map(|items| {
            items
                .iter()
                .map(|outcome| outcome.as_str().unwrap_or("").to_owned())
                .collect()
        })
        .unwrap_or_default()
}

fn assert_healthy(inst: &Instance) {
    let outcomes = outcomes(inst);
    for outcome in &outcomes {
        assert!(
            !outcome.starts_with("failed:") && !outcome.starts_with("panicked:"),
            "node failed: {outcome}; all outcomes: {outcomes:?}"
        );
    }
}

fn wait_luma(inst: &Instance, seen: &Mutex<Vec<u8>>, pred: impl Fn(u8) -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(25);
    while Instant::now() < deadline {
        assert_healthy(inst);
        if seen.lock().unwrap().iter().any(|y| pred(*y)) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    false
}

#[test]
fn mixer_cut_switches_cuda_source_luma() {
    let inst = Instance::new();
    avplumber_nodes::register_media_nodes(&inst);
    exec(&inst, r#"hwaccel.init {"name":"gpu","type":"cuda"}"#);
    let device = inst
        .services()
        .hwaccels
        .get("gpu")
        .unwrap_or_else(|err| panic!("hwaccel gpu: {err}"));

    let seen = Arc::new(Mutex::new(Vec::new()));
    let seen_factory = seen.clone();
    let device_factory = device.clone();
    register_factory(&inst, "frame_source", move |name, params| {
        let spec: serde_json::Value =
            serde_json::from_str(params).map_err(|err| err.to_string())?;
        let luma = spec
            .get("luma")
            .and_then(|value| value.as_u64())
            .ok_or("frame_source needs luma")? as u8;
        Ok(Arc::new(FrameSource::new(
            name.to_string(),
            device_factory.clone(),
            luma,
        )))
    });
    register_factory(&inst, "frame_sink", move |name, _params| {
        Ok(Arc::new(FrameSink::new(name.to_string(), seen_factory.clone())))
    });

    let _stop = StopGroup(&inst);
    let comp = |name: &str, src: &str, dst: &str, layers: &str, active: u32| {
        format!(
            r#"node.add {{"type":"mixer_compositor","name":"{name}","group":"g","hwaccel":"gpu","src":{src},"dst":"{dst}","width":64,"height":64,"sw_format":"nv12","color":"sdr","fps":"25/1","max_layers":4,"layers":{layers},"active_inputs":{active}}}"#
        )
    };
    let lines = [
        format!(
            r#"node.add {{"type":"frame_source","name":"src0","group":"g","dst":"raw0","luma":{DARK}}}"#
        ),
        format!(
            r#"node.add {{"type":"frame_source","name":"src1","group":"g","dst":"raw1","luma":{BRIGHT}}}"#
        ),
        r#"node.add {"type":"one_to_many","name":"otm0","group":"g","src":"raw0","dst":["s0a","s0b"],"outputs":1,"drop":true}"#.to_string(),
        r#"node.add {"type":"one_to_many","name":"otm1","group":"g","src":"raw1","dst":["s1a","s1b"],"outputs":0,"drop":true}"#.to_string(),
        comp(
            "comp_a",
            r#"["s0a","s1a"]"#,
            "scene_a",
            r#"[{"input":0,"dst_x":0,"dst_y":0,"dst_w":64,"dst_h":64}]"#,
            1,
        ),
        comp("comp_b", r#"["s0b","s1b"]"#, "scene_b", "[]", 0),
        r#"node.add {"type":"one_to_many","name":"otm_a","group":"g","src":"scene_a","dst":["direct_a"],"outputs":1,"drop":true}"#.to_string(),
        r#"node.add {"type":"one_to_many","name":"otm_b","group":"g","src":"scene_b","dst":["direct_b"],"outputs":0,"drop":true}"#.to_string(),
        r#"node.add {"type":"mixer_selector","name":"sel","group":"g","mixer":"mix","src":["direct_a","direct_b"],"dst":"program","active":0}"#.to_string(),
        r#"node.add {"type":"mixer_snapshot","name":"snap","group":"g","src":"program","dst":"shown","snapshot":"shown_snap","fps":"25/1","slot":-1}"#.to_string(),
        r#"node.add {"type":"frame_sink","name":"sink","group":"g","src":"shown"}"#.to_string(),
        r#"mixer.init mix {"backend":"cuda","hwaccel":"gpu","fps_num":25,"fps_den":1,"color":"sdr","source_switcher":"sel","initial_pgm_slot":"A","initial_pgm_scene":"cam1","slot_a":{"compositor":"comp_a","post_otm":"otm_a"},"slot_b":{"compositor":"comp_b","post_otm":"otm_b"}}"#.to_string(),
        r#"mixer.source mix cam1 otm0 0"#.to_string(),
        r#"mixer.source mix cam2 otm1 1"#.to_string(),
        r#"mixer.scene mix cam1 {"sources":{"cam1":{"dst_x":0,"dst_y":0,"dst_w":64,"dst_h":64}}}"#.to_string(),
        r#"mixer.scene mix cam2 {"sources":{"cam2":{"dst_x":0,"dst_y":0,"dst_w":64,"dst_h":64}}}"#.to_string(),
        "group.start g".to_string(),
    ];
    for line in &lines {
        exec(&inst, line);
    }

    let scenes = control::exec_line(&inst, "mixer.scenes mix").expect("mixer.scenes");
    assert_eq!(scenes.trim(), r#"["cam1","cam2"]"#);

    assert!(
        wait_luma(&inst, &seen, |y| (20..120).contains(&y)),
        "program never showed the dark source; samples {:?}",
        seen.lock().unwrap()
    );
    let before = seen.lock().unwrap().len();
    exec(
        &inst,
        r#"mixer.cut {"mixer":"mix","scene":"cam2"}"#,
    );
    assert!(
        wait_luma(&inst, &seen, |y| y > 160),
        "cut did not switch to the bright source; samples {:?}",
        seen.lock().unwrap()
    );
    let after = seen.lock().unwrap();
    assert!(
        after[..before].iter().any(|y| (20..120).contains(y)),
        "dark program frames were not the ones before the cut: {after:?}"
    );
    assert!(
        after[before..].iter().any(|y| *y > 160),
        "bright frame was already present before the cut: {after:?}"
    );
    drop(after);

    let status = control::exec_line(&inst, "mixer.status mix").expect("mixer.status");
    let status: serde_json::Value =
        serde_json::from_str(status.trim()).expect("mixer.status json");
    assert_eq!(status["pgm_scene"], "cam2");
    assert_eq!(status["pgm_slot"], "B");
}
