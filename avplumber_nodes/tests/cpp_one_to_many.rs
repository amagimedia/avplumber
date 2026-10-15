//! Software frames through the C++ `one_to_many` node on the Rust graph.
//!
//! The node starts with both destinations enabled. After the first frame has
//! reached both sinks, `node.object.set` of `outputs` leaves only the second
//! destination, and the next frame does not arrive at the first.
//!
//! `cargo test -p avplumber_nodes --features cpp_nodes,ffmpeg9 --test cpp_one_to_many`

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex, OnceLock};
use std::time::{Duration, Instant};

use avplumber_f7k::control;
use avplumber_f7k::{
    AvpMediaType, Edge, EdgeEvent, EdgeItem, Grain, Instance, Node, NodeError, NodeKind, NodePads,
    NodePhase, Processed, Push, register_factory,
};
use rsmpeg::UnsafeDerefMut;
use rsmpeg::avutil::AVFrame;
use rusty_ffmpeg::ffi;

struct Release {
    go: Mutex<bool>,
    cv: Condvar,
}

impl Release {
    fn new() -> Self {
        Self {
            go: Mutex::new(false),
            cv: Condvar::new(),
        }
    }

    fn wait(&self) {
        let mut go = self.go.lock().unwrap();
        while !*go {
            go = self.cv.wait(go).unwrap();
        }
    }

    fn signal(&self) {
        *self.go.lock().unwrap() = true;
        self.cv.notify_all();
    }
}

struct FrameSource {
    name: String,
    sink: OnceLock<Arc<dyn Edge>>,
    release: Arc<Release>,
    sent: AtomicBool,
}

impl FrameSource {
    fn new(name: String, release: Arc<Release>) -> Self {
        Self {
            name,
            sink: OnceLock::new(),
            release,
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
        sink.push(Grain::Video(software_frame(1)))
            .accepted(&self.name)?;
        self.release.wait();
        sink.push(Grain::Video(software_frame(2)))
            .accepted(&self.name)?;
        sink.push_event(EdgeEvent::Eof);
        Ok(Processed::Done)
    }
}

trait PushExt {
    fn accepted(self, name: &str) -> Result<(), NodeError>;
}

impl PushExt for Push {
    fn accepted(self, name: &str) -> Result<(), NodeError> {
        match self {
            Push::Accepted => Ok(()),
            other => Err(NodeError::new(
                name,
                NodePhase::Process,
                format!("source push: {other:?}"),
            )),
        }
    }
}

struct FrameSink {
    name: String,
    source: OnceLock<Arc<dyn Edge>>,
    seen: Arc<Mutex<Vec<i64>>>,
}

impl FrameSink {
    fn new(name: String, seen: Arc<Mutex<Vec<i64>>>) -> Self {
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
                self.seen.lock().unwrap().push(unsafe { (*frame.as_ptr()).pts });
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

fn software_frame(pts: i64) -> AVFrame {
    let mut frame = AVFrame::new();
    frame.set_width(16);
    frame.set_height(16);
    frame.set_format(ffi::AV_PIX_FMT_YUV420P);
    frame.alloc_buffer().expect("16x16 yuv420p");
    frame.set_pts(pts);
    frame.set_time_base(ffi::AVRational { num: 1, den: 1000 });
    unsafe {
        let raw = UnsafeDerefMut::deref_mut(&mut frame);
        fill_plane(raw.data[0], raw.linesize[0], 16, 16, 180);
        fill_plane(raw.data[1], raw.linesize[1], 8, 8, 128);
        fill_plane(raw.data[2], raw.linesize[2], 8, 8, 128);
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

struct StopGroup<'a>(&'a Instance);

impl Drop for StopGroup<'_> {
    fn drop(&mut self) {
        let _ = control::exec_line(self.0, "group.stop g");
    }
}

fn wait_pts(seen: &Mutex<Vec<i64>>, want: &[i64]) -> bool {
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if seen.lock().unwrap().as_slice() == want {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    false
}

#[test]
fn one_to_many_object_set_drops_the_other_destination() {
    let inst = Instance::new();
    avplumber_nodes::register_media_nodes(&inst);
    let release = Arc::new(Release::new());
    let first = Arc::new(Mutex::new(Vec::new()));
    let second = Arc::new(Mutex::new(Vec::new()));
    let release_factory = release.clone();
    let first_factory = first.clone();
    let second_factory = second.clone();
    register_factory(&inst, "frame_source", move |name, _params| {
        Ok(Arc::new(FrameSource::new(
            name.to_string(),
            release_factory.clone(),
        )))
    });
    register_factory(&inst, "sink_a", move |name, _params| {
        Ok(Arc::new(FrameSink::new(
            name.to_string(),
            first_factory.clone(),
        )))
    });
    register_factory(&inst, "sink_b", move |name, _params| {
        Ok(Arc::new(FrameSink::new(
            name.to_string(),
            second_factory.clone(),
        )))
    });

    let _stop = StopGroup(&inst);
    for line in [
        r#"node.add {"type":"frame_source","name":"src","group":"g","dst":"raw"}"#,
        r#"node.add {"type":"one_to_many","name":"fan","group":"g","src":"raw","dst":["to_a","to_b"],"outputs":3}"#,
        r#"node.add {"type":"sink_a","name":"a","group":"g","src":"to_a"}"#,
        r#"node.add {"type":"sink_b","name":"b","group":"g","src":"to_b"}"#,
        "group.start g",
    ] {
        control::exec_line(&inst, line).unwrap_or_else(|err| panic!("{line}: {err}"));
    }

    assert!(
        wait_pts(&first, &[1]) && wait_pts(&second, &[1]),
        "first frame did not reach both sinks: a={:?} b={:?}",
        first.lock().unwrap(),
        second.lock().unwrap()
    );
    let mask = control::exec_line(&inst, "node.object.get fan outputs").expect("get outputs");
    assert_eq!(mask, "3");

    // Bit 1 only: the second destination stays, the first is dropped.
    control::exec_line(&inst, "node.object.set fan outputs 2").expect("set outputs");
    let mask = control::exec_line(&inst, "node.object.get fan outputs").expect("get outputs");
    assert_eq!(mask, "2");
    release.signal();

    assert!(
        wait_pts(&second, &[1, 2]),
        "second destination did not receive the masked frame: {:?}",
        second.lock().unwrap()
    );
    std::thread::sleep(Duration::from_millis(50));
    assert_eq!(
        first.lock().unwrap().as_slice(),
        &[1],
        "first destination received a frame after its output bit was cleared"
    );
}
