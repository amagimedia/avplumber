//! `force_keyframe` — marks video frames the encoder must make keyframes: on a
//! period, and on demand through `node.object.set <name> trigger true`, which
//! is how an RTP receiver's PLI or FIR reaches the encoder. Port of C++
//! `src/nodes/force_keyframe.cpp`.
//!
//! Triggers coalesce: any number received between two frames force *one*
//! keyframe, so a misbehaving controller cannot spike the bitrate.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use rusty_ffmpeg::ffi;
use serde_json::Value;

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::buffer::{AvpMediaType, AvpRational};
use avplumber_f7k::graph::edge::{Edge, EdgeEvent, EdgeItem, Push};
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::media::Media;
use avplumber_f7k::graph::node::Tick;
use avplumber_f7k::graph::pad::NodePads;
use avplumber_f7k::graph::poll_ctx::NodePollContext;
use avplumber_f7k::graph::timebase::rational_from_json;
use avplumber_f7k::scaffold::{Io, PollNode, Polling, flush_at_head};

#[derive(Debug, serde::Deserialize)]
pub struct ForceKeyframeSpec {
    /// Force a keyframe whenever the timestamp crosses a multiple of this many
    /// seconds: `"1/30"`, `2`, `0.5`. Absent means on demand only.
    #[serde(default)]
    interval_sec: Option<Value>,
}

impl NodeSpec for ForceKeyframeSpec {
    const TYPE_NAME: &'static str = "force_keyframe";
    type Node = Polling<ForceKeyframe>;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        let interval = match &self.interval_sec {
            Some(value) => {
                let interval =
                    rational_from_json(value).map_err(|e| format!("interval_sec: {e}"))?;
                if interval.num <= 0 || interval.den <= 0 {
                    return Err("interval_sec must be positive".into());
                }
                Some(interval)
            }
            None => None,
        };
        Ok(Polling(ForceKeyframe {
            io: Io::new(name, NodePhase::Poll),
            interval,
            requested: AtomicU64::new(0),
            forced: AtomicU64::new(0),
            triggered_frames: AtomicU64::new(0),
            periodic_frames: AtomicU64::new(0),
            state: Mutex::new(State::default()),
        }))
    }
}

#[derive(Default)]
struct State {
    /// Which interval slot the last frame fell in; C++ `last_result_`.
    last_slot: Option<i64>,
    /// A frame the output had no room for.
    pending: Option<Media>,
}

pub struct ForceKeyframe {
    io: Io,
    interval: Option<AvpRational>,
    /// Bumped by every trigger; `forced` catches up by one frame, whatever the
    /// distance, which is the coalescing.
    requested: AtomicU64,
    forced: AtomicU64,
    triggered_frames: AtomicU64,
    periodic_frames: AtomicU64,
    state: Mutex<State>,
}

impl PollNode for ForceKeyframe {
    fn io(&self) -> &Io {
        &self.io
    }

    fn pads(&self) -> NodePads {
        NodePads::siso(AvpMediaType::VIDEO, AvpMediaType::VIDEO)
    }

    /// Nothing in `step` can fail, so it may sit behind a Direct edge, which is
    /// where the replay graph puts it: right after the pacing stage.
    fn direct_consumer_is_infallible(&self) -> bool {
        true
    }

    fn start(&self) {
        *self.state.lock().unwrap() = State::default();
    }

    fn step(&self, ctx: &mut NodePollContext) -> Result<Tick, NodeError> {
        let (Some(input), Some(output)) = (self.io.input_slot.get(), self.io.output_slot.get())
        else {
            return Ok(Tick::Idle);
        };
        let mut state = self.state.lock().unwrap();
        if state.pending.is_some() && flush_at_head(&input) {
            // Produced before the flush now at the head: stale.
            state.pending = None;
        }
        if let Some(held) = state.pending.take() {
            if !self.emit(&mut state, ctx, &output, held) {
                ctx.wait_flush(input);
                return Ok(Tick::Idle);
            }
        }
        let Some(item) = input.try_take() else {
            if input.is_closed() {
                return Ok(Tick::Done);
            }
            log::trace!("{}: input empty, waiting", self.io.name);
            ctx.wait_readable(input);
            return Ok(Tick::Idle);
        };
        log::trace!("{}: took {item:?}", self.io.name);
        match item {
            EdgeItem::Event(EdgeEvent::Eof) => {
                output.push_event(EdgeEvent::Eof);
                Ok(Tick::Done)
            }
            EdgeItem::Event(EdgeEvent::FlushStart) => {
                // The next frame after a discontinuity starts a new period.
                state.last_slot = None;
                state.pending = None;
                output.push_event(EdgeEvent::FlushStart);
                Ok(Tick::Again)
            }
            EdgeItem::Event(event) => {
                output.push_event(event);
                Ok(Tick::Again)
            }
            EdgeItem::Buffer(mut buffer) => {
                let triggered = self.take_trigger();
                let periodic = self.periodic(&mut state, &buffer);
                mark(&mut buffer, triggered || periodic);
                if self.emit(&mut state, ctx, &output, buffer) {
                    Ok(Tick::Again)
                } else {
                    ctx.wait_flush(input);
                    Ok(Tick::Idle)
                }
            }
        }
    }

    fn set_object(&self, key: &str, value: &Value) -> Result<(), String> {
        match key {
            "trigger" | "force" | "request" => {
                let enabled = match value {
                    Value::Bool(b) => *b,
                    Value::Number(n) => n.as_i64().unwrap_or(0) != 0,
                    Value::Object(o) => o.get("enable").and_then(Value::as_bool).unwrap_or(true),
                    _ => true,
                };
                if enabled {
                    self.requested.fetch_add(1, Ordering::AcqRel);
                }
                Ok(())
            }
            other => Err(format!("{}: unknown object key `{other}`", self.io.name)),
        }
    }

    fn get_object(&self, key: &str) -> Result<Value, String> {
        let requested = self.requested.load(Ordering::Acquire);
        let forced = self.forced.load(Ordering::Acquire);
        match key {
            "pending" => Ok(Value::Bool(requested != forced)),
            "status" => Ok(serde_json::json!({
                "requested_generation": requested,
                "forced_generation": forced,
                "pending": requested != forced,
                "triggered_frames": self.triggered_frames.load(Ordering::Relaxed),
                "periodic_frames": self.periodic_frames.load(Ordering::Relaxed),
                "interval_enabled": self.interval.is_some(),
            })),
            other => Err(format!("{}: unknown object key `{other}`", self.io.name)),
        }
    }
}

impl ForceKeyframe {
    /// One forced frame per batch of triggers, however many arrived.
    fn take_trigger(&self) -> bool {
        let requested = self.requested.load(Ordering::Acquire);
        if requested == self.forced.load(Ordering::Acquire) {
            return false;
        }
        self.forced.store(requested, Ordering::Release);
        self.triggered_frames.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// The first frame in each interval slot, C++ `shouldForcePeriodic`. The
    /// slot index is `floor(pts / interval)` computed in integers.
    fn periodic(&self, state: &mut State, buffer: &Media) -> bool {
        let Some(interval) = self.interval else {
            return false;
        };
        let ts = buffer.ts();
        if !ts.is_valid() || ts.tb.num == 0 || ts.tb.den == 0 {
            return false;
        }
        let slot = (ts.val as i128 * ts.tb.num as i128 * interval.den as i128)
            / (ts.tb.den as i128 * interval.num as i128);
        let slot = slot as i64;
        if state.last_slot == Some(slot) {
            return false;
        }
        state.last_slot = Some(slot);
        self.periodic_frames.fetch_add(1, Ordering::Relaxed);
        true
    }

    /// `false` when the output is full: the frame is held and the node waits.
    fn emit(
        &self,
        state: &mut State,
        ctx: &mut NodePollContext,
        output: &std::sync::Arc<dyn Edge>,
        buffer: Media,
    ) -> bool {
        match output.offer(buffer) {
            Ok(()) | Err((Push::Dropped | Push::Accepted, _)) => true,
            Err((Push::Full, buffer)) => {
                state.pending = Some(buffer);
                ctx.wait_writable(output.clone());
                false
            }
            Err((Push::Closed, _)) => {
                log::info!("{}: output closed, discarding", self.io.name);
                true
            }
        }
    }
}

/// `pict_type` is what libx264 and nvenc read to force an I-frame; an explicit
/// `NONE` on every other frame leaves the decision to the encoder, exactly as
/// C++ did. The key flag goes with it so a downstream reader agrees.
fn mark(buffer: &mut Media, force: bool) {
    if let Media::Video(frame) = buffer {
        let raw = unsafe { rsmpeg::UnsafeDerefMut::deref_mut(frame) };
        if force {
            raw.pict_type = ffi::AV_PICTURE_TYPE_I;
            raw.flags |= ffi::AV_FRAME_FLAG_KEY as i32;
        } else {
            raw.pict_type = ffi::AV_PICTURE_TYPE_NONE;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use super::*;
    use avplumber_f7k::Instance;
    use avplumber_f7k::graph::BufferedEdge;
    use avplumber_f7k::graph::edge::Wakeup;
    use avplumber_f7k::graph::media::test_media;
    use avplumber_f7k::graph::node::Node;

    fn node(interval: Option<Value>) -> (Polling<ForceKeyframe>, Arc<dyn Edge>, Arc<dyn Edge>) {
        let instance = Instance::new();
        let params = serde_json::json!({});
        let ctx = BuildCtx {
            instance: &instance,
            name: "fk",
            params: &params,
            sync_group: None,
        };
        let node = ForceKeyframeSpec {
            interval_sec: interval,
        }
        .build("fk", &ctx)
        .expect("builds");
        let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(64));
        let output: Arc<dyn Edge> = Arc::new(BufferedEdge::new(64));
        node.bind_source("in", input.clone());
        node.bind_sink("out", output.clone());
        node.start();
        (node, input, output)
    }

    /// Runs the node over what is queued and returns, per output frame, whether
    /// it was marked as a keyframe.
    fn run(node: &Polling<ForceKeyframe>, output: &Arc<dyn Edge>) -> Vec<bool> {
        let mut ctx =
            NodePollContext::new(Arc::new(AtomicBool::new(false)), Arc::new(Wakeup::new()));
        loop {
            ctx.clear_park();
            match node.poll(&mut ctx).expect("step") {
                Tick::Again => continue,
                _ => break,
            }
        }
        let mut marks = Vec::new();
        while let Some(item) = output.try_take() {
            if let EdgeItem::Buffer(Media::Video(frame)) = item {
                marks.push(frame.pict_type == ffi::AV_PICTURE_TYPE_I);
            }
        }
        marks
    }

    #[test]
    fn the_first_frame_of_each_interval_is_forced() {
        // 1/1000 stamps from `test_media`; a 100 ms interval at 40 ms frames.
        let (node, input, output) = node(Some(serde_json::json!("0.1")));
        for pts in [0, 40, 80, 120, 160, 200, 240] {
            assert!(input.offer(test_media(AvpMediaType::VIDEO, pts)).is_ok());
        }
        assert_eq!(
            run(&node, &output),
            vec![true, false, false, true, false, true, false]
        );
        assert_eq!(
            node.get_object("status").unwrap()["periodic_frames"],
            serde_json::json!(3)
        );
    }

    #[test]
    fn triggers_coalesce_into_one_forced_frame() {
        let (node, input, output) = node(None);
        assert!(input.offer(test_media(AvpMediaType::VIDEO, 0)).is_ok());
        assert_eq!(run(&node, &output), vec![false]);

        node.set_object("trigger", &Value::Bool(true)).unwrap();
        node.set_object("trigger", &Value::Bool(true)).unwrap();
        node.set_object("force", &serde_json::json!({"enable": true}))
            .unwrap();
        assert_eq!(node.get_object("pending").unwrap(), Value::Bool(true));
        for pts in [40, 80] {
            assert!(input.offer(test_media(AvpMediaType::VIDEO, pts)).is_ok());
        }
        assert_eq!(run(&node, &output), vec![true, false]);
        assert_eq!(node.get_object("pending").unwrap(), Value::Bool(false));
        assert_eq!(
            node.get_object("status").unwrap()["triggered_frames"],
            serde_json::json!(1)
        );
        // `false` is not a trigger.
        node.set_object("trigger", &Value::Bool(false)).unwrap();
        assert_eq!(node.get_object("pending").unwrap(), Value::Bool(false));
        assert!(node.set_object("nope", &Value::Null).is_err());
    }

    #[test]
    fn a_flush_restarts_the_period() {
        let (node, input, output) = node(Some(serde_json::json!(1)));
        assert!(input.offer(test_media(AvpMediaType::VIDEO, 0)).is_ok());
        assert!(input.offer(test_media(AvpMediaType::VIDEO, 40)).is_ok());
        assert_eq!(run(&node, &output), vec![true, false]);
        input.push_event(EdgeEvent::FlushStart);
        input.push_event(EdgeEvent::FlushStop { resume_at: None });
        // Same slot as before the flush, forced again because the grid restarted.
        assert!(input.offer(test_media(AvpMediaType::VIDEO, 80)).is_ok());
        assert_eq!(run(&node, &output), vec![true]);
    }
}
