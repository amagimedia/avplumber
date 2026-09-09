//! `force_fps` — conforms a video stream to a fixed frame rate: a frame that
//! arrives early is dropped, a gap is filled by repeating the last frame on the
//! grid. Port of C++ `src/nodes/force_fps.cpp`.
//!
//! No libav here: the node only restamps and clones `Media`, so it builds in
//! the default configuration and its unit tests run without FFmpeg. It also
//! publishes the conformed rate and time base in the `Spec` it forwards, which
//! is what the C++ `IFrameRateSource`/`ITimeBaseSource` interfaces told an
//! encoder downstream.

use std::collections::VecDeque;
use std::sync::Mutex;
use std::time::Instant;

use serde_json::Value;

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::buffer::{AvpMediaType, AvpRational};
use avplumber_f7k::graph::edge::{EdgeEvent, EdgeItem, Push};
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::media::{Media, Ts};
use avplumber_f7k::graph::node::Tick;
use avplumber_f7k::graph::pad::NodePads;
use avplumber_f7k::graph::poll_ctx::NodePollContext;
use avplumber_f7k::graph::spec::Spec;
use avplumber_f7k::graph::timebase::rational_from_json;
use avplumber_f7k::scaffold::{Io, PollNode, Polling, flush_at_head};

/// C++ prints its drop/duplicate statistics this often.
const STATS_PERIOD_S: u64 = 10;

#[derive(Debug, serde::Deserialize)]
pub struct ForceFpsSpec {
    /// The output rate: `"30"`, `"30000/1001"`, `25`.
    fps: Value,
    /// The output time base; the inverse of `fps` when absent.
    #[serde(default)]
    timebase: Option<Value>,
}

impl NodeSpec for ForceFpsSpec {
    const TYPE_NAME: &'static str = "force_fps";
    type Node = Polling<ForceFps>;

    fn build(self, name: &str, _ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        let fps = rational_from_json(&self.fps).map_err(|e| format!("fps: {e}"))?;
        if fps.num <= 0 || fps.den <= 0 {
            return Err(format!("fps must be positive, got {}/{}", fps.num, fps.den));
        }
        let (timebase, frame_delta) = match &self.timebase {
            Some(value) => {
                let tb = rational_from_json(value).map_err(|e| format!("timebase: {e}"))?;
                if tb.num <= 0 || tb.den <= 0 {
                    return Err(format!(
                        "timebase must be positive, got {}/{}",
                        tb.num, tb.den
                    ));
                }
                let delta = Ts {
                    val: 1,
                    tb: AvpRational {
                        num: fps.den,
                        den: fps.num,
                    },
                }
                .rescale(tb);
                (tb, delta.val)
            }
            None => (
                AvpRational {
                    num: fps.den,
                    den: fps.num,
                },
                1,
            ),
        };
        if frame_delta <= 0 {
            return Err(format!(
                "timebase {}/{} cannot represent one frame at {}/{} fps",
                timebase.num, timebase.den, fps.num, fps.den
            ));
        }
        log::info!(
            "{name}: time base {}/{}, frame rate {}/{}, frame delta {frame_delta}",
            timebase.num,
            timebase.den,
            fps.num,
            fps.den
        );
        Ok(Polling(ForceFps {
            io: Io::new(name, NodePhase::Poll),
            fps,
            timebase,
            frame_delta,
            state: Mutex::new(State::default()),
        }))
    }
}

#[derive(Default)]
struct State {
    /// The last frame that went out, on the grid.
    last_ts: Option<i64>,
    /// Where the grid expects the next frame.
    next_ts: Option<i64>,
    /// The last frame seen, kept for duplication.
    last_frame: Option<Media>,
    /// Whether `last_frame` was dropped rather than emitted; C++ `last_unused_`.
    last_unused: bool,
    dropped: u64,
    duplicated: u64,
    total_in: u64,
    total_out: u64,
    last_stats: Option<Instant>,
    /// Frames the output had no room for, oldest first.
    pending: VecDeque<Media>,
}

pub struct ForceFps {
    io: Io,
    fps: AvpRational,
    timebase: AvpRational,
    /// One frame period in `timebase` units.
    frame_delta: i64,
    state: Mutex<State>,
}

impl PollNode for ForceFps {
    fn io(&self) -> &Io {
        &self.io
    }

    fn pads(&self) -> NodePads {
        NodePads::siso(AvpMediaType::VIDEO, AvpMediaType::VIDEO)
    }

    /// Nothing in `step` can fail: the parameters were validated at build and
    /// every item kind has an arm.
    fn direct_consumer_is_infallible(&self) -> bool {
        true
    }

    fn start(&self) {
        *self.state.lock().unwrap() = State::default();
    }

    fn step(&self, ctx: &mut NodePollContext) -> Result<Tick, NodeError> {
        // Bound before start, so an empty slot only happens in a test that
        // forgot to bind; idling is the infallible answer.
        let (Some(input), Some(output)) = (self.io.input_slot.get(), self.io.output_slot.get())
        else {
            return Ok(Tick::Idle);
        };
        let mut state = self.state.lock().unwrap();

        if !state.pending.is_empty() && flush_at_head(&input) {
            // Produced before the flush now at the head: stale.
            state.pending.clear();
        }
        if !self.drain_pending(&mut state, ctx, &output) {
            ctx.wait_flush(input);
            return Ok(Tick::Idle);
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
            EdgeItem::Event(EdgeEvent::Spec(spec)) => {
                output.push_event(EdgeEvent::Spec(self.conformed_spec(spec)));
                Ok(Tick::Again)
            }
            EdgeItem::Event(EdgeEvent::FlushStart) => {
                self.reset_grid(&mut state);
                state.pending.clear();
                output.push_event(EdgeEvent::FlushStart);
                Ok(Tick::Again)
            }
            EdgeItem::Event(event @ EdgeEvent::FlushStop { .. }) => {
                output.push_event(event);
                Ok(Tick::Again)
            }
            EdgeItem::Event(EdgeEvent::Drain) => {
                // The grid holds frames on purpose; only codecs drain.
                output.push_event(EdgeEvent::Drain);
                Ok(Tick::Again)
            }
            EdgeItem::Event(EdgeEvent::Eof) => {
                self.log_stats(&mut state, true);
                output.push_event(EdgeEvent::Eof);
                Ok(Tick::Done)
            }
            EdgeItem::Buffer(buffer) => {
                let produced = self.conform(&mut state, buffer);
                state.pending.extend(produced);
                self.log_stats(&mut state, false);
                if self.drain_pending(&mut state, ctx, &output) {
                    Ok(Tick::Again)
                } else {
                    ctx.wait_flush(input);
                    Ok(Tick::Idle)
                }
            }
        }
    }

    fn get_object(&self, key: &str) -> Result<Value, String> {
        match key {
            "stats" => {
                let state = self.state.lock().unwrap();
                Ok(serde_json::json!({
                    "in": state.total_in,
                    "out": state.total_out,
                    "dropped": state.dropped,
                    "duplicated": state.duplicated,
                }))
            }
            other => Err(format!("{}: unknown object `{other}`", self.io.name)),
        }
    }
}

impl ForceFps {
    /// Offers what is queued for the output, oldest first. `false` when the
    /// output is full: the rest stays queued and the node waits on it.
    fn drain_pending(
        &self,
        state: &mut State,
        ctx: &mut NodePollContext,
        output: &std::sync::Arc<dyn avplumber_f7k::graph::edge::Edge>,
    ) -> bool {
        while let Some(buffer) = state.pending.pop_front() {
            match output.offer(buffer) {
                Ok(()) | Err((Push::Dropped | Push::Accepted, _)) => {}
                Err((Push::Full, buffer)) => {
                    state.pending.push_front(buffer);
                    ctx.wait_writable(output.clone());
                    return false;
                }
                Err((Push::Closed, _)) => {
                    log::info!("{}: output closed, discarding", self.io.name);
                    state.pending.clear();
                    return true;
                }
            }
        }
        true
    }

    fn conformed_spec(&self, spec: Spec) -> Spec {
        match spec {
            Spec::Video {
                width,
                height,
                pix_fmt,
                sw_pix_fmt,
                sar,
                ..
            } => Spec::Video {
                width,
                height,
                pix_fmt,
                sw_pix_fmt,
                frame_rate: self.fps,
                sar,
                time_base: self.timebase,
            },
            other => other,
        }
    }

    fn reset_grid(&self, state: &mut State) {
        state.last_ts = None;
        state.next_ts = None;
        state.last_frame = None;
        state.last_unused = false;
    }

    fn set_last(&self, state: &mut State, frame: Media, unused: bool) {
        if state.last_unused {
            // The previous frame was never emitted and is now overwritten.
            state.dropped += 1;
        }
        state.last_frame = Some(frame);
        state.last_unused = unused;
    }

    /// The C++ algorithm, frame by frame: duplicates for a gap, a drop for an
    /// early frame, a pass-through otherwise. A discontinuity (a jump of more
    /// than half a second past the grid, or backwards) restarts the grid
    /// without filling or dropping.
    fn conform(&self, state: &mut State, mut buffer: Media) -> Vec<Media> {
        let mut out = Vec::new();
        let ts = buffer.ts();
        if !ts.is_valid() {
            // Nothing to place on a grid; let it through untouched.
            state.total_in += 1;
            state.total_out += 1;
            out.push(buffer);
            return out;
        }
        let in_ts = ts.rescale(self.timebase).val;

        if let (Some(last), Some(mut next)) = (state.last_ts, state.next_ts) {
            let delta = in_ts - last;
            let delta_s = delta as f64 * self.timebase.num as f64 / self.timebase.den as f64;
            let frame_s =
                self.frame_delta as f64 * self.timebase.num as f64 / self.timebase.den as f64;
            let discontinuity = delta_s > frame_s + 0.5 || delta < 0;
            if discontinuity {
                log::info!("{}: discontinuity {last} -> {in_ts}", self.io.name);
            }
            if delta != self.frame_delta && !discontinuity {
                if in_ts > next {
                    let gap_start = next;
                    let mut burst = 0u64;
                    while in_ts > next {
                        if let Some(last_frame) = &state.last_frame {
                            let mut dup = last_frame.clone();
                            if !state.last_unused {
                                // Used more than once: captions must not repeat.
                                strip_captions(&mut dup);
                                state.duplicated += 1;
                            }
                            dup.set_ts(Ts {
                                val: next,
                                tb: self.timebase,
                            });
                            out.push(dup);
                            state.last_unused = false;
                            state.total_out += 1;
                            burst += 1;
                        }
                        next += self.frame_delta;
                    }
                    state.next_ts = Some(next);
                    if burst > 1 {
                        log::info!(
                            "{}: filled a gap with {burst} duplicate(s); grid {gap_start} ..< \
                             {in_ts}",
                            self.io.name
                        );
                    }
                }
                if in_ts < next {
                    // Too early: keep it as the last frame, in case a gap follows,
                    // and drop it.
                    state.total_in += 1;
                    self.set_last(state, buffer, true);
                    return out;
                }
            }
        }

        buffer.set_ts(Ts {
            val: in_ts,
            tb: self.timebase,
        });
        self.set_last(state, buffer.clone(), false);
        state.last_ts = Some(in_ts);
        state.next_ts = Some(in_ts + self.frame_delta);
        state.total_out += 1;
        state.total_in += 1;
        out.push(buffer);
        out
    }

    fn log_stats(&self, state: &mut State, force: bool) {
        let now = Instant::now();
        let due = match state.last_stats {
            None => {
                state.last_stats = Some(now);
                false
            }
            Some(last) => now.duration_since(last).as_secs() >= STATS_PERIOD_S,
        };
        if (due || force) && (state.dropped > 0 || state.duplicated > 0) {
            log::info!(
                "{}: in {}, out {}, duplicated {}, dropped {}",
                self.io.name,
                state.total_in,
                state.total_out,
                state.duplicated,
                state.dropped
            );
            state.last_stats = Some(now);
        }
    }
}

/// A repeated frame must not repeat its closed captions.
#[cfg(feature = "ffmpeg")]
fn strip_captions(media: &mut Media) {
    if let Media::Video(frame) = media {
        unsafe {
            rusty_ffmpeg::ffi::av_frame_remove_side_data(
                frame.as_mut_ptr(),
                rusty_ffmpeg::ffi::AV_FRAME_DATA_A53_CC,
            );
        }
    }
}

#[cfg(not(feature = "ffmpeg"))]
fn strip_captions(_media: &mut Media) {}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use super::*;
    use avplumber_f7k::Instance;
    use avplumber_f7k::graph::BufferedEdge;
    use avplumber_f7k::graph::edge::{Edge, Wakeup};
    use avplumber_f7k::graph::media::test_media;
    use avplumber_f7k::graph::node::Node;

    struct Harness {
        node: Polling<ForceFps>,
        input: Arc<dyn Edge>,
        output: Arc<dyn Edge>,
        ctx: NodePollContext,
    }

    fn harness(fps: Value, timebase: Option<Value>, out_capacity: usize) -> Harness {
        let instance = Instance::new();
        let params = serde_json::json!({});
        let ctx = BuildCtx {
            instance: &instance,
            name: "ff",
            params: &params,
            sync_group: None,
        };
        let node = ForceFpsSpec { fps, timebase }
            .build("ff", &ctx)
            .expect("builds");
        let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(64));
        let output: Arc<dyn Edge> = Arc::new(BufferedEdge::new(out_capacity));
        node.bind_source("in", input.clone());
        node.bind_sink("out", output.clone());
        node.start();
        Harness {
            node,
            input,
            output,
            ctx: NodePollContext::new(Arc::new(AtomicBool::new(false)), Arc::new(Wakeup::new())),
        }
    }

    impl Harness {
        fn step_until_idle(&mut self) {
            loop {
                self.ctx.clear_park();
                match self.node.poll(&mut self.ctx).expect("step") {
                    Tick::Again => continue,
                    Tick::Idle | Tick::Done => break,
                }
            }
        }

        /// Steps until the node idles, then returns the output's timestamps in
        /// the node's time base. (`Media::Stub`, the libav-free test buffer,
        /// always reports 1/1000, so the value is read through a rescale; with
        /// FFmpeg the frame carries the node's base and the rescale is exact.)
        fn run(&mut self) -> Vec<i64> {
            self.step_until_idle();
            let mut out = Vec::new();
            while let Some(item) = self.output.try_take() {
                if let EdgeItem::Buffer(media) = item {
                    out.push(media.ts().rescale(self.node.timebase).val);
                }
            }
            out
        }

        fn push_ms(&self, ms: &[i64]) {
            for &pts in ms {
                assert!(
                    self.input
                        .offer(test_media(AvpMediaType::VIDEO, pts))
                        .is_ok(),
                    "room on the input"
                );
            }
        }
    }

    /// `test_media` stamps in 1/1000; the node runs at 1/25, so 40 ms is one
    /// frame.
    #[test]
    fn aligned_input_passes_through_restamped() {
        let mut h = harness(serde_json::json!("25"), None, 64);
        h.push_ms(&[0, 40, 80, 120]);
        assert_eq!(h.run(), vec![0, 1, 2, 3]);
        assert_eq!(
            h.node.get_object("stats").unwrap(),
            serde_json::json!({"in": 4, "out": 4, "dropped": 0, "duplicated": 0})
        );
    }

    #[test]
    fn a_gap_is_filled_with_duplicates_on_the_grid() {
        let mut h = harness(serde_json::json!(25), None, 64);
        // Frames 0, 1, then 4: frames 2 and 3 are missing.
        h.push_ms(&[0, 40, 160]);
        assert_eq!(h.run(), vec![0, 1, 2, 3, 4]);
        assert_eq!(
            h.node.get_object("stats").unwrap()["duplicated"],
            serde_json::json!(2)
        );
    }

    #[test]
    fn early_frames_are_dropped() {
        let mut h = harness(serde_json::json!("25/1"), None, 64);
        // Roughly 50 fps in, 25 fps out: every other frame is early. Off-grid
        // stamps rather than exact halves, so no rounding rule decides the test.
        h.push_ms(&[0, 16, 40, 56, 80]);
        assert_eq!(h.run(), vec![0, 1, 2]);
        assert_eq!(
            h.node.get_object("stats").unwrap()["dropped"],
            serde_json::json!(2)
        );
    }

    #[test]
    fn a_jump_restarts_the_grid_without_filling() {
        let mut h = harness(serde_json::json!("25"), None, 64);
        h.push_ms(&[0, 40, 5000, 5040]);
        assert_eq!(h.run(), vec![0, 1, 125, 126]);
        assert_eq!(
            h.node.get_object("stats").unwrap()["duplicated"],
            serde_json::json!(0)
        );
    }

    #[test]
    fn flush_resets_the_grid_and_forwards_both_markers() {
        let mut h = harness(serde_json::json!("25"), None, 64);
        h.push_ms(&[0, 40]);
        assert_eq!(h.run(), vec![0, 1]);
        h.input.push_event(EdgeEvent::FlushStart);
        h.input.push_event(EdgeEvent::FlushStop { resume_at: None });
        // Far away in time: with the grid reset there is nothing to fill.
        h.push_ms(&[10_000, 10_040]);
        let mut events = Vec::new();
        h.step_until_idle();
        let mut frames = Vec::new();
        while let Some(item) = h.output.try_take() {
            match item {
                EdgeItem::Buffer(media) => frames.push(media.ts().rescale(h.node.timebase).val),
                EdgeItem::Event(EdgeEvent::FlushStart) => events.push("start"),
                EdgeItem::Event(EdgeEvent::FlushStop { .. }) => events.push("stop"),
                EdgeItem::Event(_) => events.push("other"),
            }
        }
        assert_eq!(events, vec!["start", "stop"]);
        assert_eq!(frames, vec![250, 251]);
    }

    #[test]
    fn spec_is_forwarded_with_the_conformed_rate_and_base() {
        let mut h = harness(
            serde_json::json!("30"),
            Some(serde_json::json!("1/90000")),
            64,
        );
        h.input.push_event(EdgeEvent::Spec(Spec::Video {
            width: 4,
            height: 4,
            pix_fmt: 0,
            sw_pix_fmt: -1,
            frame_rate: AvpRational { num: 60, den: 1 },
            sar: AvpRational { num: 1, den: 1 },
            time_base: AvpRational { num: 1, den: 1000 },
        }));
        h.step_until_idle();
        let Some(EdgeItem::Event(EdgeEvent::Spec(Spec::Video {
            frame_rate,
            time_base,
            ..
        }))) = h.output.try_take()
        else {
            panic!("a video spec is forwarded");
        };
        assert_eq!(frame_rate, AvpRational { num: 30, den: 1 });
        assert_eq!(time_base, AvpRational { num: 1, den: 90000 });
        assert_eq!(h.node.frame_delta, 3000);
    }

    #[test]
    fn a_full_output_holds_frames_in_order() {
        let mut h = harness(serde_json::json!("25"), None, 2);
        h.push_ms(&[0, 40, 80, 120]);
        // Two fit, two wait.
        assert_eq!(h.run(), vec![0, 1]);
        assert_eq!(h.run(), vec![2, 3]);
    }
}
