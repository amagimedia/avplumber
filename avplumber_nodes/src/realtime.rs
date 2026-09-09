//! `realtime` — the pacing stage: releases each frame when its media time,
//! mapped through the group's clock, comes due, and stamps it with the release
//! time so everything downstream sees one monotonic timeline across seeks,
//! reverse play and loops. Replaces C++ `realtime`, `speed` and `pause`
//! together with the playback service (`doc/specs/rust-refactor/rust_refactor_playback.md` §7).
//!
//! Rate and pause are properties of the clock, read at release: a rate change
//! is O(1) and touches nothing in flight. This is the one node that rewrites
//! PTS, the continuity exception the design reserves for the output stage.
//!
//! No libav: the node reads and restamps `Media`, so it builds without FFmpeg
//! and its unit tests run against a synthetic clock.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use serde_json::Value;

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::buffer::{AVP_NOPTS, AvpMediaType, AvpRational};
use avplumber_f7k::graph::edge::{EdgeEvent, EdgeItem, Push};
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::media::{Media, Ts};
use avplumber_f7k::graph::node::Tick;
use avplumber_f7k::graph::pad::NodePads;
use avplumber_f7k::graph::poll_ctx::NodePollContext;
use avplumber_f7k::graph::spec::Spec;
use avplumber_f7k::graph::timebase::{MICROSECONDS, MILLISECONDS, rational_from_json, rescale};
use avplumber_f7k::scaffold::{Io, PollNode, Polling};
use avplumber_f7k::services::clock::{SyncGroup, instant_at};
use avplumber_f7k::services::playback::Playback;

/// How often a paused node looks at the clock again. Resume latency, and the
/// cost of being paused.
const PAUSED_POLL: Duration = Duration::from_millis(15);

#[derive(Debug, serde::Deserialize)]
pub struct RealtimeSpec {
    /// Output PTS snap to this grid, e.g. `"1/25"`; the time base defaults to
    /// it too.
    #[serde(default)]
    tick_period: Option<Value>,
    /// Time base of the output PTS. Defaults to `tick_period`, else `1/1000`.
    #[serde(default)]
    timebase: Option<Value>,
    /// A frame further behind now than this is dropped instead of released
    /// late. Seconds; default two frame periods, or 80 ms without a tick.
    #[serde(default)]
    negative_time_discard: Option<f64>,
    /// Behind now by more than this means the clock is anchored wrongly:
    /// re-anchor on this frame instead of dropping. Seconds, default 0.25.
    #[serde(default)]
    negative_time_tolerance: Option<f64>,
    /// Ahead of now by more than this is a timestamp jump, not a frame to wait
    /// for: re-anchor. Seconds, default 1.
    #[serde(default)]
    discontinuity_threshold: Option<f64>,
}

impl NodeSpec for RealtimeSpec {
    const TYPE_NAME: &'static str = "realtime";
    type Node = Polling<Realtime>;

    fn build(self, name: &str, ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        let group = ctx
            .sync_group
            .ok_or("realtime needs a `sync_group`: the playback group whose clock it paces by")?;
        let playback = ctx.playback(group);
        let tick = match &self.tick_period {
            Some(value) => {
                let period = rational_from_json(value).map_err(|e| format!("tick_period: {e}"))?;
                if period.num <= 0 || period.den <= 0 {
                    return Err("tick_period must be positive".into());
                }
                Some(period)
            }
            None => None,
        };
        let timebase = match (&self.timebase, tick) {
            (Some(value), _) => rational_from_json(value).map_err(|e| format!("timebase: {e}"))?,
            (None, Some(period)) => period,
            (None, None) => MILLISECONDS,
        };
        if timebase.num <= 0 || timebase.den <= 0 {
            return Err("timebase must be positive".into());
        }
        let seconds_to_us = |s: f64| (s * 1_000_000.0).round() as i64;
        let tick_us = tick.map(|t| rescale(1, t, MICROSECONDS));
        let discard_us = self
            .negative_time_discard
            .map(seconds_to_us)
            .unwrap_or_else(|| tick_us.map_or(80_000, |t| 2 * t));
        Ok(Polling(Realtime {
            io: Io::new(name, NodePhase::Poll),
            clock: playback.clock().clone(),
            playback,
            timebase,
            tick_us,
            discard_us,
            tolerance_us: seconds_to_us(self.negative_time_tolerance.unwrap_or(0.25)),
            discontinuity_us: seconds_to_us(self.discontinuity_threshold.unwrap_or(1.0)),
            released: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            state: Mutex::new(State::default()),
        }))
    }
}

#[derive(Default)]
struct State {
    /// A frame waiting for its time, or for the output, or for a resume.
    held: Option<Media>,
    /// Release the next frame even while paused: it is the one a seek landed
    /// on, and a paused viewer must see it.
    release_one: bool,
    /// Whether the clock has been anchored on a frame of this run.
    anchored: bool,
    /// The last output PTS, so the stamped timeline never steps back.
    last_out: Option<i64>,
}

pub struct Realtime {
    io: Io,
    playback: Arc<Playback>,
    clock: Arc<dyn SyncGroup>,
    timebase: AvpRational,
    tick_us: Option<i64>,
    discard_us: i64,
    tolerance_us: i64,
    discontinuity_us: i64,
    released: AtomicU64,
    dropped: AtomicU64,
    state: Mutex<State>,
}

/// What the clock says about one frame.
enum Due {
    Now,
    /// The clock is paused: keep the frame.
    Hold,
    /// Wait until this wall time.
    Later(i64),
    /// Too late to be worth showing.
    Late,
    /// The clock and the stream disagree by more than a wait or a drop can
    /// absorb: anchor the clock on this frame and release it.
    Reanchor,
}

impl PollNode for Realtime {
    fn io(&self) -> &Io {
        &self.io
    }

    fn pads(&self) -> NodePads {
        NodePads::siso(AvpMediaType::VIDEO, AvpMediaType::VIDEO)
    }

    /// Nothing in `step` can fail, so the node may end a Direct chain and its
    /// consumer may be Direct too.
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

        loop {
            // Events at the head of the input go first, even while a frame is
            // held: a flush must reach that frame before its time comes, and a
            // resume must not wait behind it.
            while let Some(EdgeItem::Event(event)) = input.peek_clone(0) {
                log::trace!("{}: event {event:?}", self.io.name);
                input.pop();
                self.on_event(&mut state, &output, event);
            }
            let frame = match state.held.take() {
                Some(frame) => frame,
                None => match input.try_take() {
                    None => {
                        if input.is_closed() {
                            return Ok(Tick::Done);
                        }
                        ctx.wait_readable(input.clone());
                        return Ok(Tick::Idle);
                    }
                    Some(EdgeItem::Buffer(frame)) => frame,
                    Some(EdgeItem::Event(event)) => {
                        self.on_event(&mut state, &output, event);
                        if matches!(state.held, None) && input.is_closed() {
                            return Ok(Tick::Done);
                        }
                        continue;
                    }
                },
            };

            if self.playback.seek_in_flight() {
                // Between a seek and its flush: this frame is from the old
                // position, whatever the clock says.
                self.dropped.fetch_add(1, Ordering::Relaxed);
                log::trace!(
                    "{}: dropping media {:?}, a seek is in flight",
                    self.io.name,
                    frame.ts()
                );
                continue;
            }
            let snapshot = self.clock.snapshot();
            if snapshot.paused && !state.release_one {
                state.held = Some(frame);
                ctx.wait_flush(input.clone());
                ctx.wait_deadline(Instant::now() + PAUSED_POLL);
                return Ok(Tick::Idle);
            }

            let media = frame.ts();
            // The one frame a paused viewer gets is not timed: a paused clock
            // maps nothing.
            let due = if snapshot.paused {
                Due::Now
            } else {
                self.due(&mut state, media)
            };
            match due {
                Due::Hold => {
                    state.held = Some(frame);
                    ctx.wait_flush(input.clone());
                    ctx.wait_deadline(Instant::now() + PAUSED_POLL);
                    return Ok(Tick::Idle);
                }
                Due::Later(wall_us) => {
                    state.held = Some(frame);
                    ctx.wait_flush(input.clone());
                    ctx.wait_deadline(instant_at(wall_us));
                    return Ok(Tick::Idle);
                }
                Due::Late => {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                    log::debug!(
                        "{}: dropping late media {}/{}",
                        self.io.name,
                        media.val,
                        fmt_tb(media.tb)
                    );
                    continue;
                }
                Due::Reanchor => {
                    self.clock.reset(media.val, media.tb);
                    state.anchored = true;
                    log::info!(
                        "{}: re-anchored the clock on media {}/{}",
                        self.io.name,
                        media.val,
                        fmt_tb(media.tb)
                    );
                }
                Due::Now => {}
            }

            if output.is_full() {
                // Also readable: a flush arriving behind the held frame must
                // not wait for the output to drain.
                state.held = Some(frame);
                ctx.wait_flush(input.clone());
                ctx.wait_writable(output.clone());
                return Ok(Tick::Idle);
            }
            let mut frame = frame;
            frame.set_ts(self.output_stamp(&mut state));
            match output.offer(frame) {
                Ok(()) | Err((Push::Dropped | Push::Accepted, _)) => {}
                Err((Push::Full, frame)) => {
                    // Raced with a consumer that filled it since the check; the
                    // frame keeps its new stamp, which is still monotonic.
                    state.held = Some(frame);
                    ctx.wait_flush(input.clone());
                    ctx.wait_writable(output.clone());
                    return Ok(Tick::Idle);
                }
                Err((Push::Closed, _)) => {
                    log::info!("{}: output closed, finishing", self.io.name);
                    return Ok(Tick::Done);
                }
            }
            state.release_one = false;
            self.released.fetch_add(1, Ordering::Relaxed);
            self.playback
                .report_release(media.rescale(MILLISECONDS).val);
            return Ok(Tick::Again);
        }
    }

    fn get_object(&self, key: &str) -> Result<Value, String> {
        match key {
            "info" => {
                let snapshot = self.clock.snapshot();
                let state = self.state.lock().unwrap();
                Ok(serde_json::json!({
                    "released": self.released.load(Ordering::Relaxed),
                    "dropped": self.dropped.load(Ordering::Relaxed),
                    "anchored": state.anchored,
                    "holding": state.held.is_some(),
                    "rate": snapshot.rate,
                    "paused": snapshot.paused,
                    "timebase": fmt_tb(self.timebase),
                }))
            }
            other => Err(format!("{}: unknown object `{other}`", self.io.name)),
        }
    }
}

impl Realtime {
    fn on_event(
        &self,
        state: &mut State,
        output: &Arc<dyn avplumber_f7k::graph::edge::Edge>,
        event: EdgeEvent,
    ) {
        match event {
            EdgeEvent::FlushStart => {
                // Whatever was waiting is from before the discontinuity.
                if state.held.take().is_some() {
                    self.dropped.fetch_add(1, Ordering::Relaxed);
                }
                state.release_one = true;
                output.push_event(EdgeEvent::FlushStart);
            }
            EdgeEvent::FlushStop { resume_at } => {
                self.playback.report_flush_stop();
                output.push_event(EdgeEvent::FlushStop { resume_at });
            }
            EdgeEvent::Spec(spec) => output.push_event(EdgeEvent::Spec(self.restamped_spec(spec))),
            // Nothing here is held inside a codec, so a drain only travels on.
            EdgeEvent::Drain => output.push_event(EdgeEvent::Drain),
            EdgeEvent::Eof => output.push_event(EdgeEvent::Eof),
        }
    }

    /// Frames leave in this node's time base, so the description does too.
    fn restamped_spec(&self, spec: Spec) -> Spec {
        match spec {
            Spec::Video {
                width,
                height,
                pix_fmt,
                sw_pix_fmt,
                frame_rate,
                sar,
                ..
            } => Spec::Video {
                width,
                height,
                pix_fmt,
                sw_pix_fmt,
                frame_rate,
                sar,
                time_base: self.timebase,
            },
            other => other,
        }
    }

    fn due(&self, state: &mut State, media: Ts) -> Due {
        if !media.is_valid() {
            // Nothing to pace by: release as it comes.
            return Due::Now;
        }
        if !state.anchored {
            return Due::Reanchor;
        }
        let wall = self.clock.map_to_wall(media.val, media.tb);
        if wall == AVP_NOPTS {
            // Paused between the snapshot and the mapping: hold, as if the
            // snapshot had said so.
            return Due::Hold;
        }
        let now = self.clock.now_us();
        let ahead = wall.saturating_sub(now);
        if ahead > self.discontinuity_us {
            return Due::Reanchor;
        }
        if ahead > 0 {
            return Due::Later(wall);
        }
        let behind = -ahead;
        if behind > self.tolerance_us {
            return Due::Reanchor;
        }
        if behind > self.discard_us {
            return Due::Late;
        }
        Due::Now
    }

    /// The release time in the output base, on the tick grid when there is
    /// one, and never at or before the previous stamp.
    fn output_stamp(&self, state: &mut State) -> Ts {
        let mut now_us = self.clock.now_us();
        if let Some(tick) = self.tick_us {
            now_us = ((now_us + tick / 2) / tick) * tick;
        }
        let mut pts = rescale(now_us, MICROSECONDS, self.timebase);
        if let Some(last) = state.last_out
            && pts <= last
        {
            pts = last + 1;
        }
        state.last_out = Some(pts);
        Ts {
            val: pts,
            tb: self.timebase,
        }
    }
}

fn fmt_tb(tb: AvpRational) -> String {
    format!("{}/{}", tb.num, tb.den)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicI64};

    use super::*;
    use avplumber_f7k::Instance;
    use avplumber_f7k::graph::BufferedEdge;
    use avplumber_f7k::graph::edge::{Edge, Wakeup};
    use avplumber_f7k::graph::media::test_media;
    use avplumber_f7k::graph::node::Node;
    use avplumber_f7k::services::clock::ClockSnapshot;

    /// A clock whose "now" the test moves by hand. Same arithmetic as the real
    /// one, in microseconds, without the process clock.
    struct TestClock {
        now: AtomicI64,
        snap: Mutex<ClockSnapshot>,
    }

    impl TestClock {
        fn new() -> Self {
            Self {
                now: AtomicI64::new(0),
                snap: Mutex::new(ClockSnapshot {
                    origin_src_us: 0,
                    origin_wall_us: 0,
                    rate: 1.0,
                    paused: false,
                    epoch: 0,
                }),
            }
        }

        fn set_now_ms(&self, ms: i64) {
            self.now.store(ms * 1000, Ordering::Release);
        }
    }

    impl SyncGroup for TestClock {
        fn now_us(&self) -> i64 {
            self.now.load(Ordering::Acquire)
        }
        fn set_rate(&self, rate: f64) {
            let mut s = self.snap.lock().unwrap();
            let now = self.now_us();
            s.origin_src_us += ((now - s.origin_wall_us) as f64 * s.rate) as i64;
            s.origin_wall_us = now;
            s.rate = rate;
            s.epoch += 1;
        }
        fn set_paused(&self, paused: bool) {
            let mut s = self.snap.lock().unwrap();
            if s.paused == paused {
                return;
            }
            let now = self.now_us();
            if paused {
                s.origin_src_us += ((now - s.origin_wall_us) as f64 * s.rate) as i64;
                s.origin_wall_us = now;
            } else {
                s.origin_wall_us = now;
            }
            s.paused = paused;
            s.epoch += 1;
        }
        fn reset(&self, new_pos: i64, tb: AvpRational) {
            let mut s = self.snap.lock().unwrap();
            s.origin_src_us = rescale(new_pos, tb, MICROSECONDS);
            s.origin_wall_us = self.now_us();
            s.epoch += 1;
        }
        fn map_to_wall(&self, src_pts: i64, tb: AvpRational) -> i64 {
            let s = self.snap.lock().unwrap();
            let src = rescale(src_pts, tb, MICROSECONDS);
            s.origin_wall_us + ((src - s.origin_src_us) as f64 / s.rate) as i64
        }
        fn join_offset(&self, _offset_us: i64) {}
        fn snapshot(&self) -> ClockSnapshot {
            *self.snap.lock().unwrap()
        }
    }

    struct Harness {
        node: Polling<Realtime>,
        clock: Arc<TestClock>,
        playback: Arc<Playback>,
        input: Arc<dyn Edge>,
        output: Arc<dyn Edge>,
        ctx: NodePollContext,
    }

    fn harness(params: Value) -> Harness {
        let instance = Instance::new();
        let clock = Arc::new(TestClock::new());
        instance.services().clocks.insert("g", clock.clone());
        let playback = instance.services().playback("g");
        // A source to seek: the mailbox is never read here, only the clock
        // reset and the pending position matter.
        let wake: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {});
        playback.bind_source(wake, None, false, 0);
        let ctx = BuildCtx {
            instance: &instance,
            name: "rt",
            params: &params,
            sync_group: Some("g"),
        };
        let spec: RealtimeSpec = serde_json::from_value(params.clone()).unwrap();
        let node = spec.build("rt", &ctx).expect("builds");
        let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(64));
        let output: Arc<dyn Edge> = Arc::new(BufferedEdge::new(64));
        node.bind_source("in", input.clone());
        node.bind_sink("out", output.clone());
        node.start();
        Harness {
            node,
            clock,
            playback,
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
                    _ => break,
                }
            }
        }

        fn push_ms(&self, ms: &[i64]) {
            for &pts in ms {
                assert!(
                    self.input
                        .offer(test_media(AvpMediaType::VIDEO, pts))
                        .is_ok()
                );
            }
        }

        /// Output PTS in the node's base, buffers only.
        fn released(&self) -> Vec<i64> {
            let mut out = Vec::new();
            while let Some(item) = self.output.try_take() {
                if let EdgeItem::Buffer(media) = item {
                    out.push(media.ts().rescale(self.node.timebase).val);
                }
            }
            out
        }
    }

    #[test]
    fn frames_wait_for_their_time_and_are_stamped_with_release_time() {
        let mut h = harness(serde_json::json!({"tick_period": "1/25"}));
        h.clock.set_now_ms(100);
        h.push_ms(&[0, 40, 80]);
        // The first frame anchors the clock: media 0 is now (100 ms).
        h.step_until_idle();
        assert_eq!(
            h.released(),
            vec![3],
            "100 ms rounds to tick 3 of a 40 ms grid"
        );
        assert_eq!(h.playback.status()["media_ms"], serde_json::json!(0));
        assert_eq!(h.playback.serial(), 1);
        // Frame 40 is due at 140 ms: held until then.
        h.step_until_idle();
        assert_eq!(h.released(), Vec::<i64>::new());
        h.clock.set_now_ms(140);
        h.step_until_idle();
        assert_eq!(h.released(), vec![4]);
        h.clock.set_now_ms(180);
        h.step_until_idle();
        assert_eq!(h.released(), vec![5]);
        assert_eq!(h.playback.status()["media_ms"], serde_json::json!(80));
    }

    #[test]
    fn rate_scales_the_wait_and_a_late_frame_is_dropped() {
        let mut h = harness(serde_json::json!({}));
        h.clock.set_now_ms(0);
        h.push_ms(&[0]);
        h.step_until_idle();
        assert_eq!(h.released().len(), 1);
        h.clock.set_rate(2.0);
        // At 2x, media 40 is due 20 ms after media 0.
        h.push_ms(&[40]);
        h.clock.set_now_ms(10);
        h.step_until_idle();
        assert_eq!(h.released(), Vec::<i64>::new(), "not yet");
        h.clock.set_now_ms(20);
        h.step_until_idle();
        assert_eq!(h.released().len(), 1);
        // Media 80 is due at 40 ms; at 200 ms it is 160 ms late, past the
        // 80 ms discard but within the 250 ms tolerance: dropped.
        h.push_ms(&[80]);
        h.clock.set_now_ms(200);
        h.step_until_idle();
        assert_eq!(h.released(), Vec::<i64>::new());
        assert_eq!(
            h.node.get_object("info").unwrap()["dropped"],
            serde_json::json!(1)
        );
    }

    #[test]
    fn paused_holds_except_the_frame_a_seek_landed_on() {
        let mut h = harness(serde_json::json!({}));
        h.clock.set_now_ms(0);
        h.push_ms(&[0]);
        h.step_until_idle();
        assert_eq!(h.released().len(), 1);
        h.playback.pause();
        h.push_ms(&[40]);
        h.clock.set_now_ms(1000);
        h.step_until_idle();
        assert_eq!(h.released(), Vec::<i64>::new(), "paused: nothing leaves");
        // A seek: the clock is reset, then flush, then the target frame, which
        // a paused viewer must see.
        h.playback
            .seek(avplumber_f7k::services::playback::Target::MediaMs(2000))
            .unwrap();
        h.input.push_event(EdgeEvent::FlushStart);
        h.input.push_event(EdgeEvent::FlushStop { resume_at: None });
        h.push_ms(&[2000, 2040]);
        h.step_until_idle();
        assert_eq!(
            h.released().len(),
            1,
            "exactly the frame the seek landed on"
        );
        assert_eq!(h.playback.status()["media_ms"], serde_json::json!(2000));
        assert_eq!(
            h.node.get_object("info").unwrap()["dropped"],
            serde_json::json!(1),
            "the frame held across the flush was stale"
        );
        assert_eq!(h.playback.status()["pending"], serde_json::json!(false));
        // Resumed at 1040 ms showing media 2000: media 2040 follows one period
        // later, at 1080 ms.
        h.clock.set_now_ms(1040);
        h.playback.resume();
        h.step_until_idle();
        assert_eq!(h.released(), Vec::<i64>::new(), "not yet");
        h.clock.set_now_ms(1080);
        h.step_until_idle();
        assert_eq!(h.released().len(), 1, "resumed: the next frame follows");
    }

    #[test]
    fn a_jump_ahead_reanchors_instead_of_waiting() {
        let mut h = harness(serde_json::json!({"discontinuity_threshold": 0.5}));
        h.clock.set_now_ms(0);
        h.push_ms(&[0, 5000]);
        h.step_until_idle();
        assert_eq!(h.released().len(), 2, "5 s ahead is a jump: released now");
        assert_eq!(h.playback.status()["media_ms"], serde_json::json!(5000));
        // Output PTS stayed monotonic even though both left at the same instant.
        h.push_ms(&[5040]);
        h.clock.set_now_ms(40);
        h.step_until_idle();
        assert_eq!(h.released().len(), 1);
    }

    #[test]
    fn a_video_spec_leaves_in_the_output_time_base() {
        let mut h = harness(serde_json::json!({"tick_period": "1/50"}));
        h.input.push_event(EdgeEvent::Spec(Spec::Video {
            width: 2,
            height: 2,
            pix_fmt: 0,
            sw_pix_fmt: -1,
            frame_rate: AvpRational { num: 25, den: 1 },
            sar: AvpRational { num: 1, den: 1 },
            time_base: AvpRational { num: 1, den: 90000 },
        }));
        h.step_until_idle();
        let Some(EdgeItem::Event(EdgeEvent::Spec(Spec::Video { time_base, .. }))) =
            h.output.try_take()
        else {
            panic!("the spec is forwarded");
        };
        assert_eq!(time_base, AvpRational { num: 1, den: 50 });
    }
}
