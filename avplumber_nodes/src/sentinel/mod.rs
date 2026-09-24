//! `sentinel_video` — one video member of a correction group.
//!
//! The shared [`CorrectionGroup`] decides whether the next input occupies the
//! cursor, is dropped, or waits behind a repeated frame. This node only stamps
//! that decision and, when the input has been gone longer than `timeout` and
//! the live clock is ahead, emits a frozen frame or a slate. Audio in the same
//! group is `resample_audio`, which spends the same budget as a sample stretch.

pub(crate) mod backup;
pub(crate) mod report;

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::edge::{Edge, EdgeEvent, EdgeItem, Push};
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::grain::Grain;
use avplumber_f7k::graph::media::{AvpMediaType, AvpRational};
use avplumber_f7k::graph::node::Polled;
use avplumber_f7k::graph::pad::NodePads;
use avplumber_f7k::graph::poll_ctx::NodePollContext;
use avplumber_f7k::graph::spec::Spec;
use avplumber_f7k::graph::timestamp::Ts;
use avplumber_f7k::node_api::{Io, NodeObjects, PollNode, Polling};
use avplumber_f7k::services::clock::SyncGroup;
use avplumber_f7k::services::correction::{Action, CorrectionGroup, MemberKind};

use crate::correction_cfg::{self, CorrectionParams};

pub use backup::{load_picture, store_picture};
pub use report::Reporter;

#[derive(serde::Deserialize)]
pub struct SentinelSpec {
    #[serde(flatten)]
    correction: CorrectionParams,
    /// Seconds to repeat the last real frame before switching to the slate.
    #[serde(default = "default_freeze")]
    freeze: f64,
    /// Seconds of one output frame, when the input spec has no frame rate.
    #[serde(default)]
    frame_duration: Option<f64>,
    #[serde(default)]
    backup_image: Option<String>,
    #[serde(default)]
    backup_picture_buffer: Option<String>,
    #[serde(default)]
    initial_picture_buffer: Option<String>,
}

fn default_freeze() -> f64 {
    5.0
}

impl NodeSpec for SentinelSpec {
    const TYPE_NAME: &'static str = "sentinel_video";
    type Node = Polling<SentinelVideo>;

    fn build(self, name: &str, ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        let reporter = Reporter::open(
            self.correction.history_file.as_deref(),
            self.correction.history_file_text.as_deref(),
            self.correction.reporting_url.clone(),
            self.correction.track_wallclock,
            self.correction.max_wallclock_drift,
            self.correction.wallclock_drift_grace_period,
            self.correction.history_report_interval,
        )?;
        let group_name = if self.correction.correction_group.is_empty() {
            "default".to_string()
        } else {
            self.correction.correction_group.clone()
        };
        let group = ctx.correction(&group_name);
        group.register_stream(name, MemberKind::Video, 0);
        let slate = match (&self.backup_picture_buffer, &self.backup_image) {
            (Some(picture), _) => backup::load_picture(picture),
            (None, Some(path)) => Some(backup::load_image(path)?),
            (None, None) => None,
        };
        let clock = ctx.sync_group.map(|name| ctx.clock(name));
        let hold_until = self.correction.hold_until_unix_ms()?;
        Ok(Polling(SentinelVideo {
            io: Io::new(name, NodePhase::Poll),
            group,
            clock,
            hold_until,
            params: self.correction,
            freeze_s: self.freeze,
            frame_duration_s: self.frame_duration,
            slate: Mutex::new(slate),
            initial_picture: self.initial_picture_buffer,
            state: Mutex::new(State::default()),
            card: AtomicU64::new(0),
            reporter,
        }))
    }
}

pub struct SentinelVideo {
    io: Io,
    group: Arc<CorrectionGroup>,
    clock: Option<Arc<dyn SyncGroup>>,
    hold_until: Option<i64>,
    params: CorrectionParams,
    freeze_s: f64,
    frame_duration_s: Option<f64>,
    slate: Mutex<Option<Grain>>,
    initial_picture: Option<String>,
    state: Mutex<State>,
    card: AtomicU64,
    reporter: Reporter,
}

struct Pending {
    grain: Grain,
    /// `false` after a repeat, so the follow-up propose spends no more budget.
    fresh: bool,
}

struct State {
    configured: bool,
    next: Option<Ts>,
    last_input: Option<i64>,
    last_shift: Option<i64>,
    last_frame: Option<Grain>,
    pending: Option<Pending>,
    pending_out: Option<Grain>,
    backups: VecDeque<Grain>,
    conceal_origin: Option<i64>,
    /// One frame in seconds, learned from a video spec before the timebase is set.
    rate_seconds: Option<f64>,
    stall_deadline: Option<Instant>,
    quantum: i64,
    tb: AvpRational,
    start: Ts,
}

impl Default for State {
    fn default() -> Self {
        Self {
            configured: false,
            next: None,
            last_input: None,
            last_shift: None,
            last_frame: None,
            pending: None,
            pending_out: None,
            backups: VecDeque::new(),
            conceal_origin: None,
            rate_seconds: None,
            stall_deadline: None,
            quantum: 0,
            tb: AvpRational { num: 1, den: 1000 },
            start: Ts::invalid(),
        }
    }
}

impl PollNode for SentinelVideo {
    fn io(&self) -> &Io {
        &self.io
    }

    fn pads(&self) -> NodePads {
        NodePads::siso(AvpMediaType::VIDEO, AvpMediaType::VIDEO)
    }

    fn start(&self) {
        let mut state = self.state.lock().unwrap();
        *state = State::default();
        if let Some(name) = &self.initial_picture {
            state.last_frame = backup::load_picture(name);
        }
    }

    fn objects(&self) -> Option<&dyn NodeObjects> {
        Some(self)
    }

    fn step(&self, ctx: &mut NodePollContext) -> Result<Polled, NodeError> {
        let input = self.io.input()?;
        let output = self.io.output()?;
        let mut state = self.state.lock().unwrap();

        if let Some(grain) = state.pending_out.take() {
            return self.push(&output, grain, &mut state, ctx);
        }
        if let Some(grain) = state.backups.pop_front() {
            self.set_card(true, grain.ts());
            return self.push(&output, grain, &mut state, ctx);
        }

        if self.holding() {
            self.drop_while_held(&input, &output);
            ctx.wait_readable(input);
            ctx.wait_deadline(Instant::now() + Duration::from_millis(20));
            return Ok(Polled::Idle);
        }

        self.drain_events(&input, &output, &mut state);

        if state.pending.is_none() {
            match input.try_take() {
                Some(EdgeItem::Buffer(grain)) => {
                    state.stall_deadline = None;
                    state.conceal_origin = None;
                    state.backups.clear();
                    state.pending = Some(Pending { grain, fresh: true });
                }
                Some(EdgeItem::Event(event)) => {
                    self.on_event(&output, &mut state, event);
                    return Ok(Polled::Again);
                }
                None => return self.stalled(ctx, &input, &output, &mut state),
            }
        }

        self.correct(&output, &mut state, ctx)
    }
}

impl NodeObjects for SentinelVideo {
    fn get_object(&self, key: &str) -> Result<Value, String> {
        match key {
            "card_status" => {
                let raw = self.card.load(Ordering::Relaxed);
                Ok(json!({
                    "card": (raw & 1) == 1,
                    "changed_at_ms": raw >> 1,
                }))
            }
            "stats" => {
                let state = self.state.lock().unwrap();
                Ok(json!({
                    "group_shift": self.group.group_shift().map(|ts| ts.val),
                    "local_shift": self.group.local_shift(&self.io.name).map(|ts| ts.val),
                    "next_ts": state.next.map(|ts| ts.val),
                    "card": (self.card.load(Ordering::Relaxed) & 1) == 1,
                }))
            }
            other => Err(format!("{}: unknown object `{other}`", self.io.name)),
        }
    }
}

impl SentinelVideo {
    fn holding(&self) -> bool {
        self.hold_until.is_some_and(|until| unix_ms() < until)
    }

    fn drop_while_held(&self, input: &Arc<dyn Edge>, output: &Arc<dyn Edge>) {
        while let Some(item) = input.try_take() {
            if let EdgeItem::Event(event) = item {
                output.push_event(event);
            }
        }
    }

    fn drain_events(&self, input: &Arc<dyn Edge>, output: &Arc<dyn Edge>, state: &mut State) {
        while let Some(EdgeItem::Event(event)) = input.peek_clone(0) {
            input.pop();
            self.on_event(output, state, event);
        }
    }

    fn on_event(&self, output: &Arc<dyn Edge>, state: &mut State, event: EdgeEvent) {
        match &event {
            EdgeEvent::FlushStart => {
                state.pending = None;
                state.last_frame = None;
                state.last_input = None;
                state.conceal_origin = None;
            }
            EdgeEvent::Spec(spec) => {
                if let Spec::Video { frame_rate, .. } = spec
                    && self.frame_duration_s.is_none()
                    && frame_rate.num > 0
                    && frame_rate.den > 0
                {
                    state.rate_seconds = Some(frame_rate.den as f64 / frame_rate.num as f64);
                }
            }
            _ => {}
        }
        output.push_event(event);
    }

    fn correct(
        &self,
        output: &Arc<dyn Edge>,
        state: &mut State,
        ctx: &mut NodePollContext,
    ) -> Result<Polled, NodeError> {
        let Some(input_ts) = state.pending.as_ref().map(|pending| pending.grain.ts()) else {
            return Ok(Polled::Again);
        };
        let fresh = state.pending.as_ref().unwrap().fresh;
        if !input_ts.is_valid() {
            log::warn!("{}: dropping a frame with no timestamp", self.io.name);
            state.pending = None;
            return Ok(Polled::Again);
        }
        self.ensure_configured(state, input_ts.tb)?;
        let tb = state.tb;
        let input_ticks = input_ts.rescale(tb).val;
        let quantum = state.quantum.max(1);
        if state.next.is_none() {
            state.next = Some(if self.params.forward_start_shift {
                input_ts.rescale(tb)
            } else {
                state.start
            });
        }
        let next = state.next.unwrap();
        // Nominal media time, not the PTS gap. A jump is a shift error (`obs`),
        // and counting it again here would grant a slew step as large as the glitch.
        let dt = if fresh && state.last_input.is_some() {
            Ts { val: quantum, tb }
        } else {
            Ts { val: 0, tb }
        };
        let decision = self
            .group
            .propose(
                &self.io.name,
                Ts {
                    val: input_ticks,
                    tb,
                },
                next,
                Ts { val: quantum, tb },
                dt,
            )
            .map_err(|e| self.io.error(NodePhase::Poll, e))?;
        self.note_shift(state, decision.local_shift, Ts { val: decision.next_ts, tb });
        state.next = Some(Ts {
            val: decision.next_ts,
            tb,
        });

        match decision.action {
            Action::Drop => {
                state.last_input = Some(input_ticks);
                state.pending = None;
                self.set_card(false, state.next.unwrap());
                Ok(Polled::Again)
            }
            Action::Repeat { emit_pts } => {
                let Some(mut picture) = state.last_frame.clone() else {
                    return self.emit_pending(output, state, ctx, Ts { val: emit_pts, tb });
                };
                if let Some(pending) = state.pending.as_mut() {
                    pending.fresh = false;
                }
                picture.set_ts(Ts { val: emit_pts, tb });
                self.set_card(true, Ts { val: emit_pts, tb });
                self.push(output, picture, state, ctx)
            }
            Action::Emit { emit_pts } | Action::Stretch { emit_pts, .. } => {
                self.emit_pending(output, state, ctx, Ts { val: emit_pts, tb })
            }
        }
    }

    fn stalled(
        &self,
        ctx: &mut NodePollContext,
        input: &Arc<dyn Edge>,
        output: &Arc<dyn Edge>,
        state: &mut State,
    ) -> Result<Polled, NodeError> {
        if input.is_closed() {
            return self.finish_eof(output, state, ctx);
        }
        let timeout = Duration::from_secs_f64(self.params.timeout.max(0.0));
        let deadline = *state.stall_deadline.get_or_insert_with(|| Instant::now() + timeout);
        if Instant::now() < deadline {
            ctx.wait_readable(input.clone());
            ctx.wait_deadline(deadline);
            return Ok(Polled::Idle);
        }
        if let Some(clock) = &self.clock {
            self.group.advance_live(clock.as_ref());
        }
        let cursor = state.next.unwrap_or(state.start);
        if !cursor.is_valid() || state.quantum <= 0 {
            state.stall_deadline = Some(Instant::now() + timeout);
            ctx.wait_readable(input.clone());
            ctx.wait_deadline(state.stall_deadline.unwrap());
            return Ok(Polled::Idle);
        }
        let timeout_ticks = correction_cfg::seconds_to_ticks(self.params.timeout, state.tb).max(0);
        let slots = self
            .group
            .conceal(&self.io.name, cursor, timeout_ticks, state.quantum)
            .map_err(|e| self.io.error(NodePhase::Poll, e))?;
        if slots.is_empty() {
            state.stall_deadline = Some(Instant::now() + timeout.max(Duration::from_millis(20)));
            ctx.wait_readable(input.clone());
            ctx.wait_deadline(state.stall_deadline.unwrap());
            return Ok(Polled::Idle);
        }
        let mut frames = VecDeque::new();
        for slot in slots {
            if state.conceal_origin.is_none() {
                state.conceal_origin = Some(slot.val);
            }
            if let Some(frame) = self.backup_frame(state, slot) {
                frames.push_back(frame);
            }
        }
        state.next = self
            .group
            .member_cursor(&self.io.name)
            .map(|cursor| cursor.next_ts)
            .or(state.next);
        state.stall_deadline = Some(Instant::now() + timeout.max(Duration::from_millis(1)));
        let Some(frame) = frames.pop_front() else {
            ctx.wait_readable(input.clone());
            ctx.wait_deadline(state.stall_deadline.unwrap());
            return Ok(Polled::Idle);
        };
        state.backups.append(&mut frames);
        self.set_card(true, frame.ts());
        self.push(output, frame, state, ctx)
    }

    fn emit_pending(
        &self,
        output: &Arc<dyn Edge>,
        state: &mut State,
        ctx: &mut NodePollContext,
        stamp: Ts,
    ) -> Result<Polled, NodeError> {
        let Some(pending) = state.pending.take() else {
            return Ok(Polled::Again);
        };
        let input_ticks = pending.grain.ts().rescale(state.tb).val;
        state.last_input = Some(input_ticks);
        let mut grain = pending.grain;
        state.last_frame = Some(grain.clone());
        grain.set_ts(stamp);
        self.set_card(false, stamp);
        self.reporter.observe_wallclock(stamp, state.start);
        self.push(output, grain, state, ctx)
    }

    fn backup_frame(&self, state: &State, pts: Ts) -> Option<Grain> {
        let freeze_ticks = correction_cfg::seconds_to_ticks(self.freeze_s, state.tb);
        let within_freeze = state
            .conceal_origin
            .map(|origin| pts.val.saturating_sub(origin) < freeze_ticks)
            .unwrap_or(true);
        let slate = self.slate.lock().unwrap().clone();
        let source = if within_freeze && freeze_ticks > 0 {
            state.last_frame.clone().or(slate)
        } else {
            slate.or_else(|| state.last_frame.clone())
        };
        source.map(|mut grain| {
            grain.set_ts(pts);
            grain
        })
    }

    fn finish_eof(
        &self,
        output: &Arc<dyn Edge>,
        state: &mut State,
        ctx: &mut NodePollContext,
    ) -> Result<Polled, NodeError> {
        if self.params.eof_passthrough
            && let Some(pending) = state.pending.take()
        {
            return self.push(output, pending.grain, state, ctx);
        }
        Ok(Polled::Done)
    }

    fn push(
        &self,
        output: &Arc<dyn Edge>,
        grain: Grain,
        state: &mut State,
        ctx: &mut NodePollContext,
    ) -> Result<Polled, NodeError> {
        if output.is_full() {
            state.pending_out = Some(grain);
            ctx.wait_writable(output.clone());
            return Ok(Polled::Idle);
        }
        match output.offer(grain) {
            Ok(()) | Err((Push::Dropped | Push::Accepted, _)) => Ok(Polled::Again),
            Err((Push::Full, grain)) => {
                state.pending_out = Some(grain);
                ctx.wait_writable(output.clone());
                Ok(Polled::Idle)
            }
            Err((Push::Closed, _)) => Ok(Polled::Done),
        }
    }

    fn ensure_configured(&self, state: &mut State, frame_tb: AvpRational) -> Result<(), NodeError> {
        if state.configured {
            return Ok(());
        }
        state.start = correction_cfg::configure_member(&self.group, &self.params, frame_tb)
            .map_err(|e| self.io.error(NodePhase::Poll, e))?;
        state.tb = state.start.tb;
        let tb = state.tb;
        state.quantum = match self.frame_duration_s.or(state.rate_seconds) {
            Some(seconds) => correction_cfg::seconds_to_ticks(seconds, tb).max(1),
            None => {
                return Err(self.io.error(
                    NodePhase::Poll,
                    "sentinel_video needs frame_duration or a video spec with a frame rate",
                ));
            }
        };
        state.configured = true;
        Ok(())
    }

    fn note_shift(&self, state: &mut State, shift: i64, at: Ts) {
        if state.last_shift == Some(shift) {
            return;
        }
        state.last_shift = Some(shift);
        self.reporter.shift_changed(at, shift, state.start);
    }

    fn set_card(&self, card: bool, pts: Ts) {
        let bit = u64::from(card);
        let current = self.card.load(Ordering::Relaxed);
        if current & 1 == bit {
            return;
        }
        let ms = if pts.is_valid() {
            pts.rescale(avplumber_f7k::graph::timebase::MILLISECONDS).val
        } else {
            0
        };
        let ms = ms.max(0) as u64;
        self.card.store(bit | (ms << 1), Ordering::Relaxed);
    }
}

fn unix_ms() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(all(test, feature = "ffmpeg"))]
mod tests;
