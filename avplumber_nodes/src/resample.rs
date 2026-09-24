//! `resample_audio` — format conversion and the audio member of a correction group.
//!
//! A grouped node spends the group's slew budget through `swr_set_compensation`
//! and inserts silence only after `timeout`, once the resampler has been drained.
//! It does not run the standalone drift detector and does not rebuild the
//! resampler when the input timestamp jumps.
//!
//! Without a group the node keeps a private cursor. Drift past `max_drift` for
//! 40 frames inserts silence or drops output, unless `compensation` is non-zero,
//! in which case libswresample's `async` option owns that and the detector stays
//! quiet. A discontinuity larger than half a second resets the private cursor
//! and the resampler.

use std::ffi::{CString, c_void};
use std::ptr;
use std::sync::{Arc, Mutex};

use rsmpeg::avutil::AVFrame;
use rusty_ffmpeg::ffi::{self, SwrContext};

use avplumber_f7k::factory::{BuildCtx, NodeSpec};
use avplumber_f7k::graph::edge::{Edge, EdgeEvent, EdgeItem};
use avplumber_f7k::graph::error::{NodeError, NodePhase};
use avplumber_f7k::graph::grain::Grain;
use avplumber_f7k::graph::media::{AvpMediaType, AvpRational};
use avplumber_f7k::graph::node::Processed;
use avplumber_f7k::graph::pad::NodePads;
use avplumber_f7k::graph::timestamp::{Ts, TsDelta};
use avplumber_f7k::node_api::{Blocking, BlockingIo, BlockingNode};
use avplumber_f7k::services::clock::SyncGroup;
use avplumber_f7k::services::correction::{Action, CorrectionGroup, MemberKind};

use crate::correction_cfg::{self, CorrectionParams};
use crate::sentinel::report::Reporter;

const DRIFT_GRACE_FRAMES: u32 = 40;
const DISCONTINUITY_S: f64 = 0.5;

#[derive(serde::Deserialize)]
pub struct ResampleSpec {
    #[serde(flatten)]
    correction: CorrectionParams,
    #[serde(default)]
    sample_rate: Option<i32>,
    #[serde(default)]
    sample_fmt: Option<String>,
    #[serde(default)]
    channel_layout: Option<String>,
    /// Standalone only. Non-zero selects libswresample's `async` compensation
    /// and disables the silence/drop detector.
    #[serde(default)]
    compensation: f64,
    /// Standalone only, seconds.
    #[serde(default = "default_drift")]
    max_drift: f64,
    /// Seconds of one output packet. Absent means `nb_samples / sample_rate`.
    #[serde(default)]
    frame_duration: Option<f64>,
}

fn default_drift() -> f64 {
    0.001
}

impl NodeSpec for ResampleSpec {
    const TYPE_NAME: &'static str = "resample_audio";
    type Node = Blocking<ResampleAudio>;

    fn build(self, name: &str, ctx: &BuildCtx<'_>) -> Result<Self::Node, String> {
        let grouped = !self.correction.correction_group.is_empty();
        let group = grouped.then(|| {
            let group = ctx.correction(&self.correction.correction_group);
            group.register_stream(name, MemberKind::Audio, 0);
            group
        });
        let reporter = Reporter::open(
            self.correction.history_file.as_deref(),
            self.correction.history_file_text.as_deref(),
            self.correction.reporting_url.clone(),
            self.correction.track_wallclock,
            self.correction.max_wallclock_drift,
            self.correction.wallclock_drift_grace_period,
            self.correction.history_report_interval,
        )?;
        Ok(Blocking(ResampleAudio {
            io: BlockingIo::new(name),
            group,
            clock: ctx.sync_group.map(|name| ctx.clock(name)),
            params: self.correction,
            out_rate: self.sample_rate,
            out_fmt: self.sample_fmt,
            out_layout: self.channel_layout,
            compensation: self.compensation,
            max_drift: self.max_drift,
            frame_duration_s: self.frame_duration,
            reporter,
            state: Mutex::new(AudioState::default()),
        }))
    }
}

pub struct ResampleAudio {
    io: BlockingIo,
    group: Option<Arc<CorrectionGroup>>,
    clock: Option<Arc<dyn SyncGroup>>,
    params: CorrectionParams,
    out_rate: Option<i32>,
    out_fmt: Option<String>,
    out_layout: Option<String>,
    compensation: f64,
    max_drift: f64,
    frame_duration_s: Option<f64>,
    reporter: Reporter,
    state: Mutex<AudioState>,
}

struct AudioState {
    engine: Option<Engine>,
    configured: bool,
    tb: AvpRational,
    start: Ts,
    next: Option<Ts>,
    last_input: Option<i64>,
    last_shift: Option<TsDelta>,
    last_samples: i32,
    /// Standalone cursor. Unused while grouped.
    solo_next: Option<Ts>,
    drift_run: u32,
    drained_stall: bool,
}

impl Default for AudioState {
    fn default() -> Self {
        Self {
            engine: None,
            configured: false,
            tb: AvpRational { num: 1, den: 1000 },
            start: Ts::invalid(),
            next: None,
            last_input: None,
            last_shift: None,
            last_samples: 0,
            solo_next: None,
            drift_run: 0,
            drained_stall: false,
        }
    }
}

struct Engine {
    swr: *mut SwrContext,
    in_rate: i32,
    out_rate: i32,
    in_fmt: i32,
    out_fmt: i32,
    channels: i32,
    out_layout: ffi::AVChannelLayout,
}

unsafe impl Send for Engine {}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe {
            ffi::swr_free(&mut self.swr);
            ffi::av_channel_layout_uninit(&mut self.out_layout);
        }
    }
}

impl BlockingNode for ResampleAudio {
    fn io(&self) -> &BlockingIo {
        &self.io
    }

    fn pads(&self) -> NodePads {
        NodePads::siso(AvpMediaType::AUDIO, AvpMediaType::AUDIO)
    }

    fn start(&self) {
        *self.state.lock().unwrap() = AudioState::default();
    }

    fn step(&self) -> Result<Processed, NodeError> {
        let input = self.io.input()?;
        let output = self.io.output()?;
        let timeout = if self.group.is_some() {
            (self.params.timeout * 1000.0).round() as i32
        } else {
            -1
        };
        match input.take(timeout) {
            Some(EdgeItem::Event(event)) => {
                self.on_event(&output, event)?;
                Ok(Processed::Again)
            }
            Some(EdgeItem::Buffer(grain)) => self.on_frame(&input, &output, grain),
            None if input.is_closed() => {
                output.push_event(EdgeEvent::Eof);
                Ok(Processed::Done)
            }
            None => self.on_stall(&input, &output),
        }
    }
}

impl ResampleAudio {
    fn on_event(&self, output: &Arc<dyn Edge>, event: EdgeEvent) -> Result<(), NodeError> {
        if let EdgeEvent::FlushStart = &event {
            let mut state = self.state.lock().unwrap();
            state.last_input = None;
            state.drift_run = 0;
            state.drained_stall = false;
            // Grouped playback keeps its cursor and its resampler across a
            // timestamp jump. Standalone starts the cursor over.
            if self.group.is_none() {
                state.solo_next = None;
                state.engine = None;
            }
        }
        output.push_event(event);
        Ok(())
    }

    fn on_frame(
        &self,
        input: &Arc<dyn Edge>,
        output: &Arc<dyn Edge>,
        grain: Grain,
    ) -> Result<Processed, NodeError> {
        let Grain::Audio(frame) = grain else {
            return Err(self.io.error(
                NodePhase::Process,
                "resample_audio expects decoded audio frames",
            ));
        };
        let mut state = self.state.lock().unwrap();
        state.drained_stall = false;
        let in_ts = frame_ts(&frame);
        if !in_ts.is_valid() {
            return Err(self.io.error(NodePhase::Process, "audio frame has no timestamp"));
        }
        self.ensure_engine(&mut state, &frame)?;
        self.ensure_configured(&mut state, in_ts.timebase())?;
        let tb = state.tb;
        let samples = frame.nb_samples.max(1);
        state.last_samples = samples;
        let duration_ticks = self.duration_ticks(&state, samples);
        let duration = TsDelta::new(duration_ticks, tb);
        let input_ticks = in_ts.rescale(tb).ticks();

        if self.group.is_some() {
            return self.grouped(input, output, &mut state, frame, input_ticks, duration);
        }
        self.standalone(input, output, &mut state, frame, in_ts, duration)
    }

    fn grouped(
        &self,
        input: &Arc<dyn Edge>,
        output: &Arc<dyn Edge>,
        state: &mut AudioState,
        frame: AVFrame,
        input_ticks: i64,
        duration: TsDelta,
    ) -> Result<Processed, NodeError> {
        let group = self.group.as_ref().unwrap();
        let tb = state.tb;
        if state.next.is_none() {
            state.next = Some(if self.params.forward_start_shift {
                Ts::new(input_ticks, tb)
            } else {
                state.start
            });
        }
        let next = state.next.unwrap();
        let dt = if state.last_input.is_some() {
            duration
        } else {
            TsDelta::zero(tb)
        };
        let decision = group
            .propose(
                &self.io.name,
                Ts::new(input_ticks, tb),
                next,
                duration,
                dt,
            )
            .map_err(|e| self.io.error(NodePhase::Process, e))?;
        if state.last_shift != Some(decision.local_shift) {
            state.last_shift = Some(decision.local_shift);
            self.reporter.shift_changed(decision.next_ts, decision.local_shift, state.start);
        }
        state.next = Some(decision.next_ts);
        state.last_input = Some(input_ticks);
        let engine = state.engine.as_mut().unwrap();
        if decision.rebased {
            engine.reset_delay()?;
        }
        let (emit_pts, delta) = match decision.action {
            Action::Stretch {
                emit_pts,
                sample_delta,
            } => (emit_pts, sample_delta.ticks()),
            Action::Emit { emit_pts } => (emit_pts, 0),
            Action::Drop => return Ok(Processed::Again),
            Action::Repeat { emit_pts } => (emit_pts, 0),
        };
        if delta != 0 {
            let sample_delta = ticks_to_samples(delta, engine.out_rate, tb);
            engine.compensate(sample_delta, frame.nb_samples.max(1))?;
        }
        let mut out = engine.convert(Some(&frame), 0)?;
        let stamp = emit_pts;
        set_frame_ts(&mut out, stamp);
        self.reporter.observe_wallclock(stamp, state.start);
        self.io.push_from(input, output, Grain::Audio(out))
    }

    fn standalone(
        &self,
        input: &Arc<dyn Edge>,
        output: &Arc<dyn Edge>,
        state: &mut AudioState,
        frame: AVFrame,
        in_ts: Ts,
        duration: TsDelta,
    ) -> Result<Processed, NodeError> {
        let tb = state.tb;
        let input_ticks = in_ts.rescale(tb).ticks();
        let engine = state.engine.as_mut().unwrap();
        if let Some(next) = state.solo_next {
            let jump = (input_ticks - next.rescale(tb).ticks()).abs();
            if jump > correction_cfg::seconds_to_ticks(DISCONTINUITY_S, tb) {
                engine.reset_delay()?;
                state.solo_next = None;
                state.drift_run = 0;
            }
        }
        if state.solo_next.is_none() {
            state.solo_next = Some(Ts::new(input_ticks, tb));
        }
        let next_ticks = state.solo_next.unwrap().rescale(tb).ticks();
        if self.compensation == 0.0 {
            let delay = unsafe { ffi::swr_get_delay(engine.swr, engine.in_rate as i64) };
            let delay_ticks = samples_to_ticks(delay, engine.in_rate, tb);
            let drift = input_ticks - next_ticks - delay_ticks;
            let limit = correction_cfg::seconds_to_ticks(self.max_drift, tb).max(1);
            if drift.abs() > limit {
                state.drift_run += 1;
            } else {
                state.drift_run = 0;
            }
            if state.drift_run >= DRIFT_GRACE_FRAMES {
                let samples = ticks_to_samples(drift, engine.in_rate, tb);
                let cap = frame.nb_samples.max(1);
                if samples > 0 {
                    engine.inject_silence(samples.min(cap))?;
                } else if samples < 0 {
                    engine.drop_output((-samples).min(cap))?;
                }
                state.drift_run = 0;
            }
        }
        let mut out = engine.convert(Some(&frame), 0)?;
        let stamp = Ts::new(next_ticks, tb);
        set_frame_ts(&mut out, stamp);
        let produced =
            samples_to_ticks(out.nb_samples as i64, engine.out_rate, tb).max(duration.ticks());
        state.solo_next = Some(Ts::new(next_ticks + produced, tb));
        self.io.push_from(input, output, Grain::Audio(out))
    }

    fn on_stall(&self, input: &Arc<dyn Edge>, output: &Arc<dyn Edge>) -> Result<Processed, NodeError> {
        let Some(group) = &self.group else {
            return Ok(Processed::Again);
        };
        if let Some(clock) = &self.clock {
            group.advance_live(clock.as_ref());
        }
        let mut state = self.state.lock().unwrap();
        let cursor = state.next.unwrap_or(state.start);
        if !cursor.is_valid() || state.engine.is_none() {
            self.io.wait(20);
            return Ok(Processed::Again);
        }
        let timeout_ticks = correction_cfg::seconds_to_ticks(self.params.timeout, state.tb).max(0);
        let quantum = self.duration_ticks(&state, state.last_samples.max(1));
        let slots = group
            .conceal(
                &self.io.name,
                cursor,
                TsDelta::new(timeout_ticks, state.tb),
                TsDelta::new(quantum, state.tb),
            )
            .map_err(|e| self.io.error(NodePhase::Process, e))?;
        if slots.is_empty() {
            drop(state);
            self.io
                .wait((self.params.timeout * 1000.0).round().max(20.0) as u64);
            return Ok(Processed::Again);
        }
        let rate = state.engine.as_ref().unwrap().out_rate;
        let drained = if state.drained_stall {
            Vec::new()
        } else {
            state.drained_stall = true;
            state.engine.as_mut().unwrap().drain()?
        };
        let tb = state.tb;
        let mut at = cursor;
        let drained: Vec<(AVFrame, Ts)> = drained
            .into_iter()
            .map(|mut frame| {
                let stamp = at;
                let step = samples_to_ticks(frame.nb_samples as i64, rate, tb).max(1);
                set_frame_ts(&mut frame, stamp);
                at = Ts::new(stamp.ticks() + step, tb);
                (frame, stamp)
            })
            .collect();
        let samples = state.last_samples.max(1);
        let mut silence = Vec::new();
        {
            let engine = state.engine.as_mut().unwrap();
            for slot in &slots {
                let mut frame = engine.silence(samples)?;
                set_frame_ts(&mut frame, *slot);
                silence.push(frame);
            }
        }
        state.next = group.member_cursor(&self.io.name).map(|cursor| cursor.next_ts);
        drop(state);
        for (frame, _) in drained {
            self.io.push_from(input, output, Grain::Audio(frame))?;
        }
        for frame in silence {
            self.io.push_from(input, output, Grain::Audio(frame))?;
        }
        Ok(Processed::Again)
    }

    fn ensure_configured(&self, state: &mut AudioState, frame_tb: AvpRational) -> Result<(), NodeError> {
        if state.configured {
            return Ok(());
        }
        if let Some(group) = &self.group {
            state.start = correction_cfg::configure_member(group, &self.params, frame_tb)
                .map_err(|e| self.io.error(NodePhase::Process, e))?;
            state.tb = state.start.timebase();
        } else {
            state.tb = frame_tb;
            state.start = Ts::invalid();
        }
        state.configured = true;
        Ok(())
    }

    fn ensure_engine(&self, state: &mut AudioState, frame: &AVFrame) -> Result<(), NodeError> {
        let in_rate = frame.sample_rate;
        let in_fmt = frame.format;
        if let Some(engine) = &state.engine
            && engine.in_rate == in_rate
            && engine.in_fmt == in_fmt
        {
            return Ok(());
        }
        let out_rate = self.out_rate.unwrap_or(in_rate);
        let out_fmt = match &self.out_fmt {
            Some(name) => sample_fmt(name).map_err(|e| self.io.error(NodePhase::Process, e))?,
            None => in_fmt,
        };
        let out_layout = match &self.out_layout {
            Some(name) => {
                channel_layout(name).map_err(|e| self.io.error(NodePhase::Process, e))?
            }
            None => copy_layout(&frame.ch_layout),
        };
        let in_layout = copy_layout(&frame.ch_layout);
        let engine = Engine::open(in_rate, in_fmt, in_layout, out_rate, out_fmt, out_layout)
            .map_err(|e| self.io.error(NodePhase::Process, e))?;
        if self.group.is_none() && self.compensation != 0.0 {
            engine.set_async(self.compensation);
        }
        state.engine = Some(engine);
        Ok(())
    }

    fn duration_ticks(&self, state: &AudioState, samples: i32) -> i64 {
        let rate = state
            .engine
            .as_ref()
            .map(|engine| engine.out_rate)
            .unwrap_or(1)
            .max(1);
        match self.frame_duration_s {
            Some(seconds) => correction_cfg::seconds_to_ticks(seconds, state.tb).max(1),
            None => samples_to_ticks(samples as i64, rate, state.tb).max(1),
        }
    }
}

impl Engine {
    fn open(
        in_rate: i32,
        in_fmt: i32,
        mut in_layout: ffi::AVChannelLayout,
        out_rate: i32,
        out_fmt: i32,
        mut out_layout: ffi::AVChannelLayout,
    ) -> Result<Self, String> {
        if in_rate <= 0 || out_rate <= 0 {
            return Err("resample sample rate must be positive".into());
        }
        let mut swr = ptr::null_mut();
        let ret = unsafe {
            ffi::swr_alloc_set_opts2(
                &mut swr,
                &out_layout,
                out_fmt,
                out_rate,
                &in_layout,
                in_fmt,
                in_rate,
                0,
                ptr::null_mut(),
            )
        };
        if ret < 0 || swr.is_null() {
            unsafe {
                ffi::av_channel_layout_uninit(&mut in_layout);
                ffi::av_channel_layout_uninit(&mut out_layout);
            }
            return Err(format!("swr_alloc_set_opts2 failed ({ret})"));
        }
        let init = unsafe { ffi::swr_init(swr) };
        if init < 0 {
            unsafe {
                ffi::swr_free(&mut swr);
                ffi::av_channel_layout_uninit(&mut in_layout);
                ffi::av_channel_layout_uninit(&mut out_layout);
            }
            return Err(format!("swr_init failed ({init})"));
        }
        unsafe { ffi::av_channel_layout_uninit(&mut in_layout) };
        let channels = out_layout.nb_channels.max(1);
        Ok(Self {
            swr,
            in_rate,
            out_rate,
            in_fmt,
            out_fmt,
            channels,
            out_layout,
        })
    }

    fn set_async(&self, compensation: f64) {
        let key = CString::new("async").unwrap();
        unsafe {
            ffi::av_opt_set_double(self.swr as *mut c_void, key.as_ptr(), compensation, 0);
        }
    }

    fn compensate(&self, sample_delta: i32, distance: i32) -> Result<(), NodeError> {
        if sample_delta == 0 {
            return Ok(());
        }
        let ret = unsafe { ffi::swr_set_compensation(self.swr, sample_delta, distance.max(1)) };
        if ret < 0 {
            log::warn!("swr_set_compensation({sample_delta}, {distance}) failed ({ret})");
        }
        Ok(())
    }

    fn inject_silence(&self, count: i32) -> Result<(), NodeError> {
        let ret = unsafe { ffi::swr_inject_silence(self.swr, count) };
        if ret < 0 {
            return Err(NodeError::new(
                "resample_audio",
                NodePhase::Process,
                format!("swr_inject_silence failed ({ret})"),
            ));
        }
        Ok(())
    }

    fn drop_output(&self, count: i32) -> Result<(), NodeError> {
        let ret = unsafe { ffi::swr_drop_output(self.swr, count) };
        if ret < 0 {
            return Err(NodeError::new(
                "resample_audio",
                NodePhase::Process,
                format!("swr_drop_output failed ({ret})"),
            ));
        }
        Ok(())
    }

    fn reset_delay(&mut self) -> Result<(), NodeError> {
        unsafe { ffi::swr_close(self.swr) };
        let ret = unsafe { ffi::swr_init(self.swr) };
        if ret < 0 {
            return Err(NodeError::new(
                "resample_audio",
                NodePhase::Process,
                format!("swr_init after discontinuity failed ({ret})"),
            ));
        }
        Ok(())
    }

    fn convert(&mut self, input: Option<&AVFrame>, extra: i32) -> Result<AVFrame, NodeError> {
        let in_samples = input.map(|frame| frame.nb_samples).unwrap_or(0).max(0);
        let delay = unsafe { ffi::swr_get_delay(self.swr, self.out_rate as i64) };
        let out_count = (unsafe {
            ffi::av_rescale_rnd(
                delay + in_samples as i64,
                self.out_rate as i64,
                self.in_rate as i64,
                3,
            )
        }) as i32
            + extra
            + 32;
        let out_count = out_count.max(1);
        let mut frame = self.blank(out_count)?;
        let in_ptr = input
            .map(|frame| unsafe { (*frame.as_ptr()).extended_data as *mut *const u8 })
            .unwrap_or(ptr::null_mut());
        let got = unsafe {
            ffi::swr_convert(
                self.swr,
                (*frame.as_mut_ptr()).extended_data,
                out_count,
                in_ptr,
                in_samples,
            )
        };
        if got < 0 {
            return Err(NodeError::new(
                "resample_audio",
                NodePhase::Process,
                format!("swr_convert failed ({got})"),
            ));
        }
        frame.set_nb_samples(got.max(0));
        Ok(frame)
    }

    fn drain(&mut self) -> Result<Vec<AVFrame>, NodeError> {
        let mut out = Vec::new();
        loop {
            let frame = self.convert(None, 0)?;
            if frame.nb_samples <= 0 {
                break;
            }
            out.push(frame);
            if out.len() > 8 {
                break;
            }
        }
        Ok(out)
    }

    fn silence(&self, samples: i32) -> Result<AVFrame, NodeError> {
        let frame = self.blank(samples.max(1))?;
        unsafe {
            ffi::av_samples_set_silence(
                (*frame.as_ptr()).extended_data,
                0,
                frame.nb_samples,
                self.channels,
                self.out_fmt,
            );
        }
        Ok(frame)
    }

    fn blank(&self, samples: i32) -> Result<AVFrame, NodeError> {
        let mut frame = AVFrame::new();
        frame.set_nb_samples(samples.max(1));
        frame.set_format(self.out_fmt);
        frame.set_sample_rate(self.out_rate);
        unsafe {
            let ret = ffi::av_channel_layout_copy(
                &mut (*frame.as_mut_ptr()).ch_layout,
                &self.out_layout,
            );
            if ret < 0 {
                return Err(NodeError::new(
                    "resample_audio",
                    NodePhase::Process,
                    format!("channel layout copy failed ({ret})"),
                ));
            }
        }
        frame.alloc_buffer().map_err(|e| {
            NodeError::new(
                "resample_audio",
                NodePhase::Process,
                format!("audio buffer: {e}"),
            )
        })?;
        Ok(frame)
    }
}

fn frame_ts(frame: &AVFrame) -> Ts {
    Ts::new(
        frame.pts,
        AvpRational {
            num: frame.time_base.num,
            den: frame.time_base.den,
        },
    )
}

fn set_frame_ts(frame: &mut AVFrame, ts: Ts) {
    frame.set_pts(ts.ticks());
    frame.set_time_base(ffi::AVRational {
        num: ts.timebase().num,
        den: ts.timebase().den,
    });
}

fn ticks_to_samples(ticks: i64, rate: i32, tb: AvpRational) -> i32 {
    if tb.den == 0 || rate <= 0 {
        return 0;
    }
    (ticks as i128 * rate as i128 * tb.num as i128 / tb.den as i128) as i32
}

fn samples_to_ticks(samples: i64, rate: i32, tb: AvpRational) -> i64 {
    if rate <= 0 || tb.num <= 0 {
        return 0;
    }
    ((samples as i128 * tb.den as i128) / (rate as i128 * tb.num as i128)) as i64
}

fn sample_fmt(name: &str) -> Result<i32, String> {
    let c = CString::new(name).map_err(|e| e.to_string())?;
    let fmt = unsafe { ffi::av_get_sample_fmt(c.as_ptr()) };
    if fmt == ffi::AV_SAMPLE_FMT_NONE {
        return Err(format!("unknown sample_fmt `{name}`"));
    }
    Ok(fmt)
}

fn channel_layout(name: &str) -> Result<ffi::AVChannelLayout, String> {
    let mut layout = unsafe { std::mem::zeroed() };
    let c = CString::new(name).map_err(|e| e.to_string())?;
    let ret = unsafe { ffi::av_channel_layout_from_string(&mut layout, c.as_ptr()) };
    if ret < 0 {
        return Err(format!("unknown channel_layout `{name}`"));
    }
    Ok(layout)
}

fn copy_layout(src: &ffi::AVChannelLayout) -> ffi::AVChannelLayout {
    let mut layout = unsafe { std::mem::zeroed() };
    unsafe {
        ffi::av_channel_layout_copy(&mut layout, src);
    }
    layout
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::{Value, json};

    use avplumber_f7k::Instance;
    use avplumber_f7k::factory::{BuildCtx, NodeSpec};
    use avplumber_f7k::graph::buffered_edge::BufferedEdge;
    use avplumber_f7k::graph::edge::{Edge, EdgeItem};
    use avplumber_f7k::graph::grain::Grain;
    use avplumber_f7k::graph::media::AvpRational;
    use avplumber_f7k::graph::node::{Node, Processed};
    use rsmpeg::avutil::AVFrame;
    use rusty_ffmpeg::ffi;

    use super::ResampleSpec;

    const MS: AvpRational = AvpRational { num: 1, den: 1000 };

    fn audio(pts: i64, samples: i32) -> Grain {
        let mut frame = AVFrame::new();
        frame.set_nb_samples(samples);
        frame.set_sample_rate(48_000);
        frame.set_format(ffi::AV_SAMPLE_FMT_S16);
        unsafe {
            ffi::av_channel_layout_default(&mut (*frame.as_mut_ptr()).ch_layout, 1);
        }
        frame.alloc_buffer().unwrap();
        frame.set_pts(pts);
        frame.set_time_base(ffi::AVRational { num: 1, den: 1000 });
        Grain::Audio(frame)
    }

    struct AudioHarness {
        node: avplumber_f7k::Blocking<super::ResampleAudio>,
        input: Arc<dyn Edge>,
        out: Arc<dyn Edge>,
    }

    impl AudioHarness {
        fn new(params: Value, sync: Option<&str>) -> Self {
            let spec: ResampleSpec = serde_json::from_value(params.clone()).unwrap();
            let instance = Instance::new();
            let node = spec
                .build(
                    "a",
                    &BuildCtx {
                        instance: &instance,
                        name: "a",
                        params: &params,
                        sync_group: sync,
                    },
                )
                .unwrap();
            let input: Arc<dyn Edge> = Arc::new(BufferedEdge::new(8));
            let out: Arc<dyn Edge> = Arc::new(BufferedEdge::new(8));
            node.bind_source("in", input.clone());
            node.bind_sink("out", out.clone());
            node.start();
            Self { node, input, out }
        }

        fn push(&self, grain: Grain) {
            assert!(self.input.offer(grain).is_ok());
        }

        fn step(&self) -> Grain {
            assert!(matches!(self.node.process().unwrap(), Processed::Again));
            match self.out.try_take() {
                Some(EdgeItem::Buffer(grain)) => grain,
                other => panic!("expected audio, got {other:?}"),
            }
        }
    }

    fn pts_of(grain: &Grain) -> i64 {
        grain.ts().ticks()
    }

    #[test]
    fn identity_keeps_the_sample_count() {
        let node = AudioHarness::new(
            json!({"sample_rate": 48000, "sample_fmt": "s16", "channel_layout": "mono"}),
            None,
        );
        node.push(audio(0, 960));
        let out = node.step();
        let Grain::Audio(frame) = &out else { panic!("audio") };
        assert_eq!(frame.nb_samples, 960);
        assert_eq!(pts_of(&out), 0);
    }

    #[test]
    fn standalone_discontinuity_resets_the_private_cursor() {
        let node = AudioHarness::new(json!({"frame_duration": 0.02}), None);
        node.push(audio(0, 960));
        assert_eq!(pts_of(&node.step()), 0);
        node.push(audio(2_000, 960));
        assert_eq!(pts_of(&node.step()), 2_000);
    }

    #[test]
    fn grouped_discontinuity_stays_on_the_cursor() {
        let node = AudioHarness::new(
            json!({
                "correction_group": "av",
                "forward_start_shift": true,
                "start_ts": 0,
                "frame_duration": 0.02,
                "rebase_threshold": 0.5,
                "max_slew": 0.1,
                "timeout": 30
            }),
            None,
        );
        node.push(audio(0, 960));
        assert_eq!(pts_of(&node.step()), 0);
        node.push(audio(200, 960));
        let out = node.step();
        assert_eq!(pts_of(&out), 20);
        let Grain::Audio(frame) = &out else { panic!("audio") };
        assert_ne!(
            frame.nb_samples, 960,
            "swr compensation did not change the packet length"
        );
    }

    #[test]
    fn grouped_stall_emits_silence_without_moving_the_shift() {
        let node = AudioHarness::new(
            json!({
                "correction_group": "av",
                "forward_start_shift": true,
                "start_ts": 0,
                "frame_duration": 0.02,
                "timeout": 0
            }),
            Some("live"),
        );
        node.push(audio(0, 960));
        node.step();
        let shift = node.node.0.group.as_ref().unwrap().group_shift().unwrap().ticks();
        node.node.0.clock.as_ref().unwrap().reset(5_000, MS);
        let out = node.step();
        assert_eq!(pts_of(&out), 20);
        let Grain::Audio(frame) = &out else { panic!("audio") };
        assert!(frame.nb_samples > 0);
        assert_eq!(
            node.node.0.group.as_ref().unwrap().group_shift().unwrap().ticks(),
            shift
        );
    }
}
