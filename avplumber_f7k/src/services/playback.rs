//! Playback control as a service: for one group of nodes, addressed by name,
//! the clock, the seek index and history, target resolution, read planning
//! for the source, and the position the pacing stage last released. Design:
//! `doc/specs/rust-refactor/rust_refactor_playback.md`.
//!
//! Nothing here touches media or runs on an executor. A source asks
//! [`Playback::plan_read`] before each read and [`Playback::plan_tail`] at the
//! end of its container; a pacing node calls [`Playback::report_release`] and
//! [`Playback::report_flush_stop`]; the control layer calls the verbs. The
//! service owns every decision, the nodes own the `AVFormatContext` and the
//! release timing.

use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};

use crate::graph::media::Ts;
use crate::graph::timebase::MILLISECONDS;
use crate::services::clock::{ClockService, SyncGroup};

/// [`Playback::released`] before any frame was released.
pub const NO_POSITION: i64 = i64::MIN;

// ------------------------------------------------------------- SeekIndex

/// One indexed frame: the recording's own timestamp and where its packet
/// starts. What `output` writes, native-endian, sixteen bytes per record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IndexEntry {
    pub timestamp_ms: i64,
    pub byte_offset: u64,
}

/// One row of the `+history` file: from media time `changed_at` on, the
/// offsets between the recording's timeline and the input, wallclock and
/// output timelines. Only `wallclock_offset` is read here (media minus UTC).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HistoryEntry {
    pub changed_at: i64,
    pub input_offset: i64,
    pub wallclock_offset: i64,
    pub output_offset: i64,
}

/// The seek table and history of one recording. Timestamps are the
/// recording's own, in milliseconds; "frame" means an index into the table.
#[derive(Debug)]
pub struct SeekIndex {
    entries: Vec<IndexEntry>,
    history: Vec<HistoryEntry>,
}

impl SeekIndex {
    pub fn new(entries: Vec<IndexEntry>, history: Vec<HistoryEntry>) -> Result<Self, String> {
        if entries.is_empty() {
            return Err("the seek table is empty".into());
        }
        for pair in entries.windows(2) {
            if pair[1].timestamp_ms < pair[0].timestamp_ms {
                return Err("seek table timestamps decrease".into());
            }
            if pair[1].byte_offset < pair[0].byte_offset {
                return Err("seek table byte offsets decrease".into());
            }
        }
        if let Some(first) = history.first()
            && first.changed_at != 0
        {
            return Err("the timestamp history must map frame zero".into());
        }
        for pair in history.windows(2) {
            if pair[1].changed_at < pair[0].changed_at {
                return Err("timestamp history changed_at values decrease".into());
            }
        }
        Ok(Self { entries, history })
    }

    /// Reads `<recording>+seek` and, when given and present, `<recording>+history`.
    pub fn load(seek_table: &Path, history: Option<&Path>) -> Result<Self, String> {
        let table = std::fs::read(seek_table)
            .map_err(|e| format!("cannot read seek table {}: {e}", seek_table.display()))?;
        let entries = Self::parse_seek_table(&table)
            .map_err(|e| format!("seek table {}: {e}", seek_table.display()))?;
        let history = match history {
            Some(path) if path.exists() => {
                let bytes = std::fs::read(path)
                    .map_err(|e| format!("cannot read history {}: {e}", path.display()))?;
                Self::parse_history(&bytes)
                    .map_err(|e| format!("history {}: {e}", path.display()))?
            }
            _ => Vec::new(),
        };
        Self::new(entries, history)
    }

    pub fn parse_seek_table(bytes: &[u8]) -> Result<Vec<IndexEntry>, String> {
        if bytes.is_empty() {
            return Err("is empty".into());
        }
        if bytes.len() % 16 != 0 {
            return Err("size must be a multiple of 16 bytes".into());
        }
        Ok(bytes
            .chunks_exact(16)
            .map(|record| IndexEntry {
                timestamp_ms: i64::from_ne_bytes(record[..8].try_into().unwrap()),
                byte_offset: u64::from_ne_bytes(record[8..].try_into().unwrap()),
            })
            .collect())
    }

    pub fn parse_history(bytes: &[u8]) -> Result<Vec<HistoryEntry>, String> {
        if bytes.len() % 32 != 0 {
            return Err("size must be a multiple of 32 bytes".into());
        }
        let field = |record: &[u8], i: usize| {
            i64::from_ne_bytes(record[i * 8..(i + 1) * 8].try_into().unwrap())
        };
        Ok(bytes
            .chunks_exact(32)
            .map(|record| HistoryEntry {
                changed_at: field(record, 0),
                input_offset: field(record, 1),
                wallclock_offset: field(record, 2),
                output_offset: field(record, 3),
            })
            .collect())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn entry(&self, frame: usize) -> IndexEntry {
        self.entries[frame.min(self.entries.len() - 1)]
    }

    pub fn first_ms(&self) -> i64 {
        self.entries[0].timestamp_ms
    }

    pub fn last_ms(&self) -> i64 {
        self.entries[self.entries.len() - 1].timestamp_ms
    }

    pub fn duration_ms(&self) -> i64 {
        self.last_ms() - self.first_ms()
    }

    pub fn has_history(&self) -> bool {
        !self.history.is_empty()
    }

    /// The frame nearest to `media_ms`; a tie selects the later one, which is
    /// the C++ contract the demo's tests encode.
    pub fn nearest(&self, media_ms: i64) -> usize {
        let at_or_after = self.entries.partition_point(|e| e.timestamp_ms < media_ms);
        if at_or_after == self.entries.len() {
            return self.entries.len() - 1;
        }
        if at_or_after > 0 {
            let before = media_ms - self.entries[at_or_after - 1].timestamp_ms;
            let after = self.entries[at_or_after].timestamp_ms - media_ms;
            if before < after {
                return at_or_after - 1;
            }
        }
        at_or_after
    }

    /// The frame being shown at `media_ms`: the last one at or before it.
    pub fn frame_of(&self, media_ms: i64) -> usize {
        self.entries
            .partition_point(|e| e.timestamp_ms <= media_ms)
            .saturating_sub(1)
    }

    /// The frame whose packet starts at or before byte `pos`, `None` before the
    /// first one.
    pub fn frame_at_bytes(&self, pos: u64) -> Option<usize> {
        let count = self.entries.partition_point(|e| e.byte_offset <= pos);
        count.checked_sub(1)
    }

    fn history_at_media(&self, media_ms: i64) -> Option<&HistoryEntry> {
        if self.history.is_empty() {
            return None;
        }
        let index = self
            .history
            .partition_point(|h| h.changed_at <= media_ms)
            .saturating_sub(1);
        Some(&self.history[index])
    }

    /// UTC milliseconds of a media timestamp, `None` without a history.
    pub fn media_to_wallclock_ms(&self, media_ms: i64) -> Option<i64> {
        self.history_at_media(media_ms)
            .map(|h| media_ms - h.wallclock_offset)
    }

    /// Media timestamp of a UTC millisecond, `None` without a history.
    pub fn wallclock_to_media_ms(&self, wallclock_ms: i64) -> Option<i64> {
        if self.history.is_empty() {
            return None;
        }
        // Each row starts, in wallclock terms, at `changed_at - wallclock_offset`.
        let mut starts: Vec<(i64, i64)> = self
            .history
            .iter()
            .map(|h| (h.changed_at - h.wallclock_offset, h.wallclock_offset))
            .collect();
        starts.sort_by_key(|(start, _)| *start);
        let index = starts
            .partition_point(|(start, _)| *start <= wallclock_ms)
            .saturating_sub(1);
        Some(wallclock_ms + starts[index].1)
    }

    /// The integer frame rate the table implies, `None` when the cadence is
    /// not consistent enough to call it one (the demo's inference rule).
    pub fn fps(&self) -> Option<u32> {
        if self.entries.len() < 2 {
            return None;
        }
        let span = self.duration_ms();
        if span <= 0 {
            return None;
        }
        let fps = ((self.entries.len() - 1) as f64 * 1000.0 / span as f64).round();
        if !(1.0..=240.0).contains(&fps) {
            return None;
        }
        let period = 1000.0 / fps;
        let tolerance = (period * 0.08).max(2.0);
        let start = self.first_ms() as f64;
        let consistent =
            self.entries.iter().enumerate().all(|(i, e)| {
                (e.timestamp_ms as f64 - (start + i as f64 * period)).abs() <= tolerance
            });
        consistent.then_some(fps as u32)
    }
}

// ---------------------------------------------------------------- Target

/// Where to seek, as the control protocol spells it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// An absolute timestamp of the recording, in milliseconds.
    MediaMs(i64),
    /// Relative to the position being shown.
    RelativeMs(i64),
    /// A UTC timestamp in milliseconds, mapped through the history.
    Wallclock(i64),
    Frame(i64),
    RelativeFrames(i64),
    /// The newest indexed frame minus the live delay.
    Live,
    End,
}

impl Target {
    /// C++ `StreamTarget::from_string`: `12000`, `+500`, `-500`, `01:02:03`,
    /// `01:00.150`, `+0:10`, `2026-08-10T12:00:00.000`, and `live`, `end`.
    pub fn parse(text: &str) -> Result<Target, String> {
        let text = text.trim();
        if text.is_empty() {
            return Err("empty seek target".into());
        }
        match text {
            "live" => return Ok(Target::Live),
            "end" => return Ok(Target::End),
            _ => {}
        }
        if text.contains('T') {
            return parse_wallclock(text).map(Target::Wallclock);
        }
        let (relative, sign, body) = match text.as_bytes()[0] {
            b'+' => (true, 1, &text[1..]),
            b'-' => (true, -1, &text[1..]),
            _ => (false, 1, text),
        };
        let ms = if body.contains(':') {
            parse_clock_ms(body)?
        } else {
            body.parse::<i64>()
                .map_err(|_| format!("`{text}` is not a seek target"))?
        };
        Ok(if relative {
            Target::RelativeMs(sign * ms)
        } else {
            Target::MediaMs(ms)
        })
    }
}

/// `hh:mm:ss[.fff]` or `mm:ss[.fff]` to milliseconds.
fn parse_clock_ms(text: &str) -> Result<i64, String> {
    let invalid = || format!("`{text}` is not a clock time");
    let (whole, frac) = match text.split_once('.') {
        Some((whole, frac)) => (whole, Some(frac)),
        None => (text, None),
    };
    let parts: Vec<i64> = whole
        .split(':')
        .map(|p| p.parse::<i64>().map_err(|_| invalid()))
        .collect::<Result<_, _>>()?;
    let seconds = match parts.as_slice() {
        [m, s] => m * 60 + s,
        [h, m, s] => (h * 60 + m) * 60 + s,
        _ => return Err(invalid()),
    };
    Ok(seconds * 1000 + frac.map(parse_millis).transpose()?.unwrap_or(0))
}

/// Digits after the decimal point, as milliseconds however many there are.
fn parse_millis(frac: &str) -> Result<i64, String> {
    let digits: String = frac.chars().take_while(|c| c.is_ascii_digit()).collect();
    if digits.is_empty() {
        return Ok(0);
    }
    let value: i64 = digits
        .parse()
        .map_err(|_| format!("`{frac}` is not a fraction"))?;
    let scale = 10i64.pow(digits.len() as u32);
    Ok(value * 1000 / scale)
}

/// `YYYY-MM-DDTHH:MM:SS[.fff][Z|±HH:MM]` to UTC milliseconds. No offset means
/// UTC, as C++ `timegm` did.
fn parse_wallclock(text: &str) -> Result<i64, String> {
    let invalid = || format!("`{text}` is not an ISO 8601 timestamp");
    let (date, rest) = text.split_once('T').ok_or_else(invalid)?;
    let mut date_parts = date.split('-');
    let year: i64 = date_parts
        .next()
        .and_then(|p| p.parse().ok())
        .ok_or_else(invalid)?;
    let month: i64 = date_parts
        .next()
        .and_then(|p| p.parse().ok())
        .ok_or_else(invalid)?;
    let day: i64 = date_parts
        .next()
        .and_then(|p| p.parse().ok())
        .ok_or_else(invalid)?;
    if date_parts.next().is_some() || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return Err(invalid());
    }
    // Split the zone suffix off the time.
    let (time, offset_minutes) = if let Some(time) = rest.strip_suffix('Z') {
        (time, 0)
    } else if let Some(pos) = rest.rfind(['+', '-']).filter(|&pos| pos > 0) {
        let (time, zone) = rest.split_at(pos);
        let sign = if zone.starts_with('-') { -1 } else { 1 };
        let (zh, zm) = zone[1..].split_once(':').unwrap_or((&zone[1..], "0"));
        let hours: i64 = zh.parse().map_err(|_| invalid())?;
        let minutes: i64 = zm.parse().map_err(|_| invalid())?;
        (time, sign * (hours * 60 + minutes))
    } else {
        (rest, 0)
    };
    let (clock, frac) = match time.split_once('.') {
        Some((clock, frac)) => (clock, Some(frac)),
        None => (time, None),
    };
    let parts: Vec<i64> = clock
        .split(':')
        .map(|p| p.parse::<i64>().map_err(|_| invalid()))
        .collect::<Result<_, _>>()?;
    let [hour, minute, second] = parts.as_slice() else {
        return Err(invalid());
    };
    let days = days_from_civil(year, month, day);
    let seconds = days * 86_400 + hour * 3600 + minute * 60 + second - offset_minutes * 60;
    Ok(seconds * 1000 + frac.map(parse_millis).transpose()?.unwrap_or(0))
}

/// Days since 1970-01-01 of a proleptic Gregorian date (Howard Hinnant's
/// algorithm).
fn days_from_civil(year: i64, month: i64, day: i64) -> i64 {
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400;
    let mp = (month + 9) % 12;
    let doy = (153 * mp + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146_097 + doe - 719_468
}

// -------------------------------------------------------------- Playback

/// Handle of a source bound to a [`Playback`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceId(usize);

/// Where a source repositions its container.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SeekTo {
    /// An indexed frame's packet: exact, so no cutoff is needed.
    Bytes(u64),
    /// A timestamp the demuxer resolves to a keyframe at or before it; the
    /// decoder drops what lies before `resume_at`.
    Time {
        media_ms: i64,
        resume_at: Option<Ts>,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Reposition {
    pub to: SeekTo,
    /// Wrap the seek in `FlushStart`/`FlushStop`: a user seek, a direction
    /// change or a loop. A reverse step or a skip stride is not one.
    pub discontinuity: bool,
}

/// What a source does before its next read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReadPlan {
    Continue,
    Reposition(Reposition),
    /// Nothing to read in this direction: wait for a command.
    Idle,
}

struct BoundSource {
    wake: Arc<dyn Fn() + Send + Sync>,
    mailbox: Mutex<Option<Reposition>>,
}

#[derive(Default)]
struct Scheduled {
    pause_at_ms: Option<i64>,
    seeks: Vec<(i64, Target)>,
}

pub struct Playback {
    name: String,
    clock: Arc<dyn SyncGroup>,
    index: Mutex<Option<Arc<SeekIndex>>>,
    sources: Mutex<Vec<BoundSource>>,
    loop_: AtomicBool,
    live_delay_ms: AtomicI64,
    /// Media time of the last released frame, [`NO_POSITION`] before the first.
    released: AtomicI64,
    /// Bumped on every release: a client tells a fresh frame from a repeated
    /// status by it.
    serial: AtomicU64,
    /// The media time a seek asked for, until a frame is released after the
    /// flush that seek caused. Relative targets resolve against it, so rapid
    /// seeks compose instead of racing the pipeline.
    pending: Mutex<Option<i64>>,
    /// A `FlushStop` has passed the pacing stage since the last seek.
    flushed: AtomicBool,
    /// The source has nothing left to read in its direction. Frames may still
    /// be in flight, so this alone is not `at_end`.
    tail_reached: AtomicBool,
    /// The viewer is on the last frame in the current direction, with nothing
    /// more to come. Where a loop fires from, and what `status` reports.
    at_end: AtomicBool,
    scheduled: Mutex<Scheduled>,
}

impl Playback {
    fn new(name: &str, clock: Arc<dyn SyncGroup>) -> Self {
        Self {
            name: name.into(),
            clock,
            index: Mutex::new(None),
            sources: Mutex::new(Vec::new()),
            loop_: AtomicBool::new(false),
            live_delay_ms: AtomicI64::new(1000),
            released: AtomicI64::new(NO_POSITION),
            serial: AtomicU64::new(0),
            pending: Mutex::new(None),
            flushed: AtomicBool::new(false),
            tail_reached: AtomicBool::new(false),
            at_end: AtomicBool::new(false),
            scheduled: Mutex::new(Scheduled::default()),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    /// The group's master clock, the same object `BuildCtx::clock` hands out
    /// for this name.
    pub fn clock(&self) -> &Arc<dyn SyncGroup> {
        &self.clock
    }

    /// Registers a source. `wake` must reach the source wherever it may be
    /// parked; `index` is the recording's, if it has one.
    pub fn bind_source(
        &self,
        wake: Arc<dyn Fn() + Send + Sync>,
        index: Option<Arc<SeekIndex>>,
        loop_: bool,
        live_delay_ms: i64,
    ) -> SourceId {
        if let Some(index) = index {
            *self.index.lock().unwrap() = Some(index);
        }
        self.loop_.store(loop_, Ordering::Release);
        self.live_delay_ms.store(live_delay_ms, Ordering::Release);
        let mut sources = self.sources.lock().unwrap();
        sources.push(BoundSource {
            wake,
            mailbox: Mutex::new(None),
        });
        SourceId(sources.len() - 1)
    }

    pub fn index(&self) -> Option<Arc<SeekIndex>> {
        self.index.lock().unwrap().clone()
    }

    pub fn set_loop(&self, loop_: bool) {
        self.loop_.store(loop_, Ordering::Release);
    }

    pub fn is_loop(&self) -> bool {
        self.loop_.load(Ordering::Acquire)
    }

    /// The media time relative targets resolve against: the seek in flight if
    /// there is one, else the last released frame.
    fn base_ms(&self) -> Option<i64> {
        if let Some(pending) = *self.pending.lock().unwrap() {
            return Some(pending);
        }
        match self.released.load(Ordering::Acquire) {
            NO_POSITION => None,
            ms => Some(ms),
        }
    }

    /// The target as an absolute media time and where to reposition for it.
    fn resolve(&self, target: Target) -> Result<(i64, SeekTo), String> {
        let index = self.index();
        let need_index =
            |what: &str| format!("{what} needs a seek table, and {} has none", self.name);
        let need_base = || "the current position is not known yet".to_string();
        let indexed = |frame: usize| {
            let entry = index.as_ref().unwrap().entry(frame);
            (entry.timestamp_ms, SeekTo::Bytes(entry.byte_offset))
        };
        let by_time = |media_ms: i64| match &index {
            Some(index) => indexed(index.nearest(media_ms)),
            None => (
                media_ms,
                SeekTo::Time {
                    media_ms,
                    resume_at: Some(Ts {
                        val: media_ms,
                        tb: MILLISECONDS,
                    }),
                },
            ),
        };
        Ok(match target {
            Target::MediaMs(ms) => by_time(ms),
            Target::RelativeMs(delta) => by_time(self.base_ms().ok_or_else(need_base)? + delta),
            Target::Wallclock(utc_ms) => {
                let index = index
                    .as_ref()
                    .ok_or_else(|| need_index("a wallclock seek"))?;
                let media_ms = index
                    .wallclock_to_media_ms(utc_ms)
                    .ok_or_else(|| format!("{} has no timestamp history", self.name))?;
                by_time(media_ms)
            }
            Target::Frame(frame) => {
                let index = index.as_ref().ok_or_else(|| need_index("a frame seek"))?;
                indexed(frame.clamp(0, index.len() as i64 - 1) as usize)
            }
            Target::RelativeFrames(delta) => {
                let index = index.as_ref().ok_or_else(|| need_index("a frame seek"))?;
                let base = index.frame_of(self.base_ms().ok_or_else(need_base)?) as i64;
                indexed((base + delta).clamp(0, index.len() as i64 - 1) as usize)
            }
            Target::Live => {
                let index = index.as_ref().ok_or_else(|| need_index("`live`"))?;
                let delay = self.live_delay_ms.load(Ordering::Acquire);
                let ms = if index.last_ms() - index.first_ms() < delay {
                    index.first_ms()
                } else {
                    index.last_ms() - delay
                };
                indexed(index.nearest(ms))
            }
            Target::End => {
                let index = index.as_ref().ok_or_else(|| need_index("`end`"))?;
                indexed(index.len() - 1)
            }
        })
    }

    /// Resolves `target`, resets the clock to it, and asks every bound source
    /// to reposition. Returns the media time seeked to.
    pub fn seek(&self, target: Target) -> Result<i64, String> {
        if self.sources.lock().unwrap().is_empty() {
            return Err(format!("no seekable source is bound to {}", self.name));
        }
        let (media_ms, to) = self.resolve(target)?;
        self.request(media_ms, to);
        Ok(media_ms)
    }

    fn request(&self, media_ms: i64, to: SeekTo) {
        self.clock.reset(media_ms, MILLISECONDS);
        *self.pending.lock().unwrap() = Some(media_ms);
        self.flushed.store(false, Ordering::Release);
        self.tail_reached.store(false, Ordering::Release);
        self.at_end.store(false, Ordering::Release);
        let reposition = Reposition {
            to,
            discontinuity: true,
        };
        for source in self.sources.lock().unwrap().iter() {
            *source.mailbox.lock().unwrap() = Some(reposition);
            (source.wake)();
        }
    }

    pub fn rate(&self) -> f64 {
        self.clock.snapshot().rate
    }

    /// Negative plays backwards, zero pauses.
    ///
    /// A sign change is a discontinuity: the pipe is flushed and reading
    /// restarts from the frame on screen. So is a change of the read stride
    /// (2x reads every other frame): the frames already in flight were read
    /// at the old stride, and letting them out at the new pace would show a
    /// burst of the old motion. The re-seek costs one repeated frame and makes
    /// the transition frame-exact, which is what the C++ demo's drain gate
    /// bought at the price of a stall.
    pub fn set_rate(&self, rate: f64) -> Result<(), String> {
        if !rate.is_finite() {
            return Err(format!("rate {rate} is not a number"));
        }
        if rate == 0.0 {
            self.pause();
            return Ok(());
        }
        let previous = self.clock.snapshot().rate;
        self.clock.set_rate(rate);
        let stride = |r: f64| r.abs().round().max(1.0) as i64;
        if previous.is_sign_negative() != rate.is_sign_negative()
            || stride(previous) != stride(rate)
        {
            let restart = self
                .seek(Target::RelativeFrames(0))
                .or_else(|_| self.seek(Target::RelativeMs(0)));
            if let Err(error) = restart {
                log::debug!("{}: rate changed before any position: {error}", self.name);
            }
        }
        Ok(())
    }

    pub fn pause(&self) {
        self.clock.set_paused(true);
    }

    /// Resuming on the last frame with `loop` set starts over: a paused viewer
    /// at the end sees frame zero next, not a stuck picture.
    pub fn resume(&self) {
        self.clock.set_paused(false);
        if self.is_at_end() {
            self.maybe_loop();
        }
    }

    /// With `loop`, and while playing, starts over from the frame at the other
    /// end. Called when the viewer reaches an end, never when the reader does:
    /// the reader runs ahead by the pipeline's depth, and looping on its say-so
    /// would flush the frames the viewer has not seen yet.
    fn maybe_loop(&self) {
        if !self.is_loop() || self.clock.snapshot().paused {
            return;
        }
        let reverse = self.clock.snapshot().rate < 0.0;
        let target = match (self.index(), reverse) {
            (Some(_), true) => Target::End,
            (Some(_), false) => Target::Frame(0),
            (None, _) => Target::MediaMs(0),
        };
        match self.resolve(target) {
            Ok((media_ms, to)) => self.request(media_ms, to),
            Err(error) => log::warn!("{}: cannot loop: {error}", self.name),
        }
    }

    pub fn is_paused(&self) -> bool {
        self.clock.snapshot().paused
    }

    /// Pause when playback reaches `target` (absolute forms only).
    pub fn pause_at(&self, target: Target) -> Result<(), String> {
        let (media_ms, _) = self.resolve(target)?;
        self.scheduled.lock().unwrap().pause_at_ms = Some(media_ms);
        Ok(())
    }

    /// Seek to `target` when playback reaches `when`; several queue in order.
    pub fn seek_at(&self, when: Target, target: Target) -> Result<(), String> {
        let (when_ms, _) = self.resolve(when)?;
        self.scheduled.lock().unwrap().seeks.push((when_ms, target));
        Ok(())
    }

    pub fn clear_scheduled(&self) {
        *self.scheduled.lock().unwrap() = Scheduled::default();
    }

    /// From the pacing stage: a frame with this media time went out. Fires the
    /// scheduled actions and clears a seek whose flush has already passed.
    pub fn report_release(&self, media_ms: i64) {
        self.released.store(media_ms, Ordering::Release);
        self.serial.fetch_add(1, Ordering::AcqRel);
        if self.flushed.load(Ordering::Acquire) {
            *self.pending.lock().unwrap() = None;
        }
        let reverse = self.clock.snapshot().rate < 0.0;
        if self.tail_reached.load(Ordering::Acquire)
            && let Some(index) = self.index()
        {
            let frame = index.frame_of(media_ms);
            let last = if reverse { 0 } else { index.len() - 1 };
            if frame == last {
                self.at_end.store(true, Ordering::Release);
                self.maybe_loop();
            }
        }
        let reached = |at: i64| {
            if reverse {
                media_ms <= at
            } else {
                media_ms >= at
            }
        };
        let (pause_now, seek_now) = {
            let mut scheduled = self.scheduled.lock().unwrap();
            let pause_now = scheduled.pause_at_ms.is_some_and(reached);
            if pause_now {
                scheduled.pause_at_ms = None;
            }
            let seek_now = match scheduled.seeks.first() {
                Some((when, target)) if reached(*when) => {
                    let target = *target;
                    scheduled.seeks.remove(0);
                    Some(target)
                }
                _ => None,
            };
            (pause_now, seek_now)
        };
        if pause_now {
            self.pause();
        }
        if let Some(target) = seek_now
            && let Err(error) = self.seek(target)
        {
            log::warn!("{}: scheduled seek failed: {error}", self.name);
        }
    }

    /// From the pacing stage: a `FlushStop` passed, so the next release is at
    /// the position the last seek asked for.
    pub fn report_flush_stop(&self) {
        self.flushed.store(true, Ordering::Release);
    }

    pub fn serial(&self) -> u64 {
        self.serial.load(Ordering::Acquire)
    }

    /// A seek has been requested and its `FlushStop` has not passed the pacing
    /// stage yet. Whatever the pacing stage holds or receives meanwhile is from
    /// before the discontinuity: it must not be shown, however a pause or a
    /// resume lands in that window.
    pub fn seek_in_flight(&self) -> bool {
        self.pending.lock().unwrap().is_some() && !self.flushed.load(Ordering::Acquire)
    }

    /// Media time of the last released frame, if any.
    pub fn released_ms(&self) -> Option<i64> {
        match self.released.load(Ordering::Acquire) {
            NO_POSITION => None,
            ms => Some(ms),
        }
    }

    pub fn is_at_end(&self) -> bool {
        self.at_end.load(Ordering::Acquire)
    }

    /// Whether a seek is waiting in `source`'s mailbox: a source parked on a
    /// full output checks this to abandon the stale packet it holds.
    pub fn has_request(&self, source: SourceId) -> bool {
        let sources = self.sources.lock().unwrap();
        sources
            .get(source.0)
            .is_some_and(|s| s.mailbox.lock().unwrap().is_some())
    }

    fn take_request(&self, source: SourceId) -> Option<Reposition> {
        let sources = self.sources.lock().unwrap();
        sources.get(source.0)?.mailbox.lock().unwrap().take()
    }

    /// Asked by the source on its own thread before each read. `last_pos` is
    /// the byte offset of the last indexed packet it delivered, `None` before
    /// the first.
    pub fn plan_read(&self, source: SourceId, last_pos: Option<u64>) -> ReadPlan {
        if let Some(reposition) = self.take_request(source) {
            return ReadPlan::Reposition(reposition);
        }
        let Some(index) = self.index() else {
            return ReadPlan::Continue;
        };
        let rate = self.clock.snapshot().rate;
        let Some(pos) = last_pos else {
            return ReadPlan::Continue;
        };
        let Some(frame) = index.frame_at_bytes(pos) else {
            return ReadPlan::Continue;
        };
        let stride = (rate.abs().round() as i64 - 1).max(0);
        if rate < 0.0 {
            let previous = frame as i64 - 1 - stride;
            if previous < 0 {
                return self.plan_tail(source, last_pos);
            }
            return ReadPlan::Reposition(Reposition {
                to: SeekTo::Bytes(index.entry(previous as usize).byte_offset),
                discontinuity: false,
            });
        }
        if stride > 0 {
            let next = frame as i64 + 1 + stride;
            if next < index.len() as i64 {
                return ReadPlan::Reposition(Reposition {
                    to: SeekTo::Bytes(index.entry(next as usize).byte_offset),
                    discontinuity: false,
                });
            }
        }
        ReadPlan::Continue
    }

    /// Asked by the source at the end of its container (or, reversing, at its
    /// start). The source idles: the viewer still has the pipeline's worth of
    /// frames to see, and the loop, if any, fires when it has seen them
    /// ([`Self::report_release`]) or when it resumes there ([`Self::resume`]).
    pub fn plan_tail(&self, source: SourceId, _last_pos: Option<u64>) -> ReadPlan {
        if let Some(reposition) = self.take_request(source) {
            return ReadPlan::Reposition(reposition);
        }
        self.tail_reached.store(true, Ordering::Release);
        if self.index().is_none() {
            // Nothing tells where the viewer is relative to the end.
            self.at_end.store(true, Ordering::Release);
        }
        ReadPlan::Idle
    }

    /// The `playback.status` document.
    pub fn status(&self) -> Value {
        let index = self.index();
        let snapshot = self.clock.snapshot();
        let released = self.released_ms();
        let start_ms = index.as_ref().map(|i| i.first_ms());
        let position_ms = match (released, start_ms) {
            (Some(ms), Some(start)) => {
                let duration = index.as_ref().unwrap().duration_ms();
                Some((ms - start).clamp(0, duration))
            }
            (Some(ms), None) => Some(ms),
            (None, _) => None,
        };
        json!({
            "position_ms": position_ms,
            "media_ms": released,
            "frame": released.and_then(|ms| index.as_ref().map(|i| i.frame_of(ms))),
            "wallclock_ms": released.and_then(|ms| index.as_ref().and_then(|i| i.media_to_wallclock_ms(ms))),
            "start_ms": start_ms,
            "duration_ms": index.as_ref().map(|i| i.duration_ms()),
            "frame_count": index.as_ref().map(|i| i.len()),
            "fps": index.as_ref().and_then(|i| i.fps()),
            "rate": snapshot.rate,
            "paused": snapshot.paused,
            "direction": if snapshot.rate < 0.0 { "reverse" } else { "forward" },
            "at_end": self.is_at_end(),
            "serial": self.serial(),
            "pending": self.pending.lock().unwrap().is_some(),
            "loop": self.is_loop(),
        })
    }
}

/// Named [`Playback`]s, one per group name, each over the clock of the same
/// name.
pub struct PlaybackService {
    groups: Mutex<HashMap<String, Arc<Playback>>>,
}

impl PlaybackService {
    pub fn new() -> Self {
        Self {
            groups: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_or_create(&self, name: &str, clocks: &ClockService) -> Arc<Playback> {
        let mut groups = self.groups.lock().unwrap();
        groups
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(Playback::new(name, clocks.get_or_create(name))))
            .clone()
    }

    pub fn get(&self, name: &str) -> Option<Arc<Playback>> {
        self.groups.lock().unwrap().get(name).cloned()
    }
}

impl Default for PlaybackService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::services::clock::WallClock;

    fn entries(count: usize, period_ms: i64) -> Vec<IndexEntry> {
        (0..count)
            .map(|i| IndexEntry {
                timestamp_ms: 1000 + i as i64 * period_ms,
                byte_offset: 188 * 10 * i as u64,
            })
            .collect()
    }

    fn index() -> Arc<SeekIndex> {
        // 25 fps, 100 frames, media 1000..4960 ms; UTC = media + 1_700_000_000_000.
        Arc::new(
            SeekIndex::new(
                entries(100, 40),
                vec![HistoryEntry {
                    changed_at: 0,
                    input_offset: 0,
                    wallclock_offset: -1_700_000_000_000,
                    output_offset: 0,
                }],
            )
            .unwrap(),
        )
    }

    #[test]
    fn nearest_prefers_the_later_frame_on_a_tie() {
        let index = index();
        assert_eq!(index.nearest(1000), 0);
        assert_eq!(index.nearest(1019), 0);
        assert_eq!(index.nearest(1020), 1, "a tie selects the later frame");
        assert_eq!(index.nearest(1021), 1);
        assert_eq!(index.nearest(0), 0);
        assert_eq!(index.nearest(99_999), 99);
        assert_eq!(index.frame_of(1039), 0);
        assert_eq!(index.frame_of(1040), 1);
        assert_eq!(index.frame_of(0), 0);
        assert_eq!(index.frame_at_bytes(0), Some(0));
        assert_eq!(index.frame_at_bytes(1879), Some(0));
        assert_eq!(index.frame_at_bytes(1880), Some(1));
        assert_eq!(index.fps(), Some(25));
        assert_eq!(index.duration_ms(), 3960);
    }

    #[test]
    fn history_maps_media_and_wallclock_both_ways() {
        let index = index();
        assert_eq!(index.media_to_wallclock_ms(1000), Some(1_700_000_001_000));
        assert_eq!(index.wallclock_to_media_ms(1_700_000_001_000), Some(1000));
        let bare = SeekIndex::new(entries(3, 40), Vec::new()).unwrap();
        assert_eq!(bare.media_to_wallclock_ms(1000), None);
    }

    #[test]
    fn tables_parse_from_their_native_records() {
        let mut bytes = Vec::new();
        for entry in entries(3, 40) {
            bytes.extend_from_slice(&entry.timestamp_ms.to_ne_bytes());
            bytes.extend_from_slice(&entry.byte_offset.to_ne_bytes());
        }
        assert_eq!(SeekIndex::parse_seek_table(&bytes).unwrap(), entries(3, 40));
        assert!(SeekIndex::parse_seek_table(&bytes[..15]).is_err());
        let mut history = Vec::new();
        for v in [0i64, 0, -5, 0] {
            history.extend_from_slice(&v.to_ne_bytes());
        }
        assert_eq!(
            SeekIndex::parse_history(&history).unwrap(),
            vec![HistoryEntry {
                changed_at: 0,
                input_offset: 0,
                wallclock_offset: -5,
                output_offset: 0
            }]
        );
        assert!(SeekIndex::new(vec![], vec![]).is_err());
    }

    #[test]
    fn targets_parse_in_every_protocol_form() {
        assert_eq!(Target::parse("12000").unwrap(), Target::MediaMs(12000));
        assert_eq!(Target::parse("+500").unwrap(), Target::RelativeMs(500));
        assert_eq!(Target::parse("-500").unwrap(), Target::RelativeMs(-500));
        assert_eq!(
            Target::parse("01:02:03").unwrap(),
            Target::MediaMs(3_723_000)
        );
        assert_eq!(Target::parse("01:00.150").unwrap(), Target::MediaMs(60_150));
        assert_eq!(Target::parse("+0:10").unwrap(), Target::RelativeMs(10_000));
        assert_eq!(Target::parse("-0:00.5").unwrap(), Target::RelativeMs(-500));
        assert_eq!(
            Target::parse("2026-08-10T12:00:00.250").unwrap(),
            Target::Wallclock(1_786_363_200_250)
        );
        assert_eq!(
            Target::parse("2026-08-10T12:00:00Z").unwrap(),
            Target::Wallclock(1_786_363_200_000)
        );
        assert_eq!(
            Target::parse("2026-08-10T14:00:00+02:00").unwrap(),
            Target::Wallclock(1_786_363_200_000)
        );
        assert_eq!(
            Target::parse("1970-01-01T00:00:00").unwrap(),
            Target::Wallclock(0)
        );
        assert_eq!(Target::parse("live").unwrap(), Target::Live);
        assert_eq!(Target::parse("end").unwrap(), Target::End);
        assert!(Target::parse("").is_err());
        assert!(Target::parse("abc").is_err());
        assert!(Target::parse("1:2:3:4").is_err());
    }

    struct Bound {
        playback: Arc<Playback>,
        source: SourceId,
        wakes: Arc<AtomicU64>,
    }

    fn bound(loop_: bool) -> Bound {
        let playback = Arc::new(Playback::new("g", Arc::new(WallClock::new())));
        let wakes = Arc::new(AtomicU64::new(0));
        let counter = wakes.clone();
        let source = playback.bind_source(
            Arc::new(move || {
                counter.fetch_add(1, Ordering::Relaxed);
            }),
            Some(index()),
            loop_,
            1000,
        );
        Bound {
            playback,
            source,
            wakes,
        }
    }

    fn bytes_of(plan: ReadPlan) -> (u64, bool) {
        match plan {
            ReadPlan::Reposition(Reposition {
                to: SeekTo::Bytes(bytes),
                discontinuity,
            }) => (bytes, discontinuity),
            other => panic!("expected an indexed reposition, got {other:?}"),
        }
    }

    #[test]
    fn a_seek_resets_the_clock_and_reaches_the_source_as_a_discontinuity() {
        let b = bound(false);
        assert!(
            b.playback.seek(Target::RelativeMs(10)).is_err(),
            "no position yet"
        );
        assert_eq!(b.playback.seek(Target::MediaMs(2010)).unwrap(), 2000);
        assert_eq!(b.wakes.load(Ordering::Relaxed), 1);
        assert_eq!(b.playback.clock().snapshot().origin_src_us, 2_000_000);
        assert!(b.playback.has_request(b.source));
        assert_eq!(
            bytes_of(b.playback.plan_read(b.source, None)),
            (1880 * 25, true)
        );
        assert!(!b.playback.has_request(b.source));
        assert_eq!(b.playback.plan_read(b.source, None), ReadPlan::Continue);
    }

    #[test]
    fn relative_targets_compose_against_the_pending_seek_until_it_is_observed() {
        let b = bound(false);
        b.playback.seek(Target::MediaMs(2000)).unwrap();
        // Two nudges before anything came out: both count.
        assert_eq!(b.playback.seek(Target::RelativeFrames(5)).unwrap(), 2200);
        assert_eq!(b.playback.seek(Target::RelativeMs(-40)).unwrap(), 2160);
        assert_eq!(b.playback.status()["pending"], json!(true));
        // A stale release from before the flush changes nothing…
        b.playback.report_release(1000);
        assert_eq!(b.playback.status()["pending"], json!(true));
        // …the first one after the flush is the position.
        b.playback.report_flush_stop();
        b.playback.report_release(2160);
        assert_eq!(b.playback.status()["pending"], json!(false));
        assert_eq!(b.playback.status()["frame"], json!(29));
        assert_eq!(b.playback.status()["position_ms"], json!(1160));
        assert_eq!(
            b.playback.status()["wallclock_ms"],
            json!(1_700_000_002_160i64)
        );
        assert_eq!(b.playback.status()["serial"], json!(2));
        assert_eq!(b.playback.seek(Target::RelativeFrames(-1)).unwrap(), 2120);
        assert_eq!(
            b.playback
                .seek(Target::Wallclock(1_700_000_003_000))
                .unwrap(),
            3000
        );
        assert_eq!(b.playback.seek(Target::Frame(-5)).unwrap(), 1000, "clamped");
        assert_eq!(b.playback.seek(Target::End).unwrap(), 4960);
        assert_eq!(b.playback.seek(Target::Live).unwrap(), 3960);
    }

    #[test]
    fn rate_drives_stride_and_direction_and_a_sign_flip_reseeks() {
        let b = bound(false);
        b.playback.seek(Target::Frame(50)).unwrap();
        b.playback.plan_read(b.source, None);
        b.playback.report_flush_stop();
        b.playback.report_release(3000);
        let pos_of = |frame: usize| index().entry(frame).byte_offset;

        b.playback.set_rate(2.0).unwrap();
        // The stride changed: re-seek to the frame on screen first…
        assert_eq!(
            bytes_of(b.playback.plan_read(b.source, None)),
            (pos_of(50), true)
        );
        // …then every other frame is skipped.
        assert_eq!(
            bytes_of(b.playback.plan_read(b.source, Some(pos_of(50)))),
            (pos_of(52), false),
            "at 2x every other frame is skipped"
        );
        b.playback.set_rate(0.5).unwrap();
        assert_eq!(
            bytes_of(b.playback.plan_read(b.source, None)),
            (pos_of(50), true)
        );
        assert_eq!(
            b.playback.plan_read(b.source, Some(pos_of(52))),
            ReadPlan::Continue
        );
        // Same stride, no re-seek.
        b.playback.set_rate(1.0).unwrap();
        assert_eq!(
            b.playback.plan_read(b.source, Some(pos_of(52))),
            ReadPlan::Continue
        );

        b.playback.set_rate(-1.0).unwrap();
        assert_eq!(b.playback.status()["direction"], json!("reverse"));
        // The sign flip re-seeks to the frame on screen…
        assert_eq!(
            bytes_of(b.playback.plan_read(b.source, None)),
            (pos_of(50), true)
        );
        // …and from then on each read steps back one frame.
        assert_eq!(
            bytes_of(b.playback.plan_read(b.source, Some(pos_of(50)))),
            (pos_of(49), false)
        );
        b.playback.set_rate(-2.0).unwrap();
        // A stride change while reversing re-seeks to the frame on screen too…
        assert_eq!(
            bytes_of(b.playback.plan_read(b.source, None)),
            (pos_of(50), true)
        );
        // …then steps back two.
        assert_eq!(
            bytes_of(b.playback.plan_read(b.source, Some(pos_of(49)))),
            (pos_of(47), false)
        );
        // Reaching the start without loop: the reader idles, and the end is
        // reported once frame 0 has been shown.
        assert_eq!(
            b.playback.plan_read(b.source, Some(pos_of(0))),
            ReadPlan::Idle
        );
        assert_eq!(b.playback.status()["at_end"], json!(false));
        b.playback.report_release(index().entry(0).timestamp_ms);
        assert_eq!(b.playback.status()["at_end"], json!(true));

        b.playback.set_rate(0.0).unwrap();
        assert!(b.playback.is_paused());
        b.playback.resume();
        assert!(!b.playback.is_paused());
        assert!(b.playback.set_rate(f64::NAN).is_err());
    }

    #[test]
    fn the_tail_loops_when_the_viewer_gets_there_or_idles() {
        let b = bound(true);
        b.playback.seek(Target::Frame(97)).unwrap();
        b.playback.plan_read(b.source, None);
        b.playback.report_flush_stop();
        // The reader runs out of file while the viewer is two frames behind:
        // idle, not at the end, no loop yet.
        let last_pos = Some(index().entry(99).byte_offset);
        assert_eq!(b.playback.plan_tail(b.source, last_pos), ReadPlan::Idle);
        assert!(!b.playback.is_at_end());
        b.playback.report_release(index().entry(97).timestamp_ms);
        b.playback.report_release(index().entry(98).timestamp_ms);
        assert!(!b.playback.is_at_end());
        assert!(!b.playback.has_request(b.source));
        // The last frame shows: that is the end, and with loop the source is
        // sent back to frame 0 as a discontinuity.
        b.playback.report_release(index().entry(99).timestamp_ms);
        assert_eq!(
            bytes_of(b.playback.plan_read(b.source, last_pos)),
            (0, true)
        );
        assert!(!b.playback.is_at_end(), "the loop seek cleared it");

        // Paused at the end: no loop until resume.
        b.playback.pause();
        b.playback.plan_read(b.source, None);
        b.playback.report_flush_stop();
        assert_eq!(b.playback.plan_tail(b.source, last_pos), ReadPlan::Idle);
        b.playback.report_release(index().entry(99).timestamp_ms);
        assert!(b.playback.is_at_end());
        assert!(
            !b.playback.has_request(b.source),
            "paused: the last frame stays"
        );
        b.playback.resume();
        assert!(b.playback.has_request(b.source), "resume at the end loops");

        // Without loop: at the end, and staying there.
        b.playback.set_loop(false);
        b.playback.plan_read(b.source, None);
        b.playback.report_flush_stop();
        assert_eq!(b.playback.plan_tail(b.source, last_pos), ReadPlan::Idle);
        b.playback.report_release(index().entry(99).timestamp_ms);
        assert!(b.playback.is_at_end());
        assert!(!b.playback.has_request(b.source));
        // A seek clears it.
        b.playback.seek(Target::Frame(3)).unwrap();
        assert!(!b.playback.is_at_end());
    }

    #[test]
    fn scheduled_pause_and_seek_fire_on_release() {
        let b = bound(false);
        b.playback.pause_at(Target::MediaMs(2000)).unwrap();
        b.playback
            .seek_at(Target::MediaMs(3000), Target::Frame(0))
            .unwrap();
        b.playback.report_release(1500);
        assert!(!b.playback.is_paused());
        b.playback.report_release(2000);
        assert!(b.playback.is_paused());
        b.playback.report_release(3000);
        assert!(
            b.playback.has_request(b.source),
            "the scheduled seek was issued"
        );
        b.playback.clear_scheduled();
    }

    #[test]
    fn without_a_source_or_index_the_verbs_say_so() {
        let playback = Playback::new("g", Arc::new(WallClock::new()));
        assert!(playback.seek(Target::MediaMs(0)).is_err());
        let wake: Arc<dyn Fn() + Send + Sync> = Arc::new(|| {});
        let source = playback.bind_source(wake, None, false, 0);
        assert!(
            playback.seek(Target::Frame(0)).is_err(),
            "frames need an index"
        );
        assert!(playback.seek(Target::Wallclock(0)).is_err());
        assert_eq!(playback.seek(Target::MediaMs(1234)).unwrap(), 1234);
        assert!(matches!(
            playback.plan_read(source, None),
            ReadPlan::Reposition(Reposition {
                to: SeekTo::Time {
                    media_ms: 1234,
                    resume_at: Some(_)
                },
                discontinuity: true
            })
        ));
        assert_eq!(playback.status()["frame_count"], Value::Null);
    }
}
