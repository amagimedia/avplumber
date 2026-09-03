//! SyncGroup master clock — atomic snapshot, lock-free map_to_wall.

use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use arc_swap::ArcSwap;
use std::sync::OnceLock;

use crate::graph::timebase::{self, MICROSECONDS};
use crate::graph::{AVP_NOPTS, AvpRational};

#[derive(Clone, Copy, Debug)]
pub struct ClockSnapshot {
    pub origin_src_us: i64,
    pub origin_wall_us: i64,
    pub rate: f64,
    pub paused: bool,
    pub epoch: u64,
}

pub trait SyncGroup: Send + Sync {
    fn set_rate(&self, rate: f64);
    fn set_paused(&self, paused: bool);
    fn reset(&self, new_pos: i64, tb: AvpRational);
    fn map_to_wall(&self, src_pts: i64, tb: AvpRational) -> i64;
    /// Current source position for clocks backed by monotonic wall time.
    ///
    /// File/source-time clocks return `None`: callers must not manufacture
    /// live progress for timelines that only advance when input arrives.
    fn live_position(&self, _tb: AvpRational) -> Option<i64> {
        None
    }
    fn join_offset(&self, offset_us: i64);
    fn snapshot(&self) -> ClockSnapshot;
}

fn now_us() -> i64 {
    static ORIGIN: OnceLock<Instant> = OnceLock::new();
    ORIGIN.get_or_init(Instant::now).elapsed().as_micros() as i64
}

fn src_us(pts: i64, tb: AvpRational) -> i64 {
    timebase::rescale(pts, tb, MICROSECONDS)
}

fn rate_or_one(rate: f64) -> f64 {
    if rate.abs() < 1e-12 { 1e-9 } else { rate }
}

fn map_snap(s: &ClockSnapshot, pts: i64, tb: AvpRational) -> i64 {
    if s.paused {
        return AVP_NOPTS;
    }
    let src = src_us(pts, tb);
    let delta = ((src - s.origin_src_us) as f64 / rate_or_one(s.rate)) as i64;
    s.origin_wall_us + delta
}

struct SnapshotClock {
    snap: ArcSwap<ClockSnapshot>,
    write: Mutex<()>,
    offset_us: AtomicI64,
    epoch: AtomicU64,
}

impl SnapshotClock {
    fn new() -> Self {
        Self {
            snap: ArcSwap::from_pointee(ClockSnapshot {
                origin_src_us: 0,
                origin_wall_us: now_us(),
                rate: 1.0,
                paused: false,
                epoch: 0,
            }),
            write: Mutex::new(()),
            offset_us: AtomicI64::new(i64::MAX),
            epoch: AtomicU64::new(0),
        }
    }

    fn publish(&self, mut s: ClockSnapshot) {
        s.epoch = self.epoch.fetch_add(1, Ordering::AcqRel) + 1;
        self.snap.store(Arc::new(s));
    }

    fn load(&self) -> ClockSnapshot {
        **self.snap.load()
    }
}

impl SyncGroup for SnapshotClock {
    fn set_rate(&self, rate: f64) {
        let _g = self.write.lock().unwrap();
        let mut s = self.load();
        let now = now_us();
        let src_now =
            s.origin_src_us + ((now - s.origin_wall_us) as f64 * rate_or_one(s.rate)) as i64;
        s.origin_src_us = src_now;
        s.origin_wall_us = now;
        s.rate = rate;
        self.publish(s);
    }

    fn set_paused(&self, paused: bool) {
        let _g = self.write.lock().unwrap();
        let mut s = self.load();
        if s.paused == paused {
            return;
        }
        if paused {
            s.paused = true;
        } else {
            s.origin_wall_us = now_us();
            s.paused = false;
        }
        self.publish(s);
    }

    fn reset(&self, new_pos: i64, tb: AvpRational) {
        let _g = self.write.lock().unwrap();
        let mut s = self.load();
        s.origin_src_us = src_us(new_pos, tb);
        s.origin_wall_us = now_us();
        self.publish(s);
        self.offset_us.store(i64::MAX, Ordering::Release);
    }

    fn map_to_wall(&self, src_pts: i64, tb: AvpRational) -> i64 {
        map_snap(&self.load(), src_pts, tb)
    }

    fn live_position(&self, tb: AvpRational) -> Option<i64> {
        let s = self.load();
        if s.paused {
            return None;
        }
        let source_us =
            s.origin_src_us + ((now_us() - s.origin_wall_us) as f64 * rate_or_one(s.rate)) as i64;
        Some(timebase::rescale(source_us, MICROSECONDS, tb))
    }

    fn join_offset(&self, offset_us: i64) {
        let mut cur = self.offset_us.load(Ordering::Acquire);
        while offset_us < cur {
            match self.offset_us.compare_exchange_weak(
                cur,
                offset_us,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => {
                    let _g = self.write.lock().unwrap();
                    let mut s = self.load();
                    s.origin_wall_us = now_us() + offset_us.min(0);
                    self.publish(s);
                    return;
                }
                Err(v) => cur = v,
            }
        }
        if cur == i64::MAX {
            if self
                .offset_us
                .compare_exchange(i64::MAX, offset_us, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                let _g = self.write.lock().unwrap();
                let mut s = self.load();
                s.origin_wall_us = now_us() + offset_us.min(0);
                self.publish(s);
            }
        }
    }

    fn snapshot(&self) -> ClockSnapshot {
        self.load()
    }
}

pub struct WallClock(SnapshotClock);
impl WallClock {
    pub fn new() -> Self {
        Self(SnapshotClock::new())
    }
}
impl Default for WallClock {
    fn default() -> Self {
        Self::new()
    }
}
impl SyncGroup for WallClock {
    fn set_rate(&self, r: f64) {
        self.0.set_rate(r);
    }
    fn set_paused(&self, p: bool) {
        self.0.set_paused(p);
    }
    fn reset(&self, n: i64, tb: AvpRational) {
        self.0.reset(n, tb);
    }
    fn map_to_wall(&self, src: i64, tb: AvpRational) -> i64 {
        self.0.map_to_wall(src, tb)
    }
    fn live_position(&self, tb: AvpRational) -> Option<i64> {
        self.0.live_position(tb)
    }
    fn join_offset(&self, o: i64) {
        self.0.join_offset(o);
    }
    fn snapshot(&self) -> ClockSnapshot {
        self.0.snapshot()
    }
}

/// File / pass-through clock: mapping is source time in microseconds.
pub struct SourceTimeClock(SnapshotClock);
impl SourceTimeClock {
    pub fn new() -> Self {
        Self(SnapshotClock::new())
    }
}
impl Default for SourceTimeClock {
    fn default() -> Self {
        Self::new()
    }
}
impl SyncGroup for SourceTimeClock {
    fn set_rate(&self, r: f64) {
        self.0.set_rate(r);
    }
    fn set_paused(&self, p: bool) {
        self.0.set_paused(p);
    }
    fn reset(&self, n: i64, tb: AvpRational) {
        self.0.reset(n, tb);
    }
    fn map_to_wall(&self, src: i64, tb: AvpRational) -> i64 {
        let s = self.0.load();
        if s.paused {
            return src;
        }
        timebase::rescale(src, tb, MICROSECONDS)
    }
    fn join_offset(&self, o: i64) {
        self.0.join_offset(o);
    }
    fn snapshot(&self) -> ClockSnapshot {
        self.0.snapshot()
    }
}

/// Deterministic clock for tests: wall = source-delta * rate in the caller's units.
pub struct SyntheticClock {
    inner: SnapshotClock,
    position: AtomicI64,
}

impl SyntheticClock {
    pub fn new() -> Self {
        Self {
            inner: SnapshotClock::new(),
            position: AtomicI64::new(0),
        }
    }

    pub fn set_position(&self, position: i64) {
        self.position.store(position, Ordering::Release);
    }
}
impl Default for SyntheticClock {
    fn default() -> Self {
        Self::new()
    }
}
impl SyncGroup for SyntheticClock {
    fn set_rate(&self, r: f64) {
        self.inner.set_rate(r);
    }
    fn set_paused(&self, p: bool) {
        self.inner.set_paused(p);
    }
    fn reset(&self, n: i64, _tb: AvpRational) {
        let _g = self.inner.write.lock().unwrap();
        let mut s = self.inner.load();
        s.origin_src_us = n;
        s.origin_wall_us = 0;
        self.inner.publish(s);
        self.position.store(n, Ordering::Release);
    }
    fn map_to_wall(&self, src_pts: i64, _tb: AvpRational) -> i64 {
        let s = self.inner.load();
        if s.paused {
            return AVP_NOPTS;
        }
        ((src_pts - s.origin_src_us) as f64 * rate_or_one(s.rate)) as i64
    }
    fn live_position(&self, _tb: AvpRational) -> Option<i64> {
        let s = self.inner.load();
        if s.paused {
            None
        } else {
            Some(self.position.load(Ordering::Acquire))
        }
    }
    fn join_offset(&self, o: i64) {
        self.inner.join_offset(o);
    }
    fn snapshot(&self) -> ClockSnapshot {
        self.inner.snapshot()
    }
}

pub struct ClockService {
    groups: Mutex<HashMap<String, Arc<dyn SyncGroup>>>,
}

impl ClockService {
    pub fn new() -> Self {
        Self {
            groups: Mutex::new(HashMap::new()),
        }
    }
    pub fn get_or_create(&self, name: &str) -> Arc<dyn SyncGroup> {
        let mut g = self.groups.lock().unwrap();
        g.entry(name.to_string())
            .or_insert_with(|| Arc::new(WallClock::new()))
            .clone()
    }
    pub fn insert(&self, name: &str, clock: Arc<dyn SyncGroup>) {
        self.groups.lock().unwrap().insert(name.to_string(), clock);
    }
}

impl Default for ClockService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::thread;

    use super::{SourceTimeClock, SyncGroup, SyntheticClock, WallClock};
    use crate::graph::AvpRational;

    const MILLIS: AvpRational = AvpRational { num: 1, den: 1000 };

    #[test]
    fn synthetic_live_position_is_programmed_without_sleeping() {
        let clock = SyntheticClock::new();
        clock.reset(5_000, MILLIS);
        clock.set_position(5_040);

        assert_eq!(clock.live_position(MILLIS), Some(5_040));
    }

    #[test]
    fn source_time_clock_does_not_claim_live_wall_progress() {
        let clock = SourceTimeClock::new();
        clock.reset(5_000, MILLIS);

        assert_eq!(clock.live_position(MILLIS), None);
    }

    #[test]
    fn concurrent_snapshot_reads_never_mix_published_clock_states() {
        let clock = Arc::new(WallClock::new());
        clock.reset(0, MILLIS);
        let writer = {
            let clock = clock.clone();
            thread::spawn(move || {
                for update in 0..2_000 {
                    clock.set_rate(if update % 2 == 0 { 1.0 } else { 2.0 });
                }
            })
        };
        let readers = (0..4)
            .map(|_| {
                let clock = clock.clone();
                thread::spawn(move || {
                    let mut previous_epoch = 0;
                    for _ in 0..2_000 {
                        let snapshot = clock.snapshot();
                        assert!(snapshot.epoch >= previous_epoch);
                        assert!(snapshot.rate == 1.0 || snapshot.rate == 2.0);
                        previous_epoch = snapshot.epoch;
                    }
                })
            })
            .collect::<Vec<_>>();

        writer.join().unwrap();
        for reader in readers {
            reader.join().unwrap();
        }
    }
}
