//! CorrectionGroup: shared reference timeline + per-member cursors. No Sentinel policy.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use crate::graph::AvpRational;
use crate::graph::media::Ts;
use crate::graph::timebase::{self, ts_cmp};
use crate::services::clock::SyncGroup;

#[derive(Clone, Debug)]
pub struct CorrectionSnapshot {
    pub output_tb: AvpRational,
    pub start_ts: Option<Ts>,
    pub locked_shift: bool,
    pub last_discontinuity: Option<Ts>,
    pub expected_ts: Ts,
    pub generation: u64,
}

#[derive(Clone, Debug)]
pub struct MemberCursor {
    pub name: String,
    pub next_ts: Ts,
    pub generation: u64,
}

struct Member {
    next_ts: Ts,
}

struct Inner {
    output_tb: AvpRational,
    start_ts: Option<Ts>,
    locked_shift: bool,
    last_discontinuity: Option<Ts>,
    expected_ts: Ts,
    live_driven: bool,
    generation: u64,
    members: HashMap<String, Member>,
}

pub struct CorrectionGroup {
    name: String,
    inner: Mutex<Inner>,
}

impl CorrectionGroup {
    pub fn new(name: String) -> Self {
        Self {
            name,
            inner: Mutex::new(Inner {
                output_tb: AvpRational {
                    num: 1,
                    den: 1_000_000,
                },
                start_ts: None,
                locked_shift: false,
                last_discontinuity: None,
                expected_ts: Ts::invalid(),
                live_driven: false,
                generation: 0,
                members: HashMap::new(),
            }),
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn register(&self, member: &str) {
        let mut g = self.inner.lock().unwrap();
        g.members.entry(member.to_string()).or_insert(Member {
            next_ts: Ts::invalid(),
        });
        g.generation += 1;
    }

    pub fn unregister(&self, member: &str) {
        let mut g = self.inner.lock().unwrap();
        g.members.remove(member);
        g.generation += 1;
    }

    pub fn set_output_tb(&self, tb: AvpRational) {
        let mut g = self.inner.lock().unwrap();
        if g.output_tb.num == 0 {
            g.output_tb = tb;
        } else if timebase::rescale(1, tb, g.output_tb) > 1 {
            g.output_tb = tb;
        }
        g.generation += 1;
    }

    pub fn set_start_ts(&self, ts: Ts) {
        let mut g = self.inner.lock().unwrap();
        g.start_ts = Some(ts);
        if !g.expected_ts.is_valid() {
            g.expected_ts = ts;
        }
        g.generation += 1;
    }

    pub fn set_locked_shift(&self, locked: bool) {
        self.inner.lock().unwrap().locked_shift = locked;
    }

    pub fn note_discontinuity(&self, at: Ts) {
        let mut g = self.inner.lock().unwrap();
        g.last_discontinuity = Some(at);
        g.generation += 1;
    }

    pub fn snapshot(&self) -> CorrectionSnapshot {
        let g = self.inner.lock().unwrap();
        CorrectionSnapshot {
            output_tb: g.output_tb,
            start_ts: g.start_ts,
            locked_shift: g.locked_shift,
            last_discontinuity: g.last_discontinuity,
            expected_ts: g.expected_ts,
            generation: g.generation,
        }
    }

    pub fn member_cursor(&self, member: &str) -> Option<MemberCursor> {
        let g = self.inner.lock().unwrap();
        let m = g.members.get(member)?;
        Some(MemberCursor {
            name: member.to_string(),
            next_ts: m.next_ts,
            generation: g.generation,
        })
    }

    /// Commit a member's next timestamp. Stale `generation` is rejected so a
    /// lock is never held across an edge push.
    pub fn commit(&self, member: &str, next_ts: Ts, generation: u64) -> Result<(), String> {
        let mut g = self.inner.lock().unwrap();
        if generation != g.generation {
            return Err("stale correction generation".into());
        }
        let m = g
            .members
            .get_mut(member)
            .ok_or_else(|| format!("unknown member {member}"))?;
        m.next_ts = next_ts;
        if !g.live_driven
            && (!g.expected_ts.is_valid()
                || (next_ts.is_valid()
                    && ts_cmp(next_ts.val, next_ts.tb, g.expected_ts.val, g.expected_ts.tb)
                        .is_gt()))
        {
            g.expected_ts = next_ts;
        }
        g.generation += 1;
        Ok(())
    }

    pub fn advance_live(&self, clock: &dyn SyncGroup) {
        loop {
            let (output_tb, generation) = {
                let g = self.inner.lock().unwrap();
                if !g.expected_ts.is_valid() {
                    return;
                }
                (g.output_tb, g.generation)
            };
            let Some(position) = clock.live_position(output_tb) else {
                return;
            };
            let candidate = Ts {
                val: position,
                tb: output_tb,
            };

            let mut g = self.inner.lock().unwrap();
            if g.generation != generation || g.output_tb != output_tb {
                continue;
            }
            let mut changed = false;
            if !g.live_driven {
                g.live_driven = true;
                changed = true;
            }
            if !g.expected_ts.is_valid()
                || ts_cmp(
                    candidate.val,
                    candidate.tb,
                    g.expected_ts.val,
                    g.expected_ts.tb,
                )
                .is_gt()
            {
                g.expected_ts = candidate;
                changed = true;
            }
            if changed {
                g.generation += 1;
            }
            return;
        }
    }

    pub fn expected_file(&self) -> Ts {
        self.inner.lock().unwrap().expected_ts
    }
}

pub struct CorrectionService {
    groups: Mutex<HashMap<String, Arc<CorrectionGroup>>>,
}

impl CorrectionService {
    pub fn new() -> Self {
        Self {
            groups: Mutex::new(HashMap::new()),
        }
    }
    pub fn get_or_create(&self, name: &str) -> Arc<CorrectionGroup> {
        self.groups
            .lock()
            .unwrap()
            .entry(name.to_string())
            .or_insert_with(|| Arc::new(CorrectionGroup::new(name.to_string())))
            .clone()
    }
}

impl Default for CorrectionService {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Barrier};
    use std::thread;

    use super::CorrectionGroup;
    use crate::graph::{AVP_NOPTS, AvpRational, Ts};
    use crate::services::clock::{ClockSnapshot, SourceTimeClock, SyncGroup, SyntheticClock};

    const MILLIS: AvpRational = AvpRational { num: 1, den: 1000 };
    const MICROS: AvpRational = AvpRational {
        num: 1,
        den: 1_000_000,
    };

    #[test]
    fn live_expected_timestamp_advances_while_input_is_absent() {
        let correction = CorrectionGroup::new("live".into());
        let clock = SyntheticClock::new();
        clock.reset(1_000, MILLIS);
        correction.set_output_tb(MILLIS);
        correction.set_start_ts(Ts {
            val: 1_000,
            tb: MILLIS,
        });

        clock.set_position(1_030);
        correction.advance_live(&clock);

        let advanced = correction.snapshot().expected_ts;
        assert_eq!(advanced.tb, MILLIS);
        assert_eq!(advanced.val, 1_030);
    }

    #[test]
    fn concurrent_live_advancement_is_monotonic_and_preserves_member_state() {
        let correction = Arc::new(CorrectionGroup::new("live".into()));
        let clock = Arc::new(SyntheticClock::new());
        clock.reset(2_000, MILLIS);
        correction.set_output_tb(MILLIS);
        correction.set_start_ts(Ts {
            val: 2_000,
            tb: MILLIS,
        });
        correction.register("video");
        let cursor = correction.member_cursor("video").unwrap();
        correction
            .commit(
                "video",
                Ts {
                    val: 2_010,
                    tb: MILLIS,
                },
                cursor.generation,
            )
            .unwrap();
        correction.set_locked_shift(true);
        clock.set_position(3_000);

        let workers = (0..4)
            .map(|_| {
                let correction = correction.clone();
                let clock = clock.clone();
                thread::spawn(move || {
                    let mut previous = correction.snapshot().expected_ts.val;
                    for _ in 0..100 {
                        correction.advance_live(clock.as_ref());
                        let current = correction.snapshot().expected_ts.val;
                        assert!(current >= previous);
                        previous = current;
                    }
                })
            })
            .collect::<Vec<_>>();
        for worker in workers {
            worker.join().unwrap();
        }

        let snapshot = correction.snapshot();
        assert!(snapshot.locked_shift);
        assert_eq!(snapshot.output_tb, MILLIS);
        assert_eq!(
            snapshot.expected_ts,
            Ts {
                val: 3_000,
                tb: MILLIS
            }
        );
        assert_eq!(
            correction.member_cursor("video").unwrap().next_ts,
            Ts {
                val: 2_010,
                tb: MILLIS,
            }
        );
    }

    #[test]
    fn file_clock_cannot_advance_live_correction() {
        let correction = CorrectionGroup::new("file".into());
        correction.set_output_tb(MILLIS);
        correction.set_start_ts(Ts {
            val: 1_000,
            tb: MILLIS,
        });
        let clock = SourceTimeClock::new();
        clock.reset(1_000, MILLIS);

        correction.advance_live(&clock);

        assert_eq!(
            correction.snapshot().expected_ts,
            Ts {
                val: 1_000,
                tb: MILLIS
            }
        );
    }

    #[test]
    fn live_member_cursor_can_run_ahead_without_advancing_wall_expected() {
        let correction = CorrectionGroup::new("live".into());
        correction.set_output_tb(MILLIS);
        correction.set_start_ts(Ts { val: 0, tb: MILLIS });
        correction.register("sentinel");
        let clock = SyntheticClock::new();
        clock.reset(0, MILLIS);
        clock.set_position(30);
        correction.advance_live(&clock);
        let cursor = correction.member_cursor("sentinel").unwrap();

        correction
            .commit(
                "sentinel",
                Ts {
                    val: 40,
                    tb: MILLIS,
                },
                cursor.generation,
            )
            .unwrap();

        assert_eq!(
            correction.snapshot().expected_ts,
            Ts {
                val: 30,
                tb: MILLIS
            }
        );
        assert_eq!(
            correction.member_cursor("sentinel").unwrap().next_ts,
            Ts {
                val: 40,
                tb: MILLIS
            }
        );
    }

    struct RacingClock {
        entered: Arc<Barrier>,
        release: Arc<Barrier>,
        calls: std::sync::atomic::AtomicUsize,
    }

    impl SyncGroup for RacingClock {
        fn set_rate(&self, _rate: f64) {}
        fn set_paused(&self, _paused: bool) {}
        fn reset(&self, _new_pos: i64, _tb: AvpRational) {}
        fn map_to_wall(&self, _src_pts: i64, _tb: AvpRational) -> i64 {
            AVP_NOPTS
        }
        fn live_position(&self, tb: AvpRational) -> Option<i64> {
            let call = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            if call == 0 {
                self.entered.wait();
                self.release.wait();
            }
            Some(if tb == MICROS { 3_000_000 } else { 3_000 })
        }
        fn join_offset(&self, _offset_us: i64) {}
        fn snapshot(&self) -> ClockSnapshot {
            ClockSnapshot {
                origin_src_us: 0,
                origin_wall_us: 0,
                rate: 1.0,
                paused: false,
                epoch: 0,
            }
        }
    }

    #[test]
    fn output_timebase_change_retries_live_candidate_coherently() {
        let correction = Arc::new(CorrectionGroup::new("race".into()));
        correction.set_start_ts(Ts {
            val: 1_000_000,
            tb: MICROS,
        });
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let clock = Arc::new(RacingClock {
            entered: entered.clone(),
            release: release.clone(),
            calls: std::sync::atomic::AtomicUsize::new(0),
        });
        let advancing = {
            let correction = correction.clone();
            let clock = clock.clone();
            thread::spawn(move || correction.advance_live(clock.as_ref()))
        };

        entered.wait();
        correction.set_output_tb(MILLIS);
        release.wait();
        advancing.join().unwrap();

        let snapshot = correction.snapshot();
        assert_eq!(snapshot.output_tb, MILLIS);
        assert_eq!(
            snapshot.expected_ts,
            Ts {
                val: 3_000,
                tb: MILLIS
            }
        );
        assert_eq!(
            clock.calls.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "stale candidate was published without retry"
        );
    }
}
