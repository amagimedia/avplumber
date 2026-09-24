//! A media point ([`Ts`]) and a signed media span ([`TsDelta`]).
//!
//! The split matches `Instant` and `Duration`: a point plus a span is a point,
//! and two points subtract to a span. Adding two points does not compile.
//! Spans are signed, because a shift or a DTS-before-PTS gap can be negative.
//!
//! Arithmetic uses the finer time base, like C++ `addTS`. An invalid operand
//! makes the result invalid: `NOPTS + shift` is not a timestamp. To stay in one
//! specific time base — a packet's own, say — rescale the other operand into it
//! first, which makes the sum the `addTSSameTB` of C++.

use std::cmp::Ordering;

use crate::graph::media::{AVP_NOPTS, AvpRational};
use crate::graph::timebase::finer;

#[derive(Clone, Copy, Debug)]
struct Ticks {
    val: i64,
    tb: AvpRational,
}

impl Ticks {
    fn invalid() -> Self {
        Self {
            val: AVP_NOPTS,
            tb: AvpRational { num: 0, den: 0 },
        }
    }

    fn is_valid(self) -> bool {
        self.val != AVP_NOPTS
    }

    fn rescale(self, to: AvpRational) -> Self {
        if !self.is_valid() {
            return Self {
                val: AVP_NOPTS,
                tb: to,
            };
        }
        Self {
            val: rescale(self.val, self.tb, to),
            tb: to,
        }
    }
}

/// Sum or difference in the finer time base. `None` when either side is invalid.
fn combine(a: Ticks, b: Ticks) -> Option<(i64, i64, AvpRational)> {
    if !a.is_valid() || !b.is_valid() {
        return None;
    }
    let tb = finer(a.tb, b.tb);
    Some((rescale(a.val, a.tb, tb), rescale(b.val, b.tb, tb), tb))
}

fn cmp_ticks(a: Ticks, b: Ticks) -> Ordering {
    match (a.is_valid(), b.is_valid()) {
        (false, false) => Ordering::Equal,
        (false, true) => Ordering::Less,
        (true, false) => Ordering::Greater,
        (true, true) => ts_cmp(a.val, a.tb, b.val, b.tb),
    }
}

/// A point on a media timeline. `AVP_NOPTS` is invalid.
#[derive(Clone, Copy, Debug)]
pub struct Ts(Ticks);

/// A signed span of media time. Zero is a valid empty span.
#[derive(Clone, Copy, Debug)]
pub struct TsDelta(Ticks);

impl Ts {
    pub fn new(ticks: i64, tb: AvpRational) -> Self {
        Self(Ticks { val: ticks, tb })
    }

    pub fn invalid() -> Self {
        Self(Ticks::invalid())
    }

    pub fn is_valid(self) -> bool {
        self.0.is_valid()
    }

    pub fn ticks(self) -> i64 {
        self.0.val
    }

    pub fn timebase(self) -> AvpRational {
        self.0.tb
    }

    pub fn rescale(self, to: AvpRational) -> Self {
        Self(self.0.rescale(to))
    }
}

impl TsDelta {
    pub fn new(ticks: i64, tb: AvpRational) -> Self {
        Self(Ticks { val: ticks, tb })
    }

    pub fn zero(tb: AvpRational) -> Self {
        Self(Ticks { val: 0, tb })
    }

    pub fn invalid() -> Self {
        Self(Ticks::invalid())
    }

    pub fn is_valid(self) -> bool {
        self.0.is_valid()
    }

    pub fn ticks(self) -> i64 {
        self.0.val
    }

    pub fn timebase(self) -> AvpRational {
        self.0.tb
    }

    pub fn rescale(self, to: AvpRational) -> Self {
        Self(self.0.rescale(to))
    }
}

impl std::ops::Add<TsDelta> for Ts {
    type Output = Ts;

    fn add(self, delta: TsDelta) -> Ts {
        match combine(self.0, delta.0) {
            None => Ts::invalid(),
            Some((a, b, tb)) => Ts::new(a + b, tb),
        }
    }
}

impl std::ops::Add<Ts> for TsDelta {
    type Output = Ts;

    fn add(self, point: Ts) -> Ts {
        point + self
    }
}

impl std::ops::Sub<TsDelta> for Ts {
    type Output = Ts;

    fn sub(self, delta: TsDelta) -> Ts {
        match combine(self.0, delta.0) {
            None => Ts::invalid(),
            Some((a, b, tb)) => Ts::new(a - b, tb),
        }
    }
}

impl std::ops::Sub for Ts {
    type Output = TsDelta;

    fn sub(self, earlier: Ts) -> TsDelta {
        match combine(self.0, earlier.0) {
            None => TsDelta::invalid(),
            Some((a, b, tb)) => TsDelta::new(a - b, tb),
        }
    }
}

impl std::ops::Add for TsDelta {
    type Output = TsDelta;

    fn add(self, other: TsDelta) -> TsDelta {
        match combine(self.0, other.0) {
            None => TsDelta::invalid(),
            Some((a, b, tb)) => TsDelta::new(a + b, tb),
        }
    }
}

impl std::ops::Sub for TsDelta {
    type Output = TsDelta;

    fn sub(self, other: TsDelta) -> TsDelta {
        match combine(self.0, other.0) {
            None => TsDelta::invalid(),
            Some((a, b, tb)) => TsDelta::new(a - b, tb),
        }
    }
}

impl std::ops::Neg for TsDelta {
    type Output = TsDelta;

    fn neg(self) -> TsDelta {
        if !self.is_valid() {
            return TsDelta::invalid();
        }
        TsDelta::new(-self.ticks(), self.timebase())
    }
}

impl PartialEq for Ts {
    fn eq(&self, other: &Self) -> bool {
        cmp_ticks(self.0, other.0).is_eq()
    }
}

impl Eq for Ts {}

impl PartialOrd for Ts {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Ts {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_ticks(self.0, other.0)
    }
}

impl PartialEq for TsDelta {
    fn eq(&self, other: &Self) -> bool {
        cmp_ticks(self.0, other.0).is_eq()
    }
}

impl Eq for TsDelta {}

impl PartialOrd for TsDelta {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TsDelta {
    fn cmp(&self, other: &Self) -> Ordering {
        cmp_ticks(self.0, other.0)
    }
}

/// `av_rescale_q(val, from, to)`: `val * from / to`.
pub fn rescale(val: i64, from: AvpRational, to: AvpRational) -> i64 {
    if from.den == 0 || to.num == 0 {
        return val;
    }
    #[cfg(feature = "ffmpeg")]
    {
        unsafe {
            rusty_ffmpeg::ffi::av_rescale_q(
                val,
                rusty_ffmpeg::ffi::AVRational {
                    num: from.num,
                    den: from.den,
                },
                rusty_ffmpeg::ffi::AVRational {
                    num: to.num,
                    den: to.den,
                },
            )
        }
    }
    #[cfg(not(feature = "ffmpeg"))]
    {
        rescale_i128(val, from, to)
    }
}

#[allow(dead_code)]
pub fn rescale_i128(val: i64, from: AvpRational, to: AvpRational) -> i64 {
    if from.den == 0 || to.num == 0 {
        return val;
    }
    let num = (val as i128) * (from.num as i128) * (to.den as i128);
    let den = (from.den as i128) * (to.num as i128);
    if den == 0 {
        return val;
    }
    (num / den) as i64
}

/// Compare two timestamps as rationals `(val * tb.num) / tb.den`.
pub fn ts_cmp(val_a: i64, tb_a: AvpRational, val_b: i64, tb_b: AvpRational) -> Ordering {
    let a = (val_a as i128) * (tb_a.num as i128) * (tb_b.den as i128);
    let b = (val_b as i128) * (tb_b.num as i128) * (tb_a.den as i128);
    a.cmp(&b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tb(den: i32) -> AvpRational {
        AvpRational { num: 1, den }
    }

    #[test]
    fn sum_uses_the_finer_timebase() {
        let sum = Ts::new(1, tb(1)) + TsDelta::new(500, tb(1000));
        assert_eq!(sum.timebase(), tb(1000));
        assert_eq!(sum.ticks(), 1500);
    }

    #[test]
    fn difference_of_points_is_a_span() {
        let delta = Ts::new(2000, tb(1000)) - Ts::new(1, tb(1));
        assert_eq!(delta.timebase(), tb(1000));
        assert_eq!(delta.ticks(), 1000);
    }

    #[test]
    fn negative_span_moves_a_point_backward() {
        let delta = TsDelta::new(-5, tb(1000));
        let moved = Ts::new(10, tb(1000)) + delta;
        assert_eq!(moved.ticks(), 5);
        assert_eq!((-delta).ticks(), 5);
        assert_eq!((TsDelta::zero(tb(1000)) - delta).ticks(), 5);
    }

    #[test]
    fn nopts_poisons_the_result() {
        let bad = Ts::invalid();
        let delta = TsDelta::new(1, tb(1000));
        assert!(!(bad + delta).is_valid());
        assert!(!(bad - Ts::new(0, tb(1000))).is_valid());
        assert!(!(TsDelta::invalid() + delta).is_valid());
        assert!(!(-TsDelta::invalid()).is_valid());
    }

    #[test]
    fn invalid_points_compare_equal() {
        assert_eq!(Ts::invalid(), Ts::new(AVP_NOPTS, tb(1000)));
        assert!(Ts::invalid() < Ts::new(0, tb(1)));
    }
}
