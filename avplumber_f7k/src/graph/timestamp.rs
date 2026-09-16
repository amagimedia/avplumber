//! Timestamps and conversion between time bases.

use crate::graph::buffer::{AVP_NOPTS, AvpRational};
use crate::graph::timebase::finer;

#[derive(Clone, Copy, Debug)]
pub struct Ts {
    pub val: i64,
    pub tb: AvpRational,
}

impl Ts {
    pub fn invalid() -> Self {
        Self {
            val: AVP_NOPTS,
            tb: AvpRational { num: 0, den: 0 },
        }
    }

    pub fn is_valid(self) -> bool {
        self.val != AVP_NOPTS
    }

    pub fn rescale(self, to: AvpRational) -> Ts {
        if !self.is_valid() {
            return Self {
                val: AVP_NOPTS,
                tb: to,
            };
        }
        Ts {
            val: rescale(self.val, self.tb, to),
            tb: to,
        }
    }
}

/// Sum in the finer of the two time bases, like C++ `addTS`. An invalid operand
/// makes the sum invalid: `NOPTS + shift` is not a timestamp.
///
/// To stay in one specific time base — a packet's own, say — rescale the other
/// operand into it first, which makes this the `addTSSameTB` of C++.
impl std::ops::Add for Ts {
    type Output = Ts;

    fn add(self, other: Ts) -> Ts {
        if !self.is_valid() || !other.is_valid() {
            return Ts::invalid();
        }
        let tb = finer(self.tb, other.tb);
        Ts {
            val: rescale(self.val, self.tb, tb) + rescale(other.val, other.tb, tb),
            tb,
        }
    }
}

impl PartialEq for Ts {
    fn eq(&self, other: &Self) -> bool {
        match (self.is_valid(), other.is_valid()) {
            (false, false) => true,
            (true, true) => ts_cmp(self.val, self.tb, other.val, other.tb).is_eq(),
            _ => false,
        }
    }
}

impl Eq for Ts {}

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
pub fn ts_cmp(val_a: i64, tb_a: AvpRational, val_b: i64, tb_b: AvpRational) -> std::cmp::Ordering {
    let a = (val_a as i128) * (tb_a.num as i128) * (tb_b.den as i128);
    let b = (val_b as i128) * (tb_b.num as i128) * (tb_a.den as i128);
    a.cmp(&b)
}
