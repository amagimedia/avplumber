//! Single place for rational timestamp conversion.

use crate::graph::buffer::AvpRational;

pub const MICROSECONDS: AvpRational = AvpRational {
    num: 1,
    den: 1_000_000,
};
pub const MILLISECONDS: AvpRational = AvpRational { num: 1, den: 1_000 };

/// libav → native. Both directions live here so no node writes the struct
/// literal by hand.
#[cfg(feature = "ffmpeg")]
impl From<rusty_ffmpeg::ffi::AVRational> for AvpRational {
    fn from(r: rusty_ffmpeg::ffi::AVRational) -> Self {
        Self {
            num: r.num,
            den: r.den,
        }
    }
}

#[cfg(feature = "ffmpeg")]
impl From<AvpRational> for rusty_ffmpeg::ffi::AVRational {
    fn from(r: AvpRational) -> Self {
        Self {
            num: r.num,
            den: r.den,
        }
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
pub fn ts_cmp(val_a: i64, tb_a: AvpRational, val_b: i64, tb_b: AvpRational) -> std::cmp::Ordering {
    let a = (val_a as i128) * (tb_a.num as i128) * (tb_b.den as i128);
    let b = (val_b as i128) * (tb_b.num as i128) * (tb_a.den as i128);
    a.cmp(&b)
}

/// Compare two time bases by value, so "finer" and "coarser" mean the same here
/// as they do in C++ (`av::Rational`'s `<`): the *smaller* rational is the finer
/// one. A degenerate time base (`den == 0`) sorts as the coarsest.
pub fn tb_cmp(a: AvpRational, b: AvpRational) -> std::cmp::Ordering {
    match (a.den == 0, b.den == 0) {
        (true, true) => std::cmp::Ordering::Equal,
        (true, false) => std::cmp::Ordering::Greater,
        (false, true) => std::cmp::Ordering::Less,
        (false, false) => {
            ((a.num as i128) * (b.den as i128)).cmp(&((b.num as i128) * (a.den as i128)))
        }
    }
}

/// The finer of two time bases: what an addition of two timestamps happens in,
/// like C++ `addTS`'s `std::min`. An unusable one loses, so `NOPTS`'s `0/0` never
/// becomes the result.
pub fn finer(a: AvpRational, b: AvpRational) -> AvpRational {
    if a.num <= 0 || a.den <= 0 {
        return b;
    }
    if b.num <= 0 || b.den <= 0 {
        return a;
    }
    if tb_cmp(a, b).is_le() { a } else { b }
}
