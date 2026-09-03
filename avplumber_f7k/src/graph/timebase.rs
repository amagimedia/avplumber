//! Single place for rational timestamp conversion.

use crate::graph::buffer::AvpRational;

pub const MICROSECONDS: AvpRational = AvpRational {
    num: 1,
    den: 1_000_000,
};
pub const MILLISECONDS: AvpRational = AvpRational { num: 1, den: 1_000 };

/// `av_rescale_q(val, from, to)`: `val * from / to`.
pub fn rescale(val: i64, from: AvpRational, to: AvpRational) -> i64 {
    if from.den == 0 || to.num == 0 {
        return val;
    }
    #[cfg(feature = "ffmpeg")]
    {
        return unsafe {
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
        };
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
