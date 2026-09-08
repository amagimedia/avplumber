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

/// Reduces `num/den` by their gcd, keeping the sign on the numerator.
pub fn reduce(r: AvpRational) -> AvpRational {
    fn gcd(a: i64, b: i64) -> i64 {
        if b == 0 { a.abs() } else { gcd(b, a % b) }
    }
    let g = gcd(r.num as i64, r.den as i64);
    if g <= 1 {
        return r;
    }
    AvpRational {
        num: (r.num as i64 / g) as i32,
        den: (r.den as i64 / g) as i32,
    }
}

/// C++ `parseRatio`: `"30"`, `"30/1"`, `"1/25"`, `"29.97"`. A decimal is
/// scaled by a power of ten and reduced, so `"29.97"` is `2997/100`.
pub fn parse_rational(text: &str) -> Result<AvpRational, String> {
    let text = text.trim();
    let invalid = || format!("`{text}` is not a rational (want N, N/D or a decimal)");
    let parse_int = |part: &str| part.trim().parse::<i64>().map_err(|_| invalid());
    let (num, den) = match text.split_once('/') {
        Some((n, d)) => (parse_int(n)?, parse_int(d)?),
        None => match text.split_once('.') {
            Some((whole, frac)) if !frac.is_empty() && frac.len() <= 9 => {
                let scale = 10i64.pow(frac.len() as u32);
                let whole_value = if whole.is_empty() || whole == "-" {
                    0
                } else {
                    parse_int(whole)?
                };
                let frac_value = parse_int(frac)?;
                let sign = if text.starts_with('-') { -1 } else { 1 };
                (sign * (whole_value.abs() * scale + frac_value), scale)
            }
            _ => (parse_int(text)?, 1),
        },
    };
    if den == 0 {
        return Err(format!("`{text}`: the denominator is zero"));
    }
    if num.abs() > i32::MAX as i64 || den > i32::MAX as i64 {
        return Err(format!("`{text}` does not fit a rational"));
    }
    Ok(reduce(AvpRational {
        num: num as i32,
        den: den as i32,
    }))
}

/// A rational node parameter as scripts write it: a string for
/// [`parse_rational`], or a JSON number (an integer as `N/1`, a float through
/// its shortest decimal form).
pub fn rational_from_json(value: &serde_json::Value) -> Result<AvpRational, String> {
    match value {
        serde_json::Value::String(text) => parse_rational(text),
        serde_json::Value::Number(number) => parse_rational(&number.to_string()),
        other => Err(format!("`{other}` is not a rational")),
    }
}

#[cfg(test)]
mod rational_tests {
    use super::*;

    fn r(num: i32, den: i32) -> AvpRational {
        AvpRational { num, den }
    }

    #[test]
    fn parses_the_forms_scripts_use() {
        assert_eq!(parse_rational("30").unwrap(), r(30, 1));
        assert_eq!(parse_rational("30/1").unwrap(), r(30, 1));
        assert_eq!(parse_rational("1/25").unwrap(), r(1, 25));
        assert_eq!(parse_rational(" 50 / 2 ").unwrap(), r(25, 1));
        assert_eq!(parse_rational("29.97").unwrap(), r(2997, 100));
        assert_eq!(parse_rational("0.04").unwrap(), r(1, 25));
        assert_eq!(parse_rational("-1").unwrap(), r(-1, 1));
        assert_eq!(parse_rational("-0.5").unwrap(), r(-1, 2));
        assert_eq!(
            rational_from_json(&serde_json::json!(25)).unwrap(),
            r(25, 1)
        );
        assert_eq!(
            rational_from_json(&serde_json::json!(0.5)).unwrap(),
            r(1, 2)
        );
        assert_eq!(
            rational_from_json(&serde_json::json!("1/30")).unwrap(),
            r(1, 30)
        );
    }

    #[test]
    fn rejects_garbage_and_zero_denominators() {
        assert!(parse_rational("").is_err());
        assert!(parse_rational("abc").is_err());
        assert!(parse_rational("1/0").is_err());
        assert!(parse_rational("1/2/3").is_err());
        assert!(rational_from_json(&serde_json::json!(true)).is_err());
    }
}
