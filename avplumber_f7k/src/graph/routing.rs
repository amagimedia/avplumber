//! The `routing` key grammar and its resolution against a
//! [`Spec::Catalog`](crate::graph::spec::Spec::Catalog).
//!
//! A query over [`CatalogStream`], which is why it sits here beside
//! [`spec`](super::spec) rather than with its one caller,
//! [`demux`](crate::nodes::demux): [`MEDIA_TYPE_VIDEO`] and friends are a
//! hand-written shadow of the raw `AVMEDIA_TYPE_*` encoding
//! [`CatalogStream::codec_type`] carries, and the two want to be in one module
//! so a change to that encoding is visible in one place.
//!
//! Free of libav, like `spec` itself and for the same reason: the grammar and
//! the ordinal rules stay unit-tested in the default build. The node that uses
//! it needs `Media::Packet` and so needs the `ffmpeg` feature.

use crate::graph::spec::CatalogStream;

/// Raw `AVMEDIA_TYPE_*`, matching [`CatalogStream::codec_type`]. Spelled out
/// rather than pulled from `rusty_ffmpeg` so this module builds without libav;
/// `tests::media_type_constants_match_libav` checks them against the real enum
/// whenever the `ffmpeg` feature is on.
pub const MEDIA_TYPE_VIDEO: i32 = 0;
pub const MEDIA_TYPE_AUDIO: i32 = 1;
pub const MEDIA_TYPE_DATA: i32 = 2;

/// What one `routing` key selects.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RouteTarget {
    /// `v`, `a:2`, `d:0` — the `ordinal`-th stream of that media type, counting
    /// only streams the `streams_filter` matched.
    Kind { codec_type: i32, ordinal: usize },
    /// `3` — that container stream index, filter ignored.
    Index(i32),
}

/// One parsed `routing` key.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct RouteKey {
    /// `?` prefix: a missing stream is logged, not an error.
    pub optional: bool,
    pub target: RouteTarget,
}

/// `[?]{v|V|a|A|d|D}[:index]` or `[?]<absolute index>`, ported from
/// `src/nodes/demux.cpp:102-190`.
///
/// Stricter than C++ in one place: C++ reaches `std::stoi`, which stops at the
/// first non-digit and so silently accepts `v:0abc` and `1x`. Here the index
/// part must be entirely digits.
pub fn parse_route_key(key: &str) -> Result<RouteKey, String> {
    let (optional, rest) = match key.strip_prefix('?') {
        Some(rest) => (true, rest),
        None => (false, key),
    };
    let mut chars = rest.chars();
    let tag = chars
        .next()
        .ok_or_else(|| format!("routing key `{key}` is too short"))?;
    let codec_type = match tag {
        'v' | 'V' => MEDIA_TYPE_VIDEO,
        'a' | 'A' => MEDIA_TYPE_AUDIO,
        'd' | 'D' => MEDIA_TYPE_DATA,
        '0'..='9' => {
            let index = parse_index(rest)
                .ok_or_else(|| format!("routing key `{key}` is not a stream index"))?;
            return Ok(RouteKey {
                optional,
                target: RouteTarget::Index(index as i32),
            });
        }
        other => {
            return Err(format!(
                "routing key `{key}` has an invalid media type `{other}`"
            ));
        }
    };
    // The tag is ASCII, so the rest of the key starts at byte 1.
    let ordinal = if rest.len() == 1 {
        0
    } else {
        let suffix = rest[1..]
            .strip_prefix(':')
            .filter(|suffix| !suffix.is_empty())
            .ok_or_else(|| format!("routing key `{key}` must be `{tag}` or `{tag}:<index>`"))?;
        parse_index(suffix).ok_or_else(|| {
            format!("routing key `{key}` has a non-numeric stream index `{suffix}`")
        })?
    };
    Ok(RouteKey {
        optional,
        target: RouteTarget::Kind {
            codec_type,
            ordinal,
        },
    })
}

fn parse_index(text: &str) -> Option<usize> {
    if text.is_empty() || !text.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    text.parse().ok()
}

/// The catalog stream a key selects, or `None` when there is no such stream.
///
/// Matches C++ exactly on the filter's reach: a type-tag key counts only
/// streams with `matches_filter`, while an absolute index addresses the stream
/// directly and ignores the filter.
pub fn resolve(key: &RouteKey, streams: &[CatalogStream]) -> Option<i32> {
    match key.target {
        RouteTarget::Index(index) => streams
            .iter()
            .find(|stream| stream.index == index)
            .map(|stream| stream.index),
        RouteTarget::Kind {
            codec_type,
            ordinal,
        } => streams
            .iter()
            .filter(|stream| stream.codec_type == codec_type && stream.matches_filter)
            .nth(ordinal)
            .map(|stream| stream.index),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kind(codec_type: i32, ordinal: usize) -> RouteTarget {
        RouteTarget::Kind {
            codec_type,
            ordinal,
        }
    }

    fn parsed(key: &str) -> RouteKey {
        parse_route_key(key).unwrap_or_else(|e| panic!("`{key}` should parse: {e}"))
    }

    #[test]
    fn a_bare_type_tag_means_the_first_stream_of_that_type() {
        assert_eq!(parsed("v").target, kind(MEDIA_TYPE_VIDEO, 0));
        assert_eq!(parsed("a").target, kind(MEDIA_TYPE_AUDIO, 0));
        assert_eq!(parsed("d").target, kind(MEDIA_TYPE_DATA, 0));
        assert!(!parsed("v").optional);
    }

    #[test]
    fn the_type_tag_is_case_insensitive() {
        assert_eq!(parsed("V").target, parsed("v").target);
        assert_eq!(parsed("A:2").target, parsed("a:2").target);
        assert_eq!(parsed("D").target, parsed("d").target);
    }

    #[test]
    fn a_colon_suffix_overrides_the_ordinal() {
        assert_eq!(parsed("a:2").target, kind(MEDIA_TYPE_AUDIO, 2));
        assert_eq!(parsed("v:10").target, kind(MEDIA_TYPE_VIDEO, 10));
    }

    #[test]
    fn a_leading_question_mark_marks_the_route_optional() {
        let key = parsed("?d:0");
        assert!(key.optional);
        assert_eq!(key.target, kind(MEDIA_TYPE_DATA, 0));
        assert!(parsed("?3").optional);
    }

    #[test]
    fn a_leading_digit_means_an_absolute_stream_index() {
        assert_eq!(parsed("3").target, RouteTarget::Index(3));
        assert_eq!(parsed("12").target, RouteTarget::Index(12));
        assert_eq!(parsed("0").target, RouteTarget::Index(0));
    }

    #[test]
    fn malformed_keys_are_rejected() {
        for bad in ["", "?", "x", "v:", "v:x", "v2", "a::1", "1x", "-1", "v:1:2"] {
            assert!(
                parse_route_key(bad).is_err(),
                "`{bad}` should not have parsed"
            );
        }
    }

    fn catalog() -> Vec<CatalogStream> {
        // Video, then two audio tracks, then a data track — with the *first*
        // audio track filtered out, which is the case that separates ordinal
        // counting from stream indices.
        vec![
            stream(0, MEDIA_TYPE_VIDEO, true),
            stream(1, MEDIA_TYPE_AUDIO, false),
            stream(2, MEDIA_TYPE_AUDIO, true),
            stream(3, MEDIA_TYPE_DATA, true),
        ]
    }

    fn stream(index: i32, codec_type: i32, matches_filter: bool) -> CatalogStream {
        CatalogStream {
            index,
            codec_type,
            spec: Default::default(),
            matches_filter,
        }
    }

    #[test]
    fn type_tag_ordinals_count_only_streams_the_filter_matched() {
        let streams = catalog();
        assert_eq!(resolve(&parsed("v"), &streams), Some(0));
        assert_eq!(resolve(&parsed("a"), &streams), Some(2));
        assert_eq!(resolve(&parsed("a:1"), &streams), None);
        assert_eq!(resolve(&parsed("d"), &streams), Some(3));
    }

    #[test]
    fn an_absolute_index_ignores_the_filter() {
        let streams = catalog();
        assert_eq!(resolve(&parsed("1"), &streams), Some(1));
        assert_eq!(resolve(&parsed("9"), &streams), None);
    }

    #[cfg(feature = "ffmpeg")]
    #[test]
    fn media_type_constants_match_libav() {
        use rusty_ffmpeg::ffi;
        assert_eq!(MEDIA_TYPE_VIDEO, ffi::AVMEDIA_TYPE_VIDEO);
        assert_eq!(MEDIA_TYPE_AUDIO, ffi::AVMEDIA_TYPE_AUDIO);
        assert_eq!(MEDIA_TYPE_DATA, ffi::AVMEDIA_TYPE_DATA);
    }
}
