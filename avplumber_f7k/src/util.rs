//! Shared helpers that are not part of the graph, the executor, or a service.

/// Unix milliseconds. Requires `Z` or a numeric offset; a bare datetime is rejected.
pub fn parse_iso8601_ms(text: &str) -> Result<i64, String> {
    let text = text.trim();
    text.parse::<jiff::Timestamp>()
        .map(|ts| ts.as_millisecond())
        .map_err(|err| format!("`{text}` is not an ISO 8601 timestamp with a zone: {err}"))
}

#[cfg(test)]
mod tests {
    use super::parse_iso8601_ms;

    #[test]
    fn unix_epoch_offset_and_fraction() {
        assert_eq!(parse_iso8601_ms("1970-01-01T00:00:00Z").unwrap(), 0);
        assert_eq!(parse_iso8601_ms("1970-01-01T01:00:00+01:00").unwrap(), 0);
        assert_eq!(
            parse_iso8601_ms("2020-01-01T00:00:00Z").unwrap(),
            1_577_836_800_000
        );
        assert_eq!(
            parse_iso8601_ms("2026-08-10T12:00:00.250Z").unwrap(),
            1_786_363_200_250
        );
    }

    #[test]
    fn rejects_bare_and_impossible_dates() {
        assert!(parse_iso8601_ms("2020-02-31T00:00:00Z").is_err());
        assert!(parse_iso8601_ms("1970-01-01T00:00:00").is_err());
        assert!(parse_iso8601_ms("not-a-date").is_err());
    }
}
