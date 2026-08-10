import struct
from datetime import datetime, timezone

import pytest

from transcode import output_collisions, parse_args, parse_wallclock_start, write_history


def test_wallclock_now_is_resolved_once_from_supplied_clock():
    expected = datetime(2026, 8, 10, 12, 30, tzinfo=timezone.utc)
    assert parse_wallclock_start("now", now=lambda: expected) is expected


@pytest.mark.parametrize(
    "value, expected",
    [
        ("2026-08-10T12:30:00Z", "2026-08-10T12:30:00+00:00"),
        ("2026-08-10T14:30:00+02:00", "2026-08-10T12:30:00+00:00"),
    ],
)
def test_wallclock_accepts_timezone_qualified_iso8601(value, expected):
    assert parse_wallclock_start(value).isoformat() == expected


def test_wallclock_rejects_timezone_less_value():
    with pytest.raises(ValueError, match="timezone"):
        parse_wallclock_start("2026-08-10T12:30:00")


def test_cli_requires_explicit_fps(tmp_path):
    source = tmp_path / "source.mp4"
    source.write_bytes(b"vod")
    with pytest.raises(SystemExit):
        parse_args(["--input", str(source), "--output", str(tmp_path / "out.ts")])


def test_output_collisions_include_canonical_and_rotated_files(tmp_path):
    output = tmp_path / "out.ts"
    paths = (tmp_path / "out.ts+history", tmp_path / "out.ts+seek.2")
    for path in paths:
        path.write_bytes(b"old")
    assert output_collisions(output) == paths


def test_history_record_maps_first_packet_to_selected_wallclock(tmp_path):
    path = tmp_path / "out.ts+history"
    origin = datetime(2026, 8, 10, 12, tzinfo=timezone.utc)
    write_history(path, first_timestamp_ms=1_260, wallclock_start=origin)
    assert struct.unpack("=qqqq", path.read_bytes()) == (
        0,
        0,
        -1_786_363_198_740,
        0,
    )
