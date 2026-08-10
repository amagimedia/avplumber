import struct
from datetime import datetime, timezone

import pytest

from replay import (
    read_seek_table,
    read_timestamp_history,
    validate_recording,
)


def test_reads_native_packed_seek_entries(tmp_path):
    path = tmp_path / "clip.ts+seek"
    path.write_bytes(struct.pack("=qQqQ", 1_260, 188, 1_300, 752))

    assert read_seek_table(path) == (
        (1_260, 188),
        (1_300, 752),
    )


@pytest.mark.parametrize(
    "payload, message",
    [
        (b"", "empty"),
        (b"short", "multiple of 16"),
        (struct.pack("=qQqQ", 200, 188, 100, 376), "timestamps"),
        (struct.pack("=qQqQ", 100, 376, 200, 188), "offsets"),
    ],
)
def test_rejects_invalid_seek_tables(tmp_path, payload, message):
    path = tmp_path / "clip.ts+seek"
    path.write_bytes(payload)

    with pytest.raises(ValueError, match=message):
        read_seek_table(path)


def test_missing_seek_table_is_reported(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_seek_table(tmp_path / "missing")


def test_reads_history_and_maps_frame_zero_to_utc(tmp_path):
    history = tmp_path / "clip.ts+history"
    origin_ms = 1_786_314_600_000
    history.write_bytes(struct.pack("=qqqq", 0, 0, 1_260 - origin_ms, 0))

    entries = read_timestamp_history(history)

    assert entries == ((0, 0, -1_786_314_598_740, 0),)
    assert entries.media_to_wallclock_ms(1_260) == origin_ms
    assert entries.media_to_wallclock_ms(6_260) == origin_ms + 5_000
    assert entries.wallclock_to_media_ms(origin_ms + 9_000) == 10_260


@pytest.mark.parametrize("payload", [b"", b"bad"])
def test_rejects_invalid_history_size(tmp_path, payload):
    path = tmp_path / "clip.ts+history"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        read_timestamp_history(path)


def test_validates_recording_and_infers_integer_fps(tmp_path):
    recording = tmp_path / "clip.ts"
    recording.write_bytes(b"mpegts")
    timestamps = (1_260, 1_293, 1_327, 1_360, 1_393, 1_427)
    (tmp_path / "clip.ts+seek").write_bytes(
        b"".join(struct.pack("=qQ", ts, 188 * (index + 1)) for index, ts in enumerate(timestamps))
    )
    origin_ms = int(datetime(2026, 8, 10, 12, tzinfo=timezone.utc).timestamp() * 1000)
    (tmp_path / "clip.ts+history").write_bytes(
        struct.pack("=qqqq", 0, 0, timestamps[0] - origin_ms, 0)
    )

    artifact = validate_recording(recording)

    assert artifact.fps == 30
    assert artifact.frame_count == 6
    assert artifact.start_ms == 1_260
    assert artifact.duration_ms == 167
    assert artifact.wallclock_start_ms == origin_ms


def test_rejects_inconsistent_frame_cadence(tmp_path):
    recording = tmp_path / "clip.ts"
    recording.write_bytes(b"mpegts")
    (tmp_path / "clip.ts+seek").write_bytes(
        b"".join(struct.pack("=qQ", ts, 188 * index) for index, ts in enumerate((0, 33, 66, 150)))
    )
    (tmp_path / "clip.ts+history").write_bytes(struct.pack("=qqqq", 0, 0, -1_000, 0))

    with pytest.raises(ValueError, match="cadence"):
        validate_recording(recording)
