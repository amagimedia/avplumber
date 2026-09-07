import pytest

np = pytest.importorskip("numpy")
from frame_codes import frame_image, read_code
from verify_recording import continuity, visibility_errors


@pytest.mark.parametrize("source,frame", [(0, 0), (3, 2227), (15, 65535)])
def test_frame_code_survives_grid_scaling(source, frame):
    image = frame_image(source, frame)
    assert read_code(image[::2, ::2]) == (source, frame)


def test_transition_blend_is_rejected_instead_of_counted_as_a_frame():
    first = frame_image(0, 0).astype(np.uint16)
    second = frame_image(1, 123).astype(np.uint16)
    assert read_code(((first + second) // 2).astype(np.uint8)) is None


def test_returning_source_must_have_advanced_while_hidden():
    rows = [
        {"frame": 0, "codes": [{"source": 0, "id": 40}]},
        {"frame": 1, "codes": [{"source": 0, "id": 41}]},
        {"frame": 30, "codes": [{"source": 0, "id": 70}]},
        {"frame": 31, "codes": [{"source": 0, "id": 42}]},
    ]
    result = continuity(rows, 7200)
    assert result[0]["anomalies"] == [{"frame": 31, "id": 42}]


def test_recording_cannot_pass_after_sources_disappear():
    rows = [{"frame": 0, "codes": [{"source": 0, "id": 40}, {"source": 1, "id": 50}]},
            {"frame": 1, "codes": []},
            {"frame": 2, "codes": [{"source": 0, "id": 42}]}]
    errors = visibility_errors(rows, [{"start": 0, "end": 3, "sources": [0, 1]}])
    assert errors == [{"frame": 1, "missing_sources": [0, 1]},
                      {"frame": 2, "missing_sources": [1]}]


def test_only_explicit_transition_intervals_allow_unreadable_frames():
    rows = [{"frame": 0, "codes": [{"source": 0, "id": 40}]},
            {"frame": 1, "codes": []},
            {"frame": 2, "codes": [{"source": 1, "id": 50}]}]
    schedule = [{"start": 0, "end": 1, "sources": [0]},
                {"start": 1, "end": 2, "transition": True},
                {"start": 2, "end": 3, "sources": [1]}]
    assert visibility_errors(rows, schedule) == []
    with pytest.raises(ValueError, match="cover"):
        visibility_errors(rows, schedule[:2])
