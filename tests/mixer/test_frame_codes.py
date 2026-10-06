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
    rows = [{"frame": 0, "codes": [{"capacity": 2, "slot": 0, "source": 0, "id": 40},
                                    {"capacity": 2, "slot": 1, "source": 1, "id": 50}]},
            {"frame": 1, "codes": []},
            {"frame": 2, "codes": [{"capacity": 2, "slot": 0, "source": 0, "id": 42}]}]
    errors = visibility_errors(rows, [{"start": 0, "end": 3, "sources": [0, 1]}])
    assert [error["frame"] for error in errors] == [1, 2]
    assert errors[0]["missing_slots"] == {0: 0, 1: 1}
    assert errors[1]["missing_slots"] == {1: 1}


def test_only_explicit_transition_intervals_allow_unreadable_frames():
    rows = [{"frame": 0, "codes": [{"capacity": 1, "slot": 0, "source": 0, "id": 40}]},
            {"frame": 1, "codes": []},
            {"frame": 2, "codes": [{"capacity": 1, "slot": 0, "source": 1, "id": 50}]}]
    schedule = [{"start": 0, "end": 1, "sources": [0]},
                {"start": 1, "end": 2, "transition": "fade"},
                {"start": 2, "end": 3, "sources": [1]}]
    assert visibility_errors(rows, schedule) == []
    with pytest.raises(ValueError, match="cover"):
        visibility_errors(rows, schedule[:2])


def test_fullscreen_cannot_pass_with_a_grid_containing_the_expected_source():
    rows = [{"frame": 0, "codes": [
        {"capacity": 2, "slot": slot, "source": slot, "id": 40}
        for slot in range(2)]}]
    errors = visibility_errors(rows, [{"start": 0, "end": 1, "sources": [0]}])
    assert errors and errors[0]["missing_slots"] == {0: 0}


def test_permuted_tiles_and_unexpected_tiles_are_rejected():
    rows = [{"frame": 0, "codes": [
        {"capacity": 4, "slot": slot, "source": source, "id": 40}
        for slot, source in enumerate((1, 0, 3))]}]
    schedule = [{"start": 0, "end": 1, "capacity": 4, "slots": {"0": 0, "1": 1}}]
    errors = visibility_errors(rows, schedule)
    assert errors[0]["missing_slots"] == {0: 0, 1: 1}
    assert len(errors[0]["unexpected_tiles"]) == 3


@pytest.mark.parametrize("kind", [True, "cut", "cuda_wipe", "unknown"])
def test_unknown_or_uncheckable_transition_exemptions_fail_closed(kind):
    with pytest.raises(ValueError, match="transition"):
        visibility_errors([{"frame": 0, "codes": []}],
                          [{"start": 0, "end": 1, "transition": kind}])


def test_partial_grid_preserves_explicit_slot_mapping():
    row = {"frame": 0, "codes": [{"capacity": 4, "slot": 2, "source": 7, "id": 1}]}
    assert visibility_errors([row], [{"start": 0, "end": 1, "capacity": 4,
                                      "slots": {"2": 7}}]) == []
