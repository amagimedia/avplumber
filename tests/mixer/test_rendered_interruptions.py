import pytest

np = pytest.importorskip('numpy')
from check_transition_control import identity, longest_still_run
from frame_codes import frame_image


def test_identity_names_each_scene_and_rejects_a_blend():
    assert identity(frame_image(0, 27)) == ('full0', 27)
    assert identity(frame_image(1, 28)) == ('full1', 28)
    split = np.concatenate([frame_image(0, 30)[:, ::2], frame_image(1, 31)[:, ::2]], axis=1)
    assert identity(split) == ('split', 30)
    blend = ((frame_image(0, 27).astype(np.uint16) + frame_image(1, 28)) // 2).astype(np.uint8)
    assert identity(blend) == (None, None)


def test_longest_still_run_finds_a_frozen_stretch():
    assert longest_still_run([]) == 0
    assert longest_still_run([b'a', b'b', b'c']) == 1
    assert longest_still_run([b'a', b'b', b'b', b'b', b'c', b'c']) == 3
