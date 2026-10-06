import pytest

np = pytest.importorskip('numpy')
from check_rendered_interruptions import assert_frozen
from frame_codes import frame_image


def test_retained_partial_picture_rejects_a_single_endpoint_flash():
    old = frame_image(0, 27)
    target = frame_image(1, 28)
    partial = ((old.astype(np.uint16) + target) // 2).astype(np.uint8)
    held = [partial.copy() for _ in range(12)]
    assert_frozen(partial, held)
    held[6] = old
    with pytest.raises(AssertionError, match='frame 6'):
        assert_frozen(partial, held)


def test_retained_overlay_cannot_be_replaced_with_a_scene_endpoint():
    scene = frame_image(0, 30)
    overlay = scene.copy()
    overlay[50:250, 100:450] = (16, 240, 180)
    with pytest.raises(AssertionError, match='frame 0'):
        assert_frozen(overlay, [scene] * 10)


def test_reference_may_not_be_chosen_from_the_already_frozen_result():
    reference = frame_image(0, 30)
    older = frame_image(0, 29)
    with pytest.raises(AssertionError, match='frame 0'):
        assert_frozen(reference, [older] * 10)


def test_missing_output_cannot_pass_snapshot_check():
    with pytest.raises(AssertionError, match='fewer than eight'):
        assert_frozen(frame_image(0, 0), [])
