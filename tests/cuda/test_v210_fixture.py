import struct

import numpy as np
import pytest

from v210_fixture import (FAMILIES, frame_stride, hlg_planes, manifest, pack_v210,
                          sample_planes, sdr8_planes, write_fixture)


def test_known_v210_words():
    # Six pixels occupy four words: U0 Y0 V0 | Y1 U1 Y2 | V1 Y3 U2 | Y4 V2 Y5.
    y = np.array([[0, 1, 63, 64, 940, 1023]], dtype="<u2")
    u = np.array([[512, 513, 514]], dtype="<u2")
    v = np.array([[1023, 1, 0]], dtype="<u2")
    assert pack_v210((y, u, v), 16) == struct.pack(
        "<4I", 512 | (0 << 10) | (1023 << 20),
        1 | (513 << 10) | (63 << 20),
        1 | (64 << 10) | (514 << 20),
        940 | (0 << 10) | (1023 << 20))


@pytest.mark.parametrize("width", [2, 4, 6, 46, 48, 50, 1920])
def test_frame_size_and_padding(width):
    stride = frame_stride(width)
    payload = ((width * 2 + 2) // 3) * 4
    data = pack_v210(sample_planes(width, 3), stride)
    assert len(data) == stride * 3
    for row in range(3):
        assert data[row * stride + payload:(row + 1) * stride] == b"\xa5" * (stride - payload)


@pytest.mark.parametrize("width,stride", [(0, None), (3, None), (6, 12), (6, 17)])
def test_invalid_layout(width, stride):
    with pytest.raises(ValueError):
        frame_stride(width, stride)


def test_simulated_frames_change(tmp_path):
    path = tmp_path / "frames.v210"
    stride = write_fixture(path, 50, 7, 2)
    data = path.read_bytes()
    assert len(data) == 2 * stride * 7
    assert data[:stride * 7] != data[stride * 7:]


def test_hlg_checkpoints_and_range():
    y, u, v = hlg_planes(96, 8, index=3)
    # BT.2100 transfer checkpoints: E = 0, 1/12, 1 -> Y' = 0, 0.5, ~1.0.
    for i, code in enumerate((64, 502, 940)):
        np.testing.assert_array_equal(y[:2, 12 * i:12 * (i + 1)], code)
        np.testing.assert_array_equal(u[:2, 6 * i:6 * (i + 1)], 512)
        np.testing.assert_array_equal(v[:2, 6 * i:6 * (i + 1)], 512)
    assert y.min() >= 64 and y.max() <= 940
    for c in (u, v):
        assert c.min() >= 64 and c.max() <= 960
    assert not np.array_equal(y, hlg_planes(96, 8, index=4)[0])


def test_sdr8_promotion_is_shifted_8bit():
    for plane, hi in zip(sdr8_planes(48, 5, index=2), (940, 960, 960)):
        assert not np.any(plane & 3), "promoted samples must be multiples of four"
        assert plane.min() >= 64 and plane.max() <= hi


@pytest.mark.parametrize("family", sorted(FAMILIES))
def test_families_pack_and_manifest(tmp_path, family):
    path = tmp_path / f"{family}.v210"
    stride = write_fixture(path, 48, 4, 2, family=family, source=1)
    assert path.stat().st_size == 2 * stride * 4
    m = manifest(48, 4, 2, stride, family)
    assert m["frame_bytes"] == stride * 4 and m["chroma_location"] == "left"
    assert (m["color_trc"] == "arib-std-b67") == (family == "hlg")
