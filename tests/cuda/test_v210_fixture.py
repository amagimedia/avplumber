import struct

import numpy as np
import pytest

from v210_fixture import frame_stride, pack_v210, sample_planes, write_fixture


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
