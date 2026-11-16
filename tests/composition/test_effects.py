"""Compare every 8-bit source code against the actual legacy FFmpeg expression."""

import shutil
import subprocess

import pytest

from pyplumber.composition.effects import white_wash_lut


def test_white_wash_is_exactly_the_legacy_ffmpeg_lut_for_every_y_u_v_code():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("FFmpeg is needed for the independent legacy reference")
    y = bytes(range(256)) * 4
    uv = bytes(value for code in range(256) for value in (code, 255 - code))
    result = subprocess.run(
        [
            ffmpeg,
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            "nv12",
            "-video_size",
            "32x32",
            "-i",
            "pipe:0",
            "-vf",
            "lutyuv=y=val*0.7+70.5:u=(val-128)*0.7+128:v=(val-128)*0.7+128,format=nv12",
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "pipe:1",
        ],
        input=y + uv,
        capture_output=True,
        check=True,
    )
    lut = white_wash_lut()
    expected = bytes(lut[0][v] for v in y) + bytes(
        lut[1 + i % 2][v] for i, v in enumerate(uv)
    )
    assert result.stdout == expected


@pytest.mark.parametrize("amount", [-1, 1.01, float("nan")])
def test_invalid_effect_strength_is_rejected(amount):
    with pytest.raises(ValueError):
        white_wash_lut(amount)
