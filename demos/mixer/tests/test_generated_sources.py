"""Configured fixture dimensions and cadence must survive H.264 encoding."""
import json
import shutil
import subprocess

import pytest

np = pytest.importorskip("numpy")
from frame_codes import generate, read_code


@pytest.mark.parametrize("width,height,fps", [(1920, 1080, 60), (1280, 720, 24), (640, 360, 30)])
def test_generated_source_has_requested_size_rate_and_frame_ids(tmp_path, width, height, fps):
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("requires FFmpeg and ffprobe")
    encoders = subprocess.check_output(["ffmpeg", "-hide_banner", "-encoders"], stderr=subprocess.DEVNULL)
    if b"libx264" not in encoders:
        pytest.skip("requires FFmpeg with libx264")
    path = tmp_path / "source.mp4"
    generate(path, 3, fps, 1, width, height)
    stream = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_streams", "-of", "json", str(path),
    ]))["streams"][0]
    assert (stream["width"], stream["height"], stream["avg_frame_rate"]) == (width, height, f"{fps}/1")
    frames = subprocess.Popen([
        "ffmpeg", "-v", "error", "-i", str(path), "-pix_fmt", "gray", "-f", "rawvideo", "pipe:1",
    ], stdout=subprocess.PIPE)
    try:
        for number in range(fps):
            data = frames.stdout.read(width * height)
            assert len(data) == width * height
            assert read_code(np.frombuffer(data, np.uint8).reshape(height, width)) == (3, number)
        assert frames.stdout.read(1) == b""
        assert frames.wait(timeout=10) == 0
    finally:
        if frames.poll() is None:
            frames.kill()
            frames.wait()


@pytest.mark.parametrize("width,height,fps", [(1919, 1080, 60), (1920, 1079, 60), (1920, 1080, 0)])
def test_invalid_source_settings_fail_before_encoding(tmp_path, width, height, fps):
    with pytest.raises(ValueError):
        generate(tmp_path / "source.mp4", 0, fps, 1, width, height)
    assert not list(tmp_path.iterdir())


def test_existing_source_is_not_overwritten(tmp_path):
    path = tmp_path / "source.mp4"
    path.write_bytes(b"existing source")
    with pytest.raises(FileExistsError):
        generate(path, 0, 60, 1)
    assert path.read_bytes() == b"existing source"
