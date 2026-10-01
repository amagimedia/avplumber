import shutil
import subprocess
import sys
from pathlib import Path

import pytest


MIXER_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MIXER_DIR))
sys.path.insert(0, str(MIXER_DIR.parents[1]))  # repo root: pyplumber package


@pytest.fixture(scope="session")
def nvenc():
    """Skips unless FFmpeg encodes H.264 and HEVC on NVENC here: test media is never encoded in
    software. A probe encode, because an FFmpeg build lists NVENC encoders without a GPU too."""
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        pytest.skip("requires FFmpeg and ffprobe")
    for codec in ("h264_nvenc", "hevc_nvenc"):
        probe = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=size=256x144:rate=4:duration=0.25",
                                "-pix_fmt", "yuv420p", "-c:v", codec, "-f", "null", "-"], capture_output=True)
        if probe.returncode:
            pytest.skip(f"requires an NVIDIA GPU with {codec}")
