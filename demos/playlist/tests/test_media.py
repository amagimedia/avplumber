"""Five deterministic 1080p30 ten-second fixture contract."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from player import FIXTURE_NAMES, demo_clips


MEDIA_DIR = Path(__file__).parents[1] / "test-media"
GENERATOR = MEDIA_DIR / "generate.sh"


def test_default_scenario_has_exactly_five_unique_local_fixtures():
    fixture_clips = demo_clips(MEDIA_DIR)
    assert len(fixture_clips) == 5
    assert tuple(Path(clip.url).name for clip in fixture_clips) == FIXTURE_NAMES
    assert len({clip.item_id for clip in fixture_clips}) == 5
    assert all("basketball" not in clip.url.lower() for clip in fixture_clips)


def test_generator_declares_distinct_patterns_frame_counter_and_pts_clock():
    source = GENERATOR.read_text()
    for pattern in ("testsrc2", "smptebars", "smptehdbars", "rgbtestsrc",
                    "yuvtestsrc"):
        assert f"generate_clip {pattern} " in source
    assert "FRAME %{n}" in source
    assert "PTS %{pts\\\\:hms}" in source
    assert "-frames:v 300" in source
    assert "size=1920x1080:rate=30" in source
    assert "basketball" not in source.lower()


@pytest.mark.skipif(shutil.which("ffprobe") is None,
                    reason="ffprobe is not installed")
def test_generated_files_are_unique_h264_1080p30_300_frame_ten_second_clips():
    paths = [MEDIA_DIR / name for name in FIXTURE_NAMES]
    if not all(path.exists() for path in paths):
        pytest.skip("generated media artifacts are not present")

    hashes = set()
    for path in paths:
        result = subprocess.run([
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries",
            "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ], check=True, capture_output=True, text=True)
        stream = json.loads(result.stdout)["streams"][0]
        assert stream == {
            "codec_name": "h264",
            "width": 1920,
            "height": 1080,
            "r_frame_rate": "30/1",
            "duration": "10.000000",
            "nb_frames": "300",
        }
        hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
    assert len(hashes) == 5
