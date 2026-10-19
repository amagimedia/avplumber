"""Native regression: input_rec `frame_metadata` selects the keys the decoders stamp on frames.

Run in the built avplumber environment (needs the `ffmpeg` CLI with the native mpeg4 and aac encoders):
    python3 tests/smoke_input_frame_metadata.py

Every decoded frame is handed back to the input for stamping. Without a seek table the input
stamps seven of its eight video keys (`frame_no` needs the table) and both audio keys. Each case
decodes the first frames of the same clip and reports which of the keys arrived.
"""

import pathlib
import subprocess
import sys
import tempfile
import time

from pyplumber import AVPlumber
from pyplumber import node as n

VIDEO_KEYS = ("video_ts", "video_pts", "input_ts", "output_ts", "wallclock_ts", "wallclock", "frame_no", "frame_ts")
AUDIO_KEYS = ("frame_ts", "sample_rate")
FPS = 30
FRAMES = 30
NOPTS = -(1 << 63)


def make_fixture(workdir):
    mp4 = workdir / "clip.mp4"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size=160x120:rate={FPS}",
                    "-f", "lavfi", "-i", "sine=r=48000", "-frames:v", str(FRAMES), "-t", str(FRAMES / FPS),
                    "-c:v", "mpeg4", "-g", "12", "-c:a", "aac", str(mp4)], check=True)
    return mp4


def stamped_keys(url, input_params=None, frames=5):
    """The VIDEO_KEYS on the first `frames` decoded pictures and the AUDIO_KEYS on the first
    `frames` decoded sample buffers, as two sets."""
    avp = AVPlumber()
    errors = []
    avp.on_exception = lambda *error: errors.append(error)
    avp.edges.planCapacity("*", 64)
    for cls, params in (
        (n.InputRec, {"name": "input", "url": str(url), "dst": "packets", **(input_params or {})}),
        (n.Demux, {"name": "demux", "src": "packets", "routing": {"v:0": "video_packets", "a:0": "audio_packets"}}),
        (n.DecVideo, {"name": "decode", "src": "video_packets", "dst": "video"}),
        (n.DecAudio, {"name": "decode_audio", "src": "audio_packets", "dst": "audio"}),
    ):
        avp.addNode(cls({"group": "input", **params}))
    video = avp.getEdge("video", "VideoFrame")
    audio = avp.getEdge("audio", "AudioSamples")
    video_keys, audio_keys = set(), set()
    seen_video = seen_audio = 0
    try:
        avp.group("input").startNodes()
        deadline = time.monotonic() + 20
        while seen_video < frames or seen_audio < frames:
            assert not errors, errors
            assert time.monotonic() < deadline, f"decoded {seen_video} pictures and {seen_audio} sample buffers"
            frame = video.tryGet(10)
            if frame is not None and frame.pts.timestamp != NOPTS:
                seen_video += 1
                video_keys |= {key for key in VIDEO_KEYS if key in frame.metadata}
            samples = audio.tryGet(0)
            if samples is not None and samples.pts.timestamp != NOPTS:
                seen_audio += 1
                audio_keys |= {key for key in AUDIO_KEYS if key in samples.metadata}
    finally:
        avp.shutdown()
    return video_keys, audio_keys


def check(name, got, expected):
    video, audio = got
    expected_video, expected_audio = expected
    assert video == expected_video, f"{name}: video frames carry {sorted(video)}, expected {sorted(expected_video)}"
    assert audio == expected_audio, f"{name}: audio frames carry {sorted(audio)}, expected {sorted(expected_audio)}"
    print(f"PASS {name}: video {sorted(video) or 'none'}, audio {sorted(audio) or 'none'}", flush=True)


def main():
    with tempfile.TemporaryDirectory() as tmp:
        mp4 = make_fixture(pathlib.Path(tmp))
        check("default stamps every key but frame_no", stamped_keys(mp4),
              (set(VIDEO_KEYS) - {"frame_no"}, set(AUDIO_KEYS)))
        check("frame_metadata [frame_ts] stamps that key only", stamped_keys(mp4, {"frame_metadata": ["frame_ts"]}),
              ({"frame_ts"}, {"frame_ts"}))
        check("frame_metadata [] stamps nothing", stamped_keys(mp4, {"frame_metadata": []}),
              (set(), set()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
