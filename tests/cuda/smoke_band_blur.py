"""Synthetic upload/download is intentional: compare filter pixels on the GPU."""
import subprocess
import sys

import numpy as np

width, height = 64, 48
y = np.full((height, width), 160, np.uint8)
uv = np.full((height // 2, width), 128, np.uint8)


def run(luma, options):
    raw = luma.tobytes() + uv.tobytes()
    cmd = [sys.argv[1], "-v", "error", "-init_hw_device", "cuda=gpu:0",
           "-filter_hw_device", "gpu", "-f", "rawvideo", "-pixel_format", "nv12",
           "-video_size", f"{width}x{height}", "-i", "pipe:0", "-frames:v", "1",
           "-vf", f"hwupload,band_blur_cuda={options},hwdownload,format=nv12",
           "-c:v", "rawvideo", "-f", "rawvideo", "pipe:1"]
    result = subprocess.run(cmd, input=raw, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stderr.decode()
    assert len(result.stdout) == len(raw)
    return np.frombuffer(result.stdout[:width * height], np.uint8).reshape(height, width), result.stdout[width * height:]


band = "y_start=.25:y_end=.75:radius=6"
out, chroma = run(y, band + ":blur_start=0:blur_end=0:luma_start=1:luma_end=1")
assert np.array_equal(out, y) and chroma == uv.tobytes(), "identity changed pixels"
for gradient in ("linear", "smoothstep"):
    out, chroma = run(y, band + f":blur_start=1:blur_end=1:luma_start=1:luma_end=.5:gradient={gradient}")
    assert np.array_equal(out[:12], y[:12]) and np.array_equal(out[36:], y[36:])
    assert np.all(out[12] == 160) and np.all(out[35] == 80)
    assert np.all(np.diff(out[12:36, 0].astype(int)) <= 0)
    assert chroma == uv.tobytes(), "neutral chroma changed"
checker = ((np.indices(y.shape).sum(axis=0) % 2) * 200 + 20).astype(np.uint8)
for radius in (1, 12):
    out, chroma = run(checker, f"y_start=.25:y_end=.75:radius={radius}:blur_start=1:blur_end=1:luma_start=1:luma_end=1")
    assert np.array_equal(out[:12], checker[:12]) and np.array_equal(out[36:], checker[36:])
    assert np.all(out[12:36] == 120), "checkerboard blur differs from reference"
    assert chroma == uv.tobytes()
print("PASS: 5 band-blur pixel cases (identity, linear/smoothstep gradients, radius 1/12)")
