"""Generate raw v210 bytes, with no MXL, camera, or compressed-video dependency.

The moving 10-bit ramps distinguish Y/U/V, frame order, row pitch and sample
alignment. Requires NumPy. Output has no header; consumers need its dimensions,
frame rate and stride supplied separately.
"""

import argparse
from pathlib import Path

import numpy as np


def frame_stride(width, stride=None):
    if width <= 0 or width % 2:
        raise ValueError("v210 requires a positive even width")
    minimum = ((width * 2 + 2) // 3) * 4
    stride = ((width + 47) // 48) * 128 if stride is None else stride
    if stride < minimum or stride % 4:
        raise ValueError("stride must fit a packed row and be a multiple of four")
    return stride


def sample_planes(width, height, index=0):
    frame_stride(width)
    if height <= 0:
        raise ValueError("height must be positive")
    row = np.arange(height, dtype=np.uint32)[:, None]
    x = np.arange(width, dtype=np.uint32)[None, :]
    cx = x[:, :width // 2]
    y = (x + row * 17 + index * 101) % 1024
    u = (cx * 3 + row * 29 + 341 + index * 59) % 1024
    v = (cx * 7 + row * 13 + 683 + index * 83) % 1024
    return tuple(plane.astype("<u2") for plane in (y, u, v))


def pack_v210(planes, stride=None):
    y, u, v = planes
    height, width = y.shape
    stride = frame_stride(width, stride)
    if u.shape != (height, width // 2) or v.shape != u.shape:
        raise ValueError("expected planar 4:2:2 samples")
    if any(np.any(plane > 1023) or np.any(plane < 0) for plane in planes):
        raise ValueError("samples must be in the 10-bit range")
    samples = np.zeros((height, ((width * 2 + 2) // 3) * 3), dtype=np.uint32)
    active = samples[:, :width * 2]
    active[:, 0::4], active[:, 1::4] = u, y[:, 0::2]
    active[:, 2::4], active[:, 3::4] = v, y[:, 1::2]
    words = samples.reshape(height, -1, 3)
    packed = words[:, :, 0] | (words[:, :, 1] << 10) | (words[:, :, 2] << 20)
    output = np.full((height, stride), 0xa5, dtype=np.uint8)
    payload = packed.astype("<u4").view(np.uint8)
    output[:, :payload.shape[1]] = payload
    return output.tobytes()


def write_fixture(path, width, height, frames, stride=None):
    stride = frame_stride(width, stride)
    if frames <= 0 or height <= 0:
        raise ValueError("frames and height must be positive")
    with Path(path).open("wb") as stream:
        for index in range(frames):
            stream.write(pack_v210(sample_planes(width, height, index), stride))
    return stride


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--frames", type=int, default=60)
    parser.add_argument("--stride", type=int)
    args = parser.parse_args()
    stride = write_fixture(args.output, args.width, args.height, args.frames, args.stride)
    print(f"{args.width}x{args.height}, {args.frames} frames, stride={stride}, "
          f"frame_bytes={stride * args.height}; interpret at 60 fps for a 1080p60 test")
