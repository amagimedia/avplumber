"""Generate raw v210 bytes, with no MXL, camera, or compressed-video dependency.

The moving 10-bit ramps distinguish Y/U/V, frame order, row pitch and sample
alignment. Requires NumPy. Output has no header; consumers need its dimensions,
frame rate and stride supplied separately.
"""

from pathlib import Path

import numpy as np

HLG_A, HLG_B, HLG_C = 0.17883277, 0.28466892, 0.55991073


def hlg_oetf(e):
    """BT.2100 HLG OETF on scene-linear values in [0, 1]."""
    e = np.asarray(e, dtype=np.float64)
    return np.where(e <= 1 / 12, np.sqrt(3 * e), HLG_A * np.log(12 * np.maximum(e, 1 / 12) - HLG_B) + HLG_C)


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


def hlg_planes(width, height, index=0):
    """Native HLG BT.2020 family: moving scene-linear ramps through the HLG
    OETF and the bt2020nc matrix, limited range, chroma left-sited (even
    columns). Rows 0-1 carry transfer checkpoints E = 0, 1/12, 1 as 12-column
    gray patches whose expected codes are Y 64, 502, 940 with chroma 512."""
    frame_stride(width)
    row = np.arange(height, dtype=np.float64)[:, None]
    x = np.arange(width, dtype=np.float64)[None, :]
    g = ((x + row * 3 + index * 11) % width) / (width - 1)
    r = ((x * 2 + row * 7 + index * 5) % width) / (width - 1)
    b = ((x * 5 + row * 11 + index * 17) % width) / (width - 1)
    for i, e in enumerate((0.0, 1 / 12, 1.0)):
        r[:2, 12 * i:12 * (i + 1)] = g[:2, 12 * i:12 * (i + 1)] = b[:2, 12 * i:12 * (i + 1)] = e
    rp, gp, bp = hlg_oetf(r), hlg_oetf(g), hlg_oetf(b)
    yp = 0.2627 * rp + 0.6780 * gp + 0.0593 * bp
    y = np.rint(876 * yp + 64).astype("<u2")
    u = np.rint(896 * (bp - yp) / 1.8814 + 512).astype("<u2")[:, 0::2]
    v = np.rint(896 * (rp - yp) / 1.4746 + 512).astype("<u2")[:, 0::2]
    return y, u, v


def sdr8_planes(width, height, index=0):
    """SDR-promoted family: BT.709 limited-range 8-bit patterns packed as
    sample8 << 2, so every 10-bit code is a multiple of four (16 -> 64,
    235 -> 940, neutral 128 -> 512). 8-bit source precision, not HDR."""
    frame_stride(width)
    row = np.arange(height, dtype=np.uint32)[:, None]
    x = np.arange(width, dtype=np.uint32)[None, :]
    cx = x[:, :width // 2]
    y = 16 + (x + row * 17 + index * 101) % 220
    u = 16 + (cx * 3 + row * 29 + index * 59) % 225
    v = 16 + (cx * 7 + row * 13 + index * 83) % 225
    return tuple((plane << 2).astype("<u2") for plane in (y, u, v))


FAMILIES = {"ramp": sample_planes, "hlg": hlg_planes, "sdr8": sdr8_planes}

# Stream color contract per family, in avplumber/FFmpeg option spelling.
_BT709 = {"color_range": "tv", "color_primaries": "bt709", "color_trc": "bt709",
          "colorspace": "bt709", "chroma_location": "left"}
COLOR = {
    "ramp": _BT709,
    "sdr8": _BT709,
    "hlg": {"color_range": "tv", "color_primaries": "bt2020", "color_trc": "arib-std-b67",
            "colorspace": "bt2020nc", "chroma_location": "left"},
}


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


def write_fixture(path, width, height, frames, stride=None, family="ramp", source=0):
    stride = frame_stride(width, stride)
    if frames <= 0 or height <= 0:
        raise ValueError("frames and height must be positive")
    planes = FAMILIES[family]
    with Path(path).open("wb") as stream:
        for index in range(frames):
            stream.write(pack_v210(planes(width, height, index + source * 1000), stride))
    return stride
