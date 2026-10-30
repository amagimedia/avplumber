"""Headerless v210 packing and the BT.2100 HLG encoding of BT.2020 RGB, as NumPy arrays.

The demo's HLG patterns (hdr_patterns.py) and the CUDA test fixtures
(tests/cuda/v210_fixture.py) share these.
"""

import numpy as np

HLG_A, HLG_B, HLG_C = 0.17883277, 0.28466892, 0.55991073


def hlg_oetf(e):
    """BT.2100 HLG OETF on scene-linear values in [0, 1]."""
    e = np.asarray(e, dtype=np.float64)
    return np.where(e <= 1 / 12, np.sqrt(3 * e), HLG_A * np.log(12 * np.maximum(e, 1 / 12) - HLG_B) + HLG_C)


def hlg_ycbcr422(r, g, b):
    """Scene-linear BT.2020 planes through the HLG OETF and the bt2020nc matrix: 10-bit
    limited-range Y, Cb and Cr, chroma left-sited (even columns)."""
    rp, gp, bp = hlg_oetf(r), hlg_oetf(g), hlg_oetf(b)
    yp = 0.2627 * rp + 0.6780 * gp + 0.0593 * bp
    y = np.rint(876 * yp + 64).astype("<u2")
    cb = np.rint(896 * (bp - yp) / 1.8814 + 512).astype("<u2")[:, 0::2]
    cr = np.rint(896 * (rp - yp) / 1.4746 + 512).astype("<u2")[:, 0::2]
    return y, cb, cr


def frame_stride(width, stride=None):
    if width <= 0 or width % 2:
        raise ValueError("v210 requires a positive even width")
    minimum = ((width * 2 + 2) // 3) * 4
    stride = ((width + 47) // 48) * 128 if stride is None else stride
    if stride < minimum or stride % 4:
        raise ValueError("stride must fit a packed row and be a multiple of four")
    return stride


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
