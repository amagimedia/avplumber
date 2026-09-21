"""Smooth, seamlessly-looping HLG BT.2020 demo content as packed v210.

Every time dependence is a function of phase = 2*pi*i/N, so frame N wraps
back onto frame 0 with no seam: a drifting BT.2020 colour field, a slow
diagonal luminance wave, and an HDR highlight orbiting the frame. Values are
scene-referred linear, encoded through the BT.2100 HLG OETF and the bt2020nc
matrix, limited range, chroma left-sited. This is demo content, not the
correctness pattern (v210_fixture.hlg_planes keeps the checkpoints).
"""

from pathlib import Path

import numpy as np

from v210_fixture import hlg_oetf, pack_v210

TAU = 2.0 * np.pi


# Saturated BT.2020 primaries and secondaries: at full extent several of these
# fall outside the BT.709 gamut, so on a real wide-gamut HLG display they are
# visibly more saturated than any SDR path can reproduce.
WIDE_GAMUT = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (0, 1, 1), (1, 0, 1), (1, 1, 0))


def frame_planes(width, height, index, frames, source):
    theta = TAU * index / frames
    phi = source * TAU / 4.0
    u = np.broadcast_to((np.arange(width, dtype=np.float64) / (width - 1))[None, :], (height, width))
    v = np.broadcast_to((np.arange(height, dtype=np.float64) / (height - 1))[:, None], (height, width))
    # Near-black ground for real contrast against the highlights.
    r = np.full((height, width), 0.02); g = np.full((height, width), 0.02); b = np.full((height, width), 0.02)

    # Top third: wide-gamut saturated bars, cycling so each tile differs over time.
    top = v < 1 / 3
    bar = ((u * 6 + theta / TAU + source) % 6).astype(int)
    for k, (cr, cg, cb) in enumerate(WIDE_GAMUT):
        m = top & (bar == k)
        r[m] = cr * 0.9; g[m] = cg * 0.9; b[m] = cb * 0.9

    # Middle third: a full 0 -> peak luminance sweep that pans, so the whole
    # dynamic range is on screen at once (deep black through specular white).
    mid = (v >= 1 / 3) & (v < 2 / 3)
    sweep = (u + theta / TAU + phi / TAU) % 1.0
    r[mid] = sweep[mid]; g[mid] = sweep[mid]; b[mid] = sweep[mid]

    # Bottom third stays near black; two tight specular highlights orbit the
    # whole frame and reach HLG peak, popping against the dark ground.
    for k in (0, 1):
        cx = 0.5 + 0.33 * np.cos(theta + phi + k * np.pi)
        cy = 0.5 + 0.33 * np.sin(theta + phi + k * np.pi)
        hi = np.exp(-((u - cx) ** 2 + (v - cy) ** 2) / (2 * 0.06 ** 2))
        r = r + hi; g = g + hi; b = b + hi
    r = np.clip(r, 0.0, 1.0); g = np.clip(g, 0.0, 1.0); b = np.clip(b, 0.0, 1.0)
    rp, gp, bp = hlg_oetf(r), hlg_oetf(g), hlg_oetf(b)
    yp = 0.2627 * rp + 0.6780 * gp + 0.0593 * bp
    y = np.rint(876 * yp + 64).astype("<u2")
    cb = np.rint(896 * (bp - yp) / 1.8814 + 512).astype("<u2")[:, 0::2]
    cr = np.rint(896 * (rp - yp) / 1.4746 + 512).astype("<u2")[:, 0::2]
    return y, cb, cr


def write_hlg(path, width, height, frames, source=0):
    with Path(path).open("wb") as stream:
        for index in range(frames):
            stream.write(pack_v210(frame_planes(width, height, index, frames, source)))
