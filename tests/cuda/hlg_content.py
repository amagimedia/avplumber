"""Smooth, seamlessly-looping HLG BT.2020 demo content as packed v210.

Every time dependence is a function of phase = 2*pi*i/N, so frame N wraps
back onto frame 0 with no seam: a drifting BT.2020 colour field, a slow
diagonal luminance wave, and an HDR highlight orbiting the frame. Values are
scene-referred linear, encoded through the BT.2100 HLG OETF and the bt2020nc
matrix, limited range, chroma left-sited. This is demo content, not the
correctness pattern (v210_fixture.hlg_planes keeps the checkpoints).
"""

import argparse
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


CMP_LO, CMP_HI = 440, 568   # narrow 128-code luma window before the stretch


def comparison_planes(width, height, index, frames, source=0):
    """A shallow neutral-grey luma ramp in a narrow code window (CMP_LO..CMP_HI),
    panning slowly. On its own the ramp is nearly flat; the 8-bit vs 10-bit
    difference lives in the low bits and is exposed by stretch_luma()."""
    u = np.broadcast_to((np.arange(width, dtype=np.float64) / (width - 1))[None, :], (height, width))
    t = (u + index / frames) % 1.0                     # panning, seamless
    y = np.rint(CMP_LO + t * (CMP_HI - CMP_LO)).astype("<u2")
    cb = np.full((height, width // 2), 512, "<u2")
    cr = np.full((height, width // 2), 512, "<u2")
    return y, cb, cr


def stretch_luma(planes):
    """Map the narrow CMP_LO..CMP_HI window onto 64..940. Applied AFTER any 8-bit
    quantisation, it amplifies the 8-bit half's coarse steps into wide, obvious
    bands while the 10-bit half stays fine -- so the difference survives even an
    8-bit transport and display."""
    y, cb, cr = planes
    scaled = (y.astype(np.float64) - CMP_LO) * (940 - 64) / (CMP_HI - CMP_LO) + 64
    return np.clip(np.rint(scaled), 0, 1023).astype("<u2"), cb, cr


def quantize8(planes):
    """Drop to 8-bit precision inside the 10-bit container: round each code to
    the nearest multiple of four. Same pattern, but it bands like 8-bit."""
    return tuple(np.clip((np.rint(p / 4.0) * 4), 0, 1023).astype("<u2") for p in planes)


def write(path, planes_fn, width, height, frames, source, eight_bit=False):
    with Path(path).open("wb") as stream:
        for i in range(frames):
            planes = planes_fn(width, height, i, frames, source)
            if eight_bit:
                planes = quantize8(planes)
            stream.write(pack_v210(planes))
    print(f"wrote {path} ({frames} frames, seamless {frames/60:.1f}s loop"
          f"{', 8-bit quantised' if eight_bit else ''})", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("outdir", type=Path)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--frames", type=int, default=300)
    p.add_argument("--sources", type=int, default=4)
    p.add_argument("--pattern", choices=("motion", "comparison"), default="motion")
    args = p.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.pattern == "comparison":
        # One shallow ramp, written 10-bit and 8-bit, each stretched after the
        # quantisation so the bit-depth difference is visible on any display.
        base = f"{args.width}x{args.height}_{args.frames}f"
        for name, eight in (("hlgcompare10", False), ("hlgcompare8", True)):
            path = args.outdir / f"{name}_{base}.v210"
            with path.open("wb") as stream:
                for i in range(args.frames):
                    planes = comparison_planes(args.width, args.height, i, args.frames)
                    if eight:
                        planes = quantize8(planes)
                    stream.write(pack_v210(stretch_luma(planes)))
            print(f"wrote {path} ({'8-bit' if eight else '10-bit'}, stretched)", flush=True)
        return
    for s in range(args.sources):
        path = args.outdir / f"hlgmotion{s}_{args.width}x{args.height}_{args.frames}f.v210"
        write(path, frame_planes, args.width, args.height, args.frames, s)


if __name__ == "__main__":
    main()
