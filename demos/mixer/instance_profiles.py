"""Source capacity of each instance type the mixer setup supports.

Every value comes from measurements on that instance (docs/capacity.md), derived ones noted
beside them; another machine needs its own measured entry, never a scaled T4 one. Per-rate
tables cover every frame rate the setup offers. setup_runtime.py applies the selected profile
and serves it to setup.html. Standard library only: tests/check_setup.cjs loads it with any
python3.
"""

from enum import Enum


class InstanceType(str, Enum):
    TESLA_T4 = "tesla_t4"


INSTANCE_PROFILES = {
    # GCP: one Tesla T4 (one Turing NVENC, 15 GB), 16 vCPU; 1920x1080 inputs.
    InstanceType.TESLA_T4: {
        # Unique sources on an SDR canvas, keys included: 110 is the 30 fps baseline and covers 25
        # (fewer frames on every engine), 75 measured at 60. 50 fps scales the 60 fps total by frame
        # rate (90, not measured); the caps below bound it to 82, which passed.
        "sources": {25: 110, 30: 110, 50: 90, 60: 75},
        # Share of that total a "bit_depth:chroma" canvas carries. HLG 4:2:0: 90 of 110 at 30 fps,
        # above which the GPU-side SDR-to-HLG work saturates NVDEC, and 61 of 75 at 60, where 64 ran
        # out of GPU compute. HLG 4:2:2 costs more GPU per source: 55 at 60 fps keeps the margin of
        # 61 at 4:2:0 (61 and 57 ran at GPU p95 92-96%); 81 at 25/30 fps is not measured (90 passed).
        "mode_share": {"8:420": 1.0, "10:420": 0.82, "10:422": 0.74},
        # SDR + HLG decodes: about 1100 decoded frames/s keeps NVDEC near 90% (40 streams at 25 fps
        # measured 81%), at most 40 streams.
        "nvdec_decodes": {25: 40, 30: 36, 50: 22, 60: 18},
        # Five browser workers of eight windows (compose.yaml) at every rate, keys included.
        "browser_windows": 40,
        # NV12 = 1 unit, P010 = 2 (twice the bytes, not measured). 30 and 34 with pinned uploads at
        # 25 and 30 fps; 17 at 60 (1020 frames/s), scaled by frame rate to 20 at 50.
        "raw_upload_units": {25: 30, 30: 34, 50: 20, 60: 17},
        # HLG v210 4:2:2 inputs, unpacked on the GPU (the 60 fps 4:2:2 limit was measured with four).
        "hlg_v210": 4,
        # NVENC share of one 1920x1080 encode per encoded frame/s (the cost follows the pixel count),
        # by codec and preset: the offered presets are the h264 keys. Extra aux outputs fill what the
        # program, clean feed and aux encodes leave of budget_pct (setup_runtime.extra_aux_limit).
        # Bitrate does not enter: CBR NVENC time is roughly independent of bitrate (to be verified).
        "nvenc": {
            "budget_pct": 80,
            "pct_per_fps": {
                # Measured 2026-10-02, 1080p30, tune ull, CBR, no B-frames: p3 0.219 (5 encodes took
                # 32.8%, 9 took 58.2%), p5 0.466 (5 took 70.0%). p1: 0.70 of p3, NVIDIA's Turing
                # table; to be replaced by the T4 measurement.
                "h264": {"p1": 0.153, "p3": 0.219, "p5": 0.466},
                # HEVC Main10 (the HLG program): twice H.264 at the same preset, an assumption; to
                # be replaced by the T4 measurement.
                "hevc": {"p1": 0.306, "p3": 0.438, "p5": 0.932},
            },
            "bitrate_kbps": [2000, 20000],   # the range the setup offers per output
            "default_preset": "p3",          # every output's, until the setup sets another
        },
    },
}
