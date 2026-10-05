"""Source capacity of each instance type the mixer setup supports.

Every value comes from measurements on that instance (docs/capacity.md), derived ones noted
beside them; another machine needs its own measured entry, never a scaled T4 one. Per-rate
tables cover every frame rate the setup offers; a profile's "mode_limits" replace, on one canvas
("bit_depth:chroma"), the limits measured apart there. setup_runtime.py applies the selected profile
and serves it to setup.html. Standard library only: tests/check_setup.cjs loads it with any
python3.
"""

from enum import Enum


class InstanceType(str, Enum):
    TESLA_T4 = "tesla_t4"
    NVIDIA_L4 = "nvidia_l4"


# Not capacities: the range the setup offers per output, and each output's encode until the setup
# sets another: the programs (sdr, hdr) and the clean feed (sdr_clean) at p3, every aux bus at p1.
_BITRATE_KBPS = [2000, 20000]
_ENCODE_DEFAULTS = {
    "sdr": {"preset": "p3", "bitrate_kbps": 6000},
    "hdr": {"preset": "p3", "bitrate_kbps": 8000},
    "sdr_clean": {"preset": "p3", "bitrate_kbps": 6000},
    "aux": {"preset": "p1", "bitrate_kbps": 4000},
}


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
        # Combined SDR/HLG v210 inputs, unpacked on the GPU (the 60 fps limit was measured with four).
        "hlg_v210": 4,
        # NVENC share of one 1920x1080 encode per encoded frame/s (the cost follows the pixel count),
        # by codec and preset: the offered presets are the h264 keys. Extra aux outputs fill what the
        # program, clean feed and aux encodes leave of budget_pct (setup_runtime.extra_aux_limit).
        # Bitrate does not enter: CBR NVENC time is roughly independent of bitrate (to be verified).
        "nvenc": {
            "budget_pct": 80,
            "pct_per_fps": {
                # Measured 2026-10-02, 1080p30, tune ull, CBR, no B-frames: H.264 p3 and p5 on the live
                # show (5 encodes took 32.8% and 70.0%), the others from two real-time 1080p30 test
                # encodes beside it (per encode: H.264 p1 6.0%, p3 6.5%, p5 14.9%; HEVC Main10 p1
                # 4.5%, p3 9.0%, p5 13.1%), anchored to the show's 0.219 for H.264 p3.
                "h264": {"p1": 0.204, "p3": 0.219, "p5": 0.466},
                "hevc": {"p1": 0.153, "p3": 0.305, "p5": 0.442},   # Main10, the HLG program
            },
            "bitrate_kbps": _BITRATE_KBPS,
            "defaults": _ENCODE_DEFAULTS,
        },
    },
    # GCP g2-standard-16: one L4 (Ada: 2 NVENC, 4 NVDEC, 24 GB), 16 vCPU; 1920x1080 inputs.
    # Measured 2026-10-02 beside the full outputs (program at p5, clean feed, Program preview,
    # Multiviewer and three extra aux at p3, four keys), with synthetic inputs of 3-10 Mbit/s (16
    # peak) and grain. A limit keeps NVDEC and the GPU at 90% of the idle show and 10% of the CPU idle.
    # NVDEC fills first, then the browser windows, raw uploads last: an upload raises the NVDEC load
    # of the same decodes (48 at 60 fps: 91% beside none, 95% beside four v210, 98% beside eight).
    # 25 fps keeps the 30 fps numbers; 50 fps scales the 60 fps decodes and uploads by frame rate
    # (not measured).
    InstanceType.NVIDIA_L4: {
        "max_scenes": 256,
        # SDR canvas, keys included: 170 at 30 fps (83 NVDEC, 40 browser, 47 NV12), where the CPU is
        # the limit (46% idle), and 100 at 60 (41, 40, 19; CPU 47% idle) less the decode below.
        "sources": {25: 170, 30: 170, 50: 110, 60: 99},
        # The 10-bit canvases have their own limits (mode_limits), not a share of the SDR total.
        "mode_share": {"8:420": 1.0, "10:420": 1.0, "10:422": 1.0},
        # H.264: 83 decodes at 30 fps ran NVDEC at 88-89%; 41 at 60 fps ran it at 91% beside 15 and
        # beside 19 uploads, so 40.
        "nvdec_decodes": {25: 83, 30: 83, 50: 48, 60: 40},
        # Main10 surfaces use more VRAM: bound their count even when NVDEC has throughput left.
        "nvdec_hdr_decodes": {25: 53, 30: 53, 50: 32, 60: 27},
        "browser_windows": 40,
        "raw_upload_units": {25: 47, 30: 47, 50: 22, 60: 19},   # NV12 = 1, P010 = 2
        # Beside 44 decodes and the browsers at 60 fps (88 inputs): NVDEC 84%, GPU 75%. Eight ran
        # NVDEC at 86% and the GPU at 81%, 98% while cutting.
        "hlg_v210": 4,
        "mode_limits": {
            # Full 256-scene cuts at 25 fps need margin below 30 outputs; 30/50 fps
            # retain the conservative count of the neighboring validated rate.
            "8:420": {"nvenc_max_outputs": {25: 26, 30: 26, 50: 22, 60: 22}},
            # Half the decodes of an HLG canvas are HEVC Main10, which loads NVDEC less than H.264:
            # 48 at 60 fps beside the browsers (88 inputs) ran it at 89-91%, 44 at 78%, 52 at 98%
            # (gate failed). No raw upload fits beside them. The 25/30 fps counts scale the 60 fps
            # ones by frame rate (not measured).
            "10:420": {"nvdec_decodes": {25: 96, 30: 96, 50: 57, 60: 48},
                       "raw_upload_units": {25: 0, 30: 0, 50: 0, 60: 0},
                       # 88 sources and 13 extra AUX passed the 60 fps cut test with zero
                       # misses; 15 missed two deadlines. Presets match the 4:2:2 test below.
                       "nvenc_budget_pct": {60: 75},
                       "nvenc_max_outputs": {25: 20, 30: 20, 50: 18, 60: 18}},
            # NVDEC decodes 4:2:0 only, so a 4:2:2 canvas takes its native 4:2:2 inputs as v210
            # uploads (hlg_v210) and no 4:2:0 upload; four decodes make room for them.
            "10:422": {"nvdec_decodes": {25: 88, 30: 88, 50: 52, 60: 44},
                       "raw_upload_units": {25: 0, 30: 0, 50: 0, 60: 0},
                       # 88 sources, SDR p5 / HLG and AUX p3: 15 extra AUX missed deadlines
                       # on cuts; 12 held 60 fps, zero misses and 34-48 ms encoded cut latency.
                       # This modeled budget reserves the measured margin at 60 fps.
                       "nvenc_budget_pct": {60: 70},
                       "nvenc_max_outputs": {25: 20, 30: 20, 50: 17, 60: 17}},
        },
        # Not measured per preset: the T4's costs over 2.2 (two NVENC engines). The full outputs'
        # totals agree within 5 points: 22% on SDR at 30 fps, 30% at 60, 49% on HLG at 60.
        "nvenc": {
            "budget_pct": 80,
            # Largest validated output count; mode/rate limits may lower this ceiling to
            # reserve memory for input frames. Faster presets cannot bypass either limit.
            "max_outputs": 30,
            "pct_per_fps": {
                "h264": {"p1": 0.093, "p3": 0.100, "p5": 0.212},
                "hevc": {"p1": 0.070, "p3": 0.139, "p5": 0.201},
            },
            "bitrate_kbps": _BITRATE_KBPS,
            "defaults": _ENCODE_DEFAULTS,
        },
    },
}
