#!/usr/bin/env python3
"""
dmabuf_browser_to_janus.py

Single HTML page (rendered by the dma-browser sidecar) -> avplumber -> either a
Janus streaming-plugin RTP mountpoint or an in-process duplicate-frame probe.

The dma-browser sidecar auto-opens the configured browser URL and exports every
composited frame as a DMA-BUF (GPU buffer) over an fdpass unix socket. This
graph imports those frames and streams H.264 to a Janus RTP mountpoint.

Set OUTPUT_MODE=mpdecimate to replace the Janus output with:

  CUDA RGB0 -> hwdownload -> yuv444p -> mpdecimate -> null sink

The probe counts frames immediately before and after mpdecimate, then reports
the difference as duplicate frames. The explicit CPU conversion is isolated to
this diagnostic mode; the default Janus path remains GPU-native through NVENC.

Pipeline (public nodes only -- no in-house convert_cuda):

  ipc_dmabuf_source(@drm)                    DRM_PRIME frames from the browser
  assume_video_format(drm_prime / RGB0)
  drm_prime_to_cuda(@gpu)                    EGL-import the DMA-BUF honoring the
                                             NVIDIA tiling modifier -> RGB0 CUDA
  setpts/settb                               preserve the browser's shared
                                             monotonic clock at 1/FPS precision
  force_keyframe(1s)                         IDR cadence so WebRTC viewers can join
  enc_video(h264_nvenc, baseline, low latency)
  bsf: dump_extra=freq=keyframe              repeat SPS/PPS on every keyframe
  mux -> output(rtp)   rtp://JANUS:5004?rtcp_port=5005  (payload_type 96)

This mirrors the detile half of avplumber/examples/from_dmabuf.avplumber (the
proven NVIDIA DMA-BUF consumer), with the Janus RTP output constructed directly
from public node wrappers.
The Janus streaming mountpoint must be video H264, payload type 96, on
JANUS_VIDEO_PORT (see janus.plugin.streaming.jcfg).

WHY drm_prime_to_cuda (and not a plain DRM hwdownload): on NVIDIA the browser's
GPU render target is always tiled (block-linear) -- the driver refuses a linear
*renderable* RGBA buffer -- so a plain DRM hwdownload maps the tiled bytes
linearly and yields a sheared/garbage image. drm_prime_to_cuda imports the
DMA-BUF via EGL, which honors the modifier and detiles into a linear CUDA frame.
This needs a consumer built with HAVE_CUDA+HAVE_GL+HAVE_DRM (see
demos/dmabuf-browser/consumer/Dockerfile.cuda).

Env overrides:
  SOCKET (default /tmp/dma-page/overlay.sock), WIDTH, HEIGHT, FPS, RENDER_NODE,
  OUTPUT_MODE (janus or mpdecimate), MPDECIMATE_DURATION_SEC,
  MPDECIMATE_REPORT_INTERVAL_SEC, MPDECIMATE_FILTER,
  JANUS_HOST, JANUS_VIDEO_PORT, JANUS_VIDEO_RTCP_PORT, JANUS_VIDEO_PT,
  JANUS_VIDEO_SSRC, VIDEO_BITRATE, PYPLUMBER_PATH.
"""

import os
import sys
import time

sys.path.insert(
    0,
    os.environ.get(
        "PYPLUMBER_PATH", os.path.join(os.path.dirname(__file__), "..", "..")
    ),
)

import pyplumber
from pyplumber.node import (
    AssumeVideoFormat,
    FilterVideo,
)

from dmabuf_browser_common import (
    CountInputFrames,
    CountOutputFrames,
    FrameCounts,
    duplicate_stats,
    make_dmabuf_cuda_input_nodes,
    make_janus_h264_output_nodes,
)

SOCKET = os.environ.get("SOCKET", "/tmp/dma-page/overlay.sock")
W = int(os.environ.get("WIDTH", "1920"))
H = int(os.environ.get("HEIGHT", "1080"))
FPS = int(os.environ.get("FPS", "30"))
RENDER_NODE = os.environ.get("RENDER_NODE", "/dev/dri/renderD128")

J_HOST = os.environ.get("JANUS_HOST", "127.0.0.1")
J_PORT = int(os.environ.get("JANUS_VIDEO_PORT", "5004"))
J_RTCP = int(os.environ.get("JANUS_VIDEO_RTCP_PORT", str(J_PORT + 1)))
J_PT = int(os.environ.get("JANUS_VIDEO_PT", "96"))
J_SSRC = int(os.environ.get("JANUS_VIDEO_SSRC", str(0x41565001)))
BITRATE = os.environ.get("VIDEO_BITRATE", "4000k")
OUTPUT_MODE = os.environ.get("OUTPUT_MODE", "janus").strip().lower()
MPDECIMATE_DURATION_SEC = float(os.environ.get("MPDECIMATE_DURATION_SEC", "30"))
MPDECIMATE_REPORT_INTERVAL_SEC = float(
    os.environ.get("MPDECIMATE_REPORT_INTERVAL_SEC", "5")
)
MPDECIMATE_FILTER = os.environ.get("MPDECIMATE_FILTER", "mpdecimate").strip()

if OUTPUT_MODE not in {"janus", "mpdecimate"}:
    sys.exit("OUTPUT_MODE must be 'janus' or 'mpdecimate'")
if MPDECIMATE_DURATION_SEC < 0:
    sys.exit("MPDECIMATE_DURATION_SEC must be non-negative")
if MPDECIMATE_REPORT_INTERVAL_SEC <= 0:
    sys.exit("MPDECIMATE_REPORT_INTERVAL_SEC must be positive")
if not MPDECIMATE_FILTER:
    sys.exit("MPDECIMATE_FILTER must not be empty")

# Output is Janus RTP by default. Override to dump a file for one-frame
# verification of the detile, e.g. OUTPUT_FORMAT=mpegts OUTPUT_URL=/out/test.ts
OUT_FORMAT = os.environ.get("OUTPUT_FORMAT", "rtp")
OUT_URL = os.environ.get(
    "OUTPUT_URL",
    "rtp://%s:%d?pkt_size=1200&rtcp_port=%d" % (J_HOST, J_PORT, J_RTCP),
)

# Wait (up to 2 min) for the dma-browser sidecar to create the fdpass socket.
for _ in range(240):
    if os.path.exists(SOCKET):
        break
    time.sleep(0.5)
else:
    sys.exit(f"[dmabuf_browser_to_janus] fdpass socket {SOCKET} never appeared")

avp = pyplumber.AVPlumber()

# @drm imports the incoming DMA-BUF as a DRM_PRIME frame; @gpu (CUDA) is where
# the tiled buffer is detiled. On NVIDIA the browser render target is always
# tiled (block-linear) -- a plain DRM hwdownload reads sheared garbage -- so
# drm_prime_to_cuda imports the DMA-BUF via EGL *honoring the NVIDIA modifier*
# into an RGB0 CUDA frame that NVENC can encode directly.
avp.executeCommandsFromString(
    'hwaccel.init { "name": "@drm", "type": "drm", "device": "%s" }' % RENDER_NODE
)
avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')


def print_mpdecimate_stats(counts, final=False):
    input_frames, output_frames, duplicates, duplicate_pct = duplicate_stats(counts)
    label = "final" if final else "progress"
    print(
        "[dmabuf_mpdecimate] %s input=%d unique=%d duplicates=%d duplicate_pct=%.3f"
        % (label, input_frames, output_frames, duplicates, duplicate_pct),
        flush=True,
    )


nodes, cuda_edge = make_dmabuf_cuda_input_nodes(
    prefix="v",
    socket=SOCKET,
    width=W,
    height=H,
    fps=FPS,
    drm_hwaccel="@drm",
    cuda_hwaccel="@gpu",
    source_group="in",
    processing_group="out",
)

if OUTPUT_MODE == "janus":
    nodes.extend(
        make_janus_h264_output_nodes(
            prefix="v_janus",
            src=cuda_edge,
            group="out",
            cuda_hwaccel="@gpu",
            fps=FPS,
            bitrate=BITRATE,
            output_format=OUT_FORMAT,
            output_url=OUT_URL,
            output_options=(
                {"payload_type": J_PT, "rtpflags": "skip_rtcp", "ssrc": J_SSRC}
                if OUT_FORMAT == "rtp"
                else {}
            ),
        )
    )
    mpdecimate_counts = None
else:
    mpdecimate_counts = FrameCounts()
    nodes.extend(
        [
            AssumeVideoFormat(
                {
                    "width": W,
                    "height": H,
                    "pixel_format": "cuda",
                    "real_pixel_format": "rgb0",
                    "src": cuda_edge,
                    "dst": "v_mp_cuda",
                    "group": "out",
                    "auto_restart": "panic",
                }
            ),
            # mpdecimate is a CPU libavfilter. Keep the download explicit and local to
            # this diagnostic mode, then give it planar 4:4:4 frames for comparison.
            FilterVideo(
                {
                    "graph": "hwdownload,format=rgb0,format=yuv444p",
                    "hwaccel": "@gpu",
                    "src": "v_mp_cuda",
                    "dst": "v_mp_444",
                    "dst_width": W,
                    "dst_height": H,
                    "dst_pixel_format": "yuv444p",
                    "dst_frame_rate": "%d/1" % FPS,
                    "group": "out",
                    "name": "Duplicate_Download_444",
                    "auto_restart": "panic",
                }
            ),
            CountInputFrames(
                {
                    "src": "v_mp_444",
                    "dst": "v_mp_counted",
                    "group": "out",
                    "name": "Duplicate_Input_Count",
                },
                mpdecimate_counts,
            ),
            FilterVideo(
                {
                    "graph": MPDECIMATE_FILTER,
                    "src": "v_mp_counted",
                    "dst": "v_mp_unique",
                    "dst_width": W,
                    "dst_height": H,
                    "dst_pixel_format": "yuv444p",
                    "dst_frame_rate": "%d/1" % FPS,
                    "group": "out",
                    "name": "Duplicate_Mpdecimate",
                    "auto_restart": "panic",
                }
            ),
            CountOutputFrames(
                {
                    "src": "v_mp_unique",
                    "group": "out",
                    "name": "Duplicate_Output_Count",
                },
                mpdecimate_counts,
            ),
        ]
    )

for node in nodes:
    avp.addNode(node)

if OUTPUT_MODE == "janus":
    print(
        "[dmabuf_browser_to_janus] %s (%dx%d@%d) -> rtp://%s:%d pt=%d ssrc=0x%08x"
        % (SOCKET, W, H, FPS, J_HOST, J_PORT, J_PT, J_SSRC),
        flush=True,
    )
else:
    print(
        "[dmabuf_mpdecimate] %s (%dx%d@%d) filter=%s duration_sec=%.3f"
        % (SOCKET, W, H, FPS, MPDECIMATE_FILTER, MPDECIMATE_DURATION_SEC),
        flush=True,
    )

avp.group("out").startNodes()
avp.group("in").startNodes()

started = time.monotonic()
next_report = started + MPDECIMATE_REPORT_INTERVAL_SEC
try:
    while True:
        time.sleep(1)
        avp.heartbeat()
        if OUTPUT_MODE != "mpdecimate":
            continue
        now = time.monotonic()
        if now >= next_report:
            print_mpdecimate_stats(mpdecimate_counts)
            next_report = now + MPDECIMATE_REPORT_INTERVAL_SEC
        if MPDECIMATE_DURATION_SEC and now - started >= MPDECIMATE_DURATION_SEC:
            break
except KeyboardInterrupt:
    pass
finally:
    if OUTPUT_MODE == "mpdecimate":
        avp.group("in").stopNodes()
        time.sleep(1)
        avp.group("out").stopNodes()
        print_mpdecimate_stats(mpdecimate_counts, final=True)
