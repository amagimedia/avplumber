#!/usr/bin/env python3
"""
dmabuf_browser_to_janus.py

Single HTML page (rendered by the dma-browser sidecar) -> avplumber -> Janus
streaming-plugin RTP mountpoint, viewable in the janus-preview page.

The dma-browser sidecar auto-opens the configured browser URL and exports every
composited frame as a DMA-BUF (GPU buffer) over an fdpass unix socket. This
graph imports those frames and streams H.264 to a Janus RTP mountpoint.

Pipeline (public nodes only -- no in-house convert_cuda):

  ipc_dmabuf_source(@drm)                    DRM_PRIME frames from the browser
  assume_video_format(drm_prime / cuda)
  force_fps
  drm_prime_to_cuda(@gpu)                    EGL-import the DMA-BUF honoring the
                                             NVIDIA tiling modifier -> linear CUDA
  filter_video: hwdownload,format=bgra,scale=WxH,format=yuv420p (@gpu -> CPU)
  force_keyframe(1s)                         IDR cadence so WebRTC viewers can join
  enc_video(libx264, baseline, zerolatency)
  bsf: dump_extra=freq=keyframe              repeat SPS/PPS on every keyframe
  mux -> output(rtp)   rtp://JANUS:5004?rtcp_port=5005  (payload_type 96)

This mirrors the detile half of avplumber/examples/from_dmabuf.avplumber (the
proven NVIDIA DMA-BUF consumer) but swaps the in-house convert_cuda for a public
hwdownload, with the Janus RTP output taken from pyplumber/auto_mixer/outputs.py.
The Janus streaming mountpoint must be video H264, payload type 96, on
JANUS_VIDEO_PORT (see janus.plugin.streaming.jcfg).

WHY drm_prime_to_cuda (and not a plain DRM hwdownload): on NVIDIA the browser's
GPU render target is always tiled (block-linear) -- the driver refuses a linear
*renderable* RGBA buffer -- so a plain DRM hwdownload maps the tiled bytes
linearly and yields a sheared/garbage image. drm_prime_to_cuda imports the
DMA-BUF via EGL, which honors the modifier and detiles into a linear CUDA frame.
This needs a consumer built with HAVE_CUDA+HAVE_GL+HAVE_DRM (see
dmabuf-demo/consumer/Dockerfile.cuda). On a GPU that CAN give a linear renderable
buffer (Intel/AMD, dma-browser shim GBM_LINEAR_SHIM_FORCE_LINEAR=1) the DRM-only
build + a plain hwdownload works and this CUDA step can be dropped.

Env overrides:
  SOCKET (default /tmp/dma-page/overlay.sock), WIDTH, HEIGHT, FPS, RENDER_NODE,
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
    IpcDmabufSource,
    AssumeVideoFormat,
    ForceFPS,
    DrmPrimeToCuda,
    FilterVideo,
    ForceKeyFrame,
    EncVideo,
    Bsf,
    Mux,
    Output,
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
# into a linear CUDA frame that a normal hwdownload can then read correctly.
avp.executeCommandsFromString(
    'hwaccel.init { "name": "@drm", "type": "drm", "device": "%s" }' % RENDER_NODE
)
avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')

nodes = [
    IpcDmabufSource(
        {
            "socket": SOCKET,
            "dst": "v_ipc",
            "hwaccel": "@drm",
            "group": "in",
            "name": "Overlay_Receive",
            "auto_restart": "group",
        }
    ),
    AssumeVideoFormat(
        {
            "pixel_format": "drm_prime",
            "real_pixel_format": "cuda",
            "src": "v_ipc",
            "dst": "v_assumed",
            "group": "out",
            "auto_restart": "panic",
        }
    ),
    ForceFPS(
        {
            "fps": "%d/1" % FPS,
            "src": "v_assumed",
            "dst": "v_fps",
            "group": "out",
            "auto_restart": "panic",
        }
    ),
    # Detile: EGL-import the DMA-BUF (honoring the NVIDIA tiling modifier) into a
    # linear CUDA frame. This is the step a DRM-only build cannot do.
    DrmPrimeToCuda(
        {
            "hwaccel": "@gpu",
            "drop_alpha": True,
            "src": "v_fps",
            "dst": "v_cuda",
            "group": "out",
            "name": "Overlay_ToCuda",
            "auto_restart": "group",
        }
    ),
    ForceKeyFrame(
        {
            "interval_sec": "1/1",
            "src": "v_cuda",
            "dst": "v_kf",
            "group": "out",
            "auto_restart": "panic",
        }
    ),
    # Encode the CUDA frame directly with NVENC (it accepts the rgba/bgr0 CUDA
    # surface and does the CSC to h264 internally). We skip hwdownload -- ffmpeg's
    # cuda hwcontext can't init a download for drm_prime_to_cuda's frame -- and we
    # skip the in-house convert_cuda, keeping this to public nodes + system ffmpeg.
    EncVideo(
        {
            "codec": "h264_nvenc",
            "hwaccel": "@gpu",
            "src": "v_kf",
            "dst": "v_enc",
            "group": "out",
            "name": "Video_Encode",
            "auto_restart": "panic",
            "options": {
                "b": BITRATE,
                "maxrate": BITRATE,
                "bufsize": BITRATE,
                "g": FPS,
                "bf": 0,
                "rc": "cbr",
                "preset": "p4",
                "tune": "ll",
                "profile": "baseline",
                "forced-idr": 1,
            },
        }
    ),
    Bsf(
        {
            "bsf": "dump_extra=freq=keyframe",
            "src": "v_enc",
            "dst": "v_enc_hdr",
            "group": "out",
            "auto_restart": "panic",
        }
    ),
    Mux(
        {
            "src": ["v_enc_hdr"],
            "dst": "mux_v",
            "ts_sort_wait": 0,
            "group": "out",
            "auto_restart": "panic",
        }
    ),
    Output(
        {
            "format": OUT_FORMAT,
            "url": OUT_URL,
            "options": (
                {"payload_type": J_PT, "rtpflags": "skip_rtcp", "ssrc": J_SSRC}
                if OUT_FORMAT == "rtp"
                else {}
            ),
            "src": "mux_v",
            "group": "out",
            "name": "Janus_Out",
            "auto_restart": "panic",
        }
    ),
]

for node in nodes:
    avp.addNode(node)

print(
    "[dmabuf_browser_to_janus] %s (%dx%d@%d) -> rtp://%s:%d pt=%d ssrc=0x%08x"
    % (SOCKET, W, H, FPS, J_HOST, J_PORT, J_PT, J_SSRC),
    flush=True,
)

avp.group("out").startNodes()
avp.group("in").startNodes()

while True:
    time.sleep(1)
    avp.heartbeat()
