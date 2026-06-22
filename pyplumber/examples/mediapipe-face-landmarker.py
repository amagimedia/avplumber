#!/usr/bin/env python3
"""MediaPipe Face Landmarker — GPU zero-copy via avplumber C++ node.

Pipeline (all GPU, zero-copy):
  InputRec → Demux → DecVideo(cuda) → Realtime → FilterVideo(scale_cuda)
  → Split → [v_for_egl] CudaToEglImage → MediaPipeFaceMeshGpu → face_metadata
  → JoinMetadata(v_for_draw + face_metadata)
  → FaceMeshOverlayNode [CuPy zero-copy on NV12 luma]
  → AssumeVideoFormat → EncVideo(h264_nvenc) → Mux → Output(rtp Janus)

Requires avplumber built with HAVE_MEDIAPIPE=1 (Dockerfile.mediapipe).

Environment variables
─────────────────────
  AVP_INPUT               Input URL/file (default: talkshow/Genaro_norm.ts)
  AVP_OUTPUT              RTP output URL (default: rtp://127.0.0.1:5004)
  AVP_OUTPUT_FORMAT       Output container format (default: rtp)
  AVP_WIDTH               Processing + output width (default: 960)
  AVP_HEIGHT              Processing + output height (default: 540)
  AVP_USE_REALTIME        Enable real-time sync (default: true)
  AVP_LOOP                Loop input file (default: true)
  AVP_FACE_MAX_FACES      Max faces to detect (default: 4)
  AVP_FACE_WITH_ATTENTION Enable attention landmarks (default: true)
  AVP_FACE_INFER_EVERY_N  Run inference every N frames (default: 1)
  AVP_FACE_RESOURCE_ROOT  MediaPipe resource root (default: auto from build)
  AVP_FACE_METADATA_KEY   Landmark metadata key (default: face_landmarks_v1)
  AVP_FACE_DOT_RADIUS     Landmark dot radius in pixels (default: 2)
  AVP_FACE_DOT_LUMA       Luma value for dots, 0-255 (default: 235)
  AVP_WEBUI_API           Web UI endpoint, e.g. http://127.0.0.1:22222
  AVP_REMOTE_CONTROL_PORT Control server TCP port (required with AVP_WEBUI_API)
  AVP_INSTANCE_NAME       Web UI instance label (default: mediapipe-face-landmarker)
  AVP_LOGFILE             Write avplumber log to this file
"""

from __future__ import annotations

import collections
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pyplumber
from pyplumber.node import (
    AssumeVideoFormat,
    CudaToEglImage,
    DecVideo,
    Demux,
    EncVideo,
    FilterVideo,
    InputRec,
    MediaPipeFaceMeshGpu,
    Mux,
    Output,
    PythonNode,
    Realtime,
    Split,
)


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


# ── Config ────────────────────────────────────────────────────────────────────
INPUT_URL  = os.environ.get("AVP_INPUT", "/home/fedora/test-content/talkshow/Genaro_norm.ts")
OUTPUT_URL = os.environ.get("AVP_OUTPUT", "rtp://127.0.0.1:5004?pkt_size=1316&ssrc=0x41565001")
OUTPUT_FORMAT = os.environ.get("AVP_OUTPUT_FORMAT", "rtp")
WIDTH      = int(os.environ.get("AVP_WIDTH",  "960"))
HEIGHT     = int(os.environ.get("AVP_HEIGHT", "540"))
USE_REALTIME   = _env_bool("AVP_USE_REALTIME", True)
LOOP_INPUT     = _env_bool("AVP_LOOP", True)
AUTO_RESTART   = "group" if USE_REALTIME else "off"

METADATA_KEY   = os.environ.get("AVP_FACE_METADATA_KEY", "face_landmarks_v1")
MAX_FACES      = int(os.environ.get("AVP_FACE_MAX_FACES", "4"))
WITH_ATTENTION = _env_bool("AVP_FACE_WITH_ATTENTION", True)
INFER_EVERY_N  = int(os.environ.get("AVP_FACE_INFER_EVERY_N", "1"))
RESOURCE_ROOT  = os.environ.get("AVP_FACE_RESOURCE_ROOT", "")
DOT_RADIUS     = int(os.environ.get("AVP_FACE_DOT_RADIUS", "2"))
DOT_LUMA       = int(os.environ.get("AVP_FACE_DOT_LUMA", "235"))

WEBUI_API           = os.environ.get("AVP_WEBUI_API", "")
INSTANCE_NAME       = os.environ.get("AVP_INSTANCE_NAME", "mediapipe-face-landmarker")
LOGFILE             = os.environ.get("AVP_LOGFILE", "")
REMOTE_CONTROL_PORT = int(os.environ.get("AVP_REMOTE_CONTROL_PORT", "0") or "0")

if WEBUI_API and not REMOTE_CONTROL_PORT:
    raise SystemExit("AVP_WEBUI_API requires AVP_REMOTE_CONTROL_PORT.")


# ── Zero-copy CUDA plane view ─────────────────────────────────────────────────
class _CudaPlaneView:
    def __init__(self, ptr: int, shape: tuple, strides: tuple, typestr: str = "|u1"):
        self._iface = {"shape": shape, "strides": strides,
                       "typestr": typestr, "data": (ptr, False), "version": 3}

    @property
    def __cuda_array_interface__(self):
        return self._iface


# ── Face mesh overlay Python node ─────────────────────────────────────────────
class FaceMeshOverlayNode(PythonNode):
    """Read VideoFrame + MetadataFrame from separate edges; draw face mesh dots zero-copy."""

    def __init__(self, args: dict):
        # siso VideoFrame in/out; grabs MetadataFrame edge manually in _avplumber_initialized
        super().__init__({"data_type": "VideoFrame"} | args)
        self._metadata_key = str(self.parameters.get("metadata_key", METADATA_KEY))
        self._metadata_edge_name = str(self.parameters.get("metadata_src", "face_md"))
        self._radius = int(self.parameters.get("dot_radius", DOT_RADIUS))
        self._luma   = int(self.parameters.get("dot_luma",   DOT_LUMA))
        self._cp = None
        self._md_src = None
        self._fps_times: collections.deque = collections.deque(maxlen=30)
        self._frame_count = 0

    def _avplumber_initialized(self):
        super()._avplumber_initialized()
        self._md_src = self._avplumber.getEdge(self._metadata_edge_name, "MetadataFrame")
        assert self._md_src is not None, f"MetadataFrame edge {self._metadata_edge_name!r} not found"

    def process(self):
        frame = self._src.get()
        if frame is None:
            return

        now = time.monotonic()
        self._fps_times.append(now)
        self._frame_count += 1
        fps = (len(self._fps_times) - 1) / (self._fps_times[-1] - self._fps_times[0]) \
            if len(self._fps_times) >= 2 else 0.0

        try:
            # Blocking get — face_md and v_for_draw are both fed from Split at the same rate.
            md_frame = self._md_src.get()
            if self._frame_count <= 5 and md_frame is not None:
                print(
                    f"[FaceMeshOverlay] pts_check frame_pts={frame.pts.timestamp}"
                    f" md_pts={md_frame.pts.timestamp}",
                    flush=True,
                )

            if md_frame is not None:
                md = md_frame.metadata
                md_dict = md.as_dict if hasattr(md, 'as_dict') else md
                raw = md_dict.get(self._metadata_key)
                if raw is not None:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                    status = data.get("status", "?")
                    faces = data.get("faces", [])
                    face_info = ", ".join(
                        f"face{i}:{len(f.get('landmarks', []))}lm"
                        for i, f in enumerate(faces)
                    ) or "no_faces"
                    print(
                        f"[FaceMeshOverlay] frame={self._frame_count}"
                        f" pts={frame.pts.timestamp}"
                        f" fps={fps:.1f}"
                        f" status={status}"
                        f" faces={len(faces)}"
                        f" [{face_info}]",
                        flush=True,
                    )
                    if status == "ok":
                        self._draw(frame, data)
            else:
                print(
                    f"[FaceMeshOverlay] frame={self._frame_count}"
                    f" pts={frame.pts.timestamp}"
                    f" fps={fps:.1f}"
                    f" no_metadata",
                    flush=True,
                )
        except Exception as e:
            print(f"[FaceMeshOverlay] frame={self._frame_count} error: {e}", flush=True)
            self._cp = None  # reset CuPy so next frame retries import
        finally:
            self._dst.enqueue(frame)  # always forward frame, even on draw error

    def _draw(self, frame, data: dict) -> None:
        if self._cp is None:
            import cupy as cp
            self._cp = cp
        cp = self._cp
        w = int(frame.width)
        h = int(frame.height)
        stride = int(frame.linesize[0])
        ptr    = int(frame.data_ptr[0])
        if ptr == 0 or stride == 0:
            return
        view    = _CudaPlaneView(ptr=ptr, shape=(h, stride), strides=(stride, 1))
        y_plane = cp.asarray(view)
        r = self._radius
        luma = self._luma
        for face in data.get("faces", []):
            for lm in face.get("landmarks", []):
                px = int(lm[0] * w)
                py = int(lm[1] * h)
                y0 = max(0, py - r);  y1 = min(h, py + r + 1)
                x0 = max(0, px - r);  x1 = min(w, px + r + 1)
                if y1 > y0 and x1 > x0:
                    y_plane[y0:y1, x0:x1] = luma
        cp.cuda.Stream.null.synchronize()


# ── Build and run graph ───────────────────────────────────────────────────────
def main() -> None:
    print(f"Input:      {INPUT_URL}")
    print(f"Output:     {OUTPUT_URL} ({OUTPUT_FORMAT})")
    print(f"Resolution: {WIDTH}x{HEIGHT}  realtime={USE_REALTIME}  loop={LOOP_INPUT}")
    print(f"Inference:  GPU (MediaPipeFaceMeshGpu C++ node, zero-copy EGL)")

    avp = pyplumber.AVPlumber()
    avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')
    avp.edges.planCapacity("*", 8)

    if LOGFILE:
        avp.setLogFile(LOGFILE)
    if REMOTE_CONTROL_PORT:
        avp.enableControlServer(REMOTE_CONTROL_PORT)
    if WEBUI_API:
        avp.registerWithWebUI(WEBUI_API, INSTANCE_NAME, LOGFILE)

    nodes = [
        InputRec({
            "name": "Input", "url": INPUT_URL, "dst": "in_mux",
            "group": "in", "initial_timeout": 20, "timeout": 10,
            "loop": LOOP_INPUT, "auto_restart": AUTO_RESTART,
        }),
        Demux({
            "src": "in_mux", "routing": {"?v:0": "v_pkt"},
            "wait_for_keyframe": False, "group": "in", "auto_restart": AUTO_RESTART,
        }),
        DecVideo({
            "src": "v_pkt", "dst": "v_dec", "name": "Video_Dec",
            "pixel_format": "?cuda", "hwaccel": "@gpu",
            "codec_map": {"h264": "h264_cuvid", "hevc": "hevc_cuvid"},
            "hwaccel_only_for_codecs": ["h264", "hevc"],
            "group": "in", "auto_restart": AUTO_RESTART,
        }),
        Realtime({
            "src": "v_dec", "dst": "v_rt", "set_pts": True,
            "group": "proc", "auto_restart": AUTO_RESTART,
        }) if USE_REALTIME else None,
        FilterVideo({
            "src": "v_rt" if USE_REALTIME else "v_dec",
            "dst": "v_scaled",
            "graph": f"scale_cuda=w={WIDTH}:h={HEIGHT}",
            "dst_width": WIDTH, "dst_height": HEIGHT,
            "dst_pixel_format": "cuda", "hwaccel": "@gpu",
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        Split({
            "src": "v_scaled", "dst": ["v_for_egl", "v_for_draw"],
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        # MediaPipe GPU path
        CudaToEglImage({
            "src": "v_for_egl", "dst": "v_egl",
            "pool_id": "mediapipe_face_mesh", "pool_size": 8, "pool_max_size": 16,
            "sync": True,
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        MediaPipeFaceMeshGpu({
            "src": "v_egl", "dst": "face_md",
            "metadata_key": METADATA_KEY,
            **({"resource_root": RESOURCE_ROOT} if RESOURCE_ROOT else {}),
            "max_faces": MAX_FACES,
            "with_attention": WITH_ATTENTION,
            "infer_every_n": INFER_EVERY_N,
            "emit_dropped_metadata": True,
            "debug_log_every_n": int(os.environ.get("AVP_FACE_DEBUG_LOG_EVERY_N", "1")),
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        # Draw zero-copy on GPU luma plane; reads MetadataFrame edge face_md directly
        FaceMeshOverlayNode({
            "src": "v_for_draw", "dst": "v_overlay",
            "name": "FaceMesh_Overlay",
            "metadata_key": METADATA_KEY,
            "metadata_src": "face_md",
            "dot_radius": DOT_RADIUS, "dot_luma": DOT_LUMA,
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        AssumeVideoFormat({
            "src": "v_overlay", "dst": "v_preenc",
            "width": WIDTH, "height": HEIGHT,
            "pixel_format": "cuda", "real_pixel_format": "nv12",
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        EncVideo({
            "src": "v_preenc", "dst": "v_enc", "name": "Video_Enc",
            "codec": "h264_nvenc", "hwaccel": "@gpu",
            "options": {"b": "4000k", "maxrate": "6000k", "bufsize": "8000k",
                        "preset": "p4", "profile": "high", "bf": 0},
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        Mux({"src": ["v_enc"], "dst": "muxed", "group": "proc", "auto_restart": AUTO_RESTART}),
        Output({
            "src": "muxed", "format": OUTPUT_FORMAT, "url": OUTPUT_URL,
            "name": "Output", "group": "proc", "auto_restart": AUTO_RESTART,
        }),
    ]

    for node in nodes:
        if node is not None:
            avp.addNode(node)

    avp.group("in").startNodes()
    time.sleep(1)
    avp.group("proc").startNodes()

    print(f"Graph running — preview: http://172.17.44.114:8081/")
    if WEBUI_API:
        print(f"Web UI: {WEBUI_API}  instance={INSTANCE_NAME}")

    try:
        while True:
            time.sleep(1)
            avp.heartbeat()
    except KeyboardInterrupt:
        avp.shutdown()


if __name__ == "__main__":
    main()
