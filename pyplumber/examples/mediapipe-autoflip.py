#!/usr/bin/env python3
"""MediaPipe Auto-flip — face-tracking 9:16 crop via GPU zero-copy pipeline.

Uses MediaPipeFaceDetectionGpu (C++) to detect faces, then:
  1. A Python bridge node copies detection metadata from the MetadataFrame edge
     onto each VideoFrame's AVDictionary.
  2. The C++ mediapipe_autoflip_crop_metadata node runs Google's
     KinematicPathSolver for temporally-stabilised horizontal cropping.
  3. crop_metadata_cuda applies the CUDA crop.

KinematicPathSolver (Google AutoFlip quality):
  - Causal online filter: position + velocity state machine
  - Constraints: min_motion_to_reframe, max_velocity
  - No lookahead required — smoothing is kinematic, not predictive

Pipeline (all GPU, zero-copy):
  InputRec → Demux → DecVideo(cuda) → Realtime → FilterVideo(scale_cuda)
  → Split [v_for_egl, v_for_autoflip]
     ├─ CudaToEglImage → MediaPipeFaceDetectionGpu → face_det_md (MetadataFrame)
     └─ DetectionBridgeNode (Python, copies face_det_md JSON onto VideoFrame metadata)
  → MediaPipeAutoflipCropMetadata (C++, KinematicPathSolver)
  → CropMetadataCuda → scale_cuda → EncVideo → Mux → Output

Environment variables
─────────────────────
  AVP_INPUT               Input URL/file (default: input.ts)
  AVP_OUTPUT              Output URL/file (default: output.ts)
  AVP_OUTPUT_FORMAT       Output container format (default: mpegts)
  AVP_SRC_WIDTH           Source decode width  (default: 1920)
  AVP_SRC_HEIGHT          Source decode height (default: 1080)
  AVP_OUT_WIDTH           Output width  (default: 540)
  AVP_OUT_HEIGHT          Output height (default: 960)
  AVP_USE_REALTIME        Enable real-time sync (default: true)
  AVP_LOOP                Loop input file (default: true)
  AVP_FACE_RESOURCE_ROOT  MediaPipe resource root (default: auto from build)
  AVP_WEBUI_API           Web UI endpoint, e.g. http://127.0.0.1:22222
  AVP_REMOTE_CONTROL_PORT Control server TCP port (required with AVP_WEBUI_API)
  AVP_INSTANCE_NAME       Web UI instance label (default: mediapipe-autoflip)
  AVP_LOGFILE             Write avplumber log to this file
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pyplumber
from pyplumber.node import (
    AssumeVideoFormat,
    CudaToEglImage,
    DecVideo,
    Demux,
    DrawBBox,
    EncVideo,
    FilterVideo,
    InputRec,
    MediaPipeAutoflipCropMetadata,
    MediaPipeFaceDetectionGpu,
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
INPUT_URL     = os.environ.get("AVP_INPUT",  "input.ts")
OUTPUT_URL    = os.environ.get("AVP_OUTPUT", "output.ts")
OUTPUT_FORMAT = os.environ.get("AVP_OUTPUT_FORMAT", "mpegts")

SRC_W  = int(os.environ.get("AVP_SRC_WIDTH",  "1920"))
SRC_H  = int(os.environ.get("AVP_SRC_HEIGHT", "1080"))
OUT_W  = int(os.environ.get("AVP_OUT_WIDTH",  "540"))
OUT_H  = int(os.environ.get("AVP_OUT_HEIGHT", "960"))

# 9:16 crop at source resolution: width = height * 9/16, rounded to even
CROP_H = SRC_H
CROP_W = int(SRC_H * 9 / 16) & ~1   # 1080 * 9/16 = 607.5 → 606

USE_REALTIME = _env_bool("AVP_USE_REALTIME", True)
LOOP_INPUT   = _env_bool("AVP_LOOP", True)
AUTO_RESTART = "group" if USE_REALTIME else "off"

RESOURCE_ROOT       = os.environ.get("AVP_FACE_RESOURCE_ROOT", "")
WEBUI_API           = os.environ.get("AVP_WEBUI_API", "")
INSTANCE_NAME       = os.environ.get("AVP_INSTANCE_NAME", "mediapipe-autoflip")
LOGFILE             = os.environ.get("AVP_LOGFILE", "")
REMOTE_CONTROL_PORT = int(os.environ.get("AVP_REMOTE_CONTROL_PORT", "0") or "0")

if WEBUI_API and not REMOTE_CONTROL_PORT:
    raise SystemExit("AVP_WEBUI_API requires AVP_REMOTE_CONTROL_PORT.")




# ── Detection bridge node ────────────────────────────────────────────────────
class DetectionBridgeNode(PythonNode):
    """Copies face detection JSON from MetadataFrame edge onto VideoFrame metadata.

    MediaPipeFaceDetectionGpu outputs to a separate MetadataFrame edge.
    mediapipe_autoflip_crop_metadata (C++) reads from the VideoFrame's
    AVDictionary.  This node bridges the two by consuming both streams and
    copying the JSON string from the MetadataFrame onto the VideoFrame.
    """

    def __init__(self, args: dict):
        super().__init__({"data_type": "VideoFrame"} | args)
        self._det_edge_name = str(self.parameters.get("face_det_src", "face_det_md"))
        self._metadata_key  = str(self.parameters.get("metadata_key", "face_detections_v1"))
        self._det_src       = None
        self._frame_count   = 0

    def _avplumber_initialized(self):
        super()._avplumber_initialized()
        self._det_src = self._avplumber.getEdge(self._det_edge_name, "MetadataFrame")
        assert self._det_src is not None, f"MetadataFrame edge {self._det_edge_name!r} not found"

    def process(self):
        frame = self._src.get()
        if frame is None:
            return

        self._frame_count += 1
        det_frame = self._det_src.get()

        if det_frame is not None:
            try:
                md = det_frame.metadata
                md_dict = md.as_dict if hasattr(md, "as_dict") else md
                raw = md_dict.get(self._metadata_key)
                if raw is not None:
                    frame.metadata[self._metadata_key] = raw if isinstance(raw, str) else json.dumps(raw)
            except Exception as exc:
                if self._frame_count % 150 == 1:
                    print(f"[DetectionBridge] metadata copy error: {exc}", flush=True)

        self._dst.enqueue(frame)


# ── Build and run graph ───────────────────────────────────────────────────────
def main() -> None:
    print(f"Input:      {INPUT_URL}")
    print(f"Output:     {OUTPUT_URL} ({OUTPUT_FORMAT})")
    print(f"Source:     {SRC_W}x{SRC_H}  →  crop {CROP_W}x{CROP_H}  →  {OUT_W}x{OUT_H} (9:16)")
    print(f"Realtime:   {USE_REALTIME}  loop={LOOP_INPUT}")
    print(f"Solver:     KinematicPathSolver (Google AutoFlip quality)")

    avp = pyplumber.AVPlumber()
    avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')
    avp.edges.planCapacity("*", 8)
    avp.edges.planCapacity("v_for_egl", 11)

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
            "graph": f"scale_cuda=w={SRC_W}:h={SRC_H}",
            "dst_width": SRC_W, "dst_height": SRC_H,
            "dst_pixel_format": "cuda", "hwaccel": "@gpu",
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        Split({
            "src": "v_scaled", "dst": ["v_for_egl", "v_for_autoflip"],
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        # C++ GPU face detection
        CudaToEglImage({
            "src": "v_for_egl", "dst": "v_egl",
            "pool_id": "mediapipe_face_det", "pool_size": 8, "pool_max_size": 16,
            "sync": True,
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        MediaPipeFaceDetectionGpu({
            "src": "v_egl", "dst": "face_det_md",
            "metadata_key": "face_detections_v1",
            **({"resource_root": RESOURCE_ROOT} if RESOURCE_ROOT else {}),
            "min_detection_confidence": 0.5,
            "emit_dropped_metadata": True,
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        # Python bridge: copies face_det_md JSON from MetadataFrame edge
        # onto VideoFrame AVDictionary so the C++ kinematic node can read it
        DetectionBridgeNode({
            "src": "v_for_autoflip", "dst": "v_with_det",
            "name": "DetectionBridge",
            "face_det_src": "face_det_md",
            "metadata_key": "face_detections_v1",
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        # C++ KinematicPathSolver (Google AutoFlip quality) — reads face_detections_v1,
        # writes smoothed_crop_viewport_v1
        MediaPipeAutoflipCropMetadata({
            "src": "v_with_det", "dst": "v_for_crop",
            "name": "AutoflipKinematic",
            "metadata_key_ins": ["face_detections_v1"],
            "metadata_key_out": "smoothed_crop_viewport_v1",
            "crop_w": CROP_W, "crop_h": CROP_H,
            "debug_log_every_n": 150,
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        # Draw white viewport rectangle instead of cropping
        DrawBBox({
            "src": "v_for_crop", "dst": "v_drawn",
            "name": "Draw_Viewport",
            "metadata_key": "smoothed_crop_viewport_v1",
            "bbox_thickness": 2, "min_conf": 0.0,
            "width": SRC_W, "height": SRC_H,
            "pixel_format": "cuda", "real_pixel_format": "nv12",
            "debug_log_every_n": 150,
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        AssumeVideoFormat({
            "src": "v_drawn", "dst": "v_enc_in",
            "width": SRC_W, "height": SRC_H,
            "pixel_format": "cuda", "real_pixel_format": "nv12",
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        EncVideo({
            "src": "v_enc_in", "dst": "v_enc", "name": "Video_Enc",
            "codec": "h264_nvenc", "hwaccel": "@gpu",
            "options": {"b": "4000k", "maxrate": "4000k", "bufsize": "4000k",
                        "rc": "cbr", "g": 25, "bf": 0, "preset": "p6",
                        "profile": "baseline", "level": "4.0", "tune": "ull",
                        "zerolatency": 1, "delay": 0},
            "group": "proc", "auto_restart": AUTO_RESTART,
        }),
        Mux({"src": ["v_enc"], "dst": "muxed",
             "group": "proc", "auto_restart": AUTO_RESTART}),
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

    print("Graph running.")
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
