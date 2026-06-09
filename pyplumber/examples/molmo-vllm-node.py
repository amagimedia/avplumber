#!/usr/bin/env python3
"""CUDA Molmo/vLLM async node smoke-test graph.

Default backend is ``mock`` so the graph can validate CUDA preprocessing,
windowing, metadata projection, and CUDA overlays before the vLLM direct-tensor
runner hook is installed.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import pyplumber
from pyplumber.molmo_vllm import MolmoVllmAsync
from pyplumber.node import (
    AssumeVideoFormat,
    DecVideo,
    Demux,
    DrawBBox,
    DrawBBoxLabels,
    DrawKeypoints,
    EncVideo,
    FilterVideo,
    InputRec,
    Mux,
    Output,
)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


INPUT_URL = os.environ.get("AVP_INPUT", "input.mp4")
OUTPUT_URL = os.environ.get("AVP_OUTPUT", "molmo-vllm-node.mp4")
OUTPUT_FORMAT = os.environ.get("AVP_OUTPUT_FORMAT", "mp4")
WIDTH = int(os.environ.get("AVP_WIDTH", "1280"))
HEIGHT = int(os.environ.get("AVP_HEIGHT", "720"))

MOLMO_BACKEND = os.environ.get("AVP_MOLMO_BACKEND", "mock")
MOLMO_STRICT = _env_bool("AVP_MOLMO_STRICT", False)
MOLMO_RUNNER_FACTORY = os.environ.get("AVP_MOLMO_RUNNER_FACTORY", "")
MOLMO_MODEL_ID = os.environ.get("AVP_MOLMO_MODEL_ID", "allenai/Molmo2-VideoPoint-4B")
MOLMO_PROMPT = os.environ.get(
    "AVP_MOLMO_PROMPT",
    "Find people, vehicles, balls, and important objects. Return JSON only with objects containing label, confidence, bbox, and point. Coordinates must be integers from 0 to 1000 relative to the image.",
)


avp = pyplumber.AVPlumber()
avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')
avp.edges.planCapacity("*", 7)

nodes = [
    InputRec(
        {
            "url": INPUT_URL,
            "dst": "in_pkt",
            "group": "in",
            "auto_restart": "off",
            "name": "Input",
            "timeout": -1,
            "loop": False,
        }
    ),
    Demux(
        {
            "src": "in_pkt",
            "routing": {"v:0": "v_pkt"},
            "group": "in",
            "auto_restart": "off",
        }
    ),
    DecVideo(
        {
            "src": "v_pkt",
            "dst": "v_dec_cuda",
            "group": "in",
            "name": "Video_Decode",
            "auto_restart": "off",
            "pixel_format": "?cuda",
            "hwaccel": "@gpu",
            "codec_map": {"h264": "h264_cuvid", "hevc": "hevc_cuvid"},
            "hwaccel_only_for_codecs": ["h264", "hevc"],
        }
    ),
    FilterVideo(
        {
            "graph": f"scale_cuda=w={WIDTH}:h={HEIGHT}",
            "src": "v_dec_cuda",
            "dst": "v_scaled_cuda",
            "group": "g1",
            "name": "Scale_For_Molmo",
            "auto_restart": "group",
            "dst_width": WIDTH,
            "dst_height": HEIGHT,
            "dst_pixel_format": "cuda",
            "hwaccel": "@gpu",
        }
    ),
    MolmoVllmAsync(
        {
            "src": "v_scaled_cuda",
            "dst": "v_molmo_md",
            "group": "g1",
            "name": "Molmo_Vllm_Async",
            "backend": MOLMO_BACKEND,
            "strict_zero_copy": MOLMO_STRICT,
            "runner_factory": MOLMO_RUNNER_FACTORY,
            "model_id": MOLMO_MODEL_ID,
            "prompt": MOLMO_PROMPT,
            "prompt_id": "example_objects",
            "molmo_frame_size": int(os.environ.get("AVP_MOLMO_FRAME_SIZE", "378")),
            "allow_non_default_frame_size": _env_bool("AVP_MOLMO_ALLOW_NON_DEFAULT_SIZE", False),
            "patch_size": 14,
            "sample_fps": float(os.environ.get("AVP_MOLMO_SAMPLE_FPS", "8")),
            "window_frames": int(os.environ.get("AVP_MOLMO_WINDOW_FRAMES", "16")),
            "window_stride": int(os.environ.get("AVP_MOLMO_WINDOW_STRIDE", os.environ.get("AVP_MOLMO_WINDOW_FRAMES", "16"))),
            "window_queue_size": 1,
            "max_inflight": 1,
            "visualize_ttl_frames": int(os.environ.get("AVP_MOLMO_VISUALIZE_TTL_FRAMES", "32")),
            "gpu_memory_utilization": float(os.environ.get("AVP_MOLMO_GPU_MEMORY_UTILIZATION", "0.75")),
            "max_model_len": int(os.environ.get("AVP_MOLMO_MAX_MODEL_LEN", "8192")),
            "max_new_tokens": int(os.environ.get("AVP_MOLMO_MAX_NEW_TOKENS", "512")),
            "temperature": float(os.environ.get("AVP_MOLMO_TEMPERATURE", "0.0")),
            "tensor_dtype": os.environ.get("AVP_MOLMO_TENSOR_DTYPE", "fp16"),
            "auto_restart": "group",
        }
    ),
    DrawBBox(
        {
            "src": "v_molmo_md",
            "dst": "v_molmo_bbox",
            "group": "g1",
            "name": "Draw_Molmo_BBox",
            "metadata_key": "molmo_detections",
            "bbox_thickness": 2,
            "min_conf": 0.0,
            "width": WIDTH,
            "height": HEIGHT,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "auto_restart": "group",
        }
    ),
    DrawKeypoints(
        {
            "src": "v_molmo_bbox",
            "dst": "v_molmo_points",
            "group": "g1",
            "name": "Draw_Molmo_Points",
            "metadata_key": "molmo_points",
            "radius": 5,
            "color": "yellow",
            "min_conf": 0.0,
            "width": WIDTH,
            "height": HEIGHT,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "auto_restart": "group",
        }
    ),
    DrawBBoxLabels(
        {
            "src": "v_molmo_points",
            "dst": "v_molmo_labels",
            "group": "g1",
            "name": "Draw_Molmo_Labels",
            "metadata_key": "molmo_detections",
            "min_conf": 0.0,
            "width": WIDTH,
            "height": HEIGHT,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "auto_restart": "group",
        }
    ),
    AssumeVideoFormat(
        {
            "src": "v_molmo_labels",
            "dst": "v_preenc",
            "group": "g1",
            "width": WIDTH,
            "height": HEIGHT,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "auto_restart": "group",
        }
    ),
    EncVideo(
        {
            "src": "v_preenc",
            "dst": "v_outenc",
            "group": "g1",
            "name": "Video_Encode",
            "codec": "h264_nvenc",
            "hwaccel": "@gpu",
            "auto_restart": "group",
            "options": {
                "b": "6000k",
                "maxrate": "8000k",
                "bufsize": "12000k",
                "preset": "p5",
                "profile": "high",
                "bf": 0,
            },
        }
    ),
    Mux(
        {
            "src": ["v_outenc"],
            "dst": "muxed",
            "group": "g1",
        }
    ),
    Output(
        {
            "src": "muxed",
            "format": OUTPUT_FORMAT,
            "url": OUTPUT_URL,
            "group": "g1",
            "auto_restart": "group",
        }
    ),
]

for node in nodes:
    avp.addNode(node)

print(
    "Starting Molmo/vLLM example:",
    f"input={INPUT_URL}",
    f"output={OUTPUT_URL}",
    f"backend={MOLMO_BACKEND}",
    f"strict={MOLMO_STRICT}",
)

avp.group("in").startNodes()
avp.group("g1").startNodes()

while True:
    time.sleep(1)
    avp.heartbeat()
