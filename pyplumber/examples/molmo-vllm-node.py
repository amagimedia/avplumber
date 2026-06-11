#!/usr/bin/env python3
"""CUDA Molmo/vLLM async node smoke-test graph.

Default backend is ``mock`` so the graph can validate CUDA preprocessing,
windowing, metadata projection, and CUDA overlays cheaply. Set
``AVP_MOLMO_BACKEND=vllm`` for local Molmo2 inference.
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
    RealtimeVideoFrame,
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
USE_REALTIME = _env_bool("AVP_USE_REALTIME", False)
EXIT_ON_EOF = _env_bool("AVP_EXIT_ON_EOF", not USE_REALTIME)
VOD_IDLE_SECONDS = float(os.environ.get("AVP_VOD_IDLE_SECONDS", "5"))
VOD_MAX_SECONDS = float(os.environ.get("AVP_VOD_MAX_SECONDS", "0"))
VOD_MIN_OUTPUT_BYTES = int(os.environ.get("AVP_VOD_MIN_OUTPUT_BYTES", "65536"))
VOD_IDLE_FALLBACK = _env_bool("AVP_VOD_IDLE_FALLBACK", False)
VOD_START_DELAY_SECONDS = float(os.environ.get("AVP_VOD_START_DELAY_SECONDS", "2"))
PROCESS_AUTO_RESTART = os.environ.get("AVP_PROCESS_AUTO_RESTART", "group" if USE_REALTIME else "off")

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
    RealtimeVideoFrame(
        {
            "src": "v_dec_cuda",
            "dst": "v_rt_cuda",
            "group": "g1",
            "name": "Realtime_Input",
            "auto_restart": PROCESS_AUTO_RESTART,
            "team": "molmo-sync",
            "negative_time_tolerance": 0.007,
            "jitter_margin": 0,
            "discontinuity_threshold": 3,
            "set_pts": True,
        }
    )
    if USE_REALTIME
    else None,
    FilterVideo(
        {
            "graph": f"scale_cuda=w={WIDTH}:h={HEIGHT}",
            "src": "v_rt_cuda" if USE_REALTIME else "v_dec_cuda",
            "dst": "v_scaled_cuda",
            "group": "g1",
            "name": "Scale_For_Molmo",
            "dst_width": WIDTH,
            "dst_height": HEIGHT,
            "dst_pixel_format": "cuda",
            "hwaccel": "@gpu",
            "auto_restart": PROCESS_AUTO_RESTART,
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
            "prompt_style": os.environ.get("AVP_MOLMO_PROMPT_STYLE", ""),
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
            "visualize_ttl_seconds": os.environ.get("AVP_MOLMO_VISUALIZE_TTL_SECONDS"),
            "blocking_visualization": _env_bool("AVP_MOLMO_BLOCKING_VISUALIZATION", False),
            "gpu_memory_utilization": float(os.environ.get("AVP_MOLMO_GPU_MEMORY_UTILIZATION", "0.75")),
            "max_model_len": int(os.environ.get("AVP_MOLMO_MAX_MODEL_LEN", "8192")),
            "max_num_batched_tokens": int(os.environ.get("AVP_MOLMO_MAX_NUM_BATCHED_TOKENS", "8192")),
            "max_new_tokens": int(os.environ.get("AVP_MOLMO_MAX_NEW_TOKENS", "512")),
            "temperature": float(os.environ.get("AVP_MOLMO_TEMPERATURE", "0.0")),
            "tensor_dtype": os.environ.get("AVP_MOLMO_TENSOR_DTYPE", "fp16"),
            "model_dtype": os.environ.get("AVP_MOLMO_MODEL_DTYPE", "float16"),
            "mm_processor_cache_gb": float(os.environ.get("AVP_MOLMO_MM_PROCESSOR_CACHE_GB", "0")),
            "cpu_offload_gb": float(os.environ.get("AVP_MOLMO_CPU_OFFLOAD_GB", "0")),
            "tensor_parallel_size": int(os.environ.get("AVP_MOLMO_TENSOR_PARALLEL_SIZE", "1")),
            "enforce_eager": _env_bool("AVP_MOLMO_ENFORCE_EAGER", True),
            "seed": int(os.environ.get("AVP_MOLMO_SEED", "0")),
            "result_policy": os.environ.get("AVP_MOLMO_RESULT_POLICY", "hold_latest"),
            "debug_log_every_n": int(os.environ.get("AVP_MOLMO_DEBUG_LOG_EVERY_N", "0")),
            "delayed_track_hold_seconds": float(os.environ.get("AVP_MOLMO_DELAYED_TRACK_HOLD_SECONDS", "0")),
            "auto_restart": PROCESS_AUTO_RESTART,
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
            "debug_log_every_n": int(os.environ.get("AVP_DRAW_KEYPOINTS_DEBUG_LOG_EVERY_N", "0")),
            "auto_restart": PROCESS_AUTO_RESTART,
        }
    ),
    DrawKeypoints(
        {
            "src": "v_molmo_bbox",
            "dst": "v_molmo_points",
            "group": "g1",
            "name": "Draw_Molmo_Points",
            "metadata_key": "molmo_points",
            "radius": int(os.environ.get("AVP_MOLMO_POINT_RADIUS", "5")),
            "color": os.environ.get("AVP_MOLMO_POINT_COLOR", "yellow"),
            "min_conf": 0.0,
            "width": WIDTH,
            "height": HEIGHT,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "auto_restart": PROCESS_AUTO_RESTART,
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
            "auto_restart": PROCESS_AUTO_RESTART,
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
            "auto_restart": PROCESS_AUTO_RESTART,
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
            "auto_restart": PROCESS_AUTO_RESTART,
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
            "auto_restart": PROCESS_AUTO_RESTART,
        }
    ),
    Output(
        {
            "src": "muxed",
            "format": OUTPUT_FORMAT,
            "url": OUTPUT_URL,
            "group": "g1",
            "name": "Output",
            "auto_restart": PROCESS_AUTO_RESTART,
        }
    ),
]

for node in nodes:
    if node is not None:
        avp.addNode(node)

if _env_bool("AVP_MOLMO_PRELOAD", False):
    for node in nodes:
        runner = getattr(node, "_runner", None)
        if isinstance(node, MolmoVllmAsync) and hasattr(runner, "_ensure_llm"):
            print("Preloading Molmo/vLLM model before graph start")
            runner._ensure_llm()
            print("Molmo/vLLM model preload done")
            break

print(
    "Starting Molmo/vLLM example:",
    f"input={INPUT_URL}",
    f"output={OUTPUT_URL}",
    f"backend={MOLMO_BACKEND}",
    f"strict={MOLMO_STRICT}",
)

avp.group("in").startNodes()
if not USE_REALTIME and VOD_START_DELAY_SECONDS > 0:
    time.sleep(VOD_START_DELAY_SECONDS)
avp.group("g1").startNodes()


def _stop_vod_graph() -> None:
    avp.group("g1").stopNodes()
    avp.group("in").stopNodes()
    avp.shutdown()


def _node_is_working(name: str) -> bool:
    try:
        return bool(avp.node(name).isWorking)
    except Exception:
        return True


last_size = -1
last_change = time.monotonic()
start_time = time.monotonic()
output_seen_working = False
while True:
    time.sleep(1)
    avp.heartbeat()
    if not EXIT_ON_EOF:
        continue

    output_path = Path(OUTPUT_URL)
    output_working = _node_is_working("Output")
    if output_working:
        output_seen_working = True
    elif output_seen_working:
        size = output_path.stat().st_size if output_path.exists() else 0
        print(
            "VOD output node finished; stopping:",
            f"output={OUTPUT_URL}",
            f"size={size}",
        )
        _stop_vod_graph()
        break

    if output_path.exists():
        size = output_path.stat().st_size
        if size != last_size:
            last_size = size
            last_change = time.monotonic()
        elif VOD_IDLE_FALLBACK and size >= VOD_MIN_OUTPUT_BYTES and time.monotonic() - last_change >= VOD_IDLE_SECONDS:
            print(
                "VOD output idle; stopping:",
                f"output={OUTPUT_URL}",
                f"size={size}",
                f"idle_seconds={VOD_IDLE_SECONDS}",
                f"min_output_bytes={VOD_MIN_OUTPUT_BYTES}",
            )
            _stop_vod_graph()
            break

    if VOD_MAX_SECONDS > 0 and time.monotonic() - start_time >= VOD_MAX_SECONDS:
        print(
            "VOD max runtime reached; stopping:",
            f"output={OUTPUT_URL}",
            f"max_seconds={VOD_MAX_SECONDS}",
        )
        _stop_vod_graph()
        break
