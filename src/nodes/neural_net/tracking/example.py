#!/usr/bin/env python3

import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT))

import pyplumber
from pyplumber.node import (
    AssumeVideoFormat,
    DecVideo,
    Demux,
    EncVideo,
    FilterVideo,
    InputRec,
    Mux,
    Output,
    RescaleVideo,
)

from src.nodes.neural_net.tracking.ultralytics_bytetrack import UltralyticsByteTrackNode


INPUT_URL = os.environ.get("AVP_INPUT", "input.mp4")
OUTPUT_URL = os.environ.get("AVP_OUTPUT", "ultralytics-bytetrack.mp4")
OUTPUT_FORMAT = os.environ.get("AVP_OUTPUT_FORMAT", "mp4")
WEIGHTS = os.environ.get("AVP_ULTRALYTICS_WEIGHTS", "yolo11n.pt")
DEVICE = os.environ.get("AVP_ULTRALYTICS_DEVICE", "")
TARGET_LABELS = [
    item.strip()
    for item in os.environ.get("AVP_ULTRALYTICS_TARGET_LABELS", "person").split(",")
    if item.strip()
]

WIDTH = int(os.environ.get("AVP_WIDTH", "1280"))
HEIGHT = int(os.environ.get("AVP_HEIGHT", "720"))
METADATA_KEY = os.environ.get("AVP_ULTRALYTICS_METADATA_KEY", "ultralytics_tracks")
COUNT_KEY = os.environ.get("AVP_ULTRALYTICS_COUNT_KEY", "ultralytics_track_count")

_FILTER_GRAPH = (
    "drawtext=text='TRACKS\\: %{metadata\\:" + COUNT_KEY + "}':"
    " x=24: y=24: box=1: boxborderw=8: fontsize=28: fontcolor=white"
)


avp = pyplumber.AVPlumber()

nodes = [
    InputRec({
        "url": INPUT_URL,
        "dst": "in",
        "group": "in",
        "auto_restart": "off",
        "name": "input",
        "timeout": -1,
    }),
    Demux({
        "src": "in",
        "routing": {"v:0": "v_in"},
        "group": "in",
        "auto_restart": "off",
    }),
    DecVideo({
        "name": "decode",
        "group": "in",
        "src": "v_in",
        "dst": "v_decoded",
        "auto_restart": "off",
    }),
    RescaleVideo({
        "name": "scale_to_tracking",
        "group": "tracking",
        "src": "v_decoded",
        "dst": "v_yuv420p",
        "dst_width": WIDTH,
        "dst_height": HEIGHT,
        "dst_pixel_format": "yuv420p",
        "flags": ["SWS_BILINEAR"],
        "auto_restart": "group",
    }),
    UltralyticsByteTrackNode({
        "name": "ultralytics_bytetrack",
        "group": "tracking",
        "src": "v_yuv420p",
        "dst": "v_tracked",
        "weights": WEIGHTS,
        "device": DEVICE,
        "metadata_key": METADATA_KEY,
        "count_metadata_key": COUNT_KEY,
        "target_labels": TARGET_LABELS,
        "tracker": os.environ.get("AVP_ULTRALYTICS_TRACKER", "bytetrack.yaml"),
        "conf": float(os.environ.get("AVP_ULTRALYTICS_CONF", "0.25")),
        "debug_log_every_n": int(os.environ.get("AVP_DEBUG_EVERY_N", "30")),
    }),
    FilterVideo({
        "name": "draw_count",
        "group": "tracking",
        "src": "v_tracked",
        "dst": "v_annotated",
        "graph": _FILTER_GRAPH,
        "auto_restart": "group",
    }),
    AssumeVideoFormat({
        "name": "assume_encoder_format",
        "group": "tracking",
        "src": "v_annotated",
        "dst": "v_encoder_in",
        "width": WIDTH,
        "height": HEIGHT,
        "pixel_format": "yuv420p",
        "auto_restart": "group",
    }),
    EncVideo({
        "name": "encode",
        "group": "tracking",
        "src": "v_encoder_in",
        "dst": "v_encoded",
        "codec": "libx264",
        "auto_restart": "group",
    }),
    Mux({
        "name": "mux",
        "group": "tracking",
        "src": ["v_encoded"],
        "dst": "muxed",
    }),
    Output({
        "name": "output",
        "group": "tracking",
        "src": "muxed",
        "url": OUTPUT_URL,
        "format": OUTPUT_FORMAT,
        "auto_restart": "group",
    }),
]

for node in nodes:
    avp.addNode(node)

print(f"Input: {INPUT_URL}", flush=True)
print(f"Output: {OUTPUT_URL}", flush=True)
print(f"Ultralytics weights: {WEIGHTS}", flush=True)
print(f"Target labels: {TARGET_LABELS or '<all>'}", flush=True)

avp.group("in").startNodes()
avp.group("tracking").startNodes()

while True:
    time.sleep(1)
    avp.heartbeat()
