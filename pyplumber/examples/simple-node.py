#!/usr/bin/env python3
# Simple node that just prints the input data

import time
import sys
import os
sys.path.append("../..")

import pyplumber
from pyplumber.node import (
    PythonNode,
    InputRec,
    Demux,
    DecVideo,
    SpeedVideo,
    ForceFPS,
    Pause,
    RealtimeVideoFrame,
    RescaleVideo,
    FilterVideo,
    AssumeVideoFormat,
    EncVideo,
    Mux,
    Output,
)

INPUT_URL = os.environ.get("AVP_INPUT", "input.mp4")
OUTPUT_URL = os.environ.get("AVP_OUTPUT", "output.mp4")
OUTPUT_FORMAT = os.environ.get("AVP_OUTPUT_FORMAT", "mpegts")

_FILTER_GRAPH = (
    "drawtext=text='MSG\\: %{metadata\\:msg}\n"
    "FRAME\\: %{metadata\\:frame_no}\n"
    "VIDEO\\: %{metadata\\:video_ts}\n"
    "PTS\\: %{metadata\\:video_pts}\n"
    "INPUT\\: %{metadata\\:input_ts}\n"
    "OUTPUT\\: %{metadata\\:output_ts}\n"
    "WALLCLOCK\\: %{metadata\\:wallclock_ts}': x=20: y=100: box=1: boxborderw=2"
)

avp = pyplumber.AVPlumber()

input_rec = InputRec({
    "url": INPUT_URL,
    "dst": "in10",
    "group": "in",
    "auto_restart": "off",
    "name": "input",
    "timeout": -1,
    "preseek": 0,
    "seek_table": "",
    "ts_offsets": "",
    "team": "@input-seek-team",
    "timestamp_source": "wallclock",
    "loop": True,
    "pause_team": "pause-team",
    "speed_team": "speed-team",
})
demux = Demux({
    "src": "in10",
    "routing": {"v:0": "v_in", "?d:0": "d_in"},
    "group": "in",
    "auto_restart": "off",
})
dec_video = DecVideo({
    "auto_restart": "off",
    "name": "Video_Decode",
    "group": "in",
    "src": "v_in",
    "dst": "Video_Decode_out",
})
speed_video = SpeedVideo({
    "name": "speed",
    "team": "speed-team",
    "group": "g1",
    "src": "Video_Decode_out",
    "dst": "speeded",
    "speed": 1,
    "sync_team": "sync-team",
    "sync_node": "rtsync",
})
force_fps = ForceFPS({
    "fps": "50",
    "group": "g1",
    "src": "speeded",
    "dst": "forcedfps",
})
pause = Pause({
    "name": "pause",
    "team": "pause-team",
    "group": "g1",
    "src": "forcedfps",
    "dst": "pause_out",
    "paused": False,
    "x--sync_team": "x--sync-team",
})
realtime_vf = RealtimeVideoFrame({
    "team": "sync-team",
    "negative_time_tolerance": 0.007,
    "jitter_margin": 0,
    "discontinuity_threshold": 3,
    "src": "pause_out",
    "dst": "v_rt",
    "group": "g1",
    "name": "rtsync",
    "set_pts": True,
})
rescale = RescaleVideo({
    "dst_width": 1280,
    "dst_height": 720,
    "flags": ["SWS_LANCZOS"],
    "dst_pixel_format": "yuv420p",
    "auto_restart": "panic",
    "name": "scale",
    "group": "g1",
    "src": "v_rt",
    "dst": "scaled_to_python",
})
filter_video = FilterVideo({
    "graph": _FILTER_GRAPH,
    "auto_restart": "group",
    "name": "Video_Filter",
    "group": "g1",
    "src": "scaled",
    "dst": "v_to_encoder",
})
assume_video_format = AssumeVideoFormat({
    "width": 1280,
    "height": 720,
    "pixel_format": "yuv420p",
    "group": "g1",
    "src": "v_to_encoder",
    "dst": "v_to_encoder_assumed",
    "auto_restart": "group",
})
enc_video = EncVideo({
    "auto_restart": "group",
    "name": "Video_Encode",
    "group": "g1",
    "src": "v_to_encoder_assumed",
    "dst": "v_outenc",
    "codec": "libx264",
})
mux = Mux({
    "src": ["v_outenc"],
    "dst": "mux0-1",
    "group": "g1",
})
out = Output({
    "format": OUTPUT_FORMAT,
    "url": OUTPUT_URL,
    "src": "mux0-1",
    "group": "in",
    "auto_restart": "group",
})
        

for node in (
    input_rec,
    demux,
    dec_video,
    speed_video,
    force_fps,
    pause,
    realtime_vf,
    rescale,
    filter_video,
    assume_video_format,
    enc_video,
    mux,
    out,
):
    avp.addNode(node)


class SimpleNode(PythonNode):
    def process(self):
        p = self._src.get()
        if p:
            p.metadata["msg"] = f"Hello from python: {p}"
            if isinstance(self._dst, dict):
                for dst_name, dst in self._dst.items():
                    p.metadata["msg"] += f" {dst_name}"
                    dst.enqueue(p)
            else:
                self._dst.enqueue(p)


node = SimpleNode({"src": "scaled_to_python", "dst": ["scaled"], "group": "g1", "name": "python-test-node"})
avp.addNode(node)

print("Starting groups")
avp.group("in").startNodes()
avp.group("g1").startNodes()

while True:
    time.sleep(1)
    avp.heartbeat()
