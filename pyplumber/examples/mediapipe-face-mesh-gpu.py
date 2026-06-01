#!/usr/bin/env python3

import json
import os
import sys
import time
from pathlib import Path

sys.path.append("../..")

import pyplumber  # pyright: ignore[reportMissingImports]
from pyplumber.node import (  # pyright: ignore[reportMissingImports]
    CudaToEglImage,
    DecVideo,
    Demux,
    FilterVideo,
    ForceFPS,
    InputRec,
    MediaPipeFaceMeshGpu,
    PythonNode,
    Realtime,
)

INPUT_URL = os.environ.get("AVP_INPUT", "input.mp4")
METADATA_JSONL = os.environ.get("AVP_FACE_MESH_JSONL", "face-mesh-events.jsonl")
MODEL_WIDTH = int(os.environ.get("AVP_FACE_MODEL_WIDTH", "960"))
MODEL_HEIGHT = int(os.environ.get("AVP_FACE_MODEL_HEIGHT", "540"))
FPS = os.environ.get("AVP_FPS", "30/1")
METADATA_KEY = os.environ.get("AVP_FACE_METADATA_KEY", "face_landmarks_v1")
RESOURCE_ROOT = os.environ.get("AVP_FACE_RESOURCE_ROOT")


class MetadataJsonlSink(PythonNode):
    def __init__(self, args: dict):
        super().__init__({"data_type": "MetadataFrame"} | args)
        self.metadata_key = str(self.parameters.get("metadata_key", METADATA_KEY))
        self.output_path = Path(str(self.parameters.get("output_path", METADATA_JSONL)))
        self._fh = None

    def doStart(self):
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.output_path.open("w", encoding="utf-8")

    def doStop(self):
        if self._fh:
            self._fh.close()
            self._fh = None

    def process(self):
        frame = self._src.get()
        if frame is None or not frame.isComplete:
            return

        payload = frame.metadata.as_dict
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        print(line, flush=True)
        if self._fh:
            self._fh.write(line + "\n")
            self._fh.flush()

        face_block = payload.get(self.metadata_key, {})
        for event in face_block.get("events", []):
            print(
                f"face_event pts={face_block.get('pts')} type={event.get('type')} face_id={event.get('face_id')}",
                flush=True,
            )


def main() -> None:
    print(f"Input: {INPUT_URL}", flush=True)
    print(f"Metadata JSONL: {METADATA_JSONL}", flush=True)

    avp = pyplumber.AVPlumber()
    avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')
    avp.edges.planCapacity("*", 8)

    nodes = [
        InputRec({
            "name": "input",
            "url": INPUT_URL,
            "dst": "in_mux",
            "group": "in",
            "initial_timeout": 20,
            "timeout": 10,
            "loop": True,
            "auto_restart": "group",
        }),
        Demux({
            "src": "in_mux",
            "routing": {"?v:0": "v_pkt"},
            "wait_for_keyframe": False,
            "group": "in",
            "auto_restart": "group",
        }),
        DecVideo({
            "src": "v_pkt",
            "dst": "v_dec_cuda",
            "name": "Video_Dec",
            "pixel_format": "?cuda",
            "hwaccel": "@gpu",
            "group": "in",
            "auto_restart": "group",
            "codec_map": {"h264": "h264_cuvid", "hevc": "hevc_cuvid"},
            "hwaccel_only_for_codecs": ["h264", "hevc"],
        }),
        Realtime({
            "src": "v_dec_cuda",
            "dst": "v_rt",
            "set_pts": True,
            "group": "in",
            "auto_restart": "group",
        }),
        ForceFPS({
            "src": "v_rt",
            "dst": "v_fps",
            "fps": FPS,
            "group": "face",
            "auto_restart": "group",
        }),
        FilterVideo({
            "src": "v_fps",
            "dst": "v_face_cuda",
            "graph": f"scale_cuda=w={MODEL_WIDTH}:h={MODEL_HEIGHT}",
            "dst_width": MODEL_WIDTH,
            "dst_height": MODEL_HEIGHT,
            "dst_pixel_format": "cuda",
            "hwaccel": "@gpu",
            "group": "face",
            "auto_restart": "group",
        }),
        CudaToEglImage({
            "src": "v_face_cuda",
            "dst": "v_face_egl",
            "pool_id": "mediapipe_face_mesh",
            "pool_size": 8,
            "pool_max_size": 16,
            "sync": True,
            "group": "face",
            "auto_restart": "group",
        }),
        MediaPipeFaceMeshGpu({
            "src": "v_face_egl",
            "dst": "face_metadata",
            "metadata_key": METADATA_KEY,
            **({"resource_root": RESOURCE_ROOT} if RESOURCE_ROOT else {}),
            "max_faces": 1,
            "with_attention": True,
            "speaking_start_open_ratio": float(os.environ.get("AVP_SPEAKING_START_OPEN_RATIO", "0.045")),
            "speaking_stop_open_ratio": float(os.environ.get("AVP_SPEAKING_STOP_OPEN_RATIO", "0.030")),
            "speaking_start_confirm_frames": int(os.environ.get("AVP_SPEAKING_START_CONFIRM_FRAMES", "2")),
            "speaking_stop_confirm_frames": int(os.environ.get("AVP_SPEAKING_STOP_CONFIRM_FRAMES", "5")),
            "debug_log_every_n": int(os.environ.get("AVP_DEBUG_LOG_EVERY_N", "30")),
            "group": "face",
            "auto_restart": "group",
        }),
        MetadataJsonlSink({
            "src": "face_metadata",
            "name": "face-metadata-jsonl",
            "metadata_key": METADATA_KEY,
            "output_path": METADATA_JSONL,
            "group": "sink",
        }),
    ]

    for node in nodes:
        avp.addNode(node)

    avp.group("in").startNodes()
    avp.group("face").startNodes()
    avp.group("sink").startNodes()

    try:
        while True:
            time.sleep(1)
            avp.heartbeat()
    except KeyboardInterrupt:
        avp.shutdown()


if __name__ == "__main__":
    main()
