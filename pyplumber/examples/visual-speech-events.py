#!/usr/bin/env python3

import ast
import json
import os
import sys
import time
from pathlib import Path

sys.path.append("../..")
import pyplumber  # pyright: ignore[reportMissingImports]
from pyplumber.node import (  # pyright: ignore[reportMissingImports]
    CudaInferYolo,
    DecVideo,
    Demux,
    FilterVideo,
    InputRec,
    JoinMetadata,
    NullSink,
    Split,
)
from pyplumber.mouth_tracker import FaceAnchoredMouthTrackerNode  # pyright: ignore[reportMissingImports]
from pyplumber.visual_speech import VisualSpeechGateNode  # pyright: ignore[reportMissingImports]


INPUT_URL = os.environ.get("AVP_INPUT", "input.ts")
EVENTS_JSONL = os.environ.get("AVP_EVENTS_JSONL", "visual-speech-events.jsonl")
SUMMARY_JSON = os.environ.get("AVP_SUMMARY_JSON", "visual-speech-summary.json")

MODEL_DIR = Path(os.environ.get("AVP_MODEL_DIR", "/home/user/tensorrt/face-recognition-1.2"))
DATA_YAML = Path(os.environ.get("AVP_DATA_YAML", str(MODEL_DIR / "data.yaml")))
FACE_ENGINE = os.environ.get("AVP_FACE_ENGINE")

MODEL_WIDTH = int(os.environ.get("AVP_MODEL_WIDTH", "960"))
MODEL_HEIGHT = int(os.environ.get("AVP_MODEL_HEIGHT", "544"))
MODEL_CONTENT_HEIGHT = int(os.environ.get("AVP_MODEL_CONTENT_HEIGHT", "540"))
MODEL_CONTENT_OFFSET_Y = int(os.environ.get("AVP_MODEL_CONTENT_OFFSET_Y", "2"))
CONF_THRESH = float(os.environ.get("AVP_CONF_THRESH", "0.25"))
MAX_DET = int(os.environ.get("AVP_MAX_DET", "300"))
METADATA_KEY = os.environ.get("AVP_FACE_METADATA_KEY", "face_parts")
MOUTH_METADATA_KEY = os.environ.get("AVP_MOUTH_METADATA_KEY", "mouth_rois_v1")
VISUAL_METADATA_KEY = os.environ.get("AVP_VISUAL_METADATA_KEY", "visual_speech_v1")
SOURCE_NAME = os.environ.get("AVP_SOURCE_NAME", Path(INPUT_URL).stem)


def _parse_yaml_names(data_yaml: Path) -> list[str]:
    if not data_yaml.exists():
        return []

    lines = data_yaml.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped.startswith("names:"):
            continue

        value = stripped.removeprefix("names:").strip()
        if value:
            try:
                parsed = ast.literal_eval(value)
            except (SyntaxError, ValueError):
                return []
            if isinstance(parsed, dict):
                return [str(parsed[key]) for key in sorted(parsed)]
            if isinstance(parsed, (list, tuple)):
                return [str(name) for name in parsed]

        names: list[str] = []
        for child in lines[i + 1:]:
            child_stripped = child.strip()
            if not child.startswith((" ", "\t")) or not child_stripped:
                break
            if ":" in child_stripped:
                _, name = child_stripped.split(":", 1)
                names.append(name.strip().strip("'\""))
        return names

    return []


def class_names() -> list[str]:
    env_names = os.environ.get("AVP_CLASS_NAMES")
    if env_names:
        return [name.strip() for name in env_names.split(",") if name.strip()]
    names = _parse_yaml_names(DATA_YAML)
    return names or ["Eye", "Face", "MakeUp", "Mouth", "Nose", "Tooth", "Topping"]


def face_engine_path() -> str:
    if FACE_ENGINE:
        return FACE_ENGINE
    candidates = [
        MODEL_DIR / "face-recognition_960x544.plan",
        MODEL_DIR / "face-recognition_960x544.engine",
        MODEL_DIR / "best.plan",
        MODEL_DIR / "best.engine",
        MODEL_DIR / "train" / "weights" / "best.plan",
        MODEL_DIR / "train" / "weights" / "best.engine",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(MODEL_DIR / "face-recognition_960x544.plan")


def visual_targets() -> list[dict]:
    env_targets = os.environ.get("AVP_VISUAL_TARGETS")
    if env_targets:
        parsed = json.loads(env_targets)
        if not isinstance(parsed, list):
            raise ValueError("AVP_VISUAL_TARGETS must be a JSON array")
        return parsed
    return [{"name": "primary"}]


def wait_until_done(avp, sink_name: str, timeout_s: float) -> None:
    sink = avp.node(sink_name)
    deadline = time.time() + timeout_s
    while not sink.isWorking and time.time() < deadline:
        time.sleep(0.1)
    if not sink.isWorking:
        raise TimeoutError(f"{sink_name} did not start")
    while sink.isWorking and time.time() < deadline:
        time.sleep(0.5)
    if sink.isWorking:
        raise TimeoutError(f"{sink_name} did not finish")
    sink.join()


def main() -> None:
    names = class_names()
    engine = face_engine_path()
    targets = visual_targets()

    print(f"Input: {INPUT_URL}", flush=True)
    print(f"Events JSONL: {EVENTS_JSONL}", flush=True)
    print(f"Summary JSON: {SUMMARY_JSON}", flush=True)
    print(f"Face engine: {engine}", flush=True)
    print(f"Face class names: {names}", flush=True)
    print(f"Visual targets: {json.dumps(targets, sort_keys=True)}", flush=True)
    print(f"Mouth metadata key: {MOUTH_METADATA_KEY}", flush=True)

    avp = pyplumber.AVPlumber()
    avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')
    avp.edges.planCapacity("*", 12)

    nodes = [
        InputRec({
            "url": INPUT_URL,
            "dst": "in_mux0",
            "group": "visual_speech",
            "name": "Input",
            "initial_timeout": 20,
            "timeout": 10,
        }),
        Demux({
            "src": "in_mux0",
            "wait_for_keyframe": False,
            "routing": {"?v:0": "v_pkt"},
            "group": "visual_speech",
            "name": "Demux",
        }),
        DecVideo({
            "src": "v_pkt",
            "dst": "v_dec_cuda",
            "group": "visual_speech",
            "name": "Decode_Video_CUDA",
            "optional": True,
            "pixel_format": "?cuda",
            "hwaccel": "@gpu",
            "codec_map": {"h264": "h264_cuvid", "hevc": "hevc_cuvid"},
            "hwaccel_only_for_codecs": ["h264", "hevc"],
        }),
        Split({
            "src": "v_dec_cuda",
            "dst": ["v_dec_fullres", "v_dec_for_yolo"],
            "group": "visual_speech",
            "name": "Split_For_Yolo",
        }),
        FilterVideo({
            "graph": (
                f"scale_cuda=w={MODEL_WIDTH}:h={MODEL_CONTENT_HEIGHT},"
                f"pad_cuda={MODEL_WIDTH}:{MODEL_HEIGHT}:0:{MODEL_CONTENT_OFFSET_Y}"
            ),
            "src": "v_dec_for_yolo",
            "dst": "v_pre_yolo",
            "group": "visual_speech",
            "name": "Scale_Yolo",
            "dst_width": MODEL_WIDTH,
            "dst_height": MODEL_HEIGHT,
            "dst_pixel_format": "cuda",
            "hwaccel": "@gpu",
        }),
        CudaInferYolo({
            "src": "v_pre_yolo",
            "dst": "v_post_faces",
            "group": "visual_speech",
            "name": "Yolo_FaceParts",
            "input_format": "RGB",
            "conf_thresh": CONF_THRESH,
            "max_det": MAX_DET,
            "infer_every_n": 1,
            "metadata_key_detection": METADATA_KEY,
            "debug_log_metadata": False,
            "debug_log_every_n": 300,
            "mask_gpu_every_n": 0,
            "mask_cpu_every_n": 0,
            "models": [{
                "engine": engine,
                "task_type": "detection",
                "class_names": names,
                "output_box_format": "end2end_xyxy",
            }],
        }),
        JoinMetadata({
            "src": ["v_dec_fullres", "v_post_faces"],
            "dst": "v_with_faces",
            "group": "visual_speech",
            "name": "Join_Face_Metadata",
        }),
        FaceAnchoredMouthTrackerNode({
            "src": "v_with_faces",
            "dst": "v_with_mouth_rois",
            "group": "visual_speech",
            "name": "Mouth_Tracker",
            "run_in_wrapper_thread": True,
            "source": SOURCE_NAME,
            "input_metadata_key": METADATA_KEY,
            "output_metadata_key": MOUTH_METADATA_KEY,
            "targets": targets,
            "min_conf": CONF_THRESH,
            "mouth_tracker_enabled": os.environ.get("AVP_MOUTH_TRACKER_ENABLED", "1") not in {"0", "false", "False"},
            "mouth_track_hold_ms": float(os.environ.get("AVP_MOUTH_TRACK_HOLD_MS", "60000")),
            "mouth_track_alpha": float(os.environ.get("AVP_MOUTH_TRACK_ALPHA", "0.35")),
            "mouth_estimate_enabled": os.environ.get("AVP_MOUTH_ESTIMATE_ENABLED", "1") not in {"0", "false", "False"},
            "mouth_default_rel_w": float(os.environ.get("AVP_MOUTH_DEFAULT_REL_W", "0.24")),
            "mouth_default_rel_h": float(os.environ.get("AVP_MOUTH_DEFAULT_REL_H", "0.07")),
            "mouth_default_rel_y": float(os.environ.get("AVP_MOUTH_DEFAULT_REL_Y", "0.70")),
            "mouth_nose_rel_offset_y": float(os.environ.get("AVP_MOUTH_NOSE_REL_OFFSET_Y", "0.18")),
            "log_every_n": int(os.environ.get("AVP_MOUTH_LOG_EVERY_N", "0")),
        }),
        VisualSpeechGateNode({
            "src": "v_with_mouth_rois",
            "dst": "v_visual_speech",
            "group": "visual_speech",
            "name": "Visual_Speech_Gate",
            "run_in_wrapper_thread": True,
            "source": SOURCE_NAME,
            "mouth_metadata_key": MOUTH_METADATA_KEY,
            "output_metadata_key": VISUAL_METADATA_KEY,
            "event_jsonl_path": EVENTS_JSONL,
            "summary_json_path": SUMMARY_JSON,
            "targets": targets,
            "start_threshold": float(os.environ.get("AVP_VISUAL_START_THRESHOLD", "0.14")),
            "stop_threshold": float(os.environ.get("AVP_VISUAL_STOP_THRESHOLD", "0.05")),
            "start_confirm_ms": float(os.environ.get("AVP_VISUAL_START_CONFIRM_MS", "250")),
            "stop_confirm_ms": float(os.environ.get("AVP_VISUAL_STOP_CONFIRM_MS", "900")),
            "motion_weight": float(os.environ.get("AVP_VISUAL_MOTION_WEIGHT", "6.0")),
            "open_weight": float(os.environ.get("AVP_VISUAL_OPEN_WEIGHT", "0.55")),
            "open_floor_margin": float(os.environ.get("AVP_VISUAL_OPEN_FLOOR_MARGIN", "0.012")),
            "open_range": float(os.environ.get("AVP_VISUAL_OPEN_RANGE", "0.055")),
            "log_every_n": int(os.environ.get("AVP_VISUAL_LOG_EVERY_N", "300")),
        }),
        NullSink({
            "src": "v_visual_speech",
            "group": "visual_speech",
            "name": "Visual_Speech_Sink",
        }),
    ]

    for node in nodes:
        avp.addNode(node)

    avp.group("visual_speech").startNodes()
    wait_until_done(avp, "Visual_Speech_Sink", float(os.environ.get("AVP_TIMEOUT_S", "240")))
    avp.group("visual_speech").stopNodes()


if __name__ == "__main__":
    main()
