#!/usr/bin/env python3

import ast
import json
import os
import sys
import time
from pathlib import Path

sys.path.append("../..")
import pyplumber  # pyright: ignore[reportMissingImports]
from pyplumber.mouth_tracker import FaceAnchoredMouthTrackerNode  # pyright: ignore[reportMissingImports]
from pyplumber.node import (  # pyright: ignore[reportMissingImports]
    AssumeVideoFormat,
    CudaInferYolo,
    DecVideo,
    Demux,
    DrawBBox,
    DrawBBoxLabels,
    EncVideo,
    FilterVideo,
    ForceFPS,
    InputRec,
    JoinMetadata,
    Mux,
    Output,
    Split,
)


INPUT_URL = os.environ.get("AVP_INPUT", "input.ts")
OUTPUT_URL = os.environ.get("AVP_OUTPUT", "mouth-roi-bboxes.ts")
OUTPUT_FORMAT = os.environ.get("AVP_OUTPUT_FORMAT", "mpegts")

MODEL_DIR = Path(os.environ.get("AVP_MODEL_DIR", "/home/user/tensorrt/face-recognition-1.2"))
DATA_YAML = Path(os.environ.get("AVP_DATA_YAML", str(MODEL_DIR / "data.yaml")))
FACE_ENGINE = os.environ.get("AVP_FACE_ENGINE")

MODEL_WIDTH = int(os.environ.get("AVP_MODEL_WIDTH", "960"))
MODEL_HEIGHT = int(os.environ.get("AVP_MODEL_HEIGHT", "544"))
MODEL_CONTENT_HEIGHT = int(os.environ.get("AVP_MODEL_CONTENT_HEIGHT", "540"))
MODEL_CONTENT_OFFSET_Y = int(os.environ.get("AVP_MODEL_CONTENT_OFFSET_Y", "2"))
OUTPUT_WIDTH = int(os.environ.get("AVP_OUTPUT_WIDTH", "1920"))
OUTPUT_HEIGHT = int(os.environ.get("AVP_OUTPUT_HEIGHT", "1080"))
FPS = os.environ.get("AVP_FPS", "30000/1001")
CONF_THRESH = float(os.environ.get("AVP_CONF_THRESH", "0.25"))
MAX_DET = int(os.environ.get("AVP_MAX_DET", "300"))

FACE_METADATA_KEY = os.environ.get("AVP_FACE_METADATA_KEY", "face_parts")
MOUTH_METADATA_KEY = os.environ.get("AVP_MOUTH_METADATA_KEY", "mouth_rois_v1")
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
    return [{"name": "primary", "x1": 0, "y1": 0, "x2": MODEL_WIDTH, "y2": MODEL_HEIGHT}]


def draw_labels(targets: list[dict]) -> list[str]:
    return ["M", "M (interpolated)"]


def label_colors(labels: list[str]) -> dict[str, str]:
    return {
        "M": "green",
        "M (interpolated)": "yellow",
    }


def wait_until_done(avp, node_name: str, timeout_s: float) -> None:
    node = avp.node(node_name)
    deadline = time.time() + timeout_s
    while not node.isWorking and time.time() < deadline:
        time.sleep(0.1)
    if not node.isWorking:
        raise TimeoutError(f"{node_name} did not start")
    while node.isWorking and time.time() < deadline:
        time.sleep(0.5)
    if node.isWorking:
        raise TimeoutError(f"{node_name} did not finish")
    node.join()


def main() -> None:
    names = class_names()
    engine = face_engine_path()
    targets = visual_targets()
    allowed_labels = draw_labels(targets)

    print(f"Input: {INPUT_URL}", flush=True)
    print(f"Output: {OUTPUT_URL}", flush=True)
    print(f"Face engine: {engine}", flush=True)
    print(f"Face class names: {names}", flush=True)
    print(f"Visual targets: {json.dumps(targets, sort_keys=True)}", flush=True)
    print(f"Mouth labels: {allowed_labels}", flush=True)

    avp = pyplumber.AVPlumber()
    avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')
    avp.edges.planCapacity("*", 12)

    nodes = [
        InputRec({
            "url": INPUT_URL,
            "dst": "in_mux0",
            "group": "mouth_roi_viz",
            "name": "Input",
            "initial_timeout": 20,
            "timeout": 10,
        }),
        Demux({
            "src": "in_mux0",
            "wait_for_keyframe": False,
            "routing": {"?v:0": "v_pkt"},
            "group": "mouth_roi_viz",
            "name": "Demux",
        }),
        DecVideo({
            "src": "v_pkt",
            "dst": "v_dec_cuda",
            "group": "mouth_roi_viz",
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
            "group": "mouth_roi_viz",
            "name": "Split_For_Yolo",
        }),
        FilterVideo({
            "graph": (
                f"scale_cuda=w={MODEL_WIDTH}:h={MODEL_CONTENT_HEIGHT},"
                f"pad_cuda={MODEL_WIDTH}:{MODEL_HEIGHT}:0:{MODEL_CONTENT_OFFSET_Y}"
            ),
            "src": "v_dec_for_yolo",
            "dst": "v_pre_yolo",
            "group": "mouth_roi_viz",
            "name": "Scale_Yolo",
            "dst_width": MODEL_WIDTH,
            "dst_height": MODEL_HEIGHT,
            "dst_pixel_format": "cuda",
            "hwaccel": "@gpu",
        }),
        CudaInferYolo({
            "src": "v_pre_yolo",
            "dst": "v_post_faces",
            "group": "mouth_roi_viz",
            "name": "Yolo_FaceParts",
            "input_format": "RGB",
            "conf_thresh": CONF_THRESH,
            "max_det": MAX_DET,
            "infer_every_n": 1,
            "metadata_key_detection": FACE_METADATA_KEY,
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
            "group": "mouth_roi_viz",
            "name": "Join_Face_Metadata",
        }),
        FaceAnchoredMouthTrackerNode({
            "src": "v_with_faces",
            "dst": "v_with_mouth_rois",
            "group": "mouth_roi_viz",
            "name": "Mouth_Tracker",
            "run_in_wrapper_thread": True,
            "source": SOURCE_NAME,
            "input_metadata_key": FACE_METADATA_KEY,
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
            "log_every_n": int(os.environ.get("AVP_MOUTH_LOG_EVERY_N", "300")),
        }),
        DrawBBox({
            "src": "v_with_mouth_rois",
            "dst": "v_mouth_bboxes",
            "group": "mouth_roi_viz",
            "name": "Draw_Mouth_ROI_BBoxes",
            "metadata_key": MOUTH_METADATA_KEY,
            "bbox_thickness": int(os.environ.get("AVP_BBOX_THICKNESS", "4")),
            "min_conf": 0.0,
            "allowed_labels": allowed_labels,
            "label_colors": label_colors(allowed_labels),
            "model_content_width": MODEL_WIDTH,
            "model_content_height": MODEL_HEIGHT,
            "model_content_offset_x": 0,
            "model_content_offset_y": 0,
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "debug_log_every_n": 300,
        }),
        DrawBBoxLabels({
            "src": "v_mouth_bboxes",
            "dst": "v_mouth_labels",
            "group": "mouth_roi_viz",
            "name": "Draw_Mouth_ROI_Labels",
            "metadata_key": MOUTH_METADATA_KEY,
            "label_template": "{label}",
            "allowed_labels": allowed_labels,
            "min_conf": 0.0,
            "show_predicted_labels": True,
            "show_untracked": True,
            "model_content_width": MODEL_WIDTH,
            "model_content_height": MODEL_HEIGHT,
            "model_content_offset_x": 0,
            "model_content_offset_y": 0,
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "text_color": "white",
            "background_color": "black",
            "font_scale": 1,
            "debug_log_every_n": 300,
        }),
        ForceFPS({
            "fps": FPS,
            "group": "mouth_roi_viz",
            "src": "v_mouth_labels",
            "dst": "v_mouth_fps",
        }),
        AssumeVideoFormat({
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "group": "mouth_roi_viz",
            "src": "v_mouth_fps",
            "dst": "v_preenc",
        }),
        EncVideo({
            "src": "v_preenc",
            "dst": "v_outenc",
            "group": "mouth_roi_viz",
            "name": "Video_Encode_NVENC",
            "codec": "h264_nvenc",
            "hwaccel": "@gpu",
            "options": {
                "b": "8000k",
                "maxrate": "8000k",
                "bufsize": "8000k",
                "rc": "cbr",
                "g": 75,
                "bf": 0,
                "preset": "p7",
                "tune": "ll",
                "profile": "high",
                "multipass": "disabled",
                "zerolatency": 1,
                "spatial_aq": 1,
                "temporal_aq": 1,
            },
        }),
        Mux({
            "src": ["v_outenc"],
            "dst": "mux_v",
            "group": "mouth_roi_viz",
            "ts_sort_wait": 0,
        }),
        Output({
            "format": OUTPUT_FORMAT,
            "url": OUTPUT_URL,
            "src": "mux_v",
            "group": "mouth_roi_viz",
            "name": "Output",
        }),
    ]

    for node in nodes:
        avp.addNode(node)

    avp.group("mouth_roi_viz").startNodes()
    wait_until_done(avp, "Output", float(os.environ.get("AVP_TIMEOUT_S", "240")))
    avp.group("mouth_roi_viz").stopNodes()


if __name__ == "__main__":
    main()
