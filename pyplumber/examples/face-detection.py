#!/usr/bin/env python3

import ast
import os
import sys
import time
from pathlib import Path

sys.path.append("../..")
import pyplumber  # pyright: ignore[reportMissingImports]
from pyplumber.node import (  # pyright: ignore[reportMissingImports]
    AssumeVideoFormat,
    CropMetadataCuda,
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
    PlayerTracker,
    Realtime,
    SmoothCropViewport,
    Split,
)

INPUT_URL = os.environ.get("AVP_INPUT", "input.mp4")
OUTPUT_URL = os.environ.get("AVP_OUTPUT", "face-recognition-output.mp4")
OUTPUT_FORMAT = os.environ.get("AVP_OUTPUT_FORMAT", "mpegts")

MODEL_DIR = Path(os.environ.get("AVP_MODEL_DIR", "/home/user/tensorrt/face-recognition-1.2"))
DATA_YAML = Path(os.environ.get("AVP_DATA_YAML", str(MODEL_DIR / "data.yaml")))
FACE_ENGINE = os.environ.get("AVP_FACE_ENGINE")

MODEL_WIDTH = int(os.environ.get("AVP_MODEL_WIDTH", "960"))
MODEL_HEIGHT = int(os.environ.get("AVP_MODEL_HEIGHT", "544"))
MODEL_CONTENT_HEIGHT = int(os.environ.get("AVP_MODEL_CONTENT_HEIGHT", "540"))
OUTPUT_WIDTH = int(os.environ.get("AVP_OUTPUT_WIDTH", "1920"))
OUTPUT_HEIGHT = int(os.environ.get("AVP_OUTPUT_HEIGHT", "1080"))
VIEWPORT_WIDTH = int(os.environ.get("AVP_VIEWPORT_WIDTH", "608"))
VIEWPORT_HEIGHT = int(os.environ.get("AVP_VIEWPORT_HEIGHT", "1080"))
FPS = os.environ.get("AVP_FPS", "30/1")
CONF_THRESH = float(os.environ.get("AVP_CONF_THRESH", "0.25"))
MAX_DET = int(os.environ.get("AVP_MAX_DET", "300"))
TRACKER_FRAME_RATE = int(os.environ.get("AVP_TRACKER_FRAME_RATE", "30"))

METADATA_KEY = "yolo_faces"
VIEWPORT_METADATA_KEY = "smoothed_crop_viewport_v1"
TRACKED_FACE_PARTS = ["Eye", "Nose", "Mouth"]
COLORS = [
    "green",
    "yellow",
    "cyan",
    "orange",
    "purple",
    "light_blue",
    "magenta",
    "white",
    "red",
]


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
    if names:
        return names

    print(
        f"Warning: no class names found in {DATA_YAML}. "
        "Set AVP_CLASS_NAMES='face,...' if your model has multiple classes."
    )
    return ["face"]


def face_engine_path() -> str:
    if FACE_ENGINE:
        return FACE_ENGINE

    candidates = [
        MODEL_DIR / "face-recognition_960x544.plan",
        MODEL_DIR / "face-recognition_960x544.engine",
        MODEL_DIR / "best.plan",
        MODEL_DIR / "best.engine",
        MODEL_DIR / "train" / "weights" / "best.plan",
        MODEL_DIR / "train" / "weights" / "best.engine"
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)

    return str(MODEL_DIR / "train" / "weights" / "best.engine")


def label_colors(names: list[str]) -> dict[str, str]:
    return {name: COLORS[i % len(COLORS)] for i, name in enumerate(names)}


def tracked_face_part_names(names: list[str]) -> list[str]:
    by_lower = {name.lower(): name for name in names}
    return [by_lower.get(label.lower(), label) for label in TRACKED_FACE_PARTS]


def main() -> None:
    names = class_names()
    tracked_labels = tracked_face_part_names(names)
    engine = face_engine_path()

    print(f"Face model engine: {engine}")
    print(f"Face class names: {names}")
    print(f"Tracked face parts: {tracked_labels}")

    avp = pyplumber.AVPlumber()
    avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')
    avp.edges.planCapacity("*", 7)

    nodes = [
        InputRec({
            "url": INPUT_URL,
            "dst": "in_mux0",
            "group": "in",
            "name": "input",
            "initial_timeout": 20,
            "timeout": 10,
            "loop": True,
            "auto_restart": "group",
        }),
        Demux({
            "src": "in_mux0",
            "wait_for_keyframe": False,
            "routing": {"?v:0": "v_pkt"},
            "group": "in",
            "auto_restart": "group",
        }),
        DecVideo({
            "src": "v_pkt",
            "dst": "v_dec_cuda",
            "group": "in",
            "name": "Video_Dec",
            "auto_restart": "group",
            "optional": True,
            "pixel_format": "?cuda",
            "hwaccel": "@gpu",
            "codec_map": {"h264": "h264_cuvid", "hevc": "hevc_cuvid"},
            "hwaccel_only_for_codecs": ["h264", "hevc"],
        }),
        Realtime({
            "src": "v_dec_cuda",
            "dst": "v_dec_rt",
            "group": "in",
            "auto_restart": "group",
            "set_pts": True,
        }),
        Split({
            "src": "v_dec_rt",
            "dst": ["v_dec_fullres", "v_dec_for_yolo"],
            "group": "in",
            "auto_restart": "group",
        }),
        FilterVideo({
            "graph": f"scale_cuda=w={MODEL_WIDTH}:h={MODEL_CONTENT_HEIGHT},pad_cuda={MODEL_WIDTH}:{MODEL_HEIGHT}:0:2",
            "src": "v_dec_for_yolo",
            "dst": "v_pre_yolo",
            "group": "in",
            "name": "Scale_Yolo",
            "auto_restart": "group",
            "dst_width": MODEL_WIDTH,
            "dst_height": MODEL_HEIGHT,
            "dst_pixel_format": "cuda",
            "hwaccel": "@gpu",
        }),
        CudaInferYolo({
            "src": "v_pre_yolo",
            "dst": "v_post_faces",
            "group": "in",
            "name": "Yolo_Faces",
            "auto_restart": "group",
            "input_format": "RGB",
            "conf_thresh": CONF_THRESH,
            "max_det": MAX_DET,
            "infer_every_n": 1,
            "metadata_key_detection": METADATA_KEY,
            "debug_log_metadata": True,
            "debug_log_every_n": 30,
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
            "dst": "v_fullres_with_faces",
            "group": "in",
            "auto_restart": "group",
        }),
        PlayerTracker({
            "src": "v_fullres_with_faces",
            "dst": "v_tracked_faces",
            "group": "in",
            "metadata_key": METADATA_KEY,
            "target_labels": tracked_labels,
            "frame_rate": TRACKER_FRAME_RATE,
            "track_buffer": 90,
            "predict_on_empty": True,
            "track_thresh": 0.2,
            "high_thresh": 0.85,
            "debug_log_every_n": 30,
            "auto_restart": "group",
        }),
        DrawBBox({
            "src": "v_tracked_faces",
            "dst": "v_bboxes_drawn",
            "group": "in",
            "name": "Draw_Face_BBoxes",
            "metadata_key": METADATA_KEY,
            "bbox_thickness": 2,
            "min_conf": CONF_THRESH,
            "allowed_labels": names,
            "label_colors": label_colors(names),
            "model_content_width": MODEL_WIDTH,
            "model_content_height": MODEL_HEIGHT,
            "model_content_offset_x": 0,
            "model_content_offset_y": 0,
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "debug_log_every_n": 30,
            "auto_restart": "group",
        }),
        DrawBBoxLabels({
            "src": "v_bboxes_drawn",
            "dst": "v_labels_drawn",
            "group": "in",
            "name": "Draw_Face_Labels",
            "metadata_key": METADATA_KEY,
            "label_template": "{label}",
            "allowed_labels": names,
            "min_conf": CONF_THRESH,
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
            "debug_log_every_n": 30,
            "auto_restart": "group",
        }),
        SmoothCropViewport({
            "src": "v_labels_drawn",
            "dst": "v_viewport_md",
            "group": "in",
            "name": "Smooth_Crop_Viewport",
            "metadata_key_ins": [METADATA_KEY],
            "metadata_key_out": VIEWPORT_METADATA_KEY,
            "viewport_dst_width": VIEWPORT_WIDTH,
            "viewport_dst_height": VIEWPORT_HEIGHT,
            "model_content_width": MODEL_WIDTH,
            "model_content_height": MODEL_HEIGHT,
            "model_content_offset_x": 0,
            "model_content_offset_y": 0,
            "min_conf": CONF_THRESH,
            "allowed_labels": tracked_labels,
            "focus_mode": "label_priority",
            "label_priority": tracked_labels,
            "lost_target": "hold_last",
            "filter_type": "kalman",
            "lookahead_frames": 8,
            "hold_enabled": True,
            "max_velocity_px_per_s": 2400,
            "max_acceleration_px_per_s2": 80000,
            "viewport_marker_half_extent": 16,
            "debug_log_every_n": 30,
            "viewport_fit": "contain",
            "viewport_contain_margin_px": 24,
            "auto_restart": "group",
        }),
        DrawBBox({
            "src": "v_viewport_md",
            "dst": "v_viewport_drawn",
            "group": "in",
            "name": "Draw_Viewport",
            "metadata_key": VIEWPORT_METADATA_KEY,
            "bbox_thickness": 2,
            "min_conf": 0.0,
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "debug_log_every_n": 30,
            "auto_restart": "group",
        }),
        ForceFPS({
            "fps": FPS,
            "group": "in",
            "src": "v_viewport_drawn",
            "dst": "v_labels_fps",
            "auto_restart": "group",
        }),
        FilterVideo({
            "graph": f"scale_cuda=w={OUTPUT_WIDTH}:h={OUTPUT_HEIGHT}",
            "src": "v_labels_fps",
            "dst": "v_scaled_out",
            "group": "in",
            "name": "Scale_Output",
            "auto_restart": "group",
            "dst_width": OUTPUT_WIDTH,
            "dst_height": OUTPUT_HEIGHT,
            "dst_pixel_format": "cuda",
            "hwaccel": "@gpu",
        }),
        CropMetadataCuda({
            "src": "v_scaled_out",
            "dst": "v_cropped_cuda",
            "group": "in",
            "name": "Crop_Viewport",
            "metadata_key": VIEWPORT_METADATA_KEY,
            "debug_log_every_n": 30,
            "auto_restart": "group",
        }),
        AssumeVideoFormat({
            "width": VIEWPORT_WIDTH,
            "height": VIEWPORT_HEIGHT,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "group": "in",
            "src": "v_cropped_cuda",
            "dst": "v_preenc",
            "auto_restart": "group",
        }),
        EncVideo({
            "src": "v_preenc",
            "dst": "v_outenc",
            "group": "in",
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
            "auto_restart": "group",
        }),
        Mux({
            "src": ["v_outenc"],
            "dst": "mux_v",
            "group": "in",
            "ts_sort_wait": 0,
            "auto_restart": "group",
        }),
        Output({
            "format": OUTPUT_FORMAT,
            "url": OUTPUT_URL,
            "src": "mux_v",
            "group": "in",
            "auto_restart": "group",
        }),
    ]

    for node in nodes:
        avp.addNode(node)

    print("Starting group: in")
    avp.group("in").startNodes()

    while True:
        time.sleep(1)
        avp.heartbeat()


if __name__ == "__main__":
    main()
