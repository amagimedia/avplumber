#!/usr/bin/env python3

import ast
import json
import os
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

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
    PlayerTracker,
    PythonNode,
    SmoothCropViewport,
    Split,
)
from pyplumber.visual_utils import valid_pts  # pyright: ignore[reportMissingImports]


INPUT_URL = os.environ.get("AVP_INPUT", "shaky-samples/talkshow-Genaro-12m00s-20s.ts")
OUTPUT_URL = os.environ.get("AVP_OUTPUT", "shaky-samples/talkshow-Genaro-12m00s-debug.ts")
OUTPUT_FORMAT = os.environ.get("AVP_OUTPUT_FORMAT", "mpegts")

MODEL_DIR = Path(os.environ.get("AVP_MODEL_DIR", "models/face-recognition-1.2"))
DATA_YAML = Path(os.environ.get("AVP_DATA_YAML", str(MODEL_DIR / "data.yaml")))
FACE_ENGINE = os.environ.get("AVP_FACE_ENGINE")

MODEL_WIDTH = int(os.environ.get("AVP_MODEL_WIDTH", "960"))
MODEL_HEIGHT = int(os.environ.get("AVP_MODEL_HEIGHT", "544"))
MODEL_CONTENT_HEIGHT = int(os.environ.get("AVP_MODEL_CONTENT_HEIGHT", "540"))
MODEL_CONTENT_OFFSET_Y = int(os.environ.get("AVP_MODEL_CONTENT_OFFSET_Y", "2"))
OUTPUT_WIDTH = int(os.environ.get("AVP_OUTPUT_WIDTH", "1920"))
OUTPUT_HEIGHT = int(os.environ.get("AVP_OUTPUT_HEIGHT", "1080"))
FACE_CROP_W = int(os.environ.get("AVP_FACE_CROP_W", "608"))
FACE_CROP_H = int(os.environ.get("AVP_FACE_CROP_H", "1080"))
FPS = os.environ.get("AVP_FPS", "30000/1001")
CONF_THRESH = float(os.environ.get("AVP_CONF_THRESH", "0.25"))
MAX_DET = int(os.environ.get("AVP_MAX_DET", "300"))

FACE_METADATA_KEY = os.environ.get("AVP_FACE_METADATA_KEY", "face_parts")
MOUTH_METADATA_KEY = os.environ.get("AVP_MOUTH_METADATA_KEY", "mouth_rois_v1")
LIVE_VIEWPORT_KEY = os.environ.get("AVP_LIVE_VIEWPORT_KEY", "live_viewport_v1")
FACE_ONLY_VIEWPORT_KEY = os.environ.get("AVP_FACE_ONLY_VIEWPORT_KEY", "face_only_viewport_v1")
LIVE_VIEWPORT_BOX_KEY = os.environ.get("AVP_LIVE_VIEWPORT_BOX_KEY", "live_viewport_bbox_v1")
FACE_ONLY_VIEWPORT_BOX_KEY = os.environ.get("AVP_FACE_ONLY_VIEWPORT_BOX_KEY", "face_only_viewport_bbox_v1")
GROUP = "genaro_debug"


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
    return _parse_yaml_names(DATA_YAML) or ["Eye", "Face", "MakeUp", "Mouth", "Nose", "Tooth", "Topping"]


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


class ViewportBBoxNode(PythonNode):
    def __init__(self, args: dict):
        super().__init__({"data_type": "VideoFrame"} | args)
        p = self.parameters
        self.viewport_metadata_key = str(p["viewport_metadata_key"])
        self.output_metadata_key = str(p["output_metadata_key"])
        self.label = str(p["label"])
        self.model_width = int(p.get("model_width", MODEL_WIDTH))
        self.model_height = int(p.get("model_height", MODEL_HEIGHT))
        self.frame_width = int(p.get("frame_width", OUTPUT_WIDTH))
        self.frame_height = int(p.get("frame_height", OUTPUT_HEIGHT))

    def process(self):
        frame = self._src.get()
        if frame is None:
            return
        if not valid_pts(frame):
            self._dst.enqueue(frame)
            return

        detections = []
        try:
            metadata = json.loads(str(frame.metadata[self.viewport_metadata_key]))
            viewport_box = metadata.get("viewport_bbox")
            viewport_w = float(metadata.get("viewport_dst_width", FACE_CROP_W))
            viewport_h = float(metadata.get("viewport_dst_height", FACE_CROP_H))
            frame_w = float(metadata.get("full_frame_width", self.frame_width))
            frame_h = float(metadata.get("full_frame_height", self.frame_height))
            if isinstance(viewport_box, list) and len(viewport_box) >= 4:
                cx = (float(viewport_box[0]) + float(viewport_box[2])) * 0.5
                cy = (float(viewport_box[1]) + float(viewport_box[3])) * 0.5
                x1 = max(0.0, min(cx - viewport_w * 0.5, frame_w - viewport_w))
                y1 = max(0.0, min(cy - viewport_h * 0.5, frame_h - viewport_h))
                x2 = min(frame_w, x1 + viewport_w)
                y2 = min(frame_h, y1 + viewport_h)
                sx = self.model_width / max(frame_w, 1.0)
                sy = self.model_height / max(frame_h, 1.0)
                detections.append({
                    "label": self.label,
                    "conf": 1.0,
                    "xyxy": [
                        round(x1 * sx, 3),
                        round(y1 * sy, 3),
                        round(x2 * sx, 3),
                        round(y2 * sy, 3),
                    ],
                })
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            detections = []

        frame.metadata[self.output_metadata_key] = json.dumps({
            "version": 1,
            "coord_space": "model",
            "model_width": self.model_width,
            "model_height": self.model_height,
            "detections": detections,
        }, sort_keys=True)
        self._dst.enqueue(frame)


def label_colors() -> dict[str, str]:
    return {
        "Eye": "yellow",
        "Face": "cyan",
        "MakeUp": "white",
        "Mouth": "green",
        "Nose": "magenta",
        "Tooth": "white",
        "Topping": "orange",
        "M": "green",
        "M (interpolated)": "yellow",
        "LIVE_VIEWPORT": "red",
        "FACE_ONLY_VIEWPORT": "cyan",
    }


def draw_bbox_node(name: str, src: str, dst: str, metadata_key: str, allowed_labels: list[str], thickness: int) -> DrawBBox:
    return DrawBBox({
        "src": src,
        "dst": dst,
        "group": GROUP,
        "name": name,
        "metadata_key": metadata_key,
        "bbox_thickness": thickness,
        "min_conf": 0.0,
        "allowed_labels": allowed_labels,
        "label_colors": label_colors(),
        "model_content_width": MODEL_WIDTH,
        "model_content_height": MODEL_HEIGHT,
        "model_content_offset_x": 0,
        "model_content_offset_y": 0,
        "width": OUTPUT_WIDTH,
        "height": OUTPUT_HEIGHT,
        "pixel_format": "cuda",
        "real_pixel_format": "nv12",
        "debug_log_every_n": 300,
    })


def draw_label_node(name: str, src: str, dst: str, metadata_key: str, allowed_labels: list[str]) -> DrawBBoxLabels:
    return DrawBBoxLabels({
        "src": src,
        "dst": dst,
        "group": GROUP,
        "name": name,
        "metadata_key": metadata_key,
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
    })


def main() -> None:
    names = class_names()
    engine = face_engine_path()

    print(f"Input: {INPUT_URL}", flush=True)
    print(f"Output: {OUTPUT_URL}", flush=True)
    print(f"Face engine: {engine}", flush=True)
    print(f"Face class names: {names}", flush=True)
    print("Overlay legend: raw face-part boxes, tracked Face boxes, mouth ROI, red live viewport, cyan Face-only viewport", flush=True)

    avp = pyplumber.AVPlumber()
    avp.executeCommandsFromString('hwaccel.init { "name": "@gpu", "type": "cuda" }')
    avp.edges.planCapacity("*", 16)

    nodes = [
        InputRec({
            "url": INPUT_URL,
            "dst": "in_mux0",
            "group": GROUP,
            "name": "Input",
            "initial_timeout": 20,
            "timeout": 10,
        }),
        Demux({
            "src": "in_mux0",
            "wait_for_keyframe": False,
            "routing": {"?v:0": "v_pkt"},
            "group": GROUP,
            "name": "Demux",
        }),
        DecVideo({
            "src": "v_pkt",
            "dst": "v_dec_cuda",
            "group": GROUP,
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
            "group": GROUP,
            "name": "Split_For_Yolo",
        }),
        FilterVideo({
            "graph": (
                f"scale_cuda=w={MODEL_WIDTH}:h={MODEL_CONTENT_HEIGHT},"
                f"pad_cuda=w={MODEL_WIDTH}:h={MODEL_HEIGHT}:x=0:y={MODEL_CONTENT_OFFSET_Y}"
            ),
            "src": "v_dec_for_yolo",
            "dst": "v_pre_yolo",
            "group": GROUP,
            "name": "Scale_Yolo",
            "dst_width": MODEL_WIDTH,
            "dst_height": MODEL_HEIGHT,
            "dst_pixel_format": "cuda",
            "hwaccel": "@gpu",
        }),
        CudaInferYolo({
            "src": "v_pre_yolo",
            "dst": "v_post_faces",
            "group": GROUP,
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
            "group": GROUP,
            "name": "Join_Face_Metadata",
        }),
        draw_bbox_node("Draw_Raw_FacePart_BBoxes", "v_with_faces", "v_raw_boxes", FACE_METADATA_KEY, names, 2),
        draw_label_node("Draw_Raw_FacePart_Labels", "v_raw_boxes", "v_raw_labels", FACE_METADATA_KEY, names),
        PlayerTracker({
            "src": "v_raw_labels",
            "dst": "v_tracked_faces",
            "group": GROUP,
            "name": "Track_Face_Only",
            "metadata_key": FACE_METADATA_KEY,
            "target_labels": ["Face"],
            "frame_rate": float(os.environ.get("AVP_TRACKER_FRAME_RATE", "29.97")),
            "track_buffer": 90,
            "predict_on_empty": True,
            "track_thresh": 0.2,
            "high_thresh": 0.85,
            "debug_log_every_n": 300,
        }),
        draw_bbox_node("Draw_Tracked_Face_BBoxes", "v_tracked_faces", "v_tracked_boxes", FACE_METADATA_KEY, ["Face"], 4),
        draw_label_node("Draw_Tracked_Face_Labels", "v_tracked_boxes", "v_tracked_labels", FACE_METADATA_KEY, ["Face"]),
        SmoothCropViewport({
            "src": "v_tracked_labels",
            "dst": "v_live_viewport",
            "group": GROUP,
            "name": "Smooth_LiveStyle_Viewport",
            "metadata_key_ins": [FACE_METADATA_KEY],
            "metadata_key_out": LIVE_VIEWPORT_KEY,
            "viewport_dst_width": FACE_CROP_W,
            "viewport_dst_height": FACE_CROP_H,
            "focus_mode": "label_priority",
            "filter_type": "kalman",
            "lost_target": "hold_last",
            "debug_log_every_n": 300,
        }),
        ViewportBBoxNode({
            "src": "v_live_viewport",
            "dst": "v_live_viewport_box",
            "group": GROUP,
            "name": "Live_Viewport_As_BBox",
            "viewport_metadata_key": LIVE_VIEWPORT_KEY,
            "output_metadata_key": LIVE_VIEWPORT_BOX_KEY,
            "label": "LIVE_VIEWPORT",
        }),
        SmoothCropViewport({
            "src": "v_live_viewport_box",
            "dst": "v_face_only_viewport",
            "group": GROUP,
            "name": "Smooth_FaceOnly_Viewport",
            "metadata_key_ins": [FACE_METADATA_KEY],
            "metadata_key_out": FACE_ONLY_VIEWPORT_KEY,
            "viewport_dst_width": FACE_CROP_W,
            "viewport_dst_height": FACE_CROP_H,
            "focus_mode": "label_priority",
            "filter_type": "kalman",
            "lost_target": "hold_last",
            "allowed_labels": ["Face"],
            "label_priority": ["Face"],
            "debug_log_every_n": 300,
        }),
        ViewportBBoxNode({
            "src": "v_face_only_viewport",
            "dst": "v_face_only_viewport_box",
            "group": GROUP,
            "name": "FaceOnly_Viewport_As_BBox",
            "viewport_metadata_key": FACE_ONLY_VIEWPORT_KEY,
            "output_metadata_key": FACE_ONLY_VIEWPORT_BOX_KEY,
            "label": "FACE_ONLY_VIEWPORT",
        }),
        FaceAnchoredMouthTrackerNode({
            "src": "v_face_only_viewport_box",
            "dst": "v_mouth_rois",
            "group": GROUP,
            "name": "Mouth_Tracker",
            "run_in_wrapper_thread": True,
            "source": "genaro",
            "input_metadata_key": FACE_METADATA_KEY,
            "output_metadata_key": MOUTH_METADATA_KEY,
            "targets": [{"name": "primary", "x1": 0, "y1": 0, "x2": MODEL_WIDTH, "y2": MODEL_HEIGHT}],
            "min_conf": CONF_THRESH,
            "log_every_n": 300,
        }),
        draw_bbox_node(
            "Draw_Live_Viewport_BBox",
            "v_mouth_rois",
            "v_live_viewport_drawn",
            LIVE_VIEWPORT_BOX_KEY,
            ["LIVE_VIEWPORT"],
            5,
        ),
        draw_label_node(
            "Draw_Live_Viewport_Label",
            "v_live_viewport_drawn",
            "v_live_viewport_label",
            LIVE_VIEWPORT_BOX_KEY,
            ["LIVE_VIEWPORT"],
        ),
        draw_bbox_node(
            "Draw_FaceOnly_Viewport_BBox",
            "v_live_viewport_label",
            "v_face_viewport_drawn",
            FACE_ONLY_VIEWPORT_BOX_KEY,
            ["FACE_ONLY_VIEWPORT"],
            5,
        ),
        draw_label_node(
            "Draw_FaceOnly_Viewport_Label",
            "v_face_viewport_drawn",
            "v_face_viewport_label",
            FACE_ONLY_VIEWPORT_BOX_KEY,
            ["FACE_ONLY_VIEWPORT"],
        ),
        draw_bbox_node(
            "Draw_Mouth_ROI_BBoxes",
            "v_face_viewport_label",
            "v_mouth_boxes",
            MOUTH_METADATA_KEY,
            ["M", "M (interpolated)"],
            4,
        ),
        draw_label_node(
            "Draw_Mouth_ROI_Labels",
            "v_mouth_boxes",
            "v_mouth_labels",
            MOUTH_METADATA_KEY,
            ["M", "M (interpolated)"],
        ),
        ForceFPS({
            "fps": FPS,
            "group": GROUP,
            "src": "v_mouth_labels",
            "dst": "v_fps",
        }),
        AssumeVideoFormat({
            "width": OUTPUT_WIDTH,
            "height": OUTPUT_HEIGHT,
            "pixel_format": "cuda",
            "real_pixel_format": "nv12",
            "group": GROUP,
            "src": "v_fps",
            "dst": "v_preenc",
        }),
        EncVideo({
            "src": "v_preenc",
            "dst": "v_outenc",
            "group": GROUP,
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
            "group": GROUP,
            "ts_sort_wait": 0,
        }),
        Output({
            "format": OUTPUT_FORMAT,
            "url": OUTPUT_URL,
            "src": "mux_v",
            "group": GROUP,
            "name": "Output",
        }),
    ]

    for node in nodes:
        avp.addNode(node)

    avp.group(GROUP).startNodes()
    wait_until_done(avp, "Output", float(os.environ.get("AVP_TIMEOUT_S", "240")))
    avp.group(GROUP).stopNodes()


if __name__ == "__main__":
    main()
