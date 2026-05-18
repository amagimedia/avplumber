"""Debug overlay helper nodes for visual speech diagnostics."""

from __future__ import annotations

import json

from pyplumber.audio_vad import Speaker
from pyplumber.node import PythonNode
from pyplumber.visual_utils import valid_pts

from .config import FACE_CROP_H, FACE_CROP_W, FACE_MODEL_H, FACE_MODEL_W


class StaticViewportMetadataNode(PythonNode):
    """Attach fixed crop dimensions so CropMetadataCuda uses its center fallback."""

    def __init__(self, args: dict):
        super().__init__({"data_type": "VideoFrame"} | args)
        p = self.parameters
        self.metadata_key = str(p["metadata_key"])
        self._metadata_json = json.dumps({
            "viewport_dst_width": int(p["viewport_dst_width"]),
            "viewport_dst_height": int(p["viewport_dst_height"]),
        }, sort_keys=True)

    def process(self):
        frame = self._src.get()
        if frame is None:
            return
        frame.metadata[self.metadata_key] = self._metadata_json
        self._dst.enqueue(frame)


class SpeakingStatusLabelNode(PythonNode):
    """Convert registry state into small A/V labels for the debug overlay."""

    def __init__(self, args: dict, index: int, registry: Speaker):
        super().__init__({"data_type": "VideoFrame"} | args)
        p = self.parameters
        self.index = index
        self.registry = registry
        self.visual_metadata_key = str(p["visual_metadata_key"])
        self.viewport_metadata_key = str(p["viewport_metadata_key"])
        self.output_metadata_key = str(p["output_metadata_key"])
        self.model_width = int(p.get("model_width", FACE_MODEL_W))
        self.model_height = int(p.get("model_height", FACE_MODEL_H))
        self.frame_width = int(p.get("frame_width", 1920))
        self.frame_height = int(p.get("frame_height", 1080))
        self.static_face_crop = bool(p.get("static_face_crop", False))
        self.fallback_x = float(p.get("fallback_x", 24))
        self.fallback_y = float(p.get("fallback_y", 8))

    def _crop_origin(self, frame) -> tuple[float, float]:
        if self.static_face_crop:
            return (
                max(0.0, (self.frame_width - FACE_CROP_W) * 0.5),
                max(0.0, (self.frame_height - FACE_CROP_H) * 0.5),
            )
        try:
            metadata = json.loads(str(frame.metadata[self.viewport_metadata_key]))
            viewport_box = metadata.get("viewport_bbox")
            viewport_w = float(metadata.get("viewport_dst_width", FACE_CROP_W))
            viewport_h = float(metadata.get("viewport_dst_height", FACE_CROP_H))
            frame_w = float(metadata.get("full_frame_width", self.frame_width))
            frame_h = float(metadata.get("full_frame_height", self.frame_height))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            return self.fallback_x, self.fallback_y

        if not isinstance(viewport_box, list) or len(viewport_box) < 4:
            return self.fallback_x, self.fallback_y
        try:
            cx = (float(viewport_box[0]) + float(viewport_box[2])) * 0.5
            cy = (float(viewport_box[1]) + float(viewport_box[3])) * 0.5
        except (TypeError, ValueError):
            return self.fallback_x, self.fallback_y

        x = max(0.0, min(cx - viewport_w * 0.5, frame_w - viewport_w))
        y = max(0.0, min(cy - viewport_h * 0.5, frame_h - viewport_h))
        return x, y

    def _label_box(self, frame, offset_y: float) -> list[float]:
        crop_x, crop_y = self._crop_origin(frame)
        x = crop_x + self.fallback_x
        y = crop_y + self.fallback_y + offset_y
        return [x, y, x + 64, y + 20]

    def _detection(self, frame, label: str, offset_y: float) -> dict:
        return {
            "label": label,
            "conf": 1.0,
            "xyxy": [round(float(v), 3) for v in self._label_box(frame, offset_y)],
        }

    def process(self):
        frame = self._src.get()
        if frame is None:
            return
        if not valid_pts(frame) or frame.width <= 0 or frame.height <= 0:
            return

        detections = []
        entry = self.registry.get(self.index)
        video_speaking = bool(entry.visual_speaking) if entry is not None else False
        try:
            metadata = json.loads(str(frame.metadata[self.visual_metadata_key]))
        except (KeyError, json.JSONDecodeError):
            metadata = {}

        for target in metadata.get("targets", []):
            if isinstance(target, dict):
                video_speaking = video_speaking or bool(target.get("speaking"))

        if video_speaking:
            detections.append(self._detection(frame, "V", 0))

        if entry is not None and entry.speaking:
            detections.append(self._detection(frame, "A", 26))

        frame.metadata[self.output_metadata_key] = json.dumps({
            "version": 1,
            "coord_space": "model",
            "model_width": self.model_width,
            "model_height": self.model_height,
            "detections": detections,
        }, sort_keys=True)
        self._dst.enqueue(frame)


# Visual speech metadata key names (per-input, so no collisions on the same frame).
VS_MOUTH_KEY_PREFIX = "vs_mouth_rois"
VS_VISUAL_KEY_PREFIX = "vs_visual_speech"
VS_SPEAKING_LABEL_KEY_PREFIX = "vs_speaking_labels"
DEBUG_MOUTH_LABELS = ["M", "M (interpolated)"]
DEBUG_MOUTH_LABEL_COLORS = {
    "M": "green",
    "M (interpolated)": "yellow",
}
DEBUG_VIDEO_SPEAKING_LABELS = ["V"]
DEBUG_AUDIO_SPEAKING_LABELS = ["A"]
