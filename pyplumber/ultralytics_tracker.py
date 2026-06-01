from __future__ import annotations

import ctypes
import json
import time
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .node import PythonNode


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _as_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def _as_optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _as_list(value: Any) -> list[Any]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, Iterable):
        return list(value)
    return [value]


def _format_name(frame) -> str:
    fmt = getattr(frame, "format", None)
    if fmt is None:
        fmt = getattr(frame, "pixel_format", None)
    if fmt is None:
        return ""
    name = getattr(fmt, "name", None)
    if callable(name):
        name = name()
    if name:
        return str(name).lower()
    return str(fmt).lower()


class UltralyticsByteTrackNode(PythonNode):
    """Run Ultralytics YOLO tracking on CPU-readable video frames.

    The node writes AVPlumber's standard YOLO-style detection metadata onto the
    input frame and forwards that same frame downstream. It is intended as a
    reusable Python/prototyping node; use AVPlumber's native cuda_infer_yolo and
    player_tracker nodes for the high-throughput CUDA/TensorRT path.
    """

    def __init__(self, args: dict[str, Any]):
        params = {"data_type": "VideoFrame"} | args
        super().__init__(params)

        p = self.parameters
        weights = p.get("weights", p.get("model", ""))
        if not weights:
            raise ValueError("UltralyticsByteTrackNode requires 'weights' or 'model'")

        self.weights = str(weights)
        self.metadata_key = str(p.get("metadata_key", "ultralytics_tracks"))
        self.count_metadata_key = str(p.get("count_metadata_key", ""))
        self.status_metadata_key = str(p.get("status_metadata_key", ""))
        self.device = p.get("device", None)
        self.tracker = str(p.get("tracker", "bytetrack.yaml"))
        self.conf = float(p.get("conf", 0.25))
        self.iou = _as_optional_float(p.get("iou", None))
        self.imgsz = p.get("imgsz", None)
        self.max_det = _as_optional_int(p.get("max_det", None))
        self.half = p.get("half", None)
        self.agnostic_nms = p.get("agnostic_nms", None)
        self.verbose = _as_bool(p.get("verbose", False))
        self.persist = _as_bool(p.get("persist", True), True)
        self.include_untracked = _as_bool(p.get("include_untracked", True), True)
        self.emit_empty = _as_bool(p.get("emit_empty", True), True)
        self.error_policy = str(p.get("error_policy", "raise")).lower()
        self.debug_log_every_n = int(p.get("debug_log_every_n", 0) or 0)
        self.source_name = str(p.get("source", p.get("name", "ultralytics_bytetrack")))
        self.model_index = int(p.get("model_index", 0) or 0)

        class_specs = _as_list(p.get("classes", p.get("target_classes", None)))
        label_specs = _as_list(p.get("target_labels", p.get("allowed_labels", None)))
        self._class_specs = class_specs + label_specs
        self._label_overrides = self._parse_label_overrides(p.get("label_map", {}))
        self._model = None
        self._classes: list[int] | None = None
        self._frame_index = 0
        self._last_detection_count = 0
        self._last_track_count = 0
        self._total_ms = 0.0

        if self.error_policy not in {"raise", "passthrough"}:
            raise ValueError("error_policy must be 'raise' or 'passthrough'")

    @staticmethod
    def _parse_label_overrides(value: Any) -> dict[int, str]:
        if not value:
            return {}
        if not isinstance(value, Mapping):
            raise ValueError("label_map must be an object mapping class id to label")
        overrides: dict[int, str] = {}
        for key, label in value.items():
            try:
                cls = int(key)
            except (TypeError, ValueError):
                continue
            overrides[cls] = str(label)
        return overrides

    def _ensure_model(self):
        if self._model is not None:
            return self._model
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "UltralyticsByteTrackNode requires the optional 'ultralytics' package"
            ) from exc

        self._model = YOLO(self.weights)
        self._classes = self._resolve_classes()
        return self._model

    def _model_names(self) -> dict[int, str]:
        model = self._model
        names = getattr(model, "names", {}) if model is not None else {}
        if isinstance(names, Mapping):
            return {int(k): str(v) for k, v in names.items()}
        return {idx: str(name) for idx, name in enumerate(names)}

    def _resolve_classes(self) -> list[int] | None:
        if not self._class_specs:
            return None

        names = self._model_names()
        by_label = {label.lower(): cls for cls, label in names.items()}
        resolved: list[int] = []
        for spec in self._class_specs:
            if isinstance(spec, int):
                resolved.append(int(spec))
                continue
            text = str(spec).strip()
            if text == "":
                continue
            try:
                resolved.append(int(text))
                continue
            except ValueError:
                pass
            cls = by_label.get(text.lower())
            if cls is None:
                raise ValueError(f"unknown Ultralytics class label: {text}")
            resolved.append(cls)
        return sorted(set(resolved))

    def _plane_u8(self, frame, plane: int, rows: int) -> np.ndarray | None:
        linesize = frame.linesize
        data_ptr = frame.data_ptr
        stride = int(linesize[plane]) if len(linesize) > plane else 0
        ptr = int(data_ptr[plane]) if len(data_ptr) > plane else 0
        if rows <= 0 or stride <= 0 or ptr <= 0:
            return None
        buf_type = ctypes.c_uint8 * (stride * rows)
        buf = buf_type.from_address(ptr)
        return np.ctypeslib.as_array(buf).reshape(rows, stride)

    def _packed_frame_to_bgr(self, frame, channels: int, order: str) -> np.ndarray:
        width = int(frame.width)
        height = int(frame.height)
        plane = self._plane_u8(frame, 0, height)
        if plane is None:
            raise RuntimeError("invalid packed frame plane")
        row_bytes = width * channels
        if plane.shape[1] < row_bytes:
            raise RuntimeError("packed frame stride is smaller than visible row")
        img = plane[:, :row_bytes].reshape(height, width, channels)
        if channels == 1:
            return np.ascontiguousarray(np.repeat(img, 3, axis=2))
        if order == "rgb":
            img = img[:, :, ::-1]
        return np.ascontiguousarray(img)

    def _yuv420p_to_bgr(self, frame) -> np.ndarray:
        width = int(frame.width)
        height = int(frame.height)
        uv_w = (width + 1) // 2
        uv_h = (height + 1) // 2

        y_plane = self._plane_u8(frame, 0, height)
        u_plane = self._plane_u8(frame, 1, uv_h)
        v_plane = self._plane_u8(frame, 2, uv_h)
        if y_plane is None or u_plane is None or v_plane is None:
            raise RuntimeError("invalid yuv420p frame planes")

        y = y_plane[:, :width].astype(np.float32)
        u = u_plane[:, :uv_w].astype(np.float32)
        v = v_plane[:, :uv_w].astype(np.float32)
        u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)[:height, :width]
        v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)[:height, :width]
        d = u - 128.0
        e = v - 128.0
        r = y + 1.4020 * e
        g = y - 0.3441 * d - 0.7141 * e
        b = y + 1.7720 * d
        return np.ascontiguousarray(np.clip(np.stack((b, g, r), axis=2), 0, 255).astype(np.uint8))

    def _nv12_to_bgr(self, frame) -> np.ndarray:
        width = int(frame.width)
        height = int(frame.height)
        uv_w = (width + 1) // 2
        uv_h = (height + 1) // 2

        y_plane = self._plane_u8(frame, 0, height)
        uv_plane = self._plane_u8(frame, 1, uv_h)
        if y_plane is None or uv_plane is None:
            raise RuntimeError("invalid nv12 frame planes")
        if uv_plane.shape[1] < uv_w * 2:
            raise RuntimeError("nv12 chroma stride is smaller than visible row")

        y = y_plane[:, :width].astype(np.float32)
        uv = uv_plane[:, : uv_w * 2]
        u = uv[:, 0::2].astype(np.float32)
        v = uv[:, 1::2].astype(np.float32)
        u = np.repeat(np.repeat(u, 2, axis=0), 2, axis=1)[:height, :width]
        v = np.repeat(np.repeat(v, 2, axis=0), 2, axis=1)[:height, :width]
        d = u - 128.0
        e = v - 128.0
        r = y + 1.4020 * e
        g = y - 0.3441 * d - 0.7141 * e
        b = y + 1.7720 * d
        return np.ascontiguousarray(np.clip(np.stack((b, g, r), axis=2), 0, 255).astype(np.uint8))

    def _frame_to_bgr(self, frame) -> np.ndarray:
        fmt = _format_name(frame)
        if fmt in {"rgb24", "rgb"}:
            return self._packed_frame_to_bgr(frame, 3, "rgb")
        if fmt in {"bgr24", "bgr"}:
            return self._packed_frame_to_bgr(frame, 3, "bgr")
        if fmt in {"gray", "gray8", "y8"}:
            return self._packed_frame_to_bgr(frame, 1, "gray")
        if fmt in {"yuv420p", "yuvj420p"}:
            return self._yuv420p_to_bgr(frame)
        if fmt == "nv12":
            return self._nv12_to_bgr(frame)
        raise RuntimeError(
            "unsupported frame format for UltralyticsByteTrackNode: "
            f"{fmt or '<unknown>'}; insert filter_video format=rgb24, bgr24, "
            "yuv420p, or nv12 before this node"
        )

    def _track_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "conf": self.conf,
            "persist": self.persist,
            "tracker": self.tracker,
            "verbose": self.verbose,
        }
        if self.device is not None and self.device != "":
            kwargs["device"] = self.device
        if self.iou is not None:
            kwargs["iou"] = self.iou
        if self.imgsz is not None and self.imgsz != "":
            kwargs["imgsz"] = self.imgsz
        if self.max_det is not None:
            kwargs["max_det"] = self.max_det
        if self.half is not None:
            kwargs["half"] = _as_bool(self.half)
        if self.agnostic_nms is not None:
            kwargs["agnostic_nms"] = _as_bool(self.agnostic_nms)
        if self._classes is not None:
            kwargs["classes"] = self._classes
        return kwargs

    def _label_for_class(self, cls: int) -> str:
        if cls in self._label_overrides:
            return self._label_overrides[cls]
        return self._model_names().get(cls, str(cls))

    def _serialize_results(self, result, width: int, height: int, elapsed_ms: float) -> dict[str, Any]:
        detections: list[dict[str, Any]] = []
        boxes = getattr(result, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy() if boxes.conf is not None else np.zeros((len(xyxy),), dtype=np.float32)
            classes = boxes.cls.cpu().numpy() if boxes.cls is not None else np.full((len(xyxy),), -1, dtype=np.float32)
            ids = boxes.id.cpu().numpy() if boxes.id is not None else None
            for idx, box in enumerate(xyxy):
                track_id = int(ids[idx]) if ids is not None else None
                if track_id is None and not self.include_untracked:
                    continue
                cls = int(classes[idx]) if idx < len(classes) else -1
                det = {
                    "xyxy": [float(box[0]), float(box[1]), float(box[2]), float(box[3])],
                    "conf": float(confs[idx]) if idx < len(confs) else 0.0,
                    "cls": cls,
                    "label": self._label_for_class(cls),
                    "model_index": self.model_index,
                }
                if track_id is not None:
                    det["track_id"] = track_id
                detections.append(det)

        self._last_detection_count = len(detections)
        self._last_track_count = len({d["track_id"] for d in detections if "track_id" in d})

        return {
            "schema": "yolo_detections_v1",
            "source": self.source_name,
            "coord_space": "model",
            "model_width": width,
            "model_height": height,
            "frame": self._frame_index,
            "inference_ms": round(elapsed_ms, 3),
            "tracker": self.tracker,
            "detections": detections,
        }

    def _empty_metadata(self, frame, *, status: str, error: str | None = None) -> dict[str, Any]:
        md = {
            "schema": "yolo_detections_v1",
            "source": self.source_name,
            "coord_space": "model",
            "model_width": int(frame.width),
            "model_height": int(frame.height),
            "frame": self._frame_index,
            "status": status,
            "tracker": self.tracker,
            "detections": [],
        }
        if error:
            md["error"] = error
        return md

    def _write_metadata(self, frame, payload: dict[str, Any]) -> None:
        frame.metadata[self.metadata_key] = json.dumps(payload, sort_keys=True)
        if self.count_metadata_key:
            frame.metadata[self.count_metadata_key] = str(len(payload.get("detections", [])))
        if self.status_metadata_key:
            frame.metadata[self.status_metadata_key] = str(payload.get("status", "ok"))

    def _enqueue(self, frame) -> None:
        if isinstance(self._dst, dict):
            for dst in self._dst.values():
                dst.enqueue(frame)
        else:
            self._dst.enqueue(frame)

    def _process_frame(self, frame) -> None:
        model = self._ensure_model()
        image = self._frame_to_bgr(frame)

        started = time.perf_counter()
        results = model.track(source=image, **self._track_kwargs())
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        self._total_ms += elapsed_ms

        result = results[0] if results else None
        if result is None:
            payload = self._empty_metadata(frame, status="no_result")
        else:
            payload = self._serialize_results(result, int(frame.width), int(frame.height), elapsed_ms)
        self._write_metadata(frame, payload)

    def process(self):
        frame = self._src.tryGet(0)
        if frame is None:
            time.sleep(0.001)
            return

        if int(frame.width) <= 0 or int(frame.height) <= 0:
            self._enqueue(frame)
            return

        processed_count = self._frame_index + 1
        try:
            self._process_frame(frame)
        except Exception as exc:
            if self.error_policy == "raise":
                raise
            if self.emit_empty:
                self._write_metadata(
                    frame,
                    self._empty_metadata(frame, status="error", error=f"{type(exc).__name__}: {exc}"),
                )

        if self.debug_log_every_n > 0 and processed_count % self.debug_log_every_n == 0:
            avg_ms = self._total_ms / max(1, processed_count)
            print(
                "UltralyticsByteTrackNode:"
                f" frame={self._frame_index}"
                f" detections={self._last_detection_count}"
                f" tracks={self._last_track_count}"
                f" avg_ms={avg_ms:.2f}",
                flush=True,
            )

        self._enqueue(frame)
        self._frame_index += 1


def make_ultralytics_bytetrack_node(
    *,
    src: str,
    dst: str,
    weights: str | Path,
    metadata_key: str = "ultralytics_tracks",
    group: str = "ultralytics",
    name: str = "ultralytics_bytetrack",
    **kwargs: Any,
) -> UltralyticsByteTrackNode:
    """Convenience factory for graph builders that prefer keyword arguments."""

    return UltralyticsByteTrackNode(
        {
            "name": name,
            "group": group,
            "src": src,
            "dst": dst,
            "weights": str(weights),
            "metadata_key": metadata_key,
        }
        | kwargs
    )
