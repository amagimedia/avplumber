import json
from dataclasses import dataclass

from .node import PythonNode
from .visual_utils import (
    absolute_box,
    blend_box,
    box_area,
    box_center,
    box_contains,
    box_height,
    box_width,
    clamp,
    clamp_box_to_face,
    clamp_relative_box,
    intersects_region,
    relative_box,
    timestamp_ms_delta,
    timestamp_seconds,
    valid_pts,
)


@dataclass
class _MouthTrackState:
    mouth_rel_xyxy: list[float] | None = None
    mouth_last_detected_pts: int | None = None
    mouth_last_roi_pts: int | None = None


class FaceAnchoredMouthTrackerNode(PythonNode):
    """Attach detected, tracked, or estimated mouth ROIs to named face targets."""

    def __init__(self, args: dict):
        super().__init__({"data_type": "VideoFrame"} | args)
        p = self.parameters
        self.input_metadata_key = str(p.get("input_metadata_key", "face_parts"))
        self.output_metadata_key = str(p.get("output_metadata_key", "mouth_rois_v1"))
        self.source_name = str(p.get("source", p.get("name", "mouth_tracker")))
        self.min_conf = float(p.get("min_conf", 0.25))
        self.targets = list(p.get("targets", [{"name": "primary"}]))
        self.mouth_tracker_enabled = bool(p.get("mouth_tracker_enabled", True))
        self.mouth_track_hold_ms = float(p.get("mouth_track_hold_ms", 60000))
        self.mouth_track_alpha = float(p.get("mouth_track_alpha", 0.35))
        self.mouth_estimate_enabled = bool(p.get("mouth_estimate_enabled", True))
        self.mouth_default_rel_w = float(p.get("mouth_default_rel_w", 0.24))
        self.mouth_default_rel_h = float(p.get("mouth_default_rel_h", 0.07))
        self.mouth_default_rel_y = float(p.get("mouth_default_rel_y", 0.70))
        self.mouth_nose_rel_offset_y = float(p.get("mouth_nose_rel_offset_y", 0.18))
        self.log_every_n = int(p.get("log_every_n", 0))
        self._states: dict[str, _MouthTrackState] = {}
        self._frame_count = 0

    def _state(self, name: str) -> _MouthTrackState:
        state = self._states.get(name)
        if state is None:
            state = _MouthTrackState()
            self._states[name] = state
        return state

    def _enqueue(self, frame):
        if isinstance(self._dst, dict):
            for dst in self._dst.values():
                dst.enqueue(frame)
        else:
            self._dst.enqueue(frame)

    def _active_targets(self, sec: float) -> dict[str, list[dict]]:
        active: dict[str, list[dict]] = {}
        for target in self.targets:
            start_sec = target.get("start_sec")
            end_sec = target.get("end_sec")
            if start_sec is not None and sec < float(start_sec):
                continue
            if end_sec is not None and sec >= float(end_sec):
                continue
            name = str(target.get("name", "primary"))
            active.setdefault(name, []).append(target)
        return active

    def _parse_detections(self, frame) -> tuple[dict, list[dict]]:
        try:
            raw = frame.metadata[self.input_metadata_key]
        except KeyError:
            return {}, []
        try:
            metadata = json.loads(str(raw))
        except json.JSONDecodeError:
            return {}, []

        detections = []
        for item in metadata.get("detections", []):
            label = str(item.get("label", ""))
            conf = float(item.get("conf", 0.0))
            xyxy = item.get("xyxy")
            if conf < self.min_conf or not isinstance(xyxy, list) or len(xyxy) != 4:
                continue
            detections.append({
                "label": label,
                "conf": conf,
                "xyxy": [float(v) for v in xyxy],
            })
        return metadata, detections

    def _pick_face(self, faces: list[dict], regions: list[dict]) -> dict | None:
        candidates = []
        for face in faces:
            box = face["xyxy"]
            if not regions or any(intersects_region(box, region) for region in regions):
                candidates.append(face)
        if not candidates:
            return None
        return max(candidates, key=lambda d: (float(d["conf"]), box_area(d["xyxy"])))

    def _pick_part(self, parts: list[dict], face_box: list[float]) -> dict | None:
        fc_x, _ = box_center(face_box)
        face_w = max(1.0, box_width(face_box))
        face_h = max(1.0, box_height(face_box))

        candidates = []
        for part in parts:
            box = part["xyxy"]
            cx, cy = box_center(box)
            if not box_contains(face_box, cx, cy):
                continue
            dx = abs(cx - fc_x) / face_w
            dy = abs(cy - (face_box[1] + 0.68 * face_h)) / face_h
            candidates.append((dx + dy, part))
        if not candidates:
            return None
        return min(candidates, key=lambda item: item[0])[1]

    def _estimate_mouth_rel(self, state: _MouthTrackState, face_box: list[float], nose: dict | None) -> list[float]:
        if state.mouth_rel_xyxy is not None:
            prev = state.mouth_rel_xyxy
            width = max(0.04, prev[2] - prev[0])
            height = max(0.025, prev[3] - prev[1])
            cx = (prev[0] + prev[2]) * 0.5
            cy = (prev[1] + prev[3]) * 0.5
        else:
            width = self.mouth_default_rel_w
            height = self.mouth_default_rel_h
            cx = 0.5
            cy = self.mouth_default_rel_y

        if nose is not None:
            face_w = max(1.0, box_width(face_box))
            face_h = max(1.0, box_height(face_box))
            nx, ny = box_center(nose["xyxy"])
            cx = clamp((nx - face_box[0]) / face_w, 0.25, 0.75)
            cy = clamp((ny - face_box[1]) / face_h + self.mouth_nose_rel_offset_y, 0.58, 0.84)

        return clamp_relative_box([
            cx - width * 0.5,
            cy - height * 0.5,
            cx + width * 0.5,
            cy + height * 0.5,
        ])

    def _track_mouth(
        self,
        state: _MouthTrackState,
        face: dict | None,
        mouth: dict | None,
        nose: dict | None,
        frame,
    ) -> tuple[dict | None, dict]:
        info = {
            "source": "none",
            "raw_detected": False,
            "age_ms": None,
        }
        if face is None:
            return None, info

        face_box = face["xyxy"]
        pts = int(frame.pts.timestamp)
        if mouth is not None:
            rel = clamp_relative_box(relative_box(mouth["xyxy"], face_box))
            if state.mouth_rel_xyxy is None:
                state.mouth_rel_xyxy = rel
            else:
                state.mouth_rel_xyxy = clamp_relative_box(
                    blend_box(state.mouth_rel_xyxy, rel, self.mouth_track_alpha)
                )
            state.mouth_last_detected_pts = pts
            state.mouth_last_roi_pts = pts
            info.update({
                "source": "detected",
                "raw_detected": True,
                "age_ms": 0.0,
            })
            return {**mouth, "source": "detected", "estimated": False}, info

        if not self.mouth_tracker_enabled:
            return None, info

        rel = None
        source = "none"
        age_ms = None
        if state.mouth_rel_xyxy is not None and state.mouth_last_detected_pts is not None:
            age_ms = timestamp_ms_delta(frame, state.mouth_last_detected_pts)
            if age_ms <= self.mouth_track_hold_ms:
                rel = state.mouth_rel_xyxy
                source = "tracked"

        if rel is None and self.mouth_estimate_enabled:
            rel = self._estimate_mouth_rel(state, face_box, nose)
            source = "estimated"

        if rel is None:
            return None, info

        box = clamp_box_to_face(absolute_box(rel, face_box), face_box)
        state.mouth_last_roi_pts = pts
        info.update({
            "source": source,
            "raw_detected": False,
            "age_ms": age_ms,
        })
        return {
            "label": "Mouth",
            "conf": 0.0,
            "xyxy": box,
            "source": source,
            "estimated": source != "detected",
        }, info

    def _target_record(self, name: str, face: dict | None, mouth: dict | None, nose: dict | None, info: dict) -> dict:
        record = {
            "target": name,
            "visible": face is not None,
            "mouth_roi_available": mouth is not None,
            "mouth_source": str(info.get("source", "none")),
            "mouth_raw_detected": bool(info.get("raw_detected", False)),
        }
        if face is not None:
            record["face_xyxy"] = [round(float(v), 3) for v in face["xyxy"]]
            record["face_conf"] = round(float(face["conf"]), 6)
        if nose is not None:
            record["nose_xyxy"] = [round(float(v), 3) for v in nose["xyxy"]]
        if mouth is not None:
            record["mouth_xyxy"] = [round(float(v), 3) for v in mouth["xyxy"]]
            record["mouth_conf"] = round(float(mouth.get("conf", 0.0)), 6)
        if info.get("age_ms") is not None:
            record["mouth_track_age_ms"] = round(float(info["age_ms"]), 3)
        return record

    def _draw_detection(self, name: str, record: dict) -> dict | None:
        if not record.get("mouth_roi_available") or "mouth_xyxy" not in record:
            return None
        source = str(record.get("mouth_source", "none"))
        label = "M" if source == "detected" else "M (interpolated)"
        conf = {
            "detected": 1.0,
            "tracked": 0.75,
            "estimated": 0.45,
        }.get(source, 0.0)
        return {
            "label": label,
            "target": name,
            "mouth_source": source,
            "conf": conf,
            "xyxy": record["mouth_xyxy"],
            "predicted": source != "detected",
        }

    def process(self):
        frame = self._src.get()
        if frame is None:
            return

        self._frame_count += 1
        if not valid_pts(frame):
            self._enqueue(frame)
            return

        sec = timestamp_seconds(frame)
        pts = int(frame.pts.timestamp)
        input_metadata, detections = self._parse_detections(frame)
        faces = [d for d in detections if d["label"].lower() == "face"]
        mouths = [d for d in detections if d["label"].lower() == "mouth"]
        noses = [d for d in detections if d["label"].lower() == "nose"]

        targets = []
        draw_detections = []
        for name, regions in self._active_targets(sec).items():
            state = self._state(name)
            face = self._pick_face(faces, regions)
            nose = None
            mouth = None
            info = {"source": "none", "raw_detected": False, "age_ms": None}
            if face is not None:
                nose = self._pick_part(noses, face["xyxy"])
                raw_mouth = self._pick_part(mouths, face["xyxy"])
                mouth, info = self._track_mouth(state, face, raw_mouth, nose, frame)
            record = self._target_record(name, face, mouth, nose, info)
            targets.append(record)
            draw_detection = self._draw_detection(name, record)
            if draw_detection is not None:
                draw_detections.append(draw_detection)

        metadata = {
            "version": 1,
            "source": self.source_name,
            "pts": pts,
            "sec": round(sec, 6),
            "coord_space": input_metadata.get("coord_space", "model"),
            "model_width": input_metadata.get("model_width", 960),
            "model_height": input_metadata.get("model_height", 544),
            "targets": targets,
            "detections": draw_detections,
        }
        frame.metadata[self.output_metadata_key] = json.dumps(metadata, sort_keys=True)

        if self.log_every_n > 0 and self._frame_count % self.log_every_n == 0:
            preview = ", ".join(
                f"{item['target']}:{item['mouth_source']}" for item in targets
            )
            print(f"FaceAnchoredMouthTrackerNode frame={self._frame_count} sec={sec:.3f} {preview}", flush=True)

        self._enqueue(frame)
