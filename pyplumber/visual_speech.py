import json
from dataclasses import dataclass, field
from pathlib import Path

from .node import PythonNode
from .visual_utils import (
    box_area,
    box_center,
    box_height,
    box_width,
    clamp,
    smoothstep,
    timestamp_ms_delta,
    timestamp_seconds,
    valid_pts,
)


@dataclass
class _TargetState:
    speaking: bool = False
    candidate_start_pts: int | None = None
    candidate_stop_pts: int | None = None
    segment_start_pts: int | None = None
    prev_features: dict | None = None
    motion_ema: float = 0.0
    mouth_h_floor: float | None = None
    last_seen_pts: int | None = None
    last_score: float = 0.0
    emitted_events: int = 0
    active_frames: int = 0
    visible_frames: int = 0
    mouth_visible_frames: int = 0
    mouth_roi_frames: int = 0
    mouth_roi_detected_frames: int = 0
    mouth_roi_tracked_frames: int = 0
    mouth_roi_estimated_frames: int = 0
    score_count: int = 0
    score_sum: float = 0.0
    max_score: float = 0.0
    segments: list[dict] = field(default_factory=list)


class VisualSpeechGateNode(PythonNode):
    """Pass video frames through and annotate visual speaking state from mouth ROI metadata."""

    def __init__(self, args: dict):
        super().__init__({"data_type": "VideoFrame"} | args)
        p = self.parameters
        self.mouth_metadata_key = str(p.get("mouth_metadata_key", "mouth_rois_v1"))
        self.output_metadata_key = str(p.get("output_metadata_key", "visual_speech_v1"))
        self.source_name = str(p.get("source", p.get("name", "visual_speech")))
        self.start_threshold = float(p.get("start_threshold", 0.14))
        self.stop_threshold = float(p.get("stop_threshold", 0.05))
        self.start_confirm_ms = float(p.get("start_confirm_ms", 250))
        self.stop_confirm_ms = float(p.get("stop_confirm_ms", 900))
        self.max_gap_ms = float(p.get("max_gap_ms", 300))
        self.motion_alpha = float(p.get("motion_alpha", 0.35))
        self.motion_weight = float(p.get("motion_weight", 6.0))
        self.open_weight = float(p.get("open_weight", 0.55))
        self.mouth_floor_rise_alpha = float(p.get("mouth_floor_rise_alpha", 0.01))
        self.mouth_floor_fall_alpha = float(p.get("mouth_floor_fall_alpha", 0.20))
        self.open_floor_margin = float(p.get("open_floor_margin", 0.012))
        self.open_range = float(p.get("open_range", 0.055))
        self.log_every_n = int(p.get("log_every_n", 0))
        self.event_jsonl_path = p.get("event_jsonl_path")
        self.summary_json_path = p.get("summary_json_path")
        self.targets = list(p.get("targets", [{"name": "primary"}]))
        self._states: dict[str, _TargetState] = {}
        self._frame_count = 0
        self._fh = None
        self._last_valid_pts: int | None = None
        self._last_valid_sec: float | None = None
        self._finished = False

    def _ensure_event_file(self):
        if self.event_jsonl_path and self._fh is None:
            path = Path(self.event_jsonl_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = path.open("w", encoding="utf-8")

    def doStart(self):
        self._ensure_event_file()
        super().doStart()

    def doStop(self):
        super().doStop()
        self._finish_stream()
        self._write_summary()
        if self._fh:
            self._fh.close()
            self._fh = None

    def _state(self, name: str) -> _TargetState:
        state = self._states.get(name)
        if state is None:
            state = _TargetState()
            self._states[name] = state
        return state

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

    def _parse_mouth_metadata(self, frame) -> tuple[dict, dict[str, dict]]:
        try:
            raw = frame.metadata[self.mouth_metadata_key]
        except KeyError:
            return {}, {}
        try:
            metadata = json.loads(str(raw))
        except json.JSONDecodeError:
            return {}, {}

        by_target = {}
        for item in metadata.get("targets", []):
            if not isinstance(item, dict):
                continue
            name = item.get("target")
            if name is None:
                continue
            by_target[str(name)] = item
        return metadata, by_target

    def _features_from_target(self, item: dict | None) -> dict | None:
        if not item or not item.get("visible"):
            return None
        face_box = item.get("face_xyxy")
        if not isinstance(face_box, list) or len(face_box) != 4:
            return None
        face_box = [float(v) for v in face_box]
        face_w = max(1.0, box_width(face_box))
        face_h = max(1.0, box_height(face_box))
        face_area = max(1.0, box_area(face_box))

        mouth_source = str(item.get("mouth_source", "none"))
        out = {
            "face_xyxy": face_box,
            "face_conf": float(item.get("face_conf", 0.0)),
            "mouth_visible": mouth_source == "detected",
            "mouth_roi_available": bool(item.get("mouth_roi_available", False)),
            "mouth_source": mouth_source,
            "mouth_conf": 0.0,
            "mouth_x": 0.0,
            "mouth_y": 0.0,
            "mouth_w": 0.0,
            "mouth_h": 0.0,
            "mouth_area": 0.0,
            "nose_mouth_y": 0.0,
        }

        mouth_box = item.get("mouth_xyxy")
        if isinstance(mouth_box, list) and len(mouth_box) == 4:
            mouth_box = [float(v) for v in mouth_box]
            mx, my = box_center(mouth_box)
            out.update({
                "mouth_xyxy": mouth_box,
                "mouth_conf": float(item.get("mouth_conf", 0.0)),
                "mouth_x": (mx - face_box[0]) / face_w,
                "mouth_y": (my - face_box[1]) / face_h,
                "mouth_w": max(0.0, mouth_box[2] - mouth_box[0]) / face_w,
                "mouth_h": max(0.0, mouth_box[3] - mouth_box[1]) / face_h,
                "mouth_area": box_area(mouth_box) / face_area,
            })
            nose_box = item.get("nose_xyxy")
            if isinstance(nose_box, list) and len(nose_box) == 4:
                _, ny = box_center([float(v) for v in nose_box])
                out["nose_mouth_y"] = (my - ny) / face_h
        return out

    def _record_score(self, state: _TargetState, score: float, features: dict | None):
        state.active_frames += 1
        if features is not None:
            state.visible_frames += 1
        if features is not None and features.get("mouth_visible"):
            state.mouth_visible_frames += 1
        if features is not None and features.get("mouth_roi_available"):
            state.mouth_roi_frames += 1
            source = str(features.get("mouth_source", "none"))
            if source == "detected":
                state.mouth_roi_detected_frames += 1
            elif source == "tracked":
                state.mouth_roi_tracked_frames += 1
            elif source == "estimated":
                state.mouth_roi_estimated_frames += 1
        state.score_count += 1
        state.score_sum += float(score)
        state.max_score = max(state.max_score, float(score))

    def _score(self, state: _TargetState, features: dict | None, frame) -> tuple[float, float, float]:
        if features is None or not features["mouth_visible"]:
            state.motion_ema *= (1.0 - self.motion_alpha)
            return 0.0, 0.0, state.motion_ema

        pts = int(frame.pts.timestamp)
        mouth_h = float(features["mouth_h"])
        if state.mouth_h_floor is None:
            state.mouth_h_floor = mouth_h
        elif mouth_h < state.mouth_h_floor:
            state.mouth_h_floor = (
                (1.0 - self.mouth_floor_fall_alpha) * state.mouth_h_floor
                + self.mouth_floor_fall_alpha * mouth_h
            )
        else:
            state.mouth_h_floor = (
                (1.0 - self.mouth_floor_rise_alpha) * state.mouth_h_floor
                + self.mouth_floor_rise_alpha * mouth_h
            )

        open_start = state.mouth_h_floor + self.open_floor_margin
        open_end = open_start + self.open_range
        open_score = smoothstep(open_start, open_end, mouth_h)

        raw_motion = 0.0
        if state.prev_features is not None and timestamp_ms_delta(frame, state.last_seen_pts) <= self.max_gap_ms:
            prev = state.prev_features
            raw_motion = (
                abs(float(features["mouth_h"]) - float(prev["mouth_h"])) * 3.0
                + abs(float(features["mouth_y"]) - float(prev["mouth_y"])) * 1.6
                + abs(float(features["mouth_area"]) - float(prev["mouth_area"])) * 2.0
                + abs(float(features["nose_mouth_y"]) - float(prev["nose_mouth_y"])) * 1.2
            )
        state.motion_ema = (
            (1.0 - self.motion_alpha) * state.motion_ema
            + self.motion_alpha * raw_motion
        )
        state.prev_features = features
        state.last_seen_pts = pts

        motion_score = clamp(state.motion_ema * self.motion_weight, 0.0, 1.0)
        score = clamp(max(motion_score, open_score * self.open_weight), 0.0, 1.0)
        return score, open_score, motion_score

    def _update_state(self, name: str, state: _TargetState, score: float, frame) -> list[dict]:
        pts = int(frame.pts.timestamp)
        sec = timestamp_seconds(frame)
        events = []

        if score >= self.start_threshold:
            state.candidate_stop_pts = None
            if state.speaking:
                state.last_score = score
                return events
            if state.candidate_start_pts is None:
                state.candidate_start_pts = pts
            if timestamp_ms_delta(frame, state.candidate_start_pts) >= self.start_confirm_ms:
                state.speaking = True
                state.segment_start_pts = state.candidate_start_pts
                event = {
                    "event": "visual_speech_start",
                    "source": self.source_name,
                    "target": name,
                    "pts": pts,
                    "sec": round(sec, 6),
                    "speech_start_pts": int(state.segment_start_pts),
                    "score": round(score, 6),
                }
                events.append(event)
                state.emitted_events += 1
        elif score <= self.stop_threshold:
            state.candidate_start_pts = None
            if state.speaking:
                if state.candidate_stop_pts is None:
                    state.candidate_stop_pts = pts
                if timestamp_ms_delta(frame, state.candidate_stop_pts) >= self.stop_confirm_ms:
                    start_pts = state.segment_start_pts
                    event = {
                        "event": "visual_speech_stop",
                        "source": self.source_name,
                        "target": name,
                        "pts": pts,
                        "sec": round(sec, 6),
                        "speech_start_pts": int(start_pts) if start_pts is not None else None,
                        "speech_end_pts": int(state.candidate_stop_pts),
                        "score": round(score, 6),
                    }
                    state.segments.append(event)
                    events.append(event)
                    state.emitted_events += 1
                    state.speaking = False
                    state.segment_start_pts = None
                    state.candidate_stop_pts = None
        else:
            state.candidate_start_pts = None
            state.candidate_stop_pts = None

        state.last_score = score
        return events

    def _force_stop(
        self,
        name: str,
        state: _TargetState,
        pts: int,
        sec: float,
        reason: str,
        score: float | None = None,
    ) -> dict | None:
        if not state.speaking:
            return None
        start_pts = state.segment_start_pts
        end_pts = state.candidate_stop_pts if state.candidate_stop_pts is not None else pts
        event = {
            "event": "visual_speech_stop",
            "source": self.source_name,
            "target": name,
            "pts": int(pts),
            "sec": round(float(sec), 6),
            "speech_start_pts": int(start_pts) if start_pts is not None else None,
            "speech_end_pts": int(end_pts),
            "score": round(float(state.last_score if score is None else score), 6),
            "reason": reason,
        }
        state.segments.append(event)
        state.emitted_events += 1
        state.speaking = False
        state.segment_start_pts = None
        state.candidate_start_pts = None
        state.candidate_stop_pts = None
        return event

    def _write_event(self, event: dict):
        self._ensure_event_file()
        if self._fh:
            self._fh.write(json.dumps(event, sort_keys=True) + "\n")
            self._fh.flush()
        print(json.dumps(event, sort_keys=True), flush=True)

    def _write_summary(self):
        if not self.summary_json_path:
            return
        summary = {
            "source": self.source_name,
            "mouth_metadata_key": self.mouth_metadata_key,
            "output_metadata_key": self.output_metadata_key,
            "frames": self._frame_count,
            "config": {
                "start_threshold": self.start_threshold,
                "stop_threshold": self.stop_threshold,
                "start_confirm_ms": self.start_confirm_ms,
                "stop_confirm_ms": self.stop_confirm_ms,
                "motion_weight": self.motion_weight,
                "open_weight": self.open_weight,
            },
            "targets": {},
        }
        for name, state in self._states.items():
            active_frames = int(state.active_frames)
            summary["targets"][name] = {
                "speaking_at_end": state.speaking,
                "last_score": round(float(state.last_score), 6),
                "max_score": round(float(state.max_score), 6),
                "mean_score": round(float(state.score_sum / state.score_count), 6) if state.score_count else 0.0,
                "active_frames": active_frames,
                "visible_frames": int(state.visible_frames),
                "mouth_visible_frames": int(state.mouth_visible_frames),
                "mouth_raw_detected_frames": int(state.mouth_visible_frames),
                "mouth_roi_frames": int(state.mouth_roi_frames),
                "mouth_roi_detected_frames": int(state.mouth_roi_detected_frames),
                "mouth_roi_tracked_frames": int(state.mouth_roi_tracked_frames),
                "mouth_roi_estimated_frames": int(state.mouth_roi_estimated_frames),
                "mouth_raw_detected_pct_of_active": round(100.0 * state.mouth_visible_frames / active_frames, 3) if active_frames else 0.0,
                "mouth_roi_pct_of_active": round(100.0 * state.mouth_roi_frames / active_frames, 3) if active_frames else 0.0,
                "events": int(state.emitted_events),
                "segments": state.segments,
            }
        path = Path(self.summary_json_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def _enqueue(self, frame):
        if isinstance(self._dst, dict):
            for dst in self._dst.values():
                dst.enqueue(frame)
        else:
            self._dst.enqueue(frame)

    def _finish_stream(self):
        if self._finished:
            return
        self._finished = True
        if self._last_valid_pts is None or self._last_valid_sec is None:
            return
        for name, state in self._states.items():
            event = self._force_stop(
                name,
                state,
                self._last_valid_pts,
                self._last_valid_sec,
                "stream_end",
            )
            if event is not None:
                self._write_event(event)
        self._write_summary()

    def process(self):
        frame = self._src.get()
        if frame is None:
            return

        self._frame_count += 1
        if not valid_pts(frame):
            self._finish_stream()
            self._enqueue(frame)
            return

        sec = timestamp_seconds(frame)
        pts = int(frame.pts.timestamp)
        self._last_valid_pts = pts
        self._last_valid_sec = sec

        _, mouth_by_target = self._parse_mouth_metadata(frame)
        target_results = []
        frame_events = []
        active_targets = self._active_targets(sec)
        active_names = set(active_targets)
        for name in active_targets:
            state = self._state(name)
            item = mouth_by_target.get(name)
            features = self._features_from_target(item)
            score, open_score, motion_score = self._score(state, features, frame)
            self._record_score(state, score, features)
            events = self._update_state(name, state, score, frame)
            for event in events:
                self._write_event(event)
            frame_events.extend(events)

            result = {
                "target": name,
                "visible": bool(features),
                "mouth_visible": bool(features and features.get("mouth_visible")),
                "mouth_roi_available": bool(features and features.get("mouth_roi_available")),
                "mouth_source": str(features.get("mouth_source", "none")) if features else "none",
                "speaking": bool(state.speaking),
                "score": round(float(score), 6),
                "mouth_open_score": round(float(open_score), 6),
                "mouth_motion_score": round(float(motion_score), 6),
            }
            if features:
                result["face_xyxy"] = [round(float(v), 3) for v in features["face_xyxy"]]
                if "mouth_xyxy" in features:
                    result["mouth_xyxy"] = [round(float(v), 3) for v in features["mouth_xyxy"]]
                    result["mouth_h_rel"] = round(float(features["mouth_h"]), 6)
                if item and item.get("mouth_track_age_ms") is not None:
                    result["mouth_track_age_ms"] = round(float(item["mouth_track_age_ms"]), 3)
            target_results.append(result)

        for name, state in self._states.items():
            if name in active_names:
                continue
            event = self._force_stop(name, state, pts, sec, "target_inactive")
            if event is not None:
                self._write_event(event)
                frame_events.append(event)

        frame_metadata = {
            "version": 1,
            "source": self.source_name,
            "pts": pts,
            "sec": round(sec, 6),
            "targets": target_results,
            "events": frame_events,
        }
        frame.metadata[self.output_metadata_key] = json.dumps(frame_metadata, sort_keys=True)

        if self.log_every_n > 0 and self._frame_count % self.log_every_n == 0:
            preview = ", ".join(
                f"{item['target']}:{item['score']:.2f}:{'talk' if item['speaking'] else 'idle'}"
                for item in target_results
            )
            print(f"VisualSpeechGateNode frame={self._frame_count} sec={sec:.3f} {preview}", flush=True)

        self._enqueue(frame)
