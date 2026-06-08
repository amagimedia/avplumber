from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .core import AVPlumber
from .node import (
    DecVideo,
    Demux,
    FilterVideo,
    ForceFPS,
    InputRec,
    PythonNode,
    TrackNetBall,
)


@dataclass(frozen=True)
class TrackNetRawJsonResult:
    output_path: Path
    video_width: int
    video_height: int
    n_frames: int
    frames_with_metadata: int
    frames_with_ball: int
    metadata_decode_errors: int


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def parse_size(value: str) -> tuple[int, int]:
    parts = value.lower().replace(":", "x").split("x", 1)
    if len(parts) != 2:
        raise ValueError("expected WIDTHxHEIGHT")
    width = int(parts[0])
    height = int(parts[1])
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return width, height


def ratio_from_fps(value: str | int | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "off", "skip"}:
        return None
    if "/" in text:
        return text
    fps = int(text)
    if fps <= 0:
        raise ValueError("fps must be positive")
    return f"{fps}/1"


def _json_command(name: str, params: dict[str, Any]) -> str:
    return f"{name} {json.dumps(params, separators=(',', ':'))}"


def timestamp_payload(frame) -> dict[str, Any]:
    pts = frame.pts
    tb = pts.timebase
    return {
        "pts": int(pts.timestamp),
        "timebase": [int(tb.num), int(tb.den)],
    }


class TrackNetRawJsonSink(PythonNode):
    def __init__(self, args: dict[str, Any]):
        super().__init__({"data_type": "VideoFrame"} | args)
        params = self.parameters
        self.metadata_key = str(params.get("metadata_key", "tracknet_ball"))
        self.output_path = Path(str(params["output_path"]))
        metadata_jsonl = params.get("metadata_jsonl")
        self.metadata_jsonl_path = Path(str(metadata_jsonl)) if metadata_jsonl else None
        self.input_url = str(params.get("input_url", ""))
        self.target_label = str(params.get("target_label", "basketball"))
        self.include_timing = bool(params.get("include_timing", False))
        self.contract_width = int(params.get("contract_width", 0) or 0)
        self.contract_height = int(params.get("contract_height", 0) or 0)
        self.frames: list[dict[str, Any]] = []
        self._frame_index = 0
        self._metadata_jsonl_fh = None
        self._written = False
        self._last_width = 0
        self._last_height = 0
        self.frames_with_metadata = 0
        self.frames_with_ball = 0
        self.metadata_decode_errors = 0

    def _open_metadata_jsonl(self) -> None:
        if self.metadata_jsonl_path is None or self._metadata_jsonl_fh is not None:
            return
        self.metadata_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._metadata_jsonl_fh = self.metadata_jsonl_path.open("w", encoding="utf-8")

    def _read_payload(self, frame) -> dict[str, Any] | None:
        try:
            raw = frame.metadata[self.metadata_key]
        except KeyError:
            return None
        if isinstance(raw, dict):
            return raw
        try:
            payload = json.loads(str(raw))
        except json.JSONDecodeError:
            self.metadata_decode_errors += 1
            return None
        return payload if isinstance(payload, dict) else None

    def _bboxes_from_srs_payload(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        bboxes = []
        for item in payload.get("bboxes") or []:
            if not isinstance(item, dict) or "x" not in item or "y" not in item:
                continue
            bbox = {
                "x": clamp01(float(item["x"])),
                "y": clamp01(float(item["y"])),
                "score": item.get("score"),
            }
            if bbox["score"] is not None:
                bbox["score"] = float(bbox["score"])
            bboxes.append(bbox)
        return bboxes

    def _bboxes_from_detection_payload(self, payload: dict[str, Any], frame) -> list[dict[str, Any]]:
        model_width = float(payload.get("model_width") or frame.width)
        model_height = float(payload.get("model_height") or frame.height)
        if model_width <= 0.0 or model_height <= 0.0:
            return []

        self._last_width = int(model_width)
        self._last_height = int(model_height)

        bboxes = []
        for item in payload.get("detections") or []:
            if not isinstance(item, dict):
                continue
            if item.get("label") not in (None, self.target_label):
                continue
            xyxy = item.get("xyxy")
            if not isinstance(xyxy, list) or len(xyxy) != 4:
                continue
            x1, y1, x2, y2 = [float(v) for v in xyxy]
            score = item.get("conf", item.get("score", item.get("tracknet_score")))
            bboxes.append(
                {
                    "x": clamp01((x1 + x2) * 0.5 / model_width),
                    "y": clamp01((y1 + y2) * 0.5 / model_height),
                    "score": float(score) if score is not None else 0.0,
                }
            )
        return bboxes

    def _bboxes_from_payload(self, payload: dict[str, Any] | None, frame) -> list[dict[str, Any]]:
        if payload is None:
            return []
        self.frames_with_metadata += 1
        if "bboxes" in payload:
            return self._bboxes_from_srs_payload(payload)
        if "detections" in payload:
            return self._bboxes_from_detection_payload(payload, frame)
        return []

    def _write_metadata_jsonl(self, frame_record: dict[str, Any], payload: dict[str, Any] | None) -> None:
        if self.metadata_jsonl_path is None:
            return
        self._open_metadata_jsonl()
        assert self._metadata_jsonl_fh is not None
        self._metadata_jsonl_fh.write(
            json.dumps(
                {
                    "frame": frame_record["frame"],
                    "metadata_key": self.metadata_key,
                    "tracknet": payload,
                    **(
                        timestamp_payload(frame_record["_frame"])
                        if "_frame" in frame_record
                        else {}
                    ),
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        )

    def process(self):
        frame = self._src.get()
        if frame is None or frame.width <= 0 or frame.height <= 0:
            return
        self._last_width = int(frame.width)
        self._last_height = int(frame.height)

        payload = self._read_payload(frame)
        bboxes = self._bboxes_from_payload(payload, frame)
        if bboxes:
            self.frames_with_ball += 1

        record: dict[str, Any] = {
            "frame": self._frame_index,
            "bboxes": bboxes,
        }
        if self.include_timing:
            record.update(timestamp_payload(frame))

        if self.metadata_jsonl_path is not None:
            debug_record = dict(record)
            debug_record["_frame"] = frame
            self._write_metadata_jsonl(debug_record, payload)

        self.frames.append(record)
        self._frame_index += 1

    def result(self) -> TrackNetRawJsonResult:
        width = self.contract_width or self._last_width
        height = self.contract_height or self._last_height
        return TrackNetRawJsonResult(
            output_path=self.output_path,
            video_width=int(width),
            video_height=int(height),
            n_frames=len(self.frames),
            frames_with_metadata=self.frames_with_metadata,
            frames_with_ball=self.frames_with_ball,
            metadata_decode_errors=self.metadata_decode_errors,
        )

    def write_output(self) -> None:
        if self._written:
            return
        result = self.result()
        payload: dict[str, Any] = {
            "video": str(Path(self.input_url).resolve()) if self.input_url else self.input_url,
            "metadata_key": self.metadata_key,
            "video_width": result.video_width,
            "video_height": result.video_height,
            "frames": self.frames,
            "summary": {
                "n_frames": result.n_frames,
                "frames_with_metadata": result.frames_with_metadata,
                "frames_with_ball": result.frames_with_ball,
                "metadata_decode_errors": result.metadata_decode_errors,
            },
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        if self._metadata_jsonl_fh:
            self._metadata_jsonl_fh.close()
            self._metadata_jsonl_fh = None
        self._written = True
        print(
            "TrackNet raw metadata:",
            f"frames={result.n_frames}",
            f"with_ball={result.frames_with_ball}",
            "->",
            self.output_path,
            flush=True,
        )

    def doStop(self):
        self.write_output()
        super().doStop()


def wait_until_done(avp: AVPlumber, node_name: str, timeout_s: float) -> None:
    node = avp.node(node_name)
    deadline = None if timeout_s <= 0 else time.time() + timeout_s

    def has_time() -> bool:
        return deadline is None or time.time() < deadline

    while not node.isWorking and has_time():
        avp.heartbeat()
        time.sleep(0.1)
    if not node.isWorking:
        raise TimeoutError(f"{node_name} did not start")
    while node.isWorking and has_time():
        avp.heartbeat()
        time.sleep(0.5)
    if node.isWorking:
        raise TimeoutError(f"{node_name} did not finish")
    node.join()


def build_tracknet_raw_json_graph(
    *,
    input_path: Path,
    engine_path: Path,
    output_path: Path,
    metadata_jsonl: Path | None = None,
    metadata_key: str = "tracknet_ball",
    target_label: str = "basketball",
    conf_thresh: float = 0.04,
    visible_thresh: float = 0.5,
    output_mode: str = "detection",
    triplet_alignment: str = "latest",
    preprocess_mode: str | None = None,
    sample_every_n: int = 1,
    fps: str | int | None = "30/1",
    tracknet_scale: tuple[int, int] | None = None,
    contract_width: int = 0,
    contract_height: int = 0,
    include_timing: bool = False,
    use_cuda_graph: bool = True,
    debug_log_metadata: bool = False,
    debug_log_every_n: int = 0,
    queue_capacity: int = 12,
    initial_timeout: int = 20,
    input_timeout: int = 10,
    cuda_device: int | str | None = None,
    stop_ts: str = "",
    remote_control_port: int = 0,
    webui_api: str = "",
    instance_name: str = "sports-reframe-ball",
    logfile: str = "",
) -> tuple[AVPlumber, TrackNetRawJsonSink]:
    fps_ratio = ratio_from_fps(fps)
    avp = AVPlumber()
    if remote_control_port:
        avp.enableControlServer(remote_control_port)
    if webui_api:
        avp.registerWithWebUI(webui_api, instance_name, logfile)

    hwaccel_params: dict[str, Any] = {"name": "@gpu", "type": "cuda"}
    if cuda_device is not None and str(cuda_device) != "":
        hwaccel_params["device"] = str(cuda_device)
    avp.executeCommandsFromString(_json_command("hwaccel.init", hwaccel_params))
    avp.edges.planCapacity("*", queue_capacity)

    video_edge = "v_dec_cuda"
    input_params: dict[str, Any] = {
        "name": "Input",
        "url": str(input_path),
        "dst": "in_mux",
        "group": "analysis",
        "initial_timeout": initial_timeout,
        "timeout": input_timeout,
        "loop": False,
        "on_error": "panic",
    }
    if stop_ts:
        input_params["stop_ts"] = stop_ts

    nodes = [
        InputRec(input_params),
        Demux(
            {
                "name": "Demux",
                "src": "in_mux",
                "routing": {"?v:0": "v_pkt"},
                "wait_for_keyframe": False,
                "group": "analysis",
            }
        ),
        DecVideo(
            {
                "name": "Decode_CUDA",
                "src": "v_pkt",
                "dst": video_edge,
                "pixel_format": "?cuda",
                "hwaccel": "@gpu",
                "codec_map": {"h264": "h264_cuvid", "hevc": "hevc_cuvid"},
                "hwaccel_only_for_codecs": ["h264", "hevc"],
                "group": "analysis",
                "on_error": "panic",
            }
        ),
    ]

    if fps_ratio:
        nodes.append(
            ForceFPS(
                {
                    "name": "Force_FPS",
                    "src": video_edge,
                    "dst": "v_proc_fps",
                    "fps": fps_ratio,
                    "group": "analysis",
                }
            )
        )
        video_edge = "v_proc_fps"

    if tracknet_scale:
        width, height = tracknet_scale
        nodes.append(
            FilterVideo(
                {
                    "name": "Scale_TrackNet",
                    "src": video_edge,
                    "dst": "v_tracknet_input",
                    "graph": f"scale_cuda=w={width}:h={height}",
                    "dst_width": width,
                    "dst_height": height,
                    "dst_pixel_format": "cuda",
                    "hwaccel": "@gpu",
                    "group": "analysis",
                }
            )
        )
        video_edge = "v_tracknet_input"

    resolved_preprocess_mode = preprocess_mode or (
        "srs_affine" if output_mode == "srs_ball" else "resize"
    )

    nodes.append(
        TrackNetBall(
            {
                "name": "TrackNet_Ball",
                "src": video_edge,
                "dst": "v_tracknet_metadata",
                "engine": str(engine_path),
                "metadata_key": metadata_key,
                "target_label": target_label,
                "conf_thresh": conf_thresh,
                "visible_thresh": visible_thresh,
                "output_mode": output_mode,
                "triplet_alignment": triplet_alignment,
                "preprocess_mode": resolved_preprocess_mode,
                "sample_every_n": sample_every_n,
                "use_cuda_graph": use_cuda_graph,
                "debug_log_metadata": debug_log_metadata,
                "debug_log_every_n": debug_log_every_n,
                "group": "analysis",
            }
        )
    )

    sink = TrackNetRawJsonSink(
        {
            "name": "TrackNet_Raw_JSON",
            "src": "v_tracknet_metadata",
            "metadata_key": metadata_key,
            "target_label": target_label,
            "output_path": str(output_path),
            "metadata_jsonl": str(metadata_jsonl) if metadata_jsonl else "",
            "input_url": str(input_path),
            "contract_width": contract_width,
            "contract_height": contract_height,
            "include_timing": include_timing,
            "group": "analysis",
        }
    )
    nodes.append(sink)

    for node in nodes:
        avp.addNode(node)

    return avp, sink


def run_tracknet_ball_to_raw_json(
    *,
    input_path: Path,
    engine_path: Path,
    output_path: Path,
    metadata_jsonl: Path | None = None,
    timeout_s: float = 600.0,
    **graph_options,
) -> TrackNetRawJsonResult:
    avp, sink = build_tracknet_raw_json_graph(
        input_path=input_path,
        engine_path=engine_path,
        output_path=output_path,
        metadata_jsonl=metadata_jsonl,
        **graph_options,
    )
    try:
        avp.group("analysis").startNodes()
        wait_until_done(avp, "TrackNet_Raw_JSON", timeout_s)
    finally:
        avp.group("analysis").stopNodes()
        sink.write_output()
        avp.shutdown()
    return sink.result()
