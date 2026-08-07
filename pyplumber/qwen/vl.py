"""Qwen3-VL async CUDA video node.

The node reuses Molmo's async pass-through plumbing but has its own strict
Qwen3 video tensor preprocessor and parser. Strict mode never copies video
buffers to CPU; text tokenization and JSON parsing still happen on CPU.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable

from pyplumber.molmo.vllm import (
    MolmoVllmAsync,
    _MolmoPreprocessor,
    _MolmoResult,
    _TensorBuffer,
    _find_repo_file,
    _json_dumps,
)


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)


def _coerce_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _extract_json_value(text: str) -> tuple[Any | None, str]:
    if not text:
        return None, "empty"

    candidates: list[str] = []
    for match in _JSON_FENCE_RE.finditer(text):
        candidates.append(match.group(1).strip())
    candidates.append(text.strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, (dict, list)):
                return parsed, "ok"
        except json.JSONDecodeError:
            pass

        starts = [(candidate.find("{"), "{", "}"), (candidate.find("["), "[", "]")]
        starts = [item for item in starts if item[0] >= 0]
        starts.sort(key=lambda item: item[0])
        for start, open_char, close_char in starts:
            depth = 0
            in_string = False
            escape = False
            for index in range(start, len(candidate)):
                char = candidate[index]
                if in_string:
                    if escape:
                        escape = False
                    elif char == "\\":
                        escape = True
                    elif char == '"':
                        in_string = False
                    continue
                if char == '"':
                    in_string = True
                elif char == open_char:
                    depth += 1
                elif char == close_char:
                    depth -= 1
                    if depth == 0:
                        snippet = candidate[start : index + 1]
                        try:
                            parsed = json.loads(snippet)
                        except json.JSONDecodeError:
                            break
                        if isinstance(parsed, (dict, list)):
                            return parsed, "ok"
                        break

    return None, "invalid_json"


def _object_list(parsed: Any) -> list[dict[str, Any]]:
    if isinstance(parsed, list):
        return [item for item in parsed if isinstance(item, dict)]
    if isinstance(parsed, dict):
        objects = parsed.get("objects", parsed.get("detections", parsed.get("items")))
        if isinstance(objects, list):
            return [item for item in objects if isinstance(item, dict)]
        if any(key in parsed for key in ("bbox_2d", "bbox", "box", "xyxy", "point_2d", "point")):
            return [parsed]
    return []


def _parse_label(obj: dict[str, Any], fallback_index: int) -> str:
    label = obj.get("label", obj.get("name", obj.get("object", "object")))
    if not isinstance(label, str) or not label.strip():
        return f"object_{fallback_index}"
    return label.strip()


def _parse_conf(obj: dict[str, Any]) -> float:
    conf = _coerce_float(obj.get("confidence", obj.get("conf", obj.get("score", 1.0))))
    if conf is None:
        return 1.0
    return min(1.0, max(0.0, conf))


def _coord_to_pixel(value: Any, *, dim: int, coordinate_scale: float) -> float | None:
    number = _coerce_float(value)
    if number is None:
        return None
    number = min(coordinate_scale, max(0.0, number))
    return number * float(dim) / coordinate_scale


def _build_point_metadata(points: list[tuple[float, float, float]], *, width: int, height: int) -> dict[str, Any] | None:
    if not points:
        return None

    keypoints: list[float] = []
    for px, py, conf in points:
        keypoints.extend([px, py, conf])

    return {
        "schema": "pose_keypoints_v1",
        "coord_space": "model",
        "model_width": int(width),
        "model_height": int(height),
        "num_keypoints": len(keypoints) // 3,
        "poses": [
            {
                "label": "qwen_points",
                "conf": max(keypoints[2::3]) if len(keypoints) >= 3 else 1.0,
                "keypoints": keypoints,
            }
        ],
    }


def parse_qwen_generated_text(
    generated_text: str,
    *,
    model_width: int,
    model_height: int,
    prompt_id: str,
    window_start_pts: str,
    window_end_pts: str,
    coordinate_scale: float = 1000.0,
    latency_ms: float | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    parsed, parse_status = _extract_json_value(generated_text)
    raw_md: dict[str, Any] = {
        "schema": "qwen_raw_v1",
        "prompt_id": prompt_id,
        "window_start_pts": window_start_pts,
        "window_end_pts": window_end_pts,
        "generated_text": generated_text,
        "parse_status": parse_status,
    }
    if latency_ms is not None:
        raw_md["latency_ms"] = round(float(latency_ms), 3)

    if parsed is None:
        return None, None, raw_md

    objects = _object_list(parsed)
    if not objects:
        raw_md["parse_status"] = "no_objects"
        return None, None, raw_md

    detections: list[dict[str, Any]] = []
    points: list[tuple[float, float, float]] = []
    invalid_count = 0
    for index, obj in enumerate(objects):
        label = _parse_label(obj, index)
        conf = _parse_conf(obj)
        emitted = False

        bbox = obj.get("bbox_2d", obj.get("bbox", obj.get("box", obj.get("xyxy"))))
        if isinstance(bbox, list) and len(bbox) >= 4:
            x1 = _coord_to_pixel(bbox[0], dim=model_width, coordinate_scale=coordinate_scale)
            y1 = _coord_to_pixel(bbox[1], dim=model_height, coordinate_scale=coordinate_scale)
            x2 = _coord_to_pixel(bbox[2], dim=model_width, coordinate_scale=coordinate_scale)
            y2 = _coord_to_pixel(bbox[3], dim=model_height, coordinate_scale=coordinate_scale)
            if None not in (x1, y1, x2, y2):
                assert x1 is not None and y1 is not None and x2 is not None and y2 is not None
                if x2 < x1:
                    x1, x2 = x2, x1
                if y2 < y1:
                    y1, y2 = y2, y1
                if x2 > x1 and y2 > y1:
                    detections.append(
                        {
                            "label": label,
                            "conf": conf,
                            "xyxy": [x1, y1, x2, y2],
                            "source": "qwen",
                        }
                    )
                    emitted = True

        point = obj.get("point_2d", obj.get("point", obj.get("center", obj.get("xy"))))
        if isinstance(point, list) and len(point) >= 2:
            px = _coord_to_pixel(point[0], dim=model_width, coordinate_scale=coordinate_scale)
            py = _coord_to_pixel(point[1], dim=model_height, coordinate_scale=coordinate_scale)
            if px is not None and py is not None:
                points.append((px, py, conf))
                emitted = True

        if not emitted:
            invalid_count += 1

    raw_md["object_count"] = len(objects)
    raw_md["invalid_object_count"] = invalid_count
    raw_md["detection_count"] = len(detections)
    raw_md["point_count"] = len(points)
    if not detections and not points:
        raw_md["parse_status"] = "no_valid_objects"

    det_md = None
    if detections:
        det_md = {
            "schema": "yolo_detections_v1",
            "coord_space": "model",
            "model_width": int(model_width),
            "model_height": int(model_height),
            "detections": detections,
        }

    return det_md, _build_point_metadata(points, width=model_width, height=model_height), raw_md


def pack_qwen3_video_patches(
    frames_nchw: Any,
    *,
    sample_count: int,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> tuple[Any, Any]:
    """Pack normalized ``[T,C,H,W]`` frames into Qwen3 ``pixel_values_videos``."""

    torch = importlib.import_module("torch")
    frames = frames_nchw[: int(sample_count)]
    if frames.shape[0] % temporal_patch_size != 0:
        repeat_count = temporal_patch_size - (frames.shape[0] % temporal_patch_size)
        frames = torch.cat([frames, frames[-1:].repeat(repeat_count, 1, 1, 1)], dim=0)

    padded_frames, channels, height, width = frames.shape
    grid_t = padded_frames // temporal_patch_size
    grid_h = height // patch_size
    grid_w = width // patch_size
    if height % patch_size or width % patch_size:
        raise ValueError("Qwen3 target dimensions must be divisible by patch_size")
    if grid_h % merge_size or grid_w % merge_size:
        raise ValueError("Qwen3 patch grid must be divisible by merge_size")

    patches = frames.reshape(
        grid_t,
        temporal_patch_size,
        channels,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    )
    patches = patches.permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    flatten = patches.reshape(
        grid_t * grid_h * grid_w,
        channels * temporal_patch_size * patch_size * patch_size,
    ).contiguous()
    video_grid_thw = torch.tensor(
        [[grid_t, grid_h, grid_w]],
        device=frames.device,
        dtype=torch.long,
    )
    return flatten, video_grid_thw


class _Qwen3Preprocessor(_MolmoPreprocessor):
    def __init__(
        self,
        *,
        torch_mod: Any,
        cupy_mod: Any,
        target_height: int,
        target_width: int,
        patch_size: int,
        temporal_patch_size: int,
        merge_size: int,
        window_frames: int,
        dtype_name: str,
        kernel_path: Path,
    ) -> None:
        self.torch = torch_mod
        self.cp = cupy_mod
        self.target_height = target_height
        self.target_width = target_width
        self.frame_size = target_width
        self.patch_size = patch_size
        self.temporal_patch_size = temporal_patch_size
        self.merge_size = merge_size
        self.window_frames = window_frames
        self.dtype_name = dtype_name
        self.kernel_path = kernel_path

        code = kernel_path.read_text(encoding="utf-8")
        self.module = self.cp.RawModule(
            code=code,
            options=self._raw_module_options(),
            name_expressions=(
                "kQwen3VlPreprocessNV12_fp16",
                "kQwen3VlPreprocessNV12_fp32",
            ),
        )
        kernel_name = "kQwen3VlPreprocessNV12_fp16" if dtype_name == "fp16" else "kQwen3VlPreprocessNV12_fp32"
        self.kernel = self.module.get_function(kernel_name)
        self._active_buffer: _TensorBuffer | None = None

    def make_buffer(self, index: int) -> _TensorBuffer:
        tensor = self.torch.empty(
            (self.window_frames, 3, self.target_height, self.target_width),
            device="cuda",
            dtype=self._torch_dtype(),
        )
        cupy_view = self._torch_to_cupy(tensor)
        return _TensorBuffer(index=index, tensor=tensor, cupy_view=cupy_view)

    def preprocess_frame(self, frame: Any, buffer: _TensorBuffer, sample_index: int) -> None:
        self._active_buffer = buffer
        width = int(frame.width)
        height = int(frame.height)
        if width <= 0 or height <= 0:
            raise ValueError("invalid frame dimensions")
        if sample_index < 0 or sample_index >= self.window_frames:
            raise ValueError("sample index out of range")

        y_plane, pitch_y = self.wrap_plane(frame, 0, height)
        uv_plane, pitch_uv = self.wrap_plane(frame, 1, max(1, height // 2))
        total_pixels = self.target_width * self.target_height
        block = (256,)
        grid = ((total_pixels + block[0] - 1) // block[0],)
        self.kernel(
            grid,
            block,
            (
                y_plane,
                pitch_y,
                uv_plane,
                pitch_uv,
                buffer.cupy_view,
                int(sample_index),
                int(width),
                int(height),
                int(self.target_width),
                int(self.target_height),
            ),
        )

    def build_video_inputs(self, sample_count: int) -> dict[str, Any]:
        if self._active_buffer is None:
            raise RuntimeError("Qwen3 preprocessor has no active tensor buffer")
        pixel_values, video_grid_thw = pack_qwen3_video_patches(
            self._active_buffer.tensor,
            sample_count=sample_count,
            patch_size=self.patch_size,
            temporal_patch_size=self.temporal_patch_size,
            merge_size=self.merge_size,
        )
        return {
            "pixel_values_videos": pixel_values,
            "video_grid_thw": video_grid_thw,
            "sample_fps": float(getattr(self, "sample_fps", 2.0)),
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
            "patch_size": self.patch_size,
        }


class _QwenMockRunner:
    def generate(self, job: Any) -> str:
        return _json_dumps(
            [
                {
                    "label": "mock_qwen_object",
                    "confidence": 0.99,
                    "bbox_2d": [300, 250, 700, 800],
                    "point_2d": [500, 500],
                }
            ]
        )


def _load_runner_factory(path: str) -> Callable[..., Any]:
    module_name, sep, attr_name = path.partition(":")
    if not sep or not module_name or not attr_name:
        raise ValueError("runner_factory must be in module:attribute form")
    module = importlib.import_module(module_name)
    factory = getattr(module, attr_name)
    if not callable(factory):
        raise TypeError(f"runner_factory is not callable: {path}")
    return factory


class QwenVlAsync(MolmoVllmAsync):
    """Pass-through CUDA video node with async Qwen3-VL metadata projection."""

    def __init__(self, args: dict[str, Any]):
        params = dict(args)
        params.setdefault("model_id", "Qwen/Qwen3-VL-8B-Instruct")
        params.setdefault("backend", "transformers_direct")
        params.setdefault(
            "prompt",
            "Analyze the video window. Return JSON only as a list of visible objects with label, confidence, bbox_2d, and point_2d. Coordinates must be integers from 0 to 999 relative to the video frame.",
        )
        params.setdefault("prompt_id", "qwen_objects")
        params.setdefault("metadata_key_detections", "qwen_detections")
        params.setdefault("metadata_key_points", "qwen_points")
        params.setdefault("metadata_key_raw", "qwen_raw")
        params.setdefault("patch_size", 16)
        params.setdefault("qwen_target_height", 448)
        params.setdefault("qwen_target_width", 800)
        params.setdefault("molmo_frame_size", params["qwen_target_width"])
        params.setdefault("strict_zero_copy", True)
        super().__init__(params)

    def _validate_config(self) -> None:
        self.target_height = int(self._args.get("qwen_target_height", 448))
        self.target_width = int(self._args.get("qwen_target_width", 800))
        self.temporal_patch_size = int(self._args.get("temporal_patch_size", 2))
        self.merge_size = int(self._args.get("merge_size", 2))
        self.coordinate_scale = float(self._args.get("coordinate_scale", 1000.0))
        self.frame_size = self.target_width

        if self.backend not in ("transformers_direct", "mock"):
            raise ValueError("Qwen backend must be transformers_direct or mock")
        if self.target_height <= 0 or self.target_width <= 0:
            raise ValueError("qwen_target_height and qwen_target_width must be positive")
        if self.patch_size <= 0 or self.temporal_patch_size <= 0 or self.merge_size <= 0:
            raise ValueError("patch_size, temporal_patch_size, and merge_size must be positive")
        if self.temporal_patch_size != self.merge_size:
            raise ValueError("phase 1 requires temporal_patch_size to equal merge_size for Qwen3 timestamp expansion")
        if self.target_height % self.patch_size or self.target_width % self.patch_size:
            raise ValueError("Qwen target dimensions must be divisible by patch_size")
        if (self.target_height // self.patch_size) % self.merge_size:
            raise ValueError("Qwen target height patch grid must be divisible by merge_size")
        if (self.target_width // self.patch_size) % self.merge_size:
            raise ValueError("Qwen target width patch grid must be divisible by merge_size")
        if self.tensor_dtype not in ("fp16", "fp32"):
            raise ValueError("tensor_dtype must be fp16 or fp32")
        if self.sample_fps <= 0:
            raise ValueError("sample_fps must be positive")
        if self.window_frames <= 0:
            raise ValueError("window_frames must be positive")
        if self.window_stride != self.window_frames:
            raise ValueError("phase 1 supports tumbling windows only: window_stride must equal window_frames")
        if self.window_queue_size < 0:
            raise ValueError("window_queue_size must be non-negative")
        if self.max_inflight <= 0:
            raise ValueError("max_inflight must be positive")
        if self.visualize_ttl_frames < 0:
            raise ValueError("visualize_ttl_frames must be non-negative")
        if self.visualize_ttl_seconds is not None and self.visualize_ttl_seconds < 0:
            raise ValueError("visualize_ttl_seconds must be non-negative")
        if self.fallback_input_fps <= 0:
            raise ValueError("fallback_input_fps must be positive")
        if self.result_policy not in ("hold_latest", "no_hold"):
            raise ValueError("Qwen result_policy must be hold_latest or no_hold")
        if self.coordinate_scale <= 0:
            raise ValueError("coordinate_scale must be positive")

    def _init_runtime(self) -> None:
        if self.backend == "mock":
            self._runner = _QwenMockRunner()
        elif self.runner_factory:
            factory = _load_runner_factory(self.runner_factory)
            self._runner = factory(self._runner_config())
        else:
            from pyplumber.qwen.transformers_runner import TransformersQwen3VlDirectRunner

            self._runner = TransformersQwen3VlDirectRunner(self._runner_config())

        try:
            import cupy  # type: ignore
            import torch  # type: ignore

            if not torch.cuda.is_available():
                raise RuntimeError("torch CUDA is unavailable")

            kernel_path = _find_repo_file(
                "src/nodes/neural_net/vlm/qwen3_vl_preprocess.cu",
                Path(__file__).resolve(),
            )
            self._torch = torch
            self._cp = cupy
            self._preprocessor = _Qwen3Preprocessor(
                torch_mod=torch,
                cupy_mod=cupy,
                target_height=self.target_height,
                target_width=self.target_width,
                patch_size=self.patch_size,
                temporal_patch_size=self.temporal_patch_size,
                merge_size=self.merge_size,
                window_frames=self.window_frames,
                dtype_name=self.tensor_dtype,
                kernel_path=kernel_path,
            )
            self._preprocessor.sample_fps = self.sample_fps

            buffer_count = self.max_inflight + self.window_queue_size + 1
            for index in range(buffer_count):
                self._free_buffers.append(self._preprocessor.make_buffer(index))

            self._available = True
            for index in range(self.max_inflight):
                worker = threading.Thread(
                    target=self._worker_main,
                    name=f"QwenVlAsyncWorker-{index}",
                    daemon=True,
                )
                self._workers.append(worker)
                worker.start()
        except Exception as exc:
            self._available = False
            self._unavailable_reason = f"{type(exc).__name__}: {exc}"
            if self.strict_zero_copy:
                raise

    def _runner_config(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "model_dtype": str(self._args.get("model_dtype", "auto")),
            "attn_implementation": str(self._args.get("attn_implementation", "sdpa")),
            "device_map": str(self._args.get("device_map", "auto")),
            "trust_remote_code": bool(self._args.get("trust_remote_code", True)),
            "sample_fps": self.sample_fps,
            "temporal_patch_size": self.temporal_patch_size,
            "merge_size": self.merge_size,
        }

    def _run_backend(self, job: Any) -> str:
        runner = self._runner
        if runner is None:
            raise RuntimeError("Qwen runner is not initialized")
        if hasattr(runner, "generate"):
            return str(runner.generate(job))
        if callable(runner):
            return str(runner(job))
        raise TypeError("Qwen runner must be callable or expose generate(job)")

    def _execute_job(self, job: Any) -> _MolmoResult:
        latency_start = time.perf_counter()
        generated_text = ""
        try:
            generated_text = self._run_backend(job)
            latency_ms = (time.perf_counter() - latency_start) * 1000.0
            det_md, point_md, raw_md = parse_qwen_generated_text(
                generated_text,
                model_width=self.target_width,
                model_height=self.target_height,
                prompt_id=job.prompt_id,
                window_start_pts=job.start_pts,
                window_end_pts=job.end_pts,
                coordinate_scale=self.coordinate_scale,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            det_md = None
            point_md = None
            raw_md = {
                "schema": "qwen_raw_v1",
                "prompt_id": job.prompt_id,
                "window_start_pts": job.start_pts,
                "window_end_pts": job.end_pts,
                "generated_text": "",
                "parse_status": "backend_error",
                "error": f"{type(exc).__name__}: {exc}",
            }
        finally:
            self._release_buffer(job.buffer)

        expires_seconds = None
        if job.end_seconds is not None:
            expires_seconds = job.end_seconds + self._ttl_seconds()

        if self.debug_log_every_n > 0 and job.sequence % self.debug_log_every_n == 0:
            print(
                "qwen_vl_result",
                _json_dumps(
                    {
                        "sequence": job.sequence,
                        "sample_count": job.sample_count,
                        "start_seconds": job.start_seconds,
                        "end_seconds": job.end_seconds,
                        "has_points_json": point_md is not None,
                        "latency_ms": raw_md.get("latency_ms"),
                        "parse_status": raw_md.get("parse_status"),
                        "error": raw_md.get("error"),
                        "generated_text": generated_text,
                    }
                ),
                flush=True,
            )

        return _MolmoResult(
            detection_json=_json_dumps(det_md) if det_md else None,
            points_json=_json_dumps(point_md) if point_md else None,
            raw_json=_json_dumps(raw_md),
            end_seconds=job.end_seconds,
            expires_seconds=expires_seconds,
            end_frame_index=job.end_frame_index,
            expires_frame_index=job.end_frame_index + self.visualize_ttl_frames,
        )

    def _attach_unavailable(self, frame: Any) -> None:
        if self.strict_zero_copy:
            return
        payload = {
            "schema": "qwen_raw_v1",
            "prompt_id": self.prompt_id,
            "parse_status": "unavailable",
            "status": self._unavailable_reason or "unavailable",
        }
        frame.metadata[self.metadata_key_raw] = _json_dumps(payload)
