"""Molmo2/vLLM async video node for CUDA AVPlumber frames.

This module keeps the first AVP integration deliberately Python-side:

* AVFrame CUDA NV12 planes are wrapped as CuPy arrays without CPU copies.
* A CuPy RawModule kernel writes sampled frames into a torch CUDA tensor pool.
* A background worker consumes completed window tensors and publishes metadata.

The built-in ``mock`` backend is useful for graph, CUDA preprocess, and
visualization smoke tests. The ``vllm`` backend invokes the local vLLM Molmo2
runner; today's public vLLM path consumes CPU video arrays, while the CUDA
patch tensor path remains available for a future direct-tensor runner.
"""

from __future__ import annotations

import importlib
import json
import math
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

try:
    from pyplumber.node import PythonNode
except ModuleNotFoundError as exc:
    if exc.name != "_avplumber":
        raise

    class PythonNode:  # type: ignore[no-redef]
        """Minimal shim for parser/config tests before native AVP is built."""

        TYPE = "python_node"

        def __init__(self, args: dict[str, Any]):
            explicit_type = args.get("type")
            src_edges = args.get("src")
            dst_edges = args.get("dst")
            no_inputs = src_edges is None
            single_input = isinstance(src_edges, str)
            multi_input = isinstance(src_edges, list)
            no_outputs = dst_edges is None
            single_output = isinstance(dst_edges, str)
            multi_output = isinstance(dst_edges, list)

            if explicit_type is not None:
                node_type = str(explicit_type)
            elif single_input and single_output:
                node_type = "python_node_siso"
            elif single_input and multi_output:
                node_type = "python_node_simo"
            elif multi_input and single_output:
                node_type = "python_node_miso"
            elif multi_input and multi_output:
                node_type = "python_node_mimo"
            elif no_inputs and single_output:
                node_type = "python_node_so"
            elif no_inputs and multi_output:
                node_type = "python_node_mo"
            elif single_input and no_outputs:
                node_type = "python_node_si"
            elif multi_input and no_outputs:
                node_type = "python_node_mi"
            else:
                raise ValueError(f"Invalid number of source or destination edges: {src_edges} / {dst_edges}")

            self._args = args | {"type": node_type, "type_python": self.__class__.__name__}
            self._src = None
            self._dst = None

        @property
        def parameters(self) -> dict[str, Any]:
            return self._args


_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.IGNORECASE | re.DOTALL)
_NATIVE_POINT_TAG_RE = re.compile(r"<(?:points|tracks)\b(?P<attrs>[^>]*)/?>", re.IGNORECASE | re.DOTALL)
_NATIVE_COORD_ATTR_RE = re.compile(r'\bcoords="(?P<coords>[0-9\t:;, .]+)"', re.IGNORECASE)
_NATIVE_LABEL_ATTR_RE = re.compile(r'\b(?:alt|label|name)="(?P<label>[^"]+)"', re.IGNORECASE)


def _find_repo_file(relative_path: str, start: Path) -> Path:
    for parent in (start, *start.parents):
        candidate = parent / relative_path
        if candidate.exists():
            return candidate
    raise RuntimeError(f"missing repository file: {relative_path}")


def _json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _string_list(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        value = value.replace(";", ",")
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    raise TypeError(f"expected string or list, got {type(value).__name__}")


def _coerce_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _extract_json_object(text: str) -> tuple[dict[str, Any] | None, str]:
    """Extract the first JSON object from model output text."""

    if not text:
        return None, "empty"

    candidates: list[str] = []
    for match in _JSON_FENCE_RE.finditer(text):
        candidates.append(match.group(1).strip())
    candidates.append(text.strip())

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed, "ok"
        except json.JSONDecodeError:
            pass

        start = candidate.find("{")
        if start < 0:
            continue
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
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    snippet = candidate[start : index + 1]
                    try:
                        parsed = json.loads(snippet)
                    except json.JSONDecodeError:
                        break
                    if isinstance(parsed, dict):
                        return parsed, "ok"
                    break

    return None, "invalid_json"


def _scale_1000_coord(value: Any, frame_size: int) -> float | None:
    number = _coerce_float(value)
    if number is None:
        return None
    number = min(1000.0, max(0.0, number))
    return number * float(frame_size) / 1000.0


def _parse_object_conf(obj: dict[str, Any]) -> float:
    conf = _coerce_float(obj.get("confidence", obj.get("conf", 1.0)))
    if conf is None:
        return 1.0
    return min(1.0, max(0.0, conf))


def _parse_object_label(obj: dict[str, Any], fallback_index: int) -> str:
    label = obj.get("label", obj.get("name", "object"))
    if not isinstance(label, str) or not label.strip():
        return f"object_{fallback_index}"
    return label.strip()


def _build_point_metadata(
    points: list[tuple[float, float, float]],
    *,
    frame_size: int,
) -> dict[str, Any] | None:
    if not points:
        return None

    keypoints: list[float] = []
    for px, py, conf in points:
        keypoints.extend([px, py, conf])

    return {
        "schema": "pose_keypoints_v1",
        "coord_space": "model",
        "model_width": frame_size,
        "model_height": frame_size,
        "num_keypoints": len(keypoints) // 3,
        "poses": [
            {
                "label": "molmo_points",
                "conf": max(keypoints[2::3]) if len(keypoints) >= 3 else 1.0,
                "keypoints": keypoints,
            }
        ],
    }


def _extract_native_molmo_timed_points(
    generated_text: str,
    *,
    frame_size: int,
) -> tuple[list[_TimedPoint], int]:
    """Extract Molmo2 native ``<points>/<tracks coords="...">`` outputs.

    Molmo2 VideoPoint commonly returns coordinates as text tags instead of
    JSON. The coordinate payload uses 0..1000 coordinates and commonly groups
    tracks as ``time_seconds object_id x y ...``.
    """

    points: list[_TimedPoint] = []
    invalid = 0

    for tag_match in _NATIVE_POINT_TAG_RE.finditer(generated_text):
        attrs = tag_match.group("attrs")
        coord_match = _NATIVE_COORD_ATTR_RE.search(attrs)
        if coord_match is None:
            invalid += 1
            continue

        label_match = _NATIVE_LABEL_ATTR_RE.search(attrs)
        label_prefix = label_match.group("label").strip() if label_match else "molmo"
        label_prefix = label_prefix or "molmo"

        for segment in re.split(r"[\t:;,]+", coord_match.group("coords")):
            numbers = re.findall(r"[0-9]+(?:\.[0-9]+)?", segment)
            if len(numbers) < 3:
                continue

            if len(numbers) % 3 == 0:
                frame_id = 0.0
                triples = numbers
            elif (len(numbers) - 1) % 3 == 0:
                frame_value = _coerce_float(numbers[0])
                frame_id = frame_value if frame_value is not None else 0.0
                triples = numbers[1:]
            else:
                invalid += 1
                continue

            for index in range(0, len(triples) - 2, 3):
                object_id = triples[index]
                px = _scale_1000_coord(triples[index + 1], frame_size)
                py = _scale_1000_coord(triples[index + 2], frame_size)
                if px is None or py is None:
                    invalid += 1
                    continue

                points.append(
                    _TimedPoint(
                        time_seconds=frame_id,
                        object_id=f"{label_prefix}:{object_id}" if object_id else label_prefix,
                        x=px,
                        y=py,
                        conf=1.0,
                    )
                )

    return points, invalid


def _extract_native_molmo_points(
    generated_text: str,
    *,
    frame_size: int,
) -> tuple[list[tuple[float, float, float]], int]:
    """Return one display point per object from native Molmo point output."""

    timed_points, invalid = _extract_native_molmo_timed_points(
        generated_text,
        frame_size=frame_size,
    )

    latest_by_id: dict[str, _TimedPoint] = {}
    for point in timed_points:
        current = latest_by_id.get(point.object_id)
        if current is None or point.time_seconds >= current.time_seconds:
            latest_by_id[point.object_id] = point

    return [(point.x, point.y, point.conf) for point in latest_by_id.values()], invalid


def parse_molmo_generated_text(
    generated_text: str,
    *,
    frame_size: int,
    prompt_id: str,
    window_start_pts: str,
    window_end_pts: str,
    latency_ms: float | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
    """Parse Molmo JSON-ish output into AVP draw metadata payloads.

    Input coordinates are expected to be integers/floats in 0..1000 relative to
    the square Molmo input. Output coordinates are model-space pixels relative
    to ``frame_size``.
    """

    parsed, parse_status = _extract_json_object(generated_text)
    raw_md: dict[str, Any] = {
        "schema": "molmo_raw_v1",
        "prompt_id": prompt_id,
        "window_start_pts": window_start_pts,
        "window_end_pts": window_end_pts,
        "generated_text": generated_text,
        "parse_status": parse_status,
    }
    if latency_ms is not None:
        raw_md["latency_ms"] = round(float(latency_ms), 3)

    if parsed is None:
        native_points, invalid_native_count = _extract_native_molmo_points(
            generated_text,
            frame_size=frame_size,
        )
        if native_points:
            raw_md["parse_status"] = "native_points"
            raw_md["object_count"] = len(native_points)
            raw_md["invalid_object_count"] = invalid_native_count
            raw_md["detection_count"] = 0
            raw_md["point_count"] = len(native_points)
            return None, _build_point_metadata(native_points, frame_size=frame_size), raw_md

        return None, None, raw_md

    objects = parsed.get("objects", [])
    if not isinstance(objects, list):
        raw_md["parse_status"] = "no_objects"
        return None, None, raw_md

    detections: list[dict[str, Any]] = []
    keypoints: list[float] = []
    invalid_object_count = 0

    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            invalid_object_count += 1
            continue

        label = _parse_object_label(obj, index)
        conf = _parse_object_conf(obj)
        emitted = False

        bbox = obj.get("bbox", obj.get("box", obj.get("xyxy")))
        if isinstance(bbox, list) and len(bbox) >= 4:
            x1 = _scale_1000_coord(bbox[0], frame_size)
            y1 = _scale_1000_coord(bbox[1], frame_size)
            x2 = _scale_1000_coord(bbox[2], frame_size)
            y2 = _scale_1000_coord(bbox[3], frame_size)
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
                            "source": "molmo",
                        }
                    )
                    emitted = True

        point = obj.get("point", obj.get("center", obj.get("xy")))
        if isinstance(point, list) and len(point) >= 2:
            px = _scale_1000_coord(point[0], frame_size)
            py = _scale_1000_coord(point[1], frame_size)
            if px is not None and py is not None:
                keypoints.extend([px, py, conf])
                emitted = True

        if not emitted:
            invalid_object_count += 1

    raw_md["object_count"] = len(objects)
    raw_md["invalid_object_count"] = invalid_object_count
    raw_md["detection_count"] = len(detections)
    raw_md["point_count"] = len(keypoints) // 3
    if not detections and not keypoints:
        raw_md["parse_status"] = "no_valid_objects"

    det_md = None
    if detections:
        det_md = {
            "schema": "yolo_detections_v1",
            "coord_space": "model",
            "model_width": frame_size,
            "model_height": frame_size,
            "detections": detections,
        }

    points = []
    for index in range(0, len(keypoints), 3):
        points.append((keypoints[index], keypoints[index + 1], keypoints[index + 2]))
    point_md = _build_point_metadata(points, frame_size=frame_size)

    return det_md, point_md, raw_md


@dataclass
class _TimedPoint:
    time_seconds: float
    object_id: str
    x: float
    y: float
    conf: float = 1.0


@dataclass
class _TensorBuffer:
    index: int
    tensor: Any
    cupy_view: Any


@dataclass
class _BufferedFrame:
    frame: Any
    seconds: float | None
    frame_index: int


@dataclass
class _WindowJob:
    sequence: int
    buffer: _TensorBuffer
    prompt: str
    prompt_id: str
    sample_count: int
    start_pts: str
    end_pts: str
    start_seconds: float | None
    end_seconds: float | None
    start_frame_index: int
    end_frame_index: int
    video_inputs: dict[str, Any]
    enqueue_time: float


@dataclass
class _MolmoResult:
    detection_json: str | None
    points_json: str | None
    raw_json: str
    end_seconds: float | None
    expires_seconds: float | None
    end_frame_index: int
    expires_frame_index: int
    timed_points: tuple[_TimedPoint, ...] = ()


class _MolmoPreprocessor:
    def __init__(
        self,
        *,
        torch_mod: Any,
        cupy_mod: Any,
        frame_size: int,
        patch_size: int,
        window_frames: int,
        dtype_name: str,
        kernel_path: Path,
    ) -> None:
        self.torch = torch_mod
        self.cp = cupy_mod
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.window_frames = window_frames
        self.dtype_name = dtype_name
        self.kernel_path = kernel_path
        self.patch_grid = frame_size // patch_size
        self.patches_per_frame = self.patch_grid * self.patch_grid
        self.patch_vec = patch_size * patch_size * 3

        code = kernel_path.read_text(encoding="utf-8")
        self.module = self.cp.RawModule(
            code=code,
            options=self._raw_module_options(),
            name_expressions=(
                "kMolmo2PreprocessNV12_fp16",
                "kMolmo2PreprocessNV12_fp32",
            ),
        )
        kernel_name = "kMolmo2PreprocessNV12_fp16" if dtype_name == "fp16" else "kMolmo2PreprocessNV12_fp32"
        self.kernel = self.module.get_function(kernel_name)

    def _raw_module_options(self) -> tuple[str, ...]:
        include_dirs = (
            str(self.kernel_path.parent),
            "/usr/local/cuda/include",
            "/usr/local/cuda-13.0/include",
            "/usr/local/cuda-13.0/targets/x86_64-linux/include",
        )
        return ("--std=c++14",) + tuple(f"-I{path}" for path in include_dirs if Path(path).exists())

    def _torch_dtype(self) -> Any:
        if self.dtype_name == "fp16":
            return self.torch.float16
        if self.dtype_name == "fp32":
            return self.torch.float32
        raise ValueError(f"unsupported tensor dtype: {self.dtype_name}")

    def _torch_to_cupy(self, tensor: Any) -> Any:
        if hasattr(self.cp, "from_dlpack"):
            try:
                return self.cp.from_dlpack(tensor)
            except TypeError:
                capsule = self.torch.utils.dlpack.to_dlpack(tensor)
                return self.cp.from_dlpack(capsule)
        capsule = self.torch.utils.dlpack.to_dlpack(tensor)
        return self.cp.fromDlpack(capsule)

    def make_buffer(self, index: int) -> _TensorBuffer:
        tensor = self.torch.empty(
            (self.window_frames, self.patches_per_frame, self.patch_vec),
            device="cuda",
            dtype=self._torch_dtype(),
        )
        cupy_view = self._torch_to_cupy(tensor)
        return _TensorBuffer(index=index, tensor=tensor, cupy_view=cupy_view)

    def wrap_plane(self, frame: Any, plane_index: int, rows: int) -> Any:
        ptr = int(frame.data_ptr[plane_index]) if len(frame.data_ptr) > plane_index else 0
        pitch = int(frame.linesize[plane_index]) if len(frame.linesize) > plane_index else 0
        if ptr <= 0 or pitch <= 0 or rows <= 0:
            raise ValueError("invalid CUDA frame plane pointer or pitch")
        size = pitch * rows
        flat = self.cp.ndarray(
            shape=(size,),
            dtype=self.cp.uint8,
            memptr=self.cp.cuda.MemoryPointer(self.cp.cuda.UnownedMemory(ptr, size, owner=frame), 0),
        )
        return flat.reshape(rows, pitch), pitch

    def preprocess_frame(self, frame: Any, buffer: _TensorBuffer, sample_index: int) -> None:
        width = int(frame.width)
        height = int(frame.height)
        if width <= 0 or height <= 0:
            raise ValueError("invalid frame dimensions")
        if sample_index < 0 or sample_index >= self.window_frames:
            raise ValueError("sample index out of range")

        y_plane, pitch_y = self.wrap_plane(frame, 0, height)
        uv_plane, pitch_uv = self.wrap_plane(frame, 1, max(1, height // 2))

        total_pixels = self.frame_size * self.frame_size
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
                int(self.frame_size),
                int(self.patch_size),
            ),
        )

    def synchronize(self) -> None:
        self.cp.cuda.get_current_stream().synchronize()

    def build_video_inputs(self, sample_count: int) -> dict[str, Any]:
        torch = self.torch
        grid_h = self.patch_grid
        grid_w = self.patch_grid
        pool_h = 3
        pool_w = 3
        idx = torch.arange(self.patches_per_frame, device="cuda", dtype=torch.long).reshape(grid_h, grid_w)

        h_pad = pool_h * ((grid_h + pool_h - 1) // pool_h) - grid_h
        w_pad = pool_w * ((grid_w + pool_w - 1) // pool_w) - grid_w
        if h_pad or w_pad:
            idx = torch.nn.functional.pad(
                idx,
                (w_pad // 2, (w_pad + 1) // 2, h_pad // 2, (h_pad + 1) // 2),
                value=-1,
            )

        padded_h, padded_w = idx.shape
        pooled = (
            idx.reshape(padded_h // pool_h, pool_h, padded_w // pool_w, pool_w)
            .permute(0, 2, 1, 3)
            .reshape(-1, pool_h * pool_w)
        )

        per_frame: list[Any] = []
        for frame_index in range(sample_count):
            offset = frame_index * self.patches_per_frame
            per_frame.append(torch.where(pooled >= 0, pooled + offset, pooled))
        token_pooling = torch.cat(per_frame, dim=0)
        token_count = int(token_pooling.shape[0])
        return {
            "token_pooling": token_pooling,
            "num_pooled_patches": torch.tensor([token_count], device="cuda", dtype=torch.long),
            "video_tokens": torch.ones((token_count,), device="cuda", dtype=torch.bool),
            "num_video_tokens": torch.tensor([token_count], device="cuda", dtype=torch.long),
            "video_grids": torch.tensor(
                [[padded_h // pool_h, padded_w // pool_w]],
                device="cuda",
                dtype=torch.long,
            ),
        }


class _MockRunner:
    def generate(self, job: _WindowJob) -> str:
        # Stable, visible center-ish object for draw node smoke tests.
        return _json_dumps(
            {
                "objects": [
                    {
                        "label": "mock_object",
                        "confidence": 0.99,
                        "bbox": [360, 260, 640, 720],
                        "point": [500, 500],
                    }
                ]
            }
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


class MolmoVllmAsync(PythonNode):
    """Pass-through CUDA video node with async Molmo/vLLM metadata projection."""

    def __init__(self, args: dict[str, Any]):
        super().__init__(args)
        params = args

        self.model_id = str(params.get("model_id", "allenai/Molmo2-VideoPoint-4B"))
        self.prompt = str(
            params.get(
                "prompt",
                "Find people, vehicles, balls, and important objects. Return JSON only with objects containing label, confidence, bbox, and point. Coordinates must be integers from 0 to 1000 relative to the image.",
            )
        )
        self.prompt_id = str(params.get("prompt_id", "default_objects"))
        self.backend = str(params.get("backend", "vllm")).lower()
        self.runner_factory = str(params.get("runner_factory", ""))
        self.strict_zero_copy = bool(params.get("strict_zero_copy", True))

        self.frame_size = int(params.get("molmo_frame_size", 378))
        self.patch_size = int(params.get("patch_size", 14))
        self.allow_non_default_frame_size = bool(params.get("allow_non_default_frame_size", False))
        self.tensor_dtype = str(params.get("tensor_dtype", "fp16")).lower()

        self.sample_fps = float(params.get("sample_fps", 8.0))
        self.window_frames = int(params.get("window_frames", 16))
        self.window_stride = int(params.get("window_stride", self.window_frames))
        self.window_queue_size = int(params.get("window_queue_size", 1))
        self.sidecar_urls = _string_list(params.get("sidecar_urls"))
        max_inflight_param = params.get("max_inflight")
        self.max_inflight = (
            int(max_inflight_param) if max_inflight_param not in (None, "") else max(1, len(self.sidecar_urls))
        )
        self.visualize_ttl_frames = int(params.get("visualize_ttl_frames", self.window_frames))
        ttl_seconds_param = params.get("visualize_ttl_seconds")
        self.visualize_ttl_seconds = None if ttl_seconds_param in (None, "") else float(ttl_seconds_param)
        self.blocking_visualization = bool(params.get("blocking_visualization", False))
        result_policy_param = str(params.get("result_policy", "hold_latest")).lower()
        default_worker_join_ms = 300000 if result_policy_param == "delayed_tracks" else 2000
        worker_join_timeout_param = params.get("worker_join_timeout_ms", default_worker_join_ms)
        self.worker_join_timeout_ms = int(
            default_worker_join_ms if worker_join_timeout_param in (None, "") else worker_join_timeout_param
        )
        self.fallback_input_fps = float(params.get("fallback_input_fps", 30.0))

        self.metadata_key_detections = str(params.get("metadata_key_detections", "molmo_detections"))
        self.metadata_key_points = str(params.get("metadata_key_points", "molmo_points"))
        self.metadata_key_raw = str(params.get("metadata_key_raw", "molmo_raw"))
        self.result_policy = result_policy_param
        self.debug_log_every_n = int(params.get("debug_log_every_n", 0))
        self.delayed_track_hold_seconds = float(params.get("delayed_track_hold_seconds", 0.0))

        self.max_new_tokens = int(params.get("max_new_tokens", 512))
        self.temperature = float(params.get("temperature", 0.0))
        self.gpu_memory_utilization = float(params.get("gpu_memory_utilization", 0.75))
        self.max_model_len = int(params.get("max_model_len", 8192))
        self.max_num_seqs = int(params.get("max_num_seqs", 1))
        self.max_num_batched_tokens = int(params.get("max_num_batched_tokens", 8192))

        self._validate_config()

        self._torch = None
        self._cp = None
        self._preprocessor: _MolmoPreprocessor | None = None
        self._runner: Any = None
        self._available = False
        self._unavailable_reason = ""

        self._frame_index = 0
        self._next_sample_seconds: float | None = None
        self._fallback_sample_every_n = max(1, round(self.fallback_input_fps / max(self.sample_fps, 0.001)))
        self._current_buffer: _TensorBuffer | None = None
        self._current_sample_count = 0
        self._current_start_pts = ""
        self._current_start_seconds: float | None = None
        self._current_start_frame_index = 0
        self._current_end_pts = ""
        self._current_end_seconds: float | None = None
        self._current_end_frame_index = 0
        self._sequence = 0

        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._stop_event = threading.Event()
        self._free_buffers: deque[_TensorBuffer] = deque()
        self._pending_jobs: deque[_WindowJob] = deque()
        self._active_jobs = 0
        self._latest_result: _MolmoResult | None = None
        self._completed_delayed_jobs: dict[int, tuple[_WindowJob, _MolmoResult]] = {}
        self._next_delayed_sequence = 1
        self._delayed_frames: deque[_BufferedFrame] = deque()
        self._eof_flushed = False
        self._workers: list[threading.Thread] = []

        self._init_runtime()

    def _validate_config(self) -> None:
        if self.frame_size <= 0:
            raise ValueError("molmo_frame_size must be positive")
        if self.patch_size <= 0:
            raise ValueError("patch_size must be positive")
        if self.frame_size % self.patch_size != 0:
            raise ValueError("molmo_frame_size must be divisible by patch_size")
        if self.frame_size != 378 and not self.allow_non_default_frame_size:
            raise ValueError("non-default molmo_frame_size requires allow_non_default_frame_size=true")
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
        if self.backend not in ("vllm", "mock"):
            raise ValueError("backend must be vllm or mock")
        if self.result_policy not in ("hold_latest", "no_hold", "delayed_tracks"):
            raise ValueError("result_policy must be hold_latest, no_hold, or delayed_tracks")
        if self.delayed_track_hold_seconds < 0:
            raise ValueError("delayed_track_hold_seconds must be non-negative")

    def _init_runtime(self) -> None:
        if self.backend == "mock":
            self._runner = _MockRunner()
        elif self.runner_factory:
            factory = _load_runner_factory(self.runner_factory)
            self._runner = factory(
                {
                    "model_id": self.model_id,
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "gpu_memory_utilization": self.gpu_memory_utilization,
                    "max_model_len": self.max_model_len,
                    "max_num_seqs": self.max_num_seqs,
                    "max_num_batched_tokens": self.max_num_batched_tokens,
                    "frame_size": self.frame_size,
                    "patch_size": self.patch_size,
                    "sample_fps": self.sample_fps,
                    "sidecar_url": str(self._args.get("sidecar_url", "")),
                    "sidecar_urls": self.sidecar_urls,
                    "model_dtype": str(self._args.get("model_dtype", "float16")),
                    "prompt_style": str(self._args.get("prompt_style", "")),
                    "mm_processor_cache_gb": float(self._args.get("mm_processor_cache_gb", 0)),
                    "cpu_offload_gb": float(self._args.get("cpu_offload_gb", 0)),
                    "tensor_parallel_size": int(self._args.get("tensor_parallel_size", 1)),
                    "enforce_eager": bool(self._args.get("enforce_eager", True)),
                    "seed": int(self._args.get("seed", 0)),
                }
            )
        elif self.backend == "vllm":
            from pyplumber.molmo.vllm_runner import VllmMolmo2VideoRunner

            self._runner = VllmMolmo2VideoRunner(
                {
                    "model_id": self.model_id,
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": self.temperature,
                    "gpu_memory_utilization": self.gpu_memory_utilization,
                    "max_model_len": self.max_model_len,
                    "max_num_seqs": self.max_num_seqs,
                    "max_num_batched_tokens": self.max_num_batched_tokens,
                    "frame_size": self.frame_size,
                    "patch_size": self.patch_size,
                    "sample_fps": self.sample_fps,
                    "sidecar_url": str(self._args.get("sidecar_url", "")),
                    "sidecar_urls": self.sidecar_urls,
                    "model_dtype": str(self._args.get("model_dtype", "float16")),
                    "prompt_style": str(self._args.get("prompt_style", "")),
                    "mm_processor_cache_gb": float(self._args.get("mm_processor_cache_gb", 0)),
                    "cpu_offload_gb": float(self._args.get("cpu_offload_gb", 0)),
                    "tensor_parallel_size": int(self._args.get("tensor_parallel_size", 1)),
                    "enforce_eager": bool(self._args.get("enforce_eager", True)),
                    "seed": int(self._args.get("seed", 0)),
                }
            )
        else:
            self._unavailable_reason = (
                "vLLM direct tensor runner is not configured; use backend='mock' for smoke tests "
                "or pass runner_factory='module:factory'"
            )
            if self.strict_zero_copy:
                raise RuntimeError(self._unavailable_reason)
            return

        try:
            import torch  # type: ignore
            import cupy  # type: ignore

            if not torch.cuda.is_available():
                raise RuntimeError("torch CUDA is unavailable")

            kernel_path = _find_repo_file(
                "src/nodes/neural_net/preprocess/molmo2_preprocess.cu",
                Path(__file__).resolve(),
            )

            self._torch = torch
            self._cp = cupy
            self._preprocessor = _MolmoPreprocessor(
                torch_mod=torch,
                cupy_mod=cupy,
                frame_size=self.frame_size,
                patch_size=self.patch_size,
                window_frames=self.window_frames,
                dtype_name=self.tensor_dtype,
                kernel_path=kernel_path,
            )

            buffer_count = self.max_inflight + self.window_queue_size + 1
            for index in range(buffer_count):
                self._free_buffers.append(self._preprocessor.make_buffer(index))

            self._available = True
            for index in range(self.max_inflight):
                worker = threading.Thread(
                    target=self._worker_main,
                    name=f"MolmoVllmAsyncWorker-{index}",
                    daemon=True,
                )
                self._workers.append(worker)
                worker.start()
        except Exception as exc:
            self._available = False
            self._unavailable_reason = f"{type(exc).__name__}: {exc}"
            if self.strict_zero_copy:
                raise

    def _frame_seconds(self, frame: Any) -> float | None:
        try:
            pts = frame.pts
            timestamp = int(pts.timestamp)
            timebase = pts.timebase
            if timestamp <= -9223372036854775800 or not timebase or int(timebase.den) == 0:
                return None
            return float(timestamp) * float(timebase.num) / float(timebase.den)
        except Exception:
            return None

    def _is_eof_frame(self, frame: Any) -> bool:
        try:
            return int(frame.pts.timestamp) <= -9223372036854775800
        except Exception:
            return False

    def _frame_pts_string(self, frame: Any) -> str:
        try:
            return str(frame.pts)
        except Exception:
            return ""

    def _should_sample(self, frame_seconds: float | None) -> bool:
        if frame_seconds is None:
            return (self._frame_index % self._fallback_sample_every_n) == 1

        interval = 1.0 / self.sample_fps
        if self._next_sample_seconds is None:
            self._next_sample_seconds = frame_seconds
        if frame_seconds + 1e-6 < self._next_sample_seconds:
            return False
        while self._next_sample_seconds <= frame_seconds + 1e-6:
            self._next_sample_seconds += interval
        return True

    def _acquire_buffer(self) -> _TensorBuffer | None:
        with self._lock:
            if not self._free_buffers:
                return None
            return self._free_buffers.popleft()

    def _release_buffer(self, buffer: _TensorBuffer) -> None:
        with self._lock:
            self._free_buffers.append(buffer)

    def _enqueue_job(self, job: _WindowJob) -> None:
        with self._condition:
            pending_capacity = max(1, self.window_queue_size)
            while len(self._pending_jobs) >= pending_capacity:
                dropped = self._pending_jobs.popleft()
                self._release_buffer(dropped.buffer)
            self._pending_jobs.append(job)
            self._condition.notify()

    def _finish_current_job(self) -> _WindowJob | None:
        if self._current_buffer is None or self._current_sample_count <= 0 or self._preprocessor is None:
            return None

        self._preprocessor.synchronize()
        self._sequence += 1
        video_inputs = self._preprocessor.build_video_inputs(self._current_sample_count)
        job = _WindowJob(
            sequence=self._sequence,
            buffer=self._current_buffer,
            prompt=self.prompt,
            prompt_id=self.prompt_id,
            sample_count=self._current_sample_count,
            start_pts=self._current_start_pts,
            end_pts=self._current_end_pts,
            start_seconds=self._current_start_seconds,
            end_seconds=self._current_end_seconds,
            start_frame_index=self._current_start_frame_index,
            end_frame_index=self._current_end_frame_index,
            video_inputs=video_inputs,
            enqueue_time=time.perf_counter(),
        )
        self._current_buffer = None
        self._current_sample_count = 0
        self._current_start_pts = ""
        self._current_start_seconds = None
        self._current_start_frame_index = 0
        self._current_end_pts = ""
        self._current_end_seconds = None
        self._current_end_frame_index = 0
        return job

    def _process_sample(self, frame: Any, frame_seconds: float | None) -> _WindowJob | None:
        if not self._available or self._preprocessor is None:
            return None

        if self._current_buffer is None:
            self._current_buffer = self._acquire_buffer()
            if self._current_buffer is None:
                return None
            self._current_sample_count = 0
            self._current_start_pts = self._frame_pts_string(frame)
            self._current_start_seconds = frame_seconds
            self._current_start_frame_index = self._frame_index

        try:
            self._preprocessor.preprocess_frame(frame, self._current_buffer, self._current_sample_count)
        except Exception as exc:
            self._unavailable_reason = f"preprocess_error:{type(exc).__name__}: {exc}"
            self._release_buffer(self._current_buffer)
            self._current_buffer = None
            self._current_sample_count = 0
            return None

        self._current_sample_count += 1
        self._current_end_pts = self._frame_pts_string(frame)
        self._current_end_seconds = frame_seconds
        self._current_end_frame_index = self._frame_index
        if self._current_sample_count < self.window_frames:
            return None

        job = self._finish_current_job()
        if job is None:
            return None
        if self.blocking_visualization or self.result_policy == "delayed_tracks":
            return job
        self._enqueue_job(job)
        return None

    def _run_backend(self, job: _WindowJob) -> str:
        runner = self._runner
        if runner is None:
            raise RuntimeError("Molmo runner is not initialized")
        if hasattr(runner, "generate"):
            return str(runner.generate(job))
        if callable(runner):
            return str(runner(job))
        raise TypeError("Molmo runner must be callable or expose generate(job)")

    def _ttl_seconds(self) -> float:
        if self.visualize_ttl_seconds is not None:
            return float(self.visualize_ttl_seconds)
        return float(self.visualize_ttl_frames) / self.fallback_input_fps

    def _execute_job(self, job: _WindowJob) -> _MolmoResult:
        latency_start = time.perf_counter()
        timed_points: tuple[_TimedPoint, ...] = ()
        generated_text = ""
        try:
            generated_text = self._run_backend(job)
            latency_ms = (time.perf_counter() - latency_start) * 1000.0
            native_timed_points, _ = _extract_native_molmo_timed_points(
                generated_text,
                frame_size=self.frame_size,
            )
            timed_points = tuple(native_timed_points)
            det_md, point_md, raw_md = parse_molmo_generated_text(
                generated_text,
                frame_size=self.frame_size,
                prompt_id=job.prompt_id,
                window_start_pts=job.start_pts,
                window_end_pts=job.end_pts,
                latency_ms=latency_ms,
            )
        except Exception as exc:
            det_md = None
            point_md = None
            raw_md = {
                "schema": "molmo_raw_v1",
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
                "molmo_vllm_result",
                _json_dumps(
                    {
                        "sequence": job.sequence,
                        "sample_count": job.sample_count,
                        "start_seconds": job.start_seconds,
                        "end_seconds": job.end_seconds,
                        "timed_point_count": len(timed_points),
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
            timed_points=timed_points,
        )

    def _worker_main(self) -> None:
        while not self._stop_event.is_set():
            with self._condition:
                while not self._pending_jobs and not self._stop_event.is_set():
                    self._condition.wait(timeout=0.1)
                if self._stop_event.is_set():
                    break
                job = self._pending_jobs.popleft()
                self._active_jobs += 1

            if job is None:
                continue

            try:
                result = self._execute_job(job)
                with self._lock:
                    self._latest_result = result
                    if self.result_policy == "delayed_tracks":
                        self._completed_delayed_jobs[job.sequence] = (job, result)
            finally:
                with self._condition:
                    self._active_jobs -= 1
                    self._condition.notify_all()

    def _attach_unavailable(self, frame: Any) -> None:
        if self.strict_zero_copy:
            return
        payload = {
            "schema": "molmo_raw_v1",
            "prompt_id": self.prompt_id,
            "parse_status": "unavailable",
            "status": self._unavailable_reason or "unavailable",
        }
        frame.metadata[self.metadata_key_raw] = _json_dumps(payload)

    def _result_is_fresh(self, result: _MolmoResult, frame_seconds: float | None) -> bool:
        if frame_seconds is not None and result.end_seconds is not None and result.expires_seconds is not None:
            return result.end_seconds <= frame_seconds <= result.expires_seconds
        return result.end_frame_index <= self._frame_index <= result.expires_frame_index

    def _attach_result(self, frame: Any, result: _MolmoResult) -> None:
        if result.detection_json:
            frame.metadata[self.metadata_key_detections] = result.detection_json
        if result.points_json:
            frame.metadata[self.metadata_key_points] = result.points_json
        frame.metadata[self.metadata_key_raw] = result.raw_json

    def _attach_cached_result(self, frame: Any, frame_seconds: float | None) -> None:
        with self._lock:
            result = self._latest_result
        if result is None:
            return

        if not self._result_is_fresh(result, frame_seconds):
            return

        self._attach_result(frame, result)

    def _nearest_delayed_frame(
        self,
        *,
        target_seconds: float | None,
        target_frame_index: int,
        max_seconds_error: float,
    ) -> _BufferedFrame | None:
        best: _BufferedFrame | None = None
        best_error = float("inf")
        for buffered in self._delayed_frames:
            if target_seconds is not None and buffered.seconds is not None:
                error = abs(buffered.seconds - target_seconds)
                if error > max_seconds_error:
                    continue
            else:
                error = abs(buffered.frame_index - target_frame_index)

            if error < best_error:
                best = buffered
                best_error = error
        return best

    def _delayed_frames_near_target(
        self,
        *,
        target_seconds: float | None,
        target_frame_index: int,
        max_seconds_error: float,
    ) -> list[tuple[_BufferedFrame, float]]:
        if self.delayed_track_hold_seconds <= 0:
            buffered = self._nearest_delayed_frame(
                target_seconds=target_seconds,
                target_frame_index=target_frame_index,
                max_seconds_error=max_seconds_error,
            )
            if buffered is None:
                return []
            if target_seconds is not None and buffered.seconds is not None:
                return [(buffered, abs(buffered.seconds - target_seconds))]
            return [(buffered, float(abs(buffered.frame_index - target_frame_index)))]

        max_frame_error = max(1, int(round(self.delayed_track_hold_seconds * self.fallback_input_fps / 2.0)))
        max_time_error = max(max_seconds_error, self.delayed_track_hold_seconds / 2.0)
        matches: list[tuple[_BufferedFrame, float]] = []
        for buffered in self._delayed_frames:
            if target_seconds is not None and buffered.seconds is not None:
                error = abs(buffered.seconds - target_seconds)
                if error <= max_time_error:
                    matches.append((buffered, error))
            else:
                frame_error = abs(buffered.frame_index - target_frame_index)
                if frame_error <= max_frame_error:
                    matches.append((buffered, float(frame_error)))

        return matches

    def _attach_delayed_track_result(self, job: _WindowJob, result: _MolmoResult) -> None:
        points_by_frame_object: dict[int, dict[str, tuple[float, float, float, float]]] = {}
        raw_frame_index = job.end_frame_index
        max_seconds_error = max(0.075, 0.5 / max(self.fallback_input_fps, 0.001))

        for point in result.timed_points:
            target_seconds = None
            if job.start_seconds is not None:
                target_seconds = job.start_seconds + point.time_seconds
            target_frame_index = job.start_frame_index + int(round(point.time_seconds * self.fallback_input_fps))
            matches = self._delayed_frames_near_target(
                target_seconds=target_seconds,
                target_frame_index=target_frame_index,
                max_seconds_error=max_seconds_error,
            )
            for buffered, error in matches:
                frame_points = points_by_frame_object.setdefault(buffered.frame_index, {})
                current = frame_points.get(point.object_id)
                if current is None or error < current[3]:
                    frame_points[point.object_id] = (point.x, point.y, point.conf, error)
                raw_frame_index = buffered.frame_index

        points_by_frame: dict[int, list[tuple[float, float, float]]] = {
            frame_index: [(x, y, conf) for x, y, conf, _error in points_by_object.values()]
            for frame_index, points_by_object in points_by_frame_object.items()
        }

        if self.debug_log_every_n > 0 and job.sequence % self.debug_log_every_n == 0:
            print(
                "molmo_vllm_delayed_attach",
                _json_dumps(
                    {
                        "sequence": job.sequence,
                        "timed_point_count": len(result.timed_points),
                        "hold_seconds": self.delayed_track_hold_seconds,
                        "matched_frame_count": len(points_by_frame),
                        "matched_frame_indices": sorted(points_by_frame.keys()),
                    }
                ),
                flush=True,
            )

        for buffered in self._delayed_frames:
            points = points_by_frame.get(buffered.frame_index)
            if points:
                point_md = _build_point_metadata(points, frame_size=self.frame_size)
                if point_md is not None:
                    buffered.frame.metadata[self.metadata_key_points] = _json_dumps(point_md)
                buffered.frame.metadata[self.metadata_key_raw] = result.raw_json
            elif buffered.frame_index == raw_frame_index:
                buffered.frame.metadata[self.metadata_key_raw] = result.raw_json

        if not result.timed_points and result.points_json:
            buffered = self._nearest_delayed_frame(
                target_seconds=job.end_seconds,
                target_frame_index=job.end_frame_index,
                max_seconds_error=max_seconds_error,
            )
            if buffered is not None:
                buffered.frame.metadata[self.metadata_key_points] = result.points_json
                buffered.frame.metadata[self.metadata_key_raw] = result.raw_json

    def _delayed_tracks_async_enabled(self) -> bool:
        return self.result_policy == "delayed_tracks" and not self.blocking_visualization and self.max_inflight > 1

    def _drain_completed_delayed_jobs(self) -> None:
        while True:
            with self._lock:
                completed = self._completed_delayed_jobs.pop(self._next_delayed_sequence, None)
            if completed is None:
                return
            job, result = completed
            self._attach_delayed_track_result(job, result)
            self._flush_delayed_frames(job.end_frame_index)
            self._next_delayed_sequence += 1

    def _wait_for_delayed_jobs(self) -> None:
        deadline = time.perf_counter() + max(0.0, self.worker_join_timeout_ms / 1000.0)
        while True:
            self._drain_completed_delayed_jobs()
            with self._condition:
                if not self._pending_jobs and self._active_jobs == 0:
                    return
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    return
                self._condition.wait(timeout=min(0.1, remaining))

    def _flush_delayed_frames(self, through_frame_index: int | None = None) -> None:
        while self._delayed_frames:
            if through_frame_index is not None and self._delayed_frames[0].frame_index > through_frame_index:
                break
            buffered = self._delayed_frames.popleft()
            self._dst.enqueue(buffered.frame)

    def _flush_partial_delayed_window(self) -> None:
        if self.result_policy != "delayed_tracks" or self._current_buffer is None or self._current_sample_count <= 0:
            return
        job = self._finish_current_job()
        if job is None:
            return
        if self._delayed_tracks_async_enabled():
            self._enqueue_job(job)
            return
        result = self._execute_job(job)
        self._attach_delayed_track_result(job, result)
        with self._lock:
            self._latest_result = result
        self._flush_delayed_frames(job.end_frame_index)

    def _process_delayed_tracks(self, frame: Any, frame_seconds: float | None) -> None:
        self._delayed_frames.append(
            _BufferedFrame(
                frame=frame,
                seconds=frame_seconds,
                frame_index=self._frame_index,
            )
        )

        job = None
        if self._should_sample(frame_seconds):
            job = self._process_sample(frame, frame_seconds)

        if job is None:
            self._drain_completed_delayed_jobs()
            return

        if self._delayed_tracks_async_enabled():
            self._enqueue_job(job)
            self._drain_completed_delayed_jobs()
            return

        result = self._execute_job(job)
        self._attach_delayed_track_result(job, result)
        with self._lock:
            self._latest_result = result
        self._flush_delayed_frames(job.end_frame_index)

    def _flush_eof_delayed_tracks(self) -> None:
        if self._eof_flushed:
            return
        if self.result_policy == "delayed_tracks":
            self._flush_partial_delayed_window()
            if self._delayed_tracks_async_enabled():
                self._wait_for_delayed_jobs()
                self._drain_completed_delayed_jobs()
            self._flush_delayed_frames()
        self._eof_flushed = True

    def process(self) -> None:
        wait_peek = getattr(self._src, "wait_peek", None)
        if callable(wait_peek):
            next_frame = wait_peek(-1)
            if self._is_eof_frame(next_frame):
                self._flush_eof_delayed_tracks()
                return

        frame = self._src.get()
        if not frame or self._is_eof_frame(frame):
            self._flush_eof_delayed_tracks()
            return

        self._frame_index += 1
        frame_seconds = self._frame_seconds(frame)

        if not self._available:
            self._attach_unavailable(frame)
            self._dst.enqueue(frame)
            return

        if self.result_policy == "delayed_tracks":
            self._process_delayed_tracks(frame, frame_seconds)
            return

        job = None
        if self._should_sample(frame_seconds):
            job = self._process_sample(frame, frame_seconds)

        if job is not None:
            result = self._execute_job(job)
            self._attach_result(frame, result)
            with self._lock:
                self._latest_result = result
        elif self.result_policy == "hold_latest":
            self._attach_cached_result(frame, frame_seconds)
        self._dst.enqueue(frame)

    def doStop(self) -> None:
        self._flush_eof_delayed_tracks()
        self._stop_event.set()
        with self._condition:
            if self._current_buffer is not None:
                self._release_buffer(self._current_buffer)
                self._current_buffer = None
                self._current_sample_count = 0
            while self._pending_jobs:
                self._release_buffer(self._pending_jobs.popleft().buffer)
            self._completed_delayed_jobs.clear()
            self._condition.notify_all()
        join_deadline = time.perf_counter() + max(0.0, self.worker_join_timeout_ms / 1000.0)
        for worker in self._workers:
            worker.join(timeout=max(0.0, join_deadline - time.perf_counter()))
        self._workers.clear()
        self._delayed_frames.clear()
