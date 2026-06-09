#!/usr/bin/env bash
set -euo pipefail

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDA_PATH="${CUDA_PATH:-/usr/local/cuda-13.0}"
export HF_HOME="${HF_HOME:-/models/huggingface}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/lib:/opt/tensorrt/lib:/usr/local/cuda-13.0/lib64:/usr/local/cuda-13.0/targets/x86_64-linux/lib}"
export PATH="/usr/local/bin:/usr/local/cuda-13.0/bin:${PATH}"
export PYTHONPATH="${PYTHONPATH:-/opt/avplumber}"

python3 <<'PY'
import json
import shutil
import subprocess
from pathlib import Path

print("== ffmpeg ==")
filters = subprocess.check_output(["ffmpeg", "-hide_banner", "-filters"], stderr=subprocess.STDOUT, text=True)
for filter_name in ("scale_cuda", "overlay_many_cuda"):
    if filter_name not in filters:
        raise SystemExit(f"missing FFmpeg CUDA filter: {filter_name}")
    print(f"filter ok: {filter_name}")

nvcc = shutil.which("nvcc")
if not nvcc:
    raise SystemExit("missing CUDA compiler: nvcc")
print(f"nvcc={nvcc}")
curand_h = Path("/usr/local/cuda-13.0/include/curand.h")
if not curand_h.exists():
    raise SystemExit(f"missing CUDA cuRAND header: {curand_h}")
print(f"curand.h={curand_h}")

print("== imports ==")
import _avplumber  # noqa: F401
import pyplumber  # noqa: F401
import torch
import cupy as cp
import vllm
import transformers

print(f"torch={torch.__version__}")
print(f"cupy={cp.__version__}")
print(f"vllm={vllm.__version__}")
print(f"transformers={transformers.__version__}")

if not torch.cuda.is_available():
    raise SystemExit("torch CUDA is unavailable")

device_name = torch.cuda.get_device_name(0)
capability = torch.cuda.get_device_capability(0)
print(f"cuda device={device_name} capability={capability}")

print("== parser ==")
from pyplumber.molmo_vllm import MolmoVllmAsync, _MolmoPreprocessor, parse_molmo_generated_text
from pyplumber.molmo_vllm_runner import _format_molmo2_video_prompt, _metadata_for_window

det_md, point_md, raw_md = parse_molmo_generated_text(
    json.dumps({"objects": [{"label": "ball", "confidence": 0.9, "bbox": [400, 400, 600, 600], "point": [500, 500]}]}),
    frame_size=378,
    prompt_id="smoke",
    window_start_pts="0",
    window_end_pts="1",
)
assert raw_md["parse_status"] == "ok"
assert det_md and det_md["detections"][0]["label"] == "ball"
assert point_md and point_md["num_keypoints"] == 1
print("parser ok")

assert _format_molmo2_video_prompt("Point to the ball.").startswith("<|video|><|im_start|>user")
metadata = _metadata_for_window(sample_count=4, frame_size=378, sample_fps=8)
assert metadata["frames_indices"] == [0, 1, 2, 3]
assert metadata["width"] == 378 and metadata["height"] == 378
print("runner helpers ok")

print("== node init ==")
node = MolmoVllmAsync(
    {
        "src": "video_in",
        "dst": "video_out",
        "backend": "mock",
        "strict_zero_copy": True,
        "window_frames": 1,
        "window_stride": 1,
        "window_queue_size": 0,
        "visualize_ttl_frames": 1,
    }
)
assert node._available
node.doStop()
print("mock strict node ok")

print("== preprocess kernel ==")
kernel_path = Path("/opt/avplumber/src/nodes/neural_net/preprocess/molmo2_preprocess.cu")
pre = _MolmoPreprocessor(
    torch_mod=torch,
    cupy_mod=cp,
    frame_size=378,
    patch_size=14,
    window_frames=1,
    dtype_name="fp16",
    kernel_path=kernel_path,
)
buffer = pre.make_buffer(0)


class SyntheticNV12Frame:
    def __init__(self) -> None:
        self.width = 640
        self.height = 360
        self.y = cp.full((self.height, self.width), 128, dtype=cp.uint8)
        self.uv = cp.full((self.height // 2, self.width), 128, dtype=cp.uint8)
        self.data_ptr = [self.y.data.ptr, self.uv.data.ptr]
        self.linesize = [self.y.strides[0], self.uv.strides[0]]


frame = SyntheticNV12Frame()
pre.preprocess_frame(frame, buffer, 0)
pre.synchronize()

tensor = buffer.tensor
assert tuple(tensor.shape) == (1, 729, 588)
if not bool(torch.isfinite(tensor).all().item()):
    raise SystemExit("preprocess tensor contains non-finite values")

min_value = float(tensor.min().item())
max_value = float(tensor.max().item())
if min_value < -1.1 or max_value > 1.1:
    raise SystemExit(f"preprocess tensor outside normalized range: {min_value}..{max_value}")
print(f"preprocess ok: shape={tuple(tensor.shape)} range={min_value:.4f}..{max_value:.4f}")

torch.cuda.synchronize()
print("molmo docker smoke ok")
PY

if [[ "${AVP_MOLMO_RUN_GRAPH:-0}" == "1" ]]; then
    : "${AVP_INPUT:?AVP_INPUT must point to a mounted media file}"
    mkdir -p "$(dirname "${AVP_OUTPUT:-/artifacts/molmo-smoke.ts}")"
    export AVP_MOLMO_BACKEND="${AVP_MOLMO_BACKEND:-mock}"
    export AVP_MOLMO_STRICT="${AVP_MOLMO_STRICT:-true}"
    export AVP_OUTPUT="${AVP_OUTPUT:-/artifacts/molmo-smoke.ts}"
    export AVP_OUTPUT_FORMAT="${AVP_OUTPUT_FORMAT:-mpegts}"
    timeout_seconds="${AVP_MOLMO_GRAPH_TIMEOUT:-45}"

    set +e
    timeout "${timeout_seconds}" python3 /opt/avplumber/pyplumber/examples/molmo-vllm-node.py
    status=$?
    set -e

    if [[ "${status}" != "0" && "${status}" != "124" ]]; then
        exit "${status}"
    fi
    if [[ ! -s "${AVP_OUTPUT}" ]]; then
        echo "graph output is missing or empty: ${AVP_OUTPUT}" >&2
        exit 1
    fi
    echo "molmo graph smoke ok: ${AVP_OUTPUT}"
fi
