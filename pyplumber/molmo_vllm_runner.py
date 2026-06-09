"""Local vLLM runner for AVPlumber Molmo2 video windows."""

from __future__ import annotations

import os
from typing import Any


def _get_bool(config: dict[str, Any], name: str, default: bool) -> bool:
    value = config.get(name, default)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).lower() in ("1", "true", "yes", "on")


def _get_float(config: dict[str, Any], name: str, default: float) -> float:
    value = config.get(name, default)
    if value is None:
        return default
    return float(value)


def _get_int(config: dict[str, Any], name: str, default: int) -> int:
    value = config.get(name, default)
    if value is None:
        return default
    return int(value)


def _format_molmo2_video_prompt(prompt: str) -> str:
    return f"<|video|><|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"


def _metadata_for_window(*, sample_count: int, frame_size: int, sample_fps: float) -> dict[str, Any]:
    fps = max(float(sample_fps), 0.001)
    return {
        "fps": fps,
        "duration": float(sample_count) / fps,
        "total_num_frames": int(sample_count),
        "frames_indices": list(range(int(sample_count))),
        "video_backend": "avplumber",
        "do_sample_frames": False,
        "height": int(frame_size),
        "width": int(frame_size),
    }


class VllmMolmo2VideoRunner:
    """Generate Molmo2 outputs for sampled AVP video windows.

    The current public vLLM Molmo2 processor accepts video arrays plus metadata.
    AVP's CUDA preprocessor already stores normalized patch-packed RGB tensors,
    so this runner reconstructs a CPU RGB video array for vLLM. That copy is the
    compatibility path; a future runner can use ``job.buffer.tensor`` and
    ``job.video_inputs`` directly when vLLM exposes a stable tensor input API.
    """

    def __init__(self, config: dict[str, Any]):
        self.model_id = str(config.get("model_id", "allenai/Molmo2-VideoPoint-4B"))
        self.frame_size = _get_int(config, "frame_size", 378)
        self.patch_size = _get_int(config, "patch_size", 14)
        self.sample_fps = _get_float(config, "sample_fps", 8.0)
        self.max_new_tokens = _get_int(config, "max_new_tokens", 256)
        self.temperature = _get_float(config, "temperature", 0.0)
        self.gpu_memory_utilization = _get_float(config, "gpu_memory_utilization", 0.85)
        self.max_model_len = _get_int(config, "max_model_len", 4096)
        self.max_num_seqs = _get_int(config, "max_num_seqs", 1)
        self.max_num_batched_tokens = _get_int(config, "max_num_batched_tokens", 8192)
        self.tensor_parallel_size = _get_int(config, "tensor_parallel_size", 1)
        self.cpu_offload_gb = _get_float(config, "cpu_offload_gb", 0.0)
        self.mm_processor_cache_gb = _get_float(config, "mm_processor_cache_gb", 0.0)
        self.model_dtype = str(config.get("model_dtype", "float16"))
        self.enforce_eager = _get_bool(config, "enforce_eager", True)
        self.seed = _get_int(config, "seed", 0)

        self._llm: Any = None
        self._sampling_params: Any = None

    def _ensure_llm(self) -> None:
        if self._llm is not None:
            return

        os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        from vllm import LLM, SamplingParams

        llm_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "trust_remote_code": True,
            "dtype": self.model_dtype,
            "limit_mm_per_prompt": {"video": 1},
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "tensor_parallel_size": self.tensor_parallel_size,
            "cpu_offload_gb": self.cpu_offload_gb,
            "mm_processor_cache_gb": self.mm_processor_cache_gb,
            "enforce_eager": self.enforce_eager,
            "seed": self.seed,
        }
        self._llm = LLM(**llm_kwargs)
        self._sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )

    def _window_to_video_array(self, job: Any) -> Any:
        tensor = job.buffer.tensor[: job.sample_count]
        if not tensor.__class__.__module__.startswith("torch"):
            raise TypeError("Molmo vLLM runner expects a torch CUDA tensor buffer")

        import torch as torch_mod

        patch_grid = self.frame_size // self.patch_size
        video = ((tensor.float() + 1.0) * 127.5).clamp(0, 255).to(dtype=torch_mod.uint8)
        video = video.reshape(
            int(job.sample_count),
            patch_grid,
            patch_grid,
            self.patch_size,
            self.patch_size,
            3,
        )
        video = video.permute(0, 1, 3, 2, 4, 5).contiguous()
        video = video.reshape(int(job.sample_count), self.frame_size, self.frame_size, 3)
        return video.cpu().numpy()

    def generate(self, job: Any) -> str:
        self._ensure_llm()
        assert self._llm is not None
        assert self._sampling_params is not None

        video = self._window_to_video_array(job)
        metadata = _metadata_for_window(
            sample_count=int(job.sample_count),
            frame_size=self.frame_size,
            sample_fps=self.sample_fps,
        )
        request = {
            "prompt": _format_molmo2_video_prompt(str(job.prompt)),
            "multi_modal_data": {"video": [(video, metadata)]},
            "multi_modal_uuids": {"video": f"avp-molmo-window-{int(job.sequence)}"},
        }
        outputs = self._llm.generate(request, sampling_params=self._sampling_params)
        if not outputs or not outputs[0].outputs:
            return ""
        return str(outputs[0].outputs[0].text)
