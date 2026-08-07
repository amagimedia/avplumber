"""Transformers runner for Molmo2 video windows."""

from __future__ import annotations

import os
from typing import Any


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


def _metadata_for_transformers_window(*, sample_count: int, frame_size: int, sample_fps: float) -> dict[str, Any]:
    """Build metadata accepted by transformers.video_utils.VideoMetadata."""

    fps = max(float(sample_fps), 0.001)
    return {
        "fps": fps,
        "duration": float(sample_count) / fps,
        "total_num_frames": int(sample_count),
        "frames_indices": list(range(int(sample_count))),
        "video_backend": "avplumber",
        "height": int(frame_size),
        "width": int(frame_size),
    }


def _messages_for_transformers(prompt: str, video: Any, prompt_style: str = "") -> list[dict[str, Any]]:
    text_content: dict[str, Any] = {"type": "text", "text": prompt}
    if prompt_style:
        text_content["style"] = prompt_style

    return [
        {
            "role": "user",
            "content": [
                text_content,
                {"type": "video", "video": video},
            ],
        }
    ]


def _generated_suffix(generated_ids: Any, prompt_tokens: int) -> Any:
    return generated_ids[:, prompt_tokens:]


def window_to_video_array(job: Any, *, frame_size: int, patch_size: int) -> Any:
    tensor = job.buffer.tensor[: job.sample_count]
    if not tensor.__class__.__module__.startswith("torch"):
        raise TypeError("Molmo Transformers runner expects a torch CUDA tensor buffer")

    import torch as torch_mod

    patch_grid = frame_size // patch_size
    video = ((tensor.float() + 1.0) * 127.5).clamp(0, 255).to(dtype=torch_mod.uint8)
    video = video.reshape(
        int(job.sample_count),
        patch_grid,
        patch_grid,
        patch_size,
        patch_size,
        3,
    )
    video = video.permute(0, 1, 3, 2, 4, 5).contiguous()
    video = video.reshape(int(job.sample_count), frame_size, frame_size, 3)
    return video.cpu().numpy()


class TransformersMolmo2VideoRunner:
    """Generate Molmo2 outputs through the official Transformers processor."""

    def __init__(self, config: dict[str, Any]):
        self.model_id = str(config.get("model_id", "allenai/Molmo2-8B"))
        self.frame_size = _get_int(config, "frame_size", 378)
        self.patch_size = _get_int(config, "patch_size", 14)
        self.sample_fps = _get_float(config, "sample_fps", 0.5)
        self.max_new_tokens = _get_int(config, "max_new_tokens", 512)
        self.model_dtype = str(config.get("model_dtype", "auto"))
        self.prompt_style = str(config.get("prompt_style", ""))

        self._processor: Any = None
        self._model: Any = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        from transformers import AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            trust_remote_code=True,
            dtype=self.model_dtype,
            device_map="auto",
        )
        self._model.eval()

    def _move_inputs_to_model_device(self, inputs: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.to(self._model.device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }

    def generate_from_video(self, *, prompt: str, video: Any, metadata: dict[str, Any]) -> str:
        self._ensure_model()
        assert self._processor is not None
        assert self._model is not None

        inputs = self._processor.apply_chat_template(
            _messages_for_transformers(prompt, video, self.prompt_style),
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
            video_metadata=[metadata],
            do_sample_frames=False,
        )
        inputs = self._move_inputs_to_model_device(dict(inputs))

        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        with torch.inference_mode():
            generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        prompt_tokens = inputs["input_ids"].size(1)
        generated_tokens = _generated_suffix(generated_ids, prompt_tokens)
        return str(
            self._processor.post_process_image_text_to_text(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )

    def generate(self, job: Any) -> str:
        video = window_to_video_array(job, frame_size=self.frame_size, patch_size=self.patch_size)
        metadata = _metadata_for_transformers_window(
            sample_count=int(job.sample_count),
            frame_size=self.frame_size,
            sample_fps=self.sample_fps,
        )
        return self.generate_from_video(prompt=str(job.prompt), video=video, metadata=metadata)


def create_runner(config: dict[str, Any]) -> TransformersMolmo2VideoRunner:
    return TransformersMolmo2VideoRunner(config)
