"""Strict direct-tensor Transformers runner for Qwen3-VL video windows."""

from __future__ import annotations

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


def _format_qwen3_video_prompt_fallback(prompt: str) -> str:
    return (
        "<|im_start|>user\n"
        "<|vision_start|><|video_pad|><|vision_end|>"
        f"{prompt}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _qwen3_timestamps(*, grid_t: int, sample_fps: float, temporal_patch_size: int) -> list[float]:
    fps = max(float(sample_fps), 0.001)
    return [
        ((frame_index * temporal_patch_size) + ((temporal_patch_size - 1) / 2.0)) / fps
        for frame_index in range(int(grid_t))
    ]


def expand_qwen3_video_placeholders(
    text: str,
    *,
    grid_t: int,
    grid_h: int,
    grid_w: int,
    sample_fps: float,
    temporal_patch_size: int,
    merge_size: int,
    vision_start_token: str = "<|vision_start|>",
    vision_end_token: str = "<|vision_end|>",
    video_token: str = "<|video_pad|>",
) -> str:
    """Expand one Qwen3 video token into timestamped per-frame placeholders."""

    merge_length = int(merge_size) * int(merge_size)
    frame_seqlen = int(grid_h) * int(grid_w) // merge_length
    if frame_seqlen <= 0:
        raise ValueError("Qwen3 video frame token length must be positive")

    placeholder = ""
    for timestamp in _qwen3_timestamps(
        grid_t=int(grid_t),
        sample_fps=float(sample_fps),
        temporal_patch_size=int(temporal_patch_size),
    ):
        placeholder += f"<{timestamp:.1f} seconds>"
        placeholder += vision_start_token + (video_token * frame_seqlen) + vision_end_token

    wrapped_token = vision_start_token + video_token + vision_end_token
    if wrapped_token in text:
        return text.replace(wrapped_token, placeholder, 1)
    if video_token in text:
        return text.replace(video_token, placeholder, 1)
    return placeholder + text


class TransformersQwen3VlDirectRunner:
    """Generate Qwen3-VL output from prebuilt CUDA ``pixel_values_videos`` tensors.

    This runner intentionally never calls a video processor. It accepts only
    tensors produced upstream by the AVP CUDA preprocessor and fails if they are
    not CUDA tensors.
    """

    def __init__(self, config: dict[str, Any]):
        self.model_id = str(config.get("model_id", "Qwen/Qwen3-VL-8B-Instruct"))
        self.max_new_tokens = _get_int(config, "max_new_tokens", 512)
        self.temperature = _get_float(config, "temperature", 0.0)
        self.model_dtype = str(config.get("model_dtype", "auto"))
        self.attn_implementation = str(config.get("attn_implementation", "sdpa"))
        self.device_map = str(config.get("device_map", "auto"))
        self.trust_remote_code = _get_bool(config, "trust_remote_code", True)
        self.sample_fps = _get_float(config, "sample_fps", 2.0)
        self.temporal_patch_size = _get_int(config, "temporal_patch_size", 2)
        self.merge_size = _get_int(config, "merge_size", 2)

        self._model: Any = None
        self._tokenizer: Any = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return

        from transformers import AutoTokenizer

        try:
            from transformers import Qwen3VLForConditionalGeneration
        except ImportError:
            Qwen3VLForConditionalGeneration = None

        if Qwen3VLForConditionalGeneration is None:
            from transformers import AutoModelForImageTextToText

            model_cls = AutoModelForImageTextToText
        else:
            model_cls = Qwen3VLForConditionalGeneration

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            trust_remote_code=self.trust_remote_code,
        )
        model_kwargs: dict[str, Any] = {
            "dtype": self.model_dtype,
            "attn_implementation": self.attn_implementation,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.device_map.lower() not in ("", "none", "null"):
            model_kwargs["device_map"] = self.device_map
        self._model = model_cls.from_pretrained(self.model_id, **model_kwargs)
        if "device_map" not in model_kwargs:
            import torch

            if torch.cuda.is_available():
                self._model.to("cuda")
        self._model.eval()

    def _model_device(self) -> Any:
        if hasattr(self._model, "device"):
            return self._model.device
        return next(self._model.parameters()).device

    def _model_dtype(self) -> Any:
        dtype = getattr(self._model, "dtype", None)
        if dtype is not None:
            return dtype
        return next(self._model.parameters()).dtype

    def _chat_text(self, prompt: str) -> str:
        assert self._tokenizer is not None
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": "avp://window"},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        try:
            text = self._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            text = _format_qwen3_video_prompt_fallback(prompt)
        if "<|video_pad|>" not in text:
            text = _format_qwen3_video_prompt_fallback(prompt)
        return str(text)

    def _prepare_text_inputs(self, prompt: str, video_grid_thw: Any) -> dict[str, Any]:
        assert self._tokenizer is not None
        grid = video_grid_thw[0].detach().cpu().tolist()
        text = expand_qwen3_video_placeholders(
            self._chat_text(prompt),
            grid_t=int(grid[0]),
            grid_h=int(grid[1]),
            grid_w=int(grid[2]),
            sample_fps=self.sample_fps,
            temporal_patch_size=self.temporal_patch_size,
            merge_size=self.merge_size,
        )
        return dict(self._tokenizer([text], return_tensors="pt"))

    def generate(self, job: Any) -> str:
        self._ensure_model()
        assert self._model is not None

        import torch

        video_inputs = dict(job.video_inputs)
        pixel_values = video_inputs.get("pixel_values_videos")
        video_grid_thw = video_inputs.get("video_grid_thw")
        if pixel_values is None or video_grid_thw is None:
            raise ValueError("Qwen3 direct runner requires pixel_values_videos and video_grid_thw")
        if not getattr(pixel_values, "is_cuda", False):
            raise RuntimeError("strict_no_cpu_video_copy requires CUDA pixel_values_videos")

        device = self._model_device()
        model_dtype = self._model_dtype()
        text_inputs = self._prepare_text_inputs(str(job.prompt), video_grid_thw)
        text_inputs = {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in text_inputs.items()
        }
        text_inputs["pixel_values_videos"] = pixel_values.to(device=device, dtype=model_dtype, non_blocking=True)
        text_inputs["video_grid_thw"] = video_grid_thw.to(device=device, dtype=torch.long, non_blocking=True)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        generate_kwargs: dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "do_sample": self.temperature > 0,
        }
        if self.temperature > 0:
            generate_kwargs["temperature"] = self.temperature
        with torch.inference_mode():
            generated_ids = self._model.generate(**text_inputs, **generate_kwargs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        prompt_tokens = text_inputs["input_ids"].size(1)
        generated_tokens = generated_ids[:, prompt_tokens:]
        return str(
            self._tokenizer.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        )


def create_runner(config: dict[str, Any]) -> TransformersQwen3VlDirectRunner:
    return TransformersQwen3VlDirectRunner(config)
