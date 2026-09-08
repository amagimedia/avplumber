import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.nodes.neural_net.vlm.molmo import node as molmo_vllm
from src.nodes.neural_net.vlm.qwen import node as qwen_vl
from src.nodes.neural_net.vlm.qwen import transformers_runner as qwen_runner


def _parse(text, width=800, height=448):
    return qwen_vl.parse_qwen_generated_text(
        text,
        model_width=width,
        model_height=height,
        prompt_id="test",
        window_start_pts="1",
        window_end_pts="2",
        latency_ms=12.3456,
    )


def test_parse_qwen_generated_text_accepts_qwen3_list_format():
    det_md, point_md, raw_md = _parse(
        json.dumps(
            [
                {
                    "label": "ball handler",
                    "confidence": 0.8,
                    "bbox_2d": [100, 200, 300, 400],
                    "point_2d": [250, 300],
                }
            ]
        )
    )

    assert raw_md["parse_status"] == "ok"
    assert raw_md["detection_count"] == 1
    assert raw_md["point_count"] == 1
    assert det_md["schema"] == "yolo_detections_v1"
    assert det_md["model_width"] == 800
    assert det_md["model_height"] == 448
    assert det_md["detections"][0]["xyxy"] == [80.0, 89.6, 240.0, 179.2]
    assert point_md["poses"][0]["keypoints"] == [200.0, 134.4, 0.8]


def test_parse_qwen_generated_text_accepts_json_fence_and_objects_key():
    det_md, point_md, raw_md = _parse(
        """```json
{"objects":[{"label":"shot clock","score":0.7,"bbox":[0,0,1000,1000]}]}
```"""
    )

    assert point_md is None
    assert raw_md["parse_status"] == "ok"
    assert det_md["detections"][0]["label"] == "shot clock"
    assert det_md["detections"][0]["conf"] == 0.7
    assert det_md["detections"][0]["xyxy"] == [0.0, 0.0, 800.0, 448.0]


def test_parse_qwen_generated_text_reports_invalid_json():
    det_md, point_md, raw_md = _parse("not json")

    assert det_md is None
    assert point_md is None
    assert raw_md["parse_status"] == "invalid_json"
    assert raw_md["generated_text"] == "not json"


def test_expand_qwen3_video_placeholders_repeats_per_merged_patch_grid():
    text = "<|im_start|>user\n<|vision_start|><|video_pad|><|vision_end|>Find ball<|im_end|>\n"

    expanded = qwen_runner.expand_qwen3_video_placeholders(
        text,
        grid_t=2,
        grid_h=4,
        grid_w=6,
        sample_fps=2.0,
        temporal_patch_size=2,
        merge_size=2,
    )

    assert "<0.2 seconds>" in expanded
    assert "<1.2 seconds>" in expanded
    assert expanded.count("<|video_pad|>") == 12


def test_format_qwen3_video_prompt_fallback_contains_video_token_before_prompt():
    prompt = qwen_runner._format_qwen3_video_prompt_fallback("Find the shooter.")

    assert "<|vision_start|><|video_pad|><|vision_end|>" in prompt
    assert prompt.index("<|video_pad|>") < prompt.index("Find the shooter.")


def test_transformers_runner_omits_empty_device_map_private(monkeypatch):
    captured = {}

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            captured["model_id"] = model_id
            captured["kwargs"] = kwargs
            return cls()

        def to(self, device):
            captured["to"] = device
            return self

        def eval(self):
            captured["eval"] = True

    class FakeTorch:
        class cuda:
            @staticmethod
            def is_available():
                return True

    def fake_import(name):
        if name == "transformers":
            return type(
                "FakeTransformers",
                (),
                {
                    "AutoTokenizer": FakeTokenizer,
                    "Qwen3VLForConditionalGeneration": FakeModel,
                },
            )
        if name == "torch":
            return FakeTorch
        raise ImportError(name)

    monkeypatch.setattr("builtins.__import__", lambda name, *args, **kwargs: fake_import(name))

    runner = qwen_runner.TransformersQwen3VlDirectRunner(
        {
            "model_id": "Qwen/Qwen3-VL-8B-Instruct",
            "device_map": "",
        }
    )
    runner._ensure_model()

    assert "device_map" not in captured["kwargs"]
    assert captured["to"] == "cuda"
    assert captured["eval"] is True


def test_pack_qwen3_video_patches_shape_and_grid():
    torch = pytest.importorskip("torch")
    frames = torch.arange(3 * 3 * 4 * 4, dtype=torch.float32).reshape(3, 3, 4, 4)

    pixel_values, video_grid_thw = qwen_vl.pack_qwen3_video_patches(
        frames,
        sample_count=3,
        patch_size=2,
        temporal_patch_size=2,
        merge_size=1,
    )

    assert video_grid_thw.tolist() == [[2, 2, 2]]
    assert pixel_values.shape == (8, 24)
    assert pixel_values.device == frames.device


def _qwen_node_for_config_tests(**overrides):
    node = object.__new__(qwen_vl.QwenVlAsync)
    node._args = {
        "qwen_target_height": 448,
        "qwen_target_width": 800,
        "temporal_patch_size": 2,
        "merge_size": 2,
        "coordinate_scale": 1000,
    }
    node.backend = "transformers_direct"
    node.patch_size = 16
    node.tensor_dtype = "fp16"
    node.sample_fps = 2.0
    node.window_frames = 8
    node.window_stride = 8
    node.window_queue_size = 1
    node.max_inflight = 1
    node.visualize_ttl_frames = 8
    node.visualize_ttl_seconds = None
    node.fallback_input_fps = 30.0
    node.result_policy = "delayed_tracks"
    for name, value in overrides.items():
        if name in node._args:
            node._args[name] = value
        setattr(node, name, value)
    return node


def test_qwen_preprocessor_config_rejects_delayed_tracks_private():
    node = _qwen_node_for_config_tests()

    with pytest.raises(ValueError, match="result_policy"):
        node._validate_config()


def test_qwen_preprocessor_config_rejects_temporal_merge_mismatch_private():
    node = _qwen_node_for_config_tests(merge_size=1, result_policy="hold_latest")

    with pytest.raises(ValueError, match="temporal_patch_size"):
        node._validate_config()
