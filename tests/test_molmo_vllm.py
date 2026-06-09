import json
import importlib.util
import sys
from pathlib import Path

MODULE_PATH = Path(__file__).resolve().parents[1] / "pyplumber" / "molmo_vllm.py"
sys.path.insert(0, str(MODULE_PATH.parents[1]))
SPEC = importlib.util.spec_from_file_location("molmo_vllm_for_tests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
molmo_vllm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = molmo_vllm
SPEC.loader.exec_module(molmo_vllm)
parse_molmo_generated_text = molmo_vllm.parse_molmo_generated_text

RUNNER_MODULE_PATH = Path(__file__).resolve().parents[1] / "pyplumber" / "molmo_vllm_runner.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("molmo_vllm_runner_for_tests", RUNNER_MODULE_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
molmo_vllm_runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = molmo_vllm_runner
RUNNER_SPEC.loader.exec_module(molmo_vllm_runner)


def _parse(text, frame_size=378):
    return parse_molmo_generated_text(
        text,
        frame_size=frame_size,
        prompt_id="test",
        window_start_pts="1",
        window_end_pts="2",
        latency_ms=12.3456,
    )


def test_parse_molmo_generated_text_emits_bbox_and_point_metadata():
    det_md, point_md, raw_md = _parse(
        json.dumps(
            {
                "objects": [
                    {
                        "label": "person",
                        "confidence": 0.82,
                        "bbox": [100, 200, 300, 400],
                        "point": [250, 300],
                    }
                ]
            }
        )
    )

    assert raw_md["parse_status"] == "ok"
    assert raw_md["detection_count"] == 1
    assert raw_md["point_count"] == 1

    assert det_md["schema"] == "yolo_detections_v1"
    assert det_md["coord_space"] == "model"
    assert det_md["model_width"] == 378
    det = det_md["detections"][0]
    assert det["label"] == "person"
    assert det["conf"] == 0.82
    assert det["xyxy"] == [37.8, 75.6, 113.4, 151.2]

    assert point_md["schema"] == "pose_keypoints_v1"
    assert point_md["num_keypoints"] == 1
    assert point_md["poses"][0]["keypoints"] == [94.5, 113.4, 0.82]


def test_parse_molmo_generated_text_accepts_json_fence():
    det_md, point_md, raw_md = _parse(
        """Some text
```json
{"objects":[{"label":"ball","confidence":1,"bbox":[450,450,550,550],"point":[500,500]}]}
```
"""
    )

    assert raw_md["parse_status"] == "ok"
    assert det_md["detections"][0]["label"] == "ball"
    assert point_md["poses"][0]["keypoints"] == [189.0, 189.0, 1.0]


def test_parse_molmo_generated_text_skips_invalid_objects_independently():
    det_md, point_md, raw_md = _parse(
        json.dumps(
            {
                "objects": [
                    {"label": "bad"},
                    {"label": "point", "point": [1000, 0], "confidence": 0.5},
                ]
            }
        )
    )

    assert det_md is None
    assert raw_md["invalid_object_count"] == 1
    assert raw_md["point_count"] == 1
    assert point_md["poses"][0]["keypoints"] == [378.0, 0.0, 0.5]


def test_parse_molmo_generated_text_reports_invalid_json():
    det_md, point_md, raw_md = _parse("not json")

    assert det_md is None
    assert point_md is None
    assert raw_md["parse_status"] == "invalid_json"
    assert raw_md["generated_text"] == "not json"


def test_parse_molmo_generated_text_accepts_native_video_points():
    det_md, point_md, raw_md = _parse(
        '<points alt="black jersey players" coords="0 1 500 400 2 620 450; 1 1 510 410 2 630 460"/>'
    )

    assert det_md is None
    assert raw_md["parse_status"] == "native_points"
    assert raw_md["point_count"] == 2
    assert point_md["schema"] == "pose_keypoints_v1"
    assert point_md["num_keypoints"] == 2
    assert point_md["poses"][0]["keypoints"] == [192.78, 154.98, 1.0, 238.14, 173.88, 1.0]


def test_vllm_runner_helpers_build_prompt_and_metadata():
    prompt = molmo_vllm_runner._format_molmo2_video_prompt("Point to black jerseys.")
    metadata = molmo_vllm_runner._metadata_for_window(sample_count=4, frame_size=378, sample_fps=8)

    assert prompt == "<|video|><|im_start|>user\nPoint to black jerseys.<|im_end|>\n<|im_start|>assistant\n"
    assert metadata["fps"] == 8
    assert metadata["duration"] == 0.5
    assert metadata["frames_indices"] == [0, 1, 2, 3]
    assert metadata["width"] == 378
