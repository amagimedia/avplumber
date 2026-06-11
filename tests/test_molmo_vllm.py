import json
import importlib.util
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

MODULE_PATH = REPO_ROOT / "pyplumber" / "molmo" / "vllm.py"
SPEC = importlib.util.spec_from_file_location("molmo_vllm_for_tests", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
molmo_vllm = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = molmo_vllm
SPEC.loader.exec_module(molmo_vllm)
parse_molmo_generated_text = molmo_vllm.parse_molmo_generated_text

RUNNER_MODULE_PATH = REPO_ROOT / "pyplumber" / "molmo" / "vllm_runner.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("molmo_vllm_runner_for_tests", RUNNER_MODULE_PATH)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
molmo_vllm_runner = importlib.util.module_from_spec(RUNNER_SPEC)
sys.modules[RUNNER_SPEC.name] = molmo_vllm_runner
RUNNER_SPEC.loader.exec_module(molmo_vllm_runner)

TRANSFORMERS_RUNNER_MODULE_PATH = (
    REPO_ROOT / "pyplumber" / "molmo" / "transformers_runner.py"
)
TRANSFORMERS_RUNNER_SPEC = importlib.util.spec_from_file_location(
    "molmo_transformers_runner_for_tests",
    TRANSFORMERS_RUNNER_MODULE_PATH,
)
assert TRANSFORMERS_RUNNER_SPEC is not None and TRANSFORMERS_RUNNER_SPEC.loader is not None
molmo_transformers_runner = importlib.util.module_from_spec(TRANSFORMERS_RUNNER_SPEC)
sys.modules[TRANSFORMERS_RUNNER_SPEC.name] = molmo_transformers_runner
TRANSFORMERS_RUNNER_SPEC.loader.exec_module(molmo_transformers_runner)

SIDECAR_MODULE_PATH = REPO_ROOT / "pyplumber" / "molmo" / "transformers_sidecar.py"
sys.path.insert(0, str(SIDECAR_MODULE_PATH.parent))
SIDECAR_SPEC = importlib.util.spec_from_file_location("molmo_transformers_sidecar_for_tests", SIDECAR_MODULE_PATH)
assert SIDECAR_SPEC is not None and SIDECAR_SPEC.loader is not None
molmo_transformers_sidecar = importlib.util.module_from_spec(SIDECAR_SPEC)
sys.modules[SIDECAR_SPEC.name] = molmo_transformers_sidecar
SIDECAR_SPEC.loader.exec_module(molmo_transformers_sidecar)


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


def test_transformers_runner_metadata_omits_vllm_sampling_fields():
    metadata = molmo_transformers_runner._metadata_for_transformers_window(
        sample_count=4,
        frame_size=378,
        sample_fps=0.5,
    )

    assert metadata["fps"] == 0.5
    assert metadata["duration"] == 8.0
    assert metadata["total_num_frames"] == 4
    assert metadata["frames_indices"] == [0, 1, 2, 3]
    assert metadata["video_backend"] == "avplumber"
    assert metadata["width"] == 378
    assert "do_sample_frames" not in metadata


def test_transformers_runner_messages_embed_video_payload():
    video = object()
    messages = molmo_transformers_runner._messages_for_transformers(
        "Point to the ball.",
        video,
        "pointing",
    )

    assert messages == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Point to the ball.", "style": "pointing"},
                {"type": "video", "video": video},
            ],
        }
    ]


def test_transformers_runner_generated_suffix_preserves_batch_dimension():
    generated_ids = np.array([[10, 11, 12, 13, 14]])

    suffix = molmo_transformers_runner._generated_suffix(generated_ids, 3)

    assert suffix.tolist() == [[13, 14]]
    assert suffix.ndim == 2


def test_sidecar_request_roundtrip_preserves_video_prompt_and_metadata():
    video = np.zeros((2, 4, 4, 3), dtype=np.uint8)
    metadata = {"fps": 2.0, "frames_indices": [0, 1], "width": 4, "height": 4}

    body = molmo_transformers_sidecar._encode_request(
        prompt="Point to the ball.",
        video=video,
        metadata=metadata,
        max_new_tokens=128,
    )
    prompt, decoded_video, decoded_metadata, max_new_tokens = molmo_transformers_sidecar._decode_request(body)

    assert prompt == "Point to the ball."
    assert np.array_equal(decoded_video, video)
    assert decoded_metadata == metadata
    assert max_new_tokens == 128


def test_sidecar_runner_posts_encoded_window(monkeypatch):
    sent = {}
    runner = molmo_transformers_sidecar.SidecarMolmo2VideoRunner(
        {
            "frame_size": 4,
            "patch_size": 2,
            "sample_fps": 2.0,
            "max_new_tokens": 64,
        }
    )

    video = np.ones((3, 4, 4, 3), dtype=np.uint8)
    monkeypatch.setattr(
        molmo_transformers_sidecar,
        "window_to_video_array",
        lambda job, *, frame_size, patch_size: video,
    )

    def post(body):
        prompt, decoded_video, metadata, max_new_tokens = molmo_transformers_sidecar._decode_request(body)
        sent["prompt"] = prompt
        sent["video"] = decoded_video
        sent["metadata"] = metadata
        sent["max_new_tokens"] = max_new_tokens
        return {"generated_text": "<points coords=\"0.0 1 500 500\"/>"}

    monkeypatch.setattr(runner, "_post", post)

    class Job:
        sample_count = 3
        prompt = "Track the player."

    assert runner.generate(Job()) == '<points coords="0.0 1 500 500"/>'
    assert sent["prompt"] == "Track the player."
    assert np.array_equal(sent["video"], video)
    assert sent["metadata"]["fps"] == 2.0
    assert sent["metadata"]["duration"] == 1.5
    assert sent["metadata"]["frames_indices"] == [0, 1, 2]
    assert sent["max_new_tokens"] == 64


def test_sidecar_runner_uses_configured_urls_round_robin():
    runner = molmo_transformers_sidecar.SidecarMolmo2VideoRunner(
        {
            "sidecar_urls": [
                "http://127.0.0.1:8765/generate",
                "http://127.0.0.1:8766/generate",
            ],
        }
    )

    assert runner._next_url() == "http://127.0.0.1:8765/generate"
    assert runner._next_url() == "http://127.0.0.1:8766/generate"
    assert runner._next_url() == "http://127.0.0.1:8765/generate"


def _molmo_node_for_private_tests():
    node = object.__new__(molmo_vllm.MolmoVllmAsync)
    node.visualize_ttl_frames = 30
    node.visualize_ttl_seconds = None
    node.fallback_input_fps = 30.0
    node.frame_size = 378
    node.result_policy = "hold_latest"
    node.debug_log_every_n = 0
    node.delayed_track_hold_seconds = 0.0
    node.worker_join_timeout_ms = 2000
    node.max_inflight = 1
    node.blocking_visualization = False
    node._delayed_frames = deque()
    node._completed_delayed_jobs = {}
    node._next_delayed_sequence = 1
    node._active_jobs = 0
    node._eof_flushed = False
    node._frame_index = 0
    node.metadata_key_detections = "molmo_detections"
    node.metadata_key_points = "molmo_points"
    node.metadata_key_raw = "molmo_raw"
    node._lock = threading.RLock()
    node._condition = threading.Condition(node._lock)
    return node


def test_cached_result_ttl_frames_use_video_fps_when_timestamps_exist():
    node = _molmo_node_for_private_tests()
    result = molmo_vllm._MolmoResult(
        detection_json=None,
        points_json="{}",
        raw_json="{}",
        end_seconds=10.0,
        expires_seconds=10.0 + node._ttl_seconds(),
        end_frame_index=300,
        expires_frame_index=330,
    )

    assert node._ttl_seconds() == 1.0
    assert node._result_is_fresh(result, 10.5)
    assert node._result_is_fresh(result, 11.0)
    assert not node._result_is_fresh(result, 11.01)


def test_explicit_ttl_seconds_overrides_frame_ttl():
    node = _molmo_node_for_private_tests()
    node.visualize_ttl_seconds = 0.25

    assert node._ttl_seconds() == 0.25


def test_attach_result_writes_configured_metadata_keys():
    node = _molmo_node_for_private_tests()

    class Frame:
        def __init__(self):
            self.metadata = {}

    frame = Frame()
    result = molmo_vllm._MolmoResult(
        detection_json='{"detections":[]}',
        points_json='{"poses":[]}',
        raw_json='{"parse_status":"native_points"}',
        end_seconds=1.0,
        expires_seconds=2.0,
        end_frame_index=1,
        expires_frame_index=2,
    )

    node._attach_result(frame, result)

    assert frame.metadata["molmo_detections"] == '{"detections":[]}'
    assert frame.metadata["molmo_points"] == '{"poses":[]}'
    assert frame.metadata["molmo_raw"] == '{"parse_status":"native_points"}'


def test_native_tracks_preserve_timed_points():
    points, invalid = molmo_vllm._extract_native_molmo_timed_points(
        '<tracks coords="0.0 1 205 564;0.5 1 198 615;1.0 1 241 629">ball handler</tracks>',
        frame_size=378,
    )

    assert invalid == 0
    assert [point.time_seconds for point in points] == [0.0, 0.5, 1.0]
    assert [point.object_id for point in points] == ["molmo:1", "molmo:1", "molmo:1"]
    assert points[0].x == 77.49
    assert points[0].y == 213.192


def test_delayed_tracks_attach_only_to_matching_buffered_frames():
    node = _molmo_node_for_private_tests()

    class Frame:
        def __init__(self):
            self.metadata = {}

    frames = [Frame(), Frame(), Frame()]
    node._delayed_frames.extend(
        [
            molmo_vllm._BufferedFrame(frame=frames[0], seconds=10.0, frame_index=1),
            molmo_vllm._BufferedFrame(frame=frames[1], seconds=10.5, frame_index=16),
            molmo_vllm._BufferedFrame(frame=frames[2], seconds=11.0, frame_index=31),
        ]
    )
    job = molmo_vllm._WindowJob(
        sequence=1,
        buffer=None,
        prompt="track",
        prompt_id="track",
        sample_count=2,
        start_pts="10",
        end_pts="11",
        start_seconds=10.0,
        end_seconds=11.0,
        start_frame_index=1,
        end_frame_index=31,
        video_inputs={},
        enqueue_time=0.0,
    )
    result = molmo_vllm._MolmoResult(
        detection_json=None,
        points_json=None,
        raw_json='{"parse_status":"native_points"}',
        end_seconds=11.0,
        expires_seconds=11.0,
        end_frame_index=31,
        expires_frame_index=31,
        timed_points=(
            molmo_vllm._TimedPoint(time_seconds=0.0, object_id="track:1", x=100.0, y=120.0),
            molmo_vllm._TimedPoint(time_seconds=1.0, object_id="track:1", x=130.0, y=150.0),
        ),
    )

    node._attach_delayed_track_result(job, result)

    assert "molmo_points" in frames[0].metadata
    assert "molmo_points" not in frames[1].metadata
    assert "molmo_points" in frames[2].metadata
    first_points = json.loads(frames[0].metadata["molmo_points"])
    last_points = json.loads(frames[2].metadata["molmo_points"])
    assert first_points["poses"][0]["keypoints"] == [100.0, 120.0, 1.0]
    assert last_points["poses"][0]["keypoints"] == [130.0, 150.0, 1.0]


def test_delayed_tracks_hold_seconds_persists_nearby_frames():
    node = _molmo_node_for_private_tests()
    node.delayed_track_hold_seconds = 0.2

    class Frame:
        def __init__(self):
            self.metadata = {}

    frames = [Frame(), Frame(), Frame()]
    node._delayed_frames.extend(
        [
            molmo_vllm._BufferedFrame(frame=frames[0], seconds=10.00, frame_index=1),
            molmo_vllm._BufferedFrame(frame=frames[1], seconds=10.08, frame_index=3),
            molmo_vllm._BufferedFrame(frame=frames[2], seconds=10.30, frame_index=9),
        ]
    )
    job = molmo_vllm._WindowJob(
        sequence=1,
        buffer=None,
        prompt="track",
        prompt_id="track",
        sample_count=1,
        start_pts="10",
        end_pts="10",
        start_seconds=10.0,
        end_seconds=10.0,
        start_frame_index=1,
        end_frame_index=1,
        video_inputs={},
        enqueue_time=0.0,
    )
    result = molmo_vllm._MolmoResult(
        detection_json=None,
        points_json=None,
        raw_json='{"parse_status":"native_points"}',
        end_seconds=10.0,
        expires_seconds=10.0,
        end_frame_index=1,
        expires_frame_index=1,
        timed_points=(
            molmo_vllm._TimedPoint(time_seconds=0.0, object_id="track:1", x=100.0, y=120.0),
        ),
    )

    node._attach_delayed_track_result(job, result)

    assert "molmo_points" in frames[0].metadata
    assert "molmo_points" in frames[1].metadata
    assert "molmo_points" not in frames[2].metadata


def test_async_delayed_tracks_drain_completed_jobs_in_sequence_order():
    node = _molmo_node_for_private_tests()
    node.result_policy = "delayed_tracks"
    node.max_inflight = 2

    class Frame:
        def __init__(self, index):
            self.index = index
            self.metadata = {}

    class Dst:
        def __init__(self):
            self.frames = []

        def enqueue(self, frame):
            self.frames.append(frame)

    frames = [Frame(1), Frame(2)]
    node._dst = Dst()
    node._delayed_frames.extend(
        [
            molmo_vllm._BufferedFrame(frame=frames[0], seconds=10.0, frame_index=1),
            molmo_vllm._BufferedFrame(frame=frames[1], seconds=11.0, frame_index=2),
        ]
    )

    def job(sequence, frame_index):
        return molmo_vllm._WindowJob(
            sequence=sequence,
            buffer=None,
            prompt="track",
            prompt_id="track",
            sample_count=1,
            start_pts=str(frame_index),
            end_pts=str(frame_index),
            start_seconds=float(9 + frame_index),
            end_seconds=float(9 + frame_index),
            start_frame_index=frame_index,
            end_frame_index=frame_index,
            video_inputs={},
            enqueue_time=0.0,
        )

    def result(frame_index, x):
        return molmo_vllm._MolmoResult(
            detection_json=None,
            points_json=None,
            raw_json='{"parse_status":"native_points"}',
            end_seconds=float(9 + frame_index),
            expires_seconds=float(9 + frame_index),
            end_frame_index=frame_index,
            expires_frame_index=frame_index,
            timed_points=(
                molmo_vllm._TimedPoint(time_seconds=0.0, object_id="track:1", x=x, y=120.0),
            ),
        )

    node._completed_delayed_jobs[2] = (job(2, 2), result(2, 200.0))
    node._drain_completed_delayed_jobs()

    assert node._dst.frames == []
    assert node._next_delayed_sequence == 1

    node._completed_delayed_jobs[1] = (job(1, 1), result(1, 100.0))
    node._drain_completed_delayed_jobs()

    assert node._dst.frames == frames
    assert "molmo_points" in frames[0].metadata
    assert "molmo_points" in frames[1].metadata
    assert node._next_delayed_sequence == 3


def test_delayed_tracks_completes_window_synchronously():
    node = _molmo_node_for_private_tests()
    node.result_policy = "delayed_tracks"
    node.blocking_visualization = False
    job = object()
    node._finish_current_job = lambda: job
    node._enqueue_job = lambda _job: (_ for _ in ()).throw(AssertionError("delayed tracks must not queue jobs"))

    class Preprocessor:
        def preprocess_frame(self, frame, buffer, sample_index):
            pass

    node._available = True
    node._preprocessor = Preprocessor()
    node._current_buffer = object()
    node._current_sample_count = 1
    node.window_frames = 2

    assert node._process_sample(object(), 1.0) is job


def test_delayed_tracks_stop_flushes_buffered_tail_frames():
    node = _molmo_node_for_private_tests()
    node.result_policy = "delayed_tracks"
    node._stop_event = threading.Event()
    node._condition = threading.Condition(threading.RLock())
    node._current_buffer = None
    node._current_sample_count = 0
    node._pending_jobs = deque()
    node._workers = []

    class Frame:
        def __init__(self, index):
            self.index = index
            self.metadata = {}

    class Dst:
        def __init__(self):
            self.frames = []

        def enqueue(self, frame):
            self.frames.append(frame)

    frames = [Frame(1), Frame(2), Frame(3)]
    node._dst = Dst()
    node._delayed_frames.extend(
        molmo_vllm._BufferedFrame(frame=frame, seconds=float(frame.index), frame_index=frame.index)
        for frame in frames
    )

    node.doStop()

    assert node._dst.frames == frames
    assert not node._delayed_frames
