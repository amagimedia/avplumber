# `UltralyticsByteTrackNode`

`UltralyticsByteTrackNode` is a pyplumber Python node that runs Ultralytics
YOLO tracking on CPU-readable `VideoFrame` objects and writes AVPlumber
YOLO-style detection metadata onto each frame.

It performs both model inference and tracking in one Python node by calling:

```python
model.track(source=image, persist=True, tracker="bytetrack.yaml")
```

Use this node for prototyping, VOD tools, and graphs that need to reuse an
existing `.pt` Ultralytics model without exporting it to TensorRT. For
high-throughput CUDA/TensorRT production graphs, prefer the native split:
`cuda_infer_yolo` for inference followed by `player_tracker` for C++ tracking.

## Data Flow

```text
VideoFrame -> UltralyticsByteTrackNode -> same VideoFrame with metadata
```

The input frame must be CPU-readable. Supported formats are:

- `rgb24`
- `bgr24`
- `gray8`
- `yuv420p` / `yuvj420p`
- `nv12`

If the upstream graph produces another format, insert `filter_video` or
`rescale_video` first. CUDA frames need to be downloaded to CPU memory before
this node; keep CUDA frames on the native `cuda_infer_yolo` path when
throughput matters.

## Metadata

The node writes JSON under `metadata_key` using AVPlumber's standard YOLO-style
shape:

```json
{
  "schema": "yolo_detections_v1",
  "source": "ultralytics_bytetrack",
  "coord_space": "model",
  "model_width": 1280,
  "model_height": 720,
  "frame": 42,
  "inference_ms": 18.4,
  "tracker": "bytetrack.yaml",
  "detections": [
    {
      "xyxy": [125.0, 64.0, 241.0, 390.0],
      "conf": 0.91,
      "cls": 0,
      "label": "person",
      "model_index": 0,
      "track_id": 7
    }
  ]
}
```

Downstream nodes such as `draw_bbox`, `draw_bbox_labels`,
`smooth_crop_viewport`, and custom Python nodes can consume this metadata.

## Parameters

Required:

- `src` - input video edge.
- `dst` - output video edge.
- `weights` / `model` - Ultralytics model path or model name.

Optional:

- `metadata_key` - output metadata key. Default: `ultralytics_tracks`.
- `device` - Ultralytics device string or index, such as `0`, `cuda:0`, or
  `cpu`. Default: Ultralytics default.
- `tracker` - Ultralytics tracker config. Default: `bytetrack.yaml`.
- `conf` - confidence threshold. Default: `0.25`.
- `iou` - optional IoU threshold.
- `imgsz` - optional inference image size.
- `max_det` - optional maximum detections.
- `half` - optional Ultralytics half-precision flag. Ultralytics still owns the
  tensor conversion and preprocessing.
- `classes` / `target_classes` - optional class IDs or class names to pass to
  Ultralytics.
- `target_labels` / `allowed_labels` - alias for class-name filtering.
- `include_untracked` - include detections without a tracker ID. Default: true.
- `emit_empty` - write empty metadata on failures when `error_policy` is
  `passthrough`. Default: true.
- `error_policy` - `raise` or `passthrough`. Default: `raise`.
- `count_metadata_key` - optional scalar metadata key for detection count.
- `status_metadata_key` - optional scalar metadata key for status.
- `debug_log_every_n` - print periodic node statistics. Default: `0`.

## Example

```python
from pyplumber.ultralytics_tracker import UltralyticsByteTrackNode

avp.addNode(UltralyticsByteTrackNode({
    "name": "Track_Players",
    "group": "tracking",
    "src": "v_yuv420p",
    "dst": "v_tracked",
    "weights": "/models/player_referee_ball_yolo_best.pt",
    "metadata_key": "yolo_players",
    "target_labels": ["player"],
    "tracker": "bytetrack.yaml",
    "device": 0,
}))
```
