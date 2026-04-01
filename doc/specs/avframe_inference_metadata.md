# AVFrame Inference Metadata Specification

This document describes the metadata and side data attached to an `AVFrame` as it flows through the YOLO inference pipeline and into downstream nodes like `basketball_analysis`, `smooth_crop_viewport`, and `reframer`.

All JSON metadata is stored in the AVFrame's `metadata` dictionary (`AVDictionary`) as string values, keyed by configurable names. Binary data (segmentation masks) is attached as `AVFrameSideData`.

## Detection Metadata

**Default key:** `yolo_detections_v1`

Produced by `cuda_infer_yolo` when running detection-class models. Multiple models can contribute detections to a single frame; each detection records which model produced it.

```json
{
  "coord_space": "model",
  "model_width": 640,
  "model_height": 640,
  "models": [
    {
      "model_index": 0,
      "engine": "/models/yolov8n-basketball.engine",
      "engine_name": "yolov8n-basketball.engine"
    },
    {
      "model_index": 1,
      "engine": "/models/yolov8n-hoop.engine",
      "engine_name": "yolov8n-hoop.engine"
    },
    {
      "model_index": 2,
      "engine": "/models/yolov8n-player.engine",
      "engine_name": "yolov8n-player.engine"
    }
  ],
  "detections": [
    {
      "cls": 0,
      "conf": 0.91,
      "xyxy": [245.3, 180.7, 278.1, 213.5],
      "model_index": 0,
      "engine_name": "yolov8n-basketball.engine",
      "label": "ball"
    },
    {
      "cls": 1,
      "conf": 0.87,
      "xyxy": [300.0, 50.2, 360.5, 110.8],
      "model_index": 1,
      "engine_name": "yolov8n-hoop.engine",
      "label": "hoop"
    },
    {
      "cls": 0,
      "conf": 0.95,
      "xyxy": [100.2, 150.0, 180.6, 400.3],
      "model_index": 2,
      "engine_name": "yolov8n-player.engine",
      "label": "player"
    },
    {
      "cls": 0,
      "conf": 0.93,
      "xyxy": [350.1, 160.5, 430.0, 410.2],
      "model_index": 2,
      "engine_name": "yolov8n-player.engine",
      "label": "player"
    },
    {
      "cls": 0,
      "conf": 0.72,
      "xyxy": [500.0, 200.1, 560.3, 420.7],
      "model_index": 2,
      "engine_name": "yolov8n-player.engine",
      "label": "player"
    }
  ]
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `coord_space` | string | Coordinate space for bounding boxes. Always `"model"`. |
| `model_width` | int | Width of the model input (e.g. 640). Bounding boxes are in this space. |
| `model_height` | int | Height of the model input. |
| `models` | array | List of models that contributed detections. |
| `models[].model_index` | int | Index of this model in the inference node's model list. |
| `models[].engine` | string | Full path to the TensorRT engine file. |
| `models[].engine_name` | string | Filename only of the engine. |
| `detections` | array | All detections from all models, merged. |
| `detections[].cls` | int | Class index from model output. |
| `detections[].conf` | float | Confidence score, 0.0-1.0. |
| `detections[].xyxy` | [float; 4] | Bounding box `[x1, y1, x2, y2]` in model coordinate space. |
| `detections[].model_index` | int | Which model produced this detection (-1 if unknown). |
| `detections[].engine_name` | string | Engine filename that produced this detection. |
| `detections[].label` | string | Human-readable class name. Present only if `class_names` is configured on the inference node. |

## Segmentation Metadata

**Default key:** `yolo_segmentation`

Produced by `cuda_infer_yolo` when running segmentation-class models. The JSON metadata contains detection boxes; the actual pixel masks are in binary side data (see below).

```json
{
  "detections": [
    {
      "cls": 0,
      "conf": 0.89,
      "xyxy": [98.5, 148.2, 182.1, 402.0],
      "model_index": 0,
      "engine_name": "yolov8n-seg-court.engine",
      "label": "court"
    },
    {
      "cls": 1,
      "conf": 0.76,
      "xyxy": [280.0, 30.0, 380.5, 130.2],
      "model_index": 0,
      "engine_name": "yolov8n-seg-court.engine",
      "label": "backboard"
    }
  ]
}
```

Each detection in the array has the same fields as detection metadata (see table above). The ordering of detections corresponds 1:1 with the mask index in the binary side data.

### Segmentation Binary Side Data

Masks are attached as `AVFrameSideData`, not in the JSON dictionary. Two formats exist:

#### CPU Masks (`AV_FRAME_DATA_YOLO_SEG_MASKS`, type `0x59534D00`)

Attached when the node parameter `mask_cpu_every_n > 0`.

```
Offset  Size     Field
0       4 bytes  uint32  num_masks     Number of masks
4       4 bytes  uint32  cpu_mask_w    Mask width in pixels
8       4 bytes  uint32  cpu_mask_h    Mask height in pixels
12      4 bytes  uint32  reserved      (always 0)
16      variable float[] mask_data     num_masks * cpu_mask_w * cpu_mask_h floats
```

Each mask is a 2D float array of size `cpu_mask_w * cpu_mask_h`. Masks are stored contiguously: mask 0 first, then mask 1, etc. Values are per-pixel confidence/probability.

#### GPU Masks (`AV_FRAME_DATA_YOLO_SEG_MASKS_GPU`, type `0x59534D01`)

Attached when the node parameter `mask_gpu_every_n > 0`.

```
Offset  Size     Field
0       8 bytes  uint64  gpu_ptr       CUdeviceptr to mask data in GPU memory
8       4 bytes  uint32  num_masks     Number of masks
12      4 bytes  uint32  proto_w       Prototype mask width
16      4 bytes  uint32  proto_h       Prototype mask height
20      4 bytes  uint32  model_w       Full model input width
24      4 bytes  uint32  model_h       Full model input height
```

The GPU buffer pointed to by `gpu_ptr` contains float data of shape `[num_masks, proto_h, proto_w]`.

## Pose Metadata

**Default key:** `yolo_pose`

Produced by `cuda_infer_yolo` when running pose-estimation models. Each pose includes a bounding box and an array of keypoints.

```json
{
  "coord_space": "model",
  "model_width": 640,
  "model_height": 640,
  "num_keypoints": 17,
  "poses": [
    {
      "cls": 0,
      "conf": 0.94,
      "xyxy": [100.2, 150.0, 180.6, 400.3],
      "model_index": 0,
      "engine_name": "yolov8n-pose.engine",
      "label": "person",
      "keypoints": [
        140.5, 155.2, 0.92,
        142.1, 152.0, 0.88,
        138.9, 152.3, 0.90,
        148.0, 158.1, 0.85,
        133.2, 158.5, 0.83,
        155.3, 195.0, 0.91,
        125.8, 198.2, 0.89,
        162.0, 240.5, 0.87,
        118.3, 243.1, 0.86,
        158.7, 280.0, 0.80,
        121.5, 278.3, 0.78,
        150.1, 300.2, 0.90,
        130.4, 302.0, 0.88,
        152.3, 350.1, 0.85,
        128.0, 348.5, 0.84,
        155.0, 395.2, 0.82,
        125.5, 393.8, 0.81
      ]
    },
    {
      "cls": 0,
      "conf": 0.91,
      "xyxy": [350.1, 160.5, 430.0, 410.2],
      "model_index": 0,
      "engine_name": "yolov8n-pose.engine",
      "label": "person",
      "keypoints": [
        390.2, 165.0, 0.93,
        392.5, 162.1, 0.90,
        387.8, 162.5, 0.89,
        398.0, 170.3, 0.86,
        382.1, 170.0, 0.84,
        405.5, 210.2, 0.92,
        375.0, 212.0, 0.90,
        412.3, 255.1, 0.88,
        368.2, 258.0, 0.85,
        408.0, 295.3, 0.82,
        372.5, 293.0, 0.80,
        400.1, 320.5, 0.91,
        380.0, 322.1, 0.89,
        402.5, 365.0, 0.86,
        378.3, 363.2, 0.84,
        405.0, 405.1, 0.83,
        376.0, 403.5, 0.81
      ]
    }
  ]
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `coord_space` | string | Always `"model"`. |
| `model_width` | int | Model input width. |
| `model_height` | int | Model input height. |
| `num_keypoints` | int | Number of keypoints per pose (e.g. 17 for COCO human pose). |
| `poses` | array | One entry per detected person/object. |
| `poses[].cls` | int | Class index. |
| `poses[].conf` | float | Detection confidence, 0.0-1.0. |
| `poses[].xyxy` | [float; 4] | Bounding box `[x1, y1, x2, y2]` in model coordinates. |
| `poses[].model_index` | int | Model index. |
| `poses[].engine_name` | string | Engine filename. |
| `poses[].label` | string | Class name (optional, requires `class_names` config). |
| `poses[].keypoints` | [float] | Flat array of `[x, y, confidence]` triplets. Length = `num_keypoints * 3`. |

### COCO Keypoint Order (17 keypoints)

| Index | Keypoint | Index | Keypoint |
|-------|----------|-------|----------|
| 0 | nose | 9 | right wrist |
| 1 | left eye | 10 | left hip |
| 2 | right eye | 11 | right hip |
| 3 | left ear | 12 | left knee |
| 4 | right ear | 13 | right knee |
| 5 | left shoulder | 14 | left ankle |
| 6 | right shoulder | 15 | right ankle |
| 7 | left elbow | 16 | |
| 8 | right elbow | | |

## Reframer Viewport Metadata

**Default key:** `reframer_bbox`

Produced by the `reframer` node. Describes the current viewport/crop region chosen by the reframing algorithm.

```json
{
  "version": 1,
  "bbox_norm": [0.1, 0.05, 0.75, 0.95],
  "viewport_bbox": [192, 36, 1440, 684],
  "viewport_center_x": 816,
  "full_frame_width": 1920,
  "full_frame_height": 720,
  "latency_ms": 2.3
}
```

### Field Reference

| Field | Type | Description |
|-------|------|-------------|
| `version` | int | Schema version, currently 1. |
| `bbox_norm` | [float; 4] | Viewport `[x1, y1, x2, y2]` normalized to 0.0-1.0 range. |
| `viewport_bbox` | [int; 4] | Viewport `[x1, y1, x2, y2]` in full-frame pixel coordinates. |
| `viewport_center_x` | int | Horizontal center of viewport in pixels. |
| `full_frame_width` | int | Width of the full (uncropped) frame. |
| `full_frame_height` | int | Height of the full (uncropped) frame. |
| `latency_ms` | float | Processing latency of the reframer in milliseconds. |

## Metadata Propagation

The `join_metadata` node merges metadata from multiple sources onto a single frame:
- Dictionary metadata is merged via `av_dict_copy()`.
- Side data is copied via `av_frame_new_side_data_from_buf()`. If a side data type already exists on the primary frame, it is not overwritten.

This is how detections from separate inference nodes (running different models) end up on the same frame before reaching downstream consumers.

## How basketball_analysis Consumes Metadata

The `basketball_analysis` node reads the configured metadata key (default `yolo_detections_v1`) from the AVFrame dictionary and parses it into:

- A `MetadataEnvelope` with `coord_space`, `model_width`, `model_height`
- A vector of `DetectionBox` structs, each with `cls`, `conf`, `x1/y1/x2/y2`, `label`, `model_index`, `engine_name`

It filters detections by label (`ball`, `hoop`, `player`) and `model_index` to drive shot detection, ball tracking, and reframe target computation. Pose keypoints (specifically ankle positions) are used as foot-bbox proximity anchors for shot detection logic.

## Source Files

| File | Role |
|------|------|
| `src/nodes/hwaccel/cuda_infer_yolo.cpp` | Inference orchestration and metadata serialization |
| `src/nodes/hwaccel/cuda_infer_yolo_base.hpp` | Result structs and side data type constants |
| `src/nodes/hwaccel/yolo_decode_detection.hpp` | Detection box decoding |
| `src/nodes/hwaccel/yolo_decode_segmentation.hpp` | Segmentation mask decoding and assembly |
| `src/nodes/hwaccel/yolo_decode_pose.hpp` | Pose keypoint decoding |
| `src/nodes/basketball_analysis.cpp` | Metadata parsing and basketball logic |
| `src/nodes/hwaccel/reframer.cpp` | Viewport metadata output |
| `src/nodes/smooth_crop_viewport.cpp` | Detection-driven crop filtering |
| `src/nodes/join_metadata.cpp` | Metadata merge across streams |
