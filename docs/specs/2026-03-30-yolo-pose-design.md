# YOLO Pose Decoder and draw_keypoints Node

## Summary

Add pose/keypoint support to the YOLO inference pipeline. A new `PoseDecoder` decodes keypoint data from YOLO pose models (e.g., basketball court landmark detection with 34 keypoints). A new `draw_keypoints` CUDA node renders the keypoints as filled circles on NV12 frames.

## Context

The court-pose model (`court-pose.plan`) is a YOLOv8n-pose model trained on basketball court landmarks:
- Input: `1x3x960x960`
- Output: `1x107x18900` (raw_cxcywh format, no NMS)
- 1 class: `court`
- 34 keypoints per detection, each with (x, y, visibility)
- Output attrs: 4 (box) + 1 (class) + 102 (34 * 3 keypoints) = 107

Existing infrastructure: `DetectionDecoder` and `SegmentationDecoder` in separate header files, `CudaOverlayBase` for draw nodes, coordinate remapping for padded model input to full-resolution output.

## Components

### 1. PoseResult struct (cuda_infer_yolo_base.hpp)

```cpp
struct PoseResult : DetectionResult {
    std::vector<float> keypoints;  // flat [x, y, conf, x, y, conf, ...] per detection
    int num_keypoints = 0;         // keypoints per detection (e.g. 34)
};
```

### 2. PoseDecoder (yolo_decode_pose.hpp)

New file following the same pattern as `yolo_decode_detection.hpp` and `yolo_decode_segmentation.hpp`.

- Same layout auto-detection (attrs-first vs dets-first) as existing decoders
- Output tensor layout: `[4 box] + [num_classes scores] + [num_keypoints * 3]`
- `num_keypoints` derived from: `(attrs - 4 - num_classes) / 3`
- `num_classes` provided via model param (default 1)
- Decode loop: threshold on class conf, extract bbox (cxcywh -> xyxy), extract flat keypoint triplets
- Pure CPU decode, no GPU work needed

### 3. ModelRunner changes (cuda_infer_yolo_base.hpp)

- Add `std::unique_ptr<PoseDecoder> pose_decoder` field
- Add `int num_classes = -1` field (default -1 for auto-detect; required for pose models)

### 4. Process loop integration (cuda_infer_yolo.cpp)

New param: `metadata_key_pose_` (default `"yolo_pose"`)

New `else if (model.task_type == TaskType::Pose)` branch:
1. Call `pose_decoder->decode(host_outputs, output_dims, dp)`
2. Add bounding box detections to `all_dets` (included in main detection metadata)
3. Build pose metadata JSON and attach to frame via `av_dict_set` under `metadata_key_pose_`

### 5. Pose metadata format

```json
{
  "coord_space": "model",
  "model_width": 960,
  "model_height": 960,
  "num_keypoints": 34,
  "poses": [
    {
      "cls": 0,
      "conf": 0.85,
      "label": "court",
      "xyxy": [100, 200, 800, 700],
      "model_index": 3,
      "engine_name": "court-pose.plan",
      "keypoints": [120.5, 340.2, 0.95, 450.1, 210.8, 0.88, ...]
    }
  ]
}
```

Keypoints are flat triplets: `[x1, y1, conf1, x2, y2, conf2, ...]` for compact serialization.

### 6. draw_keypoints CUDA kernel (draw_keypoints.cu)

Kernel `kDrawKeypoints`:
- Input: array of (x, y) positions in frame coords, radius, Y/UV plane pointers, frame stride/dims
- Each thread checks pixel distance to each keypoint center; if within radius, writes color (default white: Y=235, U=128, V=128)

### 7. draw_keypoints node (draw_keypoints.cpp)

Extends `CudaOverlayBase`. Params:

| Param | Default | Description |
|-------|---------|-------------|
| `metadata_key` | `"yolo_pose"` | Metadata key to read |
| `radius` | `3` | Circle radius in pixels |
| `color` | `"white"` | Circle color (DrawColor) |
| `min_conf` | `0.0` | Minimum keypoint confidence to draw |
| `model_content_width` | `0` | Coordinate remapping (same as draw_segmask) |
| `model_content_height` | `0` | Coordinate remapping |
| `model_content_offset_x` | `0` | Coordinate remapping |
| `model_content_offset_y` | `0` | Coordinate remapping |
| `debug_log_every_n` | `0` | Debug logging interval |

Process flow:
1. Read frame metadata at `metadata_key`
2. Parse JSON, extract flat keypoint arrays from each pose
3. Filter by `min_conf` per keypoint
4. Remap from model coords to frame coords (existing `remapModelCoord` pattern)
5. Upload keypoint positions to GPU scratch buffer
6. Launch kernel to draw circles
7. Pass frame through

### 8. Example pipeline update (yolo_reframe_crop.avplumber)

Add court-pose model (5th model) to `cuda_infer_yolo` models array:
```json
{
  "engine": "/home/fedora/tensorrt/court-pose-2/court-pose.plan",
  "task_type": "pose",
  "class_names": ["court"],
  "num_classes": 1,
  "output_box_format": "raw_cxcywh"
}
```

Draw chain order:
```
v_1080p_with_md -> draw_segmask -> draw_keypoints -> draw_bbox -> draw_text -> ...
```

Keypoints rendered after segmask overlay and before bounding boxes.

The `draw_keypoints` node uses the same coordinate remapping as the other draw nodes:
`model_content_width: 960, model_content_height: 540, model_content_offset_x: 0, model_content_offset_y: 210`

## New files

- `src/nodes/hwaccel/yolo_decode_pose.hpp`
- `src/nodes/hwaccel/draw_keypoints.cu`
- `src/nodes/hwaccel/draw_keypoints.cpp`

## Modified files

- `src/nodes/hwaccel/cuda_infer_yolo_base.hpp` — PoseResult struct, ModelRunner fields
- `src/nodes/hwaccel/cuda_infer_yolo.cpp` — pose branch in process loop, metadata building, param parsing
- `examples/yolo/yolo_reframe_crop.avplumber` — add court-pose model, add draw_keypoints node
- `Makefile` — only if the existing `.cu` pattern rule doesn't cover `draw_keypoints.cu`
