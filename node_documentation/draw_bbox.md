# `draw_bbox`

`draw_bbox` draws bounding boxes onto CUDA `NV12` video frames using metadata attached to each frame.

## What It Does

1. Reads a CUDA `av::VideoFrame` from `src`.
2. Copies the frame into a new CUDA output frame.
3. Parses bbox metadata from `metadata_key`.
4. Draws one or more boxes onto the copied frame.
5. Forwards the annotated frame to `dst`.

## Supported Metadata

`draw_bbox` accepts two metadata shapes:

- single-box metadata such as `viewport_bbox` or `bbox_norm`
- YOLO-style metadata with:
  - `coord_space`
  - `model_width`
  - `model_height`
  - `detections`

Each YOLO detection may include:

- `cls`
- `label`
- `conf`
- `xyxy`

## Letterbox Remap

When YOLO runs on a padded model canvas, detections are still emitted in model-space coordinates.

Use the optional `model_content_*` parameters to map detections from the active image region inside that canvas back onto the full output frame:

- `model_content_width`
- `model_content_height`
- `model_content_offset_x`
- `model_content_offset_y`

Example:

- source frame `1920x1080`
- YOLO input canvas `960x960`
- active image area `960x540`
- centered vertical padding `y=210`

Then set:

- `model_content_width: 960`
- `model_content_height: 540`
- `model_content_offset_x: 0`
- `model_content_offset_y: 210`

## Parameters

### Required

- `src`
  Input video edge.

- `dst`
  Output video edge.

### Optional

- `metadata_key`
  Metadata key to read bounding boxes from. Default: `reframer_bbox`.

- `bbox_thickness`
  Outline thickness in pixels. Default: `2`.

- `min_conf`
  Minimum confidence for YOLO detections. Default: `0.0`.

- `allowed_classes`
  Optional array of class IDs to draw.

- `allowed_labels`
  Optional array of labels to draw.

- `model_content_width`
  Width of the real image area inside the model canvas. Default: disabled.

- `model_content_height`
  Height of the real image area inside the model canvas. Default: disabled.

- `model_content_offset_x`
  Left offset of the real image area inside the model canvas. Default: `0`.

- `model_content_offset_y`
  Top offset of the real image area inside the model canvas. Default: `0`.

- `width`
- `height`
- `pixel_format`
- `real_pixel_format`
  Output-format hints used by the node graph.

- `debug_log_every_n`
  Logging interval. Default: `0`.

## Example

```txt
node.add {
  "type": "draw_bbox",
  "src": "v_1080p_with_md",
  "dst": "v_annotated_cuda",
  "metadata_key": "yolo_detections_v1",
  "allowed_labels": ["ball", "foot", "player"],
  "model_content_width": 960,
  "model_content_height": 540,
  "model_content_offset_x": 0,
  "model_content_offset_y": 210,
  "width": 1920,
  "height": 1080,
  "pixel_format": "cuda",
  "real_pixel_format": "nv12"
}
```
