# `cuda_infer_yolo`

`cuda_infer_yolo` runs TensorRT-based YOLO inference on CUDA video frames and writes detection metadata onto each output frame. The node forwards the original frame downstream and augments it with JSON detection results.

## What It Does

1. Reads CUDA `av::VideoFrame` input from `src` and forwards the same frame to `dst`.
2. Initializes TensorRT from `engine` and discovers the image input tensor and first output tensor.
3. Uses a CUDA preprocess kernel to convert the incoming `NV12` frame into the model's input tensor format.
4. Runs TensorRT inference on the current frame.
5. Decodes YOLO-style output tensors into bounding boxes, class IDs, and confidences.
6. Applies class-aware non-maximum suppression (NMS).
7. Writes JSON detections metadata onto the output frame.

## Input Requirements

- Input frames must be CUDA frames: `AV_PIX_FMT_CUDA`
- The hardware-backed software format must be `NV12`
- The incoming frame dimensions must match the model input tensor dimensions
- The TensorRT engine must expose at least one image input tensor and one output tensor
- Dynamic TensorRT dimensions are not supported in this version
- Supported input tensor layouts:
  - `CHW` with `C=3`
  - `NCHW` with `N=1, C=3`
- Supported input/output tensor data types:
  - `float`
  - `half`

## Output Decoding

The node supports several common YOLO export layouts:

- `2D` outputs such as `[84, N]` or `[N, 84]`
- `3D` outputs such as:
  - `[1, 84, N]`
  - `[1, N, 84]`
  - end-to-end layouts `[1, N, 6]` or `[1, 6, N]` with `[x1, y1, x2, y2, conf, cls]`

For the ambiguous `[1, N, 6]` / `[1, 6, N]` case, the default decoder keeps the legacy interpretation above.
Use `output_box_format: raw_cxcywh` to instead treat those tensors as raw YOLO output with
`[cx, cy, w, h, score0, score1]`.

For class-score layouts, the node:

- selects the best-scoring class per candidate
- filters by `conf_thresh`
- converts center-width-height boxes to `xyxy`
- applies class-aware NMS using `iou_thresh`
- limits final results to `max_det`

## Output Metadata

By default, the node stores JSON metadata under `yolo_detections_v1`. The JSON contains:

- `version`
- `coord_space`
- `thresholds`
- `detections`

Each detection entry contains:

- `cls`
- `label` if class names are available
- `conf`
- `xyxy`

Coordinates are emitted in model input space, not remapped back to an original source resolution. The metadata marks this explicitly with `coord_space = "model"`.

## Parameters

### Required

- `src`
  Input video edge.

- `dst`
  Output video edge.

- `engine`
  Path to the TensorRT engine file.

- `hwaccel`
  Name of the shared `HWAccelDevice` instance to use.

### Optional

- `conf_thresh`
  Confidence threshold applied before NMS. Default: `0.25`.

- `iou_thresh`
  IoU threshold used for class-aware NMS. Default: `0.45`.

- `max_det`
  Maximum number of detections kept after NMS. Default: `300`.

- `infer_every_n`
  Run inference only on every Nth frame. Frames skipped by this setting are passed through without new detection metadata. Default: `1`.

- `metadata_key_out`
  Metadata key written to the outgoing frame. Default: `yolo_detections_v1`.

- `debug_log_metadata`
  If `true`, log serialized detection metadata periodically. Default: `false`.

- `debug_log_every_n`
  Logging interval used together with `debug_log_metadata`. Default: `30`.

- `input_format`
  Channel order for RGB preprocessing. Values `BGR` or `bgr` enable BGR ordering; any other value is treated as RGB.

- `output_box_format`
  Optional override for ambiguous `[1, N, 6]` / `[1, 6, N]` outputs.
  Supported values:
  - `end2end_xyxy` for `[x1, y1, x2, y2, conf, cls]` (default)
  - `raw_cxcywh` for `[cx, cy, w, h, score0, score1]`

- `yolo_classes`
  Optional path to a whitespace-separated class-label file.
  The node reads the file at startup and maps class IDs to labels by index.

- `class_names`
  Optional class label mapping.
  Supported form:
  - array of class-name strings

- `class_index_remap_per_model`
  Optional array matching `engines`, where each item is an integer array used to remap decoded class IDs for that model.
  Example: `[[], [], [1, 0]]` swaps class `0` and `1` for the third model before labels are attached.

## Class Names

If neither `yolo_classes` nor `class_names` is provided, detections still include numeric `cls` IDs.

If `yolo_classes` is provided:

- the file is parsed as whitespace-separated labels
- labels are mapped to class IDs by index
- use `class_names` instead if a label itself contains spaces

If `class_names` is provided:

- a string array maps class IDs to labels by index

Do not set both `yolo_classes` and `class_names` at the same time.

## Runtime Notes

- Auxiliary engine input tensors may exist, but the node preprocesses only the detected image input tensor
- The node binds tensor addresses by name using the TensorRT 10 API
- Only the first output tensor is decoded in this version
- On preprocessing, TensorRT, or copy failures, the node logs the error and returns without producing output for that frame

## Example

```txt
cuda_infer_yolo:
  src: decoded_cuda
  dst: detected_cuda
  engine: /models/yolo.engine
  hwaccel: cuda0
  conf_thresh: 0.25
  iou_thresh: 0.45
  max_det: 300
  infer_every_n: 1
  metadata_key_out: yolo_detections_v1
  input_format: RGB
  output_box_format: raw_cxcywh
  yolo_classes: /models/yolo_classes.txt
```
