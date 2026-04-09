# `cuda_infer_yolo`

`cuda_infer_yolo` runs TensorRT-based YOLO inference on CUDA video frames and writes detection (and optionally segmentation) metadata onto each output frame. The node forwards the original frame downstream and augments it with JSON detection results and optional mask data.

## What It Does

1. Reads CUDA `av::VideoFrame` input from `src` and forwards the same frame to `dst`.
2. Initializes TensorRT from each model's `engine` and discovers input/output tensors.
3. Uses a CUDA preprocess kernel to convert the incoming `NV12` frame into the model's input tensor format.
4. Runs TensorRT inference on the current frame (all models).
5. Decodes YOLO-style output tensors into bounding boxes, class IDs, and confidences.
6. For segmentation models: assembles per-detection masks from prototype masks and coefficients via GPU kernel.
7. Writes JSON detections metadata onto the output frame.
8. Optionally outputs GPU masks on `dst_seg` and/or CPU masks as AVFrame side data.

## Input Requirements

- Input frames must be CUDA frames: `AV_PIX_FMT_CUDA`
- The hardware-backed software format must be `NV12`
- The incoming frame dimensions must match the model input tensor dimensions
- The TensorRT engine must expose at least one image input tensor and one output tensor
- Dynamic TensorRT dimensions are not supported
- Supported input tensor layouts:
  - `CHW` with `C=3`
  - `NCHW` with `N=1, C=3`
- Supported input/output tensor data types:
  - `float`
  - `half`

## Task Types

Each model has a `task_type` that determines how its output is decoded:

### Detection (`task_type: "detection"`)

Decodes bounding boxes from a single output tensor. Supports several common YOLO export layouts:

- `2D` outputs such as `[84, N]` or `[N, 84]`
- `3D` outputs such as:
  - `[1, 84, N]`
  - `[1, N, 84]`
  - end-to-end layouts `[1, N, 6]` or `[1, 6, N]` with `[x1, y1, x2, y2, conf, cls]`

### Segmentation (`task_type: "segmentation"`)

Decodes bounding boxes plus 32 mask coefficients from output tensor 0, and mask prototypes from output tensor 1. The segmentation model must have exactly 2 output tensors:

- **output0**: detections + 32 mask coefficients per detection (same layout as detection but with 32 extra attrs after class scores)
- **output1**: mask prototypes `[1, 32, proto_h, proto_w]` or `[32, proto_h, proto_w]`

Mask assembly is performed on GPU: matrix multiply of coefficients x prototypes followed by sigmoid activation.

**GPU mask output** (`dst_seg` edge): AVFrame containing assembled masks at prototype resolution, with PTS matching the input frame. Controlled by `mask_gpu_every_n`.

**CPU mask output** (AVFrame side data): Downsampled masks attached as binary side data on the passthrough frame. Controlled by `mask_cpu_every_n`.

## Output Metadata

The node stores JSON metadata under `metadata_key_detection` (default: `yolo_detections`). The JSON contains:

- `coord_space`
- `model_width`, `model_height`
- `models` array
- `detections` array

Each detection entry contains:

- `cls`
- `label` (if class names are available)
- `conf`
- `xyxy`
- `model_index`
- `engine_name`

Coordinates are emitted in model input space, not remapped back to an original source resolution. The metadata marks this explicitly with `coord_space = "model"`.

## CPU Mask Side Data Format

When segmentation models emit CPU masks, they are stored as AVFrame side data with type `0x59534D00`. Binary layout:

```
Header (16 bytes):
  uint32_t  num_detections    -- number of masks (matches detection count)
  uint32_t  mask_width        -- mask_cpu_resolution
  uint32_t  mask_height       -- mask_cpu_resolution
  uint32_t  reserved          -- padding, set to 0

Masks (num_detections * mask_width * mask_height * sizeof(float) bytes):
  float[]   mask_0            -- mask_width * mask_height floats, row-major
  float[]   mask_1
  ...
```

Masks are in the same order as detections in the JSON metadata.

## Parameters

### Required

- `src`
  Input video edge.

- `dst`
  Output video edge (passthrough frame with detection metadata).

- `models`
  Array of model objects. Each model object has:
  - `engine` (required) — path to the TensorRT engine file.
  - `task_type` (optional) — `"detection"` (default), `"segmentation"`, or `"pose"` (future).
  - `output_box_format` (optional) — `"end2end_xyxy"` (default) or `"raw_cxcywh"`.
  - `class_names` (optional) — array of class-name strings, mapped to class IDs by index.
  - `class_index_remap` (optional) — integer array used to remap decoded class IDs.
  - `include_in_detection_metadata` (optional) — boolean, defaults to `true`. When `false`, the model still runs and can emit masks/side data, but its decoded detections are excluded from the shared JSON metadata list.

  All models must share the same input dimensions and data type.

### Optional

- `dst_seg`
  Optional output edge for GPU segmentation masks. Only used when segmentation models are present.

- `conf_thresh`
  Confidence threshold. Default: `0.25`.

- `max_det`
  Maximum number of detections kept after sorting. Default: `300`.

- `infer_every_n`
  Run inference only on every Nth frame. Default: `1`.

- `metadata_key_detection`
  Metadata key for detection JSON. Default: `"yolo_detections"`.

- `metadata_key_segmentation`
  Metadata key for segmentation info. Default: `"yolo_segmentation"`.

- `mask_gpu_every_n`
  Output GPU masks every Nth inference frame. Default: `1`.

- `mask_cpu_every_n`
  Attach CPU mask side data every Nth inference frame. Default: `2`.

- `mask_cpu_resolution`
  Square resolution for downsampled CPU masks. Default: `120`.

- `debug_log_metadata`
  If `true`, log serialized detection metadata periodically. Default: `false`.

- `debug_log_every_n`
  Logging interval used together with `debug_log_metadata`. Default: `30`.

- `input_format`
  Channel order for RGB preprocessing. Values `BGR` or `bgr` enable BGR ordering; any other value is treated as RGB.

## Runtime Notes

- The CUDA context is obtained from the incoming frame's `hw_frames_ctx` (no `hwaccel` parameter needed)
- Auxiliary engine input tensors may exist but only the detected image input tensor is preprocessed
- The node binds tensor addresses by name using the TensorRT 10 API
- On preprocessing, TensorRT, or copy failures, the node logs the error and returns without producing output for that frame
- `mask_gpu_every_n` and `mask_cpu_every_n` count inference frames, not total frames

## Examples

Detection only:

```json
node.add { "type": "cuda_infer_yolo", "src": "v_pre_yolo", "dst": "v_post_yolo",
  "input_format": "RGB", "conf_thresh": 0.25, "max_det": 300,
  "models": [
    { "engine": "/models/yolo.engine", "task_type": "detection", "class_names": ["person", "car", "dog"] }
  ]
}
```

Multiple models with segmentation:

```json
node.add { "type": "cuda_infer_yolo", "src": "v_pre_yolo", "dst": "v_post_yolo", "dst_seg": "v_seg_masks",
  "input_format": "RGB", "conf_thresh": 0.20, "max_det": 20,
  "mask_gpu_every_n": 1, "mask_cpu_every_n": 3, "mask_cpu_resolution": 120,
  "models": [
    { "engine": "/models/ball.plan", "task_type": "detection", "class_names": ["basketball"], "output_box_format": "end2end_xyxy" },
    { "engine": "/models/yolo-seg.plan", "task_type": "segmentation", "class_names": ["person"], "output_box_format": "raw_cxcywh" }
  ]
}
```
