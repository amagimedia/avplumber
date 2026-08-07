# `cuda_infer_scene_cut_onnx`

`cuda_infer_scene_cut_onnx` runs a pairwise CNN-embedding scene-cut classifier on
CUDA video frames and writes the final
`camera_shot_transition` decision directly onto each output frame. Unlike `luma_diff`
/ `hog_diff`, this model's cut threshold is baked into the ONNX graph at export time,
so no external Python threshold/debounce decider node is needed downstream.

## What It Does

1. Reads CUDA `av::VideoFrame` input from `src` and forwards the same frame to `dst`.
2. Initializes TensorRT from `engine` and resolves the `frame_a`/`frame_b`/
   `scene_cut_flag` tensors by exact name.
3. Uses a dedicated CUDA preprocess kernel to convert the incoming `NV12` frame into
   an RGB `float32` CHW tensor normalized to `[0,1]` (no ImageNet mean/std) — matching
   the model's documented preprocessing contract exactly.
4. Retains the previous call's preprocessed tensor as `frame_a` via a device-to-device
   copy each frame (no re-preprocessing of the previous frame, no retained
   `av::VideoFrame` history).
5. Runs TensorRT inference and reads the `int64`/`int32` `scene_cut_flag` output.
6. Writes `{"camera_shot_transition": bool, "camera_shot_type": "unknown"}` onto the
   output frame's metadata (schema-compatible with the `luma_scene_cut` Python node's
   output, so downstream consumers — `player_tracker`'s `camera_shot_metadata_key`,
   a metadata consumer or debug scene-cut counter — need no changes).

## Input Requirements

- Input frames must be CUDA frames: `AV_PIX_FMT_CUDA`, hardware sw_format `NV12`.
- Incoming frame dimensions must exactly match `input_width`/`input_height` (the
  caller's avplumber graph is responsible for scaling to this resolution beforehand,
  e.g. via `scale_npp`; this node does not resize).
- The TensorRT engine must expose exactly two input tensors (`frame_a`, `frame_b` by
  default) and one output tensor (`scene_cut_flag` by default).
- Both input tensors must be `float32` `(1,3,H,W)` NCHW.
- The output tensor must be `int64` or `int32`, shape `(1,)`.
- Dynamic-shape engines (built with `--minShapes`/`--optShapes`/`--maxShapes`) are
  supported: the node resolves dynamic input dims via `setInputShape()` using
  `input_width`/`input_height` before allocating device buffers.

## First-frame behavior

On the very first frame seen after (re)initialization, there is no real previous
frame yet. The node preprocesses the current frame into the `frame_b` slot (so the
next call's device-to-device shift has real data) but does **not** run inference
against an undefined `frame_a`; it writes `camera_shot_transition: false` and moves
on. From the second frame onward, inference runs normally every frame.

## Output Metadata

Written under `metadata_key` (default: `camera_shot_info`):

```json
{"camera_shot_transition": false, "camera_shot_type": "unknown"}
```

## Parameters

- `src`, `dst` (string) — edge names.
- `engine` (string, required) — path to the TensorRT `.plan` file.
- `hwaccel` (string, required) — shared `HWAccelDevice` object name.
- `metadata_key` (string, default `"camera_shot_info"`) — output metadata key.
- `frame_a_tensor_name` (string, default `"frame_a"`) — previous-frame input tensor name.
- `frame_b_tensor_name` (string, default `"frame_b"`) — current-frame input tensor name.
- `output_tensor_name` (string, default `"scene_cut_flag"`) — output tensor name.
- `input_width` (int, default `360`), `input_height` (int, default `202`) — expected
  frame dimensions, matching the model's documented `(N,3,202,360)` NCHW contract.
- `debug_log_every_n` (int, default `0`) — periodic debug logging cadence; `0` disables.

## Example

```json
{
  "type": "scale_npp",
  "src": "v_for_scene_cut_onnx",
  "dst": "v_pre_scene_cut_onnx",
  "params": {"w": 360, "h": 202, "format": "nv12"}
},
{
  "type": "cuda_infer_scene_cut_onnx",
  "src": "v_pre_scene_cut_onnx",
  "dst": "v_scene_cut_onnx",
  "params": {
    "engine": "/models/scene_cut/scene_cut_detector_202x360_fp16.plan",
    "hwaccel": "@gpu",
    "metadata_key": "camera_shot_info"
  }
}
```
