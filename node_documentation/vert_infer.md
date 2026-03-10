# `vert_infer`

`vert_infer` runs a TensorRT model over a short history of CUDA video frames and writes crop metadata onto each output frame. The node is intended for vertical reframing: it predicts one of five horizontal actions, updates an internal viewport state, and stores the resulting crop box in frame metadata.

## What It Does

1. Reads CUDA `av::VideoFrame` input from `src` and forwards the same frame to `dst`.
2. Initializes TensorRT from `engine` and discovers the model input/output tensors.
3. Uses the model's visual tensor shape to determine the expected frame width, frame height, and history length.
4. Collects a rolling history of frames plus viewport kinematics.
5. Preprocesses the frame history into the visual input tensor.
6. Builds a kinematics tensor from:
   - normalized viewport center `center_x / input_width`
   - viewport velocity
   - normalized acceleration
7. Runs inference and takes the argmax of the 5 action scores.
8. Maps the winning action to a horizontal force:
   - `0 -> -1.0`
   - `1 -> -0.3`
   - `2 -> 0.0`
   - `3 -> 0.3`
   - `4 -> 1.0`
9. Updates the viewport center, clamps it to the valid crop range, and writes crop metadata into the outgoing frame.

## Input Requirements

- Input frames must be CUDA frames: `AV_PIX_FMT_CUDA`
- The hardware-backed software format must be `NV12`
- The incoming frame dimensions must match the model's visual tensor dimensions
- The TensorRT engine must expose:
  - one visual input tensor
  - one viewport kinematics input tensor
  - one output tensor containing exactly 5 action scores
- Dynamic TensorRT dimensions are not supported in this version

## Warmup Behavior

The node needs `history_length` frames before it can run inference.

- `warmup_mode=wait`: pass frames through without crop metadata until enough history is available
- `warmup_mode=center_crop`: emit centered crop metadata during warmup

## Output Metadata

By default, the node stores JSON metadata under `vert_crop_v1`. The JSON contains:

- `version`
- `x1`
- `x2`
- `y`
- `w`
- `h`
- `viewport_center_x`
- `viewport_width`
- `action`
- `latency_ms`

If `debug_log_action_scores=true`, the metadata also includes:

- `action_scores`

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

- `metadata_key_out`
  Metadata key written to the outgoing frame. Default: `vert_crop_v1`.

- `history_length`
  Number of frames kept in history. If omitted, the node uses the history length implied by the model input tensor. If provided, it must match the model.

- `viewport_width`
  Output crop width in pixels. If omitted, the node derives a vertical 9:16 crop width from the input height.

- `friction`
  Damping factor applied to per-frame horizontal movement. Default: `0.1`.

- `full_frame_width`
  Full-frame width used to calibrate horizontal movement magnitude. Default: `1920`.

- `full_frame_height`
  Full-frame height configuration value. Default: `1080`.
  Note: the current implementation validates that it is positive, but movement scaling is derived from `full_frame_width`.

- `fallback_fps`
  Optional fallback frame rate used only when frame timestamps are missing, invalid, or unusable. If omitted, the node still falls back internally to `30 FPS` as a last-resort safety default.

- `debug_log_every_n`
  If greater than `0`, log every Nth processed frame.

- `debug_log_action_scores`
  If `true`, include action scores in debug logs and output metadata.

- `visual_tensor_name`
  Explicit TensorRT tensor name for the visual input. Usually not needed if autodetection succeeds.

- `kinematics_tensor_name`
  Explicit TensorRT tensor name for the viewport kinematics input. Usually not needed if autodetection succeeds.

- `output_tensor_name`
  Explicit TensorRT tensor name for the action-score output. Usually not needed if autodetection succeeds.

- `input_format`
  Channel order for RGB preprocessing. Values `BGR` or `bgr` enable BGR ordering; any other value is treated as RGB.

- `visual_mode`
  Visual input mode. Allowed values:
  - `grayscale`
  - `rgb`
  Default: `grayscale`.

- `warmup_mode`
  Warmup behavior before enough history has been collected. Allowed values:
  - `wait`
  - `center_crop`
  Default: `wait`.

## Timing

The node uses frame timestamps first. `fallback_fps` is used only when it cannot compute a reliable delta from frame `pts` values, such as:

- the first frame
- missing timestamps
- non-monotonic timestamps
- large timestamp jumps

## Motion Scaling

Horizontal motion is calibrated relative to `full_frame_width`.

Current behavior:

- full force corresponds to roughly `10` pixels of movement on a `1920`-pixel-wide full frame
- movement is scaled proportionally for other configured full-frame widths
- `full_frame_height` does not currently affect motion scaling

## Example

```txt
vert_infer:
  src: decoded_cuda
  dst: reframed_cuda
  engine: /models/vert.engine
  hwaccel: cuda0
  metadata_key_out: vert_crop_v1
  viewport_width: 608
  full_frame_width: 1920
  full_frame_height: 1080
  fallback_fps: 30
  visual_mode: grayscale
  warmup_mode: center_crop
```
