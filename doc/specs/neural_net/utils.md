# neural_net/utils — Viewport and Reframing Nodes

## Node: `mediapipe_autoflip_crop_metadata`
Metadata-only crop-region solver using Google MediaPipe's `FrameCropRegionComputer` + `KinematicPathSolver`. Reads detection bboxes from frame metadata, computes the tightest covering crop window, feeds the window center through a velocity-limited 1D integrator, and writes the smoothed crop rectangle as JSON metadata. No GPU frames are read or written — the node is a pure metadata passthrough.

Requires build flag `HAVE_MEDIAPIPE_AUTOFLIP=1` and the `libavp_mediapipe_autoflip.so` bridge library.

### Parameters
| Param | Default | Description |
|-------|---------|-------------|
| `saliency` | `[]` | Array of saliency source objects (see below). If empty, falls back to `metadata_key_ins` |
| `metadata_key_ins` | `["face_detections_v1"]` | Legacy: list of metadata keys in face_detections_v1 format (used only when `saliency` is empty) |
| `metadata_key_out` | `"smoothed_crop_viewport_v1"` | Output metadata key |
| `crop_w` | `0` | Crop width in pixels (0 = 9:16 of frame height) |
| `crop_h` | `0` | Crop height in pixels (0 = full frame height) |
| `min_motion_to_reframe` | `0.0` | KinematicPathSolver deadzone in pixels |
| `max_velocity` | `0.0` | KinematicPathSolver max speed in px/sec at `pixels_per_degree=1.0` (0 = unlimited) |
| `lookahead_frames` | `6` | Lookahead buffer (set to 0 for fully causal/live operation) |
| `reset_on_scene_cut` | `true` | Clear solver history when scene cut metadata fires |
| `scene_cut_metadata_key` | `""` | Metadata key to check for scene cut signal |
| `scene_cut_field` | `"scene_cut"` | JSON field inside the scene cut metadata |
| `model_content_width` | `0` | Fallback model content width for coord remapping |
| `model_content_height` | `0` | Fallback model content height |
| `model_content_offset_x` | `0` | Fallback horizontal padding offset |
| `model_content_offset_y` | `0` | Fallback vertical padding offset |
| `debug_log_every_n` | `0` | Log a line every N frames (0 = disabled) |

### Saliency source object fields
Each entry in the `saliency` array:

| Field | Default | Description |
|-------|---------|-------------|
| `metadata_key` | required | Frame metadata key to read detections from |
| `role` | `"context"` | `"preferred"` or `"context"` — preferred gets full weight, context gets 1/10 weight |
| `weight` | `1.0` | Detection weight passed to `FrameCropRegionComputer` |
| `min_conf` | `0.0` | Minimum detection confidence to include |
| `optional_input` | `false` | If true, missing metadata key is not an error |
| `shot_type_key` | `""` | When non-empty, gate this source on a shot-type match |
| `shot_type_field` | `"camera_shot"` | JSON field inside the shot-type metadata to compare |
| `shot_type_value` | `""` | Required value of `shot_type_field` for gate to pass |
| `fallback_when_empty` | `false` | When true, this source is only used if all non-fallback sources yielded zero detections for the current frame |

### Detection gathering — two-pass priority fallback
1. **Pass 1 (primary):** all sources with `fallback_when_empty: false` are queried. Their detections are collected.
2. **Pass 2 (fallback):** if pass 1 yielded zero detections, sources with `fallback_when_empty: true` are queried.

This allows player detections to serve as a fallback tracking target when ball and ball-handler are absent, without contributing to crop decisions on frames where the ball is visible.

### Detection format support
- **YOLO/tracker format:** `{"model_width":W,"model_height":H,"detections":[{"xyxy":[x1,y1,x2,y2],"conf":0.85,"label":"..."}]}` — remapped from model space to frame space using `model_width/height` from the JSON (falling back to `model_content_*` node params).
- **Face detection format:** top-level array or `{"faces":[...]}` with `bbox` (pixel) or `bbox_norm` (normalised) fields.

### Output JSON (`metadata_key_out`)
```json
{
  "viewport_bbox": [x1, 0, x2, 1080],
  "viewport_dst_width": 608,
  "viewport_dst_height": 1080,
  "full_frame_width": 1920,
  "full_frame_height": 1080,
  "status": "ok"
}
```
`status` values: `"ok"`, `"no_subjects"` (solver coasting, no detections), `"fallback_center"`, `"error:<msg>"`.

---

## Node: `smooth_crop_viewport`
Smooth viewport center tracking with configurable filtering.

### Parameters
| Param | Default | Description |
|-------|---------|-------------|
| `metadata_keys` | required | Detection metadata keys to read bboxes from |
| `viewport_width/height` | required | Fixed viewport dimensions |
| `filter_type` | "kalman" | "kalman", "butterworth" (1-8th order), "lti", "none" |
| `max_velocity/acceleration/jerk/snap` | — | Derivative motion limits |
| `kalman_q_pos/q_vel/r_meas` | — | Kalman filter tuning |
| `cutoff_freq_hz` / `quality_factor` | — | Butterworth filter tuning |

### Output
Attaches `viewport_center_x` metadata to frame for downstream crop nodes.

---

## Node: `reframer`
Dynamic video reframing using a TensorRT trajectory prediction model.

### Parameters
| Param | Default | Description |
|-------|---------|-------------|
| `engine_path` | required | TensorRT model for viewport prediction |
| `visual_model_w/h` | required | Model input resolution |
| `history_length` | required | Frame history for model input |
| `kinematics_dim` | required | State vector dimensionality |

### Pipeline
1. Encode visual features: NV12 → NCHW tensor (CUDA kernel)
2. Maintain kinematic state (viewport center, velocity)
3. Run TensorRT inference for next-frame viewport prediction
4. Output viewport crop coordinates with smooth transitions

### CUDA kernels
Uses `kNV12_to_NCHW_fp32/fp16` for visual feature encoding, plus `reframer.cu` for crop extraction.
