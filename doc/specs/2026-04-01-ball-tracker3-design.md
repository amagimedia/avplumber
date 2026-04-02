# ball_tracker — Single-Ball Tracker with Physics Gating

## Overview

A new C++ node (`src/nodes/ball_tracker.cpp`) that replaces raw ball detections in frame metadata with a single tracked ball. Inspired by the `demo_soccer.py` ball tracking logic from `gst-python-ml`, adapted for basketball.

The node maintains a Kalman-filtered ball state, selects the best detection candidate per frame via multi-criteria scoring, gates it for physical plausibility, and coasts through detection gaps. Non-ball detections pass through unchanged.

No multi-track association — this is a single-object tracker optimized for a ball in sports video.

Replaces the old tracker nodes (`ball_tracker2`, `byte_track_ball`, `track_ball`, `track_ball_holder`) which are deleted as part of this change.

## Pipeline Position

```
cuda_infer_yolo ──> ball_tracker ──> join_metadata ──> draw_bbox
                                                   ──> basketball_analysis
                                                   ──> smooth_crop_viewport
```

Reads and **overwrites the same metadata key** (default `yolo_detections`). Ball detections are removed from the array and replaced with the single tracked ball (or nothing if no valid candidate and coasting exhausted). All other detections (players, hoops, etc.) pass through untouched.

**Important:** The upstream `cuda_infer_yolo` node's `conf_thresh` must be set <= the tracker's `min_conf` (default 0.04) to ensure the tracker receives low-confidence candidates. If `conf_thresh` is higher, the tracker will never see candidates below that threshold.

## Prerequisite Refactor: Extract Kalman1D

`smooth_crop_viewport.cpp` contains a `Kalman1D` class (~55 lines) in an anonymous namespace. Extract it to a shared header so both nodes can use it.

**New file:** `src/kalman1d.hpp`

Contents: the existing `Kalman1D` class (position + velocity state, 2x2 covariance, predict/correct cycle) with `#pragma once` guard.

**Modified file:** `src/nodes/smooth_crop_viewport.cpp`

Remove the local `Kalman1D` class, replace with `#include "../kalman1d.hpp"`. The `LowpassKalman2` wrapper stays in `smooth_crop_viewport.cpp` since it implements the `LowpassBackend` interface specific to that node.

## Node Type

Blocking `NodeSISO<av::VideoFrame, av::VideoFrame>` with `IInputReset`. Runs in its own thread. Registered as `ball_tracker`.

## Data Flow Per Frame

```
1. Parse metadata key -> split into ball_dets[] and other_dets[]
   (filter by target_label / target_class)

2. Kalman predict step (every frame, dt = 1.0)

3. select_best_ball(ball_dets)
   -> score each candidate with multi-criteria function
   -> return best candidate index (or -1)

4. gate_accept(best_candidate)
   -> 5-stage physical plausibility check
   -> accept or reject

5. If rejected: detection_override()
   -> force accept high-confidence detection after gap/streak
   -> on override: hard-reset Kalman to new position (velocity=0, high covariance)

6. If accepted (gated or override):
   -> Kalman correct step (measurement update)
   -> update last_emitted_box, trail, recent_speed
   -> reset coast_streak, det_reject_streak
   -> source = "detected" or "override"

7. If not accepted and coasting enabled:
   -> Kalman predict-only (no correction, covariance grows)
   -> validate predicted position (in bounds, distance check)
   -> update trail, recent_speed from coasted step
   -> source = "coasted", confidence decays 0.85^coast_streak
   -> stop after coast_max consecutive frames

8. If not accepted and not coasting:
   -> no ball emitted this frame

9. Rebuild metadata:
   -> other_dets[] + [tracked_ball or nothing]
   -> include trail array in metadata
   -> write back to same metadata key
```

## Component 1: Kalman Ball State

Two independent `Kalman1D` filters, one for X and one for Y (center position). Each filter maintains position, velocity, and 2x2 covariance.

**State per axis:** `[position, velocity]` with covariance `P(2x2)`

**Time step:** `dt = 1.0` (one frame). All parameters (gate sizes, speed) are in pixels-per-frame units. The upstream `force_fps` node ensures constant frame rate.

**Predict step** (every frame):
- `x += v * dt`
- Covariance grows by process noise `q_pos`, `q_vel`

**Correct step** (when detection accepted):
- Innovation = measured_center - predicted_center
- Kalman gain from covariance and measurement noise `r_meas`
- Updates position, velocity, covariance

**Predict-only** (coasting):
- Only the predict step runs. Covariance grows each frame, naturally expressing increasing uncertainty.

**Hard reset** (on detection override):
- Reinitialize both filters at the new detection's center position
- Velocity = 0, covariance = high (1e3)
- Old state is meaningless after camera cut or long gap

**Confidence during coasting:**
- Starts from the last accepted detection's confidence
- Decays by multiplying `0.85` per coast frame: `conf = last_conf * 0.85^coast_streak`
- Simple and predictable; the Kalman covariance growth handles prediction quality internally

**Default tuning (faster than crop viewport):**

| Parameter | Default | Crop viewport default | Why different |
|-----------|---------|----------------------|---------------|
| `kalman_q_pos` | `1.0` | `0.01` | Ball moves much faster than a smooth crop pan |
| `kalman_q_vel` | `0.1` | `0.001` | Ball velocity changes rapidly (bounces, passes) |
| `kalman_r_meas` | `4.0` | `4.0` | Same — detection noise is similar |

## Component 2: select_best_ball()

Scores each ball detection candidate:

```
score = w_conf * conf
      - w_dist * normalized_distance_to_prediction
      - w_size * size_penalty
      - w_round * roundness_penalty
```

**Distance to prediction:** Euclidean distance from candidate center to Kalman-predicted position, normalized by `0.5 * (model_w + model_h)`. Zero if no prior state.

**Size penalty:**
```
target = target_ball_size_rel * min(model_w, model_h)
penalty = clip(|max(bbox_w, bbox_h) - target| / target, 0, 2) * 0.5
```

Based on nba-ball training data analysis (4,022 annotations):
- Mean ball max dimension = 3.6% of min(H, W)
- P5-P95 range = 2.8% - 4.7%
- Default `target_ball_size_rel = 0.036`

**Roundness penalty:**
```
aspect_ratio = bbox_w / bbox_h
penalty = 1.0 - exp(-(aspect_ratio - 1.0)^2 / 0.15)
```

Basketball aspect ratio from training data: mean 0.98, median 1.0 (nearly perfect circle).

**Default weights:** `w_conf=1.0, w_dist=0.015, w_size=0.5, w_round=0.4`

Returns index of highest-scoring candidate, or -1 if no candidates.

## Component 3: gate_accept()

Five-stage physical plausibility check. Uses `last_emitted_box` (detected, override, or coasted) as reference, matching the demo_soccer approach.

**Stage 1 — Hard cap:**
```
if distance_to_last_trail_point > max_jump_rel * min(H, W):
    reject immediately
```
Default `max_jump_rel = 0.12` (12% of frame). Prevents teleportation.

**Stage 2 — Distance gate:**
```
base_gate = max(gate_min_px, gate_rel * min(H, W))
pass = (no trail yet) OR (distance_to_last <= base_gate)
```
Default `gate_rel = 0.06`, `gate_min_px = 12`.

**Stage 3 — Prediction gate (OR with stage 2):**
```
if gate_use_pred AND kalman_prediction exists:
    pass = distance_to_kalman_prediction <= base_gate * 1.25
```
Default `gate_use_pred = true`. Allows accepting a ball that moved along the expected trajectory even if it exceeded the raw distance gate.

**Stage 4 — IoU continuity:**
```
if last_emitted_box exists:
    pass = iou(candidate_box, last_emitted_box) >= min_iou
```
Default `min_iou = 0.20`. Uses last emitted box (detected, override, or coasted) as reference.

**Stage 5 — Speed constraint:**
```
if recent_speed > 0:
    pass = distance_to_last <= speed_mult * recent_speed
```
Default `speed_mult = 3.0`. `recent_speed` is an EMA: `0.8 * old + 0.2 * current_step`. Updated from both detected and coasted steps.

**Combined logic:**
```
accepted = NOT hard_cap_exceeded
       AND ((distance_ok AND iou_ok AND speed_ok) OR prediction_ok)
```

First frame (empty trail): all spatial gates pass automatically.

## Component 4: Detection Override

Handles camera cuts and reacquisition when gating blocks valid detections.

When gating rejects a candidate:
- Increment `det_reject_streak`
- Check override conditions:
  ```
  if conf >= override_conf AND any of:
      reject_streak >= override_after
      gap_frames >= reacquire_frames
      distance_to_last <= 2.5 * base_gate
  then:
      force accept, source = "override"
      hard-reset Kalman state to new position (velocity=0, covariance=high)
  ```

When gating accepts or no candidate exists: reset `det_reject_streak = 0`.

**Defaults:** `override_conf = 0.28`, `override_after = 2`, `reacquire_frames = 6`

**Camera cut behavior:** All detections rejected by hard cap -> reject streak grows -> after 2 frames a high-confidence detection anywhere gets force-accepted -> Kalman hard-resets to new position. Natural reacquisition in 2-6 frames.

## Component 5: Coasting

When no candidate is accepted and coasting is enabled:

```
kalman predict-only step (no measurement correction)
predicted_position = kalman.pos()

if predicted_position is in frame bounds
   AND distance_from_last_trail <= 1.25 * hard_cap:
    emit predicted position as tracked ball
    source = "coasted"
    update recent_speed from coasted step
    coast_streak++
```

Stop coasting after `coast_max` consecutive frames without a detection.

Reset `coast_streak = 0` when a detection is accepted.

**Defaults:** `coast = true`, `coast_max = 6`

## Component 6: Trail

A `deque<TrailPoint>` (maxlen `trail_max`, default 200) storing `{x, y, frame_index}` for each accepted position (detected, override, or coasted).

**Densification:** When a new point is added with a gap of 2-5 frames from the previous point, linearly interpolated points are inserted to fill the gap. This produces a smooth trail for visualization.

**Serialized in output metadata** for a future `draw_trail` node:

```json
{
  "trail": [
    [245, 180, 100],
    [248, 183, 101],
    [251, 187, 102]
  ]
}
```

Array of `[x, y, frame_index]` tuples.

## Component 7: Output Metadata

Overwrites the input metadata key. The `detections` array contains:
- All non-ball detections **unchanged** (preserving `model_index`, `engine_name`, `label`, etc.)
- The single tracked ball detection (if any), with extra tracker fields:

```json
{
  "cls": 0,
  "label": "basketball",
  "conf": 0.85,
  "xyxy": [245.3, 180.7, 278.1, 213.5],
  "model_index": 0,
  "engine_name": "ball_960x544.plan",
  "track_id": 1,
  "source": "detected",
  "predicted": false,
  "missed_frames": 0,
  "coast_streak": 0,
  "gate_status": "accepted",
  "ball_score": 0.72,
  "candidates_count": 3,
  "velocity_px_per_frame": 12.3,
  "velocity_x": 8.1,
  "velocity_y": 9.3
}
```

**`source` values:** `"detected"` (gating passed), `"override"` (detection override fired), `"coasted"` (predicted from Kalman)

**`predicted`:** `true` when source is `"coasted"`, `false` otherwise.

**Preserved fields on coasted outputs:** `model_index`, `engine_name`, `label`, `cls` are carried forward from the last detected ball. Downstream nodes that filter by these fields continue to work.

**Velocity fields:** `velocity_x` and `velocity_y` are from the Kalman filter's velocity estimate. `velocity_px_per_frame` is `sqrt(vx^2 + vy^2)`.

The top-level metadata structure (`coord_space`, `model_width`, `model_height`, `models`) is preserved from the input. The `trail` array is added at the top level alongside `detections`.

## Component 8: Destructor Statistics

On node destruction, log a summary:

```
ball_tracker: === tracking summary ===
  total frames:          1500
  detected (gated):      1180  (78.7%)
  detected (override):     45  (3.0%)
  coasted:                120  (8.0%)
  no ball:                155  (10.3%)
  dropped by gate:        210
  avg accepted ball size: 3.4% of min(H,W)
ball_tracker: confidence histogram (accepted detections):
  0.0-0.1:   5
  0.1-0.2:  12
  0.2-0.3:  34
  ...
ball_tracker: velocity stats (px/frame):
  min: 0.2  max: 85.3  avg: 12.7
```

Velocity stats use running min/max/sum/count (O(1) memory, no vector needed).

## Parameters

### Target Filtering

| Parameter | Default | Description |
|-----------|---------|-------------|
| `metadata_key` | `"yolo_detections"` | Metadata key to read and overwrite |
| `target_label` | `"basketball"` | Primary label to track |
| `target_labels` | `[]` | Additional labels to track |
| `target_class` | `-1` | Class ID to track (-1 = use labels only) |
| `min_conf` | `0.04` | Minimum confidence to consider a detection |

### Best Ball Selection

| Parameter | Default | Description |
|-----------|---------|-------------|
| `target_ball_size_rel` | `0.036` | Expected ball size as fraction of min(H,W) |
| `w_conf` | `1.0` | Scoring weight: confidence |
| `w_dist` | `0.015` | Scoring weight: distance to prediction penalty |
| `w_size` | `0.5` | Scoring weight: size deviation penalty |
| `w_round` | `0.4` | Scoring weight: non-roundness penalty |

### Gating

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_jump_rel` | `0.12` | Hard cap: max jump as fraction of min(H,W) |
| `gate_rel` | `0.06` | Distance gate as fraction of min(H,W) |
| `gate_min_px` | `12` | Minimum gate radius in pixels |
| `gate_use_pred` | `true` | Enable Kalman prediction gate as fallback |
| `min_iou` | `0.20` | IoU continuity threshold |
| `speed_mult` | `3.0` | Speed gate multiplier |

### Detection Override

| Parameter | Default | Description |
|-----------|---------|-------------|
| `override_conf` | `0.28` | Confidence threshold for force-accept |
| `override_after` | `2` | Rejection streak frames before override |
| `reacquire_frames` | `6` | Gap frames before reacquisition |

### Coasting

| Parameter | Default | Description |
|-----------|---------|-------------|
| `coast` | `true` | Enable coasting (predict through gaps) |
| `coast_max` | `6` | Maximum consecutive coast frames |

### Kalman Filter

| Parameter | Default | Description |
|-----------|---------|-------------|
| `kalman_q_pos` | `1.0` | Process noise: position |
| `kalman_q_vel` | `0.1` | Process noise: velocity |
| `kalman_r_meas` | `4.0` | Measurement noise |

### Trail

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trail_max` | `200` | Maximum trail points stored |

### Debug

| Parameter | Default | Description |
|-----------|---------|-------------|
| `debug_log_every_n` | `0` | Log every Nth frame (0 = disabled) |

## Example Pipeline Script

```
hwaccel.init { "name": "@gpu", "type": "cuda" }

node.add { "type": "input", "url": "/home/fedora/nba.mp4", "dst": "in_v", "group": "in", "name": "input", "timeout": -1, "auto_restart": "panic" }
node.add { "type": "demux", "src": "in_v", "wait_for_keyframe": false, "routing": { "?v:0": "v_pkt" }, "group": "in" }

# CUDA HW decode
node.add { "type": "dec_video", "src": "v_pkt", "dst": "v_dec_cuda", "group": "in", "name": "Video_Dec", "optional": true, "pixel_format": "?cuda", "hwaccel": "@gpu", "codec_map": { "h264": "h264_cuvid", "hevc": "hevc_cuvid" }, "hwaccel_only_for_codecs": ["h264", "hevc"] }
node.add { "type": "force_fps", "fps": "30", "group": "in", "src": "v_dec_cuda", "dst": "v_dec_30fps" }

# Split: full-res for drawing, downscaled for inference
node.add { "type": "split", "src": "v_dec_30fps", "dst": ["v_dec_1080p", "v_dec_for_yolo"], "group": "in" }

# Scale to exact 16:9 (960x540), then pad 4px to reach model input 960x544
node.add { "type": "filter_video", "graph": "scale_cuda=w=960:h=540", "src": "v_dec_for_yolo", "dst": "v_scaled_960x540", "group": "in", "name": "Cuda_Scale_Yolo", "auto_restart": "group", "dst_width": 960, "dst_height": 540, "dst_pixel_format": "cuda", "hwaccel": "@gpu" }
node.add { "type": "filter_video", "graph": "pad_cuda=w=960:h=544:x=0:y=2", "src": "v_scaled_960x540", "dst": "v_pre_yolo", "group": "in", "name": "Cuda_Pad_Yolo", "auto_restart": "group", "dst_width": 960, "dst_height": 544, "dst_pixel_format": "cuda", "hwaccel": "@gpu" }

# YOLO inference — conf_thresh must be <= ball_tracker min_conf (0.04)
node.add { "type": "cuda_infer_yolo", "src": "v_pre_yolo", "dst": "v_post_yolo", "group": "in", "name": "Yolo_Infer", "auto_restart": "group", "input_format": "RGB", "conf_thresh": 0.04, "max_det": 20, "infer_every_n": 1, "metadata_key_detection": "yolo_detections", "debug_log_metadata": true, "debug_log_every_n": 30, "mask_gpu_every_n": 0, "mask_cpu_every_n": 0, "models": [{ "engine": "/home/fedora/tensorrt/ball_960x544.plan", "task_type": "detection", "class_names": ["basketball"], "output_box_format": "end2end_xyxy" }] }

# Ball tracker — filters raw detections, emits single tracked ball
node.add { "type": "ball_tracker", "src": "v_post_yolo", "dst": "v_tracked", "group": "in", "metadata_key": "yolo_detections", "target_label": "basketball", "coast": true }

# Merge tracked detections onto 1080p branch by PTS
node.add { "type": "join_metadata", "src": ["v_dec_1080p", "v_tracked"], "dst": "v_1080p_with_md", "group": "in" }

# Draw tracked ball bbox on full-res frames
node.add { "type": "draw_bbox", "src": "v_1080p_with_md", "dst": "v_annotated_cuda", "group": "in", "name": "Draw_BBoxes", "metadata_key": "yolo_detections", "bbox_thickness": 2, "min_conf": 0.04, "allowed_labels": ["basketball"], "model_colors": { "0": "red" }, "model_content_width": 960, "model_content_height": 540, "model_content_offset_x": 0, "model_content_offset_y": 2, "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12", "debug_log_every_n": 30 }

# Encode and output
node.add { "type": "force_fps", "fps": "30/1", "group": "in", "src": "v_annotated_cuda", "dst": "v_annotated_fps" }
node.add { "type": "assume_video_format", "width": 1920, "height": 1080, "pixel_format": "cuda", "real_pixel_format": "nv12", "group": "in", "src": "v_annotated_fps", "dst": "v_preenc", "auto_restart": "panic" }
node.add { "type": "enc_video", "src": "v_preenc", "dst": "v_outenc", "group": "in", "name": "Video_Encode_NVENC", "codec": "h264_nvenc", "hwaccel": "@gpu", "options": { "b": "6000k", "maxrate": "6000k", "bufsize": "12000k", "g": 60, "bf": 0, "preset": "p3", "profile": "high" } }

node.add { "type": "mux", "src": ["v_outenc"], "dst": "mux_v", "group": "in", "ts_sort_wait": 0 }
node.add { "type": "output", "format": "mpegts", "url": "yolo_ball_tracked.ts", "src": "mux_v", "group": "in", "auto_restart": "panic" }

group.start in
```

## Files Changed

| File | Action |
|------|--------|
| `src/kalman1d.hpp` | **New** — shared Kalman1D class extracted from smooth_crop_viewport |
| `src/nodes/smooth_crop_viewport.cpp` | **Modified** — remove local Kalman1D, `#include "../kalman1d.hpp"` |
| `src/nodes/ball_tracker.cpp` | **New** — the tracker node |
| `src/nodes/ball_tracker2.cpp` | **Deleted** |
| `src/nodes/byte_track_ball.cpp` | **Deleted** |
| `src/nodes/track_ball.cpp` | **Deleted** |
| `src/nodes/track_ball_holder.cpp` | **Deleted** |

## Design Decisions

**Why no multi-track association:**
For a single ball, ByteTrack-style multi-hypothesis tracking is overkill. The ball-specific scoring (size, roundness, prediction distance) is more discriminative than IoU-based association. The gating + override logic handles the same scenarios (occlusion, reacquisition) more simply.

**Why Kalman over raw velocity:**
Raw frame-to-frame velocity is noisy. The Kalman filter smooths velocity estimates and provides principled uncertainty growth during coasting, replacing ad-hoc `velocity_smoothing` and `coast_decay` parameters. The existing `Kalman1D` class from `smooth_crop_viewport.cpp` is extracted to a shared header to avoid duplication.

**Why dt = 1.0 (per-frame) instead of real timestamps:**
All parameters (gate sizes, speed multiplier) are in pixels-per-frame units, matching how the soccer demo computes velocity. The upstream `force_fps` node already ensures constant frame rate.

**Why overwrite the same metadata key:**
Downstream nodes (`draw_bbox`, `basketball_analysis`, `smooth_crop_viewport`) already read from `yolo_detections`. Writing to the same key means zero pipeline reconfiguration — the tracker is a drop-in filter.

**Why hard-reset Kalman on detection override:**
Detection override fires after camera cuts or long gaps. The old Kalman state (velocity, covariance) is meaningless at the new position. A hard reset gives clean behavior immediately after reacquisition.

**Why use last_emitted_box for IoU gating (not last_detected_box):**
Matches the demo_soccer approach. During coasting, the predicted position is our best estimate. Comparing the returning detection against that estimate is more accurate than comparing against a stale detected position from several frames ago.

**Why delete old tracker nodes:**
`ball_tracker2`, `byte_track_ball`, `track_ball`, and `track_ball_holder` are unused in any example pipeline and did not work well in practice. The new `ball_tracker` replaces all of them with a single, better-designed node.

**Why 3.6% target ball size:**
Empirically measured from 4,022 nba-ball training annotations. Mean max dimension = 3.6% of min(H,W) at 1080p (median 39px width, 22px height, aspect ratio ~1.0).
