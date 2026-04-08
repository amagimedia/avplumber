# `track_ball_holder`

`track_ball_holder` is a lightweight metadata-only tracker that pairs `sports ball` and `person` detections, picks the most likely ball-handler when one exists, and falls back to ball-only tracking when the ball is in flight.

## What It Does

1. Reads `av::VideoFrame` input from `src` and forwards the same frame to `dst`.
2. Parses YOLO-compatible metadata from `metadata_key_in`.
3. Splits detections into `sports ball` and `person` candidates.
4. Scores candidate person-ball pairs using a hand-side / upper-half heuristic.
5. Builds one combined union box for the best pair.
6. Falls back to the standalone ball when no good holder pair exists.
7. Tracks the active target over time with short-gap velocity prediction.
8. Writes YOLO-compatible tracked metadata to `metadata_key_out`.

## Input Metadata

The node expects YOLO-style metadata with:

- `version`
- `coord_space`
- `model_width`
- `model_height`
- `detections`

Each detection should contain:

- `cls`
- optional `label`
- `conf`
- `xyxy`

## Output Metadata

The output remains YOLO-compatible so `draw_bbox` can consume it directly.

When a tracked target is available, the first detection includes:

- `label` default `ball_holder` in holder mode, or `sports ball` in ball-only mode
- `conf`
- `xyxy`
- `track_id`
- `predicted`
- `missed_frames`
- `source`
- `tracking_mode`
- `pair_score`
- `hand_side`
- `person_xyxy`
- `ball_xyxy`
- `person_conf`
- `ball_conf`

The node also writes a top-level `tracker` object for debugging.

## Pairing Heuristic

For each candidate `person` / `sports ball` pair, the node:

- computes the ball center
- computes proxy left/right hand positions near the upper half of the person box
- rejects pairs that are too far from both proxy hand positions
- requires the ball to sit in a configurable upper vertical band of the person box
- gives a bonus when the ball center is inside an expanded person box
- adds a temporal preference for pairs near the previously tracked combined target

The highest-scoring valid pair becomes the current `ball_holder`.

If no valid holder pair exists but the ball is still detected, the node switches to ball-only mode and tracks the ball box directly.

## Tracking Behavior

This first version keeps one active target:

- if no track is active, it starts one from the best current person-ball pair
- if no good pair exists but the ball is visible, it starts or updates a ball-only track
- if a track is active, it prefers a holder pair when one is available and otherwise follows the standalone ball
- if no suitable target matches, it can emit a predicted box for a short miss window
- once the miss window expires, the track is dropped

This is intended to bridge short misses, not long occlusions or full re-identification.

## Parameters

### Required

- `src`
  Input video edge.

- `dst`
  Output video edge.

### Optional

- `metadata_key_in`
  Metadata key to read YOLO detections from. Default: `yolo_detections_v1`.

- `metadata_key_out`
  Metadata key to write tracked output to. Default: `ball_holder_track_v1`.

- `ball_label`
  Detection label to treat as the ball. Default: `sports ball`.

- `person_label`
  Detection label to treat as the handler candidate. Default: `person`.

- `output_label`
  Label written for the combined target. Default: `ball_holder`.

- `ball_class`
  Optional class ID to use for ball matching. Default: disabled.

- `person_class`
  Optional class ID to use for person matching. Default: disabled.

- `min_ball_conf`
  Minimum confidence required for ball detections. Default: `0.10`.

- `min_person_conf`
  Minimum confidence required for person detections. Default: `0.25`.

- `max_hand_distance_px`
  Maximum distance from the ball center to the nearest proxy hand anchor. Default: `180`.

- `hand_y_ratio`
  Vertical position of the proxy hand anchors within the person box. Default: `0.35`.

- `side_outset_ratio`
  Horizontal offset beyond the left/right edge used for proxy hand anchors. Default: `0.05`.

- `upper_y_min_ratio`
  Minimum allowed normalized vertical ball position in the person box. Default: `0.05`.

- `upper_y_max_ratio`
  Maximum allowed normalized vertical ball position in the person box. Default: `0.72`.

- `expanded_person_margin_x_ratio`
  Horizontal expansion used when checking whether the ball is near the person body. Default: `0.10`.

- `expanded_person_margin_y_ratio`
  Vertical expansion used when checking whether the ball is near the person body. Default: `0.05`.

- `combined_padding_x_ratio`
  Horizontal padding added to the combined union box. Default: `0.04`.

- `combined_padding_y_ratio`
  Vertical padding added to the combined union box. Default: `0.04`.

- `max_missed_frames`
  Maximum number of consecutive missed frames for which predicted boxes may be emitted. Default: `4`.

- `max_center_distance`
  Maximum center distance in metadata pixel space for matching a new combined pair to the tracked target unless IoU also passes. Default: `220`.

- `min_iou_match`
  Minimum IoU allowed for matching. Default: `0.0`.

- `emit_predicted`
  If `true`, emit predicted combined boxes during short misses. Default: `true`.

- `prediction_decay`
  Velocity/confidence decay applied while predicting through misses. Default: `0.85`.

- `velocity_smoothing`
  Blend factor for reusing previous velocity when a new pair arrives. Default: `0.60`.

- `debug_log_every_n`
  Logging interval for tracker status. Default: `0`.

## Runtime Notes

- The node is algorithmic only. It does not require an ONNX file or TensorRT engine.
- The node forwards the original frame and only mutates metadata.
- The node implements `IInputReset` so track state clears on upstream resets.
- The output is a single active target. In holder mode that target is the combined person+ball box. In fallback mode it is the ball box itself.

## Example

```txt
track_ball_holder:
  src: v_post_yolo
  dst: v_tracked_yolo
  metadata_key_in: yolo_detections_v1
  metadata_key_out: ball_holder_track_v1
  ball_label: sports ball
  person_label: person
  output_label: ball_holder
  min_ball_conf: 0.10
  min_person_conf: 0.25
  max_hand_distance_px: 180
  emit_predicted: true
```
