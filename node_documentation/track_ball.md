# `track_ball`

`track_ball` is a lightweight metadata-only tracker for basketball workflows. It reads YOLO-style detection metadata from each incoming frame, keeps one active target track, and can emit predicted boxes through short detection gaps.

## What It Does

1. Reads `av::VideoFrame` input from `src` and forwards the same frame to `dst`.
2. Parses YOLO-compatible metadata from `metadata_key_in`.
3. Filters detections to the configured ball target by `label` and/or `cls`.
4. When no track is active, only acquires a new ball from detections that show frame-to-frame motion.
5. Maintains a rolling history of accepted ball positions and derives recent motion statistics from that history.
6. Matches the best current ball detection against the active track using predicted position plus motion-vector sanity checks.
7. Rejects candidates that are too static, jump too far, or disagree strongly with the recent motion history.
8. When the detector misses briefly, optionally predicts the next box for a configurable miss window.
9. Writes YOLO-compatible tracked metadata to `metadata_key_out`.

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

The node preserves the top-level coordinate metadata so downstream consumers such as `draw_bbox` can continue scaling model-space coordinates back onto the target frame.

## Output Metadata

The output remains YOLO-compatible:

- `version`
- `coord_space`
- `model_width`
- `model_height`
- `detections`

When a tracked ball is available, the first detection includes:

- `cls`
- optional `label`
- `conf`
- `xyxy`
- `track_id`
- `predicted`
- `missed_frames`
- `source`

The node also writes a top-level `tracker` object for debugging.

## Tracking Behavior

This first version keeps only one active track:

- if no track is active, it starts one only after seeing a ball candidate move across consecutive frames
- if a track is active, it tries to match the best current ball detection using predicted position plus motion-vector sanity checks
- if no suitable detection matches, it can emit a predicted box for up to `max_missed_frames`
- once the miss window expires, the track is dropped

This is intended to bridge short gaps and avoid latching onto static false positives such as crowd objects, reflections, or far-away distractors, not long occlusions.

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
  Metadata key to write tracked output to. Default: `ball_track_v1`.

- `target_label`
  Detection label to track. Default: `sports ball`.

- `target_labels`
  Optional array of detection labels to track.

- `target_class`
  Optional class ID to track. Default: disabled.

- `target_classes`
  Optional array of class IDs to track.

- `min_conf`
  Minimum confidence required for incoming detections. Default: `0.10`.

- `max_missed_frames`
  Maximum number of consecutive missed frames for which predicted boxes may be emitted. Default: `8`.

- `max_center_distance`
  Maximum center distance in metadata pixel space for detection-to-track matching unless IoU also passes. Default: `160`.

- `min_iou_match`
  Minimum IoU allowed for matching. Default: `0.0`.

- `match_min_motion`
  Minimum per-frame center motion in metadata pixel space required for an active-track detection match. Detections that are effectively static relative to the previous tracked ball are ignored. Default: `2.0`.

- `match_max_motion`
  Maximum per-frame center motion in metadata pixel space allowed for an active-track detection match. Detections that jump too far from the previous tracked ball are ignored. Default: `64.0`.

- `match_min_cosine_similarity`
  Minimum cosine similarity between the current track velocity and a candidate detection motion vector. Higher values require a more consistent direction of travel. Default: `-0.2`.

- `match_max_prediction_error`
  Maximum center-distance error allowed between the predicted ball position for the current frame and a candidate detection. Detections that land too far from the predicted path are ignored. Default: `28.0`.

- `match_max_velocity_delta`
  Maximum allowed change in per-frame velocity between the active track and a candidate detection. Detections that imply an abrupt velocity jump are ignored. Default: `24.0`.

- `slow_match_max_prediction_error`
  Maximum center-distance error allowed for a slow or nearly static candidate to still match the active track, as long as it remains close to the predicted path. Default: `12.0`.

- `slow_match_min_overlap`
  Minimum IoU overlap that allows a slow or nearly static candidate to match the active track when it is close to the predicted path. Default: `0.10`.

- `huge_jump_frame_fraction`
  Fraction of the frame span treated as a huge jump when examining recent track history. Slow/static candidates are rejected if the recent history contains a jump at or above this threshold. Default: `0.25`.

- `huge_jump_history_window`
  Number of most-recent track-history steps checked for a huge jump before allowing a slow/static candidate. Default: `8`.

- `history_size`
  Maximum number of accepted tracked positions to retain for motion modeling. Default: `30`.

- `history_motion_window`
  Number of most-recent history steps used to compute average motion statistics for candidate filtering. Default: `12`.

- `history_match_min_cosine_similarity`
  Minimum cosine similarity between the recent history motion vector and a candidate detection motion vector. Higher values require a more consistent direction of travel relative to the longer history. Default: `-0.1`.

- `history_max_motion_scale`
  Multiplier applied to the recent history max per-frame speed when deriving the allowed motion ceiling for active matches. Default: `2.5`.

- `history_max_motion_slack`
  Extra per-frame motion slack added on top of the history-derived motion ceiling for active matches. Default: `16.0`.

- `acquisition_min_motion`
  Minimum center motion in metadata pixel space between consecutive frames before a new idle track may be acquired. Default: `4.0`.

- `acquisition_max_match_distance`
  Maximum center distance used when comparing current detections to the previous frame during idle-track acquisition. Default: `120`.

- `acquisition_min_cosine_similarity`
  Minimum cosine similarity between two consecutive idle-track motion vectors before a new track may be acquired. Higher values require a more consistent direction of travel. Default: `0.2`.

- `emit_predicted`
  If `true`, emit predicted boxes during short misses. Default: `true`.

- `prediction_decay`
  Velocity/confidence decay applied while predicting through misses. Default: `0.92`.

- `velocity_smoothing`
  Blend factor for reusing previous velocity when a new detection arrives. Default: `0.60`.

- `debug_log_every_n`
  Logging interval for tracker status. Default: `0`.

## Runtime Notes

- The node is algorithmic only. It does not require an ONNX file or TensorRT engine.
- The node forwards the original frame and only mutates metadata.
- The node implements `IInputReset` so track state clears on upstream resets.
- The tracker is ball-focused and intentionally does not keep multiple simultaneous tracks.
- Idle-track acquisition prefers moving detections and ignores nearly static candidates, which helps suppress persistent far-away false positives.
- Idle-track acquisition also requires a short, roughly consistent motion vector across consecutive frames before a new target is accepted.
- Active-track matching uses a rolling history of accepted ball positions plus the predicted current-frame position to reject candidates that jump too far in one frame, land too far from the predicted path, or move in a direction that is strongly inconsistent with the recent ball trajectory.
- Slow or nearly static candidates can still match when they stay close to the predicted path, but only if the recent history did not include a huge jump across the frame.
- If multiple target filters are configured, a detection is accepted when it matches any configured label or class.

## Example

```txt
track_ball:
  src: v_post_yolo
  dst: v_tracked_yolo
  metadata_key_in: yolo_detections_v1
  metadata_key_out: ball_track_v1
  target_labels: [ball, foot]
  min_conf: 0.10
  match_min_motion: 3.0
  match_max_motion: 40.0
  match_min_cosine_similarity: 0.0
  match_max_prediction_error: 22.0
  match_max_velocity_delta: 18.0
  slow_match_max_prediction_error: 10.0
  slow_match_min_overlap: 0.08
  huge_jump_frame_fraction: 0.25
  huge_jump_history_window: 8
  history_size: 30
  history_motion_window: 12
  history_match_min_cosine_similarity: 0.1
  history_max_motion_scale: 1.8
  history_max_motion_slack: 10.0
  acquisition_min_motion: 4.0
  acquisition_max_match_distance: 120
  acquisition_min_cosine_similarity: 0.4
  max_missed_frames: 8
  max_center_distance: 160
  prediction_decay: 0.92
  emit_predicted: true
```
