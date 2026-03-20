# `ball_tracker2`

`ball_tracker2` is a metadata-only single-ball tracker for basketball workflows. It reads YOLO-style detection metadata, keeps a long rolling history of candidate trajectories, scores them for continuity, and emits exactly one YOLO-compatible ball box.

## What It Does

1. Reads `av::VideoFrame` input from `src` and forwards the same frame to `dst`.
2. Parses YOLO-compatible detection metadata from `metadata_key_in`.
3. Filters detections to the configured ball target by `label` and/or `cls`.
4. Maintains multiple internal trajectory hypotheses over a long history window.
5. Scores candidate assignments using predicted position, continuity, acceleration, jerk, overlap, and confidence.
6. Rejects large single-frame discontinuities relative to frame size.
7. Allows short slow/static phases when they still align with the predicted path.
8. Predicts through short detector gaps when needed.
9. Emits only the single best current track under `metadata_key_out`.

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

Coordinates are expected in model space, matching `cuda_infer_yolo`.

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
- `hits`
- `age`

The node also writes a top-level `tracker` object for debugging.

## Tracking Behavior

`ball_tracker2` is designed for cases where the true ball is often detected, but reflections or crowd distractors produce plausible false positives.

- it keeps several internal trajectory hypotheses rather than following a single greedy chain
- it scores hypotheses over a longer time window instead of trusting only the previous frame
- it rejects assignments that imply a very large jump in one frame
- it penalizes abrupt acceleration and jerk spikes
- it can keep a slow or briefly static ball if it still aligns with the predicted trajectory
- it applies output-selection hysteresis so the final box does not bounce back and forth between far-away hypotheses
- it emits only one selected track downstream even though several hypotheses may exist internally

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
  Metadata key to write tracked output to. Default: `ball_track_v2`.

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

- `history_size`
  Maximum number of recent samples retained per hypothesis. Default: `120`.

- `history_motion_window`
  Number of recent samples used when computing motion statistics. Default: `12`.

- `max_missed_frames`
  Maximum number of consecutive predicted frames before a hypothesis is dropped. Default: `8`.

- `max_jump_frame_fraction`
  Maximum allowed single-frame jump as a fraction of the larger frame dimension. Default: `0.25`.

- `match_max_center_distance`
  Maximum center-distance error allowed between the predicted path and a candidate detection. Default: `140`.

- `min_iou_match`
  Minimum IoU allowed to rescue a candidate when center-distance is large. Default: `0.01`.

- `max_acceleration`
  Maximum allowed per-frame velocity change for a matched detection. Default: `28.0`.

- `max_jerk`
  Maximum allowed jerk spike relative to recent history. Default: `28.0`.

- `slow_mode_max_prediction_error`
  Maximum prediction error allowed for a slow/static candidate to remain on-track. Default: `12.0`.

- `min_track_quality_margin`
  Minimum score margin required to switch away from the currently selected track when two hypotheses are close. Default: `2.0`.

- `max_output_jump_frame_fraction`
  Maximum allowed jump for switching the emitted output to a different hypothesis, expressed as a fraction of the larger frame dimension. Default: `0.12`.

- `output_switch_margin`
  Minimum score advantage required before the tracker will switch the emitted output to a distant competing hypothesis. Default: `8.0`.

- `min_switch_hits`
  Minimum hit count required before a competing hypothesis may steal the emitted output after a large spatial jump. Default: `4`.

- `prediction_decay`
  Velocity/confidence decay applied while predicting through misses. Default: `0.92`.

- `velocity_smoothing`
  Blend factor for reusing previous velocity when a new detection arrives. Default: `0.60`.

- `min_confirmed_hits`
  Minimum number of matched detections required before a hypothesis may be emitted. Default: `2`.

- `min_track_age`
  Minimum age required before a hypothesis may be emitted. Default: `2`.

- `max_hypotheses`
  Maximum number of internal hypotheses retained after pruning. Default: `6`.

- `debug_log_every_n`
  Logging interval for tracker status. Default: `0`.

## Runtime Notes

- The node is algorithmic only. It does not require an ONNX file or TensorRT engine.
- The node forwards the original frame and only mutates metadata.
- The node implements `IInputReset` so tracker state clears on upstream resets.
- Downstream nodes such as `draw_bbox` can consume the output directly because the metadata remains YOLO-compatible.
- The internal algorithm is beam-search-like: it expands and scores several hypotheses each frame, then keeps only the best few.

## Example

```txt
ball_tracker2:
  src: v_post_yolo
  dst: v_tracked_yolo
  metadata_key_in: yolo_detections_v1
  metadata_key_out: ball_track_v2
  target_label: basketball
  min_conf: 0.20
  history_size: 120
  history_motion_window: 12
  max_missed_frames: 8
  max_jump_frame_fraction: 0.25
  match_max_center_distance: 140
  min_iou_match: 0.01
  max_acceleration: 28.0
  max_jerk: 28.0
  slow_mode_max_prediction_error: 12.0
  min_track_quality_margin: 2.0
  max_output_jump_frame_fraction: 0.12
  output_switch_margin: 8.0
  min_switch_hits: 4
  prediction_decay: 0.92
  velocity_smoothing: 0.60
  min_confirmed_hits: 2
  min_track_age: 2
  max_hypotheses: 6
```
