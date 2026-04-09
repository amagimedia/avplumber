# `byte_track_ball`

`byte_track_ball` is a metadata-only, ByteTrack-style tracker for basketball workflows. It reads YOLO-style detection metadata, keeps multiple internal candidate ball tracks, matches high-confidence detections first, uses lower-confidence detections to recover existing tracks, and emits only the single best current ball track.

## What It Does

1. Reads `av::VideoFrame` input from `src` and forwards the same frame to `dst`.
2. Parses YOLO-compatible detection metadata from `metadata_key_in`.
3. Filters detections to the configured ball target by `label` and/or `cls`.
4. Splits detections into high-confidence and low-confidence sets.
5. Predicts each active track forward using a simple velocity model.
6. Matches confirmed tracks to high-confidence detections first.
7. Tries to recover unmatched tracks using lower-confidence detections.
8. Starts new tracks from unmatched strong detections.
9. Emits only the best current track as YOLO-compatible metadata under `metadata_key_out`.

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
- `confirmed`
- `hits`

The node also writes a top-level `tracker` object for debugging.

## Tracking Behavior

This node is ByteTrack-style rather than detector-only:

- high-confidence detections are used first to update existing tracks
- lower-confidence detections may still recover unmatched tracks
- tracks can survive short detector dropouts when `emit_predicted` is enabled
- several internal tracks may exist at once, but only the best current track is emitted downstream

This is useful when YOLO sometimes produces multiple basketball candidates in the same frame, but you only want one stable output box.

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

- `high_conf_thresh`
  Threshold separating high-confidence detections from low-confidence recovery detections. Default: `0.35`.

- `new_track_conf_thresh`
  Minimum confidence required to start a brand-new track from an unmatched detection. Default: `0.35`.

- `match_iou_thresh`
  Minimum IoU-like association threshold for the first matching pass. Default: `0.05`.

- `low_match_iou_thresh`
  Minimum IoU-like association threshold for the low-confidence recovery pass. Default: `0.01`.

- `match_max_center_distance`
  Maximum center distance used in the first matching pass. Default: `120`.

- `low_match_max_center_distance`
  Maximum center distance used in the low-confidence recovery pass. Default: `160`.

- `max_time_lost`
  Maximum number of consecutive missed frames before a track is dropped. Default: `8`.

- `min_confirmed_hits`
  Number of successful matches required before a track is treated as confirmed output. Default: `2`.

- `max_tracks`
  Maximum number of internal tracks kept after pruning. Default: `8`.

- `emit_predicted`
  If `true`, emit predicted positions for short detection gaps. Default: `true`.

- `prediction_decay`
  Velocity/confidence decay applied while predicting through misses. Default: `0.92`.

- `velocity_smoothing`
  Blend factor for reusing previous velocity when a new detection arrives. Default: `0.60`.

- `debug_log_every_n`
  Logging interval for tracker status. Default: `0`.

## Runtime Notes

- The node is algorithmic only. It does not require an ONNX file or TensorRT engine.
- The node forwards the original frame and only mutates metadata.
- The node implements `IInputReset` so tracker state clears on upstream resets.
- Internally the node may maintain several ball tracks, but downstream consumers see only one best track.
- This is a ByteTrack-style implementation tailored for single-ball basketball pipelines rather than a full generic MOT implementation.

## Example

```txt
byte_track_ball:
  src: v_post_yolo
  dst: v_tracked_yolo
  metadata_key_in: yolo_detections_v1
  metadata_key_out: ball_track_v1
  target_label: basketball
  min_conf: 0.20
  high_conf_thresh: 0.35
  new_track_conf_thresh: 0.35
  match_iou_thresh: 0.05
  low_match_iou_thresh: 0.01
  match_max_center_distance: 120
  low_match_max_center_distance: 160
  max_time_lost: 8
  min_confirmed_hits: 2
  emit_predicted: true
  prediction_decay: 0.92
```
