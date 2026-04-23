# Cropped Tracker With Player-Foot Follow Fallback

## Goal

Add a new cropped tracker example that keeps the existing ball tracker behavior, but gives `smooth_crop_viewport` another detection source to follow when ball and handler signals are weak.

The new source is the existing TensorRT model:

- engine: `/home/fedora/tensorrt/player-ball.plan`
- classes: `["foot", "player", "ball"]`

This should live in a new example file so the current cropped baseline stays unchanged.

## Scope

In scope:

- add a new `.avplumber` example derived from the current cropped tracker example
- add a dedicated split and `cuda_infer_yolo` branch for `player-ball.plan`
- feed that metadata into `smooth_crop_viewport`
- expand crop-follow label selection to include `foot` and `player`

Out of scope:

- changing `ball_tracker`
- changing the existing cropped example
- using the new model as a tracker source
- adding new drawing overlays unless needed later for debugging

## Recommended Approach

Use a separate inference node and a separate metadata key.

Why:

- isolates the new model cleanly from existing player, ball, and segmentation branches
- keeps debugging simple because the reframer input source is explicit
- avoids changing tracker semantics

## Example Shape

Create a new example file alongside the current cropped tracker example, based on:

- `examples/yolo/yolo_infer_all_players_tracker_cropped.avplumber`

Recommended new file name:

- `examples/yolo/yolo_infer_all_players_tracker_cropped_player_foot_follow.avplumber`

## Graph Changes

Add a new split from the pre-yolo path and a new inference node:

1. split `v_pre_yolo` into a new branch for the player-foot model
2. run `cuda_infer_yolo` with:
   - engine `/home/fedora/tensorrt/player-ball.plan`
   - classes `["foot", "player", "ball"]`
   - a dedicated metadata key such as `yolo_player_foot`
3. join that metadata into the 1080p metadata stream before `smooth_crop_viewport`

The branch should be independent of the dedicated ball tracker branch.

## Reframer Selection

The new example should configure `smooth_crop_viewport` with:

- `metadata_key_ins`: include the new metadata key in addition to existing inputs
- `allowed_labels`: `["basketball", "BallHandler", "foot", "player", "Player"]`
- `focus_mode`: `"label_priority"`
- `label_priority`: `["basketball", "BallHandler", "foot", "player", "Player"]`

Interpretation:

1. follow tracked ball first
2. then ball handler
3. then foot from the new model
4. then lower-case `player` from the new model
5. then upper-case `Player` from the existing full-scene model

## Error Handling

- If the new model branch fails, it should behave like the other infer nodes in the example and stay scoped to the same group restart behavior.
- If the new metadata stream is empty on a frame, `smooth_crop_viewport` should continue using the existing fallback order.

## Verification

Verify by:

- loading the new example on the remote host
- confirming the graph starts with the additional infer branch
- confirming `smooth_crop_viewport` reads the new metadata key
- rerunning against `nba-full.mp4` or `nba.mp4` and checking that output is produced

## Self-Review

- no placeholders remain
- scope is limited to a new example and reframer inputs
- tracker behavior stays unchanged
