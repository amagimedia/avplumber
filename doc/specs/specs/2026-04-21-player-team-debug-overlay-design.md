# Player Team Debug Overlay

## Goal

Add always-on debug overlays to the main player-segmentation example so team flips can be diagnosed visually during live RTMP runs without changing the underlying tracking or team-classification semantics.

The overlay must make it easy to answer two questions:
- which tracked player is which over time
- when a given track flips between team assignments

## Current State

The main example already renders:
- court segmentation masks
- team-colored player segmentation masks
- ball trail and handler overlays

What is missing for debugging:
- a stable visual anchor for each tracked player
- a human-readable per-track ID
- a direct text rendering of current team assignment on the same track

The current `draw_bbox_labels` node can render:
- `{track_id}`
- `{conf}`
- `{velocity}`
- `{label}`
- `{cls}`

It cannot render the `team_classifier` output field directly.

## Constraints

1. Keep the main example graph as the source of truth. Do not create a separate debug-only variant.
2. Keep player segmentation masks as the primary visual layer.
3. Keep generic player debug boxes subtle.
4. Do not overload `label` to mean team assignment.
5. Team display must come from the explicit team-classification output field.
6. Keep the existing thicker magenta ball-handler bbox unchanged.

## Recommended Design

Add two always-on debug layers for tracked players in the main example:
- a subtle neutral `draw_bbox` overlay for all tracked players
- a `draw_bbox_labels` overlay showing track ID and team assignment

Add one new render-only token to `draw_bbox_labels`:
- `{team_ab}`

This token reads the existing `team` metadata field and formats it for quick human inspection:
- `0 -> A`
- `1 -> B`
- missing or negative team -> `?`

This keeps semantics clean:
- `track_id` remains identity
- `team` remains team assignment
- `label` remains the detector/class label

## Overlay Behavior

### Player bbox overlay

Add a neutral player bbox overlay using `draw_bbox` on `yolo_players`.

Parameters:
- `bbox_thickness: 1`
- neutral color
- only player detections

Purpose:
- make track motion and tracker drift visible even when masks look correct
- provide a stable visual anchor for the text label

### Player label overlay

Add a `draw_bbox_labels` node on `yolo_players`.

Label template:
```text
ID:{track_id}
T:{team_ab}
```

Expected rendering:
- `ID:17`
- `T:A`

If `team` is unavailable:
- `T:?`

### Ball-handler overlay

Keep the existing ball-handler overlay unchanged:
- same metadata source
- same thicker magenta bbox

The handler overlay remains visually distinct from the generic player debug bboxes.

## Code Changes

### Parsed detection metadata

Extend parsed overlay detection support so the draw pipeline can read optional `team` from detection metadata.

This must be additive only:
- if `team` is absent, existing behavior stays unchanged
- existing templates continue to work

### `draw_bbox_labels` token support

Add token support in `draw_bbox_labels`:
- token name: `team_ab`
- source field: `team`
- formatter:
  - `0 -> A`
  - `1 -> B`
  - otherwise `?`

No other template tokens need to change.

## Graph Changes

Update the main example graph:
- keep `Draw_Player_Teams`
- insert a subtle player `draw_bbox` overlay after it
- insert `draw_bbox_labels` after the player bbox overlay
- keep downstream ball, handler, viewport, and output stages intact

The debug overlays must use the tracked player metadata stream so labels stay attached to track IDs rather than raw per-frame detections.

## Files

Code:
- `src/nodes/neural_net/draw/cuda_overlay_base.hpp`
- `src/nodes/neural_net/draw/cuda_overlay_base.cpp`
- `src/nodes/neural_net/draw/draw_bbox_labels.cpp`

Graph:
- `examples/yolo/yolo_infer_all_players_tracker_pose_live_teams.avplumber`

## Failure Handling

If a player detection has no `track_id`:
- suppress label for that player unless existing node settings explicitly allow untracked labels

If a player detection has no `team`:
- render `T:?`

No failure in the debug overlay path should interfere with player segmentation-mask rendering.

## Validation

1. Local graph parse/build.
2. Remote rebuild on the Fedora host.
3. Run the RTMP example.
4. Verify that:
- player masks remain pixel-perfect
- generic player boxes are thin and unobtrusive
- labels show stable `ID:<n>` values
- second line shows `T:A`, `T:B`, or `T:?`
- team flips are directly visible on the same track ID over time
- ball-handler bbox remains the thicker magenta overlay

## Scope

In scope:
- `team_ab` label token
- subtle always-on player bbox overlay
- always-on player ID/team labels in the main example

Out of scope:
- changing team-classifier assignment logic
- changing tracker behavior
- adding true bbox opacity support
- mutating player `label` to carry team information
