# Ball Tracker Coast Edge-Jump Veto

## Goal

Prevent abrupt ball-tracker reacquires when the tracker is already uncertain and a new detection would snap the ball close to the far left or far right side of the full model frame.

This is intended to reduce visually bad reframer shifts caused by suspicious edge detections while the tracker is coasting or stale.

## Scope

In scope:

- add a narrow guard inside `ball_tracker`
- evaluate full-frame `cx` only
- apply only when the tracker is already coasting or stale
- prefer continued coasting over immediate acceptance of suspicious edge detections

Out of scope:

- changing normal confident tracking
- changing `smooth_crop_viewport`
- changing viewport-space logic
- using segmentation for this rule

## Recommended Approach

Apply the rule in the override / stale reacquire path, with the current coasting state as an additional trigger.

Why:

- keeps the existing normal gate behavior untouched
- targets the exact failure mode: uncertain tracker plus abrupt edge snap
- avoids suppressing valid detections during stable tracking

## Rule

For a candidate detection:

1. only consider the rule if the tracker is stale or currently coasting
2. compute jump distance from the last accepted or coasted position
3. compute whether candidate `cx` lies inside a left/right edge zone of the full model frame
4. if both are true, treat the detection as suspicious and do not immediately accept it
5. continue coasting and wait for better detections

The suspicious candidate should also be prevented from immediately winning the stale override path. It must persist for a small number of frames before override can use it.

## Coordinates

The edge test uses the candidate center `cx` in full model coordinates, not viewport coordinates.

If `model_w = 960`, an edge-zone test works directly on that width.

## Proposed Parameters

- `coast_edge_jump_veto_enabled: true`
- `coast_edge_zone_rel: 0.14`
- `coast_edge_jump_rel: 0.18`
- `coast_edge_confirm_frames: 9`

Interpretation:

- edge zone is the leftmost/rightmost `14%` of the model width
- jump must be at least `18%` of the model minimum dimension
- suspicious edge candidates must persist for `9` frames before override can use them

## Behavior

Stable tracking:

- unchanged

Coasting or stale tracking:

- large jump to near-edge candidate is ignored initially
- tracker keeps coasting
- if the same near-edge candidate keeps appearing consistently for enough frames, override may eventually accept it

## Error Handling

- If model dimensions are missing or invalid, the rule fails open and tracking falls back to current behavior.
- If there is no last accepted/coasted position, the rule does not apply.

## Debugging

Add per-frame debug fields when this logic is evaluated:

- candidate `cx`
- jump distance
- `edge_zone_hit`
- `edge_jump_veto`
- confirmation hit count if a suspicious candidate is being accumulated

## Verification

Verify by rerunning the affected tracker example and checking:

- suspicious edge snaps while coasting are no longer accepted immediately
- normal on-court, non-edge reacquires still work
- logs show the veto only in stale/coasted situations

## Self-Review

- no placeholders remain
- scope is limited to a single tracker rule
- the rule explicitly uses full-frame `cx`
- stable tracking remains unchanged
