# Ball Tracker Court Bounds Veto Design

## Goal

Improve the existing CPU-mask off-court veto used by `ball_tracker` so that stale ball reacquires can be rejected when the candidate ball box is horizontally outside the detected court bounds.

This change is intended to fix cases like the `nba.mp4` frame-`1581` false reacquire, where:

- the raw ball detector confidence is strong
- the candidate is repeatedly seen near the far right side
- the candidate is slightly off court
- the current center-sample CPU veto is too weak to reject it

The new check must:

- use the existing CPU segmentation mask side data
- ignore top/bottom court separation
- only reason about left/right court bounds
- apply to reacquires for now
- be enabled by default
- keep geometry code out of `ball_tracker.cpp`

## Non-Goals

- no new CUDA shader or GPU-only bound check
- no increase in segmentation mask resolution
- no filtering of normal non-reacquire detections
- no attempt to solve all false positives with confidence thresholding

## Current State

`ball_tracker` already has:

- CPU court-mask parsing from `AV_FRAME_DATA_YOLO_SEG_MASKS`
- an `override_off_court_veto_enabled` path
- a center-support based veto used only for provisional far reacquires / overrides

Problems with the current approach:

- it samples around the ball center, which is not precise enough when the ball is only slightly off court
- it does not model the actual horizontal overlap between the ball bbox and the court mask
- most graph variants do not enable it
- the logic lives in `ball_tracker.cpp`, which is already too large

## Proposed Approach

Replace the current center-support off-court veto with a horizontal bbox overlap check against the CPU court mask.

For a reacquire candidate:

1. Sample several rows across the candidate bbox vertical span.
2. For each sampled row, find the leftmost and rightmost on-court mask pixels.
3. Treat that row as the valid horizontal court interval.
4. Compute how much of the candidate bbox width overlaps that interval.
5. Aggregate the sampled rows into a single veto decision.

Top/bottom separation is ignored by design. A ball can be vertically above the court and still be valid. Only left/right exclusion matters.

## Architecture

### Code Placement

Move the geometry and court-bound logic into `src/nodes/neural_net/sport_specific/ball_tracker_utils.hpp`.

Keep in `ball_tracker.cpp`:

- state management
- frame parsing
- reacquire / override state machine
- logging orchestration
- parameter wiring

Move to `ball_tracker_utils.hpp`:

- court mask row scanning
- court interval extraction
- horizontal overlap math
- structured veto decision result

This keeps `ball_tracker.cpp` focused on tracking policy rather than mask geometry.

### Utility Types

Add utility-side config and result structs.

`CourtBoundsCheckConfig`

- `enabled = true`
- `seg_metadata_key = "yolo_seg"`
- `court_class_indices = {1}`
- `mask_threshold = 0.5`
- `sample_rows = 3`
- `min_horizontal_overlap_ratio`
- `max_horizontal_outside_px`
- `reacquire_only = true`

`CourtBoundsCheckResult`

- `usable`
- `veto`
- `sample_rows_used`
- `best_overlap_ratio`
- `worst_overlap_ratio`
- `max_left_out_px`
- `max_right_out_px`
- `reason`

`reason` should be a short enum-like string such as:

- `disabled`
- `no_mask`
- `no_court_rows`
- `inside_bounds`
- `outside_left`
- `outside_right`
- `low_overlap`

### Utility Functions

Add pure helper functions under `ball_tracker_detail`:

- row sampling across bbox height
- court interval extraction for one mask row
- overlap computation between bbox x-range and court interval
- aggregate bounds decision for a detection box

The utility entry point should take:

- `DetectionBox`
- `ParsedFrameMetadata`
- `CourtBoundsCheckConfig`

and return `CourtBoundsCheckResult`.

## Data Flow

### Parsing

Keep the existing CPU mask parsing path in `ball_tracker.cpp` and reuse `ParsedFrameMetadata`.

No new side-data format is needed.

### Reacquire Hook

Apply the new court-bounds check only when the tracker has already decided the candidate is a reacquire candidate.

This means:

- normal continuous detections are unaffected
- stale far reacquires are checked
- edge-jump provisional reacquires are checked

If the court-bounds result says `veto=true`, the reacquire is rejected immediately.

### Logging

When debug logging is enabled, emit a structured summary for reacquire candidates:

- bbox center
- bbox width
- sample rows used
- worst overlap ratio
- left/right outside distance
- veto result
- reason

This replaces the current center-support style logging for off-court veto decisions.

## Decision Rule

The decision rule should be conservative.

Recommended logic:

- if no usable court rows are found, do not veto
- if the bbox has enough horizontal overlap with court on any sampled row, do not veto
- veto only when the bbox is clearly outside the left or right court bounds across the sampled rows

Recommended first-pass aggregation:

- sample `3` rows: top-third, center, bottom-third of the bbox
- compute overlap ratio per row:
  - `overlap_ratio = overlap_width / bbox_width`
- compute outside distances:
  - `left_out_px = max(0, court_left - bbox_left)`
  - `right_out_px = max(0, bbox_right - court_right)`
- veto when both conditions hold:
  - worst or median overlap ratio is below threshold
  - outside distance exceeds threshold

This two-part rule reduces accidental vetoes when the bbox only barely touches the court edge.

## Defaults

Make the improved court-bounds veto active by default for reacquires.

Default settings:

- `enabled = true`
- `reacquire_only = true`
- `mask_threshold = 0.5`
- `sample_rows = 3`
- `min_horizontal_overlap_ratio = 0.35`
- `max_horizontal_outside_px = 2.0` in mask-space pixels

These values are intentionally conservative and should be tuned from real runs after implementation.

## Example Cleanup

Clean up the ball-tracker example surface to match what is actually used.

Goals:

- standardize example behavior around the new default-on court-bounds veto
- remove explicit graph-level config that becomes redundant
- keep example-specific knobs only where they differ intentionally

Expected cleanup:

- standard tracker examples should no longer need explicit `override_off_court_*` parameters just to enable basic court-bound veto
- cropped and non-cropped tracker examples should share the same default veto behavior unless they intentionally override thresholds
- unused legacy off-court center-sample parameters should be removed or deprecated if they no longer reflect the implementation

We should only remove parameters from examples after confirming the new defaults cover their current behavior.

## Error Handling

If any of these are missing, the veto must fail open:

- no segmentation metadata
- no CPU mask side data
- no court class detections
- empty court interval on sampled rows

Fail-open is required so ball tracking does not silently stop working when segmentation is absent.

## Testing Plan

Manual verification on `nba.mp4`:

1. Reproduce the frame `1580-1582` sequence.
2. Confirm the frame `1581` reacquire candidate is vetoed by the new bounds check.
3. Confirm normal ball motion above the court is still accepted.
4. Confirm normal in-bounds reacquires still pass.
5. Review logs for overlap ratios and outside distances.

Additional checks:

- compare tracker output before/after on the same `nba.mp4`
- inspect a few good edge-side ball sequences
- verify examples without segmentation metadata fail open

## Risks

- mask-space thresholds may be sensitive because CPU masks are low resolution
- row-based court interval extraction may be noisy where the segmentation mask has gaps
- too aggressive overlap thresholds could suppress valid sideline balls

Mitigations:

- keep defaults conservative
- fail open when court data is weak
- log detailed overlap metrics for tuning

## Implementation Notes

- prefer utility functions in `ball_tracker_utils.hpp` over adding more private helpers in `ball_tracker.cpp`
- keep the tracker callsite thin
- preserve existing behavior for non-reacquire detections
- do not introduce GPU dependencies into the tracker for this work

## Open Questions Resolved

- Use CPU mask path, not CUDA: yes
- Ignore top/bottom separation: yes
- Apply to reacquires for now: yes
- Keep logic out of `ball_tracker.cpp` if possible: yes
- Make the check active by default: yes
