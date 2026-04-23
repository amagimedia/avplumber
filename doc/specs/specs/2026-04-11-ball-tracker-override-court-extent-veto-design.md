# Ball Tracker Override Court-Extent Veto

## Goal

Block long-running false ball reacquisitions near the end of the cropped run, where the tracker can eventually jump to detections far outside the playable court area, without touching normal gated detections.

The target failure mode is the late stale-reacquire path around frames like `1590+`, where false candidates appear far left of the court and can survive long enough to become override candidates.

## What Failed

The previous center-point court-mask check was too aggressive.

- It was applied in the wrong place first: normal gate instead of override-only.
- Even after narrowing to override-only, sampling court support at the ball center still failed.
- In the late segment, real-ball detections near `x≈772, y≈227` also got `support=0`, because the ball center is often above the visible segmented court pixels.

So segmentation can still help, but not as a center-point “ball must lie on the court mask” test.

## Proposed Behavior

Use court segmentation only inside stale far-reacquire / override logic.

When a candidate is being considered for override-style reacquisition:

1. Estimate the horizontal extent of the segmented court from the CPU court masks.
2. Check whether the candidate lies well outside that court extent.
3. If it is far outside, block pending override accumulation and keep coasting / no-ball.

Normal gated detections remain unchanged.

## Why This Fits the Observed Failure

In the late bad case:

- false candidate: around `x≈170`
- true candidate: around `x≈772`

Even if the ball is vertically above non-court pixels, the true candidate is still horizontally over the court area, while the false one is not.

That makes a horizontal extent / hull check much better than a center-pixel mask-support test.

## Data Source

Use the existing:

- segmentation metadata key: `yolo_seg`
- CPU segmentation side data: `AV_FRAME_DATA_YOLO_SEG_MASKS`
- court class index: `1` (`"basketball-court"`)

## Court Extent Computation

For each frame:

1. Union the selected court masks.
2. For each mask column, mark whether any pixel in that column exceeds threshold.
3. Compute:
   - leftmost occupied court column
   - rightmost occupied court column
4. Map candidate `cx` from model coordinates into mask-column coordinates.

Optional robustness:

- expand the court extent by a small margin
- ignore tiny isolated occupied columns if needed

## Veto Rule

Apply only when `far_reacquire == true`.

If the candidate’s mapped x-position lies outside the expanded court extent:

- do not start a new pending override
- do not accumulate pending override hits
- clear pending override if appropriate
- return `false` from override logic

If the court data is missing or malformed:

- fail open
- preserve current override behavior

## Parameters

- `override_off_court_veto_enabled`
- `override_off_court_seg_metadata_key`
- `override_off_court_class_indices`
- `override_off_court_mask_threshold`
- `override_off_court_extent_margin_px`

Recommended defaults:

- `override_off_court_veto_enabled = true`
- `override_off_court_seg_metadata_key = "yolo_seg"`
- `override_off_court_class_indices = [1]`
- `override_off_court_mask_threshold = 0.5`
- `override_off_court_extent_margin_px = 4`

## Debug

Log only in the override path:

- candidate `cx`
- mapped mask `x`
- court extent `[left, right]`
- margin-expanded extent
- veto result

This is enough to verify the late bad frames without polluting normal candidate logs.

## Expected Outcome

- Late far-left false detections around `x≈170` are prevented from becoming override reacquires.
- Normal true-ball detections near `x≈772` remain eligible.
- Earlier valid play is unaffected because ordinary gated detections are unchanged.
