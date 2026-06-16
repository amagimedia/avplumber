# Court Calibration Current Implementation

Status: current Python implementation as of 2026-06-12. This document is the
source of truth for the tactical-view court homography path. Older pose-based,
async, spillover-reconciliation, and court-only matching paths were removed or
left in git history.

The current priority is accuracy and coverage. Speed work is next.

## Runtime Graph

The demo path is `pyplumber/examples/tactical_view.py`.

1. Decode video on CUDA, force 25 fps, and scale/pad the YOLO input to
   960x544 (`scale_cuda=960:540,pad_cuda=960:544:0:2`).
2. Run player, ball, court segmentation, and player segmentation TensorRT
   heads.
3. `court_seg_evidence_cuda` consumes GPU court-seg masks and emits:
   - CPU mask side-data at 272x272 for Python calibration,
   - luma side-data for painted-line snapping,
   - `court_seg_evidence` metadata for shot classification.
4. `CourtCalibrationNode` consumes frame metadata and side-data, publishes
   `court_calib`, `court_proj`, and `court_evidence`.
5. `draw_keypoints` draws `court_proj` in magenta and `court_evidence` in red.
   These are debug overlays only.
6. `draw_tactical_court` consumes `court_calib.h` and player feet metadata,
   draws the larger tactical panel, and filters player dots in court feet.

The neural-demo templates no longer wire the old pose/court-pose homography or
the old tactical panel path.

## Data Contracts

Input side-data:

- `AV_FRAME_DATA_YOLO_SEG_MASKS`: packed float masks from YOLO segmentation.
- `AV_FRAME_DATA_COURT_LUMA`: packed uint8 luma plane from the 960x544 model
  frame, including the 2 px vertical pad.

Input metadata:

- `yolo_seg`: segmentation detections and class labels.
- `yolo_players`: player/hoop detections, used for the hoop anchor and closeup
  guard.
- `camera_shot_info`: wide/close shot classification.

Published metadata:

- `court_calib`: source-frame to court homography and diagnostics.
- `court_proj`: projected model court points for magenta debug visualization.
- `court_evidence`: current-frame arc evidence points for red debug
  visualization.

`court_calib.h` is the real product. `court_proj` is generated from that same
homography so the magenta overlay does not drift away from the tactical view.

## Court And Camera Model

The physical court model lives in `pyplumber/court_segm/geometry.py`.

- Court size: 94 ft x 50 ft.
- Hoop ground centers: `(5.25, 25)` and `(88.75, 25)`.
- 3-pt region boundary: baseline, far corner straight, arc radius 23.75 ft,
  near corner straight.
- Homography convention: normalized source image coordinates to normalized
  court coordinates.
- Camera model: six PTZ/rig parameters `(pan, tilt, focal, cx, cy, cz)`.
  Live solves optimize pan, tilt, and focal. The venue rig is fixed at the
  nominal broadcast position so noisy masks cannot invent a free 8-DOF warp.

The right-side court model is produced by mirroring the left-side model.

## Evidence Extraction

Implemented in `pyplumber/court_segm/evidence.py`.

1. Read packed mask side-data and luma side-data.
2. Split segmentation masks by label:
   - `basketball-court` becomes the floor mask,
   - `three point line` becomes candidate 3-pt region masks.
3. Crop each 3-pt region mask to its detection box. The C++ mask assembly does
   not crop instance masks the way Ultralytics does, so Python restores the
   instance crop before contour extraction.
4. Clean each 3-pt region mask with `clean_region_mask`:
   - threshold,
   - keep the largest connected component,
   - fill enclosed holes,
   - preserve float values inside the kept region for sub-pixel contours.
5. If the explicit 3-pt class is missing or weak, synthesize the 3-pt region
   from the floor mask. The hole inside the floor mask is often the cleanest
   outline of the 3-pt region.
6. Pick the 3-pt candidate closest to the detected hoop, or the largest
   candidate when no hoop is present.
7. Extract source-pixel evidence:
   - sub-pixel 3-pt boundary contours,
   - floor/court boundary contours,
   - far sideline from the top edge of the floor mask,
   - near court/crowd cut from the bottom edge of the court mask,
   - hoop center from the `Hoop` detection.

All mask-to-source mappings undo the 960x544 letterbox geometry so the 272x272
mask grid maps back to the original frame correctly.

## Acquisition

The main solve starts in `CourtCalibrationNode.process()`.

### Template Search

`_lines_h` in `pyplumber/court_segm/matching.py` searches a pre-rendered grid
of physically valid broadcast cameras.

- Template resolution: 54x96.
- Grid: 35 pan values x 11 tilt values x 12 zoom values, pruned by court
  visibility and rim projectability.
- Each template stores a court mask, a 3-pt-region mask, and expected rim
  locations for left/right hoop sides.
- Per frame, observed court and 3-pt masks are downsampled to the same 54x96
  grid.
- The 3-pt term scores IoU and is weighted by 2.0.
- The floor term scores observed-floor coverage and penalizes templates that
  claim floor above the visible court.
- The hoop term penalizes rim mismatch. Floor-only solves require a nearby
  hoop because the floor trapezoid is too symmetric on its own.
- If 3-pt evidence exists, a 27-candidate local fine grid around the best
  template removes coarse quantization.
- The previous match lightly smooths pan/tilt/focal when the new camera is
  close enough.

The returned homography is still a physical PTZ camera, not a free-form H.

### PTZ Refinement

`_refine_ptz` in `pyplumber/court_segm/fitting.py` polishes the template
camera in physical camera space.

- Boundary points are assigned to analytic model segments under the current
  homography.
- Residuals are curve distances in court feet:
  - arc radius residual,
  - corner-straight y residuals,
  - baseline x residual.
- Far-sideline points add a `Y = 0` residual.
- Bottom court/crowd points add a hinge constraint: the near sideline must lie
  above the crowd cut, in a bounded band.
- The detected rim is projected through the ground homography and compared to
  its analytic ground-plane intersection. This anchors scale when the mask
  boundary is ambiguous.
- Optimization is bounded least squares over pan, tilt, and focal.

Acceptance gates reject bad refinements:

- median curve error in feet,
- model coverage,
- convex/large court quad sanity,
- hoop error,
- far-sideline error,
- minimum angular arc coverage.

### Luma Snap

`_luma_snap` refines the camera against painted-line ridges in the luma plane.

- It projects the model arc, far sideline, and hoop-side baseline into the
  960x544 luma image.
- For each projected sample it searches along the local normal for a bright
  ridge.
- It rejects frames where ridges are too sparse or too far from the projection.
- Degrees of freedom depend on available evidence:
  - arc: pan, tilt, focal,
  - baseline plus sideline: pan and tilt,
  - sideline only: tilt,
  - baseline only: pan.
- The snap is bounded and never accepted when it makes the residual worse.

This is the step that can beat the proto-mask quantization floor.

### Boundary Snap

`_boundary_snap` is the current-frame segmentation-boundary tracker.

- It assigns current-frame boundary points to model segments under the prior
  homography.
- It requires enough arc evidence.
- It runs a small bounded pan/tilt/focal least-squares solve.
- It is accepted only if it improves and stays within the visual evidence
  gate.

This is useful on frames where the blue court segmentation outlines the arc
better than the explicit yellow 3-pt class.

## Temporal Behavior

There is no async calibration wrapper anymore. `CourtCalibrationNode` owns the
whole state machine synchronously.

Per frame:

1. Reject or reset on non-wide shots and closeups.
2. Read masks and luma. If masks are missing, try to publish a held
   homography.
3. Try a fresh acquisition through template search, PTZ refine, luma snap, and
   boundary snap.
4. If the fresh acquisition fails a gate, `_try_track_current` attempts to
   update the last valid physical camera using current-frame boundary/luma
   evidence.
5. If tracking fails, `_publish_hold` reuses a recent homography for up to
   `hold_frames`.
6. If no valid hold exists, publish invalid metadata for the frame.

State split:

- `last_valid`: any homography good enough for tactical projection.
- `last_good`: a homography whose projected arc also agrees with visible arc
  evidence, so it is safe for full debug line visualization.

This split is intentional. The tactical view can stay correct while the source
overlay suppresses or de-emphasizes visually unsupported arc projections.

The publish source field records provenance, for example:

- `acquire`
- `track_acquire_template_failed`
- `track_acquire_sideline_gate`
- `hold_acquire_template_failed`
- `hold_acquire_hoop_gate`
- `no_masks`

## Debug Visualization

`court_proj` is the magenta source-frame overlay. It is generated from the
published `court_calib.h` in `pyplumber/court_segm/temporal.py`, then smoothed
for display only.

Display smoothing knobs:

- `court_proj_smooth_alpha`
- `court_proj_reset_px`
- `court_proj_max_gap_frames`

The smoothing does not change `court_calib.h` and does not affect the tactical
view. It only reduces flicker in the magenta debug overlay.

`court_evidence` is the red source-frame evidence overlay. It shows sampled
arc evidence points from the current segmentation masks. It is not a model
projection.

`project_arc` in `court_calib` means the homography passed the visual arc
conformance gate for the frame. A valid homography can still have
`project_arc=false` when the tactical projection is usable but the source
overlay should not claim a visually unsupported arc.

## Tactical View

`src/nodes/neural_net/draw/draw_tactical_court.cpp` consumes `court_calib`.

- It parses `court_calib.h`, `source_w`, `source_h`, and `hoop_on_left`.
- Player feet from `player_feet` are projected to court feet through the same
  homography.
- Points more than 2 ft outside the court are rejected instead of clamped.
- Per-player alpha-beta filtering runs in court feet, not source pixels.
- Fallback bbox-bottom feet are accepted with lower trust.
- Lost players can coast for `player_hold_frames`.
- The static tactical court is drawn from the known court model, not from the
  source-frame magenta points.

The tactical panel in `tactical_view.py` is scaled about 50 percent larger:
540x330 with inner padding 21 and line thickness 3.

## CUDA Pieces

Implemented in `pyplumber/court_segm/cuda.py`.

Active CUDA/CuPy pieces:

- largest-component and hole-fill mask cleanup,
- synthetic floor-hole 3-pt region generation,
- cuBLAS template-mask GEMM,
- 27-candidate fine-grid IoU scoring.

Boundary extraction intentionally stays on CPU via `skimage.measure.find_contours`
because the solver needs a small set of sub-pixel contour points, not a dense
GPU raster.

## Removed Paths

Removed from the current implementation:

- `AsyncCourtCalibrationNode`
- old pose/court-pose homography in neural-demo templates,
- old `draw_tactical_court` neural-demo wiring,
- YOLO pose dependency in neural-demo,
- `reconcile_spillover`,
- `_court_only_match`,
- async/prewarm/track-disagreement switches in the Python examples.

The C++ `draw_tactical_court` node remains because the new Python tactical
view still uses it to render the top-down panel.

## Current Benchmark

Accuracy-first run on the strict continuous wide BBL clip, 38.0 s output:

| metric | value |
|---|---:|
| frames logged | 950 |
| valid homography coverage | 100.0% |
| `project_arc` coverage | 61.7% |
| wall time | 113.23 s |
| effective processing FPS | 8.39 |
| realtime speed factor | 0.336x |
| hold age max | 27 frames |
| hold age p90 | 3 frames |
| `err_ft` p50 | 1.388 ft |
| `err_ft` p90 | 2.071 ft |
| arc source-pixel error p50 | 18.6 px |
| arc source-pixel error p90 | 436.0 px |

Source provenance in that run:

| source | frames |
|---|---:|
| `acquire` | 766 |
| `hold_acquire_sideline_gate` | 109 |
| `hold_acquire_hoop_gate` | 35 |
| `track_acquire_sideline_gate` | 18 |
| `hold_acquire_template_failed` | 15 |
| `track_acquire_hoop_gate` | 6 |
| `track_acquire_template_failed` | 1 |

Interpretation: tactical homography coverage is high, but `project_arc`
coverage and speed are the next two work items. The current source-frame arc
metric has a long tail because visually unsupported frames are still valid for
tactical projection and may be held or tracked through partial evidence.

## Main Knobs

Environment knobs in `tactical_view.py`:

- `AVP_CALIB_EVERY_N`: calibration cadence. Accuracy testing currently uses 1.
- `AVP_COURT_CPU_MASK_EVERY_N`: mask side-data cadence.
- `AVP_EMIT_LUMA`: enable luma side-data.
- `AVP_RELATIVE_REFINE`: enable PTZ refine.
- `AVP_COURT_PROJ_SMOOTH_ALPHA`: magenta overlay smoother alpha.
- `AVP_COURT_PROJ_RESET_PX`: display-smoother reset threshold.
- `AVP_COURT_PROJ_MAX_GAP_FRAMES`: display-smoother max gap.
- `AVP_CALIB_LOG`: write per-frame calibration ndjson.
- `AVP_DOT_LOG`: write tactical player-dot logs.

Node parameters worth tuning:

- `evidence_gate_px`
- `visual_evidence_p90_px`
- `hold_frames`
- `hoop_gate_ft`
- `sideline_gate_ft`
- `min_arc_coverage_deg`
- `gate_median_ft`
- `min_coverage`

## Speed Work Candidates

The current slow path is Python-heavy by design. Likely speed targets:

1. Avoid full template search on every frame when a recent valid PTZ exists.
2. Make tracking the common path and acquisition the recovery path.
3. Reduce luma ridge sampling or cache projected model samples.
4. Reduce `least_squares` calls when boundary/luma evidence is weak.
5. Move the current-frame track snap into C++ once the algorithm stabilizes.
6. Keep `AVP_CALIB_EVERY_N=1` for accuracy tests, then measure cadences only
   after tracking can fill the skipped frames without visual gaps.
