# Tactical Court Calibration v2 — Plan

Goal: top-down tactical view from a single wide broadcast camera with player
positions accurate to ~1–3 ft for ≥99% of visible players in wide shots.
Court boundary, paint, hoops, and three-point lines rendered from a calibrated
ground-plane homography. Calibration is driven by the court segmentation
model. The YOLO court-pose model is kept at most as an optional cold-start
initializer (its accuracy is too low for anything else — it never enters the
refinement, and the fitted result must pass the same residual gates
regardless of init source); if M0 shows curve-based cold start is reliable,
the pose head is dropped to free the engine slot.

Content assumption: always basketball. Court dimensions are a configurable
model (NBA 94×50 ft default: hoop center 5.25 ft from baseline, arc radius
23.75 ft, corner-3 lateral 22 ft; league presets for FIBA/NCAA later).

## Evidence base (probe on ~95 s NBA clip, 2026-06-10)

Ten frames were run through `court-segmentation.pt` (same weights as the
deployed `court-segmentation_960x544.plan`):

- The `three point line` class segments the **filled region enclosed by the
  3-pt line** (baseline to arc), not a thin line. Its outer boundary is the
  3-pt arc + corner straights; its lower boundary is the baseline — all known
  court-model curves.
- RANSAC ellipse fit to the 3-pt region boundary: **inlier RMS 1.3–1.5 px at
  1920×1080** on 7/9 frames with mutually consistent ellipse axes
  (~815–880 × 226–242 px). The two degenerate frames had low detection conf
  (0.25, 0.67) and implausible axes — both detectable by gates.
- The court-region mask's lower boundary follows the courtside crowd, not the
  near sideline. The near sideline is **never observable** in this framing.
- Per-frame failure modes that the design must absorb: class dropout (a frame
  with only one of the two classes), duplicate instances of the same class,
  contour fragmentation from players standing on lines.
- The Ultralytics probe upsamples masks to input resolution; the C++ TRT path
  stores masks at proto resolution in side data. M0 must check the actual
  slot-0 CPU mask resolution and fit with sub-pixel boundary interpolation.

## Why the current implementation is inaccurate (summary)

`src/nodes/neural_net/draw/draw_tactical_court.cpp` has two mapper paths:

1. Pose-keypoint homography: 12 court-boundary keypoints (often near-collinear
   in broadcast framing), 8 ft inlier tolerance, accepts minimal 4-point
   solves with no degeneracy check, hoop consistency check disabled by default.
2. Fallback "hoop-ray" mapper: affine, single constant px/ft depth scale —
   geometrically wrong under perspective. This is the inaccurate "depth
   estimation".

Player smoothing/jump-gating happens in panel pixel space (conflates camera
pan with player motion); projections are clamped onto the court edge instead
of rejected; hold/cached-overlay logic masks failures. The segmentation —
the densest, most accurate signal available — was used only as a direction
hint in the fallback.

Known small bug to fix along the way: `metadata_dump` scales model→source
coordinates as `source/model` ignoring the `pad_cuda 960:544:0:2` content
offset (`metadata_dump.cpp` ~line 1639).

## Architecture

Decisions locked in: NBA court dimensions only for now (league presets in
M3); linear algebra via **Eigen** added as a header-only submodule in
`deps/` (LM, conic fitting, covariance extraction).

1. **New node `court_calibration`**
   (`src/nodes/neural_net/sport_specific/court_calibration.cpp`).
   Inputs (frame metadata + side data): `yolo_seg` CPU masks (slot 0,
   court + 3pt region), `yolo_players_seg` (slot 1, for boundary excision),
   `yolo_players` (Hoop detection for side disambiguation),
   `camera_shot_info` (wide-shot gate).
   Output metadata `court_calib`: normalized 3×3 source→court homography,
   hoop side, residual stats, inlier counts, valid flag, age.
2. **`draw_tactical_court` rewrite**: delete both mapper paths; consume
   `court_calib`; render the full court model (boundary, paint, center
   circle, both 3-pt lines, hoops); per-player smoothing moves to court space.
   Add a debug mode that reprojects the court model onto the broadcast frame
   (H⁻¹) for visual calibration QA.
3. **No `metadata_dump` dependency**: the current `draw_tactical_court`
   consumes the `frame_dump` JSON produced by `metadata_dump`; the rewrite
   reads `yolo_players` (track ids, teams), `player_feet`, `ball_handler`,
   and `court_calib` metadata directly. `metadata_dump` and the game-world
   nodes are out of scope for this work. (If `metadata_dump` is used later,
   its pad-offset scaling bug still needs the fix noted above.)
4. **Target graph (M1)**: a lean VOD graph derived from
   `examples/yolo/metadata_dump_bbl.avplumber` minus `metadata_dump`,
   `scoreboard_roi`/`scoreboard_ocr`, `game_state`, `court_zone`,
   `shot_attempt_detector`, and `possession_tracker`. Kept: players + ball +
   court-seg + player-seg heads, `shot_classifier` (wide gate),
   `player_tracker`, `player_feet_seg`, `player_torso_seg` +
   `jersey_color_extract` + `torso_team_classifier`, `ball_tracker` +
   `ball_handler`, then `court_calibration` → `draw_tactical_court` → NVENC.
   `Yolo_Pose` head kept only while it serves as a cold-start candidate; drop
   it (freeing the engine slot) once M0 confirms curve-based cold start.
   Keep seg at 960×544 initially, raise to 1280×736 only if M0 shows the
   precision budget needs it.
5. **Follow-up consolidation**: `court_zone` consumes `court_calib` instead of
   its own 3-pt mask logic; `court_polygon` (pose-based) becomes unused —
   deprecate.

## Calibration algorithm

### Stage A — observation extraction (per wide frame)
- Merge duplicate instances per class (max over masks), threshold 0.5.
- Excise dilated player-seg masks and the score-bug region from boundaries;
  drop boundary pixels within ~8 px of frame edges (clipping, not court).
- Extract sub-pixel boundary points from the float mask (0.5-crossing).
- Court region: keep only the upper boundary chain (far-sideline candidate);
  the lower chain is crowd — never used.
- 3-pt region: full outer boundary (arc + corner straights + baseline).

### Stage B — per-frame fit
- **Warm init**: previous frame's solution.
- **Cold init** (first frame, cuts, side switch): RANSAC line fits (far
  sideline from court mask; baseline as the dominant hoop-side line of the
  3-pt boundary), RANSAC ellipse fit for the arc with plausibility gates
  (detection conf, axis ratio/size, inlier fraction — the probe's two bad
  frames fail these). Derive **virtual keypoints** from fitted curves:
  far corner (sideline∩baseline), two baseline∩corner-3 points, two
  straight↔arc junctions, arc apex. Hoop detection picks left/right half.
  DLT on the virtual correspondences → H₀. The pose model, if kept, is only
  another H₀ candidate here (useful when masks alone are ambiguous); whichever
  init is used, only the curve-refined, gate-passing result is ever published.
- **Refinement**: robust (Huber) Levenberg–Marquardt minimizing point-to-curve
  residuals **measured in court feet** between all boundary observations and
  the reprojected model curves (far sideline, baseline, corner straights, arc).
- **Precision stage (optional, for sub-foot)**: snap to the painted line —
  1-D luma ridge search along reprojected curve normals in the source frame,
  then refit. Removes any constant segmentation-boundary bias.
- **Acceptance gates**: median residual ≤ ~1.5 ft, minimum arc angular
  coverage, minimum baseline support; otherwise hold last good H (age-limited),
  then hide player dots when stale.

### Stage C — temporal model
- Parametrize as a fixed-position PTZ camera: estimate the camera center
  recursively across frames (constant per venue/shot), then solve only
  pan/tilt/focal (+ small roll) per frame — 3–4 DOF instead of 8. A single
  bad frame can no longer produce a garbage mapping.
- Constant-velocity Kalman smoothing on the PTZ parameters (not element-wise
  H lerp). Side-switch confirm logic retained from the current node.
- When segmentation degrades, propagate by parameter velocity for N frames;
  later option: sparse optical flow on court texture for frame-to-frame delta.

### Stage D — players
- Ground point: midpoint of planted feet from `player_feet_seg` (left/right
  points already computed). Fallback to bbox-bottom-center is allowed only
  with a bias and variance **measured on frames where both are available**
  (per-depth-band statistics, computed in M0) — a calibrated estimator, not a
  guess; its inflated variance feeds the uncertainty budget below.
- Project via H, then track per id **in court feet**: constant-velocity
  Kalman, gating at ~25 ft/s. The filter's covariance is the player's
  position uncertainty; during occlusion or airborne phases (foot rising vs
  court fit) no fabricated measurement is injected — the covariance grows on
  prediction alone until the dot is hidden by the rendering gate.
- Projections landing > 3 ft outside the court are rejected, not clamped.
- Ball on the tactical view: drawn only when its position is provable — i.e.
  when `ball_handler` associates it to a player (position = that player's
  feet) or it is detected at rest on the floor. During flight it is hidden
  (a ground homography cannot place an airborne ball).

### Feet measurement quality and improvements (probe findings, 2026-06-10)

Zoomed inspection of `player-seg_960x544.onnx` masks on the test clip shows
the raw signal is excellent: shoe-tight contours, left/right feet separable
as distinct blobs on nearly every player. The accuracy losses are in the
extraction, all fixable:

1. **Proto-pixel quantization (biggest, cheapest fix)**: the TensorRT engine
   outputs 240×136 mask protos and `player_feet_seg.cu` takes the lowest
   proto pixel above threshold — foot y is quantized to 4 px steps in model
   space (8 px at 1080p). With 3–5× perspective depth amplification at far
   court this alone is worth ~1.5–4 ft of depth error. Fix: sub-pixel
   0.5-crossing interpolation along columns of the float proto mask (and a
   parabola fit over the lowest rows per blob for sub-pixel x) in the same
   kernel.
2. **Planted-foot selection instead of blob midpoint**: track left/right
   foot points separately in court space; the planted foot is identified by
   near-zero court-space velocity (a provable test against its track
   covariance, consistent with the no-guessing rule). Use the planted foot
   as the position anchor; in double support average both; when neither foot
   is stationary (running/airborne) rely on filter prediction with growing
   covariance.
3. **Occluded feet**: when the mask bottom sits well above the bbox bottom
   or foot confidence is low, use the calibrated bbox-bottom fallback
   (bias/variance measured per depth band on frames where both estimators
   exist) with inflated measurement noise.
4. **Resolution headroom**: raising seg input to 1280×736 makes protos
   320×184 (~33% finer) — apply only if the sub-pixel fix leaves the feet
   term dominant in the error budget.

### Uncertainty budget — no unproven positions

Every rendered dot must carry a mathematically propagated error bound; if the
bound cannot be established or exceeds the display threshold, the dot is
hidden (or visually marked uncertain), never silently guessed.

- **Homography uncertainty**: the LM fit yields the parameter covariance
  Σ_H = (JᵀWJ)⁻¹ (first-order, from the same Jacobian the solver already
  computes). For each player foot point x, propagate through the projection:
  Σ_player = J_proj(x) Σ_H J_proj(x)ᵀ + J_H(x) Σ_foot J_H(x)ᵀ, where Σ_foot
  is the foot-point measurement covariance (seg-derived feet vs calibrated
  bbox fallback).
- **Filter consistency**: the Kalman update consumes Σ_player as measurement
  noise, so track covariance is honest by construction; prediction-only
  frames (occlusion, stale H) grow it with no artificial cap.
- **Rendering gate**: draw a dot only while √trace(Σ_track) ≤ threshold
  (default 3 ft). Stale-calibration hold frames inflate Σ_H with age instead
  of pretending the old H is exact.
- **Covariance calibration check** (M0/M2 eval): normalized residuals
  (z-scores) of the held-out-curve cross-validation and of player
  innovations must be ~N(0,1); if the bounds are overconfident, scale noise
  models until they are. This makes the "≤ 3 ft" claim verifiable without
  hand labels.

## Near-sideline occlusion (the stated biggest issue)

The near sideline is never observed and never needs to be: it is implied by H
("interpolated" by reprojecting the y = 50 ft model line). What bounds the
extrapolation error:

- The 3-pt region constrains geometry deep into the frame: arc apex ~29 ft
  from the baseline, corner straights at y = 3/47 ft — i.e. observations span
  nearly the full depth range players occupy; true extrapolation is only the
  last few feet.
- The PTZ parametrization + temporal smoothing bound per-frame wobble.
- The cross-validation metric below explicitly measures near-court error.

## Evaluation (no hand labels)

- **Held-out-curve cross-validation**: fit without the arc, measure arc
  reprojection error in feet (and inversely for the baseline). Automatic,
  per frame, in the target unit.
- Per-frame residual, % of wide frames calibrated, H jitter time series.
- Player physics: implied speed distribution (flag > 25 ft/s), stationary-dot
  jitter RMS.
- Overlay videos: court model reprojected on broadcast + minimap side by side,
  full bbl clip + one other arena for generalization.
- Targets: median cross-val curve error ≤ 1.5 ft; ≥ 98% of wide frames
  calibrated; stationary-player dot jitter ≤ 0.5 ft RMS.

## Milestones

- **M0 — offline prototype (Python, extends the probe)**: full Stage A–D on
  the bbl clip masks; overlay video; verify TRT proto mask resolution and the
  court-mask top edge vs painted far sideline; tune gates. Go/no-go against
  the targets before any C++.
- **M1 — C++ `court_calibration` node** + `draw_tactical_court` consumption +
  graph updates (pose head removed). Parity with M0 on the same clip.
- **M2 — temporal/PTZ model, line-snap refinement, `metadata_dump`
  integration** (court-space coords in `frame_dump`, pad-offset fix).
- **M3 — consolidation and hardening**: `court_zone` on `court_calib`,
  multi-league court dims, perf pass, optional 1280×736 seg engine,
  deprecate `court_polygon`.

## Risks

- TRT proto mask resolution much lower than the probe's (sub-pixel fitting
  mitigates; else raise seg input res or upsample protos). Verify in M0.
- Court-mask top edge may be the apron/crowd line rather than the painted far
  sideline in some arenas; if biased, weight the 3-pt region higher and rely
  on the line-snap stage. Verify in M0.
- Arenas where the crowd also occludes the baseline: gates + temporal hold;
  worst case the arc alone still gives 5 constraints + temporal continuity.
- Fit cost is negligible (thousands of boundary points, LM on ≤ 8 params,
  CPU at 25 fps is fine); no CUDA work needed for calibration itself.
- Future accuracy lever if ever needed: add seg classes (paint, center
  circle, center line) — same method, more constraints; no pose model ever.
