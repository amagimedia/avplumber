# Cloth-Biased Jersey Color Extraction And Stable Team Assignment

## Goal

Reduce false team flips for tracked players without adding a new segmentation model.

The system should stop treating the full player silhouette as if it were the jersey. Instead, it should estimate a cloth-biased team color from the existing player segmentation mask and use a more stable temporal assignment policy.

## Problem

The current pipeline uses player segmentation masks, but it does not segment the jersey itself.

Today:
- `jersey_color_extract` samples color from the player mask, restricted only by a coarse torso mode
- `team_classifier` uses the extracted UV values with online centroids and per-frame nearest-centroid assignment

This is fragile for three reasons:
1. the sampled color can still be contaminated by skin, arms, head/neck spill, shorts, socks, and shadows
2. the tracked player to segmentation-instance match can be wrong on some frames
3. team assignment can flip on a single noisy frame because there is no assignment hysteresis

## Constraints

1. Keep per-pixel operations on CUDA.
2. Do not add a new neural model in this phase.
3. Keep the existing graph topology mostly intact.
4. Preserve the current player segmentation model and side-data flow.
5. Keep CPU work limited to compact per-player summaries and temporal logic.

## Recommended Design

Split the problem into two layers:
- GPU: extract a cloth-biased per-player color summary from the player segmentation mask
- CPU: classify teams from that summary with temporal hysteresis

This keeps expensive pixel work in CUDA and keeps the tracker/team state machine in C++.

## `jersey_color_extract` V2

### High-level behavior

Replace the current masked UV mean with a cloth-biased dominant-color estimator.

For each player mask:
1. restrict processing to a tighter torso ROI inside the bbox
2. ignore pixels outside the player mask
3. reject likely skin pixels in YUV space
4. accumulate a coarse color histogram over remaining pixels
5. choose the dominant cloth color mode
6. emit a compact per-player summary to metadata

### ROI policy

The ROI should be stricter than the current coarse torso crop.

Recommended initial ROI:
- vertical band centered on upper/mid torso
- exclude the top head band
- exclude the lower leg/shoe region
- optionally shrink left/right edges slightly to reduce arm contamination

The exact constants should be simple configurable fractions, not hardcoded to one broadcast geometry.

### Color space

Use YUV-derived logic, not UV-only logic.

Reason:
- colorful jerseys are often separable in UV
- white/black/gray uniforms differ more in luminance than in chroma

Recommended approach:
- use YUV for skin rejection
- accumulate either UV histogram plus luma summary, or a full coarse YUV histogram if needed

Start simpler:
- coarse UV histogram
- plus luma mean / range for the selected mode

### Skin rejection

Add a simple heuristic skin gate in shader code.

Initial design:
- approximate skin region in YUV using a configurable range / ellipse
- reject pixels that fall in likely-skin region before histogram accumulation

This is not expected to be perfect. It is only a first-pass filter to stop bare skin from dominating the jersey estimate.

### CUDA output summary

Do not read back raw pixels.

For each player, return only a compact summary such as:
- `jersey_uv`: dominant cloth chroma center
- `jersey_y`: representative luma for that mode
- `jersey_pixels`: count of accepted cloth pixels
- `skin_pixels`: count of rejected skin-like pixels
- `jersey_confidence`: a separation / dominance score

This summary is written back into segmentation metadata entries, similar to current `jersey_uv` behavior.

## `team_classifier` V2

### Input

Consume the cloth-biased summary instead of the raw mixed UV average.

### Assignment stability

Add hysteresis to team switching.

A track should not switch teams on one contrary frame.

Recommended policy:
- compute current candidate team from the extracted cloth summary
- keep a per-track assigned team and a switch counter / evidence score
- only switch if the contrary team wins by margin and persists for several frames

### Centroid stability

Do not let low-confidence assignments update centroids aggressively.

Recommended behavior:
- update centroids only for sufficiently confident assignments
- ignore low-confidence or ambiguous frames
- optionally update track EMA and centroid EMA with different gates

### Unknown state

Allow tracks to remain unknown when evidence is weak.

This is better than forcing a noisy `A/B` assignment every frame.

## Why Not Full K-Means First

A full per-player k-means inside CUDA is possible, but it is not the best first implementation.

Reasons:
- more complexity
- more convergence behavior to debug
- less deterministic than a simple histogram/mode extractor

A coarse dominant-color histogram is easier to reason about and should already solve most of the observed contamination problem.

## Data Flow

1. player segmentation masks already exist on GPU
2. `jersey_color_extract` processes masked torso pixels in CUDA
3. shader emits compact cloth summary per player
4. metadata is updated with the new summary
5. `team_classifier` reads the summary and applies hysteresis-based assignment
6. draw/debug overlays continue to visualize resulting track/team state

## Files

Expected code changes:
- `src/nodes/neural_net/sport_specific/jersey_color_extract.cpp`
- `src/nodes/neural_net/sport_specific/jersey_color_extract.cu`
- `src/nodes/neural_net/sport_specific/team_classifier.cpp`

Possible graph/config changes:
- `examples/yolo/yolo_infer_all_players_tracker_pose_live_teams.avplumber`
  Only if new thresholds or debug parameters need exposure.

## Validation

1. Build locally and on remote.
2. Run the main live example.
3. Verify the player segmentation path remains stable.
4. Inspect debug labels for fixed track IDs and ensure team flips are substantially reduced.
5. Confirm that ambiguous frames tend toward `unknown` or stable hold rather than rapid `A/B` oscillation.

## Scope

In scope:
- CUDA cloth-biased jersey color extraction
- simple skin rejection in shader code
- dominant-color summary output
- team-assignment hysteresis and confidence gating

Out of scope:
- new jersey segmentation model
- retraining existing models
- full k-means or heavy iterative clustering unless the simpler histogram design proves insufficient
