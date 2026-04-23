## Goal

Improve realtime team classification by replacing the current single-color torso summary with a GPU-computed torso appearance histogram that preserves the full cloth color distribution.

The immediate target is to reduce obvious human-visible team mistakes in cases where the torso mask is already mostly correct but the classifier still picks the wrong team because one dominant color bin is too lossy.

## Problem

The current pipeline extracts a torso cloth mask, summarizes it to one dominant `YUV` color, and classifies teams from that compressed sample. This loses too much structure:

- trim, folds, shadows, and compression can move the dominant bin
- two jerseys with similar average tone but different full color distribution can collapse together
- a visually obvious team difference across the torso can disappear when reduced to one mode
- tracker instability amplifies the problem by carrying a bad reduced sample forward

The torso mask itself is now considered good enough for the next step. The weak link is the appearance descriptor and the team matching algorithm built on top of it.

## Design

### Descriptor change

`jersey_color_extract` will stop treating a single dominant `YUV` mode as the primary jersey descriptor. Instead it will produce a compact histogram over all kept torso cloth pixels.

The first implementation will use:

- a normalized `UV` histogram for chroma structure
- a normalized `L` histogram for brightness structure
- coverage metrics such as cloth pixel count and usable histogram mass

This is intentionally `UV + L`, not full dense `LAB`, because it preserves the core perceptual structure while staying practical for the existing CUDA path. If needed later, the `L` path can be upgraded to fuller `LAB` features without changing the classifier shape.

### GPU histogram extraction

Histogram accumulation should happen in the existing CUDA extraction stage, reusing the already computed torso cloth mask.

Per kept cloth pixel:

1. Read the pixel color from the decoded frame.
2. Convert it into histogram coordinates.
3. Accumulate it into:
   - one `UV` bin
   - one `L` bin

The kernel output for each player detection will include:

- normalized `UV` histogram
- normalized `L` histogram
- dominant-bin summaries for debug logs
- existing cloth coverage stats

The extractor should continue to emit the debug torso mask side data unchanged so live visual verification still works.

### Team prototype model

`team_classifier` will replace centroid-by-single-color logic with histogram prototype matching.

Bootstrap will collect early torso histograms and build two team prototypes from them:

- prototype `0`: average normalized histogram of cluster `0`
- prototype `1`: average normalized histogram of cluster `1`

The bootstrap step remains symmetric only during initial team discovery. After bootstrap, team identity remains frozen as it does today.

### Distance function

Each current player sample will be scored against both team prototypes using histogram distance rather than dominant-color distance.

Recommended first-pass metric:

- weighted `L1` or chi-square distance on the normalized `UV` histogram
- plus weighted `L1` distance on the normalized `L` histogram

The combined score should look like:

- `dist = uv_weight * D_uv + l_weight * D_l`

`UV` carries the main team signal. `L` remains a secondary structural signal for dark/bright separation and neutral jerseys.

### Current-frame-first assignment

The current detection histogram becomes the primary classification signal.

Per-frame assignment order:

1. score the current torso histogram against both team prototypes
2. choose the nearer prototype if the margin is sufficient
3. use stored track state only for smoothing, lock confirmation, and short-lived ID handoff

This keeps the previous fix that reduces dependence on tracker correctness.

### Persistence and handoff

The current track soft-persistence logic stays, but it operates on histogram evidence instead of single-color evidence.

That means:

- track EMA state becomes histogram EMA state
- recent-appearance handoff uses histogram similarity instead of single-color similarity
- already locked players remain pinned unless the run resets on a shot cut

### Debug visibility

The frame log should expose histogram-based behavior clearly enough to diagnose misclassifications.

Add or update logs for:

- per-frame team counts
- histogram match margin to both prototypes
- whether assignment used current-frame evidence or handoff
- bootstrap prototype separation statistics
- top `UV` bins and top `L` bins for sampled detections

The existing red/blue upper-torso mask overlay remains the primary visual debug layer.

## Data Flow

1. `player-seg` produces the player mask.
2. `jersey_color_extract` keeps only torso cloth pixels.
3. CUDA accumulates per-player `UV` and `L` histograms over those pixels.
4. Extractor writes histograms and coverage stats into segmentation metadata.
5. `team_classifier` matches player detections to segmentation detections.
6. Bootstrap builds two team histogram prototypes.
7. Each current detection is classified by histogram distance to the two prototypes.
8. Track persistence and recent-appearance handoff smooth the result without becoming the source of truth.

## Error Handling

- If torso cloth coverage is too small, skip histogram-based assignment for that detection.
- If bootstrap prototypes are too close, remain in bootstrap mode and keep players unassigned rather than forcing a weak split.
- If the current histogram is weak but a recent-appearance handoff is strong, allow the handoff path to preserve team continuity.
- If the shot classifier reports a cut, reset prototypes, per-track history, and recent-appearance state as the system already does.

## Performance

This design must remain compatible with the current realtime C++/TensorRT pipeline.

Constraints:

- histogram extraction runs in CUDA, not on CPU
- descriptor size stays compact enough for metadata transport
- classifier comparison is linear in histogram size and player count
- the first version should prefer a small fixed bin count over a richer but slower descriptor

Recommended starting point:

- `UV`: `16 x 16` bins or a smaller packed variant if metadata size becomes an issue
- `L`: `16` bins

If that is too heavy in practice, reduce bin count before changing the overall design.

## Testing

- Realtime verification on the current `bbl_alt.mp4` wide example.
- Confirm that team mistakes drop in scenes where the torso mask already looks correct.
- Compare frame logs before and after the change for:
  - exact team split counts
  - assignment margin stability
  - mismatch frequency
  - handoff behavior during tracker ID changes
- Manually inspect the stream to confirm that red/blue torso masks now align better with obvious team appearance.

## Non-Goals

- adding a new neural network for team classification
- moving classification out of the current C++/TensorRT/CUDA path
- restoring full player segmentation overlays
- solving every neutral-jersey edge case in the first iteration
