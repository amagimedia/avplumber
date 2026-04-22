## Goal

Replace the single dominant YUV color descriptor with full UV+L histograms throughout the jersey color extraction and team classification pipeline. Use chi-square histogram distance for team matching instead of Euclidean color distance. Improve classification accuracy without fallback to the old single-color path.

## Problem

The current pipeline extracts a single dominant YUV color per player torso and classifies teams by Euclidean distance to two cluster centroids. This loses too much structure:

- shadows, folds, compression artifacts, and trim can shift the dominant UV bin
- two jerseys with similar average tone but different full color distributions collapse together
- the single-mode confidence metric is fragile when cloth pixels are distributed across multiple bins
- tracker instability carries a bad reduced sample forward through EMA smoothing

The torso cloth mask produced by `jersey_color_extract` is already good. The weak link is the appearance descriptor and the matching algorithm built on it.

## Scope

Three files change:

- `src/nodes/neural_net/sport_specific/jersey_color_extract.cu` - kernel output
- `src/nodes/neural_net/sport_specific/jersey_color_extract.cpp` - new GPU buffers, metadata output
- `src/nodes/neural_net/sport_specific/team_classifier.cpp` - histogram-based classification

Plus param updates in:

- `examples/yolo/yolo_infer_all_players_tracker_pose_live_teams.avplumber`

Nothing else changes. No draw node changes, no tracker changes, no side data changes.

## Design

### CUDA kernel changes

The kernel `kJerseyUVMean` in `jersey_color_extract.cu` already accumulates a 16x16 UV histogram in shared memory (`s_hist_count[256]`) but discards it, outputting only the dominant bin's average YUV.

Changes:

1. Add a shared memory L histogram: `__shared__ int s_l_hist[16]`. Each kept cloth pixel quantizes its Y value to one of 16 bins and atomicAdds into `s_l_hist`.

2. Add two new output buffers to the kernel signature:
   - `float* out_uv_hist` - `[num_dets * 256]`, normalized 16x16 UV histogram per detection
   - `float* out_l_hist` - `[num_dets * 16]`, normalized 16-bin L histogram per detection

3. Thread 0 normalizes both histograms by dividing each bin count by `s_cloth_count` and writes to the output buffers.

4. Existing outputs (`out_best_yuv`, `out_best_count`, `out_cloth_count`, `out_skin_count`, `out_confidence`) remain for debug logging and gating.

Shared memory increase: 64 bytes (16 ints for L histogram). Total shared memory per block: ~5.2 KB. Well within limits.

DtoH copy increase: from ~`5 * N * 4` to ~`(256 + 16 + 5) * N * 4` bytes per frame. At 10 players: ~11 KB. Negligible.

### jersey_color_extract.cpp changes

New persistent device buffers added to the node:

- `d_out_uv_hist_` - `CUdeviceptr`, sized `capacity * 256 * sizeof(float)`
- `d_out_l_hist_` - `CUdeviceptr`, sized `capacity * 16 * sizeof(float)`

Allocated in `ensureCapacity()`, freed in `releaseBuffers()`, passed as kernel args, DtoH-copied after kernel launch.

Metadata output per seg detection changes:

- Remove: `jersey_y`, `jersey_uv`
- Add: `jersey_uv_hist` (array of 256 floats, normalized), `jersey_l_hist` (array of 16 floats, normalized)
- Keep: `jersey_cloth_pixels`, `jersey_skin_pixels`, `jersey_confidence`, `jersey_mode_pixels`, `jersey_mode_ratio`

### team_classifier.cpp changes

#### Per-track state

`TrackColor` stores histogram EMAs instead of 3 floats:

```
struct TrackColor {
    float uv_hist_ema[256] = {};
    float l_hist_ema[16] = {};
    float last_confidence = 0.0f;
    int assigned_team = -1;
    int initial_candidate_team = -1;
    int initial_candidate_frames = 0;
    uint32_t hits = 0;
    uint64_t last_frame = 0;
};
```

EMA update: element-wise `hist_ema[i] = (1 - alpha) * hist_ema[i] + alpha * current_hist[i]`, then re-normalize the histogram to sum to 1.

#### Team prototypes

Two prototype histograms replace the two 3-float centroids:

```
float proto_uv_[2][256] = {};
float proto_l_[2][16] = {};
```

#### Distance function

Chi-square distance on both histogram types, combined with configurable weights:

```
D_uv = sum_i( (a[i] - b[i])^2 / (a[i] + b[i] + eps) )  // over 256 UV bins
D_l  = sum_i( (a[i] - b[i])^2 / (a[i] + b[i] + eps) )   // over 16 L bins
dist = uv_weight * D_uv + l_weight * D_l
```

`eps` = 1e-7 to avoid division by zero on empty bins.

#### Bootstrap

K-means on track histogram EMAs using chi-square distance:

1. Pick first track with sufficient hits as seed 0.
2. Pick farthest track from seed 0 (by chi-square) as seed 1.
3. Run up to 10 iterations: assign each track to nearest prototype by chi-square, recompute prototypes as element-wise mean of assigned histograms (then re-normalize).
4. Check prototype separation: `chi_square(proto_0, proto_1) >= bootstrap_min_prototype_distance`. If too close, remain in bootstrap mode.

After bootstrap succeeds, prototypes are frozen as the identity reference.

#### Assignment

Replaces the frozen axis projection with direct prototype distance comparison:

1. Compute chi-square distance from the current detection histogram (or track EMA if current frame has insufficient pixels) to both team prototypes.
2. Candidate team = argmin distance.
3. Margin = `abs(dist_to_team0 - dist_to_team1)`.

Two-tier lock policy (same structure as current):

- Strong lock: margin >= `initial_assignment_margin` and sufficient hits and confidence. Assign immediately.
- Weak lock: margin >= `soft_assignment_margin`. Track consecutive same-team frames. Assign after `initial_assignment_confirm_frames` consistent frames.

Current-frame-first: the current detection histogram is the primary signal. Track EMA is the fallback when the current frame has insufficient cloth pixels or confidence.

#### Online prototype update

After assignment, update the assigned team's prototype from confirmed same-team tracks:

```
proto[team][i] = (1 - ema_alpha_centroid) * proto[team][i] + ema_alpha_centroid * sample[i]
```

Then re-normalize. Only update when the track's margin exceeds `assignment_margin` and the track agrees with its assigned team (same gating as current centroid updates).

Frozen identity is preserved because prototypes only drift within their own team's distribution space. No global swap is possible since each track pushes only its own team's prototype.

#### Handoff

`RecentAppearance` stores the histogram pair instead of single y/u/v:

```
struct RecentAppearance {
    float x1, y1, x2, y2;
    float uv_hist[256];
    float l_hist[16];
    float confidence;
    int team;
    uint64_t frame;
};
```

`findRecentHandoff` compares using chi-square histogram distance instead of weighted color distance. The `handoff_max_color_distance` param is replaced with `handoff_max_hist_distance` in chi-square units.

### Parameter changes

Removed from team_classifier:

- `luma_weight` - replaced by separate histogram weights

Added to team_classifier:

- `uv_weight` (default 1.0) - weight for UV histogram chi-square in combined distance
- `l_weight` (default 0.5) - weight for L histogram chi-square in combined distance
- `bootstrap_min_prototype_distance` (default 0.1) - minimum chi-square between prototypes to accept bootstrap

Changed defaults (chi-square distance scale differs from Euclidean color distance):

- `soft_assignment_margin` - new default 0.02
- `initial_assignment_margin` - new default 0.05
- `assignment_margin` - new default 0.08
- `handoff_max_hist_distance` (replaces `handoff_max_color_distance`) - new default 0.15

These are conservative starting guesses. Exact values will be tuned during live testing.

Unchanged params: `iou_match_threshold`, `ema_alpha_track`, `ema_alpha_centroid`, `bootstrap_frames`, `bootstrap_min_tracks`, `track_idle_frames`, `handoff_max_age_frames`, `handoff_max_center_distance_rel`, `handoff_min_size_ratio`, `initial_assignment_min_hits`, `initial_assignment_confirm_frames`, `min_jersey_pixels`, `min_jersey_confidence`, all write-back/rewrite params, `debug_log_every_n`.

### Debug logging

Update the per-frame debug log line to include:

- chi-square distances to both prototypes for sampled detections
- margin (distance difference)
- top-3 UV bins and top L bin per sampled detection
- prototype separation (chi-square between the two team prototypes)

Keep existing counters: `tracked`, `seg`, `matched`, `assigned_known`, `assigned_t0/t1`, `sticky_mismatch`, `handoff_matches/locks`, `strong/weak_locks`, `bootstrapped`.

### Example script changes

Update `team_classifier` params in `yolo_infer_all_players_tracker_pose_live_teams.avplumber`:

- Remove: `luma_weight`
- Add: `uv_weight`, `l_weight`, `bootstrap_min_prototype_distance`
- Update margin defaults to chi-square scale values
- Replace `handoff_max_color_distance` with `handoff_max_hist_distance`

No other nodes in the example change.

## Data Flow

1. `player-seg` produces the player mask (unchanged).
2. `jersey_color_extract` CUDA kernel keeps only torso cloth pixels (unchanged selection).
3. Kernel accumulates 16x16 UV histogram and 16-bin L histogram over cloth pixels.
4. Kernel normalizes and outputs both histograms plus coverage stats.
5. Node DtoH copies histograms and writes `jersey_uv_hist`, `jersey_l_hist` into seg metadata.
6. `team_classifier` matches player detections to seg detections by IoU (unchanged).
7. Per-track histogram EMA is updated from matched detections.
8. Bootstrap builds two team histogram prototypes from track EMAs using k-means with chi-square.
9. Each detection is classified by chi-square distance to the two frozen prototypes.
10. Track persistence and handoff smooth the result using histogram similarity.

## Error Handling

- If torso cloth pixel count is below `min_jersey_pixels`, skip histogram for that detection (same gate as current). Output zero histograms.
- If bootstrap prototypes are too close (below `bootstrap_min_prototype_distance`), remain in bootstrap mode.
- If current histogram is weak but handoff match is strong, allow handoff to preserve team continuity.
- Shot transition resets all state (prototypes, track EMAs, recent appearances) as current code does.
- Empty histogram (all zeros) treated as no evidence - skip EMA update, skip assignment.

## Performance

- Kernel: shared memory increases by 64 bytes. No new global memory reads. One additional shared-memory reduction (16 bins for L) trivially fits in existing block. Kernel runtime increase is negligible.
- DtoH: ~11 KB at 10 players vs ~200 bytes before. Still negligible vs frame decode/encode time.
- JSON: 272 floats per player serialized. At 10 players, ~27 KB JSON per frame for the seg metadata key. This is the largest cost but still small relative to video frame data.
- CPU classifier: chi-square over 272 bins per player per team = ~544 multiplies per player. At 10 players: ~5440 ops. Negligible.
- Memory: `RecentAppearance` grows from ~40 bytes to ~1.1 KB per entry. With `handoff_max_age_frames=3` and 10 players, ~33 KB total. Fine.
- `TrackColor` grows from ~28 bytes to ~1.1 KB per entry. With ~20 active tracks, ~22 KB total. Fine.

## Testing

- Remote build and run on the fedora T4 box against `bbl_alt.mp4`.
- Confirm `jersey_color_extract` debug logs show non-zero histogram bins and reasonable cloth counts.
- Confirm `team_classifier` bootstrap succeeds and prototype chi-square separation is well above threshold.
- Compare assignment stability before and after: track team flips over time, assignment margin distribution.
- Visual inspection of red/blue torso masks on the RTMP stream - white jerseys consistently one color, black jerseys consistently the other.
- Verify no regression in tracking, ball detection, viewport, or encode quality.

## Non-Goals

- Adding a neural network for team classification.
- Moving classification out of the C++/CUDA pipeline.
- Changing the torso mask extraction logic (already good).
- Solving referee/coach classification (separate problem).
- Hardcoding jersey colors (algorithm must be general).
