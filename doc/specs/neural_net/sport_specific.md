# neural_net/sport_specific — Basketball Analysis Nodes

## Node: `shot_classifier`
Classifies each frame as `wide`, `closeup`, or `ambiguous` based on court segmentation mask coverage and player detection count. Writes a hysteresis-smoothed shot type to frame metadata.

### Parameters
| Param | Default | Description |
|-------|---------|-------------|
| `seg_metadata_key` | `"yolo_seg"` | Frame metadata key for court segmentation JSON |
| `seg_side_data_slot` | `0` | CPU side-data slot index for segmentation masks |
| `seg_mask_threshold` | `0.5` | Per-pixel threshold for counting a mask pixel as positive |
| `court_class_indices` | `[0, 1]` | Segmentation class indices that count as court |
| `player_metadata_key` | `"yolo_players"` | Frame metadata key for player detections |
| `player_labels` | `["Player"]` | Detection labels counted as players |
| `player_min_conf` | `0.25` | Minimum confidence for a player detection to count |
| `wide_court_threshold` | `0.25` | Court coverage ≥ this → candidate for wide |
| `closeup_court_threshold` | `0.05` | Court coverage ≤ this → candidate for closeup |
| `ambiguous_min_players` | `3` | Minimum valid-sized players required to call wide |
| `high_player_override` | `7` | Player count ≥ this → wide regardless of court coverage |
| `player_height_fraction` | `0.25` | Expected player height as fraction of frame height |
| `player_height_tolerance` | `0.45` | Fractional ± tolerance on expected player height |
| `player_min_aspect_ratio` | `0.75` | Minimum h/w ratio to accept a player as standing |
| `min_stable_frames` | `6` | Frames a candidate type must hold before switching |
| `reuse_last_court_coverage_frames` | `0` | Reuse last seen coverage for this many frames when seg unavailable |
| `metadata_key_out` | `"camera_shot_info"` | Output metadata key (JSON) |
| `debug_log_every_n` | `1` | Log a line every N frames (0 = disabled) |

### Court coverage
Coverage is computed as the **union** of all court-class mask pixels: a pixel is counted once regardless of how many court classes overlap it, so coverage is always in [0, 1].

### Classification logic
1. If `valid_player_count ≥ high_player_override` → `wide` (overrides court coverage).
2. Else if `court_coverage ≤ closeup_court_threshold` OR `valid_player_count < ambiguous_min_players` → `closeup`.
3. Else → `wide`.

Then hysteresis: the raw type must match the current candidate for `min_stable_frames` consecutive frames before the output type changes.

### Output JSON (`metadata_key_out`)
```json
{
  "camera_shot_type": "wide",        // "wide" | "closeup"
  "camera_shot_transition": false,   // true on the first frame of a type change
  "court_coverage": 0.31,            // [0.0, 1.0]
  "court_coverage_cached": true,     // present only when value is reused
  "court_coverage_age_frames": 3     // present only when value is reused
}
```

---

## Node: `ball_tracker`
Kalman-filtered basketball tracking with multi-criteria selection and coasting.

### Parameters
| Param | Default | Description |
|-------|---------|-------------|
| `metadata_key` | "yolo_detections" | Detection input key |
| `target_label` | "basketball" | Ball class label filter |
| `target_class` | — | Alternative: filter by class index |
| `min_conf` | 0.04 | Minimum detection confidence |
| `coast` | false | Enable Kalman coasting when no detection |
| `coast_max` | 6 | Max consecutive coast frames |
| `trail_max` | 200 | Trail history length |
| `dump_file` | — | CSV output path for analysis |

### Selection weights
`w_conf` (1.0), `w_dist` (1.0), `w_size` (0.5), `w_round` (0.3) — score candidates by confidence, distance to prediction, size match, roundness.

### 5-stage gating
1. **Hard cap** — reject if jump > `max_jump_rel` × frame diagonal
2. **Distance gate** — reject if distance > `gate_rel` × diagonal (min `gate_min_px`)
3. **Prediction gate** — use Kalman predicted position if `gate_use_pred`
4. **IoU continuity** — reject if IoU < `min_iou` with previous box
5. **Speed constraint** — reject if speed > `speed_mult` × recent average

### Override
After `override_after` consecutive rejections, accept any detection above `override_conf`.

### Output
Replaces detection metadata with single tracked ball. Appends `trail` array of [x,y] points for `draw_trail`. CSV columns: frame, x, y, conf, source (detected/coasted/override/""), dx, dy, dist.

---

## Node: `basketball_analysis`
Shot detection and game event analysis from detection metadata.

### Parameters
| Param | Default | Description |
|-------|---------|-------------|
| `metadata_key_in` | "yolo_detections_v1" | Detection source |
| `metadata_key_out` | "basketball_analysis_v1" | Analysis output key |
| `ball_label` / `player_label` | "basketball"/"player" | Class labels |
| `min_speed_px_per_frame` | 5.0 | Ball speed threshold |
| `arm_frames` | 2 | Consecutive near-player frames to arm |
| `confirm_frames` | 3 | Frames to confirm shot |
| `shot_make_min_travel_px` | 70.0 | Minimum ball travel for shot |
| `shot_hoop_memory_frames` | 24 | Hoop detection lookback window |

### Pipeline
1. Match ball detections across frames
2. Track ball proximity to player foot bboxes
3. Arm shot state when ball near player for `arm_frames`
4. Detect release event (ball speed exceeds threshold)
5. Fit quadratic trajectory to confirm shot arc
6. Check hoop proximity within memory window
7. Output shot events with type and confidence
