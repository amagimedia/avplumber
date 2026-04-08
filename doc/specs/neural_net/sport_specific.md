# neural_net/sport_specific — Basketball Analysis Nodes

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
