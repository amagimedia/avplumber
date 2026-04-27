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

## Node: `shot_classifier`
Camera shot classification from court segmentation and player detections.

### Parameters
| Param | Default | Description |
|-------|---------|-------------|
| `seg_metadata_key` | "yolo_seg" | Court segmentation metadata |
| `player_metadata_key` | "yolo_players" | Player detection metadata |
| `metadata_key_out` | "camera_shot_info" | Output metadata key |
| `wide_court_threshold` | 0.25 | Court coverage above this is wide |
| `closeup_court_threshold` | 0.05 | Court coverage below this is closeup |
| `ambiguous_min_players` | 3 | Minimum players required to keep wide |
| `high_player_override` | 5 | High player count forces wide |
| `min_stable_frames` | 6 | Hysteresis before switching camera shot |

### Pipeline
1. Measure visible court coverage from segmentation masks
2. Count valid player detections sized like wide-shot players
3. Classify the frame as `wide` or `closeup`
4. Apply hysteresis before switching the reported camera shot
5. Output `camera_shot_type`, `camera_shot_transition`, and `court_coverage`
