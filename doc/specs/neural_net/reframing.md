# neural_net/reframing — Generic Reframing Nodes

## Node: `smooth_crop_viewport`
Smooth viewport center tracking with configurable filtering.

### Parameters
| Param | Default | Description |
|-------|---------|-------------|
| `metadata_keys` | required | Detection metadata keys to read bboxes from |
| `viewport_width/height` | required | Fixed viewport dimensions |
| `filter_type` | "kalman" | "kalman", "butterworth" (1-8th order), "lti", "none" |
| `max_velocity/acceleration/jerk/snap` | — | Derivative motion limits |
| `kalman_q_pos/q_vel/r_meas` | — | Kalman filter tuning |
| `cutoff_freq_hz` / `quality_factor` | — | Butterworth filter tuning |

### Output
Attaches `viewport_center_x` metadata to frame for downstream crop nodes.
