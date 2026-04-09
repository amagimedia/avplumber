# neural_net/utils — Viewport and Reframing Nodes

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

---

## Node: `reframer`
Dynamic video reframing using a TensorRT trajectory prediction model.

### Parameters
| Param | Default | Description |
|-------|---------|-------------|
| `engine_path` | required | TensorRT model for viewport prediction |
| `visual_model_w/h` | required | Model input resolution |
| `history_length` | required | Frame history for model input |
| `kinematics_dim` | required | State vector dimensionality |

### Pipeline
1. Encode visual features: NV12 → NCHW tensor (CUDA kernel)
2. Maintain kinematic state (viewport center, velocity)
3. Run TensorRT inference for next-frame viewport prediction
4. Output viewport crop coordinates with smooth transitions

### CUDA kernels
Uses `kNV12_to_NCHW_fp32/fp16` for visual feature encoding, plus `reframer.cu` for crop extraction.
