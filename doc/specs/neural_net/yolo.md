# neural_net/yolo — YOLO TensorRT Inference

## Node: `cuda_infer_yolo`

Multi-model YOLO inference supporting detection, segmentation, and pose tasks.

### Parameters
| Param | Default | Description |
|-------|---------|-------------|
| `models` | required | Array of model configs (engine path, task_type, class_names, output_box_format) |
| `src` / `dst` | required | Input/output video frame edges |
| `dst_seg` | — | Optional segmentation mask output edge |
| `conf_thresh` | 0.25 | Confidence threshold |
| `max_det` | 300 | Max detections per frame |
| `infer_every_n` | 1 | Process every Nth frame |
| `metadata_key_detection` | "yolo_detections" | Detection metadata key |
| `metadata_key_segmentation` | "yolo_segmentation" | Segmentation metadata key |
| `metadata_key_pose` | "yolo_pose" | Pose metadata key |
| `input_format` | "RGB" | RGB or BGR |
| `mask_gpu_every_n` | 0 | GPU mask emission frequency (0=disabled) |
| `mask_cpu_every_n` | 0 | CPU mask emission frequency |
| `mask_cpu_resolution` | 120 | CPU mask downscale target |

### Pipeline
1. NV12→NCHW preprocess per model (CUDA kernel)
2. TensorRT enqueueV3 per model
3. D2H async copy + stream sync
4. Decode with task-specific decoder
5. Merge all model detections, sort by confidence, truncate
6. Attach JSON metadata to output frame
7. Optional: emit GPU mask side data on `dst_seg`

### Decoders
- **DetectionDecoder**: Parses [N, 4+C] raw or [N, 6] end2end boxes
- **SegmentationDecoder**: Detections + 32 mask coefficients; GPU kernel `kMaskAssemble` computes coeff·proto dot product + sigmoid; optional CPU downsample via `kMaskDownsample`
- **PoseDecoder**: Detections + keypoints [x,y,conf,...] with class-score NMS

### CUDA kernels
- `nv12_to_nchw.cu`: Preprocess (shared with rtdetr)
- `mask_assemble.cu`: `kMaskAssemble` (coeff×proto), `kMaskDownsample` (bilinear)
