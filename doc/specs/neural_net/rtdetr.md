# neural_net/rtdetr — RT-DETR TensorRT Inference

## Node: `cuda_infer_rtdetr`

Single-model RT-DETR detection inference with two output contract formats.

### Parameters
| Param | Default | Description |
|-------|---------|-------------|
| `models` | required | Array of exactly 1 model (engine, output_contract, class_names) |
| `src` / `dst` | required | Input/output video frame edges |
| `output_contract` | "rtdetr_e2e_v1" | Output format contract |
| `conf_thresh` | 0.25 | Confidence threshold |
| `max_det` | 300 | Max detections |
| `infer_every_n` | 1 | Process every Nth frame |
| `metadata_key_detection` | "yolo_detections" | Output metadata key |
| `input_format` | "RGB" | RGB or BGR |

### Output contracts

**`rtdetr_e2e_v1`** — Separate tensors: boxes[1,N,4] (xyxy pixel), scores[1,N] or [N] (float), labels[1,N] or [N] (int32/int64). Labels identified by integer dtype; scores by float dtype.

**`rtdetr_combined_v1`** — Single tensor [1,N,4+C] or [N,4+C] with normalized cxcywh boxes followed by C class scores. Boxes are denormalized by model input width/height during decode.

### Pipeline
Same preprocess/inference/sync as YOLO base, then contract-specific decode → JSON metadata on output frame.

### Destructor stats
Logs detection count histogram and per-frame detection rate on shutdown.
