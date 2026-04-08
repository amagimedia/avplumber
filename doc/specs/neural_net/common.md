# neural_net/common — TensorRT Inference Base

Shared infrastructure for all TensorRT-based inference nodes.

## Files
- `infer_trt_base.hpp` / `infer_trt_base.cpp`

## CudaInferTrtBase

Base class handling the full TensorRT lifecycle: CUDA context init from incoming hw frames, engine deserialization, binding allocation, NV12→NCHW preprocessing, inference dispatch, and D2H output sync.

### ModelRunner

Per-engine state holder:
- TRT runtime/engine/context
- IO tensor GPU allocations and name→index map
- Preprocess CUDA kernel handle + stream
- Task type (Detection / Segmentation / Pose)
- Output box format: `EndToEndXYXY` or `RawCXCYWH`
- Class names and optional class index remap

### Supported output dtypes
FLOAT, HALF (auto-converted to float), INT32, INT64 (auto-converted to float).

### Preprocessing kernel
`kNV12_to_NCHW_fp32` / `kNV12_to_NCHW_fp16` — BT.709 limited-range YUV→RGB, normalized to [0,1]. Supports BGR order. Grid: (W+31)/32 × (H+7)/8.

### Data structs
- `Detection`: x1, y1, x2, y2, conf, cls, model_index
- `DetectionResult`, `SegmentationResult` (+ GPU mask buf), `PoseResult` (+ keypoints)
